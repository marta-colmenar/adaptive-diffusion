import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import inspect
from ablation_models_diffusion_wa import DiffusionModelLoc # DiffusionModelSeenLocations, DiffusionModel,
from train_sequential_observed import (
    DATASETS,
    build_diffusion_schedule,
    build_model_inputs,
    choose_order,
    inverse_transform_batch,
    process_df,
)

@torch.no_grad()
def sample_mc_for_single_prefix(
    model,
    cond_v,
    cond_times,
    dt_future,
    target,
    seen_loc_idx, # Expects 1D tensor [k] or None
    alphas,
    betas,
    cumalphas,
    num_mc,
    mc_chunk,
):
    device = target.device
    bsz, length = target.shape
    num_steps = cumalphas.shape[0]
    produced = 0
    chunks = []

    while produced < num_mc:
        m = min(mc_chunk, num_mc - produced)
        rep = m

        cond_v_rep = [v.repeat_interleave(rep, dim=0) for v in cond_v]
        cond_times_rep = [t.repeat_interleave(rep, dim=0) for t in cond_times]
        dt_future_rep = dt_future.repeat_interleave(rep, dim=0)
        locs = [torch.arange(length, device=device).float() for _ in range(len(cond_v_rep))]
        
        if seen_loc_idx is not None and seen_loc_idx.numel() > 0:
            target_seen = target[:, seen_loc_idx].repeat_interleave(rep, dim=0)
            cond_v_rep.append(target_seen)
            cond_times_rep.append(dt_future_rep)
            locs.append(seen_loc_idx.float())

        locs.append(torch.arange(length, device=device).float())
        x_t = torch.randn(bsz * rep, length, device=device)

        # Autocast FP16 unlocks Tensor Cores on RTX 4090
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for t_step_eval in range(num_steps - 1, 0, -1):
                alpha_t = alphas[t_step_eval - 1]
                cumalpha_t = cumalphas[t_step_eval - 1]
                sigma = torch.sqrt(betas[t_step_eval - 1])
                factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)

                t_tensor = torch.full((bsz * rep,), t_step_eval, device=device, dtype=torch.long)

                noise_pred = model(
                    x_t=x_t,
                    v_hist=cond_v_rep,
                    dt_hist=cond_times_rep,
                    t_step=t_tensor,
                    dt_future=dt_future_rep,
                    locs=locs,
                )

                z = torch.randn_like(x_t) if t_step_eval > 1 else torch.zeros_like(x_t)
                x_t = 1.0 / torch.sqrt(alpha_t) * (x_t - factor * noise_pred) + sigma * z

        chunk = x_t.view(bsz, rep, length).detach().cpu().numpy().astype(np.float32)
        chunks.append(chunk)
        produced += rep

    return np.concatenate(chunks, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Monte Carlo DDPM samples for every ordered-prefix level "
            "(k=0..L) on the test set."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--n_vf", type=int, default=3, choices=[3, 4])
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained sequential checkpoint")
    parser.add_argument("--num_steps", type=int, default=52)
    parser.add_argument("--max_cond", type=int, default=2)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_mc", type=int, default=100, help="MC samples per test sample and prefix")
    parser.add_argument("--mc_chunk", type=int, default=10, help="Generate MC samples in chunks to limit memory")
    parser.add_argument("--max_test_samples", type=int, default=0, help="0 = full test set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/storage/homefs/mc21s017/data_vf/")
    parser.add_argument("--output_dir", type=str, default="/storage/homefs/mc21s017/")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--greedy", action="store_true", help="Use greedy prefix selection based on MC variance")
    parser.add_argument("--sample_offset", type=int, default=0, help="Starting test sample index")
    # parser.add_argument("--use_diffusion_model", action="store_true",
    #                     help="Use `DiffusionModel` (no seen-locations stream) and append seen-prefix as extra VF")
    # parser.add_argument("--seen_scale", type=float, default=0.5,
    #                     help="Scale applied to seen-prefix values when passed to DiffusionModel or DiffusionModelSeenLocations")
    args = parser.parse_args()

    if args.mc_chunk <= 0:
        raise ValueError("--mc_chunk must be > 0")
    if args.num_mc <= 0:
        raise ValueError("--num_mc must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = DATASETS[args.dataset]
    vf_key = f"{args.n_vf}vf"

    test_file = os.path.join(args.data_path, cfg["test_files"][vf_key])
    scaler_path = os.path.join(args.data_path, cfg["scaler_file"])

    df_test = pd.read_csv(test_file)
    vf_scaler = np.load(scaler_path, allow_pickle=True).item()

    proc_test = process_df(df_test, vf_scaler, cfg, args.n_vf)
    test_ds_full = TensorDataset(*proc_test)

    if args.max_test_samples > 0:
        end_idx = min(args.sample_offset + args.max_test_samples, len(test_ds_full))
        test_ds = Subset(test_ds_full, list(range(args.sample_offset, end_idx)))
    else:
        test_ds = test_ds_full


    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    order = choose_order(args.dataset, cfg)
    length = len(order)
    order_tensor = torch.tensor(order, dtype=torch.long, device=device)
    # permute randomly order
    order_tensor = order_tensor[torch.randperm(length, device=device)]
    print(f"Randomized order: {order_tensor.cpu().numpy()}")
    max_cond_eff = min(args.max_cond, args.n_vf - 1)

    ckpt_path = args.checkpoint.strip()
    if not ckpt_path:
        ckpt_path = os.path.join(
            args.output_dir,
            f"sequential_observed_{args.dataset}_{args.n_vf}vf.pth",
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Try to infer architecture from checkpoint metadata
    inferred_layers = args.layers
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Count sa_blocks in checkpoint to infer K
        state_dict = ckpt["model_state_dict"]
        sa_block_keys = [k for k in state_dict.keys() if k.startswith("sa_blocks.")]
        if sa_block_keys:
            max_sa_idx = max([int(k.split(".")[1]) for k in sa_block_keys])
            inferred_layers = max_sa_idx + 1
            print(f"Inferred K={inferred_layers} from checkpoint")
    
    model = DiffusionModelLoc(
        L=length,
        D=args.D,
        K=inferred_layers,
        max_cond=max_cond_eff,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    alphas, betas, cumalphas = build_diffusion_schedule(args.num_steps, device)

    n_test = len(test_ds)
    n_prefix = length + 1

    if not args.output_name:
        args.output_name = f"sequential_prefix_mc_{args.dataset}_{args.n_vf}vf_test_mc{args.num_mc}"

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"{args.output_name}_pred.npy")
    target_path = os.path.join(args.output_dir, f"{args.output_name}_target.npy")
    meta_path = os.path.join(args.output_dir, f"{args.output_name}_meta.npz")

    pred_mem = np.lib.format.open_memmap(
        pred_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_test, n_prefix, args.num_mc, length),
    )
    target_mem = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_test, length),
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test samples: {n_test}")
    print(f"Locations (L): {length}")
    print(f"Prefix levels: {n_prefix} (k=0..{length})")
    print(f"MC samples per prefix: {args.num_mc}")
    print(f"Output pred: {pred_path}")
    print(f"Output target: {target_path}")

    t0 = time.time()
    offset = 0
    if args.greedy:
        all_greedy_orders = np.empty((n_test, length), dtype=np.int64)
        print("Running Per-Sample Greedy Active Selection (FP16 Accelerated)...")

        for batch_idx, batch in enumerate(test_loader):
            batch = [b.to(device, non_blocking=True) for b in batch]
            cond_v, cond_times, _target_time, dt_future, target = build_model_inputs(
                batch, max_cond=max_cond_eff
            )

            bsz = target.size(0)
            target_np = target.detach().cpu().numpy().astype(np.float32)
            target_mem[offset:offset + bsz, :] = target_np

            for i in range(bsz):
                global_idx = offset + i
                t_sample_0 = time.time()

                cond_v_i = [v[i:i+1] for v in cond_v]
                cond_times_i = [t[i:i+1] for t in cond_times]
                dt_future_i = dt_future[i:i+1]
                target_i = target[i:i+1]

                seen_prefix_i = []
                unseen_set = set(range(length))

                # Step k = 0
                pred_0 = sample_mc_for_single_prefix(
                    model=model, cond_v=cond_v_i, cond_times=cond_times_i, dt_future=dt_future_i,
                    target=target_i, seen_loc_idx=None, alphas=alphas, betas=betas,
                    cumalphas=cumalphas, num_mc=args.num_mc, mc_chunk=args.mc_chunk,
                )
                pred_mem[global_idx, 0, :, :] = pred_0[0]

                # Step k = 1..L
                for k in range(1, length + 1):
                    var_prev = np.var(pred_mem[global_idx, k - 1, :, :], axis=0)

                    unseen_list = list(unseen_set)
                    best_next_loc = unseen_list[np.argmax(var_prev[unseen_list])]
                    
                    seen_prefix_i.append(best_next_loc)
                    unseen_set.remove(best_next_loc)

                    seen_loc_tensor = torch.tensor(seen_prefix_i, dtype=torch.long, device=device)

                    pred_k = sample_mc_for_single_prefix(
                        model=model, cond_v=cond_v_i, cond_times=cond_times_i, dt_future=dt_future_i,
                        target=target_i, seen_loc_idx=seen_loc_tensor, alphas=alphas, betas=betas,
                        cumalphas=cumalphas, num_mc=args.num_mc, mc_chunk=args.mc_chunk,
                    )
                    pred_mem[global_idx, k, :, :] = pred_k[0]

                all_greedy_orders[global_idx, :] = np.array(seen_prefix_i, dtype=np.int64)

                sample_dt = time.time() - t_sample_0
                print(
                    f"Sample {global_idx + 1:4d}/{n_test} done in {sample_dt:.1f}s "
                    f"| total elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

            offset += bsz


    else:
        for batch_idx, batch in enumerate(test_loader):
            batch = [b.to(device, non_blocking=True) for b in batch]
            cond_v, cond_times, _target_time, dt_future, target = build_model_inputs(
                batch,
                max_cond=max_cond_eff,
            )

            bsz = target.size(0)
            target_np = target.detach().cpu().numpy().astype(np.float32)
            target_mem[offset:offset + bsz, :] = target_np

            seen_loc_idx = None
            
            for k in range(n_prefix):
                if k > 0:
                    seen_loc_idx = order_tensor[:k]
                else:
                    seen_loc_idx = None

                pred_k = sample_mc_for_single_prefix(
                    model=model,
                    cond_v=cond_v,
                    cond_times=cond_times,
                    dt_future=dt_future,
                    target=target,
                    seen_loc_idx=seen_loc_idx,
                    alphas=alphas,
                    betas=betas,
                    cumalphas=cumalphas,
                    num_mc=args.num_mc,
                    mc_chunk=args.mc_chunk,
                )

                pred_mem[offset:offset + bsz, k, :, :] = pred_k

            offset += bsz
            elapsed = time.time() - t0
            print(
                f"Batch {batch_idx + 1:4d} | processed={offset}/{n_test} "
                f"| elapsed={elapsed:.1f}s",
                flush=True,
            )

    np.savez_compressed(
        meta_path,
        dataset=np.asarray(args.dataset),
        n_vf=np.int64(args.n_vf),
        num_steps=np.int64(args.num_steps),
        max_cond=np.int64(max_cond_eff),
        num_mc=np.int64(args.num_mc),
        length=np.int64(length),
        n_test=np.int64(n_test),
        prefix_ks=np.arange(n_prefix, dtype=np.int64),
        order=np.asarray(order, dtype=np.int64),
        pred_path=np.asarray(pred_path),
        target_path=np.asarray(target_path),
        checkpoint=np.asarray(ckpt_path),
        greedy_orders=all_greedy_orders if args.greedy else None,
    )

    total_dt = time.time() - t0
    print(f"Done. Total time: {total_dt:.1f}s")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
