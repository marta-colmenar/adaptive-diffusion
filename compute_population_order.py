"""
compute_population_order.py
============================
Compute the A6 population-optimised acquisition order from the train/val set.

Algorithm (greedy, population-level)
--------------------------------------
  1. Sample k=0 ensemble for every train/val patient (no observed locations).
  2. For each unseen location j, compute mean predictive variance across patients:
         score(j) = mean_patients[ Var_mc[prediction at j] ]
  3. Select j* = argmax score(j).  Add j* to the fixed order.
  4. Resample all patients with j* observed (append to seen list).
  5. Repeat until all L locations are ordered.

The result is a single integer array of shape (L,) saved as a .npy file.
Pass it to sample_policies_mc.py via --order_file.

Usage
-----
python compute_population_order.py \
    --dataset bern \
    --n_vf 3 \
    --checkpoint /storage/homefs/mc21s017/sequential_observed_bern_3vf.pth \
    --num_mc 50 \
    --mc_chunk 50 \
    --batch_size 64 \
    --output_dir /storage/homefs/mc21s017/results/
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ablation_models_diffusion_wa import DiffusionModelLoc
from train_sequential_observed import (
    DATASETS,
    build_diffusion_schedule,
    build_model_inputs,
    choose_order,
    process_df,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling kernel (same as sample_policies_mc.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_mc_batch(
    model, cond_v, cond_times, dt_future, target,
    seen_loc_idx,
    alphas, betas, cumalphas,
    num_mc, mc_chunk,
):
    """Return (bsz, num_mc, L) float32 numpy array."""
    device      = target.device
    bsz, length = target.shape
    num_steps   = cumalphas.shape[0]
    produced    = 0
    chunks      = []

    while produced < num_mc:
        m   = min(mc_chunk, num_mc - produced)
        rep = m

        cv  = [v.repeat_interleave(rep, dim=0) for v in cond_v]
        ct  = [t.repeat_interleave(rep, dim=0) for t in cond_times]
        dtf = dt_future.repeat_interleave(rep, dim=0)
        locs = [torch.arange(length, device=device).float() for _ in cv]

        if seen_loc_idx is not None and seen_loc_idx.numel() > 0:
            cv.append(target[:, seen_loc_idx].repeat_interleave(rep, dim=0))
            ct.append(dtf)
            locs.append(seen_loc_idx.float())

        locs.append(torch.arange(length, device=device).float())
        x_t = torch.randn(bsz * rep, length, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for t_step in range(num_steps - 1, 0, -1):
                alpha_t    = alphas[t_step - 1]
                cumalpha_t = cumalphas[t_step - 1]
                sigma      = torch.sqrt(betas[t_step - 1])
                factor     = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                t_tensor   = torch.full((bsz * rep,), t_step,
                                        device=device, dtype=torch.long)
                noise_pred = model(
                    x_t=x_t, v_hist=cv, dt_hist=ct,
                    t_step=t_tensor, dt_future=dtf, locs=locs,
                )
                z   = torch.randn_like(x_t) if t_step > 1 else torch.zeros_like(x_t)
                x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - factor * noise_pred) + sigma * z

        chunks.append(x_t.view(bsz, rep, length).detach().cpu().numpy().astype(np.float32))
        produced += rep

    return np.concatenate(chunks, axis=1)   # (bsz, num_mc, L)


# ─────────────────────────────────────────────────────────────────────────────
# Population variance helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_population_variance_scores(
    model, loader, device, max_cond_eff,
    seen_prefix,          # list of int, observed location indices so far
    alphas, betas, cumalphas,
    num_mc, mc_chunk,
):
    """
    For each location j, compute mean-over-patients of Var_mc[pred_j].
    Returns array of shape (L,).
    """
    seen_tensor = (
        torch.tensor(seen_prefix, dtype=torch.long, device=device)
        if seen_prefix else None
    )

    sum_var = None
    n_total = 0

    for batch in loader:
        batch  = [b.to(device, non_blocking=True) for b in batch]
        cond_v, cond_times, _, dt_future, target = build_model_inputs(
            batch, max_cond=max_cond_eff,
        )
        bsz = target.size(0)

        samples = sample_mc_batch(
            model, cond_v, cond_times, dt_future, target,
            seen_tensor, alphas, betas, cumalphas, num_mc, mc_chunk,
        )   # (bsz, num_mc, L)

        # variance per patient per location, then sum across patients
        batch_var = np.var(samples, axis=1)   # (bsz, L)
        if sum_var is None:
            sum_var = batch_var.sum(axis=0)   # (L,)
        else:
            sum_var += batch_var.sum(axis=0)
        n_total += bsz

    return sum_var / n_total   # (L,)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute A6 population-optimised acquisition order from train/val."
    )
    parser.add_argument("--dataset",    type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--n_vf",       type=int, default=3,  choices=[3, 4])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_steps",  type=int, default=52)
    parser.add_argument("--max_cond",   type=int, default=2)
    parser.add_argument("--D",          type=int, default=64)
    parser.add_argument("--layers",     type=int, default=4)
    parser.add_argument("--num_mc",     type=int, default=50,
                        help="MC samples per patient per step (50 is usually enough).")
    parser.add_argument("--mc_chunk",   type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--data_path",  type=str,
                        default="/storage/homefs/mc21s017/data_vf/")
    parser.add_argument("--output_dir", type=str,
                        default="/storage/homefs/mc21s017/results/")
    parser.add_argument("--use_val",    action="store_true",
                        help="If set, use validation split; otherwise use training split.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg    = DATASETS[args.dataset]
    vf_key = f"{args.n_vf}vf"

    # ── load train (or val) data ───────────────────────────────────────────────
    # Train file is used by default; pass --use_val to use the test file as proxy
    # (adjust if you have an explicit val split)
    if args.use_val:
        data_file = os.path.join(args.data_path, cfg["test_files"][vf_key])
        split_name = "val(test-proxy)"
    else:
        data_file = os.path.join(args.data_path, cfg["train_files"][vf_key])
        split_name = "train"

    scaler_path = os.path.join(args.data_path, cfg["scaler_file"])
    df          = pd.read_csv(data_file)
    vf_scaler   = np.load(scaler_path, allow_pickle=True).item()
    proc        = process_df(df, vf_scaler, cfg, args.n_vf)
    dataset     = TensorDataset(*proc)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    n_patients  = len(dataset)
    order_ref   = choose_order(args.dataset, cfg)
    length      = len(order_ref)
    max_cond_eff = min(args.max_cond, args.n_vf - 1)

    print(f"Dataset:    {args.dataset} ({split_name}), {n_patients} patients")
    print(f"L={length},  num_mc={args.num_mc},  batch_size={args.batch_size}")

    # ── model ─────────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint.strip() or os.path.join(
        "/storage/homefs/mc21s017/",
        f"sequential_observed_{args.dataset}_{args.n_vf}vf.pth",
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    inferred_layers = args.layers
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sa_keys = [k for k in ckpt["model_state_dict"] if k.startswith("sa_blocks.")]
        if sa_keys:
            inferred_layers = max(int(k.split(".")[1]) for k in sa_keys) + 1
            print(f"Inferred K={inferred_layers} from checkpoint.")

    model = DiffusionModelLoc(
        L=length, D=args.D, K=inferred_layers, max_cond=max_cond_eff,
    ).to(device)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    alphas, betas, cumalphas = build_diffusion_schedule(args.num_steps, device)

    # ── greedy population order ───────────────────────────────────────────────
    pop_order   = []
    unseen_set  = set(range(length))
    t0          = time.time()

    # We store per-step variance scores for diagnostics
    var_scores_history = np.zeros((length, length), dtype=np.float32)

    print("\nStarting greedy population order computation...")
    for step in range(length):
        t_step = time.time()
        scores = compute_population_variance_scores(
            model, loader, device, max_cond_eff,
            seen_prefix=pop_order,
            alphas=alphas, betas=betas, cumalphas=cumalphas,
            num_mc=args.num_mc, mc_chunk=args.mc_chunk,
        )
        var_scores_history[step] = scores

        unseen_arr = np.array(sorted(unseen_set), dtype=np.int64)
        best       = int(unseen_arr[np.argmax(scores[unseen_arr])])
        pop_order.append(best)
        unseen_set.remove(best)

        print(
            f"Step {step + 1:3d}/{length}  selected={best:3d}"
            f"  score={scores[best]:.4f}"
            f"  ({time.time() - t_step:.1f}s / total {time.time() - t0:.1f}s)",
            flush=True,
        )

    pop_order_arr = np.array(pop_order, dtype=np.int64)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    order_path  = os.path.join(
        args.output_dir, f"pop_order_{args.dataset}_{args.n_vf}vf.npy"
    )
    scores_path = os.path.join(
        args.output_dir, f"pop_order_{args.dataset}_{args.n_vf}vf_scores.npy"
    )
    np.save(order_path,  pop_order_arr)
    np.save(scores_path, var_scores_history)

    print(f"\nPopulation order saved → {order_path}")
    print(f"Variance score history → {scores_path}")
    print(f"Order (first 20): {pop_order_arr[:20]}")
    print(f"Total time: {time.time() - t0:.1f}s")


def build_model_inputs(batch, max_cond):
    from train_sequential_observed import build_model_inputs as _bmi
    return _bmi(batch, max_cond)


if __name__ == "__main__":
    main()
