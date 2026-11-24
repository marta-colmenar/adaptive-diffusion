"""
sample_policies_mc.py
=====================
Unified Monte Carlo sampler for all acquisition policies.

Policies
--------
  A0   random       – uniform random location selection
  A1   max_variance – per-patient greedy: location with max marginal variance
  A2   max_entropy  – per-patient greedy: location with max differential entropy
                      (Gaussian approx; argmax identical to A1, kept separate for
                      future kernel-density implementation)
  A5   growing      – fixed order from choose_order() (clinical spiral)
  A6   pop_order    – fixed order loaded from --order_file (A6 population-
                      optimised sequence, computed by compute_population_order.py)
  mse_oracle        – per-patient oracle: max |mean_pred - true_target|
                      (upper bound; requires ground truth at test time)

All policies use U3: full conditional re-sampling with the trained partial-
observation diffusion model at every new measurement.

Usage
-----
# A0 – random
python sample_policies_mc.py --dataset bern --policy random ...

# A1 – max variance (adaptive, per-patient)
python sample_policies_mc.py --dataset bern --policy max_variance ...

# A5 – growing (fixed clinical order)
python sample_policies_mc.py --dataset bern --policy growing ...

# A6 – population-optimised fixed order
python sample_policies_mc.py --dataset bern --policy pop_order \
    --order_file /storage/homefs/mc21s017/results/pop_order_bern_3vf.npy ...

# MSE oracle
python sample_policies_mc.py --dataset bern --policy mse_oracle ...
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from ablation_models_diffusion_wa import DiffusionModelLoc
from train_sequential_observed import (
    DATASETS,
    build_diffusion_schedule,
    build_model_inputs,
    choose_order,
    process_df,
)

# ─────────────────────────────────────────────────────────────────────────────
# Core sampling kernel  (identical to original sample_sequential_ordered_prefix_mc.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_mc_for_single_prefix(
    model, cond_v, cond_times, dt_future, target,
    seen_loc_idx,           # 1-D LongTensor of observed location indices, or None
    alphas, betas, cumalphas,
    num_mc, mc_chunk,
):
    """Return samples array of shape (bsz, num_mc, L)."""
    device     = target.device
    bsz, length = target.shape
    num_steps  = cumalphas.shape[0]
    produced   = 0
    chunks     = []

    while produced < num_mc:
        m   = min(mc_chunk, num_mc - produced)
        rep = m

        cond_v_rep     = [v.repeat_interleave(rep, dim=0) for v in cond_v]
        cond_times_rep = [t.repeat_interleave(rep, dim=0) for t in cond_times]
        dt_future_rep  = dt_future.repeat_interleave(rep, dim=0)
        locs = [torch.arange(length, device=device).float()
                for _ in range(len(cond_v_rep))]

        if seen_loc_idx is not None and seen_loc_idx.numel() > 0:
            target_seen = target[:, seen_loc_idx].repeat_interleave(rep, dim=0)
            cond_v_rep.append(target_seen)
            cond_times_rep.append(dt_future_rep)
            locs.append(seen_loc_idx.float())

        locs.append(torch.arange(length, device=device).float())
        x_t = torch.randn(bsz * rep, length, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for t_step in range(num_steps - 1, 0, -1):
                alpha_t    = alphas[t_step - 1]
                cumalpha_t = cumalphas[t_step - 1]
                sigma      = torch.sqrt(betas[t_step - 1])
                factor     = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)

                t_tensor = torch.full((bsz * rep,), t_step,
                                      device=device, dtype=torch.long)
                noise_pred = model(
                    x_t=x_t, v_hist=cond_v_rep, dt_hist=cond_times_rep,
                    t_step=t_tensor, dt_future=dt_future_rep, locs=locs,
                )
                z   = torch.randn_like(x_t) if t_step > 1 else torch.zeros_like(x_t)
                x_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - factor * noise_pred) + sigma * z

        chunks.append(x_t.view(bsz, rep, length).detach().cpu().numpy().astype(np.float32))
        produced += rep

    return np.concatenate(chunks, axis=1)   # (bsz, num_mc, L)


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition functions  (greedy, per-patient)
# ─────────────────────────────────────────────────────────────────────────────

def acq_random(samples_k, unseen, target_np, rng):
    """A0 – uniform random."""
    return rng.choice(unseen)

def acq_max_variance(samples_k, unseen, target_np, rng):
    """A1 – location with greatest marginal predictive variance."""
    var = np.var(samples_k, axis=0)          # (L,)
    return unseen[np.argmax(var[unseen])]

def acq_max_entropy(samples_k, unseen, target_np, rng):
    """A2 – greatest differential entropy (Gaussian approx).
    H ∝ log σ² → argmax same as A1; kept as a separate policy for
    potential future replacement with a non-parametric estimator.
    """
    var = np.var(samples_k, axis=0) + 1e-12
    return unseen[np.argmax(var[unseen])]

def acq_mse_oracle(samples_k, unseen, target_np, rng):
    """MSE oracle – max |mean_pred − target| (upper bound, needs true labels)."""
    err = np.abs(np.mean(samples_k, axis=0) - target_np)   # (L,)
    return unseen[np.argmax(err[unseen])]

GREEDY_POLICIES = {
    "random":      acq_random,
    "max_variance": acq_max_variance,
    "max_entropy":  acq_max_entropy,
    "mse_oracle":   acq_mse_oracle,
}
FIXED_POLICIES = {"growing", "pop_order"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample under a specified acquisition policy (U3 diffusion)."
    )
    # --- data / model ---
    parser.add_argument("--dataset",    type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--n_vf",       type=int, default=3,  choices=[3, 4])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_steps",  type=int, default=52)
    parser.add_argument("--max_cond",   type=int, default=2)
    parser.add_argument("--D",          type=int, default=64)
    parser.add_argument("--layers",     type=int, default=4)
    parser.add_argument("--data_path",  type=str,
                        default="/storage/homefs/mc21s017/data_vf/")
    # --- policy ---
    parser.add_argument(
        "--policy", type=str, required=True,
        choices=list(GREEDY_POLICIES.keys()) + list(FIXED_POLICIES),
        help=(
            "random       A0  – uniform random\n"
            "max_variance A1  – per-patient max marginal variance\n"
            "max_entropy  A2  – per-patient max differential entropy\n"
            "growing      A5  – fixed clinical order (choose_order)\n"
            "pop_order    A6  – fixed population-optimised order (--order_file)\n"
            "mse_oracle       – per-patient oracle upper bound\n"
        ),
    )
    parser.add_argument("--order_file", type=str, default="",
                        help="Path to .npy order array for pop_order (A6) policy.")
    # --- sampling ---
    parser.add_argument("--num_mc",           type=int, default=100)
    parser.add_argument("--mc_chunk",         type=int, default=10)
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--num_workers",      type=int, default=2)
    parser.add_argument("--max_test_samples", type=int, default=0,
                        help="0 = full test set")
    parser.add_argument("--sample_offset",    type=int, default=0)
    parser.add_argument("--seed",             type=int, default=0)
    # --- output ---
    parser.add_argument("--output_dir",  type=str,
                        default="/storage/homefs/mc21s017/results/")
    parser.add_argument("--output_name", type=str, default="")

    args = parser.parse_args()

    # ── sanity checks ─────────────────────────────────────────────────────────
    if args.mc_chunk <= 0:
        raise ValueError("--mc_chunk must be > 0")
    if args.policy == "pop_order" and not args.order_file:
        raise ValueError("--policy pop_order requires --order_file")

    # fixed policies can use batched inference; greedy must be per-sample
    is_greedy = args.policy in GREEDY_POLICIES
    batch_size = args.batch_size
    if is_greedy and batch_size != 1:
        print(f"[INFO] Policy '{args.policy}' is per-patient. Forcing batch_size=1.")
        batch_size = 1

    # ── reproducibility ───────────────────────────────────────────────────────
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg    = DATASETS[args.dataset]
    vf_key = f"{args.n_vf}vf"

    # ── data ──────────────────────────────────────────────────────────────────
    test_file   = os.path.join(args.data_path, cfg["test_files"][vf_key])
    scaler_path = os.path.join(args.data_path, cfg["scaler_file"])
    df_test     = pd.read_csv(test_file)
    vf_scaler   = np.load(scaler_path, allow_pickle=True).item()

    proc_test    = process_df(df_test, vf_scaler, cfg, args.n_vf)
    test_ds_full = TensorDataset(*proc_test)

    if args.max_test_samples > 0:
        end_idx = min(args.sample_offset + args.max_test_samples, len(test_ds_full))
        test_ds = Subset(test_ds_full, list(range(args.sample_offset, end_idx)))
    else:
        test_ds = test_ds_full

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── fixed order ───────────────────────────────────────────────────────────
    clinical_order = choose_order(args.dataset, cfg)   # A5 / fallback
    length         = len(clinical_order)
    max_cond_eff   = min(args.max_cond, args.n_vf - 1)

    if args.policy == "growing":
        fixed_order = clinical_order
    elif args.policy == "pop_order":
        fixed_order = np.load(args.order_file).tolist()
        assert len(fixed_order) == length, (
            f"pop_order length {len(fixed_order)} != VF length {length}"
        )
    else:
        fixed_order = None   # greedy policies don't use a fixed order

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
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    alphas, betas, cumalphas = build_diffusion_schedule(args.num_steps, device)

    # ── output arrays ─────────────────────────────────────────────────────────
    n_test   = len(test_ds)
    n_prefix = length + 1   # k = 0 … L

    if not args.output_name:
        args.output_name = (
            f"policy_{args.policy}_{args.dataset}_{args.n_vf}vf"
            f"_offset{args.sample_offset}_mc{args.num_mc}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path   = os.path.join(args.output_dir, f"{args.output_name}_pred.npy")
    target_path = os.path.join(args.output_dir, f"{args.output_name}_target.npy")
    timing_path = os.path.join(args.output_dir, f"{args.output_name}_timing.npy")
    meta_path   = os.path.join(args.output_dir, f"{args.output_name}_meta.npz")

    # shape: (n_test, n_prefix, num_mc, L)
    pred_mem = np.lib.format.open_memmap(
        pred_path,  mode="w+", dtype=np.float32,
        shape=(n_test, n_prefix, args.num_mc, length),
    )
    target_mem = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=(n_test, length),
    )
    # wall-clock seconds per (sample, prefix-step) for the sampling call alone
    timing_mem = np.lib.format.open_memmap(
        timing_path, mode="w+", dtype=np.float32, shape=(n_test, n_prefix),
    )
    # the order used for each test sample (L,) – same for all if fixed policy
    all_orders = np.empty((n_test, length), dtype=np.int64)

    print(f"Policy:        {args.policy}")
    print(f"Device:        {device}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Test samples:  {n_test}  (offset {args.sample_offset})")
    print(f"L={length}, n_prefix={n_prefix}, num_mc={args.num_mc}")
    print(f"pred  → {pred_path}")
    print(f"target→ {target_path}")

    t0     = time.time()
    offset = 0

    # ═══════════════════════════════════════════════════════════════════════════
    # FIXED POLICIES  (A5 growing, A6 pop_order) – batch_size can be > 1
    # ═══════════════════════════════════════════════════════════════════════════
    if not is_greedy:
        order_tensor = torch.tensor(fixed_order, dtype=torch.long, device=device)

        for batch_idx, batch in enumerate(test_loader):
            batch = [b.to(device, non_blocking=True) for b in batch]
            cond_v, cond_times, _, dt_future, target = build_model_inputs(
                batch, max_cond=max_cond_eff,
            )
            bsz = target.size(0)
            target_mem[offset:offset + bsz] = (
                target.detach().cpu().numpy().astype(np.float32)
            )
            for gi in range(bsz):
                all_orders[offset + gi] = np.array(fixed_order, dtype=np.int64)

            for k in range(n_prefix):
                seen_loc_idx = order_tensor[:k] if k > 0 else None
                t_k  = time.time()
                pred = sample_mc_for_single_prefix(
                    model, cond_v, cond_times, dt_future, target,
                    seen_loc_idx, alphas, betas, cumalphas,
                    args.num_mc, args.mc_chunk,
                )
                dt_k = time.time() - t_k
                pred_mem[offset:offset + bsz, k]    = pred
                timing_mem[offset:offset + bsz, k]  = dt_k / bsz

            offset += bsz
            print(
                f"Batch {batch_idx + 1:4d} | processed={offset}/{n_test}"
                f" | elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # GREEDY POLICIES  (A0 random, A1 max_variance, A2 max_entropy, mse_oracle)
    # batch_size forced to 1 above
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        acq_fn = GREEDY_POLICIES[args.policy]

        for batch_idx, batch in enumerate(test_loader):
            batch = [b.to(device, non_blocking=True) for b in batch]
            cond_v, cond_times, _, dt_future, target = build_model_inputs(
                batch, max_cond=max_cond_eff,
            )
            # bsz == 1 always
            target_np = target.detach().cpu().numpy().astype(np.float32)   # (1, L)
            target_mem[offset] = target_np[0]

            t_total = time.time()

            seen_prefix = []
            unseen_set  = set(range(length))

            # ── k = 0 : no observed locations ─────────────────────────────────
            t_k   = time.time()
            pred0 = sample_mc_for_single_prefix(
                model, cond_v, cond_times, dt_future, target,
                None, alphas, betas, cumalphas, args.num_mc, args.mc_chunk,
            )
            timing_mem[offset, 0] = time.time() - t_k
            pred_mem[offset, 0]   = pred0[0]

            # ── k = 1 … L : greedy selection ──────────────────────────────────
            for k in range(1, length + 1):
                samples_prev = pred_mem[offset, k - 1]           # (num_mc, L)
                unseen_arr   = np.array(sorted(unseen_set), dtype=np.int64)

                next_loc = acq_fn(
                    samples_prev, unseen_arr, target_np[0], rng,
                )
                seen_prefix.append(int(next_loc))
                unseen_set.remove(int(next_loc))

                seen_tensor = torch.tensor(
                    seen_prefix, dtype=torch.long, device=device,
                )
                t_k   = time.time()
                pred_k = sample_mc_for_single_prefix(
                    model, cond_v, cond_times, dt_future, target,
                    seen_tensor, alphas, betas, cumalphas,
                    args.num_mc, args.mc_chunk,
                )
                timing_mem[offset, k] = time.time() - t_k
                pred_mem[offset, k]   = pred_k[0]

            all_orders[offset] = np.array(seen_prefix, dtype=np.int64)

            print(
                f"Sample {offset + 1:4d}/{n_test}"
                f" | {time.time() - t_total:.1f}s"
                f" | elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
            offset += 1

    # ── metadata ──────────────────────────────────────────────────────────────
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
        policy=np.asarray(args.policy),
        fixed_order=np.asarray(fixed_order if fixed_order else clinical_order,
                               dtype=np.int64),
        all_orders=all_orders,
        pred_path=np.asarray(pred_path),
        target_path=np.asarray(target_path),
        checkpoint=np.asarray(ckpt_path),
        sample_offset=np.int64(args.sample_offset),
    )

    print(f"\nDone. Total: {time.time() - t0:.1f}s")
    print(f"Metadata → {meta_path}")
    print(f"Timing   → {timing_path}")


def build_model_inputs(batch, max_cond):
    """Re-export from train_sequential_observed (avoid circular import)."""
    from train_sequential_observed import build_model_inputs as _bmi
    return _bmi(batch, max_cond)


if __name__ == "__main__":
    main()
