"""
merge_policy_parts.py
======================
Merge partial output files produced by SLURM array jobs into a single
pred/target/timing triple ready for compute_metrics_policies.py.

Usage
-----
python merge_policy_parts.py \
    --results_dir /storage/homefs/mc21s017/results/ \
    --policy max_variance \
    --dataset bern \
    --n_vf 3 \
    --n_parts 4 \
    --num_mc 100
"""

import argparse
import os
import glob
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--policy",      type=str, required=True)
    parser.add_argument("--dataset",     type=str, required=True)
    parser.add_argument("--n_vf",        type=int, default=3)
    parser.add_argument("--n_parts",     type=int, required=True,
                        help="Number of SLURM array tasks (parts).")
    parser.add_argument("--num_mc",      type=int, default=100)
    args = parser.parse_args()

    stem = f"policy_{args.policy}_{args.dataset}_{args.n_vf}vf"

    # Discover parts
    pred_parts   = []
    target_parts = []
    timing_parts = []

    for i in range(args.n_parts):
        p = os.path.join(args.results_dir, f"{stem}_part{i}_pred.npy")
        t = os.path.join(args.results_dir, f"{stem}_part{i}_target.npy")
        ti= os.path.join(args.results_dir, f"{stem}_part{i}_timing.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing part {i}: {p}")
        pred_parts.append(p)
        target_parts.append(t)
        timing_parts.append(ti if os.path.exists(ti) else None)

    # Read first part to infer shape
    p0        = np.load(pred_parts[0], mmap_mode="r")
    _, n_prefix, num_mc, L = p0.shape
    n_total   = sum(np.load(p, mmap_mode="r").shape[0] for p in pred_parts)

    print(f"Merging {args.n_parts} parts → N={n_total}, n_prefix={n_prefix}, L={L}")

    out_pred   = os.path.join(args.results_dir, f"{stem}_offset0_mc{args.num_mc}_pred.npy")
    out_target = os.path.join(args.results_dir, f"{stem}_offset0_mc{args.num_mc}_target.npy")
    out_timing = os.path.join(args.results_dir, f"{stem}_offset0_mc{args.num_mc}_timing.npy")

    merged_pred   = np.lib.format.open_memmap(
        out_pred,   mode="w+", dtype=np.float32, shape=(n_total, n_prefix, num_mc, L))
    merged_target = np.lib.format.open_memmap(
        out_target, mode="w+", dtype=np.float32, shape=(n_total, L))
    has_timing = all(t is not None for t in timing_parts)
    if has_timing:
        merged_timing = np.lib.format.open_memmap(
            out_timing, mode="w+", dtype=np.float32, shape=(n_total, n_prefix))

    offset = 0
    for i, (pp, tp, tip) in enumerate(zip(pred_parts, target_parts, timing_parts)):
        pred_i   = np.load(pp,  mmap_mode="r")
        target_i = np.load(tp)
        n_i      = pred_i.shape[0]
        merged_pred[offset:offset + n_i]   = pred_i
        merged_target[offset:offset + n_i] = target_i
        if has_timing and tip:
            timing_i = np.load(tip)
            merged_timing[offset:offset + n_i] = timing_i
        offset += n_i
        print(f"  Part {i}: {n_i} samples (cumulative {offset}/{n_total})")

    # Write a minimal meta file so compute_metrics_policies.py can find the paths
    meta_path = os.path.join(
        args.results_dir,
        f"{stem}_offset0_mc{args.num_mc}_meta.npz",
    )
    np.savez_compressed(
        meta_path,
        policy=np.asarray(args.policy),
        dataset=np.asarray(args.dataset),
        n_vf=np.int64(args.n_vf),
        num_mc=np.int64(args.num_mc),
        length=np.int64(L),
        n_test=np.int64(n_total),
        n_prefix=np.int64(n_prefix),
        prefix_ks=np.arange(n_prefix, dtype=np.int64),
        pred_path=np.asarray(out_pred),
        target_path=np.asarray(out_target),
    )

    print(f"\nMerged pred   → {out_pred}")
    print(f"Merged target → {out_target}")
    if has_timing:
        print(f"Merged timing → {out_timing}")
    print(f"Meta          → {meta_path}")


if __name__ == "__main__":
    main()
