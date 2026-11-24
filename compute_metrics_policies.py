"""
compute_metrics_policies.py
============================
Compute reconstruction metrics from policy sampling outputs.

Metrics
-------
  RMSE   – root mean squared error
  MAE    – mean absolute error
  MD     – mean deviation (mean of reconstructed field, clinical VF summary)
  sLV    – square-root loss variance (sLV = sqrt( Var(pointwise errors) ))
           Following Heijl et al.: sLV ≈ std of per-location TD errors

All metrics are reported:
  • Per k (number of observed locations, k = 0 … L)       → curves
  • Overall (mean over test set)
  • Per severity stratum at every k                        → strata curves
  • Pointwise (per VF location, mean over test patients)   → heatmap inputs

Severity strata (based on target MD, dB scale after inverse transform)
  normal/near-normal   MD > -6
  early                -12 < MD ≤ -6
  moderate             -20 < MD ≤ -12
  advanced             -28 < MD ≤ -20
  severe               MD ≤ -28

Usage
-----
# single policy output
python compute_metrics_policies.py \
    --pred_file  /storage/homefs/mc21s017/results/policy_max_variance_bern_3vf_pred.npy \
    --target_file /storage/homefs/mc21s017/results/policy_max_variance_bern_3vf_target.npy \
    --dataset bern \
    --output_dir /storage/homefs/mc21s017/results/metrics/

# or point to the meta file and let it resolve paths automatically
python compute_metrics_policies.py \
    --meta_file /storage/homefs/mc21s017/results/policy_max_variance_bern_3vf_offset0_mc100_meta.npz \
    --dataset bern \
    --output_dir /storage/homefs/mc21s017/results/metrics/
"""

import argparse
import os

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Inverse scaler helper
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform(arr, scaler):
    """
    Undo StandardScaler normalisation.
    arr: (..., L) numpy array in normalised space.
    Returns array of same shape in original dB space.
    """
    mean = scaler.mean_        # (L,)
    std  = np.sqrt(scaler.var_)  # (L,)
    return arr * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Clinical summary metrics on a VF field (in dB)
# ─────────────────────────────────────────────────────────────────────────────

def field_md(field_db):
    """MD = mean of all TD values.  field_db: (..., L)."""
    return np.mean(field_db, axis=-1)

def field_slv(field_db):
    """
    sLV ≈ std of pointwise TD values across locations.
    field_db: (..., L)  → returns (...,)
    Following convention: sLV = sqrt( mean((TD_i - MD)^2 ) ).
    """
    return np.std(field_db, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-k metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_per_k(
    pred_mean,   # (N, K+1, L)  MC mean, normalised space
    target,      # (N, L)       ground truth, normalised space
    scaler,      # sklearn scaler for inverse transform
):
    """
    Returns dict with keys: rmse, mae, md_mae, slv_mae
    Each value is a 1-D array of length K+1 (one entry per prefix level).
    """
    N, n_prefix, L = pred_mean.shape

    rmse   = np.zeros(n_prefix)
    mae    = np.zeros(n_prefix)
    md_mae = np.zeros(n_prefix)
    slv_mae= np.zeros(n_prefix)

    target_db  = inverse_transform(target, scaler)         # (N, L)
    target_md  = field_md(target_db)                       # (N,)
    target_slv = field_slv(target_db)                      # (N,)

    for k in range(n_prefix):
        p_db   = inverse_transform(pred_mean[:, k, :], scaler)  # (N, L)
        err    = p_db - target_db                                # (N, L)

        rmse[k]    = np.sqrt(np.mean(err ** 2))
        mae[k]     = np.mean(np.abs(err))

        pred_md    = field_md(p_db)                              # (N,)
        md_mae[k]  = np.mean(np.abs(pred_md - target_md))

        pred_slv   = field_slv(p_db)                             # (N,)
        slv_mae[k] = np.mean(np.abs(pred_slv - target_slv))

    return dict(rmse=rmse, mae=mae, md_mae=md_mae, slv_mae=slv_mae)


# ─────────────────────────────────────────────────────────────────────────────
# Severity stratification
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_BINS = {
    "normal":   (  -6.0,   np.inf),
    "early":    ( -12.0,   -6.0),
    "moderate": ( -20.0,  -12.0),
    "advanced": ( -28.0,  -20.0),
    "severe":   (-np.inf, -28.0),
}


def severity_mask(target_db, scaler=None):
    """
    Returns dict {stratum_name: boolean array (N,)}.
    target_db: (N, L) in dB space.
    """
    md = field_md(target_db)   # (N,)
    masks = {}
    for name, (lo, hi) in SEVERITY_BINS.items():
        masks[name] = (md > lo) & (md <= hi)
    return masks


def compute_metrics_per_k_per_severity(pred_mean, target, scaler):
    """
    Returns dict {stratum: {metric: array(n_prefix)}}.
    """
    target_db = inverse_transform(target, scaler)
    masks     = severity_mask(target_db)
    results   = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            results[name] = None
            continue
        results[name] = compute_metrics_per_k(
            pred_mean[mask], target[mask], scaler,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise metrics (mean over patients, for each VF location)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pointwise_metrics(pred_mean, target, scaler, ks=None):
    """
    Returns dict {k_label: array(L)} for each requested k value.
    Default ks: [0, 10, 20, 30, len-1]
    """
    N, n_prefix, L = pred_mean.shape
    if ks is None:
        ks = sorted(set([0, 10, 20, 30, n_prefix - 1]))
        ks = [k for k in ks if k < n_prefix]

    target_db = inverse_transform(target, scaler)   # (N, L)
    out = {}
    for k in ks:
        p_db      = inverse_transform(pred_mean[:, k, :], scaler)  # (N, L)
        mae_pw    = np.mean(np.abs(p_db - target_db), axis=0)       # (L,)
        out[f"k{k:03d}_mae"] = mae_pw
        rmse_pw   = np.sqrt(np.mean((p_db - target_db) ** 2, axis=0))
        out[f"k{k:03d}_rmse"] = rmse_pw
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Timing / compute cost summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_timing_summary(timing):
    """
    timing: (N, n_prefix) wall-clock seconds for sampling call at each step.
    Returns dict with mean/median/total per step.
    """
    return dict(
        mean_per_step=timing.mean(axis=0),     # (n_prefix,)
        median_per_step=np.median(timing, axis=0),
        total_per_sample=timing.sum(axis=1).mean(),  # scalar
        per_step_cumulative=np.cumsum(timing.mean(axis=0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute reconstruction metrics for a policy sampling output."
    )
    parser.add_argument("--meta_file",   type=str, default="",
                        help="Path to _meta.npz from sample_policies_mc.py "
                             "(auto-resolves pred/target paths).")
    parser.add_argument("--pred_file",   type=str, default="")
    parser.add_argument("--target_file", type=str, default="")
    parser.add_argument("--timing_file", type=str, default="",
                        help="Optional _timing.npy for compute-cost metrics.")
    parser.add_argument("--dataset",   type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--n_vf",      type=int, default=3,  choices=[3, 4])
    parser.add_argument("--data_path", type=str,
                        default="/storage/homefs/mc21s017/data_vf/")
    parser.add_argument("--output_dir",  type=str,
                        default="/storage/homefs/mc21s017/results/metrics/")
    parser.add_argument("--output_name", type=str, default="",
                        help="Stem for output files; inferred from meta if blank.")
    parser.add_argument("--pointwise_ks", type=int, nargs="*", default=None,
                        help="Specific k values for pointwise metrics. "
                             "Default: 0,10,20,30,L-1.")
    args = parser.parse_args()

    # ── resolve paths ─────────────────────────────────────────────────────────
    if args.meta_file:
        meta      = np.load(args.meta_file, allow_pickle=True)
        pred_path = str(meta["pred_path"])
        tgt_path  = str(meta["target_path"])
        policy    = str(meta.get("policy", "unknown"))
        # timing lives beside the meta
        timing_path = args.meta_file.replace("_meta.npz", "_timing.npy")
        if not args.output_name:
            args.output_name = os.path.basename(args.meta_file).replace("_meta.npz", "")
    else:
        if not args.pred_file or not args.target_file:
            raise ValueError("Provide either --meta_file or both --pred_file and --target_file.")
        pred_path   = args.pred_file
        tgt_path    = args.target_file
        timing_path = args.timing_file
        policy      = "unknown"
        if not args.output_name:
            args.output_name = os.path.splitext(os.path.basename(pred_path))[0]

    # ── load scaler ───────────────────────────────────────────────────────────
    from train_sequential_observed import DATASETS
    cfg         = DATASETS[args.dataset]
    scaler_path = os.path.join(args.data_path, cfg["scaler_file"])
    vf_scaler   = np.load(scaler_path, allow_pickle=True).item()

    # ── load arrays ───────────────────────────────────────────────────────────
    print(f"Loading pred  → {pred_path}")
    pred_full  = np.load(pred_path, mmap_mode="r")   # (N, n_prefix, num_mc, L)
    print(f"Loading target→ {tgt_path}")
    target     = np.load(tgt_path)                    # (N, L)

    N, n_prefix, num_mc, L = pred_full.shape
    print(f"N={N}, n_prefix={n_prefix}, num_mc={num_mc}, L={L}")

    # MC mean as point estimate
    print("Computing MC mean...")
    pred_mean = pred_full.mean(axis=2)   # (N, n_prefix, L)

    # ── overall per-k metrics ─────────────────────────────────────────────────
    print("Computing per-k metrics...")
    metrics = compute_metrics_per_k(pred_mean, target, vf_scaler)

    # ── per-severity per-k metrics ────────────────────────────────────────────
    print("Computing per-severity metrics...")
    sev_metrics = compute_metrics_per_k_per_severity(pred_mean, target, vf_scaler)

    # ── pointwise metrics ─────────────────────────────────────────────────────
    print("Computing pointwise metrics...")
    pw_ks    = args.pointwise_ks if args.pointwise_ks else None
    pw_metrics = compute_pointwise_metrics(pred_mean, target, vf_scaler, ks=pw_ks)

    # ── timing ────────────────────────────────────────────────────────────────
    timing_summary = None
    if timing_path and os.path.exists(timing_path):
        timing = np.load(timing_path)   # (N, n_prefix)
        timing_summary = compute_timing_summary(timing)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}_metrics.npz")

    save_dict = dict(
        policy=np.asarray(policy),
        dataset=np.asarray(args.dataset),
        n_vf=np.int64(args.n_vf),
        n_test=np.int64(N),
        n_prefix=np.int64(n_prefix),
        L=np.int64(L),
        num_mc=np.int64(num_mc),
        prefix_ks=np.arange(n_prefix, dtype=np.int64),
        # overall
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        md_mae=metrics["md_mae"],
        slv_mae=metrics["slv_mae"],
    )

    # severity strata
    for sname, smet in sev_metrics.items():
        if smet is None:
            continue
        for mname, arr in smet.items():
            save_dict[f"sev_{sname}_{mname}"] = arr

    # pointwise
    for key, arr in pw_metrics.items():
        save_dict[f"pw_{key}"] = arr

    # timing
    if timing_summary:
        for key, val in timing_summary.items():
            save_dict[f"timing_{key}"] = val

    np.savez_compressed(out_path, **save_dict)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Policy: {policy}  |  Dataset: {args.dataset}  |  N={N}")
    print(f"{'k':>4}  {'RMSE':>8}  {'MAE':>8}  {'MD-MAE':>8}  {'sLV-MAE':>8}")
    report_ks = [0, 5, 10, 15, 20, 25, 30, n_prefix - 1]
    for k in report_ks:
        if k >= n_prefix:
            continue
        print(
            f"{k:4d}  {metrics['rmse'][k]:8.3f}  {metrics['mae'][k]:8.3f}"
            f"  {metrics['md_mae'][k]:8.3f}  {metrics['slv_mae'][k]:8.3f}"
        )
    print(f"\nMetrics saved → {out_path}")


if __name__ == "__main__":
    main()
