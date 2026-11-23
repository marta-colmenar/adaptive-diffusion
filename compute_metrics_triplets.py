"""
Compute metrics for diffusion samples generated from triplet parquet data.

This script scans a samples directory for files produced by
`generate_triplets_diffusion_samples.py`, computes FID, MSE, and MAE for each
conditioning variant, and writes a CSV summary.

The evaluation is performed directly on the saved `.npy` files, so no scaler or
CSV test-set loading is required.
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import sqrtm
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "scipy is required for FID computation in compute_metrics_triplets.py"
    ) from exc


DEFAULT_SAMPLES_DIR = "/storage/homefs/mc21s017/samples_triplets/"
DEFAULT_OUTPUT_DIR = "/storage/homefs/mc21s017/metrics_triplets/"
SCALER = "/storage/homefs/mc21s017/data_vf/scaler_bern_3vf_new.npy"


stages = np.load("/storage/homefs/mc21s017/data_vf/class_2.npy", allow_pickle=True)

@dataclass
class TripletRun:
    variant: str
    pred_file: str
    real_file: str


def calculate_fid(features_real: np.ndarray, features_generated: np.ndarray) -> float:
    """Compute Fréchet Inception Distance from feature matrices."""
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_generated, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_generated, rowvar=False)

    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    cov_sqrt, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    cov_sqrt = np.real(cov_sqrt)

    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return float(np.real(fid))


def discover_runs(samples_dir: str):
    """Discover triplet sample runs from the saved file naming convention."""
    pattern = re.compile(
        r"^(?P<variant>.*?)_pred\.npy$"
    )
    runs = []

    if not os.path.isdir(samples_dir):
        return runs

    for name in sorted(os.listdir(samples_dir)):
        match = pattern.match(name)
        if match is None:
            continue
        pred_file = os.path.join(samples_dir, name)
        real_file = pred_file.replace("_pred.npy", "_real.npy")
        if not os.path.exists(real_file):
            continue
        runs.append(
            TripletRun(
                variant=match.group("variant"),
                pred_file=pred_file,
                real_file=real_file,
            )
        )

    return runs


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray):
    """Compute FID, MSE, MAE from saved prediction arrays."""
    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions with shape (n_test, n_samples, L), got {predictions.shape}")
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected ground_truth with shape (n_test, L), got {ground_truth.shape}")
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError(
            f"Mismatched patient counts: predictions {predictions.shape[0]} vs ground_truth {ground_truth.shape[0]}"
        )
    if predictions.shape[2] != ground_truth.shape[1]:
        raise ValueError(
            f"Mismatched VF lengths: predictions {predictions.shape[2]} vs ground_truth {ground_truth.shape[1]}"
        )

    pred_mean = predictions.mean(axis=1)
    n_patients, n_samples, length = predictions.shape

    pred_flat = predictions.reshape(-1, length)

    fid_samples = calculate_fid(ground_truth, pred_flat)
    fid_mean = calculate_fid(ground_truth, pred_mean)

    mse = float(np.mean((ground_truth - pred_mean) ** 2))
    mae = float(np.mean(np.abs(ground_truth - pred_mean)))
    mse_std = float(np.std(np.mean((ground_truth - pred_mean) ** 2, axis=1)))
    mae_std = float(np.std(np.mean(np.abs(ground_truth - pred_mean), axis=1)))

    return {
        "n_patients": int(n_patients),
        "n_samples": int(n_samples),
        "length": int(length),
        "fid_samples": fid_samples,
        "fid_mean": fid_mean,
        "mse": mse,
        "mae": mae,
        "mse_std": mse_std,
        "mae_std": mae_std,
    }



def main():
    parser = argparse.ArgumentParser(description="Compute metrics for triplet diffusion samples")
    parser.add_argument("--samples_dir", type=str, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        help="Variant to evaluate, or 'all' to process every discovered triplet output",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = discover_runs(args.samples_dir)
    if args.variant != "all":
        runs = [run for run in runs if run.variant == args.variant]

    if not runs:
        raise FileNotFoundError(
            f"No triplet sample runs found in {args.samples_dir}. Expected files like "
            f"triplets_oct_vf_diffusion_bern_3vf_<variant>_pred.npy"
        )

    scaler = np.load(SCALER, allow_pickle=True).item()
    summary_rows = []
    for run in runs:
        predictions = np.load(run.pred_file, mmap_mode="r")
        ground_truth = np.load(run.real_file, mmap_mode="r")
        predictions = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
        ground_truth = scaler.inverse_transform(ground_truth)
        metrics = compute_metrics(predictions, ground_truth)

        row = {
            "variant": run.variant,
            "pred_file": os.path.basename(run.pred_file),
            "real_file": os.path.basename(run.real_file),
            **metrics,
        }
        summary_rows.append(row)

        print(
            f"{run.variant}: MAE={metrics['mae']:.2f} ± {metrics['mae_std']:.2f}"
        )

        for stage in range(5):
            stage_mask = stages == stage
            if not np.any(stage_mask):
                continue
            stage_metrics = compute_metrics(predictions[stage_mask], ground_truth[stage_mask])
            row = {
                "variant": f"{run.variant}_stage{stage}",
                "pred_file": os.path.basename(run.pred_file),
                "real_file": os.path.basename(run.real_file),
                **stage_metrics,
            }
            summary_rows.append(row)

            print(
                f"  Stage {stage}: MAE={stage_metrics['mae']:.2f} ± {stage_metrics['mae_std']:.2f}"
            )

    csv_path = os.path.join(args.output_dir, "triplets_metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()