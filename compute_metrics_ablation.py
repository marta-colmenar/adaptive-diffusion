"""
Compute Metrics for Ablation Study Evaluation
----------------------------------------------
Loads saved samples and computes FID, MSE, MAE for each model variant
and dataset. Creates comparison tables and visualizations.
"""

import os, re, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ablation_models_diffusion_wa import MODEL_REGISTRY
from model_diffusion_wa import calculate_fid

# Suppress sklearn version warning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Compute metrics from saved samples')
parser.add_argument('--samples_dir', type=str, default='/storage/homefs/mc21s017/samples/',
                    help='Directory containing saved samples')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/metrics/',
                    help='Output directory for results')
args = parser.parse_args()

SAMPLES_DIR = args.samples_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Models and Datasets
# -------------------------------------------------------------------------
MODELS = list(MODEL_REGISTRY.keys())
DATASETS = ['wa', 'bern']

# -------------------------------------------------------------------------
# Disease stage classification functions
# -------------------------------------------------------------------------
def classify_stage_wa(md):
    """Classify Washington dataset by mean deviation (MD) stage."""
    if md >= 0:
        return 0
    elif md < 0 and md >= -5:
        return 1
    elif md < -5 and md >= -12:
        return 2
    elif md < -12 and md >= -20:
        return 3
    elif md < -20:
        return 4

def classify_stage_bern(md):
    """Classify Bern dataset by mean deviation (MD) stage."""
    if md <= -0.8:
        return 0
    elif md > -0.8 and md <= 4.4:
        return 1
    elif md > 4.4 and md <= 9.5:
        return 2
    elif md > 9.5 and md <= 15.3:
        return 3
    elif md > 15.3 and md <= 23.1:
        return 4
    else:
        return 4

# Dataset registry (same as in training/sampling scripts)
DATASETS_INFO = {
    "wa": {
        "test_file": "df_test_wa_3vf_normal.csv",
        "scaler_file": "scaler_wa_3vf.npy",
        "data_path": "/storage/homefs/mc21s017/data_vf/",
        "md_col":"MD_1",
        "classify_fn": classify_stage_wa,
    },
    "bern": {
        "test_file": "df_test_bern_3vf_normal_new.csv",
        "scaler_file": "scaler_bern_3vf_new.npy",
        "data_path": "/storage/homefs/mc21s017/data_vf/",
        "md_col": "MD_1",
        "classify_fn": classify_stage_bern,
    },
}

MODEL_DATASET_INFO = {
    "wa": DATASETS_INFO["wa"],
    "bern": DATASETS_INFO["bern"],
    "joint_orig": DATASETS_INFO["bern"],
    "merged_G": DATASETS_INFO["bern"],
    "merged_G_clipm30": DATASETS_INFO["bern"],
    "merged_24-2": DATASETS_INFO["wa"],
    "merged_242": DATASETS_INFO["wa"],
    "merged_24-2_clipm30": DATASETS_INFO["wa"],
    "merged_242_clipm30": DATASETS_INFO["wa"],
}

# -------------------------------------------------------------------------
# Metric computation functions
# -------------------------------------------------------------------------
def compute_metrics_with_fid_variants(predictions, ground_truth, scaler=None):
    """
    Compute FID, MSE, MAE with both scaled and original data scales.
    Also compute baseline FID values for reference.
    
    Args:
        predictions: (n_test, n_samples, L) or (n_test, L) array of predictions (scaled)
        ground_truth: (n_test, L) array of ground truth (scaled)
        scaler: optional sklearn scaler to inverse transform to original scale
    
    Returns:
        dict with FID (scaled/original), MSE, MAE, and baseline FID values
    """
    # Predictions can be shape (n_test, n_samples, L) 
    if predictions.ndim == 3:
        pred_mean = predictions.mean(axis=1)  # (n_test, L)
        pred_samples = predictions  # Keep for FID
        n_test, n_samp, L = predictions.shape
    else:
        pred_mean = predictions
        pred_samples = None
        n_test, L = predictions.shape
        n_samp = 1
    
    # ===== Compute FID on SCALED data =====
    if pred_samples is not None:
        pred_flat = pred_samples.reshape(-1, L)
        gt_flat = np.repeat(ground_truth, n_samp, axis=0)
    else:
        pred_flat = pred_mean
        gt_flat = ground_truth
    
    fid_scaled = calculate_fid(gt_flat, pred_flat)
    
    # Compute baseline FID on scaled data
    # 1. Real vs Real (should be 0 or very low)
    fid_baseline_real = calculate_fid(gt_flat, gt_flat)
    
    # 2. Gaussian noise vs Real (upper bound)
    gaussian_noise = np.random.randn(*gt_flat.shape).astype(np.float32)
    fid_baseline_gaussian = calculate_fid(gt_flat, gaussian_noise)
    
    # ===== Inverse transform to ORIGINAL scale =====
    if scaler is not None:
        pred_mean_orig = scaler.inverse_transform(pred_mean)
        gt_orig = scaler.inverse_transform(ground_truth)
        
        if pred_samples is not None:
            pred_flat_orig = pred_samples.reshape(-1, L)
            pred_flat_orig = scaler.inverse_transform(pred_flat_orig)
            gt_flat_orig = np.repeat(gt_orig, n_samp, axis=0)
        else:
            pred_flat_orig = pred_mean_orig
            gt_flat_orig = gt_orig
        
        fid_original = calculate_fid(gt_flat_orig, pred_flat_orig)
        fid_baseline_real_orig = calculate_fid(gt_flat_orig, gt_flat_orig)
        gaussian_noise_orig = np.random.randn(*gt_flat_orig.shape).astype(np.float32)
        fid_baseline_gaussian_orig = calculate_fid(gt_flat_orig, gaussian_noise_orig)
    else:
        fid_original = None
        fid_baseline_real_orig = None
        fid_baseline_gaussian_orig = None
        pred_mean_orig = None
        gt_orig = None
    
    # ===== Compute MSE and MAE on ORIGINAL scale =====
    if scaler is not None:
        mse_mean = mean_squared_error(gt_orig, pred_mean_orig)
        mae_mean = mean_absolute_error(gt_orig, pred_mean_orig)
        mse_std = np.std(((gt_orig - pred_mean_orig) ** 2).mean(axis=1))
        mae_std = np.std(np.abs(gt_orig - pred_mean_orig).mean(axis=1))
    else:
        mse_mean = mean_squared_error(ground_truth, pred_mean)
        mae_mean = mean_absolute_error(ground_truth, pred_mean)
        mse_std = np.std(((ground_truth - pred_mean) ** 2).mean(axis=1))
        mae_std = np.std(np.abs(ground_truth - pred_mean).mean(axis=1))
    
    return {
        'fid_scaled': fid_scaled,
        'fid_original': fid_original,
        'fid_baseline_real_scaled': fid_baseline_real,
        'fid_baseline_gaussian_scaled': fid_baseline_gaussian,
        'fid_baseline_real_original': fid_baseline_real_orig,
        'fid_baseline_gaussian_original': fid_baseline_gaussian_orig,
        'mse': mse_mean,
        'mae': mae_mean,
        'mse_std': mse_std,
        'mae_std': mae_std,
    }

def compute_metrics(predictions, ground_truth, scaler=None):
    """
    Legacy wrapper for backward compatibility.
    """
    result = compute_metrics_with_fid_variants(predictions, ground_truth, scaler)
    return {
        'fid': result['fid_original'] if result['fid_original'] is not None else result['fid_scaled'],
        'mse': result['mse'],
        'mae': result['mae'],
        'mse_std': result['mse_std'],
        'mae_std': result['mae_std'],
    }

# -------------------------------------------------------------------------
# Load samples and compute metrics
# -------------------------------------------------------------------------

def discover_sample_runs(samples_dir, models, datasets):
    """Discover both legacy and cross-eval sample files."""
    runs = []
    if not os.path.exists(samples_dir):
        return runs

    pred_files = [f for f in os.listdir(samples_dir) if f.startswith("samples_") and f.endswith("_pred.npy")]
    legacy_re = re.compile(r"^samples_(?P<model>.+)_(?P<dataset>wa|bern)_pred\.npy$")
    cross_re = re.compile(
        r"^samples_(?P<model>.+)_(?P<model_dataset>wa|bern|joint_orig|merged_G|merged_24-2|merged_242|merged_G_clipm30|merged_24-2_clipm30|merged_242_clipm30)_on_(?P<data_dataset>wa|bern)_(?P<conversion>24toG|Gto24|none|auto)_pred\.npy$"
    )

    for fname in sorted(pred_files):
        m_cross = cross_re.match(fname)
        if m_cross is not None:
            model = m_cross.group('model')
            model_dataset = m_cross.group('model_dataset')
            data_dataset = m_cross.group('data_dataset')
            conversion = m_cross.group('conversion')
            if model not in models:
                continue

            real_file = fname.replace('_pred.npy', '_real.npy')
            real_path = os.path.join(samples_dir, real_file)
            if not os.path.exists(real_path):
                continue

            runs.append({
                'model': model,
                'model_dataset': model_dataset,
                'data_dataset': data_dataset,
                'conversion': conversion,
                'sample_file': os.path.join(samples_dir, fname),
                'real_file': real_path,
            })
            continue

        m_legacy = legacy_re.match(fname)
        if m_legacy is not None:
            model = m_legacy.group('model')
            dataset = m_legacy.group('dataset')
            if model not in models or dataset not in datasets:
                continue

            real_file = fname.replace('_pred.npy', '_real.npy')
            real_path = os.path.join(samples_dir, real_file)
            if not os.path.exists(real_path):
                continue

            runs.append({
                'model': model,
                'model_dataset': dataset,
                'data_dataset': dataset,
                'conversion': 'none',
                'sample_file': os.path.join(samples_dir, fname),
                'real_file': real_path,
            })

    return runs


print("=" * 70)
print("Computing Metrics for All Model Variants and Datasets")
print("=" * 70)

results = []
stage_results = []
runs = discover_sample_runs(SAMPLES_DIR, MODELS, DATASETS)

if len(runs) == 0:
    print("\nERROR: No samples found! Make sure all sampling jobs have completed.")
    exit(1)

print(f"Found {len(runs)} run(s) in {SAMPLES_DIR}")

scaler_cache = {}
stage_cache = {}

for run in runs:
    model = run['model']
    data_dataset = run['data_dataset']
    model_dataset = run['model_dataset']
    conversion = run['conversion']

    print(f"\n{data_dataset.upper()} Dataset | model={model} | model_dataset={model_dataset} | conversion={conversion}")
    print("-" * 70)

    # joint_orig checkpoints are evaluated in the selected data domain,
    # so use the data_dataset scaler (WA or Bern) instead of a single model-domain scaler.
    scaler_key = data_dataset if model_dataset == "joint_orig" else model_dataset

    if scaler_key not in scaler_cache:
        if scaler_key not in MODEL_DATASET_INFO:
            print(f"  WARNING: Unsupported scaler domain={scaler_key}. Run skipped.")
            continue
        model_info = MODEL_DATASET_INFO[scaler_key]
        scaler_file = os.path.join(model_info["data_path"], model_info["scaler_file"])
        scaler_cache[scaler_key] = np.load(scaler_file, allow_pickle=True).item()
    scaler = scaler_cache[scaler_key]

    if data_dataset not in stage_cache:
        data_info = DATASETS_INFO[data_dataset]
        stage_labels = None
        test_file_path = os.path.join(data_info["data_path"], data_info["test_file"])
        if os.path.exists(test_file_path):
            df_test = pd.read_csv(test_file_path)
            md_col = data_info["md_col"]
            if md_col in df_test.columns:
                classify_fn = data_info["classify_fn"]
                stage_labels = df_test[md_col].apply(classify_fn).astype(int).to_numpy()
            else:
                print(f"  WARNING: MD column '{md_col}' not found in {test_file_path}. Stage results skipped.")
        else:
            print(f"  WARNING: Test file not found ({test_file_path}). Stage results skipped.")
        stage_cache[data_dataset] = stage_labels

    stage_labels = stage_cache[data_dataset]

    print(f"  Loading {os.path.basename(run['sample_file'])}...", end=" ", flush=True)
    predictions = np.load(run['sample_file'])
    ground_truth = np.load(run['real_file'])
    print(f"Shape: {predictions.shape} | {ground_truth.shape}")

    print(f"    Computing metrics (scaled & original scale)...", end=" ", flush=True)
    metrics_full = compute_metrics_with_fid_variants(predictions, ground_truth, scaler=scaler)
    print("Done")

    result_row = {
        'Dataset': data_dataset.upper(),
        'Data_Dataset': data_dataset,
        'Model_Dataset': model_dataset,
        'Conversion': conversion,
        'Run_Name': os.path.basename(run['sample_file']).replace('_pred.npy', ''),
        'Model': model,
        'FID_Scaled': metrics_full['fid_scaled'],
        'FID_Original': metrics_full['fid_original'],
        'FID_Baseline_Real_Scaled': metrics_full['fid_baseline_real_scaled'],
        'FID_Baseline_Gaussian_Scaled': metrics_full['fid_baseline_gaussian_scaled'],
        'FID_Baseline_Real_Original': metrics_full['fid_baseline_real_original'],
        'FID_Baseline_Gaussian_Original': metrics_full['fid_baseline_gaussian_original'],
        'MSE': metrics_full['mse'],
        'MAE': metrics_full['mae'],
        'MSE_std': metrics_full['mse_std'],
        'MAE_std': metrics_full['mae_std'],
    }
    results.append(result_row)

    print(f"      FID (scaled): {metrics_full['fid_scaled']:.4f} | FID (original): {metrics_full['fid_original']:.4f}")
    print(f"      MSE: {metrics_full['mse']:.6f} | MAE: {metrics_full['mae']:.6f}")
    print(f"      Baseline (scaled) - Real: {metrics_full['fid_baseline_real_scaled']:.6f}, Gaussian: {metrics_full['fid_baseline_gaussian_scaled']:.4f}")
    print(f"      (Metrics computed on original data scale)")

    # if "all_together" in model and stage_labels is not None:
    if len(stage_labels) != ground_truth.shape[0]:
        print("      WARNING: Stage labels length mismatch. Stage results skipped.")
    else:
        print("      Stage-wise results (all_together):")
        for stage in sorted(np.unique(stage_labels)):
            idx = np.where(stage_labels == stage)[0]
            if idx.size == 0:
                continue
            stage_predictions = predictions[idx]
            stage_ground_truth = ground_truth[idx]
            stage_metrics = compute_metrics_with_fid_variants(
                stage_predictions, stage_ground_truth, scaler=scaler
            )

            stage_results.append({
                'Dataset': data_dataset.upper(),
                'Data_Dataset': data_dataset,
                'Model_Dataset': model_dataset,
                'Conversion': conversion,
                'Run_Name': os.path.basename(run['sample_file']).replace('_pred.npy', ''),
                'Model': model,
                'Stage': int(stage),
                'N': int(idx.size),
                'FID_Scaled': stage_metrics['fid_scaled'],
                'FID_Original': stage_metrics['fid_original'],
                'MSE': stage_metrics['mse'],
                'MAE': stage_metrics['mae'],
                'MSE_std': stage_metrics['mse_std'],
                'MAE_std': stage_metrics['mae_std'],
            })
            print(
                f"        Stage {stage} (n={idx.size}): "
                f"FID (original) {stage_metrics['fid_original']:.4f} | "
                f"MSE {stage_metrics['mse']:.6f} (sd {stage_metrics['mse_std']:.6f}) | "
                f"MAE {stage_metrics['mae']:.6f} (sd {stage_metrics['mae_std']:.6f})"
            )

# -------------------------------------------------------------------------
# Create results dataframe
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Results Summary")
print("=" * 70)
print(df_results.to_string(index=False))

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------
results_csv = os.path.join(OUTPUT_DIR, "metrics_summary_vf1.csv")
df_results.to_csv(results_csv, index=False)
print(f"\n✓ Results saved to: {results_csv}")

# -------------------------------------------------------------------------
# Save stage-wise results (all_together only)
# -------------------------------------------------------------------------
if len(stage_results) > 0:
    df_stage = pd.DataFrame(stage_results)
    stage_csv = os.path.join(OUTPUT_DIR, "metrics_stages_all_together.csv")
    df_stage.to_csv(stage_csv, index=False)
    print(f"\n✓ Stage-wise results saved to: {stage_csv}")

# -------------------------------------------------------------------------
# Create comparison tables (per dataset)
# -------------------------------------------------------------------------
for dataset in DATASETS:
    df_dataset = df_results[df_results['Data_Dataset'] == dataset].copy()
    if len(df_dataset) == 0:
        continue
    
    # Sort by FID on original scale (best first)
    df_dataset_sorted = df_dataset.sort_values('FID_Original').reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"{dataset.upper()} Dataset - Ranked by FID (Lower is Better)")
    print(f"(Metrics in original data scale)")
    print(f"{'='*70}")
    print(df_dataset_sorted[['Model', 'Model_Dataset', 'Conversion', 'FID_Original', 'MSE', 'MAE']].to_string(index=True))
    
    # Save dataset-specific results
    dataset_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset.lower()}.csv")
    df_dataset_sorted.to_csv(dataset_csv, index=False)
    print(f"\n✓ Saved: {dataset_csv}")
 
print("\n" + "=" * 70)
print("Metrics computation complete!")
print("=" * 70)
