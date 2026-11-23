"""
Evaluate Models with Varying Number of Conditioning VFs (N)
------------------------------------------------------------
Evaluates three models trained with different max_cond values:
    - N=1: denoiser_ablation_all_together_vf1_{dataset}_3vf.pth (max_cond=1)
    - N=2: denoiser_ablation_all_together_{dataset}_3vf.pth (max_cond=2)
    - N=3: denoiser_ablation_all_together_{dataset}_4vf.pth (max_cond=3)

Generates 1000 samples per test sample for each model.
Tests prediction tasks appropriate for each model's max_cond.
"""

import os, gc, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ablation_models_diffusion_wa import get_model, MODEL_REGISTRY
from model_diffusion_wa import f, calculate_fid

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate varying N models (WA or BERN)')
parser.add_argument('--dataset', type=str, default='wa', choices=['wa', 'bern'],
                    help='Dataset variant')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='Number of samples per test sample')
parser.add_argument('--num_steps', type=int, default=52,
                    help='Number of diffusion steps')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for test data')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--model_path', type=str, default='/storage/homefs/mc21s017/',
                    help='Path to saved models')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/results_varying_n/',
                    help='Output directory for results')
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = args.num_samples
NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
MODEL_NAME = 'all_together'
DATASET_NAME = args.dataset
DATA_PATH = args.data_path
MODEL_PATH = args.model_path
OUTPUT_DIR = os.path.join(args.output_dir, DATASET_NAME)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Dataset processing
# -------------------------------------------------------------------------
DATASETS = {
    "wa": {
        "test_file": "df_test_wa_4vf_normal.csv",
        "scaler_file": "scaler_wa_3vf.npy",
        "vf_size": 54,
        "exclude_cols": [25, 34],
        "age_scale": 365.0,
    },
    "bern": {
        "test_file": "df_test_bern_4vf_normal.csv",
        "scaler_file": "scaler_bern_3vf.npy",
        "vf_size": 59,
        "exclude_cols": [],
        "age_scale": 1.0,
    },
}

dataset_cfg = DATASETS[DATASET_NAME]

def process_df_4vf(df, scaler, cfg):
    """Process 4VF dataframe"""
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)

    vf_size = cfg["vf_size"]
    exclude_cols = cfg.get("exclude_cols", [])

    X_list = [arr[:, i*vf_size:(i+1)*vf_size] for i in range(4)]
    if exclude_cols:
        X_list = [np.delete(Xi, exclude_cols, axis=1) for Xi in X_list]

    X_list = [scaler.transform(Xi) for Xi in X_list]
    age_scale = cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f'age_{i}'].values.astype(np.float32)/age_scale) for i in range(4)]
    
    X_tensors = [torch.from_numpy(Xi.astype(np.float32)) for Xi in X_list]
    return tuple(X_tensors + t_list)

# -------------------------------------------------------------------------
# Metric computation
# -------------------------------------------------------------------------
def compute_metrics_on_original_scale(predictions, ground_truth, scaler):
    """
    Compute FID on scaled data, MSE/MAE on original (non-transformed) data scale.
    
    Args:
        predictions: (n_test, n_samples, L) array of predictions (scaled)
        ground_truth: (n_test, L) array of ground truth (scaled)
        scaler: sklearn scaler to inverse transform
    
    Returns:
        dict with FID (on scaled data) and MSE/MAE (on original scale)
    """
    n_test, n_samp, L = predictions.shape
    
    # Average predictions across samples
    pred_mean = predictions.mean(axis=1)  # (n_test, L)
    
    # Prepare data for FID (flatten all samples) - COMPUTE ON SCALED DATA
    pred_flat = predictions.reshape(-1, L)
    gt_flat = np.repeat(ground_truth, n_samp, axis=0)
    
    # Compute FID on SCALED data
    fid_scaled = calculate_fid(gt_flat, pred_flat)
    
    # Inverse transform to original scale for MSE/MAE
    pred_mean_orig = scaler.inverse_transform(pred_mean)
    gt_orig = scaler.inverse_transform(ground_truth)
    
    # Compute MSE and MAE on original scale
    mse = mean_squared_error(gt_orig, pred_mean_orig)
    mae = mean_absolute_error(gt_orig, pred_mean_orig)
    mse_std = np.std(((gt_orig - pred_mean_orig) ** 2).mean(axis=1))
    mae_std = np.std(np.abs(gt_orig - pred_mean_orig).mean(axis=1))
    
    return {
        'fid': fid_scaled,
        'mse': mse,
        'mae': mae,
        'mse_std': mse_std,
        'mae_std': mae_std,
    }

def compute_subgroup_metrics(predictions, ground_truth, time_horizons, scaler):
    """
    Compute FID and MAE for subgroups based on time intervals.
    
    Time intervals: (0, 2), [2, 4), [4, inf)
    
    Args:
        predictions: (n_test, n_samples, L) array of predictions (scaled)
        ground_truth: (n_test, L) array of ground truth (scaled)
        time_horizons: (n_test,) array of time horizons
        scaler: sklearn scaler
    
    Returns:
        dict with subgroup results
    """
    subgroup_results = []
    
    # Define time bins
    bins = [(0, 2), (2, 4), (4, np.inf)]
    bin_labels = ['0<t<2', '2<=t<4', 't>=4']
    
    for (t_min, t_max), label in zip(bins, bin_labels):
        if t_max == np.inf:
            mask = time_horizons >= t_min
        else:
            mask = (time_horizons >= t_min) & (time_horizons < t_max)
        
        n_in_group = np.sum(mask)
        if n_in_group == 0:
            continue
        
        # Extract subgroup data
        pred_sub = predictions[mask]  # (n_sub, n_samples, L)
        gt_sub = ground_truth[mask]   # (n_sub, L)
        
        # Compute metrics
        n_sub, n_samp, L = pred_sub.shape
        pred_mean_sub = pred_sub.mean(axis=1)  # (n_sub, L)
        
        # FID on scaled data
        pred_flat = pred_sub.reshape(-1, L)
        gt_flat = np.repeat(gt_sub, n_samp, axis=0)
        fid_sub = calculate_fid(gt_flat, pred_flat)
        
        # Inverse transform and compute MAE
        pred_mean_orig = scaler.inverse_transform(pred_mean_sub)
        gt_orig = scaler.inverse_transform(gt_sub)
        
        mae_sub = mean_absolute_error(gt_orig, pred_mean_orig)
        mae_std_sub = np.std(np.abs(gt_orig - pred_mean_orig).mean(axis=1))
        
        subgroup_results.append({
            'time_interval': label,
            'count': n_in_group,
            'fid': fid_sub,
            'mae': mae_sub,
            'mae_std': mae_std_sub,
        })
    
    return subgroup_results

# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
print("=" * 80)
print(f"Evaluating Models with Varying N (Number of Conditioning VFs)")
print(f"Dataset: {DATASET_NAME} | Samples per test: {NUM_SAMPLES}")
print("=" * 80)
print("Loading data...", flush=True)

test_file = os.path.join(DATA_PATH, dataset_cfg["test_file"])
scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

df_test = pd.read_csv(test_file)
scaler = np.load(scaler_file, allow_pickle=True).item()

proc_test = process_df_4vf(df_test, scaler, dataset_cfg)
x1_all, x2_all, x3_all, x4_all, t1_all, t2_all, t3_all, t4_all = proc_test

L = x1_all.shape[1]

test_ds = TensorDataset(*proc_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

print(f"Test samples: {len(test_ds)}")
print(f"VF locations: {L}")

# -------------------------------------------------------------------------
# Define model configurations
# -------------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        'N': 1,
        'checkpoint': f"denoiser_ablation_all_together_vf1_{DATASET_NAME}_3vf.pth",
        'max_cond': 1,
        'D': 64,
        'K': 2,
        'tasks': [
            {
                'name': 'VF3→VF4',
                'cond_vfs': [2],  # x3
                'target_vf': 3,  # x4
                'description': 'N=1: VF3 → VF4',
            },
        ],
    },
    {
        'N': 2,
        'checkpoint': f"denoiser_ablation_all_together_{DATASET_NAME}_3vf.pth",
        'max_cond': 2,
        'D': 64,
        'K': 2,
        'tasks': [
            {
                'name': 'VF2,VF3→VF4',
                'cond_vfs': [1, 2],  # x2, x3
                'target_vf': 3,  # x4
                'description': 'N=2: VF2,VF3 → VF4',
            },
        ],
    },
    {
        'N': 3,
        'checkpoint': f"denoiser_ablation_all_together_{DATASET_NAME}_4vf.pth",
        'max_cond': 3,
        'D': 64,
        'K': 2,
        'tasks': [
            {
                'name': 'VF1,VF2,VF3→VF4',
                'cond_vfs': [0, 1, 2],  # x1, x2, x3
                'target_vf': 3,  # x4
                'description': 'N=3: VF1,VF2,VF3 → VF4',
            },
        ],
    },
]

# -------------------------------------------------------------------------
# Diffusion schedule
# -------------------------------------------------------------------------
cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(torch.tensor(0.0, device=device), NUM_STEPS)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# -------------------------------------------------------------------------
# Evaluate each model
# -------------------------------------------------------------------------
all_results = []
n_data_store = {}  # Store data for each N to compute subgroup metrics later

for model_config in MODEL_CONFIGS:
    N = model_config['N']
    checkpoint_name = model_config['checkpoint']
    max_cond = model_config['max_cond']
    D = model_config['D']
    K = model_config['K']
    
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: N={N} (max_cond={max_cond})")
    print(f"Checkpoint: {checkpoint_name}")
    print("=" * 80)
    
    # Load model
    checkpoint_path = os.path.join(MODEL_PATH, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
        continue
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}, best FID: {checkpoint['best_fid']:.4f}")
    
    # Build model
    model_kwargs = {'L': L, 'D': D, 'max_cond': max_cond, 'K': K}
    denoiser = get_model(MODEL_NAME, **model_kwargs).to(device)
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    denoiser.eval()
    
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Evaluate each task for this model
    for task_idx, task in enumerate(model_config['tasks']):
        print("\n" + "-" * 80)
        print(f"Task {task_idx+1}/{len(model_config['tasks'])}: {task['name']}")
        print(f"Description: {task['description']}")
        print("-" * 80)
        
        task_name = task['name']
        cond_vfs = task['cond_vfs']
        target_vf_idx = task['target_vf']
        
        all_samples = []
        all_reals = []
        all_time_horizons = []
        
        print(f"Generating {NUM_SAMPLES} samples per test sample...")
        start_time = time.time()
        num_test_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_start = time.time()
                x1_b, x2_b, x3_b, x4_b, t1_b, t2_b, t3_b, t4_b = batch
                batch_size_actual = x1_b.size(0)
                
                x1_b = x1_b.to(device, non_blocking=True)
                x2_b = x2_b.to(device, non_blocking=True)
                x3_b = x3_b.to(device, non_blocking=True)
                x4_b = x4_b.to(device, non_blocking=True)
                
                dt_1 = (t1_b - t1_b).to(device, non_blocking=True)  # Always 0
                dt_2 = (t2_b - t1_b).to(device, non_blocking=True)
                dt_3 = (t3_b - t1_b).to(device, non_blocking=True)
                dt_4 = (t4_b - t1_b).to(device, non_blocking=True)
                
                # Select target VF and dt
                all_vfs = [x1_b, x2_b, x3_b, x4_b]
                all_dts = [dt_1, dt_2, dt_3, dt_4]
                
                target = all_vfs[target_vf_idx]
                dt_target = all_dts[target_vf_idx]
                
                # Calculate time horizon between VF3 and VF4 for subgrouping
                time_horizon = (dt_4 - dt_3).cpu().numpy()
                all_time_horizons.append(time_horizon)
                
                # Ground truth
                all_reals.append(target.cpu().numpy())
                
                batch_samples = []
                
                for sample_idx in range(NUM_SAMPLES):
                    xt = torch.randn_like(target, device=device)
                    
                    for t_step in range(NUM_STEPS - 1, 0, -1):
                        alpha_t = alphas[t_step - 1]
                        cumalpha_t = cumalphas[t_step - 1]
                        sigma = torch.sqrt(betas[t_step - 1])
                        factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                        t_tensor = torch.tensor([t_step], device=device).expand(batch_size_actual)
                        
                        # Select conditioning VFs and their time deltas
                        cond_vf_list = [all_vfs[i] for i in cond_vfs]
                        cond_dt_list = [all_dts[i] for i in cond_vfs]
                        
                        pred_noise = denoiser(xt, cond_vf_list, cond_dt_list, t_tensor, dt_target)
                        
                        z = torch.randn_like(xt) if t_step > 1 else torch.zeros_like(xt)
                        xt = 1/torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z
                    
                    batch_samples.append(xt.cpu().numpy())
                
                # Stack samples: (batch_size, num_samples, L)
                batch_samples = np.stack(batch_samples, axis=1)
                all_samples.append(batch_samples)
                
                batch_time = time.time() - batch_start
                elapsed_total = time.time() - start_time
                num_test_samples += batch_size_actual
                
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    print(f"  Batch {batch_idx+1:3d} | Samples: {num_test_samples:4d} | "
                          f"Batch: {batch_time:5.1f}s | Total: {elapsed_total:6.1f}s", flush=True)
                
                gc.collect()
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_samples = np.concatenate(all_samples, axis=0)  # (n_test, num_samples, L)
        all_reals = np.concatenate(all_reals, axis=0)      # (n_test, L)
        all_time_horizons = np.concatenate(all_time_horizons, axis=0)  # (n_test,)
        
        # Calculate average time horizon
        avg_time_horizon = np.mean(all_time_horizons)
        std_time_horizon = np.std(all_time_horizons)
        
        print(f"\nSampling complete! Time: {time.time() - start_time:.1f}s")
        print(f"Generated shape: {all_samples.shape}")
        print(f"Average time horizon: {avg_time_horizon:.4f} ± {std_time_horizon:.4f}")
        
        # Compute metrics
        print("Computing metrics...")
        metrics = compute_metrics_on_original_scale(all_samples, all_reals, scaler)
        
        print(f"\nResults for {task_name}:")
        print(f"  FID: {metrics['fid']:.4f}")
        print(f"  MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")
        
        # Store results
        result_row = {
            'N': N,
            'Model': f'N={N}',
            'Checkpoint': checkpoint_name,
            'Task': task_name,
            'Num_Cond_VFs': len(cond_vfs),
            'Avg_Time_Horizon': avg_time_horizon,
            'Std_Time_Horizon': std_time_horizon,
            'FID': metrics['fid'],
            'MSE': metrics['mse'],
            'MSE_std': metrics['mse_std'],
            'MAE': metrics['mae'],
            'MAE_std': metrics['mae_std'],
            'Num_Samples': NUM_SAMPLES,
            'Num_Test': all_samples.shape[0],
        }
        all_results.append(result_row)
        
        # Accumulate data for subgroup analysis by N
        if N not in n_data_store:
            n_data_store[N] = {
                'all_samples': [],
                'all_reals': [],
                'all_horizons': [],
            }
        n_data_store[N]['all_samples'].append(all_samples)
        n_data_store[N]['all_reals'].append(all_reals)
        n_data_store[N]['all_horizons'].append(all_time_horizons)
        
        # Save samples for this task
        output_prefix = os.path.join(OUTPUT_DIR, f"N{N}_{task_name.replace(',', '').replace('→', 'to')}")
        samples_file = f"{output_prefix}_pred.npy"
        reals_file = f"{output_prefix}_real.npy"
        horizons_file = f"{output_prefix}_horizons.npy"
        
        np.save(samples_file, all_samples.astype(np.float32))
        np.save(reals_file, all_reals.astype(np.float32))
        np.save(horizons_file, all_time_horizons.astype(np.float32))
        
        print(f"\n✓ Saved samples:")
        print(f"  {samples_file}")
        print(f"  {reals_file}")
        print(f"  {horizons_file}")
    
    # Clean up model from GPU
    del denoiser
    gc.collect()
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# Summary and save results
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EVALUATION COMPLETE - SUMMARY")
print("=" * 80)

df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))

# Save results CSV
results_file = os.path.join(OUTPUT_DIR, f"results_varying_n_{NUM_SAMPLES}samples.csv")
df_results.to_csv(results_file, index=False)
print(f"\n✓ Results saved to: {results_file}")

# -------------------------------------------------------------------------
# Compute and display subgroup metrics by time interval
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TIME INTERVAL SUBGROUP ANALYSIS")
print("=" * 80)

subgroup_summary = []

for N in sorted(n_data_store.keys()):
    print(f"\nProcessing subgroups for N={N}...")
    
    # Concatenate all samples, reals, horizons for this N across all tasks
    all_samples_N = np.concatenate(n_data_store[N]['all_samples'], axis=0)
    all_reals_N = np.concatenate(n_data_store[N]['all_reals'], axis=0)
    all_horizons_N = np.concatenate(n_data_store[N]['all_horizons'], axis=0)
    
    print(f"  Total samples for N={N}: {all_samples_N.shape[0]} test cases")
    print(f"  Time horizon range: {all_horizons_N.min():.4f} to {all_horizons_N.max():.4f}")
    
    # Compute subgroup metrics
    subgroups = compute_subgroup_metrics(all_samples_N, all_reals_N, all_horizons_N, scaler)
    
    for subgroup in subgroups:
        subgroup['N'] = N
        subgroup_summary.append(subgroup)
        print(f"    {subgroup['time_interval']:8s}: n={subgroup['count']:4d} | FID={subgroup['fid']:.4f} | MAE={subgroup['mae']:.6f} ± {subgroup['mae_std']:.6f}")

# Create and display summary table
df_subgroup = pd.DataFrame(subgroup_summary)
print("\n" + "=" * 80)
print("SUBGROUP METRICS TABLE")
print("=" * 80)

if len(df_subgroup) > 0:
    # Pivot table: rows = time intervals, cols = N values
    pivot_table = df_subgroup.pivot_table(
        index='time_interval',
        columns='N',
        values=['count', 'fid', 'mae', 'mae_std'],
        aggfunc='first'
    )
    print(pivot_table)
    
    # Save subgroup metrics
    subgroup_file = os.path.join(OUTPUT_DIR, f"subgroup_metrics_{DATASET_NAME}_{NUM_SAMPLES}samples.csv")
    df_subgroup.to_csv(subgroup_file, index=False)
    print(f"\n✓ Subgroup metrics saved to: {subgroup_file}")
    
    # Print formatted table
    print("\n" + "-" * 80)
    print("Formatted Summary by N and Time Interval:")
    print("-" * 80)
    for N in sorted(df_subgroup['N'].unique()):
        print(f"\nN={N}:")
        subset = df_subgroup[df_subgroup['N'] == N]
        for _, row in subset.iterrows():
            print(f"  {row['time_interval']:8s}: count={row['count']:4d} | FID={row['fid']:.4f} | MAE={row['mae']:.6f} ± {row['mae_std']:.6f}")

# Print comparison by N
print("\n" + "-" * 80)
print("Performance Comparison by N:")
print("-" * 80)
for idx, row in df_results.iterrows():
    print(f"N={row['N']} | {row['Task']:20s} | Δt: {row['Avg_Time_Horizon']:6.4f} ± {row['Std_Time_Horizon']:6.4f} | "
          f"FID: {row['FID']:7.4f} | MSE: {row['MSE']:9.6f} ± {row['MSE_std']:9.6f} | "
          f"MAE: {row['MAE']:9.6f} ± {row['MAE_std']:9.6f}")

print("=" * 80)
print("All evaluations completed successfully!")
print("=" * 80)
