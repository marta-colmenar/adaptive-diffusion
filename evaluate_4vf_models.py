"""
Evaluate 4VF Models - Multiple Prediction Tasks
------------------------------------------------
Generates 150 samples per test sample for models trained on 4VFs.
Tests three prediction tasks:
  - VF3 → VF4 (1 conditioning VF)
  - VF2, VF3 → VF4 (2 conditioning VFs)  
  - VF1, VF2, VF3 → VF4 (3 conditioning VFs)

Computes FID and MSE on non-transformed (original scale) data for each task.
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
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate 4VF models on multiple tasks')
parser.add_argument('--model', type=str, required=True,
                    choices=['all_together','cross_attn', 'full', 'baseline', 'self_attn', 'spatial', 'temporal'],
                    help='Model variant to evaluate')
parser.add_argument('--dataset', type=str, default='wa', choices=['wa', 'bern'],
                    help='Dataset variant')
parser.add_argument('--num_samples', type=int, default=150,
                    help='Number of samples per test sample')
parser.add_argument('--num_steps', type=int, default=52,
                    help='Number of diffusion steps')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for test data')
parser.add_argument('--D', type=int, default=32,
                    help='Model dimension')
parser.add_argument('--K', type=int, default=2,
                    help='Number of attention blocks')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--model_path', type=str, default='/storage/homefs/mc21s017/',
                    help='Path to saved models')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/results_4vf/',
                    help='Output directory for results')
args = parser.parse_args()

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = args.num_samples
NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
MODEL_NAME = args.model
DATASET_NAME = args.dataset
DATA_PATH = args.data_path
MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset registry
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

# -------------------------------------------------------------------------
# Dataset processing
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
print("=" * 80)
print(f"Evaluating 4VF Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
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
# Model Loading
# -------------------------------------------------------------------------
print(f"\nLoading trained model: {MODEL_NAME}...", flush=True)

checkpoint_path = os.path.join(MODEL_PATH, f"denoiser_ablation_{MODEL_NAME}_{DATASET_NAME}_4vf.pth")
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"Checkpoint loaded from epoch {checkpoint['epoch']}, best FID: {checkpoint['best_fid']:.4f}")

# Build model with same parameters as training (max_cond=3 for 4VF models)
model_kwargs = {'L': L, 'D': args.D, 'max_cond': 3}
if MODEL_NAME in ['self_attn', 'cross_attn', 'full']:
    model_kwargs['K'] = args.K

denoiser = get_model(MODEL_NAME, **model_kwargs).to(device)
denoiser.load_state_dict(checkpoint['model_state_dict'])
denoiser.eval()

n_params = sum(p.numel() for p in denoiser.parameters())
print(f"Model parameters: {n_params:,}")

# -------------------------------------------------------------------------
# Diffusion schedule
# -------------------------------------------------------------------------
cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(torch.tensor(0.0, device=device), NUM_STEPS)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# -------------------------------------------------------------------------
# Define prediction tasks
# -------------------------------------------------------------------------
TASKS = [
    {
        'name': 'VF1→VF4',
        'max_cond': 1,
        'cond_vfs': [0],  # x1_b
        'description': 'Single conditioning VF (VF1) predicting VF4',
    },
    {
        'name': 'VF2→VF4',
        'max_cond': 1,
        'cond_vfs': [1],  # x2_b
        'description': 'Single conditioning VF (VF2) predicting VF4',
    },
    {
        'name': 'VF1,VF2→VF4',
        'max_cond': 2,
        'cond_vfs': [0, 1],  # x1_b, x2_b
        'description': 'Two conditioning VFs (VF1, VF2) predicting VF4',
    },
    {
        'name': 'VF1,VF3→VF4',
        'max_cond': 2,
        'cond_vfs': [0, 2],  # x1_b, x2_b
        'description': 'Two conditioning VFs (VF1, VF2) predicting VF4',
    },
    {
        'name': 'VF3→VF4',
        'max_cond': 1,
        'cond_vfs': [2],  # x3_b
        'description': 'Single conditioning VF (VF3) predicting VF4',
    },
    {
        'name': 'VF2,VF3→VF4',
        'max_cond': 2,
        'cond_vfs': [1, 2],  # x2_b, x3_b
        'description': 'Two conditioning VFs (VF2, VF3) predicting VF4',
    },
    {
        'name': 'VF1,VF2,VF3→VF4',
        'max_cond': 3,
        'cond_vfs': [0, 1, 2],  # x1_b, x2_b, x3_b
        'description': 'Three conditioning VFs (VF1, VF2, VF3) predicting VF4',
    },
]

# -------------------------------------------------------------------------
# Run evaluation for each task
# -------------------------------------------------------------------------
all_results = []

for task_idx, task in enumerate(TASKS):
    print("\n" + "=" * 80)
    print(f"Task {task_idx+1}/6: {task['name']}")
    print(f"Description: {task['description']}")
    print("=" * 80)
    
    task_name = task['name']
    max_cond = task['max_cond']
    cond_vfs = task['cond_vfs']
    
    all_samples = []
    all_reals = []
    all_time_horizons = []  # Store time horizons for averaging
    
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
            
            # Calculate time horizon: from last conditioned VF to VF4
            last_cond_idx = min(cond_vfs)
            all_dts_list = [dt_1, dt_2, dt_3]
            time_horizon = (dt_4 - all_dts_list[last_cond_idx]).cpu().numpy()
            all_time_horizons.append(time_horizon)
            
            target = x4_b
            
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
                    
                    # Condition based on task - use cond_vfs to select which VFs to use
                    all_vfs = [x1_b, x2_b, x3_b]
                    all_dts = [dt_1, dt_2, dt_3]
                    
                    cond_vf_list = [all_vfs[i] for i in cond_vfs]
                    cond_dt_list = [all_dts[i] for i in cond_vfs]
                    
                    pred_noise = denoiser(xt, cond_vf_list, cond_dt_list, t_tensor, dt_4)
                    
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
    print(f"Average time horizon (last cond VF to VF4): {avg_time_horizon:.4f} ± {std_time_horizon:.4f}")
    
    # -------------------------------------------------------------------------
    # Compute metrics (FID on scaled, MSE/MAE on original scale)
    # -------------------------------------------------------------------------
    print("Computing metrics (FID on scaled data, MSE/MAE on original scale)...")
    metrics = compute_metrics_on_original_scale(all_samples, all_reals, scaler)
    
    print(f"\nResults for {task_name}:")
    print(f"  FID: {metrics['fid']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")
    
    # Store results
    result_row = {
        'Model': MODEL_NAME,
        'Dataset': DATASET_NAME.upper(),
        'Task': task_name,
        'Conditioning_VFs': max_cond,
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
    
    # Save samples for this task
    output_prefix = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{DATASET_NAME}_{task_name.replace(',', '').replace('→', 'to')}")
    samples_file = f"{output_prefix}_pred.npy"
    reals_file = f"{output_prefix}_real.npy"
    
    np.save(samples_file, all_samples.astype(np.float32))
    np.save(reals_file, all_reals.astype(np.float32))
    
    print(f"\n✓ Saved samples:")
    print(f"  {samples_file}")
    print(f"  {reals_file}")

# -------------------------------------------------------------------------
# Summary and save results
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EVALUATION COMPLETE - SUMMARY")
print("=" * 80)

df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))

# Save results CSV
results_file = os.path.join(OUTPUT_DIR, f"results_{MODEL_NAME}_{DATASET_NAME}_4vf.csv")
df_results.to_csv(results_file, index=False)
print(f"\n✓ Results saved to: {results_file}")

# Print comparison
print("\n" + "-" * 80)
print("Performance Comparison:")
print("-" * 80)
for task_name in ['VF1→VF4', 'VF2→VF4', 'VF1,VF2→VF4', 'VF3→VF4', 'VF2,VF3→VF4', 'VF1,VF2,VF3→VF4']:
    row = df_results[df_results['Task'] == task_name].iloc[0]
    print(f"{task_name:20s} | Δt: {row['Avg_Time_Horizon']:6.4f} ± {row['Std_Time_Horizon']:6.4f} | FID: {row['FID']:7.4f} | MSE: {row['MSE']:9.6f} ± {row['MSE_std']:9.6f} | MAE: {row['MAE']:9.6f} ± {row['MAE_std']:9.6f}")

print("=" * 80)
print("All tasks completed successfully!")
print("=" * 80)
