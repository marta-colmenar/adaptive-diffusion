"""
Analyze Ablation Study Results
-------------------------------
Compares all trained ablation models and generates performance table.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ablation_models_diffusion_wa import get_model, MODEL_REGISTRY
from model_diffusion_wa import f, calculate_fid
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/storage/homefs/mc21s017/data_vf/"
OUTPUT_DIR = "/storage/homefs/mc21s017/"
NUM_STEPS = 52
NUM_MC = 100  # Monte Carlo samples per test sample
L = 52
D = 32
K = 2
MAX_COND = 2

# -------------------------------------------------------------------------
# Load test data
# -------------------------------------------------------------------------
def process_df(df, scaler):
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    exclude_cols = [25, 34]
    X1, X2, X3 = arr[:, :54], arr[:, 54:108], arr[:, 108:]
    X1, X2, X3 = np.delete(X1, exclude_cols, axis=1), np.delete(X2, exclude_cols, axis=1), np.delete(X3, exclude_cols, axis=1)
    X1, X2, X3 = scaler.transform(X1), scaler.transform(X2), scaler.transform(X3)
    return (
        torch.from_numpy(X1.astype(np.float32)),
        torch.from_numpy(X2.astype(np.float32)),
        torch.from_numpy(X3.astype(np.float32)),
        torch.from_numpy(df['age_0'].values.astype(np.float32)/365),
        torch.from_numpy(df['age_1'].values.astype(np.float32)/365),
        torch.from_numpy(df['age_2'].values.astype(np.float32)/365),
    )

print("Loading test data...")
df_test = pd.read_csv(f'{DATA_PATH}df_test_wa_3vf_normal.csv')
scaler = np.load(f'{DATA_PATH}scaler_wa_3vf.npy', allow_pickle=True).item()

X1_test, X2_test, X3_test, t1_test, t2_test, t3_test = process_df(df_test, scaler)
test_ds = TensorDataset(X1_test, X2_test, X3_test, t1_test, t2_test, t3_test)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Diffusion schedule
cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(torch.tensor(0.0, device=device), NUM_STEPS)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# -------------------------------------------------------------------------
# Evaluation function
# -------------------------------------------------------------------------
def evaluate_model(model, model_name):
    """Evaluate model on all three tasks"""
    model.eval()
    results = {}
    
    def eval_task(task_id, task_name):
        preds, reals = [], []
        mse_mean_sum, mae_mean_sum, n = 0.0, 0.0, 0
        
        with torch.no_grad():
            for x1_b, x2_b, x3_b, t1_b, t2_b, t3_b in test_loader:
                x1_b, x2_b, x3_b = x1_b.to(device), x2_b.to(device), x3_b.to(device)
                t_diff_2 = (t2_b - t1_b).to(device)
                t_diff_3 = (t3_b - t1_b).to(device)
                dt_0 = torch.zeros_like(t_diff_2)

                # Task configuration
                if task_id == 1:
                    target = x2_b
                    cond_v = [x1_b]
                    cond_dt = [dt_0]
                    dt_future = t_diff_2
                elif task_id == 2:
                    target = x3_b
                    cond_v = [x1_b, x2_b]
                    cond_dt = [dt_0, t_diff_2]
                    dt_future = t_diff_3
                else:
                    target = x3_b
                    cond_v = [x2_b]
                    cond_dt = [dt_0]
                    dt_future = t_diff_3 - t_diff_2

                # Monte Carlo DDPM sampling
                sum_preds = torch.zeros_like(target)
                for _ in range(NUM_MC):
                    xt = torch.randn_like(target, device=device)
                    for t_step in range(NUM_STEPS - 1, 0, -1):
                        alpha_t = alphas[t_step - 1]
                        cumalpha_t = cumalphas[t_step - 1]
                        sigma = torch.sqrt(betas[t_step - 1])
                        factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                        t_tensor = torch.tensor([t_step], device=device).expand(xt.shape[0])

                        pred_noise = model(xt, cond_v, cond_dt, t_tensor, dt_future)
                        z = torch.randn_like(xt) if t_step > 1 else torch.zeros_like(xt)
                        xt = 1/torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z

                    sum_preds += xt
                    preds.append(xt.cpu().numpy())

                reals.append(target.cpu().numpy())
                mean_pred = sum_preds / float(NUM_MC)
                mse_mean_sum += nn.functional.mse_loss(mean_pred, target, reduction="sum").item()
                mae_mean_sum += nn.functional.l1_loss(mean_pred, target, reduction="sum").item()
                n += target.numel()

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)
        
        fid = calculate_fid(reals, preds)
        mse = mse_mean_sum / n
        mae = mae_mean_sum / n
        rmse = np.sqrt(mse)
        
        return {'FID': fid, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}
    
    print(f"  Evaluating {model_name}...")
    results['Task1_VF1->VF2'] = eval_task(1, "VF1->VF2")
    results['Task2_VF1+VF2->VF3'] = eval_task(2, "VF1+VF2->VF3")
    results['Task3_VF2->VF3'] = eval_task(3, "VF2->VF3")
    
    return results

# -------------------------------------------------------------------------
# Load and evaluate all models
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("ABLATION STUDY RESULTS")
print("="*70 + "\n")

all_results = {}

for model_name in MODEL_REGISTRY.keys():
    model_path = os.path.join(OUTPUT_DIR, f"denoiser_ablation_{model_name}_wa_3vf.pth")
    
    if not os.path.exists(model_path):
        print(f"⚠ Warning: Model {model_name} not found at {model_path}")
        continue
    
    print(f"\nLoading {model_name} model...")
    
    # Create model
    model_kwargs = {'L': L, 'D': D, 'max_cond': MAX_COND}
    if model_name in ['self_attn', 'cross_attn', 'full']:
        model_kwargs['K'] = K
    
    model = get_model(model_name, **model_kwargs).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        best_fid = checkpoint.get('best_fid', 'N/A')
        print(f"  Loaded from epoch {epoch}, best FID: {best_fid}")
    else:
        model.load_state_dict(checkpoint)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Evaluate
    results = evaluate_model(model, model_name)
    all_results[model_name] = {
        'n_params': n_params,
        'results': results
    }

# -------------------------------------------------------------------------
# Create results table
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70 + "\n")

# Prepare data for table
table_data = []
model_order = ['baseline', 'spatial', 'self_attn', 'cross_attn', 'full']

for model_name in model_order:
    if model_name not in all_results:
        continue
    
    data = all_results[model_name]
    n_params = data['n_params']
    
    # Task 2 results (primary task)
    task2 = data['results']['Task2_VF1+VF2->VF3']
    
    row = [
        model_name.replace('_', ' ').title(),
        f"{n_params:,}",
        f"{task2['FID']:.4f}",
        f"{task2['MSE']:.6f}",
        f"{task2['MAE']:.6f}",
        f"{task2['RMSE']:.6f}"
    ]
    table_data.append(row)

headers = ['Model', 'Params', 'FID ↓', 'MSE ↓', 'MAE ↓', 'RMSE ↓']
print(tabulate(table_data, headers=headers, tablefmt='grid'))

# -------------------------------------------------------------------------
# Detailed results for all tasks
# -------------------------------------------------------------------------
print("\n" + "="*70)
print("DETAILED RESULTS (ALL TASKS)")
print("="*70 + "\n")

for model_name in model_order:
    if model_name not in all_results:
        continue
    
    print(f"\n{model_name.upper().replace('_', ' ')}:")
    print("-" * 70)
    
    results = all_results[model_name]['results']
    
    for task_name, metrics in results.items():
        print(f"\n  {task_name}:")
        print(f"    FID:  {metrics['FID']:.4f}")
        print(f"    MSE:  {metrics['MSE']:.6f}")
        print(f"    MAE:  {metrics['MAE']:.6f}")
        print(f"    RMSE: {metrics['RMSE']:.6f}")

# -------------------------------------------------------------------------
# Export to CSV
# -------------------------------------------------------------------------
output_csv = os.path.join(OUTPUT_DIR, "ablation_results.csv")
df_results = pd.DataFrame(table_data, columns=headers)
df_results.to_csv(output_csv, index=False)
print(f"\n\n✓ Results saved to: {output_csv}")

# -------------------------------------------------------------------------
# Create visualization
# -------------------------------------------------------------------------
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')

models = [m for m in model_order if m in all_results]
model_labels = [m.replace('_', '\n').title() for m in models]

# Extract metrics
fid_scores = [all_results[m]['results']['Task2_VF1+VF2->VF3']['FID'] for m in models]
mse_scores = [all_results[m]['results']['Task2_VF1+VF2->VF3']['MSE'] for m in models]
mae_scores = [all_results[m]['results']['Task2_VF1+VF2->VF3']['MAE'] for m in models]
params = [all_results[m]['n_params'] / 1000 for m in models]  # in thousands

# FID
axes[0, 0].bar(model_labels, fid_scores, color='steelblue', alpha=0.7)
axes[0, 0].set_title('FID Score (↓ better)', fontweight='bold')
axes[0, 0].set_ylabel('FID')
axes[0, 0].grid(axis='y', alpha=0.3)

# MSE
axes[0, 1].bar(model_labels, mse_scores, color='coral', alpha=0.7)
axes[0, 1].set_title('MSE (↓ better)', fontweight='bold')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].grid(axis='y', alpha=0.3)

# MAE
axes[1, 0].bar(model_labels, mae_scores, color='seagreen', alpha=0.7)
axes[1, 0].set_title('MAE (↓ better)', fontweight='bold')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].grid(axis='y', alpha=0.3)

# Parameters
axes[1, 1].bar(model_labels, params, color='mediumpurple', alpha=0.7)
axes[1, 1].set_title('Model Parameters', fontweight='bold')
axes[1, 1].set_ylabel('Parameters (K)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
output_fig = os.path.join(OUTPUT_DIR, "ablation_results.png")
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_fig}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
