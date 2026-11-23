"""
Merged Dataset Training Script (Washington + Bern)
---------------------------------------------------
Trains models on merged Washington and Bern data with coordinate conversions.
Creates two models:
  1. all_together_merged_G: trained on G pattern (59 locations)
  2. all_together_merged_242: trained on 24-2 pattern (52 locations)

Validation is kept simple: uses Bern data with G conversions for G model,
                          uses Washington data for 24-2 model.
"""

import os, gc, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import sqrtm
import wandb
from ablation_models_diffusion_wa import get_model, MODEL_REGISTRY
from model_diffusion_wa import f, calculate_fid

# =========================================================================
# Coordinate Conversion Utilities
# =========================================================================
DEFAULT_MAP_G_FROM_24_2 = {
    0: [15], 1: [12], 2: [38], 3: [41], 4: [23], 5: [7], 6: [17, 26], 7: [32], 8: [47],
    9: [35, 43], 10: [31], 11: [46], 12: [28], 13: [22], 14: [6], 15: [19], 16: [23],
    17: [22], 18: [31], 19: [32], 20: [2], 21: [1], 22: [51], 23: [52], 24: [20],
    25: [29], 26: [26], 27: [35], 28: [18], 29: [27], 30: [3, 9, 17], 31: [43, 49, 53],
    32: [22, 23, 31, 32], 33: [24], 34: [21], 35: [30], 36: [33], 37: [3, 9], 38: [0, 4],
    39: [44, 50], 40: [49, 53], 41: [9, 17], 42: [4, 10], 43: [36, 44], 44: [43, 49],
    45: [14], 46: [13], 47: [39], 48: [40], 49: [8, 9, 15, 16], 50: [4, 5, 11, 12],
    51: [37, 38, 44, 45], 52: [41, 42, 48, 49], 53: [0, 4, 10], 54: [36, 44, 50],
    55: [2, 3], 56: [0, 1], 57: [50, 51], 58: [52, 53]
}

def convert_24_to_g_batch(x_24):
    """Convert 24-2 (54 locations) to G (59 locations) via averaging."""
    if x_24.shape[1] != 54:
        raise ValueError(f"24toG conversion requires input dim 54, got {x_24.shape[1]}")
    out = np.zeros((x_24.shape[0], 59), dtype=np.float32)
    for g_idx, src_24_idxs in DEFAULT_MAP_G_FROM_24_2.items():
        out[:, g_idx] = x_24[:, src_24_idxs].mean(axis=1)
    return out


def build_map_24_from_g(map_g_from_24):
    map_24_from_g = {k: [] for k in range(54)}
    for g_idx, source_24_list in map_g_from_24.items():
        for source_24 in source_24_list:
            map_24_from_g[source_24].append(g_idx)
    return map_24_from_g


MAP_24_FROM_G = build_map_24_from_g(DEFAULT_MAP_G_FROM_24_2)


def convert_g_to_24_batch(x_g):
    """Convert G (59 locations) to 24-2 raw space (54 locations) via inverse mapping average."""
    if x_g.shape[1] != 59:
        raise ValueError(f"Gto24 conversion requires input dim 59, got {x_g.shape[1]}")
    out = np.zeros((x_g.shape[0], 54), dtype=np.float32)
    for idx_24, src_g_idxs in MAP_24_FROM_G.items():
        if len(src_g_idxs) == 0:
            out[:, idx_24] = x_g[:, min(idx_24, x_g.shape[1] - 1)]
        else:
            out[:, idx_24] = x_g[:, src_g_idxs].mean(axis=1)
    return out

# =========================================================================
# Argument Parser
# =========================================================================
parser = argparse.ArgumentParser(description='Train all_together model on merged WA+Bern data')
parser.add_argument('--target_pattern', type=str, default='G', choices=['G', '24-2'],
                    help='Target coordinate pattern for training (G=59 locations, 24-2=52 locations)')
parser.add_argument('--epochs', type=int, default=15000,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--num_steps', type=int, default=52,
                    help='Number of diffusion steps')
parser.add_argument('--eval_every', type=int, default=50,
                    help='Evaluate every N epochs')
parser.add_argument('--log_every', type=int, default=10,
                    help='Log every N batches')
parser.add_argument('--D', type=int, default=32,
                    help='Model dimension')
parser.add_argument('--K', type=int, default=2,
                    help='Number of attention blocks')
parser.add_argument('--max_cond', type=int, default=2,
                    help='Maximum conditioning VFs')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/',
                    help='Output directory for models')
parser.add_argument('--clip_min_minus30', action='store_true',
                    help='Clip VF values to a minimum of -30 before scaling')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable wandb logging')
args = parser.parse_args()

# =========================================================================
# Dataset Registry
# =========================================================================
DATASETS = {
    "wa": {
        "train_files": {"3vf": "df_train_wa_3vf_augmented.csv"},
        "test_files":  {"3vf": "df_test_wa_3vf_normal.csv"},
        "scaler_file": "scaler_wa_3vf.npy",
        "vf_size": 54,
        "exclude_cols": [25, 34],
        "age_scale": 365.0,
    },
    "bern": {
        "train_files": {"3vf": "df_train_bern_3vf.csv"},
        "test_files":  {"3vf": "df_test_bern_3vf_normal.csv"},
        "scaler_file": "scaler_bern_3vf.npy",
        "vf_size": 59,
        "exclude_cols": [],
        "age_scale": 1.0,
    },
}

# =========================================================================
# Config
# =========================================================================
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

TARGET_PATTERN = args.target_pattern
L_TARGET = 59 if TARGET_PATTERN == 'G' else 52 if TARGET_PATTERN == '24-2' else None
if L_TARGET is None:
    raise ValueError(f"Invalid target_pattern: {TARGET_PATTERN}")

NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LOG_EVERY = args.log_every
EVAL_EVERY = args.eval_every
LR = args.lr
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir
MODEL_NAME = 'all_together'
CLIP_MIN_VALUE = -30.0 if args.clip_min_minus30 else None
CLIP_SUFFIX = 'clipm30' if args.clip_min_minus30 else 'noclip'
MERGED_SUFFIX = f'merged_{TARGET_PATTERN}_{CLIP_SUFFIX}'

MODEL_PATH = os.path.join(OUTPUT_DIR, f"denoiser_ablation_{MODEL_NAME}_{MERGED_SUFFIX}_3vf.pth")

slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

if not args.no_wandb:
    wandb.init(
        entity="marta-colmenar-h",
        project="glaucoma-diffusion-ablation",
        name=(
            f"merged_{TARGET_PATTERN}_{CLIP_SUFFIX}_{slurm_job_id}"
            if slurm_job_id else f"merged_{TARGET_PATTERN}_{CLIP_SUFFIX}"
        ),
        tags=["merged", "all_together", TARGET_PATTERN, CLIP_SUFFIX],
        config={
            "model": MODEL_NAME,
            "target_pattern": TARGET_PATTERN,
            "clip_min_value": CLIP_MIN_VALUE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_steps": NUM_STEPS,
            "L": L_TARGET,
            "D": args.D,
            "K": args.K,
            "max_cond": args.max_cond,
        },
        resume="never"
    )

print("=" * 70)
print(f"Training {MODEL_NAME} on merged WA+Bern data")
print(f"Target pattern: {TARGET_PATTERN} (L={L_TARGET})")
print(f"Input clipping: min={CLIP_MIN_VALUE}" if CLIP_MIN_VALUE is not None else "Input clipping: disabled")
print("=" * 70)

# =========================================================================
# Dataset Processing
# =========================================================================
def process_df_merged(data_path, target_pattern, clip_min_value=None):
    """
    Merge WA and Bern data, converting both to target pattern.
    
    Target pattern 'G': convert WA 24-2 -> G, keep Bern G
    Target pattern '24-2': keep WA 24-2, convert Bern G -> 24-2 (not fully implemented, use G only)
    
    Returns: merged tensors (X_train_list, t_train_list), (X_test_list, t_test_list)
    """
    
    def _extract_vf_raw(df, cfg, n_vf):
        arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
        vf_size = cfg["vf_size"]
        X_list = [arr[:, i*vf_size:(i+1)*vf_size] for i in range(n_vf)]
        return X_list

    # Load WA data
    wa_cfg = DATASETS["wa"]
    wa_train_file = os.path.join(data_path, wa_cfg["train_files"]["3vf"])
    wa_test_file = os.path.join(data_path, wa_cfg["test_files"]["3vf"])
    
    df_wa_train = pd.read_csv(wa_train_file)
    df_wa_test = pd.read_csv(wa_test_file)
    scaler_wa = np.load(os.path.join(data_path, wa_cfg["scaler_file"]), allow_pickle=True).item()
    
    X_wa_train = _extract_vf_raw(df_wa_train, wa_cfg, 3)  # 3 VF snapshots
    X_wa_test = _extract_vf_raw(df_wa_test, wa_cfg, 3)
    t_wa_train = [torch.from_numpy(df_wa_train[f'age_{i}'].values.astype(np.float32)/wa_cfg["age_scale"]) for i in range(3)]
    t_wa_test = [torch.from_numpy(df_wa_test[f'age_{i}'].values.astype(np.float32)/wa_cfg["age_scale"]) for i in range(3)]
    
    # Load Bern data
    bern_cfg = DATASETS["bern"]
    bern_train_file = os.path.join(data_path, bern_cfg["train_files"]["3vf"])
    bern_test_file = os.path.join(data_path, bern_cfg["test_files"]["3vf"])
    
    df_bern_train = pd.read_csv(bern_train_file)
    df_bern_test = pd.read_csv(bern_test_file)
    scaler_bern = np.load(os.path.join(data_path, bern_cfg["scaler_file"]), allow_pickle=True).item()
    
    X_bern_train = _extract_vf_raw(df_bern_train, bern_cfg, 3)
    X_bern_test = _extract_vf_raw(df_bern_test, bern_cfg, 3)
    t_bern_train = [torch.from_numpy(df_bern_train[f'age_{i}'].values.astype(np.float32)/bern_cfg["age_scale"]) for i in range(3)]
    t_bern_test = [torch.from_numpy(df_bern_test[f'age_{i}'].values.astype(np.float32)/bern_cfg["age_scale"]) for i in range(3)]
    
    # Convert based on target pattern
    if target_pattern == 'G':
        # Convert WA 24-2 -> G to match Bern G
        print("  Converting WA data from 24-2 to G...")
        X_wa_train_conv = [convert_24_to_g_batch(Xi) for Xi in X_wa_train]
        
        # Merge training data from both datasets
        X_train_merged = [np.concatenate([X_wa_train_conv[i], X_bern_train[i]], axis=0) for i in range(3)]
        X_test_merged = X_bern_test  # Use Bern test only for validation
        t_train_merged = [torch.cat([t_wa_train[i], t_bern_train[i]]) for i in range(3)]
        t_test_merged = t_bern_test
        
        # Use Bern scaler (G pattern)
        scaler_target = scaler_bern
        
    else:  # '24-2'
        # Convert Bern G -> raw 24-2 then drop WA blindspot cols to match 52-loc space
        print("  Converting Bern data from G to 24-2...")
        wa_exclude_cols = wa_cfg.get("exclude_cols", [])

        X_wa_train_52 = [np.delete(Xi, wa_exclude_cols, axis=1) for Xi in X_wa_train]
        X_wa_test_52 = [np.delete(Xi, wa_exclude_cols, axis=1) for Xi in X_wa_test]

        X_bern_train_24_raw = [convert_g_to_24_batch(Xi) for Xi in X_bern_train]
        X_bern_train_52 = [np.delete(Xi, wa_exclude_cols, axis=1) for Xi in X_bern_train_24_raw]

        X_train_merged = [np.concatenate([X_wa_train_52[i], X_bern_train_52[i]], axis=0) for i in range(3)]
        X_test_merged = X_wa_test_52  # Keep WA validation for simplicity
        t_train_merged = [torch.cat([t_wa_train[i], t_bern_train[i]]) for i in range(3)]
        t_test_merged = t_wa_test
        scaler_target = scaler_wa
    
    if clip_min_value is not None:
        X_train_merged = [np.clip(Xi, clip_min_value, None) for Xi in X_train_merged]
        X_test_merged = [np.clip(Xi, clip_min_value, None) for Xi in X_test_merged]

    # Scale merged data
    X_train_scaled = [scaler_target.transform(Xi) for Xi in X_train_merged]
    X_test_scaled = [scaler_target.transform(Xi) for Xi in X_test_merged]
    
    X_train_tensors = [torch.from_numpy(Xi.astype(np.float32)) for Xi in X_train_scaled]
    X_test_tensors = [torch.from_numpy(Xi.astype(np.float32)) for Xi in X_test_scaled]
    
    return (X_train_tensors, t_train_merged), (X_test_tensors, t_test_merged), scaler_target

print("Loading merged data...")
(X_train_list, t_train_list), (X_test_list, t_test_list), scaler = process_df_merged(
    DATA_PATH, TARGET_PATTERN, CLIP_MIN_VALUE
)

# Unpack
x1_train, x2_train, x3_train = X_train_list
t1_train, t2_train, t3_train = t_train_list
x1_test, x2_test, x3_test = X_test_list
t1_test, t2_test, t3_test = t_test_list

# Build datasets
train_ds = TensorDataset(x1_train, x2_train, x3_train, t1_train, t2_train, t3_train)
test_ds = TensorDataset(x1_test, x2_test, x3_test, t1_test, t2_test, t3_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

L_data = x1_train.shape[1]
print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
print(f"VF locations: L={L_data}")

if L_data != L_TARGET:
    raise ValueError(f"Data L={L_data} does not match target L={L_TARGET}")

# =========================================================================
# Model + Optimizer
# =========================================================================
print(f"Building all_together model (L={L_TARGET})...")

model_kwargs = {'L': L_TARGET, 'D': args.D, 'max_cond': args.max_cond, 'K': args.K}
denoiser = get_model('all_together', **model_kwargs).to(device)

n_params = sum(p.numel() for p in denoiser.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)
scaler_grad = torch.cuda.amp.GradScaler()

# Diffusion schedule
cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(torch.tensor(0.0, device=device), NUM_STEPS)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# =========================================================================
# Training Loop
# =========================================================================
print("Starting training...")
print("=" * 70)

best_loss = float("inf")
best_fid = float("inf")

for epoch in range(1, EPOCHS + 1):
    denoiser.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = batch
        x1_b = x1_b.to(device, non_blocking=True)
        x2_b = x2_b.to(device, non_blocking=True)
        x3_b = x3_b.to(device, non_blocking=True)
        t_diff_2 = (t2_b - t1_b).to(device, non_blocking=True)
        t_diff_3 = (t3_b - t1_b).to(device, non_blocking=True)
        B_cur = x3_b.size(0)

        t = torch.randint(1, NUM_STEPS + 1, (B_cur,), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            # Primary task: VF1+VF2 -> VF3
            target = x3_b
            cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
            xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
            dt_0 = torch.zeros_like(t_diff_2)
            noise_pred = denoiser(xt, [x1_b, x2_b], [dt_0, t_diff_2], t, t_diff_3)
            noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
            loss = ((noise_pred - noise_target) ** 2).mean()

        scaler_grad.scale(loss).backward()
        scaler_grad.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        scaler_grad.step(optimizer)
        scaler_grad.update()

        running_loss += loss.item()

        if (batch_idx + 1) % LOG_EVERY == 0:
            print(f"Epoch {epoch:5d} | Batch {batch_idx+1:5d} | Loss: {loss.item():.6f}", flush=True)

    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - t0

    # Evaluation
    if epoch % EVAL_EVERY == 0:
        denoiser.eval()
        test_loss = 0.0
        pred_x0_all = []
        real_x0_all = []

        with torch.no_grad():
            for batch in test_loader:
                x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = batch
                x1_b = x1_b.to(device, non_blocking=True)
                x2_b = x2_b.to(device, non_blocking=True)
                x3_b = x3_b.to(device, non_blocking=True)
                t_diff_2 = (t2_b - t1_b).to(device, non_blocking=True)
                t_diff_3 = (t3_b - t1_b).to(device, non_blocking=True)
                B_cur = x3_b.size(0)

                t = torch.ones(B_cur, dtype=torch.long, device=device) * (NUM_STEPS // 2)
                cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                xt = torch.sqrt(cumalphas_t) * x3_b + torch.sqrt(1 - cumalphas_t) * torch.randn_like(x3_b, device=device)
                dt_0 = torch.zeros_like(t_diff_2)
                noise_pred = denoiser(xt, [x1_b, x2_b], [dt_0, t_diff_2], t, t_diff_3)
                noise_target = (xt - torch.sqrt(cumalphas_t) * x3_b) / torch.sqrt(1 - cumalphas_t)
                test_loss += ((noise_pred - noise_target) ** 2).mean().item()

                pred_x0 = (xt - torch.sqrt(1 - cumalphas_t) * noise_pred) / torch.sqrt(cumalphas_t)
                pred_x0_all.append(pred_x0.detach().cpu().numpy())
                real_x0_all.append(x3_b.detach().cpu().numpy())

        test_loss /= len(test_loader)
        pred_x0_all = np.concatenate(pred_x0_all, axis=0)
        real_x0_all = np.concatenate(real_x0_all, axis=0)
        eval_fid = calculate_fid(real_x0_all, pred_x0_all)

        if eval_fid < best_fid:
            best_fid = eval_fid
            best_loss = test_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fid': best_fid,
                'best_loss': best_loss,
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"  ✓ Best model saved (FID={eval_fid:.6f}, loss={test_loss:.6f})")

        print(
            f"Epoch {epoch:5d} | Train loss: {epoch_loss:.6f} | "
            f"Test loss: {test_loss:.6f} | FID: {eval_fid:.6f} | Time: {epoch_time:.1f}s",
            flush=True
        )

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "test_loss": test_loss,
                "eval_fid": eval_fid,
                "best_fid": best_fid,
                "time_per_epoch": epoch_time,
            })

        gc.collect()
        torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(f"Training complete!")
print(f"Best validation FID: {best_fid:.6f}")
print(f"Best model saved to: {MODEL_PATH}")
print("=" * 70)

if not args.no_wandb:
    wandb.finish()
