import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ablation_models_diffusion_wa import DiffusionModelLoc


DATASETS = {
    "wa": {
        "train_files": {
            "3vf": "df_train_wa_3vf_augmented.csv",
            "4vf": "df_train_wa_4vf_augmented.csv",
        },
        "test_files": {
            "3vf": "df_test_wa_3vf_normal.csv",
            "4vf": "df_test_wa_4vf_normal.csv",
        },
        "scaler_file": "scaler_wa_3vf.npy",
        "vf_size": 54,
        "exclude_cols": [25, 34],
        "age_scale": 365.0,
    },
    "bern": {
        "train_files": {
            "3vf": "df_train_bern_3vf_augmented_new.csv",
            "4vf": "df_train_bern_4vf_augmented.csv",
        },
        "test_files": {
            "3vf": "df_test_bern_3vf_normal_new.csv",
            "4vf": "df_test_bern_4vf_normal.csv",
        },
        "scaler_file": "scaler_bern_3vf_new.npy",
        "vf_size": 59,
        "exclude_cols": [],
        "age_scale": 1.0,
    },
}


WA_ORDER_RAW = [49, 7, 30, 25, 19, 18, 32, 51, 35, 22, 26, 4, 36, 28, 40, 34, 16, 5, 24, 45, 23, 54, 2, 44, 20, 10, 41, 43, 37, 13, 47, 3, 33, 15, 11, 27, 52, 1, 39, 9, 31, 29, 42, 53, 50, 17, 8, 38, 21, 14, 48, 46, 6, 12]
WA_ORDER_RAW = [v - 1 for v in WA_ORDER_RAW]  # Convert to 0-based indexing

BERN_ORDER = [18, 50, 40, 4, 25, 37, 13, 54, 27, 56, 48, 28, 6, 55, 19, 5, 11, 57, 53, 30, 29, 31, 21, 0, 32, 20, 24, 38, 35, 23, 36, 43, 9, 49, 17, 39, 41, 45, 58, 42, 14, 3, 16, 22, 12, 26, 52, 10, 8, 15, 51, 46, 44, 47, 33, 7, 34, 2, 1]


# ============================================================================
# Diffusion Schedule Functions (DDPM)
# ============================================================================
def f(t, num_steps, s=8e-3):
    """Diffusion schedule for cumulative alpha."""
    t = torch.as_tensor(t)
    return torch.cos((t/num_steps + s)/(1 + s) * torch.pi/2)**2


def build_diffusion_schedule(num_steps, device):
    """Build diffusion schedule for DDPM training and sampling."""
    cumalphas = f(torch.arange(num_steps + 1, device=device), num_steps) / f(torch.tensor(0.0, device=device), num_steps)
    alphas = cumalphas[1:] / cumalphas[:-1]
    betas = 1 - alphas
    cumalphas = cumalphas[1:]
    return alphas, betas, cumalphas


def load_xy_coords(dataset_name, data_path, cfg):
    if dataset_name == "wa":
        xy_file = os.path.join(data_path, "p24d2.csv")
        df_xy = pd.read_csv(xy_file)
        x_coords = df_xy["x"].values.astype(np.float32)
        y_coords = df_xy["y"].values.astype(np.float32)
    elif dataset_name == "bern":
        xy_file = os.path.join(data_path, "x_y.xlsx")
        df_xy = pd.read_excel(xy_file)
        x_coords = df_xy.iloc[:, 0].values.astype(np.float32)
        y_coords = df_xy.iloc[:, 1].values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if cfg.get("exclude_cols"):
        ex = cfg["exclude_cols"]
        x_coords = np.delete(x_coords, ex)
        y_coords = np.delete(y_coords, ex)

    return x_coords, y_coords


def process_df(df, scaler, cfg, n_vf):
    arr = df[df.columns[df.columns.str.contains("TD_")]].values.astype(np.float32)
    vf_size = cfg["vf_size"]

    x_list = [arr[:, i * vf_size:(i + 1) * vf_size] for i in range(n_vf)]
    if cfg.get("exclude_cols"):
        ex = cfg["exclude_cols"]
        x_list = [np.delete(xi, ex, axis=1) for xi in x_list]

    x_list = [scaler.transform(xi) for xi in x_list]

    age_scale = cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f"age_{i}"].values.astype(np.float32) / age_scale) for i in range(n_vf)]
    x_tensors = [torch.from_numpy(xi.astype(np.float32)) for xi in x_list]
    return tuple(x_tensors + t_list)


def normalize_order(order_raw, raw_l, exclude_cols, subtract_one=False):
    order = list(order_raw)

    if subtract_one:
        candidate = [v - 1 for v in order]
        if min(candidate) >= 0 and max(candidate) < raw_l:
            order = candidate

    if min(order) < 0 or max(order) >= raw_l:
        raise ValueError(f"Order indices out of range [0, {raw_l - 1}] after normalization")

    ex = set(exclude_cols)
    kept = []
    seen = set()
    for idx in order:
        if idx in ex or idx in seen:
            continue
        kept.append(idx)
        seen.add(idx)

    # Remap raw indices to compact indices after column deletion.
    remap = {}
    new_idx = 0
    for i in range(raw_l):
        if i in ex:
            continue
        remap[i] = new_idx
        new_idx += 1

    remapped_order = [remap[i] for i in kept]

    l_post = raw_l - len(ex)
    remaining = [i for i in range(l_post) if i not in set(remapped_order)]
    remapped_order.extend(remaining)

    return remapped_order


def choose_order(dataset_name, cfg):
    raw_l = cfg["vf_size"]
    ex = cfg.get("exclude_cols", [])
    if dataset_name == "wa":
        return normalize_order(WA_ORDER_RAW, raw_l, ex, subtract_one=True)
    if dataset_name == "bern":
        return normalize_order(BERN_ORDER, raw_l, ex, subtract_one=False)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

def build_model_inputs(batch, max_cond):
    # batch layout: x_0 ... x_{n-1}, t_0 ... t_{n-1}
    n_total = len(batch) // 2
    x_list = list(batch[:n_total])
    t_list = list(batch[n_total:])

    target = x_list[-1]
    target_time = t_list[-1]

    available_cond = n_total - 1
    use_cond = min(max_cond, available_cond)

    cond_v = x_list[available_cond - use_cond:available_cond]
    cond_t = [target_time - t for t in t_list[available_cond - use_cond:available_cond]]

    dt_future = torch.zeros_like(target_time)
    return cond_v, cond_t, target_time, dt_future, target

def build_seen_prefix_mask(target, order_tensor, k):
    """
    Build ordered-prefix seen mask from the first k entries of order_tensor.
    """
    B, L = target.shape
    seen_mask = torch.zeros(B, L, device=target.device, dtype=target.dtype)
    if k > 0:
        seen_idx = order_tensor[:k]
        seen_mask[:, seen_idx] = 1.0
    seen_values = target * seen_mask
    return seen_values, seen_mask


def inverse_transform_batch(batch_array, scaler):
    """Map standardized VF values back to the original scale."""
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(batch_array)
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return batch_array * scaler.scale_ + scaler.mean_
    raise ValueError("Scaler does not support inverse transformation")


def compute_split_mae(pred_all, target_all, seen_idx):
    """Compute overall, seen, and unseen MAE on the same scale as the inputs."""
    abs_err = np.abs(pred_all - target_all)
    overall_mae = float(abs_err.mean())

    if seen_idx is None or len(seen_idx) == 0:
        return overall_mae, float("nan"), float(abs_err.mean())

    seen_mask = np.zeros(target_all.shape[1], dtype=bool)
    seen_mask[np.asarray(seen_idx, dtype=np.int64)] = True
    unseen_mask = ~seen_mask

    seen_mae = float(abs_err[:, seen_mask].mean()) if seen_mask.any() else float("nan")
    unseen_mae = float(abs_err[:, unseen_mask].mean()) if unseen_mask.any() else float("nan")
    return overall_mae, seen_mae, unseen_mae


def compute_fid_from_arrays(real_array, fake_array, eps=1e-6):
    """Compute FID between two sets of feature vectors."""
    real_array = np.asarray(real_array, dtype=np.float64)
    fake_array = np.asarray(fake_array, dtype=np.float64)

    mu_real = real_array.mean(axis=0)
    mu_fake = fake_array.mean(axis=0)

    cov_real = np.cov(real_array, rowvar=False)
    cov_fake = np.cov(fake_array, rowvar=False)

    if cov_real.ndim == 0:
        cov_real = np.array([[cov_real]])
    if cov_fake.ndim == 0:
        cov_fake = np.array([[cov_fake]])

    dim = mu_real.shape[0]
    cov_real = cov_real + np.eye(dim, dtype=np.float64) * eps
    cov_fake = cov_fake + np.eye(dim, dtype=np.float64) * eps

    cov_sqrt = _matrix_sqrt_psd(cov_real)
    middle = cov_sqrt @ cov_fake @ cov_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle)

    mean_diff = mu_real - mu_fake
    fid = mean_diff.dot(mean_diff) + np.trace(cov_real + cov_fake - 2.0 * middle_sqrt)
    return float(np.real(fid))


def _matrix_sqrt_psd(matrix):
    """Matrix square root for symmetric PSD matrices."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    order_tensor,
    max_cond,
    cumalphas,
    one_location,
    location_index = None,
    grad_scaler=None,
):
    """DDPM-style training with ordered-prefix conditioning."""
    model.train()

    L = order_tensor.numel()
    num_steps = cumalphas.shape[0]

    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]

        cond_v, cond_times, _ , dt_future, target = build_model_inputs(
            batch, max_cond=max_cond
        )
        locs = [torch.arange(L, device=device).float() for _ in range(len(cond_v))]

        B = target.shape[0]

        # ---- Sample random timestep ----
        t = torch.randint(1, num_steps + 1, (B,), device=device)
        
        # ---- Sample random noise ----
        noise = torch.randn_like(target)
        
        # ---- Add noise to target using DDPM schedule ----
        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
        x_t_noisy = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * noise
        
        # ---- Sample k (ordered-prefix conditioning) with bias toward lower k ----
        # Exponential decay: P(k) ∝ exp(-decay*k), heavily favoring low-k during training
        # k_decay = 0.1  # Decay parameter; lower values increase bias toward k=0
        # k_weights = torch.exp(-k_decay * torch.arange(L+1, dtype=torch.float32, device=device))
        # k_weights = k_weights / k_weights.sum()  # Normalize to probabilities
        # k = torch.multinomial(k_weights, 1, replacement=False).item()
        # ---- Build full-length seen-prefix VF and mask ----
        # if one_location:
        #     k = location_index
        # if k > 0:
        #     seen_idx = order_tensor[:k]
        #     target_seen = target[:, seen_idx]
        #     cond_v.append(target_seen)
        #     cond_times.append(dt_future)
        #     locs.append(seen_idx.float())

        # randomly select 10 k values
        # choose randomly number L from 0 to L-1
        num_k = torch.randint(0, L+1, (1,)).item()

        if num_k > 0:
            indexes_k = torch.randperm(L)[:num_k].to(device)
            seen_idx = indexes_k
            target_seen = target[:, seen_idx]
            cond_v.append(target_seen)
            cond_times.append(dt_future)
            locs.append(seen_idx.float())
        
        locs.append(torch.arange(L, device=device).float())

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            # ---- Predict noise from noisy input conditioned on seen prefix ----
            noise_pred = model(
                x_t=x_t_noisy,
                v_hist=cond_v,
                dt_hist=cond_times,
                t_step=t,
                dt_future=dt_future,
                locs=locs,
            )

            loss = nn.MSELoss()(noise_pred, noise)

        # ---- backward ----
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

    train_mse = float(loss.detach().cpu())

    return train_mse

@torch.no_grad()
def sample_ddpm_with_fixed_k(
    model,
    loader,
    device,
    order_tensor,
    max_cond,
    alphas,
    betas,
    cumalphas,
    k_value,
    max_batches=0,
):
    """DDPM sampling with fixed k (for stratified evaluation)."""
    model.eval()

    L = order_tensor.numel()
    num_steps = cumalphas.shape[0]
    pred_list = []
    target_list = []

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        batch = [b.to(device, non_blocking=True) for b in batch]
        cond_v, cond_times, _ , dt_future, target = build_model_inputs(
            batch, max_cond=max_cond
        )
        locs = [torch.arange(L, device=device).float() for _ in range(len(cond_v))]

        B = target.size(0)

        if k_value > 0:
            seen_idx = order_tensor[:k_value]
            target_seen = target[:, seen_idx]
            cond_v.append(target_seen)
            cond_times.append(dt_future)
            locs.append(seen_idx.float())
        
        locs.append(torch.arange(L, device=device).float())

        # ---- Start from pure noise ----
        x_t = torch.randn(B, L, device=device)

        # ---- DDPM reverse process ----
        for t_step_eval in range(num_steps - 1, 0, -1):
            alpha_t = alphas[t_step_eval - 1]
            cumalpha_t = cumalphas[t_step_eval - 1]
            sigma = torch.sqrt(betas[t_step_eval - 1])
            factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)

            t_tensor = torch.tensor([t_step_eval], device=device).expand(B)

            noise_pred = model(
                x_t=x_t,
                v_hist=cond_v,
                dt_hist=cond_times,
                t_step=t_tensor,
                dt_future=dt_future,
                locs=locs
            )

            z = torch.randn_like(x_t) if t_step_eval > 1 else torch.zeros_like(x_t)
            x_t = 1 / torch.sqrt(alpha_t) * (x_t - factor * noise_pred) + sigma * z

        pred_list.append(x_t.detach().cpu())
        target_list.append(target.detach().cpu())

    if not pred_list:
        return np.empty((0, L), dtype=np.float32), np.empty((0, L), dtype=np.float32)

    pred_all = torch.cat(pred_list, dim=0).numpy().astype(np.float32)
    target_all = torch.cat(target_list, dim=0).numpy().astype(np.float32)

    return pred_all, target_all

@torch.no_grad()
def evaluate_stratified(
    model,
    loader,
    device,
    order_tensor,
    max_cond,
    alphas,
    betas,
    cumalphas,
    vf_scaler,
    one_location,
    location_index=None,
    max_batches=0,
):
    """
    Evaluate model with stratified k ranges and return weighted average metrics.
    k_ranges: list of (k_min, k_max, name) tuples
    """
    k_ranges = [
        (0, 0, "k=0"),
        (1, 10, "k=[1-10]"),
        # (21, 30, "k=[21-30]"),
    ]
    if one_location:
        k_ranges = [
            (location_index, location_index, f"k={location_index}")
        ]

    results = {}
    total_fid = 0.0
    total_mae = 0.0
    num_ranges = len(k_ranges)

    for k_min, k_max, k_name in k_ranges:
        # For each range, evaluate at a random value (or at k_min for single value)
        k_eval = k_min if k_min == k_max else int(torch.randint(k_min, k_max + 1, (1,), device=device).item())
        seen_idx = order_tensor[:k_eval].detach().cpu().numpy() if k_eval > 0 else None

        pred_all, target_all = sample_ddpm_with_fixed_k(
            model,
            loader,
            device,
            order_tensor,
            max_cond=max_cond,
            alphas=alphas,
            betas=betas,
            cumalphas=cumalphas,
            k_value=k_eval,
            max_batches=max_batches,
        )

        if pred_all.size == 0:
            fid = float("inf")
            mae = float("inf")
            mae_seen = float("inf")
            mae_unseen = float("inf")
        else:
            fid = compute_fid_from_arrays(target_all, pred_all)
            pred_all = inverse_transform_batch(pred_all, vf_scaler)
            target_all = inverse_transform_batch(target_all, vf_scaler)
            mae, mae_seen, mae_unseen = compute_split_mae(pred_all, target_all, seen_idx)

        results[k_name] = {"fid": fid, "mae": mae, "mae_seen": mae_seen, "mae_unseen": mae_unseen, "k_eval": k_eval}
        total_fid += fid
        total_mae += mae

    avg_fid = total_fid / num_ranges
    avg_mae = total_mae / num_ranges

    return results, avg_fid, avg_mae


def load_checkpoint_state(model, checkpoint_path, device):
    """Load a pretrained checkpoint into the current model for fine-tuning."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded pretrained weights from: {checkpoint_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train sequential observed-location model")
    parser.add_argument("--dataset", type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=52, help="Number of diffusion steps")
    parser.add_argument("--max_cond", type=int, default=2)
    parser.add_argument("--n_vf", type=int, default=3, choices=[3, 4])
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_max_batches", type=int, default=0, help="0 means full test set (default: full test set for accurate model selection)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile (can add startup overhead)")
    parser.add_argument("--sample_max_samples", type=int, default=512)
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Optional pretrained checkpoint used to initialize model before fine-tuning",
    )
    parser.add_argument("--data_path", type=str, default="/storage/homefs/mc21s017/data_vf/")
    parser.add_argument("--output_dir", type=str, default="/storage/homefs/mc21s017/")
    parser.add_argument("--one_location", action="store_true")
    parser.add_argument("--location_index", type=int, default=16)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = DATASETS[args.dataset]
    vf_key = f"{args.n_vf}vf"

    train_file = os.path.join(args.data_path, cfg["train_files"][vf_key])
    test_file = os.path.join(args.data_path, cfg["test_files"][vf_key])
    scaler_path = os.path.join(args.data_path, cfg["scaler_file"])

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    vf_scaler = np.load(scaler_path, allow_pickle=True).item()

    required_cols = [col for col in df_train.columns if col.startswith("TD_")]
    required_cols.extend([f"age_{i}" for i in range(args.n_vf)])

    df_train = df_train.dropna(subset=required_cols).reset_index(drop=True)
    df_test = df_test.dropna(subset=required_cols).reset_index(drop=True)

    proc_train = process_df(df_train, vf_scaler, cfg, args.n_vf)
    proc_test = process_df(df_test, vf_scaler, cfg, args.n_vf)

    train_ds = TensorDataset(*proc_train)
    test_ds = TensorDataset(*proc_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    x_coords_np, _ = load_xy_coords(args.dataset, args.data_path, cfg)
    L = x_coords_np.shape[0]

    order = choose_order(args.dataset, cfg)
    if len(order) != L:
        raise ValueError(f"Order length {len(order)} does not match L={L}")

    order_tensor = torch.tensor(order, dtype=torch.long, device=device)
    max_cond_eff = min(args.max_cond, args.n_vf - 1)

    model = DiffusionModelLoc(
        L=L,
        D=args.D,
        K=args.layers,
        max_cond=max_cond_eff,
    ).to(device)

    if args.init_checkpoint:
        load_checkpoint_state(model, args.init_checkpoint, device)

    if hasattr(vf_scaler, "scale_"):
        feature_scale = torch.tensor(vf_scaler.scale_, dtype=torch.float32, device=device)
    else:
        feature_scale = torch.ones(L, dtype=torch.float32, device=device)

    # Debug: print scaler info
    print(f"\n=== Scaler Info ===")
    print(f"vf_scaler type: {type(vf_scaler)}")
    print(f"vf_scaler.scale_ shape: {vf_scaler.scale_.shape}")
    print(f"vf_scaler.scale_ min/max: {vf_scaler.scale_.min():.6f} / {vf_scaler.scale_.max():.6f}")
    print(f"vf_scaler.mean_ min/max: {vf_scaler.mean_.min():.6f} / {vf_scaler.mean_.max():.6f}")
    print(f"feature_scale on device: {feature_scale.device}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- Build diffusion schedule ----
    alphas, betas, cumalphas = build_diffusion_schedule(args.num_steps, device)

    best_fid = float("inf")
    best_mae = float("inf")
    if args.one_location:
        save_path = os.path.join(args.output_dir, f"sequential_observed_{args.dataset}_{args.n_vf}vf_one_location_{args.location_index}.pth")
    else:
        save_path = os.path.join(args.output_dir, f"sequential_observed_{args.dataset}_{args.n_vf}vf.pth")

    use_amp = torch.cuda.is_available()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if args.use_compile:
        model = torch.compile(model)

    print(f"Dataset: {args.dataset} | L={L} | n_vf={args.n_vf} | max_cond={max_cond_eff}")
    print(f"Diffusion training: num_steps={args.num_steps}")
    print("Train conditioning: ordered future prefix appended as a full-length VF")
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Eval every: {max(args.eval_every, 1)} epochs")
    print(f"Model selection: STRATIFIED by k ranges with WEIGHTED AVERAGE")
    print(f"  k=0, k=[1-10], k=[11-20], k=[21-30], k=[31-40]")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"AMP: {use_amp} | torch.compile: {args.use_compile}")
    print(f"Saving to: {save_path}")

    for epoch in range(1, args.epochs + 1):
        t0_train = time.time()
        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            order_tensor,
            max_cond=max_cond_eff,
            cumalphas=cumalphas,
            one_location=args.one_location,
            location_index=args.location_index,
            grad_scaler=grad_scaler,
        )

        train_dt = time.time() - t0_train
        eval_dt = 0.0

        eval_every = max(args.eval_every, 1)
        if epoch % eval_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | starting stratified evaluation...", flush=True)
            t0_eval = time.time()
            results, avg_fid, avg_mae = evaluate_stratified(
                model,
                test_loader,
                device,
                order_tensor,
                max_cond=max_cond_eff,
                alphas=alphas,
                betas=betas,
                cumalphas=cumalphas,
                one_location=args.one_location,
                location_index=args.location_index,
                vf_scaler=vf_scaler,
            )
            eval_dt = time.time() - t0_eval

            # Print detailed results
            print(f"Epoch {epoch:4d} | Stratified Results:", flush=True)
            for k_name, metrics in results.items():
                print(
                    f"  {k_name:12s} | FID={metrics['fid']:8.6f} | MAE={metrics['mae']:8.6f} | "
                    f"seen={metrics['mae_seen']:8.6f} | unseen={metrics['mae_unseen']:8.6f}",
                    flush=True,
                )
            print(
                f"  {'WEIGHTED_AVG':12s} | FID={avg_fid:8.6f} | MAE={avg_mae:8.6f} | time={eval_dt:.1f}s",
                flush=True,
            )

            if avg_fid < best_fid:
                best_fid = avg_fid
                best_mae = avg_mae
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": args.dataset,
                        "n_vf": args.n_vf,
                        "L": L,
                        "order": order,
                        "best_test_mae": best_mae,
                        "best_test_fid": best_fid,
                        "stratified_results": results,
                    },
                    save_path,
                )
                print(
                    f"Epoch {epoch:4d} | *** New best WEIGHTED_AVG FID: {best_fid:.6f} | MAE: {best_mae:.6f} *** saved model",
                    flush=True,
                )

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d} | train_next_mse_std={train_mse:.6f} | "
                f"train={train_dt:.1f}s"
            )

if __name__ == "__main__":
    main()
