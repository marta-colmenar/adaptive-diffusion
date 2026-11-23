"""
Train a simple baseline model for VF1 -> VF2 prediction (pairwise regression).
Uses only the first two VF snapshots from the dataset files.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CascadeNet5(nn.Module):
    """
    Inspired by Wen et al. CascadeNet-5:
    - multiple conv stages
    - feature concatenation ("cascade")
    - outputs 1 channel 8x9 map
    """

    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()

        # Stage 1
        self.s1 = nn.Sequential(
            ConvBNReLU(in_channels, base_ch),
            ConvBNReLU(base_ch, base_ch),
        )

        # Stage 2
        self.s2 = nn.Sequential(
            ConvBNReLU(base_ch, base_ch),
            ConvBNReLU(base_ch, base_ch),
        )

        # Stage 3
        self.s3 = nn.Sequential(
            ConvBNReLU(base_ch * 2, base_ch),
            ConvBNReLU(base_ch, base_ch),
        )

        # Stage 4
        self.s4 = nn.Sequential(
            ConvBNReLU(base_ch * 3, base_ch),
            ConvBNReLU(base_ch, base_ch),
        )

        # Stage 5
        self.s5 = nn.Sequential(
            ConvBNReLU(base_ch * 4, base_ch),
            ConvBNReLU(base_ch, base_ch),
        )

        self.head = nn.Conv2d(base_ch * 5, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, 8, 9)
        """
        f1 = self.s1(x)                 # (B, ch, 8, 9)
        f2 = self.s2(f1)                # (B, ch, 8, 9)

        f3_in = torch.cat([f1, f2], dim=1)
        f3 = self.s3(f3_in)

        f4_in = torch.cat([f1, f2, f3], dim=1)
        f4 = self.s4(f4_in)

        f5_in = torch.cat([f1, f2, f3, f4], dim=1)
        f5 = self.s5(f5_in)

        out_in = torch.cat([f1, f2, f3, f4, f5], dim=1)
        y = self.head(out_in)           # (B, 1, 8, 9)
        
        return y


# ------------------------------------------------------------
# VF vector <-> 8x9 grid helpers
# ------------------------------------------------------------

GRID_H = 8
GRID_W = 9
GRID_SIZE = GRID_H * GRID_W


def vf_to_grid(x, L):
    """
    Pad a VF vector (B, L) to (B, 1, 8, 9).
    """
    if L > GRID_SIZE:
        raise ValueError(f"VF length L={L} exceeds grid size {GRID_SIZE}.")
    pad = GRID_SIZE - L
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)
    return x.view(x.size(0), 1, GRID_H, GRID_W)


# ------------------------------------------------------------
# Dataset utils
# ------------------------------------------------------------

def process_df_pairs(df, scaler, cfg):
    """
    Returns tensors for VF1, VF2 and their time difference.
    """
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    vf_size = cfg["vf_size"]
    exclude_cols = cfg.get("exclude_cols", [])

    x1 = arr[:, 0 * vf_size:1 * vf_size]
    x2 = arr[:, 1 * vf_size:2 * vf_size]

    if exclude_cols:
        x1 = np.delete(x1, exclude_cols, axis=1)
        x2 = np.delete(x2, exclude_cols, axis=1)

    # Scale using provided scaler
    x1 = scaler.transform(x1)
    x2 = scaler.transform(x2)

    age_scale = cfg.get("age_scale", 1.0)
    t1 = df["age_0"].values.astype(np.float32) / age_scale
    t2 = df["age_1"].values.astype(np.float32) / age_scale
    dt_12 = t2 - t1

    x1_t = torch.from_numpy(x1.astype(np.float32))
    x2_t = torch.from_numpy(x2.astype(np.float32))
    dt_t = torch.from_numpy(dt_12.astype(np.float32))

    return x1_t, x2_t, dt_t


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VF1->VF2 baseline model")
    parser.add_argument("--dataset", type=str, default="wa", choices=["wa", "bern"],
                        help="Dataset variant")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--D", type=int, default=64,
                        help="Base channels for CascadeNet5")
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="/storage/homefs/mc21s017/data_vf/")
    parser.add_argument("--output_dir", type=str, default="/storage/homefs/mc21s017/")
    args = parser.parse_args()

    # Dataset registry
    DATASETS = {
        "wa": {
            "train_files": {"3vf": "df_train_wa_3vf_normal.csv"},
            "test_files": {"3vf": "df_test_wa_3vf_normal.csv"},
            "scaler_file": "scaler_wa_3vf.npy",
            "vf_size": 54,
            "exclude_cols": [25, 34],
            "age_scale": 365.0,
        },
        "bern": {
            "train_files": {"3vf": "df_train_bern_3vf_normal.csv"},
            "test_files": {"3vf": "df_test_bern_3vf_normal.csv"},
            "scaler_file": "scaler_bern_3vf.npy",
            "vf_size": 59,
            "exclude_cols": [],
            "age_scale": 1.0,
        },
    }

    cfg = DATASETS[args.dataset]
    train_file = os.path.join(args.data_path, cfg["train_files"]["3vf"])
    test_file = os.path.join(args.data_path, cfg["test_files"]["3vf"])
    scaler_file = os.path.join(args.data_path, cfg["scaler_file"])

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    scaler = np.load(scaler_file, allow_pickle=True).item()

    x1_train, x2_train, dt_train = process_df_pairs(df_train, scaler, cfg)
    x1_test, x2_test, dt_test = process_df_pairs(df_test, scaler, cfg)

    L_data = x1_train.shape[1]
    L_model = args.L if args.L is not None else L_data
    if args.L is not None and args.L != L_data:
        print(f"Warning: model L={args.L} differs from data L={L_data}; ensure this is intended.")

    train_ds = TensorDataset(x1_train, x2_train, dt_train)
    test_ds = TensorDataset(x1_test, x2_test, dt_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = CascadeNet5(in_channels=1, base_ch=args.D).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_mse = float("inf")
    model_path = os.path.join(args.output_dir, f"vfpair_cascadenet5_{args.dataset}_L{L_model}.pth")

    # Mask for valid VF locations in the 8x9 grid
    mask_1d = torch.zeros(GRID_SIZE, dtype=torch.float32)
    mask_1d[:L_model] = 1.0
    mask = mask_1d.view(1, 1, GRID_H, GRID_W).to(device)

    print("Training VF1 -> VF2 with CascadeNet5...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x1_b, x2_b, dt_b in train_loader:
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            dt_b = dt_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            x1_grid = vf_to_grid(x1_b, L_model)
            x2_grid = vf_to_grid(x2_b, L_model)
            pred = model(x1_grid)
            diff = (pred - x2_grid) * mask
            loss = (diff ** 2).sum() / mask.sum()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - t0
        print(f"Epoch {epoch:5d}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.6f} | Time: {epoch_time:.1f}s")

        # Eval
        if epoch % 50 == 0:
            model.eval()
            mse_sum, n = 0.0, 0
            with torch.no_grad():
                for x1_b, x2_b, dt_b in test_loader:
                    x1_b = x1_b.to(device)
                    x2_b = x2_b.to(device)
                    dt_b = dt_b.to(device)
                    x1_grid = vf_to_grid(x1_b, L_model)
                    x2_grid = vf_to_grid(x2_b, L_model)
                    pred = model(x1_grid)
                    diff = (pred - x2_grid) * mask
                    mse_sum += (diff ** 2).sum().item()
                    n += mask.sum().item()
            mse = mse_sum / max(n, 1)
            print(f"Eval MSE: {mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mse": best_mse,
                    "config": vars(args),
                }, model_path)
                print(f"✓ New best model saved! MSE: {best_mse:.6f}")

    print(f"Done. Best MSE: {best_mse:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
