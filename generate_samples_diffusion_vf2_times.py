"""
Sample VF3 From Two Observed VFs At Fixed Offsets From VF2
---------------------------------------------------------
Uses exactly two observed VFs (VF1, VF2) as conditioning and generates
samples for a hypothetical future VF3 at predefined offsets from VF2.

The model is conditioned in the same way as the standard 3VF task:
    cond_v = [VF1, VF2]
    cond_dt = [0, t2 - t1]

The requested future time is specified relative to VF2, but the model
expects the target time relative to VF1. Therefore:

    dt_future = (t2 - t1) + offset_from_vf2

Outputs are written one horizon at a time to keep host RAM bounded.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ablation_models_diffusion_wa import MODEL_REGISTRY, get_model
from model_diffusion_wa import f


TIME_LABELS = ["0m", "2m", "4m", "8m", "16m", "32m", "64m"]
TIME_MONTHS = np.array([0, 2, 4, 8, 16, 32, 64], dtype=np.float32)
TIME_YEARS = TIME_MONTHS / 12.0


parser = argparse.ArgumentParser(
    description="Generate samples for VF1,VF2 -> VF3 at fixed offsets from VF2"
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=list(MODEL_REGISTRY.keys()),
    help="Model variant to evaluate",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="wa",
    choices=["wa", "bern"],
    help="Dataset variant",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=150,
    help="Number of generated samples per test case and horizon",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=52,
    help="Number of diffusion steps",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for test data",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="DataLoader worker processes",
)
parser.add_argument(
    "--D",
    type=int,
    default=32,
    help="Model dimension",
)
parser.add_argument(
    "--K",
    type=int,
    default=2,
    help="Number of attention blocks",
)
parser.add_argument(
    "--L",
    type=int,
    default=None,
    help="Number of VF locations",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/storage/homefs/mc21s017/data_vf/",
    help="Path to data directory",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/storage/homefs/mc21s017/",
    help="Path to saved models",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/storage/homefs/mc21s017/samples/",
    help="Output directory for samples",
)
parser.add_argument(
    "--save_original_scale",
    action="store_true",
    help="Inverse-transform samples before saving",
)
args = parser.parse_args()


np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = args.num_samples
NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
MODEL_NAME = args.model
DATASET_NAME = args.dataset
DATA_PATH = args.data_path
MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)


DATASETS = {
    "wa": {
        "test_files": {"3vf": "df_test_wa_3vf_normal.csv"},
        "scaler_file": "scaler_wa_3vf.npy",
        "vf_size": 54,
        "exclude_cols": [25, 34],
        "age_scale": 365.0,
    },
    "bern": {
        "test_files": {"3vf": "df_test_bern_3vf_normal.csv"},
        "scaler_file": "scaler_bern_3vf.npy",
        "vf_size": 59,
        "exclude_cols": [],
        "age_scale": 1.0,
    },
}


def process_df(df, scaler, cfg):
    arr = df[df.columns[df.columns.str.contains("TD_")]].values.astype(np.float32)
    vf_size = cfg["vf_size"]
    exclude_cols = cfg.get("exclude_cols", [])

    x_list = [arr[:, i * vf_size:(i + 1) * vf_size] for i in range(3)]
    if exclude_cols:
        x_list = [np.delete(x_i, exclude_cols, axis=1) for x_i in x_list]

    x_list = [scaler.transform(x_i).astype(np.float32) for x_i in x_list]
    age_scale = cfg.get("age_scale", 1.0)
    t_list = [
        torch.from_numpy(df[f"age_{i}"].values.astype(np.float32) / age_scale)
        for i in range(3)
    ]

    x_tensors = [torch.from_numpy(x_i) for x_i in x_list]
    return tuple(x_tensors + t_list)


def load_model(model_name, l_model):
    checkpoint_path = os.path.join(
        MODEL_PATH,
        f"denoiser_ablation_{model_name}_{DATASET_NAME}_3vf.pth",
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = {"L": l_model, "D": args.D, "max_cond": 2}
    if model_name in ["self_attn", "cross_attn", "full", "all_together"]:
        model_kwargs["K"] = args.K

    denoiser = get_model(model_name, **model_kwargs).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])
    denoiser.eval()

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Loaded {model_name} | epoch {checkpoint.get('epoch', '?')} | params: {n_params:,}")
    return denoiser


dataset_cfg = DATASETS[DATASET_NAME]
print("=" * 70)
print(f"Generating per-time samples from VF2 offsets | Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
print("=" * 70)
print("Loading data...", flush=True)

test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"]["3vf"])
scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

df_test = pd.read_csv(test_file)
scaler = np.load(scaler_file, allow_pickle=True).item()
proc_test = process_df(df_test, scaler, dataset_cfg)

x1_all, x2_all, x3_all, t1_all, t2_all, t3_all = proc_test
test_ds = TensorDataset(x1_all, x2_all, x3_all, t1_all, t2_all, t3_all)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

l_data = x1_all.shape[1]
l_model = args.L if args.L is not None else l_data
n_test_total = len(test_ds)

print(f"Test samples: {n_test_total}")
print(f"VF locations (data): {l_data}, VF locations (model): {l_model}")


cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(
    torch.tensor(0.0, device=device), NUM_STEPS
)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]


denoiser = load_model(MODEL_NAME, l_model)
start_time = time.time()

output_prefix = os.path.join(OUTPUT_DIR, f"samples_vf2_time_{MODEL_NAME}_{DATASET_NAME}")
prediction_files = {}
dt_target_files = {}

# Save reference information once for downstream analysis.
actual_vf3_file = f"{output_prefix}_actual_vf3.npy"
actual_dt_from_vf1_file = f"{output_prefix}_actual_dt_from_vf1.npy"
actual_dt_from_vf2_file = f"{output_prefix}_actual_dt_from_vf2.npy"
cond_vfs_file = f"{output_prefix}_cond_vfs.npy"
cond_times_file = f"{output_prefix}_cond_times.npy"

if args.save_original_scale:
    actual_vf3 = scaler.inverse_transform(x3_all.numpy()).astype(np.float32)
else:
    actual_vf3 = x3_all.numpy().astype(np.float32)
np.save(actual_vf3_file, actual_vf3)

actual_dt_from_vf1 = (t3_all - t1_all).numpy().astype(np.float32)
actual_dt_from_vf2 = (t3_all - t2_all).numpy().astype(np.float32)
np.save(actual_dt_from_vf1_file, actual_dt_from_vf1)
np.save(actual_dt_from_vf2_file, actual_dt_from_vf2)

cond_vfs = np.stack([x1_all.numpy(), x2_all.numpy()], axis=1).astype(np.float32)
if args.save_original_scale:
    cond_vfs_orig = np.empty_like(cond_vfs)
    cond_vfs_orig[:, 0, :] = scaler.inverse_transform(cond_vfs[:, 0, :])
    cond_vfs_orig[:, 1, :] = scaler.inverse_transform(cond_vfs[:, 1, :])
    cond_vfs = cond_vfs_orig.astype(np.float32)
np.save(cond_vfs_file, cond_vfs)

cond_times = np.stack([
    np.zeros(n_test_total, dtype=np.float32),
    (t2_all - t1_all).numpy().astype(np.float32),
], axis=1)
np.save(cond_times_file, cond_times)


with torch.no_grad():
    for label, offset_years in zip(TIME_LABELS, TIME_YEARS):
        print(f"\nTime horizon {label} from VF2 ({offset_years:.4f} years)", flush=True)

        samples_file = f"{output_prefix}_{label}_pred.npy"
        samples_mm = np.lib.format.open_memmap(
            samples_file,
            mode="w+",
            dtype=np.float32,
            shape=(n_test_total, NUM_SAMPLES, l_model),
        )

        dt_file = f"{output_prefix}_{label}_dt_target.npy"
        dt_mm = np.lib.format.open_memmap(
            dt_file,
            mode="w+",
            dtype=np.float32,
            shape=(n_test_total,),
        )

        write_idx = 0
        num_test_samples = 0

        for batch_idx, (x1_b, x2_b, x3_b, t1_b, t2_b, _t3_b) in enumerate(test_loader):
            batch_start = time.time()
            batch_size_actual = x1_b.size(0)

            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            t1_b = t1_b.to(device, non_blocking=True)
            t2_b = t2_b.to(device, non_blocking=True)

            dt_from_vf1_to_vf2 = t2_b - t1_b
            dt_hist = [torch.zeros_like(dt_from_vf1_to_vf2), dt_from_vf1_to_vf2]
            dt_target = dt_from_vf1_to_vf2 + torch.full(
                (batch_size_actual,),
                float(offset_years),
                device=device,
            )
            dt_mm[write_idx:write_idx + batch_size_actual] = dt_target.detach().cpu().numpy().astype(np.float32)

            batch_samples = []
            for _sample_idx in range(NUM_SAMPLES):
                xt = torch.randn_like(x3_b, device=device, dtype=torch.float32)

                for t_step in range(NUM_STEPS - 1, 0, -1):
                    alpha_t = alphas[t_step - 1]
                    cumalpha_t = cumalphas[t_step - 1]
                    sigma = torch.sqrt(betas[t_step - 1])
                    factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                    t_tensor = torch.full(
                        (batch_size_actual,),
                        t_step,
                        device=device,
                        dtype=torch.float32,
                    )

                    pred_noise = denoiser(
                        xt,
                        [x1_b, x2_b],
                        dt_hist,
                        t_tensor,
                        dt_target,
                    )
                    z = torch.randn_like(xt) if t_step > 1 else torch.zeros_like(xt)
                    xt = 1 / torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z

                batch_np = xt.float().cpu().numpy().astype(np.float32)
                if args.save_original_scale:
                    batch_np = scaler.inverse_transform(batch_np).astype(np.float32)
                batch_samples.append(batch_np)

            batch_samples = np.stack(batch_samples, axis=1).astype(np.float32)
            samples_mm[write_idx:write_idx + batch_size_actual] = batch_samples
            write_idx += batch_size_actual

            batch_time = time.time() - batch_start
            num_test_samples += batch_size_actual
            print(
                f"  Batch {batch_idx + 1:3d} | Test samples: {num_test_samples:4d} | batch: {batch_time:6.1f}s",
                flush=True,
            )

            gc.collect()
            torch.cuda.empty_cache()

        samples_mm.flush()
        dt_mm.flush()
        del samples_mm
        del dt_mm

        prediction_files[label] = os.path.basename(samples_file)
        dt_target_files[label] = os.path.basename(dt_file)

        print(f"  ✓ Time {label}: {n_test_total} test × {NUM_SAMPLES} samples × {l_model} VF")


with open(f"{output_prefix}_meta.json", "w") as f_meta:
    json.dump(
        {
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "task": "VF1,VF2 -> VF3 at offsets from VF2",
            "time_labels": TIME_LABELS,
            "time_months": TIME_MONTHS.tolist(),
            "time_years": TIME_YEARS.tolist(),
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "vf_locations": int(l_model),
            "n_test": int(n_test_total),
            "save_original_scale": args.save_original_scale,
            "prediction_files": prediction_files,
            "dt_target_files": dt_target_files,
            "actual_vf3_file": os.path.basename(actual_vf3_file),
            "actual_dt_from_vf1_file": os.path.basename(actual_dt_from_vf1_file),
            "actual_dt_from_vf2_file": os.path.basename(actual_dt_from_vf2_file),
            "cond_vfs_file": os.path.basename(cond_vfs_file),
            "cond_times_file": os.path.basename(cond_times_file),
        },
        f_meta,
        indent=2,
    )


print("\n" + "=" * 70)
print(f"Saved per-time samples under prefix: {output_prefix}_<time>_pred.npy")
print(f"Total time: {time.time() - start_time:.1f}s")
print("=" * 70)