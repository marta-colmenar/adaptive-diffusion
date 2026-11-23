"""
Generate 150 Samples per VF for Ablation Study Evaluation
----------------------------------------------------------
Loads trained ablation models and generates 150 samples per test sample.
For most models: primary task (VF1+VF2 -> VF3)
For n_cond=1 models: VF2 -> VF3 task only
Saves predictions and ground truth for later metric computation.

Large-scale generation: ~10-20 hours per model depending on hardware.
"""

import os, gc, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ablation_models_diffusion_wa import get_model, MODEL_REGISTRY
from model_diffusion_wa import f

# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate samples for ablation evaluation')
parser.add_argument('--model', type=str, required=True,
                    choices=list(MODEL_REGISTRY.keys()),
                    help='Model variant to evaluate')
parser.add_argument('--dataset', type=str, default='wa', choices=['wa', 'bern'],
                    help='Source dataset used for test CSV loading')
parser.add_argument('--model_dataset', type=str, default=None,
                    choices=['wa', 'bern', 'joint_orig', 'merged_G', 'merged_24-2', 'merged_242',
                             'merged_G_clipm30', 'merged_24-2_clipm30', 'merged_242_clipm30'],
                    help='Dataset domain of the trained checkpoint/scaler (defaults to --dataset)')
parser.add_argument('--vf_conversion', type=str, default='auto',
                    choices=['auto', 'none', '24toG', 'Gto24'],
                    help='VF geometry conversion mode between WA 24-2 and Bern G spaces')
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
parser.add_argument('--max_cond', type=int, default=2,
                    help='Maximum conditioning VFs')
parser.add_argument('--L', type=int, default=None,
                    help='Number of VF locations')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--model_path', type=str, default='/storage/homefs/mc21s017/',
                    help='Path to saved models')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/samples/',
                    help='Output directory for samples')
parser.add_argument('--pair_only', action='store_true',
                    help='Sample pair tasks (VF1->VF3 or VF2->VF3) using 3VF data')
parser.add_argument('--sample_task', type=int, default=2,
                    help='Sampling task: 1/2 for 3VF, 1/2/3 for 4VF')
args = parser.parse_args()

# -------------------------------------------------------------------------
# Helper function for augmented pair dataset
# -------------------------------------------------------------------------
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


def build_map_24_from_g(map_g_from_24):
    map_24_from_g = {k: [] for k in range(54)}
    for g_idx, source_24_list in map_g_from_24.items():
        for source_24 in source_24_list:
            map_24_from_g[source_24].append(g_idx)
    return map_24_from_g


MAP_24_FROM_G = build_map_24_from_g(DEFAULT_MAP_G_FROM_24_2)


def convert_24_to_g_batch(x_24):
    if x_24.shape[1] != 54:
        raise ValueError(f"24toG conversion requires input dim 54, got {x_24.shape[1]}")
    out = np.zeros((x_24.shape[0], 59), dtype=np.float32)
    for g_idx, src_24_idxs in DEFAULT_MAP_G_FROM_24_2.items():
        out[:, g_idx] = x_24[:, src_24_idxs].mean(axis=1)
    return out


def convert_g_to_24_batch(x_g):
    if x_g.shape[1] != 59:
        raise ValueError(f"Gto24 conversion requires input dim 59, got {x_g.shape[1]}")
    out = np.zeros((x_g.shape[0], 54), dtype=np.float32)
    for idx_24, src_g_idxs in MAP_24_FROM_G.items():
        if len(src_g_idxs) == 0:
            out[:, idx_24] = x_g[:, min(idx_24, x_g.shape[1] - 1)]
        else:
            out[:, idx_24] = x_g[:, src_g_idxs].mean(axis=1)
    return out


def normalize_model_dataset_name(model_dataset):
    if model_dataset == 'merged_242':
        return 'merged_24-2'
    if model_dataset == 'merged_242_clipm30':
        return 'merged_24-2_clipm30'
    return model_dataset


def model_domain_geometry(model_dataset):
    if model_dataset == 'joint_orig':
        return 'mixed'
    if model_dataset in ('bern', 'merged_G', 'merged_G_clipm30'):
        return 'G'
    if model_dataset in ('wa', 'merged_24-2', 'merged_24-2_clipm30'):
        return '24-2'
    raise ValueError(f"Unsupported model_dataset for geometry: {model_dataset}")


def resolve_conversion_mode(mode, data_dataset, model_dataset):
    if model_dataset == 'joint_orig':
        return 'none'
    if mode != 'auto':
        return mode
    data_geometry = '24-2' if data_dataset == 'wa' else 'G'
    model_geometry = model_domain_geometry(model_dataset)
    if data_geometry == model_geometry:
        return 'none'
    if data_geometry == '24-2' and model_geometry == 'G':
        return '24toG'
    if data_geometry == 'G' and model_geometry == '24-2':
        return 'Gto24'
    raise ValueError(f"Cannot resolve conversion mode for data={data_dataset}, model={model_dataset}")


def build_pair_dataset(proc_tuple):
    x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = proc_tuple
    cond = torch.cat([x1_b, x1_b, x2_b], dim=0)
    target = torch.cat([x2_b, x3_b, x3_b], dim=0)
    dt_future = torch.cat([t2_b - t1_b, t3_b - t1_b, t3_b - t2_b], dim=0)
    return cond, target, dt_future

def build_pair_dataset_vf1_only(proc_tuple):
    """Build pair dataset with only VF2->VF3 task (n_cond=1)."""
    x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = proc_tuple
    # Only task 3: x2 -> x3 (conditioning on x2, predicting x3)
    cond = x2_b
    target = x3_b
    dt_future = t3_b - t2_b
    return cond, target, dt_future

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
MODEL_DATASET_NAME = normalize_model_dataset_name(args.model_dataset if args.model_dataset is not None else args.dataset)
CLIP_MIN_VALUE = -30.0 if 'clipm30' in MODEL_DATASET_NAME else None
DATA_PATH = args.data_path
MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset registry (same as in training_ablation_wa.py)
DATASETS = {
    "wa": {
        "train_files": {"3vf": "df_train_wa_3vf_augmented.csv", "4vf": "df_train_wa_4vf_augmented.csv"},
        "test_files":  {"3vf": "df_test_wa_3vf_normal.csv",    "4vf": "df_test_wa_4vf_normal.csv"},
        "scaler_file": "scaler_wa_3vf.npy",
        "vf_size": 54,
        "exclude_cols": [25, 34],
        "age_scale": 365.0,
    },
    "bern": {
        "train_files": {"3vf": "df_train_bern_3vf_augmented_new.csv", "4vf": "df_train_bern_4vf_normal.csv"},
        "test_files":  {"3vf": "df_test_bern_3vf_normal_new.csv",  "4vf": "df_test_bern_4vf_normal.csv"},
        "scaler_file": "scaler_bern_3vf_new.npy",
        "vf_size": 59,
        "exclude_cols": [],
        "age_scale": 1.0,
    },
}

dataset_cfg = DATASETS[DATASET_NAME]

MODEL_DATASET_DOMAINS = {
    'wa': DATASETS['wa'],
    'bern': DATASETS['bern'],
    'joint_orig': {
        # Placeholder values; for joint checkpoints we use dataset-specific scaler
        # selected by --dataset (handled below in data loading).
        'scaler_file': DATASETS['bern']['scaler_file'],
        'vf_size': DATASETS['bern']['vf_size'],
        'exclude_cols': DATASETS['bern']['exclude_cols'],
        'age_scale': DATASETS['bern']['age_scale'],
    },
    'merged_G': {
        'scaler_file': DATASETS['bern']['scaler_file'],
        'vf_size': DATASETS['bern']['vf_size'],
        'exclude_cols': DATASETS['bern']['exclude_cols'],
        'age_scale': DATASETS['bern']['age_scale'],
    },
    'merged_G_clipm30': {
        'scaler_file': DATASETS['bern']['scaler_file'],
        'vf_size': DATASETS['bern']['vf_size'],
        'exclude_cols': DATASETS['bern']['exclude_cols'],
        'age_scale': DATASETS['bern']['age_scale'],
    },
    'merged_24-2': {
        'scaler_file': DATASETS['wa']['scaler_file'],
        'vf_size': DATASETS['wa']['vf_size'],
        'exclude_cols': DATASETS['wa']['exclude_cols'],
        'age_scale': DATASETS['wa']['age_scale'],
    },
    'merged_24-2_clipm30': {
        'scaler_file': DATASETS['wa']['scaler_file'],
        'vf_size': DATASETS['wa']['vf_size'],
        'exclude_cols': DATASETS['wa']['exclude_cols'],
        'age_scale': DATASETS['wa']['age_scale'],
    },
}

if MODEL_DATASET_NAME not in MODEL_DATASET_DOMAINS:
    raise ValueError(f"Unsupported --model_dataset={MODEL_DATASET_NAME}")

model_dataset_cfg = MODEL_DATASET_DOMAINS[MODEL_DATASET_NAME]
CONVERSION_MODE = resolve_conversion_mode(args.vf_conversion, DATASET_NAME, MODEL_DATASET_NAME)
MODEL_N_VF = args.max_cond + 1
DATA_N_VF = 3 if args.pair_only else MODEL_N_VF
VF_KEY = f"{DATA_N_VF}vf"
use_xy_var = MODEL_NAME == 'diffusion_xy_var'

if args.pair_only and args.max_cond != 1:
    raise ValueError("--pair_only requires --max_cond=1")

if MODEL_DATASET_NAME == 'joint_orig' and not use_xy_var:
    raise ValueError("--model_dataset joint_orig is supported only with --model diffusion_xy_var")

if args.pair_only:
    args.sample_task = 1
    valid_tasks = (1,)
else:
    valid_tasks = (1, 2) if DATA_N_VF == 3 else (1, 2, 3)
    if args.sample_task not in valid_tasks:
        raise ValueError(f"Invalid --sample_task={args.sample_task} for {DATA_N_VF}VF; allowed: {valid_tasks}")

if CONVERSION_MODE == '24toG' and dataset_cfg['vf_size'] != 54:
    raise ValueError(f"24toG expects source vf_size=54 but got {dataset_cfg['vf_size']} ({DATASET_NAME})")
if CONVERSION_MODE == 'Gto24' and dataset_cfg['vf_size'] != 59:
    raise ValueError(f"Gto24 expects source vf_size=59 but got {dataset_cfg['vf_size']} ({DATASET_NAME})")

# -------------------------------------------------------------------------
# Dataset processing
# -------------------------------------------------------------------------
def process_df(df, scaler, source_cfg, target_cfg, n_vf: int, conversion_mode: str = 'none', clip_min_value=None):
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    source_vf_size = source_cfg["vf_size"]
    target_exclude_cols = target_cfg.get("exclude_cols", [])

    X_list = [arr[:, i * source_vf_size:(i + 1) * source_vf_size] for i in range(n_vf)]

    if conversion_mode == '24toG':
        X_list = [convert_24_to_g_batch(Xi) for Xi in X_list]
    elif conversion_mode == 'Gto24':
        X_list = [convert_g_to_24_batch(Xi) for Xi in X_list]

    if target_exclude_cols:
        X_list = [np.delete(Xi, target_exclude_cols, axis=1) for Xi in X_list]

    if clip_min_value is not None:
        X_list = [np.clip(Xi, clip_min_value, None) for Xi in X_list]

    X_list = [scaler.transform(Xi) for Xi in X_list]
    age_scale = source_cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f'age_{i}'].values.astype(np.float32)/age_scale) for i in range(n_vf)]

    X_tensors = [torch.from_numpy(Xi.astype(np.float32)) for Xi in X_list]
    return tuple(X_tensors + t_list)


def process_df_padded(df, scaler, source_cfg, n_vf: int, l_target: int, clip_min_value=None):
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    vf_size = source_cfg["vf_size"]
    exclude_cols = source_cfg.get("exclude_cols", [])

    x_list = [arr[:, i * vf_size:(i + 1) * vf_size] for i in range(n_vf)]
    if exclude_cols:
        x_list = [np.delete(xi, exclude_cols, axis=1) for xi in x_list]

    if clip_min_value is not None:
        x_list = [np.clip(xi, clip_min_value, None) for xi in x_list]

    x_list = [scaler.transform(xi) for xi in x_list]
    x_list = [
        np.pad(xi, ((0, 0), (0, l_target - xi.shape[1])), mode='constant') if xi.shape[1] < l_target else xi
        for xi in x_list
    ]

    age_scale = source_cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f'age_{i}'].values.astype(np.float32) / age_scale) for i in range(n_vf)]
    x_tensors = [torch.from_numpy(xi.astype(np.float32)) for xi in x_list]
    return tuple(x_tensors + t_list)


def load_xy_coords(dataset_name, data_path, cfg):
    if dataset_name == 'wa':
        xy_file = os.path.join(data_path, 'p24d2.csv')
        df_xy = pd.read_csv(xy_file)
        x_coords = df_xy['x'].values.astype(np.float32)
        y_coords = df_xy['y'].values.astype(np.float32)
        if cfg.get("exclude_cols"):
            x_coords = np.delete(x_coords, cfg["exclude_cols"])
            y_coords = np.delete(y_coords, cfg["exclude_cols"])
    elif dataset_name == 'bern':
        xy_file = os.path.join(data_path, 'x_y.xlsx')
        df_xy = pd.read_excel(xy_file)
        x_coords = df_xy.iloc[:, 0].values.astype(np.float32)
        y_coords = df_xy.iloc[:, 1].values.astype(np.float32)
    else:
        raise ValueError(f"Unknown dataset for XY coords: {dataset_name}")
    return x_coords, y_coords


def make_coord_tensors(n_samples, x_coords, y_coords, l_target):
    l_src = len(x_coords)
    x_pad = np.zeros((n_samples, l_target), dtype=np.float32)
    y_pad = np.zeros((n_samples, l_target), dtype=np.float32)
    valid = np.zeros((n_samples, l_target), dtype=np.float32)
    x_pad[:, :l_src] = x_coords[None, :]
    y_pad[:, :l_src] = y_coords[None, :]
    valid[:, :l_src] = 1.0
    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(y_pad),
        torch.from_numpy(valid),
    ), l_src


def make_dataset_id_tensor(n_samples, dataset_id):
    return torch.full((n_samples,), dataset_id, dtype=torch.long)

# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
print("=" * 70)
print(f"Generating {NUM_SAMPLES} Samples per Test Sample")
print(f"Model: {MODEL_NAME} | Data dataset: {DATASET_NAME} | Model dataset: {MODEL_DATASET_NAME}")
print(f"VF conversion: {CONVERSION_MODE}")
print(f"Test-set clipping: min={CLIP_MIN_VALUE}" if CLIP_MIN_VALUE is not None else "Test-set clipping: disabled")
print("=" * 70)
print("Loading data...", flush=True)

test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"][VF_KEY])
# For joint_orig checkpoints we use the selected dataset scaler (WA or Bern),
# matching how each source was scaled during joint training.
if MODEL_DATASET_NAME == 'joint_orig':
    scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])
else:
    scaler_file = os.path.join(DATA_PATH, model_dataset_cfg["scaler_file"])

df_test = pd.read_csv(test_file)
scaler = np.load(scaler_file, allow_pickle=True).item()

data_valid_l = None
if use_xy_var:
    l_target = args.L if args.L is not None else 59
    proc_test = process_df_padded(
        df_test,
        scaler,
        dataset_cfg,
        DATA_N_VF,
        l_target=l_target,
        clip_min_value=CLIP_MIN_VALUE,
    )
    x_coords, y_coords = load_xy_coords(DATASET_NAME, DATA_PATH, dataset_cfg)
    coord_tensors, data_valid_l = make_coord_tensors(len(df_test), x_coords, y_coords, l_target)
    if MODEL_DATASET_NAME == 'joint_orig':
        dataset_id = 0 if DATASET_NAME == 'wa' else 1
        dataset_id_tensor = make_dataset_id_tensor(len(df_test), dataset_id)
        proc_test = proc_test + (dataset_id_tensor,) + coord_tensors
    else:
        dataset_id = 0 if DATASET_NAME == 'wa' else 1
        dataset_id_tensor = make_dataset_id_tensor(len(df_test), dataset_id)
        proc_test = proc_test + (dataset_id_tensor,) + coord_tensors
else:
    proc_test = process_df(
        df_test,
        scaler,
        dataset_cfg,
        model_dataset_cfg,
        DATA_N_VF,
        conversion_mode=CONVERSION_MODE,
        clip_min_value=CLIP_MIN_VALUE,
    )

if args.pair_only:
    if args.max_cond == 1:
        proc_test = build_pair_dataset_vf1_only(proc_test)
    else:
        proc_test = build_pair_dataset(proc_test)
L_data = proc_test[0].shape[1]
L_requested = args.L if args.L is not None else L_data

test_ds = TensorDataset(*proc_test)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

print(f"Test samples: {len(test_ds)}")
print(f"VF locations (data): {L_data}, requested model L: {L_requested}")

# -------------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------------
print(f"\nLoading trained model: {MODEL_NAME}...", flush=True)

checkpoint_path = os.path.join(MODEL_PATH, f"denoiser_ablation_{MODEL_NAME}_{MODEL_DATASET_NAME}_{VF_KEY}.pth")
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"Checkpoint loaded from epoch {checkpoint['epoch']}, best FID: {checkpoint['best_fid']:.4f}")


def infer_checkpoint_l(state_dict):
    for key in ("loc", "module.loc"):
        if key in state_dict and state_dict[key].ndim == 1:
            return int(state_dict[key].shape[0])
    for key, value in state_dict.items():
        if key.endswith(".loc") and value.ndim == 1:
            return int(value.shape[0])
    return None


checkpoint_l = infer_checkpoint_l(checkpoint['model_state_dict'])
L_model = L_requested
if checkpoint_l is not None and L_model != checkpoint_l:
    print(
        f"WARNING: requested L={L_model} but checkpoint expects L={checkpoint_l}. "
        f"Using checkpoint L={checkpoint_l} to avoid shape mismatch.",
        flush=True,
    )
    L_model = checkpoint_l
elif checkpoint_l is not None:
    print(f"Model L resolved to checkpoint L={checkpoint_l}", flush=True)

# Build model with same parameters as training
model_kwargs = {'L': L_model, 'D': args.D, 'max_cond': args.max_cond}
if MODEL_NAME in ['diffusion', 'diffusion_xy', 'diffusion_xy_var', 'diffusion_loc']:
    model_kwargs['K'] = args.K

# load coordinates files
if MODEL_NAME == 'diffusion_xy':
    if DATASET_NAME == 'wa':
        xy_file = os.path.join(DATA_PATH, 'p24d2.csv')
        df_xy = pd.read_csv(xy_file)
        x_coords = df_xy['x'].values.astype(np.float32)
        y_coords = df_xy['y'].values.astype(np.float32)
        # exclude exclude_cols
        if dataset_cfg.get("exclude_cols"):
            exclude_cols = dataset_cfg["exclude_cols"]
            x_coords = np.delete(x_coords, exclude_cols)
            y_coords = np.delete(y_coords, exclude_cols)
    elif DATASET_NAME == 'bern':
        xy_file = os.path.join(DATA_PATH, 'x_y.xlsx')
        df_xy = pd.read_excel(xy_file)
        x_coords = df_xy.iloc[:, 0].values.astype(np.float32)  # V1 column
        y_coords = df_xy.iloc[:, 1].values.astype(np.float32)  # V2 column
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")
    model_kwargs['x_coords'] = x_coords
    model_kwargs['y_coords'] = y_coords

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
# Sampling Loop (VF1+VF2 -> VF3, primary task; or VF2->VF3 for n_cond=1)
# -------------------------------------------------------------------------
print(f"\nGenerating {NUM_SAMPLES} samples per test sample...")
print("=" * 70)

all_samples = []      # Shape: (num_test_samples, num_samples, L)
all_reals = []        # Shape: (num_test_samples, L)
all_cond_vfs = []     # Shape: (num_test_samples, max_cond, L) - conditioning VFs
all_cond_times = []   # Shape: (num_test_samples, max_cond+1) - conditioning times + horizon
num_test_samples = 0

start_time = time.time()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        x_coords_b = None
        y_coords_b = None
        valid_mask_b = None
        dataset_id_b = None
        core_batch = batch
        if use_xy_var:
            if MODEL_DATASET_NAME == 'joint_orig':
                dataset_id_b = batch[-4].to(device, non_blocking=True)
                x_coords_b = batch[-3].to(device, non_blocking=True)
                y_coords_b = batch[-2].to(device, non_blocking=True)
                valid_mask_b = batch[-1].to(device, non_blocking=True)
                core_batch = batch[:-4]
            else:
                x_coords_b = batch[-3].to(device, non_blocking=True)
                y_coords_b = batch[-2].to(device, non_blocking=True)
                valid_mask_b = batch[-1].to(device, non_blocking=True)
                core_batch = batch[:-3]

        batch_start = time.time()
        if args.pair_only:
            cond_b, target_b, dt_future_b = core_batch
            batch_size_actual = target_b.size(0)
            cond_b = cond_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            dt_future_b = dt_future_b.to(device, non_blocking=True)
            target = target_b
        elif DATA_N_VF == 3:
            x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = core_batch
            batch_size_actual = x1_b.size(0)
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            t_diff_1 = (t3_b - t1_b).to(device, non_blocking=True)
            t_diff_2 = (t3_b - t2_b).to(device, non_blocking=True)
            target = x3_b
            if args.sample_task == 1:
                target = x2_b
                cond_v = [x1_b]
                cond_dt = [torch.zeros_like(t_diff_2)]
                dt_future = t_diff_2
            else:
                target = x3_b
                cond_v = [x1_b, x2_b]
                cond_dt = [t_diff_1, t_diff_2]
                dt_future = torch.zeros_like(t_diff_2)
                locs = [torch.arange(args.L, device=device).float() for _ in range(3)]
        else:
            x1_b, x2_b, x3_b, x4_b, t1_b, t2_b, t3_b, t4_b = core_batch
            batch_size_actual = x1_b.size(0)
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            x4_b = x4_b.to(device, non_blocking=True)
            dt_2 = (t2_b - t1_b).to(device, non_blocking=True)
            dt_3 = (t3_b - t1_b).to(device, non_blocking=True)
            dt_4 = (t4_b - t1_b).to(device, non_blocking=True)
            target = x4_b
            if args.sample_task == 1:
                cond_v = [x1_b]
                cond_dt = [torch.zeros_like(dt_4)]
                dt_future = dt_4
            elif args.sample_task == 2:
                cond_v = [x1_b, x2_b]
                cond_dt = [torch.zeros_like(dt_4), dt_2]
                dt_future = dt_4
            else:
                cond_v = [x1_b, x2_b, x3_b]
                cond_dt = [torch.zeros_like(dt_4), dt_2, dt_3]
                dt_future = dt_4
        
        # Ground truth
        all_reals.append(target.cpu().numpy())
        
        # Save conditioning VFs (for all models)
        # Store as (batch_size, num_cond_vfs, L)
        if args.pair_only:
            # For pair_only, cond_b shape is (batch_size, L), expand to (batch_size, 1, L)
            batch_cond = cond_b.cpu().numpy()[:, np.newaxis, :]
            all_cond_vfs.append(batch_cond)
            # Save times: conditioning time + horizon (dt_future)
            batch_cond_times = np.zeros((batch_size_actual, 1), dtype=np.float32)
            batch_horizon = dt_future_b.cpu().numpy()[:, np.newaxis]
            batch_times_with_horizon = np.concatenate([batch_cond_times, batch_horizon], axis=1)
            all_cond_times.append(batch_times_with_horizon)
        elif DATA_N_VF == 3:
            # Stack along axis 1: (batch_size, num_cond_vfs, L)
            batch_cond = np.stack([v.cpu().numpy() for v in cond_v], axis=1)
            all_cond_vfs.append(batch_cond)
            # Stack times: (batch_size, num_cond_vfs) + horizon (batch_size, 1)
            batch_cond_times = np.stack([t.cpu().numpy() for t in cond_dt], axis=1)
            batch_horizon = dt_future.cpu().numpy()[:, np.newaxis]
            batch_times_with_horizon = np.concatenate([batch_cond_times, batch_horizon], axis=1)
            all_cond_times.append(batch_times_with_horizon)
        else:
            # Stack along axis 1: (batch_size, num_cond_vfs, L)
            batch_cond = np.stack([v.cpu().numpy() for v in cond_v], axis=1)
            all_cond_vfs.append(batch_cond)
            # Stack times: (batch_size, num_cond_vfs) + horizon (batch_size, 1)
            batch_cond_times = np.stack([t.cpu().numpy() for t in cond_dt], axis=1)
            batch_horizon = dt_future.cpu().numpy()[:, np.newaxis]
            batch_times_with_horizon = np.concatenate([batch_cond_times, batch_horizon], axis=1)
            all_cond_times.append(batch_times_with_horizon)
        
        batch_samples = []
        
        for sample_idx in range(NUM_SAMPLES):
            xt = torch.randn_like(target, device=device)
            if use_xy_var:
                xt = xt * valid_mask_b
            
            for t_step in range(NUM_STEPS - 1, 0, -1):
                alpha_t = alphas[t_step - 1]
                cumalpha_t = cumalphas[t_step - 1]
                sigma = torch.sqrt(betas[t_step - 1])
                factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                t_tensor = torch.tensor([t_step], device=device).expand(batch_size_actual)
                
                if args.pair_only:
                    if use_xy_var:
                        pred_noise = denoiser(
                            xt, [cond_b], [torch.zeros_like(dt_future_b)], t_tensor, dt_future_b,
                            x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                            dataset_id=dataset_id_b,
                        )
                    else:
                        pred_noise = denoiser(xt, [cond_b], [torch.zeros_like(dt_future_b)], t_tensor, dt_future_b)
                else:
                    if use_xy_var:
                        pred_noise = denoiser(
                            xt, cond_v, cond_dt, t_tensor, dt_future,
                            x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                            dataset_id=dataset_id_b,
                        )
                    else:
                        pred_noise = denoiser(xt, cond_v, cond_dt, t_tensor, dt_future, locs)
                z = torch.randn_like(xt) if t_step > 1 else torch.zeros_like(xt)
                xt = 1/torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z
                if use_xy_var:
                    xt = xt * valid_mask_b
            
            batch_samples.append(xt.cpu().numpy())
        
        # Stack samples: (batch_size, num_samples, L)
        batch_samples = np.stack(batch_samples, axis=1)
        all_samples.append(batch_samples)
        
        batch_time = time.time() - batch_start
        elapsed_total = time.time() - start_time
        num_test_samples += batch_size_actual
        avg_time_per_patient_sample = batch_time / (batch_size_actual * NUM_SAMPLES)
        est_time_for_one_sample_per_patient = elapsed_total / NUM_SAMPLES
        
        print(f"Batch {batch_idx+1:3d} | Test samples: {num_test_samples:4d} | "
              f"Batch time: {batch_time:6.1f}s | Total time: {elapsed_total:7.1f}s | "
              f"Avg: {avg_time_per_patient_sample*1000:7.2f} ms/sample/patient | "
              f"Est. 1 sample/patient: {est_time_for_one_sample_per_patient:7.1f}s total", flush=True)
        
        gc.collect()
        torch.cuda.empty_cache()

# Concatenate all batches
all_samples = np.concatenate(all_samples, axis=0)  # (n_test, num_samples, L)
all_reals = np.concatenate(all_reals, axis=0)      # (n_test, L)

if use_xy_var and data_valid_l is not None and data_valid_l < all_samples.shape[2]:
    all_samples = all_samples[:, :, :data_valid_l]
    all_reals = all_reals[:, :data_valid_l]
    if len(all_cond_vfs) > 0:
        all_cond_vfs = [arr[:, :, :data_valid_l] for arr in all_cond_vfs]

print("\n" + "=" * 70)
print(f"Sampling complete!")
total_time = time.time() - start_time
print(f"Total time: {total_time:.1f}s")
print(f"Estimated time for 1 sample per patient: {total_time / NUM_SAMPLES:.1f}s")
print(f"Average time per patient for 1 sample: {((total_time / max(num_test_samples, 1)) / NUM_SAMPLES) * 1000:.2f} ms")
print(f"Test samples: {all_samples.shape[0]}")
print(f"Samples per test: {all_samples.shape[1]}")
print(f"VF dimension: {all_samples.shape[2]}")

# -------------------------------------------------------------------------
# Save samples
# -------------------------------------------------------------------------
task_tag = f"task{args.sample_task}"
# output_prefix = os.path.join(OUTPUT_DIR, f"samples_{MODEL_NAME}_{DATASET_NAME}_{task_tag}")
if DATASET_NAME == MODEL_DATASET_NAME and CONVERSION_MODE == 'none':
    output_prefix = os.path.join(OUTPUT_DIR, f"samples_{MODEL_NAME}_{DATASET_NAME}")
else:
    output_prefix = os.path.join(
        OUTPUT_DIR,
        f"samples_{MODEL_NAME}_{MODEL_DATASET_NAME}_on_{DATASET_NAME}_{CONVERSION_MODE}"
    )


samples_file = f"{output_prefix}_pred.npy"
reals_file = f"{output_prefix}_real.npy"

np.save(samples_file, all_samples.astype(np.float32))
np.save(reals_file, all_reals.astype(np.float32))

print(f"\n✓ Saved {NUM_SAMPLES*all_samples.shape[0]} total samples ({all_samples.shape[0]} test × {NUM_SAMPLES} samples)")
print(f"  Predictions: {samples_file} ({all_samples.nbytes / 1e9:.2f} GB)")
print(f"  Ground truth: {reals_file} ({all_reals.nbytes / 1e9:.2f} GB)")

# Save conditioning VFs
if len(all_cond_vfs) > 0:
    all_cond_vfs = np.concatenate(all_cond_vfs, axis=0)
    cond_file = f"{output_prefix}_cond.npy"
    np.save(cond_file, all_cond_vfs.astype(np.float32))
    print(f"  Conditioning VFs: {cond_file} ({all_cond_vfs.nbytes / 1e9:.2f} GB)")
    print(f"    Shape: {all_cond_vfs.shape}")
    
    # Save conditioning times (including horizon)
    all_cond_times = np.concatenate(all_cond_times, axis=0)
    cond_times_file = f"{output_prefix}_cond_times.npy"
    np.save(cond_times_file, all_cond_times.astype(np.float32))
    print(f"  Conditioning times + horizon: {cond_times_file} ({all_cond_times.nbytes / 1e9:.2f} GB)")
    print(f"    Shape: {all_cond_times.shape} (last column is horizon)")

print("=" * 70)
