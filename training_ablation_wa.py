"""
Ablation Study Training Script
-------------------------------
Trains different model variants (baseline, spatial, self_attn, cross_attn, full)
for systematic evaluation of each component's contribution.
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

# -------------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train ablation models')
parser.add_argument('--model', type=str, required=True, 
                    choices=list(MODEL_REGISTRY.keys()),
                    help='Model variant to train')
parser.add_argument('--dataset', type=str, default='wa', choices=['wa', 'bern', 'joint_orig', 'oct_vf', 'oct_vf4', 'vf_small'],
                    help='Dataset variant (controls files and preprocessing)')
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
parser.add_argument('--L', type=int, default=None,
                    help='Number of VF locations (overrides dataset default)')
parser.add_argument('--K', type=int, default=2,
                    help='Number of attention blocks')
parser.add_argument('--max_cond', type=int, default=2,
                    help='Maximum conditioning VFs')
parser.add_argument('--data_path', type=str, default='/storage/homefs/mc21s017/data_vf/',
                    help='Path to data directory')
parser.add_argument('--output_dir', type=str, default='/storage/homefs/mc21s017/',
                    help='Output directory for models')
parser.add_argument('--train_task', type=int, default=0,
                    help='Train only a specific task (0=all tasks, 1/2/3 for a single task)')
parser.add_argument('--pair_only', action='store_true',
                    help='Augment 3VF data into pair samples and train a single pair task')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable wandb logging')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='Path to pretrained model checkpoint for initialization')
parser.add_argument('--name-model-tag', type=str, default=None,
                    help='Optional tag to append to model name in wandb')

args = parser.parse_args()

# Dataset registry
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
        "vf_size": 54,  # raw VF size before exclusions
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
    "oct_vf": {
        "train_files": {
            "3vf": "df_train_oct_vf.parquet",
        },
        "test_files": {
            "3vf": "df_val_oct_vf.parquet",
        },
        "scaler_file": "scaler_bern_3vf_new.npy",
        "vf_size": 64,
        "exclude_cols": [],
        "age_scale": 365.0,
    },
    "oct_vf4": {
        "train_files": {
            "3vf": "df_train_oct_vf.parquet",
        },
        "test_files": {
            "3vf": "df_val_oct_vf.parquet",
        },
        "scaler_file": "scaler_bern_3vf_new.npy",
        "vf_size": 64,
        "exclude_cols": [],
        "age_scale": 365.0,
    },
    "vf_small": {
        "train_files": {
            "3vf": "df_train_oct_vf.parquet",
        },
        "test_files": {
            "3vf": "df_val_oct_vf.parquet",
        },
        "scaler_file": "scaler_bern_3vf_new.npy",
        "vf_size": 64,
        "exclude_cols": [],
        "age_scale": 365.0,
    }
}

use_xy_var = args.model == 'diffusion_xy_var'

if args.dataset == 'joint_orig':
    if not use_xy_var:
        raise ValueError("--dataset joint_orig requires --model diffusion_xy_var")
    dataset_cfg = None
else:
    dataset_cfg = DATASETS[args.dataset]
MODEL_N_VF = args.max_cond + 1  # number of conditioning snapshots + target
DATA_N_VF = 3 if args.pair_only else MODEL_N_VF
VF_KEY = f"{DATA_N_VF}vf"

if args.pair_only and args.max_cond != 1:
    raise ValueError("--pair_only requires --max_cond=1")

if args.pair_only and use_xy_var:
    raise ValueError("--pair_only is not supported yet with --model diffusion_xy_var")

if args.pair_only:
    valid_tasks = (0, 1)
else:
    valid_tasks = (0, 1, 2) if MODEL_N_VF == 3 else (0, 1, 2, 3)
if args.train_task not in valid_tasks:
    raise ValueError(f"Invalid --train_task={args.train_task} for {MODEL_N_VF}VF; allowed: {valid_tasks}")

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LOG_EVERY = args.log_every
EVAL_EVERY = args.eval_every
LR = args.lr
MODEL_NAME = args.model
DATASET_NAME = args.dataset
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir

if DATASET_NAME == 'joint_orig':
    L_DEFAULT = 59
else:
    L_DEFAULT = dataset_cfg["vf_size"] - len(dataset_cfg.get("exclude_cols", []))
L_PLANNED = args.L if args.L is not None else L_DEFAULT

if args.pretrained_model:
    MODEL_PATH = os.path.join(OUTPUT_DIR, f"denoiser_ablation_{MODEL_NAME}_{DATASET_NAME}_{VF_KEY}_pretrained.pth")
else:
    if args.name_model_tag:
        MODEL_PATH = os.path.join(OUTPUT_DIR, f"denoiser_ablation_{MODEL_NAME}_{DATASET_NAME}_{VF_KEY}_{args.name_model_tag}.pth")
    else:
        MODEL_PATH = os.path.join(OUTPUT_DIR, f"denoiser_ablation_{MODEL_NAME}_{DATASET_NAME}_{VF_KEY}.pth")
best_fid = float('inf')

slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

# -------------------------------------------------------------------------
# W&B init
# -------------------------------------------------------------------------
if not args.no_wandb:
    wandb.init(
        entity="marta-colmenar-h",
        project="glaucoma-diffusion-ablation",
        name=(
            f"ablation_{MODEL_NAME}_{DATASET_NAME}_{slurm_job_id}"
            if slurm_job_id else f"ablation_{MODEL_NAME}_{DATASET_NAME}"
        ),
        tags=["ablation", MODEL_NAME, DATASET_NAME, f"{DATA_N_VF}VF"],
        config={
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_steps": NUM_STEPS,
            "L": L_PLANNED,
            "D": args.D,
            "K": args.K,
            "max_cond": args.max_cond,
            "data_n_vf": DATA_N_VF,
            "pair_only": args.pair_only,
        },
        resume="never" 
    )

# -------------------------------------------------------------------------
# Dataset processing
# -------------------------------------------------------------------------


def process_df(df, scaler, cfg, n_vf: int):
    X_list = extract_vf_arrays(df, cfg, n_vf)

    # Scale each VF snapshot
    X_list = [scaler.transform(Xi) for Xi in X_list]
    age_scale = cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f'age_{i}'].values.astype(np.float32)/age_scale) for i in range(n_vf)]

    X_tensors = [torch.from_numpy(Xi.astype(np.float32)) for Xi in X_list]
    return tuple(X_tensors + t_list)


def process_df_padded(df, scaler, cfg, n_vf: int, l_target: int):
    x_list = extract_vf_arrays(df, cfg, n_vf)

    x_list = [scaler.transform(xi) for xi in x_list]
    x_list = [
        np.pad(xi, ((0, 0), (0, l_target - xi.shape[1])), mode='constant') if xi.shape[1] < l_target else xi
        for xi in x_list
    ]

    age_scale = cfg.get("age_scale", 1.0)
    t_list = [torch.from_numpy(df[f'age_{i}'].values.astype(np.float32) / age_scale) for i in range(n_vf)]
    x_tensors = [torch.from_numpy(xi.astype(np.float32)) for xi in x_list]
    return tuple(x_tensors + t_list)


def extract_vf_arrays(df, cfg, n_vf: int):
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    vf_size = cfg["vf_size"]
    exclude_cols = cfg.get("exclude_cols", [])

    x_list = [arr[:, i * vf_size:(i + 1) * vf_size] for i in range(n_vf)]
    if exclude_cols:
        x_list = [np.delete(xi, exclude_cols, axis=1) for xi in x_list]
    return x_list


def drop_missing_triplet_rows(df, n_vf: int):
    """Remove rows with missing values in any triplet field used by training."""
    required_cols = []
    for i in range(n_vf):
        required_cols.extend([f'age_{i}', f'MD_{i}', f'class_{i}', f'examination_{i}'])
    required_cols.extend([c for c in df.columns if c.startswith('TD_')])
    required_cols = [c for c in required_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Dropped {removed} rows with missing triplet values out of {before} total rows.")
    return df

def load_triplets(path, scaler):
    df = pd.read_parquet(path)

    def vf(name):
        cols = [c for c in df.columns if c.startswith(name + "_deviation")]
        vf = (-1)*(df[cols].values)
        return torch.tensor(scaler.transform(vf)).float().cpu()
    
    def years_between(a, b):
        date_a = pd.to_datetime(df[a])
        date_b = pd.to_datetime(df[b])

        years = (date_b - date_a).dt.days / 365.25
        return torch.tensor(years.values, dtype=torch.float32).cpu()
    
    vf1_target_cols = [f"vf1_deviation_{i}" for i in range(59)]
    vf2_target_cols = [f"vf2_deviation_{i}" for i in range(59)]

    mask1 = df[vf1_target_cols].isna().all(axis=1)
    df = df[~mask1].reset_index(drop=True)
    mask2 = df[vf2_target_cols].isna().all(axis=1)
    df = df[~mask2].reset_index(drop=True)

    X_tensors = [vf(f"vf{i}") for i in range(1, 4)]
    t_list = [years_between("acquisition_date_3", f"acquisition_date_{i}") for i in range(1, 4)]

    return tuple(X_tensors + t_list)

def load_triplets_small(path, scaler):
    df = pd.read_parquet(path)

    def vf(name):
        cols = [c for c in df.columns if c.startswith(name + "_deviation")]
        vf = (-1)*(df[cols].values)
        return torch.tensor(scaler.transform(vf)).float().cpu()
    
    def years_between(a, b):
        date_a = pd.to_datetime(df[a])
        date_b = pd.to_datetime(df[b])

        years = (date_b - date_a).dt.days / 365.25
        return torch.tensor(years.values, dtype=torch.float32).cpu()

    X_tensors = [vf("vf1_target"), vf("vf2_target"), vf("vf3")]
    t_list = [years_between("acquisition_date_3", f"acquisition_date_{i}") for i in range(1, 4)]

    return tuple(X_tensors + t_list)

def load_triplets4(path, scaler):
    df = pd.read_parquet(path)

    def vf(name):
        cols = [c for c in df.columns if c.startswith(name + "_deviation")]
        vf = (-1) * df[cols].values
        return torch.tensor(scaler.transform(vf)).float().cpu()

    def years_between(a, b):
        date_a = pd.to_datetime(df[a])
        date_b = pd.to_datetime(df[b])

        years = (date_b - date_a).dt.days / 365.25
        return torch.tensor(years.values, dtype=torch.float32).cpu()
    
    vf1_target_cols = [f"vf1_deviation_{i}" for i in range(59)]
    vf2_target_cols = [f"vf2_deviation_{i}" for i in range(59)]

    mask1 = df[vf1_target_cols].isna().all(axis=1)
    df = df[~mask1].reset_index(drop=True)
    mask2 = df[vf2_target_cols].isna().all(axis=1)
    df = df[~mask2].reset_index(drop=True)

    X_tensors = [
        vf("vf1"),
        vf("vf1_target"),
        vf("vf2"),
        vf("vf2_target"),
        vf("vf3"),
    ]

    t_list = [
        years_between("acquisition_date_3", "acquisition_date_1"), 
        years_between("acquisition_date_3", "acquisition_date_1"),  
        years_between("acquisition_date_3", "acquisition_date_2"),
        years_between("acquisition_date_3", "acquisition_date_2"),
        years_between("acquisition_date_3", "acquisition_date_3"),
    ]

    return tuple(X_tensors + t_list)

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
    )


def make_dataset_id_tensor(n_samples, dataset_id):
    return torch.full((n_samples,), dataset_id, dtype=torch.long)


def concat_tuples(tuple_a, tuple_b):
    return tuple(torch.cat([a, b], dim=0) for a, b in zip(tuple_a, tuple_b))

def build_pair_dataset(proc_tuple):
    x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = proc_tuple
    cond = torch.cat([x1_b, x1_b, x2_b], dim=0)
    target = torch.cat([x2_b, x3_b, x3_b], dim=0)
    dt_future = torch.cat([t2_b - t1_b, t3_b - t1_b, t3_b - t2_b], dim=0)
    return cond, target, dt_future

# -------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------
print("=" * 60)
print(f"Training Ablation Model: {MODEL_NAME} | Dataset: {DATASET_NAME}")
print("=" * 60)
print("Loading data...", flush=True)
if DATASET_NAME == 'joint_orig':
    wa_cfg = DATASETS['wa']
    bern_cfg = DATASETS['bern']

    wa_train_file = os.path.join(DATA_PATH, wa_cfg["train_files"][VF_KEY])
    wa_test_file = os.path.join(DATA_PATH, wa_cfg["test_files"][VF_KEY])
    bern_train_file = os.path.join(DATA_PATH, bern_cfg["train_files"][VF_KEY])
    bern_test_file = os.path.join(DATA_PATH, bern_cfg["test_files"][VF_KEY])

    print(f"Dataset: joint_orig | WA Train: {wa_train_file} | WA Test: {wa_test_file}")
    print(f"Dataset: joint_orig | BERN Train: {bern_train_file} | BERN Test: {bern_test_file}")

    df_train_wa = pd.read_csv(wa_train_file, low_memory=False)
    df_test_wa = pd.read_csv(wa_test_file, low_memory=False)
    df_train_bern = pd.read_csv(bern_train_file, low_memory=False)
    df_test_bern = pd.read_csv(bern_test_file, low_memory=False)

    df_train_wa = drop_missing_triplet_rows(df_train_wa, DATA_N_VF)
    df_test_wa = drop_missing_triplet_rows(df_test_wa, DATA_N_VF)
    df_train_bern = drop_missing_triplet_rows(df_train_bern, DATA_N_VF)
    df_test_bern = drop_missing_triplet_rows(df_test_bern, DATA_N_VF)

    scaler_wa = np.load(os.path.join(DATA_PATH, wa_cfg["scaler_file"]), allow_pickle=True).item()
    scaler_bern = np.load(os.path.join(DATA_PATH, bern_cfg["scaler_file"]), allow_pickle=True).item()

    L_target = 59
    proc_train_wa = process_df_padded(df_train_wa, scaler_wa, wa_cfg, DATA_N_VF, L_target)
    proc_test_wa = process_df_padded(df_test_wa, scaler_wa, wa_cfg, DATA_N_VF, L_target)
    proc_train_bern = process_df_padded(df_train_bern, scaler_bern, bern_cfg, DATA_N_VF, L_target)
    proc_test_bern = process_df_padded(df_test_bern, scaler_bern, bern_cfg, DATA_N_VF, L_target)

    x_wa, y_wa = load_xy_coords('wa', DATA_PATH, wa_cfg)
    x_bern, y_bern = load_xy_coords('bern', DATA_PATH, bern_cfg)

    coords_train_wa = make_coord_tensors(len(df_train_wa), x_wa, y_wa, L_target)
    coords_test_wa = make_coord_tensors(len(df_test_wa), x_wa, y_wa, L_target)
    coords_train_bern = make_coord_tensors(len(df_train_bern), x_bern, y_bern, L_target)
    coords_test_bern = make_coord_tensors(len(df_test_bern), x_bern, y_bern, L_target)

    dataset_train_wa = make_dataset_id_tensor(len(df_train_wa), 0)
    dataset_test_wa = make_dataset_id_tensor(len(df_test_wa), 0)
    dataset_train_bern = make_dataset_id_tensor(len(df_train_bern), 1)
    dataset_test_bern = make_dataset_id_tensor(len(df_test_bern), 1)

    proc_train = concat_tuples(proc_train_wa + (dataset_train_wa,) + coords_train_wa, proc_train_bern + (dataset_train_bern,) + coords_train_bern)
    proc_test = concat_tuples(proc_test_wa + (dataset_test_wa,) + coords_test_wa, proc_test_bern + (dataset_test_bern,) + coords_test_bern)
elif DATASET_NAME == 'oct_vf':
    train_file = os.path.join(DATA_PATH, dataset_cfg["train_files"][VF_KEY])
    test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"][VF_KEY])
    scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

    print(f"Dataset: {DATASET_NAME} | Train: {train_file} | Test: {test_file}")

    proc_train = load_triplets(train_file, np.load(scaler_file, allow_pickle=True).item())
    proc_test = load_triplets(test_file, np.load(scaler_file, allow_pickle=True).item())
elif DATASET_NAME == 'vf_small':
    train_file = os.path.join(DATA_PATH, dataset_cfg["train_files"][VF_KEY])
    test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"][VF_KEY])
    scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

    print(f"Dataset: {DATASET_NAME} | Train: {train_file} | Test: {test_file}")

    proc_train = load_triplets_small(train_file, np.load(scaler_file, allow_pickle=True).item())
    proc_test = load_triplets_small(test_file, np.load(scaler_file, allow_pickle=True).item())
elif DATASET_NAME == 'oct_vf4':
    train_file = os.path.join(DATA_PATH, dataset_cfg["train_files"][VF_KEY])
    test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"][VF_KEY])
    scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

    print(f"Dataset: {DATASET_NAME} | Train: {train_file} | Test: {test_file}")

    proc_train = load_triplets4(train_file, np.load(scaler_file, allow_pickle=True).item())
    proc_test = load_triplets4(test_file, np.load(scaler_file, allow_pickle=True).item())
else:
    train_file = os.path.join(DATA_PATH, dataset_cfg["train_files"][VF_KEY])
    test_file = os.path.join(DATA_PATH, dataset_cfg["test_files"][VF_KEY])
    scaler_file = os.path.join(DATA_PATH, dataset_cfg["scaler_file"])

    print(f"Dataset: {DATASET_NAME} | Train: {train_file} | Test: {test_file}")

    df_train = pd.read_csv(train_file, low_memory=False)
    df_test = pd.read_csv(test_file, low_memory=False)
    df_train = drop_missing_triplet_rows(df_train, DATA_N_VF)
    df_test = drop_missing_triplet_rows(df_test, DATA_N_VF)
    scaler = np.load(scaler_file, allow_pickle=True).item()

    # Dynamic preprocessing based on N_VF
    proc_train = process_df(df_train, scaler, dataset_cfg, DATA_N_VF)
    proc_test = process_df(df_test, scaler, dataset_cfg, DATA_N_VF)

    if use_xy_var:
        x_coords, y_coords = load_xy_coords(DATASET_NAME, DATA_PATH, dataset_cfg)
        coords_train = make_coord_tensors(len(df_train), x_coords, y_coords, L_PLANNED)
        coords_test = make_coord_tensors(len(df_test), x_coords, y_coords, L_PLANNED)
        dataset_train = make_dataset_id_tensor(len(df_train), 0 if DATASET_NAME == 'wa' else 1)
        dataset_test = make_dataset_id_tensor(len(df_test), 0 if DATASET_NAME == 'wa' else 1)
        proc_train = proc_train + (dataset_train,) + coords_train
        proc_test = proc_test + (dataset_test,) + coords_test

if args.pair_only:
    proc_train = build_pair_dataset(proc_train)
    proc_test = build_pair_dataset(proc_test)

L_data = proc_train[0].shape[1]
L_model = args.L if args.L is not None else L_data
if args.L is not None and args.L != L_data:
    print(f"Warning: model L={args.L} differs from data L={L_data}; ensure this is intended.")
if use_xy_var and L_model != L_data:
    raise ValueError(f"For diffusion_xy_var, model L must match data L ({L_data}), got {L_model}")
print(f"VF locations (after preprocessing): data L={L_data}, model L={L_model}")

train_ds = TensorDataset(*proc_train)
test_ds  = TensorDataset(*proc_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

# -------------------------------------------------------------------------
# Model + optimizer
# -------------------------------------------------------------------------
print(f"Building model: {MODEL_NAME}...", flush=True)

# Get model with appropriate parameters
model_kwargs = {'L': L_model, 'D': args.D, 'max_cond': args.max_cond}
if MODEL_NAME in ['diffusion', 'diffusion_xy', 'diffusion_xy_var']:
    model_kwargs['K'] = args.K

# Load XY coordinates if using diffusion_xy model
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
    
    print(f"  Loaded XY coordinates: x_shape={x_coords.shape}, y_shape={y_coords.shape}")
    model_kwargs['x_coords'] = x_coords
    model_kwargs['y_coords'] = y_coords

denoiser = get_model(MODEL_NAME, **model_kwargs).to(device)
if args.pretrained_model:
    print(f"Loading pretrained model from {args.pretrained_model}...")
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    denoiser.load_state_dict(checkpoint['model_state_dict'])

n_params = sum(p.numel() for p in denoiser.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)
scaler_grad = torch.cuda.amp.GradScaler()

# Diffusion schedule
cumalphas = f(torch.arange(NUM_STEPS + 1, device=device), NUM_STEPS) / f(torch.tensor(0.0, device=device), NUM_STEPS)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------
print("Starting training...")
print("=" * 60)

best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    denoiser.train()
    running_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        x_coords_b = None
        y_coords_b = None
        valid_mask_b = None
        dataset_id_b = None
        core_batch = batch
        if use_xy_var:
            if DATASET_NAME == 'joint_orig':
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

        if args.pair_only:
            cond_b, target_b, dt_future_b = core_batch
            cond_b = cond_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            dt_future_b = dt_future_b.to(device, non_blocking=True)
            B_cur = target_b.size(0)
        elif DATA_N_VF == 3 and DATASET_NAME != "oct_vf4":
            x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = core_batch
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            if DATASET_NAME == 'oct_vf' or DATASET_NAME == 'vf_small':
                t_diff_2 = t2_b.to(device, non_blocking=True)
                t_diff_3 = t3_b.to(device, non_blocking=True)
            else:
                t_diff_1 = (t3_b - t1_b).to(device, non_blocking=True)
                t_diff_2 = (t3_b - t2_b).to(device, non_blocking=True)
                t_diff_23 = (t3_b - t2_b).to(device, non_blocking=True)
            locs = [torch.arange(args.L, device=device).float() for _ in range(3)]
            B_cur = x3_b.size(0)
        elif DATA_N_VF == 3 and DATASET_NAME == "oct_vf4":
            x1_b, x2_b, x3_b, x4_b, x5_b, t1_b, t2_b, t3_b, t4_b, t5_b = batch
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            x4_b = x4_b.to(device, non_blocking=True)
            x5_b = x5_b.to(device, non_blocking=True)
            t_diff_2 = t2_b.to(device, non_blocking=True)
            t_diff_3 = t3_b.to(device, non_blocking=True)
            t_diff_4 = t4_b.to(device, non_blocking=True)
            t_diff_5 = t5_b.to(device, non_blocking=True)
            locs = [torch.arange(args.L, device=device).float() for _ in range(5)]
            B_cur = x5_b.size(0)
        else:
            x1_b, x2_b, x3_b, x4_b, t1_b, t2_b, t3_b, t4_b = batch
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            x3_b = x3_b.to(device, non_blocking=True)
            x4_b = x4_b.to(device, non_blocking=True)
            dt_1 = (t4_b - t1_b).to(device, non_blocking=True)
            dt_2 = (t4_b - t2_b).to(device, non_blocking=True)
            dt_3 = (t4_b - t3_b).to(device, non_blocking=True)
            B_cur = x4_b.size(0)

        t = torch.randint(1, NUM_STEPS + 1, (B_cur,), device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            if args.pair_only:
                target = target_b
                cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                dt_0 = torch.zeros_like(dt_future_b)
                if use_xy_var:
                    noise_pred = denoiser(
                        xt, [cond_b], [dt_0], t, dt_future_b,
                        x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                        dataset_id=dataset_id_b
                    )
                else:
                    noise_pred = denoiser(xt, [cond_b], [dt_0], t, dt_future_b)
                noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
            else:
                if args.train_task != 0:
                    task_id = args.train_task
                else:
                    task_val = torch.rand(1).item()
                    if DATA_N_VF == 3:
                        task_id = 1 if task_val < 0.4 else 2
                    else:
                        task_id = 1 if task_val < 0.2 else (2 if task_val < 0.4 else 3)

                if DATA_N_VF == 3:
                    if task_id == 1:
                        # VF1 -> VF2
                        target = x3_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_3 = torch.zeros_like(t_diff_1)
                        if use_xy_var:
                            noise_pred = denoiser(
                                xt, [x1_b], [dt_0], t, t_diff_3,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            noise_pred = denoiser(xt, [x1_b], [t_diff_1], t, dt_3)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                    elif DATASET_NAME == "oct_vf4":
                        target = x5_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_0 = torch.zeros_like(t_diff_5)
                        noise_pred = denoiser(
                            xt, [x1_b, x2_b, x3_b, x4_b], [dt_0, t_diff_2, t_diff_3, t_diff_4], t, t_diff_5, locs=locs)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                    elif DATASET_NAME == "oct_vf" or DATASET_NAME == "vf_small":
                        target = x3_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_0 = torch.zeros_like(t_diff_3)
                        noise_pred = denoiser(
                            xt, [x1_b, x2_b], [dt_0, t_diff_2], t, t_diff_3, locs=locs)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                    else:
                        # VF1+VF2 -> VF3 (primary)
                        target = x3_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_3 = torch.zeros_like(t_diff_2)
                        if use_xy_var:
                            noise_pred = denoiser(
                                xt, [x1_b, x2_b], [dt_0, t_diff_2], t, t_diff_3,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            noise_pred = denoiser(xt, [x1_b, x2_b], [t_diff_1, t_diff_2], t, dt_3, locs)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                else:
                    # 4VF tasks: VF1->VF4 (0.2), VF1+VF2->VF4 (0.2), VF1+VF2+VF3->VF4 (0.6)
                    if task_id == 1:
                        target = x4_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_0 = torch.zeros_like(dt_4)
                        if use_xy_var:
                            noise_pred = denoiser(
                                xt, [x1_b], [dt_0], t, dt_4,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            noise_pred = denoiser(xt, [x1_b], [dt_0], t, dt_4)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                    elif task_id == 2:
                        target = x4_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_0 = torch.zeros_like(dt_4)
                        if use_xy_var:
                            noise_pred = denoiser(
                                xt, [x1_b, x2_b], [dt_0, dt_2], t, dt_4,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            noise_pred = denoiser(xt, [x1_b, x2_b], [dt_0, dt_2], t, dt_4)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)
                    else:
                        target = x4_b
                        cumalphas_t = cumalphas.gather(0, t - 1).unsqueeze(1)
                        xt = torch.sqrt(cumalphas_t) * target + torch.sqrt(1 - cumalphas_t) * torch.randn_like(target, device=device)
                        dt_0 = torch.zeros_like(dt_4)
                        if use_xy_var:
                            noise_pred = denoiser(
                                xt, [x1_b, x2_b, x3_b], [dt_0, dt_2, dt_3], t, dt_4,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            noise_pred = denoiser(xt, [x1_b, x2_b, x3_b], [dt_0, dt_2, dt_3], t, dt_4)
                        noise_target = (xt - torch.sqrt(cumalphas_t) * target) / torch.sqrt(1 - cumalphas_t)

            if use_xy_var:
                loss = ((noise_pred - noise_target) ** 2 * valid_mask_b).sum() / valid_mask_b.sum().clamp_min(1.0)
            else:
                loss = nn.functional.mse_loss(noise_pred, noise_target)

        scaler_grad.scale(loss).backward()
        scaler_grad.step(optimizer)
        scaler_grad.update()

        running_loss += loss.item()
        if (batch_idx + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            if not args.no_wandb:
                wandb.log({"train/batch_loss": float(avg_loss), "epoch": epoch})
            running_loss = 0.0

    # ------------------ Epoch End ------------------
    epoch_time = time.time() - t0
    if not args.no_wandb:
        wandb.log({"train/epoch_loss": float(loss.item()), "train/epoch_time": epoch_time, "epoch": epoch})
    if epoch % 20 == 0:
        print(f"Epoch {epoch:5d}/{EPOCHS} | Loss: {loss.item():.6f} | Time: {epoch_time:.1f}s", flush=True)

    # ------------------ Evaluation ------------------
    if epoch % EVAL_EVERY == 0:
        print(f"\nEvaluating at epoch {epoch}...")
        denoiser.eval()

        def evaluate_task(task_id, task_name):
            """Evaluate a specific prediction task"""
            preds, reals = [], []
            mse_sum, n = 0.0, 0
            with torch.no_grad():
                for batch in test_loader:
                    x_coords_b = None
                    y_coords_b = None
                    valid_mask_b = None
                    dataset_id_b = None
                    core_batch = batch
                    if use_xy_var:
                        if DATASET_NAME == 'joint_orig':
                            dataset_id_b = batch[-4].to(device)
                            x_coords_b = batch[-3].to(device)
                            y_coords_b = batch[-2].to(device)
                            valid_mask_b = batch[-1].to(device)
                            core_batch = batch[:-4]
                        else:
                            x_coords_b = batch[-3].to(device)
                            y_coords_b = batch[-2].to(device)
                            valid_mask_b = batch[-1].to(device)
                            core_batch = batch[:-3]

                    if DATA_N_VF == 3:
                        if args.pair_only:
                            cond_b, target_b, dt_future_b = core_batch
                            cond_b = cond_b.to(device)
                            target_b = target_b.to(device)
                            dt_future_b = dt_future_b.to(device)
                            target = target_b
                            cond_v = [cond_b]
                            cond_dt = [torch.zeros_like(dt_future_b)]
                            dt_future = dt_future_b
                        elif DATASET_NAME == "oct_vf4":
                            x1_b, x2_b, x3_b, x4_b, x5_b, t1_b, t2_b, t3_b, t4_b, t5_b = core_batch
                            x1_b = x1_b.to(device)
                            x2_b = x2_b.to(device)
                            x3_b = x3_b.to(device)
                            x4_b = x4_b.to(device)
                            x5_b = x5_b.to(device)
                            t_diff_2 = t2_b.to(device)
                            t_diff_3 = t3_b.to(device)
                            t_diff_4 = t4_b.to(device)
                            t_diff_5 = t5_b.to(device)
                            target = x5_b
                            cond_v = [x1_b, x2_b, x3_b, x4_b]
                            cond_dt = [torch.zeros_like(t_diff_5), t_diff_2, t_diff_3, t_diff_4]
                            dt_future = t_diff_5
                            locs = [torch.arange(args.L, device=device).float() for _ in range(5)]
                        elif DATASET_NAME == "oct_vf" or DATASET_NAME == "vf_small":
                            x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = core_batch
                            x1_b = x1_b.to(device)
                            x2_b = x2_b.to(device)
                            x3_b = x3_b.to(device)
                            t_diff_2 = t2_b.to(device)
                            t_diff_3 = t3_b.to(device)
                            target = x3_b
                            cond_v = [x1_b, x2_b]
                            cond_dt = [torch.zeros_like(t_diff_3), t_diff_2]
                            dt_future = t_diff_3
                            locs = [torch.arange(args.L, device=device).float() for _ in range(3)]
                        else:
                            x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = core_batch
                            x1_b, x2_b, x3_b = x1_b.to(device), x2_b.to(device), x3_b.to(device)
                            t_diff_1 = (t3_b - t1_b).to(device)
                            t_diff_2 = (t3_b - t2_b).to(device)
                            dt_3 = torch.zeros_like(t_diff_1)
                            if task_id == 1:
                                target = x2_b
                                cond_v = [x1_b]
                                cond_dt = [dt_0]
                                dt_future = t_diff_2
                            else:
                                target = x3_b
                                cond_v = [x1_b, x2_b]
                                cond_dt = [t_diff_1, t_diff_2]
                                dt_future = dt_3
                                locs = [torch.arange(args.L, device=device).float() for _ in range(3)]
                    else:
                        x1_b, x2_b, x3_b, x4_b, t1_b, t2_b, t3_b, t4_b = core_batch
                        x1_b, x2_b, x3_b, x4_b = x1_b.to(device), x2_b.to(device), x3_b.to(device), x4_b.to(device)
                        dt_2 = (t2_b - t1_b).to(device)
                        dt_3 = (t3_b - t1_b).to(device)
                        dt_4 = (t4_b - t1_b).to(device)
                        dt_0 = torch.zeros_like(dt_4)
                        if task_id == 1:
                            target = x4_b
                            cond_v = [x1_b]
                            cond_dt = [dt_0]
                            dt_future = dt_4
                        elif task_id == 2:
                            target = x4_b
                            cond_v = [x1_b, x2_b]
                            cond_dt = [dt_0, dt_2]
                            dt_future = dt_4
                        else:
                            target = x4_b
                            cond_v = [x1_b, x2_b, x3_b]
                            cond_dt = [dt_0, dt_2, dt_3]
                            dt_future = dt_4

                    # DDPM sampling
                    xt = torch.randn_like(target, device=device)
                    if use_xy_var:
                        xt = xt * valid_mask_b
                    for t_step_eval in range(NUM_STEPS - 1, 0, -1):
                        alpha_t = alphas[t_step_eval - 1]
                        cumalpha_t = cumalphas[t_step_eval - 1]
                        sigma = torch.sqrt(betas[t_step_eval - 1])
                        factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                        t_tensor = torch.tensor([t_step_eval], device=device).expand(xt.shape[0])
                        if use_xy_var:
                            pred_noise = denoiser(
                                xt, cond_v, cond_dt, t_tensor, dt_future,
                                x_coords=x_coords_b, y_coords=y_coords_b, valid_mask=valid_mask_b,
                                dataset_id=dataset_id_b
                            )
                        else:
                            pred_noise = denoiser(xt, cond_v, cond_dt, t_tensor, dt_future, locs)
                        z = torch.randn_like(xt) if t_step_eval > 1 else torch.zeros_like(xt)
                        xt = 1/torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z
                        if use_xy_var:
                            xt = xt * valid_mask_b
                    preds.append(xt.cpu().numpy())
                    reals.append(target.cpu().numpy())
                    if use_xy_var:
                        mse_sum += (((xt - target) ** 2) * valid_mask_b).sum().item()
                        n += valid_mask_b.sum().item()
                    else:
                        mse_sum += nn.functional.mse_loss(xt, target, reduction="sum").item()
                        n += target.numel()

            preds = np.concatenate(preds, axis=0)
            reals = np.concatenate(reals, axis=0)
            fid = calculate_fid(reals, preds)
            mse = mse_sum / max(n, 1)
            return mse, fid

        if DATA_N_VF == 3:
            if args.pair_only:
                mse1, fid1 = evaluate_task(1, "pair")
                primary_mse, primary_fid = mse1, fid1
            else:
                if args.train_task == 1:
                    mse1, fid1 = evaluate_task(1, "VF1->VF2")
                    primary_mse, primary_fid = mse1, fid1
                elif args.train_task == 2:
                    mse2, fid2 = evaluate_task(2, "VF1+VF2->VF3")
                    primary_mse, primary_fid = mse2, fid2
                else:
                    mse1, fid1 = evaluate_task(1, "VF1->VF2")
                    mse2, fid2 = evaluate_task(2, "VF1+VF2->VF3")
                    primary_mse, primary_fid = mse2, fid2
        else:
            if args.train_task == 1:
                mse1, fid1 = evaluate_task(1, "VF1->VF4")
                primary_mse, primary_fid = mse1, fid1
            elif args.train_task == 2:
                mse2, fid2 = evaluate_task(2, "VF1+VF2->VF4")
                primary_mse, primary_fid = mse2, fid2
            elif args.train_task == 3:
                mse3, fid3 = evaluate_task(3, "VF1+VF2+VF3->VF4")
                primary_mse, primary_fid = mse3, fid3
            else:
                mse1, fid1 = evaluate_task(1, "VF1->VF4")
                mse2, fid2 = evaluate_task(2, "VF1+VF2->VF4")
                mse3, fid3 = evaluate_task(3, "VF1+VF2+VF3->VF4")
                primary_mse, primary_fid = mse3, fid3

        if not args.no_wandb:
            log_dict = {"epoch": epoch}
            if args.pair_only:
                log_dict.update({"eval/MSE_task1": float(mse1), "eval/FID_task1": float(fid1)})
            else:
                if args.train_task in (0, 1):
                    log_dict.update({"eval/MSE_task1": float(mse1), "eval/FID_task1": float(fid1)})
                if args.train_task in (0, 2):
                    log_dict.update({"eval/MSE_task2": float(mse2), "eval/FID_task2": float(fid2)})
                if DATA_N_VF == 4 and args.train_task in (0, 3):
                    log_dict.update({"eval/MSE_task3": float(mse3), "eval/FID_task3": float(fid3)})
            wandb.log(log_dict)

        if DATA_N_VF == 3:
            if args.pair_only:
                print(f"Pair task (cond->target): MSE: {mse1:.6f} | FID: {fid1:.4f} ← primary")
            else:
                if args.train_task in (0, 1):
                    primary_marker = " ← primary" if args.train_task == 1 else ""
                    print(f"Task 1 (VF1->VF2):      MSE: {mse1:.6f} | FID: {fid1:.4f}{primary_marker}")
                if args.train_task in (0, 2):
                    primary_marker = " ← primary" if args.train_task in (0, 2) else ""
                    print(f"Task 2 (VF1+VF2->VF3):  MSE: {mse2:.6f} | FID: {fid2:.4f}{primary_marker}")
        else:
            if args.train_task in (0, 1):
                primary_marker = " ← primary" if args.train_task == 1 else ""
                print(f"Task 1 (VF1->VF4):               MSE: {mse1:.6f} | FID: {fid1:.4f}{primary_marker}")
            if args.train_task in (0, 2):
                primary_marker = " ← primary" if args.train_task == 2 else ""
                print(f"Task 2 (VF1+VF2->VF4):           MSE: {mse2:.6f} | FID: {fid2:.4f}{primary_marker}")
            if args.train_task in (0, 3):
                primary_marker = " ← primary" if args.train_task in (0, 3) else ""
                print(f"Task 3 (VF1+VF2+VF3->VF4):       MSE: {mse3:.6f} | FID: {fid3:.4f}{primary_marker}")

        # Save best model based on primary task
        if primary_fid < best_fid:
            best_fid = primary_fid
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fid': best_fid,
                'config': vars(args),
            }, MODEL_PATH)
            print(f"✓ New best model saved! FID: {primary_fid:.4f}\n")
        else:
            print()

        gc.collect()
        torch.cuda.empty_cache()

print("=" * 60)
print("Training completed!")
print(f"Best FID: {best_fid:.4f}")
print(f"Model saved to: {MODEL_PATH}")
print("=" * 60)

if not args.no_wandb:
    wandb.finish()
