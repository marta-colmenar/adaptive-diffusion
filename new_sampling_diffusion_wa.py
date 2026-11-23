import torch, numpy as np, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from new_model_diffusion_12_wa import VFConditionalDiffusion
from model_diffusion_wa import f, calculate_fid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_df(df):
    arr = df[df.columns[df.columns.str.contains('TD_')]].values.astype(np.float32)
    exclude_cols = [25,34]
    X1, X2, X3= arr[:, :54], arr[:, 54:108], arr[:, 108:]
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

# ------------------ Config ------------------
DATA_PATH = "/storage/homefs/mc21s017/data_vf/"
MODEL_PATH = "/storage/homefs/mc21s017/denoiser_diffusion_3vf_wa_augmented_sum_12.pth"
num_steps = 52
num_samples = 150  # number of samples per patient

# ------------------ Load Data ------------------
df_test = pd.read_csv(f'{DATA_PATH}df_test_wa_3vf_normal.csv')
scaler = np.load(f'{DATA_PATH}scaler_wa_3vf.npy', allow_pickle=True).item()

# Preprocess once
arr = df_test[df_test.columns[df_test.columns.str.contains('TD_')]].values.astype(np.float32)

X1,  X2,  X3,  t1,  t2,  t3  = process_df(df_test)

dataset_test = TensorDataset(X1, X2, X3, t1, t2, t3)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False, pin_memory=True)

# ------------------ Model ------------------
denoiser = VFConditionalDiffusion(L=52, D=32, K=2, max_cond=2).to(device)
denoiser.load_state_dict(torch.load(MODEL_PATH, map_location=device))
denoiser.eval()

# Precompute alphas, betas
cumalphas = f(torch.arange(num_steps + 1), num_steps) / f(0, num_steps)
alphas = cumalphas[1:] / cumalphas[:-1]
betas = 1 - alphas
cumalphas = cumalphas[1:]

# ------------------ Sampling ------------------
predefined_times = torch.tensor([2, 4, 8, 16, 32, 64], dtype=torch.float32) / 12  # in years
names = ["2m", "4m", "8m", "16m", "32m", "64m"]

samples_per_time = {name: [] for name in names}
t_diffs_per_time = {}

with torch.no_grad():
    for name, t_diff_target in zip(names, predefined_times):
        print(f"Sampling for t_diff = {t_diff_target.item()}", flush=True)
        t_diff_exp = t_diff_target.expand(len(dataset_test)).to(device)

        batch_outputs = []

        for batch in tqdm(test_loader, desc=f"Time {name}"):
            x1_b, x2_b, x3_b, t1_b, t2_b, t3_b = [x.to(device) for x in batch]
            t_diff_2 = (t2_b - t1_b).to(device)
            batch_size = x3_b.shape[0]

            # Repeat batch num_samples times
            x1_b_rep = x1_b.unsqueeze(1).repeat(1, num_samples, 1).view(-1, x1_b.shape[1])
            x2_b_rep = x2_b.unsqueeze(1).repeat(1, num_samples, 1).view(-1, x2_b.shape[1])
            dt_0_rep = torch.zeros(batch_size).unsqueeze(1).repeat(1, num_samples).view(-1).to(device)
            t_diff_2_rep = t_diff_2.unsqueeze(1).repeat(1, num_samples).view(-1)
            t_diff_target_rep = t_diff_exp[:batch_size].unsqueeze(1).repeat(1, num_samples).view(-1)

            # Start with noise
            xt = torch.randn(batch_size * num_samples, x3_b.shape[1], device=device)

            # Reverse diffusion
            for t_step in range(num_steps - 1, 0, -1):
                alpha_t = alphas[t_step - 1]
                cumalpha_t = cumalphas[t_step - 1]
                sigma = torch.sqrt(betas[t_step - 1])
                factor = (1 - alpha_t) / torch.sqrt(1 - cumalpha_t)
                t_tensor = torch.tensor([t_step], device=device).expand(xt.shape[0])
                
                # Call model with list format: [x1, x2], [dt_0, dt_2], t, dt_target
                pred_noise = denoiser(xt, [x1_b_rep], [dt_0_rep], t_tensor, t_diff_target_rep)
                z = torch.randn_like(xt) if t_step > 1 else torch.zeros_like(xt)
                xt = 1 / torch.sqrt(alpha_t) * (xt - factor * pred_noise) + sigma * z

            # Reshape to (batch_size, num_samples, num_loc)
            xt = xt.view(batch_size, num_samples, -1).cpu().numpy()
            batch_outputs.append(xt)

            # Save t_diff for this batch
            t_diffs_per_time[name] = t_diff_target_rep.cpu().view(batch_size, num_samples).numpy()[:, 0]  # first sample is enough

        # Concatenate all batches along patients axis: (num_patients, num_samples, num_loc)
        arr = np.concatenate(batch_outputs, axis=0)
        # Swap axes to get (num_samples, num_patients, num_loc)
        samples_per_time[name] = np.swapaxes(arr, 0, 1)

# ------------------ Save ------------------
np.save(f'{DATA_PATH}/samples_per_time_wa_augmented_sum_12_VF1.npy', samples_per_time)
np.save(f'{DATA_PATH}/t_diffs_per_time_wa_augmented_sum_12_VF1.npy', t_diffs_per_time)
print("Sampling completed and saved.")