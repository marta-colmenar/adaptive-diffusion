import torch
import numpy as np
import pandas as pd

from ablation_models_diffusion_wa import get_model
from model_diffusion_wa import f


CHECKPOINT = "denoiser_ablation_diffusion_loc_vf_small_3vf.pth"
PARQUET = "data_vf/df_test_oct_vf.parquet"
SCALER = "data_vf/scaler_bern_3vf_new.npy"

NUM_SAMPLES = 2500
NUM_STEPS = 52
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_triplets(path, scaler_path=SCALER):
    df = pd.read_parquet(path)
    scaler = np.load(scaler_path, allow_pickle=True).item()

    def vf(name):
        cols = [c for c in df.columns if c.startswith(name + "_deviation")]
        vf = (-1)*(df[cols].values)
        return torch.tensor(scaler.transform(vf)).float().to(DEVICE)
    
    def years_between(a, b):
        date_a = pd.to_datetime(df[a])
        date_b = pd.to_datetime(df[b])

        years = (date_b - date_a).dt.days / 365.25
        return torch.tensor(years.values, dtype=torch.float32).cpu()

    return {
        "vf1_target": vf("vf1_target"),
        "vf1": vf("vf1"),
        "vf2_target": vf("vf2_target"),
        "vf2": vf("vf2"),
        "vf3": vf("vf3"),
        "dt1": torch.tensor(years_between("acquisition_date_3", "acquisition_date_1")),
        "dt2": torch.tensor(years_between("acquisition_date_3", "acquisition_date_2")),
        "dt3": torch.tensor(years_between("acquisition_date_3", "acquisition_date_3")),
    }


def diffusion_schedule(T, device):
    cum = f(torch.arange(T + 1, device=device), T)
    cum = cum / cum[0]
    alpha_bar = cum[1:]
    alpha = alpha_bar / cum[:-1]
    beta = 1 - alpha
    return alpha, beta, alpha_bar


@torch.no_grad()
def sample(model, cond_v, cond_dt, dt_future, shape,
           alpha, beta, alpha_bar):

    x = torch.randn(shape, device=DEVICE)

    for t in reversed(range(1, len(alpha_bar))):
        a = alpha[t - 1]
        ab = alpha_bar[t - 1]
        b = beta[t - 1]

        tt = torch.full((shape[0],), t,
                        device=DEVICE,
                        dtype=torch.long)

        eps = model(x, cond_v, cond_dt, tt, dt_future)

        noise = torch.randn_like(x) if t > 1 else 0

        x = (
            (x - (1 - a) / torch.sqrt(1 - ab) * eps)
            / torch.sqrt(a)
            + torch.sqrt(b) * noise
        )

    return x


triplets = load_triplets(PARQUET)

L = triplets["vf3"].shape[1]

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state = ckpt["model_state_dict"]

model = get_model(
    "diffusion",
    L=L,
    D=64,
    K=2,
    max_cond=2,
).to(DEVICE)

model.load_state_dict(state, strict=False)
model.eval()

alpha, beta, alpha_bar = diffusion_schedule(NUM_STEPS, DEVICE)

variants = {
    "vf1_vf2": ("vf1", "vf2"),
    "vf1t_vf2t": ("vf1_target", "vf2_target"),
    # "vf1_vf2t": ("vf1", "vf2_target"),
    # "vf1t_vf2": ("vf1_target", "vf2"),
    # "noise_noise": ("vf1_target", "vf2_target", "vf1_target", "vf2_target"),
    # "vf1t_vf1_vf2t_vf2": ("vf1_target", "vf1", "vf2_target", "vf2"),
    # "vf1t_vf1t_vf2t_vf2t": ("vf1_target", "vf1_target", "vf2_target", "vf2_target"),
    # "vf1_vf1t_vf2_vf2t": ("vf1", "vf1_target", "vf2", "vf2_target"),
    # "vf1t_vf2t_vf1_vf2": ("vf1_target", "vf2_target", "vf1", "vf2"),
    # "vf1_vf2_vf1t_vf2t": ("vf1", "vf2", "vf1_target", "vf2_target"),
}

for name, (a, b) in variants.items():
    cond_v = [
        triplets[a].to(DEVICE),
        triplets[b].to(DEVICE),
    ]

    cond_dt = [
        triplets["dt1"].to(DEVICE),
        triplets["dt2"].to(DEVICE)
    ]

# for name, (a, b, c, d) in variants.items():
#     cond_v = [
#         triplets[a].to(DEVICE),
#         triplets[b].to(DEVICE),
#         triplets[c].to(DEVICE),
#         triplets[d].to(DEVICE),
#     ]
#     if name == "vf1t_vf1_vf2t_vf2":
#         cond_dt = [
#             triplets["dt1"].to(DEVICE),
#             triplets["dt1"].to(DEVICE),
#             triplets["dt2"].to(DEVICE),
#             triplets["dt2"].to(DEVICE)
#          ]
        
        
    # elif name == "vf1t_vf1t_vf2t_vf2t":
    #     cond_dt = [
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #     ]

    # elif name == "vf1_vf1t_vf2_vf2t":
    #     cond_dt = [
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #     ]
    # elif name == "vf1t_vf2t_vf1_vf2":
    #     cond_dt = [
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #     ]
    # elif name == "vf1_vf2_vf1t_vf2t":
    #     cond_dt = [
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #         torch.zeros_like(triplets["dt2"]).to(DEVICE),
    #         triplets["dt2"].to(DEVICE),
    #     ]

#     elif name == "noise_noise":
#         cond_v = [
#             torch.zeros_like(triplets["vf1_target"]).to(DEVICE),
#             torch.zeros_like(triplets["vf2_target"]).to(DEVICE),
#         ]

#         cond_dt = [
#             torch.zeros_like(triplets["dt2"]).to(DEVICE),
#             triplets["dt2"].to(DEVICE),
#         ]
    
    dt_future = triplets["dt3"].to(DEVICE)

    preds = []

    for _ in range(NUM_SAMPLES):
        preds.append(
            sample(
                model,
                cond_v,
                cond_dt,
                dt_future,
                triplets["vf3"].shape,
                alpha,
                beta,
                alpha_bar,
            ).cpu().numpy()
        )

    preds = np.stack(preds, axis=1)

    np.save(f"/storage/homefs/mc21s017/samples_triplets/{name}_pred.npy", preds)
    np.save(f"/storage/homefs/mc21s017/samples_triplets/{name}_real.npy", triplets["vf3"].cpu().numpy())

    print(f"Finished {name}")