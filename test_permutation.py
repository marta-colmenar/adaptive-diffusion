import torch
from ablation_models_diffusion_wa import get_model, positional_encoding_td
from types import MethodType

def permute_tokens(x, perm):
    return x[:, perm, :]


def equiv_error(a, b):
    return (a - b).abs().max().item()

@torch.no_grad()
def test_permutation_equivariance_no_pos(
    model,
    batch_size=8,
    atol=1e-5,
    rtol=1e-4,
):

    # --------------------------------------------------
    # Save original encoder
    # --------------------------------------------------

    original_encode_vf = model._encode_vf

    # --------------------------------------------------
    # Encoder WITHOUT positional encoding
    # --------------------------------------------------

    def encode_vf_no_pos(self, x, dt, pos):

        z = positional_encoding_td(x, self.D * 2)

        z = self.td_enc(z)

        z = z + self.time_enc(dt).unsqueeze(1)

        # intentionally omit:
        # z = z + pos.permute(0,2,1)

        return z

    model._encode_vf = MethodType(
        encode_vf_no_pos,
        model,
    )

    try:

        result = test_permutation_equivariance(
            model,
            batch_size=batch_size,
            atol=atol,
            rtol=rtol,
        )

    finally:

        # restore original model
        model._encode_vf = original_encode_vf

    return result

@torch.no_grad()
def test_permutation_equivariance(
    model,
    batch_size=4,
    atol=1e-5,
    rtol=1e-4,
    device=None,
):
    """
    Tests whether

        model(Px) == P model(x)

    where P is a random permutation of the token dimension.

    Returns:
        dict with statistics.
    """

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    L = model.L
    max_cond = model.max_cond

    # --------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------

    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [
        torch.randn(batch_size, L, device=device)
        for _ in range(max_cond)
    ]

    dt_hist = [
        torch.rand(batch_size, device=device)
        for _ in range(max_cond)
    ]

    t_step = torch.rand(batch_size, device=device)

    dt_future = torch.rand(batch_size, device=device)

    # --------------------------------------------------
    # Random permutation
    # --------------------------------------------------

    perm = torch.randperm(L, device=device)

    x_t_perm = x_t[:, perm]

    v_hist_perm = [
        v[:, perm]
        for v in v_hist
    ]

    # --------------------------------------------------
    # Original forward pass
    # --------------------------------------------------

    y = model(
        x_t=x_t,
        v_hist=v_hist,
        dt_hist=dt_hist,
        t_step=t_step,
        dt_future=dt_future,
    )

    # --------------------------------------------------
    # Permuted forward pass
    # --------------------------------------------------

    y_perm = model(
        x_t=x_t_perm,
        v_hist=v_hist_perm,
        dt_hist=dt_hist,
        t_step=t_step,
        dt_future=dt_future,
    )

    # --------------------------------------------------
    # Equivariance check
    # --------------------------------------------------

    y_expected = y[:, perm]

    abs_err = (y_perm - y_expected).abs()

    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()

    equivariant = torch.allclose(
        y_perm,
        y_expected,
        atol=atol,
        rtol=rtol,
    )

    print("=" * 60)
    print("Permutation Equivariance Test")
    print("=" * 60)
    print(f"Sequence length : {L}")
    print(f"Batch size      : {batch_size}")
    print(f"Max error       : {max_err:.6e}")
    print(f"Mean error      : {mean_err:.6e}")
    print(f"Equivariant     : {equivariant}")
    print("=" * 60)

    return {
        "equivariant": equivariant,
        "max_error": max_err,
        "mean_error": mean_err,
        "perm": perm.cpu(),
    }

@torch.no_grad()
def inspect_encoder_equivariance(model, batch_size=2):

    device = next(model.parameters()).device

    B = batch_size
    L = model.L

    x = torch.randn(B, L, device=device)
    dt = torch.rand(B, device=device)

    pos = model.pos_enc(model.loc).T.unsqueeze(0)

    perm = torch.randperm(L, device=device)

    z1 = model._encode_vf(x, dt, pos)
    z2 = model._encode_vf(x[:, perm], dt, pos)

    z1_perm = z1[:, perm]

    err = (z2 - z1_perm).abs().max()

    print("encoder equivariance error =", err.item())

@torch.no_grad()
def inspect_encoder_equivariance_no_pos(model, batch_size=2):

    device = next(model.parameters()).device

    B = batch_size
    L = model.L

    x = torch.randn(B, L, device=device)
    dt = torch.rand(B, device=device)

    pos = torch.zeros(
        1,
        model.D,
        model.L,
        device=device,
    )

    perm = torch.randperm(L, device=device)

    z1 = model._encode_vf(x, dt, pos)
    z2 = model._encode_vf(x[:, perm], dt, pos)

    z1_perm = z1[:, perm]

    err = (z2 - z1_perm).abs().max()

    print("encoder equivariance error =", err.item())

@torch.no_grad()
def test_token_equiv_with_conditioning(model, batch_size=8):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond

    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [
        torch.randn(batch_size, L, device=device)
        for _ in range(K)
    ]

    dt_hist = [
        torch.rand(batch_size, device=device)
        for _ in range(K)
    ]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    perm = torch.randperm(L, device=device)

    # permute inputs
    x_perm = x_t[:, perm]
    v_perm = [v[:, perm] for v in v_hist]

    y = model(
        x_t=x_t,
        v_hist=v_hist,
        dt_hist=dt_hist,
        t_step=t_step,
        dt_future=dt_future,
    )

    y_perm = model(
        x_t=x_perm,
        v_hist=v_perm,
        dt_hist=dt_hist,
        t_step=t_step,
        dt_future=dt_future,
    )

    err = (y_perm - y[:, perm]).abs()

    print("TOKEN equiv max error:", err.max().item())
    print("TOKEN equiv mean error:", err.mean().item())

@torch.no_grad()
def test_vf_order_symmetry(model, batch_size=8):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond

    assert K >= 2, "Need at least 2 conditioning VFs"

    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [
        torch.randn(batch_size, L, device=device)
        for _ in range(K)
    ]

    dt_hist = [
        torch.rand(batch_size, device=device)
        for _ in range(K)
    ]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    # swap VF1 and VF2
    v_swapped = v_hist.copy()
    v_swapped[0], v_swapped[1] = v_swapped[1], v_swapped[0]

    dt_swapped = dt_hist.copy()
    dt_swapped[0], dt_swapped[1] = dt_swapped[1], dt_swapped[0]

    y = model(
        x_t=x_t,
        v_hist=v_hist,
        dt_hist=dt_hist,
        t_step=t_step,
        dt_future=dt_future,
    )

    y_swap = model(
        x_t=x_t,
        v_hist=v_swapped,
        dt_hist=dt_swapped,
        t_step=t_step,
        dt_future=dt_future,
    )

    err = (y_swap - y).abs()

    print("VF ORDER symmetry max error:", err.max().item())
    print("VF ORDER symmetry mean error:", err.mean().item())

@torch.no_grad()
def test_full_symmetry(model, batch_size=8):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond

    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [
        torch.randn(batch_size, L, device=device)
        for _ in range(K)
    ]

    dt_hist = [
        torch.rand(batch_size, device=device)
        for _ in range(K)
    ]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    perm = torch.randperm(L, device=device)

    # swap VF order
    v_swapped = v_hist.copy()
    v_swapped[0], v_swapped[1] = v_swapped[1], v_swapped[0]

    dt_swapped = dt_hist.copy()
    dt_swapped[0], dt_swapped[1] = dt_swapped[1], dt_swapped[0]

    y = model(x_t, v_hist, dt_hist, t_step, dt_future)

    y2 = model(x_t[:, perm], v_hist, dt_hist, t_step, dt_future)
    y2 = y2[:, perm]

    y3 = model(x_t, v_swapped, dt_swapped, t_step, dt_future)

    print("token equiv error:", (y2 - y).abs().max().item())
    print("vf swap error   :", (y3 - y).abs().max().item())

@torch.no_grad()
def test_self_attention_block(sa_block, dim=64, L=59, B=4, device="cuda"):
    sa_block.eval().to(device)

    x = torch.randn(B, L, dim, device=device)
    perm = torch.randperm(L, device=device)

    x_perm = permute_tokens(x, perm)

    y = sa_block(x)
    y_perm = sa_block(x_perm)

    err = equiv_error(y_perm, permute_tokens(y, perm))

    print("SelfAttention equiv error:", err)

@torch.no_grad()
def test_cross_attention_block(ca_block, dim=64, L=59, B=4, device="cuda"):
    ca_block.eval().to(device)

    x = torch.randn(B, L, dim, device=device)
    cond = torch.randn(B, L, dim, device=device)

    perm = torch.randperm(L, device=device)

    x_p = permute_tokens(x, perm)
    cond_p = permute_tokens(cond, perm)

    y = ca_block(x, cond)
    y_p = ca_block(x_p, cond_p)

    err = equiv_error(y_p, permute_tokens(y, perm))

    print("CrossAttention equiv error:", err)

@torch.no_grad()
def test_encoder(model, B=4, L=59, device="cuda"):

    x = torch.randn(B, L, device=device)
    dt = torch.rand(B, device=device)

    perm = torch.randperm(L, device=device)

    pos = torch.zeros(1, model.D, model.L, device=device)

    z = model._encode_vf(x, dt, pos)
    z_p = model._encode_vf(x[:, perm], dt, pos)

    err = equiv_error(z_p, permute_tokens(z, perm))

    print("Encoder equiv error:", err)

@torch.no_grad()
def test_sa_no_mlp(sa_block, dim=64, L=59, B=4, device="cuda"):
    sa_block.eval().to(device)

    x = torch.randn(B, L, dim, device=device)
    perm = torch.randperm(L, device=device)

    x_p = permute_tokens(x, perm)

    # forward with frozen MLP
    h1 = sa_block.norm1(x)
    attn_out, _ = sa_block.attn(h1, h1, h1, need_weights=False)
    y = x + attn_out

    h2 = sa_block.norm1(x_p)
    attn_out2, _ = sa_block.attn(h2, h2, h2, need_weights=False)
    y2 = x_p + attn_out2

    err = equiv_error(y2, permute_tokens(y, perm))

    print("SelfAttn (no MLP) equiv error:", err)

@torch.no_grad()
def test_layernorm(L=59, dim=64, B=4, device="cuda"):

    ln = torch.nn.LayerNorm(dim).to(device).eval()

    x = torch.randn(B, L, dim, device=device)
    perm = torch.randperm(L, device=device)

    x_p = permute_tokens(x, perm)

    y = ln(x)
    y_p = ln(x_p)

    err = equiv_error(y_p, permute_tokens(y, perm))

    print("LayerNorm equiv error:", err)

@torch.no_grad()
def full_model_debug(model, batch_size=4):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond

    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [torch.randn(batch_size, L, device=device) for _ in range(K)]
    dt_hist = [torch.rand(batch_size, device=device) for _ in range(K)]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    perm = torch.randperm(L, device=device)

    # permute inputs
    x_p = x_t[:, perm]
    v_p = [v[:, perm] for v in v_hist]

    # forward original
    y = model(x_t, v_hist, dt_hist, t_step, dt_future)

    # forward permuted
    y_p = model(x_p, v_p, dt_hist, t_step, dt_future)

    # compare
    err = equiv_error(y_p, y[:, perm])

    print("FULL MODEL equiv error:", err)

@torch.no_grad()
def layerwise_equivariance_trace(model, batch_size=4):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond
    D = model.D

    model.eval()

    # -----------------------------
    # helpers
    # -----------------------------
    def err_tokens(a, b, perm):
        return (a - b[:, perm, :]).abs().max().item()

    def err_full(a, b, perm):
        a = a[:, :-1, :]
        b = b[:, :-1, :]
        return (a - b[:, perm, :]).abs().max().item()

    # -----------------------------
    # inputs
    # -----------------------------
    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [torch.randn(batch_size, L, device=device) for _ in range(K)]
    dt_hist = [torch.rand(batch_size, device=device) for _ in range(K)]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    perm = torch.randperm(L, device=device)

    # permuted inputs
    x_p = x_t[:, perm]
    v_p = [v[:, perm] for v in v_hist]

    # -----------------------------
    # encoder (NO positional encoding assumed)
    # -----------------------------
    pos = torch.zeros(1, D, L, device=device)

    x0 = model._encode_vf(x_t, dt_future, pos)
    x0_p = model._encode_vf(x_p, dt_future, pos)

    print("=" * 60)
    print("LAYERWISE EQUIVARIANCE TRACE")
    print("=" * 60)

    print("Encoder:", err_tokens(x0_p, x0, perm))

    # -----------------------------
    # step token
    # -----------------------------
    s = model.step_enc(t_step).unsqueeze(1)

    x = torch.cat([x0, s], dim=1)
    x_p = torch.cat([x0_p, s], dim=1)

    # -----------------------------
    # conditioning
    # -----------------------------
    cond = torch.cat(
        [model._encode_vf(v, dt, pos) for v, dt in zip(v_hist, dt_hist)],
        dim=1,
    )

    cond_p = torch.cat(
        [model._encode_vf(v, dt, pos) for v, dt in zip(v_p, dt_hist)],
        dim=1,
    )

    # -----------------------------
    # layerwise propagation
    # -----------------------------
    for i, (sa, ca) in enumerate(zip(model.sa_blocks, model.ca_blocks)):

        # Self-attention
        x = sa(x)
        x_p = sa(x_p)

        print(f"SA block {i}: {err_full(x_p, x, perm)}")

        # Cross-attention
        x = x + ca(x, cond)
        x_p = x_p + ca(x_p, cond_p)

        print(f"CA block {i}: {err_full(x_p, x, perm)}")

    # -----------------------------
    # final projection
    # -----------------------------
    x = x[:, :-1, :]
    x_p = x_p[:, :-1, :]

    x = model.out(x.permute(0, 2, 1)).squeeze(1)
    x_p = model.out(x_p.permute(0, 2, 1)).squeeze(1)

    print("FINAL:", (x_p - x[:, perm]).abs().max().item())

    print("=" * 60)

@torch.no_grad()
def residual_growth_test(model):

    device = next(model.parameters()).device
    L = model.L
    K = model.max_cond
    D = model.D

    model.eval()

    # -------------------------
    # proper input shape
    # -------------------------
    x = torch.randn(4, L, D, device=device)

    cond = torch.randn(4, L * K, D, device=device)

    perm = torch.randperm(L, device=device)

    x_p = x[:, perm, :]

    errors = []

    for i in range(len(model.sa_blocks)):

        # SA
        x = model.sa_blocks[i](x)
        x_p = model.sa_blocks[i](x_p)

        # CA
        x = x + model.ca_blocks[i](x, cond)
        x_p = x_p + model.ca_blocks[i](x_p, cond)

        err = (x_p - x[:, perm, :]).abs().max().item()
        errors.append(err)

        print(f"layer {i}: {err}")

    print("error growth:", errors)

@torch.no_grad()
def attention_weight_stability(sa_block, L=59, B=2, dim=64, device=None):

    if device is None:
        device = next(sa_block.parameters()).device

    sa_block = sa_block.to(device).eval()

    x = torch.randn(B, L, dim, device=device)
    perm = torch.randperm(L, device=device)

    x_p = x[:, perm, :]

    # forward
    q1 = sa_block.norm1(x)
    q2 = sa_block.norm1(x_p)

    _, w1 = sa_block.attn(q1, q1, q1, need_weights=True)
    _, w2 = sa_block.attn(q2, q2, q2, need_weights=True)

    # permute attention map properly
    w1_p = w1[:, perm][:, :, perm]

    print("attention mismatch:", (w1_p - w2).abs().max().item())

@torch.no_grad()
def equivariance_check(model, batch_size=8, atol=1e-5):

    device = next(model.parameters()).device
    model.eval()

    L = model.L
    K = model.max_cond
    D = model.D

    # -------------------------------------------------
    # FORCE NO POSITIONAL INFORMATION
    # -------------------------------------------------
    def encode_no_pos(x, dt):
        pos = torch.zeros(1, D, L, device=device)
        return model._encode_vf(x, dt, pos)

    # -------------------------------------------------
    # error function (handles step token correctly)
    # -------------------------------------------------
    def err(a, b, perm):

        # -------------------------
        # CASE 1: final output (B, L)
        # -------------------------
        if a.dim() == 2:
            return (a - b[:, perm]).abs().max().item()

        # -------------------------
        # CASE 2: embeddings (B, L, D)
        # -------------------------
        elif a.dim() == 3:
            a = a[:, :-1, :]
            b = b[:, :-1, :]
            return (a - b[:, perm, :]).abs().max().item()

        else:
            raise ValueError(f"Unexpected tensor shape: {a.shape}")

    # -------------------------------------------------
    # inputs
    # -------------------------------------------------
    x_t = torch.randn(batch_size, L, device=device)

    v_hist = [torch.randn(batch_size, L, device=device) for _ in range(K)]
    dt_hist = [torch.rand(batch_size, device=device) for _ in range(K)]

    t_step = torch.rand(batch_size, device=device)
    dt_future = torch.rand(batch_size, device=device)

    # -------------------------------------------------
    # permutation (ONLY token space)
    # -------------------------------------------------
    perm = torch.randperm(L, device=device)

    x_t_p = x_t[:, perm]
    v_hist_p = [v[:, perm] for v in v_hist]

    # -------------------------------------------------
    # forward pass (NO positional encoding)
    # -------------------------------------------------
    def forward(x_t, v_hist):
        # encode VF history
        cond_chunks = [
            encode_no_pos(v, dt)
            for v, dt in zip(v_hist, dt_hist)
        ]
        cond = torch.cat(cond_chunks, dim=1)

        # main input
        x = encode_no_pos(x_t, dt_future)

        # step token (NOT permuted)
        s = model.step_enc(t_step).unsqueeze(1)
        x = torch.cat([x, s], dim=1)

        # attention stack
        for sa, ca in zip(model.sa_blocks, model.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # output projection
        x = x[:, :-1, :]
        x = model.out(x.permute(0, 2, 1)).squeeze(1)

        return x

    # -------------------------------------------------
    # compute outputs
    # -------------------------------------------------
    y = forward(x_t, v_hist)
    y_p = forward(x_t_p, v_hist_p)

    # -------------------------------------------------
    # equivariance error
    # -------------------------------------------------
    max_err = err(y_p, y, perm)
    mean_err = err(y_p, y, perm)

    print("=" * 60)
    print("PERMUTATION EQUIVARIANCE CHECK (NO POS ENC)")
    print("=" * 60)
    print(f"Sequence length : {L}")
    print(f"Batch size      : {batch_size}")
    print(f"Max error       : {max_err:.6e}")
    print(f"Mean error      : {mean_err:.6e}")

    if max_err < atol:
        print("Result          : ✔ EQUIVARIANT")
    else:
        print("Result          : ❌ NOT EQUIVARIANT")

    print("=" * 60)

    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "equivariant": max_err < atol,
        "perm": perm.cpu(),
    }

@torch.no_grad()
def vf_permutation_repetition_test(model):

    device = next(model.parameters()).device
    model.eval()

    L = model.L
    K = model.max_cond
    D = model.D

    # --------------------------------------------------
    # base VF inputs
    # --------------------------------------------------
    VF1 = torch.randn(1, L, device=device)
    VF2 = torch.randn(1, L, device=device)

    dt1 = torch.rand(1, device=device)
    dt2 = torch.rand(1, device=device)

    # reuse same target + noise
    x_t = torch.randn(1, L, device=device)
    t_step = torch.rand(1, device=device)
    dt_future = torch.rand(1, device=device)

    # --------------------------------------------------
    # helper: build model input
    # --------------------------------------------------
    def run(vf_list, dt_list):

        # encode VF history
        cond_chunks = [
            model._encode_vf(v, dt, torch.zeros(1, D, L, device=device))
            for v, dt in zip(vf_list, dt_list)
        ]

        cond = torch.cat(cond_chunks, dim=1)

        x = model._encode_vf(x_t, dt_future, torch.zeros(1, D, L, device=device))

        s = model.step_enc(t_step).unsqueeze(1)
        x = torch.cat([x, s], dim=1)

        for sa, ca in zip(model.sa_blocks, model.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        x = x[:, :-1, :]
        x = model.out(x.permute(0, 2, 1)).squeeze(1)

        return x

    # --------------------------------------------------
    # configurations
    # --------------------------------------------------
    configs = {
        "VF1 VF2 VF1 VF2": [VF1, VF2, VF1, VF2],
        "VF2 VF1 VF2 VF1": [VF2, VF1, VF2, VF1],
        "VF1 VF1 VF2 VF2": [VF1, VF1, VF2, VF2],
        "VF2 VF2 VF1 VF1": [VF2, VF2, VF1, VF1],
    }

    dt_map = {
        "VF1 VF2 VF1 VF2": [dt1, dt2, dt1, dt2],
        "VF2 VF1 VF2 VF1": [dt2, dt1, dt2, dt1],
        "VF1 VF1 VF2 VF2": [dt1, dt1, dt2, dt2],
        "VF2 VF2 VF1 VF1": [dt2, dt2, dt1, dt1],
    }

    outputs = {}

    for name in configs:
        outputs[name] = run(configs[name], dt_map[name])

    # --------------------------------------------------
    # comparisons
    # --------------------------------------------------
    def diff(a, b):
        return (a - b).abs().max().item()

    print("=" * 70)
    print("VF PERMUTATION + REPETITION SYMMETRY TEST")
    print("=" * 70)

    base = outputs["VF1 VF2 VF1 VF2"]

    for name, out in outputs.items():
        print(f"{name:25s} max error vs base: {diff(out, base):.6e}")

    print("=" * 70)

    return outputs

# ======================================================
# Example usage
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(
        "diffusion",
        L=59,
        D=64,
        K=2,
        max_cond=2,
    ).to(DEVICE)

ckpt = torch.load("denoiser_ablation_diffusion_oct_vf4_3vf_pretrained.pth", map_location=DEVICE)
state = ckpt["model_state_dict"]
model.load_state_dict(state, strict=False)
model.eval()

result = test_permutation_equivariance(
    model,
    batch_size=8,
)

result_no_pos = test_permutation_equivariance_no_pos(
    model,
    batch_size=8,
)

result_encoder = inspect_encoder_equivariance(model, batch_size=2)

result_encoder_no_pos = inspect_encoder_equivariance_no_pos(model, batch_size=2)

result_token_equiv = test_token_equiv_with_conditioning(model, batch_size=8)
result_vf_order_symmetry = test_vf_order_symmetry(model, batch_size=8)
result_full_symmetry = test_full_symmetry(model, batch_size=8)

result_sa_no_mlp = test_sa_no_mlp(model.sa_blocks[0], dim=64, L=59, B=4, device=DEVICE)
result_layernorm = test_layernorm(L=59, dim=64, B=4, device=DEVICE)
result_cross_attention = test_cross_attention_block(model.ca_blocks[0], dim=64, L=59, B=4, device=DEVICE)
result_self_attention = test_self_attention_block(model.sa_blocks[0], dim=64, L=59, B=4, device=DEVICE)
result_full_model_debug = full_model_debug(model, batch_size=4)
result_encoder_debug = test_encoder(model, B=4, L=59, device=DEVICE)

result_layerwise_trace = layerwise_equivariance_trace(model, batch_size=4)
result_residual_growth = residual_growth_test(model)
result_attention_stability = attention_weight_stability(model.sa_blocks[0], L=59, B=2, dim=64)

result_equivariance_check = equivariance_check(model, batch_size=8, atol=1e-5)

result_vf_permutation_repetition = vf_permutation_repetition_test(model)