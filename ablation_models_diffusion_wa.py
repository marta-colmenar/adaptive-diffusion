import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------
def positional_encoding(t, out_features):
    """
    Computes positional encoding for a given time step t
    """
    i = torch.arange(out_features // 2, device=t.device)
    angles = t[:, None] / (10000 ** (2 * i / out_features))
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class PEBlock(nn.Module):
    """Sinusoidal PE + linear projection"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, t):
        return self.fc(positional_encoding(t, self.fc.in_features))

def positional_encoding_td(t, out_features, t_min=-6.0, t_max=3.0, new_min=0, new_max=470.0):
    t = (t - t_min) / (t_max - t_min)         
    t = t * (new_max - new_min) + new_min     

    i = torch.arange(out_features // 2, device=t.device)
    denom = 10000 ** (2 * i / out_features)
    angles = t.unsqueeze(-1) / denom
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# ------------------------------------------------------------
# Attention blocks
# ------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        h, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond):
        x_norm = self.norm1(x)
        h, _ = self.attn(x_norm, cond, cond, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

''' OLD MODELS 
# ============================================================
# ABLATION MODEL 1: Baseline (No Attention, No Spatial Encoding)
# ============================================================
class VFDiffusionBaseline(nn.Module):
    """
    Simplest model: projection + MLP only.
    No spatial encoding, no temporal encoding, no attention.
    """

    def __init__(self, L=52, D=32, max_cond=2, **kwargs):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # layer norm
        self.norm = nn.LayerNorm(D * L)

        # MLP for processing
        self.mlp = nn.Sequential(
            nn.Linear(D * L, 4*(D * L)),
            nn.ReLU(),
            nn.Linear(4*(D * L), 4*(D * L)),
            nn.ReLU(),
            nn.Linear(4*(D * L), L)
        )

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, _dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            cond_sum += self.vf_weights[i] * z

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)

        # Flatten spatial dimensions
        x_flat = x.view(B, -1)                                   # (B, D*L)
        cond_flat = cond_sum.view(B, -1)                         # (B, D*L)
        
        # Concatenate all features
        features = x_flat + cond_flat

        # Predict noise
        eps_hat = self.mlp(self.norm(features))                   # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 1: Baseline (No Attention, No Spatial Encoding)
# ============================================================
class VFDiffusionWithTemporalEncoding(nn.Module):
    """
    Simplest model: Only temporal encodings and MLPs.
    No spatial encoding, no attention.
    """

    def __init__(self, L=52, D=32, max_cond=2, **kwargs):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # layer norm
        self.norm = nn.LayerNorm(D * L + 2 * D)

        # MLP for processing
        self.mlp = nn.Sequential(
            nn.Linear(D * L + 2 * D, 4*(D * L + 2 * D)),
            nn.ReLU(),
            nn.Linear(4*(D * L + 2 * D), 4*(D * L + 2 * D)),
            nn.ReLU(),
            nn.Linear(4*(D * L + 2 * D), L)
        )

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)

        # Flatten spatial dimensions
        x_flat = x.view(B, -1)                                   # (B, D*L)
        cond_flat = cond_sum.view(B, -1)                         # (B, D*L)
        
        # Conditioning tokens
        s_token = self.step_enc(t_step)                          # (B, D)
        h_token = self.horizon_enc(dt_future)                    # (B, D)

        # Concatenate all features
        features = torch.cat([x_flat + cond_flat, s_token, h_token], dim=-1)

        # Predict noise
        eps_hat = self.mlp(self.norm(features))                   # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 2: With Spatial Encoding (No Attention)
# ============================================================
class VFDiffusionWithSpatialEncoding(nn.Module):
    """
    Adds spatial positional encoding to the baseline.
    Still no attention mechanism.
    """

    def __init__(self, L=52, D=32, max_cond=2, **kwargs):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)  # Spatial encoding
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # layer norm
        self.norm = nn.LayerNorm(D * L + 2 * D)

        # MLP for processing
        self.mlp = nn.Sequential(
            nn.Linear(D * L + 2 * D, 4*(D * L + 2 * D)),
            nn.ReLU(),
            nn.Linear(4*(D * L + 2 * D), 4*(D * L + 2 * D)),
            nn.ReLU(),
            nn.Linear(4*(D * L + 2 * D), L)
        )

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        cond_sum = cond_sum + pos  # Add spatial encoding

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos  # Add spatial encoding

        # Flatten spatial dimensions
        x_flat = x.view(B, -1)                                   # (B, D*L)
        cond_flat = cond_sum.view(B, -1)                         # (B, D*L)
        
        # Conditioning tokens
        s_token = self.step_enc(t_step)                          # (B, D)
        h_token = self.horizon_enc(dt_future)                    # (B, D)

        # Concatenate all features
        features = torch.cat([x_flat + cond_flat, s_token, h_token], dim=-1)

        # Predict noise
        eps_hat = self.mlp(self.norm(features))                   # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 3: With Self-Attention (No Cross-Attention)
# ============================================================
class VFDiffusionWithSelfAttention(nn.Module):
    """
    Adds self-attention blocks to capture spatial dependencies.
    No cross-attention with conditioning yet.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # Self-attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(D) for _ in range(K)
        ])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs (combined into x)
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        cond_sum = cond_sum + pos

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos

        # Combine conditioning with noisy input
        x = x + cond_sum

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)            # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)      # (B, 1, D)

        x = x.permute(0, 2, 1)                                  # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)             # (B, L+2, D)

        # Self-attention blocks
        for sa in self.sa_blocks:
            x = sa(x)

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                        # (B, L, D)
        x = x.permute(0, 2, 1)                                  # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                        # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 4: With Self-Attention and MLP Conditioning
# ============================================================
class VFDiffusionWithSelfAttentionMLPCond(nn.Module):
    """
    Self-attention blocks with MLP-based conditioning.
    Conditions via MLPs instead of cross-attention mechanism.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # Self-attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(D) for _ in range(K)
        ])

        # MLP-based conditioning layers
        self.cond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D * L, 2 * D * L),
                nn.ReLU(),
                nn.Linear(2 * D * L, D * L)
            ) for _ in range(K)
        ])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        cond = cond_sum + pos                                   # (B, D, L)
        cond_flat = cond.view(B, -1)                            # (B, D*L)

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)            # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)      # (B, 1, D)

        x = x.permute(0, 2, 1)                                  # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)             # (B, L+2, D)

        # Self-attention + MLP conditioning blocks
        for sa, cond_mlp in zip(self.sa_blocks, self.cond_mlps):
            x = sa(x)  # Self-attention already includes residual connection
            
            # Process conditioning through MLP and add to features
            cond_processed = cond_mlp(cond_flat)                # (B, D*L)
            cond_processed = cond_processed.view(B, self.L, self.D)  # (B, L, D)
            
            # Add conditioning to spatial features only (not tokens)
            x[:, :-2, :] = x[:, :-2, :] + cond_processed

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                        # (B, L, D)
        x = x.permute(0, 2, 1)                                  # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                        # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 5: With Cross-Attention (Separate Conditioning)
# ============================================================
class VFDiffusionWithCrossAttention(nn.Module):
    """
    Full model with both self-attention and cross-attention.
    Conditioning VFs are processed separately via cross-attention.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # Attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(D) for _ in range(K)
        ])

        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(D) for _ in range(K)
        ])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        cond = cond_sum + pos
        cond = cond.permute(0, 2, 1)                            # (B, L, D)

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)            # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)      # (B, 1, D)

        x = x.permute(0, 2, 1)                                  # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)             # (B, L+2, D)

        # Attention blocks
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                        # (B, L, D)
        x = x.permute(0, 2, 1)                                  # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                        # (B, L)
        return eps_hat

# ============================================================
# ABLATION MODEL 6: Full Model with Lambda Conditioning
# ============================================================
class VFDiffusionFull(nn.Module):
    """
    Complete model with learnable conditioning strength λ.
    This is identical to the original VFConditionalDiffusion.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # Attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(D) for _ in range(K)
        ])

        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(D) for _ in range(K)
        ])

        # Conditioning strength λ
        self.lambda_cond = nn.Parameter(torch.tensor(1.0))

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += self.vf_weights[i] * z

        cond = cond_sum + pos
        cond = cond.permute(0, 2, 1)                            # (B, L, D)

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)            # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)      # (B, 1, D)

        x = x.permute(0, 2, 1)                                  # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)             # (B, L+2, D)

        # Attention blocks
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + self.lambda_cond * ca(x, cond)

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                        # (B, L, D)
        x = x.permute(0, 2, 1)                                  # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                        # (B, L)
        return eps_hat


class VFDiffusionWithCrossAttentionSinId(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Per-VF linear layers for VALUE sin encodings
        self.vf_value_linear = nn.ModuleList([nn.Linear(D*2, D) for _ in range(max_cond)])
        self.vf_xt_linear = nn.Linear(D*2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)              # (1, D, L)

        # Conditioning VFs (summed, like original SinId)
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = positional_encoding_td(v_i.flatten(), self.D*2)  
            z = self.vf_value_linear[i](z).reshape(B, self.L, self.D).permute(0, 2, 1)  # (B, D, L)
            z = z + self.time_enc(dt_i).unsqueeze(-1)

            cond_sum += z

        cond = cond_sum + pos
        cond = cond.permute(0, 2, 1)                             # (B, L, D)

        # Noisy target VF
        x = positional_encoding_td(x_t.flatten(), self.D*2) 
        x = self.vf_xt_linear(x).reshape(B, self.L, self.D).permute(0, 2, 1)  # (B, D, L)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)             # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)       # (B, 1, D)

        x = x.permute(0, 2, 1)                                   # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)              # (B, L+2, D)

        # Attention blocks
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                         # (B, L, D)
        x = x.permute(0, 2, 1)                                   # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                         # (B, L)
        return eps_hat

class VFDiffusionWithCrossAttentionBeforeTime(nn.Module):
    """
    Full model with both self-attention and cross-attention.
    Conditioning VFs are processed separately via cross-attention.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond

        # Fixed spatial positions
        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)

        # VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(max_cond, D, 1))

        # Learnable VF weights
        self.vf_weights = nn.Parameter(torch.ones(max_cond))

        # Attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttentionBlock(D) for _ in range(K)
        ])

        self.ca_blocks = nn.ModuleList([
            CrossAttentionBlock(D) for _ in range(K)
        ])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU() #??

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)             # (1, D, L)

        # Conditioning VFs
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = self.relu(self.vf_proj(v_i.unsqueeze(1)))      # (B, D, L)
            z = z + self.vf_id_emb[i]
            z = self.vf_weights[i] * z
            z = z + self.time_enc(dt_i).unsqueeze(-1)
            cond_sum += z

        cond = cond_sum + pos
        cond = cond.permute(0, 2, 1)                            # (B, L, D)

        # Noisy target VF
        x = self.relu(self.vf_proj(x_t.unsqueeze(1)))           # (B, D, L)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)            # (B, 1, D)
        h_token = self.horizon_enc(dt_future).unsqueeze(1)      # (B, 1, D)

        x = x.permute(0, 2, 1)                                  # (B, L, D)
        x = torch.cat([x, s_token, h_token], dim=1)             # (B, L+2, D)

        # Attention blocks
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove tokens and predict noise
        x = x[:, :-2, :]                                        # (B, L, D)
        x = x.permute(0, 2, 1)                                  # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                        # (B, L)
        return eps_hat

class VFDiffusionWithCrossAttentionSinIdSumTime(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()

        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.vf_proj = nn.Conv1d(1, D, kernel_size=1)
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.horizon_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Per-VF linear layers for VALUE sin encodings
        self.vf_value_linear = nn.ModuleList([nn.Linear(D*2, D) for _ in range(max_cond)])
        self.vf_xt_linear = nn.Linear(D*2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)
        n_cond = len(v_hist)

        # Spatial positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)              # (1, D, L)

        # Conditioning VFs (summed, like original SinId)
        cond_sum = 0.0
        for i, (v_i, dt_i) in enumerate(zip(v_hist, dt_hist)):
            z = positional_encoding_td(v_i.flatten(), self.D*2)  
            z = self.vf_value_linear[i](z).reshape(B, self.L, self.D).permute(0, 2, 1)  # (B, D, L)
            z = z + self.time_enc(dt_i).unsqueeze(-1)

            cond_sum += z

        cond = cond_sum + pos
        cond = cond.permute(0, 2, 1)                             # (B, L, D)

        # Noisy target VF
        x = positional_encoding_td(x_t.flatten(), self.D*2) 
        x = self.vf_xt_linear(x).reshape(B, self.L, self.D).permute(0, 2, 1)  # (B, D, L)
        x = x + self.horizon_enc(dt_future).unsqueeze(-1)       # (B, 1, D)
        x = x + pos

        # Conditioning tokens
        s_token = self.step_enc(t_step).unsqueeze(1)             # (B, 1, D)        

        x = x.permute(0, 2, 1)                                   # (B, L, D)
        x = torch.cat([x, s_token], dim=1)              # (B, L+2, D)

        # Attention blocks
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove tokens and predict noise
        x = x[:, :-1, :]                                         # (B, L, D)
        x = x.permute(0, 2, 1)                                   # (B, D, L)

        eps_hat = self.out(x).squeeze(1)                         # (B, L)
        return eps_hat

class NewModelAllTogether(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        # time_enc(dt): (B, D) -> (B, 1, D)
        z = z + self.time_enc(dt).unsqueeze(1)

        # pos: (1, D, L) -> (1, L, D)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)

        # Spatial positional encoding: (1, D, L) -> (1, D, L)
        pos = self.pos_enc(self.loc).T.unsqueeze(0)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            # (B, L, D)
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        # (B, L, D)
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat

class NewModelAllTogetherCondSA(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])
        self.cond_sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        # time_enc(dt): (B, D) -> (B, 1, D)
        z = z + self.time_enc(dt).unsqueeze(1)

        # pos: (1, D, L) -> (1, L, D)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)

        # Spatial positional encoding: (1, D, L) -> (1, D, L)
        pos = self.pos_enc(self.loc).T.unsqueeze(0)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            # (B, L, D)
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        # (B, L, D)
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca, cond_sa in zip(self.sa_blocks, self.ca_blocks, self.cond_sa_blocks):
            x = sa(x)
            cond_sa =  cond_sa(x)
            x = x + ca(x, cond_sa)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat

class NewModelAllTogetherSA(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        # time_enc(dt): (B, D) -> (B, 1, D)
        z = z + self.time_enc(dt).unsqueeze(1)

        # pos: (1, D, L) -> (1, L, D)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)

        # Spatial positional encoding: (1, D, L) -> (1, D, L)
        pos = self.pos_enc(self.loc).T.unsqueeze(0)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            # (B, L, D)
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        # (B, L, D)
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([cond, x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa in self.sa_blocks:
            x = sa(x)

        # Remove token -> (B, L, D)
        x = x[:, cond.size(1):-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat
'''

# ============================================================
# FINAL MODELS
# ============================================================
class DiffusionModel(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        # time_enc(dt): (B, D) -> (B, 1, D)
        z = z + self.time_enc(dt).unsqueeze(1)

        # pos: (1, D, L) -> (1, L, D)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        B = x_t.size(0)

        # Spatial positional encoding: (1, D, L) -> (1, D, L)
        pos = self.pos_enc(self.loc).T.unsqueeze(0)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            # (B, L, D)
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        # (B, L, D)
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat

class DiffusionModelSeenLocations(nn.Module):
    """
    Same architecture as DiffusionModel, with one extra conditioning stream that
    encodes already-seen future locations.
    """

    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        z = positional_encoding_td(x, self.D * 2)
        z = self.td_enc(z)
        z = z + self.time_enc(dt).unsqueeze(1)
        z = z + pos.permute(0, 2, 1)
        return z
    
    def _encode_seen_tokens(self, seen_values, seen_loc_idx, dt_future, pos):
        if seen_values is None or seen_loc_idx is None:
            return None

        z = positional_encoding_td(seen_values, self.D * 2)
        z = self.td_enc(z)

        z = z + self.time_enc(dt_future).unsqueeze(1)

        pos_l = self.pos_enc(self.loc[seen_loc_idx]).unsqueeze(0)
        z = z + pos_l

        return z

    def forward(
        self,
        x_t,
        v_hist,
        dt_hist,
        t_step,
        dt_future,
        seen_values=None,
        seen_loc_idx=None,
    ):
        # Spatial positional encoding: (1, D, L)
        pos = self.pos_enc(self.loc).T.unsqueeze(0)
        pos = pos.expand(x_t.size(0), -1, -1)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        seen_cond = self._encode_seen_tokens(seen_values, seen_loc_idx, dt_future, pos)
        if seen_cond is not None:
            cond_chunks.append(seen_cond)

        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token
        s_token = self.step_enc(t_step).unsqueeze(1)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)
        return eps_hat

class DiffusionModelXY(nn.Module):
    def __init__(
        self,
        L=52,
        D=32,
        K=2,
        max_cond=2,
        value_scale=1.0,
        x_coords=None,
        y_coords=None,
    ):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        if x_coords is None or y_coords is None:
            raise ValueError("NewModelAllTogetherXY requires x_coords and y_coords")

        x_coords = torch.as_tensor(x_coords, dtype=torch.float32)
        y_coords = torch.as_tensor(y_coords, dtype=torch.float32)

        if x_coords.numel() != L or y_coords.numel() != L:
            raise ValueError(
                f"x_coords and y_coords must both have length L={L}, got {x_coords.numel()} and {y_coords.numel()}"
            )

        self.register_buffer("x_coord", x_coords)
        self.register_buffer("y_coord", y_coords)

        # Separate positional encoders for X and Y coordinates
        self.pos_enc_x = PEBlock(D, D)
        self.pos_enc_y = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _spatial_encoding(self):
        # x/y encodings are both (L, D); combine position-wise
        pos_x = self.pos_enc_x(self.x_coord)
        pos_y = self.pos_enc_y(self.y_coord)
        pos = pos_x + pos_y
        return pos.T.unsqueeze(0)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        z = z + self.time_enc(dt).unsqueeze(1)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future):
        # Spatial positional encoding from x/y coordinates
        pos = self._spatial_encoding()

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i in zip(v_hist, dt_hist):
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat


class DiffusionModelLoc(nn.Module):
    def __init__(self, L=52, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        # self.register_buffer("loc", torch.arange(L).float())

        # Encoders
        self.pos_enc = PEBlock(D, D)
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

    def _encode_vf(self, x, dt, pos):
        """
        x:  (B, L)
        dt: (B,) or scalar-like
        pos: (1, D, L)
        returns: (B, L, D)
        """
        # (B, L, 2D)
        z = positional_encoding_td(x, self.D * 2)

        # (B, L, D)
        z = self.td_enc(z)

        # Add time + spatial encodings
        # time_enc(dt): (B, D) -> (B, 1, D)
        z = z + self.time_enc(dt).unsqueeze(1)

        # pos: (1, D, L) -> (1, L, D)
        z = z + pos.permute(0, 2, 1)

        return z

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future, locs):
        B = x_t.size(0)

        # ---- Conditioning VFs ----
        cond_chunks = []
        for v_i, dt_i, loc_i in zip(v_hist, dt_hist, locs[:len(v_hist)]):
            # Spatial positional encoding: (1, D, L) -> (1, D, L)
            pos = self.pos_enc(loc_i).T.unsqueeze(0)
            # (B, L, D)
            cond_chunks.append(self._encode_vf(v_i, dt_i, pos))

        # (B, max_cond*L, D)
        cond = torch.cat(cond_chunks, dim=1)

        # ---- Noisy target VF ----
        pos = self.pos_enc(locs[len(v_hist)]).T.unsqueeze(0)
        # (B, L, D)
        x = self._encode_vf(x_t, dt_future, pos)

        # Step token: (B, 1, D)
        s_token = self.step_enc(t_step).unsqueeze(1)

        # Append token: (B, L+1, D)
        x = torch.cat([x, s_token], dim=1)

        # ---- Attention blocks ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + ca(x, cond)

        # Remove token -> (B, L, D)
        x = x[:, :-1, :]

        # Conv1d expects (B, D, L)
        x = x.permute(0, 2, 1).contiguous()

        eps_hat = self.out(x).squeeze(1)  # (B, L)
        return eps_hat



class SelfAttentionBlockMask(nn.Module):
    def __init__(self, D, n_heads=4):
        super().__init__()
        self.D = D
        self.n_heads = n_heads
        self.head_dim = D // n_heads

        assert D % n_heads == 0, "D must be divisible by n_heads"

        self.qkv = nn.Linear(D, D * 3)
        self.out = nn.Linear(D, D)

        self.norm = nn.LayerNorm(D)

    def forward(self, x, valid_mask=None):
        """
        x: (B, L, D)
        valid_mask: (B, L) with 1 for valid, 0 for padded
        """
        B, L, D = x.shape

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head
        def reshape_heads(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            # (B, H, L, head_dim)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if valid_mask is not None:
            # mask keys (prevent reading from padded positions)
            key_mask = valid_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(key_mask == 0, mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

        # merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        if valid_mask is not None:
            # mask queries (prevent padded positions from computing anything)
            query_mask = valid_mask.unsqueeze(-1)  # (B, L, 1)
            attn_out = attn_out * query_mask

        out = self.out(attn_out)

        return x + out  # residual

class CrossAttentionBlockMask(nn.Module):
    def __init__(self, D, n_heads=4):
        super().__init__()
        self.D = D
        self.n_heads = n_heads
        self.head_dim = D // n_heads

        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, D * 2)
        self.out = nn.Linear(D, D)

        self.norm = nn.LayerNorm(D)

    def forward(self, x, cond, valid_mask=None, cond_mask=None):
        """
        x: (B, L, D)
        cond: (B, Lc, D)
        valid_mask: (B, L) - mask for x queries
        cond_mask: (B, Lc) - mask for cond keys
        """
        B, L, D = x.shape
        Lc = cond.shape[1]

        x_norm = self.norm(x)

        q = self.q(x_norm)
        kv = self.kv(cond)
        k, v = kv.chunk(2, dim=-1)

        def reshape_heads(t, length):
            return t.view(B, length, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, L)
        k = reshape_heads(k, Lc)
        v = reshape_heads(v, Lc)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, Lc)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if cond_mask is not None:
            # mask conditioning keys
            mask = cond_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lc)
            mask_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask == 0, mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        if valid_mask is not None:
            # mask query positions (prevent padded x from computing anything)
            query_mask = valid_mask.unsqueeze(-1)  # (B, L, 1)
            attn_out = attn_out * query_mask

        out = self.out(attn_out)

        return out

class DiffusionModelXYVariable(nn.Module):
    """
    Improved diffusion model with per-sample x/y coordinates and optional valid-location mask.
    - Fused coordinate encoding (learnable 2D->D projection)
    - Dataset-aware affine scaling (per-domain spatial modulation)
    - Vectorized VF encoding (batch processing)
    - Unified mask handling without mask tokens
    """

    def __init__(self, L=59, D=32, K=2, max_cond=2, value_scale=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.max_cond = max_cond
        self.value_scale = value_scale

        # Fused coordinate encoder: (2,) -> D
        self.coord_enc = nn.Sequential(
            nn.Linear(2, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, D)
        )
        
        self.step_enc = PEBlock(D, D)
        self.time_enc = PEBlock(D, D)

        # Maps (2D) sinusoidal encoding -> D
        self.td_enc = nn.Linear(D * 2, D)

        # Attention blocks
        self.sa_blocks = nn.ModuleList([SelfAttentionBlockMask(D) for _ in range(K)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlockMask(D) for _ in range(K)])

        # Output
        self.out = nn.Conv1d(D, 1, kernel_size=1)

        # Dataset-aware affine transformation (learnable per-domain scaling)
        self.dataset_scale = nn.Embedding(2, D)
        self.dataset_shift = nn.Embedding(2, D)

    def _spatial_encoding(self, x_coords, y_coords):
        """
        Fused coordinate encoding with learnable projection.
        x_coords, y_coords: (B, L)
        returns: (B, L, D)
        """
        bsz, ldim = x_coords.shape
        if ldim != self.L:
            raise ValueError(f"Expected coords with L={self.L}, got L={ldim}")

        # Stack coordinates: (B, L, 2)
        coords = torch.stack([x_coords * 100, y_coords * 100], dim=-1)
        
        # Reshape for batch processing: (B*L, 2)
        coords_flat = coords.view(-1, 2)
        
        # Encode: (B*L, D) -> reshape to (B, L, D)
        pos = self.coord_enc(coords_flat).view(bsz, ldim, self.D)
        
        return pos

    def _encode_vf_batch(self, v_all, dt_all, pos):
        """
        Vectorized VF encoding for multiple timesteps.
        v_all: (T, B, L) - stacked VF history + target
        dt_all: (T, B) - stacked time deltas
        pos: (B, L, D) - spatial encoding
        returns: (T, B, L, D)
        """
        T, B, L = v_all.shape
        
        # TD encoding for all timesteps at once
        z_list = []
        for t in range(T):
            v_t = v_all[t]  # (B, L)
            dt_t = dt_all[t]  # (B,)
            
            # Sinusoidal encoding: (B, L) -> (B, L, 2D)
            z_t = positional_encoding_td(v_t, self.D * 2)
            
            # Project to D: (B, L, 2D) -> (B, L, D)
            z_t = self.td_enc(z_t)
            
            # Add time encoding: (B, D) -> (B, 1, D)
            z_t = z_t + self.time_enc(dt_t).unsqueeze(1)
            
            # Add spatial position: (B, L, D)
            z_t = z_t + pos
            
            z_list.append(z_t)
        
        return torch.stack(z_list, dim=0)  # (T, B, L, D)

    def forward(self, x_t, v_hist, dt_hist, t_step, dt_future, x_coords=None, y_coords=None, valid_mask=None, dataset_id=None):
        if x_coords is None or y_coords is None:
            raise ValueError("DiffusionModelXYVariable requires x_coords and y_coords in forward")

        if x_coords.ndim != 2 or y_coords.ndim != 2:
            raise ValueError("x_coords and y_coords must have shape (B, L)")

        B = x_t.size(0)
        device = x_t.device

        if valid_mask is not None:
            valid_mask = valid_mask.to(x_t.dtype)

        # Compute spatial encoding
        pos = self._spatial_encoding(x_coords, y_coords)  # (B, L, D)

        # Apply dataset-aware affine transformation if dataset_id provided
        if dataset_id is not None:
            dataset_id = torch.as_tensor(dataset_id, device=device, dtype=torch.long)
            if dataset_id.ndim == 0:
                dataset_id = dataset_id.unsqueeze(0)
            
            scale = self.dataset_scale(dataset_id)  # (B, D)
            shift = self.dataset_shift(dataset_id)  # (B, D)
            
            # Apply affine: (B, L, D) * (B, 1, D) + (B, 1, D)
            pos = pos * scale.unsqueeze(1) + shift.unsqueeze(1)

        # ---- Vectorized conditioning encoding ----
        v_hist_stacked = torch.stack(v_hist, dim=0)  # (T, B, L)
        dt_hist_stacked = torch.stack(dt_hist, dim=0)  # (T, B)
        cond_encoded = self._encode_vf_batch(v_hist_stacked, dt_hist_stacked, pos)  # (T, B, L, D)
        
        # Concatenate history: (B, T*L, D)
        cond = cond_encoded.permute(1, 0, 2, 3).reshape(B, -1, self.D)

        # ---- Target VF encoding ----
        x_encoded = self._encode_vf_batch(x_t.unsqueeze(0), dt_future.unsqueeze(0), pos)  # (1, B, L, D)
        x = x_encoded.squeeze(0)  # (B, L, D)

        # ---- Build attention input with step token ----
        s_token = self.step_enc(t_step).unsqueeze(1)  # (B, 1, D)
        x = torch.cat([x, s_token], dim=1)  # (B, L+1, D)

        # ---- Build unified masks ----
        x_mask = None
        cond_mask = None
        
        if valid_mask is not None:
            # Mask for target + always-valid step token
            step_mask = torch.ones(B, 1, device=device)
            x_mask = torch.cat([valid_mask, step_mask], dim=1)  # (B, L+1)
            
            # Mask for conditioning (repeated for each history item)
            cond_mask = valid_mask.repeat(1, len(v_hist))  # (B, T*L)

        # ---- Attention layers ----
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x, valid_mask=x_mask)
            x = x + ca(x, cond, valid_mask=x_mask, cond_mask=cond_mask)

        # ---- Output projection ----
        x = x[:, :-1, :]  # Remove step token: (B, L, D)
        x = x.permute(0, 2, 1).contiguous()  # (B, D, L)
        eps_hat = self.out(x).squeeze(1)  # (B, L)

        # Apply mask to output if provided
        if valid_mask is not None:
            eps_hat = eps_hat * valid_mask

        return eps_hat

'''============================================================
Model Registry for Easy Ablation Studies
============================================================
MODEL_REGISTRY = {
    'baseline': VFDiffusionBaseline,
    'temporal': VFDiffusionWithTemporalEncoding, 
    'spatial': VFDiffusionWithSpatialEncoding,
    'self_attn': VFDiffusionWithSelfAttention,
    'self_attn_mlp_cond': VFDiffusionWithSelfAttentionMLPCond,
    'cross_attn': VFDiffusionWithCrossAttention,
    'cross_attn_id': VFDiffusionWithCrossAttentionSinId,
    'cross_attn_id_scaled': VFDiffusionWithCrossAttentionSinId,
    'cross_attn_before_time': VFDiffusionWithCrossAttentionBeforeTime,
    'cross_attn_id_sum_time': VFDiffusionWithCrossAttentionSinIdSumTime,
    'full': VFDiffusionFull,
    'all_together': NewModelAllTogether
    'cross_attn': VFDiffusionWithCrossAttention,
    'all_together': NewModelAllTogether,
    'all_together_vf1': NewModelAllTogether,
    'all_together_vf1_new_training': NewModelAllTogether,
    'only_self_attn': NewModelAllTogetherSA,
}'''

MODEL_REGISTRY = {
    'diffusion': DiffusionModel,
    'diffusion_seen_locations': DiffusionModelSeenLocations,
    'diffusion_xy': DiffusionModelXY,
    'diffusion_xy_var': DiffusionModelXYVariable,
    'diffusion_loc': DiffusionModelLoc,
}


def get_model(model_name, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name: One of ['baseline', 'spatial', 'self_attn', 'cross_attn', 'full']
        **kwargs: Model hyperparameters (L, D, K, max_cond)
    
    Returns:
        Instantiated model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)


# ============================================================
# Example Usage
# ============================================================
# if __name__ == "__main__":
#     # Create all models for ablation study
#     L, D, K = 59, 32, 2
#     max_cond = 2
    
#     models = {
#         name: get_model(name, L=L, D=D, K=K, max_cond=max_cond)
#         for name in MODEL_REGISTRY.keys()
#     }
    
#     # Print parameter counts
#     print("Ablation Study - Model Complexity:\n")
#     print(f"{'Model':<15} {'Parameters':>12}")
#     print("-" * 30)
#     for name, model in models.items():
#         n_params = sum(p.numel() for p in model.parameters())
#         print(f"{name:<15} {n_params:>12,}")
    
#     # Test forward pass
#     print("\n\nTesting forward pass...")
#     batch_size = 4
#     x_t = torch.randn(batch_size, L)
#     v_hist = [torch.randn(batch_size, L) for _ in range(max_cond)]
#     dt_hist = [torch.randn(batch_size) for _ in range(max_cond)]
#     t_step = torch.randn(batch_size)
#     dt_future = torch.randn(batch_size)
    
#     for name, model in models.items():
#         try:
#             out = model(x_t, v_hist, dt_hist, t_step, dt_future)
#             print(f"✓ {name:<15} output shape: {out.shape}")
#         except Exception as e:
#             print(f"✗ {name:<15} failed: {e}")
