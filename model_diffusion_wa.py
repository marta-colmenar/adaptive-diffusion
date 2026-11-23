import torch, numpy as np, pandas as pd
import torch.nn as nn
from scipy.linalg import sqrtm

# -------------------------------------------------------------------------
# Dataset, model and utils
# --------------------------------------------------------------------

class PEBlock(nn.Module):
    def __init__(self, out_features_pe, out_features):
        super().__init__()
        self.fc = nn.Linear(out_features_pe, out_features)
        self.relu = nn.ReLU()
        self.out_features_pe = out_features_pe

    def forward(self, loc):
        loc_enc = positional_encoding(loc, self.out_features_pe)
        return self.relu(self.fc(loc_enc))

class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + out

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=16, num_heads=4):
        super(SelfAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.conv1x1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x) 
        attn_output += x  # Residual connection
        feed_forward_output = self.relu(self.conv1x1(attn_output.permute(0, 2, 1)))  # Conv1D expects (batch, channels, seq_len)
        feed_forward_output =  feed_forward_output.permute(0, 2, 1) # Convert back to (batch, seq_len, channels)
        return   feed_forward_output + attn_output# residual connection 

class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

    def forward(self, x, cond):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, cond, cond)
        return x + out

class VFNet(nn.Module):
    def __init__(self):
        super(VFNet, self).__init__()
        
        # Positional Encoding
        self.register_buffer("loc", torch.arange(52).float())
        self.pos_enc = PEBlock(32, 8)
        self.time_enc = PEBlock(32, 8)
        self.step_enc = PEBlock(16, 24)
        self.time_diff_enc = PEBlock(32,24)
        
        # Initial Convolutions
        self.conv_vf = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)
        
        # Self-Attention Blocks repeated 3 times
        self.self_attention_blocks = nn.ModuleList([SelfAttentionBlock(24) for _ in range(3)])
        
        # Output Layer
        self.conv_out = nn.Conv1d(24, 1, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, vf1, vf2, vf3, t_step, t2_diff, t3_diff):
        # Apply positional encoding
        pos_enc = self.pos_enc(self.loc).T  # Shape: (8, 52)
        
        # Convolutional encoding
        vf1 = self.relu(self.conv_vf(vf1.unsqueeze(1))) # Shape: (batch, 8, 52)
        vf2 = self.relu(self.conv_vf(vf2.unsqueeze(1))) # Shape: (batch, 8, 52)
        vf3 = self.relu(self.conv_vf(vf3.unsqueeze(1))) # Shape: (batch, 8, 52)
        # vf4 = self.relu(self.conv_vf(vf4.unsqueeze(1))) # Shape: (batch, 8, 52)
        
        # Add position encoding of locations
        vf1 = vf1 + pos_enc.unsqueeze(0)
        vf2 = vf2 + pos_enc.unsqueeze(0)
        vf3 = vf3 + pos_enc.unsqueeze(0)
        # vf4 = vf4 + pos_enc.unsqueeze(0)

        # Add time encoding of condition VFs
        vf1 = vf1 + self.time_enc(torch.zeros_like(t2_diff)).unsqueeze(-1)
        vf2 = vf2 + self.time_enc(t2_diff).unsqueeze(-1)
        # vf3 = vf3 + self.time_enc(t3_diff).unsqueeze(-1)
        x = torch.cat([vf1, vf2, vf3], dim=1) # Shape: (batch, 24, 52)

        # Add the step encoding
        step_enc = self.step_enc(t_step) # Shape: (batch, 32)

        # Add the diff time encoding
        time_diff_enc = self.time_diff_enc(t3_diff)
        
        x = torch.cat([x, step_enc.unsqueeze(-1), time_diff_enc.unsqueeze(-1)], dim=-1) # Shape: (batch, 32, 61)  
        x = x.permute(0, 2, 1) # Change to (batch, 61, 16) 

        # Self-Attention Blocks (x3)
        for block in self.self_attention_blocks:
            x = block(x) # Shape: (batch, 61, 16)

        # Final output
        x = x[:, :-2, :]  # Drop the last row (batch, 52, 16)
        x = x.permute(0, 2, 1) # Change back to (batch, 16, 52)
        x = self.conv_out(x).squeeze(1)   # Shape: (batch, 52)

        return x # Shape: (batch, 52)

class VFNetSum(nn.Module):
    def __init__(self):
        super(VFNetSum, self).__init__()
        
        # Positional Encoding
        self.register_buffer("loc", torch.arange(52).float())
        self.pos_enc = PEBlock(32, 32)
        self.time_enc = PEBlock(32, 32)
        self.step_enc = PEBlock(16, 32)
        self.time_diff_enc = PEBlock(32, 32)
        
        # Initial Convolution
        self.conv_vf = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1)
        
        # Learnable VF weights (importance)
        self.vf_weights = nn.Parameter(torch.ones(3))
        
        # Learned VF identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(3, 32, 1))
        
        # Self-Attention Blocks
        self.self_attention_blocks = nn.ModuleList(
            [SelfAttentionBlock(32) for _ in range(3)]
        )
        
        # Output
        self.conv_out = nn.Conv1d(32, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, vf1, vf2, vf3, t_step, t2_diff, t3_diff):
        # Positional encoding
        pos_enc = self.pos_enc(self.loc).T  # (32, 52)
        
        # Encode VFs
        vf1 = self.relu(self.conv_vf(vf1.unsqueeze(1)))  # (B, 32, 52)
        vf2 = self.relu(self.conv_vf(vf2.unsqueeze(1)))
        vf3 = self.relu(self.conv_vf(vf3.unsqueeze(1)))

        # Add time encoding per VF
        vf1 = vf1 + self.time_enc(torch.zeros_like(t2_diff)).unsqueeze(-1)
        vf2 = vf2 + self.time_enc(t2_diff).unsqueeze(-1)
        vf3 = vf3 + self.time_enc(t3_diff).unsqueeze(-1)

        # Add VF identity embeddings
        vf1 = vf1 + self.vf_id_emb[0]
        vf2 = vf2 + self.vf_id_emb[1]
        vf3 = vf3 + self.vf_id_emb[2]

        # Weighted summation
        x = (
            self.vf_weights[0] * vf1 +
            self.vf_weights[1] * vf2 +
            self.vf_weights[2] * vf3
        )

        # Add positional encoding
        x = x + pos_enc.unsqueeze(0)  # (B, 32, 52)

        # Global conditioning tokens
        step_enc = self.step_enc(t_step)          # (B, 32)
        time_diff_enc = self.time_diff_enc(t3_diff)  # (B, 32)

        # Append as tokens
        x = torch.cat(
            [x, step_enc.unsqueeze(-1), time_diff_enc.unsqueeze(-1)],
            dim=-1
        )  # (B, 32, 54)

        x = x.permute(0, 2, 1)  # (B, 54, 32)

        # Attention
        for block in self.self_attention_blocks:
            x = block(x)

        # Remove conditioning tokens
        x = x[:, :-2, :]  # (B, 52, 32)
        x = x.permute(0, 2, 1)  # (B, 32, 52)

        # Output
        x = self.conv_out(x).squeeze(1)  # (B, 52)
        return x

class VFNetDiffusion(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 32

        # Location buffer
        self.register_buffer("loc", torch.arange(52).float())

        # Encodings
        self.pos_enc = PEBlock(dim, dim)
        self.time_enc = PEBlock(dim, dim)
        self.step_enc = PEBlock(dim, dim)
        self.time_diff_enc = PEBlock(dim, dim)  # t3_diff

        # VF encoders
        self.conv_vf = nn.Conv1d(1, dim, kernel_size=1)

        # Identity embeddings
        self.vf_id_emb = nn.Parameter(torch.randn(2, dim, 1))

        # Attention blocks
        self.sa_blocks = nn.ModuleList([
            SelfAttnBlock(dim),
            SelfAttnBlock(dim)
        ])

        self.ca_blocks = nn.ModuleList([
            CrossAttnBlock(dim),
            CrossAttnBlock(dim)
        ])

        # Output
        self.conv_out = nn.Conv1d(dim, 1, kernel_size=1)

        # Conditioning strength
        self.cond_scale = nn.Parameter(torch.tensor(1.0))

        self.relu = nn.ReLU()

    def forward(self, vf1, vf2,  x_t, t_step, t2_diff, t3_diff):
        B = x_t.shape[0]

        # Positional encoding
        pos = self.pos_enc(self.loc).T.unsqueeze(0)  # (1, 32, 52)

        # --- Encode conditioning VFs ---
        vf1 = self.relu(self.conv_vf(vf1.unsqueeze(1)))
        vf2 = self.relu(self.conv_vf(vf2.unsqueeze(1)))

        vf1 = vf1 + self.vf_id_emb[0]
        vf2 = vf2 + self.vf_id_emb[1]

        vf1 = vf1 + self.time_enc(torch.zeros_like(t2_diff)).unsqueeze(-1)
        vf2 = vf2 + self.time_enc(t2_diff).unsqueeze(-1)

        # Add future horizon encoding to conditioning
        horizon_enc = self.time_diff_enc(t3_diff).unsqueeze(-1)
        vf1 = vf1 + horizon_enc
        vf2 = vf2 + horizon_enc

        cond = torch.cat([vf1, vf2], dim=-1)  # (B, 32, 104)
        cond = cond + pos.repeat(1, 1, 2)
        cond = cond.permute(0, 2, 1)  # (B, 104, 32)

        # --- Encode noisy target ---
        x = self.relu(self.conv_vf(x_t.unsqueeze(1)))
        x = x + pos

        # Diffusion step embedding
        t_emb = self.step_enc(t_step).unsqueeze(-1)

        # Future horizon embedding
        x = x + t_emb + horizon_enc

        x = x.permute(0, 2, 1)  # (B, 52, 32)

        # --- SA + CA blocks ---
        for sa, ca in zip(self.sa_blocks, self.ca_blocks):
            x = sa(x)
            x = x + self.cond_scale * ca(x, cond)

        # --- Output ---
        x = x.permute(0, 2, 1)
        x = self.conv_out(x).squeeze(1)

        return x


def f(t, num_steps, s=8e-3):
    t = torch.as_tensor(t)
    return torch.cos((t/num_steps + s)/(1 + s) * torch.pi/2)**2

def calculate_fid(features_real, features_generated):
    mu_real, sigma_real = np.mean(features_real, 0), np.cov(features_real, rowvar=False)
    mu_gen, sigma_gen = np.mean(features_generated, 0), np.cov(features_generated, rowvar=False)

    mean_diff = np.sum((mu_real - mu_gen) ** 2)

    cov_sqrt, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    # Take real part to avoid complex number warnings (imaginary part should be ~0)
    cov_sqrt = np.real(cov_sqrt)
    
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return np.real(fid)  # Ensure final result is real

def positional_encoding(t, out_features):
    """
    Computes positional encoding for a given time step t
    """
    i = torch.arange(out_features // 2, device=t.device)
    angles = t[:, None] / (10000 ** (2 * i / out_features))
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
