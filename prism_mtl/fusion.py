"""Task-Specific Modality Fusion (TSMF) and the task-agnostic baseline."""
import torch
import torch.nn as nn


def _ffn(d_model, dropout):
    return nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * 4, d_model),
        nn.Dropout(dropout),
    )


class TSMF(nn.Module):
    """Learnable task tokens cross-attend to the modality tokens of each time step.

    in : modality tokens (B, T, M, E)
    out: task tokens     (B, K, T, D)
    """

    def __init__(self, embed=128, d_model=256, num_tasks=4, num_modalities=5,
                 heads=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_tasks = int(num_tasks)
        self.num_modalities = int(num_modalities)

        self.mod_proj = nn.Linear(embed, d_model)
        self.task_tokens = nn.Parameter(torch.randn(num_tasks, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_dropout)

    def forward(self, modality_tokens, return_attn=False):
        B, T, M, E = modality_tokens.shape
        assert M == self.num_modalities, f"expected {self.num_modalities} modalities, got {M}"

        kv = self.mod_proj(modality_tokens).view(B * T, M, self.d_model)
        q = self.task_tokens.unsqueeze(0).expand(B * T, -1, -1)

        kvn = self.norm1(kv)
        out, attn_w = self.attn(self.norm1(q), kvn, kvn, need_weights=return_attn)
        out = q + out
        out = out + self.ffn(self.norm2(out))
        out = out.view(B, T, self.num_tasks, self.d_model).permute(0, 2, 1, 3).contiguous()

        if return_attn:
            return out, attn_w.view(B, T, self.num_tasks, M)
        return out


class StaticTSMF(nn.Module):
    """Single-task TSMF for static images (one task token, no time axis).

    in : modality tokens (B, M, E)
    out: fused task token (B, D)
    """

    def __init__(self, embed=128, d_model=256, num_modalities=2,
                 heads=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)

        self.mod_proj = nn.Linear(embed, d_model)
        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=attn_dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_dropout)

    def forward(self, modality_tokens, return_attn=False):
        B, M, E = modality_tokens.shape
        assert M == self.num_modalities, f"expected {self.num_modalities} modalities, got {M}"

        kv = self.mod_proj(modality_tokens)
        q = self.task_token.unsqueeze(0).expand(B, -1, -1)

        kvn = self.norm_kv(kv)
        out, attn_w = self.attn(self.norm_q(q), kvn, kvn, need_weights=return_attn)
        out = q + out
        out = out + self.ffn(self.norm2(out))

        if return_attn:
            return out[:, 0, :], attn_w
        return out[:, 0, :]


class TaskAgnosticFusion(nn.Module):
    """Baseline for Table 12: self-attention among modality tokens without task tokens.

    in : modality tokens        (B, T, M, E)
    out: one shared token per t (B, T, D)
    """

    def __init__(self, embed=128, d_model=256, num_modalities=5,
                 heads=4, attn_dropout=0.1, ffn_dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)

        self.mod_proj = nn.Linear(embed, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, ffn_dropout)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def forward(self, modality_tokens):
        B, T, M, E = modality_tokens.shape
        assert M == self.num_modalities, f"expected {self.num_modalities} modalities, got {M}"

        x = self.mod_proj(modality_tokens).view(B * T, M, self.d_model)
        xn = self.norm1(x)
        attn_out, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return self.out_proj(x.mean(dim=1)).view(B, T, self.d_model)
