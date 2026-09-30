"""Non-visual modality encoders: frame-wise MLP followed by Token-SE."""
import torch
import torch.nn as nn


class TokenSE(nn.Module):
    """Squeeze-and-excitation over a token sequence.

    x: (B, T, E). The sequence is squeezed over T with average and max pooling,
    and the resulting channel weights rescale every token.
    """

    def __init__(self, embed_dim, reduction=16):
        super().__init__()
        hidden = max(4, embed_dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim, bias=False),
        )

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean(dim=1)) + self.mlp(x.amax(dim=1)))
        return x * w.unsqueeze(1)


class KeypointEncoder(nn.Module):
    """Joint keypoints (B, 3, T, K, 1) -> joint modality tokens (B, T, E)."""

    def __init__(self, K, embed=128, hidden=256, se_reduction=16):
        super().__init__()
        self.in_dim = 3 * K
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, embed),
            nn.GELU(),
        )
        self.se = TokenSE(embed_dim=embed, reduction=se_reduction)

    def forward(self, kp):
        kp = kp.squeeze(-1).permute(0, 2, 1, 3).contiguous()  # (B, T, 3, K)
        B, T, _, K = kp.shape
        x = self.mlp(kp.reshape(B * T, 3 * K)).view(B, T, -1)
        return self.se(x)


class SensorEncoder(nn.Module):
    """Vehicle sensor group (B, T, D) -> sensor modality tokens (B, T, E)."""

    def __init__(self, in_dim, embed=128, hidden=128, dropout=0.1, se_reduction=16):
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
            nn.GELU(),
        )
        self.se = TokenSE(embed_dim=embed, reduction=se_reduction)

    def forward(self, x):
        B, T, D = x.shape
        x = self.mlp(x.reshape(B * T, D)).view(B, T, -1)
        return self.se(x)
