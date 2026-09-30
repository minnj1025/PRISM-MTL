"""Hierarchical Stage-wise Attention network (HSA-Net).

A torchvision ResNet-18 is kept intact (ImageNet weights load without key changes),
and a different attention operator is applied after each residual stage:

    Stage 1: identity
    Stage 2: directional spatial attention (vertical k x 1 + horizontal 1 x k)
    Stage 3: spatial attention (k x k)
    Stage 4: channel attention (avg + max pooled MLP)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class IdentityAttn(nn.Module):
    def forward(self, x):
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class DirectionalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        k = kernel_size
        self.conv_v = nn.Conv2d(2, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=False)
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(1, k), padding=(0, k // 2), bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv_v(cat) + self.conv_h(cat))
        return x * attn


def make_stage_attn(stage_id, channels, sa_kernel=7, dsa_kernel=7, ca_reduction=16):
    if stage_id == 2:
        return DirectionalSpatialAttention(kernel_size=dsa_kernel)
    if stage_id == 3:
        return SpatialAttention(kernel_size=sa_kernel)
    if stage_id == 4:
        return ChannelAttention(channels, reduction=ca_reduction)
    return IdentityAttn()


class HSANet(nn.Module):
    """ResNet-18 with stage-wise attention. Returns a 512-d pooled feature."""

    def __init__(self, pretrained=True, sa_kernel=7, dsa_kernel=7, ca_reduction=16):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.attn1 = make_stage_attn(1, 64, sa_kernel, dsa_kernel, ca_reduction)
        self.attn2 = make_stage_attn(2, 128, sa_kernel, dsa_kernel, ca_reduction)
        self.attn3 = make_stage_attn(3, 256, sa_kernel, dsa_kernel, ca_reduction)
        self.attn4 = make_stage_attn(4, 512, sa_kernel, dsa_kernel, ca_reduction)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = self.attn1(b.layer1(x))
        x = self.attn2(b.layer2(x))
        x = self.attn3(b.layer3(x))
        x = self.attn4(b.layer4(x))
        return torch.flatten(b.avgpool(x), 1)


def make_hsa_embed(out_dim=128, pretrained=True):
    """HSA-Net encoder and its linear projection to the modality embedding size."""
    return HSANet(pretrained=pretrained), nn.Linear(512, out_dim)
