"""PRISM-MTL and its adaptations to NTHU-DDD and HDD.

Attribute names follow the experiment code so that the checkpoints used for the
paper load with ``strict=True``.
"""
import torch
import torch.nn as nn

from .encoders import KeypointEncoder, SensorEncoder
from .fusion import TSMF, StaticTSMF, TaskAgnosticFusion
from .hsa import make_hsa_embed

MODALITIES = ("face", "body", "scene", "post", "gest")
AIDE_TASKS = ("emotion", "behavior", "scene", "vehicle")
AIDE_NUM_CLASSES = (5, 7, 3, 5)

# AIDE provides 136 whole-body keypoints per frame (COCO-WholeBody order).
POSTURE_SLICE = slice(0, 26)   # body + feet
GESTURE_SLICE = slice(94, 136)  # both hands


class PRISM_MTL(nn.Module):
    """PRISM-MTL on AIDE.

    Face, body and scene frames are encoded by independent HSA-Nets, posture and
    gesture keypoints by Token-SE encoders. TSMF lets one task token per task
    select modalities at every time step, and each task token is mean-pooled over
    time before its classification head.

    Args:
        modalities: subset of MODALITIES (Table 8). Unused encoders are not built.
        fusion: "tsmf" (default) or "agnostic" (Table 12 baseline).
        heads: number of TSMF attention heads (Table 13).
    """

    def __init__(self, modalities=MODALITIES, fusion="tsmf", embed=128, d_model=256,
                 heads=4, dropout=0.1, pretrained=True):
        super().__init__()
        unknown = set(modalities) - set(MODALITIES)
        if unknown or not modalities:
            raise ValueError(f"modalities must be a non-empty subset of {MODALITIES}")
        self.modalities = [m for m in MODALITIES if m in modalities]
        self.fusion_type = fusion

        self.face = self.fp = self.body = self.bp = self.scene = self.sp = None
        self.post = self.gest = None
        if "face" in self.modalities:
            self.face, self.fp = make_hsa_embed(embed, pretrained)
        if "body" in self.modalities:
            self.body, self.bp = make_hsa_embed(embed, pretrained)
        if "scene" in self.modalities:
            self.scene, self.sp = make_hsa_embed(embed, pretrained)
        if "post" in self.modalities:
            self.post = KeypointEncoder(K=26, embed=embed)
        if "gest" in self.modalities:
            self.gest = KeypointEncoder(K=42, embed=embed)

        M = len(self.modalities)
        if fusion == "tsmf":
            self.fusion = TSMF(embed, d_model, num_tasks=4, num_modalities=M, heads=heads,
                               attn_dropout=dropout, ffn_dropout=dropout)
        elif fusion == "agnostic":
            self.fusion = TaskAgnosticFusion(embed, d_model, num_modalities=M, heads=heads,
                                             attn_dropout=dropout, ffn_dropout=dropout)
        else:
            raise ValueError(f"unknown fusion: {fusion}")

        self.pool_norm = nn.LayerNorm(d_model)
        self.head_emotion = nn.Linear(d_model, AIDE_NUM_CLASSES[0])
        self.head_behavior = nn.Linear(d_model, AIDE_NUM_CLASSES[1])
        self.head_scene = nn.Linear(d_model, AIDE_NUM_CLASSES[2])
        self.head_vehicle = nn.Linear(d_model, AIDE_NUM_CLASSES[3])

    @staticmethod
    def _encode_frames(net, proj, x):
        B, C, T, H, W = x.shape
        f = proj(net(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)))
        return f.view(B, T, -1)

    @staticmethod
    def _encode_context_4view(net, proj, ctx):
        # ctx: (B, 3, 4T, H, W) holding the in-car, front, left and right views.
        B, C, TT, H, W = ctx.shape
        T = TT // 4
        x = ctx.view(B, C, 4, T, H, W).permute(0, 3, 2, 1, 4, 5).contiguous().view(B * T * 4, 3, H, W)
        return proj(net(x)).view(B, T, 4, -1).mean(dim=2)

    def forward(self, kp, ctx, body, face):
        encode = {
            "face": lambda: self._encode_frames(self.face, self.fp, face),
            "body": lambda: self._encode_frames(self.body, self.bp, body),
            "scene": lambda: self._encode_context_4view(self.scene, self.sp, ctx),
            "post": lambda: self.post(kp[:, :, :, POSTURE_SLICE, :]),
            "gest": lambda: self.gest(kp[:, :, :, GESTURE_SLICE, :]),
        }
        tokens = torch.stack([encode[m]() for m in self.modalities], dim=2)  # (B, T, M, E)

        if self.fusion_type == "tsmf":
            z = self.pool_norm(self.fusion(tokens).mean(dim=2))  # (B, 4, D)
            z0, z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        else:
            z0 = z1 = z2 = z3 = self.pool_norm(self.fusion(tokens).mean(dim=1))

        return (self.head_emotion(z0), self.head_behavior(z1),
                self.head_scene(z2), self.head_vehicle(z3))


class PRISM_NTHU(nn.Module):
    """Static single-task adaptation for NTHU-DDD (face crop + full frame, 2 classes)."""

    def __init__(self, embed=128, d_model=256, heads=4, dropout=0.1, num_classes=2, pretrained=True):
        super().__init__()
        self.face_encoder, self.face_proj = make_hsa_embed(embed, pretrained)
        self.orig_encoder, self.orig_proj = make_hsa_embed(embed, pretrained)
        self.fusion = StaticTSMF(embed, d_model, num_modalities=2, heads=heads,
                                 attn_dropout=dropout, ffn_dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, face, original):
        tokens = torch.stack([self.face_proj(self.face_encoder(face)),
                              self.orig_proj(self.orig_encoder(original))], dim=1)  # (B, 2, E)
        return self.head(self.norm(self.fusion(tokens)))


class PRISM_HDD(nn.Module):
    """Two-task adaptation for HDD.

    Modalities: front RGB (HSA-Net) and four CAN sensor groups (Token-SE encoders).
    Tasks: VBR (12-class, single-label) and TCR (6 positive classes, multi-label).
    """

    def __init__(self, num_vbr=12, num_tcr=6, embed=128, d_model=256, heads=4, dropout=0.1,
                 pretrained=True):
        super().__init__()
        self.rgb_encoder, self.rgb_proj = make_hsa_embed(embed, pretrained)
        self.pedal_encoder = SensorEncoder(2, embed, hidden=128, dropout=dropout)
        self.steering_encoder = SensorEncoder(2, embed, hidden=128, dropout=dropout)
        self.motion_encoder = SensorEncoder(2, embed, hidden=128, dropout=dropout)
        self.signal_encoder = SensorEncoder(2, embed, hidden=128, dropout=dropout)
        self.fusion = TSMF(embed, d_model, num_tasks=2, num_modalities=5, heads=heads,
                           attn_dropout=dropout, ffn_dropout=dropout)
        self.pool_norm = nn.LayerNorm(d_model)
        self.head_vbr = nn.Linear(d_model, num_vbr)
        self.head_tcr = nn.Linear(d_model, num_tcr)

    def forward(self, rgb, pedal, steering, motion, signal):
        B, C, T, H, W = rgb.shape
        rgb_seq = self.rgb_proj(self.rgb_encoder(rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)))
        tokens = torch.stack([
            rgb_seq.view(B, T, -1),
            self.pedal_encoder(pedal),
            self.steering_encoder(steering),
            self.motion_encoder(motion),
            self.signal_encoder(signal),
        ], dim=2)
        z = self.pool_norm(self.fusion(tokens).mean(dim=2))
        return self.head_vbr(z[:, 0]), self.head_tcr(z[:, 1])
