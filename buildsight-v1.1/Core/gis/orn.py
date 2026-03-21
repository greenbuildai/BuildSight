from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ORNConfig:
    input_channels: int = 5
    input_size: Tuple[int, int] = (128, 224)  # (h, w)
    feature_dim: int = 128
    gru_hidden: int = 128
    num_body_parts: int = 4  # head, torso, legs, hands
    num_ppe_items: int = 2  # helmet, vest
    num_ppe_classes: int = 3  # present, absent, uncertain
    occlusion_classes: int = 2  # visible, occluded


class OcclusionAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.gate = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, occ_logits: torch.Tensor) -> torch.Tensor:
        # Use occlusion logits to gate spatial features.
        occ_prob = torch.softmax(occ_logits, dim=1)[:, 1:2]
        attn = torch.sigmoid(self.gate(self.conv(feats)))
        return feats * attn * (1.0 - occ_prob)


class ORN(nn.Module):
    """
    Occlusion Reasoning Network (ORN).
    Inputs: person crop (RGB) + PPE heatmap channels (helmet, vest).
    Outputs:
      - occlusion_logits: per-pixel visible/occluded
      - visibility_logits: body-part visibility scores
      - ppe_logits: per-PPE item 3-class logits
      - next_hidden: GRU hidden state for temporal modeling
    """

    def __init__(self, config: ORNConfig):
        super().__init__()
        self.config = config

        # Keep encoder shallow for Jetson-class latency; export via ONNX or TensorRT.
        self.encoder = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.feature_dim),
            nn.ReLU(inplace=True),
        )

        self.occlusion_head = nn.Sequential(
            nn.Conv2d(config.feature_dim, config.feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.feature_dim // 2, config.occlusion_classes, kernel_size=1),
        )

        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.feature_dim, config.num_body_parts),
        )

        self.attn = OcclusionAttention(config.feature_dim)

        self.temporal_gru = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.ppe_head = nn.Linear(config.gru_hidden, config.num_ppe_items * config.num_ppe_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        occ_logits = self.occlusion_head(feats)
        feats_attn = self.attn(feats, occ_logits)

        visibility_logits = self.visibility_head(feats_attn)

        pooled = F.adaptive_avg_pool2d(feats_attn, 1).flatten(1)
        gru_in = pooled.unsqueeze(1)
        gru_out, next_hidden = self.temporal_gru(gru_in, hidden)
        temporal_feat = gru_out[:, -1, :]

        ppe_logits = self.ppe_head(temporal_feat)
        ppe_logits = ppe_logits.view(-1, self.config.num_ppe_items, self.config.num_ppe_classes)

        return ppe_logits, visibility_logits, occ_logits, next_hidden


def decode_ppe_logits(ppe_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to class indices: 0=present, 1=absent, 2=uncertain.
    """
    return torch.argmax(ppe_logits, dim=-1)


def ppe_class_names() -> Tuple[str, str, str]:
    return ("PPE_PRESENT", "PPE_ABSENT", "PPE_UNCERTAIN")
