#!/usr/bin/env python
"""Learnable variable-count point detector with a DINOv2 feature backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DecodedPoint:
    x: float
    y: float
    confidence: float


class AdaptivePointDetector(nn.Module):
    """Dense objectness heatmap; point count follows thresholded local maxima.

    The network has no slot count and no fixed semantic point identities.  A
    finite ``max_points`` is used only as an inference safety guard.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int = 14,
        decoder_dim: int = 192,
        output_stride: int = 4,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.output_stride = output_stride
        embed_dim = int(getattr(backbone, "embed_dim", 384))
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, 3, padding=1),
            nn.GroupNorm(8, decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.GroupNorm(8, decoder_dim // 2),
            nn.GELU(),
            nn.Conv2d(decoder_dim // 2, 1, 1),
        )
        self.configure_backbone(freeze_backbone, unfreeze_last_blocks)

    def configure_backbone(self, freeze: bool, unfreeze_last_blocks: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze
        if freeze and unfreeze_last_blocks > 0 and hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            for block in blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features: Any = self.backbone.forward_features(images)
        if isinstance(features, dict):
            tokens = features.get("x_norm_patchtokens")
            if tokens is None:
                raise KeyError("DINOv2 forward_features did not return x_norm_patchtokens")
        else:
            tokens = features
        batch, token_count, channels = tokens.shape
        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        if grid_h * grid_w != token_count:
            raise ValueError(f"Patch-token count {token_count} does not match grid {grid_h}x{grid_w}")
        feature_map = tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
        logits = self.decoder(feature_map)
        output_h = max(1, images.shape[-2] // self.output_stride)
        output_w = max(1, images.shape[-1] // self.output_stride)
        return F.interpolate(logits, size=(output_h, output_w), mode="bilinear", align_corners=False)

    @staticmethod
    def decode(
        logits: torch.Tensor,
        input_hw: tuple[int, int],
        threshold: float = 0.35,
        nms_kernel: int = 5,
        max_points: int = 64,
        fixed_k: int = 0,
    ) -> list[list[DecodedPoint]]:
        probability = logits.sigmoid()
        pooled = F.max_pool2d(probability, nms_kernel, stride=1, padding=nms_kernel // 2)
        maxima = probability.eq(pooled)
        results: list[list[DecodedPoint]] = []
        output_h, output_w = logits.shape[-2:]
        input_h, input_w = input_hw
        for batch_index in range(logits.shape[0]):
            scores = probability[batch_index, 0]
            valid = maxima[batch_index, 0]
            if fixed_k > 0:
                candidate_scores = scores.masked_fill(~valid, -1.0).flatten()
                count = min(fixed_k, int(valid.sum()))
                values, indices = torch.topk(candidate_scores, k=count)
                keep = values >= 0.0
                values, indices = values[keep], indices[keep]
            else:
                flat_mask = (valid & (scores >= threshold)).flatten()
                indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
                values = scores.flatten()[indices]
                if len(indices) > max_points:
                    values, order = torch.topk(values, k=max_points)
                    indices = indices[order]
            order = torch.argsort(values, descending=True)
            points: list[DecodedPoint] = []
            for index in order:
                flat_index = int(indices[index])
                y, x = divmod(flat_index, output_w)
                source_x = x * (input_w - 1) / max(output_w - 1, 1)
                source_y = y * (input_h - 1) / max(output_h - 1, 1)
                points.append(DecodedPoint(float(source_x), float(source_y), float(values[index])))
            results.append(points)
        return results
