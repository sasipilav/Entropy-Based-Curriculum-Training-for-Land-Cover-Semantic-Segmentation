from __future__ import annotations

from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = str(cfg.get("arch", cfg.get("model_name", "deeplabv3plus"))).lower()
    backbone = str(cfg.get("backbone", "resnet152"))
    in_channels = int(cfg.get("in_channels", 3))
    num_classes = int(cfg["num_classes"])
    encoder_weights = cfg.get("encoder_weights", "imagenet")

    if arch in {"deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    raise ValueError(f"Unknown arch/model_name: {arch}")
