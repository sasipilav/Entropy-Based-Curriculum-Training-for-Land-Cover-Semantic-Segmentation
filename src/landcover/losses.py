from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class soft dice loss with ignore_index.

    logits: (B, C, H, W)
    targets: (B, H, W) long
    """
    b, c, h, w = logits.shape
    probs = logits.softmax(1)
    valid = (targets != ignore_index)

    # one-hot targets
    tgt = torch.zeros((b, c, h, w), device=logits.device, dtype=probs.dtype)
    t = targets.clone()
    t[~valid] = 0
    tgt.scatter_(1, t.unsqueeze(1), 1.0)
    tgt = tgt * valid.unsqueeze(1)

    probs = probs * valid.unsqueeze(1)

    inter = (probs * tgt).sum(dim=(0, 2, 3))
    union = (probs + tgt).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, float, float]:
    """Total loss = CE + Dice (weights optional)."""
    ignore_index = int(cfg.get("ignore_index", cfg.get("IGNORE_INDEX", 255)))
    ce_weight = float(cfg.get("ce_weight", 1.0))
    dice_weight = float(cfg.get("dice_weight", 1.0))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))

    ce = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    d = soft_dice_loss(logits, targets, ignore_index=ignore_index)
    loss = ce_weight * ce + dice_weight * d
    return loss, float(d.detach().cpu()), float(ce.detach().cpu())
