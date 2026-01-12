from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def confusion_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute confusion matrix for (B,H,W) preds/targets tensors on CPU."""
    if preds.ndim == 2:
        preds = preds.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)

    preds = preds.reshape(-1).to(torch.long)
    targets = targets.reshape(-1).to(torch.long)

    mask = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        mask = mask & (targets != int(ignore_index))

    preds = preds[mask]
    targets = targets[mask]

    idx = targets * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def metrics_from_confusion(cm: torch.Tensor, ignore_index: Optional[int] = None) -> Dict[str, object]:
    cm = cm.float()
    num_classes = cm.shape[0]
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    pred_sum = cm.sum(dim=0)
    fp = pred_sum - tp
    fn = support - tp

    precision_c = tp / (tp + fp).clamp_min(1e-9)
    recall_c = tp / (tp + fn).clamp_min(1e-9)
    f1_c = (2 * precision_c * recall_c / (precision_c + recall_c).clamp_min(1e-9))

    valid = support > 0
    if ignore_index is not None and 0 <= int(ignore_index) < num_classes:
        valid[int(ignore_index)] = False

    iou_c = tp / (tp + fp + fn).clamp_min(1e-9)

    macro_precision = precision_c[valid].mean().item() if valid.any() else float("nan")
    macro_recall = recall_c[valid].mean().item() if valid.any() else float("nan")
    macro_f1 = f1_c[valid].mean().item() if valid.any() else float("nan")
    mean_iou = iou_c[valid].mean().item() if valid.any() else float("nan")

    acc = (tp.sum() / cm.sum().clamp_min(1e-9)).item()

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "accuracy": acc,
        "mean_iou": mean_iou,
        "iou_per_class": iou_c.detach().cpu().numpy(),
    }
