from __future__ import annotations

import math
from typing import Any


def poly_lr_epoch(base_lr: float, epoch: int, max_epoch: int, power: float = 0.9, min_lr: float = 0.0) -> float:
    t = min(max(epoch / float(max_epoch), 0.0), 1.0)
    lr = base_lr * ((1.0 - t) ** power)
    return max(lr, min_lr)


def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)
