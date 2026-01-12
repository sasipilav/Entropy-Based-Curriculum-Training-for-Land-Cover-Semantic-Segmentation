from __future__ import annotations

import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_u32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def resolve_device(device: str) -> str:
    if device.lower() == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


@dataclass
class Checkpoint:
    path: str
    score: float
    epoch: int


def topk_insert(checkpoints: list[Checkpoint], new_item: Checkpoint, k: int = 3) -> list[Checkpoint]:
    checkpoints = checkpoints + [new_item]
    checkpoints = sorted(checkpoints, key=lambda x: x.score, reverse=True)[:k]
    return checkpoints
