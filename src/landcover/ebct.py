from __future__ import annotations

import random
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch

from .utils import stable_u32


def curriculum_pct(epoch: int, cfg: Dict[str, Any]) -> float:
    eb = cfg.get("ebct", {})
    warm = int(eb.get("warmup_epochs", 0))
    end_epoch = int(eb.get("end_epoch", 0))
    start_pct = float(eb.get("start_pct", 1.0))
    end_pct = float(eb.get("end_pct", 1.0))

    if epoch < warm:
        return 1.0
    if epoch >= end_epoch:
        return end_pct
    if end_epoch <= warm:
        return end_pct

    t = (epoch - warm) / float(end_epoch - warm)
    return start_pct + (end_pct - start_pct) * t


@torch.no_grad()
def recalc_entropy(
    model,
    df: pd.DataFrame,
    transform_entropy,
    device: str,
    cfg: Dict[str, Any],
    epoch: int,
    num_crops: int = 1,
) -> pd.DataFrame:
    """Compute per-image entropy scores and return df sorted by entropy (ascending)."""
    model.eval()
    entropies = []
    cs = int(cfg["crop_size"])
    base_seed = int(cfg.get("seed", 0))

    for row in df.itertuples():
        image = cv2.imread(row.sat_image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        img_id = str(getattr(row, "image_id", getattr(row, "Index", "0")))
        sid = stable_u32(img_id)

        hs = []
        for k in range(int(num_crops)):
            r = random.Random(base_seed + 1000003 * epoch + 10007 * k + sid)
            y0 = r.randint(0, h_img - cs)
            x0 = r.randint(0, w_img - cs)

            patch = image[y0 : y0 + cs, x0 : x0 + cs]
            dummy = np.zeros((cs, cs), dtype=np.uint8)

            aug = transform_entropy(image=patch, mask=dummy)
            img_t = aug["image"].unsqueeze(0).to(device)

            logits = model(img_t)
            prob = logits.softmax(1)
            hmap = -(prob * prob.clamp_min(1e-8).log()).sum(1)  # 1 x cs x cs
            hs.append(float(hmap.mean().item()))

        entropies.append(float(np.mean(hs)))

    out = df.copy()
    out["entropy"] = entropies
    return out.sort_values("entropy").reset_index(drop=True)
