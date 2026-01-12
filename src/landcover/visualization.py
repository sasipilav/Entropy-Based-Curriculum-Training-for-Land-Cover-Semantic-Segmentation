from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """C,H,W tensor -> H,W,3 float in [0,1]"""
    img = img_t.detach().cpu().float().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = img * std + mean
    return np.clip(img, 0, 1)


def visualize_pairs(
    pairs: Sequence[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
    title: str,
    index_to_color_fn,
    max_rows: int = 12,
) -> None:
    pairs = list(pairs)[:max_rows]
    n = len(pairs)
    cols = 1 + int(any(p[1] is not None for p in pairs)) + int(any(p[2] is not None for p in pairs))
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    if axes.ndim == 1:
        axes = axes.reshape(n, cols)

    for row_i, (img, mask_idx, pred_idx) in enumerate(pairs):
        col = 0
        axes[row_i, col].imshow(img)
        axes[row_i, col].set_title("Image")
        axes[row_i, col].axis("off")
        col += 1
        if mask_idx is not None:
            axes[row_i, col].imshow(index_to_color_fn(mask_idx))
            axes[row_i, col].set_title("Mask")
            axes[row_i, col].axis("off")
            col += 1
        if pred_idx is not None:
            axes[row_i, col].imshow(index_to_color_fn(pred_idx))
            axes[row_i, col].set_title("Prediction")
            axes[row_i, col].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: Sequence[str],
    normalize: str | None = "true",
    title: str = "Confusion",
    ax=None,
):
    cm = cm.clone().float()
    if normalize == "true":
        cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1e-9)
    elif normalize == "pred":
        cm = cm / cm.sum(dim=0, keepdim=True).clamp_min(1e-9)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    cm_np = cm.cpu().numpy()
    im = ax.imshow(cm_np, vmin=0, vmax=1 if normalize else None)
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fmt = ".3f" if normalize else "d"
    thresh = (cm_np.max() + cm_np.min()) / 2.0
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            val = cm_np[i, j]
            text = f"{val:{fmt}}" if fmt != "d" else f"{int(round(val))}"
            color = "black" if val > thresh else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    return ax


def plot_curves(logs: list[dict], keys: Sequence[str], title: str, xlabel: str = "epoch"):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r.get("epoch", i) for i, r in enumerate(logs)]
    for k in keys:
        ys = [r.get(k, np.nan) for r in logs]
        ax.plot(xs, ys, label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    plt.tight_layout()
    plt.show()
