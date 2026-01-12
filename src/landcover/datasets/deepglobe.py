from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset


CLASS_NAMES = [
    "urban_land",
    "agriculture_land",
    "rangeland",
    "forest_land",
    "water",
    "barren_land",
    "unknown",
]

# DeepGlobe class-color map (RGB) -> index
COLOR_MAP: Dict[Tuple[int, int, int], int] = {
    (0, 255, 255): 0,      # urban_land
    (255, 255, 0): 1,      # agriculture_land
    (255, 0, 255): 2,      # rangeland
    (0, 255, 0): 3,        # forest_land
    (0, 0, 255): 4,        # water
    (255, 255, 255): 5,    # barren_land
    (0, 0, 0): 6,          # unknown
}

IGNORE_INDEX = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_metadata(data_dir: str, valid_frac: float = 0.1, seed: int = 42):
    """Load DeepGlobe metadata.csv from a local folder.

    Expect:
      data_dir/metadata.csv
      and paths inside metadata.csv are relative to data_dir.
    """
    meta_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    metadata_df = pd.read_csv(meta_path)
    metadata_df = metadata_df[["image_id", "sat_image_path", "mask_path", "split"]].copy()
    metadata_df["sat_image_path"] = metadata_df["sat_image_path"].apply(lambda p: os.path.join(data_dir, p))
    metadata_df["mask_path"] = metadata_df["mask_path"].apply(
        lambda p: os.path.join(data_dir, p) if isinstance(p, str) else p
    )

    labeled_df = metadata_df[metadata_df["mask_path"].apply(lambda p: isinstance(p, str))].copy()
    labeled_df = labeled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    valid_df = labeled_df.sample(frac=valid_frac, random_state=seed)
    train_df = labeled_df.drop(valid_df.index).reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    test_df = metadata_df[metadata_df["split"] == "test"].copy().reset_index(drop=True)
    return train_df, valid_df, test_df


def rgb_to_index_mask(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    mask = np.zeros((h, w), dtype=np.int64)
    for color, idx in COLOR_MAP.items():
        matches = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        mask[matches] = idx
    return mask


def index_to_color(mask_idx: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask_idx.shape, 3), dtype=np.uint8)
    for color, idx in COLOR_MAP.items():
        canvas[mask_idx == idx] = np.array(color, dtype=np.uint8)
    return canvas


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_train_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transform(crop_size: int, center: bool = True, do_crop: bool = True) -> A.Compose:
    ops = []
    if do_crop:
        ops.append(A.CenterCrop(crop_size, crop_size) if center else A.CropNonEmptyMaskIfExists(crop_size, crop_size))
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


# ---- boundary-guided crop helpers (moved out of Dataset) ----
def _rand_crop_xy(h: int, w: int, cs: int) -> Tuple[int, int]:
    return random.randint(0, h - cs), random.randint(0, w - cs)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _boundary_points(mask: np.ndarray, focus_class: int, neighbor_classes: Iterable[int], kernel_size: int = 3):
    focus = (mask == focus_class).astype(np.uint8)
    neigh = np.isin(mask, list(neighbor_classes)).astype(np.uint8)

    k = np.ones((kernel_size, kernel_size), np.uint8)
    dil_focus = cv2.dilate(focus, k, iterations=1)
    boundary = (dil_focus & neigh).astype(bool)

    ys, xs = np.where(boundary)
    if len(ys) == 0:
        return None
    return np.stack([ys, xs], axis=1)


def _guided_crop_xy(
    mask: np.ndarray,
    cs: int,
    focus_class: int,
    neighbor_classes: Iterable[int],
    kernel_size: int = 3,
    max_tries: int = 15,
):
    h, w = mask.shape
    pts = _boundary_points(mask, focus_class, neighbor_classes, kernel_size=kernel_size)
    if pts is None:
        return None

    for _ in range(max_tries):
        y, x = pts[random.randint(0, len(pts) - 1)]
        y0 = _clamp(int(y) - cs // 2, 0, h - cs)
        x0 = _clamp(int(x) - cs // 2, 0, w - cs)
        return y0, x0
    return None


def five_crop_coords(h: int, w: int, cs: int):
    y_max = h - cs
    x_max = w - cs
    yc = y_max // 2
    xc = x_max // 2
    return [(0, 0), (0, x_max), (y_max, 0), (y_max, x_max), (yc, xc)]


class DeepGlobeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        crop_size: int,
        num_crops_per_image: int = 1,
        manual_crop: bool = False,
        boundary_sampling: Optional[dict] = None,
        epoch: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cs = int(crop_size)
        self.num_crops = int(num_crops_per_image)
        self.manual_crop = bool(manual_crop)
        self.boundary_sampling = boundary_sampling or {}
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.df) * self.num_crops

    def __getitem__(self, idx: int):
        img_idx = idx // self.num_crops
        row = self.df.iloc[img_idx]
        guided_flag = 0

        image = cv2.imread(row.sat_image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(row.mask_path, cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_to_index_mask(mask_rgb)

        if self.manual_crop:
            h, w = mask.shape
            if h >= self.cs and w >= self.cs:
                bs = self.boundary_sampling
                yx = None

                use_bs = bool(bs.get("use", False))
                if use_bs and self.epoch >= int(bs.get("warmup_epochs", 0)):
                    p = float(bs.get("p_max", 0.0))
                    if random.random() < p:
                        yx = _guided_crop_xy(
                            mask,
                            self.cs,
                            focus_class=int(bs.get("focus_class", 2)),
                            neighbor_classes=bs.get("neighbor_classes", (1,)),
                            kernel_size=int(bs.get("kernel_size", 3)),
                            max_tries=int(bs.get("max_tries", 15)),
                        )

                if yx is None:
                    y0, x0 = _rand_crop_xy(h, w, self.cs)
                else:
                    guided_flag = 1
                    y0, x0 = yx

                image = image[y0 : y0 + self.cs, x0 : x0 + self.cs]
                mask = mask[y0 : y0 + self.cs, x0 : x0 + self.cs]

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].long()
        return image_t, mask_t, torch.tensor(guided_flag, dtype=torch.long)
