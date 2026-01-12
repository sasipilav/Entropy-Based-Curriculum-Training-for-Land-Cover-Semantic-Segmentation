from __future__ import annotations

import os
import glob
import zipfile
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]

# Visualization palette used in the original notebook.
LOVEDA_COLORS = [
    (0, 0, 0),       # background
    (255, 0, 0),     # building
    (255, 255, 0),   # road
    (0, 0, 255),     # water
    (139, 69, 19),   # barren
    (0, 255, 0),     # forest
    (255, 0, 255),   # agriculture
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def extract_loveda_zips(zip_dir: str, extract_dir: str) -> None:
    """Optional helper if your LoveDA dataset is stored as multiple .zip files."""
    os.makedirs(extract_dir, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))
    if not zips:
        raise FileNotFoundError(f"No .zip files found in: {zip_dir}")
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(extract_dir)


def load_metadata(root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test dataframes.

    Expected structure:
      root/Train/{Urban,Rural}/images_png/*.png
      root/Train/{Urban,Rural}/masks_png/*.png
      root/Val/{Urban,Rural}/...
    """

    def _gather(split: str) -> pd.DataFrame:
        rows = []
        split_dir = os.path.join(root, split)
        for scene in ["Urban", "Rural"]:
            img_dir = os.path.join(split_dir, scene, "images_png")
            msk_dir = os.path.join(split_dir, scene, "masks_png")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                if not fn.lower().endswith(".png"):
                    continue
                img_path = os.path.join(img_dir, fn)
                msk_path = os.path.join(msk_dir, fn)
                rows.append(
                    dict(
                        image_id=f"{split}_{scene}_{os.path.splitext(fn)[0]}",
                        sat_image_path=img_path,
                        mask_path=msk_path if os.path.exists(msk_path) else None,
                        split=split.lower(),
                        scene=scene.lower(),
                    )
                )
        return pd.DataFrame(rows)

    train_df = _gather("Train")
    valid_df = _gather("Val")
    test_df = _gather("Test") if os.path.isdir(os.path.join(root, "Test")) else pd.DataFrame(columns=train_df.columns)

    train_df = train_df[train_df["mask_path"].notna()].reset_index(drop=True)
    valid_df = valid_df[valid_df["mask_path"].notna()].reset_index(drop=True)
    return train_df, valid_df, test_df


def mask_to_train_ids(mask_u8: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """LoveDA masks: 0=no-data(ignore), 1..7=classes. Map 1..7 -> 0..6, 0 -> ignore_index."""
    m = mask_u8.astype(np.int64)
    out = np.full_like(m, fill_value=ignore_index, dtype=np.int64)
    valid = (m >= 1) & (m <= 7)
    out[valid] = m[valid] - 1
    return out


def get_train_transform(crop_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transform(crop_size: int, center: bool = True, do_crop: bool = True) -> A.Compose:
    tf = []
    if do_crop:
        tf.append(A.CenterCrop(crop_size, crop_size) if center else A.RandomCrop(crop_size, crop_size))
    tf += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(tf)


def get_eval_transform_full() -> A.Compose:
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class LoveDADataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose, ignore_index: int, num_crops_per_image: int = 1):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.ignore_index = ignore_index
        self.num_crops_per_image = int(num_crops_per_image)

    def __len__(self) -> int:
        return len(self.df) * self.num_crops_per_image

    def __getitem__(self, idx: int):
        row_idx = idx // self.num_crops_per_image
        row = self.df.iloc[row_idx]

        image = cv2.imread(row.sat_image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_u8 = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask_to_train_ids(mask_u8, ignore_index=self.ignore_index).astype(np.uint8)

        aug = self.transform(image=image, mask=mask)
        img_t = aug["image"]
        mask_t = aug["mask"].long()

        # third return for API compatibility with your original code
        return img_t, mask_t, torch.tensor(0, dtype=torch.long)
