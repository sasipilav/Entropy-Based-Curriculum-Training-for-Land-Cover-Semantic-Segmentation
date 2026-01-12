from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import models
from .losses import segmentation_loss
from .metrics import confusion_from_preds, metrics_from_confusion
from .schedulers import poly_lr_epoch, set_lr
from .utils import Checkpoint, ensure_dir, resolve_device, set_seed, topk_insert
from .ebct import curriculum_pct, recalc_entropy

from .datasets import loveda as loveda_ds
from .datasets import deepglobe as deepglobe_ds


def make_loader(ds, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool):
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
    )
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def train_one_epoch(model, loader, optimizer, device: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    loss_sum, dice_sum, ce_sum = 0.0, 0.0, 0.0
    count = 0
    cm_total = torch.zeros(int(cfg["num_classes"]), int(cfg["num_classes"]), dtype=torch.long)

    ignore_index = int(cfg.get("ignore_index", cfg.get("IGNORE_INDEX", 255)))

    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss, d_val, ce_val = segmentation_loss(logits, masks, cfg)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1).detach().cpu()
        cm_total += confusion_from_preds(preds, masks.detach().cpu(), int(cfg["num_classes"]), ignore_index=ignore_index)

        bsz = int(images.size(0))
        loss_sum += float(loss.item()) * bsz
        dice_sum += float(d_val) * bsz
        ce_sum += float(ce_val) * bsz
        count += bsz

    m = metrics_from_confusion(cm_total, ignore_index=ignore_index)
    return {
        "loss": loss_sum / max(1, count),
        "dice": dice_sum / max(1, count),
        "ce": ce_sum / max(1, count),
        "iou": float(m["mean_iou"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "accuracy": float(m["accuracy"]),
    }


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device: str,
    cfg: Dict[str, Any],
    five_crop: bool = False,
) -> Dict[str, float]:
    model.eval()
    loss_sum, dice_sum, ce_sum = 0.0, 0.0, 0.0
    count = 0
    cm_total = torch.zeros(int(cfg["num_classes"]), int(cfg["num_classes"]), dtype=torch.long)
    ignore_index = int(cfg.get("ignore_index", cfg.get("IGNORE_INDEX", 255)))

    # five-crop is only meaningful if loader yields full images (no crop in transform)
    cs = int(cfg.get("crop_size", 0))

    for images, masks, _ in loader:
        if not five_crop:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss, d_val, ce_val = segmentation_loss(logits, masks, cfg)

            preds = torch.argmax(logits, dim=1).detach().cpu()
            cm_total += confusion_from_preds(preds, masks.detach().cpu(), int(cfg["num_classes"]), ignore_index=ignore_index)

            bsz = int(images.size(0))
            loss_sum += float(loss.item()) * bsz
            dice_sum += float(d_val) * bsz
            ce_sum += float(ce_val) * bsz
            count += bsz
            continue

        # five-crop path: keep full image on CPU, crop -> GPU per crop
        bsz, _, h, w = images.shape
        coords = deepglobe_ds.five_crop_coords(h, w, cs)

        for (y0, x0) in coords:
            img_c = images[:, :, y0 : y0 + cs, x0 : x0 + cs].to(device)
            msk_c = masks[:, y0 : y0 + cs, x0 : x0 + cs].to(device)

            logits = model(img_c)
            loss, d_val, ce_val = segmentation_loss(logits, msk_c, cfg)

            preds = torch.argmax(logits, dim=1).detach().cpu()
            cm_total += confusion_from_preds(preds, msk_c.detach().cpu(), int(cfg["num_classes"]), ignore_index=ignore_index)

            loss_sum += float(loss.item()) * int(bsz)
            dice_sum += float(d_val) * int(bsz)
            ce_sum += float(ce_val) * int(bsz)
            count += int(bsz)

    m = metrics_from_confusion(cm_total, ignore_index=ignore_index)
    return {
        "loss": loss_sum / max(1, count),
        "dice": dice_sum / max(1, count),
        "ce": ce_sum / max(1, count),
        "iou": float(m["mean_iou"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        "accuracy": float(m["accuracy"]),
    }


def _dataset_bundle(cfg: Dict[str, Any]):
    name = str(cfg["dataset"]).lower()
    if name == "loveda":
        train_df, valid_df, test_df = loveda_ds.load_metadata(cfg["data_dir"])
        return name, (train_df, valid_df, test_df)
    if name == "deepglobe":
        train_df, valid_df, test_df = deepglobe_ds.load_metadata(cfg["data_dir"], valid_frac=float(cfg.get("valid_frac", 0.1)), seed=int(cfg.get("seed", 42)))
        return name, (train_df, valid_df, test_df)
    raise ValueError(f"Unknown dataset: {cfg['dataset']}")


def run_experiment(cfg: Dict[str, Any], tag: str = "run") -> Tuple[torch.nn.Module, list[dict]]:
    """Train and return (best_model, logs)."""
    cfg = dict(cfg)  # shallow copy
    cfg["device"] = resolve_device(str(cfg.get("device", "cuda")))
    set_seed(int(cfg.get("seed", 0)))

    dataset_name, (train_df, valid_df, test_df) = _dataset_bundle(cfg)

    model = models.build_model(cfg).to(cfg["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    # dataset-specific transforms + dataset classes
    if dataset_name == "loveda":
        train_tf = loveda_ds.get_train_transform(int(cfg["crop_size"]))
        eval_tf = loveda_ds.get_eval_transform(int(cfg.get("eval_crop_size", cfg["crop_size"])), center=True, do_crop=True)
        entropy_tf = loveda_ds.get_eval_transform(int(cfg["crop_size"]), center=True, do_crop=False)

        mk_train_ds = lambda df, epoch: loveda_ds.LoveDADataset(
            df, train_tf, ignore_index=int(cfg.get("ignore_index", 255)), num_crops_per_image=int(cfg.get("crops_per_image", 1))
        )
        mk_val_ds = lambda df: loveda_ds.LoveDADataset(
            df, eval_tf, ignore_index=int(cfg.get("ignore_index", 255)), num_crops_per_image=1
        )
        five_crop = False

    elif dataset_name == "deepglobe":
        train_tf = deepglobe_ds.get_train_transform()
        eval_tf = deepglobe_ds.get_eval_transform(int(cfg["crop_size"]), center=True, do_crop=True)
        entropy_tf = deepglobe_ds.get_eval_transform(int(cfg["crop_size"]), center=True, do_crop=False)

        bs_cfg = cfg.get("boundary_sampling", {})
        mk_train_ds = lambda df, epoch: deepglobe_ds.DeepGlobeDataset(
            df,
            train_tf,
            crop_size=int(cfg["crop_size"]),
            num_crops_per_image=int(cfg.get("crops_per_image", 1)),
            manual_crop=True,
            boundary_sampling=bs_cfg,
            epoch=int(epoch),
        )
        mk_val_ds = lambda df: deepglobe_ds.DeepGlobeDataset(
            df,
            eval_tf,
            crop_size=int(cfg["crop_size"]),
            num_crops_per_image=1,
            manual_crop=False,
            epoch=0,
        )
        five_crop = bool(cfg.get("five_crop_eval", False))

        # keep original default ignore index behavior
        cfg.setdefault("ignore_index", deepglobe_ds.IGNORE_INDEX)

    else:
        raise AssertionError("unreachable")

    valid_loader = make_loader(
        mk_val_ds(valid_df),
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=False,
        drop_last=False,
    )

    # EBCT state
    train_df_entropy = None
    logs: list[dict] = []
    topk: list[Checkpoint] = []

    save_dir = ensure_dir(cfg.get("save_dir", "checkpoints"))
    num_epochs = int(cfg["num_epochs"])
    poly_power = float(cfg.get("poly_power", 0.9))
    min_lr = float(cfg.get("min_lr", 0.0))

    eb = cfg.get("ebct", {})
    eb_use = bool(eb.get("use", False))
    eb_warmup = int(eb.get("warmup_epochs", 0))
    eb_recalc = int(eb.get("recalc_interval", 1))
    eb_num_crops = int(eb.get("entropy_num_crops", eb.get("num_crops", 1)))

    for epoch in range(num_epochs):
        t0 = time.time()

        lr_epoch = poly_lr_epoch(float(cfg["lr"]), epoch, num_epochs, power=poly_power, min_lr=min_lr)
        set_lr(optimizer, lr_epoch)

        # ---- EBCT subset selection ----
        p_used = 1.0
        if eb_use and epoch >= eb_warmup:
            if (train_df_entropy is None) or (epoch % eb_recalc == 0):
                train_df_entropy = recalc_entropy(
                    model=model,
                    df=train_df,
                    transform_entropy=entropy_tf,
                    device=cfg["device"],
                    cfg=cfg,
                    epoch=epoch,
                    num_crops=eb_num_crops,
                )
            p_used = float(curriculum_pct(epoch, cfg))
            subset_size = max(1, int(len(train_df_entropy) * p_used))
            train_subset_df = train_df_entropy.iloc[:subset_size].reset_index(drop=True)
        else:
            train_subset_df = train_df

        train_ds = mk_train_ds(train_subset_df, epoch)
        train_loader = make_loader(
            train_ds,
            batch_size=int(cfg.get("batch_size", 8)),
            num_workers=int(cfg.get("num_workers", 4)),
            shuffle=True,
            drop_last=True,
        )

        train_stats = train_one_epoch(model, train_loader, optimizer, cfg["device"], cfg)
        val_stats = eval_one_epoch(model, valid_loader, cfg["device"], cfg, five_crop=five_crop)

        row = {
            "epoch": epoch,
            "lr": lr_epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "p_used": p_used,
            "t_epoch_sec": time.time() - t0,
        }
        logs.append(row)

        # save top-3 by val_iou
        score = float(val_stats["iou"])
        ckpt_path = f"{save_dir}/{dataset_name}_{tag}_epoch{epoch:03d}_miou{score:.4f}.pth"
        # write checkpoint when it makes top-k
        topk = topk_insert(topk, Checkpoint(path=ckpt_path, score=score, epoch=epoch), k=3)
        if any(c.path == ckpt_path for c in topk):
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"[{dataset_name}/{tag}] epoch={epoch:03d} p={p_used:.2f} lr={lr_epoch:.2e} "
            f"train_iou={train_stats['iou']:.4f} val_iou={val_stats['iou']:.4f} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f}"
        )

    # load best checkpoint back
    best = sorted(topk, key=lambda x: x.score, reverse=True)[0]
    state = torch.load(best.path, map_location=cfg["device"])
    model.load_state_dict(state)
    model.eval()
    return model, logs
