"""
Model Training Script for Multimodal Flood Segmentation
========================================================
Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
Features:
- PyTorch AMP (Automatic Mixed Precision) for memory efficiency & 2x speedup
- Strict 2-channel SAR input (VV, VH) directly aligned with Sen1Floods11
- Robust handling of Sen1Floods11 no-data pixels (ignore_index = -1)
- Mathematically rigorous IoU computation (absent classes are NaN, never awarded 1.0)
- Dual-dataset audit: verifies BOTH training AND validation splits are genuine real data
- Real-time VRAM tracking & best checkpoint saving with data provenance metadata
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import MultimodalFloodUNet
from dataset import get_dataloaders


class DiceLoss(nn.Module):
    """
    Multiclass / Binary Dice Loss with ignore_index masking.
    Excludes invalid (-1 / unlabelled / cloudy) pixels from both intersection and union.
    """
    def __init__(self, smooth=1e-5, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # Mask out ignore_index and out-of-bounds pixels
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < num_classes)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        safe_targets = torch.clamp(targets, 0, num_classes - 1)
        targets_one_hot = torch.zeros_like(probs).scatter_(1, safe_targets.unsqueeze(1), 1)

        valid_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs_valid = probs * valid_expanded
        targets_valid = targets_one_hot * valid_expanded

        dice_per_class = []
        for c in range(num_classes):
            p = probs_valid[:, c].contiguous().view(-1)
            t = targets_valid[:, c].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class.append(dice)

        mean_dice = torch.stack(dice_per_class).mean()
        return 1.0 - mean_dice


def compute_iou(preds, targets, num_classes=2, ignore_index=-1):
    """
    Computes per-class and mean Intersection-over-Union,
    strictly ignoring unlabelled (-1) pixels.

    Evaluation Policy for Absent Classes:
    When a class has union == 0 (meaning it appears neither in the ground-truth
    labels nor in the model predictions), its IoU is mathematically undefined (NaN).
    It is NOT awarded a 1.0 score (which would falsely inflate mIoU).
    Instead, it is recorded as np.nan and excluded from the mean IoU computation.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if not valid.any():
        return [np.nan] * num_classes, 0.0

    valid_preds = preds[valid]
    valid_targets = targets[valid]

    ious = []
    valid_ious_for_mean = []
    for c in range(num_classes):
        pred_c = (valid_preds == c)
        targ_c = (valid_targets == c)
        intersection = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        if union == 0:
            # Class is absent from both predictions and targets
            ious.append(np.nan)
        else:
            c_iou = intersection / union
            ious.append(c_iou)
            valid_ious_for_mean.append(c_iou)

    miou = float(np.mean(valid_ious_for_mean)) if valid_ious_for_mean else 0.0
    return ious, miou


def train(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("==================================================")
        print(f"[GPU DETECTED] {gpu_name} ({total_vram:.2f} GB VRAM)")
        print("[OPT] PyTorch AMP Mixed Precision Active")
        print("==================================================")
    else:
        device = torch.device("cpu")
        print("[WARNING] Running on CPU. GPU recommended for satellite training.")

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_flood_model.pth")

    # Dataloaders
    print(f"\n[1/4] Loading dataset (Data directory: {args.data_dir or 'Sen1Floods11 / Procedural Fallback'})...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir if args.data_dir else None,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        sample_count=args.samples,
        auto_download=False,
    )

    # ── Strict Dual Dataset Source Audit ──────────────────────────────────────
    train_src = train_loader.dataset.data_source
    val_src = val_loader.dataset.data_source
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    real_sources = ("sen1floods11", "kuro_siwo", "user_supplied")
    train_is_real = (train_src in real_sources) and (n_train > 0)
    val_is_real = (val_src in real_sources) and (n_val > 0)
    both_real = train_is_real and val_is_real

    if both_real:
        print("\n+" + "=" * 66 + "+")
        print("| [AUDIT: PASS] VERIFIED REAL SATELLITE DATASETS LOADED            |")
        print(f"| Training Source:     {train_src.upper():<43} |")
        print(f"| Training Samples:    {n_train:<43} |")
        print(f"| Validation Source:   {val_src.upper():<43} |")
        print(f"| Validation Samples:  {n_val:<43} |")
        print(f"| Input Channels:      {args.in_channels} (SAR VV, VH bands){'':<27} |")
        print("| Procedural Fallback: DISABLED FOR BOTH TRAIN AND VALIDATION      |")
        print("+" + "=" * 66 + "+\n")
    else:
        print("\n!" + "=" * 66 + "!")
        print("| [AUDIT: WARNING] PROCEDURAL FALLBACK IN USE                      |")
        print(f"| Training Set:   source='{train_src}', samples={n_train:<24} |")
        print(f"| Validation Set: source='{val_src}', samples={n_val:<24} |")
        if not train_is_real:
            print("|  -> Training data is procedural synthetic!                       |")
        if not val_is_real:
            print("|  -> Validation data is procedural synthetic!                     |")
        print("| To train on real labeled data, first run:                        |")
        print("|   python ai_model/dataset.py --download                          |")
        print("!" + "=" * 66 + "!\n")
        if args.require_real:
            raise RuntimeError(
                f"[AUDIT FAILED] --require-real was specified, but datasets failed audit: "
                f"train_source='{train_src}' (n={n_train}), val_source='{val_src}' (n={n_val}). "
                f"Both training and validation sets must be verified real datasets with nonzero samples."
            )

    # Model
    print(f"[2/4] Initializing MultimodalFloodUNet (Channels: {args.in_channels} -> Classes: {args.num_classes})...")
    model = MultimodalFloodUNet(in_channels=args.in_channels, num_classes=args.num_classes, base_channels=32).to(device)

    # Loss Functions: Class-weighted CrossEntropy + Dice Loss with ignore_index=-1
    if args.num_classes == 2:
        class_weights = torch.tensor([1.0, 4.0], device=device)  # Rare water class weighting
    else:
        class_weights = torch.tensor([1.0, 3.0, 3.5], device=device)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    dice_loss_fn = DiceLoss(ignore_index=-1)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    print(f"\n[3/4] Starting Training for {args.epochs} Epochs (Batch Size: {args.batch_size})...")
    best_water_iou = 0.0
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        train_batches = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            valid_mask = (masks != -1) & (masks >= 0) & (masks < args.num_classes)
            if not valid_mask.any():
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                loss_ce = ce_loss_fn(outputs, masks)
                loss_dice = dice_loss_fn(outputs, masks)
                loss = loss_ce + 0.8 * loss_dice

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        train_loss = (train_loss / train_batches) if train_batches > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                valid_mask = (masks != -1) & (masks >= 0) & (masks < args.num_classes)
                if not valid_mask.any():
                    continue

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = ce_loss_fn(outputs, masks) + 0.8 * dice_loss_fn(outputs, masks)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        val_loss = (val_loss / val_batches) if val_batches > 0 else 0.0
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        class_ious, miou = compute_iou(cat_preds, cat_targets, num_classes=args.num_classes, ignore_index=-1)

        def fmt_iou(v):
            return f"{v*100:.1f}%" if not np.isnan(v) else "N/A"

        water_iou = class_ious[1] if (args.num_classes >= 2 and not np.isnan(class_ious[1])) else (class_ious[0] if not np.isnan(class_ious[0]) else 0.0)
        elapsed = time.time() - epoch_start
        vram_used = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0

        if args.num_classes == 2:
            class_str = f"Dry: {fmt_iou(class_ious[0])}, Water: {fmt_iou(class_ious[1])}"
        else:
            class_str = f"Dry: {fmt_iou(class_ious[0])}, Water: {fmt_iou(class_ious[1])}, Debris: {fmt_iou(class_ious[2])}"

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Water IoU: {fmt_iou(water_iou)} | mIoU: {miou * 100:.2f}% ({class_str}) | "
              f"VRAM: {vram_used:.1f} MB | Time: {elapsed:.1f}s")

        # Save checkpoint if Water IoU improves or at epoch 1
        if water_iou > best_water_iou or epoch == 1:
            if water_iou > best_water_iou:
                best_water_iou = water_iou
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "water_iou": water_iou,
                "miou": miou,
                "class_ious": class_ious,
                "data_source": train_src,
                "val_data_source": val_src,
                "is_real_data_trained": both_real,
                "config": {
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "patch_size": args.patch_size,
                    "batch_size": args.batch_size,
                }
            }, best_model_path)
            print(f"  --> Saved new best model (Water IoU: {best_water_iou * 100:.2f}%) to {best_model_path}")

    print(f"\n[4/4] Training Complete! Best Validation Water IoU: {best_water_iou * 100:.2f}% (mIoU: {best_miou * 100:.2f}%)")
    print(f"Data Source Verified: {'REAL BENCHMARK' if both_real else 'PROCEDURAL SYNTHETIC'}")
    print(f"Model saved at: {os.path.abspath(best_model_path)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Flood Segmentation UNet")
    parser.add_argument("--data-dir", type=str, default="", help="Path to pre-processed Sen1Floods11 dataset folder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (15-20 recommended)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (4 recommended for 6GB RTX 4050)")
    parser.add_argument("--patch-size", type=int, default=256, help="Image patch size (H, W)")
    parser.add_argument("--in-channels", type=int, default=2, help="Number of input SAR channels (2: VV/VH)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes (2: water/non-water)")
    parser.add_argument("--samples", type=int, default=320, help="Number of samples if using procedural generator")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save checkpoint")
    parser.add_argument("--require-real", action="store_true", help="Raise error if either train or val datasets fall back to procedural data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
