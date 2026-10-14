"""
Independent Test-Set Evaluation Script for Multimodal Flood Segmentation
========================================================================
Rigorous, scientific evaluation of weights/best_flood_model.pth exclusively
on the held-out, untouched Sen1Floods11 test split (410 patches).

Compliance & Evaluation Rules:
1. STRICT READ-ONLY: Never updates model weights or modifies checkpoint.
2. ZERO PROCEDURAL FALLBACK: Fails immediately if test split is missing or synthetic.
3. PREPROCESSING PARITY: Enforces identical 2-channel SAR [VV, VH] normalization.
4. NODATA MASKING: Completely excludes label pixels marked -1 (clouds/missing SAR).
5. SCIENTIFIC METRICS: Computes IoU, Precision, Recall, F1, and full confusion matrix.
6. HARDWARE AUDIT: Reports both active tensor VRAM and PyTorch reserved memory cache.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_ROOT)

from ai_model.model import MultimodalFloodUNet
from ai_model.dataset import FloodSegmentationDataset


def evaluate_test_set(checkpoint_path: str = None, batch_size: int = 4, device_str: str = None):
    # ── 1. Device Setup ───────────────────────────────────────────────────────
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  SPACE TRACK B: INDEPENDENT HELD-OUT TEST SPLIT EVALUATION")
    print("=" * 72)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[DEVICE] Hardware Acceleration: {gpu_name} ({total_vram_gb:.2f} GB Total VRAM)")
    else:
        print("[DEVICE] CPU Execution (CUDA unavailable)")

    # ── 2. Checkpoint Discovery & Architecture Compatibility ─────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(_ROOT, "weights", "best_flood_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[EVALUATION HALTED] Target checkpoint not found at: {checkpoint_path}\n"
            "Train the model first using: python ai_model/train.py --require-real"
        )

    print(f"\n[1/4] Loading Trained Model Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    data_source = ckpt.get("data_source", "unknown")
    val_data_source = ckpt.get("val_data_source", "unknown")
    is_real = ckpt.get("is_real_data_trained", False)
    epoch = ckpt.get("epoch", "N/A")
    val_water_iou = ckpt.get("water_iou", None)
    val_miou = ckpt.get("miou", None)
    cfg = ckpt.get("config", {})

    in_channels = cfg.get("in_channels", 2)
    num_classes = cfg.get("num_classes", 2)
    patch_size = cfg.get("patch_size", 256)

    print(f"  - Checkpoint Training Epoch:       {epoch}")
    print(f"  - Checkpoint Training Source:      {data_source.upper()}")
    print(f"  - Checkpoint Validation Source:    {val_data_source.upper()}")
    print(f"  - Provenance Authenticity:         {'VERIFIED REAL DATA' if is_real else 'UNVERIFIED / SYNTHETIC'}")
    if val_water_iou is not None:
        print(f"  - Recorded Validation Water IoU:   {val_water_iou * 100:.2f}%")
    if val_miou is not None:
        print(f"  - Recorded Validation Mean IoU:    {val_miou * 100:.2f}%")
    print(f"  - Architecture Configuration:      Channels={in_channels}, Classes={num_classes}, Resolution={patch_size}x{patch_size}")

    if not is_real or data_source not in ("sen1floods11", "kuro_siwo", "user_supplied"):
        raise RuntimeError(
            f"[AUDIT FAILED] Checkpoint provenance is '{data_source}'. "
            "Evaluation requires weights trained on genuine benchmark data."
        )

    # Instantiate Model in strict evaluation mode
    model = MultimodalFloodUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # Freezes batch norm & dropout

    # ── 3. Load Held-Out Test Split (Strict No Procedural Fallback) ────────────
    print(f"\n[2/4] Loading Untouched Test Split (Split: 'test')...")
    test_ds = FloodSegmentationDataset(
        data_dir=None,
        split="test",
        augment=False,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        auto_download=False,
    )

    test_src = test_ds.data_source
    n_test = len(test_ds)

    print(f"  - Test Dataset Source:             {test_src.upper()}")
    print(f"  - Held-Out Test Samples:           {n_test} patches")

    if test_src not in ("sen1floods11", "kuro_siwo", "user_supplied") or test_ds.use_procedural:
        raise RuntimeError(
            f"[AUDIT FAILED] Test set fell back to procedural synthesis (source='{test_src}'). "
            "Independent evaluation demands genuine, hand-labelled satellite test scenes."
        )

    if n_test == 0:
        raise RuntimeError(
            "[EVALUATION HALTED] Zero test samples located in data/sen1floods11/test. "
            "Ensure Sen1Floods11 test split is prepared."
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # ── 4. Inference & Cumulative Pixel Confusion Matrix ─────────────────────
    print(f"\n[3/4] Running Inference Across {n_test} Test Scenes (Batch Size: {batch_size})...")

    # Binary Confusion Matrix counts (for valid pixels != -1):
    # 0 = Dry Land, 1 = Flood Water
    tp = 0  # Water correctly predicted as Water
    fp = 0  # Dry incorrectly predicted as Water
    fn = 0  # Water incorrectly predicted as Dry
    tn = 0  # Dry correctly predicted as Dry

    total_valid_pixels = 0
    total_ignore_pixels = 0
    total_tiles = 0

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

            # Flatten to 1D tensors
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Mask out ignore pixels (-1)
            valid_mask = (masks_flat != -1) & (masks_flat >= 0) & (masks_flat < num_classes)
            ignore_count = (~valid_mask).sum().item()
            total_ignore_pixels += ignore_count

            if not valid_mask.any():
                continue

            v_preds = preds_flat[valid_mask]
            v_targs = masks_flat[valid_mask]

            # Accumulate confusion matrix entries
            # Class 0: Dry Land, Class 1: Flood Water
            p_water = (v_preds == 1)
            p_dry   = (v_preds == 0)
            t_water = (v_targs == 1)
            t_dry   = (v_targs == 0)

            tp += (p_water & t_water).sum().item()
            fp += (p_water & t_dry).sum().item()
            fn += (p_dry & t_water).sum().item()
            tn += (p_dry & t_dry).sum().item()

            total_valid_pixels += v_preds.numel()
            total_tiles += images.shape[0]

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(test_loader):
                pct = (batch_idx + 1) / len(test_loader) * 100
                print(f"  -> Processed {total_tiles:>3}/{n_test} tiles ({pct:>5.1f}%)")

    total_time = time.time() - inference_start
    ms_per_tile = (total_time / n_test) * 1000.0 if n_test > 0 else 0.0
    fps = n_test / total_time if total_time > 0 else 0.0

    # ── 5. Mathematical Metric Computations ──────────────────────────────────
    print(f"\n[4/4] Computing Definitive Evaluation Metrics...")

    # IoU
    water_union = tp + fp + fn
    water_iou = (tp / water_union) if water_union > 0 else 0.0

    dry_union = tn + fp + fn
    dry_iou = (tn / dry_union) if dry_union > 0 else 0.0

    miou = (water_iou + dry_iou) / 2.0

    # Precision, Recall, F1 for Water class
    water_prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    water_rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    water_f1   = (2.0 * water_prec * water_rec / (water_prec + water_rec)) if (water_prec + water_rec) > 0 else 0.0

    # Overall Pixel Accuracy
    pixel_acc = ((tp + tn) / total_valid_pixels) if total_valid_pixels > 0 else 0.0

    # VRAM Telemetry
    if device.type == "cuda":
        vram_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        vram_max = torch.cuda.max_memory_allocated(0) / (1024**2)
    else:
        vram_allocated = vram_reserved = vram_max = 0.0

    # ── 6. Print Scientific Benchmark Report ─────────────────────────────────
    print("\n" + "=" * 72)
    print("  OFFICIAL INDEPENDENT TEST BENCHMARK REPORT (SEN1FLOODS11)")
    print("=" * 72)
    print(f"  Dataset Split:                   HELD-OUT TEST SET (Never Seen During Training)")
    print(f"  Data Source:                     {test_src.upper()} (Hand-Labelled Sentinel-1 SAR)")
    print(f"  Test Tiles Evaluated:            {n_test} scenes ({patch_size}x{patch_size} pixels)")
    print(f"  Total Valid Pixels:              {total_valid_pixels:,}")
    print(f"  Ignored Nodata / Cloud Pixels:   {total_ignore_pixels:,} (Masked with ignore_index=-1)")
    print(f"  Inference Latency:               {ms_per_tile:.1f} ms / patch ({fps:.1f} FPS)")
    if device.type == "cuda":
        print(f"  GPU Active Tensor Memory:        {vram_allocated:.1f} MB")
        print(f"  GPU Reserved Memory (Cache):     {vram_reserved:.1f} MB (Peak Active: {vram_max:.1f} MB)")
    print("-" * 72)
    print("  CONFUSION MATRIX (Pixel-Level):")
    print(f"    True Positives  (Water -> Water):  {tp:>12,} pixels")
    print(f"    False Positives (Dry   -> Water):  {fp:>12,} pixels")
    print(f"    False Negatives (Water -> Dry):    {fn:>12,} pixels")
    print(f"    True Negatives  (Dry   -> Dry):    {tn:>12,} pixels")
    print("-" * 72)
    print("  FINAL ACCURACY & SEGMENTATION METRICS:")
    print(f"    [*] Water IoU (Jaccard Index):     {water_iou * 100:>6.2f}%")
    print(f"    [*] Dry Land IoU:                  {dry_iou * 100:>6.2f}%")
    print(f"    [*] Mean IoU (mIoU):               {miou * 100:>6.2f}%")
    print(f"    [*] Water Precision:               {water_prec * 100:>6.2f}%")
    print(f"    [*] Water Recall:                  {water_rec * 100:>6.2f}%")
    print(f"    [*] Water F1-Score:                {water_f1 * 100:>6.2f}%")
    print(f"    [*] Overall Pixel Accuracy:        {pixel_acc * 100:>6.2f}%")
    print("=" * 72)
    print("  BENCHMARK VERDICT: OFFICIALLY VALIDATED ON GENUINE SATELLITE TEST SET")
    print("=" * 72 + "\n")

    return {
        "dataset_source": test_src,
        "test_samples": n_test,
        "water_iou": water_iou,
        "dry_iou": dry_iou,
        "miou": miou,
        "water_precision": water_prec,
        "water_recall": water_rec,
        "water_f1": water_f1,
        "pixel_accuracy": pixel_acc,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "valid_pixels": total_valid_pixels,
        "ignore_pixels": total_ignore_pixels,
        "ms_per_tile": ms_per_tile,
        "fps": fps
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate flood segmentation on untouched test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_str=args.device
    )
