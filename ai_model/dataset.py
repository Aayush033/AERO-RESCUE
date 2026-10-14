"""
Flood & Debris Segmentation Dataset Loader
==========================================
Priority order (highest to lowest):
  1. Real Sen1Floods11 benchmark dataset (Bonafilia et al., 2020)
       - Hand-labelled multi-continent Sentinel-1 flood benchmark
       - Strict scene-level splitting (no spatial leakage between train/val/test)
       - Official splits: train (252 scenes), valid (89 scenes), test (90 scenes)
       - 2-class ground truth: 0 = non-water / dry terrain, 1 = water, -1 = ignore / no-data
       - Local cache: data/sen1floods11/{train,val,test}/*.npz
  2. Real Kuro Siwo benchmark dataset (Bountos et al., NeurIPS 2024)
       - Local cache: data/kuro_siwo/
  3. User-supplied directory with .npz tiles (<data_dir>/<split>/*.npz)
  4. Himalayan Valley Case-Study Tile Generator
       - Physics-based procedural synthesis used ONLY as last-resort offline fallback
       - Clearly labelled "procedural" in reporting

Compliance:
  - Supports 2-class water segmentation (Sen1Floods11 standard)
  - Properly preserves -1 ignore_index for no-data / cloud-masked pixels
  - Scene-level splitting prevents patch-level spatial leakage
"""

import os
import glob
import logging
import math
import shutil
import tarfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# Project root and cache directories
_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_S1F11_DIR = _DATA_DIR / "sen1floods11"
_KURO_DIR = _DATA_DIR / "kuro_siwo"

try:
    from huggingface_hub import hf_hub_download
    _HF_OK = True
except ImportError:
    _HF_OK = False

try:
    import rasterio
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Sen1Floods11 Downloader & Scene-Level Patch Preprocessor
# ──────────────────────────────────────────────────────────────────────────────

def download_and_prepare_sen1floods11(
    target_dir: Optional[Path] = None,
    max_scenes: Optional[int] = None,
    patch_size: int = 256,
) -> Path:
    """
    Downloads official Sen1Floods11 v1.1 hand-labelled dataset from Hugging Face
    (blumenstiel/Sen1Floods11), splits scenes by official train/valid/test splits,
    and extracts non-overlapping 256x256 patches.

    Strict scene-level splitting ensures zero spatial leakage between train and val.
    Saves pre-processed patches as .npz into target_dir/{train,val,test}/.
    """
    out_dir = Path(target_dir or _S1F11_DIR)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"

    # If already prepared with sufficient tiles, return early
    existing_train = list(train_dir.glob("*.npz"))
    existing_val = list(val_dir.glob("*.npz"))
    if len(existing_train) >= 50 and len(existing_val) >= 20:
        log.info(f"[DATASET] Sen1Floods11 cache ready: {len(existing_train)} train, {len(existing_val)} val patches.")
        return out_dir

    if not _HF_OK:
        raise RuntimeError("huggingface_hub is required to download Sen1Floods11. Run: pip install huggingface_hub")

    log.info("[DATASET] Downloading official Sen1Floods11 v1.1 archive (~1.65 GB) from Hugging Face...")
    archive_path = hf_hub_download(
        repo_id="blumenstiel/Sen1Floods11",
        filename="sen1floods11_v1_1.tar.gz",
        repo_type="dataset",
    )
    log.info(f"[DATASET] Archive downloaded to {archive_path}. Extracting...")

    extract_tmp = out_dir / "_raw_extracted"
    extract_tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_tmp)

    log.info("[DATASET] Archive extracted. Discovering files and official splits...")

    # Locate S1 and Label folders inside extracted directory
    s1_files = {p.stem.replace("_S1Hand", ""): p for p in extract_tmp.glob("**/*_S1Hand.tif")}
    lbl_files = {p.stem.replace("_LabelHand", ""): p for p in extract_tmp.glob("**/*_LabelHand.tif")}
    common_scenes = sorted(list(set(s1_files.keys()) & set(lbl_files.keys())))

    log.info(f"[DATASET] Found {len(common_scenes)} matched hand-labelled scenes.")

    # Locate official split files if present
    train_split_files = list(extract_tmp.glob("**/flood_train_data.txt")) + list(extract_tmp.glob("**/flood_train_data.csv"))
    val_split_files = list(extract_tmp.glob("**/flood_valid_data.txt")) + list(extract_tmp.glob("**/flood_valid_data.csv"))

    train_names, val_names = set(), set()
    if train_split_files:
        with open(train_split_files[0], "r") as f:
            train_names = {line.strip().replace(".tif", "").replace("_LabelHand", "") for line in f if line.strip()}
    if val_split_files:
        with open(val_split_files[0], "r") as f:
            val_names = {line.strip().replace(".tif", "").replace("_LabelHand", "") for line in f if line.strip()}

    # Fallback to deterministic 80/20 scene split if official text splits missing
    if not train_names:
        n_tr = int(len(common_scenes) * 0.8)
        train_names = set(common_scenes[:n_tr])
        val_names = set(common_scenes[n_tr:])

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    if not _RASTERIO_OK:
        raise RuntimeError("rasterio is required to read Sen1Floods11 GeoTIFFs.")

    for scene_id in common_scenes:
        if max_scenes and (counts["train"] + counts["val"] >= max_scenes):
            break

        split_name = "train" if scene_id in train_names else ("val" if scene_id in val_names else "test")
        save_folder = out_dir / split_name

        s1_path = s1_files[scene_id]
        lbl_path = lbl_files[scene_id]

        try:
            with rasterio.open(s1_path) as src_s1, rasterio.open(lbl_path) as src_lbl:
                # S1 has 2 bands: VV and VH (dB)
                s1_data = src_s1.read()  # (2, 512, 512)
                lbl_data = src_lbl.read(1)  # (512, 512) with values {-1, 0, 1}

            # Clean NaNs/Infs from nodata SAR border pixels
            s1_clean = np.nan_to_num(s1_data, nan=-35.0, posinf=0.0, neginf=-35.0)

            # Normalise SAR dB: [-35, 0] -> [0, 1]
            vv = np.clip((s1_clean[0] + 35.0) / 35.0, 0.0, 1.0).astype(np.float32)
            vh = np.clip((s1_clean[1] + 35.0) / 35.0, 0.0, 1.0).astype(np.float32)

            # Genuine 2-channel SAR input: Band 0 = VV, Band 1 = VH
            two_chan = np.stack([vv, vh], axis=0)  # (2, 512, 512)

            H, W = lbl_data.shape
            # Extract 256x256 patches strictly within this scene
            p_idx = 0
            for r in range(0, H - patch_size + 1, patch_size):
                for c in range(0, W - patch_size + 1, patch_size):
                    patch_img = two_chan[:, r:r + patch_size, c:c + patch_size]
                    patch_lbl = lbl_data[r:r + patch_size, c:c + patch_size].astype(np.int64)

                    # Skip patches that are 100% no-data
                    if np.all(patch_lbl == -1):
                        continue

                    out_name = f"{scene_id}_p{p_idx}.npz"
                    np.savez_compressed(
                        save_folder / out_name,
                        image=patch_img,
                        label=patch_lbl,
                        scene_id=scene_id,
                    )
                    counts[split_name] += 1
                    p_idx += 1

        except Exception as exc:
            log.warning(f"[DATASET] Failed processing scene {scene_id}: {exc}")

    # Clean up raw extracted GeoTIFFs to save disk space
    shutil.rmtree(extract_tmp, ignore_errors=True)

    log.info(f"[DATASET] Sen1Floods11 preparation complete! Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────────────────────────────────────

class FloodSegmentationDataset(Dataset):
    """
    Multimodal Flood & Terrain Segmentation Dataset.

    Channels (4):
      0: Pre-event Sentinel-1 SAR VV   (normalized to [0, 1])
      1: Post-event Sentinel-1 SAR VV  (normalized to [0, 1])
      2: Post-event Sentinel-1 SAR VH  (normalized to [0, 1])
      3: Copernicus DEM Slope           (normalized to [0, 1])

    Labels:
      2-class mode (num_classes=2, standard for Sen1Floods11):
        0: Non-water / Dry terrain
        1: Water / Inundated surface
        -1: Ignore / No-data (masked / cloudy / unlabelled pixels)
      3-class mode (num_classes=3, procedural multi-hazard fallback):
        0: Normal / Dry terrain
        1: Water / Flooded riverbed
        2: Mud & debris flow deposit
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        augment: bool = True,
        patch_size: int = 256,
        in_channels: int = 2,
        sample_count: int = 320,
        num_classes: int = 2,
        auto_download: bool = False,
    ):
        self.split = split
        self.augment = augment
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.sample_count = sample_count
        self.num_classes = num_classes

        self.file_list: List[str] = []
        self.data_source = "procedural"

        # ── 1. Check user-supplied data directory ─────────────────────────────
        if data_dir and os.path.exists(data_dir):
            patterns = [
                os.path.join(data_dir, split, "*.npz"),
                os.path.join(data_dir, "*.npz"),
            ]
            for pat in patterns:
                found = sorted(glob.glob(pat))
                if found:
                    self.file_list = found
                    self.data_source = "user_supplied"
                    log.info(f"[DATASET] User-supplied: {len(self.file_list)} tiles ({split})")
                    break

        # ── 2. Check local Sen1Floods11 cache ────────────────────────────────
        if not self.file_list:
            split_dir = _S1F11_DIR / split
            found = sorted(glob.glob(str(split_dir / "*.npz")))
            if found:
                self.file_list = found
                self.data_source = "sen1floods11"
                log.info(f"[DATASET] Sen1Floods11: {len(self.file_list)} real tiles loaded ({split})")
            elif auto_download:
                try:
                    download_and_prepare_sen1floods11(patch_size=patch_size)
                    found = sorted(glob.glob(str(split_dir / "*.npz")))
                    if found:
                        self.file_list = found
                        self.data_source = "sen1floods11"
                except Exception as exc:
                    log.warning(f"[DATASET] Auto-download failed: {exc}")

        # ── 3. Check local Kuro Siwo cache ───────────────────────────────────
        if not self.file_list:
            found = sorted(glob.glob(str(_KURO_DIR / split / "*.npz"))) or sorted(glob.glob(str(_KURO_DIR / "*.npz")))
            if found:
                n_tr = int(len(found) * 0.8)
                tiles = found[:n_tr] if split == "train" else found[n_tr:]
                self.file_list = tiles
                self.data_source = "kuro_siwo"
                log.info(f"[DATASET] Kuro Siwo: {len(self.file_list)} tiles ({split})")

        # ── 4. Procedural fallback (air-gapped or pre-download demo) ─────────
        self.use_procedural = len(self.file_list) == 0
        if self.use_procedural:
            log.info(
                f"[DATASET] Using physics-based Himalayan procedural tile generator ({split}). "
                f"Run 'python ai_model/dataset.py --download' to download real Sen1Floods11 imagery."
            )

    def __len__(self) -> int:
        if self.use_procedural:
            return self.sample_count if self.split == "train" else max(40, self.sample_count // 4)
        return len(self.file_list)

    def __repr__(self) -> str:
        return (
            f"FloodSegmentationDataset(split={self.split!r}, "
            f"source={self.data_source!r}, channels={self.in_channels}, classes={self.num_classes}, n={len(self)})"
        )

    # ── Procedural Fallback Tile Generator ────────────────────────────────────
    def _generate_synthetic_himalayan_tile(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Physics-based SAR + DEM + flood tile generator for offline fallback."""
        rng = np.random.RandomState(seed)
        H, W = self.patch_size, self.patch_size

        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x, y)

        meander_freq = rng.uniform(1.5, 3.5)
        meander_amp = rng.uniform(0.15, 0.4)
        river_cx = meander_amp * np.sin(meander_freq * yy * np.pi) + rng.uniform(-0.1, 0.1)
        dist_to_river = np.abs(xx - river_cx)

        elevation = 1500.0 + 1200.0 * dist_to_river + 100.0 * np.sin(3 * xx)
        gy, gx = np.gradient(elevation)
        slope = np.sqrt(gx**2 + gy**2)
        slope_norm = np.clip(slope / 50.0, 0.0, 1.0).astype(np.float32)

        normal_river = (dist_to_river < 0.04) & (slope_norm < 0.3)
        flood_width = rng.uniform(0.10, 0.22)
        flood_cor = (dist_to_river < flood_width) & (slope_norm < 0.25)
        water_mask = flood_cor & (dist_to_river < flood_width * 0.55)
        debris_mask = flood_cor & (~water_mask)

        label = np.zeros((H, W), dtype=np.int64)
        if self.num_classes == 3:
            label[debris_mask] = 2
            label[water_mask] = 1
        else:
            # 2-class mode: 0 = non-water, 1 = water (debris is non-water in pure flood model)
            label[water_mask] = 1

        pre_vv = rng.normal(-12.0, 3.0, (H, W)) + 6.0 * slope_norm
        pre_vv[normal_river] = rng.normal(-24.0, 2.0, int(normal_river.sum()))
        post_vv = pre_vv.copy()
        post_vh = rng.normal(-19.0, 3.5, (H, W)) + 4.0 * slope_norm
        post_vv[water_mask] = rng.normal(-24.5, 2.0, int(water_mask.sum()))
        post_vh[water_mask] = rng.normal(-31.0, 2.5, int(water_mask.sum()))
        post_vv[debris_mask] = rng.normal(-9.0, 2.5, int(debris_mask.sum()))
        post_vh[debris_mask] = rng.normal(-15.0, 2.8, int(debris_mask.sum()))

        def norm(a):
            return np.clip((a + 35.0) / 35.0, 0.0, 1.0).astype(np.float32)

        if self.in_channels == 2:
            image = np.stack([norm(post_vv), norm(post_vh)], axis=0)
        else:
            image = np.stack([norm(pre_vv), norm(post_vv), norm(post_vh), slope_norm], axis=0)
        return image, label

    # ── Real Tile Loader ──────────────────────────────────────────────────────
    def _load_real_tile(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and formats a .npz tile with in_channels dimensions."""
        data = np.load(path)
        image = np.nan_to_num(data["image"].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        label = data["label"].astype(np.int64)

        # Match requested in_channels
        if self.in_channels == 2:
            if image.shape[0] == 4:
                # [pre_vv, post_vv, post_vh, slope] -> pick genuine post_vv & post_vh
                image = image[1:3]
            elif image.shape[0] > 2:
                image = image[:2]
            elif image.shape[0] < 2:
                padding = np.zeros((2 - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=0)
        elif self.in_channels == 4:
            if image.shape[0] < 4:
                padding = np.zeros((4 - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=0)
            elif image.shape[0] > 4:
                image = image[:4]

        image = np.clip(image, 0.0, 1.0)

        # In 2-class mode, map any class >= 2 to 0 (non-water), while preserving -1 ignore_index
        if self.num_classes == 2:
            mask_debris = (label >= 2)
            label[mask_debris] = 0

        # Resize to patch_size if needed
        if image.shape[1] != self.patch_size or image.shape[2] != self.patch_size:
            from scipy.ndimage import zoom
            zy = self.patch_size / image.shape[1]
            zx = self.patch_size / image.shape[2]
            image = zoom(image, (1, zy, zx), order=1)
            label = zoom(label, (zy, zx), order=0)

        return image, label

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_procedural:
            seed = (idx * 17 + (1000 if self.split == "val" else 0)) % 1_000_000
            image, label = self._generate_synthetic_himalayan_tile(seed)
        else:
            image, label = self._load_real_tile(self.file_list[idx])

        # Augmentation (train split only)
        if self.augment and self.split == "train":
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                image = np.rot90(image, k, (1, 2)).copy()
                label = np.rot90(label, k, (0, 1)).copy()

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 4,
    patch_size: int = 256,
    in_channels: int = 2,
    num_classes: int = 2,
    sample_count: int = 320,
    auto_download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FloodSegmentationDataset(
        data_dir=data_dir, split="train", augment=True,
        patch_size=patch_size, in_channels=in_channels,
        sample_count=sample_count, num_classes=num_classes,
        auto_download=auto_download,
    )
    val_ds = FloodSegmentationDataset(
        data_dir=data_dir, split="val", augment=False,
        patch_size=patch_size, in_channels=in_channels,
        sample_count=sample_count, num_classes=num_classes,
        auto_download=auto_download,
    )

    log.info(f"[DATASET] Train: {train_ds} | Val: {val_ds}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# CLI: python ai_model/dataset.py --download
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Sen1Floods11 Dataset Preparation & Verification")
    parser.add_argument("--download", action="store_true", help="Download and extract official Sen1Floods11 benchmark")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (4 recommended for RTX 4050 6GB)")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional max scenes to process")
    args = parser.parse_args()

    if args.download:
        print("\n=======================================================")
        print("Downloading & preparing official Sen1Floods11 dataset...")
        print("=======================================================")
        download_and_prepare_sen1floods11(max_scenes=args.max_scenes)

    tl, vl = get_dataloaders(batch_size=args.batch_size, num_classes=2)
    x, y = next(iter(tl))
    print(f"\n[OK] Loaded batch: Images shape={x.shape}, Labels shape={y.shape}")
    print(f"[OK] Label classes in batch: {torch.unique(y).tolist()}")
    print(f"[OK] Train dataset: {tl.dataset}")
    print(f"[OK] Val dataset:   {vl.dataset}")
