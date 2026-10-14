# AERO-RESCUE: Independent Scientific Benchmark Report

**Space Track B — Satellite-Based Flood & Debris Mapping for Himalayan Disaster Response**

---

## 1. Model Architecture

| Parameter | Value |
|-----------|-------|
| **Architecture** | U-Net (Encoder-Decoder with Skip Connections) |
| **Input Channels** | 2 (Sentinel-1 SAR: VV + VH) |
| **Output Classes** | 2 (Dry Land, Flood Water) |
| **Base Channels** | 32 |
| **Resolution** | 256 × 256 pixels per patch |
| **Parameters** | ~9.6 M |
| **Framework** | PyTorch 2.x with CUDA AMP |
| **Training Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |

## 2. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999) |
| **Loss Function** | Cross-Entropy with ignore_index=-1 (nodata masking) |
| **Batch Size** | 4 |
| **Epochs** | 15 |
| **Mixed Precision** | Enabled (PyTorch AMP) |
| **Augmentation** | Random horizontal/vertical flips, rotation |

## 3. Dataset Provenance

| Split | Source | Samples | Verification |
|-------|--------|---------|-------------|
| **Train** | Sen1Floods11 (Official) | 984 patches | ✅ `--require-real` flag enforced |
| **Validation** | Sen1Floods11 (Official) | 246 patches | ✅ Verified during training |
| **Test** | Sen1Floods11 (Official) | 410 patches | ✅ Held-out, never seen during training |

**Sen1Floods11** is a globally-distributed, hand-labelled Sentinel-1 SAR flood segmentation benchmark with 446 scenes across 11 flood events worldwide.

- **No procedural/synthetic data** was used at any stage.
- **No test-set leakage**: Test split was strictly withheld during all training and hyperparameter tuning.
- **Preprocessing**: VV and VH bands calibrated from digital numbers to σ⁰ decibels using `20·log₁₀(max(DN, 1.0)) − 57.0`, then normalized to [0, 1] from the range [−35, 0] dB.

## 4. Independent Test-Set Evaluation Results

Evaluated on **410 held-out test tiles** (256×256 pixels each), totaling **23,385,182 valid pixels** after masking **3,484,578 nodata/cloud pixels**.

### Confusion Matrix (Pixel-Level)

|  | **Predicted: Water** | **Predicted: Dry** |
|--|--:|--:|
| **Actual: Water** | 2,328,395 (TP) | 692,570 (FN) |
| **Actual: Dry** | 710,769 (FP) | 19,653,448 (TN) |

### Segmentation Metrics

| Metric | Score |
|--------|------:|
| **Water IoU (Jaccard Index)** | **62.39%** |
| **Dry Land IoU** | **93.34%** |
| **Mean IoU (mIoU)** | **77.86%** |
| **Water Precision** | **76.61%** |
| **Water Recall** | **77.07%** |
| **Water F1-Score** | **76.84%** |
| **Overall Pixel Accuracy** | **94.00%** |

### Inference Performance

| Metric | Value |
|--------|------:|
| **Latency per Patch** | 14.6 ms |
| **Throughput** | 68.5 FPS |
| **GPU Active Memory** | 56.0 MB |
| **GPU Peak Memory** | 212.8 MB |

## 5. Context & Interpretation

- **Water IoU of 62.39%** is competitive with published Sen1Floods11 baselines (U-Net range: 55–68% depending on configuration).
- **94% pixel accuracy** reflects the high-quality true negative classification — the model reliably identifies non-flooded terrain.
- **Balanced precision/recall** (76.6% / 77.1%) means the model avoids both excessive false alarms and missed detections.
- **14.6 ms inference latency** enables real-time interactive mapping on consumer-grade GPUs.

## 6. Debris Flow Detection (Multi-Temporal SAR Change Detection)

Since Sen1Floods11 does not include debris labels, debris flow detection uses a **documented, physics-based method** (not learned from training):

1. **Backscatter roughness change**: `Δσ⁰ = post_VV − pre_VV`
2. **DEM slope gating**: Debris is expected on moderate valley slopes (< 25°), not on flat water surfaces.
3. **Sigmoid activation**: `debris_prob = σ((Δσ⁰ − 0.04) × 6.0) × slope_gate`
4. **Decision rule**: Debris where `debris_prob > 0.55` AND `water_prob < 0.45`

This is **explicitly documented as a heuristic**, not a trained classifier.

## 7. Live System Integration

The trained model is integrated into a full-stack disaster mapping system:

| Component | Technology |
|-----------|-----------|
| **AI Inference** | PyTorch U-Net, CUDA-accelerated |
| **Satellite Data** | Microsoft Planetary Computer STAC (Sentinel-1 GRD + Copernicus DEM) |
| **Backend API** | FastAPI with 7 REST endpoints |
| **Infrastructure Analysis** | OpenStreetMap Overpass API + NetworkX graph isolation |
| **Frontend** | Leaflet.js interactive GIS dashboard with real-time layer controls |
| **SitRep Generation** | Zero-hallucination bilingual copilot (English + Nepali) |
| **Validation** | EMSR927 Copernicus EMS concordance benchmark |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | System status, GPU info, model provenance |
| `/api/analyze` | POST | Full satellite analysis pipeline |
| `/api/trace_flow` | POST | DEM hydrological flow tracing (bonus task) |
| `/api/sitrep` | POST | Bilingual situation report generation |
| `/api/copilot/chat` | POST | Factual rescuer Q&A copilot |
| `/api/benchmark/emsr927` | GET | Copernicus EMS validation |
| `/api/evaluate` | GET | Live independent test-set evaluation |

## 8. Reproducibility

```powershell
# 1. Download Sen1Floods11 dataset
.\venv\Scripts\python.exe ai_model\dataset.py --download --batch-size 4

# 2. Train model (15 epochs, verified real data)
.\venv\Scripts\python.exe ai_model\train.py --epochs 15 --batch-size 4 --patch-size 256 --in-channels 2 --num-classes 2 --lr 0.001 --require-real

# 3. Run independent test-set evaluation
.\venv\Scripts\python.exe ai_model\evaluate.py --batch-size 4

# 4. Launch full system
.\venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

## 9. Required Attributions

- "Contains modified Copernicus Sentinel data 2026."
- "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
- "© OpenStreetMap contributors."
- Training Benchmark: Sen1Floods11 v1.1 (Bonafilia et al., 2020).
- Validation Benchmark: Copernicus Emergency Management Service (EMSR927).

---

*Report generated: 2026-10-13 | AERO-RESCUE Space Track B*
