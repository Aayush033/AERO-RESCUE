"""
FastAPI Disaster Management & Satellite Analysis Server
Unifies:
- Copernicus Sentinel-1/2 & WorldDEM-30 pipeline
- Multimodal AI Flood & Debris Segmentation
- OpenStreetMap Pre-Event Infrastructure Damage Overlay
- Network Isolation & Cut-Off Village Routing Analysis
- D8 Hydrological Valley Flow Tracing (Bonus)
- Factual Bilingual SitRep Copilot (Zero Hallucination)
- Copernicus EMS EMSR927 Validation Benchmark
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from copernicus_client import CopernicusDataClient, TRISHULI_BBOX, DEFAULT_EVENT_DATE
from sar_processor import SARProcessor
from dem_hydrology import DEMHydrologyTracer
from osm_analyzer import OSMDamageAnalyzer
from isolation_network import SettlementIsolationAnalyzer
from sitrep_copilot import SitRepCopilot
from emsr927_validator import EMSR927Validator

try:
    from ai_model.infer import FloodSegmentationInference, raster_to_geojson_features
except ImportError:
    from ..ai_model.infer import FloodSegmentationInference, raster_to_geojson_features

app = FastAPI(
    title="Himalaya Flood & Debris Mapping System - Space Track B",
    description="Multimodal AI Satellite Damage Mapping & Rescuer Isolation Analysis System",
    version="1.0.0"
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory pipeline cache
STATE = {
    "latest_analysis": None,
    "dem_tracer": None,
    "ai_engine": None,
    "copernicus_client": CopernicusDataClient(),
    "sar_processor": SARProcessor(),
    "osm_analyzer": OSMDamageAnalyzer(),
    "sitrep_copilot": SitRepCopilot(),
    "emsr927_validator": EMSR927Validator()
}


class AnalyzeRequest(BaseModel):
    bbox: Optional[List[float]] = TRISHULI_BBOX
    event_date: Optional[str] = DEFAULT_EVENT_DATE
    use_ai_model: Optional[bool] = True


class TraceFlowRequest(BaseModel):
    lon: float
    lat: float


class ChatRequest(BaseModel):
    query: str


class SitRepRequest(BaseModel):
    language: Optional[str] = "en"


@app.on_event("startup")
def startup_event():
    print("[SERVER] Initializing Multimodal AI Flood & Debris Mapping Engine...")
    STATE["ai_engine"] = FloodSegmentationInference()
    # Pre-run baseline analysis for Trishuli corridor so dashboard is immediately primed
    run_full_pipeline(TRISHULI_BBOX, DEFAULT_EVENT_DATE, use_ai=True)
    print("[SERVER] Initialization complete. Prime case study loaded.")


def run_full_pipeline(bbox, event_date, use_ai=True):
    client = STATE["copernicus_client"]
    sar = STATE["sar_processor"]
    osm = STATE["osm_analyzer"]
    ai = STATE["ai_engine"]

    # 1. Fetch satellite cube and DEM
    cube, elev, slope, meta = client.fetch_scene_cube(bbox, event_date)
    pre_vv, post_vv, post_vh, slope_norm = cube[0], cube[1], cube[2], cube[3]

    # Initialize DEM hydrological tracer
    STATE["dem_tracer"] = DEMHydrologyTracer(elev, bbox)

    # 2. AI Flood & Debris Segmentation
    if use_ai:
        preds, probs = ai.predict(cube)
        water_features = raster_to_geojson_features(
            preds, bbox, class_id=1, label_name="water", prob_map=probs[1] if probs.ndim == 3 else None
        )
        debris_features = raster_to_geojson_features(
            preds, bbox, class_id=2, label_name="debris", prob_map=probs[2] if probs.ndim == 3 else None
        )
    else:
        _, f_mask, d_mask = sar.detect_changes(pre_vv, post_vv, slope)
        water_features = raster_to_geojson_features(f_mask.astype(int), bbox, class_id=1, label_name="water")
        debris_features = raster_to_geojson_features(d_mask.astype(int), bbox, class_id=2, label_name="debris")

    # 2b. Supplement with SAR Δσ⁰ change-detection for a complete hazard footprint.
    #     The AI model detects absolute post-event water extent (river + inundated areas).
    #     The SAR change detector catches newly flooded terrain where Δσ⁰ dropped > 3.8 dB —
    #     this specifically picks up the Bhote Koshi upper-valley inundation that may be missed
    #     by the AI because pre-event backscatter in steep gorges can look similar to flood SAR.
    #     Using both methods is documented as complementary (EMSR927 methodology).
    try:
        _, sar_f_mask, sar_d_mask = sar.detect_changes(pre_vv, post_vv, slope)
        sar_water_feats  = raster_to_geojson_features(
            sar_f_mask.astype(np.uint8), bbox, class_id=1, label_name="water"
        )
        sar_debris_feats = raster_to_geojson_features(
            sar_d_mask.astype(np.uint8), bbox, class_id=1, label_name="debris"
        )
        # Union: AI polygons carry confidence scores; SAR change polygons extend the footprint
        all_water_features  = water_features  + sar_water_feats
        all_debris_features = debris_features + sar_debris_feats
    except Exception as exc:
        all_water_features  = water_features
        all_debris_features = debris_features

    # 3. Retrieve pre-event OSM infrastructure
    osm_data = osm.get_preevent_infrastructure(bbox)

    # 4. Damage Assessment using FULL merged hazard footprint
    damage_assessment = osm.assess_damage(osm_data, all_water_features, all_debris_features)

    # 5. Network Isolation Analysis
    isolation_analyzer = SettlementIsolationAnalyzer(
        osm_data,
        damage_assessment["damaged_roads"],
        damage_assessment["damaged_bridges"]
    )
    isolation_results = isolation_analyzer.analyze_isolation()

    # Area totals: use original AI features for display (SAR change is for damage only)
    water_sqkm = round(sum(f["properties"]["area_sqkm"] for f in all_water_features), 2)
    debris_sqkm = round(sum(f["properties"]["area_sqkm"] for f in all_debris_features), 2)

    summary_metrics = {
        "location_name": "Bhote Koshi – Trishuli River Corridor, Rasuwa, Nepal",
        "event_date": event_date,
        "flood_area_sqkm": water_sqkm,
        "debris_area_sqkm": debris_sqkm,
        "buildings_affected": damage_assessment["summary"]["total_buildings_affected"],
        "roads_severed_km": damage_assessment["summary"]["total_roads_severed_km"],
        "bridges_destroyed": damage_assessment["summary"]["total_bridges_destroyed"],
        "isolated_settlements_count": isolation_results["total_isolated_settlements"],
        "isolated_population": isolation_results["total_isolated_population"],
        "isolated_villages": isolation_results["isolated_names"]
    }

    result = {
        "metadata": meta,
        "summary": summary_metrics,
        "bbox": bbox,
        "layers": {
            "water_polygons": {"type": "FeatureCollection", "features": all_water_features},
            "debris_polygons": {"type": "FeatureCollection", "features": all_debris_features},
            "settlements": isolation_results["settlements_status"],
            "damaged_roads": damage_assessment["damaged_roads"],
            "intact_roads": damage_assessment["intact_roads"],
            "damaged_bridges": damage_assessment["damaged_bridges"],
            "damaged_buildings_sample": damage_assessment["damaged_buildings_sample"]
        },
        "_water_features": all_water_features,
        "_debris_features": all_debris_features,
        "raw_osm_settlements": osm_data["settlements"]
    }

    STATE["latest_analysis"] = result
    return result


@app.get("/api/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Only"
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if gpu_available else 0
    ai_engine = STATE.get("ai_engine")

    return {
        "status": "healthy",
        "gpu_accelerated": gpu_available,
        "gpu_device": gpu_name,
        "gpu_vram_mb": vram_mb,
        "ai_model_loaded": ai_engine.is_trained if ai_engine else False,
        "ai_model_data_source": ai_engine.data_source if ai_engine else "none",
        "ai_model_real_data_verified": ai_engine.is_real_data_trained if ai_engine else False,
        "ai_model_input_channels": ai_engine.in_channels if ai_engine else None,
        "case_study_loaded": STATE["latest_analysis"] is not None
    }


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    try:
        data = run_full_pipeline(req.bbox, req.event_date, use_ai=req.use_ai_model)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trace_flow")
def trace_flow_endpoint(req: TraceFlowRequest):
    if not STATE.get("dem_tracer") or not STATE.get("latest_analysis"):
        raise HTTPException(status_code=400, detail="Run analysis first.")
    
    settlements = STATE["latest_analysis"]["raw_osm_settlements"]
    path_coords, geojson, impacted = STATE["dem_tracer"].trace_flow_path(
        req.lon, req.lat, settlements
    )

    return {
        "path_geojson": geojson,
        "path_coordinates": path_coords,
        "impacted_settlements": impacted,
        "attribution": "Produced using Copernicus WorldDEM-30 © DLR e.V. and Airbus Defence and Space GmbH"
    }


@app.post("/api/sitrep")
def get_sitrep(req: SitRepRequest):
    if not STATE.get("latest_analysis"):
        raise HTTPException(status_code=400, detail="No active analysis available.")
    
    text = STATE["sitrep_copilot"].generate_situation_report(
        STATE["latest_analysis"]["summary"], language=req.language
    )
    return {"language": req.language, "report_text": text}


@app.post("/api/copilot/chat")
def copilot_chat(req: ChatRequest):
    if not STATE.get("latest_analysis"):
        raise HTTPException(status_code=400, detail="No active analysis available.")
    
    reply = STATE["sitrep_copilot"].answer_query(
        req.query, STATE["latest_analysis"]["summary"]
    )
    return {"query": req.query, "reply": reply}


@app.get("/api/benchmark/emsr927")
def benchmark_emsr927():
    if not STATE.get("latest_analysis"):
        raise HTTPException(status_code=400, detail="No active analysis available.")

    analysis = STATE["latest_analysis"]
    benchmark = STATE["emsr927_validator"].evaluate_system_performance(
        analysis["summary"],
        pred_water_features=analysis.get("_water_features", []),
        pred_debris_features=analysis.get("_debris_features", []),
        bbox=analysis.get("bbox"),
    )
    return benchmark


@app.get("/api/evaluate")
def evaluate_model():
    """Run independent held-out test set evaluation and return scientific metrics."""
    try:
        from ai_model.evaluate import evaluate_test_set
        metrics = evaluate_test_set(batch_size=4)
        return metrics
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static assets if directory exists
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
