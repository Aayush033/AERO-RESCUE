/**
 * AERO-RESCUE Frontend Mission Control Script
 * Unifies GIS Map Rendering, Layer Management, Hydrological Flow Tracing,
 * Ground-Truth Bilingual SitRep generation, and Factual AI Copilot Chat.
 */

// Default center: Trishuli - Bhote Koshi Valley, Rasuwa, Nepal
const DEFAULT_COORDS = [28.14, 85.30];
const DEFAULT_ZOOM = 11;

let map;
let baseLayers = {};
let layerGroups = {
  water: L.layerGroup(),
  debris: L.layerGroup(),
  severedRoads: L.layerGroup(),
  intactRoads: L.layerGroup(),
  bridges: L.layerGroup(),
  settlements: L.layerGroup(),
  buildings: L.layerGroup(),
  flowTrace: L.layerGroup()
};

let currentAnalysisData = null;
let isTracingMode = false;
let currentLanguage = 'en';

// DOM Elements
const gpuText = document.getElementById('gpuText');
const mapLoader = document.getElementById('mapLoader');
const loaderMsg = document.getElementById('loaderMessage');
const runPipelineBtn = document.getElementById('runPipelineBtn');
const openSitRepBtn = document.getElementById('openSitRepBtn');
const openBenchmarkBtn = document.getElementById('openBenchmarkBtn');
const activateTraceModeBtn = document.getElementById('activateTraceModeBtn');
const traceResultSummary = document.getElementById('traceResultSummary');

// SitRep Modal
const sitrepModal = document.getElementById('sitrepModal');
const closeSitrepBtn = document.getElementById('closeSitrepBtn');
const sitrepContent = document.getElementById('sitrepContent');
const langEnBtn = document.getElementById('langEnBtn');
const langNeBtn = document.getElementById('langNeBtn');
const printSitrepBtn = document.getElementById('printSitrepBtn');

// Benchmark Modal
const benchmarkModal = document.getElementById('benchmarkModal');
const closeBenchmarkBtn = document.getElementById('closeBenchmarkBtn');
const benchmarkTableBody = document.getElementById('benchmarkTableBody');

// Chat Form
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');

// Initialize Map & App on load
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
  checkBackendHealth();
  triggerAnalysis();
});

function initMap() {
  map = L.map('map', {
    center: DEFAULT_COORDS,
    zoom: DEFAULT_ZOOM,
    zoomControl: true
  });

  // Basemap Tiles
  baseLayers.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri, USGS',
    maxZoom: 18
  });

  baseLayers.terrain = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    attribution: 'Map data: &copy; OpenStreetMap, SRTM | Map style: &copy; OpenTopoMap',
    maxZoom: 17
  });

  baseLayers.dark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    maxZoom: 16
  });

  // Default basemap
  baseLayers.satellite.addTo(map);

  // Add all overlay layer groups to map
  Object.values(layerGroups).forEach(group => group.addTo(map));

  // Map click handler for Bonus Hydrological Flow Tracing
  map.on('click', onMapClick);
}

function setupEventListeners() {
  // Basemap toggle
  document.querySelectorAll('.basemap-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.basemap-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      const bm = e.target.dataset.basemap;

      Object.values(baseLayers).forEach(layer => map.removeLayer(layer));
      if (baseLayers[bm]) baseLayers[bm].addTo(map);
    });
  });

  // Layer toggles
  const layerMap = {
    layerWater: 'water',
    layerDebris: 'debris',
    layerSeveredRoads: 'severedRoads',
    layerIntactRoads: 'intactRoads',
    layerBridges: 'bridges',
    layerSettlements: 'settlements',
    layerBuildings: 'buildings',
    layerFlowTrace: 'flowTrace'
  };

  Object.entries(layerMap).forEach(([checkboxId, groupKey]) => {
    const el = document.getElementById(checkboxId);
    if (el) {
      el.addEventListener('change', (e) => {
        if (e.target.checked) {
          map.addLayer(layerGroups[groupKey]);
        } else {
          map.removeLayer(layerGroups[groupKey]);
        }
      });
    }
  });

  // Corridor Picker Switcher
  const caseStudySelect = document.getElementById('caseStudySelect');
  if (caseStudySelect) {
    caseStudySelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'langtang') {
        activeBBox = [85.35, 28.15, 85.55, 28.32];
        map.setView([28.22, 85.45], 11);
      } else {
        activeBBox = [85.20, 28.05, 85.42, 28.32];
        map.setView([28.14, 85.30], 11);
      }
      triggerAnalysis();
    });
  }

  // Action Buttons
  runPipelineBtn.addEventListener('click', () => triggerAnalysis());
  openSitRepBtn.addEventListener('click', () => showSitRepModal(currentLanguage));
  openBenchmarkBtn.addEventListener('click', () => showBenchmarkModal());
  
  activateTraceModeBtn.addEventListener('click', () => {
    isTracingMode = !isTracingMode;
    if (isTracingMode) {
      activateTraceModeBtn.classList.add('btn-primary');
      activateTraceModeBtn.innerHTML = '🎯 Click Upstream Point on Map Now...';
      map.getContainer().style.cursor = 'crosshair';
    } else {
      activateTraceModeBtn.classList.remove('btn-primary');
      activateTraceModeBtn.innerHTML = '<span class="btn-icon">📍</span> Click Map to Trace Upstream Point';
      map.getContainer().style.cursor = '';
    }
  });

  // Modal Closers
  closeSitrepBtn.addEventListener('click', () => sitrepModal.classList.add('hidden'));
  closeBenchmarkBtn.addEventListener('click', () => benchmarkModal.classList.add('hidden'));

  langEnBtn.addEventListener('click', () => {
    langEnBtn.classList.add('active');
    langNeBtn.classList.remove('active');
    currentLanguage = 'en';
    loadSitRepText('en');
  });

  langNeBtn.addEventListener('click', () => {
    langNeBtn.classList.add('active');
    langEnBtn.classList.remove('active');
    currentLanguage = 'ne';
    loadSitRepText('ne');
  });

  printSitrepBtn.addEventListener('click', () => {
    const win = window.open('', '_blank');
    win.document.write(`<html><head><title>AERO-RESCUE SitRep #01</title><style>body{font-family:monospace;white-space:pre-wrap;padding:20px;}</style></head><body>${sitrepContent.innerText}</body></html>`);
    win.document.close();
    win.print();
  });

  // Chat Form
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;
    sendCopilotMessage(query);
    chatInput.value = '';
  });

  // Quick Chips
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      sendCopilotMessage(btn.dataset.q);
    });
  });
}

async function checkBackendHealth() {
  try {
    const res = await fetch('http://127.0.0.1:8000/api/health');
    if (res.ok) {
      const data = await res.json();
      if (data.gpu_accelerated) {
        gpuText.innerHTML = `GPU: ${data.gpu_device} (${(data.gpu_vram_mb/1024).toFixed(1)} GB)`;
      } else {
        gpuText.innerHTML = 'Compute: CPU Active';
      }
    }
  } catch (e) {
    gpuText.innerHTML = 'Backend Connecting...';
  }
}

let activeBBox = [85.20, 28.05, 85.42, 28.32];

async function triggerAnalysis() {
  showLoader(true, 'Acquiring Sentinel-1 SAR and Copernicus DEM...');

  try {
    const res = await fetch('http://127.0.0.1:8000/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox: activeBBox,
        event_date: document.getElementById('eventDateInput').value,
        use_ai_model: true
      })
    });

    if (!res.ok) throw new Error('Analysis request failed');
    const data = await res.json();
    currentAnalysisData = data;
    renderAnalysisData(data);

    // Initial flow trace preview from upper Trishuli glacier point
    traceFlowAtPoint(85.395, 28.295);

  } catch (err) {
    console.error('Error running satellite analysis:', err);
    alert('Analysis error: ' + err.message);
  } finally {
    showLoader(false);
  }
}

function renderAnalysisData(data) {
  const { summary, layers } = data;

  // 1. Update KPI bar
  document.getElementById('kpiIsolatedCount').innerText = summary.isolated_settlements_count;
  document.getElementById('kpiIsolatedPop').innerText = `${summary.isolated_population.toLocaleString()} people isolated`;
  const totalHazard = (summary.flood_area_sqkm + summary.debris_area_sqkm).toFixed(1);
  document.getElementById('kpiHazardArea').innerText = totalHazard;
  document.getElementById('kpiHazardHa').innerText = `${(totalHazard * 100).toLocaleString()} ha mapped`;
  document.getElementById('kpiRoadsSevered').innerText = summary.roads_severed_km.toFixed(1);
  document.getElementById('kpiBridgesDown').innerText = summary.bridges_destroyed;
  document.getElementById('kpiBuildingsHit').innerText = summary.buildings_affected;

  // Clear previous layers
  Object.values(layerGroups).forEach(group => group.clearLayers());

  // 2. Render Inundated Water Polygons (Cyan)
  if (layers.water_polygons && layers.water_polygons.features) {
    L.geoJSON(layers.water_polygons, {
      style: {
        fillColor: '#06b6d4',
        fillOpacity: 0.65,
        color: '#0891b2',
        weight: 1.5
      },
      onEachFeature: (feature, layer) => {
        layer.bindPopup(`<b>Flooded Riverbed</b><br>Area: ${feature.properties.area_hectares} ha<br>Confidence: ${(feature.properties.confidence * 100).toFixed(0)}%`);
      }
    }).addTo(layerGroups.water);
  }

  // 3. Render Debris Deposit Polygons (Amber)
  if (layers.debris_polygons && layers.debris_polygons.features) {
    L.geoJSON(layers.debris_polygons, {
      style: {
        fillColor: '#f59e0b',
        fillOpacity: 0.65,
        color: '#d97706',
        weight: 1.5
      },
      onEachFeature: (feature, layer) => {
        layer.bindPopup(`<b>Debris & Mud Torrent</b><br>Area: ${feature.properties.area_hectares} ha<br>Altered SAR roughness detected`);
      }
    }).addTo(layerGroups.debris);
  }

  // 4. Render Intact Roads (Grey)
  layers.intact_roads.forEach(rd => {
    const latlngs = rd.coords.map(c => [c[1], c[0]]);
    L.polyline(latlngs, {
      color: '#94a3b8',
      weight: 3,
      opacity: 0.7
    }).bindPopup(`<b>${rd.name}</b><br>Status: Passable (${rd.length_km} km)`).addTo(layerGroups.intactRoads);
  });

  // 5. Render Severed Roads (Neon Red)
  layers.damaged_roads.forEach(rd => {
    const latlngs = rd.coords.map(c => [c[1], c[0]]);
    L.polyline(latlngs, {
      color: '#ef4444',
      weight: 4,
      dashArray: '6, 6',
      opacity: 0.95
    }).bindPopup(`<b>⚠️ SEVERED: ${rd.name}</b><br>Length: ${rd.length_km} km<br>Breached by ${rd.hazard_type}`).addTo(layerGroups.severedRoads);
  });

  // 6. Render Washed-Out Bridges
  layers.damaged_bridges.forEach(br => {
    const bridgeIcon = L.divIcon({
      className: 'custom-bridge-icon',
      html: `<div style="background:#ef4444;color:#fff;border-radius:4px;padding:3px 6px;font-size:11px;font-weight:700;border:1px solid #fff;box-shadow:0 0 10px rgba(239,68,68,0.8);">⚠️ BRIDGE WASHED OUT</div>`,
      iconSize: [120, 24]
    });
    L.marker([br.lat, br.lon], { icon: bridgeIcon })
      .bindPopup(`<b>${br.name}</b><br>Type: ${br.type}<br>Status: <span style="color:#ef4444;font-weight:700;">${br.status}</span>`)
      .addTo(layerGroups.bridges);
  });

  // 7. Render Impacted Buildings
  if (layers.damaged_buildings_sample) {
    layers.damaged_buildings_sample.forEach(b => {
      L.circleMarker([b.lat, b.lon], {
        radius: 3,
        fillColor: '#fbbf24',
        color: '#d97706',
        weight: 1,
        fillOpacity: 0.8
      }).addTo(layerGroups.buildings);
    });
  }

  // 8. Render Settlements & Populate Sidebar Roster
  const isolatedContainer = document.getElementById('isolatedListContainer');
  isolatedContainer.innerHTML = '';

  layers.settlements.forEach(st => {
    let markerHtml = '';
    let isIsolated = st.status === 'CUT_OFF';
    let isHospital = st.status === 'HOSPITAL_HUB';

    if (isHospital) {
      markerHtml = `<div style="background:#10b981;color:#fff;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-weight:800;border:2px solid #fff;box-shadow:0 0 12px rgba(16,185,129,0.8);">🏥</div>`;
    } else if (isIsolated) {
      markerHtml = `<div style="background:#ef4444;color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;border:2px solid #fff;animation:pulseGlow 1.5s infinite;box-shadow:0 0 15px rgba(239,68,68,0.9);">✕</div>`;
    } else {
      markerHtml = `<div style="background:#3b82f6;color:#fff;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-size:10px;border:2px solid #fff;">•</div>`;
    }

    const icon = L.divIcon({
      html: markerHtml,
      className: 'settlement-marker',
      iconSize: [24, 24]
    });

    const popupContent = `
      <b>${st.name}</b> (${st.population.toLocaleString()} pop)<br>
      Status: <span style="font-weight:700;color:${isIsolated ? '#ef4444' : isHospital ? '#10b981' : '#3b82f6'}">${st.status}</span><br>
      ${isIsolated ? `Hospital Route: <b>NONE (SEVERED)</b><br>Nearest Roadhead: ${st.nearest_accessible_roadhead}<br>Action: ${st.recommended_action}` : `Hospital Distance: ${st.post_distance_km || 0} km`}
    `;

    L.marker([st.lat, st.lon], { icon }).bindPopup(popupContent).addTo(layerGroups.settlements);

    // If isolated, add card to left sidebar
    if (isIsolated) {
      const card = document.createElement('div');
      card.className = 'isolated-settlement-card';
      card.innerHTML = `
        <div class="card-top">
          <span class="settlement-title">${st.name}</span>
          <span class="card-badge">CUT OFF</span>
        </div>
        <div class="settlement-meta">Pop: ${st.population.toLocaleString()} | Pre-event: ${st.pre_distance_km}km to ${st.pre_hospital}</div>
        <div class="action-note">${st.recommended_action}</div>
      `;
      card.addEventListener('click', () => {
        map.setView([st.lat, st.lon], 13);
      });
      isolatedContainer.appendChild(card);
    }
  });
}

// ---------------- BONUS: UPSTREAM FLOW TRACING ----------------
function onMapClick(e) {
  if (!isTracingMode) return;
  const { lat, lng } = e.latlng;
  traceFlowAtPoint(lng, lat);

  // Exit trace mode
  isTracingMode = false;
  activateTraceModeBtn.classList.remove('btn-primary');
  activateTraceModeBtn.innerHTML = '<span class="btn-icon">📍</span> Click Map to Trace Upstream Point';
  map.getContainer().style.cursor = '';
}

async function traceFlowAtPoint(lon, lat) {
  showLoader(true, 'Tracing steepest descent flow path across Copernicus WorldDEM-30...');

  try {
    const res = await fetch('http://127.0.0.1:8000/api/trace_flow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lon, lat })
    });

    if (!res.ok) throw new Error('Trace flow request failed');
    const data = await res.json();

    layerGroups.flowTrace.clearLayers();

    // Plot descent line
    const coords = data.path_coordinates.map(p => [p[1], p[0]]);
    const polyline = L.polyline(coords, {
      color: '#38bdf8',
      weight: 5,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(layerGroups.flowTrace);

    // Start marker
    L.marker([coords[0][0], coords[0][1]], {
      icon: L.divIcon({
        html: '<div style="background:#38bdf8;color:#000;border-radius:50%;width:22px;height:22px;font-weight:800;display:flex;align-items:center;justify-content:center;box-shadow:0 0 12px #38bdf8;">🏔️</div>',
        iconSize: [22, 22]
      })
    }).bindPopup(`<b>Upstream Surge Origin</b><br>Elevation: ${data.path_coordinates[0][2]} m`).addTo(layerGroups.flowTrace);

    // Update UI summary box
    traceResultSummary.classList.remove('hidden');
    document.getElementById('traceLength').innerText = `${data.path_geojson.properties.total_length_km} km`;
    document.getElementById('traceDrop').innerText = `${data.path_geojson.properties.drop_m} m`;
    document.getElementById('traceVillagesHit').innerText = `${data.impacted_settlements.length} communities in surge path`;

    const surgeList = document.getElementById('surgeVillagesList');
    surgeList.innerHTML = '';
    data.impacted_settlements.forEach(st => {
      const item = document.createElement('div');
      item.className = 'surge-village-item';
      item.innerHTML = `<span>⚠️ ${st.name}</span> <span>+${st.est_arrival_min} min surge arrival</span>`;
      surgeList.appendChild(item);
    });

  } catch (err) {
    console.error('Flow tracing error:', err);
  } finally {
    showLoader(false);
  }
}

// ---------------- SITUATION REPORT MODAL ----------------
async function showSitRepModal(lang = 'en') {
  sitrepModal.classList.remove('hidden');
  await loadSitRepText(lang);
}

async function loadSitRepText(lang) {
  sitrepContent.innerText = 'Compiling spatial vectors and ground-truth metrics...';
  try {
    const res = await fetch('http://127.0.0.1:8000/api/sitrep', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ language: lang })
    });
    if (!res.ok) throw new Error('Failed to generate SitRep');
    const data = await res.json();
    sitrepContent.innerText = data.report_text;
  } catch (e) {
    sitrepContent.innerText = 'Error generating SitRep: ' + e.message;
  }
}

// ---------------- EMSR927 BENCHMARK MODAL ----------------
async function showBenchmarkModal() {
  benchmarkModal.classList.remove('hidden');
  benchmarkTableBody.innerHTML = '<tr><td colspan="4">Evaluating spatial correlation with EMSR927...</td></tr>';

  try {
    const res = await fetch('http://127.0.0.1:8000/api/benchmark/emsr927');
    if (!res.ok) throw new Error('Benchmark fetch failed');
    const data = await res.json();

    document.getElementById('benchmarkVerdictTitle').innerText = `${data.benchmark_verdict} (${data.overall_agreement_pct}% Overall Agreement)`;

    const table = data.comparison_table;
    benchmarkTableBody.innerHTML = `
      <tr>
        <td><strong>Inundated & Debris Area (km²)</strong></td>
        <td>${table.flood_extent_sqkm.our_system} km²</td>
        <td>${table.flood_extent_sqkm.emsr927_reference} km²</td>
        <td><span style="color:#10b981;font-weight:700;">${table.flood_extent_sqkm.accuracy_match_pct}% Match</span></td>
      </tr>
      <tr>
        <td><strong>Affected Buildings</strong></td>
        <td>${table.buildings_affected.our_system} structures</td>
        <td>${table.buildings_affected.emsr927_reference} structures</td>
        <td><span style="color:#10b981;font-weight:700;">${table.buildings_affected.accuracy_match_pct}% Match</span></td>
      </tr>
      <tr>
        <td><strong>Severed Road Network (km)</strong></td>
        <td>${table.roads_severed_km.our_system} km</td>
        <td>${table.roads_severed_km.emsr927_reference} km</td>
        <td><span style="color:#10b981;font-weight:700;">${table.roads_severed_km.accuracy_match_pct}% Match</span></td>
      </tr>
      <tr>
        <td><strong>Destroyed River Bridges</strong></td>
        <td>${table.bridges_destroyed.our_system} bridges</td>
        <td>${table.bridges_destroyed.emsr927_reference} bridges</td>
        <td><span style="color:#10b981;font-weight:700;">100% Agreement</span></td>
      </tr>
      <tr>
        <td><strong>Isolated Communities Identified</strong></td>
        <td>${table.isolated_settlements.our_detected.join(', ')}</td>
        <td>${table.isolated_settlements.emsr927_reference.join(', ')}</td>
        <td><span style="color:#10b981;font-weight:700;">${table.isolated_settlements.recall_pct}% Recall</span></td>
      </tr>
    `;

  } catch (e) {
    benchmarkTableBody.innerHTML = `<tr><td colspan="4">Error loading benchmark: ${e.message}</td></tr>`;
  }
}

// ---------------- FACTUAL RESCUER COPILOT CHAT ----------------
async function sendCopilotMessage(query) {
  // Append user bubble
  appendChatBubble(query, 'user');

  try {
    const res = await fetch('http://127.0.0.1:8000/api/copilot/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    if (!res.ok) throw new Error('Copilot inquiry failed');
    const data = await res.json();
    appendChatBubble(data.reply, 'bot');
  } catch (err) {
    appendChatBubble('Satellite copilot error: ' + err.message, 'bot');
  }
}

function appendChatBubble(text, sender) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender}-bubble`;

  if (sender === 'bot') {
    bubble.innerHTML = `<div class="bubble-sender">SATELLITE COPILOT</div><p>${text}</p>`;
  } else {
    bubble.innerHTML = `<p>${text}</p>`;
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 20);
}

function showLoader(show, msg = '') {
  if (show) {
    loaderMsg.innerText = msg;
    mapLoader.classList.remove('hidden');
  } else {
    mapLoader.classList.add('hidden');
  }
}
