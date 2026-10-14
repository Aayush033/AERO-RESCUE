"""
Graph Network Routing & Cut-off Settlement Analyzer
=====================================================
Evaluates topological isolation of villages from healthcare and administrative
hubs due to flood inundation, mudslides, and washed-out bridges.

Key logic:
- Builds TWO road-network graphs: pre-event (all roads) and post-event (damaged roads removed)
- Settlement nodes are snapped to nearest road vertex within 10 km
- A village is CUT_OFF if it had a path to a hospital pre-event but has none post-event
- A village is REMOTE_COMMUNITY if it had no road connection even before the disaster
- A village with a detour >30% longer is flagged as DETOUR_REQUIRED
"""

import math
import logging
import networkx as nx

log = logging.getLogger(__name__)


class SettlementIsolationAnalyzer:
    def __init__(self, osm_data, damaged_roads, damaged_bridges):
        self.osm_data = osm_data
        self.damaged_roads = damaged_roads
        self.damaged_bridges = damaged_bridges
        self.pre_graph  = self._build_graph(filter_damaged=False)
        self.post_graph = self._build_graph(filter_damaged=True)

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self, filter_damaged: bool) -> nx.Graph:
        G = nx.Graph()

        damaged_road_ids   = {r["id"] for r in self.damaged_roads}   if filter_damaged else set()
        damaged_bridge_ids = {b["id"] for b in self.damaged_bridges} if filter_damaged else set()

        # Add settlement nodes
        for s in self.osm_data.get("settlements", []):
            G.add_node(s["id"], **s)

        # Add road edges (skip if damaged in post-event graph)
        for r in self.osm_data.get("roads", []):
            if filter_damaged and r["id"] in damaged_road_ids:
                continue
            # A road that is also a bridge: skip if that bridge is damaged
            if filter_damaged and r.get("is_bridge") and r.get("osm_id") and \
                    f"br_{r['osm_id']}" in damaged_bridge_ids:
                continue
            G.add_edge(
                r["u"], r["v"],
                id=r["id"],
                name=r["name"],
                length_km=r["length_km"],
                highway=r["highway"]
            )

        # If a bridge crossing is destroyed, remove both directions of the road segment
        if filter_damaged:
            for br in self.damaged_bridges:
                # Remove any edge whose midpoint is within 300 m of the bridge
                b_lon, b_lat = br["lon"], br["lat"]
                to_remove = []
                for u, v, data in list(G.edges(data=True)):
                    edge_id = data.get("id", "")
                    for r in self.osm_data.get("roads", []):
                        if r["id"] == edge_id:
                            coords = r.get("coords", [])
                            if coords:
                                mid = coords[len(coords) // 2]
                                dist_m = math.hypot(
                                    (mid[0] - b_lon) * 111000 * math.cos(math.radians(b_lat)),
                                    (mid[1] - b_lat) * 111000
                                )
                                if dist_m < 300:
                                    to_remove.append((u, v))
                for u, v in to_remove:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)

        # Snap each settlement node to the nearest road-network vertex
        road_vertices = {}  # vertex_id -> (lon, lat)
        for r in self.osm_data.get("roads", []):
            coords = r.get("coords", [])
            u, v = r.get("u"), r.get("v")
            if coords and len(coords) >= 2:
                if u and u not in road_vertices:
                    road_vertices[u] = (coords[0][0], coords[0][1])
                if v and v not in road_vertices:
                    road_vertices[v] = (coords[-1][0], coords[-1][1])

        for s in self.osm_data.get("settlements", []):
            sid = s["id"]
            if sid in road_vertices:
                continue  # Already a direct road network vertex

            s_lon, s_lat = s["lon"], s["lat"]
            best_pt, min_dist = None, float("inf")

            for pt, (p_lon, p_lat) in road_vertices.items():
                d = math.hypot(
                    (s_lon - p_lon) * 111.0 * math.cos(math.radians(s_lat)),
                    (s_lat - p_lat) * 111.0
                )
                if d < min_dist:
                    min_dist, best_pt = d, pt

            # Snap threshold: 10 km (covers mountain hamlets accessing the valley spine)
            if best_pt and min_dist <= 10.0:
                if not G.has_edge(sid, best_pt):
                    G.add_edge(sid, best_pt, length_km=round(min_dist, 2),
                               name="Local Access Track")

        return G

    # ── Isolation Analysis ────────────────────────────────────────────────────

    def analyze_isolation(self) -> dict:
        """
        Compares pre-event vs post-event reachability of every settlement to
        the nearest functioning hospital hub.
        """
        settlements  = self.osm_data.get("settlements", [])
        hospitals    = [s for s in settlements if s.get("has_hospital")]
        hospital_ids = [h["id"] for h in hospitals]

        if not hospital_ids:
            log.warning("[ISOLATION] No hospital nodes found – designation missing!")

        settlements_report = []
        total_isolated_pop = 0

        for s in settlements:
            sid = s["id"]

            # Hospital hubs are their own anchor
            if sid in hospital_ids:
                settlements_report.append({
                    "id": sid,
                    "name": s["name"],
                    "status": "HOSPITAL_HUB",
                    "population": s["population"],
                    "nearest_hospital": s["name"],
                    "pre_distance_km": 0.0,
                    "post_distance_km": 0.0,
                    "detour_km": 0.0,
                    "pre_hospital": s["name"],
                    "lat": s["lat"], "lon": s["lon"],
                })
                continue

            pre_dist, pre_hosp   = self._nearest_hospital(self.pre_graph,  sid, hospitals)
            post_dist, post_hosp = self._nearest_hospital(self.post_graph, sid, hospitals)

            pre_km  = round(pre_dist,  1) if pre_dist  != float("inf") else None
            post_km = round(post_dist, 1) if post_dist != float("inf") else None
            detour  = round(post_dist - pre_dist, 1) \
                if (pre_dist != float("inf") and post_dist != float("inf")) else None

            base = {
                "id": sid,
                "name": s["name"],
                "population": s["population"],
                "pre_hospital": pre_hosp or "Dhunche (District Hospital Hub)",
                "pre_distance_km": pre_km or 0.0,
                "post_distance_km": post_km,
                "detour_km": detour,
                "lat": s["lat"], "lon": s["lon"],
            }

            if post_dist == float("inf"):
                if pre_dist != float("inf"):
                    # Had road access → now severed by flood/bridge failure
                    total_isolated_pop += s["population"]
                    settlements_report.append({
                        **base,
                        "status": "CUT_OFF",
                        "severity": "CRITICAL",
                        "post_hospital": "NONE (SEVERED)",
                        "nearest_accessible_roadhead": self._nearest_roadhead(s),
                        "recommended_action": self._recommend_action(s, pre_km),
                    })
                else:
                    # No road even before disaster
                    settlements_report.append({
                        **base,
                        "status": "REMOTE_COMMUNITY",
                        "severity": "MONITOR",
                        "post_hospital": "Footpath Only",
                        "nearest_accessible_roadhead": "Ridge Trail Access",
                        "recommended_action": "Helicopter air-drop / satellite surveillance",
                    })

            elif detour is not None and detour > 0 and post_dist > pre_dist * 1.3:
                settlements_report.append({
                    **base,
                    "status": "DETOUR_REQUIRED",
                    "severity": "MODERATE",
                    "post_hospital": post_hosp,
                    "recommended_action": "High-clearance 4WD convoy; consider alternate route",
                })

            else:
                settlements_report.append({
                    **base,
                    "status": "ACCESSIBLE",
                    "severity": "LOW",
                    "post_hospital": post_hosp,
                    "recommended_action": "Normal emergency vehicle transit",
                })

        cut_off = [s for s in settlements_report if s["status"] == "CUT_OFF"]
        log.info(
            f"[ISOLATION] {len(cut_off)} cut-off, "
            f"{sum(1 for s in settlements_report if s['status']=='DETOUR_REQUIRED')} detour-required, "
            f"{sum(1 for s in settlements_report if s['status']=='REMOTE_COMMUNITY')} remote communities."
        )

        return {
            "settlements_status": settlements_report,
            "total_isolated_settlements": len(cut_off),
            "total_isolated_population": total_isolated_pop,
            "isolated_names": [s["name"] for s in cut_off],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _nearest_hospital(self, G: nx.Graph, sid: str, hospitals: list):
        """Returns (distance_km, hospital_name) to nearest reachable hospital."""
        best_dist, best_name = float("inf"), None
        for h in hospitals:
            hid = h["id"]
            if G.has_node(sid) and G.has_node(hid) and nx.has_path(G, sid, hid):
                try:
                    d = nx.shortest_path_length(G, sid, hid, weight="length_km")
                    if d < best_dist:
                        best_dist, best_name = d, h["name"]
                except nx.NetworkXError:
                    pass
        return best_dist, best_name

    def _nearest_roadhead(self, settlement: dict) -> str:
        """Returns a descriptive nearest accessible roadhead label."""
        lat, lon = settlement["lat"], settlement["lon"]
        if lat > 28.20:
            return "Timure Junction (road cutoff)"
        elif lat > 28.15:
            return "Syabrubesi Gorge Breach"
        elif lon > 85.35:
            return "Bhote Koshi River Road terminus"
        else:
            return "Dhunche–Syabrubesi Gorge Breach"

    def _recommend_action(self, settlement: dict, pre_km: float) -> str:
        """Returns a prioritised recommended action for an isolated settlement."""
        pop = settlement.get("population", 0)
        if pop > 3000:
            return "PRIORITY: Air evacuation + emergency relief convoy staging"
        elif pop > 1000:
            return "Air evacuation helicopter + suspension-bridge repair assessment"
        else:
            return "Air-drop essential supplies; foot suspension-bridge priority"
