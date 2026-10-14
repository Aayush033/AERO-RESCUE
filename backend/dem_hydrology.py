"""
Copernicus WorldDEM-30 Hydrological Valley Flow Tracing (Bonus Task)
Implements:
- D8 Steepest Descent Elevation Flow Path tracing from any upstream coordinate
- Hydraulic valley corridor width estimation
- Downstream settlement impact sequence ordering
Attribution:
Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018
provided under COPERNICUS by the European Union and ESA; all rights reserved.
"""

import math
import numpy as np


class DEMHydrologyTracer:
    def __init__(self, elevation_raster, bbox):
        """
        elevation_raster: (H, W) numpy array of elevation in meters
        bbox: [min_lon, min_lat, max_lon, max_lat]
        """
        self.dem = elevation_raster
        self.bbox = bbox
        self.H, self.W = elevation_raster.shape
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bbox

    def coord_to_cell(self, lon, lat):
        x = int(((lon - self.min_lon) / (self.max_lon - self.min_lon)) * (self.W - 1))
        y = int(((self.max_lat - lat) / (self.max_lat - self.min_lat)) * (self.H - 1))
        x = max(0, min(self.W - 1, x))
        y = max(0, min(self.H - 1, y))
        return y, x

    def cell_to_coord(self, y, x):
        lon = self.min_lon + (x / (self.W - 1)) * (self.max_lon - self.min_lon)
        lat = self.max_lat - (y / (self.H - 1)) * (self.max_lat - self.min_lat)
        return lon, lat

    def trace_flow_path(self, start_lon, start_lat, settlements, max_steps=500):
        """
        Traces D8 steepest gradient descent path from upstream point down to valley outlet.
        Returns:
          - flow_path: list of [lon, lat, elevation]
          - path_geojson: GeoJSON LineString
          - settlements_impacted: list of settlements encountered in downstream order
        """
        cur_y, cur_x = self.coord_to_cell(start_lon, start_lat)
        visited = set()
        path_cells = [(cur_y, cur_x)]
        visited.add((cur_y, cur_x))

        d8_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for _ in range(max_steps):
            cur_elev = self.dem[cur_y, cur_x]
            best_drop = 0.0
            next_step = None

            for dy, dx in d8_offsets:
                ny, nx = cur_y + dy, cur_x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    if (ny, nx) not in visited:
                        dist = math.sqrt(dy**2 + dx**2)
                        drop = (cur_elev - self.dem[ny, nx]) / dist
                        if drop > best_drop:
                            best_drop = drop
                            next_step = (ny, nx)

            if next_step is None or best_drop <= 0:
                # River pit or reached lowest point; step toward local valley minimum
                grad_y, grad_x = np.gradient(self.dem)
                step_y = int(np.sign(grad_y[cur_y, cur_x]))
                step_x = int(-np.sign(grad_x[cur_y, cur_x]))
                ny = max(0, min(self.H - 1, cur_y + (1 if cur_y < self.H - 1 else 0)))
                nx = max(0, min(self.W - 1, cur_x + step_x))
                if (ny, nx) in visited:
                    break
                next_step = (ny, nx)

            cur_y, cur_x = next_step
            visited.add((cur_y, cur_x))
            path_cells.append((cur_y, cur_x))

            # Stop if reached bottom edge of tile
            if cur_y >= self.H - 2:
                break

        # Convert cells to geographic coordinates
        path_coords = []
        total_length_km = 0.0
        prev_pt = None

        for y, x in path_cells:
            lon, lat = self.cell_to_coord(y, x)
            elev = float(self.dem[y, x])
            pt = [round(lon, 5), round(lat, 5), round(elev, 1)]
            path_coords.append(pt)

            if prev_pt:
                # Haversine distance
                dlat = (lat - prev_pt[1]) * 111.0
                dlon = (lon - prev_pt[0]) * 111.0 * math.cos(math.radians(lat))
                total_length_km += math.sqrt(dlat**2 + dlon**2)
            prev_pt = (lon, lat)

        # Check settlements in proximity to flow path corridor
        impacted_settlements = []
        for st in settlements:
            s_lon = st["lon"]
            s_lat = st["lat"]
            min_dist_km = 999.0
            dist_downstream = 0.0
            accum_dist = 0.0

            for i in range(1, len(path_coords)):
                p1 = path_coords[i - 1]
                p2 = path_coords[i]
                dlat = (p2[1] - p1[1]) * 111.0
                dlon = (p2[0] - p1[0]) * 111.0 * math.cos(math.radians(p1[1]))
                seg_len = math.sqrt(dlat**2 + dlon**2)
                accum_dist += seg_len

                # Distance from settlement to segment
                c_dlat = (s_lat - p1[1]) * 111.0
                c_dlon = (s_lon - p1[0]) * 111.0 * math.cos(math.radians(s_lat))
                d = math.sqrt(c_dlat**2 + c_dlon**2)
                if d < min_dist_km:
                    min_dist_km = d
                    dist_downstream = accum_dist

            # If within 1.2 km of valley flow center, it is directly along the flood path
            if min_dist_km <= 1.2:
                # Approximate surge arrival time assuming 28 km/h debris flood velocity
                est_arrival_min = round((dist_downstream / 28.0) * 60, 1)
                impacted_settlements.append({
                    "name": st["name"],
                    "population": st.get("population", 500),
                    "distance_downstream_km": round(dist_downstream, 2),
                    "distance_from_river_m": round(min_dist_km * 1000, 0),
                    "est_arrival_min": est_arrival_min,
                    "risk_level": "CRITICAL" if min_dist_km < 0.4 else "HIGH"
                })

        impacted_settlements.sort(key=lambda s: s["distance_downstream_km"])

        geojson = {
            "type": "Feature",
            "properties": {
                "name": "Upstream Valley Flood Descent Path",
                "total_length_km": round(total_length_km, 2),
                "start_elevation_m": path_coords[0][2],
                "end_elevation_m": path_coords[-1][2],
                "drop_m": round(path_coords[0][2] - path_coords[-1][2], 1),
                "settlements_in_path": len(impacted_settlements)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[p[0], p[1]] for p in path_coords]
            }
        }

        return path_coords, geojson, impacted_settlements
