"""
Sentinel-1 SAR Change Detection & Flood Processor
Algorithms:
- Calibrated Log-Ratio backscatter change detection (Delta sigma0 in dB)
- Lee / Adaptive Speckle Filtering
- Topographic Slope masking using Copernicus WorldDEM-30
- Morphological cleaning and vectorization
"""

import numpy as np
from scipy.ndimage import uniform_filter


class SARProcessor:
    def __init__(self, water_threshold_db=-3.8, debris_threshold_db=3.2, max_slope_deg=18.0):
        self.water_threshold_db = water_threshold_db
        self.debris_threshold_db = debris_threshold_db
        self.max_slope_deg = max_slope_deg

    def apply_lee_filter(self, img, size=5):
        """
        Lee speckle filter for SAR intensity images
        Reduces radar granular noise while preserving edges and river corridors
        """
        mean = uniform_filter(img, size)
        sq_mean = uniform_filter(img**2, size)
        variance = np.maximum(sq_mean - mean**2, 1e-8)
        overall_variance = np.var(img) + 1e-8

        weight = variance / (variance + overall_variance)
        filtered = mean + weight * (img - mean)
        return filtered

    def detect_changes(self, pre_vv_norm, post_vv_norm, slope_deg):
        """
        Performs radar backscatter change detection:
        Inputs: pre_vv_norm, post_vv_norm in range [0, 1] (representing [-35, 0] dB)
        Returns:
          - change_db: Difference map in dB
          - flood_mask: Boolean water inundation mask
          - debris_mask: Boolean sediment / debris deposit mask
        """
        # Convert back from [0, 1] to physical dB
        pre_vv_db = pre_vv_norm * 35.0 - 35.0
        post_vv_db = post_vv_norm * 35.0 - 35.0

        # Filter speckle
        pre_filtered = self.apply_lee_filter(pre_vv_db, size=5)
        post_filtered = self.apply_lee_filter(post_vv_db, size=5)

        # Delta sigma0 in dB
        change_db = post_filtered - pre_filtered

        # Topographic slope constraint: standing water and thick debris cannot remain on steep Himalayan cliff faces
        valid_terrain = slope_deg <= self.max_slope_deg

        # Flood detection: specular drop in backscatter (surface becomes mirror-like smooth water)
        flood_mask = (change_db <= self.water_threshold_db) & (post_filtered < -18.0) & valid_terrain

        # Debris flow detection: rough deposit of mud, boulders, and churned earth raises roughness
        debris_mask = (change_db >= self.debris_threshold_db) & (post_filtered > -14.0) & valid_terrain

        # Mutual exclusivity
        debris_mask = debris_mask & (~flood_mask)

        return change_db, flood_mask, debris_mask
