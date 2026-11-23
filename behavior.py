"""
===============================================================================
Fish_AI_Project - Behavior Analysis Module (behavior.py)
===============================================================================
Calculates fish swimming metrics: speed, surface time ratio, feeding zone
presence, movement intensity, and composite behavior score. Integrates environmental
modifiers from water_quality.py to simulate hypoxia gasping, temperature
slowdowns, and stress-induced behavioral changes.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BehaviorAnalyzer:
    """Analyzes raw fish tracking trajectories and applies water quality stress coupling."""

    def __init__(self) -> None:
        """Initialize behavior analyzer."""
        pass

    def analyze_tracks(
        self,
        tracks: List[Dict[str, Any]],
        water_quality_modifiers: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute spatial and kinematic behavior metrics adjusted for water quality.

        Args:
            tracks: List of active track dictionaries from tracker.py.
            water_quality_modifiers: Environmental modifiers dict from water_quality.py.

        Returns:
            Dict containing average speed, surface ratio, feeding zone presence,
            movement intensity score, and environmental stress flags.
        """
        if not tracks:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "surface_time_ratio": 0.0,
                "feeding_zone_presence_ratio": 0.0,
                "movement_intensity": 0.0,
                "behavior_score": 50.0,
                "environmental_stress_applied": False,
                "stress_description": "No active fish detected.",
                "raw_metrics": {"speed": 0.0, "surface": 0.0, "zone": 0.0}
            }

        # Raw kinematics from tracked fish
        speeds = [t.get("speed", 0.0) for t in tracks]
        raw_avg_speed = float(np.mean(speeds))
        raw_max_speed = float(np.max(speeds))

        total_near_surface = sum(t.get("time_near_surface", 0.0) for t in tracks)
        total_in_feeding_zone = sum(t.get("time_in_feeding_zone", 0.0) for t in tracks)

        # Estimate tracking duration window
        total_time_tracked = max(1.0, sum(len(t.get("history", [])) / 30.0 for t in tracks))

        raw_surface_ratio = min(1.0, total_near_surface / max(1.0, total_time_tracked * len(tracks)))
        raw_feeding_zone_ratio = min(1.0, total_in_feeding_zone / max(1.0, total_time_tracked * len(tracks)))

        # Extract water quality modifiers if available
        if water_quality_modifiers is None:
            speed_mult = 1.0
            surface_boost = 0.0
            feeding_dampener = 1.0
            reasons = []
        else:
            speed_mult = water_quality_modifiers.get("speed_modifier", 1.0)
            surface_boost = water_quality_modifiers.get("surface_boost", 0.0)
            feeding_dampener = water_quality_modifiers.get("feeding_dampener", 1.0)
            reasons = water_quality_modifiers.get("reasons", [])

        # Apply environmental stress coupling
        adjusted_avg_speed = round(max(0.0, raw_avg_speed * speed_mult), 2)
        adjusted_surface_ratio = round(min(1.0, max(0.0, raw_surface_ratio + surface_boost)), 2)
        adjusted_feeding_zone_ratio = round(min(1.0, max(0.0, raw_feeding_zone_ratio * feeding_dampener)), 2)

        # Calculate Movement Intensity (0 to 100)
        movement_intensity = min(100.0, (adjusted_avg_speed / 120.0) * 100.0)

        # Calculate composite Behavior Score (0 to 100)
        # High score indicates active, healthy, feeding-ready swimming
        behavior_score = (
            (movement_intensity * 0.40) +
            (adjusted_feeding_zone_ratio * 100.0 * 0.40) +
            ((1.0 - abs(0.20 - adjusted_surface_ratio)) * 100.0 * 0.20)
        )
        behavior_score = round(max(0.0, min(100.0, behavior_score)), 1)

        has_stress = len(reasons) > 0
        stress_desc = "; ".join(reasons) if has_stress else "Nominal environment. Normal fish movement."

        return {
            "avg_speed": adjusted_avg_speed,
            "max_speed": round(raw_max_speed * speed_mult, 2),
            "surface_time_ratio": adjusted_surface_ratio,
            "feeding_zone_presence_ratio": adjusted_feeding_zone_ratio,
            "movement_intensity": round(movement_intensity, 1),
            "behavior_score": behavior_score,
            "environmental_stress_applied": has_stress,
            "stress_description": stress_desc,
            "raw_metrics": {
                "raw_avg_speed": round(raw_avg_speed, 2),
                "raw_surface_ratio": round(raw_surface_ratio, 2),
                "raw_feeding_zone_ratio": round(raw_feeding_zone_ratio, 2)
            }
        }
