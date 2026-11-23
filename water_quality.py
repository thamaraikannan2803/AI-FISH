"""
===============================================================================
Fish_AI_Project - Water Quality Module (water_quality.py)
===============================================================================
Monitors and simulates key water parameters: pH, Temperature, Turbidity,
and Dissolved Oxygen (DO). Computes a overall Water Stress Index and exports
environmental behavior modifiers used by behavior.py to adjust swimming
speed, surface gasping ratio, and feeding zone presence.
"""

import time
import random
import logging
from typing import Dict, Any, List, Tuple
from config import WATER_QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)


class WaterQualityMonitor:
    """Water Quality Telemetry Engine & Environmental Stress Simulator."""

    def __init__(self) -> None:
        """Initialize parameters within nominal ranges."""
        self.ph = 7.4
        self.temp_c = 26.5
        self.turbidity_ntu = 4.2
        self.dissolved_oxygen_mg_l = 7.2

        # Manual overrides for simulation testing
        self.override_active = False
        self.override_params: Dict[str, float] = {}

    def get_telemetry(self) -> Dict[str, Any]:
        """Fetch current water quality values with realistic natural fluctuations."""
        if not self.override_active:
            # Simulate subtle drift
            self.ph = max(6.0, min(9.0, round(self.ph + random.uniform(-0.03, 0.03), 2)))
            self.temp_c = max(18.0, min(34.0, round(self.temp_c + random.uniform(-0.08, 0.08), 1)))
            self.turbidity_ntu = max(0.5, min(25.0, round(self.turbidity_ntu + random.uniform(-0.1, 0.1), 1)))
            self.dissolved_oxygen_mg_l = max(2.0, min(11.0, round(self.dissolved_oxygen_mg_l + random.uniform(-0.05, 0.05), 2)))
        else:
            self.ph = self.override_params.get("ph", self.ph)
            self.temp_c = self.override_params.get("temp_c", self.temp_c)
            self.turbidity_ntu = self.override_params.get("turbidity_ntu", self.turbidity_ntu)
            self.dissolved_oxygen_mg_l = self.override_params.get("dissolved_oxygen_mg_l", self.dissolved_oxygen_mg_l)

        is_safe, alerts = self.evaluate_safety()
        stress_index, stress_factors = self.calculate_water_stress_index()
        behavior_modifiers = self.get_environmental_behavior_modifiers()

        return {
            "ph": self.ph,
            "temp_c": self.temp_c,
            "turbidity_ntu": self.turbidity_ntu,
            "dissolved_oxygen_mg_l": self.dissolved_oxygen_mg_l,
            "is_safe": is_safe,
            "alerts": alerts,
            "stress_index": round(stress_index, 2),
            "stress_factors": stress_factors,
            "behavior_modifiers": behavior_modifiers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def set_simulation_override(
        self,
        ph: float = 7.4,
        temp_c: float = 26.5,
        turbidity_ntu: float = 4.2,
        dissolved_oxygen_mg_l: float = 7.2
    ) -> None:
        """Set manual simulation sliders to test environmental stress effects on fish."""
        self.override_active = True
        self.override_params = {
            "ph": ph,
            "temp_c": temp_c,
            "turbidity_ntu": turbidity_ntu,
            "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l
        }

    def clear_simulation_override(self) -> None:
        """Clear manual sliders and return to natural drift."""
        self.override_active = False

    def evaluate_safety(self) -> Tuple[bool, List[str]]:
        """Evaluate if water quality metrics satisfy safe aquaculture boundaries."""
        limits = WATER_QUALITY_THRESHOLDS
        alerts = []

        if not (limits["ph"]["min"] <= self.ph <= limits["ph"]["max"]):
            alerts.append(f"pH out of safe range ({self.ph:.1f} | limit {limits['ph']['min']}-{limits['ph']['max']})")

        if not (limits["temp_c"]["min"] <= self.temp_c <= limits["temp_c"]["max"]):
            alerts.append(f"Temperature warning ({self.temp_c:.1f}°C | limit {limits['temp_c']['min']}-{limits['temp_c']['max']}°C)")

        if self.turbidity_ntu > limits["turbidity_ntu"]["max"]:
            alerts.append(f"High Turbidity ({self.turbidity_ntu:.1f} NTU | limit < {limits['turbidity_ntu']['max']} NTU)")

        if self.dissolved_oxygen_mg_l < limits["dissolved_oxygen_mg_l"]["min"]:
            alerts.append(f"CRITICAL: Low Dissolved Oxygen ({self.dissolved_oxygen_mg_l:.2f} mg/L | min {limits['dissolved_oxygen_mg_l']['min']} mg/L)")

        is_safe = len(alerts) == 0
        return is_safe, alerts

    def calculate_water_stress_index(self) -> Tuple[float, Dict[str, bool]]:
        """Compute continuous environmental stress index (0.0 = pristine, 1.0 = lethal)."""
        stress_score = 0.0
        factors = {
            "hypoxia": False,
            "thermal_stress": False,
            "ph_irritation": False,
            "turbidity_obscuration": False
        }

        # Dissolved Oxygen stress (highest priority factor)
        if self.dissolved_oxygen_mg_l < 5.0:
            factors["hypoxia"] = True
            # Linear scaling from DO 5.0 (0.0 stress) to 2.0 (1.0 severe stress)
            do_stress = (5.0 - self.dissolved_oxygen_mg_l) / 3.0
            stress_score += max(0.0, min(0.60, do_stress * 0.60))

        # Temperature stress
        if self.temp_c < 22.0 or self.temp_c > 30.0:
            factors["thermal_stress"] = True
            temp_diff = min(abs(self.temp_c - 22.0), abs(self.temp_c - 30.0))
            stress_score += max(0.0, min(0.25, temp_diff * 0.05))

        # pH stress
        if self.ph < 6.5 or self.ph > 8.5:
            factors["ph_irritation"] = True
            ph_diff = min(abs(self.ph - 6.5), abs(self.ph - 8.5))
            stress_score += max(0.0, min(0.25, ph_diff * 0.15))

        # Turbidity stress
        if self.turbidity_ntu > 15.0:
            factors["turbidity_obscuration"] = True
            stress_score += min(0.15, (self.turbidity_ntu - 15.0) * 0.01)

        total_stress = min(1.0, stress_score)
        return total_stress, factors

    def get_environmental_behavior_modifiers(self) -> Dict[str, Any]:
        """Generate environmental behavior adjustments for behavior.py and hunger_prediction.py.

        Returns:
            Dict containing speed_modifier, surface_boost, feeding_dampener, and descriptions.
        """
        stress_index, factors = self.calculate_water_stress_index()

        speed_modifier = 1.0
        surface_boost = 0.0
        feeding_dampener = 1.0
        reasons = []

        # Hypoxia effect: Low DO reduces swimming speed & drastically increases surface gasping time
        if factors["hypoxia"]:
            do_deficit = 5.0 - self.dissolved_oxygen_mg_l
            speed_modifier *= max(0.35, 1.0 - (do_deficit * 0.18))
            surface_boost += min(0.75, do_deficit * 0.25)
            feeding_dampener *= max(0.1, 1.0 - (do_deficit * 0.25))
            reasons.append("Hypoxia lethargy & surface gasping behavior")

        # Thermal stress effect: Extreme temperatures cause sluggish movement or erratic stress
        if factors["thermal_stress"]:
            if self.temp_c < 22.0:
                speed_modifier *= 0.65  # Cold slows metabolic activity
                reasons.append("Cold water metabolic slowdown")
            else:
                speed_modifier *= 1.25  # Heat causes short erratic darting
                reasons.append("Thermal stress hyper-activity")

        # pH stress effect
        if factors["ph_irritation"]:
            feeding_dampener *= 0.50
            reasons.append("pH skin & gill irritation suppressing appetite")

        return {
            "speed_modifier": round(speed_modifier, 2),
            "surface_boost": round(surface_boost, 2),
            "feeding_dampener": round(feeding_dampener, 2),
            "reasons": reasons,
            "stress_index": round(stress_index, 2)
        }

    def generate_time_series(self, steps: int = 20) -> List[Dict[str, Any]]:
        """Generate a time-series of simulated water quality readings.

        Args:
            steps: Number of consecutive telemetry samples to generate.

        Returns:
            List of telemetry dicts, one per step.
        """
        self.clear_simulation_override()
        series = []
        for _ in range(steps):
            series.append(self.get_telemetry())
        return series

    def get_status_summary(self) -> Dict[str, Any]:
        """Return a human-readable status summary of current water conditions.

        Returns:
            Dict with label, color_code, and all parameter readings.
        """
        is_safe, alerts = self.evaluate_safety()
        stress_index, factors = self.calculate_water_stress_index()

        if stress_index < 0.10:
            label, color = "OPTIMAL", "green"
        elif stress_index < 0.30:
            label, color = "NORMAL", "blue"
        elif stress_index < 0.55:
            label, color = "CAUTION", "yellow"
        else:
            label, color = "CRITICAL", "red"

        return {
            "label": label,
            "color_code": color,
            "is_safe": is_safe,
            "alerts": alerts,
            "stress_index": round(stress_index, 3),
            "active_factors": [k for k, v in factors.items() if v],
            "ph": self.ph,
            "temp_c": self.temp_c,
            "turbidity_ntu": self.turbidity_ntu,
            "dissolved_oxygen_mg_l": self.dissolved_oxygen_mg_l,
        }


# ===========================================================================
# Standalone Demo — Module 7
# ===========================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 65)
    print("  MODULE 7 — Water Quality Simulation (Standalone Demo)")
    print("=" * 65)

    monitor = WaterQualityMonitor()

    # --- 1. Live telemetry (natural drift) ---
    print("\n  [1] Live Telemetry Readings (5 natural-drift samples)")
    print(f"  {'#':<4} {'pH':<8} {'Temp°C':<10} {'Turb NTU':<12} {'DO mg/L':<10} {'Safe':<6} {'Stress'}")
    print("  " + "─" * 58)
    for i in range(5):
        t = monitor.get_telemetry()
        safe_icon = "✓" if t["is_safe"] else "✗"
        print(f"  {i+1:<4} {t['ph']:<8.2f} {t['temp_c']:<10.1f} "
              f"{t['turbidity_ntu']:<12.1f} {t['dissolved_oxygen_mg_l']:<10.2f} "
              f"{safe_icon:<6} {t['stress_index']:.3f}")

    # --- 2. Manual override ---
    print("\n  [2] Manual Override — Simulated Hypoxia Scenario")
    monitor.set_simulation_override(ph=7.1, temp_c=31.5, turbidity_ntu=18.0,
                                     dissolved_oxygen_mg_l=3.2)
    t_stress = monitor.get_telemetry()
    print(f"  pH={t_stress['ph']}, Temp={t_stress['temp_c']}°C, "
          f"Turbidity={t_stress['turbidity_ntu']} NTU, DO={t_stress['dissolved_oxygen_mg_l']} mg/L")
    print(f"  Safe: {t_stress['is_safe']}  |  Stress Index: {t_stress['stress_index']}")
    if t_stress["alerts"]:
        print("  Alerts:")
        for a in t_stress["alerts"]:
            print(f"    ⚠  {a}")
    mods = t_stress["behavior_modifiers"]
    print(f"  Behavior: speed_modifier={mods['speed_modifier']}, "
          f"surface_boost={mods['surface_boost']}, "
          f"feeding_dampener={mods['feeding_dampener']}")

    # --- 3. Status summary ---
    print("\n  [3] Status Summary")
    summary = monitor.get_status_summary()
    print(f"  Label         : {summary['label']} ({summary['color_code'].upper()})")
    print(f"  Stress Index  : {summary['stress_index']}")
    print(f"  Active Factors: {summary['active_factors'] or 'None'}")

    # --- 4. Clear override ---
    print("\n  [4] Clearing Override → Back to Natural Drift")
    monitor.clear_simulation_override()
    t_clear = monitor.get_telemetry()
    print(f"  Override active: {monitor.override_active}")
    print(f"  Current pH     : {t_clear['ph']}")

    # --- 5. Time series ---
    print("\n  [5] 10-step Time-Series Generation")
    series = monitor.generate_time_series(steps=10)
    temps = [s["temp_c"] for s in series]
    print(f"  Temp range over 10 steps: {min(temps):.1f}°C → {max(temps):.1f}°C")

    print(f"\n{'='*65}")
    print("  MODULE 7 DEMO COMPLETE — Water quality simulation ready.")
    print(f"{'='*65}")
