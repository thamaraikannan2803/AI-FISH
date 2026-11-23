"""
===============================================================================
Fish_AI_Project - Utility Module (utils.py)
===============================================================================
Helper functions for frame rendering overlays, mathematical vector calculations,
unit conversions, and academic PDF/CSV report generation exports.
"""

import math
import time
from typing import Tuple, List, Dict, Any
import cv2
import numpy as np


def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def calculate_vector_angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Calculate angle in degrees between two 2D vectors."""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def format_timestamp(dt_seconds: float = None) -> str:
    """Format Unix timestamp to ISO standard datetime string."""
    if dt_seconds is None:
        dt_seconds = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dt_seconds))


def compute_feed_dosage(growth_stage: str, fish_count: int) -> float:
    """Compute recommended feed dosage in grams per session."""
    rates = {
        "Fry": 0.05,
        "Fingerling": 0.35,
        "Juvenile": 1.20,
        "Adult": 4.50
    }
    rate = rates.get(growth_stage, 0.35)
    return round(rate * max(1, fish_count), 2)


def draw_hud_overlay(
    frame: np.ndarray,
    telemetry: Dict[str, Any],
    prediction: Dict[str, Any],
    feeder_status: Dict[str, Any],
    fish_count: int,
    dominant_stage: str
) -> np.ndarray:
    """Render a comprehensive telemetry Heads-Up Display (HUD) overlay on the video frame."""
    hud = frame.copy()
    h, w, _ = frame.shape

    # Semi-transparent top bar (Water Quality)
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)

    # Water Quality Telemetry Text
    wq_safe = telemetry.get("is_safe", True)
    safe_color = (0, 255, 120) if wq_safe else (0, 60, 255)
    safe_badge = "WATER: SAFE" if wq_safe else "WATER: ALERT"
    cv2.putText(hud, safe_badge, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, safe_color, 2)

    wq_str = (
        f"pH: {telemetry.get('ph', 7.4):.2f} | "
        f"Temp: {telemetry.get('temp_c', 26.5):.1f}C | "
        f"Turb: {telemetry.get('turbidity_ntu', 4.0):.1f} NTU | "
        f"DO: {telemetry.get('dissolved_oxygen_mg_l', 7.0):.2f} mg/L"
    )
    cv2.putText(hud, wq_str, (160, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Semi-transparent bottom bar (Predictions & Feeder)
    overlay_bot = hud.copy()
    cv2.rectangle(overlay_bot, (0, h - 50), (w, h), (15, 20, 30), -1)
    cv2.addWeighted(overlay_bot, 0.85, hud, 0.15, 0, hud)

    # Hunger Prediction Badge
    p_status = prediction.get("status", "Unknown").upper()
    p_score = prediction.get("score", 0.0)
    p_color = (0, 255, 120) if p_status == "HUNGRY" else ((255, 200, 0) if p_status == "FEEDING" else (180, 180, 180))
    if prediction.get("suppressed_by_environment", False):
        p_status = "SUPPRESSED"
        p_color = (0, 100, 255)

    status_str = f"PREDICTION: {p_status} ({p_score:.0f}%)"
    cv2.putText(hud, status_str, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.60, p_color, 2)

    # Fish Population
    pop_str = f"Fish: {fish_count} ({dominant_stage})"
    cv2.putText(hud, pop_str, (370, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Feeder Status
    f_state = feeder_status.get("status", "Idle")
    f_hopper = feeder_status.get("hopper_level_grams", 2000.0)
    f_str = f"Feeder: {f_state} | Hopper: {f_hopper:.0f}g"
    cv2.putText(hud, f_str, (630, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # Dispense animation indicator
    if f_state == "Dispensing":
        cv2.putText(hud, "[DISPENSING FEED...]", (w - 240, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    return hud
