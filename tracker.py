"""
===============================================================================
Fish_AI_Project - Module 5: ByteTrack Fish Tracker (tracker.py)
===============================================================================
Senior AI Engineer Implementation
Purpose: Multi-object fish tracking across video frames using the ByteTrack
association algorithm. Computes fish kinematics:
- Swimming speed (instantaneous & smoothed pixels/sec)
- Movement (step displacement, cumulative distance, movement intensity)
- Direction (heading angle in degrees & compass bearing)
- Activity level (Low, Moderate, High, and 0-100 continuous score)

Adheres to:
- Object-Oriented Architecture (ByteTrackFishTracker)
- PEP-8 Standards & Comprehensive Docstrings
- Robust Logging and Exception Handling
- 100% Executable Standalone Script
===============================================================================
"""

import os
import sys
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Union

import cv2
import numpy as np

# Import configuration parameters
try:
    from config import (
        FEEDING_ZONE,
        SURFACE_ZONE_Y_MAX,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        TARGET_FPS,
        SAMPLE_VIDEOS_DIR
    )
except ImportError:
    FEEDING_ZONE = {"x1": 0.35, "y1": 0.10, "x2": 0.65, "y2": 0.45}
    SURFACE_ZONE_Y_MAX = 0.25
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    SAMPLE_VIDEOS_DIR = Path("data/sample_videos")

logger = logging.getLogger("ByteTrackFishTracker")


class ByteTrackFishTracker:
    """
    ByteTrack Multi-Object Fish Tracker and Kinematic Behavior Estimator.
    Associates high-confidence and low-confidence detections in a two-stage
    cascade to maintain continuous tracklet identity through occlusions.
    """

    def __init__(
        self,
        high_thresh: float = 0.50,
        low_thresh: float = 0.20,
        max_disappeared: int = 20,
        max_distance: float = 120.0,
        history_len: int = 30
    ) -> None:
        """
        Initialize the ByteTrack Fish Tracker.

        Args:
            high_thresh: Confidence threshold for first-stage primary association.
            low_thresh: Confidence threshold for second-stage recovery association.
            max_disappeared: Max consecutive frames a track can be missing before deregistration.
            max_distance: Maximum pixel distance for valid spatial matching.
            history_len: Number of trajectory points preserved for trails & kinematics.
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_len = history_len

        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.disappeared: Dict[int, int] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        fps: float = TARGET_FPS
    ) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from the current frame using the ByteTrack paradigm.

        Args:
            detections: List of detection dictionaries from detector module.
            fps: Current frame rate for time delta (dt) calculations.

        Returns:
            List of active track dictionaries with updated kinematics:
            id, bbox, center, speed, movement, direction, activity_level.
        """
        dt = 1.0 / max(1.0, fps)

        if not detections:
            # Increment missing counter for all existing tracks
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return list(self.tracks.values())

        # Stage 1: Partition detections into High-Score and Low-Score pools
        high_dets: List[Dict[str, Any]] = []
        low_dets: List[Dict[str, Any]] = []

        for d in detections:
            conf = d.get("confidence", 0.0)
            if conf >= self.high_thresh:
                high_dets.append(d)
            elif conf >= self.low_thresh:
                low_dets.append(d)

        # Fallback if all detections are below high_thresh (e.g. heuristic CV detections)
        if not high_dets and detections:
            high_dets = detections
            low_dets = []

        track_ids = list(self.tracks.keys())
        matched_tracks: Set[int] = set()
        matched_high_dets: Set[int] = set()

        # --- First-Stage Association: Tracks <-> High-Score Detections ---
        if track_ids and high_dets:
            matched_t_indices, matched_d_indices = self._associate(
                track_ids=track_ids,
                detections=high_dets,
                max_dist=self.max_distance
            )
            for t_idx, d_idx in zip(matched_t_indices, matched_d_indices):
                track_id = track_ids[t_idx]
                self._update_track(track_id, high_dets[d_idx], dt)
                matched_tracks.add(t_idx)
                matched_high_dets.add(d_idx)

        # --- Second-Stage Association: Remaining Tracks <-> Low-Score Detections ---
        unmatched_track_indices = [idx for idx in range(len(track_ids)) if idx not in matched_tracks]
        if unmatched_track_indices and low_dets:
            rem_track_ids = [track_ids[idx] for idx in unmatched_track_indices]
            m_rem_t, m_low_d = self._associate(
                track_ids=rem_track_ids,
                detections=low_dets,
                max_dist=self.max_distance * 1.25  # Slightly relaxed for recovery
            )
            for r_idx, l_idx in zip(m_rem_t, m_low_d):
                track_id = rem_track_ids[r_idx]
                self._update_track(track_id, low_dets[l_idx], dt)
                # Mark original track index as matched
                matched_tracks.add(unmatched_track_indices[r_idx])

        # --- Third Stage: Initialize New Tracks for Unmatched High-Score Detections ---
        for d_idx in range(len(high_dets)):
            if d_idx not in matched_high_dets:
                self._register(high_dets[d_idx])

        # --- Fourth Stage: Handle Missing/Lost Tracks ---
        for idx in range(len(track_ids)):
            if idx not in matched_tracks:
                track_id = track_ids[idx]
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)

        return list(self.tracks.values())

    def _associate(
        self,
        track_ids: List[int],
        detections: List[Dict[str, Any]],
        max_dist: float
    ) -> Tuple[List[int], List[int]]:
        """
        Associates tracks with detections via greedy Euclidean distance matching.

        Args:
            track_ids: List of track ID keys.
            detections: List of candidate detections.
            max_dist: Cutoff distance threshold.

        Returns:
            Tuple of (matched_track_indices, matched_detection_indices).
        """
        track_centers = np.array([self.tracks[tid]["center"] for tid in track_ids], dtype=np.float32)
        det_centers = np.array([d["center"] for d in detections], dtype=np.float32)

        # Distance matrix (M tracks x N detections)
        dists = np.linalg.norm(track_centers[:, np.newaxis] - det_centers, axis=2)

        matched_tracks: List[int] = []
        matched_detections: List[int] = []
        used_tracks: Set[int] = set()
        used_dets: Set[int] = set()

        # Sort all (track, det) pairs by ascending distance
        pairs = []
        for t in range(len(track_ids)):
            for d in range(len(detections)):
                pairs.append((dists[t, d], t, d))
        pairs.sort(key=lambda x: x[0])

        for dist, t, d in pairs:
            if t in used_tracks or d in used_dets:
                continue
            if dist <= max_dist:
                matched_tracks.append(t)
                matched_detections.append(d)
                used_tracks.add(t)
                used_dets.add(d)

        return matched_tracks, matched_detections

    def _register(self, detection: Dict[str, Any]) -> None:
        """Registers a new tracklet with initial kinematics."""
        track_id = self.next_track_id
        self.next_track_id += 1

        center = detection["center"]
        self.tracks[track_id] = {
            "id": track_id,
            "bbox": detection["bbox"],
            "center": center,
            "history": [center],
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,                      # Instantaneous swimming speed (px/s)
            "movement": 0.0,                   # Cumulative displacement (px)
            "direction_deg": 0.0,              # Heading angle (-180 to +180)
            "direction_label": "N",            # Compass heading
            "activity_level": "Low",           # 'Low', 'Moderate', 'High'
            "activity_score": 0.0,             # 0 to 100 continuous score
            "growth_stage": detection.get("growth_stage", "Adult"),
            "confidence": detection.get("confidence", 0.0),
            "time_in_feeding_zone": 0.0,
            "time_near_surface": 0.0,
            "first_seen": time.time()
        }
        self.disappeared[track_id] = 0

    def _update_track(self, track_id: int, detection: Dict[str, Any], dt: float) -> None:
        """
        Updates an existing track with new spatial observation and calculates
        swimming speed, movement, direction, and activity level.
        """
        track = self.tracks[track_id]
        old_center = track["center"]
        new_center = detection["center"]

        # 1. Movement & Velocity Calculation
        dx = float(new_center[0] - old_center[0])
        dy = float(new_center[1] - old_center[1])
        step_distance = math.sqrt(dx**2 + dy**2)

        vx_inst = dx / dt
        vy_inst = dy / dt
        speed_inst = step_distance / dt

        # Smooth velocity with exponential moving average
        track["vx"] = 0.65 * track["vx"] + 0.35 * vx_inst
        track["vy"] = 0.65 * track["vy"] + 0.35 * vy_inst
        track["speed"] = round(0.65 * track["speed"] + 0.35 * speed_inst, 2)
        track["movement"] = round(track["movement"] + step_distance, 1)

        # 2. Direction Calculation (Degrees & 8-Point Compass Bearing)
        if step_distance > 1.5:
            deg = math.degrees(math.atan2(dy, dx))
            track["direction_deg"] = round(deg, 1)
            track["direction_label"] = self._calculate_compass_direction(deg)

        # 3. Activity Level Calculation
        # Low: < 25 px/s, Moderate: 25 - 65 px/s, High: >= 65 px/s
        speed = track["speed"]
        if speed < 25.0:
            activity = "Low"
        elif speed < 65.0:
            activity = "Moderate"
        else:
            activity = "High"

        track["activity_level"] = activity
        track["activity_score"] = round(min(100.0, (speed / 100.0) * 100.0), 1)

        # 4. State updates
        track["center"] = new_center
        track["bbox"] = detection["bbox"]
        track["confidence"] = detection.get("confidence", track["confidence"])
        track["growth_stage"] = detection.get("growth_stage", track["growth_stage"])

        track["history"].append(new_center)
        if len(track["history"]) > self.history_len:
            track["history"].pop(0)

        # 5. Operational Zones Integration
        norm_x = new_center[0] / FRAME_WIDTH
        norm_y = new_center[1] / FRAME_HEIGHT

        if (FEEDING_ZONE["x1"] <= norm_x <= FEEDING_ZONE["x2"]) and (FEEDING_ZONE["y1"] <= norm_y <= FEEDING_ZONE["y2"]):
            track["time_in_feeding_zone"] += dt

        if norm_y <= SURFACE_ZONE_Y_MAX:
            track["time_near_surface"] += dt

        self.disappeared[track_id] = 0

    def _calculate_compass_direction(self, degrees: float) -> str:
        """
        Maps a mathematical direction angle (-180 to +180 deg) to an 8-point compass bearing.
        """
        # Normalize to 0 - 360 where 0 = East, 90 = South, 180 = West, 270 = North
        norm_deg = degrees % 360
        compass_sectors = [
            ("E", 22.5), ("SE", 67.5), ("S", 112.5), ("SW", 157.5),
            ("W", 202.5), ("NW", 247.5), ("N", 292.5), ("NE", 337.5)
        ]
        for label, threshold in compass_sectors:
            if norm_deg < threshold:
                return label
        return "E"

    def _deregister(self, track_id: int) -> None:
        """Removes a lost fish track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
            del self.disappeared[track_id]

    def compute_population_kinematics(self) -> Dict[str, Any]:
        """
        Computes aggregate kinematics and population activity metrics across all tracked fish.

        Returns:
            Dictionary with average speed, max speed, total movement, dominant direction,
            and average activity score.
        """
        if not self.tracks:
            return {
                "active_fish": 0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "total_movement": 0.0,
                "avg_activity_score": 0.0,
                "overall_activity_level": "Low",
                "dominant_direction": "N/A"
            }

        tracks = list(self.tracks.values())
        speeds = [t["speed"] for t in tracks]
        movements = [t["movement"] for t in tracks]
        act_scores = [t["activity_score"] for t in tracks]
        directions = [t["direction_label"] for t in tracks]

        avg_spd = round(float(np.mean(speeds)), 2)
        max_spd = round(float(np.max(speeds)), 2)
        tot_mov = round(float(np.sum(movements)), 1)
        avg_act = round(float(np.mean(act_scores)), 1)

        # Categorical overall activity
        if avg_act < 25.0:
            overall_level = "Low"
        elif avg_act < 65.0:
            overall_level = "Moderate"
        else:
            overall_level = "High"

        # Dominant direction (mode)
        dominant_dir = max(set(directions), key=directions.count) if directions else "N/A"

        return {
            "active_fish": len(tracks),
            "avg_speed": avg_spd,
            "max_speed": max_spd,
            "total_movement": tot_mov,
            "avg_activity_score": avg_act,
            "overall_activity_level": overall_level,
            "dominant_direction": dominant_dir
        }

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_zones: bool = True
    ) -> np.ndarray:
        """
        Renders trajectory trails, velocity arrows, speed tags, direction, and activity badges.

        Args:
            frame: Input video image frame.
            tracks: List of active track dictionaries.
            draw_zones: Whether to draw feeding and surface boundary zones.

        Returns:
            Annotated image frame.
        """
        annotated = frame.copy()
        h, w, _ = frame.shape

        # 1. Draw Operational Zones
        if draw_zones:
            fz_x1 = int(FEEDING_ZONE["x1"] * w)
            fz_y1 = int(FEEDING_ZONE["y1"] * h)
            fz_x2 = int(FEEDING_ZONE["x2"] * w)
            fz_y2 = int(FEEDING_ZONE["y2"] * h)
            cv2.rectangle(annotated, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 220, 100), 2)
            cv2.putText(
                annotated, "FEEDING ZONE",
                (fz_x1 + 8, fz_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA
            )

            surf_y = int(h * SURFACE_ZONE_Y_MAX)
            cv2.line(annotated, (0, surf_y), (w, surf_y), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(
                annotated, "SURFACE WATER ZONE (Y < 25%)",
                (15, surf_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA
            )

        activity_colors = {
            "Low": (100, 200, 255),       # Cyan / calm
            "Moderate": (50, 255, 120),   # Green / active
            "High": (50, 100, 255)        # Red-Orange / high activity
        }

        # 2. Render Trajectories, Direction Vectors, and Badges per Fish
        for track in tracks:
            tid = track["id"]
            history = track.get("history", [])
            act_level = track.get("activity_level", "Low")
            color = activity_colors.get(act_level, (0, 255, 255))

            # Motion Trail
            if len(history) > 1:
                for i in range(1, len(history)):
                    thickness = int(np.sqrt(12 * (float(i) / len(history)))) + 1
                    cv2.line(annotated, history[i - 1], history[i], color, thickness, cv2.LINE_AA)

            # Center position
            center = track["center"]
            cv2.circle(annotated, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

            # Heading Direction Vector Arrow
            vx = track.get("vx", 0.0)
            vy = track.get("vy", 0.0)
            tip_x = int(center[0] + np.clip(vx * 0.40, -45, 45))
            tip_y = int(center[1] + np.clip(vy * 0.40, -45, 45))
            cv2.arrowedLine(annotated, center, (tip_x, tip_y), (255, 255, 255), 2, tipLength=0.35)

            # Kinematics Badge: ID | Speed | Dir | Activity
            spd = track.get("speed", 0.0)
            dir_lbl = track.get("direction_label", "N")
            badge = f"Fish #{tid} | {spd:.1f}px/s | {dir_lbl} | Act: {act_level}"

            bx = max(10, center[0] - 60)
            by = max(24, center[1] - 16)
            text_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            cv2.rectangle(annotated, (bx - 2, by - text_size[1] - 4), (bx + text_size[0] + 4, by + 2), (20, 20, 25), -1)
            cv2.putText(annotated, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # 3. HUD Summary Panel: Population Kinematics
        kin = self.compute_population_kinematics()
        hud_h = 36
        cv2.rectangle(annotated, (0, h - hud_h), (w, h), (15, 20, 30), -1)
        cv2.line(annotated, (0, h - hud_h), (w, h - hud_h), (50, 60, 75), 1)

        pop_str = (
            f"Active: {kin['active_fish']} | Avg Speed: {kin['avg_speed']:.1f} px/s | "
            f"Total Mov: {kin['total_movement']:.0f} px | Activity Level: {kin['overall_activity_level']} "
            f"({kin['avg_activity_score']:.1f}%) | Heading: {kin['dominant_direction']}"
        )
        cv2.putText(
            annotated, pop_str,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA
        )

        return annotated


# Backwards compatibility alias
FishTracker = ByteTrackFishTracker


if __name__ == "__main__":
    from detector import FishDetector, FishVideoDetector

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 65)
    print("Running Module 5: ByteTrack Fish Tracker Standalone Execution")
    print("=" * 65)

    sample_video = SAMPLE_VIDEOS_DIR / "fish_sample.mp4"
    src = str(sample_video) if sample_video.is_file() else "synthetic"
    print(f"Tracking Source: {src}")

    detector = FishDetector()
    tracker = ByteTrackFishTracker()
    video_proc = FishVideoDetector(detector=detector, source=src)
    video_proc.open()

    frame_count = 0
    try:
        for raw_frame, detections, summary in video_proc.stream_detections(max_frames=45):
            frame_count += 1
            fps_val = summary.get("fps", 30.0)

            # Update ByteTrack tracker
            active_tracks = tracker.update(detections, fps=fps_val)
            annotated_frame = tracker.draw_tracks(raw_frame, active_tracks)
            kinematics = tracker.compute_population_kinematics()

            if frame_count % 15 == 0 or frame_count == 45:
                print(
                    f"Frame {frame_count:02d} | Tracked Fish: {len(active_tracks)} | "
                    f"Avg Speed: {kinematics['avg_speed']:5.1f} px/s | "
                    f"Activity: {kinematics['overall_activity_level']:<8} "
                    f"({kinematics['avg_activity_score']:.1f}%) | "
                    f"Direction: {kinematics['dominant_direction']}"
                )
    finally:
        video_proc.close()

    print("\nModule 5 ByteTrack tracker executed successfully.")
