"""
===============================================================================
Fish_AI_Project - Main Application Entry Point (app.py)
===============================================================================
Orchestrates real-time pipeline: Camera -> YOLO Detector -> ByteTrack Tracker ->
Water Quality Monitor -> Behavior Analyzer -> Hunger Predictor -> Decision Engine ->
Virtual Feeder -> SQLite Database Logger.
"""

import time
import argparse
import logging
import cv2
import numpy as np

from camera import CameraModule
from detector import FishDetector
from tracker import FishTracker
from behavior import BehaviorAnalyzer
from hunger_prediction import HungerPredictor
from water_quality import WaterQualityMonitor
from decision_engine import DecisionEngine
from virtual_feeder import VirtualFeeder
from database import DatabaseManager
from utils import draw_hud_overlay
from config import WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(module)s: %(message)s")
logger = logging.getLogger("MainApp")


def parse_args():
    parser = argparse.ArgumentParser(description="AI-Based Smart Fish Feeding System")
    parser.add_argument("--source", default=None, help="Camera index, video path, or 'synthetic'")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 for infinite loop)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing AI-Based Smart Fish Feeding System...")

    cam_source = args.source if args.source is not None else WEBCAM_INDEX

    db = DatabaseManager()
    camera = CameraModule(source=cam_source, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    detector = FishDetector()
    tracker = FishTracker()
    behavior_analyzer = BehaviorAnalyzer()
    water_monitor = WaterQualityMonitor()
    hunger_predictor = HungerPredictor()
    decision_engine = DecisionEngine()
    virtual_feeder = VirtualFeeder()

    if not camera.start():
        logger.error("Failed to start camera feed. Falling back to synthetic simulation.")
        camera = CameraModule(source="synthetic", width=FRAME_WIDTH, height=FRAME_HEIGHT)
        camera.start()

    logger.info(f"System loop started with source: {camera.source}. Press 'q' or CTRL+C to quit.")
    db.log_system_event("INFO", "MainApp", f"System started successfully with source {camera.source}.")

    frame_counter = 0

    try:
        while camera.is_running:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue

            frame_counter += 1

            # 1. Object Detection
            detections = detector.detect(frame)

            # 2. Tracking & Trajectory Analysis
            tracks = tracker.update(detections, fps=camera.fps)

            # 3. Water Quality Telemetry & Stress Index
            wq_telemetry = water_monitor.get_telemetry()

            # 4. Behavior Analysis coupled with Water Quality
            behavior_metrics = behavior_analyzer.analyze_tracks(
                tracks,
                water_quality_modifiers=wq_telemetry["behavior_modifiers"]
            )

            # Determine dominant growth stage
            growth_stages = [t.get("growth_stage", "Fingerling") for t in tracks]
            dominant_stage = max(set(growth_stages), key=growth_stages.count) if growth_stages else "Fingerling"

            # 5. ML Hunger & Satiety Prediction
            prediction = hunger_predictor.predict(
                behavior_metrics=behavior_metrics,
                water_telemetry=wq_telemetry,
                fish_count=len(tracks),
                growth_stage=dominant_stage
            )

            # 6. Automated Decision Engine
            decision = decision_engine.evaluate(
                water_telemetry=wq_telemetry,
                hunger_prediction=prediction,
                fish_count=len(tracks),
                growth_stage=dominant_stage
            )

            # 7. Virtual Feeder Progress Update & Decision Logging
            if decision["should_feed"] and virtual_feeder.status == "Idle":
                virtual_feeder.trigger_dispense(decision["recommended_grams"])
                db.log_feeding_decision(
                    should_feed=decision["should_feed"],
                    amount=decision["recommended_grams"],
                    stage=dominant_stage,
                    count=len(tracks),
                    hunger_status=prediction["status"],
                    wq_status=decision["water_quality_status"],
                    reason=decision["reason"]
                )

            feeder_state = virtual_feeder.update()

            # 8. Log telemetry periodically to database
            if int(time.time()) % 5 == 0:
                avg_confidence = float(np.mean([t["confidence"] for t in tracks])) if tracks else 0.0
                db.log_detection(
                    fish_count=len(tracks),
                    growth_stage=dominant_stage,
                    avg_conf=avg_confidence,
                    avg_speed=behavior_metrics["avg_speed"],
                    zone_count=int(sum(1 for t in tracks if t.get("time_in_feeding_zone", 0) > 0))
                )
                db.log_water_quality(
                    ph=wq_telemetry["ph"],
                    temp=wq_telemetry["temp_c"],
                    turbidity=wq_telemetry["turbidity_ntu"],
                    do=wq_telemetry["dissolved_oxygen_mg_l"],
                    is_safe=wq_telemetry["is_safe"],
                    stress_index=wq_telemetry["stress_index"],
                    alerts="; ".join(wq_telemetry["alerts"])
                )
                db.log_prediction(
                    status=prediction["status"],
                    conf=prediction["confidence"],
                    score=prediction["score"],
                    engine=prediction["predictor_engine"],
                    suppressed=prediction["suppressed_by_environment"],
                    factors="; ".join(prediction["primary_factors"])
                )

            # 9. Annotate Frame with Detections, Trajectories, and Telemetry HUD
            annotated_frame = detector.draw_detections(frame, detections, draw_zones=True)
            annotated_frame = tracker.draw_tracks(annotated_frame, tracks)
            annotated_frame = draw_hud_overlay(
                annotated_frame,
                wq_telemetry,
                prediction,
                feeder_state,
                len(tracks),
                dominant_stage
            )

            cv2.imshow("AI Smart Fish Feeding System - OpenCV Feed", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if args.max_frames > 0 and frame_counter >= args.max_frames:
                logger.info(f"Reached max-frames limit ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Termination signal received.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        db.log_system_event("INFO", "MainApp", "System shutdown cleanly.")
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()
