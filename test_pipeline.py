import numpy as np
from camera import CameraModule
from detector import FishDetector
from tracker import FishTracker
from behavior import BehaviorAnalyzer
from water_quality import WaterQualityMonitor
from hunger_prediction import HungerPredictor
from decision_engine import DecisionEngine
from virtual_feeder import VirtualFeeder
from database import DatabaseManager
from utils import draw_hud_overlay

def run_tests():
    print("1. Testing Camera...")
    cam = CameraModule(source="synthetic")
    assert cam.start()
    ret, frame = cam.read_frame()
    assert ret and frame is not None
    print("   -> Frame OK:", frame.shape)

    print("2. Testing Detector...")
    det = FishDetector()
    detections = det.detect(frame)
    annotated_det = det.draw_detections(frame, detections)
    print(f"   -> Detections: {len(detections)}")

    print("3. Testing Tracker...")
    tracker = FishTracker()
    tracks = tracker.update(detections, fps=30)
    annotated_tracks = tracker.draw_tracks(annotated_det, tracks)
    print(f"   -> Tracks: {len(tracks)}")

    print("4. Testing Water Quality Monitor...")
    wq = WaterQualityMonitor()
    telemetry = wq.get_telemetry()
    assert "ph" in telemetry and "dissolved_oxygen_mg_l" in telemetry
    print("   -> Telemetry OK. Stress index:", telemetry["stress_index"])

    print("5. Testing Behavior Analyzer...")
    behavior = BehaviorAnalyzer()
    metrics = behavior.analyze_tracks(tracks, telemetry["behavior_modifiers"])
    assert "avg_speed" in metrics and "behavior_score" in metrics
    print("   -> Behavior Score:", metrics["behavior_score"])

    print("6. Testing Hunger Predictor (ML)...")
    pred = HungerPredictor()
    assert pred.is_ml_active
    prediction = pred.predict(metrics, telemetry, len(tracks), "Fingerling")
    print(f"   -> Prediction: {prediction['status']} (Score: {prediction['score']}, Conf: {prediction['confidence']})")

    print("7. Testing Decision Engine...")
    engine = DecisionEngine()
    decision = engine.evaluate(telemetry, prediction, len(tracks), "Fingerling")
    print("   -> Decision:", decision["reason"])

    print("8. Testing Virtual Feeder...")
    feeder = VirtualFeeder()
    status_before = feeder.get_status()
    if decision["should_feed"]:
        feeder.trigger_dispense(decision["recommended_grams"])
    status_after = feeder.get_status()
    print("   -> Feeder Status:", status_after["status"])

    print("9. Testing Database Manager...")
    db = DatabaseManager()
    stats = db.get_summary_stats()
    print("   -> DB Summary Stats:", stats)

    print("10. Testing HUD Overlay...")
    hud = draw_hud_overlay(annotated_tracks, telemetry, prediction, status_after, len(tracks), "Fingerling")
    assert hud.shape == frame.shape
    print("   -> HUD Render OK!")

    print("11. Testing Bio-Environmental Safety Override during Hypoxia (DO = 3.0 mg/L)...")
    wq.set_simulation_override(ph=7.4, temp_c=26.5, turbidity_ntu=4.0, dissolved_oxygen_mg_l=3.0)
    hypoxia_telemetry = wq.get_telemetry()
    hypoxia_pred = pred.predict(metrics, hypoxia_telemetry, len(tracks), "Fingerling")
    assert hypoxia_pred["suppressed_by_environment"] is True
    assert hypoxia_pred["water_quality_safe"] is False
    print("   -> Hypoxia Safety Override Triggered:", hypoxia_pred["primary_factors"][0])

    hypoxia_decision = engine.evaluate(hypoxia_telemetry, hypoxia_pred, len(tracks), "Fingerling")
    assert hypoxia_decision["should_feed"] is False
    assert "FEEDING BLOCKED" in hypoxia_decision["reason"]
    print("   -> Feeding safely blocked:", hypoxia_decision["reason"])

    cam.release()
    print("\n=== ALL 11 MODULE & BIO-COUPLING TESTS PASSED FLAWLESSLY ===")

if __name__ == "__main__":
    run_tests()
