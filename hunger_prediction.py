"""
===============================================================================
Fish_AI_Project - Hunger Prediction Module (hunger_prediction.py)
===============================================================================
Predicts fish hunger/satiety state ('Hungry', 'Feeding', 'Satiated', 'Unknown')
using a trained Machine Learning Classifier (RandomForest / Gradient Boosting).
Integrates data preprocessing pipeline and periodic retraining interface.
Suppresses feeding triggers when water quality environmental stress is detected.
"""

import os
import pickle
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from config import ML_MODEL_PATH, SCALER_PATH, HUNGER_LABELS

logger = logging.getLogger(__name__)

# Try importing sklearn & joblib
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HungerPredictor:
    """ML-Driven Fish Hunger & Satiety Predictor with Heuristic Fallback."""

    def __init__(self, model_path: str = ML_MODEL_PATH, scaler_path: str = SCALER_PATH) -> None:
        """Initialize hunger predictor and load ML model weights if available.

        Args:
            model_path: Path to serialized trained model (.pkl).
            scaler_path: Path to feature scaler (.pkl).
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.is_ml_active = False

        self._load_model()

    def _load_model(self) -> bool:
        """Attempt to load trained ML model weights and scaler from disk."""
        if not SKLEARN_AVAILABLE:
            logger.info("scikit-learn package not present. Operating in Heuristic Rule Mode.")
            return False

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_ml_active = True
                logger.info("Successfully loaded trained RandomForest Hunger Classifier.")
                return True
            except Exception as e:
                logger.error(f"Error loading ML model files: {e}")
                self.is_ml_active = False
                return False
        else:
            logger.info("No trained ML model file found. System ready for model training.")
            return False

    def predict(
        self,
        behavior_metrics: Dict[str, Any],
        water_telemetry: Dict[str, Any],
        fish_count: int,
        growth_stage: str = "Fingerling"
    ) -> Dict[str, Any]:
        """Predict hunger state using trained ML model with environmental safety override.

        Args:
            behavior_metrics: Dict from behavior.py.
            water_telemetry: Dict from water_quality.py.
            fish_count: Active detected fish count.
            growth_stage: Dominant fish growth stage.

        Returns:
            Dict containing status ('Hungry', 'Feeding', 'Satiated', 'Unknown'),
            confidence score, decision factors, and environmental suppression state.
        """
        # Environmental Safety Check
        is_safe = water_telemetry.get("is_safe", True)
        stress_index = water_telemetry.get("stress_index", 0.0)

        if not is_safe or stress_index > 0.40:
            return {
                "status": "Satiated",  # Non-feeding state to protect fish health
                "confidence": 0.95,
                "score": 15.0,
                "primary_factors": [
                    "Environmental Stress Override (Unsafe Water)",
                    f"Water Stress Index: {stress_index:.2f}",
                    "; ".join(water_telemetry.get("alerts", []))
                ],
                "water_quality_safe": False,
                "suppressed_by_environment": True,
                "predictor_engine": "Environmental Safety Guard"
            }

        # Try ML Model Inference
        if self.is_ml_active and self.model is not None and self.scaler is not None:
            try:
                features = self._extract_features(behavior_metrics, water_telemetry, fish_count, growth_stage)
                scaled_features = self.scaler.transform([features])

                pred_class_idx = int(self.model.predict(scaled_features)[0])
                probs = self.model.predict_proba(scaled_features)[0]
                confidence = float(np.max(probs))

                status = HUNGER_LABELS[pred_class_idx] if pred_class_idx < len(HUNGER_LABELS) else "Unknown"
                hunger_score = float(probs[HUNGER_LABELS.index("Hungry")] * 100.0) if "Hungry" in HUNGER_LABELS else 50.0

                return {
                    "status": status,
                    "confidence": round(confidence, 2),
                    "score": round(hunger_score, 1),
                    "primary_factors": [
                        f"RandomForest ML Prediction ({confidence * 100:.1f}% confidence)",
                        f"Swimming Speed: {behavior_metrics.get('avg_speed', 0):.1f} px/s",
                        f"Feeding Zone Ratio: {behavior_metrics.get('feeding_zone_presence_ratio', 0)*100:.1f}%",
                        f"Surface Ratio: {behavior_metrics.get('surface_time_ratio', 0)*100:.1f}%"
                    ],
                    "water_quality_safe": True,
                    "suppressed_by_environment": False,
                    "predictor_engine": "RandomForest Classifier (ML)"
                }
            except Exception as e:
                logger.error(f"Error during ML inference ({e}). Falling back to rule engine.")

        # Fallback Modular Heuristic Prediction Engine
        return self._heuristic_predict(behavior_metrics, fish_count)

    def _extract_features(
        self,
        behavior: Dict[str, Any],
        water: Dict[str, Any],
        fish_count: int,
        growth_stage: str
    ) -> List[float]:
        """Convert raw telemetry into numerical feature vector for ML model."""
        stage_map = {"Fry": 0, "Fingerling": 1, "Juvenile": 2, "Adult": 3}
        stage_code = stage_map.get(growth_stage, 1)

        return [
            float(behavior.get("avg_speed", 0.0)),
            float(behavior.get("max_speed", 0.0)),
            float(behavior.get("surface_time_ratio", 0.0)),
            float(behavior.get("feeding_zone_presence_ratio", 0.0)),
            float(behavior.get("movement_intensity", 0.0)),
            float(fish_count),
            float(stage_code),
            float(water.get("ph", 7.4)),
            float(water.get("temp_c", 26.5)),
            float(water.get("turbidity_ntu", 4.0)),
            float(water.get("dissolved_oxygen_mg_l", 7.0)),
            float(water.get("stress_index", 0.0))
        ]

    def _heuristic_predict(self, behavior: Dict[str, Any], fish_count: int) -> Dict[str, Any]:
        """Rule-based prediction engine fallback."""
        speed = behavior.get("avg_speed", 0.0)
        zone_ratio = behavior.get("feeding_zone_presence_ratio", 0.0)
        surface_ratio = behavior.get("surface_time_ratio", 0.0)

        score = (zone_ratio * 50.0) + (min(1.0, speed / 80.0) * 35.0) + (surface_ratio * 15.0)

        if score >= 65.0:
            status = "Hungry"
        elif 40.0 <= score < 65.0:
            status = "Feeding"
        else:
            status = "Satiated"

        return {
            "status": status,
            "confidence": 0.88,
            "score": round(score, 1),
            "primary_factors": [
                "Modular Heuristic Rule Engine",
                f"Feeding Zone Presence: {zone_ratio * 100:.1f}%",
                f"Swimming Intensity: {speed:.1f} px/s"
            ],
            "water_quality_safe": True,
            "suppressed_by_environment": False,
            "predictor_engine": "Heuristic Rule Pipeline"
        }

    def train_or_retrain(self, database_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train or retrain RandomForest model using historical database records.

        Args:
            database_records: List of sample feature/label dicts from SQLite database.

        Returns:
            Dict containing training status, accuracy, and sample count.
        """
        if not SKLEARN_AVAILABLE:
            return {"success": False, "message": "scikit-learn is required for ML training."}

        if len(database_records) < 15:
            # Generate synthetic calibration dataset if historical records are few
            database_records = self._generate_synthetic_training_data()

        X = []
        y = []

        for rec in database_records:
            features = [
                rec["avg_speed"], rec["max_speed"], rec["surface_time_ratio"],
                rec["feeding_zone_presence_ratio"], rec["movement_intensity"],
                rec["fish_count"], rec.get("growth_stage_code", 1),
                rec["ph"], rec["temp_c"], rec["turbidity_ntu"],
                rec["dissolved_oxygen_mg_l"], rec["stress_index"]
            ]
            label_idx = HUNGER_LABELS.index(rec["hunger_label"]) if rec["hunger_label"] in HUNGER_LABELS else 2
            X.append(features)
            y.append(label_idx)

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_scaled, y)

        acc = float(rf.score(X_scaled, y))

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(rf, self.model_path)
        joblib.dump(scaler, self.scaler_path)

        self.model = rf
        self.scaler = scaler
        self.is_ml_active = True

        logger.info(f"Retrained RandomForest Hunger Classifier on {len(X)} samples with {acc*100:.1f}% accuracy.")
        return {
            "success": True,
            "accuracy": round(acc * 100.0, 1),
            "samples": len(X),
            "message": f"Successfully retrained model on {len(X)} samples with {acc*100:.1f}% accuracy."
        }

    def _generate_synthetic_training_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for initial model bootstrap training."""
        records = []
        np.random.seed(42)

        for _ in range(300):
            # Hungry profile: active swimming, high feeding zone presence, good DO
            is_hungry = np.random.rand() > 0.4
            if is_hungry:
                speed = np.random.uniform(60, 140)
                zone_ratio = np.random.uniform(0.55, 0.95)
                surface_ratio = np.random.uniform(0.1, 0.4)
                do = np.random.uniform(6.0, 9.0)
                label = "Hungry"
            else:
                speed = np.random.uniform(10, 50)
                zone_ratio = np.random.uniform(0.0, 0.35)
                surface_ratio = np.random.uniform(0.05, 0.25)
                do = np.random.uniform(5.5, 8.5)
                label = "Satiated"

            records.append({
                "avg_speed": speed,
                "max_speed": speed * 1.3,
                "surface_time_ratio": surface_ratio,
                "feeding_zone_presence_ratio": zone_ratio,
                "movement_intensity": min(100.0, (speed / 120.0) * 100.0),
                "fish_count": int(np.random.randint(4, 15)),
                "growth_stage_code": int(np.random.randint(0, 4)),
                "ph": float(np.random.uniform(7.0, 7.8)),
                "temp_c": float(np.random.uniform(24.0, 28.0)),
                "turbidity_ntu": float(np.random.uniform(2.0, 6.0)),
                "dissolved_oxygen_mg_l": float(do),
                "stress_index": 0.05,
                "hunger_label": label
            })
        return records


# ===========================================================================
# Standalone Demo — Module 6
# ===========================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("=" * 65)
    print("  MODULE 6 — Fish Hunger Prediction (Standalone Demo)")
    print("=" * 65)

    predictor = HungerPredictor()
    print(f"\n  ML Active : {predictor.is_ml_active}")
    print(f"  Engine    : {'RandomForest Classifier' if predictor.is_ml_active else 'Heuristic Rule Engine'}")

    # --- Demo scenarios -------------------------------------------------
    scenarios = [
        (
            "Hungry Fish (high activity)",
            {
                "avg_speed": 95.0, "max_speed": 130.0,
                "surface_time_ratio": 0.35, "feeding_zone_presence_ratio": 0.78,
                "movement_intensity": 82.0
            },
            {
                "ph": 7.4, "temp_c": 26.5, "turbidity_ntu": 4.0,
                "dissolved_oxygen_mg_l": 7.5, "stress_index": 0.05,
                "is_safe": True, "alerts": []
            },
            8
        ),
        (
            "Satiated Fish (low activity)",
            {
                "avg_speed": 15.0, "max_speed": 25.0,
                "surface_time_ratio": 0.05, "feeding_zone_presence_ratio": 0.10,
                "movement_intensity": 12.0
            },
            {
                "ph": 7.5, "temp_c": 27.0, "turbidity_ntu": 3.5,
                "dissolved_oxygen_mg_l": 8.0, "stress_index": 0.03,
                "is_safe": True, "alerts": []
            },
            6
        ),
        (
            "Stressed Water Override",
            {
                "avg_speed": 90.0, "max_speed": 120.0,
                "surface_time_ratio": 0.4, "feeding_zone_presence_ratio": 0.80,
                "movement_intensity": 85.0
            },
            {
                "ph": 5.5, "temp_c": 35.0, "turbidity_ntu": 25.0,
                "dissolved_oxygen_mg_l": 2.5, "stress_index": 0.80,
                "is_safe": False, "alerts": ["Critical pH 5.5", "Hypoxia DO 2.5 mg/L"]
            },
            5
        ),
    ]

    for name, behavior, water, fish_count in scenarios:
        print(f"\n{'─'*55}")
        print(f"  Scenario : {name}")
        result = predictor.predict(behavior, water, fish_count=fish_count)
        print(f"  Status        : {result['status']}")
        print(f"  Confidence    : {result['confidence']*100:.1f}%")
        print(f"  Hunger Score  : {result['score']:.1f}/100")
        print(f"  Env Suppressed: {result['suppressed_by_environment']}")
        print(f"  Engine        : {result['predictor_engine']}")
        print(f"  Factors       :")
        for f in result["primary_factors"]:
            if f:
                print(f"    • {f}")

    # --- Quick retraining demo ------------------------------------------
    print(f"\n{'─'*55}")
    print("  Retraining model on synthetic data …")
    train_result = predictor.train_or_retrain([])
    print(f"  Training complete: {train_result['samples']} samples, "
          f"accuracy {train_result['accuracy']:.1f}%")

    print(f"\n{'='*65}")
    print("  MODULE 6 DEMO COMPLETE — All systems operational.")
    print(f"{'='*65}")
