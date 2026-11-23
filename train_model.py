"""
===============================================================================
Fish_AI_Project - Machine Learning Training Script (train_model.py)
===============================================================================
Standalone dataset generator and training pipeline for the Hunger Prediction
RandomForest classifier. Preprocesses historical records, scales features,
trains model, and saves weights to models/hunger_rf_model.pkl.
"""

import os
import sys
import logging
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
except ImportError:
    print("scikit-learn and joblib are required for model training. Install via: pip install scikit-learn joblib")
    sys.exit(1)

from config import MODELS_DIR, ML_MODEL_PATH, SCALER_PATH, HUNGER_LABELS
from database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(num_samples: int = 600):
    """Generate realistic synthetic aquaculture dataset for bootstrap model training."""
    np.random.seed(42)
    X = []
    y = []

    for _ in range(num_samples):
        state_rnd = np.random.rand()

        if state_rnd < 0.45:
            # Hungry class (0): high speed, high feeding zone presence, high movement, nominal water DO
            label_idx = 0  # Hungry
            speed = np.random.uniform(65.0, 140.0)
            max_speed = speed * np.random.uniform(1.2, 1.5)
            surface_ratio = np.random.uniform(0.15, 0.45)
            zone_ratio = np.random.uniform(0.55, 0.95)
            movement_intensity = min(100.0, (speed / 120.0) * 100.0)
            do = np.random.uniform(6.0, 9.5)
            temp = np.random.uniform(24.0, 28.5)
            ph = np.random.uniform(7.0, 7.8)
            turbidity = np.random.uniform(1.0, 6.0)
            stress = 0.05
        elif state_rnd < 0.75:
            # Feeding class (1): moderate speed, very high feeding zone presence
            label_idx = 1  # Feeding
            speed = np.random.uniform(40.0, 85.0)
            max_speed = speed * np.random.uniform(1.1, 1.3)
            surface_ratio = np.random.uniform(0.10, 0.35)
            zone_ratio = np.random.uniform(0.70, 0.98)
            movement_intensity = min(100.0, (speed / 120.0) * 100.0)
            do = np.random.uniform(5.5, 9.0)
            temp = np.random.uniform(23.5, 29.0)
            ph = np.random.uniform(6.8, 8.0)
            turbidity = np.random.uniform(2.0, 8.0)
            stress = 0.10
        else:
            # Satiated class (2): low speed, low feeding zone presence, resting
            label_idx = 2  # Satiated
            speed = np.random.uniform(8.0, 35.0)
            max_speed = speed * np.random.uniform(1.1, 1.4)
            surface_ratio = np.random.uniform(0.05, 0.25)
            zone_ratio = np.random.uniform(0.02, 0.25)
            movement_intensity = min(100.0, (speed / 120.0) * 100.0)
            do = np.random.uniform(5.0, 8.5)
            temp = np.random.uniform(22.0, 30.0)
            ph = np.random.uniform(6.5, 8.5)
            turbidity = np.random.uniform(1.0, 10.0)
            stress = 0.15

        fish_count = np.random.randint(3, 16)
        growth_stage_code = np.random.randint(0, 4)

        features = [
            speed, max_speed, surface_ratio, zone_ratio, movement_intensity,
            fish_count, growth_stage_code, ph, temp, turbidity, do, stress
        ]
        X.append(features)
        y.append(label_idx)

    return np.array(X), np.array(y)


def train_pipeline():
    """Execute model training, validation, and artifact export."""
    logger.info("Starting Machine Learning Training Pipeline for Hunger Prediction...")

    X, y = generate_synthetic_dataset(num_samples=800)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RandomForest Classifier
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    logger.info(f"Training Accuracy:   {train_acc * 100:.2f}%")
    logger.info(f"Validation Accuracy: {test_acc * 100:.2f}%")

    y_pred = clf.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Hungry", "Feeding", "Satiated"]))

    # Export Model Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, ML_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    logger.info(f"Model successfully saved to {ML_MODEL_PATH}")
    logger.info(f"Scaler successfully saved to {SCALER_PATH}")


if __name__ == "__main__":
    train_pipeline()
