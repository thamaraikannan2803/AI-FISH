"""
===============================================================================
Fish_AI_Project - Configuration Module (config.py)
===============================================================================
Central configuration store for system parameters, thresholds, paths, and
model settings. Follows PEP-8 standards.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
TELEMETRY_DIR = DATA_DIR / "telemetry"

MODELS_DIR = BASE_DIR / "models"
YOLO_MODELS_DIR = MODELS_DIR / "yolo11"
HUNGER_ML_MODELS_DIR = MODELS_DIR / "hunger_ml"

VIDEOS_DIR = BASE_DIR / "videos"
IMAGES_DIR = BASE_DIR / "images"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure required directories exist
for folder in [
    DATA_DIR, DATASET_DIR, RAW_IMAGES_DIR, RAW_VIDEOS_DIR, SAMPLE_VIDEOS_DIR,
    TELEMETRY_DIR, MODELS_DIR, YOLO_MODELS_DIR, HUNGER_ML_MODELS_DIR,
    VIDEOS_DIR, IMAGES_DIR, DATABASE_DIR, LOGS_DIR, NOTEBOOKS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_PATH = str(DATABASE_DIR / "fish_farm.db")

# Log Configuration
LOG_FILE = str(LOGS_DIR / "app.log")

# Camera Configuration
WEBCAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
CAMERA_TIMEOUT_SEC = 5

# YOLO11 Detection Configuration
YOLO_MODEL_PATH = str(YOLO_MODELS_DIR / "yolo11n.pt") if (YOLO_MODELS_DIR / "yolo11n.pt").exists() else str(MODELS_DIR / "yolo11n.pt")
YOLO_FALLBACK_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
TARGET_CLASSES = ["fish"]  # Fish detection class

# Bounding Box Sizes to Growth Stage Mapping (Height/Width in pixels)
GROWTH_STAGE_THRESHOLDS = {
    "Fry": (0, 35),
    "Fingerling": (35, 75),
    "Juvenile": (75, 130),
    "Adult": (130, 9999)
}

# Feeding Rate Formula per Growth Stage (grams of feed per fish per day)
FEED_RATES_GRAMS = {
    "Fry": 0.05,
    "Fingerling": 0.35,
    "Juvenile": 1.20,
    "Adult": 4.50
}

# Feeding Zone Box Coordinates (Normalized 0.0 to 1.0)
FEEDING_ZONE = {
    "x1": 0.35,
    "y1": 0.10,
    "x2": 0.65,
    "y2": 0.45
}

# Surface Water Zone Definition (Top 25% of the tank)
SURFACE_ZONE_Y_MAX = 0.25

# Water Quality Thresholds (Ideal & Safe Boundaries)
WATER_QUALITY_THRESHOLDS = {
    "ph": {"min": 6.5, "max": 8.5, "ideal_min": 7.0, "ideal_max": 7.8},
    "temp_c": {"min": 22.0, "max": 30.0, "ideal_min": 25.0, "ideal_max": 28.0},
    "turbidity_ntu": {"min": 0.0, "max": 15.0, "ideal_max": 8.0},
    "dissolved_oxygen_mg_l": {"min": 5.0, "max": 12.0, "critical_min": 4.0, "ideal_min": 6.5}
}

# Machine Learning Model Configuration
ML_MODEL_PATH = str(MODELS_DIR / "hunger_rf_model.pkl")
SCALER_PATH = str(MODELS_DIR / "scaler.pkl")

# Hunger State Labels
HUNGER_LABELS = ["Hungry", "Feeding", "Satiated", "Unknown"]

# Feeder Configuration
MAX_FEED_PER_SESSION_GRAMS = 150.0
FEEDING_DISPENSE_DURATION_SEC = 5.0
