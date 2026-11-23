# AI-Based Smart Fish Feeding System for Hunger/Satiety Prediction Across Fish Growth Stages with Water Quality Monitoring

🎓 **Final Year Computer Science Engineering Project**

---

## 📌 Executive Summary
This project presents an end-to-end AI-driven aquaculture management system that integrates computer vision (YOLOv8 + ByteTrack), environmental water quality telemetry (pH, Temperature, Turbidity, Dissolved Oxygen), machine learning classification (RandomForest), and an automated decision engine to predict fish hunger/satiety states across growth stages (Fry, Fingerling, Juvenile, Adult).

The system runs completely on a Windows 10/11 laptop using a webcam or synthetic tank simulation without requiring physical hardware.

---

## 🛠️ Technology Stack
- **Language:** Python 3.11
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Object Tracking:** ByteTrack / Centroid Tracking
- **Machine Learning:** scikit-learn (RandomForest, StandardScaler, joblib)
- **Dashboard:** Streamlit & Plotly
- **Database:** SQLite3
- **Simulation:** Pure Python software-only virtual feeder & sensor telemetry

---

## 📁 Modular Project Architecture

```
Fish_AI_Project/
├── app.py                # Main application CLI orchestration loop
├── camera.py             # Live webcam / synthetic video stream module
├── detector.py           # YOLOv8 fish detection & growth stage estimator
├── tracker.py            # ByteTrack multi-fish tracking & speed calculator
├── water_quality.py      # Water quality sensor telemetry & bio-stress engine
├── behavior.py           # Kinematic behavior analysis & water stress coupling
├── hunger_prediction.py  # ML RandomForest hunger & satiety predictor
├── decision_engine.py    # Rule & safety decision engine
├── virtual_feeder.py     # Software virtual automated feeder dispenser
├── database.py           # SQLite database persistence layer
├── dashboard.py          # Multi-page Streamlit engineering GUI
├── train_model.py        # ML data preprocessing & model training script
├── utils.py              # Math, vector, and formatting utilities
├── config.py             # System thresholds, paths, and constants
├── models/               # Serialized YOLOv8 and RandomForest model weights
├── data/                 # Exported datasets and telemetry logs
├── videos/               # Sample input video clips
├── images/               # Snapshots and graphics
├── database/             # SQLite database file (fish_farm.db)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🌊 Bio-Environmental Coupling: Water Quality Impact on Behavior
A core innovation of this system is the **dynamic bio-environmental coupling**:
- **Low Dissolved Oxygen (Hypoxia < 5.0 mg/L):** Causes hypoxia distress, reducing cruising speed (`speed_modifier = 0.45`) and triggering surface-gasping behavior (`surface_time_ratio` increases up to 90%).
- **Environmental Safety Override:** When water quality parameters fall outside safe boundaries, the `HungerPredictor` suppresses feeding triggers regardless of surface activity to prevent feeding stressed fish, saving feed costs and preserving water quality.

---

## 🚀 Quick Start & Installation

### Step 1: Clone or Extract Repository
```bash
cd Fish_AI_Project
```

### Step 2: Create Virtual Environment (Python 3.11)
```bash
python -m venv venv
# On Windows Command Prompt:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Machine Learning Training (Bootstrap)
```bash
python train_model.py
```

### Step 5: Run Automated Verification Test Suite
```bash
python test_pipeline.py
```

### Step 6: Launch Streamlit Web Engineering Dashboard
```bash
streamlit run dashboard.py
```

### Step 7: Run Full Standalone Pipeline with OpenCV
```bash
# Synthetic camera mode (no physical camera required):
python app.py --source synthetic

# Or default webcam mode:
python app.py
```

---

## 📊 Streamlit Dashboard Modules
1. **Overview Dashboard:** Live KPI cards, water quality trends, feeder hopper capacity gauge.
2. **Live Camera & Detection:** Real-time video canvas with bounding boxes, track IDs, motion trails, and growth stage labels.
3. **Behavior Analysis:** Swimming speed, surface time ratio, feeding zone presence, and 2D spatial coordinate mapping.
4. **Water Quality Telemetry:** Live sensor telemetry with interactive stress simulation sliders and alert boundaries.
5. **Hunger & Satiety Predictor:** RandomForest prediction probabilities, feature breakdown, and retrain interface.
6. **Virtual Feeder:** Dispensing timer, pellet count, stage-specific dosage calculator, and hopper refill controls.
7. **Feeding History & DB:** Multi-tab SQLite database query browser and one-click CSV data exports.
8. **Analytics & Reports:** Cross-parameter correlations (DO vs speed), feed consumption charts, and academic viva voce Q&A guide.
9. **System Diagnostic Logs:** Real-time application logs with severity filter and TXT export.
10. **Settings & Configuration:** Configurable safety thresholds, cooldown timers, and feeding zone boundaries.

---

## 🎓 Academic Defense & Viva Q&A Guide
- **Q: Why use YOLOv8 for fish detection?**  
  *A:* YOLOv8 offers real-time inference speeds (>30 FPS) with high mean Average Precision (mAP) even under challenging underwater refraction and partial fish occlusion.
- **Q: How does the system prevent false-positive hunger predictions during low oxygen?**  
  *A:* The system couples water quality telemetry with behavior analysis. Low DO forces fish to the surface for gasping. The `HungerPredictor` detects hypoxia stress and suppresses feeding triggers, preventing false-positive surface hunger predictions.

---

© 2026 AI Smart Fish Feeding Project. Open-Source Academic Reference.
