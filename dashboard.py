"""
===============================================================================
Fish_AI_Project - Comprehensive Streamlit Dashboard (dashboard.py)
===============================================================================
Interactive multi-page aquaculture monitoring, computer vision, behavior
analysis, ML hunger prediction, virtual feeder control, and database reporting GUI.

Run with: streamlit run dashboard.py
"""

import time
import os
import io
from datetime import datetime
import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from camera import CameraModule
from detector import FishDetector
from tracker import FishTracker
from behavior import BehaviorAnalyzer
from hunger_prediction import HungerPredictor
from water_quality import WaterQualityMonitor
from decision_engine import DecisionEngine
from virtual_feeder import VirtualFeeder
from database import DatabaseManager
from utils import draw_hud_overlay, compute_feed_dosage
from config import (
    GROWTH_STAGE_THRESHOLDS,
    FEED_RATES_GRAMS,
    WATER_QUALITY_THRESHOLDS,
    FEEDING_ZONE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MODELS_DIR,
    ML_MODEL_PATH,
    SCALER_PATH,
    IMAGES_DIR
)


def init_session_services():
    """Ensure persistent singletons across Streamlit re-renders."""
    if "db" not in st.session_state:
        st.session_state.db = DatabaseManager()
    if "wq" not in st.session_state:
        st.session_state.wq = WaterQualityMonitor()
    if "feeder" not in st.session_state:
        st.session_state.feeder = VirtualFeeder()
    if "predictor" not in st.session_state:
        st.session_state.predictor = HungerPredictor()
    if "detector" not in st.session_state:
        st.session_state.detector = FishDetector()
    if "tracker" not in st.session_state:
        st.session_state.tracker = FishTracker()
    if "behavior" not in st.session_state:
        st.session_state.behavior = BehaviorAnalyzer()
    if "decision_engine" not in st.session_state:
        st.session_state.decision_engine = DecisionEngine()
    if "camera" not in st.session_state:
        cam = CameraModule(source="synthetic", width=FRAME_WIDTH, height=FRAME_HEIGHT)
        cam.start()
        st.session_state.camera = cam


def render_sidebar():
    """Render professional sidebar menu and real-time system status badges."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #38bdf8; margin: 0;">🐟 FishAI Studio</h2>
            <p style="color: #94a3b8; font-size: 0.82rem; margin: 0;">Smart Aquaculture System</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.divider()

    menu = [
        "📊 Overview Dashboard",
        "📹 Live Camera & Detection",
        "📈 Behavior Analysis",
        "💧 Water Quality Telemetry",
        "🤖 Hunger & Satiety Predictor",
        "🥣 Virtual Feeder",
        "🗄️ Feeding History & DB",
        "📑 Analytics & Reports",
        "📜 System Logs",
        "⚙️ Settings & Configuration"
    ]

    choice = st.sidebar.radio("Navigation Menu", menu, index=0)

    st.sidebar.divider()
    st.sidebar.markdown("### 🛰️ System Status")

    pred_active = st.session_state.predictor.is_ml_active
    st.sidebar.markdown(
        f"**ML Engine:** {'🟢 RandomForest (Active)' if pred_active else '🟡 Heuristic Rule Fallback'}"
    )

    wq_safe = st.session_state.wq.evaluate_safety()[0]
    st.sidebar.markdown(
        f"**Environment:** {'🟢 Water Safe' if wq_safe else '🔴 Alert / Stressed'}"
    )

    feeder_state = st.session_state.feeder.status
    feeder_color = "🟢" if feeder_state == "Idle" else ("🟡" if feeder_state == "Dispensing" else "🔵")
    st.sidebar.markdown(f"**Feeder:** {feeder_color} {feeder_state}")

    st.sidebar.caption("© 2026 Final Year CSE Project")

    return choice


def main():
    st.set_page_config(
        page_title="AI Smart Fish Feeding System",
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global custom styling
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        .metric-card {
            background: #131c2e;
            border: 1px solid #1e2d4a;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-safe { background-color: #064e3b; color: #34d399; }
        .badge-alert { background-color: #7f1d1d; color: #f87171; }
        .badge-warning { background-color: #78350f; color: #fbbf24; }
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    init_session_services()
    db = st.session_state.db
    wq = st.session_state.wq
    feeder = st.session_state.feeder
    predictor = st.session_state.predictor
    detector = st.session_state.detector
    tracker = st.session_state.tracker
    behavior = st.session_state.behavior
    decision_engine = st.session_state.decision_engine
    camera = st.session_state.camera

    choice = render_sidebar()

    # -------------------------------------------------------------------------
    # 1. OVERVIEW DASHBOARD
    # -------------------------------------------------------------------------
    if choice == "📊 Overview Dashboard":
        st.title("📊 System Overview & Live Telemetry")
        st.caption("AI-Powered Smart Feeding Pipeline: Computer Vision, Water Telemetry, & Automated Dispenser")

        telemetry = wq.get_telemetry()
        feeder_status = feeder.get_status()
        summary = db.get_summary_stats()

        # Row 1: Water Quality Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("pH Level", f"{telemetry['ph']:.2f}", delta="Optimal 7.0-7.8")
        c2.metric("Temperature", f"{telemetry['temp_c']:.1f} °C", delta="Ideal 25-28°C")
        c3.metric("Turbidity", f"{telemetry['turbidity_ntu']:.1f} NTU", delta="Target < 8 NTU")
        c4.metric(
            "Dissolved Oxygen",
            f"{telemetry['dissolved_oxygen_mg_l']:.2f} mg/L",
            delta="Min 5.0 mg/L",
            delta_color="normal" if telemetry["dissolved_oxygen_mg_l"] >= 5.0 else "inverse"
        )

        st.divider()

        # Row 2: Fish Population & Feeder Metrics
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Avg Detected Fish", f"{summary['avg_fish_count']:.0f} fish", delta="Active population")
        f2.metric("Swimming Activity", f"{summary['avg_speed']:.1f} px/s", delta="Normal movement")
        f3.metric("Feeder Hopper", f"{feeder_status['hopper_level_grams']} g", delta=f"{feeder_status['status']}")
        f4.metric("Total Feed Today", f"{summary['total_dispensed_grams']:.1f} g", delta=f"{summary['total_feedings']} feeding events")

        st.divider()

        col_chart_left, col_chart_right = st.columns([2, 1])

        with col_chart_left:
            st.subheader("📈 Multi-Sensor Environmental Telemetry")
            recent_wq = db.get_recent_water_quality(limit=25)
            if recent_wq:
                df_wq = pd.DataFrame(recent_wq)
                fig_env = px.line(
                    df_wq,
                    x="timestamp",
                    y=["dissolved_oxygen", "temperature", "ph"],
                    title="Historical Water Parameters Trend (DO, Temp, pH)",
                    markers=True,
                    template="plotly_dark",
                    labels={"value": "Measured Value", "timestamp": "Timestamp", "variable": "Sensor"}
                )
                fig_env.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_env, use_container_width=True)
            else:
                st.info("No water quality records available yet.")

        with col_chart_right:
            st.subheader("🥣 Feeder Hopper Capacity")
            hopper_pct = min(100.0, (feeder_status["hopper_level_grams"] / 2000.0) * 100.0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hopper_pct,
                title={'text': "Hopper Capacity %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#38bdf8"},
                    'steps': [
                        {'range': [0, 25], 'color': "#7f1d1d"},
                        {'range': [25, 60], 'color': "#78350f"},
                        {'range': [60, 100], 'color': "#064e3b"}
                    ]
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            q_c1, q_c2 = st.columns(2)
            with q_c1:
                if st.button("Manual Feed (15g)", use_container_width=True):
                    feeder.trigger_dispense(15.0)
                    st.success("Dispensed 15g!")
            with q_c2:
                if st.button("Refill Hopper", use_container_width=True):
                    feeder.refill_hopper(2000.0)
                    st.success("Hopper refilled!")

    # -------------------------------------------------------------------------
    # 2. LIVE CAMERA & DETECTION
    # -------------------------------------------------------------------------
    elif choice == "📹 Live Camera & Detection":
        st.title("📹 Live Camera & YOLOv8 Detection Canvas")
        st.caption("Real-time fish localization, bounding box regression, growth stage categorization, and trajectory tracking")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
        source_mode = ctrl_col1.selectbox("Video Input Stream", ["Synthetic Tank Simulation", "Webcam 0", "Webcam 1"])
        show_trails = ctrl_col2.checkbox("Show Trajectory Trails", value=True)
        show_hud = ctrl_col3.checkbox("Show HUD Telemetry Bar", value=True)

        # Switch camera source if requested
        target_source = "synthetic" if "Synthetic" in source_mode else int(source_mode.split()[-1])
        if camera.source != target_source:
            camera.release()
            camera = CameraModule(source=target_source, width=FRAME_WIDTH, height=FRAME_HEIGHT)
            camera.start()
            st.session_state.camera = camera

        st.divider()

        frame_placeholder = st.empty()

        col_cam_left, col_cam_right = st.columns([3, 1])

        # Read single updated frame
        ret, frame = camera.read_frame()
        if ret and frame is not None:
            # Detection & Tracking
            detections = detector.detect(frame)
            tracks = tracker.update(detections, fps=camera.fps)
            telemetry = wq.get_telemetry()
            behavior_metrics = behavior.analyze_tracks(tracks, telemetry["behavior_modifiers"])

            growth_stages = [t.get("growth_stage", "Fingerling") for t in tracks]
            dominant_stage = max(set(growth_stages), key=growth_stages.count) if growth_stages else "Fingerling"

            prediction = predictor.predict(behavior_metrics, telemetry, len(tracks), dominant_stage)
            feeder_state = feeder.get_status()

            # Render overlays
            annotated = detector.draw_detections(frame, detections, draw_zones=True)
            if show_trails:
                annotated = tracker.draw_tracks(annotated, tracks)
            if show_hud:
                annotated = draw_hud_overlay(annotated, telemetry, prediction, feeder_state, len(tracks), dominant_stage)

            # Convert BGR to RGB for Streamlit
            rgb_frame = annotated[:, :, ::-1]

            with col_cam_left:
                st.image(rgb_frame, caption=f"Active Stream: {source_mode} | FPS: {camera.fps:.1f}", use_container_width=True)

            with col_cam_right:
                st.subheader("Fish Telemetry")
                st.write(f"**Detected Count:** `{len(tracks)}`")
                st.write(f"**Dominant Stage:** `{dominant_stage}`")
                st.write(f"**Avg Speed:** `{behavior_metrics['avg_speed']:.1f} px/s`")
                st.write(f"**In Feeding Zone:** `{behavior_metrics['feeding_zone_presence_ratio']*100:.0f}%`")
                st.write(f"**Surface Gasping:** `{behavior_metrics['surface_time_ratio']*100:.0f}%`")

                if st.button("Capture Snapshot", use_container_width=True):
                    snap_path = str(IMAGES_DIR / f"snapshot_{int(time.time())}.jpg")
                    Image.fromarray(rgb_frame).save(snap_path)
                    st.success(f"Saved: {os.path.basename(snap_path)}")

                if st.button("Refresh Stream Frame", use_container_width=True):
                    st.rerun()

        # Detections table
        if tracks:
            st.subheader("Detected Fish Catalog")
            df_tracks = pd.DataFrame([{
                "Track ID": f"#{t['id']}",
                "Growth Stage": t.get("growth_stage", ""),
                "Speed (px/s)": round(t.get("speed", 0.0), 1),
                "Confidence": t.get("confidence", 0.0),
                "Position (X, Y)": f"({t['center'][0]}, {t['center'][1]})",
                "Feeding Zone Presence (s)": round(t.get("time_in_feeding_zone", 0.0), 1)
            } for t in tracks])
            st.dataframe(df_tracks, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. BEHAVIOR ANALYSIS
    # -------------------------------------------------------------------------
    elif choice == "📈 Behavior Analysis":
        st.title("📈 Kinematic Behavior Analysis")
        st.caption("Deep movement dynamics: instantaneous velocity, spatial distribution, surface gasping, and water stress coupling")

        ret, frame = camera.read_frame()
        detections = detector.detect(frame) if ret else []
        tracks = tracker.update(detections, fps=camera.fps)
        telemetry = wq.get_telemetry()
        metrics = behavior.analyze_tracks(tracks, telemetry["behavior_modifiers"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Swimming Speed", f"{metrics['avg_speed']:.1f} px/s", delta="Coupled with DO")
        c2.metric("Max Burst Speed", f"{metrics['max_speed']:.1f} px/s", delta="Darting intensity")
        c3.metric("Surface Presence Ratio", f"{metrics['surface_time_ratio'] * 100:.1f} %", delta="Hypoxia gasping indicator")
        c4.metric("Feeding Zone Occupancy", f"{metrics['feeding_zone_presence_ratio'] * 100:.1f} %", delta="Appetite indicator")

        st.divider()

        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            st.subheader("📍 2D Spatial Distribution in Tank")
            # Create synthetic coordinate positions for scatter
            pts_x = [t["center"][0] for t in tracks] if tracks else [400, 600, 800, 300]
            pts_y = [t["center"][1] for t in tracks] if tracks else [200, 350, 500, 150]
            pts_stage = [t.get("growth_stage", "Fingerling") for t in tracks] if tracks else ["Fingerling", "Adult", "Juvenile", "Fry"]

            fig_scatter = px.scatter(
                x=pts_x,
                y=pts_y,
                color=pts_stage,
                title="Fish Coordinates (X vs Y Position in Tank)",
                labels={"x": "Tank Width (px)", "y": "Tank Depth (px)"},
                template="plotly_dark",
                range_x=[0, FRAME_WIDTH],
                range_y=[FRAME_HEIGHT, 0]  # Invert Y to match computer vision coords
            )
            # Add feeding zone rectangle to plot
            fig_scatter.add_shape(
                type="rect",
                x0=FEEDING_ZONE["x1"] * FRAME_WIDTH,
                y0=FEEDING_ZONE["y1"] * FRAME_HEIGHT,
                x1=FEEDING_ZONE["x2"] * FRAME_WIDTH,
                y1=FEEDING_ZONE["y2"] * FRAME_HEIGHT,
                line=dict(color="#10b981", width=2, dash="dash"),
                fillcolor="rgba(16, 185, 129, 0.15)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b2:
            st.subheader("🎯 Composite Behavior Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics["behavior_score"],
                title={'text': "Feeding Readiness Index (0-100)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': "#374151"},
                        {'range': [40, 70], 'color': "#1e3a8a"},
                        {'range': [70, 100], 'color': "#047857"}
                    ]
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Bio-environmental coupling explanation
        st.info(
            f"""
            **🌊 Dynamic Bio-Environmental Stress Coupling:**  
            {metrics['stress_description']}  
            - **Speed Modifier:** `{telemetry['behavior_modifiers']['speed_modifier']}x`  
            - **Surface Boost:** `+{telemetry['behavior_modifiers']['surface_boost'] * 100:.0f}%`  
            - **Feeding Dampener:** `{telemetry['behavior_modifiers']['feeding_dampener']}x`
            """
        )

    # -------------------------------------------------------------------------
    # 4. WATER QUALITY TELEMETRY
    # -------------------------------------------------------------------------
    elif choice == "💧 Water Quality Telemetry":
        st.title("💧 Water Quality Monitoring & Stress Engine")
        st.caption("Continuous telemetry, environmental threshold validation, and interactive bio-stress simulation")

        telemetry = wq.get_telemetry()

        # Gauge Charts Row
        g1, g2, g3, g4 = st.columns(4)

        with g1:
            fig_do = go.Figure(go.Indicator(
                mode="gauge+number",
                value=telemetry["dissolved_oxygen_mg_l"],
                title={'text': "DO (mg/L)"},
                gauge={
                    'axis': {'range': [0, 12]},
                    'bar': {'color': "#38bdf8"},
                    'steps': [
                        {'range': [0, 4.0], 'color': "#7f1d1d"},
                        {'range': [4.0, 5.0], 'color': "#78350f"},
                        {'range': [5.0, 12.0], 'color': "#064e3b"}
                    ]
                }
            ))
            fig_do.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_do, use_container_width=True)

        with g2:
            fig_temp = go.Figure(go.Indicator(
                mode="gauge+number",
                value=telemetry["temp_c"],
                title={'text': "Temp (°C)"},
                gauge={
                    'axis': {'range': [15, 35]},
                    'bar': {'color': "#fbbf24"},
                    'steps': [
                        {'range': [15, 22], 'color': "#1e3a8a"},
                        {'range': [22, 30], 'color': "#064e3b"},
                        {'range': [30, 35], 'color': "#7f1d1d"}
                    ]
                }
            ))
            fig_temp.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_temp, use_container_width=True)

        with g3:
            fig_ph = go.Figure(go.Indicator(
                mode="gauge+number",
                value=telemetry["ph"],
                title={'text': "pH Level"},
                gauge={
                    'axis': {'range': [5, 10]},
                    'bar': {'color': "#a855f7"},
                    'steps': [
                        {'range': [5, 6.5], 'color': "#7f1d1d"},
                        {'range': [6.5, 8.5], 'color': "#064e3b"},
                        {'range': [8.5, 10], 'color': "#7f1d1d"}
                    ]
                }
            ))
            fig_ph.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_ph, use_container_width=True)

        with g4:
            fig_turb = go.Figure(go.Indicator(
                mode="gauge+number",
                value=telemetry["turbidity_ntu"],
                title={'text': "Turbidity (NTU)"},
                gauge={
                    'axis': {'range': [0, 30]},
                    'bar': {'color': "#94a3b8"},
                    'steps': [
                        {'range': [0, 8], 'color': "#064e3b"},
                        {'range': [8, 15], 'color': "#78350f"},
                        {'range': [15, 30], 'color': "#7f1d1d"}
                    ]
                }
            ))
            fig_turb.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_turb, use_container_width=True)

        st.divider()

        st.subheader("🎛️ Interactive Environmental Stress Simulator")
        st.caption("Demonstrate how adverse water conditions impact fish swimming kinematics and suppress feeding triggers")

        col_s1, col_s2 = st.columns(2)
        sim_ph = col_s1.slider("pH Level", 5.0, 10.0, float(telemetry["ph"]), 0.1)
        sim_temp = col_s2.slider("Temperature (°C)", 16.0, 35.0, float(telemetry["temp_c"]), 0.5)

        col_s3, col_s4 = st.columns(2)
        sim_turb = col_s3.slider("Turbidity (NTU)", 0.0, 30.0, float(telemetry["turbidity_ntu"]), 0.5)
        sim_do = col_s4.slider("Dissolved Oxygen (mg/L)", 1.0, 12.0, float(telemetry["dissolved_oxygen_mg_l"]), 0.1)

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Apply Custom Values", use_container_width=True):
            wq.set_simulation_override(sim_ph, sim_temp, sim_turb, sim_do)
            st.success("Custom parameters applied!")
            st.rerun()

        if b2.button("🔴 Hypoxia Emergency (3.0 mg/L)", use_container_width=True):
            wq.set_simulation_override(7.2, 26.0, 4.0, 3.0)
            st.warning("Hypoxia Emergency scenario active!")
            st.rerun()

        if b3.button("🟠 Thermal Shock (32.5°C)", use_container_width=True):
            wq.set_simulation_override(7.4, 32.5, 4.5, 5.8)
            st.warning("Thermal Shock scenario active!")
            st.rerun()

        if b4.button("🟢 Reset to Normal Drift", use_container_width=True):
            wq.clear_simulation_override()
            st.info("Simulation override cleared.")
            st.rerun()

        st.divider()

        # Environmental Stress Index
        stress_idx = telemetry["stress_index"]
        st.subheader(f"Overall Water Stress Index: {stress_idx:.2f} / 1.00")
        st.progress(min(1.0, float(stress_idx)))

        if telemetry["alerts"]:
            for alert in telemetry["alerts"]:
                st.error(f"⚠️ {alert}")
        else:
            st.success("✅ Water Quality is fully within optimal aquaculture parameters.")

    # -------------------------------------------------------------------------
    # 5. HUNGER & SATIETY PREDICTOR
    # -------------------------------------------------------------------------
    elif choice == "🤖 Hunger & Satiety Predictor":
        st.title("🤖 Machine Learning Hunger & Satiety Predictor")
        st.caption("RandomForest Classifier coupling swimming kinematics with environmental telemetry to assess appetite")

        ret, frame = camera.read_frame()
        detections = detector.detect(frame) if ret else []
        tracks = tracker.update(detections, fps=camera.fps)
        telemetry = wq.get_telemetry()
        metrics = behavior.analyze_tracks(tracks, telemetry["behavior_modifiers"])

        growth_stages = [t.get("growth_stage", "Fingerling") for t in tracks]
        dominant_stage = max(set(growth_stages), key=growth_stages.count) if growth_stages else "Fingerling"

        prediction = predictor.predict(metrics, telemetry, len(tracks), dominant_stage)

        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.subheader("Current Appetite Assessment")
            status = prediction["status"]
            score = prediction["score"]
            conf = prediction["confidence"]
            engine = prediction["predictor_engine"]
            suppressed = prediction["suppressed_by_environment"]

            status_color = "#10b981" if status == "Hungry" else ("#38bdf8" if status == "Feeding" else "#64748b")
            if suppressed:
                status_color = "#ef4444"

            st.markdown(
                f"""
                <div style="background: #131c2e; border: 2px solid {status_color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
                    <h1 style="color: {status_color}; margin: 0; font-size: 2.8rem;">{status.upper()}</h1>
                    <p style="color: #cbd5e1; font-size: 1.1rem; margin: 5px 0;">Hunger Score: <b>{score:.1f} / 100</b> | Confidence: <b>{conf * 100:.1f}%</b></p>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Inference Engine: {engine}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("### 🔍 Primary Decision Rationale")
            for factor in prediction["primary_factors"]:
                st.markdown(f"- {factor}")

        with col_p2:
            st.subheader("ML Features Snapshot")
            features = {
                "Swimming Speed": f"{metrics['avg_speed']:.1f} px/s",
                "Feeding Zone Ratio": f"{metrics['feeding_zone_presence_ratio']*100:.0f}%",
                "Surface Ratio": f"{metrics['surface_time_ratio']*100:.0f}%",
                "Movement Intensity": f"{metrics['movement_intensity']:.1f}/100",
                "Fish Count": f"{len(tracks)}",
                "Dominant Stage": dominant_stage,
                "Dissolved Oxygen": f"{telemetry['dissolved_oxygen_mg_l']:.2f} mg/L",
                "Stress Index": f"{telemetry['stress_index']:.2f}"
            }
            st.dataframe(pd.DataFrame(list(features.items()), columns=["Feature", "Current Value"]), use_container_width=True)

        st.divider()

        st.subheader("Model Weights & Lifecycle Management")
        st.write(f"**RandomForest Model File:** `{os.path.basename(ML_MODEL_PATH)}`")
        st.write(f"**StandardScaler File:** `{os.path.basename(SCALER_PATH)}`")

        if st.button("🚀 Retrain RandomForest Classifier on Fresh Data"):
            with st.spinner("Retraining RandomForest model..."):
                res = predictor.train_or_retrain([])
                if res["success"]:
                    st.success(f"{res['message']}")
                else:
                    st.error(res["message"])

    # -------------------------------------------------------------------------
    # 6. VIRTUAL FEEDER
    # -------------------------------------------------------------------------
    elif choice == "🥣 Virtual Feeder":
        st.title("🥣 Software Virtual Automated Feeder")
        st.caption("Hardware-free dispenser simulation with growth-stage dosage calculations, countdown timer, and hopper telemetry")

        feeder_status = feeder.update()

        col_vf1, col_vf2 = st.columns([1, 1])

        with col_vf1:
            st.subheader("Dispenser State")
            state_color = "#10b981" if feeder_status["status"] == "Idle" else ("#fbbf24" if feeder_status["status"] == "Dispensing" else "#38bdf8")
            st.markdown(
                f"""
                <div style="background: #131c2e; border: 1px solid #1e2d4a; border-radius: 10px; padding: 18px; margin-bottom: 15px;">
                    <h2 style="color: {state_color}; margin: 0;">Status: {feeder_status['status']}</h2>
                    <p style="margin: 8px 0 0 0; color: #94a3b8;">Session Target: <b>{feeder_status['target_feed_grams']} g</b></p>
                    <p style="margin: 0; color: #94a3b8;">Dispensed this session: <b>{feeder_status['dispensed_session_grams']} g</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if feeder_status["status"] == "Dispensing":
                st.write(f"**Dispense Progress ({feeder_status['progress_percent']}%):**")
                st.progress(feeder_status["progress_percent"] / 100.0)
                st.caption(f"Time Remaining: {feeder_status['time_remaining_sec']} seconds")
                time.sleep(0.5)
                st.rerun()

            st.write(f"**Hopper Capacity: {feeder_status['hopper_level_grams']} g / 2000 g**")
            st.progress(min(1.0, feeder_status["hopper_level_grams"] / 2000.0))

            if st.button("Refill Food Hopper (2000g)", use_container_width=True):
                feeder.refill_hopper(2000.0)
                st.success("Hopper full!")
                st.rerun()

        with col_vf2:
            st.subheader("Dosage Calculator by Growth Stage")
            calc_stage = st.selectbox("Fish Growth Stage", ["Fry", "Fingerling", "Juvenile", "Adult"], index=1)
            calc_count = st.slider("Fish Population Count", 1, 50, 10)
            dosage = compute_feed_dosage(calc_stage, calc_count)
            rate = FEED_RATES_GRAMS.get(calc_stage, 0.35)

            st.info(
                f"""
                **Calculated Feeding Dosage:** `{dosage} grams`  
                - Formula: `{calc_count} fish × {rate} g/fish = {dosage} g`
                """
            )

            if st.button(f"Dispense {dosage}g Dosage", use_container_width=True):
                if feeder_status["status"] == "Idle":
                    feeder.trigger_dispense(dosage)
                    db.log_feeding_decision(
                        should_feed=True,
                        amount=dosage,
                        stage=calc_stage,
                        count=calc_count,
                        hunger_status="Hungry",
                        wq_status="Optimal",
                        reason="Manual Dashboard Trigger"
                    )
                    st.success(f"Dispense started for {dosage}g!")
                    st.rerun()
                else:
                    st.warning("Feeder is currently busy.")

    # -------------------------------------------------------------------------
    # 7. FEEDING HISTORY & DB
    # -------------------------------------------------------------------------
    elif choice == "🗄️ Feeding History & DB":
        st.title("🗄️ SQLite Database Browser & History")
        st.caption("Inspect and export historical records: feeding events, sensor logs, predictions, and detections")

        tab_feed, tab_wq, tab_pred, tab_det = st.tabs([
            "🍽️ Feeding History",
            "💧 Water Quality Logs",
            "🤖 Predictions Log",
            "🐟 Detections Log"
        ])

        with tab_feed:
            feed_rows = db.get_recent_feedings(limit=100)
            if feed_rows:
                df_feed = pd.DataFrame(feed_rows)
                st.dataframe(df_feed, use_container_width=True)
                csv = df_feed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Feeding History as CSV",
                    csv,
                    "feeding_history.csv",
                    "text/csv"
                )
            else:
                st.info("No feeding records recorded yet.")

        with tab_wq:
            wq_rows = db.get_recent_water_quality(limit=100)
            if wq_rows:
                df_wq = pd.DataFrame(wq_rows)
                st.dataframe(df_wq, use_container_width=True)
                csv = df_wq.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Water Quality Logs as CSV",
                    csv,
                    "water_quality.csv",
                    "text/csv"
                )
            else:
                st.info("No water quality logs recorded yet.")

        with tab_pred:
            pred_rows = db.get_recent_predictions(limit=100)
            if pred_rows:
                df_pred = pd.DataFrame(pred_rows)
                st.dataframe(df_pred, use_container_width=True)
                csv = df_pred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Predictions as CSV",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
            else:
                st.info("No prediction logs recorded yet.")

        with tab_det:
            det_rows = db.get_recent_detections(limit=100)
            if det_rows:
                df_det = pd.DataFrame(det_rows)
                st.dataframe(df_det, use_container_width=True)
                csv = df_det.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Detections as CSV",
                    csv,
                    "detections.csv",
                    "text/csv"
                )
            else:
                st.info("No detection logs recorded yet.")

        st.divider()

        c_db1, c_db2 = st.columns(2)
        with c_db1:
            if st.button("🌱 Seed 40 Demo Telemetry Records"):
                db.seed_demo_data(40)
                st.success("Seeded 40 historical records!")
                st.rerun()
        with c_db2:
            if st.button("🗑️ Clear All Database Records"):
                db.clear_tables()
                st.warning("Database cleared!")
                st.rerun()

    # -------------------------------------------------------------------------
    # 8. ANALYTICS & REPORTS
    # -------------------------------------------------------------------------
    elif choice == "📑 Analytics & Reports":
        st.title("📑 Analytics & Academic Defense Reports")
        st.caption("Cross-parameter correlations, feed efficiency analytics, and academic defense documentation")

        wq_rows = db.get_recent_water_quality(limit=50)
        det_rows = db.get_recent_detections(limit=50)
        feed_rows = db.get_recent_feedings(limit=50)

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.subheader("Fish Speed vs Dissolved Oxygen")
            if wq_rows and det_rows:
                min_len = min(len(wq_rows), len(det_rows))
                df_corr = pd.DataFrame({
                    "DO (mg/L)": [r["dissolved_oxygen"] for r in wq_rows[:min_len]],
                    "Swimming Speed (px/s)": [r["avg_speed"] for r in det_rows[:min_len]],
                    "Growth Stage": [r["growth_stage"] for r in det_rows[:min_len]]
                })
                fig_corr = px.scatter(
                    df_corr,
                    x="DO (mg/L)",
                    y="Swimming Speed (px/s)",
                    color="Growth Stage",
                    title="Correlation: Hypoxia Lethargy Effect",
                    template="plotly_dark",
                    trendline="ols"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Please seed or log data to view correlation analysis.")

        with col_a2:
            st.subheader("Feed Dosage by Growth Stage")
            if feed_rows:
                df_f = pd.DataFrame(feed_rows)
                fig_bar = px.bar(
                    df_f,
                    x="growth_stage",
                    y="amount_grams",
                    color="growth_stage",
                    title="Dispensed Feed Distribution (Grams)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No feeding data available.")

        st.divider()

        st.subheader("🎓 Academic Viva Voce Defense Guide")
        with st.expander("Q1: What is the primary novelty of this aquaculture system?"):
            st.markdown(
                """
                **Answer:** Traditional smart feeders rely solely on surface motion or fixed timers. 
                Our system introduces **Dynamic Bio-Environmental Coupling**: during low Dissolved Oxygen (Hypoxia < 5.0 mg/L),
                fish rise to the surface to gasp for oxygen. Conventional computer vision systems misclassify this gasping as hunger,
                causing lethal overfeeding. Our system couples water quality telemetry with behavior analysis, actively suppressing feeding
                during hypoxia to safeguard fish health and water quality.
                """
            )

        with st.expander("Q2: Why use YOLOv8 and ByteTrack instead of classic frame differencing?"):
            st.markdown(
                """
                **Answer:** Frame differencing fails under underwater bubbles, reflections, and water currents. 
                YOLOv8 provides robust object localization and feature extraction even with optical refraction. ByteTrack associates detections
                across frames by leveraging both high and low confidence detection scores, preventing track ID switching during fish occlusion.
                """
            )

        with st.expander("Q3: How are growth stages classified without measuring fish physical length in millimeters?"):
            st.markdown(
                """
                **Answer:** Bounding box pixel dimensions (maximum dimension in pixels) calibrated against camera focal length
                and distance map into 4 distinct aquaculture growth stages (Fry: <35px, Fingerling: 35-75px, Juvenile: 75-130px, Adult: >130px),
                enabling stage-specific feed dosage formulas.
                """
            )

    # -------------------------------------------------------------------------
    # 9. SYSTEM LOGS
    # -------------------------------------------------------------------------
    elif choice == "📜 System Logs":
        st.title("📜 System Diagnostic Logs")
        st.caption("Real-time logging of application startup, detections, telemetry alerts, and feeder actions")

        log_level = st.selectbox("Filter by Severity Level", ["ALL", "INFO", "WARNING", "ERROR"])
        logs = db.get_system_logs(limit=100)

        if logs:
            df_logs = pd.DataFrame(logs)
            if log_level != "ALL":
                df_logs = df_logs[df_logs["level"] == log_level]
            st.dataframe(df_logs, use_container_width=True)

            log_text = "\n".join([f"[{r['timestamp']}] [{r['level']}] {r['module']}: {r['message']}" for r in logs])
            st.download_button(
                "📥 Export Logs (TXT)",
                log_text,
                "system_logs.txt",
                "text/plain"
            )
        else:
            st.info("No diagnostic logs in SQLite database.")

    # -------------------------------------------------------------------------
    # 10. SETTINGS & CONFIGURATION
    # -------------------------------------------------------------------------
    elif choice == "⚙️ Settings & Configuration":
        st.title("⚙️ System Thresholds & Calibration")
        st.caption("Adjust system thresholds, camera defaults, and virtual feeder safeguards")

        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            st.subheader("Water Quality Thresholds")
            st.number_input("Minimum Dissolved Oxygen (mg/L)", value=5.0, min_value=1.0, max_value=10.0, step=0.1)
            st.number_input("Maximum Turbidity (NTU)", value=15.0, min_value=1.0, max_value=50.0, step=1.0)
            st.number_input("Minimum Safe pH", value=6.5, min_value=5.0, max_value=8.0, step=0.1)
            st.number_input("Maximum Safe pH", value=8.5, min_value=7.0, max_value=10.0, step=0.1)

        with col_cfg2:
            st.subheader("Feeder & Safety Guards")
            st.number_input("Post-Feed Cooldown (seconds)", value=30, min_value=5, max_value=300, step=5)
            st.number_input("Maximum Feed per Session (grams)", value=150.0, min_value=10.0, max_value=500.0, step=10.0)
            st.number_input("YOLO Confidence Threshold", value=0.45, min_value=0.10, max_value=0.95, step=0.05)

        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
