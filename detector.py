"""
===============================================================================
Fish_AI_Project - Module 4: Fish Detector Module (detector.py)
===============================================================================
Senior AI Engineer Implementation
Purpose: Performs real-time fish detection from Webcam or Video files using
Ultralytics YOLO11, extracts bounding boxes, confidence scores, fish counts,
and real-time FPS calculations.

Key Requirements:
1. Detect fish from:
   - Laptop Webcam (index 0 or external USB camera)
   - Video file (e.g., MP4, AVI video clips)
2. Display & Calculate:
   - Bounding boxes (coordinates [x1, y1, x2, y2])
   - Confidence score per fish (0.00 - 1.00)
   - Fish count (total detected fish)
   - Real-time FPS (Frames Per Second)
3. Provide resilient fallback detection for offline laptop environments.
4. Support standalone CLI execution with webcam, video, or synthetic input.

Adheres to:
- Object-Oriented Architecture (FishDetector, FishVideoDetector)
- PEP-8 Standards & Comprehensive Docstrings
- Robust Logging and Exception Handling
- 100% Executable Standalone Script
===============================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Generator

import cv2
import numpy as np

# Try importing ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import configuration constants
try:
    from config import (
        YOLO_MODEL_PATH,
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        GROWTH_STAGE_THRESHOLDS,
        FEEDING_ZONE,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        TARGET_FPS,
        SAMPLE_VIDEOS_DIR
    )
except ImportError:
    YOLO_MODEL_PATH = "models/yolo11/yolo11n.pt"
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.50
    GROWTH_STAGE_THRESHOLDS = {
        "Fry": (0, 35),
        "Fingerling": (35, 75),
        "Juvenile": (75, 130),
        "Adult": (130, 9999)
    }
    FEEDING_ZONE = {"x1": 0.35, "y1": 0.10, "x2": 0.65, "y2": 0.45}
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30
    SAMPLE_VIDEOS_DIR = Path("data/sample_videos")


logger = logging.getLogger("FishDetector")


class FishDetector:
    """
    YOLO11 Fish Detection, Bounding Box Extraction, Confidence Scoring,
    and Growth Stage Estimator.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        iou_thresh: float = IOU_THRESHOLD
    ) -> None:
        """
        Initialize the FishDetector.

        Args:
            model_path: Path to YOLO weights (.pt). Defaults to YOLO_MODEL_PATH.
            conf_thresh: Minimum confidence threshold for positive detection.
            iou_thresh: Intersection Over Union (IoU) threshold for NMS.
        """
        self.model_path = model_path or YOLO_MODEL_PATH
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.is_deep_learning_active = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempts to load trained YOLO11 model weights with graceful fallback."""
        if not YOLO_AVAILABLE:
            logger.info("Ultralytics library not present; using heuristic computer-vision detector.")
            return

        resolved_path = Path(self.model_path).resolve()
        if not resolved_path.is_file():
            # Check for alternative best.pt or yolo11n.pt
            alt_best = resolved_path.parent / "best.pt"
            if alt_best.is_file():
                resolved_path = alt_best

        if resolved_path.is_file():
            try:
                self.model = YOLO(str(resolved_path))
                self.is_deep_learning_active = True
                logger.info(f"Loaded YOLO11 model successfully from: {resolved_path}")
            except Exception as e:
                logger.warning(f"Could not load custom YOLO11 model from {resolved_path} ({e}). Using CV detector fallback.")
        else:
            logger.info(f"Model file {resolved_path} not found. Operating with resilient CV detector.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects fish in a video frame.

        Args:
            frame: Input BGR image matrix.

        Returns:
            List of detected fish dictionaries with bounding box [x1, y1, x2, y2],
            confidence score, fish class name, dimensions, center, and growth stage.
        """
        if frame is None:
            return []

        height, width, _ = frame.shape
        detections: List[Dict[str, Any]] = []

        if self.is_deep_learning_active and self.model is not None:
            try:
                results = self.model(
                    frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    verbose=False
                )
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names.get(cls_id, "fish") if hasattr(self.model, "names") else "fish"

                        box_width = max(1, x2 - x1)
                        box_height = max(1, y2 - y1)
                        max_dim = max(box_width, box_height)
                        growth_stage = self._classify_growth_stage(max_dim)

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(conf, 3),
                            "class_name": cls_name,
                            "width": box_width,
                            "height": box_height,
                            "growth_stage": growth_stage,
                            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        })
                return detections
            except Exception as e:
                logger.error(f"Error during YOLO11 inference: {e}. Falling back to heuristic CV detection.")

        # Heuristic computer-vision contour fallback
        return self._heuristic_contour_detection(frame)

    def _classify_growth_stage(self, dimension_px: int) -> str:
        """
        Classifies fish growth stage based on bounding box dimension in pixels.

        Args:
            dimension_px: Maximum dimension (width or height) in pixels.

        Returns:
            Growth stage label ('Fry', 'Fingerling', 'Juvenile', 'Adult').
        """
        for stage, (min_dim, max_dim) in GROWTH_STAGE_THRESHOLDS.items():
            if min_dim <= dimension_px < max_dim:
                return stage
        return "Adult"

    def _heuristic_contour_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Resilient computer vision contour detection fallback for testing
        without deep learning GPU weights.
        """
        detections: List[Dict[str, Any]] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask underwater fish color range (orange/gold/cyan/green contrasts)
        mask1 = cv2.inRange(hsv, np.array([5, 60, 60]), np.array([30, 255, 255]))    # Warm orange/gold
        mask2 = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([130, 255, 255]))  # Green/cyan
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                max_dim = max(w, h)
                stage = self._classify_growth_stage(max_dim)
                confidence = round(min(0.98, max(0.55, 0.65 + (area / 3500.0))), 2)

                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": confidence,
                    "class_name": "fish",
                    "width": w,
                    "height": h,
                    "growth_stage": stage,
                    "center": (x + w // 2, y + h // 2)
                })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float = 0.0,
        draw_zones: bool = True
    ) -> np.ndarray:
        """
        Renders HUD with Bounding Boxes, Confidence Scores, Fish Count, and FPS.

        Args:
            frame: Input image frame.
            detections: List of detection dictionaries.
            fps: Current measured Frames Per Second.
            draw_zones: Whether to draw feeding and surface zones.

        Returns:
            Annotated BGR numpy image.
        """
        annotated = frame.copy()
        h, w, _ = frame.shape
        fish_count = len(detections)

        # 1. Draw Operational Zones
        if draw_zones:
            # Feeding Zone
            fz_x1 = int(FEEDING_ZONE["x1"] * w)
            fz_y1 = int(FEEDING_ZONE["y1"] * h)
            fz_x2 = int(FEEDING_ZONE["x2"] * w)
            fz_y2 = int(FEEDING_ZONE["y2"] * h)
            cv2.rectangle(annotated, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 220, 100), 2)
            cv2.putText(
                annotated, "FEEDING ZONE",
                (fz_x1 + 8, fz_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA
            )

            # Surface Water Boundary (top 25%)
            surf_y = int(h * 0.25)
            cv2.line(annotated, (0, surf_y), (w, surf_y), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(
                annotated, "SURFACE WATER ZONE (Y < 25%)",
                (15, surf_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA
            )

        # 2. Draw Fish Bounding Boxes & Confidence Badges
        stage_colors = {
            "Fry": (255, 180, 50),        # Gold
            "Fingerling": (50, 220, 255),  # Cyan
            "Juvenile": (180, 100, 255),  # Purple
            "Adult": (50, 255, 100)        # Bright Green
        }

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            stage = d.get("growth_stage", "Adult")
            conf = d.get("confidence", 0.0)
            color = stage_colors.get(stage, (0, 255, 0))

            # Draw Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw Center Dot
            cx, cy = d.get("center", (int((x1 + x2) / 2), int((y1 + y2) / 2)))
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Draw Label Tag: Class | Conf%
            label = f"fish: {conf:.2f} ({stage})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            bg_y1 = max(0, y1 - text_size[1] - 8)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_size[0] + 6, bg_y2), (20, 20, 25), -1)
            cv2.putText(
                annotated, label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

        # 3. Top HUD: FPS & Detection Status
        hud_bg_h = 42
        cv2.rectangle(annotated, (0, 0), (w, hud_bg_h), (15, 20, 30), -1)
        cv2.line(annotated, (0, hud_bg_h), (w, hud_bg_h), (50, 60, 75), 1)

        # FPS indicator
        fps_color = (0, 255, 120) if fps >= 20.0 else ((0, 200, 255) if fps >= 10.0 else (0, 100, 255))
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(annotated, fps_str, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

        # Fish count indicator
        count_str = f"Fish Count: {fish_count}"
        cv2.putText(annotated, count_str, (180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Architecture & Model badge
        mode_str = "Engine: YOLO11 [Active]" if self.is_deep_learning_active else "Engine: CV Heuristic [Fallback]"
        mode_color = (0, 220, 100) if self.is_deep_learning_active else (255, 200, 0)
        cv2.putText(annotated, mode_str, (w - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

        return annotated

    def detect_and_annotate(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        draw_zones: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convenience wrapper that runs detection, annotates the frame with HUD,
        and returns structured summary telemetry.

        Args:
            frame: Input BGR frame.
            fps: Current frame rate.
            draw_zones: Whether to draw operational zones.

        Returns:
            Tuple of:
            - annotated frame (numpy array)
            - list of detection dicts
            - summary dictionary (fish_count, fps, avg_confidence)
        """
        detections = self.detect(frame)
        annotated_frame = self.draw_detections(frame, detections, fps=fps, draw_zones=draw_zones)

        avg_conf = 0.0
        if detections:
            avg_conf = round(float(np.mean([d["confidence"] for d in detections])), 3)

        summary = {
            "fish_count": len(detections),
            "fps": round(fps, 1),
            "avg_confidence": avg_conf,
            "timestamp": time.time()
        }

        return annotated_frame, detections, summary


class FishVideoDetector:
    """
    High-level Stream Processor for detecting fish from live Webcam or Video files.
    """

    def __init__(
        self,
        detector: Optional[FishDetector] = None,
        source: Union[int, str] = 0,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        target_fps: int = TARGET_FPS
    ) -> None:
        """
        Initialize the FishVideoDetector.

        Args:
            detector: Instance of FishDetector.
            source: Webcam index (0), video file path, or 'synthetic'.
            width: Target frame width.
            height: Target frame height.
            target_fps: Target frame rate.
        """
        self.detector = detector or FishDetector()
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.is_running = False

    def open(self) -> bool:
        """Opens video capture device or prepares synthetic stream."""
        if str(self.source).lower() == "synthetic":
            logger.info("FishVideoDetector operating in synthetic stream mode.")
            self.is_running = True
            return True

        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.warning(f"Could not open source {self.source}. Falling back to synthetic stream.")
                self.source = "synthetic"
                self.is_running = True
                return True

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.is_running = True
            logger.info(f"Opened video stream from: {self.source}")
            return True

        except Exception as e:
            logger.error(f"Error opening video source: {e}. Falling back to synthetic.")
            self.source = "synthetic"
            self.is_running = True
            return True

    def _update_fps(self) -> None:
        """Calculates running exponential moving average FPS."""
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.85 * self.fps + 0.15 * instant_fps
        self.last_time = now

    def process_next_frame(self) -> Tuple[bool, Optional[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Reads next frame from source, runs detection, computes FPS, and annotates.

        Returns:
            Tuple of (success, annotated_frame, detections, summary_stats).
        """
        if not self.is_running:
            return False, None, [], {}

        self._update_fps()
        raw_frame = None

        if self.source == "synthetic":
            from camera import CameraModule
            if not hasattr(self, "_synthetic_camera"):
                self._synthetic_camera = CameraModule(source="synthetic", width=self.width, height=self.height)
                self._synthetic_camera.start()
            _, raw_frame = self._synthetic_camera.read_frame()

        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # If reading video file, loop back to start
                if isinstance(self.source, str) and Path(self.source).is_file():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret:
                    logger.warning("End of video stream or unreadable frame.")
                    return False, None, [], {}

            raw_frame = cv2.resize(frame, (self.width, self.height))

        if raw_frame is None:
            return False, None, [], {}

        annotated_frame, detections, summary = self.detector.detect_and_annotate(
            frame=raw_frame,
            fps=self.fps
        )

        return True, annotated_frame, detections, summary

    def stream_detections(
        self,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]], None, None]:
        """
        Streams detected and annotated frames as a generator.

        Args:
            max_frames: Optional limit on number of frames to yield.

        Yields:
            Tuple of (annotated_frame, detections, summary_stats).
        """
        if not self.is_running:
            self.open()

        count = 0
        try:
            while self.is_running:
                success, frame, dets, summary = self.process_next_frame()
                if not success or frame is None:
                    break

                yield frame, dets, summary
                count += 1

                if max_frames is not None and count >= max_frames:
                    break
        finally:
            self.close()

    def close(self) -> None:
        """Releases capture resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, "_synthetic_camera"):
            self._synthetic_camera.release()
        logger.info("FishVideoDetector stream closed.")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Module 4: Fish Detection from Webcam or Video File using YOLO11"
    )
    parser.add_argument(
        "--source",
        default="data/sample_videos/fish_sample.mp4",
        help="Input source: '0' for webcam, path to video file, or 'synthetic'"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Number of frames to process (default: 60)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional destination path to save annotated output video (.mp4)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    args = parse_arguments()
    print("=" * 65)
    print("Running Module 4: Fish Detection from Webcam / Video File")
    print("=" * 65)

    # Determine source type
    src: Union[int, str]
    if args.source.isdigit():
        src = int(args.source)
    elif args.source == "synthetic":
        src = "synthetic"
    else:
        # Check if video exists, otherwise fallback to sample video or synthetic
        src_path = Path(args.source)
        if not src_path.is_file():
            default_sample = SAMPLE_VIDEOS_DIR / "fish_sample.mp4"
            if default_sample.is_file():
                src = str(default_sample)
                print(f"[*] Provided source not found. Using default sample video: {src}")
            else:
                src = "synthetic"
                print("[*] No sample video found. Using synthetic camera mode.")
        else:
            src = str(src_path)

    print(f"Selected Input Source: {src}")
    print(f"Frames to Process:    {args.max_frames}")

    detector = FishDetector()
    video_processor = FishVideoDetector(detector=detector, source=src)
    video_processor.open()

    total_fish_detected = 0
    total_frames = 0
    fps_records = []

    out_writer = None
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_p), fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        for frame, detections, summary in video_processor.stream_detections(max_frames=args.max_frames):
            total_frames += 1
            fish_count = summary["fish_count"]
            fps_val = summary["fps"]
            total_fish_detected += fish_count
            fps_records.append(fps_val)

            if total_frames % 10 == 0 or total_frames == args.max_frames:
                print(
                    f"  Frame {total_frames:03d} | Fish Count: {fish_count} | "
                    f"FPS: {fps_val:4.1f} | Avg Conf: {summary['avg_confidence']:.2f}"
                )

            if out_writer is not None:
                out_writer.write(frame)

    finally:
        video_processor.close()
        if out_writer is not None:
            out_writer.release()
            print(f"[+] Output annotated video saved to: {args.output}")

    avg_fps = np.mean(fps_records) if fps_records else 0.0
    avg_count = total_fish_detected / max(1, total_frames)

    print("\nDetection Summary:")
    print(f"  • Total Frames Processed: {total_frames}")
    print(f"  • Average Real-Time FPS:  {avg_fps:.1f}")
    print(f"  • Average Fish Count:     {avg_count:.2f}")
    print("\nModule 4 fish detection pipeline executed successfully.")
