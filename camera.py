"""
===============================================================================
Fish_AI_Project - Camera Module (camera.py)
===============================================================================
Handles live webcam acquisition, video file streaming, or synthetic tank
simulation. Includes FPS measurement, frame resizing, and error handling.
"""

import time
import math
import random
import logging
from typing import Optional, Tuple, Generator
import cv2
import numpy as np
from config import WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

logger = logging.getLogger(__name__)


class CameraModule:
    """Manages video capture hardware or simulated video feeds."""

    def __init__(
        self,
        source: Optional[int | str] = WEBCAM_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = TARGET_FPS
    ) -> None:
        """Initialize camera stream parameters.

        Args:
            source: Webcam index (0), video file path, or None for synthetic mode.
            width: Target frame width.
            height: Target frame height.
            fps: Desired frame rate.
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

        # FPS metrics
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()

        # Synthetic fish state for virtual camera mode
        self._init_synthetic_fish()

    def _init_synthetic_fish(self) -> None:
        """Initialize positions and velocities for synthetic fish simulation."""
        self.synthetic_fish = []
        num_fish = 6
        stages = ["Fry", "Fingerling", "Juvenile", "Adult"]

        for i in range(num_fish):
            stage = stages[i % len(stages)]
            speed_mult = 3.5 if stage == "Fry" else (2.5 if stage == "Fingerling" else 1.8)
            self.synthetic_fish.append({
                "id": i + 1,
                "x": random.randint(150, self.width - 150),
                "y": random.randint(150, self.height - 150),
                "vx": random.uniform(-speed_mult, speed_mult),
                "vy": random.uniform(-speed_mult, speed_mult),
                "stage": stage,
                "size": 25 if stage == "Fry" else (50 if stage == "Fingerling" else (90 if stage == "Juvenile" else 140))
            })

    def start(self) -> bool:
        """Start the video capture device or simulation generator."""
        if self.source is None or str(self.source).lower() == "synthetic":
            logger.info("Camera initialized in Synthetic Simulation Mode.")
            self.is_running = True
            return True

        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.warning(f"Could not open camera source: {self.source}. Falling back to synthetic stream.")
                self.source = "synthetic"
                self.is_running = True
                return True

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.is_running = True
            logger.info(f"Camera opened successfully on source {self.source}.")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.source = "synthetic"
            self.is_running = True
            return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single video frame and update FPS calculations."""
        if not self.is_running:
            return False, None

        self._update_fps()

        if self.source == "synthetic":
            frame = self._generate_synthetic_frame()
            return True, frame

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop video file if applicable or fallback to synthetic
                if isinstance(self.source, str) and self.source != "synthetic":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to grab frame from video source. Switching to synthetic mode.")
                    self.source = "synthetic"
                    return True, self._generate_synthetic_frame()

            frame = cv2.resize(frame, (self.width, self.height))
            return True, frame

        return True, self._generate_synthetic_frame()

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic underwater tank video frame with moving fish graphics."""
        # Dark oceanic gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            ratio = y / self.height
            b = int(40 * (1 - ratio) + 15 * ratio)
            g = int(70 * (1 - ratio) + 30 * ratio)
            r = int(25 * (1 - ratio) + 10 * ratio)
            frame[y, :] = (b, g, r)

        # Draw water surface line
        cv2.line(frame, (0, int(self.height * 0.25)), (self.width, int(self.height * 0.25)), (100, 180, 220), 1)

        # Update synthetic fish positions
        for fish in self.synthetic_fish:
            fish["x"] += fish["vx"]
            fish["y"] += fish["vy"]

            # Bounce off walls
            if fish["x"] < 50 or fish["x"] > self.width - 50:
                fish["vx"] *= -1
            if fish["y"] < 50 or fish["y"] > self.height - 50:
                fish["vy"] *= -1

            # Draw fish body (ellipse + tail)
            center = (int(fish["x"]), int(fish["y"]))
            axes = (int(fish["size"] // 2), int(fish["size"] // 4))
            angle = math.degrees(math.atan2(fish["vy"], fish["vx"]))

            color = (80, 200, 120) if fish["stage"] == "Adult" else (100, 160, 240)
            cv2.ellipse(frame, center, axes, angle, 0, 360, color, -1)

            # Draw eye
            eye_x = int(fish["x"] + (axes[0] * 0.6) * math.cos(math.radians(angle)))
            eye_y = int(fish["y"] + (axes[0] * 0.6) * math.sin(math.radians(angle)))
            cv2.circle(frame, (eye_x, eye_y), 3, (255, 255, 255), -1)
            cv2.circle(frame, (eye_x, eye_y), 1, (0, 0, 0), -1)

        # Draw FPS overlay
        fps_text = f"FPS: {self.fps:.1f} | Source: {self.source}"
        cv2.putText(frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        return frame

    def _update_fps(self) -> None:
        """Calculate running average FPS."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            self.fps = 0.9 * self.fps + 0.1 * current_fps
        self.last_frame_time = now

    def release(self) -> None:
        """Release camera resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera resources released.")
