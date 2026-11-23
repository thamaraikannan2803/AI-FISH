"""
===============================================================================
Fish_AI_Project - Module 2: Fish Dataset Loader (dataset_loader.py)
===============================================================================
Senior AI Engineer Implementation
Purpose: Comprehensive dataset loader, video frame extractor, and synthetic
aquarium generator for the AI Smart Fish Feeding System (Laptop Only).

Key Capabilities:
1. Load fish images (JPEG, PNG) with normalization, color space conversion, and resizing.
2. Load and stream fish videos (MP4, AVI) with OpenCV, extracting resolution, FPS, and metadata.
3. Extract and save frames from fish videos at customizable sampling intervals.
4. Programmatically generate synthetic fish datasets with ground-truth YOLO labels
   (class 0: fish) and realistic swimming fish video clips (.mp4) for 100% hardware-free,
   offline laptop execution.
5. Provide dataset statistics, validation, and Matplotlib visualizer for inspection.

Adheres to:
- Object-Oriented Architecture (FishDatasetLoader)
- PEP-8 Standards & Comprehensive Docstrings
- Robust Logging and Exception Handling
- 100% Executable Standalone Script
===============================================================================
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Any, Union

import cv2
import numpy as np

# Configure writable cache directory for Matplotlib on Windows
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "mpl_cache"))
import matplotlib
# Use Agg backend for non-interactive headless compatibility, switchable when needed
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import configuration constants
try:
    from config import (
        BASE_DIR,
        DATA_DIR,
        DATASET_DIR,
        RAW_IMAGES_DIR,
        RAW_VIDEOS_DIR,
        SAMPLE_VIDEOS_DIR,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        TARGET_FPS
    )
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    DATASET_DIR = DATA_DIR / "dataset"
    RAW_IMAGES_DIR = DATA_DIR / "raw_images"
    RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
    SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 30


class FishDatasetLoader:
    """
    Object-Oriented Manager for loading, extracting, generating, and inspecting
    fish image and video datasets for YOLO11 training and inference.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ) -> None:
        """
        Initialize the FishDatasetLoader.

        Args:
            base_dir: Base directory of the project.
            log_level: Logging verbosity level.
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else BASE_DIR
        self.data_dir = self.base_dir / "data"
        self.dataset_dir = self.data_dir / "dataset"
        self.sample_videos_dir = self.data_dir / "sample_videos"
        self.raw_images_dir = self.data_dir / "raw_images"

        self.logger = self._configure_logger(log_level)
        self._ensure_directories()

    def _configure_logger(self, log_level: int) -> logging.Logger:
        """Sets up dedicated logger with clean formatting."""
        logger = logging.getLogger("FishDatasetLoader")
        logger.setLevel(log_level)
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _ensure_directories(self) -> None:
        """Ensures all requisite directories exist."""
        for d in [
            self.data_dir,
            self.dataset_dir,
            self.dataset_dir / "images" / "train",
            self.dataset_dir / "images" / "val",
            self.dataset_dir / "images" / "test",
            self.dataset_dir / "labels" / "train",
            self.dataset_dir / "labels" / "val",
            self.dataset_dir / "labels" / "test",
            self.sample_videos_dir,
            self.raw_images_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Image Operations
    # -------------------------------------------------------------------------

    def load_image(
        self,
        image_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Loads an image from disk, converts to RGB color space, and optionally resizes.

        Args:
            image_path: Path to target image.
            target_size: Optional (width, height) tuple to resize to.

        Returns:
            RGB numpy array image.

        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.is_file():
            self.logger.error(f"Image not found at path: {path}")
            raise FileNotFoundError(f"Image file not found: {path}")

        bgr_image = cv2.imread(str(path))
        if bgr_image is None:
            self.logger.error(f"Failed to decode image: {path}")
            raise ValueError(f"Could not read/decode image file: {path}")

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if target_size is not None:
            rgb_image = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)

        return rgb_image

    def load_images_from_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        target_size: Optional[Tuple[int, int]] = None,
        max_images: Optional[int] = None
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Loads all matching images from a directory.

        Args:
            dir_path: Target directory.
            extensions: Tuple of supported file extensions.
            target_size: Optional (width, height) to resize images.
            max_images: Optional maximum number of images to load.

        Returns:
            List of (Path, RGB numpy array) tuples.
        """
        directory = Path(dir_path)
        if not directory.is_dir():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []

        loaded: List[Tuple[Path, np.ndarray]] = []
        files = [f for f in sorted(directory.iterdir()) if f.suffix.lower() in extensions]

        if max_images is not None:
            files = files[:max_images]

        for file_path in files:
            try:
                img = self.load_image(file_path, target_size=target_size)
                loaded.append((file_path, img))
            except Exception as e:
                self.logger.warning(f"Skipping corrupt or unreadable image {file_path.name}: {e}")

        self.logger.info(f"Loaded {len(loaded)} images from {directory.name}")
        return loaded

    # -------------------------------------------------------------------------
    # Video Operations
    # -------------------------------------------------------------------------

    def get_video_metadata(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extracts dimensional, frame rate, and duration metadata from a video file.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with width, height, fps, total_frames, duration_sec.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            metadata = {
                "file_name": path.name,
                "file_path": str(path),
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_sec": round(duration_sec, 2)
            }
            return metadata
        finally:
            cap.release()

    def stream_video_frames(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        sample_interval: int = 1,
        convert_rgb: bool = True
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields frames sequentially from a video file as a generator.

        Args:
            video_path: Path to video file.
            max_frames: Maximum number of frames to yield.
            sample_interval: Step interval (1 = every frame, 2 = every 2nd frame, etc.).
            convert_rgb: Whether to convert frames to RGB (defaults to True).

        Yields:
            Tuple of (frame_index, numpy array frame).
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        frame_idx = 0
        yielded_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    if convert_rgb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield (frame_idx, frame)
                    yielded_count += 1

                    if max_frames is not None and yielded_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

    def extract_frames_to_disk(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        sample_interval: int = 5,
        max_frames: Optional[int] = None,
        prefix: str = "frame"
    ) -> List[Path]:
        """
        Extracts frames from a video and saves them as JPEG images in output_dir.

        Args:
            video_path: Source video path.
            output_dir: Target folder to save extracted frames.
            sample_interval: Frame sampling interval.
            max_frames: Optional limit on extracted frames.
            prefix: Filename prefix for saved frames.

        Returns:
            List of Paths to saved image frames.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        for idx, rgb_frame in self.stream_video_frames(
            video_path=video_path,
            max_frames=max_frames,
            sample_interval=sample_interval,
            convert_rgb=False  # Keep BGR for cv2.imwrite
        ):
            frame_filename = out_path / f"{prefix}_{idx:05d}.jpg"
            success = cv2.imwrite(str(frame_filename), rgb_frame)
            if success:
                saved_paths.append(frame_filename)

        self.logger.info(f"Extracted {len(saved_paths)} frames from {Path(video_path).name} into {out_path.name}")
        return saved_paths

    # -------------------------------------------------------------------------
    # Synthetic Aquarium Dataset & Video Generation
    # -------------------------------------------------------------------------

    def generate_synthetic_fish_image(
        self,
        canvas_size: Tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH),
        num_fish: int = 3,
        fish_size_range: Tuple[int, int] = (40, 110)
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Draws a photorealistic synthetic underwater tank scene with swimming fish,
        generating exact ground-truth bounding box coordinates in YOLO format.

        Args:
            canvas_size: (height, width) of generated canvas.
            num_fish: Number of fish to draw.
            fish_size_range: (min_length, max_length) in pixels.

        Returns:
            Tuple of:
            - RGB numpy array of rendered scene.
            - List of bounding box dicts: {'class_id': 0, 'bbox': (x1, y1, x2, y2),
              'yolo_bbox': (x_center_norm, y_center_norm, w_norm, h_norm)}
        """
        height, width = canvas_size

        # 1. Create underwater blue/cyan water gradient background
        gradient_y = np.linspace(0.85, 0.25, height)[:, np.newaxis, np.newaxis]
        top_color = np.array([12, 45, 85], dtype=np.float32)[np.newaxis, np.newaxis, :]
        bottom_color = np.array([20, 95, 130], dtype=np.float32)[np.newaxis, np.newaxis, :]
        gradient_rgb = (gradient_y * top_color + (1.0 - gradient_y) * bottom_color).astype(np.uint8)
        canvas = np.repeat(gradient_rgb, width, axis=1)

        # 2. Add subtle lighting caustics and ambient water bubbles
        for _ in range(35):
            bx = random.randint(20, width - 20)
            by = random.randint(20, height - 20)
            br = random.randint(2, 6)
            cv2.circle(canvas, (bx, by), br, (220, 240, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (bx + 1, by - 1), 1, (255, 255, 255), -1)

        # 3. Add gravel tank substrate at bottom
        substrate_height = int(height * 0.08)
        canvas[height - substrate_height:, :] = np.clip(
            canvas[height - substrate_height:, :] * 0.4 + np.array([30, 45, 55]), 0, 255
        ).astype(np.uint8)

        # 4. Render fish with realistic anatomical features (body, fins, eye, lateral line)
        detections: List[Dict[str, Any]] = []

        fish_palette = [
            {"body": (245, 130, 32), "stripe": (255, 255, 255), "fin": (230, 100, 20)},  # Clownfish orange
            {"body": (210, 50, 45),  "stripe": (255, 215, 0),   "fin": (180, 30, 25)},   # Crimson / Gold
            {"body": (40, 160, 220), "stripe": (240, 240, 60),  "fin": (30, 120, 190)},  # Tang blue / yellow
            {"body": (220, 220, 230), "stripe": (200, 40, 40),  "fin": (200, 200, 210)}, # Silver / Red Carp
            {"body": (50, 190, 130), "stripe": (255, 255, 255), "fin": (35, 150, 100)}   # Emerald green
        ]

        # Restrict fish placement within water column (avoiding substrate and extreme edges)
        y_min_bound = int(height * 0.12)
        y_max_bound = int(height * 0.85)

        for _ in range(num_fish):
            fish_len = random.randint(fish_size_range[0], fish_size_range[1])
            fish_h = int(fish_len * random.uniform(0.38, 0.52))

            cx = random.randint(fish_len + 40, width - fish_len - 40)
            cy = random.randint(y_min_bound, y_max_bound)
            heading = random.choice([-1, 1])  # 1 = facing right, -1 = facing left
            palette = random.choice(fish_palette)

            # Draw Caudal Fin (Tail)
            tail_tip_x = cx - heading * int(fish_len * 0.65)
            tail_top_y = cy - int(fish_h * 0.70)
            tail_bot_y = cy + int(fish_h * 0.70)
            tail_joint_x = cx - heading * int(fish_len * 0.35)
            tail_pts = np.array([
                [tail_joint_x, cy],
                [tail_tip_x, tail_top_y],
                [tail_tip_x, tail_bot_y]
            ], np.int32)
            cv2.fillPoly(canvas, [tail_pts], palette["fin"], lineType=cv2.LINE_AA)

            # Draw Main Body Ellipse
            axes = (int(fish_len * 0.45), int(fish_h * 0.45))
            cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, palette["body"], -1, cv2.LINE_AA)

            # Draw Anatomical Pattern Stripes
            stripe_x = cx - heading * int(fish_len * 0.05)
            cv2.ellipse(canvas, (stripe_x, cy), (int(axes[0] * 0.35), axes[1] - 2), 0, 0, 360, palette["stripe"], 3, cv2.LINE_AA)

            # Draw Dorsal Fin
            dorsal_pts = np.array([
                [cx - heading * int(fish_len * 0.15), cy - int(fish_h * 0.40)],
                [cx + heading * int(fish_len * 0.05), cy - int(fish_h * 0.80)],
                [cx + heading * int(fish_len * 0.20), cy - int(fish_h * 0.38)]
            ], np.int32)
            cv2.fillPoly(canvas, [dorsal_pts], palette["fin"], lineType=cv2.LINE_AA)

            # Draw Eye with Pupil
            eye_x = cx + heading * int(fish_len * 0.28)
            eye_y = cy - int(fish_h * 0.12)
            eye_r = max(2, int(fish_h * 0.12))
            cv2.circle(canvas, (eye_x, eye_y), eye_r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (eye_x + heading * 1, eye_y), max(1, eye_r // 2), (10, 10, 15), -1, cv2.LINE_AA)

            # Calculate tight bounding box covering body and fins
            x1 = max(0, min(tail_tip_x, cx - axes[0] - 2))
            x2 = min(width - 1, max(tail_tip_x, cx + axes[0] + 2))
            y1 = max(0, min(tail_top_y, cy - int(fish_h * 0.82)))
            y2 = min(height - 1, max(tail_bot_y, cy + axes[1] + 2))

            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Compute YOLO normalized coordinates (class 0: fish)
            box_w = x2 - x1
            box_h = y2 - y1
            x_center_norm = (x1 + box_w / 2.0) / width
            y_center_norm = (y1 + box_h / 2.0) / height
            w_norm = box_w / width
            h_norm = box_h / height

            detections.append({
                "class_id": 0,
                "class_name": "fish",
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "yolo_bbox": (
                    round(float(x_center_norm), 6),
                    round(float(y_center_norm), 6),
                    round(float(w_norm), 6),
                    round(float(h_norm), 6)
                )
            })

        return canvas, detections

    def generate_synthetic_fish_video(
        self,
        output_path: Union[str, Path],
        num_frames: int = 120,
        fps: int = TARGET_FPS,
        canvas_size: Tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH),
        num_fish: int = 4
    ) -> Path:
        """
        Renders a dynamic, kinematic video file (.mp4) of fish swimming in an aquarium tank.
        Runs completely locally with OpenCV VideoWriter (no hardware or camera required).

        Args:
            output_path: Target video file path (.mp4).
            num_frames: Total video frames to generate.
            fps: Video frames per second.
            canvas_size: (height, width) frame resolution.
            num_fish: Number of swimming fish agents.

        Returns:
            Path to generated video.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        height, width = canvas_size

        # Initialize synthetic fish physics states
        fish_states = []
        for i in range(num_fish):
            fish_states.append({
                "x": float(random.randint(150, width - 150)),
                "y": float(random.randint(120, height - 120)),
                "vx": float(random.choice([-1, 1])) * random.uniform(2.5, 5.0),
                "vy": float(random.uniform(-1.0, 1.0)),
                "length": random.randint(65, 120),
                "phase": random.uniform(0, 2 * math.pi)
            })

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"OpenCV failed to open VideoWriter for {out_file}")

        try:
            self.logger.info(f"Generating synthetic fish video: {out_file.name} ({num_frames} frames)...")
            for frame_idx in range(num_frames):
                # Gradient tank background
                gradient_y = np.linspace(0.85, 0.25, height)[:, np.newaxis, np.newaxis]
                top_color = np.array([12, 45, 85], dtype=np.float32)[np.newaxis, np.newaxis, :]
                bottom_color = np.array([20, 95, 130], dtype=np.float32)[np.newaxis, np.newaxis, :]
                gradient_rgb = (gradient_y * top_color + (1.0 - gradient_y) * bottom_color).astype(np.uint8)
                frame = np.repeat(gradient_rgb, width, axis=1)

                # Update and render each fish
                for fish in fish_states:
                    fish["x"] += fish["vx"]
                    fish["y"] += fish["vy"] + 0.6 * math.sin(frame_idx * 0.15 + fish["phase"])

                    # Bounce boundaries with soft margin
                    if fish["x"] < 100:
                        fish["x"] = 100
                        fish["vx"] = abs(fish["vx"])
                    elif fish["x"] > width - 100:
                        fish["x"] = width - 100
                        fish["vx"] = -abs(fish["vx"])

                    if fish["y"] < 80:
                        fish["y"] = 80
                        fish["vy"] = abs(fish["vy"])
                    elif fish["y"] > height - 100:
                        fish["y"] = height - 100
                        fish["vy"] = -abs(fish["vy"])

                    cx = int(fish["x"])
                    cy = int(fish["y"])
                    fish_len = fish["length"]
                    fish_h = int(fish_len * 0.44)
                    heading = 1 if fish["vx"] >= 0 else -1

                    # Body & fins
                    tail_wiggle = int(4 * math.sin(frame_idx * 0.35 + fish["phase"]))
                    tail_tip_x = cx - heading * int(fish_len * 0.65)
                    tail_pts = np.array([
                        [cx - heading * int(fish_len * 0.35), cy],
                        [tail_tip_x, cy - int(fish_h * 0.65) + tail_wiggle],
                        [tail_tip_x, cy + int(fish_h * 0.65) + tail_wiggle]
                    ], np.int32)
                    cv2.fillPoly(frame, [tail_pts], (230, 100, 20), lineType=cv2.LINE_AA)

                    # Body
                    cv2.ellipse(frame, (cx, cy), (int(fish_len * 0.45), int(fish_h * 0.45)), 0, 0, 360, (245, 130, 32), -1, cv2.LINE_AA)
                    # Stripe
                    cv2.ellipse(frame, (cx - heading * int(fish_len * 0.05), cy), (int(fish_len * 0.15), int(fish_h * 0.42)), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
                    # Eye
                    eye_x = cx + heading * int(fish_len * 0.28)
                    cv2.circle(frame, (eye_x, cy - int(fish_h * 0.12)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (eye_x + heading * 1, cy - int(fish_h * 0.12)), 2, (10, 10, 15), -1, cv2.LINE_AA)

                writer.write(frame)

            self.logger.info(f"Synthetic video successfully saved to {out_file}")
            return out_file
        finally:
            writer.release()

    def create_sample_dataset(
        self,
        num_train: int = 24,
        num_val: int = 6,
        num_test: int = 4,
        generate_video: bool = True
    ) -> Dict[str, int]:
        """
        Bootstraps a complete, self-contained YOLO11 fish dataset split
        (train, val, test) with images and YOLO formatted labels, plus a sample video clip.

        Args:
            num_train: Number of training images.
            num_val: Number of validation images.
            num_test: Number of testing images.
            generate_video: Whether to generate a sample video.

        Returns:
            Dictionary with counts of generated assets.
        """
        self.logger.info("Generating bootstrap dataset for YOLO11 training & verification...")
        splits = [
            ("train", num_train),
            ("val", num_val),
            ("test", num_test)
        ]

        summary = {"train": 0, "val": 0, "test": 0, "video_frames": 0}

        for split_name, count in splits:
            img_dir = self.dataset_dir / "images" / split_name
            lbl_dir = self.dataset_dir / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                img_path = img_dir / f"fish_{split_name}_{i+1:04d}.jpg"
                lbl_path = lbl_dir / f"fish_{split_name}_{i+1:04d}.txt"

                canvas_rgb, detections = self.generate_synthetic_fish_image(
                    canvas_size=(FRAME_HEIGHT, FRAME_WIDTH),
                    num_fish=random.randint(2, 6)
                )

                # Save RGB image in BGR format for cv2
                cv2.imwrite(str(img_path), cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

                # Save YOLO normalized annotations: class x_center y_center width height
                with open(lbl_path, "w", encoding="utf-8") as f:
                    for det in detections:
                        c = det["class_id"]
                        xc, yc, w, h = det["yolo_bbox"]
                        f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                summary[split_name] += 1

        if generate_video:
            video_out = self.sample_videos_dir / "fish_sample.mp4"
            self.generate_synthetic_fish_video(
                output_path=video_out,
                num_frames=120,
                fps=TARGET_FPS,
                num_fish=4
            )
            summary["video_frames"] = 120

        self.logger.info(
            f"Dataset creation complete: {summary['train']} train, {summary['val']} val, "
            f"{summary['test']} test, 1 sample video ({summary['video_frames']} frames)."
        )
        return summary

    # -------------------------------------------------------------------------
    # Visualization & Inspection
    # -------------------------------------------------------------------------

    def visualize_samples(
        self,
        split: str = "train",
        max_samples: int = 4,
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Visualizes dataset images overlaid with their ground-truth YOLO bounding boxes
        using Matplotlib.

        Args:
            split: Dataset split ('train', 'val', 'test').
            max_samples: Number of samples to render.
            save_path: Optional path to save visual figure.

        Returns:
            Path to saved figure or None if display only.
        """
        img_dir = self.dataset_dir / "images" / split
        lbl_dir = self.dataset_dir / "labels" / split

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            self.logger.error(f"Dataset split not found: {split}")
            return None

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:max_samples]
        if not img_files:
            self.logger.warning(f"No images found in {img_dir}")
            return None

        num_cols = min(2, len(img_files))
        num_rows = math.ceil(len(img_files) / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))

        # Flatten axes array for simple indexing
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, img_file in enumerate(img_files):
            ax = axes_flat[idx]
            img = self.load_image(img_file)
            h, w, _ = img.shape
            ax.imshow(img)
            ax.set_title(f"{img_file.name} ({w}x{h})", fontsize=11, fontweight="bold")
            ax.axis("off")

            # Check matching label file
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.is_file():
                with open(lbl_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h

                        x1 = xc - bw / 2.0
                        y1 = yc - bh / 2.0

                        rect = patches.Rectangle(
                            (x1, y1), bw, bh,
                            linewidth=2, edgecolor="#00FF7F", facecolor="none", linestyle="-"
                        )
                        ax.add_patch(rect)
                        ax.text(
                            x1, max(15, y1 - 4), "fish",
                            color="white", fontsize=9, fontweight="bold",
                            bbox=dict(facecolor="#00AA55", alpha=0.8, edgecolor="none", boxstyle="square,pad=0.2")
                        )

        # Hide any unused subplots
        for idx in range(len(img_files), len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()

        if save_path is not None:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=150, bbox_inches="tight")
            plt.close(fig)
            self.logger.info(f"Dataset visualization saved to {out_p}")
            return out_p
        else:
            default_out = self.data_dir / "dataset_preview.png"
            plt.savefig(str(default_out), dpi=150, bbox_inches="tight")
            plt.close(fig)
            return default_out

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Computes summary statistics for the dataset splits and sample videos.

        Returns:
            Dictionary with dataset counts and properties.
        """
        stats: Dict[str, Any] = {
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "sample_videos": []
        }

        for split in ["train", "val", "test"]:
            img_dir = self.dataset_dir / "images" / split
            lbl_dir = self.dataset_dir / "labels" / split

            img_count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.is_dir() else 0

            stats["splits"][split] = {
                "images": img_count,
                "labels": lbl_count
            }
            stats["total_images"] += img_count
            stats["total_labels"] += lbl_count

        # Check sample videos
        if self.sample_videos_dir.is_dir():
            video_files = list(self.sample_videos_dir.glob("*.mp4")) + list(self.sample_videos_dir.glob("*.avi"))
            for v in video_files:
                try:
                    meta = self.get_video_metadata(v)
                    stats["sample_videos"].append(meta)
                except Exception:
                    stats["sample_videos"].append({"file_name": v.name})

        return stats


if __name__ == "__main__":
    loader = FishDatasetLoader()
    print("=" * 60)
    print("Running Module 2: Fish Dataset Loader Standalone Execution")
    print("=" * 60)

    # 1. Create or bootstrap dataset if empty
    stats = loader.get_dataset_statistics()
    if stats["total_images"] == 0 or len(stats["sample_videos"]) == 0:
        print("[*] Dataset is empty. Bootstrapping synthetic dataset and video...")
        loader.create_sample_dataset(num_train=20, num_val=5, num_test=5, generate_video=True)
        stats = loader.get_dataset_statistics()

    print("\nDataset Statistics:")
    print(f"  • Total Images: {stats['total_images']}")
    for s_name, s_data in stats["splits"].items():
        print(f"    - {s_name.capitalize()}: {s_data['images']} images, {s_data['labels']} labels")
    print(f"  • Sample Videos: {len(stats['sample_videos'])}")
    for v in stats["sample_videos"]:
        print(f"    - {v.get('file_name')}: {v.get('width')}x{v.get('height')} @ {v.get('fps')} FPS ({v.get('duration_sec')}s)")

    # 2. Generate visualization preview
    preview_file = loader.visualize_samples(split="train", max_samples=4)
    if preview_file:
        print(f"\n[+] Visualization preview generated at: {preview_file}")

    print("\nModule 2 dataset loader executed successfully.")
