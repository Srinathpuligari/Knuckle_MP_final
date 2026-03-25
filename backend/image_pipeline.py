"""Image decoding, ROI extraction, alignment, and preprocessing."""

from __future__ import annotations

import os
import tempfile
import threading
from math import atan2, degrees
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import mediapipe as mp
import numpy as np

from backend.config import INDEX_MCP_LANDMARK, INDEX_PIP_LANDMARK, QUALITY_THRESHOLDS, ROI_SIZE


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _center_crop_square(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(size, height, width)
    half = size // 2
    center_x = width // 2
    center_y = height // 2
    x1 = _clamp(center_x - half, 0, max(width - size, 0))
    y1 = _clamp(center_y - half, 0, max(height - size, 0))
    return image[y1 : y1 + size, x1 : x1 + size]
class KnuckleImageProcessor:
    """Prepare uploaded captures into stable 128x128 knuckle ROIs."""

    def __init__(self) -> None:
        self._hands_lock = threading.Lock()
        self._hands = self._build_hands_detector()

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode uploaded image.")
        return image

    def process_image_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, dict[str, float]]:
        image = self.decode_image(image_bytes)
        roi = self.extract_knuckle_roi(image)
        processed = self.preprocess_roi(roi)
        return processed, self.compute_quality_metrics(processed)

    def extract_knuckle_roi(self, image: np.ndarray) -> np.ndarray:
        mediapipe_roi = self._extract_mediapipe_roi(image)
        if mediapipe_roi is not None:
            return mediapipe_roi
        return self._extract_centered_roi(image)

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        square = _center_crop_square(gray, min(gray.shape[:2]))
        resized = cv2.resize(square, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        return enhanced.astype(np.float32) / 255.0

    def compute_quality_metrics(self, roi: np.ndarray) -> dict[str, float]:
        image_u8 = np.clip(np.rint(roi * 255.0), 0, 255).astype(np.uint8)
        contrast = float(np.std(image_u8) / 255.0)
        laplacian_var = float(cv2.Laplacian(image_u8, cv2.CV_32F).var())
        edges = cv2.Canny(image_u8, 60, 150)
        edge_density = float(np.mean(edges > 0))

        contrast_score = np.clip(contrast / max(QUALITY_THRESHOLDS["min_contrast"], 1e-6), 0.0, 2.0)
        sharpness_score = np.clip(laplacian_var / max(QUALITY_THRESHOLDS["min_laplacian_var"], 1e-6), 0.0, 2.0)
        edge_score = np.clip(edge_density / max(QUALITY_THRESHOLDS["min_edge_density"], 1e-6), 0.0, 2.0)
        quality_score = float((0.25 * contrast_score) + (0.5 * sharpness_score) + (0.25 * edge_score))

        return {
            "contrast": contrast,
            "laplacian_var": laplacian_var,
            "edge_density": edge_density,
            "quality_score": quality_score,
        }

    def passes_quality_gate(self, metrics: dict[str, float]) -> bool:
        return (
            float(metrics["contrast"]) >= QUALITY_THRESHOLDS["min_contrast"]
            and float(metrics["laplacian_var"]) >= QUALITY_THRESHOLDS["min_laplacian_var"]
            and float(metrics["edge_density"]) >= QUALITY_THRESHOLDS["min_edge_density"]
        )

    def _extract_mediapipe_roi(self, image: np.ndarray) -> np.ndarray | None:
        if self._hands is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self._hands_lock:
            result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0].landmark
        height, width = image.shape[:2]
        mcp = landmarks[INDEX_MCP_LANDMARK]
        pip = landmarks[INDEX_PIP_LANDMARK]
        center = np.asarray([mcp.x * width, mcp.y * height], dtype=np.float32)
        pip_point = np.asarray([pip.x * width, pip.y * height], dtype=np.float32)
        axis = pip_point - center
        distance = float(np.linalg.norm(axis))
        if distance < 5.0:
            return None

        crop_center = center - (0.12 * axis)
        angle_deg = degrees(atan2(axis[1], axis[0]))
        rotation_deg = -90.0 - angle_deg
        rotated = self._rotate_image(image, crop_center, rotation_deg)
        crop_size = int(round(max(96.0, min(float(min(height, width)) * 0.55, distance * 6.0))))
        return self._crop_square(rotated, crop_center, crop_size)

    def _extract_centered_roi(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        coords = np.column_stack(np.where(edges > 0))
        center = np.asarray([image.shape[1] / 2.0, image.shape[0] / 2.0], dtype=np.float32)

        if len(coords) >= 40:
            centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            principal = eigenvectors[:, int(np.argmax(eigenvalues))]
            angle_deg = degrees(atan2(float(principal[0]), float(principal[1])))
            image = self._rotate_image(image, center, -90.0 - angle_deg)

        crop_size = int(round(min(image.shape[:2]) * 0.82))
        return self._crop_square(image, center, crop_size)

    def _crop_square(self, image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
        size = max(32, min(size, image.shape[0], image.shape[1]))
        half = size // 2
        center_x = int(round(float(center[0])))
        center_y = int(round(float(center[1])))
        x1 = _clamp(center_x - half, 0, max(image.shape[1] - size, 0))
        y1 = _clamp(center_y - half, 0, max(image.shape[0] - size, 0))
        return image[y1 : y1 + size, x1 : x1 + size]

    def _rotate_image(self, image: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
        rotation_matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, 1.0)
        return cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _build_hands_detector(self):
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "hands"):
            return None
        return solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.45,
        )
