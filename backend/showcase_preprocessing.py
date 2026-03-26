"""Generate showcase-only preprocessing artifacts for live registrations."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class ProcessedArtifacts:
    enhanced: np.ndarray
    pattern: np.ndarray
    blackhat: np.ndarray
    background_mask: np.ndarray


@dataclass
class RegistrationShowcaseArtifacts:
    uid: str
    subject: str
    capture_count: int
    output_root: Path
    summary_path: Path
    preview_original_vs_enhanced: Path
    preview_original_vs_pattern: Path
    average_black_pixel_ratio: float
    min_black_pixel_ratio: float
    max_black_pixel_ratio: float
    created_at: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "subject": self.subject,
            "capture_count": self.capture_count,
            "output_root": str(self.output_root),
            "summary_path": str(self.summary_path),
            "preview_original_vs_enhanced": str(self.preview_original_vs_enhanced),
            "preview_original_vs_pattern": str(self.preview_original_vs_pattern),
            "average_black_pixel_ratio": self.average_black_pixel_ratio,
            "min_black_pixel_ratio": self.min_black_pixel_ratio,
            "max_black_pixel_ratio": self.max_black_pixel_ratio,
            "created_at": self.created_at,
        }


def _slugify(text: str) -> str:
    pieces: list[str] = []
    for character in text.strip().lower():
        if character.isalnum():
            pieces.append(character)
        elif pieces and pieces[-1] != "-":
            pieces.append("-")
    return "".join(pieces).strip("-") or "registered-user"


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "none":
        return image
    if rotation == "cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


def square_crop(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return gray[top : top + size, left : left + size]


def top_bottom_bright_mask(gray: np.ndarray) -> np.ndarray:
    q975 = float(np.quantile(gray, 0.975))
    threshold = int(max(q975, float(gray.mean()) + (1.5 * float(gray.std()))))
    threshold = min(threshold, 250)

    bright = (gray >= threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    mask = np.zeros_like(bright)
    height, width = gray.shape

    for label in range(1, component_count):
        x, y, box_width, box_height, area = stats[label]
        touches_top_or_bottom = y == 0 or y + box_height >= height
        width_ratio = box_width / max(width, 1)
        if touches_top_or_bottom and area >= 200 and width_ratio >= 0.22:
            mask[labels == label] = 255

    return cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8), iterations=1)


def extract_pattern(gray: np.ndarray, output_size: int) -> ProcessedArtifacts:
    crop = square_crop(gray)
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    smooth = cv2.bilateralFilter(enhanced, 7, 30, 30)

    blackhat = cv2.morphologyEx(
        smooth,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    thresholded = cv2.adaptiveThreshold(
        blackhat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -2,
    )
    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)
    cleaned = np.zeros_like(thresholded)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 12:
            cleaned[labels == label] = 255

    background_mask = top_bottom_bright_mask(resized)
    cleaned[background_mask > 0] = 0

    pattern = np.full_like(resized, 255)
    pattern[cleaned > 0] = 0
    return ProcessedArtifacts(
        enhanced=enhanced,
        pattern=pattern,
        blackhat=blackhat,
        background_mask=background_mask,
    )


def build_point_cloud(pattern: np.ndarray, blackhat: np.ndarray, points: int, rng: np.random.Generator) -> np.ndarray:
    mask = pattern == 0
    ys, xs = np.where(mask)

    if len(xs) == 0:
        fallback = blackhat >= np.quantile(blackhat, 0.92)
        ys, xs = np.where(fallback)

    values = blackhat[ys, xs].astype(np.float32)
    weights = values + 1.0
    probabilities = weights / float(weights.sum()) if float(weights.sum()) > 0 else None

    replace = len(xs) < points
    chosen = rng.choice(len(xs), size=points, replace=replace, p=probabilities)
    xs = xs[chosen].astype(np.float32)
    ys = ys[chosen].astype(np.float32)
    values = values[chosen]

    height, width = pattern.shape
    x_coords = (xs / max(width - 1, 1)) * 2.0 - 1.0
    y_coords = 1.0 - (ys / max(height - 1, 1)) * 2.0
    z_coords = values / 255.0

    cloud = np.column_stack((x_coords, y_coords, z_coords)).astype(np.float32)
    cloud -= cloud.mean(axis=0, keepdims=True)

    xy_scale = float(np.max(np.linalg.norm(cloud[:, :2], axis=1)))
    if xy_scale > 0:
        cloud[:, :2] /= xy_scale

    z_min = float(cloud[:, 2].min())
    z_max = float(cloud[:, 2].max())
    if z_max > z_min:
        cloud[:, 2] = (cloud[:, 2] - z_min) / (z_max - z_min)
    cloud[:, 2] = (cloud[:, 2] * 0.6) - 0.3
    return cloud.astype(np.float32)


def save_contact_sheet(
    samples: list[dict[str, object]],
    output_path: Path,
    left_key: str,
    right_key: str,
    title: str,
    columns: int = 3,
) -> None:
    cards: list[np.ndarray] = []
    for sample in samples:
        left = np.asarray(sample[left_key], dtype=np.uint8)
        right = np.asarray(sample[right_key], dtype=np.uint8)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        card = np.hstack((left_bgr, right_bgr))
        label = f"{sample['subject']} | {sample['name']}"
        cv2.putText(card, label[:46], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cards.append(card)

    if not cards:
        return

    rows: list[np.ndarray] = []
    for index in range(0, len(cards), columns):
        row_cards = cards[index : index + columns]
        while len(row_cards) < columns:
            row_cards.append(np.full_like(cards[0], 255))
        rows.append(np.hstack(row_cards))

    sheet = np.vstack(rows)
    top = np.full((28, sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(top, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    final_sheet = np.vstack((top, sheet))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_sheet)


def _decode_grayscale(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to decode registration image for showcase preprocessing.")
    return image


def _coerce_gray_image(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating):
            gray = np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def generate_registration_showcase(
    uid: str,
    subject_name: str,
    image_payloads: list[bytes],
    output_root: Path,
    output_size: int = 256,
    points: int = 1024,
    rotation: str = "cw",
    source_rois: list[np.ndarray] | None = None,
) -> RegistrationShowcaseArtifacts:
    if source_rois:
        source_images = [_coerce_gray_image(roi) for roi in source_rois]
    else:
        source_images = [_decode_grayscale(image_bytes) for image_bytes in image_payloads]

    if not source_images:
        raise ValueError("At least one registration image is required for showcase preprocessing.")

    subject_slug = _slugify(subject_name)
    registration_root = output_root / f"{uid}_{subject_slug}"
    enhanced_root = registration_root / "enhanced_grayscale"
    pattern_root = registration_root / "extracted_pattern_binary"
    point_root = registration_root / "dgcnn_pointcloud_npy"
    report_root = registration_root / "reports"

    if registration_root.exists():
        shutil.rmtree(registration_root)

    enhanced_root.mkdir(parents=True, exist_ok=True)
    pattern_root.mkdir(parents=True, exist_ok=True)
    point_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    seed = sum(ord(character) for character in uid) + len(source_images)
    rng = np.random.default_rng(seed)

    preview_samples: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    black_pixel_ratios: list[float] = []

    for index, gray in enumerate(source_images, start=1):
        processed = extract_pattern(gray, output_size=output_size)
        processed = ProcessedArtifacts(
            enhanced=rotate_image(processed.enhanced, rotation),
            pattern=rotate_image(processed.pattern, rotation),
            blackhat=rotate_image(processed.blackhat, rotation),
            background_mask=rotate_image(processed.background_mask, rotation),
        )
        point_cloud = build_point_cloud(processed.pattern, processed.blackhat, points, rng)

        original = rotate_image(
            cv2.resize(square_crop(gray), (output_size, output_size), interpolation=cv2.INTER_AREA),
            rotation,
        )
        name = f"capture_{index:02d}"
        enhanced_path = enhanced_root / f"{name}.png"
        pattern_path = pattern_root / f"{name}.png"
        point_path = point_root / f"{name}.npy"

        cv2.imwrite(str(enhanced_path), processed.enhanced)
        cv2.imwrite(str(pattern_path), processed.pattern)
        np.save(point_path, point_cloud)

        black_ratio = float(np.mean(processed.pattern == 0))
        black_pixel_ratios.append(black_ratio)

        manifest_rows.append(
            {
                "capture": name,
                "enhanced_png": str(enhanced_path),
                "pattern_png": str(pattern_path),
                "pointcloud_npy": str(point_path),
                "black_pixel_ratio": f"{black_ratio:.6f}",
            }
        )
        preview_samples.append(
            {
                "subject": subject_name,
                "name": name,
                "original": original,
                "enhanced": processed.enhanced,
                "pattern": processed.pattern,
            }
        )

    manifest_path = report_root / "artifact_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["capture", "enhanced_png", "pattern_png", "pointcloud_npy", "black_pixel_ratio"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    preview_original_vs_enhanced = report_root / "preview_original_vs_enhanced.png"
    preview_original_vs_pattern = report_root / "preview_original_vs_pattern.png"
    save_contact_sheet(
        preview_samples,
        preview_original_vs_enhanced,
        left_key="original",
        right_key="enhanced",
        title=f"{subject_name}: Original vs Enhanced",
    )
    save_contact_sheet(
        preview_samples,
        preview_original_vs_pattern,
        left_key="original",
        right_key="pattern",
        title=f"{subject_name}: Original vs Extracted Pattern",
    )

    created_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "uid": uid,
        "subject": subject_name,
        "created_at": created_at,
        "capture_count": len(source_images),
        "output_root": str(registration_root),
        "output_image_size": output_size,
        "points_per_cloud": points,
        "rotation": rotation,
        "average_black_pixel_ratio": round(float(np.mean(black_pixel_ratios)), 6),
        "min_black_pixel_ratio": round(float(np.min(black_pixel_ratios)), 6),
        "max_black_pixel_ratio": round(float(np.max(black_pixel_ratios)), 6),
        "folders": {
            "enhanced_grayscale": str(enhanced_root),
            "extracted_pattern_binary": str(pattern_root),
            "dgcnn_pointcloud_npy": str(point_root),
            "reports": str(report_root),
        },
    }
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return RegistrationShowcaseArtifacts(
        uid=uid,
        subject=subject_name,
        capture_count=len(source_images),
        output_root=registration_root,
        summary_path=summary_path,
        preview_original_vs_enhanced=preview_original_vs_enhanced,
        preview_original_vs_pattern=preview_original_vs_pattern,
        average_black_pixel_ratio=summary["average_black_pixel_ratio"],
        min_black_pixel_ratio=summary["min_black_pixel_ratio"],
        max_black_pixel_ratio=summary["max_black_pixel_ratio"],
        created_at=created_at,
    )
