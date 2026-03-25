"""Shared backend data types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProcessedCapture:
    """One uploaded image after ROI extraction, preprocessing, and embedding."""

    index: int
    roi: np.ndarray
    quality: dict[str, float]
    embedding: np.ndarray | None = None


@dataclass
class EnrollmentBundle:
    """Stored template data derived from one registration."""

    template: np.ndarray
    embeddings: np.ndarray
    rois: list[np.ndarray]
    registration_quality: float
    feature_version: str


@dataclass
class StoredUser:
    """Metadata stored for one enrolled user."""

    uid: str
    name: str
    phone: str
    email: str
    dob: str
    address: str
    gender: str
    registered_at: str
    image_count: int
    registration_quality: float
    feature_version: str
    registration_dir: Path

    @property
    def template_path(self) -> Path:
        return self.registration_dir / "template.npy"

    @property
    def embeddings_path(self) -> Path:
        return self.registration_dir / "embeddings.npy"


@dataclass
class StoredEnrollment:
    """Enrollment loaded back from disk for comparison."""

    user: StoredUser
    template: np.ndarray
    embeddings: np.ndarray
    rois: list[np.ndarray]
    feature_version: str
