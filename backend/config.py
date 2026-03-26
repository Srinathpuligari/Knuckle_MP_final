"""Runtime configuration for the simplified knuckle backend."""

from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_ROOT.parent

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5001"))
BACKEND_ADMIN_CODE = os.getenv("BACKEND_ADMIN_CODE", "cbit")

STORAGE_ROOT = BACKEND_ROOT / "storage"
REGISTRATIONS_ROOT = STORAGE_ROOT / "registrations"
DATABASE_PATH = STORAGE_ROOT / "registry.sqlite3"
SHOWCASE_RUNTIME_ROOT = WORKSPACE_ROOT / "knuckle_preprocessing"
SHOWCASE_REGISTRATIONS_ROOT = SHOWCASE_RUNTIME_ROOT / "live_registrations"
SHOWCASE_REGISTRATIONS_ROOT.mkdir(parents=True, exist_ok=True)

RAW_CAPTURES_DIRNAME = "raw"
ROI_CAPTURES_DIRNAME = "roi"
TEMPLATE_FILENAME = "template.npy"
EMBEDDINGS_FILENAME = "embeddings.npy"
TEMPLATE_METADATA_NAME = "template_metadata.json"

MIN_UPLOAD_IMAGES = 5
MAX_UPLOAD_IMAGES = 20
MAX_SELECTED_IMAGES = 8
ROI_SIZE = 128
SHOWCASE_IMAGE_SIZE = int(os.getenv("KNUCKLE_SHOWCASE_IMAGE_SIZE", "256"))
SHOWCASE_POINT_COUNT = int(os.getenv("KNUCKLE_SHOWCASE_POINT_COUNT", "1024"))
SHOWCASE_ROTATION = os.getenv("KNUCKLE_SHOWCASE_ROTATION", "cw").strip().lower() or "cw"
SHOWCASE_ENABLED = os.getenv("KNUCKLE_SHOWCASE_ENABLED", "1").strip() not in {"0", "false", "False"}

FEATURE_VERSION = "efficientnet_b0_imagenet_texture_v1"
VERIFY_THRESHOLD = float(os.getenv("KNUCKLE_VERIFY_THRESHOLD", "0.80"))
VERIFY_MARGIN_THRESHOLD = float(os.getenv("KNUCKLE_VERIFY_MARGIN", "0.015"))
IDENTIFY_THRESHOLD = float(os.getenv("KNUCKLE_IDENTIFY_THRESHOLD", "0.82"))
IDENTIFY_MARGIN_THRESHOLD = float(os.getenv("KNUCKLE_IDENTIFY_MARGIN", "0.02"))
PER_IMAGE_THRESHOLD = float(os.getenv("KNUCKLE_PER_IMAGE_THRESHOLD", "0.72"))
ORB_THRESHOLD = float(os.getenv("KNUCKLE_ORB_THRESHOLD", "0.06"))
MIN_PASS_RATIO = float(os.getenv("KNUCKLE_MIN_PASS_RATIO", "0.5"))
MIN_REGISTRATION_QUALITY = float(os.getenv("KNUCKLE_MIN_REGISTRATION_QUALITY", "0.84"))

QUALITY_THRESHOLDS = {
    "min_contrast": float(os.getenv("KNUCKLE_MIN_CONTRAST", "0.04")),
    "min_laplacian_var": float(os.getenv("KNUCKLE_MIN_LAPLACIAN_VAR", "18.0")),
    "min_edge_density": float(os.getenv("KNUCKLE_MIN_EDGE_DENSITY", "0.015")),
}

INDEX_MCP_LANDMARK = 5
INDEX_PIP_LANDMARK = 6
