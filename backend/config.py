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

RAW_CAPTURES_DIRNAME = "raw"
ROI_CAPTURES_DIRNAME = "roi"
TEMPLATE_FILENAME = "template.npy"
EMBEDDINGS_FILENAME = "embeddings.npy"
TEMPLATE_METADATA_NAME = "template_metadata.json"

MIN_UPLOAD_IMAGES = 5
MAX_UPLOAD_IMAGES = 20
MAX_SELECTED_IMAGES = 8
ROI_SIZE = 128

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
