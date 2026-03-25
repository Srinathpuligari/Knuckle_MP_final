"""SQLite-backed storage for the simplified knuckle backend."""

from __future__ import annotations

import json
import secrets
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import (
    DATABASE_PATH,
    EMBEDDINGS_FILENAME,
    RAW_CAPTURES_DIRNAME,
    REGISTRATIONS_ROOT,
    ROI_CAPTURES_DIRNAME,
    STORAGE_ROOT,
    TEMPLATE_FILENAME,
    TEMPLATE_METADATA_NAME,
)
from backend.schemas import EnrollmentBundle, StoredEnrollment, StoredUser


class RegistrationStore:
    """Persistent registry for enrolled users and their templates."""

    def __init__(self) -> None:
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        REGISTRATIONS_ROOT.mkdir(parents=True, exist_ok=True)
        self.db_path = DATABASE_PATH
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    uid TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    email TEXT NOT NULL,
                    dob TEXT NOT NULL,
                    address TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    registration_quality REAL NOT NULL,
                    feature_version TEXT NOT NULL
                )
                """
            )

    def generate_uid(self) -> str:
        while True:
            candidate = str(secrets.randbelow(9) + 1) + "".join(str(secrets.randbelow(10)) for _ in range(11))
            if self.get_user(candidate) is None:
                return candidate

    def save_registration(
        self,
        uid: str,
        metadata: dict[str, str],
        original_images: list[bytes],
        enrollment: EnrollmentBundle,
    ) -> StoredUser:
        registration_dir = REGISTRATIONS_ROOT / uid
        raw_dir = registration_dir / RAW_CAPTURES_DIRNAME
        roi_dir = registration_dir / ROI_CAPTURES_DIRNAME

        if registration_dir.exists():
            shutil.rmtree(registration_dir)

        raw_dir.mkdir(parents=True, exist_ok=True)
        roi_dir.mkdir(parents=True, exist_ok=True)

        for index, image_bytes in enumerate(original_images, start=1):
            (raw_dir / f"capture_{index:02d}.jpg").write_bytes(image_bytes)

        for index, roi in enumerate(enrollment.rois, start=1):
            roi_image = np.clip(np.rint(roi * 255.0), 0, 255).astype(np.uint8)
            cv2.imwrite(str(roi_dir / f"capture_{index:02d}.png"), roi_image)

        np.save(registration_dir / TEMPLATE_FILENAME, enrollment.template.astype(np.float32))
        np.save(registration_dir / EMBEDDINGS_FILENAME, enrollment.embeddings.astype(np.float32))
        (registration_dir / TEMPLATE_METADATA_NAME).write_text(
            json.dumps({"feature_version": enrollment.feature_version}, indent=2),
            encoding="utf-8",
        )

        clean_metadata = {
            "name": metadata.get("name", "").strip(),
            "phone": metadata.get("phone", "").strip(),
            "email": metadata.get("email", "").strip(),
            "dob": metadata.get("dob", "").strip(),
            "address": metadata.get("address", "").strip(),
            "gender": metadata.get("gender", "").strip(),
        }
        registered_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO users (
                    uid,
                    name,
                    phone,
                    email,
                    dob,
                    address,
                    gender,
                    registered_at,
                    image_count,
                    registration_quality,
                    feature_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    clean_metadata["name"],
                    clean_metadata["phone"],
                    clean_metadata["email"],
                    clean_metadata["dob"],
                    clean_metadata["address"],
                    clean_metadata["gender"],
                    registered_at,
                    int(len(enrollment.rois)),
                    float(enrollment.registration_quality),
                    enrollment.feature_version,
                ),
            )

        return StoredUser(
            uid=uid,
            name=clean_metadata["name"],
            phone=clean_metadata["phone"],
            email=clean_metadata["email"],
            dob=clean_metadata["dob"],
            address=clean_metadata["address"],
            gender=clean_metadata["gender"],
            registered_at=registered_at,
            image_count=int(len(enrollment.rois)),
            registration_quality=float(enrollment.registration_quality),
            feature_version=enrollment.feature_version,
            registration_dir=registration_dir,
        )

    def _row_to_user(self, row: sqlite3.Row) -> StoredUser:
        return StoredUser(
            uid=str(row["uid"]),
            name=str(row["name"]),
            phone=str(row["phone"]),
            email=str(row["email"]),
            dob=str(row["dob"]),
            address=str(row["address"]),
            gender=str(row["gender"]),
            registered_at=str(row["registered_at"]),
            image_count=int(row["image_count"]),
            registration_quality=float(row["registration_quality"]),
            feature_version=str(row["feature_version"]),
            registration_dir=REGISTRATIONS_ROOT / str(row["uid"]),
        )

    def get_user(self, uid: str) -> StoredUser | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
        return self._row_to_user(row) if row is not None else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY registered_at DESC").fetchall()
        return [
            {
                "uid": str(row["uid"]),
                "name": str(row["name"]),
                "phone": str(row["phone"]),
                "email": str(row["email"]),
                "dob": str(row["dob"]),
                "address": str(row["address"]),
                "gender": str(row["gender"]),
                "registered_at": str(row["registered_at"]),
                "image_count": int(row["image_count"]),
                "registration_quality": float(row["registration_quality"]),
                "feature_version": str(row["feature_version"]),
            }
            for row in rows
        ]

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM users").fetchone()
        return int(row["total"]) if row is not None else 0

    def load_enrollment(self, uid: str) -> StoredEnrollment | None:
        user = self.get_user(uid)
        if user is None or not user.template_path.exists() or not user.embeddings_path.exists():
            return None

        feature_version = user.feature_version
        metadata_path = user.registration_dir / TEMPLATE_METADATA_NAME
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_version = str(metadata.get("feature_version", feature_version)).strip() or feature_version

        return StoredEnrollment(
            user=user,
            template=np.load(user.template_path).astype(np.float32),
            embeddings=np.load(user.embeddings_path).astype(np.float32),
            rois=self._load_rois(user.registration_dir / ROI_CAPTURES_DIRNAME),
            feature_version=feature_version,
        )

    def iter_enrollments(self) -> list[tuple[StoredUser, StoredEnrollment]]:
        enrollments: list[tuple[StoredUser, StoredEnrollment]] = []
        for user_data in self.list_users():
            user = self.get_user(str(user_data["uid"]))
            if user is None:
                continue
            enrollment = self.load_enrollment(user.uid)
            if enrollment is None:
                continue
            enrollments.append((user, enrollment))
        return enrollments

    def delete_user(self, uid: str) -> bool:
        user = self.get_user(uid)
        if user is None:
            return False

        with self._connect() as connection:
            connection.execute("DELETE FROM users WHERE uid = ?", (uid,))

        if user.registration_dir.exists():
            shutil.rmtree(user.registration_dir)
        return True

    def _load_rois(self, roi_dir: Path) -> list[np.ndarray]:
        rois: list[np.ndarray] = []
        for path in sorted(roi_dir.glob("*.png")):
            roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if roi is None:
                continue
            rois.append(roi.astype(np.float32) / 255.0)
        return rois
