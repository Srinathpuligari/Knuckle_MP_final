"""Verification engine for enrollment, verification, and identification."""

from __future__ import annotations

import cv2
import numpy as np

from backend.config import (
    FEATURE_VERSION,
    IDENTIFY_THRESHOLD,
    MAX_SELECTED_IMAGES,
    MAX_UPLOAD_IMAGES,
    MIN_PASS_RATIO,
    MIN_REGISTRATION_QUALITY,
    MIN_UPLOAD_IMAGES,
    ORB_THRESHOLD,
    PER_IMAGE_THRESHOLD,
    VERIFY_THRESHOLD,
)
from backend.feature_extractor import KnuckleFeatureExtractor
from backend.image_pipeline import KnuckleImageProcessor
from backend.schemas import EnrollmentBundle, ProcessedCapture, StoredEnrollment


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


class KnuckleVerificationEngine:
    """Fixed-feature backend engine with no training dependency."""

    def __init__(self) -> None:
        self.processor = KnuckleImageProcessor()
        self.extractor = KnuckleFeatureExtractor()
        self.orb = cv2.ORB_create(nfeatures=256, fastThreshold=7)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.feature_version = FEATURE_VERSION
        self.calibration = {
            "feature_version": self.feature_version,
            "verify_threshold": VERIFY_THRESHOLD,
            "identify_threshold": IDENTIFY_THRESHOLD,
            "per_image_threshold": PER_IMAGE_THRESHOLD,
            "orb_threshold": ORB_THRESHOLD,
        }

    def process_uploaded_images(self, image_payloads: list[bytes]) -> list[ProcessedCapture]:
        if not image_payloads:
            raise ValueError("At least one image is required.")
        if len(image_payloads) < MIN_UPLOAD_IMAGES:
            raise ValueError(f"Capture at least {MIN_UPLOAD_IMAGES} knuckle images.")
        if len(image_payloads) > MAX_UPLOAD_IMAGES:
            raise ValueError(f"Capture no more than {MAX_UPLOAD_IMAGES} knuckle images.")

        captures: list[ProcessedCapture] = []
        for index, image_bytes in enumerate(image_payloads):
            roi, quality = self.processor.process_image_bytes(image_bytes)
            captures.append(ProcessedCapture(index=index, roi=roi, quality=quality))

        valid_captures = [capture for capture in captures if self.processor.passes_quality_gate(capture.quality)]
        chosen = valid_captures if len(valid_captures) >= MIN_UPLOAD_IMAGES else captures
        chosen = sorted(chosen, key=lambda capture: float(capture.quality["quality_score"]), reverse=True)
        chosen = chosen[: min(MAX_SELECTED_IMAGES, len(chosen))]

        if len(chosen) < MIN_UPLOAD_IMAGES:
            raise ValueError("Use the same knuckle, keep it centered, and capture at least 5 clear images.")

        embeddings = self.extractor.extract_embeddings([capture.roi for capture in chosen])
        for capture, embedding in zip(chosen, embeddings):
            capture.embedding = embedding.astype(np.float32)
        return chosen

    def create_enrollment_bundle(self, captures: list[ProcessedCapture]) -> EnrollmentBundle:
        embeddings = self._stack_embeddings(captures)
        template = _normalize(np.mean(embeddings, axis=0))
        per_image_scores = np.asarray([float(np.dot(embedding, template)) for embedding in embeddings], dtype=np.float32)
        registration_quality = float(np.mean(per_image_scores))
        pass_ratio = float(np.mean(per_image_scores >= PER_IMAGE_THRESHOLD))

        if registration_quality < MIN_REGISTRATION_QUALITY or pass_ratio < MIN_PASS_RATIO:
            raise ValueError(
                "Enrollment captures are inconsistent. Register only one knuckle and use sharper, steadier images."
            )

        return EnrollmentBundle(
            template=template,
            embeddings=embeddings,
            rois=[capture.roi for capture in captures],
            registration_quality=registration_quality,
            feature_version=self.feature_version,
        )

    def compare_with_enrollment(
        self,
        query_captures: list[ProcessedCapture],
        enrollment: StoredEnrollment | EnrollmentBundle,
    ) -> dict[str, float | bool]:
        query_embeddings = self._stack_embeddings(query_captures)
        query_template = _normalize(np.mean(query_embeddings, axis=0))
        enrolled_template = _normalize(np.asarray(enrollment.template, dtype=np.float32))

        template_score = float(np.dot(query_template, enrolled_template))
        per_image_scores = np.asarray(
            [float(np.dot(query_embedding, enrolled_template)) for query_embedding in query_embeddings],
            dtype=np.float32,
        )
        mean_image_score = float(np.mean(per_image_scores))
        max_image_score = float(np.max(per_image_scores))
        min_image_score = float(np.min(per_image_scores))
        pass_ratio = float(np.mean(per_image_scores >= PER_IMAGE_THRESHOLD))
        orb_score = self._compute_orb_score(query_captures, enrollment)

        # Keep the extra scores for diagnostics, but let the main similarity
        # threshold drive the verification decision.
        match = bool(template_score >= VERIFY_THRESHOLD)

        return {
            "match": match,
            "score": template_score,
            "cosine_score": template_score,
            "mean_image_score": mean_image_score,
            "max_image_score": max_image_score,
            "min_image_score": min_image_score,
            "pass_ratio": pass_ratio,
            "num_query_images": int(len(query_embeddings)),
            "orb_score": orb_score,
            "cosine_threshold": VERIFY_THRESHOLD,
            "orb_threshold": ORB_THRESHOLD,
            "fused_threshold": VERIFY_THRESHOLD,
        }

    def _stack_embeddings(self, captures: list[ProcessedCapture]) -> np.ndarray:
        embeddings = [capture.embedding for capture in captures if capture.embedding is not None]
        if len(embeddings) != len(captures):
            raise ValueError("Missing embeddings for one or more captures.")
        return np.stack([np.asarray(embedding, dtype=np.float32) for embedding in embeddings], axis=0)

    def _compute_orb_score(
        self,
        query_captures: list[ProcessedCapture],
        enrollment: StoredEnrollment | EnrollmentBundle,
    ) -> float:
        enrolled_rois = [np.asarray(roi, dtype=np.float32) for roi in enrollment.rois]
        if not enrolled_rois:
            return 0.0

        enrolled_descriptors = [self._orb_descriptor(roi) for roi in enrolled_rois]
        per_query_scores: list[float] = []
        for capture in query_captures:
            query_descriptor = self._orb_descriptor(capture.roi)
            if query_descriptor is None or len(query_descriptor) == 0:
                per_query_scores.append(0.0)
                continue

            best_score = 0.0
            for enrolled_descriptor in enrolled_descriptors:
                if enrolled_descriptor is None or len(enrolled_descriptor) == 0:
                    continue
                matches = self.matcher.knnMatch(query_descriptor, enrolled_descriptor, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.78 * n.distance]
                denominator = max(min(len(query_descriptor), len(enrolled_descriptor)), 1)
                best_score = max(best_score, len(good_matches) / denominator)
            per_query_scores.append(best_score)

        return float(np.mean(per_query_scores)) if per_query_scores else 0.0

    def _orb_descriptor(self, roi: np.ndarray):
        image_u8 = np.clip(np.rint(roi * 255.0), 0, 255).astype(np.uint8)
        _keypoints, descriptor = self.orb.detectAndCompute(image_u8, None)
        return descriptor
