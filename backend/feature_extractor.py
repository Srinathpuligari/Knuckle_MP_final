"""Fixed embedding extractor used by the simplified backend."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32)


class KnuckleFeatureExtractor:
    """EfficientNet-B0 plus lightweight texture descriptors."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def extract_embeddings(self, rois: list[np.ndarray]) -> np.ndarray:
        if not rois:
            raise ValueError("At least one processed ROI is required.")

        efficientnet_embeddings = self._extract_efficientnet_embeddings(rois)
        texture_embeddings = np.stack([self._extract_texture_descriptor(roi) for roi in rois], axis=0)

        efficientnet_embeddings = _normalize_rows(efficientnet_embeddings)
        texture_embeddings = _normalize_rows(texture_embeddings)
        combined = np.concatenate(
            [
                efficientnet_embeddings,
                texture_embeddings * 0.15,
            ],
            axis=1,
        )
        return _normalize_rows(combined)

    def _extract_efficientnet_embeddings(self, rois: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([np.repeat(roi[:, :, None], 3, axis=2) for roi in rois], axis=0).astype(np.float32)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            embeddings = self.model(tensor).detach().cpu().numpy().astype(np.float32)
        return embeddings

    def _extract_texture_descriptor(self, roi: np.ndarray) -> np.ndarray:
        image_u8 = np.clip(np.rint(roi * 255.0), 0, 255).astype(np.uint8)
        downsampled = cv2.resize(image_u8, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
        downsampled /= 255.0

        hog = self._hog_descriptor(image_u8)
        lbp = self._lbp_histogram(image_u8)
        descriptor = np.concatenate([downsampled, hog, lbp], axis=0).astype(np.float32)

        norm = float(np.linalg.norm(descriptor))
        if norm <= 1e-12:
            return descriptor
        return descriptor / norm

    def _hog_descriptor(self, image_u8: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(image_u8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_u8, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        angle = np.mod(angle, 180.0)

        cells_y = 4
        cells_x = 4
        bins = 8
        cell_h = image_u8.shape[0] // cells_y
        cell_w = image_u8.shape[1] // cells_x
        features: list[np.ndarray] = []

        for cell_y in range(cells_y):
            for cell_x in range(cells_x):
                y1 = cell_y * cell_h
                y2 = (cell_y + 1) * cell_h
                x1 = cell_x * cell_w
                x2 = (cell_x + 1) * cell_w
                hist, _ = np.histogram(
                    angle[y1:y2, x1:x2].reshape(-1),
                    bins=bins,
                    range=(0.0, 180.0),
                    weights=magnitude[y1:y2, x1:x2].reshape(-1),
                )
                features.append(hist.astype(np.float32))

        descriptor = np.concatenate(features, axis=0)
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1e-12:
            return descriptor
        return descriptor / norm

    def _lbp_histogram(self, image_u8: np.ndarray) -> np.ndarray:
        center = image_u8[1:-1, 1:-1]
        codes = np.zeros_like(center, dtype=np.uint8)

        neighbors = [
            (image_u8[:-2, :-2], 0),
            (image_u8[:-2, 1:-1], 1),
            (image_u8[:-2, 2:], 2),
            (image_u8[1:-1, 2:], 3),
            (image_u8[2:, 2:], 4),
            (image_u8[2:, 1:-1], 5),
            (image_u8[2:, :-2], 6),
            (image_u8[1:-1, :-2], 7),
        ]
        for neighbor, bit in neighbors:
            codes |= ((neighbor >= center).astype(np.uint8) << bit)

        buckets = (codes // 8).astype(np.int32)
        hist, _ = np.histogram(buckets, bins=32, range=(0, 32))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        return hist
