#!/usr/bin/env python3
"""Generate a PDF report, metrics JSON, and showcase charts for the knuckle project."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.config import (
    IDENTIFY_MARGIN_THRESHOLD,
    IDENTIFY_THRESHOLD,
    MIN_UPLOAD_IMAGES,
    VERIFY_MARGIN_THRESHOLD,
    VERIFY_THRESHOLD,
)
from backend.engine import KnuckleVerificationEngine


COLOR_PRIMARY = "#1f77b4"
COLOR_ACCENT = "#2ca02c"
COLOR_WARM = "#ff7f0e"
COLOR_NEGATIVE = "#d62728"
COLOR_BG = "#f7f9fc"


@dataclass
class ClassRecord:
    class_id: str
    subject: str
    finger: str
    session1_paths: list[Path]
    session2_paths: list[Path]
    enrollment_template: np.ndarray
    query_template: np.ndarray
    registration_quality: float
    query_quality_mean: float
    enrollment_images_used: int
    query_images_used: int


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a showcase-ready knuckle project report.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("prepared_dataset"),
        help="Path to the prepared knuckle dataset.",
    )
    parser.add_argument(
        "--preprocessing-summary",
        type=Path,
        default=Path("knuckle_preprocessing/reports/summary.json"),
        help="Path to the preprocessing summary JSON.",
    )
    parser.add_argument(
        "--preprocessing-preview-enhanced",
        type=Path,
        default=Path("knuckle_preprocessing/reports/preview_original_vs_enhanced.png"),
        help="Path to the preprocessing preview image for original vs enhanced.",
    )
    parser.add_argument(
        "--preprocessing-preview-pattern",
        type=Path,
        default=Path("knuckle_preprocessing/reports/preview_original_vs_pattern.png"),
        help="Path to the preprocessing preview image for original vs pattern.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("project_showcase_report"),
        help="Output directory for the report, charts, and metrics.",
    )
    return parser.parse_args()


def load_dataset_index(dataset_root: Path) -> tuple[dict[tuple[str, str], dict[str, list[Path]]], Counter]:
    groups: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: {"session1": [], "session2": []})
    counts: Counter = Counter()
    for path in sorted(dataset_root.rglob("*.bmp")):
        pieces = path.stem.split("_")
        if len(pieces) < 2:
            continue
        session = pieces[0]
        finger = pieces[1]
        if session not in {"session1", "session2"}:
            continue
        groups[(path.parent.name, finger)][session].append(path)
        counts[(session, finger)] += 1
    return groups, counts


def build_class_records(dataset_root: Path) -> tuple[list[ClassRecord], dict[str, Any]]:
    groups, counts = load_dataset_index(dataset_root)
    engine = KnuckleVerificationEngine()

    records: list[ClassRecord] = []
    incomplete_classes = 0
    quality_rejected_classes = 0
    total_images_used = 0
    group_items = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))

    for index, ((subject, finger), sessions) in enumerate(group_items, start=1):
        session1_paths = sorted(sessions["session1"])
        session2_paths = sorted(sessions["session2"])
        if len(session1_paths) < MIN_UPLOAD_IMAGES or len(session2_paths) < MIN_UPLOAD_IMAGES:
            incomplete_classes += 1
            continue

        print(f"[prepare {index}/{len(group_items)}] {subject} | {finger}")
        session1_payloads = [path.read_bytes() for path in session1_paths]
        session2_payloads = [path.read_bytes() for path in session2_paths]

        try:
            enrollment_captures = engine.process_uploaded_images(session1_payloads)
            enrollment = engine.create_enrollment_bundle(enrollment_captures)
            query_captures = engine.process_uploaded_images(session2_payloads)
        except ValueError as exc:
            quality_rejected_classes += 1
            print(f"  -> skipped ({exc})")
            continue

        query_embeddings = np.stack([capture.embedding for capture in query_captures if capture.embedding is not None], axis=0)
        query_template = _normalize(np.mean(query_embeddings, axis=0))

        records.append(
            ClassRecord(
                class_id=f"{subject}|{finger}",
                subject=subject,
                finger=finger,
                session1_paths=session1_paths,
                session2_paths=session2_paths,
                enrollment_template=enrollment.template.astype(np.float32),
                query_template=query_template.astype(np.float32),
                registration_quality=float(enrollment.registration_quality),
                query_quality_mean=float(np.mean([capture.quality["quality_score"] for capture in query_captures])),
                enrollment_images_used=len(enrollment_captures),
                query_images_used=len(query_captures),
            )
        )
        total_images_used += len(session1_paths) + len(session2_paths)

    dataset_summary = {
        "subject_count": len({subject for subject, _finger in groups.keys()}),
        "class_count_total": len(groups),
        "class_count_evaluated": len(records),
        "class_count_incomplete": incomplete_classes,
        "class_count_rejected_by_quality_gate": quality_rejected_classes,
        "total_images_all": int(sum(counts.values())),
        "total_images_evaluated": total_images_used,
        "counts_by_session_and_finger": {f"{session}_{finger}": int(total) for (session, finger), total in sorted(counts.items())},
    }
    return records, dataset_summary


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_records(records: list[ClassRecord], dataset_summary: dict[str, Any]) -> dict[str, Any]:
    labels = [record.class_id for record in records]
    fingers = np.array([record.finger for record in records])
    enrollment_templates = np.stack([record.enrollment_template for record in records], axis=0)
    query_templates = np.stack([record.query_template for record in records], axis=0)

    score_matrix = np.matmul(query_templates, enrollment_templates.T).astype(np.float32)
    ranking = np.argsort(score_matrix, axis=1)[:, ::-1]
    top1 = ranking[:, 0]
    top3 = ranking[:, :3]
    top5 = ranking[:, :5]
    row_indices = np.arange(len(records))

    genuine_scores = score_matrix[row_indices, row_indices]
    off_diagonal_mask = ~np.eye(len(records), dtype=bool)
    impostor_scores = score_matrix[off_diagonal_mask]

    strongest_other_scores = np.max(np.where(np.eye(len(records), dtype=bool), -np.inf, score_matrix), axis=1)
    top_scores = score_matrix[row_indices, top1]
    second_best_scores = np.take_along_axis(score_matrix, ranking[:, 1:2], axis=1).reshape(-1)
    true_margins = genuine_scores - strongest_other_scores
    top_margins = top_scores - second_best_scores

    raw_top1_hits = top1 == row_indices
    raw_top3_hits = np.any(top3 == row_indices[:, None], axis=1)
    raw_top5_hits = np.any(top5 == row_indices[:, None], axis=1)

    system_ident_found = (top_scores >= IDENTIFY_THRESHOLD) & (top_margins >= IDENTIFY_MARGIN_THRESHOLD)
    system_ident_correct = system_ident_found & raw_top1_hits
    system_ident_misidentify = system_ident_found & (~raw_top1_hits)
    system_ident_reject = ~system_ident_found

    verify_accept = (
        (genuine_scores >= VERIFY_THRESHOLD)
        & raw_top1_hits
        & (true_margins >= VERIFY_MARGIN_THRESHOLD)
    )
    false_accept_query = (
        (~raw_top1_hits)
        & (top_scores >= VERIFY_THRESHOLD)
        & (top_margins >= VERIFY_MARGIN_THRESHOLD)
    )

    y_true = np.concatenate(
        [
            np.ones_like(genuine_scores, dtype=np.int32),
            np.zeros_like(impostor_scores, dtype=np.int32),
        ]
    )
    y_score = np.concatenate([genuine_scores, impostor_scores])
    y_pred = (y_score >= VERIFY_THRESHOLD).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    roc_auc = float(auc(fpr, tpr))
    average_precision = float(average_precision_score(y_true, y_score))
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])
    gar_at_far_1 = float(np.max(tpr[fpr <= 0.01])) if np.any(fpr <= 0.01) else 0.0
    gar_at_far_01 = float(np.max(tpr[fpr <= 0.001])) if np.any(fpr <= 0.001) else 0.0

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    far = _safe_divide(fp, fp + tn)
    frr = _safe_divide(fn, fn + tp)
    balanced_accuracy = (recall + specificity) / 2.0
    mcc = float(matthews_corrcoef(y_true, y_pred))

    by_finger: dict[str, dict[str, float]] = {}
    for finger in sorted(set(fingers.tolist())):
        mask = fingers == finger
        by_finger[finger] = {
            "sample_count": int(np.sum(mask)),
            "raw_top1_accuracy": float(np.mean(raw_top1_hits[mask])),
            "raw_top3_accuracy": float(np.mean(raw_top3_hits[mask])),
            "system_identification_accuracy": float(np.mean(system_ident_correct[mask])),
            "system_verification_accept_rate": float(np.mean(verify_accept[mask])),
            "mean_registration_quality": float(np.mean([record.registration_quality for record in np.array(records, dtype=object)[mask]])),
        }

    metrics = {
        "dataset": dataset_summary,
        "evaluation_protocol": {
            "gallery_session": "session1",
            "probe_session": "session2",
            "class_definition": "subject + finger",
            "verify_threshold": VERIFY_THRESHOLD,
            "verify_margin_threshold": VERIFY_MARGIN_THRESHOLD,
            "identify_threshold": IDENTIFY_THRESHOLD,
            "identify_margin_threshold": IDENTIFY_MARGIN_THRESHOLD,
        },
        "identification": {
            "raw_top1_accuracy": float(np.mean(raw_top1_hits)),
            "raw_top3_accuracy": float(np.mean(raw_top3_hits)),
            "raw_top5_accuracy": float(np.mean(raw_top5_hits)),
            "system_identification_accuracy": float(np.mean(system_ident_correct)),
            "system_rejection_rate": float(np.mean(system_ident_reject)),
            "system_misidentification_rate": float(np.mean(system_ident_misidentify)),
        },
        "verification": {
            "pairwise_accuracy_at_verify_threshold": accuracy,
            "pairwise_precision": precision,
            "pairwise_recall_tar": recall,
            "pairwise_specificity_tnr": specificity,
            "pairwise_f1_score": f1_score,
            "pairwise_far": far,
            "pairwise_frr": frr,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "gar_at_far_1_percent": gar_at_far_1,
            "gar_at_far_0_1_percent": gar_at_far_01,
            "system_genuine_accept_rate": float(np.mean(verify_accept)),
            "system_false_accept_query_rate": float(np.mean(false_accept_query)),
        },
        "score_statistics": {
            "genuine_mean": float(np.mean(genuine_scores)),
            "genuine_std": float(np.std(genuine_scores)),
            "genuine_min": float(np.min(genuine_scores)),
            "genuine_max": float(np.max(genuine_scores)),
            "impostor_mean": float(np.mean(impostor_scores)),
            "impostor_std": float(np.std(impostor_scores)),
            "impostor_min": float(np.min(impostor_scores)),
            "impostor_max": float(np.max(impostor_scores)),
            "mean_true_margin": float(np.mean(true_margins)),
            "median_true_margin": float(np.median(true_margins)),
        },
        "quality": {
            "registration_quality_mean": float(np.mean([record.registration_quality for record in records])),
            "registration_quality_std": float(np.std([record.registration_quality for record in records])),
            "registration_quality_min": float(np.min([record.registration_quality for record in records])),
            "registration_quality_max": float(np.max([record.registration_quality for record in records])),
            "query_quality_mean": float(np.mean([record.query_quality_mean for record in records])),
            "mean_enrollment_images_used": float(np.mean([record.enrollment_images_used for record in records])),
            "mean_query_images_used": float(np.mean([record.query_images_used for record in records])),
        },
        "by_finger": by_finger,
        "artifacts": {
            "labels": labels,
            "genuine_scores": genuine_scores.tolist(),
            "impostor_scores": impostor_scores.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "registration_quality": [record.registration_quality for record in records],
            "query_quality": [record.query_quality_mean for record in records],
        },
    }
    return metrics


def save_metrics_json(output_root: Path, metrics: dict[str, Any]) -> None:
    serializable = dict(metrics)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_metrics_csv(output_root: Path, metrics: dict[str, Any]) -> None:
    rows = [
        ("raw_top1_accuracy", metrics["identification"]["raw_top1_accuracy"]),
        ("raw_top3_accuracy", metrics["identification"]["raw_top3_accuracy"]),
        ("raw_top5_accuracy", metrics["identification"]["raw_top5_accuracy"]),
        ("system_identification_accuracy", metrics["identification"]["system_identification_accuracy"]),
        ("system_rejection_rate", metrics["identification"]["system_rejection_rate"]),
        ("system_misidentification_rate", metrics["identification"]["system_misidentification_rate"]),
        ("pairwise_accuracy_at_verify_threshold", metrics["verification"]["pairwise_accuracy_at_verify_threshold"]),
        ("pairwise_precision", metrics["verification"]["pairwise_precision"]),
        ("pairwise_recall_tar", metrics["verification"]["pairwise_recall_tar"]),
        ("pairwise_specificity_tnr", metrics["verification"]["pairwise_specificity_tnr"]),
        ("pairwise_f1_score", metrics["verification"]["pairwise_f1_score"]),
        ("pairwise_far", metrics["verification"]["pairwise_far"]),
        ("pairwise_frr", metrics["verification"]["pairwise_frr"]),
        ("roc_auc", metrics["verification"]["roc_auc"]),
        ("average_precision", metrics["verification"]["average_precision"]),
        ("eer", metrics["verification"]["eer"]),
        ("gar_at_far_1_percent", metrics["verification"]["gar_at_far_1_percent"]),
        ("gar_at_far_0_1_percent", metrics["verification"]["gar_at_far_0_1_percent"]),
        ("system_genuine_accept_rate", metrics["verification"]["system_genuine_accept_rate"]),
        ("system_false_accept_query_rate", metrics["verification"]["system_false_accept_query_rate"]),
    ]
    with (output_root / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def plot_dataset_composition(metrics: dict[str, Any], charts_dir: Path) -> Path:
    counts = metrics["dataset"]["counts_by_session_and_finger"]
    labels = ["Session 1\nForefinger", "Session 1\nMiddlefinger", "Session 2\nForefinger", "Session 2\nMiddlefinger"]
    values = [
        counts.get("session1_forefinger", 0),
        counts.get("session1_middlefinger", 0),
        counts.get("session2_forefinger", 0),
        counts.get("session2_middlefinger", 0),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    bars = ax.bar(labels, values, color=[COLOR_PRIMARY, "#4aa3df", COLOR_ACCENT, "#7ed957"])
    ax.set_title("Dataset Composition by Session and Finger")
    ax.set_ylabel("Image Count")
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, str(value), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    path = charts_dir / "dataset_composition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_performance_summary(metrics: dict[str, Any], charts_dir: Path) -> Path:
    metric_labels = [
        "Raw Top-1 ID",
        "System ID",
        "System Verify",
        "Pairwise Verify",
        "ROC AUC",
    ]
    metric_values = [
        metrics["identification"]["raw_top1_accuracy"] * 100.0,
        metrics["identification"]["system_identification_accuracy"] * 100.0,
        metrics["verification"]["system_genuine_accept_rate"] * 100.0,
        metrics["verification"]["pairwise_accuracy_at_verify_threshold"] * 100.0,
        metrics["verification"]["roc_auc"] * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    y = np.arange(len(metric_labels))
    bars = ax.barh(y, metric_values, color=[COLOR_PRIMARY, COLOR_ACCENT, "#0c7c59", COLOR_WARM, "#6a4c93"])
    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage")
    ax.set_title("Performance Summary")
    ax.grid(axis="x", alpha=0.18)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    for bar, value in zip(bars, metric_values):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    path = charts_dir / "performance_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_distribution(metrics: dict[str, Any], charts_dir: Path) -> Path:
    genuine = np.asarray(metrics["artifacts"]["genuine_scores"], dtype=np.float32)
    impostor = np.asarray(metrics["artifacts"]["impostor_scores"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)
    bins = np.linspace(float(min(impostor.min(), genuine.min())), float(max(impostor.max(), genuine.max())), 40)
    ax.hist(impostor, bins=bins, alpha=0.65, color=COLOR_NEGATIVE, label="Impostor scores", density=True)
    ax.hist(genuine, bins=bins, alpha=0.65, color=COLOR_ACCENT, label="Genuine scores", density=True)
    ax.axvline(VERIFY_THRESHOLD, color="#222", linestyle="--", linewidth=1.4, label=f"Verify threshold ({VERIFY_THRESHOLD:.2f})")
    ax.set_title("Verification Score Distribution")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.16)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = charts_dir / "verification_score_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(metrics: dict[str, Any], charts_dir: Path) -> Path:
    fpr = np.asarray(metrics["artifacts"]["fpr"], dtype=np.float32)
    tpr = np.asarray(metrics["artifacts"]["tpr"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=160)
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, linewidth=2.2, label=f"AUC = {metrics['verification']['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.18)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = charts_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_far_frr_curve(metrics: dict[str, Any], charts_dir: Path) -> Path:
    fpr = np.asarray(metrics["artifacts"]["fpr"], dtype=np.float32)
    tpr = np.asarray(metrics["artifacts"]["tpr"], dtype=np.float32)
    thresholds = np.asarray(metrics["artifacts"]["thresholds"], dtype=np.float32)
    fnr = 1.0 - tpr
    finite_mask = np.isfinite(thresholds)
    thresholds = thresholds[finite_mask]
    fpr = fpr[finite_mask]
    fnr = fnr[finite_mask]
    order = np.argsort(thresholds)
    thresholds_sorted = thresholds[order]
    fpr_sorted = fpr[order]
    fnr_sorted = fnr[order]

    fig, ax = plt.subplots(figsize=(8.4, 5.1), dpi=160)
    ax.plot(thresholds_sorted, fpr_sorted, color=COLOR_NEGATIVE, linewidth=2.0, label="FAR")
    ax.plot(thresholds_sorted, fnr_sorted, color=COLOR_PRIMARY, linewidth=2.0, label="FRR")
    ax.axvline(metrics["verification"]["eer_threshold"], color="#333", linestyle="--", linewidth=1.2)
    ax.set_title("FAR / FRR Across Thresholds")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(alpha=0.18)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = charts_dir / "far_frr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_identification_by_finger(metrics: dict[str, Any], charts_dir: Path) -> Path:
    fingers = sorted(metrics["by_finger"].keys())
    raw_top1 = [metrics["by_finger"][finger]["raw_top1_accuracy"] * 100.0 for finger in fingers]
    system_id = [metrics["by_finger"][finger]["system_identification_accuracy"] * 100.0 for finger in fingers]
    system_verify = [metrics["by_finger"][finger]["system_verification_accept_rate"] * 100.0 for finger in fingers]

    x = np.arange(len(fingers))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    ax.bar(x - width, raw_top1, width=width, color=COLOR_PRIMARY, label="Raw Top-1 ID")
    ax.bar(x, system_id, width=width, color=COLOR_ACCENT, label="System ID")
    ax.bar(x + width, system_verify, width=width, color=COLOR_WARM, label="System Verify")
    ax.set_xticks(x)
    ax.set_xticklabels([finger.title() for finger in fingers])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Accuracy by Finger Type")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = charts_dir / "accuracy_by_finger.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_registration_quality(metrics: dict[str, Any], charts_dir: Path) -> Path:
    quality = np.asarray(metrics["artifacts"]["registration_quality"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=160)
    ax.hist(quality, bins=24, color="#5ab1ef", edgecolor="white")
    ax.axvline(float(np.mean(quality)), color=COLOR_NEGATIVE, linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(quality):.4f}")
    ax.set_title("Registration Quality Distribution")
    ax.set_xlabel("Registration Quality")
    ax.set_ylabel("Class Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.16)
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = charts_dir / "registration_quality_histogram.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_cover_page(pdf: PdfPages, metrics: dict[str, Any], preprocessing_summary: dict[str, Any] | None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "Knuckle Biometric Project Report", fontsize=24, fontweight="bold", color="#102542")
    fig.text(0.08, 0.912, "Evaluation date: March 26, 2026", fontsize=11, color="#4f5d75")
    fig.text(0.08, 0.87, "Project snapshot", fontsize=15, fontweight="bold", color="#102542")

    dataset = metrics["dataset"]
    identification = metrics["identification"]
    verification = metrics["verification"]
    quality = metrics["quality"]

    lines = [
        f"Subjects in prepared dataset: {dataset['subject_count']}",
        f"Knuckle classes (subject + finger): {dataset['class_count_total']}",
        f"Classes evaluated with both sessions: {dataset['class_count_evaluated']}",
        f"Images used in evaluation: {dataset['total_images_evaluated']}",
        f"Raw Top-1 identification accuracy: {identification['raw_top1_accuracy'] * 100:.2f}%",
        f"System identification accuracy: {identification['system_identification_accuracy'] * 100:.2f}%",
        f"System genuine accept rate: {verification['system_genuine_accept_rate'] * 100:.2f}%",
        f"Pairwise verification accuracy: {verification['pairwise_accuracy_at_verify_threshold'] * 100:.2f}%",
        f"ROC AUC: {verification['roc_auc']:.4f}",
        f"Equal error rate (EER): {verification['eer'] * 100:.2f}%",
        f"Average registration quality: {quality['registration_quality_mean']:.4f}",
    ]
    fig.text(0.08, 0.84, "\n".join(lines), fontsize=11.5, color="#1f2d3d", va="top", linespacing=1.55)

    method_lines = [
        "Evaluation protocol",
        "1. Session 1 images are used as enrollment/gallery samples.",
        "2. Session 2 images are used as probe/query samples.",
        "3. Each class is defined as one person plus one finger, matching the project design.",
        f"4. Verification threshold = {VERIFY_THRESHOLD:.3f}, margin = {VERIFY_MARGIN_THRESHOLD:.3f}.",
        f"5. Identification threshold = {IDENTIFY_THRESHOLD:.3f}, margin = {IDENTIFY_MARGIN_THRESHOLD:.3f}.",
    ]
    fig.text(0.08, 0.53, "\n".join(method_lines), fontsize=11.2, color="#1f2d3d", va="top", linespacing=1.55)

    if preprocessing_summary:
        preprocess_lines = [
            "Preprocessing summary",
            f"Prepared images processed: {preprocessing_summary.get('total_images_processed', 'N/A')}",
            f"Subjects covered: {preprocessing_summary.get('total_subjects', 'N/A')}",
            f"Point-cloud size: {preprocessing_summary.get('points_per_cloud', 'N/A')}",
            f"Average black-pixel ratio: {preprocessing_summary.get('average_black_pixel_ratio', 'N/A')}",
        ]
        fig.text(0.08, 0.34, "\n".join(preprocess_lines), fontsize=11.2, color="#1f2d3d", va="top", linespacing=1.55)

    fig.text(
        0.08,
        0.1,
        "This report is generated directly from the current project code and local dataset, not from README notes.",
        fontsize=10.2,
        color="#4f5d75",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_image_page(pdf: PdfPages, title: str, image_paths: list[Path]) -> None:
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(8.27, 11.69), dpi=160)
    fig.patch.set_facecolor("white")
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    for axis, image_path in zip(axes, image_paths):
        axis.axis("off")
        if image_path.exists():
            axis.imshow(plt.imread(image_path))
            axis.set_title(image_path.name.replace("_", " ").replace(".png", "").title(), fontsize=11)
        else:
            axis.text(0.5, 0.5, f"Missing asset:\n{image_path}", ha="center", va="center", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(
    output_root: Path,
    metrics: dict[str, Any],
    chart_paths: list[Path],
    preprocessing_summary: dict[str, Any] | None,
    preview_enhanced: Path,
    preview_pattern: Path,
) -> Path:
    pdf_path = output_root / "knuckle_project_showcase_report.pdf"
    with PdfPages(pdf_path) as pdf:
        render_cover_page(pdf, metrics, preprocessing_summary)
        render_image_page(pdf, "Dataset and Performance Graphs", chart_paths[:3])
        render_image_page(pdf, "Verification and Quality Graphs", chart_paths[3:])
        render_image_page(pdf, "Preprocessing Showcase", [preview_enhanced, preview_pattern])
    return pdf_path


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    charts_dir = output_root / "charts"
    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    records, dataset_summary = build_class_records(args.dataset.resolve())
    metrics = evaluate_records(records, dataset_summary)
    save_metrics_json(output_root, metrics)
    save_metrics_csv(output_root, metrics)

    preprocessing_summary = None
    if args.preprocessing_summary.exists():
        preprocessing_summary = json.loads(args.preprocessing_summary.read_text(encoding="utf-8"))

    chart_paths = [
        plot_dataset_composition(metrics, charts_dir),
        plot_performance_summary(metrics, charts_dir),
        plot_score_distribution(metrics, charts_dir),
        plot_roc_curve(metrics, charts_dir),
        plot_far_frr_curve(metrics, charts_dir),
        plot_identification_by_finger(metrics, charts_dir),
        plot_registration_quality(metrics, charts_dir),
    ]

    pdf_path = build_pdf_report(
        output_root,
        metrics,
        chart_paths,
        preprocessing_summary,
        args.preprocessing_preview_enhanced.resolve(),
        args.preprocessing_preview_pattern.resolve(),
    )

    summary = {
        "pdf_report": str(pdf_path),
        "metrics_json": str(output_root / "metrics_summary.json"),
        "metrics_csv": str(output_root / "metrics_table.csv"),
        "charts": [str(path) for path in chart_paths],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
