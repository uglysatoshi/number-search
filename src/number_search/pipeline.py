"""Geometric refinement, view consensus and conditional recovery."""

import cv2
import numpy as np

from .detection import locate_candidates
from .geometry import corners, crop_plate


def refine_candidates(image, points):
    """Find plate borders at original resolution, within a CNN detection only."""
    center = points.mean(0)
    low = np.maximum(np.floor(center + (points.min(0) - center) * 1.5), 0).astype(int)
    high = np.minimum(
        np.ceil(center + (points.max(0) - center) * 1.5), [image.shape[1], image.shape[0]]
    ).astype(int)
    roi = image[low[1] : high[1], low[0] : high[0]]
    if min(roi.shape[:2]) < 3:
        return []
    factor = max(1.0, 100 / roi.shape[0])
    gray = cv2.cvtColor(cv2.resize(roi, None, fx=factor, fy=factor), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, 40, 120)
    faint_edges = cv2.morphologyEx(
        cv2.Canny(blurred, 12, 40), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    area0 = abs(cv2.contourArea(points.astype(np.float32)))
    proposals = []
    for mask in (binary, edges, faint_edges):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            if abs(cv2.contourArea(quad)) / factor**2 < max(1, area0 * 0.18):
                continue
            quad = corners(quad.reshape(4, 2).astype(np.float32) / factor + low)
            area = abs(cv2.contourArea(quad))
            if not area0 * 0.18 < area < area0 * 1.7:
                continue
            w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
            h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
            if not 2.0 < w / max(h, 1) < 7.5:
                continue
            distance = np.linalg.norm((quad.mean(0) - center) / (points.max(0) - points.min(0) + 1))
            if distance > 0.4:
                continue
            score = -distance - abs(np.log(area / area0)) * 0.15
            proposals.append((score, quad))
    proposals.sort(key=lambda v: v[0], reverse=True)
    return [p for _, p in proposals[:4]]


def select_consensus(readings, metadata):
    """Pool character likelihood across crop margins of the same polygon.

    The best view supplies coordinates/crop; other views contribute evidence
    for the text. Scores remain heuristic, not calibrated probabilities.
    """
    groups = {}
    for (text, score), (quad, detection_score, margin, group) in zip(readings, metadata):
        item = groups.setdefault(group, {"views": 0, "texts": {}})
        item["views"] += 1
        if text:
            record = {
                "text": text,
                "detected": True,
                "detection_score": detection_score,
                "recognition_score": score,
                "polygon": quad.round(2).tolist(),
                "crop_margin": margin,
            }
            item["texts"].setdefault(text, []).append(record)
    best = None
    for group in groups.values():
        for text, records in group["texts"].items():
            value = float(
                np.logaddexp.reduce([r["recognition_score"] * len(text) for r in records])
            )
            value = (value - np.log(group["views"])) / len(text)
            if best is None or value > best["selection_score"]:
                best = dict(
                    max(records, key=lambda r: r["recognition_score"]),
                    selection_score=value,
                    supporting_views=len(records),
                )
    return best


def read_candidates(model, image, detections):
    crops, metadata = [], []
    seen = set()
    for points, detection_score in detections:
        for quad in [points] + refine_candidates(image, points):
            key = quad.tobytes()
            if key in seen:
                continue
            seen.add(key)
            group = len(seen)
            for margin in (1.0, 1.1, 1.22):
                crops.append(crop_plate(image, quad, margin))
                metadata.append((quad, detection_score, margin, group))
    return select_consensus(model.read_plate_crops(crops), metadata)


def recognize(detector, model, image):
    detections = locate_candidates(detector, image)
    best = read_candidates(model, image, detections)
    recovery_threshold = model.recovery_log_score
    if best is not None and best.get("recognition_score", 0.0) >= recovery_threshold:
        return best
    # Recovery is conditional on a missing or low-scoring initial result.
    # Reuse the same CNN on a local ROI to recover fine detail lost at full-frame scale.
    zoomed = []
    for points, _ in detections[:3]:
        center = points.mean(0)
        low = np.maximum(np.floor(center + (points.min(0) - center) * 1.5), 0).astype(int)
        high = np.minimum(
            np.ceil(center + (points.max(0) - center) * 1.5), [image.shape[1], image.shape[0]]
        ).astype(int)
        roi = image[low[1] : high[1], low[0] : high[0]]
        if min(roi.shape[:2]) < 8:
            continue
        zoomed.extend((quad + low, score) for quad, score in locate_candidates(detector, roi))
    if zoomed:
        recovered = read_candidates(model, image, zoomed)
        if recovered is not None and (
            best is None or recovered.get("selection_score", 0.0) > best.get("selection_score", 0.0)
        ):
            best = dict(recovered, recovery="roi_redetection")
    if model.tile_recovery and (
        best is None or best.get("recognition_score", 0.0) < recovery_threshold
    ):
        height, width = image.shape[:2]
        tiled = []
        for x1, y1, x2, y2 in (
            (0, 0, 0.6, 0.6),
            (0.4, 0, 1, 0.6),
            (0, 0.4, 0.6, 1),
            (0.4, 0.4, 1, 1),
        ):
            left, top, right, bottom = (
                round(x1 * width),
                round(y1 * height),
                round(x2 * width),
                round(y2 * height),
            )
            tile = image[top:bottom, left:right]
            if min(tile.shape[:2]) < 8:
                continue
            tiled.extend(
                (quad + np.array([left, top], np.float32), score)
                for quad, score in locate_candidates(detector, tile)
            )
        recovered = read_candidates(model, image, tiled)
        if recovered is not None and (
            best is None or recovered["selection_score"] > best["selection_score"]
        ):
            best = dict(recovered, recovery="tile_search")
    return best or {"text": "", "detected": False, "detection_score": 0.0, "polygon": None}
