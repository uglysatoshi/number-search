"""Image/coordinate transformations shared by inference and training."""

import cv2
import numpy as np
import torch

OCR_SIZE = (160, 32)


def tensor(image):
    if image.ndim == 2:
        image = image[..., None]
    return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255)


def corners(points):
    pts = np.asarray(points, np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Expected four finite polygon corners")
    center = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
    pts = np.roll(pts, -np.argmin(pts.sum(1)), axis=0)
    if abs(cv2.contourArea(pts)) < 1:
        raise ValueError("Degenerate polygon")
    return pts


def crop_plate(image, points, margin=1.06):
    pts = corners(points)
    pts = (pts - pts.mean(0)) * margin + pts.mean(0)
    dst = np.array([[0, 0], [159, 0], [159, 31], [0, 31]], np.float32)
    result = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(pts, dst), OCR_SIZE, borderMode=cv2.BORDER_REPLICATE
    )
    return cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
