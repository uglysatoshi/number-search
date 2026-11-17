"""CNN plate candidate extraction."""

import cv2
import numpy as np
import torch

from .geometry import corners, tensor


def locate_candidates(detector, image):
    input_size = detector.inference_size
    # Match the grayscale training distribution for both color and gray uploads.
    gray_rgb = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    with torch.no_grad():
        probability = (
            detector(tensor(cv2.resize(gray_rgb, input_size)).unsqueeze(0)).sigmoid()[0, 0].numpy()
        )
    mask = (probability > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6 or w < h:
            continue
        score = float(probability[labels == i].mean()) * float(area) ** 0.5
        candidates.append((score, i))
    results = []
    for _, best in sorted(candidates, reverse=True)[:6]:
        yy, xx = np.where(labels == best)
        pts = np.stack([xx, yy], 1).astype(np.float32)
        # A low-resolution mask cannot reliably determine perspective corners.
        box = corners(cv2.boxPoints(cv2.minAreaRect(pts)))
        box *= np.array([image.shape[1] / input_size[0], image.shape[0] / input_size[1]])
        results.append((box, float(probability[labels == best].mean())))
    return results
