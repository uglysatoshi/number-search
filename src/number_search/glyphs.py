"""Glyph segmentation retained for conservative attention refinement."""

import cv2
import numpy as np

from .constants import ALPHABET, LETTERS
from .geometry import tensor

LETTER_INDICES = np.array([ALPHABET.index(c) for c in LETTERS])
DIGIT_INDICES = np.array([ALPHABET.index(c) for c in "0123456789"])


def glyph_tensor(gray):
    gray = np.asarray(gray, np.uint8)
    h, w = gray.shape
    scale = min(24 / h, 20 / w)
    resized = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.full((32, 28), 255, np.uint8)
    y, x = (32 - resized.shape[0]) // 2, (28 - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tensor(255 - canvas)


def segments(crop, mode=0):
    gray = cv2.resize(crop, (320, 64))
    if mode % 2 == 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
        )
    if mode >= 2:
        # Detach characters from the plate frame; leave short character strokes intact.
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 45), np.uint8))
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((49, 1), np.uint8))
        mask = cv2.subtract(mask, cv2.bitwise_or(horizontal, vertical))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 12 <= h <= 56 and 2 <= w <= 46 and area >= 12 and y < 36 and 0.07 < w / h < 1.5:
            boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort()
    # Reject short text (RUS), bolts and border fragments relative to main glyph height.
    if boxes:
        tallest = sorted([b[3] for b in boxes], reverse=True)[:6]
        reference = np.median(tallest)
        boxes = [b for b in boxes if b[3] >= reference * 0.48]
    glyphs = [
        glyph_tensor(gray[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w + 1])
        for x, y, w, h in boxes
    ]
    return boxes, glyphs
