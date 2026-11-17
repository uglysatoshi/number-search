"""Evaluation metrics, independent from model loading."""

import math

from .constants import ALPHABET, TRANSLATION
from .errors import DatasetError


def normalize(text):
    if not isinstance(text, str):
        raise DatasetError("Plate label must be text")
    text = text.upper().translate(TRANSLATION)
    if not text or any(c not in ALPHABET for c in text):
        raise DatasetError(f"Unsupported plate label: {text!r}")
    return text


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def wilson(correct, total, z=1.96):
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("Expected 0 <= correct <= total and total > 0")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]
