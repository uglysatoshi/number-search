"""Validated image input and atomic JSON output."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ImageError

IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png"))
MAX_PIXELS = 40_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path):
    path = Path(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ImageError(f"Supported image formats: JPG, JPEG, PNG: {path}")
    try:
        with Image.open(path) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ImageError(f"Unsupported image content: {path}")
            width, height = image.size
            if min(width, height) < 8 or width * height > MAX_PIXELS:
                raise ImageError(
                    f"Image must be at least 8×8 and at most {MAX_PIXELS} pixels: {path}"
                )
            # Keep the training/holdout pixel convention; do not rotate using EXIF.
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Cannot read image {path}: {exc}") from exc


def validate_array(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ImageError("Expected an RGB uint8 NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ImageError("Expected an RGB array shaped (height, width, 3), at least 8×8")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ImageError(f"Image exceeds {MAX_PIXELS} pixels")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
