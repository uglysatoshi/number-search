"""Real-image datasets and augmentation for offline training."""

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..errors import DatasetError
from ..geometry import corners, tensor
from ..io import read_image
from ..metrics import normalize

SIZE = (384, 288)


def load_manifest(path):
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise DatasetError("Training manifest must be a JSON object")
        if data.get("schema_version") != 1:
            raise DatasetError("Unsupported training manifest")
        result = {}
        for key in ("reader_train", "reader_valid", "detector_train", "detector_valid"):
            result[key] = [
                {**r, "file": str((path.parent / r["image"]).resolve())} for r in data[key]
            ]
            if not result[key]:
                raise DatasetError(f"Empty training split: {key}")
        blocked = set(data.get("blocked_plate_labels", []))
        reader_train = {normalize(r["expected"]) for r in result["reader_train"]}
        reader_valid = {normalize(r["expected"]) for r in result["reader_valid"]}
        for row in result["reader_train"] + result["reader_valid"]:
            row["expected"] = normalize(row["expected"])
            if not re.fullmatch(
                r"[ABCEHKMOPTXY][0-9]{3}[ABCEHKMOPTXY]{2}[0-9]{2,3}", row["expected"]
            ):
                raise DatasetError(f"Unsupported OCR training label: {row['expected']}")
        if reader_train & reader_valid or (reader_train | reader_valid) & blocked:
            raise DatasetError("Training/validation/holdout plate labels overlap")
        hashes = [{r["sha256"] for r in result[k]} for k in ("detector_train", "detector_valid")]
        if hashes[0] & hashes[1]:
            raise DatasetError("Detector training and validation images overlap")
        reader_hashes = [{r["sha256"] for r in result[k]} for k in ("reader_train", "reader_valid")]
        if reader_hashes[0] & reader_hashes[1]:
            raise DatasetError("Reader training and validation images overlap")
        if data.get("holdout_manifest"):
            held = json.loads((path.parent / data["holdout_manifest"]).read_text(encoding="utf-8"))[
                "samples"
            ]
            held_labels = {normalize(r["expected"]) for r in held}
            held_hashes = {r["sha256"] for r in held}
            if (reader_train | reader_valid) & held_labels or any(
                h & held_hashes for h in hashes + reader_hashes
            ):
                raise DatasetError("Training/validation data overlaps the declared holdout")
        result["blocked_plate_labels"] = blocked
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(f"Cannot load training manifest {path}: {exc}") from exc


def local_records(root):
    items = []
    for path in sorted(Path(root).rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        image = (path.parent / data["imagePath"]).resolve()
        shapes = [
            s
            for s in data["shapes"]
            if s.get("label") == "number" and s.get("shape_type") == "polygon"
        ]
        if len(shapes) != 1:
            raise DatasetError(f"Expected one number polygon: {path}")
        if not image.is_relative_to(path.parent.resolve()):
            raise DatasetError(f"Image path escapes annotation directory: {path}")
        items.append(
            {"path": image, "text": normalize(path.stem), "points": corners(shapes[0]["points"])}
        )
    if not items:
        raise DatasetError(f"No LabelMe annotations in {root}")
    return items


def augment_gray(image):
    image = image.astype(np.float32)
    image = image * random.uniform(0.65, 1.25) + random.uniform(-25, 25)
    image += np.random.normal(0, random.uniform(0, 7), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 0.8))
    return image


class OCRDataset(Dataset):
    def __init__(self, rows, augment=False, strong_geometry=False):
        self.items = []
        self.augment = augment
        self.strong_geometry = strong_geometry
        for r in rows:
            image = cv2.imdecode(np.fromfile(r["file"], np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise DatasetError(f"Cannot decode training image: {r['file']}")
            self.items.append((cv2.resize(image, (160, 32)), r["expected"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, text = self.items[index]
        if self.augment:
            scale = random.uniform(0.8, 1.0) if self.strong_geometry else random.uniform(0.95, 1.04)
            matrix = cv2.getRotationMatrix2D(
                (80, 16),
                random.uniform(-3, 3) if self.strong_geometry else random.uniform(-2, 2),
                scale,
            )
            matrix[0, 1] += random.uniform(-0.025, 0.025)
            shift = (
                np.array([min(4, 80 * (1 - scale)), min(2, 16 * (1 - scale))])
                if self.strong_geometry
                else np.ones(2)
            )
            matrix[:, 2] += np.random.uniform(-1, 1, 2) * shift
            image = cv2.warpAffine(image, matrix, (160, 32), borderMode=cv2.BORDER_REPLICATE)
            image = augment_gray(image)
        return tensor(image), text


class SceneDataset(Dataset):
    def __init__(self, rows, augment=False, local=()):
        self.items = []
        self.augment = augment
        for r in rows:
            image = cv2.imdecode(np.fromfile(r["file"], np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise DatasetError(f"Cannot decode training image: {r['file']}")
            image = cv2.resize(image, SIZE)
            mask = np.zeros(SIZE[::-1], np.uint8)
            for _, cx, cy, w, h in r["boxes"]:
                x1, y1 = round((cx - w / 2) * SIZE[0]), round((cy - h / 2) * SIZE[1])
                x2, y2 = round((cx + w / 2) * SIZE[0]), round((cy + h / 2) * SIZE[1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            self.items.append((image, mask))
        for r in local:
            image = read_image(r["path"])
            h, w = image.shape[:2]
            pts = r["points"] * np.array([SIZE[0] / w, SIZE[1] / h])
            mask = np.zeros(SIZE[::-1], np.uint8)
            cv2.fillPoly(mask, [pts.round().astype(np.int32)], 255)
            self.items.append((cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), SIZE), mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, mask = self.items[index]
        if self.augment:
            matrix = cv2.getRotationMatrix2D(
                (SIZE[0] / 2, SIZE[1] / 2), random.uniform(-7, 7), random.uniform(0.8, 1.2)
            )
            matrix[:, 2] += np.array([random.uniform(-15, 15), random.uniform(-12, 12)])
            image = cv2.warpAffine(image, matrix, SIZE, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, SIZE, flags=cv2.INTER_NEAREST)
            image = augment_gray(image)
        return tensor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)), tensor(mask)


class GlyphDataset(Dataset):
    def __init__(self, samples, augment=False, strong_augmentation=False):
        self.samples, self.augment = samples, augment
        self.strong_augmentation = strong_augmentation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        image, label = self.samples[i]
        if self.augment:
            a = image[0].numpy()
            matrix = cv2.getRotationMatrix2D(
                (14, 16), random.uniform(-7, 7), random.uniform(0.85, 1.1)
            )
            if self.strong_augmentation:
                matrix[0, 1] += random.uniform(-0.14, 0.14)
                matrix[0, 0] *= random.uniform(0.85, 1.15)
            matrix[:, 2] += np.random.uniform(-1.5, 1.5, 2)
            a = cv2.warpAffine(a, matrix, (28, 32))
            if random.random() < 0.3:
                a = cv2.GaussianBlur(a, (3, 3), random.uniform(0.2, 0.8))
            a = np.clip(a * random.uniform(0.6, 1.15) + np.random.normal(0, 0.02, a.shape), 0, 1)
            image = torch.from_numpy(a.astype(np.float32))[None]
        return image, label
