"""Offline trainers for the maintained architectures."""

import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..bundle import load_state
from ..errors import DatasetError
from ..io import sha256, write_json
from ..metrics import edit_distance
from ..networks import AttentionReader, CharacterCNN, PlateDetector, decode_attention, targets
from .data import SIZE, GlyphDataset, OCRDataset, SceneDataset, load_manifest, local_records

LOG = logging.getLogger(__name__)


def glyph_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += (output.argmax(1) == y).sum().item()
            count += len(y)
    return {"loss": total / count, "exact_accuracy": correct / count}


def glyph_data(rows, augment):
    from ..constants import ALPHABET
    from ..glyphs import segments

    samples = []
    for image, text in OCRDataset(rows).items:
        for mode in range(4):
            boxes, glyphs = segments(image, mode)
            if len(glyphs) == len(text):
                samples.extend((g, ALPHABET.index(c)) for g, c in zip(glyphs, text))
                break
    if not samples:
        raise DatasetError("No complete glyph segmentations in this dataset")
    return GlyphDataset(samples, augment=augment, strong_augmentation=True)


def train(args):
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise DatasetError("Learning rate must be positive and finite")
    if not 0 <= args.seed < 2**32:
        raise DatasetError("Seed must be in [0, 2**32)")
    if (args.output / "manifest.json").exists():
        raise DatasetError("Train into a new run directory, not a model release")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_manifest(args.manifest)
    prefix = "detector" if args.stage == "detector" else "reader"
    for row in data[prefix + "_train"] + data[prefix + "_valid"]:
        if not Path(row["file"]).is_file():
            raise DatasetError(f"Missing training image {row['file']}; run prepare-data first")
        if sha256(row["file"]) != row["sha256"]:
            raise DatasetError(f"Training image checksum mismatch: {row['file']}; run prepare-data")
    if args.stage == "detector":
        local = local_records(args.include_local) if args.include_local else []
        local = [r for r in local if r["text"] not in data["blocked_plate_labels"]]
        training = SceneDataset(data["detector_train"], True, local)
        validation = SceneDataset(data["detector_valid"])
        cls = PlateDetector
        run = detector_epoch
        filename = "detector.pt"
        batch = args.batch_size or 8
    elif args.stage == "reader":
        training = OCRDataset(data["reader_train"], True, args.strong_geometry)
        validation = OCRDataset(data["reader_valid"])
        cls = AttentionReader
        run = reader_epoch
        filename = "sequence.pt"
        batch = args.batch_size or 64
    else:
        training = glyph_data(data["reader_train"], True)
        validation = glyph_data(data["reader_valid"], False)
        cls = CharacterCNN
        run = glyph_epoch
        filename = "characters.pt"
        batch = args.batch_size or 128
    model = load_state(cls, args.resume) if args.resume else cls()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.1
    )
    train_loader = DataLoader(training, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=batch)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = (-1, -float("inf"))
    for epoch_number in range(1, args.epochs + 1):
        start = time.perf_counter()
        metrics = {
            "epoch": epoch_number,
            "train": run(model, train_loader, optimizer),
            "valid": run(model, valid_loader),
            "seconds": time.perf_counter() - start,
        }
        history.append(metrics)
        score = (
            metrics["valid"].get("mask_iou", metrics["valid"].get("exact_accuracy")),
            -metrics["valid"].get("cer", metrics["valid"]["loss"]),
        )
        if score > best:
            best = score
            temporary = args.output / (filename + ".tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch_number,
                    "metrics": metrics,
                    "architecture": args.stage,
                    "inference_size": list(SIZE) if args.stage == "detector" else [160, 32],
                },
                temporary,
            )
            temporary.replace(args.output / filename)
        scheduler.step()
        write_json(args.output / "history.json", history)
        LOG.info("epoch=%s validation=%s", epoch_number, metrics["valid"])


def segmentation_loss(logits, target):
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probability * target).sum(dims) + 1) / (
        probability.sum(dims) + target.sum(dims) + 1
    )
    return (
        F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(15.0))
        + (1 - dice).mean()
    )


def detector_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total = iou = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, target in loader:
            output = model(images)
            loss = segmentation_loss(output, target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            a, b = output.detach().sigmoid() > 0.5, target > 0.5
            iou += ((a & b).sum((1, 2, 3)) / (a | b).sum((1, 2, 3)).clamp_min(1)).sum().item()
            total += loss.item() * len(images)
            count += len(images)
    return {"loss": total / count, "mask_iou": iou / count}


def reader_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    count = total = correct = errors = characters = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, texts in loader:
            output = model(x)
            loss = nn.functional.cross_entropy(output.flatten(0, 1), targets(texts).flatten())
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            predicted = decode_attention(output.detach())
            correct += sum(a == b for a, b in zip(predicted, texts))
            errors += sum(edit_distance(a, b) for a, b in zip(predicted, texts))
            characters += sum(map(len, texts))
            count += len(texts)
            total += loss.item() * len(texts)
    return {"loss": total / count, "exact_accuracy": correct / count, "cer": errors / characters}
