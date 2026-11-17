import argparse
import hashlib
import io
import json
import zipfile
from unittest.mock import patch

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from number_search.errors import DatasetError
from number_search.io import sha256
from number_search.training.download import RemoteZip, prepare
from number_search.training.trainer import train


def test_range_zip_crc_without_network():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train/img/test.png", b"source-image-bytes")
    payload = stream.getvalue()

    class MemoryZip(RemoteZip):
        def fetch(self, start, end):
            assert 0 <= start <= end < len(payload)
            return payload[start : end + 1]

    remote = MemoryZip("owner/repo", "revision", len(payload), "train.zip")
    assert remote.read("train/img/test.png") == b"source-image-bytes"
    remote.entries["train/img/test.png"]["crc"] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        remote.read("train/img/test.png")


def test_cache_restore_is_verified_and_confined(tmp_path):
    manifest = tmp_path / "data/manifests/training.json"
    manifest.parent.mkdir(parents=True)
    raw = b"verified source"
    row = {
        "image": "../cache/v5/ocr/train/00000.png",
        "source": "ocr",
        "archive": "train.zip",
        "source_entry": "train/img/plate.png",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    data = {
        "schema_version": 1,
        "sources": {"ocr": {"repo": "owner/repo", "revision": "a" * 40}},
        "reader_train": [row],
        "reader_valid": [],
        "detector_train": [],
        "detector_valid": [],
    }
    manifest.write_text(json.dumps(data))

    class FakeZip:
        def __init__(self, *args):
            pass

        def read(self, name):
            assert name == "train/img/plate.png"
            return raw

    tree = io.BytesIO(json.dumps([{"path": "train.zip", "size": 12345}]).encode())
    with patch("number_search.training.download.RemoteZip", FakeZip):
        with patch("number_search.training.download.request", return_value=tree) as request:
            prepare(manifest, workers=1)
            prepare(manifest, workers=1)
            assert request.call_count == 1
    target = (manifest.parent / row["image"]).resolve()
    assert target.read_bytes() == raw
    data["reader_train"][0]["image"] = "../../../outside.png"
    manifest.write_text(json.dumps(data))
    with pytest.raises(DatasetError, match="cache"):
        prepare(manifest, workers=1)


@pytest.mark.parametrize("stage", ["reader", "detector", "glyphs"])
def test_offline_training_exports_a_valid_checkpoint(tmp_path, stage):
    labels = ["A123BC77", "B234CC199"]
    font = (
        ImageFont.truetype("DejaVuSans.ttf", 22)
        if __import__("os").name != "nt"
        else ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    )
    rows = []
    for i, text in enumerate(labels):
        image = Image.new("RGB", (160, 32), "white")
        draw = ImageDraw.Draw(image)
        for j, char in enumerate(text):
            draw.text((3 + j * 17, 2), char, font=font, fill="black")
        path = tmp_path / f"{i}.png"
        image.save(path)
        rows.append(
            {
                "image": path.name,
                "expected": text,
                "sha256": sha256(path),
                "boxes": [[0, 0.5, 0.5, 0.7, 0.5]],
            }
        )
    data = {
        "schema_version": 1,
        "reader_train": [rows[0]],
        "reader_valid": [rows[1]],
        "detector_train": [rows[0]],
        "detector_valid": [rows[1]],
        "blocked_plate_labels": [],
    }
    manifest = tmp_path / "training.json"
    manifest.write_text(json.dumps(data))
    args = argparse.Namespace(
        stage=stage,
        manifest=manifest,
        output=tmp_path / "run",
        resume=None,
        epochs=1,
        batch_size=2,
        lr=0.0001,
        seed=3,
        threads=4,
        strong_geometry=False,
        include_local=None,
    )
    train(args)
    filename = {"reader": "sequence.pt", "detector": "detector.pt", "glyphs": "characters.pt"}[
        stage
    ]
    checkpoint = torch.load(args.output / filename, weights_only=True)
    assert checkpoint["epoch"] == 1 and checkpoint["state_dict"]
    assert (args.output / "history.json").exists()
