"""Load a versioned bundle with hash verification and strict state dictionaries."""

import json
import pickle
from pathlib import Path

import torch

from .config import ModelConfig
from .constants import WEIGHT_FILES
from .errors import ModelError
from .io import sha256
from .networks import AttentionReader, CharacterCNN, PlateDetector
from .reader import PlateTextReader


def load_state(cls, path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.eval()
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise ModelError(f"Cannot load checkpoint {path}: {exc}") from exc


def load_bundle(directory):
    directory = Path(directory).expanduser().resolve()
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ModelError("Model manifest must be a JSON object")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("architecture") != "attention-glyph-ensemble"
        ):
            raise ModelError("Unsupported model bundle format")
        if not isinstance(manifest.get("model_version"), str) or not manifest["model_version"]:
            raise ModelError("Model version must be a nonempty string")
        expected = set(WEIGHT_FILES) | {"pipeline.json"}
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != expected:
            raise ModelError(
                "Model manifest must declare exactly the selected five weights and pipeline.json"
            )
        for name, digest in files.items():
            if not isinstance(digest, str) or sha256(directory / name) != digest:
                raise ModelError(f"Model integrity check failed: {name}")
        config = ModelConfig.from_dict(
            json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        raise ModelError(f"Cannot open model bundle {directory}: {exc}") from exc
    detector = load_state(PlateDetector, directory / "detector.pt")
    detector.inference_size = config.detector_size
    attention = [
        load_state(AttentionReader, directory / name) for name in ("sequence.pt", "sequence_aux.pt")
    ]
    glyphs = [
        load_state(CharacterCNN, directory / name)
        for name in ("characters.pt", "characters_aux.pt")
    ]
    reader = PlateTextReader(attention, glyphs, config).eval()
    return detector, reader, manifest
