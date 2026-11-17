import json
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from number_search.bundle import load_bundle
from number_search.config import ModelConfig, default_model_dir
from number_search.errors import ImageError, ModelError
from number_search.io import read_image, validate_array, write_json
from number_search.metrics import edit_distance, normalize, wilson


def test_rgb_unicode_path_and_input_validation(tmp_path):
    path = tmp_path / "номер.png"
    Image.new("L", (30, 20), 128).save(path)
    image = read_image(path)
    assert image.shape == (20, 30, 3) and image.dtype == np.uint8
    path.write_bytes(b"not a PNG")
    with pytest.raises(ImageError):
        read_image(path)
    with pytest.raises(ImageError):
        read_image(tmp_path / "file.txt")
    with pytest.raises(ImageError):
        validate_array(np.zeros((20, 30, 3), np.float32))
    with pytest.raises(ImageError):
        validate_array(np.zeros((20, 30), np.uint8))
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(ImageError):
        read_image(path)


def test_atomic_json_preserves_previous_output_on_failure(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"text": "A777AA77"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("temperatures", [0, 1]),
        ("temperatures", [True, 1]),
        ("glyph_fusion_weight", float("nan")),
        ("glyph_fusion_weight", 2),
        ("detector_size", [8, 8]),
        ("tile_recovery", "false"),
    ],
)
def test_invalid_model_configuration(key, value):
    config = asdict(ModelConfig())
    config[key] = value
    with pytest.raises(ModelError):
        ModelConfig.from_dict(config)


def test_model_location_is_explicit_and_invalid_bundle_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBER_SEARCH_MODEL_DIR", str(tmp_path))
    assert default_model_dir() == tmp_path
    with pytest.raises(ModelError):
        load_bundle(tmp_path)
    manifest = {
        "schema_version": 1,
        "model_version": "test",
        "architecture": "attention-glyph-ensemble",
        "files": {"../../anything.pt": "bad"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with patch("number_search.bundle.load_state", side_effect=AssertionError("must not load")):
        with pytest.raises(ModelError):
            load_bundle(tmp_path)


def test_labels_and_metrics():
    assert normalize("а333мо99") == "A333MO99"
    assert normalize("с666сс61") == "C666CC61"
    assert edit_distance("A777AA77", "A777AA7") == 1
    assert edit_distance("", "A777AA77") == 8
    assert wilson(0, 30)[0] == pytest.approx(0)
    assert wilson(30, 30)[1] == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson(0, 0)
