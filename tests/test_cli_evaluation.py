import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from number_search.errors import DatasetError
from number_search.evaluation import evaluate, load_samples
from number_search.io import sha256


class FakeRecognizer:
    manifest = {"model_version": "test", "files": {}}

    def predict_array(self, image):
        assert isinstance(image, np.ndarray)
        return {"text": "A777AA77", "polygon": None, "detected": True}


def dataset(tmp_path):
    image = tmp_path / "arbitrary_upload.png"
    Image.new("RGB", (160, 80), "white").save(image)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "id": "one",
                        "image": image.name,
                        "expected": "A777AA77",
                        "sha256": sha256(image),
                    }
                ],
            }
        )
    )
    return manifest


def test_manifest_labels_are_not_inference_features(tmp_path):
    manifest = dataset(tmp_path)
    report = evaluate(
        FakeRecognizer(),
        manifest=manifest,
        report=tmp_path / "result.json",
        visualize_dir=tmp_path / "visualizations",
    )
    assert report["exact_accuracy"] == 1
    assert report["character_error_rate"] == 0
    assert report["filename_used_for_inference"] is False
    assert (tmp_path / "visualizations/index.html").exists()
    with pytest.raises(DatasetError):
        evaluate(FakeRecognizer(), manifest=manifest, report=manifest)
    with pytest.raises(DatasetError):
        evaluate(FakeRecognizer(), manifest=manifest, visualize_dir=tmp_path)


def test_manifest_integrity_and_path_traversal(tmp_path):
    manifest = dataset(tmp_path)
    (tmp_path / "arbitrary_upload.png").write_bytes(b"corrupted")
    with pytest.raises(DatasetError, match="integrity"):
        load_samples(manifest=manifest)
    data = json.loads(manifest.read_text())
    data["samples"][0]["image"] = "../outside.png"
    manifest.write_text(json.dumps(data))
    with pytest.raises(DatasetError, match="inside"):
        load_samples(manifest=manifest)


def test_nested_equal_basenames_do_not_overwrite_visualizations(tmp_path):
    rows = []
    for name in ("first", "second"):
        path = tmp_path / name / "upload.png"
        path.parent.mkdir()
        Image.new("RGB", (160, 80), "white").save(path)
        rows.append({"id": name, "image": f"{name}/upload.png", "expected": "A777AA77"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": rows}))
    result = evaluate(FakeRecognizer(), manifest=manifest, visualize_dir=tmp_path / "gallery")
    paths = [r["visualization"]["image"] for r in result["predictions"]]
    assert len(set(paths)) == 2


def test_cli_help_and_expected_error_are_clean(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "number_search", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "predict" in result.stdout
    result = subprocess.run(
        [sys.executable, "-m", "number_search", "verify-models", "--model-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "ERROR" in result.stderr and "Traceback" not in result.stderr
    assert not result.stdout


def test_importing_inference_does_not_import_training():
    code = "import sys; from number_search import Recognizer; assert not any(n.startswith('number_search.training') for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
