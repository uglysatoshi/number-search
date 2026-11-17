"""Manifest- or folder-based evaluation with truth kept outside inference."""

import datetime
import json
from pathlib import Path

from .errors import DatasetError
from .io import IMAGE_SUFFIXES, read_image, sha256, write_json
from .metrics import edit_distance, normalize, wilson
from .visualization import save_gallery, save_prediction


def load_samples(*, manifest=None, directory=None):
    if (manifest is None) == (directory is None):
        raise DatasetError("Select exactly one manifest or image directory")
    if directory is not None:
        directory = Path(directory).resolve()
        try:
            paths = sorted(
                p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
            )
        except OSError as exc:
            raise DatasetError(f"Cannot read dataset {directory}: {exc}") from exc
        samples = [{"id": p.stem, "path": p, "expected": normalize(p.stem)} for p in paths]
    else:
        manifest = Path(manifest).resolve()
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                raise DatasetError("Dataset manifest must contain a samples array")
            if data.get("schema_version") != 1:
                raise DatasetError("Unsupported dataset manifest version")
            samples = []
            for row in data["samples"]:
                relative = Path(row["image"])
                path = (manifest.parent / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(manifest.parent):
                    raise DatasetError(
                        "Manifest image paths must stay inside the dataset directory"
                    )
                if row.get("sha256") and sha256(path) != row["sha256"]:
                    raise DatasetError(f"Dataset image integrity failed: {row['id']}")
                samples.append(
                    {"id": str(row["id"]), "path": path, "expected": normalize(row["expected"])}
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError(f"Cannot read dataset manifest {manifest}: {exc}") from exc
    if not samples:
        raise DatasetError("Dataset has no images")
    if len({s["id"] for s in samples}) != len(samples):
        raise DatasetError("Dataset sample IDs must be unique")
    return samples


def evaluate(recognizer, *, manifest=None, directory=None, report=None, visualize_dir=None):
    samples = load_samples(manifest=manifest, directory=directory)
    input_files = {s["path"] for s in samples}
    if manifest is not None:
        input_files.add(Path(manifest).resolve())
    if report is not None and Path(report).resolve() in input_files:
        raise DatasetError("Report must not overwrite a dataset input")
    if visualize_dir is not None:
        visualize_dir = Path(visualize_dir).resolve()
        if visualize_dir in {s["path"].parent for s in samples}:
            raise DatasetError("Use a separate visualization directory")
    predictions = []
    for sample in samples:
        image = read_image(sample["path"])
        result = recognizer.predict_array(image)
        result.update(
            {
                "id": sample["id"],
                "file": str(sample["path"]),
                "expected": sample["expected"],
                "correct": result["text"] == sample["expected"],
            }
        )
        if visualize_dir is not None:
            result["visualization"] = save_prediction(image, result, visualize_dir)
        predictions.append(result)
    count = len(predictions)
    correct = sum(r["correct"] for r in predictions)
    cer = sum(edit_distance(r["text"], r["expected"]) for r in predictions) / sum(
        len(r["expected"]) for r in predictions
    )
    result = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_version": recognizer.manifest["model_version"],
        "model_sha256": recognizer.manifest["files"],
        "images": count,
        "correct": correct,
        "exact_accuracy": correct / count,
        "character_error_rate": cer,
        "wilson_95_interval": wilson(correct, count),
        "annotation_used_for_inference": False,
        "filename_used_for_inference": False,
        "predictions": predictions,
    }
    if visualize_dir is not None:
        result["visualizations"] = save_gallery(predictions, visualize_dir, correct / count, cer)
    if report is not None:
        write_json(report, result)
    return result
