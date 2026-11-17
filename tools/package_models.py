"""Create a new immutable model bundle from five selected compatible checkpoints."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from number_search.bundle import load_bundle
from number_search.config import ModelConfig
from number_search.constants import WEIGHT_FILES
from number_search.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new version directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ModelConfig.from_dict(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="bundle-", dir=args.output.parent))
    try:
        for name in WEIGHT_FILES:
            shutil.copyfile(args.weights / name, stage / name)
        write_json(stage / "pipeline.json", config)
        manifest = {
            "schema_version": 1,
            "model_version": args.version,
            "architecture": "attention-glyph-ensemble",
            "files": {name: sha256(stage / name) for name in (*WEIGHT_FILES, "pipeline.json")},
        }
        write_json(stage / "manifest.json", manifest)
        load_bundle(stage)
        os.replace(stage, args.output)
    finally:
        if (
            stage.exists()
            and stage.resolve().parent == args.output.parent.resolve()
            and stage.name.startswith("bundle-")
        ):
            shutil.rmtree(stage)
    print(args.output)


if __name__ == "__main__":
    main()
