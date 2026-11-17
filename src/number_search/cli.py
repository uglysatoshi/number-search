"""Command-line interface; JSON on stdout, diagnostics on stderr."""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import default_model_dir
from .errors import NumberSearchError

LOG = logging.getLogger("number_search")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(
        prog="number-search", description="Recognize Russian license plates from JPG/PNG"
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--verbose", action="store_true")
    sub = root.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="Recognize a single image")
    predict.add_argument("--image", required=True, type=Path)
    predict.add_argument("--output", type=Path, help="Save prediction JSON")
    predict.add_argument("--visualize-dir", type=Path)
    check = sub.add_parser(
        "evaluate", help="Evaluate an explicit manifest or filename-labeled directory"
    )
    dataset = check.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--manifest", type=Path)
    dataset.add_argument("--split", type=Path)
    check.add_argument("--report", type=Path, default=Path("outputs/evaluation.json"))
    check.add_argument("--visualize-dir", type=Path)
    verify = sub.add_parser("verify-models", help="Verify the bundle and load all checkpoints")
    for command in (predict, check, verify):
        command.add_argument("--model-dir", type=Path, default=default_model_dir())
        command.add_argument("--threads", type=positive, default=4)
    train = sub.add_parser("train", help="Offline training with an explicit source manifest")
    train.add_argument("stage", choices=("detector", "reader", "glyphs"))
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--epochs", type=positive, default=25)
    train.add_argument("--batch-size", type=positive)
    train.add_argument("--lr", type=float, default=0.0005)
    train.add_argument("--seed", type=int, default=941)
    train.add_argument("--threads", type=positive, default=4)
    train.add_argument("--strong-geometry", action="store_true")
    train.add_argument(
        "--include-local", type=Path, help="Optional LabelMe directory for detector training"
    )
    download = sub.add_parser(
        "prepare-data", help="Restore selected training images from public source archives"
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--workers", type=positive, default=8)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        if args.command == "train":
            from .training.trainer import train

            train(args)
            return 0
        if args.command == "prepare-data":
            from .training.download import prepare

            prepare(args.manifest, workers=args.workers)
            return 0
        from .predictor import Recognizer

        recognizer = Recognizer(args.model_dir, threads=args.threads)
        if args.command == "verify-models":
            result = {
                "model_version": recognizer.manifest["model_version"],
                "verified": True,
                "files": recognizer.manifest["files"],
            }
        elif args.command == "predict":
            from .io import read_image, write_json

            if args.output and args.output.resolve() == args.image.resolve():
                raise NumberSearchError("Prediction output must not overwrite its image")
            image = read_image(args.image)
            result = recognizer.predict_array(image)
            if args.visualize_dir:
                from .visualization import save_prediction

                result["visualization"] = save_prediction(
                    image, {**result, "file": str(args.image)}, args.visualize_dir
                )
            if args.output:
                write_json(args.output, result)
        else:
            from .evaluation import evaluate

            result = evaluate(
                recognizer,
                manifest=args.manifest,
                directory=args.split,
                report=args.report,
                visualize_dir=args.visualize_dir,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (NumberSearchError, OSError, ValueError) as exc:
        LOG.error("%s", exc, exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
