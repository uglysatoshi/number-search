"""Public API: load once, recognize many images without filenames as features."""

from pathlib import Path

import cv2
import torch

from .bundle import load_bundle
from .config import default_model_dir
from .io import read_image, validate_array
from .pipeline import recognize


class Recognizer:
    def __init__(self, model_dir=None, *, threads=None):
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
            cv2.setNumThreads(1)
        self.model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
        self.detector, self.reader, self.manifest = load_bundle(self.model_dir)

    @torch.inference_mode()
    def predict_array(self, image):
        validate_array(image)
        result = recognize(self.detector, self.reader, image)
        return {**result, "model_version": self.manifest["model_version"]}

    def predict(self, image_path):
        return self.predict_array(read_image(image_path))
