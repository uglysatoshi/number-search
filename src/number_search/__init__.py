"""Russian license plate recognition with the selected v5 model bundle."""

__version__ = "0.1.0"
__all__ = ["Recognizer", "__version__"]


def __getattr__(name):
    if name == "Recognizer":
        from .predictor import Recognizer

        return Recognizer
    raise AttributeError(name)
