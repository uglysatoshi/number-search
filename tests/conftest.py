from pathlib import Path

import cv2
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", help="Run saved-weight/holdout regression"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="use --run-integration"))


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


@pytest.fixture(scope="session")
def recognizer():
    from number_search import Recognizer

    return Recognizer(ROOT / "models/v5", threads=4)
