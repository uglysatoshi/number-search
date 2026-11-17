import numpy as np
from PIL import Image

from number_search.visualization import render_prediction, save_gallery, save_prediction


def test_source_pixels_are_preserved_and_selected_crop_is_shown(tmp_path):
    image = np.full((100, 240, 3), 160, np.uint8)
    original = image.copy()
    result = {
        "file": "A777AA77.jpg",
        "text": "A777AA77",
        "expected": "A777AA77",
        "correct": True,
        "polygon": [[20, 30], [220, 30], [220, 70], [20, 70]],
        "crop_margin": 1.22,
    }
    canvas, crop = render_prediction(image, result)
    assert np.array_equal(image, original)
    assert crop.size == (160, 32) and canvas.width > image.shape[1]


def test_empty_prediction_gallery_has_no_fabricated_crop(tmp_path):
    result = {
        "file": "A&B.jpg",
        "text": "",
        "expected": "B777OP77",
        "correct": False,
        "polygon": None,
    }
    result["visualization"] = save_prediction(np.zeros((100, 200, 3), np.uint8), result, tmp_path)
    assert "crop" not in result["visualization"]
    links = save_gallery([result], tmp_path, 0.0, 1.0)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "A&amp;B.jpg.result.jpg" in html and "<script" not in html
    with Image.open(links["summary"]) as image:
        assert image.width == 1200
