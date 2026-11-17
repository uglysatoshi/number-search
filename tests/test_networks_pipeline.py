import math
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from number_search.config import ModelConfig
from number_search.constants import ALPHABET
from number_search.geometry import crop_plate
from number_search.networks import AttentionEnsemble, AttentionReader, decode_attention, targets
from number_search.pipeline import recognize, refine_candidates, select_consensus
from number_search.reader import PlateTextReader


def test_attention_shapes_mask_and_gradient():
    torch.manual_seed(812)
    model = AttentionReader()
    output = model(torch.rand(2, 1, 32, 160))
    assert tuple(output.shape) == (2, 9, 23)
    assert (output[:, 0, 1:11] < -999).all()
    target = targets(["A777AA77", "B123CC199"])
    assert target[0, -1] == 0
    loss = torch.nn.functional.cross_entropy(output.flatten(0, 1), target.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.features[0].weight.grad.abs().sum() > 0


def test_probability_ensemble():
    class Fixed(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.value = torch.randn(2, 9, 23, generator=torch.Generator().manual_seed(seed))

        def forward(self, x):
            return self.value

    a, b = Fixed(1), Fixed(2)
    x = torch.zeros(2, 1, 32, 160)
    actual = AttentionEnsemble([a, b], [1.3, 1.3])(x).softmax(-1)
    assert torch.allclose(
        actual, ((a(x) / 1.3).softmax(-1) + (b(x) / 1.3).softmax(-1)) / 2, atol=1e-6
    )


def test_glyph_fusion_rejects_border_artifacts():
    class Glyphs(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((len(x), 22), -20.0)
            for i, glyph in enumerate(x):
                logits[i, int(glyph[0, 0, 0])] = 10.0
            return logits

    text = "A777AA77"
    glyphs = [torch.full((1, 32, 28), float(ALPHABET.index(c))) for c in text]
    boxes = [(i * 30, 10, 15, 30) for i in range(8)]
    logits = torch.full((1, 9, 23), -30.0)
    for j, index in enumerate(targets([text])[0]):
        logits[0, j, index] = 0.0
    logits[0, 4, ALPHABET.index("A") + 1] = math.log(0.4)
    logits[0, 4, ALPHABET.index("K") + 1] = math.log(0.6)
    reader = PlateTextReader([torch.nn.Identity(), torch.nn.Identity()], [Glyphs()], ModelConfig())
    with patch("number_search.reader.segments", return_value=(boxes, glyphs)):
        assert decode_attention(reader.fuse_glyphs(logits, [np.zeros((32, 160), np.uint8)])) == [
            text
        ]
    bad = [(x, 0, w, h) for x, y, w, h in boxes]
    with patch("number_search.reader.segments", return_value=(bad, glyphs)):
        assert decode_attention(reader.fuse_glyphs(logits, [np.zeros((32, 160), np.uint8)])) == [
            "A777KA77"
        ]


def test_geometry_and_refinement():
    image = np.random.default_rng(42).integers(0, 256, (80, 200, 3), dtype=np.uint8)
    points = [[10, 10], [180, 15], [175, 65], [15, 60]]
    assert np.array_equal(crop_plate(image, points), crop_plate(image, points[::-1]))
    blank = np.full((160, 300, 3), 80, np.uint8)
    detection = np.array([[30, 40], [270, 40], [270, 120], [30, 120]], np.float32)
    assert refine_candidates(blank, detection) == []
    cv2.fillConvexPoly(
        blank, np.array([[50, 60], [250, 48], [245, 110], [55, 105]]), (240, 240, 240)
    )
    assert refine_candidates(blank, detection)


def test_consensus_and_conditional_recovery():
    quad = np.array([[40, 30], [160, 30], [160, 70], [40, 70]], np.float32)
    readings = [("M007OK163", -0.169), ("M007OK163", -0.195), ("M007PK163", -0.159)]
    result = select_consensus(readings, [(quad, 0.95, m, 1) for m in (1.0, 1.1, 1.22)])
    assert result["text"] == "M007OK163" and result["supporting_views"] == 2
    image = np.zeros((100, 200, 3), np.uint8)
    reader = SimpleNamespace(recovery_log_score=-0.3, tile_recovery=False)
    good = {"text": "A777AA77", "recognition_score": -0.01, "selection_score": -0.01}
    with patch("number_search.pipeline.locate_candidates", return_value=[(quad, 0.9)]) as locate:
        with patch("number_search.pipeline.read_candidates", return_value=good):
            assert recognize(None, reader, image) == good
            assert locate.call_count == 1
    with patch("number_search.pipeline.locate_candidates", return_value=[(quad, 0.9)]) as locate:
        with patch("number_search.pipeline.read_candidates", side_effect=[None, dict(good)]):
            assert recognize(None, reader, image)["recovery"] == "roi_redetection"
            assert locate.call_count == 2
