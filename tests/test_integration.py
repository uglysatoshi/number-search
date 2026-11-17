import json

import pytest
from conftest import ROOT

from number_search.evaluation import evaluate
from number_search.io import read_image


@pytest.mark.integration
def test_all_fifty_predictions_preserved(recognizer, tmp_path):
    result = evaluate(
        recognizer,
        manifest=ROOT / "data/external/holdout/manifest.json",
        report=tmp_path / "report.json",
    )
    original = json.loads(
        (ROOT / "reports/holdout/original_results.json").read_text(encoding="utf-8")
    )
    expected = {str(p["id"]): p["text"] for p in original["predictions"]}
    assert {p["id"]: p["text"] for p in result["predictions"]} == expected
    assert result["correct"] == 42 and result["images"] == 50
    assert result["character_error_rate"] == pytest.approx(0.06038647342995169)


@pytest.mark.integration
def test_name_independence_and_library_usage(recognizer, tmp_path):
    source = ROOT / "data/external/holdout/images/003.jpg"
    renamed = tmp_path / "unrelated.jpg"
    renamed.write_bytes(source.read_bytes())
    assert recognizer.predict(source) == recognizer.predict(renamed)
    assert recognizer.predict_array(read_image(source))["text"] == "C840BE197"
