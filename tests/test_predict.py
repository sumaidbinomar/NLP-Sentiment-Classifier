import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import LABELS, MODEL_PATH, load_model, predict

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train_model.py first")


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_result_structure(model):
    result = predict(model, "What a day")
    assert result["sentiment"] in LABELS
    assert set(result["probabilities"]) == set(LABELS)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_clear_sentiment_examples(model):
    assert predict(model, "I absolutely love this, it's amazing and wonderful!")["sentiment"] == "positive"
    assert predict(model, "This is terrible, I hate it. Worst experience ever.")["sentiment"] == "negative"


def test_empty_text_does_not_crash(model):
    assert predict(model, "")["sentiment"] in LABELS
