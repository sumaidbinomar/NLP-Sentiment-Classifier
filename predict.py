"""Predict the sentiment of custom text with the trained model.

Usage:
    python predict.py "I love this phone!"      # one or more texts as arguments
    python predict.py                           # interactive mode (empty line to quit)
"""
import argparse
from pathlib import Path

import joblib

from preprocessing import preprocess

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_model.joblib"
LABELS = ["negative", "neutral", "positive"]  # same order as label ids 0, 1, 2


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise SystemExit(f"Model not found at {path}. Run `python train_model.py` first.")
    return joblib.load(path)


def predict(model, text: str) -> dict:
    """Return the predicted label and the probability of each class."""
    probs = model.predict_proba([preprocess(text)])[0]
    scores = dict(zip(LABELS, (float(p) for p in probs)))
    return {"text": text, "sentiment": max(scores, key=scores.get), "probabilities": scores}


def show(result: dict) -> None:
    probs = ", ".join(f"{k} {v:.1%}" for k, v in result["probabilities"].items())
    print(f"  -> {result['sentiment'].upper()}  ({probs})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment prediction for custom text.")
    parser.add_argument("texts", nargs="*", help="text(s) to classify; omit for interactive mode")
    args = parser.parse_args()
    model = load_model()

    if args.texts:
        for text in args.texts:
            print(text)
            show(predict(model, text))
        return

    print("Enter text to classify (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        show(predict(model, text))


if __name__ == "__main__":
    main()
