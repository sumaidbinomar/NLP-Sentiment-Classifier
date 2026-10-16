"""Train and evaluate the sentiment classifier (TF-IDF + Logistic Regression).

Steps: load CSVs -> preprocess -> pick hyper-parameters on the validation set
-> evaluate once on the test set -> save model and results.

Outputs:
    models/sentiment_model.joblib   TF-IDF + Logistic Regression pipeline
    results/metrics.json            validation search + final test metrics
    results/classification_report.txt
    results/confusion_matrix.png
"""
import itertools
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline

from preprocessing import preprocess

ROOT = Path(__file__).parent
DATA_DIR, MODEL_DIR, RESULTS_DIR = ROOT / "data", ROOT / "models", ROOT / "results"
LABELS = ["negative", "neutral", "positive"]  # order matches label ids 0, 1, 2
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    """Read a CSV split and add a 'clean' column with the preprocessed text."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["clean"] = df["text"].astype(str).map(preprocess)
    return df


def build_pipeline(C: float, class_weight) -> Pipeline:
    return Pipeline([
        # Unigrams + bigrams; min_df=2 drops one-off typos; sublinear_tf dampens repeated words.
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=SEED)),
    ])


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Preprocessing text (this takes a moment)...")
    train, val, test = (load_split(s) for s in ("train", "validation", "test"))

    # --- Hyper-parameter search, scored on the validation set (test set stays untouched) ---
    search = []
    for C, cw in itertools.product([0.5, 1, 2, 5], [None, "balanced"]):
        model = build_pipeline(C, cw).fit(train["clean"], train["label"])
        macro_f1 = f1_score(val["label"], model.predict(val["clean"]), average="macro")
        search.append({"C": C, "class_weight": cw or "none", "val_macro_f1": round(macro_f1, 4)})
        print(f"  C={C:<4} class_weight={cw or 'none':<9} val macro-F1 = {macro_f1:.4f}")

    best = max(search, key=lambda r: r["val_macro_f1"])
    print(f"Best: C={best['C']}, class_weight={best['class_weight']}")

    # --- Final model, evaluated once on the test set ---
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    model = build_pipeline(best["C"], cw).fit(train["clean"], train["label"])
    pred = model.predict(test["clean"])

    report = classification_report(test["label"], pred, target_names=LABELS, digits=4)
    report_dict = classification_report(test["label"], pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(test["label"], pred)
    print("\nTest set results:\n" + report)
    print("Confusion matrix (rows = true, cols = predicted):\n", cm)

    # --- Save everything ---
    joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics = {
        "dataset": "cardiffnlp/tweet_eval (sentiment)",
        "sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "hyperparameter_search": search,
        "best_params": best,
        "test": {
            "accuracy": accuracy_score(test["label"], pred),
            "macro_f1": f1_score(test["label"], pred, average="macro"),
            "weighted_f1": f1_score(test["label"], pred, average="weighted"),
            "per_class": {k: report_dict[k] for k in LABELS},
            "confusion_matrix": cm.tolist(),
            "labels_order": LABELS,
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nSaved model to {MODEL_DIR} and results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
