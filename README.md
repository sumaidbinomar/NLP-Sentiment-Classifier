# NLP Sentiment Classifier

A multi-class sentiment classifier (Positive / Negative / Neutral) built with
NLTK, TF-IDF features and scikit-learn Logistic Regression, trained on the
[TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) sentiment dataset.

> Status: project scaffold. Implementation, results and usage details will be added as the project is built.

## Planned pipeline

1. `download_dataset.py` – download TweetEval sentiment data into `data/`
2. `preprocessing.py` – clean text, remove stop-words, lemmatize (NLTK)
3. `train_model.py` – TF-IDF + Logistic Regression, evaluation (precision, recall, F1, confusion matrix)
4. `predict.py` – classify custom text from the command line

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
data/       downloaded dataset (not tracked)
models/     saved trained model
results/    metrics and plots
tests/      unit tests
```
