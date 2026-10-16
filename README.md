# NLP Sentiment Classifier

A multi-class sentiment classifier (**negative / neutral / positive**) for tweets, built with
NLTK preprocessing, TF-IDF features and scikit-learn Logistic Regression. It is trained and
evaluated on the [TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset.

## How it works

1. **Preprocessing** (`preprocessing.py`): lowercase, remove URLs and `@mentions`, expand contractions
   (`don't` -> `do not`), keep letters only, remove English stop-words (negations such as
   `not`/`no`/`never` are kept because they change sentiment), then lemmatize with WordNet.
2. **Features**: TF-IDF over unigrams and bigrams (`min_df=2`, sublinear term frequency).
3. **Model**: Logistic Regression. `C` and `class_weight` are chosen on the validation set by macro-F1;
   the test set is used once, for the final evaluation.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python download_dataset.py      # download TweetEval into data/
python train_model.py           # preprocess, tune, train, evaluate, save model + results
python predict.py "I love this phone!"   # classify custom text (omit the text for interactive mode)
python -m pytest                # run tests
```

Example output:

```
Worst service ever, I will never come back.
  -> NEGATIVE  (negative 96.8%, neutral 1.8%, positive 1.4%)
```

## Results

Dataset sizes: 45,615 train / 2,000 validation / 12,284 test tweets.
Best settings: `C=2`, `class_weight="balanced"` (validation macro-F1 0.644).

Test set performance:

| Class    | Precision | Recall | F1     | Support |
|----------|-----------|--------|--------|---------|
| negative | 0.5594    | 0.6991 | 0.6215 | 3972    |
| neutral  | 0.6557    | 0.5252 | 0.5832 | 5937    |
| positive | 0.5365    | 0.5794 | 0.5571 | 2375    |

Accuracy 0.592, macro-F1 0.587, weighted-F1 0.591.

![Confusion matrix](results/confusion_matrix.png)

Full numbers are in `results/metrics.json` and `results/classification_report.txt`.

### Notes and limitations

- **Class imbalance**: negative tweets are ~16% of the training data, so balanced class weights are used
  (they scored higher than unweighted training for every value of `C` tried).
- **Distribution shift**: negative tweets are ~32% of the test set, which is part of why test scores are
  lower than validation scores.
- **Neutral is the hardest class**: many neutral tweets are predicted as negative.
- A bag-of-words model cannot capture sarcasm or context; it is a classical baseline, not a
  state-of-the-art result. Emojis are discarded during preprocessing.

## Project layout

```
download_dataset.py   fetch the dataset into data/
preprocessing.py      text cleaning, stop-word removal, lemmatization
train_model.py        tuning, training, evaluation
predict.py            command-line prediction
data/                 downloaded dataset (not tracked)
models/               saved model (sentiment_model.joblib)
results/              metrics, report, confusion matrix
tests/                unit tests
```
