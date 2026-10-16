"""Download the TweetEval sentiment dataset and save each split as a CSV.

Labels in TweetEval: 0 = negative, 1 = neutral, 2 = positive.
Output: data/train.csv, data/validation.csv, data/test.csv
        with columns: text, label, sentiment
"""
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(exist_ok=True)
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    for split_name, split in dataset.items():
        df = split.to_pandas()
        df["sentiment"] = df["label"].map(LABEL_NAMES)
        path = data_dir / f"{split_name}.csv"
        df.to_csv(path, index=False)
        counts = df["sentiment"].value_counts().to_dict()
        print(f"{split_name}: {len(df)} rows -> {path.name} {counts}")


if __name__ == "__main__":
    download()
