import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import preprocess


def test_lowercases_and_removes_stopwords():
    assert preprocess("The movie IS a great one") == "movie great one"


def test_removes_urls_and_mentions():
    assert preprocess("@user check https://t.co/abc123 now") == "check"


def test_hashtag_symbol_removed_word_kept():
    assert "happy" in preprocess("#happy day").split()


def test_lemmatizes_verbs_and_plurals():
    assert preprocess("cats running") == "cat run"


def test_negation_is_kept():
    assert "not" in preprocess("This is not good").split()


def test_contractions_become_negation():
    assert preprocess("I don't like it") == "not like"
    assert preprocess("can't wait") == "not wait"


def test_empty_and_symbol_only_input():
    assert preprocess("") == ""
    assert preprocess("!!! 123 ???") == ""
