"""Text preprocessing for tweets: clean -> tokenize -> remove stop-words -> lemmatize."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Negations flip sentiment ("not good"), so we keep them even though NLTK lists them as stop-words.
NEGATIONS = {"no", "nor", "not", "never", "cannot"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
# Expand the irregular contractions first, then the generic "n't" -> " not".
IRREGULAR_CONTRACTIONS = {"won't": "will not", "can't": "can not", "ain't": "is not"}


def _ensure_nltk_data() -> None:
    """Download the NLTK resources we need (no-op if already present)."""
    for resource, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()
_lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/@mentions/hashtag symbols, expand contractions, keep letters only."""
    text = text.lower().replace("’", "'")
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    for contraction, expansion in IRREGULAR_CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = text.replace("n't", " not")
    return re.sub(r"[^a-z\s]", " ", text)  # also drops '#', digits, punctuation, emojis


def lemmatize(word: str) -> str:
    """Reduce a word to its base form (verb first: 'running'->'run', then noun: 'cats'->'cat')."""
    return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos="v"), pos="n")


def preprocess(text: str) -> str:
    """Full pipeline: returns the cleaned, stop-word-free, lemmatized text as one string."""
    tokens = clean_text(text).split()
    tokens = [lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
