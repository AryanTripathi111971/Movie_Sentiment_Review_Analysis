"""Convert review text into the integer sequence expected by the Keras IMDB model."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
import numpy as np
from . import config

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
_HTML_BREAK_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

class EmptyTextError(ValueError):
    """Raised when the review contains no usable words."""

class VocabularyError(RuntimeError):
    """Raised when the IMDB word index cannot be loaded."""

@dataclass(frozen=True)
class Token:
    word: str
    known: bool

@dataclass(frozen=True)
class Encoded:
    ids: np.ndarray
    tokens: list[Token]
    word_count: int
    unknown_count: int
    truncated: bool

def tokenize(text: str) -> list[str]:
    text = _HTML_BREAK_RE.sub(" ", text.lower())
    return _WORD_RE.findall(text)

def load_word_index() -> dict[str, int]:
    if config.WORD_INDEX_PATH.exists():
        with open(config.WORD_INDEX_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        from keras.datasets import imdb
        index = imdb.get_word_index()
    except Exception as exc:
        raise VocabularyError(
            "Could not download the IMDB word list. Connect to the internet once and restart "
            "the app (or run 'python scripts/download_vocab.py')."
        ) from exc
    config.WORD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.WORD_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return index

class ReviewEncoder:
    def __init__(self, word_index: dict[str, int], num_words: int = config.NUM_WORDS,
                 max_len: int = config.MAX_LEN, padding: str = config.PADDING,
                 truncating: str = config.TRUNCATING):
        if padding not in {"pre", "post"} or truncating not in {"pre", "post"}:
            raise ValueError("padding / truncating must be 'pre' or 'post'")
        self.word_index = word_index
        self.num_words = num_words
        self.max_len = max_len
        self.padding = padding
        self.truncating = truncating

    def _word_id(self, word: str) -> tuple[int, bool]:
        idx = self.word_index.get(word)
        if idx is None:
            return config.OOV_ID, False
        wid = idx + config.INDEX_FROM
        if wid >= self.num_words:
            return config.OOV_ID, False
        return wid, True

    def encode(self, text: str) -> Encoded:
        words = tokenize(text)
        if not words:
            raise EmptyTextError("No words found. Write a few words about the movie.")
        items = [(config.START_ID, None, True)]
        for word in words:
            wid, known = self._word_id(word)
            items.append((wid, word, known))
        truncated = len(items) > self.max_len
        if truncated:
            items = items[-self.max_len:] if self.truncating == "pre" else items[:self.max_len]
        ids = [item[0] for item in items]
        pad = [0] * (self.max_len - len(ids))
        ids = pad + ids if self.padding == "pre" else ids + pad
        tokens = [Token(word, known) for _, word, known in items if word is not None]
        return Encoded(ids=np.asarray(ids, dtype=np.int32), tokens=tokens,
            word_count=len(words), unknown_count=sum(not token.known for token in tokens),
            truncated=truncated)
