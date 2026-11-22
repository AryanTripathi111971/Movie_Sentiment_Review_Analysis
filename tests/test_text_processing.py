import numpy as np
import pytest
from app.text_processing import EmptyTextError, ReviewEncoder, tokenize

WORDS = {"the": 1, "movie": 17, "great": 84, "terrible": 200, "rareword": 15000}

def make(max_len=10, **kw):
    return ReviewEncoder(WORDS, num_words=10_000, max_len=max_len, **kw)

def test_tokenize_lowercases_and_strips_html_and_punctuation():
    assert tokenize("The MOVIE<br />was great!!") == ["the", "movie", "was", "great"]
    assert tokenize("Don't stop") == ["don't", "stop"]

def test_word_ids_and_padding():
    enc = make().encode("The movie was great!")
    assert enc.ids.tolist() == [0, 0, 0, 0, 0, 1, 4, 20, 2, 87]
    assert enc.ids.shape == (10,) and enc.ids.dtype == np.int32

def test_unknown_words():
    enc = make().encode("the rareword blah")
    assert [token.known for token in enc.tokens] == [True, False, False]
    assert enc.unknown_count == 2

def test_post_padding():
    enc = make(padding="post").encode("great movie")
    assert enc.ids.tolist() == [1, 87, 20, 0, 0, 0, 0, 0, 0, 0]

def test_long_review_is_truncated():
    enc = make(max_len=10).encode("great " * 30)
    assert enc.truncated and len(enc.tokens) == 10 and enc.word_count == 30

def test_empty_text_is_rejected():
    for text in ["", "   ", "!!! ???"]:
        with pytest.raises(EmptyTextError):
            make().encode(text)

def test_bad_options_are_rejected():
    with pytest.raises(ValueError):
        ReviewEncoder(WORDS, padding="middle")
