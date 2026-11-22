"""Load the saved Keras model once and keep it in memory."""
from __future__ import annotations
import logging
import os
import threading
import numpy as np
from . import config

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
log = logging.getLogger("movie-sentiment")
_model = None
_lock = threading.Lock()

def load_model() -> dict:
    global _model
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {config.MODEL_PATH}\n"
            "Put your .keras file in the models folder or set MODEL_PATH."
        )
    import keras
    _model = keras.saving.load_model(config.MODEL_PATH, compile=False)
    shape = _model.input_shape
    if len(shape) != 2:
        raise RuntimeError(f"Expected a model that takes a sequence of word numbers, got input shape {shape}.")
    max_len = int(shape[1]) if shape[1] else config.MAX_LEN
    embedding = next((layer for layer in _model.layers if isinstance(layer, keras.layers.Embedding)), None)
    num_words = int(embedding.input_dim) if embedding is not None else config.NUM_WORDS
    _model.predict(np.zeros((1, max_len), dtype="int32"), verbose=0)
    log.info("Model loaded from %s (max_len=%d, vocabulary=%d)", config.MODEL_PATH.name, max_len, num_words)
    return {"max_len": max_len, "num_words": num_words}

def is_loaded() -> bool:
    return _model is not None

def predict_scores(batch: np.ndarray) -> np.ndarray:
    if _model is None:
        raise RuntimeError("Model is not loaded.")
    with _lock:
        return np.asarray(_model.predict(batch, verbose=0)).reshape(-1)
