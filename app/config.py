"""Central application settings; values can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "Movie_Sentiment_Analysis.keras"))
WORD_INDEX_PATH = Path(os.getenv("WORD_INDEX_PATH", BASE_DIR / "models" / "imdb_word_index.json"))

NUM_WORDS = 10_000
MAX_LEN = 200
START_ID = 1
OOV_ID = 2
INDEX_FROM = 3
PADDING = os.getenv("PADDING", "pre")
TRUNCATING = os.getenv("TRUNCATING", "pre")
UNCLEAR_BAND = 0.10
MAX_CHARS = 20_000
MAX_BATCH = 50
