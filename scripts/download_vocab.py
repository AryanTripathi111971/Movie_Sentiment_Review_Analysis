"""Download and cache the Keras IMDB word index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.text_processing import load_word_index

if __name__ == "__main__":
    words = load_word_index()
    print(f"Vocabulary ready: {len(words):,} words")
