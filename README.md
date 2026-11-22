# Movie Review Sentiment Analysis

An end-to-end movie-review sentiment analysis application. Users can enter a review in a browser and receive a **positive**, **negative**, or **unclear** prediction from a saved Keras neural-network model.

## Features
- FastAPI backend with interactive API documentation.
- Browser-based review analyzer with sample reviews and a sentiment score gauge.
- Single-review and batch prediction endpoints.
- IMDB/Keras word-index preprocessing with padding, truncation, and out-of-vocabulary handling.
- Health endpoint and automated tests.

## Project structure
```
.
├── app/
│   ├── static/              # HTML, CSS and JavaScript frontend
│   ├── config.py
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── text_processing.py
├── models/
│   └── README.md            # instructions for adding the model artifact
├── scripts/
│   └── download_vocab.py
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── run.py
```

## Important: model file
The trained model `Movie_Sentiment_Analysis.keras` was present in the supplied ZIP, but binary model artifacts cannot be uploaded through the available GitHub file-writing connection. The model is therefore **not included in this repository yet**. Place your existing model at:

```
models/Movie_Sentiment_Analysis.keras
```

Alternatively, set the `MODEL_PATH` environment variable to its location. The app will download and cache the IMDB word index on first startup if it is not already present.

## Setup
Python 3.10+ is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Add the trained model file as described above, then start the server:

```bash
python run.py
```

Open http://127.0.0.1:8000 in your browser. API documentation is available at http://127.0.0.1:8000/docs.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Model and service status |
| POST | `/api/predict` | Analyze one review |
| POST | `/api/predict/batch` | Analyze a list of reviews |

Example request:

```json
{ "text": "A wonderful film with brilliant acting and a moving story." }
```

The response includes the predicted label, probability of positive sentiment, confidence, token information, unknown-word count, truncation status, and inference time.

## Run tests
Install development dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest
```

The API tests require the trained model and vocabulary. The text-processing tests can run independently of TensorFlow.

## How preprocessing works
Reviews are lowercased and tokenized, mapped to the Keras IMDB word IDs, shifted by the configured index offset, and converted to fixed-length sequences. Unknown or out-of-range words use the OOV token. Sequences are pre-padded and pre-truncated by default to match the configured training convention.

## Technology stack
Python · TensorFlow/Keras · FastAPI · NumPy · HTML · CSS · JavaScript · Pytest

## Limitations
Sentiment prediction is an estimate from the supplied trained model, not a definitive assessment of a movie. Results can be less reliable for sarcasm, mixed opinions, domain-specific terms, and words outside the training vocabulary.
