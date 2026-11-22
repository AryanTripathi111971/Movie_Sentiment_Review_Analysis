"""FastAPI backend and static frontend for movie-review sentiment analysis."""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import config, model_loader
from .schemas import BatchItem, BatchRequest, BatchResponse, HealthResponse, ReviewRequest, SentimentResponse, TokenOut
from .text_processing import EmptyTextError, ReviewEncoder, load_word_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
STATIC_DIR = config.BASE_DIR / "app" / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    info = model_loader.load_model()
    app.state.encoder = ReviewEncoder(load_word_index(), info["num_words"], info["max_len"])
    yield

app = FastAPI(title="Movie Review Sentiment Analyzer",
              description="A recurrent neural network predicts sentiment from movie reviews.",
              version="1.0.0", lifespan=lifespan)

def _label(score: float) -> tuple[str, float]:
    if abs(score - 0.5) <= config.UNCLEAR_BAND:
        return "unclear", max(score, 1 - score)
    return ("positive", score) if score > 0.5 else ("negative", 1 - score)

def _encode(text: str):
    try:
        return app.state.encoder.encode(text)
    except EmptyTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

def _analyze(text: str) -> SentimentResponse:
    start = time.perf_counter()
    enc = _encode(text)
    score = float(model_loader.predict_scores(enc.ids[None, :])[0])
    label, confidence = _label(score)
    return SentimentResponse(label=label, score=score, confidence=confidence,
        tokens=[TokenOut(word=t.word, known=t.known) for t in enc.tokens],
        word_count=enc.word_count, unknown_count=enc.unknown_count,
        truncated=enc.truncated, max_len=app.state.encoder.max_len,
        inference_ms=round((time.perf_counter() - start) * 1000, 1))

def _analyze_batch(reviews: list[str]) -> BatchResponse:
    import numpy as np
    start = time.perf_counter()
    encoded = [_encode(review) for review in reviews]
    scores = model_loader.predict_scores(np.stack([item.ids for item in encoded]))
    results = []
    for review, score in zip(reviews, scores):
        label, confidence = _label(float(score))
        preview = review.strip().replace("\n", " ")
        results.append(BatchItem(preview=preview[:80] + ("…" if len(preview) > 80 else ""),
                                 label=label, score=float(score), confidence=confidence))
    return BatchResponse(results=results, inference_ms=round((time.perf_counter() - start) * 1000, 1))

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    encoder = getattr(app.state, "encoder", None)
    return HealthResponse(status="ok" if model_loader.is_loaded() and encoder else "not_ready",
        model_loaded=model_loader.is_loaded(), max_len=encoder.max_len if encoder else 0,
        vocabulary_size=encoder.num_words if encoder else 0)

@app.post("/api/predict", response_model=SentimentResponse, tags=["predict"])
async def predict(payload: ReviewRequest):
    return await run_in_threadpool(_analyze, payload.text)

@app.post("/api/predict/batch", response_model=BatchResponse, tags=["predict"])
async def predict_batch(payload: BatchRequest):
    if any(len(review) > config.MAX_CHARS for review in payload.reviews):
        raise HTTPException(status_code=422, detail=f"Each review must be at most {config.MAX_CHARS} characters.")
    return await run_in_threadpool(_analyze_batch, payload.reviews)

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
