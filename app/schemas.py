from typing import Literal
from pydantic import BaseModel, Field
from .config import MAX_BATCH, MAX_CHARS

Label = Literal["positive", "negative", "unclear"]

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS, description="The movie review")

class BatchRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)

class TokenOut(BaseModel):
    word: str
    known: bool

class SentimentResponse(BaseModel):
    label: Label
    score: float = Field(..., description="Probability that the review is positive (0 to 1)")
    confidence: float
    tokens: list[TokenOut]
    word_count: int
    unknown_count: int
    truncated: bool
    max_len: int
    inference_ms: float

class BatchItem(BaseModel):
    preview: str
    label: Label
    score: float
    confidence: float

class BatchResponse(BaseModel):
    results: list[BatchItem]
    inference_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    max_len: int
    vocabulary_size: int
