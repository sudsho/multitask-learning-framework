"""FastAPI stub: /predict_nlp and /predict_vision.

This is an endpoint-shape stub, not a serving layer for a trained MTL model.
The training scripts in `examples/` only save a metrics history file, not a
model checkpoint, so nothing is loaded at request time. The NLP handler
returns a keyword heuristic and the vision handler returns a constant shape.
Use it to exercise the request/response schemas and integration wiring.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mtl.api")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="multitask-learning-framework", version="0.1.0")


# ---------- request / response schemas ----------


class NLPRequest(BaseModel):
    text: str


class NLPResponse(BaseModel):
    sentiment: str
    sentiment_score: float
    topic: str
    ner: List[Dict[str, Any]]


class VisionRequest(BaseModel):
    # 3xHxW float list; image_b64 is not supported by this stub.
    image: Optional[List[List[List[float]]]] = None


class VisionResponse(BaseModel):
    classification: str
    classification_score: float
    seg_shape: List[int]   # H, W (full mask not returned in the JSON)


# ---------- endpoints ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_nlp", response_model=NLPResponse)
def predict_nlp(req: NLPRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    # Stub heuristics only; no model is loaded.
    sent = "positive" if any(w in req.text.lower()
                              for w in ("good", "great", "love", "enjoy")) else "negative"
    score = 0.9 if sent == "positive" else 0.1
    topic = "world"
    ner = []
    for tok in req.text.split():
        if tok.istitle() and len(tok) > 1:
            ner.append({"token": tok, "tag": "B-PER"})
    return NLPResponse(
        sentiment=sent,
        sentiment_score=score,
        topic=topic,
        ner=ner,
    )


@app.post("/predict_vision", response_model=VisionResponse)
def predict_vision(req: VisionRequest):
    if req.image is None:
        raise HTTPException(status_code=422, detail="must provide image")
    x = torch.tensor(req.image, dtype=torch.float32)
    if x.ndim != 3 or x.shape[0] != 3:
        raise HTTPException(status_code=422, detail="image must be 3xHxW")
    H, W = int(x.shape[1]), int(x.shape[2])
    # Stub: no model is loaded; returns a constant class and echoes the shape.
    return VisionResponse(
        classification="circle",
        classification_score=0.5,
        seg_shape=[H, W],
    )
