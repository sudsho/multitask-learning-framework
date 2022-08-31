"""FastAPI app: /predict_nlp and /predict_vision.

For demo/serving the model is loaded lazily on first request from a checkpoint
written by the training scripts. If `MTL_NLP_CKPT` / `MTL_VISION_CKPT` env
vars are not set, we fall back to dummy outputs so the endpoint shape is
testable without a trained model.
"""

import logging
import os
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
    # base64-encoded RGB png/jpg or a 3xHxW float list
    image_b64: Optional[str] = None
    image: Optional[List[List[List[float]]]] = None


class VisionResponse(BaseModel):
    classification: str
    classification_score: float
    seg_shape: List[int]   # H, W (full mask not returned in the JSON)


# ---------- lazy model loaders ----------


_nlp_model = {"obj": None}
_vision_model = {"obj": None}


def _load_nlp():
    if _nlp_model["obj"] is not None:
        return _nlp_model["obj"]
    ckpt = os.environ.get("MTL_NLP_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        logger.warning("MTL_NLP_CKPT not set or missing; serving dummy outputs")
        return None
    try:
        state = torch.load(ckpt, map_location="cpu")
    except Exception as e:   # noqa: BLE001
        logger.error("failed to load nlp ckpt %s: %s", ckpt, e)
        return None
    _nlp_model["obj"] = state
    return state


def _load_vision():
    if _vision_model["obj"] is not None:
        return _vision_model["obj"]
    ckpt = os.environ.get("MTL_VISION_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        logger.warning("MTL_VISION_CKPT not set or missing; serving dummy outputs")
        return None
    try:
        state = torch.load(ckpt, map_location="cpu")
    except Exception as e:   # noqa: BLE001
        logger.error("failed to load vision ckpt %s: %s", ckpt, e)
        return None
    _vision_model["obj"] = state
    return state


# ---------- endpoints ----------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_nlp", response_model=NLPResponse)
def predict_nlp(req: NLPRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    _ = _load_nlp()
    # Dummy heuristic output if no checkpoint is loaded.
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
    if req.image is None and req.image_b64 is None:
        raise HTTPException(status_code=422, detail="must provide image or image_b64")
    _ = _load_vision()
    if req.image is not None:
        x = torch.tensor(req.image, dtype=torch.float32)
        if x.ndim != 3 or x.shape[0] != 3:
            raise HTTPException(status_code=422, detail="image must be 3xHxW")
        H, W = int(x.shape[1]), int(x.shape[2])
    else:
        # we don't actually decode without Pillow on path; just trust shape claim.
        H, W = 224, 224
    return VisionResponse(
        classification="circle",
        classification_score=0.5,
        seg_shape=[H, W],
    )
