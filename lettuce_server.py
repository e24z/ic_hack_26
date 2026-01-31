from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Global model instance (loaded on first request)
_model = None


def _get_model():
    """Lazy load the LettuceDetect model."""
    global _model
    if _model is None:
        try:
            from lettucedetect import HallucinationDetector
        except ImportError as exc:
            raise RuntimeError(
                "lettucedetect is required. Install with: uv pip install lettucedetect[api]"
            ) from exc

        model_path = os.environ.get(
            "LETTUCE_MODEL", "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        )
        _model = HallucinationDetector(method="transformer", model_path=model_path)
    return _model


class LettuceRequest(BaseModel):
    contexts: list[str]
    question: str
    answer: str


class SpanPrediction(BaseModel):
    text: str
    start: int
    end: int
    hallucination_score: float


class LettuceResponse(BaseModel):
    predictions: list[SpanPrediction]


@app.post("/v1/lettucedetect/spans")
async def detect_spans(request: LettuceRequest) -> LettuceResponse:
    """
    Detect hallucinated spans in an answer given contexts and optional question.

    Args:
        request: Contains contexts (list of strings), question (string), and answer (string)

    Returns:
        Response with predictions (list of span detections with text, start, end, hallucination_score)
    """
    try:
        model = _get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Combine contexts into a single string
    context_text = "\n\n".join(request.contexts)

    # Run LettuceDetect prediction
    try:
        result = model.predict(
            context=context_text,
            question=request.question,
            answer=request.answer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"LettuceDetect prediction failed: {str(exc)}"
        ) from exc

    # Extract spans from result
    predictions = []
    if hasattr(result, "spans") and result.spans:
        for span in result.spans:
            predictions.append(
                SpanPrediction(
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    hallucination_score=span.score,
                )
            )

    return LettuceResponse(predictions=predictions)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def main() -> None:
    """Run the FastAPI server."""
    import uvicorn

    port = int(os.environ.get("LETTUCE_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
