"""Direct Lettuce backend that loads model in-process without requiring a separate server."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LettuceSpan:
    """Raw span detection from LettuceDetect model."""

    text: str
    start: int
    end: int
    hallucination_score: float


class LettuceDirectBackend:
    """Direct LettuceDetect backend (no HTTP server required).

    This backend loads the LettuceDetect model in-process and runs inference
    directly, eliminating the need for a separate server process.
    """

    def __init__(self, model_path: str):
        """Initialize Lettuce direct backend.

        Args:
            model_path: Path or identifier for the LettuceDetect model
                       (e.g., "KRLabsOrg/lettucedect-base-modernbert-en-v1")
        """
        self.model_path = model_path
        self._model = None

    def _get_model(self):
        """Lazy-load the LettuceDetect model on first use."""
        if self._model is None:
            try:
                from lettucedetect import HallucinationDetector
            except ImportError as exc:
                raise RuntimeError(
                    "lettucedetect is required. Install with: uv pip install lettucedetect[api]"
                ) from exc

            self._model = HallucinationDetector(
                method="transformer", model_path=self.model_path
            )
        return self._model

    async def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> list[LettuceSpan]:
        """Detect hallucinated spans in an answer.

        Args:
            contexts: List of context strings (reference documents)
            question: The question that was asked
            answer: The answer/summary to validate

        Returns:
            List of detected hallucination spans with scores

        Raises:
            RuntimeError: If model loading or prediction fails
        """
        model = self._get_model()

        # Combine contexts into a single string
        context_text = "\n\n".join(contexts)

        # Run LettuceDetect prediction
        try:
            result = model.predict(
                context=context_text,
                question=question,
                answer=answer,
            )
        except Exception as exc:
            raise RuntimeError(f"LettuceDetect prediction failed: {str(exc)}") from exc

        # Extract spans from result
        spans = []
        if hasattr(result, "spans") and result.spans:
            for span in result.spans:
                spans.append(
                    LettuceSpan(
                        text=span.text,
                        start=span.start,
                        end=span.end,
                        hallucination_score=span.score,
                    )
                )

        return spans
