"""Mock Lettuce backend for fast testing without model loading."""

from __future__ import annotations

from .lettuce_direct import LettuceSpan


class LettuceMockBackend:
    """Mock Lettuce backend that returns no hallucination spans.

    Useful for fast testing without loading the LettuceDetect model.
    """

    async def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> list[LettuceSpan]:
        """Return an empty list of spans (no hallucinations detected).

        Args:
            contexts: List of context strings (ignored)
            question: The question (ignored)
            answer: The answer (ignored)

        Returns:
            Empty list of spans
        """
        return []
