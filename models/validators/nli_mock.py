"""Mock NLI backend for fast testing without API calls."""

from __future__ import annotations

from .types import NLIPrediction


class NLIMockBackend:
    """Mock NLI backend that returns fixed NEUTRAL responses.

    Useful for fast testing without making actual API calls.
    """

    async def infer(self, premise: str, hypothesis: str) -> NLIPrediction:
        """Return a fixed NEUTRAL prediction.

        Args:
            premise: The premise text (ignored)
            hypothesis: The hypothesis text (ignored)

        Returns:
            Fixed NEUTRAL prediction with 0.5 confidence
        """
        return NLIPrediction(label="NEUTRAL", confidence=0.5)
