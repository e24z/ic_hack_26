"""Mock summarizer backend for fast testing without API calls."""

from __future__ import annotations


class MockBackend:
    """Mock summarizer backend that returns fixed responses.

    Useful for fast testing without making actual API calls.
    """

    async def generate_summary(
        self, data: dict | list, guidance: str | None = None
    ) -> str:
        """Return a fixed mock summary.

        Args:
            data: Data to summarize (ignored)
            guidance: Optional guidance (ignored)

        Returns:
            Fixed mock summary text
        """
        return "This is a mock summary for testing purposes."
