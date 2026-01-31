from __future__ import annotations

import os

import httpx

from .types import NLIPrediction


class NLIHttpBackend:
    def __init__(
        self,
        url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._url = url or os.environ.get("NLI_URL")
        if not self._url:
            raise ValueError("NLI URL is required")
        self._timeout_s = timeout_s

    async def infer(self, premise: str, hypothesis: str) -> NLIPrediction:
        payload = {"premise": premise, "hypothesis": hypothesis}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(self._url, json=payload)

        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"NLI request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        label = data.get("label")
        confidence = data.get("confidence")
        if not isinstance(label, str) or not isinstance(confidence, (int, float)):
            raise RuntimeError("NLI response missing label or confidence")

        return NLIPrediction(label=label.upper(), confidence=float(confidence))
