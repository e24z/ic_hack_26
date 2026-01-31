"""Direct NLI backend that calls OpenRouter API without requiring a separate server."""

from __future__ import annotations

import json

import httpx

from .types import NLIPrediction


class NLIDirectBackend:
    """Direct OpenRouter-based NLI backend (no HTTP server required).

    This backend makes direct API calls to OpenRouter for NLI inference,
    eliminating the need for a separate NLI server process.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ):
        """Initialize NLI direct backend.

        Args:
            model: Model identifier (e.g., "upstage/solar-pro-3:free")
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=60.0)

    async def infer(self, premise: str, hypothesis: str) -> NLIPrediction:
        """Run NLI inference on premise and hypothesis.

        Args:
            premise: The premise text (context/reference)
            hypothesis: The hypothesis text (summary/claim to verify)

        Returns:
            NLIPrediction with label and confidence score

        Raises:
            RuntimeError: If API call fails or response is invalid
        """
        system_prompt = (
            "You are an NLI classifier. Given a premise and a hypothesis, return a JSON "
            "object with keys: label (ENTAILMENT, CONTRADICTION, or NEUTRAL) and "
            "confidence (0 to 1). Output JSON only."
        )
        user_prompt = f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(self.base_url, json=body, headers=headers)

        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"OpenRouter request failed: {response.status_code} {response.text}"
            )

        content = (
            response.json().get("choices", [{}])[0].get("message", {}).get("content")
        )
        if not isinstance(content, str):
            raise RuntimeError("NLI response missing content")

        parsed = self._parse_json(content)
        label = parsed.get("label")
        confidence = parsed.get("confidence")

        if not isinstance(label, str) or not isinstance(confidence, (int, float)):
            raise RuntimeError("NLI response missing label or confidence")

        return NLIPrediction(label=label.upper(), confidence=float(confidence))

    def _parse_json(self, text: str) -> dict[str, object]:
        """Parse JSON from text, handling cases where it's wrapped in markdown.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON object

        Raises:
            RuntimeError: If JSON cannot be parsed
        """
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks or surrounding text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise RuntimeError("NLI response was not valid JSON")

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
