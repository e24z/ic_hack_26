from __future__ import annotations

import importlib
import json
import os


class OpenRouterBackend:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_s: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OpenRouter API key is required")
        self._model = model or os.environ.get("OPENROUTER_MODEL")
        if not self._model:
            raise ValueError("OpenRouter model is required")
        self._base_url = base_url
        self._timeout_s = timeout_s

    async def generate_summary(
        self, data: dict | list | str, guidance: str | None = None
    ) -> str:
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise RuntimeError("httpx is required for OpenRouterBackend") from exc

        payload_text = self._serialize_payload(data)
        system_prompt = (
            "Summarize whatever content you receive. The input may be messy, "
            "partial, or inconsistently structured. Return a plain text summary."
        )
        if guidance:
            system_prompt = f"{system_prompt} {guidance}"

        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": payload_text},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(self._base_url, json=body, headers=headers)

        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                f"OpenRouter request failed: {response.status_code} {response.text}"
            )

        response_json = response.json()
        content = (
            response_json.get("choices", [{}])[0].get("message", {}).get("content")
        )
        if not content or not isinstance(content, str):
            raise RuntimeError("OpenRouter response missing summary content")
        return content.strip()

    def _serialize_payload(self, data: dict | list | str) -> str:
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, ensure_ascii=True)
        except TypeError:
            return str(data)
