from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class NLIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = _read_body(self)
        if body is None:
            return

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        premise = payload.get("premise")
        hypothesis = payload.get("hypothesis")
        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            self.send_error(400, "premise and hypothesis are required")
            return

        try:
            result = _run_nli(premise=premise, hypothesis=hypothesis)
        except RuntimeError as exc:
            self.send_error(502, str(exc))
            return

        response = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


def _read_body(handler: BaseHTTPRequestHandler) -> bytes | None:
    length_header = handler.headers.get("Content-Length")
    if length_header is None:
        handler.send_error(411, "Content-Length required")
        return None
    try:
        length = int(length_header)
    except ValueError:
        handler.send_error(400, "Invalid Content-Length")
        return None
    return handler.rfile.read(length)


def _run_nli(*, premise: str, hypothesis: str) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    model = os.environ.get("NLI_MODEL") or os.environ.get("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("NLI_MODEL or OPENROUTER_MODEL is required")

    base_url = os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    system_prompt = (
        "You are an NLI classifier. Given a premise and a hypothesis, return a JSON "
        "object with keys: label (ENTAILMENT, CONTRADICTION, or NEUTRAL) and "
        "confidence (0 to 1). Output JSON only."
    )
    user_prompt = f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.post(base_url, json=body, headers=headers, timeout=60.0)
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"OpenRouter request failed: {response.status_code} {response.text}"
        )

    content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("NLI response missing content")

    parsed = _parse_json(content)
    label = parsed.get("label")
    confidence = parsed.get("confidence")
    if not isinstance(label, str) or not isinstance(confidence, (int, float)):
        raise RuntimeError("NLI response missing label or confidence")

    return {"label": label.upper(), "confidence": float(confidence)}


def _parse_json(text: str) -> dict[str, object]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise RuntimeError("NLI response was not valid JSON")


def main() -> None:
    port = int(os.environ.get("NLI_PORT", "9000"))
    server = HTTPServer(("127.0.0.1", port), NLIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
