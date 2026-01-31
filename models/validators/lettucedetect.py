from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import httpx

from .types import NLIBackend, SpanDetection, ValidationResult

if TYPE_CHECKING:
    from .lettuce_direct import LettuceDirectBackend
    from .lettuce_mock import LettuceMockBackend


class LettuceDetectValidator:
    def __init__(
        self,
        nli_backend: NLIBackend,
        lettuce_backend: LettuceDirectBackend | LettuceMockBackend | None = None,
        base_url: str | None = None,
        grounded_min: float | None = None,
        span_conf: float | None = None,
        contradiction_conf: float | None = None,
        chunk_size: int | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._nli_backend = nli_backend
        self._lettuce_backend = lettuce_backend  # Direct/mock backend (optional)
        self._base_url = base_url or os.environ.get(
            "LETTUCE_URL", "http://127.0.0.1:8000"
        )
        self._grounded_min = _env_float("LETTUCE_GROUNDED_MIN", grounded_min, 0.85)
        self._span_conf = _env_float("LETTUCE_SPAN_CONF", span_conf, 0.90)
        self._contradiction_conf = _env_float(
            "NLI_CONTRADICTION_CONF",
            contradiction_conf,
            0.90,
        )
        self._chunk_size = _env_int("NLI_CHUNK_SIZE", chunk_size, 2000)
        self._timeout_s = timeout_s

    async def validate(
        self,
        summary: str,
        context: object,
        question: str | None = None,
    ) -> ValidationResult:
        contexts = _normalize_contexts(context)

        # Use direct backend if provided, otherwise fall back to HTTP
        if self._lettuce_backend is not None:
            raw_spans = await self._lettuce_backend.detect_spans(
                contexts=contexts,
                question=question or "",
                answer=summary,
            )
            # Convert LettuceSpan objects to dict format for _coerce_span
            predictions = [
                {
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "hallucination_score": span.hallucination_score,
                }
                for span in raw_spans
            ]
            response_json = {"predictions": predictions}
        else:
            # HTTP mode - make request to Lettuce server
            payload = {
                "contexts": contexts,
                "question": question or "",
                "answer": summary,
            }
            url = f"{self._base_url.rstrip('/')}/v1/lettucedetect/spans"
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, json=payload)

            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    f"LettuceDetect request failed: {response.status_code} {response.text}"
                )

            response_json = response.json()
            predictions = response_json.get("predictions", [])
            if not isinstance(predictions, list):
                raise RuntimeError("LettuceDetect response missing predictions")

        spans: list[SpanDetection] = []
        context_chunks = _chunk_contexts(contexts, self._chunk_size)
        for item in predictions:
            span = _coerce_span(item)
            label, confidence = await self._infer_nli(span.text, context_chunks)
            severity = _severity_from_label(label)
            spans.append(
                SpanDetection(
                    text=span.text,
                    start=span.start,
                    end=span.end,
                    hallucination_score=span.hallucination_score,
                    label=label,
                    confidence=confidence,
                    severity=severity,
                )
            )

        grounded_pct = _grounded_percent(summary, spans)
        needs_fact_check = len(spans) > 0
        blocked = _should_block(
            grounded_pct=grounded_pct,
            spans=spans,
            grounded_min=self._grounded_min,
            span_conf=self._span_conf,
            contradiction_conf=self._contradiction_conf,
        )

        return ValidationResult(
            grounded_pct=grounded_pct,
            needs_fact_check=needs_fact_check,
            blocked=blocked,
            spans=spans,
            raw=response_json,
        )

    async def _infer_nli(
        self,
        text: str,
        context_chunks: list[str],
    ) -> tuple[str, float]:
        best_label = "NEUTRAL"
        best_confidence = 0.0
        for chunk in context_chunks:
            prediction = await self._nli_backend.infer(
                premise=chunk,
                hypothesis=text,
            )
            if prediction.confidence > best_confidence:
                best_label = prediction.label
                best_confidence = prediction.confidence
        return best_label, best_confidence


class _RawSpan:
    def __init__(
        self, text: str, start: int, end: int, hallucination_score: float
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.hallucination_score = hallucination_score


def _coerce_span(item: object) -> _RawSpan:
    if not isinstance(item, dict):
        raise RuntimeError("LettuceDetect span prediction is invalid")
    text = item.get("text", "")
    start = item.get("start", 0)
    end = item.get("end", 0)
    score = item.get("hallucination_score", 0.0)
    if not isinstance(text, str):
        text = str(text)
    if not isinstance(start, int):
        start = int(start) if str(start).isdigit() else 0
    if not isinstance(end, int):
        end = int(end) if str(end).isdigit() else 0
    if not isinstance(score, (int, float)):
        score = float(score) if _is_number(score) else 0.0
    return _RawSpan(text=text, start=start, end=end, hallucination_score=float(score))


def _normalize_contexts(context: object) -> list[str]:
    if context is None:
        return [""]
    if isinstance(context, (list, tuple)):
        return [_serialize_item(item) for item in context]
    return [_serialize_item(context)]


def _serialize_item(item: object) -> str:
    if isinstance(item, str):
        return item
    try:
        return json.dumps(item, ensure_ascii=True)
    except TypeError:
        return str(item)


def _chunk_contexts(contexts: list[str], chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return contexts
    chunks: list[str] = []
    for context in contexts:
        if len(context) <= chunk_size:
            chunks.append(context)
            continue
        for start in range(0, len(context), chunk_size):
            chunks.append(context[start : start + chunk_size])
    if not chunks:
        chunks.append("")
    return chunks


def _grounded_percent(summary: str, spans: list[SpanDetection]) -> float:
    if not summary:
        return 1.0
    intervals = []
    for span in spans:
        if span.label == "ENTAILMENT":
            continue
        if span.end > span.start:
            intervals.append((span.start, span.end))
        else:
            length = len(span.text)
            if length > 0:
                intervals.append((0, length))

    if not intervals:
        return 1.0

    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    hallucinated_chars = sum(end - start for start, end in merged)
    grounded_pct = 1 - (hallucinated_chars / max(len(summary), 1))
    return max(0.0, min(1.0, grounded_pct))


def _severity_from_label(label: str) -> int:
    if label == "CONTRADICTION":
        return 4
    if label == "NEUTRAL":
        return 2
    return 0


def _should_block(
    grounded_pct: float,
    spans: list[SpanDetection],
    grounded_min: float,
    span_conf: float,
    contradiction_conf: float,
) -> bool:
    if grounded_pct < grounded_min:
        return True
    for span in spans:
        if span.label == "CONTRADICTION" and span.confidence >= contradiction_conf:
            return True
        if span.hallucination_score >= span_conf and span.label != "ENTAILMENT":
            return True
    return False


def _env_float(name: str, override: float | None, default: float) -> float:
    if override is not None:
        return override
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, override: int | None, default: int) -> int:
    if override is not None:
        return override
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
