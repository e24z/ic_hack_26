from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class NLIPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class SpanDetection:
    text: str
    start: int
    end: int
    hallucination_score: float
    label: str
    confidence: float
    severity: int


@dataclass(frozen=True)
class ValidationResult:
    grounded_pct: float
    needs_fact_check: bool
    blocked: bool
    spans: list[SpanDetection]
    raw: dict[str, object] = field(default_factory=dict)


class Validator(Protocol):
    async def validate(
        self,
        summary: str,
        context: object,
        question: str | None = None,
    ) -> ValidationResult: ...


class NLIBackend(Protocol):
    async def infer(self, premise: str, hypothesis: str) -> NLIPrediction: ...
