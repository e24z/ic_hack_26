from __future__ import annotations

from typing import Protocol

from .validators.types import ValidationResult, Validator


class ModelBackend(Protocol):
    async def generate_summary(
        self, data: dict | list, guidance: str | None = None
    ) -> str: ...


class Base:
    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    async def summarise(self, data: dict | list, guidance: str | None = None) -> str:
        return await self._backend.generate_summary(data, guidance=guidance)

    def retrieve(self):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError


class Overseer:
    def __init__(self, validator: Validator, max_retries: int = 2) -> None:
        self._validator = validator
        self._max_retries = max_retries

    async def validate(
        self,
        summary: str,
        context: object,
        question: str | None = None,
    ) -> ValidationResult:
        return await self._validator.validate(
            summary=summary, context=context, question=question
        )

    async def summarise_with_validation(
        self,
        base: Base,
        data: dict | list,
        context: object,
        question: str | None = None,
        guidance: str | None = None,
    ) -> tuple[str, ValidationResult]:
        strict_guidance = (
            "Only include claims supported by the provided data. Prefer omission over "
            "speculation. If something is not stated, say it is not stated."
        )
        current_guidance = guidance
        last_result: ValidationResult | None = None
        summary = ""

        for attempt in range(self._max_retries + 1):
            summary = await base.summarise(data, guidance=current_guidance)
            last_result = await self.validate(
                summary=summary, context=context, question=question
            )
            if not last_result.blocked:
                return summary, last_result
            if attempt == self._max_retries:
                break
            current_guidance = strict_guidance

        if last_result is None:
            raise RuntimeError("Validation failed without producing a result")
        return summary, last_result

        # at this point, halugate will validate a summary yielded from the base class.
