from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    async def generate_summary(self, data: dict | list) -> str: ...


class Base:
    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    async def summarise(self, data: dict | list) -> str:
        return await self._backend.generate_summary(data)

    def retrieve(self):
        raise NotImplementedError

    def split(self):
        raise NotImplementedError


class Overseer:
    def validate(self, summary: str) -> float:
        raise NotImplementedError
        # at this point, halugate will validate a summary yielded from the base class.
