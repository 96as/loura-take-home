from __future__ import annotations

from typing import Protocol


class LLMTransientError(Exception):
    """Worth retrying: timeout-like failure, rate limit, 5xx."""


class LLMPermanentError(Exception):
    """The request was rejected; retrying will not help."""


class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str: ...
