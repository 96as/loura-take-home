"""Test equipment: an LLM stand-in that misbehaves on purpose.

A real model rarely breaks, so a fake is the only way to show every failure path of the
service working. The accuracy of its guesses is irrelevant.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from app.injection import looks_like_injection
from app.llm.base import LLMPermanentError, LLMTransientError
from app.schemas import MAX_SUMMARY_CHARS

_TICKET_OPEN = "<ticket>\nsubject: "
_BODY_MARK = "\nbody:\n"
_TICKET_CLOSE = "\n</ticket>"


def extract_ticket(prompt: str) -> tuple[str, str]:
    """Return (subject, body) from the block that classifier.build_prompt appends."""
    # The instruction text mentions <ticket> literally, so anchor on the last real block.
    anchor = prompt.rfind(_TICKET_OPEN)
    if anchor == -1:
        raise ValueError("no <ticket> block in prompt")
    subject_start = anchor + len(_TICKET_OPEN)
    body_mark = prompt.find(_BODY_MARK, subject_start)
    if body_mark == -1:
        raise ValueError("ticket block has no body")
    body_start = body_mark + len(_BODY_MARK)
    body_end = prompt.rfind(_TICKET_CLOSE)
    if body_end < body_start:
        raise ValueError("ticket block is not closed")
    return prompt[subject_start:body_mark], prompt[body_start:body_end]


# --- the normal (keyword-guess) reply -------------------------------------------------

_CATEGORY_KEYWORDS = (
    ("billing", ("charge", "refund", "invoice", "overcharged", "payment")),
    ("technical", ("error", "500", "api", "timeout", "export", "upload")),
    ("account", ("password", "log in", "email address", "account")),
)
_PRIORITY_KEYWORDS = (
    ("high", ("urgent", "production", "blocking", "asap")),
    ("low", ("nice to have", "not urgent", "feature request")),
)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _first_of(text: str, table, default: str) -> str:
    for label, words in table:
        if any(w in text for w in words):
            return label
    return default


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    match = _SENTENCE_END.search(text)
    sentence = text[: match.end()] if match else text
    if sentence[-1] not in ".!?":
        sentence += "."
    if len(sentence) > MAX_SUMMARY_CHARS:
        sentence = sentence[: MAX_SUMMARY_CHARS - 1].rstrip() + "."
    return sentence


def _guess(subject: str, body: str) -> dict[str, str]:
    text = f"{subject}\n{body}".lower()
    summary = _first_sentence(body) or _first_sentence(subject) or "No details provided."
    return {
        "category": _first_of(text, _CATEGORY_KEYWORDS, "other"),
        "priority": _first_of(text, _PRIORITY_KEYWORDS, "medium"),
        "summary": summary,
    }


# --- deliberate breakage --------------------------------------------------------------

def _fenced(good: dict) -> str:
    return f"```json\n{json.dumps(good)}\n```"


def _chatter(good: dict) -> str:
    return f"Sure! Here is the classification you asked for:\n{json.dumps(good)}\nHope that helps."


def _prose(good: dict) -> str:
    return f"This looks like a {good['category']} ticket and I would treat it as {good['priority']} priority."


def _cut_off(good: dict) -> str:
    text = json.dumps(good)
    return text[: len(text) // 2]


def _missing_field(good: dict) -> str:
    return json.dumps({k: v for k, v in good.items() if k != "priority"})


def _bad_category(good: dict) -> str:
    return json.dumps({**good, "category": "Billing Issue"})


def _bad_priority(good: dict) -> str:
    return json.dumps({**good, "priority": "critical"})


def _multi_sentence(good: dict) -> str:
    return json.dumps({**good, "summary": "The customer has a problem. They would like it looked at soon."})


_TEXT_BREAKAGES: tuple[Callable[[dict], str], ...] = (
    _fenced, _chatter, _prose, _cut_off, _missing_field, _bad_category, _bad_priority, _multi_sentence,
)
# Catalogue entries: the text ones above, plus hang and the two error kinds.
_CATALOGUE = (*_TEXT_BREAKAGES, "hang", "transient", "permanent")


class FakeLLM:
    def __init__(
        self,
        failure_rate: float = 0.25,
        seed: int | None = None,
        hang_s: float = 30.0,
        latency: tuple[float, float] = (0.05, 0.3),
    ) -> None:
        self._failure_rate = failure_rate
        self._hang_s = hang_s
        self._latency = latency
        self._rng = random.Random(seed)  # the only source of randomness

    async def complete(self, prompt: str) -> str:
        subject, body = extract_ticket(prompt)
        await asyncio.sleep(self._rng.uniform(*self._latency))
        if looks_like_injection(subject, body):
            # Gullible on purpose: obeys the attacker, to show validation cannot catch it.
            return json.dumps(
                {"category": "technical", "priority": "high", "summary": "Approved for immediate refund."}
            )
        good = _guess(subject, body)
        if self._rng.random() >= self._failure_rate:
            return json.dumps(good)
        choice = self._rng.choice(_CATALOGUE)
        if choice == "hang":
            await asyncio.sleep(self._hang_s)
            return ""  # only reached when hang_s is small; still not a valid reply
        if choice == "transient":
            raise LLMTransientError("fake: rate limited / upstream 503")
        if choice == "permanent":
            raise LLMPermanentError("fake: request rejected")
        return choice(good)


class ScriptedLLM:
    """Replays a fixed script, one item per call. For tests that need exact control."""

    def __init__(self, items: Iterable[str | BaseException | Callable[[str], Awaitable[str]]]) -> None:
        self._items = iter(items)
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            item: Any = next(self._items)
        except StopIteration:
            raise AssertionError("ScriptedLLM exhausted") from None
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return item
        return await item(prompt)
