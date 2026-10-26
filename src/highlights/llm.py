"""Ask a local LLM (Ollama) whether a candidate moment would work as a Short."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, Field

from src.common.config import OllamaConfig
from src.common.retry import with_retries
from src.common.timecode import format_timecode

_log = logging.getLogger("autoclip.highlights")

SYSTEM_PROMPT = """\
You help a small Indonesian gaming streamer grow by picking moments from a Dota 2 livestream \
to cut into YouTube Shorts. The streamer talks casually in Indonesian mixed with English while \
playing with friends.

You get part of an auto-generated transcript. Each line is: [line id] time text.
- [tertawa] is laughter, [musik] is music, other [tags] are sounds.
- *** is a censored swear word.
- anjing, anjir, anjay, bangsat, tolol, goblok, mampus and similar words are exclamations and \
banter between friends, not literal: "anjing" means something like "damn", not a dog.
- Dota terms: gank, ulti (ultimate), farm, carry, support, Roshan, Aegis, buyback, report, \
mute, smurf, and hero names such as Chrono or Void.
- Speech recognition makes mistakes: judge the moment, not the spelling.

Score 0-10 how likely a stranger scrolling Shorts would stop and watch to the end. Be \
strict: most stretches of a stream are ordinary, and these were only shortlisted because \
someone laughed or got loud. A laugh tag alone doesn't make a moment good: check that the \
transcript shows what was funny, and that a stranger would get it.
- 9-10: rare. You'd bet on it: an unmistakable punchline or reaction that needs no context.
- 7-8: clearly good: a funny line, rage moment, epic fail or clutch play you can follow \
from the transcript alone.
- 4-6: some fun, but the payoff is weak, unclear, or needs context.
- 0-3: routine talk, callouts, item or build talk, or laughter about something you can't see \
in the transcript.

Pick start_id and end_id from the given line ids: start just before the setup (no slow \
intro), end right after the payoff or the laughter.
title: short and catchy, in casual Indonesian like the streamer talks, at most 60 characters, \
no hashtags.
reason: one English sentence: what happens and why it works or doesn't.
Answer with JSON only."""

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "start_id": {"type": "integer"},
        "end_id": {"type": "integer"},
        "title": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["score", "start_id", "end_id", "title", "reason"],
}


def system_prompt(glossary: Mapping[str, str] | None = None) -> str:
    """The instructions, with the stream's own nicknames and slang (config: highlights.glossary)."""
    if not glossary:
        return SYSTEM_PROMPT
    terms = "; ".join(f"{name} = {meaning}" for name, meaning in glossary.items())
    anchor = "- Speech recognition makes mistakes"
    line = f"- Nicknames and slang on this stream: {terms}."
    return SYSTEM_PROMPT.replace(anchor, f"{line}\n{anchor}", 1)


class LlmError(RuntimeError):
    """Ollama is unreachable, the model is missing, or a prompt is too long."""


class Verdict(BaseModel):
    score: int = Field(ge=0, le=10)
    start_id: int
    end_id: int
    title: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1)


def format_excerpt(
    segments: list[dict[str, Any]], sounds: list[dict[str, Any]], ids: list[int]
) -> str:
    """One line per sentence; sound tags go on the line they follow."""
    lines = []
    for i in ids:
        segment = segments[i]
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else float("inf")
        tags = [f"[{s['tag']}]" for s in sounds if segment["start"] <= s["t"] < next_start]
        text = " ".join([segment["text"], *tags])
        lines.append(f"[{i}] {format_timecode(segment['start'])} {text}")
    return "\n".join(lines)


def build_prompt(
    excerpt: str, hint: str, first_id: int, last_id: int, min_s: float, max_s: float
) -> str:
    return (
        f"Signals for this part of the stream: {hint}.\n\n"
        f"Transcript, lines {first_id} to {last_id}:\n{excerpt}\n\n"
        f"Score this moment. Choose start_id and end_id (between {first_id} and {last_id}) "
        f"for a clip of {min_s:.0f} to {max_s:.0f} seconds, then write the title and reason."
    )


class OllamaJudge:
    def __init__(
        self,
        cfg: OllamaConfig,
        client: httpx.Client | None = None,
        glossary: Mapping[str, str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._system = system_prompt(glossary)
        self._client = client or httpx.Client(base_url=cfg.host, timeout=cfg.timeout_seconds)

    def check_ready(self) -> None:
        try:
            response = self._client.get("/api/tags", timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlmError(
                f"Ollama isn't reachable at {self._cfg.host} ({exc}). Start Ollama, or run "
                "with --no-llm to score with laughs, loudness and chat only."
            ) from exc
        names = {m["name"] for m in response.json().get("models", [])}
        if self._cfg.model not in names and f"{self._cfg.model}:latest" not in names:
            raise LlmError(
                f"Ollama has no model {self._cfg.model}. Run: ollama pull {self._cfg.model}"
            )

    def judge(
        self, excerpt: str, hint: str, ids: list[int], *, min_s: float, max_s: float
    ) -> Verdict | None:
        """The model's verdict, or None if it twice answered with something unusable."""
        prompt = build_prompt(excerpt, hint, ids[0], ids[-1], min_s, max_s)
        # ~3 characters per token for this mix of Indonesian and English. Ollama silently
        # drops whatever doesn't fit in num_ctx, so refuse instead.
        if (len(self._system) + len(prompt)) / 3 > self._cfg.num_ctx * 0.8:
            raise LlmError(
                f"Prompt is too long for num_ctx={self._cfg.num_ctx}; raise it in config.yaml"
            )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            content = self._chat(messages)
            try:
                verdict = Verdict.model_validate_json(content)
                if not ids[0] <= verdict.start_id <= verdict.end_id <= ids[-1]:
                    raise ValueError(
                        f"start_id and end_id must be line ids between {ids[0]} and {ids[-1]}, "
                        "with start_id <= end_id"
                    )
                return verdict
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                problem = str(exc).splitlines()[0][:200]
                _log.warning(
                    "LLM answer rejected: %s%s", problem, "; asking again" if not attempt else ""
                )
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That answer is invalid: {problem}. "
                        "Reply again with valid JSON only.",
                    },
                ]
        return None

    def unload(self) -> None:
        """Free the model's VRAM right away (render needs the GPU next)."""
        try:
            self._client.post(
                "/api/generate", json={"model": self._cfg.model, "keep_alive": 0}, timeout=30
            )
        except httpx.HTTPError as exc:
            _log.warning("Could not unload %s from Ollama: %s", self._cfg.model, exc)

    def close(self) -> None:
        self._client.close()

    @with_retries(attempts=3)
    def _chat(self, messages: list[dict[str, str]]) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self._cfg.model,
                "messages": messages,
                "format": VERDICT_SCHEMA,
                "stream": False,
                "keep_alive": "5m",  # stays loaded between candidates; unload() frees it at the end
                "options": {"temperature": self._cfg.temperature, "num_ctx": self._cfg.num_ctx},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
