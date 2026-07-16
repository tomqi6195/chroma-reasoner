"""LLM backend behind one interface (roadmap §4.2: API model now, open VLM
swappable later). A backend takes a conversation and returns the selection
dict; everything else (prompting, validation, KB resolution) is backend-
agnostic."""

from __future__ import annotations

import json
from typing import Protocol

from .selection_schema import SELECTION_SCHEMA

DEFAULT_MODEL = "claude-opus-4-8"


class Backend(Protocol):
    def complete(self, system: str, messages: list[dict]) -> dict:
        """messages: Anthropic-style message dicts. Returns the parsed selection."""
        ...


class AnthropicBackend:
    """Claude with vision + structured outputs: the selection JSON is
    schema-guaranteed by output_config.format, so no output parsing can fail.

    Credentials resolve from the environment (ANTHROPIC_API_KEY or an
    `ant auth login` profile) via the zero-arg client.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"format": {"type": "json_schema",
                                      "schema": SELECTION_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def image_block(image_path: str) -> dict:
    """Base64 image content block for an Anthropic message."""
    import base64
    from pathlib import Path

    suffix = Path(image_path).suffix.lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}[suffix.lstrip(".")]
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
