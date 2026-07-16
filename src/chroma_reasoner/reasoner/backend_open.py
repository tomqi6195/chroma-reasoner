"""Open-VLM backend: Qwen2.5-VL via HuggingFace transformers.

No API key, no billing — runs on any GPU with ~18 GB VRAM for the 7B model
(Colab A100/L4; use the 3B model for T4). Implements the same Backend
protocol as AnthropicBackend, so the planner is unchanged.

Open models have no structured-outputs guarantee, so this backend:
  1. appends an explicit JSON-format contract to the first user turn,
  2. extracts the first balanced JSON object from the generation,
  3. retries once with a "JSON only" nudge if extraction/parsing fails.
Content-level errors (made-up object classes) are still handled by the
planner's repair round, exactly as with the Anthropic backend.
"""

from __future__ import annotations

import base64
import io
import json

from .prompts import json_format_instructions

DEFAULT_OPEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def extract_json(text: str) -> dict:
    """First balanced {...} object in text -> dict. Raises ValueError."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in model output")


def to_qwen_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert our Anthropic-style messages to Qwen chat-template format.

    Image blocks (base64) become PIL images; text blocks pass through. The
    JSON-format contract is appended to the first user text block, since open
    backends have no structured-outputs equivalent.
    """
    from PIL import Image

    out = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    format_added = False
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"],
                        "content": [{"type": "text", "text": content}]})
            continue
        blocks = []
        for block in content:
            if block["type"] == "image":
                raw = base64.standard_b64decode(block["source"]["data"])
                blocks.append({"type": "image",
                               "image": Image.open(io.BytesIO(raw)).convert("RGB")})
            elif block["type"] == "text":
                text = block["text"]
                if message["role"] == "user" and not format_added:
                    text += "\n" + json_format_instructions()
                    format_added = True
                blocks.append({"type": "text", "text": text})
        out.append({"role": message["role"], "content": blocks})
    return out


class QwenVLBackend:
    def __init__(self, model_id: str = DEFAULT_OPEN_MODEL, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.max_new_tokens = max_new_tokens

    def _generate(self, qwen_messages: list[dict]) -> str:
        images = [b["image"] for m in qwen_messages for b in m["content"]
                  if b.get("type") == "image"]
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None,
                                return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                        do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def complete(self, system: str, messages: list[dict]) -> dict:
        qwen_messages = to_qwen_messages(system, messages)
        raw = self._generate(qwen_messages)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            # one format-repair attempt before handing back to the planner
            qwen_messages.append({"role": "assistant",
                                  "content": [{"type": "text", "text": raw}]})
            qwen_messages.append({"role": "user", "content": [{
                "type": "text",
                "text": "That was not a single valid JSON object. Re-emit the full "
                        "selection as ONLY valid JSON, matching the required shape."}]})
            return extract_json(self._generate(qwen_messages))
