"""GPU-free tests for the open-VLM backend's pure logic: message conversion
and JSON extraction. The model itself only runs on Colab."""

import base64
import io

import numpy as np
import pytest

from chroma_reasoner.reasoner.backend_open import extract_json, to_qwen_messages


def _png_b64() -> str:
    from PIL import Image

    img = Image.fromarray(np.full((8, 8), 128, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _messages():
    return [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": _png_b64()}},
        {"type": "text", "text": "Analyze the image."},
    ]}]


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_chatter_and_fences():
    text = 'Sure! Here is the selection:\n```json\n{"regions": [{"object": "sky"}]}\n```\nDone.'
    assert extract_json(text)["regions"][0]["object"] == "sky"


def test_extract_json_braces_inside_strings():
    text = 'prefix {"rationale": "uses { and } inside", "n": 2} suffix'
    assert extract_json(text)["n"] == 2


def test_extract_json_failures():
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"unbalanced": ')


def test_to_qwen_messages_structure():
    qwen = to_qwen_messages("SYSTEM TEXT", _messages())
    assert qwen[0]["role"] == "system"
    assert qwen[0]["content"][0]["text"] == "SYSTEM TEXT"
    user = qwen[1]
    kinds = [b["type"] for b in user["content"]]
    assert kinds == ["image", "text"]
    # the JSON format contract is appended to the first user text block
    assert "ONLY a JSON object" in user["content"][1]["text"]
    # image decoded to a PIL image
    assert user["content"][0]["image"].size == (8, 8)


def test_to_qwen_messages_repair_turns_pass_through():
    msgs = _messages() + [
        {"role": "assistant", "content": '{"regions": []}'},
        {"role": "user", "content": "fix the errors"},
    ]
    qwen = to_qwen_messages("S", msgs)
    assert qwen[-1]["content"][0]["text"] == "fix the errors"
    assert qwen[-2]["role"] == "assistant"
    # format contract added only once, on the first user turn
    joined = "".join(b.get("text", "") for m in qwen for b in m["content"])
    assert joined.count("ONLY a JSON object") == 1
