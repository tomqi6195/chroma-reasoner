"""The LLM's output contract: a *selection*, never colours.

The reasoner emits which KB objects are present, how to ground them, which
modifiers are operative, and an estimated luminance per region. Colours are
resolved locally against the KB (roadmap §4.2: "the KB supplies the colours").

Used as a structured-outputs JSON schema (output_config.format), so responses
are guaranteed to parse. Structured outputs don't support min/max numeric
constraints — ranges are validated in planner.py instead.
"""

SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_summary", "global_modifiers", "regions"],
    "properties": {
        "scene_summary": {
            "type": "string",
            "description": "One or two sentences: scene type, era cues, lighting, notable objects.",
        },
        "global_modifiers": {
            "type": "array",
            "description": (
                "Image-wide factors, such as film rendering or overall mood, "
                "that do not route through one object."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["family", "value", "why"],
                "properties": {
                    "family": {"type": "string"},
                    "value": {"type": "string"},
                    "why": {"type": "string"},
                },
            },
        },
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "object",
                    "grounding_phrase",
                    "estimated_L",
                    "modifiers",
                    "confidence",
                    "rationale",
                ],
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "A KB object class or alias from the vocabulary.",
                    },
                    "grounding_phrase": {
                        "type": "string",
                        "description": (
                            "Specific open-vocabulary detection phrase, such as "
                            '"the woman\'s long dress".'
                        ),
                    },
                    "estimated_L": {
                        "type": "number",
                        "description": (
                            "Estimated median CIE L of the grayscale region: "
                            "0 black, 50 mid-grey, 100 white."
                        ),
                    },
                    "modifiers": {
                        "type": "array",
                        "description": (
                            "Applicable KB (family, value) pairs in this order: era, geography, "
                            "season, weather, time_of_day, mood."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["family", "value", "why"],
                            "properties": {
                                "family": {"type": "string"},
                                "value": {"type": "string"},
                                "why": {"type": "string"},
                            },
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence from 0 to 1 given the content constraints.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this object, these modifiers, this confidence.",
                    },
                },
            },
        },
    },
}
