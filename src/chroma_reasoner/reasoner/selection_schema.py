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
            "description": "Image-wide factors (film rendering, overall mood) that don't route through a single object. May be empty.",
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
                "required": ["object", "grounding_phrase", "estimated_L",
                             "modifiers", "confidence", "rationale"],
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "A KB object class or listed alias, exactly as given in the vocabulary.",
                    },
                    "grounding_phrase": {
                        "type": "string",
                        "description": "Specific phrase for open-vocabulary detection, e.g. \"the woman's long dress\".",
                    },
                    "estimated_L": {
                        "type": "number",
                        "description": "Estimated median CIE L (0-100) of this region in the grayscale image: 0 black, 50 mid-grey, 100 white.",
                    },
                    "modifiers": {
                        "type": "array",
                        "description": "Operative (family, value) pairs from the KB catalog, in application order: era, geography, season, weather, time_of_day, mood. Only include factors that genuinely apply to this object.",
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
                        "description": "0-1: how confident the colour assignment can be, given content constraints.",
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
