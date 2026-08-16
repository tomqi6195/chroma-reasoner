"""Evaluation helpers for ablations, paired analysis, and references."""

from .ablation import llm_color_plan
from .paired import compare_arms, format_comparison
from .reference import plan_vs_reference

__all__ = ["compare_arms", "format_comparison", "llm_color_plan", "plan_vs_reference"]
