"""Knowledge-base loading and colour resolution."""

from .engine import Resolution, resolve
from .store import KnowledgeBase, load_kb

__all__ = ["KnowledgeBase", "Resolution", "load_kb", "resolve"]
