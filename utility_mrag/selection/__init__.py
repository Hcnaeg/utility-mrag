"""Top-K candidate evidence selection."""

from .surrogate_selector import SurrogateSelector
from .topk import select_top_k, sort_records_descending

__all__ = ["SurrogateSelector", "select_top_k", "sort_records_descending"]
