"""Ascend attention kernel registrations."""

from .topk_select import AscendTopKSelectKernel

__all__ = ["AscendTopKSelectKernel"]
