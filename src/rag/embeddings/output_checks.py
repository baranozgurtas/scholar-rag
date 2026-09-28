"""Validation of BGE-M3 forward-pass output (numpy only, no torch).

A forward pass that fails on the accelerator can return NaN, inf or all-zero
rows without raising. These checks turn that into `ModelOutputError`, so a
retrieval leg fails loudly instead of returning scrambled or empty results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rag.retrieval.types import ModelOutputError


def check_dense(dense: np.ndarray) -> None:
    """Raise if any dense row is non-finite or all zeros."""
    if not np.isfinite(dense).all():
        raise ModelOutputError("dense embedding contains NaN/inf")
    if (np.linalg.norm(dense, axis=1) == 0).any():
        raise ModelOutputError("dense embedding is all zeros")


def to_sparse(weights: dict[Any, Any]) -> dict[int, float]:
    """Token-id → weight, keeping positive weights; raise on non-finite ones.

    Only finite zero/negative weights are dropped. A NaN weight used to fail
    `v > 0` and vanish, so a broken forward pass left an empty sparse query
    and the sparse leg silently returned nothing.
    """
    out: dict[int, float] = {}
    for k, v in weights.items():
        f = float(v)
        if not math.isfinite(f):
            raise ModelOutputError("sparse lexical weights contain NaN/inf")
        if f > 0:
            out[int(k)] = f
    return out


__all__ = ["check_dense", "to_sparse"]
