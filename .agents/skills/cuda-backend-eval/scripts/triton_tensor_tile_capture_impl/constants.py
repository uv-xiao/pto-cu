"""Constants for Triton tensor-tile captures."""

from __future__ import annotations


DEFAULT_SHAPE = "n=1024, tensor tile 16x16x16"
DEFAULT_DTYPE = "tf32 Triton tl.dot, f32 accumulator"
DEFAULT_TOLERANCE = 1.0e-3
