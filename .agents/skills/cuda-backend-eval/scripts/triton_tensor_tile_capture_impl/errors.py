"""Error helpers for Triton tensor-tile captures."""

from __future__ import annotations


def fail(message: str) -> None:
    raise SystemExit(f"triton tensor tile capture failed: {message}")
