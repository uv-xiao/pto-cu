"""Error helpers for paper baseline environment attempts."""

from __future__ import annotations


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline environment attempt failed: {message}")
