"""Error helpers for paper baseline environment plans."""

from __future__ import annotations


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline environment plan failed: {message}")
