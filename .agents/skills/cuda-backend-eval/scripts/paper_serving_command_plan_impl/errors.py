"""Error helpers for paper serving command plans."""

from __future__ import annotations


def fail(message: str) -> None:
    raise SystemExit(f"paper serving command plan failed: {message}")
