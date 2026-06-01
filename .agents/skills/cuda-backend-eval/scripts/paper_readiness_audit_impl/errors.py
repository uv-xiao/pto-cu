from __future__ import annotations


def fail(message: str) -> None:
    raise SystemExit(f"paper readiness audit failed: {message}")
