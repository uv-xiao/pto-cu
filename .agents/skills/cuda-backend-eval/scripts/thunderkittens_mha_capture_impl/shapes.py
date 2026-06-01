"""Shape parsing for ThunderKittens MHA captures."""

from __future__ import annotations

import argparse


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be b,h,n,d")
    try:
        b, h, n, d = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape entries must be integers") from exc
    if min(b, h, n, d) <= 0:
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return b, h, n, d
