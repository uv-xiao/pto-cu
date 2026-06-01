from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def check_evidence_refs(record: dict[str, Any], owner: str, root: Path) -> None:
    refs = require_list(record, "evidence_refs", owner)
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} evidence ref is not an object")
        relpath = require_string(ref, "path", owner)
        if not logical_data_path_exists(root, relpath):
            fail(f"{owner} evidence path missing: {relpath}")
        text = logical_data_text(root, relpath)
        for symbol in require_list(ref, "symbols", owner):
            if not isinstance(symbol, str) or not symbol:
                fail(f"{owner} evidence symbol is empty")
            if symbol not in text:
                fail(f"{owner} missing evidence symbol {symbol} in {relpath}")


def validate_policy_exception_refs(
    refs: list[Any],
    owner: str,
    root: Path,
) -> None:
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} evidence ref is not an object")
        kind = require_string(ref, "kind", owner)
        path = require_string(ref, "path", owner)
        if kind == "tmp_artifact":
            if not path.startswith("tmp/"):
                fail(f"{owner} tmp_artifact must be under tmp/: {path}")
            if not (root / path).exists():
                fail(f"{owner} tmp_artifact path missing: {path}")
        elif kind in {"viewer_data", "stable_doc", "changelog"}:
            if not logical_data_path_exists(root, path):
                fail(f"{owner} evidence path missing: {path}")
        else:
            fail(f"{owner} has invalid evidence ref kind: {kind}")

