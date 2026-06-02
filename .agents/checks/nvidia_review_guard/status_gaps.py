from __future__ import annotations

from .common import *  # noqa: F403


def _linked_gap_paths(status_text: str) -> list[str]:
    try:
        body = status_text.split("\n## Remaining Gaps\n", 1)[1].split("\n## ", 1)[0]
    except IndexError:
        fail("status.md has no Remaining Gaps section")
    paths: list[str] = []
    for line in body.splitlines():
        if line.startswith("- [") and "](" in line and ")" in line:
            paths.append(line.split("](", 1)[1].split(")", 1)[0])
    return paths


def _gap_file(relpath: str):
    path = DOC_ROOT / relpath
    if path.is_dir():
        path = path / "index.md"
    require_file(path)
    return path


def check_remaining_gap_contract() -> None:
    status = DOC_ROOT / "status.md"
    require_file(status)
    status_text = status.read_text(encoding="utf-8")
    linked_gap_paths = _linked_gap_paths(status_text)
    if not linked_gap_paths and "No open backend implementation gaps." not in status_text:
        fail("status.md has no linked remaining gaps and no closure statement")
    for relpath in linked_gap_paths:
        path = _gap_file(relpath)
        text = path.read_text(encoding="utf-8")
        for heading in (
            "## Open Gap",
            "## Current Evidence",
            "## Promotion Gate",
            "## Next Actions",
        ):
            if heading not in text:
                fail(f"{path.relative_to(ROOT)} missing {heading}")
        for needle in (
            "docs/nvidia-backend/benchmark-viewer/data/",
            ".agents/checks/",
        ):
            if needle not in text:
                fail(f"{path.relative_to(ROOT)} missing evidence needle: {needle}")
        if "tmp/" not in text:
            fail(f"{path.relative_to(ROOT)} missing tmp artifact reference")
