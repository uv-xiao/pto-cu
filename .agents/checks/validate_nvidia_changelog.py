#!/usr/bin/env python3
"""Validate NVIDIA backend changelog report structure."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SECTIONS = (
    "## Code And Data Changed",
    "## Architecture Quality",
    "## Evaluation Run",
    "## Remaining Gaps",
)


def fail(message: str) -> None:
    raise SystemExit(f"nvidia changelog validation failed: {message}")


def linked_reports(index_text: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"\]\(([^)]+\.md)\)", index_text)
    }


def section_body(text: str, section: str) -> str:
    start = text.find(section)
    if start < 0:
        fail(f"missing section {section}")
    body_start = start + len(section)
    next_section = text.find("\n## ", body_start)
    if next_section < 0:
        return text[body_start:].strip()
    return text[body_start:next_section].strip()


def validate_report(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# "):
        fail(f"{path.relative_to(ROOT)} must start with an H1 title")
    for section in REQUIRED_SECTIONS:
        try:
            body = section_body(text, section)
        except SystemExit as exc:
            fail(f"{path.relative_to(ROOT)} {exc}")
        if not body:
            fail(f"{path.relative_to(ROOT)} has empty {section}")
    evaluation = section_body(text, "## Evaluation Run")
    if "```" not in evaluation and "passed" not in evaluation.lower():
        fail(f"{path.relative_to(ROOT)} lacks verification evidence")
    gaps = section_body(text, "## Remaining Gaps")
    if gaps.lower() in {"none", "none."}:
        fail(f"{path.relative_to(ROOT)} must state residual review context")


def validate_changelog(root: Path = ROOT) -> None:
    changelog_root = root / "docs" / "nvidia-backend" / "changelog"
    index = changelog_root / "index.md"
    if not index.is_file():
        fail(f"missing {index.relative_to(root)}")
    reports = {
        path.name
        for path in changelog_root.glob("*.md")
        if path.name != "index.md"
    }
    linked = linked_reports(index.read_text(encoding="utf-8"))
    missing_links = sorted(reports - linked)
    stale_links = sorted(linked - reports)
    if missing_links:
        fail(f"reports missing from index: {missing_links}")
    if stale_links:
        fail(f"index links missing reports: {stale_links}")
    for name in sorted(reports):
        validate_report(changelog_root / name)


def main() -> None:
    validate_changelog()
    print("nvidia changelog validation passed")


if __name__ == "__main__":
    main()
