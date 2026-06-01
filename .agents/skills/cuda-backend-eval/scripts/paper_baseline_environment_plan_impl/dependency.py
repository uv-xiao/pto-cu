"""Dependency-source parsing for paper baseline environment plans."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None  # type: ignore[assignment]


PACKAGE_RE = re.compile(r"^([A-Za-z0-9_.-]+)")


def package_name(requirement: str) -> str:
    text = requirement.strip()
    if not text or text.startswith("#") or text.startswith("-"):
        return ""
    text = text.split(";", 1)[0].strip()
    text = text.split("[", 1)[0].strip()
    match = PACKAGE_RE.match(text)
    return match.group(1).replace("_", "-").lower() if match else ""


def strip_inline_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        if char == "#" and not in_quote:
            return line[:index]
    return line


def fallback_toml_dependencies(path: Path) -> list[str]:
    dependencies: list[str] = []
    in_dependencies = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw_line).strip()
        if line.startswith("dependencies") or line.startswith("requires"):
            in_dependencies = "[" in line
            if line.endswith("]"):
                in_dependencies = False
            continue
        if not in_dependencies:
            continue
        if line.startswith("]"):
            in_dependencies = False
            continue
        if not line:
            continue
        dependencies.append(line.strip(",").strip("'\""))
    return dependencies


def toml_dependencies(path: Path) -> list[str]:
    if tomllib is None:
        return fallback_toml_dependencies(path)
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    dependencies: list[str] = []
    project = payload.get("project", {})
    build_system = payload.get("build-system", {})
    for value in project.get("dependencies", []):
        if isinstance(value, str):
            dependencies.append(value)
    for value in build_system.get("requires", []):
        if isinstance(value, str):
            dependencies.append(value)
    return dependencies


def requirements_dependencies(path: Path) -> list[str]:
    dependencies: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw_line).strip()
        if not line or line.startswith("-"):
            continue
        dependencies.append(line)
    return dependencies


def dependency_evidence(
    source_root: Path,
    dependency_sources: list[str],
) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    for source in dependency_sources:
        path = source_root / source
        if not path.is_file():
            continue
        dependencies = (
            toml_dependencies(path)
            if path.suffix == ".toml"
            else requirements_dependencies(path)
        )
        for dependency in dependencies:
            name = package_name(dependency)
            if not name:
                continue
            evidence.setdefault(name, []).append(f"{source}: {dependency}")
    return evidence
