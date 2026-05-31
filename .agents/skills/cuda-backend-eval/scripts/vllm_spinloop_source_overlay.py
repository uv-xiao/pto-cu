#!/usr/bin/env python3
"""Create a local vLLM source overlay with the spinloop ABI build fix."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]

SPINLOOP_TARGET = """define_extension_target(
  spinloop
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES ${VLLM_SPINLOOP_EXT_SRC}
  COMPILE_FLAGS ${SPINLOOP_COMPILE_FLAGS}
  USE_SABI 3.11
  WITH_SOABI)"""

SPINLOOP_TARGET_WITH_FLAG = (
    SPINLOOP_TARGET
    + """

target_compile_options(spinloop PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)"""
)


def fail(message: str) -> None:
    raise SystemExit(f"vLLM spinloop source overlay failed: {message}")


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "build",
        "dist",
    }
    return {
        name
        for name in names
        if name in ignored or name.endswith(".egg-info") or name.endswith(".pyc")
    }


def patch_cmake(path: Path) -> bool:
    cmake_path = path / "CMakeLists.txt"
    if not cmake_path.is_file():
        fail(f"missing {repo_relative(cmake_path)}")
    text = cmake_path.read_text(encoding="utf-8")
    marker_index = text.find("define_extension_target(\n  spinloop")
    if marker_index < 0:
        fail("could not find the expected spinloop target in CMakeLists.txt")
    next_section = text.find("\n#\n", marker_index + len(SPINLOOP_TARGET))
    spinloop_text = text[marker_index:next_section if next_section > 0 else None]
    if "-UPy_LIMITED_API" in spinloop_text:
        return False
    if SPINLOOP_TARGET not in text:
        fail("could not find the expected spinloop target in CMakeLists.txt")
    cmake_path.write_text(
        text.replace(SPINLOOP_TARGET, SPINLOOP_TARGET_WITH_FLAG, 1),
        encoding="utf-8",
    )
    return True


def create_overlay(source: Path, overlay: Path) -> dict[str, Any]:
    if not source.is_dir():
        fail(f"missing source directory: {repo_relative(source)}")
    try:
        overlay.relative_to(source)
    except ValueError:
        pass
    else:
        fail("overlay directory must not be inside the source checkout")
    if overlay.exists():
        shutil.rmtree(overlay)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, overlay, ignore=copy_ignore)
    patched = patch_cmake(overlay)
    report = {
        "status": "pass",
        "source_path": repo_relative(source),
        "overlay_path": repo_relative(overlay),
        "patched_files": [
            {
                "path": repo_relative(overlay / "CMakeLists.txt"),
                "change": (
                    "Added a spinloop-only CXX compile option "
                    "-UPy_LIMITED_API after the stable-ABI target definition."
                ),
                "applied": patched,
            }
        ],
        "upstream_checkout_mutated": False,
        "next_action": "Run vllm_spinloop_preflight.py against the overlay.",
    }
    report_path = overlay / "pto-cu-source-overlay.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--overlay", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_overlay(repo_path(args.source), repo_path(args.overlay))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
