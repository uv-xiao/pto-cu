#!/usr/bin/env python3
"""Preflight vLLM spinloop stable-ABI compatibility before long builds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def fail(message: str) -> None:
    raise SystemExit(f"vLLM spinloop preflight failed: {message}")


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def python_version(env_python: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            str(env_python),
            "-c",
            "import json, sys; print(json.dumps(tuple(sys.version_info[:3])))",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail(f"could not inspect {env_python}: {result.stderr.strip()}")
    version = json.loads(result.stdout)
    if (
        not isinstance(version, list)
        or len(version) != 3
        or not all(isinstance(item, int) for item in version)
    ):
        fail(f"unexpected version payload from {env_python}: {version!r}")
    return tuple(version)  # type: ignore[return-value]


def spinloop_uses_limited_api(cmake_text: str) -> bool:
    marker = "define_extension_target(\n  spinloop"
    target_start = cmake_text.find(marker)
    if target_start < 0:
        fail("could not find spinloop define_extension_target in CMakeLists.txt")
    next_section = cmake_text.find("\n#\n", target_start + len(marker))
    target_text = cmake_text[target_start:next_section if next_section > 0 else None]
    return "USE_SABI 3.11" in target_text and "-UPy_LIMITED_API" not in target_text


def build_report(source_root: Path, env_python: Path) -> dict[str, Any]:
    cmake_path = source_root / "CMakeLists.txt"
    spinloop_path = source_root / "csrc" / "spinloop.cpp"
    if not cmake_path.is_file():
        fail(f"missing {cmake_path}")
    if not spinloop_path.is_file():
        fail(f"missing {spinloop_path}")

    cmake_text = cmake_path.read_text(encoding="utf-8")
    spinloop_text = spinloop_path.read_text(encoding="utf-8")
    version = python_version(env_python)
    limited_api = spinloop_uses_limited_api(cmake_text)
    uses_buffer_api = "Py_buffer" in spinloop_text or "PyBuffer_Release" in spinloop_text
    incompatible = limited_api and uses_buffer_api and version < (3, 11, 0)
    return {
        "status": "fail" if incompatible else "pass",
        "source_root": str(source_root),
        "env_python": str(env_python),
        "python_version": ".".join(str(item) for item in version),
        "spinloop_uses_sabi_3_11": limited_api,
        "spinloop_uses_buffer_api": uses_buffer_api,
        "blocker": (
            "spinloop uses Py_buffer/PyBuffer_Release while the target is built "
            "with USE_SABI 3.11, but the isolated environment uses Python "
            f"{'.'.join(str(item) for item in version)} headers."
            if incompatible
            else ""
        ),
        "next_action": (
            "Use Python >=3.11 for the vLLM baseline environment, or apply a "
            "reviewed local reproducibility patch/build flag that removes "
            "Py_LIMITED_API from the spinloop CXX compile."
            if incompatible
            else "Continue with editable install."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--env-python", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        source_root=repo_path(args.source),
        env_python=repo_path(args.env_python),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
