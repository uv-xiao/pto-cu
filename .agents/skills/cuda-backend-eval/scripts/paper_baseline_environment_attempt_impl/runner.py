"""Bounded command execution for paper baseline environment attempts."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from paper_baseline_environment_attempt_impl.errors import fail
from paper_baseline_environment_attempt_impl.paths import ROOT
from paper_baseline_environment_attempt_impl.paths import repo_relative


def command_is_allowed(command: str) -> bool:
    return ".venv" not in command and "--user" not in command


def run_step(
    *,
    command: str,
    index: int,
    kind: str,
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not command_is_allowed(command):
        fail(f"refusing unsafe environment command: {command}")
    log_path = output_root / f"step-{index:02d}.log"
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - started
        output = result.stdout
        returncode: int | None = result.returncode
        status = "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = None
        status = "timeout"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return {
        "index": index,
        "kind": kind,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
        "command": command,
        "log": repo_relative(log_path),
    }
