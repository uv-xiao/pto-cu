from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .commands import run_command
from .errors import fail


def check_path_exists(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    relpath = str(check["path"])
    path = source_root / relpath
    return {
        "kind": "path_exists",
        "path": relpath,
        "why": check["why"],
        "status": "pass" if path.exists() else "fail",
    }


def check_py_compile(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    relpath = str(check["path"])
    path = source_root / relpath
    if not path.is_file():
        return {
            "kind": "py_compile",
            "path": relpath,
            "why": check["why"],
            "status": "fail",
            "output": "file missing",
        }
    result = run_command([sys.executable, "-m", "py_compile", str(path)])
    return {
        "kind": "py_compile",
        "path": relpath,
        "why": check["why"],
        "status": "pass" if result["returncode"] == 0 else "fail",
        "returncode": result["returncode"],
        "output": result["output"],
    }


def build_python_env(source_root: Path, check: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    if check.get("python_no_user_site") is True:
        env["PYTHONNOUSERSITE"] = "1"
    pythonpath = check.get("pythonpath")
    if pythonpath:
        path = source_root / str(pythonpath)
        env["PYTHONPATH"] = (
            f"{path}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(path)
        )
    return env


def check_python_module(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    module = str(check["module"])
    base = {
        "kind": "python_module",
        "module": module,
        "why": check["why"],
    }
    if check.get("python_no_user_site") is True or check.get("pythonpath"):
        env = build_python_env(source_root, check)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, sys; "
                    f"sys.exit(0 if importlib.util.find_spec({module!r}) "
                    "is not None else 1)"
                ),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
            env=env,
        )
        if check.get("pythonpath"):
            base["pythonpath"] = check.get("pythonpath")
        if check.get("python_no_user_site") is True:
            base["python_no_user_site"] = True
        base.update(
            {
                "status": "pass" if result.returncode == 0 else "fail",
                "returncode": result.returncode,
                "output": result.stdout[-4000:],
            }
        )
        return base
    found = importlib.util.find_spec(module) is not None
    base["status"] = "pass" if found else "fail"
    return base


def check_python_import(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    module = str(check["module"])
    env = build_python_env(source_root, check)
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
        env=env,
    )
    payload = {
        "kind": "python_import",
        "module": module,
        "why": check["why"],
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "output": result.stdout[-4000:],
    }
    if check.get("pythonpath"):
        payload["pythonpath"] = check.get("pythonpath")
    if check.get("python_no_user_site") is True:
        payload["python_no_user_site"] = True
    return payload


def collect_check(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    kind = check.get("kind")
    if kind == "path_exists":
        return check_path_exists(source_root, check)
    if kind == "py_compile":
        return check_py_compile(source_root, check)
    if kind == "python_module":
        return check_python_module(source_root, check)
    if kind == "python_import":
        return check_python_import(source_root, check)
    fail(f"unknown probe check kind: {kind}")
