#!/usr/bin/env python3
"""Plan or perform Qwen safetensors shard placement for PTO CUDA serving."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = (
    ROOT / "tmp" / "sources" / "qwen3-8b-model-safetensors-index-d117af2f.json"
)
DEFAULT_SHARD_DIR = ROOT / "tmp" / "sources" / "qwen3-8b-safetensors"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
DEFAULT_SOURCE_BASE_URL = (
    f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def curl_resume_command(*, url: str, target_path: str) -> str:
    return " ".join(
        [
            "curl",
            "-L",
            "-C",
            "-",
            "--fail",
            "--create-dirs",
            "-o",
            shlex.quote(target_path),
            shlex.quote(url),
        ]
    )


def shard_tensor_map(weight_map: dict[str, Any]) -> dict[str, list[str]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        by_shard[str(shard_name)].append(str(tensor_name))
    return {name: sorted(tensors) for name, tensors in by_shard.items()}


def build_shard_status(
    *,
    index_json: Path = DEFAULT_INDEX,
    shard_dir: Path = DEFAULT_SHARD_DIR,
    source_base_url: str = DEFAULT_SOURCE_BASE_URL,
    download_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not index_json.is_file():
        return {
            "schema_version": 1,
            "kind": "pto_qwen_safetensors_shard_status",
            "status": "index_missing",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "index_json": repo_relative(index_json),
            "shard_dir": repo_relative(shard_dir),
            "source_base_url": source_base_url.rstrip("/"),
            "expected_shard_count": 0,
            "present_shard_count": 0,
            "missing_shard_count": 0,
            "shards": [],
            "implemented_contracts": ["safetensors_fetch_plan"],
            "remaining_runtime_gaps": [
                "safetensors_index_capture",
                "qwen_safetensors_shard_download",
                "actual_safetensors_shape_dtype_validation",
                "cuda_device_weight_binding",
                "persistent_task_weight_arg_binding",
            ],
        }

    payload = load_json(index_json)
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map is not an object in {index_json}")

    source_base = source_base_url.rstrip("/")
    by_shard = shard_tensor_map(weight_map)
    shards = []
    missing_count = 0
    present_count = 0
    for shard_name in sorted(by_shard):
        target_path = shard_dir / shard_name
        url = f"{source_base}/{shard_name}"
        is_present = target_path.is_file()
        target_path_text = repo_relative(target_path)
        if is_present:
            present_count += 1
        else:
            missing_count += 1
        shard_record = {
            "name": shard_name,
            "status": "present" if is_present else "missing",
            "url": url,
            "target_path": target_path_text,
            "tensor_count": len(by_shard[shard_name]),
            "sample_tensors": by_shard[shard_name][:8],
            "resume_command": curl_resume_command(
                url=url,
                target_path=target_path_text,
            ),
        }
        if is_present:
            shard_record["size_bytes"] = target_path.stat().st_size
        shards.append(shard_record)

    implemented_contracts = [
        "safetensors_fetch_plan",
        "local_shard_presence_check",
    ]
    if missing_count == 0:
        implemented_contracts.append("qwen_safetensors_shards_present")
    if download_attempts:
        implemented_contracts.append("qwen_safetensors_shard_download_attempt")

    remaining_runtime_gaps = [
        "actual_safetensors_shape_dtype_validation",
        "cuda_device_weight_binding",
        "persistent_task_weight_arg_binding",
    ]
    if missing_count:
        remaining_runtime_gaps = [
            "qwen_safetensors_shard_download",
            "safetensors_tensor_open",
            *remaining_runtime_gaps,
        ]

    return {
        "schema_version": 1,
        "kind": "pto_qwen_safetensors_shard_status",
        "status": (
            "ready_for_metadata_probe" if missing_count == 0 else "shards_missing"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "shard_dir": repo_relative(shard_dir),
        "source_base_url": source_base,
        "expected_shard_count": len(shards),
        "present_shard_count": present_count,
        "missing_shard_count": missing_count,
        "expected_tensor_count": len(weight_map),
        "shards": shards,
        "download_attempts": download_attempts or [],
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": remaining_runtime_gaps,
    }


def download_missing_shards(
    *,
    index_json: Path,
    shard_dir: Path,
    source_base_url: str,
) -> list[dict[str, Any]]:
    status = build_shard_status(
        index_json=index_json,
        shard_dir=shard_dir,
        source_base_url=source_base_url,
    )
    attempts = []
    for shard in status["shards"]:
        if shard["status"] != "missing":
            continue
        target_path = Path(shard["target_path"])
        if not target_path.is_absolute():
            target_path = ROOT / target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "curl",
            "-L",
            "-C",
            "-",
            "--fail",
            "--create-dirs",
            "-o",
            str(target_path),
            shard["url"],
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        attempts.append(
            {
                "name": shard["name"],
                "url": shard["url"],
                "target_path": repo_relative(target_path),
                "returncode": result.returncode,
                "status": "downloaded" if result.returncode == 0 else "failed",
                "output_tail": result.stdout[-4000:],
            }
        )
        if result.returncode != 0:
            break
    return attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--source-base-url", default=DEFAULT_SOURCE_BASE_URL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_attempts = None
    if args.download:
        download_attempts = download_missing_shards(
            index_json=args.index_json,
            shard_dir=args.shard_dir,
            source_base_url=args.source_base_url,
        )
    payload = build_shard_status(
        index_json=args.index_json,
        shard_dir=args.shard_dir,
        source_base_url=args.source_base_url,
        download_attempts=download_attempts,
    )
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
