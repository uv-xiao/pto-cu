#!/usr/bin/env python3
"""Emit Qwen runtime token-buffer binding evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
PROMPT_ACCOUNTING = THIS_DIR / "qwen_prompt_accounting.py"
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
DEFAULT_CACHE_DIR = ROOT / "tmp" / "hf-tokenizers" / "qwen3-8b-d117af2f"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


EVIDENCE_SYMBOLS = [
    "pto_qwen_runtime_input_binding",
    "tokenizer_to_runtime_input_ids",
    "attention_mask_buffer",
    "runtime_token_buffer_plan",
    "decode_output_buffer_plan",
]


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


def load_prompt_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "qwen_prompt_accounting_for_runtime_binding",
        PROMPT_ACCOUNTING,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PROMPT_ACCOUNTING}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def token_checksum(token_ids: list[int]) -> str:
    body = ",".join(str(item) for item in token_ids).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def buffer_descriptor(
    *,
    name: str,
    dtype: str,
    shape: list[int],
    values: list[int],
    initializer: int | None = None,
) -> dict[str, Any]:
    element_count = 1
    for dim in shape:
        element_count *= dim
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "element_count": element_count,
        "byte_count": element_count * 4,
        "host_values_preview": values[: min(len(values), 16)],
        "initializer": initializer,
        "checksum": token_checksum(values),
    }


def tokenizer_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        return pad_token_id
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return eos_token_id
    return 0


def aligned_prompt_tokens(
    *,
    token_ids: list[int],
    target_prompt_tokens: int,
    pad_token_id: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    observed = len(token_ids)
    if observed == target_prompt_tokens:
        status = "exact"
        runtime_ids = list(token_ids)
    elif observed < target_prompt_tokens:
        status = "padded_to_target"
        runtime_ids = token_ids + [pad_token_id] * (target_prompt_tokens - observed)
    else:
        status = "requires_regenerated_prompt"
        runtime_ids = list(token_ids)
    attention_mask = [1] * observed + [0] * max(target_prompt_tokens - observed, 0)
    return (
        runtime_ids,
        attention_mask,
        {
            "status": status,
            "observed_prompt_tokens": observed,
            "target_prompt_tokens": target_prompt_tokens,
            "runtime_prompt_tokens": len(runtime_ids),
            "delta_tokens": target_prompt_tokens - observed,
            "pad_token_id": pad_token_id if status == "padded_to_target" else None,
        },
    )


def workload_records(
    *,
    tokenizer: Any,
    serving_workloads: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for workload in serving_workloads.get("serving_workloads", []):
        if workload.get("id") not in TARGET_WORKLOAD_IDS:
            continue
        prompt_policy = workload.get("prompt_policy", {})
        decode_policy = workload.get("decode_policy", {})
        prompt_text = str(prompt_policy.get("prompt_text", ""))
        token_ids = [
            int(item)
            for item in tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        batch_sizes = [int(item) for item in decode_policy.get("batch_sizes", [])]
        max_batch = max(batch_sizes) if batch_sizes else 0
        decode_tokens = int(decode_policy.get("decode_tokens", 0))
        target_prompt_tokens = int(prompt_policy.get("target_prompt_tokens", 0))
        runtime_ids, attention_mask, alignment = aligned_prompt_tokens(
            token_ids=token_ids,
            target_prompt_tokens=target_prompt_tokens,
            pad_token_id=tokenizer_pad_token_id(tokenizer),
        )
        repeated_input_ids = runtime_ids * max_batch
        repeated_attention_mask = attention_mask * max_batch
        output_ids = [-1] * max_batch * decode_tokens
        records.append(
            {
                "workload_id": workload.get("id"),
                "prompt_text": prompt_text,
                "target_prompt_tokens": target_prompt_tokens,
                "prompt_token_count": len(token_ids),
                "prompt_token_ids": token_ids,
                "prompt_token_checksum": token_checksum(token_ids),
                "runtime_prompt_token_count": len(runtime_ids),
                "runtime_prompt_token_ids": runtime_ids,
                "target_prompt_alignment": alignment,
                "decode_tokens": decode_tokens,
                "batch_sizes": batch_sizes,
                "input_ids_buffer": buffer_descriptor(
                    name="input_ids",
                    dtype="int32",
                    shape=[max_batch, len(runtime_ids)],
                    values=repeated_input_ids,
                ),
                "attention_mask_buffer": buffer_descriptor(
                    name="attention_mask",
                    dtype="int32",
                    shape=[max_batch, len(attention_mask)],
                    values=repeated_attention_mask,
                ),
                "output_ids_buffer": buffer_descriptor(
                    name="output_ids",
                    dtype="int32",
                    shape=[max_batch, decode_tokens],
                    values=output_ids,
                    initializer=-1,
                ),
                "scalar_bindings": {
                    "prompt_token_count": len(runtime_ids),
                    "decode_tokens": decode_tokens,
                    "max_batch_size": max_batch,
                    "first_decode_position": len(runtime_ids),
                },
                "device_binding_state": "host_materialized_not_cuda_allocated",
            }
        )
    return records


def build_runtime_input_binding(
    *,
    mode: str = "offline",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    prompt_module = load_prompt_module()
    tokenizer, load_status = prompt_module.load_tokenizer(
        mode=mode,
        cache_dir=cache_dir,
    )
    if tokenizer is None:
        return {
            "schema_version": 1,
            "kind": "pto_qwen_runtime_input_binding",
            "status": "tokenizer_unavailable",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_mode": mode,
            "cache_dir": repo_relative(cache_dir),
            "load_status": load_status,
            "workload_records": [],
            "remaining_runtime_gaps": [
                "tokenizer_cache_or_download",
                "cuda_token_buffer_allocation",
                "decode_loop_consumes_token_ids",
            ],
        }

    serving_workloads = load_json(VIEWER_DATA / "serving_workloads.json")
    records = workload_records(
        tokenizer=tokenizer,
        serving_workloads=serving_workloads,
    )
    has_target_mismatch = any(
        item["target_prompt_alignment"]["status"] == "requires_regenerated_prompt"
        for item in records
    )
    gaps = [
        "cuda_token_buffer_allocation",
        "decode_loop_consumes_token_ids",
    ]
    if has_target_mismatch:
        gaps.insert(0, "target_prompt_shape_alignment")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_runtime_input_binding",
        "status": "runtime_input_binding_plan_ready",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_mode": mode,
        "cache_dir": repo_relative(cache_dir),
        "load_status": load_status,
        "tokenizer_class": type(tokenizer).__name__,
        "workload_records": records,
        "implemented_contracts": [
            "tokenizer_to_runtime_input_ids",
            "attention_mask_buffer",
            "runtime_token_buffer_plan",
            "decode_output_buffer_plan",
            "batch_repetition_policy",
        ],
        "remaining_runtime_gaps": gaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_runtime_input_binding(mode=args.mode, cache_dir=args.cache_dir)
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
