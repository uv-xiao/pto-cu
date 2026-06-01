#!/usr/bin/env python3
"""Emit Qwen prompt-token accounting for PTO CUDA serving policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_CACHE_DIR = ROOT / "tmp" / "hf-tokenizers" / "qwen3-8b-d117af2f"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


class TokenizerLike(Protocol):
    name_or_path: str
    chat_template: str | None
    special_tokens_map: dict[str, Any]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        ...


@dataclass
class MockQwenTokenizer:
    name_or_path: str = "mock://Qwen/Qwen3-8B"
    chat_template: str | None = "mock qwen chat template"
    special_tokens_map: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.special_tokens_map is None:
            self.special_tokens_map = {
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize:
            raise ValueError("mock tokenizer only supports tokenize=True")
        word_count = sum(len(item["content"].split()) for item in conversation)
        overhead = 8 if add_generation_prompt else 5
        return list(range(word_count + overhead))


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


def load_tokenizer(*, mode: str, cache_dir: Path) -> tuple[TokenizerLike | None, str]:
    if mode == "mock":
        return MockQwenTokenizer(), "mock"
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"transformers_import_failed:{type(exc).__name__}:{exc}"

    local_only = mode == "offline"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_only,
            trust_remote_code=False,
        )
    except Exception as exc:
        return None, f"tokenizer_load_failed:{type(exc).__name__}:{exc}"
    return tokenizer, "huggingface_auto_tokenizer"


def workload_prompt_records(tokenizer: TokenizerLike) -> list[dict[str, Any]]:
    payload = load_json(VIEWER_DATA / "serving_workloads.json")
    records: list[dict[str, Any]] = []
    for workload in payload.get("serving_workloads", []):
        if workload.get("id") not in TARGET_WORKLOAD_IDS:
            continue
        prompt_policy = workload.get("prompt_policy", {})
        prompt_text = str(prompt_policy.get("prompt_text", ""))
        raw_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        target_tokens = int(prompt_policy.get("target_prompt_tokens", 0))
        records.append(
            {
                "workload_id": workload.get("id"),
                "prompt_text": prompt_text,
                "target_prompt_tokens": target_tokens,
                "raw_prompt_tokens": len(raw_ids),
                "chat_prompt_tokens": len(chat_ids),
                "target_delta_tokens": target_tokens - len(chat_ids),
                "padding_or_regeneration_required": len(chat_ids) != target_tokens,
                "tokenizer_input_policy": (
                    "chat_template with add_generation_prompt=True"
                ),
            }
        )
    return records


def build_prompt_accounting(
    *,
    mode: str = "offline",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    tokenizer, load_status = load_tokenizer(mode=mode, cache_dir=cache_dir)
    if tokenizer is None:
        return {
            "schema_version": 1,
            "kind": "pto_qwen_prompt_accounting",
            "status": "tokenizer_unavailable",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_mode": mode,
            "cache_dir": repo_relative(cache_dir),
            "load_status": load_status,
            "prompt_records": [],
            "remaining_runtime_gaps": [
                "tokenizer_cache_or_download",
                "runtime_prompt_tensor_binding",
                "decode_loop_consumes_token_ids",
            ],
        }

    records = workload_prompt_records(tokenizer)
    return {
        "schema_version": 1,
        "kind": "pto_qwen_prompt_accounting",
        "status": "pass",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_mode": mode,
        "cache_dir": repo_relative(cache_dir),
        "load_status": load_status,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", ""),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "prompt_records": records,
        "remaining_runtime_gaps": [
            "runtime_prompt_tensor_binding",
            "decode_loop_consumes_token_ids",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--mode",
        choices=["offline", "download", "mock"],
        default="offline",
        help="Tokenizer loading mode. Use download only for explicit capture.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_prompt_accounting(mode=args.mode, cache_dir=args.cache_dir)
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
