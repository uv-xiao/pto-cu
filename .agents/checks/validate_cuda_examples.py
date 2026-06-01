#!/usr/bin/env python3
"""Validate CUDA examples against benchmark-viewer contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "examples" / "cuda"
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def fail(message: str) -> None:
    raise SystemExit(f"CUDA example validation failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_examples(examples_root: Path, root: Path) -> list[dict[str, Any]]:
    manifest = load_json(examples_root / "manifest.json")
    inline_examples = manifest.get("examples")
    if isinstance(inline_examples, list):
        return inline_examples

    example_files = manifest.get("example_files")
    if not isinstance(example_files, list) or not example_files:
        fail("examples/cuda/manifest.json has no examples or example_files")

    examples: list[dict[str, Any]] = []
    for relpath in example_files:
        if not isinstance(relpath, str) or not relpath:
            fail("examples/cuda/manifest.json has an invalid example_files entry")
        shard_path = (examples_root / relpath).resolve()
        try:
            shard_path.relative_to(examples_root.resolve())
        except ValueError:
            fail(f"manifest shard escapes examples/cuda: {relpath}")
        shard = load_json(shard_path)
        shard_examples = shard.get("examples")
        if not isinstance(shard_examples, list) or not shard_examples:
            fail(f"{shard_path.relative_to(root)} has no examples")
        examples.extend(shard_examples)
    return examples


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} missing {key}")
    return value


def normalize_space(text: str) -> str:
    cleaned = text.replace("\\\n", " ")
    cleaned = cleaned.replace("`", "")
    return " ".join(cleaned.split())


def require_in_readme(readme_text: str, value: str, owner: str) -> None:
    if value in readme_text:
        return
    if normalize_space(value) in normalize_space(readme_text):
        return
    fail(f"{owner} README missing: {value}")


def example_source_text(script: Path) -> str:
    parts = [script.read_text(encoding="utf-8")]
    split_dir = script.with_name(f"{script.stem}_impl")
    if split_dir.is_dir():
        parts.extend(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(split_dir.rglob("*.py"))
        )
    return "\n".join(parts)


def validate_examples(root: Path = ROOT) -> None:
    examples_root = root / "examples" / "cuda"
    viewer_data = root / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
    readme = examples_root / "README.md"
    if not readme.is_file():
        fail(f"missing {readme.relative_to(root)}")
    readme_text = readme.read_text(encoding="utf-8")
    split_doc_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((examples_root / "docs").glob("*.md"))
    )
    review_text = f"{readme_text}\n{split_doc_text}"

    benchmarks = load_json(viewer_data / "benchmarks.json")
    methods = load_json(viewer_data / "methods.json")
    benchmark_ids = {item["id"] for item in benchmarks.get("benchmarks", [])}
    method_ids = {item["id"] for item in methods.get("methods", [])}

    examples = load_manifest_examples(examples_root, root)
    if not isinstance(examples, list) or not examples:
        fail("examples/cuda/manifest.json has no examples")

    seen_ids: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            fail("example manifest entry is not an object")
        example_id = require_string(example, "id", "example")
        if not ID_RE.fullmatch(example_id):
            fail(f"example id is not stable snake_case: {example_id}")
        if example_id in seen_ids:
            fail(f"duplicate example id: {example_id}")
        seen_ids.add(example_id)

        title = require_string(example, "title", example_id)
        script = root / require_string(example, "script", example_id)
        if not script.is_file():
            fail(f"{example_id} script missing: {script.relative_to(root)}")
        script_text = example_source_text(script)

        benchmark_id = require_string(example, "benchmark_id", example_id)
        method_id = require_string(example, "method_id", example_id)
        if benchmark_id not in benchmark_ids:
            fail(f"{example_id} unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{example_id} unknown method_id: {method_id}")

        runtime = require_string(example, "runtime", example_id)
        command = require_string(example, "command", example_id)
        expected = require_string(example, "expected_output", example_id)
        for text in (title, benchmark_id, method_id, runtime, command, expected):
            require_in_readme(review_text, text, example_id)

        symbols = example.get("evidence_symbols")
        if not isinstance(symbols, list) or not symbols:
            fail(f"{example_id} has no evidence_symbols")
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol:
                fail(f"{example_id} has empty evidence symbol")
            if symbol not in script_text:
                fail(f"{example_id} script missing evidence symbol: {symbol}")


def main() -> None:
    validate_examples()
    print("CUDA example validation passed")


if __name__ == "__main__":
    main()
