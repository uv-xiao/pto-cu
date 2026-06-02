#!/usr/bin/env python3
"""Validate CUDA examples against benchmark-viewer contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROOT / "examples" / "cuda"
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
REVIEW_DOC_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\((?P<path>docs/[^)]+\.md)\) "
    r"\((?P<lines>[0-9]+) lines\)$"
)


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


def load_review_docs(
    examples_root: Path,
    readme_text: str,
) -> dict[str, tuple[Path, str]]:
    review_docs: dict[str, tuple[Path, str]] = {}
    for line in readme_text.splitlines():
        match = REVIEW_DOC_RE.fullmatch(line)
        if not match:
            continue
        title = match.group("title")
        relpath = match.group("path")
        doc_path = examples_root / relpath
        if title in review_docs:
            fail(f"duplicate CUDA example review doc title: {title}")
        if not doc_path.is_file():
            fail(f"missing CUDA example review doc: {relpath}")
        text = doc_path.read_text(encoding="utf-8")
        actual_lines = len(text.splitlines())
        expected_lines = int(match.group("lines"))
        if actual_lines != expected_lines:
            fail(
                f"{relpath} line count is {actual_lines}, "
                f"README lists {expected_lines}"
            )
        if actual_lines > 120:
            fail(f"{relpath} is too long for focused example review")
        review_docs[title] = (doc_path, text)
    if not review_docs:
        fail("examples/cuda/README.md has no review map docs")
    return review_docs


def validate_review_doc(
    *,
    example_id: str,
    title: str,
    script: Path,
    benchmark_id: str,
    method_id: str,
    runtime: str,
    command: str,
    expected: str,
    review_docs: dict[str, tuple[Path, str]],
) -> Path:
    doc = review_docs.get(title)
    if doc is None:
        fail(f"{example_id} has no focused review doc")
    doc_path, doc_text = doc
    for value in (
        f"# CUDA Examples: {title}",
        f"## {title}",
        f"Benchmark id: `{benchmark_id}`",
        f"Runtime: `{runtime}`",
        f"Method id: `{method_id}`",
        script.name,
    ):
        if value not in doc_text:
            fail(f"{doc_path.relative_to(ROOT)} missing: {value}")
    require_in_readme(doc_text, command, example_id)
    require_in_readme(doc_text, expected, example_id)
    return doc_path


def validate_examples(root: Path = ROOT) -> None:
    examples_root = root / "examples" / "cuda"
    viewer_data = root / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
    readme = examples_root / "README.md"
    if not readme.is_file():
        fail(f"missing {readme.relative_to(root)}")
    readme_text = readme.read_text(encoding="utf-8")
    split_doc_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((examples_root / "docs").glob("*.md"))
    )
    review_text = f"{readme_text}\n{split_doc_text}"
    review_docs = load_review_docs(examples_root, readme_text)

    benchmarks = load_json(viewer_data / "benchmarks.json")
    methods = load_json(viewer_data / "methods.json")
    benchmark_ids = {item["id"] for item in benchmarks.get("benchmarks", [])}
    method_ids = {item["id"] for item in methods.get("methods", [])}

    examples = load_manifest_examples(examples_root, root)
    if not isinstance(examples, list) or not examples:
        fail("examples/cuda/manifest.json has no examples")

    seen_ids: set[str] = set()
    seen_review_docs: set[Path] = set()
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
        review_doc = validate_review_doc(
            example_id=example_id,
            title=title,
            script=script,
            benchmark_id=benchmark_id,
            method_id=method_id,
            runtime=runtime,
            command=command,
            expected=expected,
            review_docs=review_docs,
        )
        if review_doc in seen_review_docs:
            fail(f"multiple examples use {review_doc.relative_to(root)}")
        seen_review_docs.add(review_doc)

        symbols = example.get("evidence_symbols")
        if not isinstance(symbols, list) or not symbols:
            fail(f"{example_id} has no evidence_symbols")
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol:
                fail(f"{example_id} has empty evidence symbol")
            if symbol not in script_text:
                fail(f"{example_id} script missing evidence symbol: {symbol}")
    review_doc_paths = {path for path, _ in review_docs.values()}
    if seen_review_docs != review_doc_paths:
        missing = sorted(
            path.relative_to(root).as_posix()
            for path in review_doc_paths - seen_review_docs
        )
        fail(f"CUDA review docs are not tied to manifest examples: {missing}")


def main() -> None:
    validate_examples()
    print("CUDA example validation passed")


if __name__ == "__main__":
    main()
