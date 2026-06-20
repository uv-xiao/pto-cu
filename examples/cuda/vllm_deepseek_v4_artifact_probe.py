#!/usr/bin/env python3
"""Skip-safe local DeepSeek V4 artifact readiness probe."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = (
    ROOT / "tmp" / "model-artifacts" / "deepseek-ai" / "DeepSeek-V4-Flash"
)
VLLM_SOURCE_ROOT_CANDIDATES = (
    ROOT / "tmp" / "restart-sources" / "repos" / "vllm",
    ROOT / "tmp" / "sources" / "repos" / "external" / "vllm",
)
DEFAULT_VLLM_SOURCE_ROOT = next(
    (path for path in VLLM_SOURCE_ROOT_CANDIDATES if path.exists()),
    VLLM_SOURCE_ROOT_CANDIDATES[0],
)
CONFIG_FIELDS = (
    "architectures",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "quantization_config",
    "torch_dtype",
    "vocab_size",
)
TOKENIZER_CONFIG_FIELDS = (
    "bos_token",
    "eos_token",
    "model_max_length",
    "pad_token",
    "tokenizer_class",
)
TOKENIZER_FILE_CANDIDATES = ("tokenizer.json", "tokenizer.model")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    absolute_path = path if path.is_absolute() else ROOT / path
    try:
        return str(absolute_path.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_sibling_module(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_fields(data: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: data[name] for name in names if name in data}


def _required_file_summary(artifact_dir: Path) -> tuple[dict[str, Any], list[str]]:
    tokenizer_present = [
        name for name in TOKENIZER_FILE_CANDIDATES if (artifact_dir / name).is_file()
    ]
    present = []
    missing = []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        if (artifact_dir / name).is_file():
            present.append(name)
        else:
            missing.append(name)
    if tokenizer_present:
        present.extend(tokenizer_present)
    else:
        missing.append("tokenizer.json or tokenizer.model")

    return {
        "present": sorted(present),
        "missing": sorted(missing),
    }, tokenizer_present


def _read_optional_tokenizer_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    tokenizer_config = _read_json(path)
    selected = _selected_fields(tokenizer_config, TOKENIZER_CONFIG_FIELDS)
    selected["chat_template_present"] = bool(tokenizer_config.get("chat_template"))
    return selected


def _inspect_artifacts(artifact_dir: Path, require_artifacts: bool) -> dict[str, Any]:
    if not artifact_dir.is_dir():
        return {
            "status": "failed" if require_artifacts else "skipped",
            "artifact_dir": _display_path(artifact_dir),
            "reason": "artifact directory is missing",
        }

    required_files, tokenizer_present = _required_file_summary(artifact_dir)
    result: dict[str, Any] = {
        "artifact_dir": _display_path(artifact_dir),
        "required_files": required_files,
        "tokenizer_files_present": tokenizer_present,
        "config_fields": {},
        "tokenizer_config_fields": _read_optional_tokenizer_config(
            artifact_dir / "tokenizer_config.json"
        ),
    }

    config_path = artifact_dir / "config.json"
    if config_path.is_file():
        result["config_fields"] = _selected_fields(
            _read_json(config_path),
            CONFIG_FIELDS,
        )

    index_path = artifact_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "required artifact files are missing",
            }
        )
        return result

    index = _read_json(index_path)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    present_names = [name for name in shard_names if (artifact_dir / name).is_file()]
    present_name_set = set(present_names)
    missing_names = [name for name in shard_names if name not in present_name_set]
    result.update(
        {
            "index_path": _display_path(index_path),
            "indexed_tensors": len(index.get("weight_map", {})),
            "indexed_shards": len(shard_names),
            "present_shards": len(present_names),
            "missing_shards": len(missing_names),
            "present_bytes": sum(
                (artifact_dir / name).stat().st_size for name in present_names
            ),
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "missing_examples": missing_names[:5],
        }
    )

    if required_files["missing"]:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "required artifact files are missing",
            }
        )
    elif missing_names:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "indexed weight shards are missing",
            }
        )
    elif not shard_names:
        result.update(
            {
                "status": "failed" if require_artifacts else "skipped",
                "reason": "weight index does not list shards",
            }
        )
    else:
        result["status"] = "passed"
    return result


def _run_import_probe(source_root: Path) -> dict[str, Any]:
    return _load_sibling_module("vllm_deepseek_v4_import_probe").run_probe(
        source_root=source_root
    )


def _run_config_probe(
    source_root: Path,
    max_position_embeddings: int,
) -> dict[str, Any]:
    return _load_sibling_module("vllm_deepseek_v4_config_probe").run_probe(
        source_root=source_root,
        max_position_embeddings=max_position_embeddings,
    )


def _vllm_status(
    import_probe: Optional[dict[str, Any]],
    config_probe: Optional[dict[str, Any]],
) -> str:
    probes = [probe for probe in (import_probe, config_probe) if probe is not None]
    if any(probe.get("status") == "failed" for probe in probes):
        return "failed"
    if any(probe.get("vllm_import") == "available" for probe in probes):
        return "available"
    if any(probe.get("vllm_import") == "missing" for probe in probes):
        return "missing"
    return "unknown"


def _overall_status(
    artifact_status: str,
    vllm_status: str,
    require_vllm: bool,
) -> tuple[str, list[str]]:
    failures = []
    if artifact_status == "failed":
        failures.append("artifacts are required but incomplete")
    if vllm_status == "failed":
        failures.append("vLLM probe failed")
    if require_vllm and vllm_status != "available":
        failures.append("vLLM is required but not available")
    if failures:
        return "failed", failures
    if artifact_status == "skipped":
        return "skipped", []
    if vllm_status in {"missing", "unknown"}:
        return "skipped", []
    return "passed", []


def run_probe(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_VLLM_SOURCE_ROOT,
    max_position_embeddings: int = 262144,
    require_artifacts: bool = False,
    require_vllm: bool = False,
) -> dict[str, Any]:
    artifact_probe = _inspect_artifacts(artifact_dir, require_artifacts)
    import_probe = _run_import_probe(source_root)
    config_probe = _run_config_probe(source_root, max_position_embeddings)
    vllm_status = _vllm_status(import_probe, config_probe)
    status, failure_reasons = _overall_status(
        artifact_probe["status"],
        vllm_status,
        require_vllm,
    )
    if require_vllm and vllm_status != "available":
        status = "failed"

    result = {
        "status": status,
        "artifact_probe": artifact_probe,
        "vllm_status": vllm_status,
        "vllm_import_probe": import_probe,
        "vllm_config_probe": config_probe,
        "non_claim": "not model-load or serving evidence",
    }
    if failure_reasons:
        result["failure_reasons"] = failure_reasons
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="local DeepSeek-V4-Flash artifact directory under repo-relative tmp/",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_VLLM_SOURCE_ROOT,
        help="downloaded vLLM source root for static source contract checks",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=262144,
        help="synthetic DeepSeek V4 max_position_embeddings value for vLLM config probing",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="return nonzero when the artifact directory is missing or incomplete",
    )
    parser.add_argument(
        "--require-vllm",
        action="store_true",
        help="return nonzero when installed vLLM is missing or config probing fails",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run_probe(
        artifact_dir=args.artifact_dir,
        source_root=args.source_root,
        max_position_embeddings=args.max_position_embeddings,
        require_artifacts=args.require_artifacts,
        require_vllm=args.require_vllm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
