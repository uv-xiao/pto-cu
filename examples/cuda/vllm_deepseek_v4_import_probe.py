#!/usr/bin/env python3
"""Weight-free vLLM DeepSeek V4 import readiness probe."""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VLLM_SOURCE_ROOT_CANDIDATES = (
    ROOT / "tmp" / "restart-sources" / "repos" / "vllm",
    ROOT / "tmp" / "sources" / "repos" / "external" / "vllm",
)
DEFAULT_VLLM_SOURCE_ROOT = next(
    (path for path in VLLM_SOURCE_ROOT_CANDIDATES if path.exists()),
    VLLM_SOURCE_ROOT_CANDIDATES[0],
)

SOURCE_SYMBOLS = {
    "DeepseekV4ForCausalLM": (
        "vllm/models/deepseek_v4/nvidia/model.py",
        ["class DeepseekV4ForCausalLM"],
    ),
    "DeepseekV4Tokenizer": (
        "vllm/tokenizers/deepseek_v4.py",
        ["class DeepseekV4Tokenizer"],
    ),
    "DeepseekV4Config": (
        "vllm/transformers_utils/configs/deepseek_v4.py",
        ["class DeepseekV4Config"],
    ),
    "DeepseekV4FP8Config": (
        "vllm/models/deepseek_v4/quant_config.py",
        ["class DeepseekV4FP8Config", "deepseek_v4_fp8"],
    ),
}

IMPORT_TARGETS = {
    "DeepseekV4ForCausalLM": [
        "vllm.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
    ],
    "DeepseekV4Tokenizer": [
        "vllm.tokenizers.deepseek_v4",
        "DeepseekV4Tokenizer",
    ],
    "DeepseekV4Config": [
        "vllm.transformers_utils.configs.deepseek_v4",
        "DeepseekV4Config",
    ],
    "DeepseekV4FP8Config": [
        "vllm.models.deepseek_v4.quant_config",
        "DeepseekV4FP8Config",
    ],
}


def _display_path(path):
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def check_source_symbols(source_root):
    symbols = []
    missing = []
    for symbol, (relative_path, markers) in SOURCE_SYMBOLS.items():
        path = source_root / relative_path
        if not path.is_file():
            missing.append(f"{symbol}:missing-file:{relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                missing.append(f"{symbol}:missing-marker:{marker}")
        symbols.append(symbol)
    return {
        "source_root": _display_path(source_root),
        "source_status": "available" if not missing else "incomplete",
        "source_symbols": sorted(symbols),
        "source_missing": missing,
    }


def check_installed_imports():
    imported = []
    errors = []
    for symbol, (module_name, attr_name) in IMPORT_TARGETS.items():
        try:
            module = importlib.import_module(module_name)
            getattr(module, attr_name)
        except Exception as exc:
            errors.append(f"{symbol}:{type(exc).__name__}")
            continue
        imported.append(symbol)
    return {
        "imported_symbols": sorted(imported),
        "import_errors": errors,
    }


def run_probe(source_root=DEFAULT_VLLM_SOURCE_ROOT):
    result = check_source_symbols(source_root)
    if importlib.util.find_spec("vllm") is None:
        result.update(
            {
                "status": "skipped",
                "vllm_import": "missing",
                "reason": "vLLM is not installed in the active Python environment.",
            }
        )
        return result

    result["vllm_import"] = "available"
    result.update(check_installed_imports())
    if result["source_missing"] or result["import_errors"]:
        result["status"] = "failed"
    else:
        result["status"] = "passed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_VLLM_SOURCE_ROOT,
        help="downloaded vLLM source root for static source contract checks",
    )
    parser.add_argument(
        "--require-vllm",
        action="store_true",
        help="return nonzero when vLLM is not importable or imports fail",
    )
    args = parser.parse_args(argv)

    result = run_probe(source_root=args.source_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_vllm and result["status"] != "passed":
        return 2
    if result["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
