#!/usr/bin/env python3
"""Weight-free vLLM DeepSeek V4 config readiness probe."""

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
DEFAULT_MAX_POSITION_EMBEDDINGS = 262144
VLLM_DEEPSEEK_V4_DEFAULT_MAX_POSITION_EMBEDDINGS = 1048576


def _source_root_display(source_root):
    if source_root.is_relative_to(ROOT):
        return str(source_root.relative_to(ROOT))
    return str(source_root)


def check_source_contract(source_root, max_position_embeddings):
    config_relative_path = "vllm/transformers_utils/configs/deepseek_v4.py"
    quant_relative_path = "vllm/models/deepseek_v4/quant_config.py"
    config_path = source_root / config_relative_path
    quant_path = source_root / quant_relative_path
    missing = []

    if not config_path.is_file():
        missing.append(f"DeepseekV4Config:missing-file:{config_relative_path}")
    else:
        config_text = config_path.read_text(encoding="utf-8")
        for marker in [
            "class DeepseekV4Config",
            "model_type = \"deepseek_v4\"",
            "max_position_embeddings: int = 1048576",
        ]:
            if marker not in config_text:
                missing.append(f"DeepseekV4Config:missing-marker:{marker}")

    if not quant_path.is_file():
        missing.append(f"DeepseekV4FP8Config:missing-file:{quant_relative_path}")
    else:
        quant_text = quant_path.read_text(encoding="utf-8")
        for marker in [
            "class DeepseekV4FP8Config",
            "return \"deepseek_v4_fp8\"",
            "expert_dtype",
        ]:
            if marker not in quant_text:
                missing.append(f"DeepseekV4FP8Config:missing-marker:{marker}")

    return {
        "source_root": _source_root_display(source_root),
        "source_status": "available" if not missing else "incomplete",
        "source_missing": missing,
        "source_contract": {
            "config_class": "DeepseekV4Config",
            "default_max_position_embeddings": (
                VLLM_DEEPSEEK_V4_DEFAULT_MAX_POSITION_EMBEDDINGS
            ),
            "requested_max_position_embeddings": max_position_embeddings,
            "quantization_method": "deepseek_v4_fp8",
        },
    }


def check_installed_config(max_position_embeddings):
    config_module = importlib.import_module(
        "vllm.transformers_utils.configs.deepseek_v4"
    )
    quant_module = importlib.import_module("vllm.models.deepseek_v4.quant_config")
    config_cls = getattr(config_module, "DeepseekV4Config")
    quant_cls = getattr(quant_module, "DeepseekV4FP8Config")

    config = config_cls(
        max_position_embeddings=max_position_embeddings,
        expert_dtype="fp4",
        quantization_config={"quant_method": "fp8"},
    )
    quant_name = quant_cls.get_name()
    override = quant_cls.override_quantization_method(
        {"quant_method": "fp8"},
        user_quant=None,
        hf_config=config,
    )
    return {
        "config_probe": {
            "config_class": config_cls.__name__,
            "model_type": config.model_type,
            "max_position_embeddings": config.max_position_embeddings,
            "expert_dtype": getattr(config, "expert_dtype", None),
            "quantization_method": quant_name,
            "override_quantization_method": override,
        }
    }


def run_probe(
    source_root=DEFAULT_VLLM_SOURCE_ROOT,
    max_position_embeddings=DEFAULT_MAX_POSITION_EMBEDDINGS,
):
    result = check_source_contract(source_root, max_position_embeddings)
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
    try:
        result.update(check_installed_config(max_position_embeddings))
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "config_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result

    result["status"] = "failed" if result["source_missing"] else "passed"
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
        "--max-position-embeddings",
        type=int,
        default=DEFAULT_MAX_POSITION_EMBEDDINGS,
        help="synthetic DeepSeek V4 max_position_embeddings value",
    )
    parser.add_argument(
        "--require-vllm",
        action="store_true",
        help="return nonzero when vLLM is not importable or config probing fails",
    )
    args = parser.parse_args(argv)

    result = run_probe(
        source_root=args.source_root,
        max_position_embeddings=args.max_position_embeddings,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_vllm and result["status"] != "passed":
        return 2
    if result["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
