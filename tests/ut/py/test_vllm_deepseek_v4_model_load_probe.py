import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROBE_PATH = ROOT / "examples" / "cuda" / "vllm_deepseek_v4_model_load_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "vllm_deepseek_v4_model_load_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_minimal_artifact(root):
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_llm_kwargs_uses_inspected_vllm_argument_names(tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"

    kwargs = probe.build_llm_kwargs(
        artifact_dir=artifact_dir,
        max_model_len=4096,
        tensor_parallel_size=2,
        dtype="bfloat16",
        quantization="deepseek_v4_fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.82,
        enforce_eager=True,
        distributed_executor_backend="mp",
        trust_remote_code=False,
    )

    assert kwargs == {
        "model": str(artifact_dir),
        "tokenizer": str(artifact_dir),
        "tokenizer_mode": "deepseek_v4",
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "quantization": "deepseek_v4_fp8",
        "kv_cache_dtype": "fp8",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.82,
        "enforce_eager": True,
        "distributed_executor_backend": "mp",
        "trust_remote_code": False,
    }


def test_dry_run_emits_structured_plan_without_loading_vllm(tmp_path, capsys):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_minimal_artifact(artifact_dir)

    code = probe.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--dry-run",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.82",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "planned"
    assert payload["artifact_probe"]["status"] == "passed"
    assert payload["llm_kwargs"]["model"] == str(artifact_dir)
    assert payload["load_attempted"] is False
    assert payload["non_claims"] == [
        "not serving evidence",
        "not inference correctness evidence",
        "not tokenizer semantic correctness evidence",
        "not 256K context evidence",
        "not throughput or latency evidence",
    ]


def test_require_artifacts_fails_before_vllm_import_when_shard_missing(tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_minimal_artifact(artifact_dir)
    (artifact_dir / "model-00001-of-00001.safetensors").unlink()

    result = probe.run_probe(
        artifact_dir=artifact_dir,
        dry_run=False,
        require_artifacts=True,
        require_vllm=True,
    )

    assert result["status"] == "failed"
    assert result["load_attempted"] is False
    assert result["artifact_probe"]["status"] == "failed"
    assert result["failure"]["category"] == "missing_artifact"


def test_failure_parser_classifies_common_model_load_blockers():
    probe = load_probe_module()

    assert (
        probe.classify_failure("torch.OutOfMemoryError: CUDA out of memory")
        == "cuda_out_of_memory"
    )
    assert probe.classify_failure("NCCL error: unhandled cuda error") == "nccl"
    assert probe.classify_failure("No supported config format found") == "model_config"
    assert probe.classify_failure("No such file or directory") == "missing_artifact"
    assert probe.classify_failure("some other traceback") == "unknown"
