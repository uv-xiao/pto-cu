import importlib.machinery
import importlib.util
import json
import shutil
import sys
import types
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


def test_repo_local_artifact_paths_are_reported_repo_relative():
    probe = load_probe_module()
    artifact_dir = ROOT / "tmp" / "unit-model-load-paths"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    write_minimal_artifact(artifact_dir)

    try:
        result = probe.run_probe(artifact_dir=artifact_dir, dry_run=True)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)

    assert result["artifact_probe"]["artifact_dir"] == "tmp/unit-model-load-paths"
    assert result["llm_kwargs"]["model"] == "tmp/unit-model-load-paths"
    assert result["llm_kwargs"]["tokenizer"] == "tmp/unit-model-load-paths"


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


def test_missing_cuda_skips_by_default_and_fails_when_required(monkeypatch, tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_minimal_artifact(artifact_dir)
    monkeypatch.setattr(
        probe,
        "check_vllm",
        lambda: {"status": "passed", "version": "stubbed"},
    )
    monkeypatch.setattr(
        probe,
        "check_cuda",
        lambda tensor_parallel_size: {
            "status": "skipped",
            "reason": "torch.cuda is not available",
            "device_count": 0,
            "required_device_count": tensor_parallel_size,
        },
    )

    skipped = probe.run_probe(artifact_dir=artifact_dir)
    failed = probe.run_probe(artifact_dir=artifact_dir, require_cuda=True)

    assert skipped["status"] == "skipped"
    assert skipped["load_attempted"] is False
    assert skipped["cuda_probe"]["status"] == "skipped"
    assert skipped["failure"]["category"] == "cuda_unavailable"
    assert failed["status"] == "failed"
    assert failed["failure"]["category"] == "cuda_unavailable"


def test_check_cuda_suppresses_noisy_optional_import(monkeypatch, capsys):
    probe = load_probe_module()
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec("torch", loader=None)
        if name == "torch"
        else original_find_spec(name),
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = types.SimpleNamespace(cuda=FakeCuda())

    def fake_import_module(name):
        if name == "torch":
            print("noisy torch import")
            print("noisy torch stderr", file=sys.stderr)
            return fake_torch
        return importlib.import_module(name)

    monkeypatch.setattr(probe.importlib, "import_module", fake_import_module)

    result = probe.check_cuda(tensor_parallel_size=2)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert result["status"] == "skipped"
    assert result["reason"] == "torch.cuda is not available"


def test_missing_vllm_skips_by_default_and_fails_when_required(monkeypatch, tmp_path):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_minimal_artifact(artifact_dir)
    monkeypatch.setattr(
        probe,
        "check_cuda",
        lambda tensor_parallel_size: {
            "status": "passed",
            "device_count": 2,
            "required_device_count": tensor_parallel_size,
        },
    )
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: None if name == "vllm" else original_find_spec(name),
    )

    skipped = probe.run_probe(artifact_dir=artifact_dir)
    failed = probe.run_probe(artifact_dir=artifact_dir, require_vllm=True)

    assert skipped["status"] == "skipped"
    assert skipped["load_attempted"] is False
    assert skipped["vllm_probe"] == {
        "status": "skipped",
        "reason": "vLLM is not installed in the active Python environment.",
    }
    assert failed["status"] == "failed"
    assert failed["failure"]["category"] == "missing_vllm"


def test_model_load_constructs_vllm_llm_with_stubbed_cuda_and_vllm(
    monkeypatch, tmp_path
):
    probe = load_probe_module()
    artifact_dir = tmp_path / "DeepSeek-V4-Flash"
    write_minimal_artifact(artifact_dir)
    received_kwargs = {}
    shutdown_calls = []

    class FakeEngine:
        def shutdown(self):
            shutdown_calls.append("shutdown")

    class FakeLLM:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)
            self.llm_engine = FakeEngine()

    vllm_module = types.ModuleType("vllm")
    vllm_module.__version__ = "stubbed"
    vllm_module.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        probe.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec("vllm", loader=None)
        if name == "vllm"
        else original_find_spec(name),
    )
    monkeypatch.setattr(
        probe,
        "check_cuda",
        lambda tensor_parallel_size: {
            "status": "passed",
            "device_count": 2,
            "required_device_count": tensor_parallel_size,
        },
    )
    monkeypatch.setattr(probe, "_query_nvidia_smi_memory", lambda: [])

    result = probe.run_probe(
        artifact_dir=artifact_dir,
        max_model_len=4096,
        tensor_parallel_size=2,
        require_artifacts=True,
        require_vllm=True,
        require_cuda=True,
    )

    assert result["status"] == "passed"
    assert result["load_attempted"] is True
    assert result["vllm_probe"] == {"status": "passed", "version": "stubbed"}
    assert result["cuda_probe"]["status"] == "passed"
    assert result["loaded_engine_class"] == "FakeLLM"
    assert result["llm_engine_class"] == "FakeEngine"
    assert shutdown_calls == ["shutdown"]
    assert received_kwargs["model"] == str(artifact_dir)
    assert received_kwargs["tokenizer"] == str(artifact_dir)
    assert received_kwargs["max_model_len"] == 4096


def test_failure_parser_classifies_common_model_load_blockers():
    probe = load_probe_module()

    assert (
        probe.classify_failure("torch.OutOfMemoryError: CUDA out of memory")
        == "cuda_out_of_memory"
    )
    assert probe.classify_failure("NCCL error: unhandled cuda error") == "nccl"
    assert probe.classify_failure("No supported config format found") == "model_config"
    assert probe.classify_failure("No such file or directory") == "missing_artifact"
    assert probe.classify_failure("vLLM is not installed") == "missing_vllm"
    assert probe.classify_failure("torch.cuda is not available") == "cuda_unavailable"
    assert probe.classify_failure("some other traceback") == "unknown"
