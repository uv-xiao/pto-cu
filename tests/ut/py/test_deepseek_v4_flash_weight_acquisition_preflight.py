import json
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "examples"
    / "cuda"
    / "deepseek_v4_flash_weight_acquisition_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("weight_acquisition_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_index(artifact_dir, total_size, weight_map):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _run_preflight(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_acquisition_preflight_reports_remaining_capacity_and_non_claims(tmp_path):
    artifact_dir = tmp_path / "model"
    (artifact_dir / "model-00001-of-00002.safetensors").parent.mkdir()
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    _write_index(
        artifact_dir,
        12,
        {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        },
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "deepseek-ai/DeepSeek-V4-Flash",
                "usedStorage": 456,
                "safetensors": {"total": 111},
            }
        ),
        encoding="utf-8",
    )

    result = _run_preflight(
        "--artifact-dir",
        str(artifact_dir),
        "--metadata",
        str(metadata_path),
        "--download-root",
        str(tmp_path),
        "--filesystem-free-bytes",
        "20",
        "--capacity-multiplier",
        "1.5",
        "--reserve-bytes",
        "3",
        "--require-capacity",
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert payload["artifact_dir"] == "<external>/model"
    assert payload["download_root"] == "<external>"
    assert payload["indexed_shard_count"] == 2
    assert payload["present_shard_count"] == 1
    assert payload["missing_shard_count"] == 1
    assert payload["indexed_bytes"] == 12
    assert payload["present_bytes"] == 5
    assert payload["metadata_storage_bytes"] == 456
    assert payload["metadata_safetensors_total_bytes"] == 111
    assert payload["estimated_required_bytes_remaining"] == 7
    assert payload["filesystem_free_bytes"] == 20
    assert payload["capacity_multiplier"] == 1.5
    assert payload["reserve_bytes"] == 3
    assert payload["required_capacity_bytes"] == 14
    assert payload["has_required_capacity"] is True
    assert payload["can_attempt_download"] is True
    assert payload["can_attempt_model_load"] is False
    assert payload["missing_shard_examples"] == ["model-00002-of-00002.safetensors"]
    assert payload["non_claims"] == [
        "not serving evidence",
        "not model-load evidence",
        "not DeepSeek correctness evidence",
    ]
    assert str(tmp_path) not in result.stdout


def test_acquisition_preflight_require_capacity_fails_when_space_is_short(tmp_path):
    artifact_dir = tmp_path / "model"
    (artifact_dir / "model-00001-of-00002.safetensors").parent.mkdir()
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    _write_index(
        artifact_dir,
        12,
        {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        },
    )

    result = _run_preflight(
        "--artifact-dir",
        str(artifact_dir),
        "--download-root",
        str(tmp_path),
        "--filesystem-free-bytes",
        "6",
        "--require-capacity",
    )

    assert result.returncode == 3, result.stdout
    payload = json.loads(result.stdout)
    assert payload["estimated_required_bytes_remaining"] == 7
    assert payload["required_capacity_bytes"] == 7
    assert payload["has_required_capacity"] is False
    assert payload["can_attempt_download"] is False
    assert payload["can_attempt_model_load"] is False
    assert payload["preflight_status"] == "blocked_storage_capacity"


def test_acquisition_preflight_allows_model_load_only_when_manifest_is_complete(
    tmp_path,
):
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    (artifact_dir / "model-00002-of-00002.safetensors").write_bytes(b"1234567")
    _write_index(
        artifact_dir,
        12,
        {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        },
    )

    result = _run_preflight(
        "--artifact-dir",
        str(artifact_dir),
        "--download-root",
        str(tmp_path),
        "--filesystem-free-bytes",
        "0",
        "--require-capacity",
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["manifest_status"] == "complete"
    assert payload["missing_shard_count"] == 0
    assert payload["estimated_required_bytes_remaining"] == 0
    assert payload["required_capacity_bytes"] == 0
    assert payload["has_required_capacity"] is True
    assert payload["can_attempt_download"] is False
    assert payload["can_attempt_model_load"] is True
    assert payload["preflight_status"] == "ready_for_model_load"


def test_acquisition_preflight_can_estimate_capacity_from_fetched_metadata(tmp_path):
    module = _load_module()

    payload = module.build_preflight(
        tmp_path / "missing-model",
        tmp_path / "missing-metadata.json",
        filesystem_free_bytes=6,
        fetch_hf_metadata=True,
        metadata_fetcher=lambda model_id: {
            "id": model_id,
            "usedStorage": 12,
            "safetensors": {"total": 11},
        },
    )

    assert payload["metadata_source"] == "hf_api"
    assert payload["metadata_storage_bytes"] == 12
    assert payload["metadata_safetensors_total_bytes"] == 11
    assert payload["estimated_required_bytes_remaining"] == 12
    assert payload["required_capacity_bytes"] == 12
    assert payload["has_required_capacity"] is False
    assert payload["can_attempt_download"] is False
    assert payload["can_attempt_model_load"] is False
    assert payload["preflight_status"] == "blocked_storage_capacity"


def test_acquisition_preflight_reports_metadata_fetch_failure_as_json(tmp_path):
    module = _load_module()

    payload = module.build_preflight(
        tmp_path / "missing-model",
        tmp_path / "missing-metadata.json",
        fetch_hf_metadata=True,
        metadata_fetcher=lambda model_id: (_ for _ in ()).throw(
            TimeoutError("api timeout")
        ),
    )

    assert payload["metadata_source"] == "hf_api_error"
    assert payload["metadata_error"] == "TimeoutError: api timeout"
    assert payload["metadata_storage_bytes"] is None
    assert payload["estimated_required_bytes_remaining"] is None
    assert payload["has_required_capacity"] is None
    assert payload["preflight_status"] == "blocked_missing_artifact_dir"
