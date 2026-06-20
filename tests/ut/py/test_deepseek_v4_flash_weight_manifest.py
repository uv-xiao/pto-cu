import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples" / "cuda" / "deepseek_v4_flash_weight_manifest.py"


def test_manifest_reports_missing_weight_shards(tmp_path):
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    (artifact_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
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

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            "--metadata",
            str(metadata_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "incomplete"
    assert payload["model_id"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert payload["indexed_shards"] == 2
    assert payload["present_shards"] == 1
    assert payload["missing_shards"] == 1
    assert payload["present_bytes"] == 5
    assert payload["index_total_size"] == 123
    assert payload["metadata_used_storage"] == 456
    assert payload["metadata_safetensors_total"] == 111
    assert payload["non_claim"] == "not serving evidence"
    assert "can_serve" not in payload


def test_manifest_require_complete_fails_when_shards_are_missing(tmp_path):
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            "--require-complete",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 2, result.stdout
    assert json.loads(result.stdout)["status"] == "incomplete"


def test_manifest_require_complete_fails_when_artifact_dir_is_missing(tmp_path):
    artifact_dir = tmp_path / "missing-model"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            "--require-complete",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "missing"
    assert payload["reason"] == "artifact directory is missing"
    assert payload["indexed_shards"] == 0
    assert payload["present_shards"] == 0
    assert payload["missing_shards"] == 0


def test_manifest_reports_complete_weight_shards_without_serving_claim(tmp_path):
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"12345")
    (artifact_dir / "model-00002-of-00002.safetensors").write_bytes(b"1234567")
    (artifact_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-dir",
            str(artifact_dir),
            "--metadata",
            str(tmp_path / "missing-metadata.json"),
            "--require-complete",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["indexed_shards"] == 2
    assert payload["present_shards"] == 2
    assert payload["missing_shards"] == 0
    assert payload["present_bytes"] == 12
    assert payload["missing_examples"] == []
    assert payload["model_id"] == "unknown"
    assert payload["metadata_used_storage"] is None
    assert payload["metadata_safetensors_total"] is None
    assert payload["non_claim"] == "not serving evidence"
    assert "can_serve" not in payload


def test_manifest_keeps_symlinked_artifact_paths_repo_relative(tmp_path):
    target_dir = tmp_path / "target-model"
    target_dir.mkdir()
    (target_dir / "model-00001-of-00001.safetensors").write_bytes(b"12345")
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 5},
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    link_parent = ROOT / "tmp" / "unit-deepseek-manifest"
    link_path = link_parent / tmp_path.name
    shutil.rmtree(link_parent, ignore_errors=True)
    link_parent.mkdir(parents=True)
    link_path.symlink_to(target_dir, target_is_directory=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--artifact-dir",
                str(link_path),
                "--metadata",
                str(tmp_path / "missing-metadata.json"),
                "--require-complete",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        shutil.rmtree(link_parent, ignore_errors=True)

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["artifact_dir"].startswith("tmp/unit-deepseek-manifest/")
    assert payload["index_path"].startswith("tmp/unit-deepseek-manifest/")
