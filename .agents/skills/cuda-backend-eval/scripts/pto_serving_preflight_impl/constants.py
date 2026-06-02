"""Shared constants for PTO full-serving preflight capture."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
VIEWER_DATA = ROOT / "evaluations" / "nvidia" / "benchmark-viewer" / "data"
DEFAULT_OUTPUT = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-preflight"
    / "pto-serving-preflight.json"
)
SERVING_SCAFFOLD = ROOT / "examples" / "cuda" / "persistent_qwen_serving_scaffold.py"
PAPER_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
FULL_SERVING_METRIC_FIELDS = {
    "batch_size",
    "decode_tokens",
    "end_to_end_latency_ns",
    "inter_token_latency_ns",
    "throughput_tokens_per_s",
    "time_to_first_token_ns",
}
