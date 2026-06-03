# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# See LICENSE in the root of the software repository for the full text.
# ---------------------------------------------------------------------------

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts"


def load_claim_status_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        return importlib.import_module("paper_readiness_audit_impl.claim_status")
    finally:
        sys.modules.pop("paper_readiness_audit_impl.claim_status", None)
        sys.path.remove(str(SCRIPT_DIR))


def test_paper_readiness_viewer_result_refs_match_serving_coverage():
    claim_status = load_claim_status_module()
    current_results = claim_status.result_index(
        {
            "result_records": [
                {
                    "benchmark_id": "llm_serving_decode",
                    "method_id": "pto_persistent_device",
                    "hardware": {"gpu": "A100"},
                    "inputs": {
                        "shape": "Qwen/Qwen3-8B resource-backed diagnostic"
                    },
                    "statistic": {
                        "serving_coverage": "diagnostic_resource_backed_qwen_dag"
                    },
                }
            ]
        }
    )

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving"
    ]


def test_paper_readiness_viewer_result_refs_accept_matching_coverage():
    claim_status = load_claim_status_module()
    current_results = claim_status.result_index(
        {
            "result_records": [
                {
                    "benchmark_id": "llm_serving_decode",
                    "method_id": "vllm",
                    "hardware": {"gpu": "H200"},
                    "inputs": {"shape": "mpk_offline_decode,Qwen/Qwen3-8B"},
                    "statistic": {"serving_coverage": "full_serving"},
                }
            ]
        }
    )

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "vllm",
                "gpu": "H200",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == []


def test_paper_readiness_rejects_weak_pto_full_serving_row():
    claim_status = load_claim_status_module()
    current_results = claim_status.result_index(
        {
            "result_records": [
                {
                    "benchmark_id": "llm_serving_decode",
                    "method_id": "pto_persistent_device",
                    "hardware": {"gpu": "A100"},
                    "inputs": {
                        "shape": "mpk_offline_decode,Qwen/Qwen3-8B"
                    },
                    "statistic": {"serving_coverage": "full_serving"},
                    "correctness": "pass",
                    "raw_artifact": "tmp/cuda-backend/weak-pto-row/",
                }
            ]
        }
    )

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving"
    ]


def test_paper_readiness_accepts_complete_pto_full_serving_row():
    claim_status = load_claim_status_module()
    current_results = claim_status.result_index(
        {
            "result_records": [
                {
                    "benchmark_id": "llm_serving_decode",
                    "method_id": "pto_persistent_device",
                    "hardware": {"gpu": "A100"},
                    "inputs": {
                        "shape": (
                            "vdcores_offline_decode,Qwen/Qwen3-8B,"
                            "batch=8,prompt_tokens=128,decode_tokens=64"
                        )
                    },
                    "statistic": {
                        "serving_coverage": "full_serving",
                        "workload_id": "vdcores_offline_decode",
                        "sample_count": 3,
                        "host_wall_ns": 100,
                        "device_wall_ns": 90,
                        "end_to_end_latency_ns": 100,
                        "inter_token_latency_ns": 2,
                        "time_to_first_token_ns": 1,
                        "throughput_tokens_per_s": 1000.0,
                        "batch_size": 8,
                        "prompt_tokens": 128,
                        "decode_tokens": 64,
                        "completed_requests": 8,
                        "failed_requests": 0,
                        "total_input_tokens": 1024,
                        "total_output_tokens": 512,
                        "correctness_scope": "full_qwen_numerical_correctness",
                        "comparison_scope": "model_equivalent_decode",
                        "model_equivalent_ready": True,
                        "checked_token_count": 512,
                        "max_abs_error": 0.0001,
                        "correctness_tolerance": 0.001,
                    },
                    "correctness": "pass",
                    "correctness_details": {
                        "scope": "full_qwen_numerical_correctness",
                        "model_id": "Qwen/Qwen3-8B",
                        "status": "pass",
                        "token_match": True,
                        "model_equivalent_ready": True,
                        "comparison_scope": "model_equivalent_decode",
                        "checked_token_count": 512,
                        "max_abs_error": 0.0001,
                        "tolerance": 0.001,
                    },
                    "raw_artifact": "tmp/cuda-backend/complete-pto-row/",
                }
            ]
        }
    )

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == []


def test_paper_readiness_rejects_diagnostic_comparison_scope():
    claim_status = load_claim_status_module()
    row = pto_full_serving_row("mpk_offline_decode")
    row["correctness_details"]["comparison_scope"] = (
        "diagnostic_decode_without_prompt_prefill"
    )
    row["correctness_details"]["model_equivalent_ready"] = False
    current_results = claim_status.result_index({"result_records": [row]})

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving"
    ]


def pto_full_serving_row(workload_id):
    return {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "hardware": {"gpu": "A100"},
        "inputs": {
            "shape": (
                f"{workload_id},Qwen/Qwen3-8B,batch=8,"
                "prompt_tokens=128,decode_tokens=64"
            )
        },
        "statistic": {
            "serving_coverage": "full_serving",
            "workload_id": workload_id,
            "sample_count": 3,
            "host_wall_ns": 100,
            "device_wall_ns": 90,
            "end_to_end_latency_ns": 100,
            "inter_token_latency_ns": 2,
            "time_to_first_token_ns": 1,
            "throughput_tokens_per_s": 1000.0,
            "batch_size": 8,
            "prompt_tokens": 128,
            "decode_tokens": 64,
            "completed_requests": 8,
            "failed_requests": 0,
            "total_input_tokens": 1024,
            "total_output_tokens": 512,
            "correctness_scope": "full_qwen_numerical_correctness",
            "comparison_scope": "model_equivalent_decode",
            "model_equivalent_ready": True,
            "checked_token_count": 512,
            "max_abs_error": 0.0001,
            "correctness_tolerance": 0.001,
        },
        "correctness": "pass",
        "correctness_details": {
            "scope": "full_qwen_numerical_correctness",
            "model_id": "Qwen/Qwen3-8B",
            "status": "pass",
            "token_match": True,
            "model_equivalent_ready": True,
            "comparison_scope": "model_equivalent_decode",
            "checked_token_count": 512,
            "max_abs_error": 0.0001,
            "tolerance": 0.001,
        },
        "raw_artifact": f"tmp/cuda-backend/complete-pto-row/{workload_id}/",
    }


def test_paper_readiness_rejects_failed_pto_full_serving_requests():
    claim_status = load_claim_status_module()
    row = pto_full_serving_row("mpk_offline_decode")
    row["statistic"]["failed_requests"] = 1
    current_results = claim_status.result_index({"result_records": [row]})

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving"
    ]


def test_paper_readiness_rejects_underchecked_pto_decode_tokens():
    claim_status = load_claim_status_module()
    row = pto_full_serving_row("vdcores_offline_decode")
    row["statistic"]["checked_token_count"] = 511
    row["correctness_details"]["checked_token_count"] = 511
    current_results = claim_status.result_index({"result_records": [row]})

    _counts, missing = claim_status.count_evidence_refs(
        [
            {
                "kind": "viewer_result",
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "gpu": "A100",
                "shape_contains": "Qwen/Qwen3-8B",
                "serving_coverage": "full_serving",
            }
        ],
        current_results,
    )

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving"
    ]


def test_paper_readiness_requires_both_pto_full_serving_policies():
    claim_status = load_claim_status_module()
    current_results = claim_status.result_index(
        {"result_records": [pto_full_serving_row("mpk_offline_decode")]}
    )

    ref = {
        "kind": "viewer_result",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "gpu": "A100",
        "shape_contains": "Qwen/Qwen3-8B",
        "serving_coverage": "full_serving",
        "required_workload_ids": [
            "mpk_offline_decode",
            "vdcores_offline_decode",
        ],
    }
    _counts, missing = claim_status.count_evidence_refs([ref], current_results)

    assert missing == [
        "llm_serving_decode / pto_persistent_device / A100 / "
        "shape contains Qwen/Qwen3-8B / coverage full_serving / "
        "workloads mpk_offline_decode,vdcores_offline_decode"
    ]

    current_results = claim_status.result_index(
        {
            "result_records": [
                pto_full_serving_row("mpk_offline_decode"),
                pto_full_serving_row("vdcores_offline_decode"),
            ]
        }
    )
    _counts, missing = claim_status.count_evidence_refs([ref], current_results)

    assert missing == []
