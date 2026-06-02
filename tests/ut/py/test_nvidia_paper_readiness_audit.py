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
