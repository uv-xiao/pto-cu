"""Review-facing CUDA benchmark contracts."""

from __future__ import annotations

SOURCE_PAPERS = (
    {
        "id": "arXiv:2605.03190",
        "label": "VDCores",
        "path": "tmp/sources/arxiv-2605.03190-vdcores.txt",
    },
    {
        "id": "arXiv:2512.22219v1",
        "label": "MPK persistent kernel",
        "path": "tmp/sources/arxiv-2512.22219v1-mirage-persistent-kernel.txt",
    },
)

TENSOR_THROUGHPUT_BASELINES = {
    "direct_driver_sgemm",
    "direct_runtime_sgemm",
    "direct_driver_graph_sgemm",
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_tensor_core",
    "pto_persistent_dag_graph_tensor_core",
    "cublas_sgemm",
    "cublas_sgemm_graph",
}

GRAPH_ROLE_SPELLING_BASELINES = {
    "pto_persistent_dag_graph_tagged_inout",
    "pto_persistent_dag_graph_role_keyed_inout",
    "pto_persistent_dag_graph_compact_role_inout",
    "pto_persistent_dag_graph_pair_inout",
    "pto_persistent_dag_graph_role_map_inout",
}
