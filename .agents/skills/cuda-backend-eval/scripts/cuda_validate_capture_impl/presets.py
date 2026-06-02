"""Preset data for CUDA benchmark capture validation."""

PAIRED_CURRENT_MACHINES = ("hina", "dasys-h200x8")
PAIRED_CURRENT_BASELINES = (
    "cublas_sgemm",
    "cublas_sgemm_graph",
    "direct_driver",
    "direct_driver_sgemm",
    "direct_driver_graph",
    "direct_driver_graph_sgemm",
    "direct_runtime",
    "direct_runtime_sgemm",
    "pto_host_schedule",
    "pto_host_schedule_batch",
    "pto_host_schedule_compiler",
    "pto_host_schedule_generic_args",
    "pto_host_schedule_quad",
    "pto_host_schedule_unary_square",
    "pto_persistent_dag",
    "pto_persistent_dag_chain",
    "pto_persistent_dag_reuse",
    "pto_persistent_dag_scalar_affine",
    "pto_persistent_dag_scalar_axpy",
    "pto_persistent_dag_scalar_scale",
    "pto_persistent_dag_tensor",
    "pto_persistent_dag_tensor_core",
    "pto_persistent_dag_triad",
    "pto_persistent_dag_quad",
    "pto_persistent_dag_generic_args",
    "pto_persistent_dag_graph",
    "pto_persistent_dag_graph_generic_args4",
    "pto_persistent_dag_graph_node_attrs",
    "pto_persistent_dag_graph_node_io",
    "pto_persistent_dag_graph_node_link",
    "pto_persistent_dag_graph_named_callable",
    "pto_persistent_dag_graph_node_op",
    "pto_persistent_dag_graph_depends_on",
    "pto_persistent_dag_graph_scalar_axpy",
    "pto_persistent_dag_graph_scalar_scale",
    "pto_persistent_dag_graph_scalar_affine",
    "pto_persistent_dag_graph_reordered",
    "pto_persistent_dag_graph_chain",
    "pto_persistent_dag_graph_scratch_reuse",
    "pto_persistent_dag_graph_diamond",
    "pto_persistent_dag_graph_parallel_chains",
    "pto_persistent_dag_graph_wide_fanout",
    "pto_persistent_dag_graph_multi_fanin",
    "pto_persistent_dag_graph_layered_cross",
    "pto_persistent_dag_graph_tagged",
    "pto_persistent_dag_graph_tagged_inout",
    "pto_persistent_dag_graph_role_keyed_inout",
    "pto_persistent_dag_graph_compact_role_inout",
    "pto_persistent_dag_graph_pair_inout",
    "pto_persistent_dag_graph_role_map_inout",
    "pto_persistent_dag_graph_submit_groups",
    "pto_persistent_dag_graph_triad",
    "pto_persistent_dag_graph_quad",
    "pto_persistent_dag_graph_unary_square",
    "pto_persistent_dag_graph_tensor",
    "pto_persistent_dag_graph_tensor_core",
    "pto_persistent_dag_unary_square",
    "pto_persistent_device",
    "pto_persistent_device_batch",
    "pto_persistent_device_grid_batch",
    "pto_persistent_queue",
    "pto_persistent_queue_batch",
)
COMPACT_CURRENT_BASELINES = tuple(
    baseline
    for baseline in PAIRED_CURRENT_BASELINES
    if baseline
    not in {
        "pto_host_schedule_batch",
        "pto_persistent_device_batch",
        "pto_persistent_device_grid_batch",
        "pto_persistent_queue_batch",
    }
)
PAIRED_CURRENT_SIZES = (1024, 65536, 1048576)
COMPACT_CURRENT_SIZES = (1024,)
COMPACT_CURRENT_EXPECTED_REPEATS = 1
COMPACT_CURRENT_EXPECTED_RESULT_COUNT = 116
PAIRED_CURRENT_EXPECTED_RESULT_COUNT = 1422
REQUIRED_SOURCE_PAPER_IDS = ("arXiv:2605.03190", "arXiv:2512.22219v1")
REPORT_FILES = (
    "cuda-benchmark.md",
    "cuda-benchmark.svg",
    "cuda-benchmark-ratios.svg",
    "cuda-benchmark-dag-deltas.svg",
    "cuda-benchmark-throughput.svg",
)
PAIRED_CURRENT_DISPATCH = {
    "pto_persistent_dag": "1,2,1",
    "pto_persistent_dag_chain": "1,2,1,2,1",
    "pto_persistent_dag_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_scalar_axpy": "4,2,1",
    "pto_persistent_dag_scalar_scale": "11,2,1",
    "pto_persistent_dag_scalar_affine": "5,2,1",
    "pto_persistent_dag_triad": "6,2,1",
    "pto_persistent_dag_quad": "8,2,1",
    "pto_persistent_dag_generic_args": "9,2,1",
    "pto_persistent_dag_graph": "9,2,1",
    "pto_persistent_dag_graph_generic_args4": "9,2,1",
    "pto_persistent_dag_graph_node_attrs": "9,2,1",
    "pto_persistent_dag_graph_node_io": "1,2,1",
    "pto_persistent_dag_graph_node_link": "1,2,1",
    "pto_persistent_dag_graph_named_callable": "1,2,1",
    "pto_persistent_dag_graph_node_op": "1,2,1",
    "pto_persistent_dag_graph_depends_on": "1,2,1",
    "pto_persistent_dag_graph_scalar_axpy": "4,2,1",
    "pto_persistent_dag_graph_scalar_scale": "11,2,1",
    "pto_persistent_dag_graph_scalar_affine": "5,2,1",
    "pto_persistent_dag_graph_reordered": "1,9,2",
    "pto_persistent_dag_graph_chain": "1,2,1,2,1",
    "pto_persistent_dag_graph_scratch_reuse": "1,2,1,2,1,1",
    "pto_persistent_dag_graph_diamond": "9,2,1,2,1",
    "pto_persistent_dag_graph_parallel_chains": "1,2,1,2,1,1,2,1,1",
    "pto_persistent_dag_graph_wide_fanout": "1,1,2,1,1,2,1",
    "pto_persistent_dag_graph_multi_fanin": "1,2,11,6",
    "pto_persistent_dag_graph_layered_cross": "1,2,11,1,2,1,6,1,1",
    "pto_persistent_dag_graph_tagged": "9,2,1",
    "pto_persistent_dag_graph_tagged_inout": "1,1,1",
    "pto_persistent_dag_graph_role_keyed_inout": "1,1,1",
    "pto_persistent_dag_graph_compact_role_inout": "1,1,1",
    "pto_persistent_dag_graph_pair_inout": "1,1,1",
    "pto_persistent_dag_graph_role_map_inout": "1,1,1",
    "pto_persistent_dag_graph_submit_groups": "1,1,1",
    "pto_persistent_dag_graph_triad": "6,2,1",
    "pto_persistent_dag_graph_quad": "8,2,1",
    "pto_persistent_dag_graph_unary_square": "7,1,1",
    "pto_persistent_dag_unary_square": "7,1,1",
    "pto_persistent_dag_tensor": "3,1,2,1",
    "pto_persistent_dag_graph_tensor": "3,1,2,1",
    "pto_persistent_dag_tensor_core": "10,1,2,1",
    "pto_persistent_dag_graph_tensor_core": "10,1,2,1",
}
PAIRED_CURRENT_TENSOR_TILES = {
    "direct_driver_sgemm": "16x16x16",
    "direct_runtime_sgemm": "16x16x16",
    "direct_driver_graph_sgemm": "16x16x16",
    "pto_persistent_dag_tensor": "16x16x16",
    "pto_persistent_dag_graph_tensor": "16x16x16",
    "pto_persistent_dag_tensor_core": "16x16x16",
    "pto_persistent_dag_graph_tensor_core": "16x16x16",
    "cublas_sgemm": "16x16x16",
    "cublas_sgemm_graph": "16x16x16",
}
PAIRED_CURRENT_SCRATCH_REUSE = {
    "pto_persistent_dag_graph_scratch_reuse": "reused_buffer=tmp0,reuse_task=4",
}
PAIRED_CURRENT_GRAPH_TASK_ARGS = {
    "pto_persistent_dag_graph_tagged": (
        "task0=input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1];"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ),
    "pto_persistent_dag_graph_tagged_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_role_keyed_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_compact_role_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_pair_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_role_map_inout": (
        "task0=input:a,input:b,output:tmp1;task1=inout:tmp1,input:b;"
        "task2=input:tmp1,input:a,output_existing:out"
    ),
    "pto_persistent_dag_graph_submit_groups": (
        "task0=input:a,input:b,output:tmp1;"
        "task1=input:a,input:b,output:tmp2;task2=input:tmp1,input:tmp2,output_existing:out"
    ),
    "pto_persistent_dag_graph_node_io": (
        "task0=input:a,input:b,output:tmp0;task1=input:a,input:b,output:tmp1;"
        "task2=input:a,input:b,output:out"
    ),
    "pto_persistent_dag_graph_named_callable": (
        "task0=callable:add,input:a,input:b,output:tmp0;"
        "task1=callable:mul,input:a,input:b,output:tmp1;"
        "task2=callable:add,input:a,input:b,output:out"
    ),
}
PAIRED_CURRENT_GRAPH_TASK_ARG_KEYS = {
    "pto_persistent_dag_graph_tagged_inout": "tag",
    "pto_persistent_dag_graph_role_keyed_inout": "role",
    "pto_persistent_dag_graph_compact_role_inout": "compact",
    "pto_persistent_dag_graph_pair_inout": "pair",
    "pto_persistent_dag_graph_role_map_inout": "role_map",
    "pto_persistent_dag_graph_submit_groups": "submit_groups",
    "pto_persistent_dag_graph_named_callable": "named_callable",
    "pto_persistent_dag_graph_node_io": "node_io",
}
PAIRED_CURRENT_GRAPH_NODE_ATTRS = {
    "pto_persistent_dag_graph_node_attrs": "task0=attrs:tensor_args,scalar_args",
}
PAIRED_CURRENT_GRAPH_NODE_OPS = {
    "pto_persistent_dag_graph_node_link": "task0=op:add=1;task1=op:mul=2;task2=op:add=1",
    "pto_persistent_dag_graph_named_callable": "task0=op:add=1;task1=op:mul=2;task2=op:add=1",
    "pto_persistent_dag_graph_node_op": "task0=op:add=1;task1=op:mul=2;task2=op:add=1",
}
PAIRED_CURRENT_SCALAR_ARGS = {
    "pto_persistent_dag_graph_layered_cross": "scalar0=2.0",
    "pto_persistent_dag_graph_node_attrs": "scalar_args[0]=1.5,scalar_args[1]=0.25",
}
PAIRED_CURRENT_TENSOR_ARGS = {
    "pto_persistent_dag_graph_layered_cross": "c=a",
    "pto_persistent_dag_graph_node_attrs": "tensor_args[0]=tmp0,tensor_args[1]=tmp3",
}
PAIRED_CURRENT_GRAPH_ROLE_SPELLING_BASELINES = (
    "pto_persistent_dag_graph_tagged_inout",
    "pto_persistent_dag_graph_role_keyed_inout",
    "pto_persistent_dag_graph_compact_role_inout",
    "pto_persistent_dag_graph_pair_inout",
    "pto_persistent_dag_graph_role_map_inout",
)
PAIRED_CURRENT_GRAPH_FANIN = {
    "pto_persistent_dag_graph": "0,0,2",
    "pto_persistent_dag_graph_generic_args4": "0,0,2",
    "pto_persistent_dag_graph_node_attrs": "0,0,2",
    "pto_persistent_dag_graph_node_io": "0,0,2",
    "pto_persistent_dag_graph_node_link": "0,0,2",
    "pto_persistent_dag_graph_named_callable": "0,0,2",
    "pto_persistent_dag_graph_node_op": "0,0,2",
    "pto_persistent_dag_graph_depends_on": "0,0,2",
    "pto_persistent_dag_graph_scalar_axpy": "0,0,2",
    "pto_persistent_dag_graph_scalar_scale": "0,0,2",
    "pto_persistent_dag_graph_scalar_affine": "0,0,2",
    "pto_persistent_dag_graph_reordered": "2,0,0",
    "pto_persistent_dag_graph_chain": "0,0,2,1,1",
    "pto_persistent_dag_graph_scratch_reuse": "0,0,2,1,1,2",
    "pto_persistent_dag_graph_diamond": "0,0,2,2,2",
    "pto_persistent_dag_graph_parallel_chains": "0,0,0,0,2,2,2,2,2",
    "pto_persistent_dag_graph_wide_fanout": "0,1,1,1,2,2,2",
    "pto_persistent_dag_graph_multi_fanin": "0,0,0,3",
    "pto_persistent_dag_graph_layered_cross": "0,0,0,2,3,1,2,3,2",
    "pto_persistent_dag_graph_tagged": "0,0,2",
    "pto_persistent_dag_graph_tagged_inout": "0,1,1",
    "pto_persistent_dag_graph_role_keyed_inout": "0,1,1",
    "pto_persistent_dag_graph_compact_role_inout": "0,1,1",
    "pto_persistent_dag_graph_pair_inout": "0,1,1",
    "pto_persistent_dag_graph_role_map_inout": "0,1,1",
    "pto_persistent_dag_graph_submit_groups": "0,0,2",
    "pto_persistent_dag_graph_triad": "0,0,2",
    "pto_persistent_dag_graph_quad": "0,0,2",
    "pto_persistent_dag_graph_unary_square": "0,1,1",
    "pto_persistent_dag_graph_tensor": "0,1,1,2",
    "pto_persistent_dag_graph_tensor_core": "0,1,1,2",
}
PAIRED_CURRENT_GRAPH_DEPENDENTS = {
    "pto_persistent_dag_graph": "2,2",
    "pto_persistent_dag_graph_generic_args4": "2,2",
    "pto_persistent_dag_graph_node_attrs": "2,2",
    "pto_persistent_dag_graph_node_io": "2,2",
    "pto_persistent_dag_graph_node_link": "2,2",
    "pto_persistent_dag_graph_named_callable": "2,2",
    "pto_persistent_dag_graph_node_op": "2,2",
    "pto_persistent_dag_graph_depends_on": "2,2",
    "pto_persistent_dag_graph_scalar_axpy": "2,2",
    "pto_persistent_dag_graph_scalar_scale": "2,2",
    "pto_persistent_dag_graph_scalar_affine": "2,2",
    "pto_persistent_dag_graph_reordered": "0,0",
    "pto_persistent_dag_graph_chain": "2,2,3,4",
    "pto_persistent_dag_graph_scratch_reuse": "2,2,3,4,5,5",
    "pto_persistent_dag_graph_diamond": "2,3,2,3,4,4",
    "pto_persistent_dag_graph_parallel_chains": "4,4,5,5,6,7,6,7,8,8",
    "pto_persistent_dag_graph_wide_fanout": "1,2,3,4,4,5,5,6,6",
    "pto_persistent_dag_graph_multi_fanin": "3,3,3",
    "pto_persistent_dag_graph_layered_cross": "3,3,4,4,5,4,6,7,6,7,7,8,8",
    "pto_persistent_dag_graph_tagged": "2,2",
    "pto_persistent_dag_graph_tagged_inout": "1,2",
    "pto_persistent_dag_graph_role_keyed_inout": "1,2",
    "pto_persistent_dag_graph_compact_role_inout": "1,2",
    "pto_persistent_dag_graph_pair_inout": "1,2",
    "pto_persistent_dag_graph_role_map_inout": "1,2",
    "pto_persistent_dag_graph_submit_groups": "2,2",
    "pto_persistent_dag_graph_triad": "2,2",
    "pto_persistent_dag_graph_quad": "2,2",
    "pto_persistent_dag_graph_unary_square": "1,2",
    "pto_persistent_dag_graph_tensor": "1,2,3,3",
    "pto_persistent_dag_graph_tensor_core": "1,2,3,3",
}
