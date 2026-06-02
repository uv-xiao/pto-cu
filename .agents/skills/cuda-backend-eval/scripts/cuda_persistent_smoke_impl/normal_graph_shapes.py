"""Normal-graph DAG shape definitions for the CUDA persistent smoke runner."""

from __future__ import annotations

from simpler_setup.cuda_normal_graph import CudaNormalGraphNode

NORMAL_GRAPH_DAG_SHAPES = frozenset(
    {
        "normal_graph_chain",
        "normal_graph_fork_join",
        "normal_graph_layered_cross",
        "normal_graph_multi_fanin",
    }
)


def make_normal_graph_nodes(
    shape: str,
    *,
    n: int,
    dev_a: int,
    dev_b: int,
    dev_tmp0: int,
    dev_tmp1: int,
    dev_tmp2: int,
    dev_tmp3: int,
    dev_out: int,
) -> tuple[CudaNormalGraphNode, ...]:
    if shape == "normal_graph_fork_join":
        return (
            CudaNormalGraphNode("left", func_id=1, a=dev_a, b=dev_b, out=dev_tmp0, n=n),
            CudaNormalGraphNode("right", func_id=2, a=dev_a, b=dev_b, out=dev_tmp1, n=n),
            CudaNormalGraphNode(
                "join",
                depends_on=("left", "right"),
                func_id=1,
                a=dev_tmp0,
                b=dev_tmp1,
                out=dev_out,
                n=n,
            ),
        )
    if shape == "normal_graph_chain":
        return (
            CudaNormalGraphNode("root_add", func_id=1, a=dev_a, b=dev_b, out=dev_tmp0, n=n),
            CudaNormalGraphNode("root_mul", func_id=2, a=dev_a, b=dev_b, out=dev_tmp1, n=n),
            CudaNormalGraphNode(
                "join_add",
                depends_on=("root_add", "root_mul"),
                func_id=1,
                a=dev_tmp0,
                b=dev_tmp1,
                out=dev_tmp2,
                n=n,
            ),
            CudaNormalGraphNode(
                "tail_mul",
                depends_on=("join_add",),
                func_id=2,
                a=dev_tmp2,
                b=dev_b,
                out=dev_tmp3,
                n=n,
            ),
            CudaNormalGraphNode(
                "tail_add",
                depends_on=("tail_mul",),
                func_id=1,
                a=dev_tmp2,
                b=dev_tmp3,
                out=dev_out,
                n=n,
            ),
        )
    if shape == "normal_graph_multi_fanin":
        return (
            CudaNormalGraphNode("add", func_id=1, a=dev_a, b=dev_b, out=dev_tmp0, n=n),
            CudaNormalGraphNode("mul", func_id=2, a=dev_a, b=dev_b, out=dev_tmp1, n=n),
            CudaNormalGraphNode(
                "scale",
                func_id=11,
                a=dev_a,
                out=dev_tmp2,
                n=n,
                attrs={"scalar0": 2.0},
            ),
            CudaNormalGraphNode(
                "join",
                depends_on=("add", "mul", "scale"),
                func_id=6,
                a=dev_tmp0,
                b=dev_tmp1,
                out=dev_out,
                n=n,
                attrs={"c": dev_tmp2},
            ),
        )
    if shape == "normal_graph_layered_cross":
        return (
            CudaNormalGraphNode("root_add", func_id=1, a=dev_a, b=dev_b, out=dev_tmp0, n=n),
            CudaNormalGraphNode("root_mul", func_id=2, a=dev_a, b=dev_b, out=dev_tmp1, n=n),
            CudaNormalGraphNode(
                "root_scale",
                func_id=11,
                a=dev_a,
                out=dev_tmp2,
                n=n,
                attrs={"scalar0": 2.0},
            ),
            CudaNormalGraphNode(
                "mid_add",
                depends_on=("root_add", "root_mul"),
                func_id=1,
                a=dev_tmp0,
                b=dev_tmp1,
                out=dev_tmp3,
                n=n,
            ),
            CudaNormalGraphNode(
                "mid_mul",
                depends_on=("root_add", "root_mul", "root_scale"),
                func_id=2,
                a=dev_tmp1,
                b=dev_tmp2,
                out=dev_tmp0,
                n=n,
            ),
            CudaNormalGraphNode(
                "side_add",
                depends_on=("root_scale",),
                func_id=1,
                a=dev_tmp2,
                b=dev_a,
                out=dev_out,
                n=n,
            ),
            CudaNormalGraphNode(
                "fma",
                depends_on=("mid_add", "mid_mul"),
                func_id=6,
                a=dev_tmp3,
                b=dev_tmp0,
                out=dev_tmp1,
                n=n,
                attrs={"c": dev_a},
            ),
            CudaNormalGraphNode(
                "merge_side",
                depends_on=("mid_add", "mid_mul", "side_add"),
                func_id=1,
                a=dev_tmp3,
                b=dev_out,
                out=dev_tmp2,
                n=n,
            ),
            CudaNormalGraphNode(
                "final",
                depends_on=("fma", "merge_side"),
                func_id=1,
                a=dev_tmp1,
                b=dev_tmp2,
                out=dev_out,
                n=n,
            ),
        )
    raise ValueError(f"unknown normal graph shape: {shape}")
