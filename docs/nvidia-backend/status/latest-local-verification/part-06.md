# CUDA Backend Status: Latest Local Verification Part 6

Result: `tmp/cuda-backend/combined-current-b2c5c8a4/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, `cuda-benchmark-dag-deltas.svg`, and
`cuda-benchmark-throughput.svg`. The combined JSON has `60` samples and the
compact-current validator reported:
`validated tmp/cuda-backend/combined-current-b2c5c8a4/cuda-benchmark.json`.
This capture validates `30` selected benchmark rows per GPU, including
`pto_host_schedule_generic_args` and
`pto_persistent_dag_graph_generic_args4`. Selected A100 device times for host,
host-generic, base-DAG, persistent-generic, graph-generic4, tensor,
tensor-core, cuBLAS, and grid-batch were
`22528/35840/44032/29696/27648/37888/36864/37888/37888 ns`; H200 reported
`16992/31264/40320/30592/27520/48992/32480/34304/31872 ns`.

Earlier result: `tmp/cuda-backend/combined-current-d361006f/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`, and
`cuda-benchmark-ratios.svg`; it also writes
`cuda-benchmark-dag-deltas.svg`. New reports also write
`cuda-benchmark-throughput.svg` for tensor and cuBLAS rows. The combined JSON
has `50` samples and the paired-current validator reported:
`validated tmp/cuda-backend/combined-current-d361006f/cuda-benchmark.json`.
This capture proves the default paired workflow now keeps
`pto_persistent_dag_tensor`, `pto_persistent_dag_tensor_core`, and
`cublas_sgemm` in one validated current-head report on A100 and H200.

The compact paired-current gate was refreshed at commit `0b3c1699` after
adding persistent DAG no-progress diagnostics. It uses the same command shape
and validates command examples, source-paper provenance, and generated report
files:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks 2 \
    --worker-blocks-per-task 4 --sync-remote-tree
```

Result: `tmp/cuda-backend/combined-current-0b3c1699/` contains
`cuda-benchmark.json`, `cuda-benchmark.md`, `cuda-benchmark.svg`,
`cuda-benchmark-ratios.svg`, and `cuda-benchmark-dag-deltas.svg`. Regenerated
reports also include `cuda-benchmark-throughput.svg`. The combined JSON has
`50` samples and the validator reported:
`validated tmp/cuda-backend/combined-current-0b3c1699/cuda-benchmark.json`.
Selected A100 device times for
host/base-DAG/tensor/tensor-core/cuBLAS/grid-batch were
`33792/61440/44032/59392/60416/41984 ns`; H200 reported
`14848/31936/44576/36096/40959/28768 ns`.

