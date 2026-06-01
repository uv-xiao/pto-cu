# CUDA Backend Status: Evaluation And Reporting Part 2

The first cuBLAS library-backed tensor baseline adds `cublas_sgemm` to the
same compact selected-baseline report shape. It uses CUDA Runtime API events
around a warm cuBLAS `cublasSgemmStridedBatched` call over the configured
`16x16x16` descriptor. The matching `cublas_sgemm_graph` row captures that
same warmed descriptor into a CUDA Graph, instantiates it outside the measured
interval, warms graph replay once, and times `cudaGraphLaunch` with CUDA
events. In the paired A100/H200 capture under
`tmp/cuda-backend/combined-cublas-current-343924df/`, the row measured
`48128 ns` device time on A100 and `58623 ns` on H200. The matched
`pto_persistent_dag_tensor_core` rows in that report measured `33792 ns` and
`32960 ns`. This row is a CUDA library launch/compute comparison point, not a
PTO runtime path.

The first cuBLAS CUDA Graph paired capture is under
`tmp/cuda-backend/cublas-graph-compact-working/combined-current-5168f150/`.
It uses `N=1024`, one repeat, no batch rows, and the default `16x16x16`
descriptor. The paired runner synced the working tree to `bizhaoh200`,
captured A100 and H200 reports, merged `58` rows, and validated report files,
command examples, source-paper provenance, tensor descriptor metadata, PTO
dispatch sequences, and zero scheduler errors. The A100 rows measured
`cublas_sgemm=48128 ns` and `cublas_sgemm_graph=10239 ns`; H200 measured
`9119 ns` and `8543 ns`. The graph row is a launch-path comparison point
around cuBLAS graph replay, not a tuned GEMM throughput claim.

The WMMA tensor-core task body now handles a grid of 16x16 output fragments
instead of only one fragment per descriptor. Focused TDD first failed because
`pto_persistent_dag_tensor_core` rejected `32x16x16`; after generalizing the
descriptor contract and generated-dispatch task body, the paired smoke:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape tensor_core_tile --task-count 4 --n 1024 \
    --tensor-rows 32 --tensor-cols 16 --tensor-inner 16 \
    --repeat-runs 2 --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-core-wide-working
```

validated A100 and H200 JSON, Markdown, and SVG artifacts at
`tmp/cuda-backend/tensor-core-wide-working/persistent-tensor_core_tile-32x16x16-repeat2-smoke-ef475c2d/`.
Both GPUs reported dispatch `10,1,2,1`, tensor tile `32x16x16`,
`tile_count=2`, repeat completions `[4,4]`, and zero scheduler errors. The
A100 device total was `77824 ns`; H200 was `61152 ns`.

The normal L2 `SceneTestCase` adapter now uses the same descriptor rule as
the benchmark/smoke path for tensor-core tasks: `rows` and `cols` must be
multiples of `16`, and `inner` must be divisible by `8`. Focused TDD first
failed on both the fixed `persistent_dag_tensor_core_tile_f32` builder and
the explicit graph `func_id=10` path because each still required
`rows=16, cols=16`. After the guard was generalized, the no-torch ctypes
SceneTestCase selector ran the normal tensor-core path with a `32x16x16`
descriptor and the graph tensor-core path with the existing `16x16x16`
descriptor on A100 and H200:

```bash
PYTHONPATH=$PWD:$PWD/python PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -rs -k 'tensor_core_tile_with_ctypes_data' --platform cuda
```

Local A100 reported `2 passed, 143 deselected`; the synced H200 run reported
the same result after the known PTO-ISA SSH refresh warning. This closes the
normal scene-test gap for multi-fragment WMMA descriptors; broader
model-kernel families are still an evaluation and tuning task.

The compact paired benchmark then used the same `32x16x16` descriptor with
`N=1024`, one repeat, no batch rows, and the normal selected baseline set:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py \
    --sizes 1024 --repeats 1 --batch-tasks '' \
    --tensor-rows 32 --tensor-cols 16 --tensor-inner 16 \
    --sync-remote-tree \
    --output-root tmp/cuda-backend/tensor-core-wide-benchmark-working
```

The capture under
`tmp/cuda-backend/tensor-core-wide-benchmark-working/combined-current-ef475c2d/`
validated `58` rows, report files, command examples, source-paper metadata,
tensor descriptors, PTO dispatch sequences, and zero scheduler errors. In the
tensor-throughput table, A100 measured
`pto_persistent_dag_tensor_core=36864 ns`,
`cublas_sgemm=37888 ns`, and `cublas_sgemm_graph=7168 ns`; H200 measured
`33280 ns`, `38591 ns`, and `9216 ns` for the same rows.

The tensor shape sweep script now accepts `--baselines` and `--sizes`, so one
paired A100/H200 sweep can compare scalar tensor DAG, the explicit graph
tensor DAG, WMMA tensor-core DAG, and cuBLAS SGEMM rows across descriptor
shapes and problem sizes. The multi-repeat size-sweep report at commit
`e79edba2` uses a `16x16x16` descriptor with `N=256`, `4096`, and `65536`,
three repeats, and the artifact under
`tmp/cuda-backend/tensor-shape-sweep-e79edba2/`. It writes raw rows,
VDCores/MPK provenance metadata, per-baseline workload descriptions, a median
summary table with normalized GFLOP/s, and SVG charts for median device time
and median GFLOP/s with sample counts. Median device times are: A100
scalar/tensor-core/cuBLAS at
`N=256`
`47104/47104/43007 ns`, `N=4096` `79872/71680/36864 ns`, and `N=65536`
`587616/470368/38911 ns`; H200 at `N=256` `30560/28160/50496 ns`, `N=4096`
`88576/49888/37055 ns`, and `N=65536` `1032896/390368/36127 ns`. Normalized
throughput at `N=65536` is A100 scalar/tensor-core/cuBLAS
`3.57/4.46/53.90` GFLOP/s and H200 `2.03/5.37/58.05` GFLOP/s. The PTO
tensor-core row improves over the scalar tensor row as repeated tile work
grows, while the cuBLAS baseline shows the remaining gap to a tuned CUDA
library path.
A current-head one-repeat graph-tensor sweep at
`tmp/cuda-backend/tensor-shape-sweep-0e84fd26/` adds
`pto_persistent_dag_graph_tensor` to the same `16x16x16` tensor-baseline
comparison at `N=256` and `4096`. The validated median device times were:
A100 scalar/graph/tensor-core/cuBLAS `47104/47104/45056/48128 ns` at
`N=256` and `80896/80896/82944/39935 ns` at `N=4096`; H200
`29568/32800/27040/51711 ns` at `N=256` and
`89472/89152/51872/35904 ns` at `N=4096`. The graph tensor row uses the same
dispatch `3,1,2,1` as the scalar tensor row while exercising the explicit
runtime graph descriptor path.
`cuda_validate_tensor_sweep.py` checked the expected A100/H200 rows,
baselines, sizes, shape, three repeats, report files, throughput report
content, and PTO dispatch sequences before the numbers were copied into docs.
New tensor-sweep captures can also require sanitized local/remote command
examples and source-paper metadata before publishing with
`--require-command-examples` and `--require-source-papers`; the source-paper
gate verifies the referenced files exist under `tmp/sources/`.
Benchmark captures now use the same VDCores/MPK `source_papers` metadata
contract and sanitized command-example metadata contract as tensor sweeps.
They can be gated with `cuda_validate_capture.py` plus
`--require-command-examples`, `--require-zero-scheduler-errors`, and
`--require-source-papers` before new paired-current numbers are published.
The existing real-data paired A100/H200 capture from commit `61cf96cd` was
re-rendered through the updated report path under
`tmp/cuda-backend/combined-current-61cf96cd-command-source-gate/`, and the
paired-current validator passed with `--require-command-examples` and
`--require-source-papers`.

A current-HEAD one-repeat compact tensor sweep at commit `a5fd4bfd` validated
that gate against real local A100 and remote H200 data. The artifact under
`tmp/cuda-backend/tensor-shape-sweep-a5fd4bfd/` includes sanitized local
sample, remote sample, and remote tree-sync commands in generated Markdown and
JSON. The validation required A100/H200 rows, scalar tensor DAG, WMMA tensor
DAG, cuBLAS SGEMM, `N=256`, `16x16x16`, report files, command examples, and
source-paper metadata, and PTO dispatch sequences.

A follow-up working-tree sweep under
`tmp/cuda-backend/tensor-graph-library-baselines-working/`
`tensor-shape-sweep-848c4ee5/` adds the `cublas_sgemm_graph` tensor-sweep
baseline beside scalar tensor, explicit graph tensor, WMMA tensor-core, and
plain cuBLAS rows. The run uses one repeat, `N=256`, and the `16x16x16`
descriptor on local A100 and remote H200. Validated median device times were:
A100 scalar/graph/tensor-core/cuBLAS/cuBLAS-graph
`43008/41984/51200/89088/12288 ns`; H200
`37440/45120/37472/50271/10271 ns`. The cuBLAS Graph row shows the expected
benefit of replaying an already captured library call in this launch-dominated
compact descriptor, while PTO graph tensor remains close to the scalar tensor
DAG and continues to validate the explicit runtime graph descriptor path.

A current-head full compact sweep at commit `219042f5` refreshes that
baseline set with three repeats and both compact preset shapes,
`16x16x16` and `16x16x64`. The artifact under
`tmp/cuda-backend/tensor-sweep-current-working/tensor-shape-sweep-219042f5/`
validated 72 rows with `--preset compact-tensor-baselines`,
`--require-command-examples`, and `--require-source-papers`. That preset now
also requires visible Markdown/SVG throughput content and visible Markdown
baseline-comparison content. Median `16x16x64` device times were A100
scalar/graph/tensor-core/graph-tensor-core/cuBLAS/cuBLAS-graph
`52224/38912/50176/50176/74752/9216 ns`; H200 measured
`32288/32512/32480/32127/51135/10176 ns`. Median `16x16x64` GFLOP/s were
A100 `0.63/0.84/0.65/0.65/0.44/3.56` and H200
`1.01/1.01/1.01/1.02/0.64/3.22`. The regenerated Markdown now compares PTO
tensor rows against matching cuBLAS Graph rows; for example the explicit
graph tensor row is `4.00x` cuBLAS Graph device time on A100 and `3.36x` on
H200 for `16x16x16`, while explicit graph tensor-core is `5.44x` on A100 and
`3.16x` on H200 for `16x16x64`.

A current-head one-repeat size sweep at commit `76422250` extends the same
six selected tensor baselines across `N=256`, `4096`, and `65536` for
`16x16x16`. The artifact under
`tmp/cuda-backend/tensor-size-sweep-working/tensor-shape-sweep-76422250/`
validated 36 rows with required A100/H200 artifacts, report files, visible
throughput content, command examples, VDCores/MPK source-paper metadata, and
PTO dispatch sequences. At `N=65536`, A100 measured scalar/graph/tensor-core/
graph-tensor-core/cuBLAS/cuBLAS-graph device times of
`437184/439392/579360/583840/30719/11264 ns`; H200 measured
`424448/426752/343904/343136/43039/8448 ns`. The explicit graph descriptor
tracks the generated scalar tensor DAG closely at large `N`, while H200's
WMMA tensor-core rows improve over scalar tensor and A100's current WMMA row
does not in this one-repeat capture. cuBLAS Graph remains the fastest
launch-replay baseline, so the next gap is still tuned PTO tensor body work
rather than descriptor, graph, or report plumbing.

Evidence:

