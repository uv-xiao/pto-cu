# DAG Shape Results

## DAG Graph Shapes

The `pto_persistent_dag_chain` row validates that the same generated-dispatch
compiled binary can run a different runtime graph descriptor: two initial
tasks fan into an add task, then a multiply task, then a final add task. This
is still a vector microbenchmark, but it is closer to the desired persistent
runtime shape than a flat descriptor array because dependencies and fan-in
counters drive the ready queue.

The `pto_persistent_dag_reuse` row adds one more task and reuses `tmp0` after
the original `tmp0` producer's final dependent has completed. This is still
elementwise vector work, but it validates the lifecycle rule needed by a
persistent-device runtime: buffer lifetime can be represented by graph
dependencies and runtime descriptors while the generated-dispatch binary stays
unchanged.

| GPU | N | DAG ns | DAG-chain ns | Chain/DAG |
| --- | - | ------ | ------------ | --------- |
| A100 | 1024 | 32768 | 34816 | 1.06x |
| H200 | 1024 | 39584 | 46528 | 1.18x |
| A100 | 65536 | 155648 | 270336 | 1.74x |
| H200 | 65536 | 140032 | 244320 | 1.74x |
| A100 | 1048576 | 2333696 | 4242432 | 1.82x |
| H200 | 1048576 | 2010240 | 3581216 | 1.78x |

The chain row is slower than the three-task DAG because it performs more
device work and serializes two more dependency levels. That is expected here;
the useful signal is that graph shape, fan-in, and callable selection are
runtime data while the generated-dispatch device binary stays stable.

| GPU | N | DAG ns | DAG-chain ns | DAG-reuse ns | Reuse/DAG |
| --- | - | ------ | ------------ | ------------ | --------- |
| A100 | 1024 | 25600 | 36864 | 34816 | 1.36x |
| H200 | 1024 | 31456 | 39232 | 39168 | 1.25x |
| A100 | 65536 | 153600 | 268288 | 266240 | 1.73x |
| H200 | 65536 | 139328 | 244128 | 245696 | 1.76x |
| A100 | 1048576 | 2328576 | 4263936 | 3947520 | 1.70x |
| H200 | 1048576 | 2005919 | 3580768 | 3581664 | 1.79x |

The reuse row is close to the chain row because it has the same long multiply
path and one additional add branch. On A100 it is slightly faster than the
chain row for larger vectors because the final add consumes the reused
scratch branch rather than the earlier chain value; this is a microbenchmark
effect, not a claim that reuse is inherently faster.

The tensor row keeps the same persistent-DAG scheduler but extends the task
descriptor ABI with rows, columns, inner dimension, leading dimensions, and
per-tile strides. Its generated-dispatch `func_id=3` computes one or more GEMM
tiles before residual, gate, and fan-in elementwise tasks. The smoke helper now
supports non-square descriptors by allocating separate A, B, and output
extents, and the benchmark script can pass the same descriptor flags into the
`pto_persistent_dag_tensor` row. The following rows compare the older default
16x16x16 tensor DAG capture against the three-task elementwise DAG and the
one-call host-schedule vector baseline for shape context only. They are not
same-work throughput comparisons.

| GPU | N | Host ns | DAG ns | Tensor DAG ns | Tensor/DAG |
| --- | - | ------- | ------ | ------------- | ---------- |
| A100 | 1024 | 46080 | 45056 | 36864 | 0.82x |
| H200 | 1024 | 36512 | 29120 | 38912 | 1.34x |
| A100 | 65536 | 35072 | 151552 | 586752 | 3.87x |
| H200 | 65536 | 31615 | 139616 | 566656 | 4.06x |
| A100 | 1048576 | 31008 | 2296832 | 9235456 | 4.02x |
| H200 | 1048576 | 28576 | 1997568 | 8649408 | 4.33x |

At large `N`, the tensor DAG is roughly four times slower than the simple DAG
because each output element performs a 16-term dot product before the
elementwise residual, gate, and fan-in tasks. This is expected and confirms
that the persistent-device scheduler can run non-elementwise callable bodies
without changing the launch path. A metadata-carrying tensor DAG smoke after
the descriptor extension validated `N=4096` and 16 tiles with copied-back real
CUDA data on both A100 and H200. The compact smoke report and SVG are rendered
from the raw JSON with `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py`.

| GPU | PTX arch | Device ns | Rows x Cols x Inner | Tile count |
| --- | -------- | --------- | ------------------- | ---------- |
| A100 | `compute_80` | 102400 | 16 x 16 x 16 | 16 |
| H200 | `compute_90` | 70464 | 16 x 16 x 16 | 16 |
