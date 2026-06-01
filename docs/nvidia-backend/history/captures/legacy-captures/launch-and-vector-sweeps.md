# Launch And Vector Sweep Results

## Headline Results

| GPU | N | `pto_host_schedule_batch` ns | `persistent_device_batch` | Best grid blocks/task | Best grid ratio | `persistent_queue_batch` |
| --- | - | ---------------------------- | ------------------------- | --------------------- | --------------- | ------------------------ |
| A100 | 1024 | 73728 | 0.61x | 256 | 0.58x | 0.82x |
| H200 | 1024 | 66879 | 0.42x | 256 | 0.55x | 0.60x |
| A100 | 65536 | 67968 | 1.34x | 128 | 0.47x | 1.42x |
| H200 | 65536 | 61952 | 1.33x | 32 | 0.37x | 1.40x |
| A100 | 1048576 | 76768 | 16.46x | 256 | 0.53x | 16.06x |
| H200 | 1048576 | 67327 | 18.13x | 256 | 0.42x | 18.15x |

The small-vector rows show launch-amortization benefit from the persistent
paths. The large-vector rows show why the worker-grid variant matters: in the
`32,64,128,256` extended sweep, the best large-vector row uses 256 worker
blocks per descriptor on both GPUs. That reduces the A100 direct persistent
batch row from `16.46x` to `0.53x` versus the matched host-schedule batch
row, and reduces the H200 row from `18.13x` to `0.42x`. The middle `N=65536`
rows show the same shape: plain persistent batch and queue batch are slower
than host-schedule batch, while the best worker-grid row is faster. The
middle-size optimum is not monotonic: A100 ties at 128/256 blocks, while H200
is best at 32 blocks in this capture.

## Task-Count Sweep

The `7194bfc9` task-count sweep uses the same vector-add callable and compares
two vector lengths, three descriptor counts, and two grid sizes. The
worker-grid row stays below the matched host-schedule batch row for every
captured task count, while the plain one-block persistent batch row remains
too serial for larger vectors.

| GPU | N | Tasks | Best grid blocks/task | Best grid ratio |
| --- | - | ----- | --------------------- | --------------- |
| A100 | 65536 | 2 | 128 | 0.77x |
| A100 | 65536 | 6 | 128 | 0.38x |
| A100 | 65536 | 12 | 128 | 0.33x |
| A100 | 1048576 | 2 | 256 | 0.77x |
| A100 | 1048576 | 6 | 256 | 0.58x |
| A100 | 1048576 | 12 | 256 | 0.44x |
| H200 | 65536 | 2 | 128 | 0.68x |
| H200 | 65536 | 6 | 256 | 0.36x |
| H200 | 65536 | 12 | 128 | 0.22x |
| H200 | 1048576 | 2 | 256 | 0.79x |
| H200 | 1048576 | 6 | 128 | 0.40x |
| H200 | 1048576 | 12 | 256 | 0.34x |

Increasing descriptor count improves the worker-grid ratio because the
matched host-schedule reference pays more repeated launch overhead. It does
not make the one-block persistent rows acceptable: at `N=1048576`, the A100
plain persistent batch row is still `31.59x`, `16.92x`, and `10.55x` for
2, 6, and 12 tasks respectively.

## Wider Range Sweep

The `cc6869f7` wider range sweep keeps the same generated callables and
extends the descriptor-count sweep to `4,8,16` tasks across `N=16384`,
`262144`, and `4194304`. The worker-grid row stays below the matched
host-schedule batch row for every captured row, but the largest vectors become
compute-sensitive and show a smaller ratio advantage.

| GPU | N | Tasks | Best grid blocks/task | Best grid ratio |
| --- | - | ----- | --------------------- | --------------- |
| A100 | 16384 | 16 | 256 | 0.13x |
| H200 | 16384 | 16 | 128 | 0.13x |
| A100 | 262144 | 16 | 128 | 0.22x |
| H200 | 262144 | 16 | 256 | 0.22x |
| A100 | 4194304 | 16 | 128 | 0.53x |
| H200 | 4194304 | 16 | 128 | 0.51x |

The best grid size is not monotonic. H200 uses 256 blocks/task for the
`N=4194304`, four-task row, but 128 blocks/task for the eight-task and
16-task rows. A100 uses 256 blocks/task for the `N=4194304`, eight-task row,
but 128 blocks/task for the four-task and 16-task rows. This keeps `128` and
`256` as candidates for the current vector microbenchmark, not tuned defaults.

The one-block persistent rows remain too serial for large vectors. At
`N=4194304`, `pto_persistent_device_batch` is `47.18x`, `24.26x`, and
`12.38x` on A100 for 4, 8, and 16 tasks respectively; H200 is `65.28x`,
`37.53x`, and `18.46x`. The scalar tensor DAG row is also only an ABI and
scheduler validation row at this size: it reaches `355.69x` on A100 and
`461.79x` on H200 versus the matched four-task host-schedule batch row.

## PTX Sources

The A100 rows compiled PTX with local `nvcc` for `compute_80`. The H200 rows
compiled PTX with remote `nvcc` for `compute_90`, discovered from the
`/usr/local/cuda*` toolkit path. The report still marks embedded PTX rows when
fallback PTX is used, but the latest H200 report does not use that fallback.

## CUDA Graph Launch Baseline

The `direct_driver_graph` row instantiates a one-kernel CUDA Graph before the
timed interval and measures replay of that graph. This is a host-launch
amortization baseline for repeated `host_schedule` style callables; it is not
a replacement for the persistent-device scheduler because the host still owns
graph construction and replay.

| GPU | N | Host-schedule ns | Driver ns | Driver graph ns | Graph/host | Graph/driver |
| --- | - | ---------------- | --------- | --------------- | ---------- | ------------ |
| A100 | 1024 | 32768 | 38911 | 26623 | 0.81x | 0.68x |
| H200 | 1024 | 26720 | 30848 | 29311 | 1.10x | 0.95x |
| A100 | 65536 | 35648 | 27744 | 18975 | 0.53x | 0.68x |
| H200 | 65536 | 32896 | 37184 | 24351 | 0.74x | 0.65x |
| A100 | 1048576 | 28192 | 27904 | 23360 | 0.83x | 0.84x |
| H200 | 1048576 | 34048 | 40832 | 27071 | 0.80x | 0.66x |

Graph replay is faster than raw Driver API launch on every captured row. It is
also faster than the current PTO `host_schedule` path on five of six rows; the
H200 `N=1024` row is the exception, where `host_schedule` remains lower. This
keeps CUDA Graphs useful for a phase-1 repeated-launch optimization, while
leaving the phase-2 persistent-device work focused on device-side scheduling.

## Stream Concurrency

The host-schedule stream microbenchmark prepares two independent slow
vector-add callables with different `stream_id` values, runs them serially,
then launches them concurrently from host threads. The copied-back results are
validated in both cases.

| GPU | Serial ns | Parallel ns | Parallel vs serial |
| --- | --------- | ----------- | ------------------ |
| A100 | 113838981 | 57789544 | 0.51x |
| H200 | 89797063 | 46229849 | 0.51x |

This supports keeping multiple CUDA streams in the host-schedule runtime:
independent prepared callables can overlap when issued from separate host
threads. It does not solve the persistent-device scheduling problem, where the
CUDA device-side scheduler still has to run inside a persistent kernel.
