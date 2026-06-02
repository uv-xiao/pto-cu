# CUDA Backend Status: Target Role Cleanup

## Target Role Cleanup

CUDA now builds native `host`, optional `scheduler`, and `device` target roles
when a runtime build config declares them, and runtime consumers can read
binaries through roles instead of direct hardware slot names. The current
compatibility mapping is:

- Ascend: `host`, `aicpu`, and `aicore` map to their existing artifacts.
- CUDA: `host` maps to `libhost_runtime.so`, `device` maps to
  `libcuda_device_runtime.so`, `scheduler` maps to
  `libcuda_scheduler_runtime.so` when present. CUDA no longer exposes
  `aicpu_path` / `aicore_path` aliases for these role-native artifacts.

The Python `ChipWorker.init(...)` wrapper now resolves runtime binary paths
through `path_for_role(...)` / `role_paths` first. CUDA role-only binary maps
with `host` / `device` or `host` / `scheduler` / `device` initialize through
the Python API and are passed through the C++ nanobind boundary as role maps.
The underlying `ChipWorker` now probes the optional C host-runtime
`simpler_init_roles(...)` entry and passes non-host role binaries directly to
runtimes that export it. The loaded `libhost_runtime.so` represents the
`host` role, so it is not copied back through the role map. Runtimes without
the optional entry still fall back to the legacy two-binary `simpler_init`
ABI.

The scheduler-role build slice was verified with:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_runtime_builder.py tests/ut/py/test_chip_worker.py -q \
  -k 'cuda_runtime_binaries or role_only_runtime_binaries or \
      role_keyed_init or role_keyed_paths or scheduler_role'

PYTHONPATH=$PWD:$PWD/python .venv/bin/python - <<'PY'
from simpler_setup.runtime_builder import RuntimeBuilder
bins = RuntimeBuilder(platform="cuda").get_binaries(
    "persistent_device", build=True
)
print(sorted(bins.role_paths))
print(bins.path_for_role("scheduler"))
PY
```

Result: the focused selector reported `7 passed, 75 deselected` with
`--platform cuda`; the runtime builder printed
`['device', 'host', 'scheduler']`, `aicpu_path=None`, `aicore_path=None`, and
the scheduler artifact path under
`build/lib/cuda/onboard/persistent_device/`.

After adding `_ChipWorker.init_roles(...)`, the same focused selector was run
on the synced H200 checkout and reported `5 passed, 38 deselected` with the
known PTO-ISA SSH refresh warning. A real persistent graph SceneTest path was
then checked on both GPUs through the Python `Worker` / `ChipWorker` surface:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_cuda_scene_test.py \
    -q -k persistent_device_graph_with_ctypes_data --platform cuda
```

Result: local A100 reported `1 passed, 72 deselected`; synced H200 reported
`1 passed, 72 deselected` with the known PTO-ISA SSH refresh warning.

The C host-runtime role-keyed ABI slice was then verified with a fake host
runtime that fails legacy `simpler_init(...)` but succeeds
`simpler_init_roles(...)`, plus a CUDA export check:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_chip_worker.py -q \
  -k prefers_role_keyed_runtime_init

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_cuda_backend.py -q -k role_keyed_init --platform cuda
```

Result: both selectors reported `1 passed`; the fake runtime received
`device` and `scheduler` entries and no `host` entry, while the rebuilt CUDA
`persistent_device` host runtime exported `simpler_init_roles`.

The same source tree was synced to `bizhaoh200` and checked with a paired
real-data persistent graph smoke:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
    --dag-shape graph_descriptor --task-count 3 --queue-capacity 2 \
    --repeat-runs 1 --sync-remote-tree \
    --output-root tmp/cuda-backend/scheduler-role-working
```

Result: the output directory under
`tmp/cuda-backend/scheduler-role-working/`, named
`persistent-graph_descriptor-smoke-539a05b9/`, contains `a100.json`,
`h200.json`, `cuda-smoke-report.md`, and `cuda-smoke-report.svg`. The
validator required runtime `persistent_device`, mode `dag`,
`dag_shape=graph_descriptor`, dispatch `[9,2,1]`, graph fan-in `[0,0,2]`,
graph dependents `[2,2]`, one repeat launch, resource policy
`scheduler_blocks=1`, `worker_blocks=3`, `block_dim=256`, `grid_dim=4`,
report files, and zero scheduler errors. A100 reported
`device_wall_ns=41984`; H200 reported `device_wall_ns=56864`. Both hosts also
built `libcuda_scheduler_runtime.so` beside `libhost_runtime.so` and
`libcuda_device_runtime.so`.

Compatibility note:

- The legacy positional `_ChipWorker.init(...)` path remains available for
  Ascend and old runtime-binary objects without `role_paths`. CUDA runtime
  binaries use role-native paths and `simpler_init_roles(...)` where exported,
  so this compatibility path is not an open CUDA backend blocker.
