# Persistent MoE Dispatch/Combine H200 Note

This note tracks the first PR-sized persistent-device MoE dispatch/combine
slice after the Gluon MoE expert primitive.

## Scope

- Example: `examples/cuda/persistent_moe_dispatch_combine.py`
- Runtime shape: `graph_descriptor_moe_dispatch_combine`
- Tasks: four expert transforms and one weighted combine
- Default expert 0 path: `gluon_gen` persistent task-body bridge for
  `moe_expert_affine_f32` as func id `12`
- Dependency behavior: device-side fan-in releases the combine task after all
  four expert tasks complete
- Output: structured JSON for local skip/pass/fail and remote pass/fail
  evidence, including top-level `gluon_expert_bridge` metadata and the
  matching `task_bodies` entry

This is not distributed expert parallelism, a serving integration, DeepSeek
inference, direct Triton/Gluon JIT linking into the persistent kernel, or a
performance claim.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --output-json tmp/persistent-moe-dispatch-combine-local.json
```

The local command is skip-safe when CUDA tooling or a visible NVIDIA GPU is not
available. Add `--require-cuda` when a skip should fail the command.

## Remote H200 Command Shape

Use a fresh synced remote checkout or an intentionally refreshed remote branch,
then run:

```bash
REMOTE_PTO_CU=<remote-checkout> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <python> examples/cuda/persistent_moe_dispatch_combine.py \
    --device 0 --n 4096 --arch compute_90 --require-cuda
```

Record the exact remote checkout path, Python environment, command output,
pass/fail result, and these JSON fields in the PR body and worker handoff:
`status`, `dag_shape`, `completed_count`, `max_abs_error`,
`device_scheduler_errors`, `fanin_remaining`, `gluon_expert_bridge`, the
`task_bodies` entry for func id `12`, and `artifact.source_kind`. Do not treat
a skipped or setup-failed remote command as H200 correctness evidence.

## Remote H200 Result

Run method: generic remote runner with `--sync` into a temporary remote
checkout, using plain system `python3`.

```bash
REMOTE_PTO_CU=/tmp/pto-cu-persistent-moe-gluon-wrapper \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    python3 examples/cuda/persistent_moe_dispatch_combine.py \
    --device 0 --n 4096 --arch compute_90 --require-cuda \
    --output-json tmp/persistent-moe-gluon-wrapper-h200.json'
```

Result: pass.

- `status`: `passed`
- `dag_shape`: `graph_descriptor_moe_dispatch_combine`
- `arch`: `compute_90`
- `completed_count`: `5`
- `fanin_remaining`: `[0, 0, 0, 0, 0]`
- `device_scheduler_errors`: `{"count": 0, "code": 0, "task_id": 0}`
- `max_abs_error`: `0.0`
- `gluon_expert_bridge`: func id `12`, kernel
  `moe_expert_affine_f32`, task `gluon_moe_expert_affine_f32`, source kind
  `gluon-persistent-task-body-bridge`, source sha256
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `task_bodies` func id `12`: name `gluon_moe_expert_affine_f32`, source
  kind `gluon-persistent-task-body-bridge`, source sha256
  `7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f`
- `artifact.source_kind`: `generated-dispatch`

This validates only the synthetic single-process persistent-device graph shape
above. It does not validate distributed expert parallelism, serving, DeepSeek
inference, direct Triton/Gluon JIT linking into the persistent kernel, or
performance.
