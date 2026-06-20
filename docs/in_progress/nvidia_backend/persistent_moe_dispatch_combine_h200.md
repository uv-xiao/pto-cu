# Persistent MoE Dispatch/Combine H200 Note

This note tracks the first PR-sized persistent-device MoE dispatch/combine
slice after the Gluon MoE expert primitive.

## Scope

- Example: `examples/cuda/persistent_moe_dispatch_combine.py`
- Runtime shape: `graph_descriptor_moe_dispatch_combine`
- Tasks: four expert transforms and one weighted combine
- Dependency behavior: device-side fan-in releases the combine task after all
  four expert tasks complete
- Output: structured JSON for local skip/pass/fail and remote pass/fail
  evidence

This is not distributed expert parallelism, a serving integration, DeepSeek
inference, or a performance claim.

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

Record the exact remote checkout path, Python environment, command output, and
pass/fail result in the PR body and worker handoff. Do not treat a skipped or
setup-failed remote command as H200 correctness evidence.

## Remote H200 Result

Run method: generic remote runner with `--sync` into a temporary remote
checkout, using plain system `python3`.

```bash
REMOTE_PTO_CU=/tmp/pto-cu-persistent-moe-dispatch-combine \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  python3 examples/cuda/persistent_moe_dispatch_combine.py \
    --device 0 --n 4096 --arch compute_90 --require-cuda
```

Result: pass.

- `status`: `passed`
- `arch`: `compute_90`
- `completed_count`: `5`
- `fanin_remaining`: `[0, 0, 0, 0, 0]`
- `device_scheduler_errors`: `{"count": 0, "code": 0, "task_id": 0}`
- `max_abs_error`: `0.0`

This validates only the synthetic single-process persistent-device graph shape
above. It does not validate distributed expert parallelism, serving, DeepSeek
inference, or performance.
