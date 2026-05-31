# 2026-05-31 MPK Runtime Import Guard

## Code And Data Changed

- Added `mirage.mpk.base_dynamic_shard_loader` to the MPK paper-baseline
  probe contract.
- Re-ran paired A100/H200 probes under
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-04dc762d/`.
- Refreshed run-readiness, paper-readiness audit, work queue, and goal
  progress data from the stricter probe result.
- Documented the MPK runtime import gate in the CUDA backend evaluation skill.

## Architecture Quality

The previous probe only proved that the Qwen3 demo parsed and that generic
dependencies such as PyTorch, Transformers, and Triton were importable. That
was too weak for MPK because the persistent-kernel path imports
`mirage.mpk.*` from the pinned MPK checkout. A separate generic `mirage`
package can satisfy `import mirage` while still missing the MPK subpackage or
matching native symbols.

The readiness guard now checks the callable runtime import needed by the
actual MPK run commands. This makes the benchmark viewer more conservative,
but the queue now reflects executable evidence instead of source-level intent.

## Evaluation Run

Attempted a minimal local Qwen3 native bring-up run for MPK. The environment
needed user-site isolation to avoid an incompatible `torchvision`, then needed
venv-local Transformers, Safetensors, Pillow, `z3-solver`, `graphviz`, and
CUDA Python bindings. After a tmp-only `libz3.so.4.15` symlink workaround, the
local generic Mirage import succeeded, but the MPK checkout import still failed
because the available native `mirage.core` did not match the MPK Python
package expectations.

The paired probe now records the actionable blocker on both machines:

```text
python_import failed: mirage.mpk.base_dynamic_shard_loader
```

## Remaining Gaps

MPK paper-baseline runs should not be executed or imported until the pinned
MPK checkout is built/installed so `mirage.mpk.base_dynamic_shard_loader`
imports on the target evaluation host. The work queue intentionally rises
because the earlier ready state was under-verified.
