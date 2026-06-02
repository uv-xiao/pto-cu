# CUDA Pair Persistent Smoke Contract Split

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke_impl/contracts.py`
  for `PairedPersistentSmokeConfig` and tensor-tile DAG shape matching.
- Kept
  `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py`
  as the command-building entrypoint, with the same exported config name and
  `_is_tensor_tile_shape` alias used by existing checks and tests.

## Architecture Quality

The paired A100/H200 persistent-device smoke runner now separates immutable
capture configuration from command assembly. This makes the runner easier to
review without changing launch semantics, remote-sync policy, artifact naming,
or validation expectations.

## Evaluation Run

- `py_compile` covered the paired smoke script and new contracts module.
- Focused pytest covered the chain and tensor-tile descriptor paired workflows:
  `2 passed, 322 deselected`.

## Remaining Gaps

The expectation and validation-command helpers still live in the paired smoke
entrypoint. They remain candidates for a later split after this low-risk
contract extraction is reviewed.
