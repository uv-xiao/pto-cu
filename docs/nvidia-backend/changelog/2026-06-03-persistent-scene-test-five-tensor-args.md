# 2026-06-03 Persistent Scene-Test Five Tensor Args

## Code And Data Changed

- Updated `persistent_dag_generic_args_f32` scene-test packet construction to
  accept and pad five persistent tensor arguments.
- Updated `persistent_dag_graph_f32` graph task construction to accept and pad
  five tensor arguments per task.
- Left scalar argument capacity at four, matching the current
  `PtoCudaPersistentDagTask` ABI.

## Architecture Quality

Scene-test adapters now match the persistent-device ABI used by generated Qwen
tasks. This prevents tests and explicit graph descriptors from rejecting the
same fifth tensor slot that QK-norm uses for the runtime KV page table.

## Evaluation Run

- Failed before the adapter fix, then passed after it:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_scene_test.py \
    -q -k 'persistent_generic_args_five_tensor_slots or \
      persistent_graph_generic_args_five_tensor_slots'
  ```

- Passed focused persistent scene/codegen coverage:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_cuda_scene_test.py \
    tests/ut/py/test_cuda_persistent_codegen.py \
    -q -k 'persistent_generic_args or persistent_graph_generic_args or \
      generic_argument_slots'
  ```

## Remaining Gaps

This aligns review and scene-test packet construction with the current ABI; it
does not by itself produce full Qwen numerical correctness or full-serving
MPK/VDCores rows.
