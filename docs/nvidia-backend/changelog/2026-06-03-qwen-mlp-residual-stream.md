# 2026-06-03 Qwen MLP Residual Stream

## Code And Data Changed

- Updated `qwen_mlp_down` generated CUDA source to add the down projection to
  both residual inputs required by a Qwen decoder block: the attention output
  in `task->b` and the original layer input in runtime `tensor_args[1]`.
- Updated resource-backed launch packets so `qwen_mlp_down.tensor_args[1]`
  points to the matching layer's input-residual source.
- Updated Qwen task descriptors so `qwen_mlp_down` records the
  `mlp_residual` runtime tensor role in generated review artifacts.
- Added regression coverage for packet binding, descriptor materialization,
  review artifacts, and generated CUDA source.

## Architecture Quality

The previous binding added only the attention-output tensor after the MLP down
projection. `qwen_rmsnorm_post_attention` computes the pre-MLP residual stream
internally as `attention_output + layer_input`, but its output is the normalized
MLP input, so that residual stream is not otherwise stored in the activation
workspace. Passing the original layer input as a runtime tensor arg lets
`qwen_mlp_down` reconstruct the residual stream without inserting an extra
per-layer graph task.

## Evaluation Run

Focused regressions first failed on the missing `qwen_mlp_down.tensor_args[1]`
binding and the old generated source. They passed after the fix:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  -q -k 'test_launch_packet_binds_mlp_down_residual_source or test_generated_source_contains_qwen_unit_math_kernels'
```

Result: `2 passed, 44 deselected`.

Adjacent Qwen materialization, task-body, decode-loop, feedback, and review
artifact tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_decode_feedback.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: `72 passed`.

Generated Qwen task-body source compiled to PTX:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/qwen-mlp-residual-stream/qwen-persistent-task-bodies.json \
  --output-source tmp/cuda-backend/qwen-mlp-residual-stream/qwen-persistent-task-bodies.cu
/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/qwen-mlp-residual-stream/qwen-persistent-task-bodies.cu \
  -o tmp/cuda-backend/qwen-mlp-residual-stream/qwen-persistent-task-bodies.ptx
```

A one-step A100 resource-backed smoke that includes `qwen_mlp_down` passed:

```text
tmp/cuda-backend/qwen-mlp-residual-stream/qwen-decode-loop-runner.json
```

The artifact reports `resource_backed_execution.workloads[0].status = pass`,
eight task completions, and zero scheduler errors.

## Remaining Gaps

This narrows one full-serving correctness bug but does not close Qwen
paper-readiness. PTO rows still require token/logit agreement against the
Hugging Face Qwen/Qwen3-8B reference plus latency and throughput metrics for
both MPK and VDCores policies.
