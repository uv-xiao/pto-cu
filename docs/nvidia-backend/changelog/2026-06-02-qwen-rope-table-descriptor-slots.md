# 2026-06-02 Qwen RoPE Table Descriptor Slots

## Code And Data Changed

- Added runtime-generated `rope_cos_table` and `rope_sin_table` records to
  `qwen_attention_qk_norm` persistent weight descriptors.
- Kept validated Qwen weights counted through resident weight slots only, so
  RoPE tables do not appear as missing safetensor weights.
- Updated persistent weight materialization to preserve those runtime-generated
  tensor args as `requires_live_pointer` requirements.

## Architecture Quality

The branch now connects the generated QK RoPE source contract to descriptor
slots `tensor_args[2]` and `tensor_args[3]`. This still does not bind live
cos/sin table device pointers in the decode-loop launcher; that gap is now
represented explicitly in materialization output.

## Evaluation Run

- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q -k 'persistent_qwen_weight_args or persistent_qwen_weight_materialization'`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_weight_args.py --output-json tmp/cuda-backend/pto-serving-weight-args/qwen-persistent-weight-args-rope-slots-full.json`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_weight_materialization.py --weight-args-json tmp/cuda-backend/pto-serving-weight-args/qwen-persistent-weight-args-rope-slots-full.json --output-json tmp/cuda-backend/pto-serving-weight-materialization/qwen-persistent-weight-materialization-rope-slots-plan.json`.

## Remaining Gaps

PTO full-serving rows still require live RoPE table allocation and pointer
binding, decode attention reduction, complete decode-loop execution, and
viewer import for both MPK-policy and VDCores-policy workloads.
