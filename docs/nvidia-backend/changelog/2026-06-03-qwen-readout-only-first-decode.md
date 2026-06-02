# Qwen Readout-Only First Decode

## Code And Data Changed

- Added packet index offsets for resource-backed launch-packet construction.
- Added readout-only descriptor selection for prompt-prefilled Qwen runs.
- After prompt prefill, decode step 0 now runs only `qwen_final_norm` and
  `qwen_logits` against the hidden activation produced by prompt replay.
- Fixed terminal output routing so only terminal readout callables write to the
  logits buffer; a sliced prefill packet's last non-logits task stays in the
  activation chain.
- Added focused tests for descriptor splitting and offset activation binding.

## Architecture Quality

The CUDA persistent-device ABI is unchanged. The host builds smaller phase
packets from the same generated task descriptors and device task functions.
This matches serving semantics more closely: prompt replay populates KV and
the last prompt hidden state, while first-token sampling only needs readout
from that hidden state.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `55 passed`.

A100 first-layer prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prefill-readout-only-first-decode-fixed-2026-06-03/
```

The artifact reports 18 prompt positions, 144 completed prefill tasks, and a
2-task readout-only decode packet with zero scheduler errors. The logits buffer
had nonzero values, diagnostic logits reference status `pass`, and device
token feedback wrote token `67291` to the next input slot.

A100 full-prefix prompt-prefill smoke passed:

```text
tmp/cuda-backend/qwen-prefill-readout-only-full-2026-06-03/
```

The artifact reports 18 prompt positions, 4554 completed prefill tasks, and a
2-task first decode packet at offset 253 with zero scheduler errors. The
diagnostic logits reference passed and device token feedback wrote token
`116324` to the next input slot.

## Remaining Gaps

- This is still diagnostic resource-backed execution, not paper-ready full
  Qwen numerical correctness against a Hugging Face reference.
- The prompt-prefill task bodies remain scalar/diagnostic and are far too slow
  for paper-ready serving throughput.
