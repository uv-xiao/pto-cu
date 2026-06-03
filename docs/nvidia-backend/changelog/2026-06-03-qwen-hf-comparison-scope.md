# 2026-06-03 Qwen HF Comparison Scope

## Code And Data Changed

- Added `examples/cuda/qwen_hf_token_comparison.py` and
  `qwen_decode_loop_runner_impl/hf_comparison.py` to generate the PTO-vs-HF
  token comparison artifact from committed runner outputs.
- Added a regression test that requires decode artifacts without prompt
  prefill to be marked non-model-equivalent.
- Regenerated
  `tmp/cuda-backend/qwen-activation-finiteness-mpk-2026-06-03/qwen-full-prefix-hf-token-comparison.json`
  with explicit comparison scope and blocking reasons.
- Updated the Qwen full-serving gap and paper queue wording to name
  `prompt_prefill_not_executed` separately from `token_mismatch`.

## Architecture Quality

The comparison helper prevents a diagnostic decode-position artifact from
being treated as Hugging Face model-equivalent evidence. A PTO row can still
fail token agreement, but the artifact now also records whether the prerequisite
prompt-prefill state existed before comparing against the Hugging Face
`last_active_prompt_token` reference.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -q \
  -k hf_token_comparison_marks_decode_without_prefill_non_equivalent
```

Failed with `ModuleNotFoundError` before the helper existed.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -q \
  -k hf_token_comparison_marks_decode_without_prefill_non_equivalent
```

Result: `1 passed, 6 deselected`.

The regenerated comparison reports:

- `status=fail`;
- `comparison_scope=diagnostic_decode_without_prompt_prefill`;
- `model_equivalent_ready=false`;
- `blocking_reasons=[prompt_prefill_not_executed, token_mismatch]`;
- PTO top token `220` versus Hugging Face top token `151667`.

## Remaining Gaps

Full-serving promotion still requires model-equivalent prompt prefill, correct
KV-cache state, token/logit agreement against the Hugging Face reference, and
MPK/VDCores policy rows with latency and throughput metrics.
