# 2026-06-01 Qwen Prompt Accounting

## Code And Data Changed

- Added `examples/cuda/qwen_prompt_accounting.py`, an executable
  tokenizer-accounting artifact for PTO persistent-device Qwen/Qwen3-8B
  serving.
- Wired prompt accounting into `persistent_qwen_serving_scaffold.py` and
  `pto_serving_preflight.py`.
- Added the prompt-accounting example to `examples/cuda/manifest.json` and
  `examples/cuda/README.md`.
- Refreshed paper-readiness data so the LLM-serving PTO work item points at
  the tokenizer artifact alongside the lifecycle-plan, scaffold, and preflight
  artifacts.

## Architecture Quality

The PTO full-serving blocker now separates tokenizer evidence from runtime
decode execution. The prompt-accounting artifact records the Qwen tokenizer
class, chat-template availability, observed prompt-token counts, target prompt
counts, target deltas, and whether padding or prompt regeneration is required
for the MPK and VDCores serving policies.

This remains partial: token IDs are not yet bound into CUDA runtime buffers,
and the decode loop does not consume those IDs.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_prompt_accounting.py \
    --mode offline \
    --output-json tmp/cuda-backend/pto-serving-tokenizer-b95ff321/qwen-prompt-accounting.json
```

Result: `status=pass`, `tokenizer_class=Qwen2TokenizerFast`, and both
`mpk_offline_decode` and `vdcores_offline_decode` record
`chat_prompt_tokens=18`. The MPK policy target is 64 prompt tokens, and the
VDCores policy target is 128 prompt tokens, so both policies still require a
documented padding or prompt-regeneration step before full-serving import.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-b95ff321/pto-serving-preflight.json
```

Result: `status=partial`, with `qwen_prompt_accounting=pass` and full
Qwen/Qwen3-8B serving rows still absent.

## Remaining Gaps

- Bind tokenizer output IDs and prompt-padding or regeneration policy into the
  runtime decode loop.
- Implement safetensors loading, CUDA allocation and binding, generated Qwen
  kernel bodies, decode-loop execution, and viewer-result import.
- Import persistent-device Qwen/Qwen3-8B full-serving rows for
  `mpk_offline_decode` and `vdcores_offline_decode`.
