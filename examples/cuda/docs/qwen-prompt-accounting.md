# CUDA Examples: Qwen Prompt Accounting

## Qwen Prompt Accounting

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_prompt_accounting.py \
  --mode offline \
  --output-json tmp/cuda-backend/pto-serving-tokenizer/qwen-prompt-accounting.json
```

Expected output: command exits 0; output JSON records tokenizer class,
chat-template status, observed prompt-token counts, target deltas, and whether
padding or prompt regeneration is required for the MPK and VDCores serving
policies.

Offline mode requires the Qwen tokenizer to be available in the local cache.

Use `--mode download` only when intentionally capturing tokenizer evidence
from Hugging Face into `tmp/`. Use `--mode mock` for dependency-free local
contract checks; mock output must not be imported as paper evidence.

