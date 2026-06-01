# 2026-06-01 Qwen CUDA Token Buffer Binding

## Code And Data Changed

- Added `examples/cuda/qwen_cuda_token_buffer_binding.py`, which maps the
  target-length `input_ids`, `attention_mask`, and `output_ids` plans into
  CUDA device-buffer descriptors.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured live token-buffer copy evidence at
  `tmp/cuda-backend/pto-serving-token-buffer-2026-06-01/`
  `qwen-cuda-token-buffer-binding.json`.

## Architecture Quality

The token-buffer binding keeps tokenizer and shape policy in
`qwen_runtime_input_binding.py`, then makes CUDA allocation/copy a separate
runtime boundary. The live probe allocates the six paper-policy token buffers,
copies host data into CUDA memory, verifies copy-back, and frees the buffers in
one owner scope.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_cuda_token_buffer_binding.py \
  --output-json \
  tmp/cuda-backend/pto-serving-token-buffer-2026-06-01/\
qwen-cuda-token-buffer-binding.json
```

Result: `status=cuda_token_buffer_binding_ready`; the CUDA probe copied and
verified six buffers, 94,208 bytes total, covering MPK `[16,64]`
`input_ids`/`attention_mask`, MPK `[16,1024]` `output_ids`, VDCores `[16,128]`
`input_ids`/`attention_mask`, and VDCores `[16,64]` `output_ids`.

## Remaining Gaps

- Persistent decode argument binding is now tracked by
  [2026-06-01 Qwen persistent decode arguments](2026-06-01-qwen-persistent-decode-args.md);
  the decode-loop runner still needs to keep a live token pointer table open.
- Bind real CUDA KV-cache buffers, run the resident weight table in
  `cuda_live` mode through DAG submission, generate Qwen kernels, execute the
  decode loop, and import full-serving viewer rows for `Qwen/Qwen3-8B`.
