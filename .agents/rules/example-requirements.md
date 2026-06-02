# Example Requirements Rules

- Examples are part of the public review surface.
- `examples/cuda/manifest.json` is the source of truth for review-facing CUDA
  examples. Keep this catalog small and limited to representative, essential,
  end-to-end cases.
- CUDA examples should stay runnable from the repository root with
  `PYTHONPATH=$PWD:$PWD/python`.
- CUDA examples should use the same evaluated smoke paths or public runtime
  surfaces that the benchmark docs describe.
- Do not add narrow Qwen lifecycle probes as new review-facing examples. Keep
  them as support code for the advanced Qwen decode-loop example unless the
  user explicitly asks for a standalone case.
- Do not create a second example framework for CUDA unless the user asks for
  it explicitly.
- Keep example README commands synchronized with smoke scripts and evaluation
  viewer commands.
