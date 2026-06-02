# Qwen Prefill Two-Step Decode

## Code And Data Changed

No code changed in this slice. This report records an A100 evidence run for the
current Qwen live-session lifecycle:

1. replay prompt prefill without final readout tasks;
2. run first decode as a readout-only packet over the last prompt hidden state;
3. feed the sampled token back into the live input buffer;
4. run the next decode step as the full selected DAG.

## Architecture Quality

The run exercises the intended lifecycle boundary after the recent prompt
prefill and first-readout split. It confirms that the live session can reuse
prefilled hidden state for first-token readout, commit the sampled token into
the input buffer, and then return to a full selected-DAG decode packet.

## Evaluation Run

A100 first-layer live-session smoke passed:

```text
tmp/cuda-backend/qwen-prefill-two-step-first-layer-2026-06-03/
```

The artifact reports:

- prompt prefill executed 18 positions, 144 tasks, and zero scheduler errors;
- decode step 0 ran a 2-task `prefill_reused_hidden_readout_only` packet;
- step 0 sampled token `67291` and wrote it into input slot 18;
- decode step 1 ran a 10-task `full_selected_dag` packet at position 18;
- step 1 sampled token `1280` and wrote it into input slot 19;
- both decode packets reported zero scheduler errors.

This confirms the basic live-session transition from prompt prefill to
readout-only first decode to normal decode.

## Remaining Gaps

This is still diagnostic evidence. It does not prove full Qwen numerical
correctness against Hugging Face and it is not a performance-quality kernel
path. The next implementation target is still replacing scalar diagnostic task
bodies with paper-ready kernels and validating end-to-end Qwen token matching.
