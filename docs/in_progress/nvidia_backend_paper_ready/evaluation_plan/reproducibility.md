# NVIDIA Backend Paper-Ready Evaluation Plan: Reproducibility Rules

## Reproducibility Rules

Each benchmark command writes JSON under `tmp/cuda-backend/` and can be indexed
into the viewer data. A paper result is not accepted unless it records:

- command;
- pto-cu commit;
- source baseline commit or release;
- hardware;
- CUDA toolkit and driver;
- input shape;
- repeat count and warmup count;
- raw output path;
- validation command.

Current PTO microbenchmark captures can be converted to viewer result records
with `.agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py`. The
script uses the sharded `evaluations/nvidia/benchmark-viewer/data/`
`capture_imports/` collection to map raw capture baselines onto viewer
benchmark and method IDs.
The host-launch A100 capture at
`tmp/cuda-backend/host-launch-a100-8b6cdaee/` now imports its 10-repeat
`direct_driver` and `direct_driver_graph` rows into `results.json`, giving the
host-schedule launch claim explicit raw CUDA Driver launch and CUDA Driver
graph evidence on A100 while H200 Driver rows, true Runtime API rows, and
tensor-shape graph rows remain open.
The follow-up A100 capture at
`tmp/cuda-backend/host-launch-runtime-a100-e429c07b/` imports a 10-repeat
`direct_runtime` row produced by an nvcc-built shared library that calls
`cudaLaunchKernel`.
The H200 host-launch capture at
`tmp/cuda-backend/host-launch-h200-ec8f272e/` imports 10-repeat
`direct_runtime`, `direct_driver`, and `direct_driver_graph` rows for the same
`n=1024` vector shape. This closes the cross-GPU vector-launch comparison gap
for Runtime API, Driver API, and Driver graph paths.
`cuda_viewer_export.py` now imports p50, p90, p99, mean, standard deviation,
minimum, and maximum host/device latency fields for repeated raw captures, so
the current 10-repeat A100/H200 host-launch rows expose distribution shape in
the viewer instead of selected medians alone.
The selected tensor-launch captures at
`tmp/cuda-backend/tensor-launch-a100-09462d04/` and
`tmp/cuda-backend/tensor-launch-h200-09462d04/` import 10-repeat
`direct_runtime`, `direct_driver`, and `direct_driver_graph` rows for the
`n=1024`, `16x16x16` naive SGEMM tensor shape. This closes the selected
tensor-launch comparison gap for the host-schedule launch claim. Stream-count
or graph-replay sweep distributions remain open.

Paper-baseline raw captures can be converted with
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_viewer_export.py`.
The script reads `paper_baseline_runs.json`, maps each raw
`paper_baseline_run_id` to the corresponding paper-baseline method, and emits
viewer `result_records` for MPK, VDCores, vLLM, SGLang, or ThunderKittens.
This keeps paper-baseline rows generated from raw artifacts instead of
hand-edited tables.
After a capture is accepted for committed viewer evidence,
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py`
updates the viewer results, marks the matching run imported, and regenerates
the paper-readiness audit from the same inputs.
Before executing planned MPK, VDCores, vLLM, or SGLang paper-baseline runs,
refresh `paper_baseline_run_readiness.json` with
`.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`.
The viewer then shows whether model access, entrypoints, reproduction-command
contracts, expected artifacts, paired probe status, and the VDCores
`dae.runtime` build are ready. Readiness rows are pre-run evidence only; the
paper claim remains blocked until measured raw JSON is imported.

Serving baseline commands can be materialized with
`.agents/skills/cuda-backend-eval/scripts/paper_serving_command_plan.py`.
The script reads `serving_workloads.json` plus `paper_baseline_runs.json` and
emits one command-plan row per baseline, policy, and batch size. The current
primary-model command plan is committed as
`evaluations/nvidia/benchmark-viewer/data/serving_command_plan.json` for
human review and is also regenerated under `tmp/cuda-backend/` before long
runs. The current command planner emits 35 rows covering MPK, VDCores, vLLM,
SGLang, and the ThunderKittens decode-attention serving-equivalent over the
MPK/VDCores policy batch ladders. This is not a performance result; it is the
reproducible launch contract that long H200 runs must execute before their raw
JSON can be imported into the viewer. SGLang launch rows explicitly prepend the
pinned source checkout to `PYTHONPATH`, so generated commands do not
accidentally use a globally installed SGLang package. ThunderKittens rows use
the bounded MHA capture wrapper as a controlled serving-family kernel baseline
for the VDCores decode policy.

The viewer also includes one PTO `persistent_device` controlled
serving-equivalent row for `vdcores_offline_decode`: a H200 tensor-core
persistent DAG capture mapped to batch 4, 128 prompt tokens, and 64 decode
tokens. This proves the PTO side has a reviewable serving-policy-shaped row,
but it is still only an attention-tile proxy. Full MPK, VDCores, vLLM, SGLang,
ThunderKittens-family, and end-to-end PTO serving artifacts remain required
before the LLM serving claim can be paper-ready.

Before full baseline builds, paper-baseline readiness probes can be captured
with `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`. The
probe checks pinned source commits, selected entrypoint paths, Python syntax,
required Python modules, CUDA toolkit availability, and visible GPUs. The
latest paired readiness artifact is recorded at
`tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-43b927ed/`.
Use `.agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py`
for paired A100/H200 readiness. In `--sync-remote-tree` mode it copies the
local checkout to the H200 host, skips remote Git, runs the remote probe with
the explicit CUDA toolkit path, and copies `h200-probe.json` back beside the
local `a100-probe.json`. The paired runner also syncs `tmp/baselines/`
separately, because those source checkouts are required probe inputs but
generated `tmp/cuda-backend/` outputs should not be copied as part of the repo
tree sync.
The 43b927ed paired probe checks the `transformers` module for MPK and
VDCores, matching the selected Qwen3 and Llama entrypoint imports. It also
checks that the selected SGLang benchmark modules import from the pinned
source checkout, not only that the source files exist. The H200 project venv
has `transformers` installed, so MPK, VDCores, and ThunderKittens remain
setup-ready on H200 while vLLM and SGLang remain partial. vLLM now has a
reviewed Python 3.10 source-overlay path that copies the pinned checkout under
`tmp/cuda-backend/paper-baselines/source-overlays/`, unsets
`Py_LIMITED_API` only for the copied spinloop CXX target, derives the package
version from the pinned upstream checkout, builds editable vLLM from the
overlay, and passes the selected local validation imports. SGLang is partial
because H200 is missing `orjson`, while the local A100 import path currently
hits a torch/torchvision operator-registration mismatch.
ThunderKittens readiness must include the selected PyTorch-extension
dependencies (`torch`, `pybind11`, `numpy`, `pandas`, `matplotlib`, and
`tqdm`), not just source-file existence. After installing those modules in
the H200 project venv, the current H200 probe is ready for the selected
ThunderKittens setup path. The selected official H100 MHA benchmark now has
an FA3-enabled H200 run as well: FlashAttention-3 was built from the
`tmp/baselines/flash-attention/hopper` source clone with a narrowed SM90 BF16
head-dim-128 build, and the unmodified ThunderKittens benchmark was run with a
local compatibility shim that requests `return_attn_probs=True` so the current
FA3 API returns the `(out, lse)` tuple expected by ThunderKittens. The FA3
rows completed for forward/backward, causal/non-causal, and sequence lengths
768, 1536, 3072, 6144, and 12288. A follow-up isolated PyTorch reference
capture ran each selected large reference cell in a fresh H200 process with
expandable allocator segments. It recovered every 6144-token cell and left only
12288-token dense PyTorch reference cells OOM, so the remaining official-sweep
capacity issue is recorded as an accepted evidence-policy exception instead of
a tensor-core blocker. The exception is scoped only to the infeasible dense
PyTorch reference rows; paper tables may footnote those rows as
OOM/not-applicable and must not impute performance or correctness numbers from
them.
The LLM-serving claim also has a planned ThunderKittens
`thunderkittens_decode_attention_tile` run record. Its readiness is generated
from the selected source tree, repo-local capture wrapper, expected tmp
artifact paths, required metrics, and paired ThunderKittens probe status, so
the audit no longer hides ThunderKittens behind a missing-run blocker. The
current H200 capture imports five VDCores-policy batch rows. The H100 MHA
wrapper pads the 64-token decode policy to `n=256` so the kernel launch grid
is nonzero; the raw and viewer rows still preserve `prompt_tokens=128` and
`decode_tokens=64` for the serving comparison.
The first bounded ThunderKittens MHA capture is recorded under
`tmp/cuda-backend/paper-baselines/thunderkittens/mha_h100-5915346e/`. It
uses `.agents/skills/cuda-backend-eval/scripts/thunderkittens_mha_capture.py`
to run two H200 causal BF16 MHA shapes with five warmups and twenty timed
CUDA-event repeats, compare against PyTorch scaled-dot-product attention, and
export viewer-compatible paper-baseline records. This promotes the selected
ThunderKittens bounded-capture row from setup-ready to imported viewer
evidence. The full upstream correctness and benchmark sweeps are tracked by
the separate `thunderkittens_full_sweep` run. Its expected artifacts now
include the bounded repo-owned sweep, the original official benchmark probe,
and the FA3-enabled official rerun under
`tmp/cuda-backend/paper-baselines/thunderkittens/upstream-benchmark-fa3-7371626c/`.
Imported paper-baseline run records are not allowed to keep missing future
artifacts in their `expected_artifacts` lists.

Remote H200 runs should prefer Git refresh when available and use SSH
tree-sync fallback when remote Git fails. The selected path is part of the
artifact metadata.

