# CUDA Backend Status: Evaluation And Reporting Part 3

- [evaluation.md](evaluation.md) is the evaluation landing page.
- [evaluation-current.md](evaluation-current.md) summarizes the latest paired
  A100/H200 capture.
- [benchmark viewer](../../../../evaluations/nvidia/benchmark-viewer/viewer/index.html)
  is the static human-review viewer backed by committed JSON data.
- [history/index.md](history/index.md) preserves earlier captures.
- [changelog/index.md](changelog/index.md) records review-facing change
  reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py` writes JSON,
  Markdown, and SVG reports.
- `.agents/skills/cuda-backend-eval/scripts/cuda_smoke_report.py` writes
  compact smoke Markdown and SVG reports, including persistent-device dispatch
  `func_id` sequences, device scheduler error counters, repeat-run lifecycle
  counters, resource-policy metadata, tensor-core metadata, and scalar and
  tensor task arguments when present.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_benchmark.py` automates
  the local A100 run, remote H200 run, artifact copy, merge, command-example
  metadata capture, combined-artifact validation, and index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_smoke.py` automates the
  no-torch host-schedule Worker smoke on local A100 and remote H200, then
  renders the compact smoke report and refreshes the artifact index.
- `.agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py`
  automates no-torch persistent-device DAG smoke captures on local A100 and
  remote H200, including optional remote tree sync, tensor-tile descriptor
  flags, compact report rendering, smoke artifact validation, and
  artifact-index refresh.
- `.agents/skills/cuda-backend-eval/scripts/cuda_tensor_shape_sweep.py`
  automates paired A100/H200 tensor baseline sweeps over model-shaped tensor
  descriptors, records VDCores/MPK provenance and sanitized command examples
  in generated metadata, and writes JSON, Markdown, and SVG artifacts under
  `tmp/cuda-backend/`.
- `.agents/skills/cuda-backend-eval/scripts/cuda_current_summary.py` renders
  the compact benchmark tables, selected benchmark tensor-throughput table,
  and compact tensor-sweep median table used by
  [evaluation-current.md](evaluation-current.md) from raw JSON artifacts,
  including graph scratch-reuse ratios in the DAG-shapes table, explicit
  graph descriptor dispatch/fan-in/task-argument metadata plus scalar/tensor
  descriptor argument maps in the graph-metadata table, and cuBLAS Graph
  replay columns in tensor-sweep summaries.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`
  checks paired benchmark captures for expected machines, selected baselines,
  sizes, repeats, sample count, generated report files, source-paper
  metadata, sanitized command examples, dispatch IDs, tensor-tile shapes, and
  graph descriptor, graph task-argument, and scratch-reuse metadata before
  docs are refreshed. Current paired presets also require visible graph
  topology and task-argument metadata in `cuda-benchmark.md` and
  `cuda-benchmark.svg`.
- `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py` checks
  paired smoke captures for required A100/H200 artifacts, pass status, zero
  scheduler errors, expected runtime/mode, dispatch IDs, repeat-run lifecycle
  counts, tensor-tile descriptor shape, generated smoke report files, and
  visible report graph topology and graph task-argument metadata when
  requested.
- `.agents/skills/cuda-backend-eval/scripts/cuda_artifact_index.py` indexes
  local `tmp/cuda-backend/` benchmark, tensor-shape sweep, lifecycle matrix,
  and smoke artifacts, including tensor-tile shapes, persistent smoke modes,
  lifecycle scenarios, dispatch sequences, scheduler error counters,
  repeat-run counts, per-launch completion counts, graph descriptor
  fan-in/dependent arrays, graph task-argument keys, and graph task-argument
  metadata.
- `.agents/skills/cuda-backend-eval/SKILL.md` documents the current paired
  A100/H200 recipe.
