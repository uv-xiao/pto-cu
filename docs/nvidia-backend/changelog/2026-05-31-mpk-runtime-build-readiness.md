# 2026-05-31 MPK Runtime Build Readiness

## Code And Data Changed

- Built the pinned MPK checkout as an editable package on the local A100 host
  and the remote H200 host, using each host's own CUDA, CMake cache, and Python
  ABI-specific `mirage.core` extension.
- Updated `paper_baseline_pair_probe.py` so remote baseline source sync excludes
  generated `build/`, egg-info, Python cache, `.pyc`, and `*.cpython-*.so`
  artifacts.
- Updated the CUDA evaluation skill and shared contracts with the per-host MPK
  build rule and the required setup dependency on `cuda-python`.
- Re-ran the paired A100/H200 paper-baseline probe under
  `tmp/cuda-backend/paper-baselines/probes/paired-a100-h200-b138cfc5/`.
- Refreshed benchmark-viewer data so MPK source entrypoints and MPK run
  readiness are no longer listed as blocking setup gaps.

## Architecture Quality

MPK readiness now treats native build outputs as host-local artifacts instead
of portable source files. This avoids cross-host contamination from local A100
Python 3.10 extensions or CMake caches when preparing the remote H200
environment, while still letting the paired probe synchronize source checkouts
under `tmp/baselines/`.

The review queue now distinguishes setup readiness from measured evidence:
MPK can import the persistent-kernel runtime entrypoint on both machines, but
paper-grade claims still require executing the MPK run commands and importing
raw latency, throughput, correctness, and scheduler-trace artifacts.

## Evaluation Run

Validated H200 import after the remote editable build:

```text
OK mirage
OK mirage.core
OK mirage.mpk.base_dynamic_shard_loader
core dtype attrs ['float8_e4m3', 'float16', 'bfloat16']
```

Then reran the paired paper-baseline probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_pair_probe.py \
  --sync-remote-tree
```

Result:

```text
MPK source entrypoints: pass on A100 and H200
```

The refreshed work queue now reports 9 blocking work items, down from 13,
because MPK probe and run-readiness blockers were removed.

## Remaining Gaps

- MPK raw benchmark outputs have not yet been executed and imported into
  viewer results.
- vLLM and SGLang source entrypoint probes remain partial.
- VDCores run-readiness still needs execution/imported result evidence for
  paper-grade persistent-device and serving comparisons.
