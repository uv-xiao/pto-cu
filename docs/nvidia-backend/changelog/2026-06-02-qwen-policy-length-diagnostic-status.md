# Qwen Policy-Length Diagnostic Status

## Code And Data Changed

- Updated PTO full-serving gap text to recognize imported policy-length
  MPK and VDCores diagnostic decode runs.
- Kept that row explicitly scoped to
  `diagnostic_resource_backed_qwen_dag`, not `full_serving`.
- Updated the resource-backed matrix helper so future imports keep the same
  language.

## Architecture Quality

The paper-readiness status now separates two facts that were previously
blurred: the shared policy token-loop lengths have diagnostic CUDA execution
rows, but PTO still lacks numerically correct Qwen kernels and full-serving
viewer rows.

## Evaluation Run

- Added a focused viewer-data guard for the MPK 1024-step diagnostic row;
  the targeted Qwen viewer tests passed.

## Remaining Gaps

PTO Qwen rows remain diagnostic. They cannot be promoted until full Qwen
numeric kernels and shared-policy full-serving result rows are imported.
