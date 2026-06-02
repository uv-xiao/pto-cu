# Paper Readiness Serving Coverage Guard

## Code And Data Changed

- Updated paper-readiness audit result matching to include
  `statistic.serving_coverage`.
- Added focused tests proving diagnostic PTO Qwen rows cannot satisfy a
  future `full_serving` viewer-result evidence ref.

## Architecture Quality

Paper-readiness generation now shares the serving-coverage precision already
required by the paper matrix validator. This keeps diagnostic
`diagnostic_resource_backed_qwen_dag` PTO evidence from being promoted as
full-serving evidence if a future matrix ref has the same shape substring.

## Evaluation Run

- `test_nvidia_paper_readiness_audit.py` passed with pytest plugin autoload
  disabled.
- `validate_benchmark_viewer_data.py` passed and confirmed the generated
  paper-readiness audit still matches committed viewer data.

## Remaining Gaps

PTO Qwen full-serving remains open. The guard prevents accidental promotion;
it does not replace the missing numerically correct full decode-loop row.
