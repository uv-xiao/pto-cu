# Compact Attempt Guard

## Code And Data Changed

- Updated `test_benchmark_viewer_has_json_backed_review_data` so it no longer
  requires a long list of historical paper-baseline execution-attempt IDs.
- The test now validates the compact current attempt set by checking:
  - every attempt has a unique ID;
  - each attempt references a declared paper baseline and paper baseline run;
  - all five baseline families still have execution-attempt evidence;
  - representative current MPK, VDCores, vLLM, SGLang, and ThunderKittens
    attempts remain present;
  - paper-readiness audit execution-attempt statuses and next actions point to
    existing attempt IDs.

## Architecture Quality

This keeps the review guard aligned with compact evaluation data. The guard now
protects the current evidence graph instead of forcing old diagnostic rows back
into committed viewer data, which supports the policy that raw or historical
evaluation artifacts stay under `tmp/` or history notes rather than inflating
the review payload.

## Evaluation Run

Focused validation passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py -q -k \
  benchmark_viewer_has_json_backed_review_data
```

## Remaining Gaps

This does not add new paper-grade benchmark results. It removes stale pressure
to re-commit historical diagnostic rows while keeping the current evidence
references checkable.
