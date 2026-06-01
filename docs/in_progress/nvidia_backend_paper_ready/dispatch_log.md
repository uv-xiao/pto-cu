# NVIDIA Backend Paper-Ready Dispatch Log

This log records dispatcher-worker activity for the standalone pto-cu NVIDIA
backend goal. It is required review evidence; do not rely on private terminal
scrollback or unstated session memory. The active page is intentionally short;
historical entries are split under
[dispatch_log/index.md](dispatch_log/index.md).

## Logging Schema

Each entry must include:

- timestamp;
- dispatcher session or PR;
- worker id and objective;
- exact Codex command or script invocation;
- parent goal and child slice;
- branch name and PR URL or planned PR slot;
- allowed scope and files;
- dependencies and blocked assumptions;
- verification commands and results;
- merge decision and merge commit, when applicable;
- handoff summary and remaining gaps.


## Current Review Path

- Add new dispatch entries to a small dated file under
  `dispatch_log/entries/` and link it from `dispatch_log/index.md`.
- Keep each entry file under 300 lines; split by date and part when a day
  has many slices.
- Keep raw generated artifacts under `tmp/`; commit only the review
  summary, changelog, and structured viewer data.

## Latest Entries

- [2026-06-01-part-11.md](dispatch_log/entries/2026-06-01-part-11.md) (6 entries): Qwen Resident Weight Table Owner through Qwen KV-Cache Binding.
- [2026-06-01-part-12.md](dispatch_log/entries/2026-06-01-part-12.md) (6 entries): Qwen Decode Loop Runner Plan through Qwen Microdecode Live CUDA Execution.
- [2026-06-01-part-13.md](dispatch_log/entries/2026-06-01-part-13.md) (9 entries): Qwen Proxy Decode Loop Live Reuse through Qwen Unit Math Prepared Reuse.
- [2026-06-01-part-14.md](dispatch_log/entries/2026-06-01-part-14.md) (2 entries): Decode Loop Unit Math Bridge through Decode Loop Token Owner Live.
- [2026-06-02-part-01.md](dispatch_log/entries/2026-06-02-part-01.md) (1 entry): Decode Loop KV Owner Live.

## Full Archive

See [dispatch_log/index.md](dispatch_log/index.md) for all entries.
