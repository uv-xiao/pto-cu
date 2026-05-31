#!/usr/bin/env python3
"""Summarize CUDA persistent-device scheduler breakdown smoke artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _artifact_label(path: Path) -> str:
    name = path.stem.lower()
    if "a100" in name:
        return "a100"
    if "h200" in name:
        return "h200"
    return name


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            result.append(item)
    return result


def _policy_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = _as_int(payload.get(key), default)
    if value:
        return value
    policy = payload.get("resource_policy")
    if isinstance(policy, dict):
        return _as_int(policy.get(key), default)
    return default


def _completed_count(payload: dict[str, Any], processed_count: int) -> int:
    completed_counts = _int_list(payload.get("launch_completed_counts"))
    if completed_counts:
        return max(completed_counts)
    completed = _as_int(payload.get("completed_count"))
    if completed:
        return completed
    return processed_count


def _host_sync_overhead_ns(host_wall_ns: int, device_wall_ns: int) -> int:
    return max(0, host_wall_ns - device_wall_ns)


def _comma(values: list[int]) -> str:
    return ",".join(str(value) for value in values) if values else "-"


def _load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scheduler_processed_count = _as_int(payload.get("scheduler_processed_count"))
    completed_count = _completed_count(payload, scheduler_processed_count)
    device_wall_ns = _as_int(payload.get("device_wall_ns"))
    host_wall_ns = _as_int(payload.get("host_wall_ns"))
    processed_by_block = _int_list(payload.get("scheduler_processed_by_block"))
    dispatch_func_ids = _int_list(payload.get("dispatch_func_ids"))
    scheduler_blocks = _policy_int(payload, "scheduler_blocks")
    worker_blocks = _policy_int(payload, "worker_blocks")
    active_scheduler_count = sum(1 for value in processed_by_block if value > 0)

    return {
        "artifact": _artifact_label(path),
        "path": str(path),
        "status": str(payload.get("status", "unknown")),
        "runtime": str(payload.get("runtime", "unknown")),
        "mode": str(payload.get("mode", "unknown")),
        "dag_shape": str(payload.get("dag_shape", "-")),
        "n": _as_int(payload.get("n")),
        "ready_queue": {
            "scheduler_blocks": scheduler_blocks,
            "scheduler_loop_count": _as_int(payload.get("scheduler_loop_count")),
            "processed_count": scheduler_processed_count,
            "processed_by_block": processed_by_block,
            "active_scheduler_count": active_scheduler_count,
        },
        "worker_execution": {
            "worker_blocks": worker_blocks,
            "completed_task_count": completed_count,
            "dispatch_func_ids": dispatch_func_ids,
            "device_wall_ns": device_wall_ns,
            "device_ns_per_task": device_wall_ns // completed_count if completed_count > 0 else 0,
            "launch_device_wall_ns": _int_list(payload.get("launch_device_wall_ns")),
        },
        "host_synchronization": {
            "host_wall_ns": host_wall_ns,
            "host_sync_overhead_ns": _host_sync_overhead_ns(host_wall_ns, device_wall_ns),
            "launch_host_wall_ns": _int_list(payload.get("launch_host_wall_ns")),
        },
        "resource_policy": payload.get("resource_policy", {}),
        "device_scheduler_errors": payload.get("device_scheduler_errors", {}),
    }


def load_breakdown_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [_load_row(path) for path in paths]
    return sorted(rows, key=lambda row: (str(row["artifact"]), str(row["dag_shape"]), int(row["n"])))


def render_markdown_report(rows: list[dict[str, Any]], label: str) -> str:
    lines = [
        "# CUDA Scheduler Breakdown Report",
        "",
        f"- Label: `{label}`",
        "- `ready_queue` columns come from device scheduler counters.",
        "- `worker_execution` columns come from completed task counts and CUDA event timing.",
        "- `host_synchronization` is the nonnegative host wall minus device event delta.",
        "",
        (
            "| Artifact | DAG shape | Ready tasks | Processed by block | Device ns | "
            "Device ns/task | Host sync ns |"
        ),
        "| -------- | --------- | ----------- | ------------------ | --------- | -------------- | ------------ |",
    ]
    for row in rows:
        ready = row["ready_queue"]
        worker = row["worker_execution"]
        host = row["host_synchronization"]
        lines.append(
            f"| {row['artifact']} | {row['dag_shape']} | {ready['processed_count']} | "
            f"`{_comma(ready['processed_by_block'])}` | {worker['device_wall_ns']} | "
            f"{worker['device_ns_per_task']} | {host['host_sync_overhead_ns']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_svg_report(rows: list[dict[str, Any]], label: str) -> str:
    width = 860
    left = 230
    right = 40
    top = 76
    row_height = 64
    chart_width = width - left - right
    max_value = max(
        (int(row["worker_execution"].get("device_wall_ns", 0) or 0) for row in rows),
        default=1,
    )
    height = top + max(1, len(rows)) * row_height + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" font-weight="600">{html.escape(label)}</text>',
        (
            '<text x="24" y="56" font-family="sans-serif" font-size="12" fill="#555">'
            "ready_queue / worker_execution / host_synchronization breakdown</text>"
        ),
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        ready = row["ready_queue"]
        worker = row["worker_execution"]
        host = row["host_synchronization"]
        value = int(worker["device_wall_ns"])
        bar_width = int(chart_width * value / max_value) if max_value else 0
        label_text = f"{row['artifact']} {row['dag_shape']}"
        lines.extend(
            [
                f'<text x="24" y="{y + 17}" font-family="sans-serif" font-size="12">{html.escape(label_text)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="24" fill="#0369a1"/>',
                (
                    f'<text x="{left + bar_width + 8}" y="{y + 17}" '
                    f'font-family="sans-serif" font-size="12">{value} ns</text>'
                ),
                (
                    f'<text x="{left}" y="{y + 44}" font-family="sans-serif" font-size="11" fill="#555">'
                    f"ready_queue={ready['processed_count']} by_block={html.escape(_comma(ready['processed_by_block']))}; "
                    f"worker_execution={worker['completed_task_count']} tasks, {worker['device_ns_per_task']} ns/task; "
                    f"host_synchronization={host['host_sync_overhead_ns']} ns</text>"
                ),
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_breakdown_report(rows: list[dict[str, Any]], output_dir: Path, label: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cuda-scheduler-breakdown.json"
    markdown_path = output_dir / "cuda-scheduler-breakdown.md"
    svg_path = output_dir / "cuda-scheduler-breakdown.svg"
    json_path.write_text(json.dumps({"label": label, "rows": rows}, indent=2) + "\n")
    markdown_path.write_text(render_markdown_report(rows, label), encoding="utf-8")
    svg_path.write_text(render_svg_report(rows, label), encoding="utf-8")
    return json_path, markdown_path, svg_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--label", default="cuda-scheduler-breakdown")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_breakdown_rows(args.json_paths)
    for path in write_breakdown_report(rows, args.output_dir, args.label):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
