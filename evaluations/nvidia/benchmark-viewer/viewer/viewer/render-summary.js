import { fieldList, metric, table, text } from "./dom.js";

export function renderSnapshot(state) {
  const snapshot = state.results.snapshot;
  const root = document.getElementById("snapshot");
  root.replaceChildren(
    metric("Commit", snapshot.commit),
    metric("Full samples", snapshot.full_capture.samples),
    metric("Compact samples", snapshot.compact_capture.samples),
    metric("Tensor tile", snapshot.tensor_tile),
    metric("GPU targets", snapshot.gpu_targets.join(", ")),
  );
}

export function renderPlanHistory(state) {
  const root = document.getElementById("plan-history");
  const history = state.planHistory;
  const recentFocus = history.work_focus[0];
  const section = document.createElement("section");
  section.className = "item";
  const title = document.createElement("h3");
  title.append(text("Recent Work Focus"));
  section.append(
    title,
    fieldList([
      ["Current focus", history.summary.current_focus],
      ["Recent pattern", history.summary.recent_pattern],
      ["Reflection", history.summary.reflection],
      ["Test strategy", history.summary.test_strategy],
      [
        "Latest 12 commits",
        `${recentFocus.feature_or_runtime} feature/runtime, `
          + `${recentFocus.tests_or_guardrails} tests/guardrails, `
          + `${recentFocus.viewer_or_docs} viewer/docs`,
      ],
      ["Next check", history.next_reflection_check.question],
    ]),
  );
  root.replaceChildren(section);
}

export function renderHeadlineResults(state) {
  const root = document.getElementById("headline-results");
  const rows = state.results.headline_results;
  root.replaceChildren(table(
    ["GPU", "N", "Method", "Host ns", "Device ns", "Status"],
    rows.map((row) => [
      row.gpu,
      row.n,
      row.method,
      row.host_wall_ns,
      row.device_wall_ns,
      row.status,
    ]),
  ));
}

export function renderResults(state, lookup) {
  const root = document.getElementById("result-table");
  const p90 = (statistic, key) => statistic[key] ?? "-";
  root.replaceChildren(table(
    [
      "GPU",
      "Benchmark",
      "Method",
      "Inputs",
      "Samples",
      "Host ns",
      "Host p90 ns",
      "Device ns",
      "Device p90 ns",
      "Correctness",
      "Coverage",
      "Raw artifact",
    ],
    state.results.result_records.map((row) => [
      `${row.hardware.gpu} / ${row.hardware.machine} / ${row.hardware.compute_target}`,
      lookup.benchmarkTitle(row.benchmark_id),
      lookup.methodName(row.method_id),
      `${row.inputs.shape}; ${row.inputs.dtype}; ${row.inputs.repeat_policy}`,
      row.statistic.sample_count,
      row.statistic.host_wall_ns,
      p90(row.statistic, "host_wall_p90_ns"),
      row.statistic.device_wall_ns,
      p90(row.statistic, "device_wall_p90_ns"),
      row.correctness,
      row.statistic.serving_coverage || "-",
      row.raw_artifact,
    ]),
  ));
}
