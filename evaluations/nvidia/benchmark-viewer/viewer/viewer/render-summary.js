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
  const focusTotal = Math.max(
    1,
    recentFocus.feature_or_runtime
      + recentFocus.tests_or_guardrails
      + recentFocus.viewer_or_docs,
  );
  const focusBar = document.createElement("div");
  focusBar.className = "focus-bar";
  [
    ["feature", "Feature/runtime", recentFocus.feature_or_runtime],
    ["tests", "Tests/guardrails", recentFocus.tests_or_guardrails],
    ["docs", "Viewer/docs", recentFocus.viewer_or_docs],
  ].forEach(([kind, label, value]) => {
    const segment = document.createElement("span");
    segment.className = `focus-segment ${kind}`;
    segment.style.width = `${(value / focusTotal) * 100}%`;
    segment.title = `${label}: ${value}`;
    segment.setAttribute("aria-label", `${label}: ${value}`);
    focusBar.append(segment);
  });

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
    focusBar,
    table(
      ["Commit", "Focus", "Slice", "Reflection"],
      history.recent_slices.map((slice) => [
        slice.commit,
        slice.focus,
        slice.title,
        slice.reflection,
      ]),
    ),
    table(
      ["Date", "Finding", "Decision"],
      history.reflection_log.map((entry) => [
        entry.date,
        entry.finding,
        entry.decision,
      ]),
    ),
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
