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
  renderPlanHistoryInto(document.getElementById("plan-history"), state, {
    compact: true,
  });
  renderPlanHistoryInto(document.getElementById("plan-archive-history"), state, {
    compact: false,
  });
}

function renderPlanHistoryInto(root, state, { compact }) {
  if (!root) {
    return;
  }
  const history = state.planHistory;
  const recentFocus = history.work_focus[0];
  const nonFeatureTotal =
    recentFocus.tests_or_guardrails + recentFocus.viewer_or_docs;
  const focusTotal = Math.max(
    1,
    recentFocus.feature_or_runtime
      + nonFeatureTotal,
  );
  const nonFeatureRatio = Math.round((nonFeatureTotal / focusTotal) * 100);
  const reflectionStatus = nonFeatureRatio > 35
    ? "Reflection needed"
    : "Runtime-focused";
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

  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.append(
    metric("Runtime slices", recentFocus.feature_or_runtime),
    metric("Non-feature slices", nonFeatureTotal),
    metric("Non-feature share", `${nonFeatureRatio}%`),
    metric("Reflection status", reflectionStatus),
  );

  const section = document.createElement("section");
  section.className = "item";
  const title = document.createElement("h3");
  title.append(text("Recent Work Focus"));
  section.append(
    title,
    metrics,
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
  );
  if (compact) {
    section.append(table(
      ["Commit", "Focus", "Slice", "Reflection"],
      history.recent_slices.slice(0, 4).map((slice) => [
        slice.commit,
        slice.focus,
        slice.title,
        slice.reflection,
      ]),
    ));
  } else {
    section.append(
      planTimeline(history.recent_slices),
      table(
        ["Date", "Trigger", "Finding", "Decision"],
        history.reflection_log.map((entry) => [
          entry.date,
          entry.trigger,
          entry.finding,
          entry.decision,
        ]),
      ),
      fieldList([
        ["Cadence", history.next_reflection_check.cadence],
        [
          "Reporting-only action",
          history.next_reflection_check.preferred_action_if_reporting_only,
        ],
        ["Latest reviewed commit", history.latest_reviewed_commit],
      ]),
    );
  }
  root.replaceChildren(section);
}

function planTimeline(slices) {
  const list = document.createElement("ol");
  list.className = "plan-timeline";
  slices.forEach((slice) => {
    const item = document.createElement("li");
    item.className = `plan-slice ${slice.focus}`;
    const heading = document.createElement("strong");
    heading.append(text(`${slice.commit} · ${slice.title}`));
    const focus = document.createElement("span");
    focus.className = "pill";
    focus.append(text(slice.focus));
    const reflection = document.createElement("p");
    reflection.append(text(slice.reflection));
    item.append(heading, focus, reflection);
    list.append(item);
  });
  return list;
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
