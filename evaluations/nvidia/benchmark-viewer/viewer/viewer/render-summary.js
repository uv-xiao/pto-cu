import { metric, table } from "./dom.js";

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
