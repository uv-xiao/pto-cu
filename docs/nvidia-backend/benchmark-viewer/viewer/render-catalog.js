import {
  commandBlock,
  evidenceList,
  fieldList,
  namedList,
  paragraph,
  table,
  text,
} from "./dom.js";

export function renderBenchmarks(state) {
  const root = document.getElementById("benchmark-list");
  root.replaceChildren(...state.benchmarks.benchmarks.map((benchmark) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(benchmark.title));
    const desc = document.createElement("p");
    desc.append(text(benchmark.description));
    const inputs = fieldList([
      ["Shape", benchmark.run.inputs.shape],
      ["Dtype", benchmark.run.inputs.dtype],
      ["Repeat policy", benchmark.run.inputs.repeat_policy],
    ]);
    const code = document.createElement("pre");
    const codeText = document.createElement("code");
    codeText.append(text(benchmark.code));
    code.append(codeText);
    const run = document.createElement("h3");
    run.append(text("Run"));
    const evidence = document.createElement("h3");
    evidence.append(text("Evidence"));
    details.append(
      summary,
      desc,
      inputs,
      paragraph("Math", benchmark.math),
      code,
      run,
      commandBlock(benchmark.run.command),
      evidence,
      evidenceList(benchmark.evidence_refs),
    );
    return details;
  }));
}

export function renderMethods(state) {
  const root = document.getElementById("method-list");
  root.replaceChildren(...state.methods.methods.map((method) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(method.name));
    const metadata = fieldList([
      ["Category", method.category],
      ["Launch model", method.launch_model],
    ]);
    details.append(
      summary,
      metadata,
      paragraph("Runtime flow", method.runtime_flow),
      paragraph("Lifecycle mapping", method.lifecycle),
      evidenceList(method.evidence_refs),
    );
    return details;
  }));
}

export function renderSceneBuilderCoverage(state) {
  const root = document.getElementById("scene-builder-coverage");
  const metadata = state.sceneBuilderCoverage.metadata;
  const heading = document.createElement("h3");
  heading.append(text(metadata.title));
  const intro = paragraph("Status", `${metadata.status}: ${metadata.summary}`);
  const groups = state.sceneBuilderCoverage.coverage_groups.map((group) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(`${group.title} (${group.status})`));
    details.append(
      summary,
      paragraph("Summary", group.summary),
      table(["Covered Builder"], group.covered_builders.map((item) => [item])),
      ...namedList("Open Work", group.open_work),
      ...namedList(
        "Evidence",
        group.evidence_refs.map((ref) => (
          `${ref.path}: ${ref.symbols.join(", ")}`
        )),
      ),
    );
    return details;
  });
  root.replaceChildren(heading, intro, ...groups);
}

export function renderPersistentSchedulerCoverage(state) {
  const root = document.getElementById("persistent-scheduler-coverage");
  const metadata = state.persistentSchedulerCoverage.metadata;
  const heading = document.createElement("h3");
  heading.append(text(metadata.title));
  const intro = paragraph("Status", `${metadata.status}: ${metadata.summary}`);
  const groups = state.persistentSchedulerCoverage.coverage_groups.map(
    (group) => {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.append(text(`${group.title} (${group.status})`));
      details.append(
        summary,
        paragraph("Summary", group.summary),
        table(["Covered Case"], group.covered_cases.map((item) => [item])),
        ...namedList("Open Work", group.open_work),
        ...namedList(
          "Evidence",
          group.evidence_refs.map((ref) => (
            `${ref.path}: ${ref.symbols.join(", ")}`
          )),
        ),
      );
      return details;
    },
  );
  root.append(heading, intro, ...groups);
}

export function renderTensorWorkloadCoverage(state) {
  const root = document.getElementById("tensor-workload-coverage");
  const metadata = state.tensorWorkloadCoverage.metadata;
  const heading = document.createElement("h3");
  heading.append(text(metadata.title));
  const intro = paragraph("Status", `${metadata.status}: ${metadata.summary}`);
  const targets = state.tensorWorkloadCoverage.model_shape_targets || [];
  const targetRows = targets.map((target) => [
    target.title,
    `${target.tensor_tile.rows}x${target.tensor_tile.cols}x${target.tensor_tile.inner}`,
    target.status,
    target.import_smoke
      ? `${target.import_smoke.status}: ${target.import_smoke.artifact_root}`
      : "none",
    target.model_mapping,
  ]);
  const targetDetails = document.createElement("details");
  const targetSummary = document.createElement("summary");
  targetSummary.append(text("Model Shape Targets"));
  targetDetails.append(
    targetSummary,
    table(["Target", "Tile", "Status", "Import Smoke", "Mapping"], targetRows),
    ...namedList("Commands", targets.map((target) => target.run_command)),
    ...namedList(
      "Import Smoke Scope",
      targets
        .filter((target) => target.import_smoke)
        .map((target) => target.import_smoke.scope),
    ),
  );
  const groups = state.tensorWorkloadCoverage.coverage_groups.map((group) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const resultRefs = group.result_refs.map((ref) => (
      `${ref.benchmark_id}/${ref.method_id}/${ref.gpu}/${ref.shape_contains}`
    ));
    summary.append(text(`${group.title} (${group.status})`));
    details.append(
      summary,
      paragraph("Summary", group.summary),
      table(["Covered Case"], group.covered_cases.map((item) => [item])),
      ...namedList("Result Refs", resultRefs.length ? resultRefs : ["none"]),
      ...namedList("Open Work", group.open_work),
      ...namedList(
        "Evidence",
        group.evidence_refs.map((ref) => (
          `${ref.path}: ${ref.symbols.join(", ")}`
        )),
      ),
    );
    return details;
  });
  root.append(heading, intro, targetDetails, ...groups);
}

export function renderServingWorkloads(state, lookup) {
  const root = document.getElementById("serving-list");
  root.replaceChildren(...state.servingWorkloads.serving_workloads.map((workload) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(`${workload.title} (${workload.status})`));
    const metadata = fieldList([
      ["Paper source", workload.paper_source.paper],
      ["Primary model", workload.model_policy.primary_model],
      ["Bring-up model", workload.model_policy.bringup_model],
      ["Fallback model", workload.model_policy.fallback_model],
      ["Prompt target", workload.prompt_policy.target_prompt_tokens],
      ["Decode tokens", workload.decode_policy.decode_tokens],
      ["Batch sizes", workload.decode_policy.batch_sizes.join(", ")],
      ["Traffic mode", workload.decode_policy.traffic_mode],
      ["Hardware", workload.hardware_targets.join(", ")],
      [
        "Baseline runs",
        workload.baseline_run_ids
          .map(lookup.paperBaselineRunTitle)
          .join(", "),
      ],
    ]);
    const evidence = workload.evidence_refs.map((ref) => (
      `${ref.path}: ${ref.symbols.join(", ")}`
    ));
    details.append(
      summary,
      metadata,
      paragraph("Notes", workload.paper_source.notes),
      paragraph("Selection reason", workload.model_policy.selection_reason),
      paragraph("Tokenization rule", workload.prompt_policy.tokenization_rule),
      paragraph("Generation mode", workload.decode_policy.generation_mode),
      ...namedList("Required Metrics", workload.required_metrics),
      ...namedList("Current Blockers", workload.current_blockers),
      ...namedList("Evidence", evidence),
    );
    return details;
  }));
}

export function renderServingCommandPlan(state, lookup) {
  const root = document.getElementById("serving-command-list");
  const metadata = state.servingCommandPlan.metadata;
  const heading = document.createElement("h3");
  heading.append(text("Serving Command Plan"));
  const metadataFields = fieldList([
    ["Commit", metadata.pto_commit],
    ["Model tier", metadata.model_tier],
    ["Artifact root", metadata.artifact_root],
    ["Source files", metadata.source_files.join(", ")],
  ]);
  const records = state.servingCommandPlan.serving_command_plans.map((plan) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(
      `${lookup.paperBaselineRunTitle(plan.paper_baseline_run_id)} / `
      + `${lookup.servingWorkloadTitle(plan.serving_workload_id)} / `
      + `batch ${plan.batch_size}`,
    ));
    const commandRows = plan.commands.map((command) => [
      command.kind,
      command.command,
      command.raw_artifact || "-",
    ]);
    details.append(
      summary,
      fieldList([
        ["Baseline", lookup.paperBaselineName(plan.paper_baseline_id)],
        ["Model", plan.model],
        ["Prompt tokens", plan.prompt_tokens],
        ["Decode tokens", plan.decode_tokens],
        ["Traffic mode", plan.traffic_mode],
      ]),
      table(["Kind", "Command", "Raw Artifact"], commandRows),
    );
    return details;
  });
  root.replaceChildren(heading, metadataFields, ...records);
}
