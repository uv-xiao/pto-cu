const DATA_FILES = {
  benchmarks: "data/benchmarks.json",
  methods: "data/methods.json",
  paperBaselines: "data/paper_baselines.json",
  paperBaselineRuns: "data/paper_baseline_runs.json",
  paperBaselineProbes: "data/paper_baseline_probes.json",
  paperBaselineRunReadiness: "data/paper_baseline_run_readiness.json",
  paperBaselineExecutionAttempts: "data/paper_baseline_execution_attempts.json",
  servingCommandPlan: "data/serving_command_plan.json",
  servingWorkloads: "data/serving_workloads.json",
  paperEvaluation: "data/paper_evaluation_matrix.json",
  paperReadinessAudit: "data/paper_readiness_audit.json",
  paperReadinessWorkQueue: "data/paper_readiness_work_queue.json",
  goalProgress: "data/goal_progress.json",
  results: "data/results.json",
};

const state = {};

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function text(value) {
  return document.createTextNode(String(value));
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const span = document.createElement("span");
  span.append(text(label));
  const strong = document.createElement("strong");
  strong.append(text(value));
  item.append(span, strong);
  return item;
}

function paragraph(label, value) {
  const item = document.createElement("p");
  const strong = document.createElement("strong");
  strong.append(text(`${label}: `));
  item.append(strong, text(value));
  return item;
}

function fieldList(fields) {
  const list = document.createElement("dl");
  list.className = "meta-list";
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.append(text(label));
    const detail = document.createElement("dd");
    detail.append(text(value));
    list.append(term, detail);
  });
  return list;
}

function benchmarkTitle(id) {
  const benchmark = state.benchmarks.benchmarks.find((item) => item.id === id);
  return benchmark ? benchmark.title : id;
}

function methodName(id) {
  const method = state.methods.methods.find((item) => item.id === id);
  return method ? method.name : id;
}

function paperBaselineName(id) {
  const baseline = state.paperBaselines.paper_baselines.find(
    (item) => item.id === id,
  );
  return baseline ? baseline.name : id;
}

function servingWorkloadTitle(id) {
  const workload = state.servingWorkloads.serving_workloads.find(
    (item) => item.id === id,
  );
  return workload ? workload.title : id;
}

function paperBaselineRunTitle(id) {
  const run = state.paperBaselineRuns.paper_baseline_runs.find(
    (item) => item.id === id,
  );
  return run ? run.title : id;
}

function renderSnapshot() {
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

function renderHeadlineResults() {
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

function evidenceList(refs) {
  const list = document.createElement("ul");
  refs.forEach((ref) => {
    const item = document.createElement("li");
    item.append(text(`${ref.path}: ${ref.symbols.join(", ")}`));
    list.append(item);
  });
  return list;
}

function textList(items) {
  const list = document.createElement("ul");
  items.forEach((value) => {
    const item = document.createElement("li");
    item.append(text(value));
    list.append(item);
  });
  return list;
}

function namedList(title, items) {
  const heading = document.createElement("h3");
  heading.append(text(title));
  return [heading, textList(items)];
}

function commandBlock(command) {
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.append(text(command));
  pre.append(code);
  return pre;
}

function renderBenchmarks() {
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
    const math = paragraph("Math", benchmark.math);
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
      math,
      code,
      run,
      commandBlock(benchmark.run.command),
      evidence,
      evidenceList(benchmark.evidence_refs),
    );
    return details;
  }));
}

function renderMethods() {
  const root = document.getElementById("method-list");
  root.replaceChildren(...state.methods.methods.map((method) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(method.name));
    const metadata = fieldList([
      ["Category", method.category],
      ["Launch model", method.launch_model],
    ]);
    const runtime = paragraph("Runtime flow", method.runtime_flow);
    const lifecycle = paragraph("Lifecycle mapping", method.lifecycle);
    details.append(
      summary,
      metadata,
      runtime,
      lifecycle,
      evidenceList(method.evidence_refs),
    );
    return details;
  }));
}

function renderServingWorkloads() {
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
          .map((id) => {
            const run = state.paperBaselineRuns.paper_baseline_runs.find(
              (item) => item.id === id,
            );
            return run ? run.title : id;
          })
          .join(", "),
      ],
    ]);
    const notes = paragraph("Notes", workload.paper_source.notes);
    const reason = paragraph("Selection reason", workload.model_policy.selection_reason);
    const tokenizer = paragraph(
      "Tokenization rule",
      workload.prompt_policy.tokenization_rule,
    );
    const generation = paragraph(
      "Generation mode",
      workload.decode_policy.generation_mode,
    );
    const evidence = workload.evidence_refs.map((ref) => (
      `${ref.path}: ${ref.symbols.join(", ")}`
    ));
    details.append(
      summary,
      metadata,
      notes,
      reason,
      tokenizer,
      generation,
      ...namedList("Required Metrics", workload.required_metrics),
      ...namedList("Current Blockers", workload.current_blockers),
      ...namedList("Evidence", evidence),
    );
    return details;
  }));
}

function renderServingCommandPlan() {
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
      `${paperBaselineRunTitle(plan.paper_baseline_run_id)} / `
      + `${servingWorkloadTitle(plan.serving_workload_id)} / `
      + `batch ${plan.batch_size}`,
    ));
    const metadata = fieldList([
      ["Baseline", paperBaselineName(plan.paper_baseline_id)],
      ["Model", plan.model],
      ["Prompt tokens", plan.prompt_tokens],
      ["Decode tokens", plan.decode_tokens],
      ["Traffic mode", plan.traffic_mode],
    ]);
    const commandRows = plan.commands.map((command) => [
      command.kind,
      command.command,
      command.raw_artifact || "-",
    ]);
    details.append(
      summary,
      metadata,
      table(["Kind", "Command", "Raw Artifact"], commandRows),
    );
    return details;
  });
  root.replaceChildren(heading, metadataFields, ...records);
}

function renderPaperBaselines() {
  const root = document.getElementById("baseline-list");
  root.replaceChildren(...state.paperBaselines.paper_baselines.map((baseline) => {
    const runs = state.paperBaselineRuns.paper_baseline_runs.filter(
      (run) => run.paper_baseline_id === baseline.id,
    );
    const runReadiness = state.paperBaselineRunReadiness
      .paper_baseline_run_readiness
      .filter((readiness) => readiness.paper_baseline_id === baseline.id);
    const probes = state.paperBaselineProbes.paper_baseline_probes.filter(
      (probe) => probe.paper_baseline_id === baseline.id,
    );
    const executionAttempts = state.paperBaselineExecutionAttempts
      .paper_baseline_execution_attempts
      .filter((attempt) => attempt.paper_baseline_id === baseline.id);
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(`${baseline.name} (${baseline.status})`));

    const role = document.createElement("p");
    role.innerHTML = `<strong>Paper role:</strong> ${baseline.paper_role}`;
    const source = document.createElement("p");
    source.innerHTML = `<strong>Source:</strong> ${baseline.source.upstream_url} at ${baseline.source.commit}`;
    const local = document.createElement("p");
    local.innerHTML = `<strong>Local note:</strong> ${baseline.source.local_tmp_path}`;
    const reproduce = document.createElement("p");
    reproduce.innerHTML = `<strong>Paper baselines:</strong> ${baseline.paper_baselines_to_reproduce.join(", ")}`;
    const next = document.createElement("p");
    next.innerHTML = `<strong>Next action:</strong> ${baseline.next_action}`;
    const runItems = runs.map((run) => {
      const serving = (run.serving_workload_ids || [])
        .map(servingWorkloadTitle)
        .join(", ");
      const lines = [
        `${run.title} (${run.status})`,
        `Hardware: ${run.hardware_targets.join(", ")}`,
        `Claim: ${run.paper_evaluation_id}`,
        `Serving policies: ${serving || "not applicable"}`,
        `Run: ${run.run_commands.join(" | ")}`,
        `Artifacts: ${run.expected_artifacts.join(", ")}`,
      ];
      return lines.join("\n");
    });
    const readinessItems = runReadiness.map((readiness) => {
      const checks = readiness.checks.map((check) => {
        const subject = check.path || check.metric || check.name || check.kind;
        return `${check.kind}: ${subject} (${check.status})`;
      });
      const gaps = readiness.blocking_gaps.length
        ? readiness.blocking_gaps.join(" | ")
        : "none";
      return [
        `${readiness.title} (${readiness.latest_status})`,
        `Run ID: ${readiness.paper_baseline_run_id}`,
        `Artifact: ${readiness.latest_artifact_root}`,
        `Checks: ${checks.join(" | ")}`,
        `Blocking gaps: ${gaps}`,
        `Next: ${readiness.next_action}`,
      ].join("\n");
    });
    const probeItems = probes.map((probe) => {
      const checks = probe.checks.map((check) => (
        check.path ? `${check.kind}: ${check.path}` : `${check.kind}: ${check.module}`
      ));
      const machines = probe.latest_machine_status.map((item) => {
        const gaps = item.blocking_gaps.length
          ? `; gaps: ${item.blocking_gaps.join(", ")}`
          : "";
        return `${item.gpu}: ${item.status}${gaps}; artifact: ${item.artifact}`;
      });
      return [
        `${probe.title} (${probe.latest_status})`,
        `Artifact: ${probe.latest_artifact_root}`,
        `Machines: ${machines.join(" | ")}`,
        `Checks: ${checks.join(" | ")}`,
        `Next: ${probe.next_action}`,
      ].join("\n");
    });
    const executionItems = executionAttempts.map((attempt) => {
      const evidence = attempt.artifacts.join(", ");
      const blocker = attempt.blocker || "none";
      const observation = attempt.observation || "none";
      return [
        `${attempt.title} (${attempt.status})`,
        `Run ID: ${attempt.paper_baseline_run_id}`,
        `Hardware: ${attempt.hardware.gpu} ${attempt.hardware.compute_target}`,
        `Artifact root: ${attempt.artifact_root}`,
        `Command: ${attempt.command}`,
        `Observation: ${observation}`,
        `Blocker: ${blocker}`,
        `Artifacts: ${evidence}`,
      ].join("\n");
    });

    details.append(
      summary,
      role,
      source,
      local,
      reproduce,
      next,
      ...namedList("Reproduction Runs", runItems),
      ...namedList("Run Readiness", readinessItems),
      ...namedList("Readiness Probes", probeItems),
      ...namedList("Execution Attempts", executionItems),
    );
    return details;
  }));
}

function renderPaperEvaluation() {
  renderPaperReadinessAudit();
  const root = document.getElementById("paper-evaluation-list");
  root.replaceChildren(...state.paperEvaluation.paper_evaluation_matrix.map((claim) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(`${claim.title} (${claim.status})`));
    const claimText = paragraph("Claim", claim.claim);
    const metadata = fieldList([
      ["Workloads", claim.workload_ids.map(benchmarkTitle).join(", ")],
      ["Methods", claim.method_ids.map(methodName).join(", ")],
      [
        "Paper baselines",
        claim.paper_baseline_ids.map(paperBaselineName).join(", ") || "None",
      ],
      ["Hardware targets", claim.hardware_targets.join(", ")],
      ["Required metrics", claim.required_metrics.join(", ")],
    ]);
    const promotion = paragraph("Promotion gate", claim.promotion_gate);
    const evidence = claim.current_evidence_refs.map((ref) => {
      if (ref.kind === "viewer_result") {
        const benchmark = benchmarkTitle(ref.benchmark_id);
        const method = methodName(ref.method_id);
        return `${ref.kind}: ${benchmark} / ${method} / ${ref.gpu}`;
      }
      return `${ref.kind}: ${ref.path}`;
    });
    details.append(
      summary,
      claimText,
      metadata,
      ...namedList("Current Evidence", evidence),
      ...namedList("Missing Evidence", claim.missing_evidence),
      promotion,
    );
    return details;
  }));
}

function renderPaperWorkQueue() {
  const queue = state.paperReadinessWorkQueue;
  const root = document.getElementById("paper-work-queue");
  const sourceSummary = Object.entries(queue.summary.work_items_by_source)
    .map(([source, count]) => `${source}: ${count}`)
    .join(", ");
  const claimSummary = Object.entries(queue.summary.work_items_by_claim)
    .map(([claim, count]) => `${claim}: ${count}`)
    .join(", ");
  const summary = fieldList([
    ["Overall status", queue.overall_status],
    ["Ready claims", queue.ready_claims],
    ["Blocked claims", queue.blocked_claims],
    ["Total work items", queue.summary.total_work_items],
    ["By source", sourceSummary || "none"],
    ["By claim", claimSummary || "none"],
    ["Source file", queue.source_file],
  ]);
  const rows = queue.work_items.map((item) => [
    item.priority,
    item.claim_title,
    item.source,
    item.owner,
    item.status,
    item.action,
  ]);
  const heading = document.createElement("h3");
  heading.append(text("Paper Work Queue"));
  root.replaceChildren(
    heading,
    summary,
    table(["Priority", "Claim", "Source", "Owner", "Status", "Action"], rows),
  );
}

function renderGoalProgress() {
  const progress = state.goalProgress;
  const root = document.getElementById("goal-progress-list");
  const heading = document.createElement("h3");
  heading.append(text("Goal Progress"));
  const statusSummary = Object.entries(progress.summary.criteria_by_status)
    .map(([status, count]) => `${status}: ${count}`)
    .join(", ");
  const summary = fieldList([
    ["Overall status", progress.overall_status],
    ["Criteria total", progress.summary.criteria_total],
    ["Criteria met", progress.summary.criteria_met],
    ["Criteria in progress", progress.summary.criteria_in_progress],
    ["By status", statusSummary || "none"],
    ["Source files", progress.source_files.join(", ")],
  ]);
  const criteria = progress.acceptance_criteria.map((criterion) => {
    const details = document.createElement("details");
    const summaryLine = document.createElement("summary");
    summaryLine.append(text(`${criterion.title}: ${criterion.status}`));
    const extraFields = [];
    if (criterion.id === "paper_grade_results") {
      extraFields.push(
        ["Paper readiness", criterion.paper_readiness_status],
        ["Blocking work items", criterion.blocking_work_items],
      );
    }
    details.append(
      summaryLine,
      paragraph("Summary", criterion.summary),
      fieldList([
        ["Criterion ID", criterion.id],
        ["Status", criterion.status],
        ...extraFields,
      ]),
      ...namedList("Evidence", criterion.evidence_refs),
      ...namedList("Verification", criterion.verification),
      ...namedList("Gaps", criterion.gaps.length ? criterion.gaps : ["none"]),
    );
    return details;
  });
  root.replaceChildren(heading, summary, ...criteria);
}

function renderPaperReadinessAudit() {
  const audit = state.paperReadinessAudit;
  const root = document.getElementById("paper-readiness-audit");
  const summary = fieldList([
    ["Overall status", audit.overall_status],
    ["Ready claims", audit.ready_claims],
    ["Blocked claims", audit.blocked_claims],
    ["Source files", audit.source_files.join(", ")],
  ]);
  const claimRows = audit.claim_audits.map((claim) => [
    claim.title,
    claim.matrix_status,
    claim.ready_for_paper_claim ? "yes" : "no",
    claim.missing_evidence_count,
    claim.blockers.length,
  ]);
  const details = audit.claim_audits.map((claim) => {
    const item = document.createElement("details");
    const summaryLine = document.createElement("summary");
    summaryLine.append(text(`${claim.title}: ${claim.blockers.length} blockers`));
    const runStatuses = claim.paper_baseline_run_statuses.map((run) => (
      `${run.paper_baseline_id}/${run.id}: ${run.status}`
    ));
    const runReadinessStatuses = claim.paper_baseline_run_readiness_statuses
      .map((readiness) => {
        const gaps = readiness.blocking_gaps.length
          ? readiness.blocking_gaps.join(" | ")
          : "none";
        return `${readiness.paper_baseline_id}/${readiness.paper_baseline_run_id}: ${readiness.latest_status} (${gaps})`;
      });
    const probeStatuses = claim.probe_statuses.map((probe) => {
      const machines = probe.machines.map((machine) => (
        `${machine.gpu}=${machine.status}`
      )).join(", ");
      return `${probe.paper_baseline_id}: ${probe.latest_status} (${machines})`;
    });
    const nextActions = claim.next_actions.map((action) => {
      const owner = action.paper_baseline_run_id
        || action.paper_baseline_id
        || action.source;
      return `${action.source}/${owner}: ${action.action}`;
    });
    item.append(
      summaryLine,
      fieldList([
        ["Matrix status", claim.matrix_status],
        ["Evidence refs", Object.entries(claim.evidence_ref_counts)
          .map(([key, value]) => `${key}: ${value}`)
          .join(", ") || "none"],
        ["Missing viewer results", claim.missing_viewer_results.join(", ") || "none"],
      ]),
      ...namedList("Paper Baseline Runs", runStatuses.length ? runStatuses : ["none"]),
      ...namedList(
        "Run Readiness",
        runReadinessStatuses.length ? runReadinessStatuses : ["none"],
      ),
      ...namedList("Probe Status", probeStatuses.length ? probeStatuses : ["none"]),
      ...namedList("Blockers", claim.blockers),
      ...namedList("Next Actions", nextActions.length ? nextActions : ["none"]),
      paragraph("Promotion gate", claim.promotion_gate),
    );
    return item;
  });
  root.replaceChildren(
    summary,
    table(
      ["Claim", "Matrix Status", "Paper Ready", "Missing Items", "Blockers"],
      claimRows,
    ),
    ...details,
  );
}

function table(headers, rows) {
  const tableEl = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.append(text(header));
    headerRow.append(th);
  });
  thead.append(headerRow);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.append(text(cell));
      tr.append(td);
    });
    tbody.append(tr);
  });
  tableEl.append(thead, tbody);
  return tableEl;
}

function renderResults() {
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
      "Raw artifact",
    ],
    state.results.result_records.map((row) => [
      `${row.hardware.gpu} / ${row.hardware.machine} / ${row.hardware.compute_target}`,
      benchmarkTitle(row.benchmark_id),
      methodName(row.method_id),
      `${row.inputs.shape}; ${row.inputs.dtype}; ${row.inputs.repeat_policy}`,
      row.statistic.sample_count,
      row.statistic.host_wall_ns,
      p90(row.statistic, "host_wall_p90_ns"),
      row.statistic.device_wall_ns,
      p90(row.statistic, "device_wall_p90_ns"),
      row.correctness,
      row.raw_artifact,
    ]),
  ));
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.view).classList.add("active");
    });
  });
}

async function main() {
  wireTabs();
  try {
    const [
      benchmarks,
      methods,
      paperBaselines,
      paperBaselineRuns,
      paperBaselineProbes,
      paperBaselineRunReadiness,
      paperBaselineExecutionAttempts,
      servingCommandPlan,
      servingWorkloads,
      paperEvaluation,
      paperReadinessAudit,
      paperReadinessWorkQueue,
      goalProgress,
      results,
    ] = await Promise.all([
      loadJson(DATA_FILES.benchmarks),
      loadJson(DATA_FILES.methods),
      loadJson(DATA_FILES.paperBaselines),
      loadJson(DATA_FILES.paperBaselineRuns),
      loadJson(DATA_FILES.paperBaselineProbes),
      loadJson(DATA_FILES.paperBaselineRunReadiness),
      loadJson(DATA_FILES.paperBaselineExecutionAttempts),
      loadJson(DATA_FILES.servingCommandPlan),
      loadJson(DATA_FILES.servingWorkloads),
      loadJson(DATA_FILES.paperEvaluation),
      loadJson(DATA_FILES.paperReadinessAudit),
      loadJson(DATA_FILES.paperReadinessWorkQueue),
      loadJson(DATA_FILES.goalProgress),
      loadJson(DATA_FILES.results),
    ]);
    Object.assign(state, {
      benchmarks,
      methods,
      paperBaselines,
      paperBaselineRuns,
      paperBaselineProbes,
      paperBaselineRunReadiness,
      paperBaselineExecutionAttempts,
      servingCommandPlan,
      servingWorkloads,
      paperEvaluation,
      paperReadinessAudit,
      paperReadinessWorkQueue,
      goalProgress,
      results,
    });
    renderSnapshot();
    renderHeadlineResults();
    renderBenchmarks();
    renderMethods();
    renderServingWorkloads();
    renderServingCommandPlan();
    renderPaperBaselines();
    renderGoalProgress();
    renderPaperWorkQueue();
    renderPaperEvaluation();
    renderResults();
  } catch (error) {
    const errorBox = document.getElementById("load-error");
    errorBox.classList.remove("hidden");
    errorBox.textContent = `${error.message}. Serve the repo with python3 -m http.server and open the viewer through http://localhost:8000/.`;
  }
}

main();
