import { fieldList, namedList, paragraph, table, text } from "./dom.js";

export function renderPaperWorkQueue(state, lookup) {
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
    item.id,
    item.priority,
    item.claim_title,
    item.source,
    item.owner,
    item.paper_baseline_id ? lookup.paperBaselineName(item.paper_baseline_id) : "-",
    item.paper_baseline_run_id
      ? lookup.paperBaselineRunTitle(item.paper_baseline_run_id)
      : "-",
    item.execution_attempt_id || "-",
    (item.serving_workload_ids || []).map(lookup.servingWorkloadTitle).join(", ") || "-",
    item.shape_contains || "-",
    item.status,
    item.action,
    (item.evidence_summary || []).join(" | ") || "-",
    item.promotion_gate,
  ]);
  const heading = document.createElement("h3");
  heading.append(text("Paper Work Queue"));
  root.replaceChildren(
    heading,
    summary,
    table(
      [
        "Work Item",
        "Priority",
        "Claim",
        "Source",
        "Owner",
        "Baseline",
        "Run",
        "Execution Attempt",
        "Serving",
        "Shape",
        "Status",
        "Action",
        "Evidence Summary",
        "Promotion Gate",
      ],
      rows,
    ),
  );
}

export function renderGoalProgress(state) {
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

export function renderPaperReadinessAudit(state, lookup) {
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
  const details = audit.claim_audits.map((claim) => (
    renderAuditClaim(claim, lookup)
  ));
  root.replaceChildren(
    summary,
    table(
      ["Claim", "Matrix Status", "Paper Ready", "Missing Items", "Blockers"],
      claimRows,
    ),
    ...details,
  );
}

function renderAuditClaim(claim, lookup) {
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
  const nextActions = claim.next_actions.map((action) => (
    `${action.source}: ${lookup.formatServingAction(action)}`
  ));
  const actionEvidence = claim.next_actions.flatMap((action) => (
    action.evidence_summary || []
  ));
  const policyExceptions = (claim.evidence_policy_exceptions || []).map((entry) => (
    `${entry.status}: ${entry.title} - ${entry.review_rule}`
  ));
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
    ...namedList(
      "Evidence Policy Exceptions",
      policyExceptions.length ? policyExceptions : ["none"],
    ),
    ...namedList("Blockers", claim.blockers),
    ...namedList("Next Actions", nextActions.length ? nextActions : ["none"]),
    ...namedList(
      "Next Action Evidence",
      actionEvidence.length ? actionEvidence : ["none"],
    ),
    paragraph("Promotion gate", claim.promotion_gate),
  );
  return item;
}

export function renderPaperEvaluation(state, lookup) {
  renderPaperReadinessAudit(state, lookup);
  const root = document.getElementById("paper-evaluation-list");
  root.replaceChildren(...state.paperEvaluation.paper_evaluation_matrix.map((claim) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.append(text(`${claim.title} (${claim.status})`));
    const evidence = claim.current_evidence_refs.map((ref) => {
      if (ref.kind === "viewer_result") {
        const benchmark = lookup.benchmarkTitle(ref.benchmark_id);
        const method = lookup.methodName(ref.method_id);
        const coverage = ref.serving_coverage ? ` / ${ref.serving_coverage}` : "";
        return `${ref.kind}: ${benchmark} / ${method} / ${ref.gpu}${coverage}`;
      }
      return `${ref.kind}: ${ref.path}`;
    });
    const missingDetails = (claim.missing_evidence_details || []).map((entry) => (
      lookup.formatServingAction(entry)
    ));
    const policyExceptions = (claim.evidence_policy_exceptions || []).map((entry) => (
      `${entry.status}: ${entry.title} - ${entry.decision}`
    ));
    details.append(
      summary,
      paragraph("Claim", claim.claim),
      fieldList([
        ["Workloads", claim.workload_ids.map(lookup.benchmarkTitle).join(", ")],
        ["Methods", claim.method_ids.map(lookup.methodName).join(", ")],
        [
          "Paper baselines",
          claim.paper_baseline_ids.map(lookup.paperBaselineName).join(", ") || "None",
        ],
        ["Hardware targets", claim.hardware_targets.join(", ")],
        ["Required metrics", claim.required_metrics.join(", ")],
      ]),
      ...namedList("Current Evidence", evidence),
      ...namedList(
        "Evidence Policy Exceptions",
        policyExceptions.length ? policyExceptions : ["none"],
      ),
      ...namedList("Missing Evidence", claim.missing_evidence),
      ...namedList(
        "Missing Evidence Details",
        missingDetails.length ? missingDetails : ["none"],
      ),
      paragraph("Promotion gate", claim.promotion_gate),
    );
    return details;
  }));
}
