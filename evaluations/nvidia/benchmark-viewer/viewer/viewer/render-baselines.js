import { fieldList, namedList, paragraph, text } from "./dom.js";

function runItems(state, baseline, lookup) {
  return state.paperBaselineRuns.paper_baseline_runs
    .filter((run) => run.paper_baseline_id === baseline.id)
    .map((run) => {
      const serving = (run.serving_workload_ids || [])
        .map(lookup.servingWorkloadTitle)
        .join(", ");
      return [
        `${run.title} (${run.status})`,
        `Hardware: ${run.hardware_targets.join(", ")}`,
        `Claim: ${run.paper_evaluation_id}`,
        `Serving policies: ${serving || "not applicable"}`,
        `Run: ${run.run_commands.join(" | ")}`,
        `Artifacts: ${run.expected_artifacts.join(", ")}`,
      ].join("\n");
    });
}

function readinessItems(state, baseline) {
  return state.paperBaselineRunReadiness.paper_baseline_run_readiness
    .filter((readiness) => readiness.paper_baseline_id === baseline.id)
    .map((readiness) => {
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
}

function probeItems(state, baseline) {
  return state.paperBaselineProbes.paper_baseline_probes
    .filter((probe) => probe.paper_baseline_id === baseline.id)
    .map((probe) => {
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
}

function environmentItems(state, baseline) {
  return state.paperBaselineEnvironmentPlans.paper_baseline_environment_plans
    .filter((plan) => plan.paper_baseline_id === baseline.id)
    .map((plan) => {
      const critical = plan.critical_packages.map((pkg) => {
        const evidence = pkg.evidence.length ? pkg.evidence.join(", ") : "missing";
        return `${pkg.name}: ${pkg.declared ? "declared" : "missing"} (${evidence})`;
      });
      const manual = plan.manual_packages.map((pkg) => `${pkg.name}: ${pkg.why}`);
      return [
        `${plan.title} (${plan.status})`,
        `Environment: ${plan.environment_path}`,
        `Build source: ${plan.build_source_path || plan.source_path}`,
        `Python policy: ${plan.python_policy}`,
        `Dependency sources: ${plan.dependency_sources.join(", ")}`,
        `Critical packages: ${critical.join(" | ")}`,
        `Manual packages: ${manual.length ? manual.join(" | ") : "none"}`,
        `Source overlay: ${(plan.source_overlay_commands || []).length ? plan.source_overlay_commands.join(" | ") : "none"}`,
        `Install: ${plan.install_commands.join(" | ")}`,
        `Preflight: ${plan.preflight_commands.length ? plan.preflight_commands.join(" | ") : "none"}`,
        `Validate: ${plan.validation_commands.join(" | ")}`,
        `Execution gaps: ${plan.execution_gaps.join(" | ")}`,
        `Raw artifact: ${plan.raw_artifact}`,
        `Next: ${plan.next_action}`,
      ].join("\n");
    });
}

function attemptItems(state, baseline) {
  const environment = state.paperBaselineEnvironmentAttempts
    .paper_baseline_environment_attempts
    .filter((attempt) => attempt.paper_baseline_id === baseline.id)
    .map((attempt) => {
      const steps = attempt.steps.map((step) => (
        `${step.index}. ${step.kind}: ${step.status}; log: ${step.log}`
      ));
      return [
        `${attempt.title} (${attempt.status})`,
        `Environment: ${attempt.environment_path}`,
        `Window: ${attempt.start_step}-${attempt.end_step} of ${attempt.steps_total}`,
        `Captured steps: ${attempt.steps_completed}`,
        `Artifact root: ${attempt.artifact_root}`,
        `Steps: ${steps.join(" | ")}`,
        `Observation: ${attempt.observation}`,
        `Blocker: ${attempt.blocker || "none"}`,
        `Next: ${attempt.next_action}`,
      ].join("\n");
    });
  const execution = state.paperBaselineExecutionAttempts
    .paper_baseline_execution_attempts
    .filter((attempt) => attempt.paper_baseline_id === baseline.id)
    .map((attempt) => [
      `${attempt.title} (${attempt.status})`,
      `Run ID: ${attempt.paper_baseline_run_id}`,
      `Hardware: ${attempt.hardware.gpu} ${attempt.hardware.compute_target}`,
      `Artifact root: ${attempt.artifact_root}`,
      `Command: ${attempt.command}`,
      `Observation: ${attempt.observation || "none"}`,
      `Blocker: ${attempt.blocker || "none"}`,
      `Artifacts: ${attempt.artifacts.join(", ")}`,
    ].join("\n"));
  return { environment, execution };
}

function renderBaselineCard(state, baseline, lookup) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.append(text(`${baseline.name} (${baseline.status})`));
  const attempts = attemptItems(state, baseline);
  details.append(
    summary,
    paragraph("Paper role", baseline.paper_role),
    paragraph("Source", `${baseline.source.upstream_url} at ${baseline.source.commit}`),
    paragraph("Local note", baseline.source.local_tmp_path),
    paragraph("Paper baselines", baseline.paper_baselines_to_reproduce.join(", ")),
    paragraph("Next action", baseline.next_action),
    ...namedList("Reproduction Runs", runItems(state, baseline, lookup)),
    ...namedList("Run Readiness", readinessItems(state, baseline)),
    ...namedList("Readiness Probes", probeItems(state, baseline)),
    ...namedList("Environment Plans", environmentItems(state, baseline)),
    ...namedList("Environment Attempts", attempts.environment),
    ...namedList("Execution Attempts", attempts.execution),
  );
  return details;
}

export function renderPaperBaselines(state, lookup) {
  const root = document.getElementById("baseline-list");
  root.replaceChildren(
    ...state.paperBaselines.paper_baselines.map(
      (baseline) => renderBaselineCard(state, baseline, lookup),
    ),
  );
}
