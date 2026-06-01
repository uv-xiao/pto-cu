export function createLookup(state) {
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

  function formatServingAction(action) {
    const parts = [];
    if (action.method_id) {
      parts.push(`method=${methodName(action.method_id)}`);
    }
    if (action.paper_baseline_id) {
      parts.push(`baseline=${paperBaselineName(action.paper_baseline_id)}`);
    }
    if (action.paper_baseline_run_id) {
      parts.push(`run=${paperBaselineRunTitle(action.paper_baseline_run_id)}`);
    }
    if (action.serving_workload_ids && action.serving_workload_ids.length) {
      parts.push(
        `serving=${action.serving_workload_ids.map(servingWorkloadTitle).join(", ")}`,
      );
    }
    if (action.shape_contains) {
      parts.push(`shape=${action.shape_contains}`);
    }
    const prefix = parts.length ? `${parts.join(" | ")}: ` : "";
    return `${prefix}${action.action}`;
  }

  return {
    benchmarkTitle,
    formatServingAction,
    methodName,
    paperBaselineName,
    paperBaselineRunTitle,
    servingWorkloadTitle,
  };
}
