import { DATA_FILES } from "./config.js";

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

async function loadSidecarList(base, path) {
  if (path.endsWith(".json")) {
    return loadJson(`${base}/${path}`);
  }
  const index = await loadJson(`${base}/${path}/index.json`);
  return Promise.all(
    index.item_files.map((item) => loadJson(`${base}/${path}/${item}`)),
  );
}

async function expandRecord(base, record) {
  const payload = Object.assign({}, record);
  for (const field of [
    "current_evidence_refs",
    "missing_evidence_details",
    "paper_baseline_run_statuses",
    "paper_baseline_run_readiness_statuses",
    "execution_attempt_statuses",
    "probe_statuses",
    "next_actions",
  ]) {
    const pathKey = `${field}_path`;
    if (payload[pathKey]) {
      payload[field] = await loadSidecarList(base, payload[pathKey]);
      delete payload[pathKey];
    }
  }
  return payload;
}

async function loadDataFile(spec) {
  if (typeof spec === "string") {
    return loadJson(spec);
  }
  const manifest = await loadJson(spec.manifest);
  const base = spec.manifest.replace(/\/[^/]+$/, "");
  const recordFiles = manifest.record_files || await loadJson(
    `${base}/${manifest.record_files_path}`,
  );
  const records = await Promise.all(
    recordFiles.map(async (path) => {
      const record = await loadJson(`${base}/${path}`);
      return expandRecord(base, record);
    }),
  );
  const payload = Object.assign({}, manifest);
  delete payload.record_files;
  delete payload.collection;
  payload[manifest.collection] = records;
  return payload;
}

export async function loadViewerData() {
  const [
    benchmarks,
    methods,
    sceneBuilderCoverage,
    persistentSchedulerCoverage,
    paperBaselines,
    paperBaselineRuns,
    paperBaselineProbes,
    paperBaselineEnvironmentPlans,
    paperBaselineEnvironmentAttempts,
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
    loadJson(DATA_FILES.sceneBuilderCoverage),
    loadJson(DATA_FILES.persistentSchedulerCoverage),
    loadJson(DATA_FILES.paperBaselines),
    loadDataFile(DATA_FILES.paperBaselineRuns),
    loadDataFile(DATA_FILES.paperBaselineProbes),
    loadJson(DATA_FILES.paperBaselineEnvironmentPlans),
    loadDataFile(DATA_FILES.paperBaselineEnvironmentAttempts),
    loadDataFile(DATA_FILES.paperBaselineRunReadiness),
    loadDataFile(DATA_FILES.paperBaselineExecutionAttempts),
    loadDataFile(DATA_FILES.servingCommandPlan),
    loadJson(DATA_FILES.servingWorkloads),
    loadDataFile(DATA_FILES.paperEvaluation),
    loadDataFile(DATA_FILES.paperReadinessAudit),
    loadJson(DATA_FILES.paperReadinessWorkQueue),
    loadJson(DATA_FILES.goalProgress),
    loadDataFile(DATA_FILES.results),
  ]);
  return {
    benchmarks,
    methods,
    sceneBuilderCoverage,
    persistentSchedulerCoverage,
    paperBaselines,
    paperBaselineRuns,
    paperBaselineProbes,
    paperBaselineEnvironmentPlans,
    paperBaselineEnvironmentAttempts,
    paperBaselineRunReadiness,
    paperBaselineExecutionAttempts,
    servingCommandPlan,
    servingWorkloads,
    paperEvaluation,
    paperReadinessAudit,
    paperReadinessWorkQueue,
    goalProgress,
    results,
  };
}
