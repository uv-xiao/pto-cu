import { loadViewerData } from "./viewer/data.js";
import { createLookup } from "./viewer/lookup.js";
import { renderPaperBaselines } from "./viewer/render-baselines.js";
import {
  renderBenchmarks,
  renderMethods,
  renderPersistentSchedulerCoverage,
  renderSceneBuilderCoverage,
  renderServingCommandPlan,
  renderServingWorkloads,
} from "./viewer/render-catalog.js";
import {
  renderGoalProgress,
  renderPaperEvaluation,
  renderPaperWorkQueue,
} from "./viewer/render-paper.js";
import {
  renderHeadlineResults,
  renderResults,
  renderSnapshot,
} from "./viewer/render-summary.js";
import { wireTabs } from "./viewer/tabs.js";

async function main() {
  wireTabs();
  try {
    const state = await loadViewerData();
    const lookup = createLookup(state);
    renderSnapshot(state);
    renderHeadlineResults(state);
    renderBenchmarks(state);
    renderMethods(state);
    renderSceneBuilderCoverage(state);
    renderPersistentSchedulerCoverage(state);
    renderServingWorkloads(state, lookup);
    renderServingCommandPlan(state, lookup);
    renderPaperBaselines(state, lookup);
    renderGoalProgress(state);
    renderPaperWorkQueue(state, lookup);
    renderPaperEvaluation(state, lookup);
    renderResults(state, lookup);
  } catch (error) {
    const errorBox = document.getElementById("load-error");
    errorBox.classList.remove("hidden");
    errorBox.textContent = `${error.message}. Serve the repo with python3 -m http.server and open the viewer through http://localhost:8000/.`;
  }
}

main();
