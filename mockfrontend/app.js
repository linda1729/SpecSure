// 原云端地址保留： http://8.140.214.49:8000
const DEFAULT_API_BASE = "http://localhost:8000";

function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api");
  if (fromQuery) return fromQuery.replace(/\/$/, "");
  if (window.API_BASE) return window.API_BASE.replace(/\/$/, "");
  // 本地静态调试时可通过 ?api=... 覆盖；默认指向云端公网地址
  return DEFAULT_API_BASE;
}

const API_BASE = resolveApiBase();

const state = {
  datasets: [],
  defaults: null,
  svmDefaults: null,
  artifacts: { models: [], reports: [], visualizations: [] },
  svmArtifacts: { models: [], reports: [], visualizations: [] },
  evaluations: [],
  evaluationsSvm: [],
  evalSummary: { cnn: [], svm: [], comparisons: [] },
  selectedEvalDataset: null,
  classMap: {},
  lastRun: null,
  currentJobs: { cnn: null, svm: null },
  pollTimer: { cnn: null, svm: null },
};

const $ = (id) => document.getElementById(id);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

function log(message) {
  const box = $("logBox");
  if (box) {
    const time = new Date().toLocaleTimeString();
    box.textContent = `[${time}] ${message}\n` + box.textContent;
  }
  const status = $("statusText");
  if (status) status.textContent = message;
}

function setView(view) {
  qsa(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  qsa(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
}

function readableStatus(data) {
  const mode = data?.mode === "inference_only" ? "推理" : "训练";
  switch (data?.status) {
    case "running":
    case "pending":
      return `${mode}中`;
    case "succeeded":
      return `${mode}完成`;
    case "failed":
      return `${mode}失败`;
    default:
      return "等待执行";
  }
}

function updateProgressUI(model = "cnn", progress = 0, statusText = "等待执行") {
  const percent = Math.max(0, Math.min(100, Number(progress) || 0));
  const suffix = model === "svm" ? "Svm" : "Cnn";
  const fill = $(`progressFill${suffix}`);
  const num = $(`progressNumber${suffix}`);
  const label = $(`progressLabel${suffix}`);
  if (fill) fill.style.width = `${percent}%`;
  if (num) num.textContent = `${percent.toFixed(1)}%`;
  if (label) label.textContent = statusText;
}

function stopPolling(model) {
  const stopOne = (m) => {
    if (state.pollTimer[m]) {
      clearTimeout(state.pollTimer[m]);
      state.pollTimer[m] = null;
    }
    state.currentJobs[m] = null;
  };
  if (model) {
    stopOne(model);
  } else {
    stopOne("cnn");
    stopOne("svm");
  }
}

async function pollJob(jobId, model = "cnn") {
  if (!jobId) return;
  state.currentJobs[model] = jobId;
  const prefix = model === "svm" ? "/api/svm" : "/api/cnn";
  const label = model === "svm" ? "SVM" : "HybridSN";
  try {
    const data = await fetchJSON(`${API_BASE}${prefix}/train/${jobId}`);
    if (state.currentJobs[model] !== jobId) return;
    state.lastRun = data;
    renderRunResult(data);
    const statusText = readableStatus(data);
    updateProgressUI(model, data.progress ?? 0, statusText);
    if (data.status === "running" || data.status === "pending") {
      state.pollTimer[model] = setTimeout(() => pollJob(jobId, model), 1500);
    } else {
      stopPolling();
      await loadArtifacts();
      await loadEvaluations();
      log(data.message || statusText || `${label} 任务完成`);
    }
  } catch (err) {
    if (err.status === 404) {
      stopPolling();
      return log("未找到任务，可能已被清理");
    }
    log(`查询进度失败: ${err.message}`);
    updateProgressUI(model, 0, "查询进度失败，稍后重试");
    state.pollTimer[model] = setTimeout(() => pollJob(jobId, model), 2000);
  }
}

async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const txt = await resp.text();
    const error = new Error(txt || resp.statusText);
    error.status = resp.status;
    throw error;
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) return resp.json();
  return resp.text();
}

function toUrl(url) {
  if (!url) return "";
  if (url.startsWith("http")) return url;
  return `${API_BASE}${url}`;
}

function getClassNames(datasetId) {
  if (!datasetId) return null;
  return state.classMap[datasetId] || state.datasets.find((d) => d.id === datasetId)?.class_names || null;
}

function classLegendHTML(classNames) {
  if (!classNames || !Object.keys(classNames).length) {
    return '<div class="muted tiny">暂无标签 CSV</div>';
  }
  return Object.entries(classNames)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => `<span class="chip"><span class="chip-id">${k}</span>${v}</span>`)
    .join("");
}

function renderClassLegend(classNames, targetId = "classNamesBox") {
  const box = $(targetId);
  if (!box) return;
  const html = classLegendHTML(classNames);
  box.innerHTML = html || '<div class="muted tiny">当前数据集未提供标签 CSV，默认显示数字类标。</div>';
}

function chooseDataset(id) {
  if ($("trainDataset")) $("trainDataset").value = id;
  renderClassLegend(getClassNames(id));
  log(`已选择数据集 ${id}`);
}

function renderDatasets() {
  const box = $("datasetCards");
  if (!box) return;
  box.innerHTML = "";
  if (!state.datasets.length) {
    box.innerHTML = '<div class="muted">未检测到数据集，请将 .mat 文件放到项目 data/ 目录后刷新。</div>';
    return;
  }
  state.datasets.forEach((ds) => {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
      <div class="dataset-head">
        <div>
          <h4>${ds.name} (${ds.id})</h4>
          <p class="muted small">${ds.folder}</p>
        </div>
        <span class="badge ${ds.ready ? "online" : "offline"}">${ds.ready ? "就绪" : "缺文件"}</span>
      </div>
      <div class="dataset-files">
        <div><span class="meta-label">HSI</span><div class="ellipsis">${ds.data_file}</div></div>
        <div><span class="meta-label">GT</span><div class="ellipsis">${ds.gt_file}</div></div>
        <div><span class="meta-label">Key</span><div class="ellipsis">${ds.data_key}</div></div>
        <div><span class="meta-label">GT Key</span><div class="ellipsis">${ds.gt_key}</div></div>
      </div>
      <div class="class-chip-row">${classLegendHTML(ds.class_names || null)}</div>
      <div class="muted tiny">目录: ${ds.data_path.replace(ds.data_file, "")}</div>
      <div class="actions">
        <button class="btn ghost btn-use-dataset" data-id="${ds.id}" ${ds.ready ? "" : "disabled"}>使用此数据集</button>
        <span class="muted tiny">${ds.ready ? "可直接训练/推理" : "文件缺失，请按提示放置 .mat"}</span>
      </div>
    `;
    box.appendChild(card);
  });
  fillDatasetSelects();
  qsa(".btn-use-dataset").forEach((btn) => btn.addEventListener("click", () => chooseDataset(btn.dataset.id)));
}

function fillDatasetSelects() {
  const selects = [$("trainDataset")];
  selects.forEach((sel) => {
    if (!sel) return;
    sel.innerHTML = "";
    state.datasets.forEach((ds) => {
      const opt = document.createElement("option");
      opt.value = ds.id;
      opt.textContent = `${ds.id} · ${ds.name}`;
      sel.appendChild(opt);
    });
    if (sel.options.length > 0 && !sel.value) {
      sel.value = sel.options[0].value;
    }
  });
  const firstReady = state.datasets.find((d) => d.ready);
  if (firstReady) {
    if ($("trainDataset")) $("trainDataset").value = firstReady.id;
    renderClassLegend(getClassNames(firstReady.id));
  }
}

function setDefaultParams() {
  const d = state.defaults || {};
  if ($("testRatio")) $("testRatio").value = d.test_ratio ?? 0.3;
  if ($("windowSize")) $("windowSize").value = d.window_size ?? 25;
  if ($("pcaIP")) $("pcaIP").value = d.pca_components_ip ?? 30;
  if ($("pcaOther")) $("pcaOther").value = d.pca_components_other ?? 15;
  if ($("batchSize")) $("batchSize").value = d.batch_size ?? 256;
  if ($("epochs")) $("epochs").value = d.epochs ?? 100;
  if ($("learningRate")) $("learningRate").value = d.lr ?? 0.001;
}

function setSvmDefaultParams() {
  const d = state.svmDefaults || {};
  if ($("svmKernel")) $("svmKernel").value = d.kernel || "rbf";
  if ($("svmC")) $("svmC").value = d.C ?? 10;
  if ($("svmGamma")) $("svmGamma").value = d.gamma ?? "scale";
  if ($("svmDegree")) $("svmDegree").value = d.degree ?? 3;
  if ($("svmRandomState")) $("svmRandomState").value = d.random_state ?? 42;
}

const CNN_MODEL_HINTS = {
  IP: "models/cnn/trained_models/HybridSN/IndianPines_model_pca=30_window=25_lr=0.001_epochs=1.pth",
  PU: "models/cnn/trained_models/HybridSN/PaviaU_model_pca=15_window=25_lr=0.001_epochs=10.pth",
  SA: "models/cnn/trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth",
};

const SVM_MODEL_HINTS = {
  IP: "models/svm/trained_models/SVM/IndianPines_model_pca=30_window=25_lr=0.001_epochs=100.joblib",
  PU: "models/svm/trained_models/SVM/PaviaU_model_pca=15_window=25_lr=0.001_epochs=100.joblib",
  SA: "models/svm/trained_models/SVM/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib",
};

function findLatestModel(models = [], dsId, type = "cnn") {
  const ds = state.datasets.find((d) => d.id === dsId);
  const folder = ds?.folder || dsId;
  const prefix = `${folder}_model_`;
  const filtered = models.filter((m) => {
    const name = m.name || "";
    if (!name.startsWith(prefix)) return false;
    if (name.endsWith(".pca.pkl")) return false;
    if (type === "cnn" && !name.endsWith(".pth")) return false;
    if (type === "svm" && !name.endsWith(".joblib")) return false;
    return true;
  });
  if (!filtered.length) return "";
  const last = filtered[filtered.length - 1];
  return (last.path || last.name || "").replace(/\//g, "\\");
}

function defaultModelPath() {
  const dsId = $("trainDataset")?.value || "SA";
  const byArtifacts = findLatestModel(state.artifacts?.models || [], dsId, "cnn");
  if (byArtifacts) return byArtifacts;
  const hint = CNN_MODEL_HINTS[dsId];
  if (hint) return hint.replace(/\//g, "\\");
  return "";
}

function defaultSvmModelPath() {
  const dsId = $("trainDataset")?.value || "SA";
  const byArtifacts = findLatestModel(state.svmArtifacts?.models || [], dsId, "svm");
  if (byArtifacts) return byArtifacts;
  const hint = SVM_MODEL_HINTS[dsId];
  if (hint) return hint.replace(/\//g, "\\");
  return "";
}

function applyDefaultModelHints() {
  const cnnHint = defaultModelPath();
  const svmHint = defaultSvmModelPath();
  const cnnInput = $("modelPathManual");
  const svmInput = $("svmModelPathManual");
  if (cnnInput && !cnnInput.value) {
    cnnInput.placeholder = cnnHint || "可留空自动匹配最新模型";
  }
  if (svmInput && !svmInput.value) {
    svmInput.placeholder = svmHint || "可留空自动匹配最新模型";
  }
}

async function loadDefaults() {
  let ok = false;
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/defaults`);
    state.defaults = data.hyperparams || {};
    state.datasets = data.datasets || [];
    state.classMap = {};
    state.datasets.forEach((ds) => {
      if (ds.class_names) state.classMap[ds.id] = ds.class_names;
    });
    renderDatasets();
    setDefaultParams();
    ok = true;
  } catch (err) {
    log(`加载 CNN 默认参数失败: ${err.message}`);
  }

  try {
    const data = await fetchJSON(`${API_BASE}/api/svm/defaults`);
    state.svmDefaults = data.hyperparams || {};
    if (!state.datasets?.length && data.datasets) {
      state.datasets = data.datasets;
      renderDatasets();
    }
    setSvmDefaultParams();
    ok = true;
  } catch (err) {
    log(`加载 SVM 默认参数失败: ${err.message}`);
  }

  if (ok) {
    log("默认参数已加载");
  }
}

async function refreshDatasets() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/datasets`);
    state.datasets = data;
    state.classMap = {};
    state.datasets.forEach((ds) => {
      if (ds.class_names) state.classMap[ds.id] = ds.class_names;
    });
    renderDatasets();
    log("数据集状态已刷新");
  } catch (err) {
    log(`刷新数据集失败: ${err.message}`);
  }
}

function gatherParams() {
  return {
    dataset: $("trainDataset").value || "SA",
    test_ratio: Number($("testRatio").value || state.defaults?.test_ratio || 0.3),
    window_size: Number($("windowSize").value || state.defaults?.window_size || 25),
    pca_components_ip: Number($("pcaIP").value || state.defaults?.pca_components_ip || 30),
    pca_components_other: Number($("pcaOther").value || state.defaults?.pca_components_other || 15),
    batch_size: Number($("batchSize").value || state.defaults?.batch_size || 256),
    epochs: Number($("epochs").value || state.defaults?.epochs || 100),
    lr: Number($("learningRate").value || state.defaults?.lr || 0.001),
  };
}

function gatherSvmParams() {
  const base = gatherParams();
  return {
    ...base,
    kernel: $("svmKernel")?.value || state.svmDefaults?.kernel || "rbf",
    C: Number($("svmC")?.value || state.svmDefaults?.C || 10),
    gamma: $("svmGamma")?.value || state.svmDefaults?.gamma || "scale",
    degree: Number($("svmDegree")?.value || state.svmDefaults?.degree || 3),
    random_state: Number($("svmRandomState")?.value || state.svmDefaults?.random_state || 42),
  };
}

function fillModelSelect() {
  const sel = $("modelPathSelect");
  if (!sel) return;
  sel.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = "自动匹配";
  sel.appendChild(opt0);
  (state.artifacts?.models || []).forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = m.name;
    sel.appendChild(opt);
  });
}

function fillSvmModelSelect() {
  const sel = $("svmModelPathSelect");
  if (!sel) return;
  sel.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = "自动匹配";
  sel.appendChild(opt0);
  (state.svmArtifacts?.models || []).forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = m.name;
    sel.appendChild(opt);
  });
}

async function loadArtifacts() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/artifacts`);
    state.artifacts = data;
  } catch (err) {
    log(`加载 CNN 产物失败: ${err.message}`);
  }
  try {
    const data = await fetchJSON(`${API_BASE}/api/svm/artifacts`);
    state.svmArtifacts = data;
  } catch (err) {
    log(`加载 SVM 产物失败: ${err.message}`);
  }
  renderArtifacts();
  fillModelSelect();
  fillSvmModelSelect();
}

async function loadEvaluations() {
  try {
    const summary = await fetchJSON(`${API_BASE}/api/evaluations/summary`);
    state.evalSummary = summary || { cnn: [], svm: [], comparisons: [] };
    state.evaluations = summary?.cnn || [];
    state.evaluationsSvm = summary?.svm || [];
  } catch (err) {
    log(`加载评估汇总失败，将回退至 CNN：${err.message}`);
    try {
      const data = await fetchJSON(`${API_BASE}/api/cnn/evaluations`);
      state.evaluations = data || [];
      state.evaluationsSvm = [];
      state.evalSummary = { cnn: state.evaluations, svm: [], comparisons: [] };
    } catch (subErr) {
      log(`加载评估结果失败: ${subErr.message}`);
    }
  }
  renderEvaluations();
}

async function refreshEvaluations(manual = false) {
  if (manual) {
    log("手动刷新评估与对比...");
  }
  await loadEvaluations();
  // 默认选中第一条
  const comps = state.evalSummary?.comparisons || [];
  if (comps.length && !state.selectedEvalDataset) {
    state.selectedEvalDataset = comps[0].dataset;
  }
  if (manual) {
    log("评估已刷新");
  }
}

function renderArtifacts() {
  const renderList = (containerId, items, emptyText) => {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = "";
    if (!items || !items.length) {
      el.textContent = emptyText;
      return;
    }
    items.forEach((item) => {
      const a = document.createElement("a");
      a.href = toUrl(item.url || item.path);
      a.textContent = item.name;
      a.target = "_blank";
      el.appendChild(a);
    });
  };
  renderList("artifactModels", state.artifacts?.models, "暂无模型");
  renderList("artifactReports", state.artifacts?.reports, "暂无报告");
  renderList("artifactVisuals", state.artifacts?.visualizations, "暂无可视化");
  renderList("artifactModelsSvm", state.svmArtifacts?.models, "暂无模型");
  renderList("artifactReportsSvm", state.svmArtifacts?.reports, "暂无报告");
  renderList("artifactVisualsSvm", state.svmArtifacts?.visualizations, "暂无可视化");
}

function pickVisualUrl(item, key) {
  if (!item) return "";
  const urls = item.artifacts?.urls || {};
  return urls[key] || item.artifacts?.[`${key}_path`] || "";
}

function metricBlock(label, acc, kappa, cssClass) {
  const width = Math.max(0, Math.min(100, Number(acc) || 0));
  const accText = acc === null || acc === undefined ? "--" : `${Number(acc).toFixed(2)}%`;
  const kappaText = kappa === null || kappa === undefined ? "--" : Number(kappa).toFixed(2);
  return `
    <div class="metric-pair">
      <div class="label">${label}</div>
      <div class="metric-bar ${cssClass}"><div class="fill" style="width:${width}%"></div></div>
      <div class="value">${accText}</div>
      <div class="metric-meta">Kappa ${kappaText}</div>
    </div>
  `;
}

function visualBlock(label, url) {
  if (!url) {
    return `<div class="dash-visual muted"><p class="muted tiny">${label}</p><div class="muted tiny">暂无可视化</div></div>`;
  }
  const href = toUrl(url);
  return `
    <div class="dash-visual">
      <p class="muted tiny">${label}</p>
      <a href="${href}" target="_blank">
        <img src="${href}" alt="${label}" />
      </a>
    </div>
  `;
}

function renderDashboard() {
  renderEvalButtons();
  renderEvalDetail();
}

function evalCardHTML(item) {
  const metrics = item.metrics || {};
  const art = item.artifacts || {};
  const urls = art.urls || {};
  const visuals = [
    ["预测", urls.prediction || art.prediction_path],
    ["GT", urls.groundtruth || art.groundtruth_path],
    ["混淆矩阵", urls.confusion || art.confusion_path],
    ["伪彩色", urls.pseudocolor || art.pseudocolor_path],
    ["分类图", urls.classification || art.classification_path],
    ["对比图", urls.comparison || art.comparison_path],
    ["错误图", urls.error_map || art.error_map_path],
  ];
  const metricHtml = Object.entries(metrics)
    .map(([k, v]) => `<div class="metric-row"><span>${k}</span><strong>${Number(v).toFixed(3)}</strong></div>`)
    .join("");
  const visualHtml = visuals
    .filter(([, url]) => !!url)
    .map(
      ([label, url]) => `
        <div class="thumb">
          <p class="muted tiny">${label}</p>
          <a href="${toUrl(url)}" target="_blank">
            <img src="${toUrl(url)}" alt="${label}" />
          </a>
        </div>
      `
    )
    .join("");
  const legendHtml = `<div class="class-names">${classLegendHTML(item.class_names || getClassNames(item.dataset))}</div>`;
  const hyper = `PCA=${item.pca_components} · Window=${item.window_size} · LR=${item.lr} · Epochs=${item.epochs}`;
  const reportHref = toUrl(item.report_url || item.report_path);
  const reportLink = reportHref
    ? `<a class="btn ghost" href="${reportHref}" target="_blank">报告</a>`
    : '<span class="muted tiny">报告缺失</span>';
  return `
    <div class="eval-card">
      <div class="eval-head">
        <div>
          <p class="eyebrow">评估 · ${item.model?.toUpperCase() || "CNN"}</p>
          <h3>${item.dataset_name} (${item.dataset})</h3>
          <div class="muted tiny">${hyper}</div>
        </div>
        ${reportLink}
      </div>
      <div class="eval-body">
        <div class="eval-metrics">${metricHtml || '<div class="muted tiny">报告中未解析到指标</div>'}</div>
        <div class="eval-legend">${legendHtml}</div>
      </div>
      <div class="thumb-grid">${visualHtml || '<div class="muted tiny">暂未生成可视化</div>'}</div>
    </div>
  `;
}

function renderEvalSection(items, containerId, emptyText) {
  const box = $(containerId);
  if (!box) return;
  box.innerHTML = ""; // 已迁移到新对比视图
}

function renderComparisons() {
  renderEvalButtons();
}

function renderEvaluations() {
  renderDashboard();
}

function renderEvalButtons() {
  const box = $("evalButtons");
  if (!box) return;
  const comps = state.evalSummary?.comparisons || [];
  const cnnMap = Object.fromEntries((state.evalSummary?.cnn || []).map((i) => [i.dataset, i]));
  const svmMap = Object.fromEntries((state.evalSummary?.svm || []).map((i) => [i.dataset, i]));
  if (!comps.length) {
    box.innerHTML = '<div class="muted">暂无对比数据，先运行 CNN 与 SVM 训练。</div>';
    return;
  }
  if (!state.selectedEvalDataset) {
    state.selectedEvalDataset = comps[0].dataset;
  }
  const fmt = (v) => (v === null || v === undefined ? "--" : Number(v).toFixed(2));
  box.innerHTML = comps
    .map((c) => {
      const cnn = cnnMap[c.dataset];
      const svm = svmMap[c.dataset];
      const cnnAcc = c.cnn_accuracy ?? cnn?.metrics?.overall_accuracy_percent ?? null;
      const svmAcc = c.svm_accuracy ?? svm?.metrics?.overall_accuracy_percent ?? null;
      const active = state.selectedEvalDataset === c.dataset ? "active" : "";
      const badge =
        c.better === "cnn"
          ? '<span class="badge online">CNN</span>'
          : c.better === "svm"
          ? '<span class="badge online">SVM</span>'
          : '<span class="badge ghost">待对比</span>';
      return `
        <button class="eval-pill ${active}" data-ds="${c.dataset}">
          <div>
            <div class="pill-title">${c.dataset_name} (${c.dataset})</div>
            <div class="pill-sub">CNN ${fmt(cnnAcc)} · SVM ${fmt(svmAcc)}</div>
          </div>
          ${badge}
        </button>
      `;
    })
    .join("");
  qsa(".eval-pill").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.selectedEvalDataset = btn.dataset.ds;
      renderEvalButtons();
      renderEvalDetail();
    })
  );
}

function modelDetailHTML(item, label) {
  if (!item) {
    return `<div class="detail-card"><div class="detail-head"><strong>${label}</strong><span class="badge ghost">暂无</span></div><div class="muted tiny">尚未生成 ${label} 报告，可先运行训练。</div></div>`;
  }
  const metrics = item.metrics || {};
  const urls = item.artifacts?.urls || {};
  const visuals = [
    ["预测", urls.prediction || item.artifacts?.prediction_path],
    ["GT", urls.groundtruth || item.artifacts?.groundtruth_path],
    ["混淆矩阵", urls.confusion || item.artifacts?.confusion_path],
    ["伪彩色", urls.pseudocolor || item.artifacts?.pseudocolor_path],
    ["分类图", urls.classification || item.artifacts?.classification_path],
    ["对比图", urls.comparison || item.artifacts?.comparison_path],
    ["错误图", urls.error_map || item.artifacts?.error_map_path],
  ].filter(([, url]) => !!url);

  const metricHtml = Object.entries(metrics)
    .map(
      ([k, v]) =>
        `<div class="metric-chip"><span class="muted tiny">${k}</span><div><strong>${Number(v).toFixed(3)}</strong></div></div>`
    )
    .join("");
  const links = [
    ["报告", item.report_url || item.report_path],
    ["模型", urls.model || item.artifacts?.model_path],
    ["预测", urls.prediction || item.artifacts?.prediction_path],
    ["混淆矩阵", urls.confusion || item.artifacts?.confusion_path],
  ]
    .filter(([, u]) => !!u)
    .map(([t, u]) => `<a class="tiny-link" href="${toUrl(u)}" target="_blank">${t}</a>`)
    .join("");
  const visualHtml = visuals
    .map(
      ([t, u]) => `
      <div>
        <p class="muted tiny">${t}</p>
        <a href="${toUrl(u)}" target="_blank"><img src="${toUrl(u)}" alt="${t}" /></a>
      </div>`
    )
    .join("");

  return `
    <div class="detail-card">
      <div class="detail-head">
        <strong>${label}</strong>
        <span class="badge ghost">PCA ${item.pca_components} · Window ${item.window_size}</span>
      </div>
      <div class="detail-metrics">${metricHtml || '<div class="muted tiny">报告中未解析到指标</div>'}</div>
      <div class="detail-links">${links || ""}</div>
      <div class="detail-visuals">${visualHtml || '<div class="muted tiny">暂无可视化</div>'}</div>
    </div>
  `;
}

function renderEvalDetail() {
  const box = $("evalDetail");
  if (!box) return;
  const comps = state.evalSummary?.comparisons || [];
  if (!comps.length) {
    box.innerHTML = '<div class="muted">暂无评估数据，先运行 CNN 与 SVM 训练。</div>';
    return;
  }
  const cnnMap = Object.fromEntries((state.evalSummary?.cnn || []).map((i) => [i.dataset, i]));
  const svmMap = Object.fromEntries((state.evalSummary?.svm || []).map((i) => [i.dataset, i]));
  const selected = state.selectedEvalDataset || comps[0].dataset;
  const comp = comps.find((c) => c.dataset === selected) || comps[0];
  const badge =
    comp?.better === "cnn"
      ? '<span class="badge online">CNN 更优</span>'
      : comp?.better === "svm"
      ? '<span class="badge online">SVM 更优</span>'
      : '<span class="badge ghost">待对比</span>';
  const cnn = cnnMap[comp.dataset];
  const svm = svmMap[comp.dataset];
  const fmt = (v) => (v === null || v === undefined ? "--" : Number(v).toFixed(2));
  const metricsToCompare = [
    ["Test accuracy (%)", "test_accuracy_percent"],
    ["Kappa accuracy (%)", "kappa_percent"],
    ["Overall accuracy (%)", "overall_accuracy_percent"],
    ["Average accuracy (%)", "average_accuracy_percent"],
    ["Test loss (%)", "test_loss_percent"],
  ];
  const compareRows = metricsToCompare
    .map(([label, key]) => {
      const cnnVal = cnn?.metrics?.[key];
      const svmVal = svm?.metrics?.[key];
      const cnnWidth = Math.max(0, Math.min(100, Number(cnnVal) || 0));
      const svmWidth = Math.max(0, Math.min(100, Number(svmVal) || 0));
      const show = cnnVal !== undefined || svmVal !== undefined;
      if (!show) return "";
      return `
        <div class="compare-row">
          <div class="compare-head">
            <span>${label}</span>
            <span class="muted tiny">CNN ${fmt(cnnVal)} · SVM ${fmt(svmVal)}</span>
          </div>
          <div class="compare-bars">
            <div>
              <div class="label">CNN</div>
              <div class="compare-bar cnn"><div class="fill" style="width:${cnnWidth}%"></div></div>
            </div>
            <div>
              <div class="label">SVM</div>
              <div class="compare-bar svm"><div class="fill" style="width:${svmWidth}%"></div></div>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
  box.innerHTML = `
    <div class="eval-detail">
      <div class="eval-detail-header">
        <div>
          <p class="eyebrow">${comp.dataset}</p>
          <h3>${comp.dataset_name}</h3>
          <div class="dashboard-meta">CNN Acc ${fmt(comp.cnn_accuracy)} · SVM Acc ${fmt(comp.svm_accuracy)}</div>
        </div>
        ${badge}
      </div>
      <div class="compare-table">${compareRows || '<div class="muted tiny">报告中未找到可对比的指标</div>'}</div>
      <div class="detail-grid">
        ${modelDetailHTML(cnn, "CNN · HybridSN")}
        ${modelDetailHTML(svm, "SVM")}
      </div>
    </div>
  `;
}

function detectModel(data, fallback = "cnn") {
  const path = data?.artifacts?.model_path || "";
  if (path.endsWith(".joblib") || path.includes("/svm/")) return "svm";
  return fallback;
}

function renderRunResult(data, modelHint) {
  if (!data) return;
  const model = modelHint || detectModel(data);
  const prefix = model === "svm" ? "Svm" : "Cnn";
  renderClassLegend(data.class_names || getClassNames(data.dataset));
  const setImg = (suffix, url) => {
    const img = $(suffix);
    if (!img) return;
    if (url) {
      img.src = toUrl(url);
      img.parentElement?.classList?.remove("muted");
    } else {
      img.removeAttribute("src");
      img.parentElement?.classList?.add("muted");
    }
  };
  const metricsBox = $(`metricsBox${prefix}`);
  if (metricsBox) {
    const metrics = data.metrics || {};
    if (!Object.keys(metrics).length) {
      const placeholder =
        data.status === "running" || data.status === "pending"
          ? "训练/推理进行中，完成后将自动更新指标"
          : "推理模式无报告，或尚未生成";
      metricsBox.innerHTML = `<div class="muted">${placeholder}</div>`;
    } else {
      metricsBox.innerHTML = Object.entries(metrics)
        .map(([k, v]) => `<div class="metric-row"><span>${k}</span><strong>${Number(v).toFixed(3)}</strong></div>`)
        .join("");
    }
  }
  const statusText = readableStatus(data);
  if ($(`runMessage${prefix}`)) $(`runMessage${prefix}`).textContent = data.message || statusText || "";
  const art = data.artifacts || {};
  const urls = art.urls || {};
  setImg(`imgPrediction${prefix}`, urls.prediction || art.prediction_path);
  setImg(`imgConfusion${prefix}`, urls.confusion || art.confusion_path);
  setImg(`imgGT${prefix}`, urls.groundtruth || art.groundtruth_path);
  setImg(`imgPseudo${prefix}`, urls.pseudocolor || art.pseudocolor_path);
  setImg(`imgClassify${prefix}`, urls.classification || art.classification_path);
  setImg(`imgError${prefix}`, urls.error_map || art.error_map_path);
  setImg(`imgCompare${prefix}`, urls.comparison || art.comparison_path);
  const linkBox = $(`artifactLinks${prefix}`);
  if (linkBox) {
    linkBox.innerHTML = "";
    const pairs = [
      ["模型", urls.model || art.model_path],
      ["PCA", urls.pca || art.pca_path],
      ["报告", urls.report || art.report_path],
      ["混淆矩阵", urls.confusion || art.confusion_path],
      ["预测", urls.prediction || art.prediction_path],
      ["GT", urls.groundtruth || art.groundtruth_path],
      ["伪彩色", urls.pseudocolor || art.pseudocolor_path],
      ["分类图", urls.classification || art.classification_path],
      ["错误图", urls.error_map || art.error_map_path],
      ["对比图", urls.comparison || art.comparison_path],
    ];
    pairs.forEach(([label, url]) => {
      if (!url) return;
      const a = document.createElement("a");
      a.href = toUrl(url);
      a.target = "_blank";
      a.textContent = label;
      linkBox.appendChild(a);
    });
  }
  const logsBox = $(`runLogs${prefix}`);
  if (logsBox) logsBox.textContent = (data.logs_tail || []).join("\n");
}

async function runHybrid(mode) {
  stopPolling("cnn");
  updateProgressUI("cnn", 2, "准备启动");
  const isInfer = mode === "infer";
  const payload = gatherParams();
  payload.inference_only = isInfer;
  if (payload.inference_only) {
    const manual = $("modelPathManual").value.trim();
    payload.input_model_path = $("modelPathSelect").value || manual || defaultModelPath();
  }
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastRun = data;
    renderRunResult(data, "cnn");
    updateProgressUI("cnn", data.progress ?? 5, readableStatus(data));
    if (data.job_id) {
      log(`已提交 HybridSN ${isInfer ? "推理" : "训练"} 任务，Job ID: ${data.job_id}`);
      pollJob(data.job_id, "cnn");
    } else {
      await loadArtifacts();
      await loadEvaluations();
      log(data.message || `HybridSN ${isInfer ? "推理" : "训练"}完成`);
    }
  } catch (err) {
    stopPolling();
    updateProgressUI("cnn", 0, "执行失败");
    log(`执行失败: ${err.message}`);
  }
}

async function runSvm(mode) {
  stopPolling("svm");
  updateProgressUI("svm", 2, "准备启动");
  const isInfer = mode === "infer";
  const payload = gatherSvmParams();
  payload.inference_only = isInfer;
  if (payload.inference_only) {
    const manual = $("svmModelPathManual")?.value.trim();
    payload.input_model_path = $("svmModelPathSelect")?.value || manual || defaultSvmModelPath();
  }
  try {
    const data = await fetchJSON(`${API_BASE}/api/svm/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastRun = data;
    renderRunResult(data, "svm");
    updateProgressUI("svm", data.progress ?? 5, readableStatus(data));
    if (data.job_id) {
      log(`已提交 SVM ${isInfer ? "推理" : "训练"} 任务，Job ID: ${data.job_id}`);
      pollJob(data.job_id, "svm");
    } else {
      await loadArtifacts();
      await loadEvaluations();
      log(data.message || `SVM ${isInfer ? "推理" : "训练"}完成`);
    }
  } catch (err) {
    stopPolling();
    updateProgressUI("svm", 0, "执行失败");
    log(`执行失败: ${err.message}`);
  }
}

function bindEvents() {
  qsa(".nav-item").forEach((btn) => {
    if (btn.dataset.view) {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    }
  });
  if ($("btnRefreshDatasets")) $("btnRefreshDatasets").addEventListener("click", refreshDatasets);
  if ($("btnTrain")) $("btnTrain").addEventListener("click", () => runHybrid("train"));
  if ($("btnInfer")) $("btnInfer").addEventListener("click", () => runHybrid("infer"));
  if ($("btnSvmTrain")) $("btnSvmTrain").addEventListener("click", () => runSvm("train"));
  if ($("btnSvmInfer")) $("btnSvmInfer").addEventListener("click", () => runSvm("infer"));
  if ($("btnResetParams"))
    $("btnResetParams").addEventListener("click", () => {
      setDefaultParams();
      setSvmDefaultParams();
      log("已恢复默认超参");
    });
  if ($("trainDataset"))
    $("trainDataset").addEventListener("change", (e) => {
      renderClassLegend(getClassNames(e.target.value));
       applyDefaultModelHints();
    });
  if ($("btnFillDefaultModel"))
    $("btnFillDefaultModel").addEventListener("click", () => {
      const path = defaultModelPath();
      const input = $("modelPathManual");
      if (input) input.value = path;
      log(`已填入默认模型路径: ${path}`);
    });
  if ($("btnFillSvmDefaultModel"))
    $("btnFillSvmDefaultModel").addEventListener("click", () => {
      const path = defaultSvmModelPath();
      const input = $("svmModelPathManual");
      if (input) input.value = path;
      log(`已填入默认 SVM 模型路径: ${path}`);
    });
  if ($("btnDocs")) $("btnDocs").addEventListener("click", () => window.open(`${API_BASE}/docs`, "_blank"));
  if ($("btnRefreshEval")) $("btnRefreshEval").addEventListener("click", () => refreshEvaluations(true));
}

async function init() {
  bindEvents();
  updateProgressUI("cnn", 0, "等待执行");
  updateProgressUI("svm", 0, "等待执行");
  await loadDefaults();
  await loadArtifacts();
  setSvmDefaultParams();
  applyDefaultModelHints();
  await loadEvaluations();
  log("前端就绪，按顺序进行数据→训练→查看产物");
}

init();
