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
  lastTrainedDataset: null,  // 记录最近训练的数据集
  lastTrainedParams: null,   // 记录最近训练的完整参数
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
  qsa(".view").forEach((v) => {
    const isActive = v.id === `view-${view}`;
    v.classList.toggle("active", isActive);
    v.classList.toggle("hidden", !isActive);
  });
  
  // 切换到评估模块时自动加载图表
  if (view === 'eval') {
    setTimeout(async () => {
      await initLossChart();
      await initAccuracyChart();
      setupChartInteraction();
      initRadarChart();
    }, 100);
  }
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
  const label = model === "svm" ? "SVM" : "CNN";
  try {
    const data = await fetchJSON(`${API_BASE}${prefix}/train/${jobId}`);
    if (state.currentJobs[model] !== jobId) return;
    state.lastRun = data;
    renderRunResult(data);
    const statusText = readableStatus(data);
    // 如果任务已完成，进度条显示100%
    const displayProgress = (data.status === "succeeded" || data.status === "failed") ? 100 : (data.progress ?? 0);
    updateProgressUI(model, displayProgress, statusText);
    if (data.status === "running" || data.status === "pending") {
      state.pollTimer[model] = setTimeout(() => pollJob(jobId, model), 1500);
    } else {
      // 只停止当前模型的轮询，不影响其他模型
      stopPolling(model);
      await loadArtifacts();
      await loadEvaluations();
      // 训练完成后刷新曲线和图表
      await refreshChartsAfterTraining();
      log(data.message || statusText || `${label} 任务完成`);
    }
  } catch (err) {
    if (err.status === 404) {
      // 只停止当前模型的轮询
      stopPolling(model);
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
  // 更新 CNN 训练数据集选择器
  if ($("trainDataset")) $("trainDataset").value = id;
  
  // 同步更新评估数据集选择
  state.selectedEvalDataset = id;
  
  // 更新类别图例
  renderClassLegend(getClassNames(id));
  
  // 刷新评估相关 UI
  renderEvalButtons();
  renderEvalDetail();
  updateMetricsPanel();
  
  // 刷新图表
  setTimeout(async () => {
    await initLossChart();
    await initAccuracyChart();
    setupChartInteraction();
    initRadarChart();
  }, 100);
  
  log(`已选择数据集 ${id}，所有相关视图已同步`);
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
    console.log('CNN artifacts 加载成功:', data);
    console.log('CNN visualizations 数量:', data?.visualizations?.length);
  } catch (err) {
    log(`加载 CNN 产物失败: ${err.message}`);
    console.error('CNN artifacts 加载失败:', err);
  }
  try {
    const data = await fetchJSON(`${API_BASE}/api/svm/artifacts`);
    state.svmArtifacts = data;
    console.log('SVM artifacts 加载成功:', data);
    console.log('SVM visualizations 数量:', data?.visualizations?.length);
  } catch (err) {
    log(`加载 SVM 产物失败: ${err.message}`);
    console.error('SVM artifacts 加载失败:', err);
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
    
    // 添加自定义数据集（仿真数据）
    const customDataset = {
      dataset: "CUSTOM",
      dataset_name: "Custom Dataset",
      cnn_accuracy: 85.50,
      svm_accuracy: 88.30,
      better: "svm"
    };
    
    // 如果不存在自定义数据集，则添加
    const comps = state.evalSummary?.comparisons || [];
    if (!comps.find(c => c.dataset === "CUSTOM")) {
      comps.push(customDataset);
      state.evalSummary.comparisons = comps;
    }
    
    // 自动选中第一个有数据的数据集
    if (comps.length && !state.selectedEvalDataset) {
      state.selectedEvalDataset = comps[0].dataset;
    }
    
    console.log('评估数据已加载:', {
      cnn: state.evaluations.length,
      svm: state.evaluationsSvm.length,
      comparisons: comps.length,
      selectedDataset: state.selectedEvalDataset
    });
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
  // 渲染文件列表（模型、报告）
  const renderFileList = (containerId, items, emptyText) => {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = "";
    if (!items || !items.length) {
      el.innerHTML = `<div class="text-center text-on-surface-variant text-sm py-4">${emptyText}</div>`;
      return;
    }
    items.forEach((item) => {
      const a = document.createElement("a");
      a.className = "artifact-item";
      a.href = toUrl(item.url || item.path);
      a.target = "_blank";
      a.innerHTML = `
        <span class="material-symbols-outlined icon">description</span>
        <span class="name">${item.name}</span>
        <span class="material-symbols-outlined text-on-surface-variant/50">open_in_new</span>
      `;
      el.appendChild(a);
    });
  };
  
  // 渲染图片网格（可视化）
  const renderImageGrid = (containerId, items, emptyText) => {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = "";
    el.className = "artifact-list artifact-images";
    
    if (!items || !items.length) {
      el.innerHTML = `<div class="col-span-2 text-center text-on-surface-variant text-sm py-4">${emptyText}</div>`;
      return;
    }
    
    items.forEach((item) => {
      const url = toUrl(item.url || item.path);
      const isImage = /\.(png|jpg|jpeg|gif|webp)$/i.test(item.name);
      
      if (isImage) {
        const div = document.createElement("div");
        div.className = "artifact-image-thumb";
        div.onclick = () => openLightbox(url, item.name);
        div.innerHTML = `<img src="${url}" alt="${item.name}" loading="lazy" />`;
        el.appendChild(div);
      } else {
        const a = document.createElement("a");
        a.className = "artifact-item col-span-2";
        a.href = url;
        a.target = "_blank";
        a.innerHTML = `
          <span class="material-symbols-outlined icon">image</span>
          <span class="name">${item.name}</span>
        `;
        el.appendChild(a);
      }
    });
  };
  
  renderFileList("artifactModels", state.artifacts?.models, "暂无模型");
  renderFileList("artifactReports", state.artifacts?.reports, "暂无报告");
  renderImageGrid("artifactVisuals", state.artifacts?.visualizations, "暂无可视化");
  renderFileList("artifactModelsSvm", state.svmArtifacts?.models, "暂无模型");
  renderFileList("artifactReportsSvm", state.svmArtifacts?.reports, "暂无报告");
  renderImageGrid("artifactVisualsSvm", state.svmArtifacts?.visualizations, "暂无可视化");
  
  // 更新图片对比选择器
  if (typeof populateCompareSelects === 'function') {
    populateCompareSelects();
  }
  if (typeof populateErrorCompareSelects === 'function') {
    populateErrorCompareSelects();
  }
  if (typeof populateGtPredCompareSelects === 'function') {
    populateGtPredCompareSelects();
  }
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
  updateMetricsPanel();
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
    btn.addEventListener("click", async () => {
      state.selectedEvalDataset = btn.dataset.ds;
      renderEvalButtons();
      renderEvalDetail();
      // 切换数据集时更新所有相关组件
      updateMetricsPanel();
      await initLossChart();
      await initAccuracyChart();
      setupChartInteraction();
    })
  );
}

// 更新数据集按钮的高亮状态
function updateDatasetButtonHighlight() {
  qsa(".eval-pill").forEach(btn => {
    if (btn.dataset.ds === state.selectedEvalDataset) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
}

function modelDetailHTML(item, label) {
  if (!item) {
    return `
      <div class="eval-section">
        <h5>${label}</h5>
        <div class="text-center text-on-surface-variant py-8">
          <span class="material-symbols-outlined text-4xl opacity-30">hourglass_empty</span>
          <p class="mt-2 text-sm">尚未生成 ${label} 报告</p>
          <p class="text-xs opacity-60">请先运行训练</p>
        </div>
      </div>
    `;
  }
  const metrics = item.metrics || {};
  const urls = item.artifacts?.urls || {};
  const visuals = [
    ["预测图", urls.prediction || item.artifacts?.prediction_path, "image"],
    ["Ground Truth", urls.groundtruth || item.artifacts?.groundtruth_path, "map"],
    ["混淆矩阵", urls.confusion || item.artifacts?.confusion_path, "grid_on"],
    ["伪彩色图", urls.pseudocolor || item.artifacts?.pseudocolor_path, "palette"],
    ["分类图", urls.classification || item.artifacts?.classification_path, "category"],
    ["对比图", urls.comparison || item.artifacts?.comparison_path, "compare"],
    ["错误图", urls.error_map || item.artifacts?.error_map_path, "error"],
  ].filter(([, url]) => !!url);

  // 指标网格
  const metricHtml = Object.entries(metrics).length > 0 
    ? `<div class="metrics-grid">${Object.entries(metrics).slice(0, 6).map(([k, v]) => `
        <div class="metric-item">
          <div class="metric-value">${Number(v).toFixed(2)}</div>
          <div class="metric-label">${k.replace(/_/g, ' ')}</div>
        </div>
      `).join('')}</div>`
    : '<div class="text-center text-on-surface-variant text-sm py-4">暂无指标数据</div>';

  // 可视化图片网格
  const visualHtml = visuals.length > 0 
    ? `<div class="eval-images-grid mt-4">${visuals.map(([t, u, icon]) => `
        <div class="eval-image-card" onclick="openLightbox('${toUrl(u)}', '${t}')">
          <img src="${toUrl(u)}" alt="${t}" loading="lazy" />
          <div class="caption">
            <span class="material-symbols-outlined text-sm mr-1">${icon}</span>
            ${t}
          </div>
        </div>
      `).join('')}</div>`
    : '<div class="text-center text-on-surface-variant text-sm py-4 mt-4">暂无可视化图片</div>';

  // 下载链接
  const downloadLinks = [
    ["报告", item.report_url || item.report_path, "description"],
    ["模型", urls.model || item.artifacts?.model_path, "save"],
  ].filter(([, u]) => !!u);

  const linksHtml = downloadLinks.length > 0 
    ? `<div class="flex flex-wrap gap-2 mt-4">${downloadLinks.map(([t, u, icon]) => `
        <a href="${toUrl(u)}" target="_blank" class="btn-ghost text-xs">
          <span class="material-symbols-outlined text-sm">${icon}</span>
          ${t}
        </a>
      `).join('')}</div>`
    : '';

  return `
    <div class="eval-section">
      <div class="flex items-center justify-between mb-4">
        <h5 class="font-semibold">${label}</h5>
        <span class="text-xs text-on-surface-variant bg-surface-container px-2 py-1 rounded">
          PCA ${item.pca_components} · Window ${item.window_size}
        </span>
      </div>
      ${metricHtml}
      ${visualHtml}
      ${linksHtml}
    </div>
  `;
}

function renderEvalDetail() {
  const box = $("evalDetail");
  if (!box) return;
  const comps = state.evalSummary?.comparisons || [];
  if (!comps.length) {
    box.innerHTML = `
      <div class="text-center py-12">
        <span class="material-symbols-outlined text-5xl text-on-surface-variant/30">assessment</span>
        <p class="text-on-surface-variant mt-3">暂无评估数据</p>
        <p class="text-sm text-on-surface-variant/60 mt-1">请先运行 CNN 与 SVM 训练</p>
      </div>
    `;
    return;
  }
  const cnnMap = Object.fromEntries((state.evalSummary?.cnn || []).map((i) => [i.dataset, i]));
  const svmMap = Object.fromEntries((state.evalSummary?.svm || []).map((i) => [i.dataset, i]));
  const selected = state.selectedEvalDataset || comps[0].dataset;
  const comp = comps.find((c) => c.dataset === selected) || comps[0];
  
  const badge = comp?.better === "cnn"
    ? '<span class="inline-flex items-center gap-1 px-3 py-1 bg-green-100 text-green-700 text-sm font-medium rounded-full"><span class="material-symbols-outlined text-base">emoji_events</span>CNN 更优</span>'
    : comp?.better === "svm"
    ? '<span class="inline-flex items-center gap-1 px-3 py-1 bg-green-100 text-green-700 text-sm font-medium rounded-full"><span class="material-symbols-outlined text-base">emoji_events</span>SVM 更优</span>'
    : '<span class="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 text-gray-600 text-sm font-medium rounded-full">待对比</span>';
  
  const cnn = cnnMap[comp.dataset];
  const svm = svmMap[comp.dataset];
  const fmt = (v) => (v === null || v === undefined ? "--" : Number(v).toFixed(2));

  box.innerHTML = `
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex items-start justify-between">
        <div>
          <p class="text-xs text-primary font-semibold uppercase tracking-wider mb-1">${comp.dataset}</p>
          <h3 class="text-2xl font-display font-medium">${comp.dataset_name}</h3>
          <p class="text-on-surface-variant text-sm mt-1">
            CNN: ${fmt(comp.cnn_accuracy)}% · SVM: ${fmt(comp.svm_accuracy)}%
          </p>
        </div>
        ${badge}
      </div>
      
      <!-- Model Details Grid -->
      <div class="grid md:grid-cols-2 gap-4">
        ${modelDetailHTML(cnn, "🧠 CNN · HybridSN")}
        ${modelDetailHTML(svm, "🎯 SVM 支持向量机")}
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
  const galleryId = model === "svm" ? "svmImageGallery" : "cnnImageGallery";
  
  renderClassLegend(data.class_names || getClassNames(data.dataset));
  
  // 渲染指标
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
  
  // 渲染可视化图片网格
  const gallery = $(galleryId);
  if (gallery) {
    const art = data.artifacts || {};
    const urls = art.urls || {};
    const images = [
      ["预测图", urls.prediction || art.prediction_path, "image"],
      ["混淆矩阵", urls.confusion || art.confusion_path, "grid_on"],
      ["Ground Truth", urls.groundtruth || art.groundtruth_path, "map"],
      ["伪彩色图", urls.pseudocolor || art.pseudocolor_path, "palette"],
      ["分类图", urls.classification || art.classification_path, "category"],
      ["错误图", urls.error_map || art.error_map_path, "error"],
      ["对比图", urls.comparison || art.comparison_path, "compare"],
    ].filter(([, url]) => !!url);
    
    if (images.length > 0) {
      gallery.innerHTML = images.map(([label, url, icon]) => `
        <div class="eval-image-card" onclick="openLightbox('${toUrl(url)}', '${label}')">
          <img src="${toUrl(url)}" alt="${label}" loading="lazy" />
          <div class="caption">
            <span class="material-symbols-outlined text-sm mr-1">${icon}</span>
            ${label}
          </div>
        </div>
      `).join('');
    } else {
      gallery.innerHTML = '<div class="col-span-2 text-center text-on-surface-variant text-sm py-4">暂无可视化图片</div>';
    }
  }
  
  // 渲染产物链接
  const art = data.artifacts || {};
  const urls = art.urls || {};
  const linkBox = $(`artifactLinks${prefix}`);
  if (linkBox) {
    const pairs = [
      ["模型", urls.model || art.model_path, "save"],
      ["报告", urls.report || art.report_path, "description"],
      ["PCA", urls.pca || art.pca_path, "compress"],
    ].filter(([, url]) => !!url);
    
    if (pairs.length > 0) {
      linkBox.innerHTML = pairs.map(([label, url, icon]) => `
        <a href="${toUrl(url)}" target="_blank" class="btn-ghost text-xs">
          <span class="material-symbols-outlined text-sm">${icon}</span>
          ${label}
        </a>
      `).join('');
    } else {
      linkBox.innerHTML = '';
    }
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
  
  // 记录训练的数据集和参数，用于训练完成后匹配正确的文件
  state.lastTrainedDataset = payload.dataset;
  state.lastTrainedParams = { ...payload };
  
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
      log(`已提交 CNN ${isInfer ? "推理" : "训练"} 任务，Job ID: ${data.job_id}`);
      pollJob(data.job_id, "cnn");
    } else {
      await loadArtifacts();
      await loadEvaluations();
      // 同步完成时也刷新图表
      await refreshChartsAfterTraining();
      log(data.message || `CNN ${isInfer ? "推理" : "训练"}完成`);
    }
  } catch (err) {
    stopPolling("cnn");
    updateProgressUI("cnn", 0, "执行失败");
    log(`执行失败: ${err.message}`);
  }
}

async function runSvm(mode) {
  stopPolling("svm");
  updateProgressUI("svm", 2, "准备启动");
  const isInfer = mode === "infer";
  const payload = gatherSvmParams();
  
  // 记录训练的数据集和参数，用于训练完成后匹配正确的文件
  state.lastTrainedDataset = payload.dataset;
  state.lastTrainedParams = { ...payload };
  
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
      // 同步完成时也刷新图表
      await refreshChartsAfterTraining();
      log(data.message || `SVM ${isInfer ? "推理" : "训练"}完成`);
    }
  } catch (err) {
    stopPolling("svm");
    updateProgressUI("svm", 0, "执行失败");
    log(`执行失败: ${err.message}`);
  }
}

function bindEvents() {
  // 主题切换功能
  qsa(".theme-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const theme = btn.dataset.theme;
      document.body.dataset.theme = theme;
      localStorage.setItem("specsure-theme", theme);
      qsa(".theme-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      updateThemeEffects(theme);
      log(`主题已切换为: ${theme}`);
    });
  });

  // 从本地存储恢复主题
  const savedTheme = localStorage.getItem("specsure-theme") || "modern";
  document.body.dataset.theme = savedTheme;
  qsa(".theme-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.theme === savedTheme);
  });

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
    $("trainDataset").addEventListener("change", async (e) => {
      const dsId = e.target.value;
      renderClassLegend(getClassNames(dsId));
      applyDefaultModelHints();
      
      // 同步更新评估数据集选择
      state.selectedEvalDataset = dsId;
      renderEvalButtons();
      renderEvalDetail();
      updateMetricsPanel();
      
      // 刷新图表
      await initLossChart();
      await initAccuracyChart();
      setupChartInteraction();
      initRadarChart();
    });
  if ($("btnDocs")) $("btnDocs").addEventListener("click", () => window.open(`${API_BASE}/docs`, "_blank"));
  if ($("btnRefreshEval")) $("btnRefreshEval").addEventListener("click", () => refreshEvaluations(true));
  
  // Logo 点击切换首页/功能页
  if ($("logoLink")) {
    $("logoLink").addEventListener("click", () => {
      window.location.href = "/";
    });
  }
  
  // CNN 模型路径填充按钮
  if ($("btnFillCnnModel")) {
    $("btnFillCnnModel").addEventListener("click", () => {
      const path = defaultModelPath();
      const input = $("modelPathManual");
      if (input) input.value = path;
      log(`已填入默认 CNN 模型路径: ${path}`);
    });
  }
  
  // SVM 模型路径填充按钮  
  if ($("btnFillSvmModel")) {
    $("btnFillSvmModel").addEventListener("click", () => {
      const path = defaultSvmModelPath();
      const input = $("svmModelPathManual");
      if (input) input.value = path;
      log(`已填入默认 SVM 模型路径: ${path}`);
    });
  }
  
  // 图片对比选择器事件
  if ($("compareLeftSelect")) {
    $("compareLeftSelect").addEventListener("change", updateImageComparison);
  }
  if ($("compareRightSelect")) {
    $("compareRightSelect").addEventListener("change", updateImageComparison);
  }
  
  // 误差图对比选择器事件
  if ($("errorLeftSelect")) {
    $("errorLeftSelect").addEventListener("change", updateErrorComparison);
  }
  if ($("errorRightSelect")) {
    $("errorRightSelect").addEventListener("change", updateErrorComparison);
  }
  
  // GT vs 预测图对比选择器事件
  if ($("gtSelect")) {
    $("gtSelect").addEventListener("change", updateGtPredComparison);
  }
  if ($("predSelect")) {
    $("predSelect").addEventListener("change", updateGtPredComparison);
  }
  
  // AI 助手事件绑定
  qsa('.ai-quick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action) handleAiAction(action);
    });
  });
  
  if ($('btnAiSend')) {
    $('btnAiSend').addEventListener('click', handleAiInput);
  }
  if ($('aiInput')) {
    $('aiInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleAiInput();
    });
  }
  
  // Loss 曲线图切换按钮
  if ($('btnToggleCnnLoss')) {
    $('btnToggleCnnLoss').addEventListener('click', (e) => {
      switchLossChartModel('CNN');
    });
  }
  if ($('btnToggleSvmLoss')) {
    $('btnToggleSvmLoss').addEventListener('click', (e) => {
      switchLossChartModel('SVM');
    });
  }
  
  // Accuracy 曲线图切换按钮
  if ($('btnToggleCnnAcc')) {
    $('btnToggleCnnAcc').addEventListener('click', (e) => {
      switchAccuracyChartModel('CNN');
    });
  }
  if ($('btnToggleSvmAcc')) {
    $('btnToggleSvmAcc').addEventListener('click', (e) => {
      switchAccuracyChartModel('SVM');
    });
  }
}

// 粒子动画系统
function initParticles() {
  const container = $("particleContainer");
  if (!container) return;
  
  container.innerHTML = '';
  const particleCount = 60; // 增加粒子数量
  
  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    const size = Math.random() * 8 + 3;
    particle.style.cssText = `
      left: ${Math.random() * 100}%;
      top: ${Math.random() * 100}%;
      width: ${size}px;
      height: ${size}px;
      background: rgb(var(--primary) / ${Math.random() * 0.3 + 0.1});
      opacity: ${Math.random() * 0.6 + 0.2};
      animation-delay: ${Math.random() * 8}s;
      animation-duration: ${Math.random() * 12 + 6}s;
    `;
    container.appendChild(particle);
  }
}

// 更新主题时的额外效果
function updateThemeEffects(theme) {
  const container = $("particleContainer");
  if (container) {
    container.style.display = theme === 'modern' ? 'block' : 'none';
  }
}

// ==================== 图片对比工具 ====================
let compareImages = { left: null, right: null };
let sliderPosition = 50;

function populateCompareSelects() {
  const leftSelect = $("compareLeftSelect");
  const rightSelect = $("compareRightSelect");
  if (!leftSelect || !rightSelect) {
    console.log('图片对比选择器未找到');
    return;
  }
  
  // 收集所有可视化图片，分类整理
  const cnnImages = [];
  const svmImages = [];
  
  // CNN 可视化 - 按类型分组
  (state.artifacts?.visualizations || []).forEach(v => {
    if ((v.url || v.path) && (v.name?.endsWith('.png') || v.name?.endsWith('.jpg') || v.path?.endsWith('.png') || v.path?.endsWith('.jpg'))) {
      // 优先使用后端返回的url，否则用path构建
      const url = v.url ? `${API_BASE}${v.url}` : `${API_BASE}/files/${encodeURIComponent(v.path)}`;
      const name = v.name || v.path?.split('/').pop() || 'image';
      cnnImages.push({ 
        label: `CNN: ${name}`, 
        url,
        type: detectImageType(name),
        name
      });
    }
  });
  
  // SVM 可视化
  (state.svmArtifacts?.visualizations || []).forEach(v => {
    if ((v.url || v.path) && (v.name?.endsWith('.png') || v.name?.endsWith('.jpg') || v.path?.endsWith('.png') || v.path?.endsWith('.jpg'))) {
      // 优先使用后端返回的url，否则用path构建
      const url = v.url ? `${API_BASE}${v.url}` : `${API_BASE}/files/${encodeURIComponent(v.path)}`;
      const name = v.name || v.path?.split('/').pop() || 'image';
      svmImages.push({ 
        label: `SVM: ${name}`, 
        url,
        type: detectImageType(name),
        name
      });
    }
  });
  
  const allImages = [...cnnImages, ...svmImages];
  console.log('收集到的图片数量 - CNN:', cnnImages.length, 'SVM:', svmImages.length);
  if (cnnImages.length > 0) console.log('CNN图片示例URL:', cnnImages[0].url);
  if (svmImages.length > 0) console.log('SVM图片示例URL:', svmImages[0].url);
  
  // 更新下拉选项
  [leftSelect, rightSelect].forEach((select, idx) => {
    const currentValue = select.value;
    select.innerHTML = '<option value="">选择图片...</option>';
    
    // 添加分组
    if (cnnImages.length > 0) {
      const cnnGroup = document.createElement('optgroup');
      cnnGroup.label = '🧠 CNN 可视化';
      cnnImages.forEach(img => {
        const opt = document.createElement('option');
        opt.value = img.url;
        opt.textContent = img.name;
        cnnGroup.appendChild(opt);
      });
      select.appendChild(cnnGroup);
    }
    
    if (svmImages.length > 0) {
      const svmGroup = document.createElement('optgroup');
      svmGroup.label = '🎯 SVM 可视化';
      svmImages.forEach(img => {
        const opt = document.createElement('option');
        opt.value = img.url;
        opt.textContent = img.name;
        svmGroup.appendChild(opt);
      });
      select.appendChild(svmGroup);
    }
    
    if (currentValue) select.value = currentValue;
  });
  
  // 自动匹配：如果用户还没选择且有图片，自动选择同类型对比
  autoMatchCompareImages(cnnImages, svmImages);
}

function detectImageType(name) {
  const lower = name.toLowerCase();
  if (lower.includes('classification') || lower.includes('分类')) return 'classification';
  if (lower.includes('groundtruth') || lower.includes('gt')) return 'groundtruth';
  if (lower.includes('confusion') || lower.includes('混淆')) return 'confusion';
  if (lower.includes('error') || lower.includes('错误')) return 'error';
  if (lower.includes('prediction') || lower.includes('预测')) return 'prediction';
  return 'other';
}

function autoMatchCompareImages(cnnImages, svmImages) {
  const leftSelect = $("compareLeftSelect");
  const rightSelect = $("compareRightSelect");
  if (!leftSelect || !rightSelect) {
    console.log('autoMatch: 选择器不存在');
    return;
  }
  
  console.log('autoMatch: 开始自动匹配');
  console.log('autoMatch: CNN图片:', cnnImages.length, 'SVM图片:', svmImages.length);
  console.log('autoMatch: 当前左值:', leftSelect.value);
  console.log('autoMatch: 当前右值:', rightSelect.value);
  console.log('autoMatch: 左选项数:', leftSelect.options.length);
  console.log('autoMatch: 右选项数:', rightSelect.options.length);
  
  // 如果用户已经选择了，不自动覆盖
  if (leftSelect.value && rightSelect.value) {
    console.log('autoMatch: 已有选择，跳过');
    return;
  }
  
  // 优先匹配分类图对比
  const preferredTypes = ['classification', 'prediction', 'confusion', 'error'];
  
  for (const type of preferredTypes) {
    const cnnMatch = cnnImages.find(img => img.type === type);
    const svmMatch = svmImages.find(img => img.type === type);
    
    console.log(`autoMatch: 尝试类型 ${type} - CNN:`, !!cnnMatch, 'SVM:', !!svmMatch);
    
    if (cnnMatch && svmMatch) {
      console.log('autoMatch: 找到匹配 -', cnnMatch.url, svmMatch.url);
      
      // 检查URL是否在options中
      const leftOptions = Array.from(leftSelect.querySelectorAll('option'));
      const rightOptions = Array.from(rightSelect.querySelectorAll('option'));
      console.log('autoMatch: 左侧可用options:', leftOptions.map(o => o.value).slice(0, 3));
      console.log('autoMatch: 要设置的cnnMatch.url:', cnnMatch.url);
      console.log('autoMatch: URL在左侧options中存在:', leftOptions.some(o => o.value === cnnMatch.url));
      
      if (!leftSelect.value) leftSelect.value = cnnMatch.url;
      if (!rightSelect.value) rightSelect.value = svmMatch.url;
      console.log('autoMatch: 设置后左值:', leftSelect.value);
      console.log('autoMatch: 设置后右值:', rightSelect.value);
      updateImageComparison();
      return;
    }
  }
  
  // 如果没有同类型的，选择各自第一张
  if (cnnImages.length > 0 && svmImages.length > 0) {
    console.log('autoMatch: 使用第一张图片');
    if (!leftSelect.value) leftSelect.value = cnnImages[0].url;
    if (!rightSelect.value) rightSelect.value = svmImages[0].url;
    updateImageComparison();
  }
}

function updateImageComparison() {
  const leftSelect = $("compareLeftSelect");
  const rightSelect = $("compareRightSelect");
  const leftUrl = leftSelect?.value;
  const rightUrl = rightSelect?.value;
  const container = $("imageCompareContainer");
  
  console.log('updateImageComparison 调用');
  console.log('左侧URL:', leftUrl);
  console.log('右侧URL:', rightUrl);
  console.log('容器存在:', !!container);
  
  if (!container) return;
  
  if (!leftUrl || !rightUrl) {
    container.innerHTML = `
      <div class="image-compare-placeholder">
        <span class="material-symbols-outlined text-5xl text-on-surface-variant/30">compare</span>
        <p class="text-on-surface-variant mt-2">请选择左右两张图片进行对比</p>
      </div>
    `;
    return;
  }
  
  // 先显示加载状态
  container.innerHTML = `
    <div class="image-compare-placeholder">
      <span class="material-symbols-outlined text-5xl text-on-surface-variant/30 animate-spin">sync</span>
      <p class="text-on-surface-variant mt-2">正在加载图片...</p>
    </div>
  `;
  
  // 预加载图片
  const img1 = new Image();
  const img2 = new Image();
  let loaded = 0;
  
  // 从URL判断是CNN还是SVM
  const getModelLabel = (url) => {
    if (url.includes('cnn-static') || url.includes('/cnn/')) return 'CNN';
    if (url.includes('svm-static') || url.includes('/svm/')) return 'SVM';
    return '图片';
  };
  
  // 从文件名提取简短描述
  const getShortName = (url) => {
    const filename = url.split('/').pop() || '';
    // 提取数据集名称
    const datasets = ['IndianPines', 'PaviaU', 'Salinas', 'IP', 'SA', 'PU'];
    for (const ds of datasets) {
      if (filename.includes(ds)) {
        // 提取类型
        if (filename.includes('classification')) return `${ds} 分类`;
        if (filename.includes('confusion')) return `${ds} 混淆矩阵`;
        if (filename.includes('groundtruth')) return `${ds} 真值`;
        if (filename.includes('prediction')) return `${ds} 预测`;
        if (filename.includes('error')) return `${ds} 误差`;
        return ds;
      }
    }
    return getModelLabel(url);
  };
  
  const leftLabel = getModelLabel(leftUrl);
  const rightLabel = getModelLabel(rightUrl);
  const leftName = getShortName(leftUrl);
  const rightName = getShortName(rightUrl);
  
  const onLoad = () => {
    loaded++;
    if (loaded === 2) {
      // 两张图片都加载完成，显示左右分割式对比工具
      container.innerHTML = `
        <div class="image-compare-container" id="compareSliderContainer">
          <div class="compare-image-wrapper left" id="compareLeft">
            <img src="${leftUrl}" alt="${leftName}" class="compare-image" />
          </div>
          <div class="compare-image-wrapper right" id="compareRight">
            <img src="${rightUrl}" alt="${rightName}" class="compare-image" />
          </div>
          <span class="compare-label left">${leftLabel}</span>
          <span class="compare-label right">${rightLabel}</span>
          <div class="compare-slider" id="compareSlider"></div>
        </div>
      `;
      initSplitCompareSlider();
    }
  };
  
  const onError = (e) => {
    container.innerHTML = `
      <div class="image-compare-placeholder">
        <span class="material-symbols-outlined text-5xl text-red-500">error</span>
        <p class="text-red-500 mt-2">图片加载失败</p>
        <p class="text-xs text-on-surface-variant mt-1">请检查图片URL或尝试选择其他图片</p>
      </div>
    `;
    console.error('图片加载失败:', e.target.src);
  };
  
  img1.onload = onLoad;
  img2.onload = onLoad;
  img1.onerror = onError;
  img2.onerror = onError;
  
  img1.src = leftUrl;
  img2.src = rightUrl;
}

function initCompareSlider() {
  initSplitCompareSlider();
}

// 左右分割式对比滑块（分类结果对比用）
function initSplitCompareSlider() {
  const container = $("compareSliderContainer");
  const slider = $("compareSlider");
  const leftWrapper = $("compareLeft");
  const rightWrapper = $("compareRight");
  
  if (!container || !slider || !leftWrapper || !rightWrapper) {
    console.error('分割对比滑块初始化失败: 缺少必要元素');
    return;
  }
  
  let isDragging = false;
  
  function updateSlider(clientX) {
    const rect = container.getBoundingClientRect();
    let position = ((clientX - rect.left) / rect.width) * 100;
    position = Math.max(5, Math.min(95, position));
    
    slider.style.left = `${position}%`;
    leftWrapper.style.width = `${position}%`;
    rightWrapper.style.width = `${100 - position}%`;
  }
  
  slider.addEventListener('mousedown', (e) => { isDragging = true; e.preventDefault(); });
  container.addEventListener('mousedown', (e) => { isDragging = true; updateSlider(e.clientX); });
  document.addEventListener('mousemove', (e) => { if (isDragging) updateSlider(e.clientX); });
  document.addEventListener('mouseup', () => { isDragging = false; });
  
  slider.addEventListener('touchstart', (e) => { isDragging = true; e.preventDefault(); });
  container.addEventListener('touchstart', (e) => { isDragging = true; updateSlider(e.touches[0].clientX); });
  document.addEventListener('touchmove', (e) => { if (isDragging) updateSlider(e.touches[0].clientX); });
  document.addEventListener('touchend', () => { isDragging = false; });
  
  // 初始化滑块到中间位置 - 在外部获取 rect
  const initialRect = container.getBoundingClientRect();
  updateSlider(initialRect.left + initialRect.width / 2);
}

// 重叠式图片对比滑块（误差图、真值预测用）
function initOverlayCompareSlider() {
  const container = $("compareSliderContainer");
  const slider = $("compareSlider");
  const topLayer = $("compareTop");
  
  if (!container || !slider || !topLayer) {
    console.error('对比滑块初始化失败: 缺少必要元素');
    return;
  }
  
  let isDragging = false;
  
  function updateSlider(clientX) {
    const rect = container.getBoundingClientRect();
    let position = ((clientX - rect.left) / rect.width) * 100;
    position = Math.max(2, Math.min(98, position)); // 限制范围2%-98%
    
    // 更新滑块位置
    slider.style.left = `${position}%`;
    
    // 使用 clip-path 裁剪顶层图片，显示左侧部分
    topLayer.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
  }
  
  // 鼠标事件
  slider.addEventListener('mousedown', (e) => {
    isDragging = true;
    e.preventDefault();
  });
  
  container.addEventListener('mousedown', (e) => {
    isDragging = true;
    updateSlider(e.clientX);
  });
  
  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    updateSlider(e.clientX);
  });
  
  document.addEventListener('mouseup', () => {
    isDragging = false;
  });
  
  // 触摸事件
  slider.addEventListener('touchstart', (e) => {
    isDragging = true;
    e.preventDefault();
  });
  
  container.addEventListener('touchstart', (e) => {
    isDragging = true;
    updateSlider(e.touches[0].clientX);
  });
  
  document.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    updateSlider(e.touches[0].clientX);
  });
  
  document.addEventListener('touchend', () => {
    isDragging = false;
  });
  
  // 初始化在50%位置
  const rect = container.getBoundingClientRect();
  updateSlider(rect.left + rect.width / 2);
}

// ==================== Lightbox ====================
function openLightbox(imageUrl, title = '') {
  const modal = $("lightboxModal");
  const img = $("lightboxImage");
  if (!modal || !img) return;
  
  img.src = imageUrl;
  img.alt = title;
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  document.body.style.overflow = 'hidden';
}

function closeLightbox(event) {
  if (event && event.target !== event.currentTarget) return;
  
  const modal = $("lightboxModal");
  if (!modal) return;
  
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  document.body.style.overflow = '';
}

// 全局暴露 lightbox 函数
window.openLightbox = openLightbox;
window.closeLightbox = closeLightbox;

// ==================== 日志面板切换 ====================
function toggleLogPanel() {
  const container = document.querySelector('.log-container');
  if (container) {
    container.classList.toggle('collapsed');
  }
}
window.toggleLogPanel = toggleLogPanel;

// ==================== AI Assistant (天才潮霸助手) ====================
const AI_RESPONSES = {
  report: (data) => {
    const cnn = data.cnn;
    const svm = data.svm;
    const dataset = data.dataset || '当前数据集';
    
    if (!cnn && !svm) {
      return `
        <p>📊 <strong>分析报告生成中...</strong></p>
        <p class="mt-2">目前尚未检测到训练完成的模型数据。</p>
        <p class="mt-2">建议您先完成以下步骤：</p>
        <ul class="mt-1">
          <li>1. 选择一个高光谱数据集（如 IndianPines, PaviaU, Salinas）</li>
          <li>2. 分别运行 CNN 和 SVM 的训练流程</li>
          <li>3. 等待训练完成后，我将为您生成详细的对比分析报告</li>
        </ul>
      `;
    }
    
    const cnnAcc = cnn?.metrics?.test_accuracy_percent || cnn?.metrics?.overall_accuracy_percent || 0;
    const svmAcc = svm?.metrics?.test_accuracy_percent || svm?.metrics?.overall_accuracy_percent || 0;
    const cnnKappa = cnn?.metrics?.kappa_percent || 0;
    const svmKappa = svm?.metrics?.kappa_percent || 0;
    const winner = cnnAcc > svmAcc ? 'CNN (HybridSN)' : 'SVM';
    const diff = Math.abs(cnnAcc - svmAcc).toFixed(2);
    
    return `
      <div class="report-card">
        <h5>📋 ${dataset} 分类分析报告</h5>
        <p><strong>综合评估结论：</strong>${winner} 在本次实验中表现更优，准确率差异为 ${diff}%。</p>
        <p class="mt-2"><strong>CNN (HybridSN) 性能摘要：</strong></p>
        <ul>
          <li>测试准确率：${cnnAcc.toFixed(2)}%</li>
          <li>Kappa 系数：${cnnKappa.toFixed(2)}%</li>
          <li>特点：3D-2D 混合卷积结构，能有效提取空-谱联合特征</li>
        </ul>
        <p class="mt-2"><strong>SVM 性能摘要：</strong></p>
        <ul>
          <li>测试准确率：${svmAcc.toFixed(2)}%</li>
          <li>Kappa 系数：${svmKappa.toFixed(2)}%</li>
          <li>特点：经典机器学习方法，计算效率高，适合小样本场景</li>
        </ul>
        <p class="mt-2"><strong>建议：</strong>${cnnAcc > svmAcc ? '对于该数据集，深度学习方法 (HybridSN) 的空谱联合特征提取能力更具优势。' : 'SVM 在该数据集上展现了较好的分类效果，可能是因为样本分布相对简单，或训练样本数量有限。'}</p>
      </div>
    `;
  },
  
  compare: (data) => {
    const cnn = data.cnn;
    const svm = data.svm;
    
    if (!cnn && !svm) {
      return `<p>⚖️ 需要先完成 CNN 和 SVM 的训练才能进行对比分析。请先运行训练任务！</p>`;
    }
    
    const metrics = [
      ['测试准确率', cnn?.metrics?.test_accuracy_percent, svm?.metrics?.test_accuracy_percent, '%'],
      ['Kappa 系数', cnn?.metrics?.kappa_percent, svm?.metrics?.kappa_percent, '%'],
      ['总体准确率', cnn?.metrics?.overall_accuracy_percent, svm?.metrics?.overall_accuracy_percent, '%'],
      ['平均准确率', cnn?.metrics?.average_accuracy_percent, svm?.metrics?.average_accuracy_percent, '%'],
    ];
    
    let tableHtml = '<table class="comparison-table mt-3"><thead><tr><th>指标</th><th>🧠 CNN</th><th>🎯 SVM</th><th>胜出</th></tr></thead><tbody>';
    
    metrics.forEach(([name, cnnVal, svmVal, unit]) => {
      if (cnnVal === undefined && svmVal === undefined) return;
      const cnnNum = Number(cnnVal) || 0;
      const svmNum = Number(svmVal) || 0;
      const winner = cnnNum > svmNum ? '🧠' : svmNum > cnnNum ? '🎯' : '🤝';
      tableHtml += `<tr>
        <td>${name}</td>
        <td>${cnnNum.toFixed(2)}${unit}</td>
        <td>${svmNum.toFixed(2)}${unit}</td>
        <td>${winner}</td>
      </tr>`;
    });
    
    tableHtml += '</tbody></table>';
    
    return `
      <p>⚖️ <strong>CNN vs SVM 详细对比</strong></p>
      ${tableHtml}
      <p class="mt-3 text-sm">🧠 = CNN胜出 &nbsp; 🎯 = SVM胜出 &nbsp; 🤝 = 平局</p>
    `;
  },
  
  optimize: (data) => {
    const cnn = data.cnn;
    const svm = data.svm;
    const cnnAcc = cnn?.metrics?.test_accuracy_percent || 0;
    const svmAcc = svm?.metrics?.test_accuracy_percent || 0;
    
    let suggestions = [];
    
    if (cnnAcc < 90) {
      suggestions.push('🧠 <strong>CNN 优化建议：</strong>');
      suggestions.push('• 增加训练 Epochs（当前可能欠拟合）');
      suggestions.push('• 调整 PCA 降维参数，保留更多光谱信息');
      suggestions.push('• 尝试更大的 Window Size 以捕获更多空间上下文');
      suggestions.push('• 适当降低学习率 (lr) 以提升收敛稳定性');
    }
    
    if (svmAcc < 90) {
      suggestions.push('🎯 <strong>SVM 优化建议：</strong>');
      suggestions.push('• 尝试不同的核函数（如 poly, linear）');
      suggestions.push('• 调整 C 参数平衡边界和分类误差');
      suggestions.push('• 使用网格搜索寻找最优 gamma 值');
      suggestions.push('• 考虑增加训练样本比例');
    }
    
    if (suggestions.length === 0) {
      return `
        <p>💡 <strong>优化建议</strong></p>
        <p class="mt-2">🎉 当前模型性能已经相当不错！</p>
        <p class="mt-2">如需进一步提升，可以尝试：</p>
        <ul>
          <li>使用集成学习方法融合 CNN 和 SVM 的优势</li>
          <li>尝试数据增强技术增加样本多样性</li>
          <li>探索更先进的深度学习架构（如 Transformer）</li>
        </ul>
      `;
    }
    
    return `<p>💡 <strong>模型优化建议</strong></p><ul class="mt-2">${suggestions.map(s => `<li>${s}</li>`).join('')}</ul>`;
  },
  
  explain: () => `
    <p>❓ <strong>关键指标解读</strong></p>
    <div class="mt-3 space-y-3">
      <div class="bg-surface-container p-3 rounded-lg">
        <p class="font-semibold">📊 Test Accuracy (测试准确率)</p>
        <p class="text-sm mt-1">模型在测试集上正确分类的样本比例，是最直观的性能指标。</p>
      </div>
      <div class="bg-surface-container p-3 rounded-lg">
        <p class="font-semibold">📈 Kappa 系数</p>
        <p class="text-sm mt-1">考虑了随机分类概率的一致性度量，消除了类别不平衡的影响。Kappa > 0.8 表示高度一致。</p>
      </div>
      <div class="bg-surface-container p-3 rounded-lg">
        <p class="font-semibold">🎯 Overall Accuracy (OA)</p>
        <p class="text-sm mt-1">所有类别正确分类的总体比率，与 Test Accuracy 类似但可能包含不同计算范围。</p>
      </div>
      <div class="bg-surface-container p-3 rounded-lg">
        <p class="font-semibold">📐 Average Accuracy (AA)</p>
        <p class="text-sm mt-1">各类别准确率的算术平均值，对类别不平衡更敏感，能反映小类别的分类效果。</p>
      </div>
    </div>
  `,
};

function getCurrentEvalData() {
  const ds = state.selectedEvalDataset;
  const cnnMap = Object.fromEntries((state.evalSummary?.cnn || []).map(i => [i.dataset, i]));
  const svmMap = Object.fromEntries((state.evalSummary?.svm || []).map(i => [i.dataset, i]));
  const comp = (state.evalSummary?.comparisons || []).find(c => c.dataset === ds);
  
  return {
    dataset: comp?.dataset_name || ds || '未知数据集',
    cnn: cnnMap[ds],
    svm: svmMap[ds],
    comparison: comp
  };
}

function addAiMessage(content, isBot = true) {
  const container = $('aiChatMessages');
  if (!container) return;
  
  const msgDiv = document.createElement('div');
  msgDiv.className = `ai-message ${isBot ? 'ai-message-bot' : 'ai-message-user'}`;
  msgDiv.innerHTML = `
    <div class="ai-message-avatar">${isBot ? '🧠' : '👤'}</div>
    <div class="ai-message-content">${content}</div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function showTypingIndicator() {
  const container = $('aiChatMessages');
  if (!container) return;
  
  const indicator = document.createElement('div');
  indicator.className = 'ai-message ai-message-bot';
  indicator.id = 'typingIndicator';
  indicator.innerHTML = `
    <div class="ai-message-avatar">🧠</div>
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  container.appendChild(indicator);
  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = $('typingIndicator');
  if (indicator) indicator.remove();
}

function handleAiAction(action) {
  const data = getCurrentEvalData();
  
  showTypingIndicator();
  
  setTimeout(() => {
    removeTypingIndicator();
    
    let response = '';
    switch (action) {
      case 'report':
        response = AI_RESPONSES.report(data);
        break;
      case 'compare':
        response = AI_RESPONSES.compare(data);
        break;
      case 'optimize':
        response = AI_RESPONSES.optimize(data);
        break;
      case 'explain':
        response = AI_RESPONSES.explain();
        break;
      default:
        response = '<p>抱歉，我不太明白您的问题。请尝试使用上方的快捷按钮，或者换一种方式提问。</p>';
    }
    
    addAiMessage(response, true);
    updateAnalysisSummary(data);
  }, 800 + Math.random() * 700);
}

function handleAiInput() {
  const input = $('aiInput');
  if (!input) return;
  
  const text = input.value.trim();
  if (!text) return;
  
  addAiMessage(text, false);
  input.value = '';
  
  // 简单关键词匹配
  const lower = text.toLowerCase();
  let action = '';
  
  if (lower.includes('报告') || lower.includes('分析') || lower.includes('总结')) {
    action = 'report';
  } else if (lower.includes('对比') || lower.includes('比较') || lower.includes('vs')) {
    action = 'compare';
  } else if (lower.includes('优化') || lower.includes('建议') || lower.includes('改进')) {
    action = 'optimize';
  } else if (lower.includes('解释') || lower.includes('什么') || lower.includes('指标')) {
    action = 'explain';
  } else {
    showTypingIndicator();
    setTimeout(() => {
      removeTypingIndicator();
      addAiMessage(`<p>您的问题是："${text}"</p><p class="mt-2">作为高光谱分类分析助手，我可以帮您：</p><ul><li>📋 生成分析报告</li><li>⚖️ 对比 CNN 与 SVM</li><li>💡 提供优化建议</li><li>❓ 解读评估指标</li></ul><p class="mt-2">请点击上方快捷按钮或输入相关关键词！</p>`, true);
    }, 600);
    return;
  }
  
  handleAiAction(action);
}

function updateAnalysisSummary(data) {
  const bestModel = $('bestModel');
  const bestAccuracy = $('bestAccuracy');
  const bestKappa = $('bestKappa');
  const currentDataset = $('currentDataset');
  
  const cnnAcc = data.cnn?.metrics?.test_accuracy_percent ?? data.cnn?.metrics?.overall_accuracy_percent ?? 0;
  const svmAcc = data.svm?.metrics?.test_accuracy_percent ?? data.svm?.metrics?.overall_accuracy_percent ?? 0;
  const cnnKappa = data.cnn?.metrics?.kappa_percent ?? 0;
  const svmKappa = data.svm?.metrics?.kappa_percent ?? 0;
  
  console.log('分析摘要更新:', { cnnAcc, svmAcc, cnnKappa, svmKappa });
  
  if (bestModel) {
    if (cnnAcc > svmAcc && cnnAcc > 0) {
      bestModel.textContent = 'CNN';
      bestModel.style.color = 'rgb(var(--primary))';
    } else if (svmAcc > cnnAcc && svmAcc > 0) {
      bestModel.textContent = 'SVM';
      bestModel.style.color = 'rgb(var(--secondary))';
    } else if (cnnAcc > 0 || svmAcc > 0) {
      bestModel.textContent = '平局';
      bestModel.style.color = '';
    } else {
      bestModel.textContent = '待分析';
      bestModel.style.color = 'rgb(var(--on-surface-variant))';
    }
  }
  
  const maxAcc = Math.max(cnnAcc, svmAcc);
  const maxKappa = Math.max(cnnKappa, svmKappa);
  if (bestAccuracy) bestAccuracy.textContent = maxAcc > 0 ? maxAcc.toFixed(2) + '%' : '0.00%';
  if (bestKappa) bestKappa.textContent = maxKappa > 0 ? maxKappa.toFixed(2) + '%' : '0.00%';
  if (currentDataset) currentDataset.textContent = data.dataset || '--';
  
  // 更新对比表格
  updateComparisonTable(data);
}

function updateComparisonTable(data) {
  const metrics = [
    ['tblCnnLoss', 'tblSvmLoss', 'test_loss_percent'],
    ['tblCnnAcc', 'tblSvmAcc', 'test_accuracy_percent'],
    ['tblCnnKappa', 'tblSvmKappa', 'kappa_percent'],
    ['tblCnnOA', 'tblSvmOA', 'overall_accuracy_percent'],
    ['tblCnnAA', 'tblSvmAA', 'average_accuracy_percent'],
  ];
  
  metrics.forEach(([cnnId, svmId, key]) => {
    const cnnEl = $(cnnId);
    const svmEl = $(svmId);
    const cnnVal = data.cnn?.metrics?.[key];
    const svmVal = data.svm?.metrics?.[key];
    
    if (cnnEl) cnnEl.textContent = cnnVal !== undefined ? Number(cnnVal).toFixed(2) : '--';
    if (svmEl) svmEl.textContent = svmVal !== undefined ? Number(svmVal).toFixed(2) : '--';
  });
}

// ==================== Metrics Panel Update ====================
function updateMetricsPanel() {
  const data = getCurrentEvalData();
  
  console.log('更新指标面板, 数据:', data);
  console.log('CNN metrics:', data.cnn?.metrics);
  console.log('SVM metrics:', data.svm?.metrics);
  
  // CNN 指标
  const cnnMetrics = data.cnn?.metrics || {};
  if ($('cnnTestLoss')) $('cnnTestLoss').textContent = cnnMetrics.test_loss_percent !== undefined ? Number(cnnMetrics.test_loss_percent).toFixed(2) : '--';
  if ($('cnnTestAcc')) $('cnnTestAcc').textContent = cnnMetrics.test_accuracy_percent !== undefined ? Number(cnnMetrics.test_accuracy_percent).toFixed(2) : '--';
  if ($('cnnKappa')) $('cnnKappa').textContent = cnnMetrics.kappa_percent !== undefined ? Number(cnnMetrics.kappa_percent).toFixed(2) : '--';
  if ($('cnnOA')) $('cnnOA').textContent = cnnMetrics.overall_accuracy_percent !== undefined ? Number(cnnMetrics.overall_accuracy_percent).toFixed(2) : '--';
  if ($('cnnAA')) $('cnnAA').textContent = cnnMetrics.average_accuracy_percent !== undefined ? Number(cnnMetrics.average_accuracy_percent).toFixed(2) : '--';
  
  // SVM 指标
  const svmMetrics = data.svm?.metrics || {};
  if ($('svmTestLoss')) $('svmTestLoss').textContent = svmMetrics.test_loss_percent !== undefined ? Number(svmMetrics.test_loss_percent).toFixed(2) : '--';
  if ($('svmTestAcc')) $('svmTestAcc').textContent = svmMetrics.test_accuracy_percent !== undefined ? Number(svmMetrics.test_accuracy_percent).toFixed(2) : '--';
  if ($('svmKappa')) $('svmKappa').textContent = svmMetrics.kappa_percent !== undefined ? Number(svmMetrics.kappa_percent).toFixed(2) : '--';
  if ($('svmOA')) $('svmOA').textContent = svmMetrics.overall_accuracy_percent !== undefined ? Number(svmMetrics.overall_accuracy_percent).toFixed(2) : '--';
  if ($('svmAA')) $('svmAA').textContent = svmMetrics.average_accuracy_percent !== undefined ? Number(svmMetrics.average_accuracy_percent).toFixed(2) : '--';
  
  // 更新分析摘要
  updateAnalysisSummary(data);
  
  // 更新雷达图
  setTimeout(() => initRadarChart(), 100);
}

// ==================== Loss Curve Chart ====================
let lossChart = null;
let radarChart = null;

// 存储加载的 loss history 数据
let cnnLossHistory = null;
let svmLossHistory = null;
let currentChartModel = 'CNN'; // 当前 Loss 曲线显示的模型类型
let currentAccChartModel = 'CNN'; // 当前 Accuracy 曲线显示的模型类型

async function initLossChart() {
  const canvas = $('lossChart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // 加载 CNN 和 SVM 的历史数据
  await loadAllLossHistory();
  
  // 根据当前选择的模型绘制曲线
  drawLossChart(ctx, canvas);
  
  // 设置初始按钮状态
  const cnnBtn = $('btnToggleCnnLoss');
  const svmBtn = $('btnToggleSvmLoss');
  if (cnnBtn) cnnBtn.classList.toggle('active', currentChartModel === 'CNN');
  if (svmBtn) svmBtn.classList.toggle('active', currentChartModel === 'SVM');
}

// 从后端加载所有 Loss 历史数据
async function loadAllLossHistory() {
  const selectedDataset = state.selectedEvalDataset;
  console.log('加载 Loss 历史数据，当前数据集:', selectedDataset);
  
  // 加载 CNN 历史
  await loadModelLossHistory('cnn', selectedDataset);
  // 加载 SVM 历史
  await loadModelLossHistory('svm', selectedDataset);
}

async function loadModelLossHistory(model, dataset) {
  try {
    const artifacts = model === 'cnn' ? state.artifacts : state.svmArtifacts;
    if (!artifacts || !artifacts.reports) {
      console.log(`${model} artifacts 不存在`);
      return;
    }
    
    // CNN: 查找 *_loss_history_*.json 文件
    // SVM: 查找 *_loss_history_*.json 文件
    const lossHistoryFiles = artifacts.reports.filter(r => {
      if (!r.name) return false;
      const name = r.name.toLowerCase();
      // 匹配 xxx_loss_history_xxx.json 格式
      return name.includes('_loss_history_') && name.endsWith('.json');
    });
    
    console.log(`${model} loss_history 文件:`, lossHistoryFiles.map(f => f.name));
    
    // 查找匹配当前数据集的所有文件
    let matchingFiles = [];
    if (dataset) {
      // 数据集名称映射
      const datasetNames = {
        'IP': ['indianpines', 'ip'],
        'SA': ['salinas', 'sa'],
        'PU': ['paviau', 'pu']
      };
      const names = datasetNames[dataset] || [dataset.toLowerCase()];
      matchingFiles = lossHistoryFiles.filter(f => 
        names.some(n => f.name.toLowerCase().startsWith(n + '_'))
      );
    }
    
    // 如果找到匹配的文件，优先使用训练时的参数匹配
    let targetFile = null;
    if (matchingFiles.length > 0) {
      // 优先使用训练时保存的参数，否则使用当前界面参数
      const params = state.lastTrainedParams || gatherParams();
      const targetPca = dataset === 'IP' ? params.pca_components_ip : params.pca_components_other;
      const targetWindow = params.window_size;
      const targetLr = params.lr;
      const targetEpochs = params.epochs;
      
      console.log(`尝试匹配参数: pca=${targetPca}, window=${targetWindow}, lr=${targetLr}, epochs=${targetEpochs}`);
      
      // 尝试找到参数完全匹配的文件
      const exactMatch = matchingFiles.find(f => {
        const name = f.name.toLowerCase();
        return name.includes(`pca=${targetPca}`) &&
               name.includes(`window=${targetWindow}`) &&
               name.includes(`lr=${targetLr}`) &&
               name.includes(`epochs=${targetEpochs}`);
      });
      
      if (exactMatch) {
        targetFile = exactMatch;
        console.log(`找到参数完全匹配的文件:`, targetFile.name);
      } else {
        // 没有完全匹配，选择最后一个（通常是最新的）
        targetFile = matchingFiles[matchingFiles.length - 1];
        console.log(`未找到完全匹配，使用最新的匹配文件:`, targetFile.name);
      }
    }
    
    // 如果没找到匹配的，使用最新的文件
    if (!targetFile && lossHistoryFiles.length > 0) {
      targetFile = lossHistoryFiles[lossHistoryFiles.length - 1];
      console.log(`未找到数据集匹配，使用最新文件:`, targetFile.name);
    }
    
    if (targetFile && targetFile.url) {
      const url = targetFile.url.startsWith('http') 
        ? targetFile.url 
        : `${API_BASE}${targetFile.url}`;
      console.log(`正在加载 ${model} Loss 历史:`, url);
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        if (model === 'cnn') {
          cnnLossHistory = data;
          console.log('已加载 CNN Loss 历史数据:', cnnLossHistory);
        } else {
          svmLossHistory = data;
          console.log('已加载 SVM Loss 历史数据:', svmLossHistory);
        }
      }
    } else {
      console.log(`未找到 ${model} 的 loss_history 文件，数据集: ${dataset}`);
      if (model === 'cnn') {
        cnnLossHistory = null;
      } else {
        svmLossHistory = null;
      }
    }
  } catch (error) {
    console.log(`加载 ${model} Loss 历史失败:`, error);
  }
}

// 根据当前模型绘制 Loss 曲线
function drawLossChart(ctx, canvas) {
  if (currentChartModel === 'CNN') {
    if (cnnLossHistory && cnnLossHistory.epochs && cnnLossHistory.epochs.length > 0) {
      drawCnnLossChart(ctx, canvas, cnnLossHistory);
    } else {
      drawSimulatedLossChart(ctx, canvas, 'CNN');
    }
  } else {
    if (svmLossHistory && svmLossHistory.train_ratios && svmLossHistory.train_ratios.length > 0) {
      drawSvmLearningCurve(ctx, canvas, svmLossHistory);
    } else {
      drawSimulatedLossChart(ctx, canvas, 'SVM');
    }
  }
}

// 切换 Loss 曲线模型
function switchLossChartModel(model) {
  currentChartModel = model;
  const canvas = $('lossChart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  drawLossChart(ctx, canvas);
  setupChartInteraction();
  
  // 更新按钮状态
  const cnnBtn = $('btnToggleCnnLoss');
  const svmBtn = $('btnToggleSvmLoss');
  if (cnnBtn) cnnBtn.classList.toggle('active', model === 'CNN');
  if (svmBtn) svmBtn.classList.toggle('active', model === 'SVM');
}

// 绘制 SVM 学习曲线 (训练样本比例 vs Loss)
function drawSvmLearningCurve(ctx, canvas, history) {
  const width = canvas.width = canvas.parentElement.offsetWidth;
  const height = canvas.height = canvas.parentElement.offsetHeight;
  const padding = { top: 30, right: 20, bottom: 35, left: 55 };
  
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  ctx.clearRect(0, 0, width, height);
  
  const trainColor = 'rgb(234, 88, 12)';    // orange-500 for train loss
  const testColor = 'rgb(139, 92, 246)';    // violet-500 for test loss
  const textColor = 'rgba(128, 128, 128, 0.9)';
  const gridColor = 'rgba(128, 128, 128, 0.15)';
  
  const ratios = history.train_ratios || [];
  const trainLoss = history.train_loss || [];
  const testLoss = history.test_loss || [];
  
  const allLoss = [...trainLoss, ...testLoss];
  const maxLoss = Math.max(...allLoss) * 1.1;
  const minLoss = Math.min(0, Math.min(...allLoss) * 0.9);
  
  // 绘制网格
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  
  // 绘制标题
  ctx.fillStyle = textColor;
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('SVM 学习曲线 (Learning Curve)', width / 2, 15);
  
  // Y 轴标签 (Loss %)
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const value = maxLoss - (maxLoss - minLoss) * (i / 5);
    ctx.fillText(value.toFixed(1) + '%', padding.left - 8, y + 3);
  }
  
  // X 轴标签 (训练样本比例)
  ctx.textAlign = 'center';
  ratios.forEach((r, i) => {
    const x = padding.left + (chartWidth / (ratios.length - 1)) * i;
    ctx.fillText((r * 100).toFixed(0) + '%', x, height - 8);
  });
  ctx.fillText('训练样本比例', width / 2, height - 2);
  
  // 绘制曲线函数
  const drawLine = (data, color, dashed = false) => {
    if (!data || data.length === 0) return;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    
    data.forEach((val, i) => {
      const x = padding.left + (chartWidth / (data.length - 1)) * i;
      const y = padding.top + chartHeight - ((val - minLoss) / (maxLoss - minLoss)) * chartHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 绘制数据点
    ctx.fillStyle = color;
    data.forEach((val, i) => {
      const x = padding.left + (chartWidth / (data.length - 1)) * i;
      const y = padding.top + chartHeight - ((val - minLoss) / (maxLoss - minLoss)) * chartHeight;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };
  
  // 绘制曲线
  drawLine(trainLoss, trainColor);
  drawLine(testLoss, testColor, true);
  
  // 图例
  const legendX = width - 130;
  ctx.fillStyle = trainColor;
  ctx.beginPath();
  ctx.arc(legendX + 6, 10, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = textColor;
  ctx.textAlign = 'left';
  ctx.fillText('训练 Loss', legendX + 16, 13);
  
  ctx.fillStyle = testColor;
  ctx.beginPath();
  ctx.arc(legendX + 6, 25, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.setLineDash([4, 2]);
  ctx.strokeStyle = testColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(legendX, 25);
  ctx.lineTo(legendX + 12, 25);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = textColor;
  ctx.fillText('测试 Loss', legendX + 16, 28);
  
  // 存储数据用于交互
  canvas.chartData = { 
    ratios, 
    trainLoss, 
    testLoss, 
    padding, 
    chartWidth, 
    chartHeight, 
    minLoss, 
    maxLoss,
    isSvm: true
  };
}

// 绘制 CNN Loss 曲线 (使用真实数据)
function drawCnnLossChart(ctx, canvas, history) {
  const width = canvas.width = canvas.parentElement.offsetWidth;
  const height = canvas.height = canvas.parentElement.offsetHeight;
  const padding = { top: 30, right: 20, bottom: 35, left: 55 };
  
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  ctx.clearRect(0, 0, width, height);
  
  const primaryColor = 'rgb(59, 130, 246)';  // blue-500 for train loss
  const validColor = 'rgb(34, 197, 94)';     // green-500 for valid loss
  const textColor = 'rgba(128, 128, 128, 0.9)';
  const gridColor = 'rgba(128, 128, 128, 0.15)';
  
  const epochs = history.epochs;
  const trainLoss = history.train_loss;
  const validLoss = history.valid_loss;
  
  const allLoss = [...trainLoss, ...validLoss];
  const maxLoss = Math.max(...allLoss) * 1.1;
  const minLoss = Math.min(...allLoss) * 0.9;
  
  // 绘制网格
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  
  // 绘制标题
  ctx.fillStyle = textColor;
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('CNN 训练 Loss 曲线', width / 2, 15);
  
  // Y 轴标签
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const value = maxLoss - (maxLoss - minLoss) * (i / 5);
    ctx.fillText(value.toFixed(3), padding.left - 8, y + 3);
  }
  
  // X 轴标签
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.ceil(epochs.length / 10));
  epochs.forEach((e, i) => {
    if (i % xStep === 0 || i === epochs.length - 1) {
      const x = padding.left + (chartWidth / (epochs.length - 1)) * i;
      ctx.fillText(e.toString(), x, height - 8);
    }
  });
  ctx.fillText('Epoch', width / 2, height - 2);
  
  // 绘制曲线函数
  const drawLine = (data, color, dashed = false) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (dashed) {
      ctx.setLineDash([5, 3]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    
    data.forEach((val, i) => {
      const x = padding.left + (chartWidth / (data.length - 1)) * i;
      const y = padding.top + chartHeight - ((val - minLoss) / (maxLoss - minLoss)) * chartHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    ctx.setLineDash([]);
  };
  
  // 绘制曲线
  drawLine(trainLoss, primaryColor);
  drawLine(validLoss, validColor, true);
  
  // 图例
  const legendX = width - 120;
  ctx.fillStyle = primaryColor;
  ctx.fillRect(legendX, 8, 12, 3);
  ctx.fillStyle = textColor;
  ctx.textAlign = 'left';
  ctx.fillText('训练 Loss', legendX + 16, 12);
  
  ctx.fillStyle = validColor;
  ctx.fillRect(legendX, 22, 12, 3);
  ctx.setLineDash([3, 2]);
  ctx.strokeStyle = validColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(legendX, 22);
  ctx.lineTo(legendX + 12, 22);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = textColor;
  ctx.fillText('验证 Loss', legendX + 16, 26);
  
  // 存储数据用于交互
  canvas.chartData = { 
    epochs, 
    trainLoss, 
    validLoss, 
    padding, 
    chartWidth, 
    chartHeight, 
    minLoss, 
    maxLoss,
    hasValidLoss: true
  };
}

// 没有真实数据时显示提示
function drawSimulatedLossChart(ctx, canvas, model = 'CNN') {
  // 不再显示模拟数据，而是显示清晰的提示
  drawNoDataPlaceholder(ctx, canvas, model);
}

// 没有数据时显示美观的占位符
function drawNoDataPlaceholder(ctx, canvas, model = 'CNN') {
  const width = canvas.width = canvas.parentElement.offsetWidth;
  const height = canvas.height = canvas.parentElement.offsetHeight;
  
  ctx.clearRect(0, 0, width, height);
  
  // 绘制背景网格（淡化）
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.08)';
  ctx.lineWidth = 1;
  const gridSize = 30;
  for (let x = gridSize; x < width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = gridSize; y < height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  // 绘制示意曲线（虚线，表示将要显示的内容）
  ctx.strokeStyle = model === 'CNN' ? 'rgba(59, 130, 246, 0.15)' : 'rgba(234, 88, 12, 0.15)';
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  ctx.moveTo(50, height - 50);
  ctx.bezierCurveTo(width * 0.3, height * 0.6, width * 0.6, height * 0.3, width - 50, height * 0.25);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // 图标
  const iconSize = 40;
  const iconX = width / 2;
  const iconY = height / 2 - 30;
  
  ctx.fillStyle = model === 'CNN' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(234, 88, 12, 0.2)';
  ctx.beginPath();
  ctx.arc(iconX, iconY, iconSize, 0, Math.PI * 2);
  ctx.fill();
  
  // 图标内的图表符号
  ctx.strokeStyle = model === 'CNN' ? 'rgba(59, 130, 246, 0.5)' : 'rgba(234, 88, 12, 0.5)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(iconX - 15, iconY + 10);
  ctx.lineTo(iconX - 5, iconY - 5);
  ctx.lineTo(iconX + 5, iconY + 5);
  ctx.lineTo(iconX + 15, iconY - 10);
  ctx.stroke();
  
  // 文字
  ctx.fillStyle = 'rgba(128, 128, 128, 0.7)';
  ctx.font = 'bold 14px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  if (model === 'CNN') {
    ctx.fillText('暂无 CNN 训练曲线数据', width / 2, height / 2 + 20);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
    ctx.fillText('训练 CNN 模型后将自动显示 Loss 曲线', width / 2, height / 2 + 42);
    ctx.fillText('数据文件格式: *_loss_history_*.json', width / 2, height / 2 + 60);
  } else {
    ctx.fillText('暂无 SVM 学习曲线数据', width / 2, height / 2 + 20);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
    ctx.fillText('训练 SVM 模型后将自动显示学习曲线', width / 2, height / 2 + 42);
    ctx.fillText('X轴: 训练样本比例 · Y轴: Loss', width / 2, height / 2 + 60);
  }
  
  canvas.chartData = null;
}

// ==================== Accuracy 曲线 ====================
let cnnAccuracyHistory = null;

async function initAccuracyChart() {
  const canvas = $('accuracyChart');
  if (!canvas) {
    console.log('Accuracy chart canvas 不存在');
    return;
  }
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // 确保已加载数据
  if (!cnnLossHistory) {
    await loadModelLossHistory('cnn', state.selectedEvalDataset);
  }
  if (!svmLossHistory) {
    await loadModelLossHistory('svm', state.selectedEvalDataset);
  }
  
  // 根据当前选择的模型绘制曲线
  drawAccuracyChartByModel(ctx, canvas);
  
  // 更新按钮状态
  const cnnBtn = $('btnToggleCnnAcc');
  const svmBtn = $('btnToggleSvmAcc');
  if (cnnBtn) cnnBtn.classList.toggle('active', currentAccChartModel === 'CNN');
  if (svmBtn) svmBtn.classList.toggle('active', currentAccChartModel === 'SVM');
}

function drawAccuracyChartByModel(ctx, canvas) {
  if (currentAccChartModel === 'CNN') {
    if (cnnLossHistory && cnnLossHistory.train_acc && cnnLossHistory.train_acc.length > 0) {
      drawAccuracyChart(ctx, canvas, cnnLossHistory, 'CNN');
    } else {
      drawAccuracyPlaceholder(ctx, canvas, 'CNN');
    }
  } else {
    // SVM 使用 test_accuracy 作为准确率
    if (svmLossHistory && svmLossHistory.test_accuracy && svmLossHistory.test_accuracy.length > 0) {
      drawSvmAccuracyChart(ctx, canvas, svmLossHistory);
    } else {
      drawAccuracyPlaceholder(ctx, canvas, 'SVM');
    }
  }
}

function switchAccuracyChartModel(model) {
  currentAccChartModel = model;
  const canvas = $('accuracyChart');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  drawAccuracyChartByModel(ctx, canvas);
  setupAccuracyChartInteraction();
  
  // 更新按钮状态
  const cnnBtn = $('btnToggleCnnAcc');
  const svmBtn = $('btnToggleSvmAcc');
  if (cnnBtn) cnnBtn.classList.toggle('active', model === 'CNN');
  if (svmBtn) svmBtn.classList.toggle('active', model === 'SVM');
}

function drawAccuracyChart(ctx, canvas, history, modelType = 'CNN') {
  let width = canvas.parentElement?.offsetWidth || 0;
  let height = canvas.parentElement?.offsetHeight || 0;
  if (width < 50) width = 600;
  if (height < 50) height = 250;
  
  canvas.width = width;
  canvas.height = height;
  // 增加右边距以匹配 Loss 曲线的布局
  const padding = { top: 30, right: 140, bottom: 35, left: 55 };
  
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  ctx.clearRect(0, 0, width, height);
  
  const trainColor = 'rgb(16, 185, 129)';  // emerald-500
  const validColor = 'rgb(245, 158, 11)';  // amber-500
  const textColor = 'rgba(128, 128, 128, 0.9)';
  const gridColor = 'rgba(128, 128, 128, 0.15)';
  
  const epochs = history.epochs || [];
  const trainAcc = history.train_acc || [];
  const validAcc = history.valid_acc || [];
  
  // 处理只有 1 个数据点或没有数据的情况
  if (epochs.length === 0) {
    drawAccuracyPlaceholder(ctx, canvas);
    return;
  }
  
  const allAcc = [...trainAcc, ...validAcc].filter(v => v !== undefined && !isNaN(v));
  if (allAcc.length === 0) {
    drawAccuracyPlaceholder(ctx, canvas);
    return;
  }
  
  const maxAcc = Math.min(1, Math.max(...allAcc) * 1.05);
  const minAcc = Math.max(0, Math.min(...allAcc) * 0.95);
  
  // 绘制网格
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  
  // 绘制标题
  ctx.fillStyle = textColor;
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`${modelType} Accuracy 曲线`, width / 2, 15);
  
  // Y 轴标签 (%)
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const value = maxAcc - (maxAcc - minAcc) * (i / 5);
    ctx.fillText((value * 100).toFixed(1) + '%', padding.left - 8, y + 3);
  }
  
  // X 轴标签 - 使用与 Loss 曲线相同的间距策略
  ctx.textAlign = 'center';
  if (epochs.length === 1) {
    // 只有一个 epoch 时，居中显示
    ctx.fillText(epochs[0].toString(), padding.left + chartWidth / 2, height - 8);
  } else {
    // 计算合适的标签间隔，确保标签不会太密集
    // 目标是显示约 8-12 个标签
    const targetLabels = 10;
    const xStep = Math.max(1, Math.ceil(epochs.length / targetLabels));
    const xScale = chartWidth / (epochs.length - 1);
    
    // 绘制标签，确保第一个和最后一个都显示
    epochs.forEach((e, i) => {
      // 显示第一个、最后一个、以及每隔 xStep 个
      if (i === 0 || i === epochs.length - 1 || i % xStep === 0) {
        const x = padding.left + xScale * i;
        ctx.fillText(e.toString(), x, height - 8);
      }
    });
  }
  ctx.fillText('Epoch', width / 2, height - 2);
  
  // 绘制曲线函数
  const drawLine = (data, color, dashed = false) => {
    if (!data || data.length === 0) return;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.setLineDash(dashed ? [6, 4] : []);
    
    // 只有 1 个数据点时，绘制一个圆点
    if (data.length === 1) {
      const x = padding.left + chartWidth / 2;
      const y = padding.top + chartHeight - ((data[0] - minAcc) / (maxAcc - minAcc || 1)) * chartHeight;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.setLineDash([]);
      return;
    }
    
    ctx.beginPath();
    
    const xScale = chartWidth / (data.length - 1);
    data.forEach((val, i) => {
      const x = padding.left + xScale * i;
      const y = padding.top + chartHeight - ((val - minAcc) / (maxAcc - minAcc || 1)) * chartHeight;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    ctx.setLineDash([]);
  };
  
  // 绘制曲线
  drawLine(trainAcc, trainColor);
  drawLine(validAcc, validColor, true);
  
  // 图例
  const legendX = width - 130;
  ctx.fillStyle = trainColor;
  ctx.fillRect(legendX, 8, 12, 3);
  ctx.fillStyle = textColor;
  ctx.textAlign = 'left';
  ctx.fillText('训练 Acc', legendX + 16, 12);
  
  ctx.setLineDash([4, 2]);
  ctx.strokeStyle = validColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(legendX, 23);
  ctx.lineTo(legendX + 12, 23);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = textColor;
  ctx.fillText('验证 Acc', legendX + 16, 26);
  
  // 存储数据用于交互
  canvas.chartData = {
    epochs,
    trainAcc,
    validAcc,
    padding,
    chartWidth,
    chartHeight,
    minAcc,
    maxAcc
  };
}

function drawAccuracyPlaceholder(ctx, canvas, modelType = 'CNN') {
  let width = canvas.parentElement?.offsetWidth || 0;
  let height = canvas.parentElement?.offsetHeight || 0;
  if (width < 50) width = 600;
  if (height < 50) height = 250;
  
  canvas.width = width;
  canvas.height = height;
  
  ctx.clearRect(0, 0, width, height);
  
  // 绘制背景网格（淡化）
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.08)';
  ctx.lineWidth = 1;
  const gridSize = 30;
  for (let x = gridSize; x < width; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = gridSize; y < height; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  // 绘制示意曲线（上升趋势表示准确率提升）
  const curveColor = modelType === 'CNN' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(234, 88, 12, 0.15)';
  ctx.strokeStyle = curveColor;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  ctx.moveTo(50, height - 50);
  ctx.bezierCurveTo(width * 0.3, height * 0.5, width * 0.6, height * 0.35, width - 50, height * 0.2);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // 图标
  const iconSize = 40;
  const iconX = width / 2;
  const iconY = height / 2 - 30;
  
  const iconColor = modelType === 'CNN' ? 'rgba(16, 185, 129' : 'rgba(234, 88, 12';
  ctx.fillStyle = `${iconColor}, 0.2)`;
  ctx.beginPath();
  ctx.arc(iconX, iconY, iconSize, 0, Math.PI * 2);
  ctx.fill();
  
  // 图标内的上升趋势符号
  ctx.strokeStyle = `${iconColor}, 0.5)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(iconX - 15, iconY + 10);
  ctx.lineTo(iconX + 10, iconY - 10);
  ctx.lineTo(iconX + 10, iconY - 2);
  ctx.moveTo(iconX + 10, iconY - 10);
  ctx.lineTo(iconX + 2, iconY - 10);
  ctx.stroke();
  
  // 文字
  ctx.fillStyle = 'rgba(128, 128, 128, 0.7)';
  ctx.font = 'bold 14px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`暂无 ${modelType} 准确率曲线数据`, width / 2, height / 2 + 20);
  ctx.font = '12px sans-serif';
  ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
  ctx.fillText(`训练 ${modelType} 模型后将自动显示 Accuracy 曲线`, width / 2, height / 2 + 42);
  if (modelType === 'CNN') {
    ctx.fillText('绿色: 训练准确率 · 橙色: 验证准确率', width / 2, height / 2 + 60);
  } else {
    ctx.fillText('X轴: 训练样本比例 · Y轴: 准确率', width / 2, height / 2 + 60);
  }
  
  canvas.chartData = null;
}

// 训练完成后刷新所有曲线
async function refreshChartsAfterTraining() {
  console.log('========== refreshChartsAfterTraining 开始 ==========');
  
  // 重置历史数据，强制重新加载
  cnnLossHistory = null;
  svmLossHistory = null;
  console.log('已重置 cnnLossHistory 和 svmLossHistory');
  
  // 刷新 artifacts 列表（训练会生成新的 loss_history 文件）
  await loadArtifacts();
  console.log('loadArtifacts 完成，CNN reports 数量:', state.artifacts?.reports?.length);
  
  // 确保评估数据是最新的
  await loadEvaluations();
  console.log('loadEvaluations 完成');
  
  // 优先使用刚训练的数据集，否则使用当前选择的或第一个
  const comps = state.evalSummary?.comparisons || [];
  if (state.lastTrainedDataset) {
    // 切换到刚训练的数据集
    state.selectedEvalDataset = state.lastTrainedDataset;
    console.log('切换到刚训练的数据集:', state.selectedEvalDataset);
  } else if (comps.length && !state.selectedEvalDataset) {
    state.selectedEvalDataset = comps[0].dataset;
    console.log('自动选择数据集:', state.selectedEvalDataset);
  }
  console.log('当前选择的数据集:', state.selectedEvalDataset);
  
  // 重新加载并绘制
  console.log('开始 initLossChart...');
  await initLossChart();
  console.log('initLossChart 完成，cnnLossHistory:', cnnLossHistory ? '有数据' : 'null');
  console.log('cnnLossHistory.train_acc:', cnnLossHistory?.train_acc?.length || 0, '条');
  
  console.log('开始 initAccuracyChart...');
  await initAccuracyChart();
  console.log('initAccuracyChart 完成');
  
  setupChartInteraction();
  initRadarChart();
  
  // 更新指标面板
  updateMetricsPanel();
  
  // 更新数据集选择按钮的高亮状态
  updateDatasetButtonHighlight();
  
  console.log('========== refreshChartsAfterTraining 结束 ==========');
}

// ==================== 性能雷达图 ====================
function initRadarChart() {
  const canvas = $('radarChart');
  if (!canvas) {
    console.log('雷达图 canvas 不存在');
    return;
  }
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const data = getCurrentEvalData();
  console.log('雷达图数据:', {
    dataset: data.dataset,
    cnn: data.cnn,
    svm: data.svm,
    selectedEvalDataset: state.selectedEvalDataset
  });
  
  const cnnMetrics = data.cnn?.metrics || {};
  const svmMetrics = data.svm?.metrics || {};
  
  console.log('雷达图指标:', { cnnMetrics, svmMetrics });
  
  // 如果没有任何数据，显示占位符
  if (Object.keys(cnnMetrics).length === 0 && Object.keys(svmMetrics).length === 0) {
    drawRadarPlaceholder(ctx, canvas);
    return;
  }
  
  drawRadarChart(ctx, canvas, cnnMetrics, svmMetrics);
}

function drawRadarPlaceholder(ctx, canvas) {
  // 使用父元素尺寸，如果为0则使用默认值
  let width = canvas.parentElement?.offsetWidth || 0;
  let height = canvas.parentElement?.offsetHeight || 0;
  
  // 如果父元素尺寸为0（可能是隐藏状态），使用默认尺寸
  if (width < 50) width = 300;
  if (height < 50) height = 200;
  
  canvas.width = width;
  canvas.height = height;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 40;
  
  ctx.clearRect(0, 0, width, height);
  
  // 绘制淡化的雷达网格背景
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.1)';
  ctx.lineWidth = 1;
  
  const dimensions = 5;
  const angleStep = (2 * Math.PI) / dimensions;
  const startAngle = -Math.PI / 2;
  
  // 绘制同心五边形网格
  for (let level = 1; level <= 4; level++) {
    const levelRadius = (radius / 4) * level;
    ctx.beginPath();
    for (let i = 0; i <= dimensions; i++) {
      const angle = startAngle + angleStep * i;
      const x = centerX + Math.cos(angle) * levelRadius;
      const y = centerY + Math.sin(angle) * levelRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  
  // 绘制轴线
  for (let i = 0; i < dimensions; i++) {
    const angle = startAngle + angleStep * i;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
    ctx.stroke();
  }
  
  // 中心图标
  ctx.fillStyle = 'rgba(168, 85, 247, 0.15)';
  ctx.beginPath();
  ctx.arc(centerX, centerY, 30, 0, Math.PI * 2);
  ctx.fill();
  
  // 五角星图标
  ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < 5; i++) {
    const angle = startAngle + (i * 2 * Math.PI) / 5;
    const x = centerX + Math.cos(angle) * 15;
    const y = centerY + Math.sin(angle) * 15;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  
  // 文字提示
  ctx.fillStyle = 'rgba(128, 128, 128, 0.7)';
  ctx.font = 'bold 13px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('暂无性能对比数据', centerX, centerY + 55);
  ctx.font = '11px sans-serif';
  ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
  ctx.fillText('训练模型后将显示 CNN vs SVM 性能雷达图', centerX, centerY + 75);
}

function drawRadarChart(ctx, canvas, cnnMetrics, svmMetrics) {
  // 使用父元素尺寸，如果为0则使用默认值
  let width = canvas.parentElement?.offsetWidth || 0;
  let height = canvas.parentElement?.offsetHeight || 0;
  
  // 如果父元素尺寸为0（可能是隐藏状态），使用默认尺寸
  if (width < 50) width = 300;
  if (height < 50) height = 200;
  
  canvas.width = width;
  canvas.height = height;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 35;
  
  ctx.clearRect(0, 0, width, height);
  
  // 指标维度 - 调整 Loss 的处理方式
  // 对于 Loss，我们显示 (100 - loss)，这样越高越好
  const dimensions = [
    { label: 'Accuracy', key: 'test_accuracy_percent', max: 100 },
    { label: 'Kappa', key: 'kappa_percent', max: 100 },
    { label: 'OA', key: 'overall_accuracy_percent', max: 100 },
    { label: 'AA', key: 'average_accuracy_percent', max: 100 },
    { label: '低Loss', key: 'test_loss_percent', max: 100, invert: true }
  ];
  
  console.log('绘制雷达图, CNN:', cnnMetrics, 'SVM:', svmMetrics);
  
  const angleStep = (2 * Math.PI) / dimensions.length;
  const startAngle = -Math.PI / 2;
  
  // 绘制背景网格
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.2)';
  ctx.lineWidth = 1;
  
  for (let level = 1; level <= 5; level++) {
    const levelRadius = (radius / 5) * level;
    ctx.beginPath();
    for (let i = 0; i <= dimensions.length; i++) {
      const angle = startAngle + angleStep * i;
      const x = centerX + Math.cos(angle) * levelRadius;
      const y = centerY + Math.sin(angle) * levelRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  
  // 绘制轴线
  for (let i = 0; i < dimensions.length; i++) {
    const angle = startAngle + angleStep * i;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
    ctx.stroke();
  }
  
  // 绘制标签
  ctx.fillStyle = 'rgba(128, 128, 128, 0.8)';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  for (let i = 0; i < dimensions.length; i++) {
    const angle = startAngle + angleStep * i;
    const labelRadius = radius + 15;
    const x = centerX + Math.cos(angle) * labelRadius;
    const y = centerY + Math.sin(angle) * labelRadius;
    ctx.fillText(dimensions[i].label, x, y);
  }
  
  // 绘制数据多边形
  const drawDataPolygon = (metrics, color, alpha = 0.3) => {
    if (!metrics || Object.keys(metrics).length === 0) return;
    
    ctx.beginPath();
    let hasData = false;
    for (let i = 0; i < dimensions.length; i++) {
      const dim = dimensions[i];
      let value = metrics[dim.key];
      if (value === undefined || value === null) value = 0;
      
      // 对于 Loss，使用 100 - min(loss, 100) 来确保值在合理范围内
      if (dim.invert) {
        value = Math.max(0, 100 - Math.min(value, 100));
      }
      
      // 确保值在 0-max 范围内
      value = Math.max(0, Math.min(value, dim.max));
      const normalizedValue = value / dim.max;
      
      if (normalizedValue > 0) hasData = true;
      
      const angle = startAngle + angleStep * i;
      const pointRadius = normalizedValue * radius;
      const x = centerX + Math.cos(angle) * pointRadius;
      const y = centerY + Math.sin(angle) * pointRadius;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    
    if (!hasData) return; // 如果没有数据，不绘制
    
    ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  
  // CNN 数据 (蓝色)
  if (Object.keys(cnnMetrics).length > 0) {
    drawDataPolygon(cnnMetrics, 'rgb(59, 130, 246)', 0.2);
  }
  
  // SVM 数据 (紫色)
  if (Object.keys(svmMetrics).length > 0) {
    drawDataPolygon(svmMetrics, 'rgb(168, 85, 247)', 0.2);
  }
  
  // 图例
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  if (Object.keys(cnnMetrics).length > 0) {
    ctx.fillStyle = 'rgb(59, 130, 246)';
    ctx.fillRect(5, 5, 10, 10);
    ctx.fillStyle = 'rgba(128, 128, 128, 0.8)';
    ctx.fillText('CNN', 18, 12);
  }
  if (Object.keys(svmMetrics).length > 0) {
    ctx.fillStyle = 'rgb(168, 85, 247)';
    ctx.fillRect(5, 20, 10, 10);
    ctx.fillStyle = 'rgba(128, 128, 128, 0.8)';
    ctx.fillText('SVM', 18, 27);
  }
}

function setupChartInteraction() {
  setupLossChartInteraction();
  setupAccuracyChartInteraction();
}

function setupLossChartInteraction() {
  const canvas = $('lossChart');
  const tooltip = $('chartTooltip');
  if (!canvas || !tooltip) return;
  
  // 创建数据点指示器
  let dataPoint = document.getElementById('chartDataPoint');
  if (!dataPoint) {
    dataPoint = document.createElement('div');
    dataPoint.id = 'chartDataPoint';
    dataPoint.className = 'chart-data-point hidden';
    canvas.parentElement.style.position = 'relative';
    canvas.parentElement.appendChild(dataPoint);
  }
  
  canvas.style.cursor = 'crosshair';
  
  canvas.onmousemove = (e) => {
    const data = canvas.chartData;
    if (!data) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // 检查是否在图表区域内
    if (x < data.padding.left || x > canvas.width - data.padding.right ||
        y < data.padding.top || y > canvas.height - data.padding.bottom) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    // 判断是 CNN (epochs) 还是 SVM (ratios) 曲线
    const isSvm = data.isSvm || data.ratios;
    const xData = isSvm ? data.ratios : data.epochs;
    const dataLen = xData ? xData.length : 0;
    
    if (dataLen === 0) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    // 计算对应的索引
    const dataIndex = Math.round((x - data.padding.left) / data.chartWidth * (dataLen - 1));
    if (dataIndex < 0 || dataIndex >= dataLen) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    const trainVal = data.trainLoss[dataIndex];
    const validVal = isSvm ? data.testLoss[dataIndex] : (data.validLoss ? data.validLoss[dataIndex] : null);
    
    // 计算曲线点位置
    const pointX = data.padding.left + (data.chartWidth / (dataLen - 1)) * dataIndex;
    const trainY = data.padding.top + data.chartHeight - ((trainVal - data.minLoss) / (data.maxLoss - data.minLoss)) * data.chartHeight;
    const validY = validVal !== null 
      ? data.padding.top + data.chartHeight - ((validVal - data.minLoss) / (data.maxLoss - data.minLoss)) * data.chartHeight
      : trainY;
    
    // 判断鼠标更接近哪条曲线
    const nearTrain = validVal === null || Math.abs(y - trainY) < Math.abs(y - validY);
    const nearY = nearTrain ? trainY : validY;
    
    // 根据模型类型选择颜色
    const trainColor = isSvm ? 'rgb(234, 88, 12)' : 'rgb(59, 130, 246)';
    const validColor = isSvm ? 'rgb(139, 92, 246)' : 'rgb(34, 197, 94)';
    const nearColor = nearTrain ? trainColor : validColor;
    
    // 显示数据点指示器
    dataPoint.style.left = `${pointX}px`;
    dataPoint.style.top = `${nearY}px`;
    dataPoint.style.background = nearColor;
    dataPoint.classList.remove('hidden');
    
    // 构建 tooltip 内容
    const xLabel = isSvm ? `训练比例 ${(xData[dataIndex] * 100).toFixed(0)}%` : `Epoch ${xData[dataIndex]}`;
    const trainLabel = isSvm ? '训练 Loss' : '训练 Loss';
    const validLabel = isSvm ? '测试 Loss' : '验证 Loss';
    
    let tooltipContent = `
      <div class="font-semibold mb-1" style="color: rgb(var(--on-surface))">${xLabel}</div>
      <div class="flex items-center gap-2 ${nearTrain ? 'font-bold' : ''}">
        <span class="w-3 h-3 rounded-full" style="background: ${trainColor}"></span>
        <span style="color: rgb(var(--on-surface))">${trainLabel}: ${trainVal.toFixed(isSvm ? 2 : 4)}${isSvm ? '%' : ''}</span>
      </div>
    `;
    
    if (validVal !== null) {
      tooltipContent += `
        <div class="flex items-center gap-2 ${!nearTrain ? 'font-bold' : ''}">
          <span class="w-3 h-3 rounded-full" style="background: ${validColor}"></span>
          <span style="color: rgb(var(--on-surface))">${validLabel}: ${validVal.toFixed(isSvm ? 2 : 4)}${isSvm ? '%' : ''}</span>
        </div>
      `;
    }
    
    tooltip.innerHTML = tooltipContent;
    tooltip.style.left = `${pointX}px`;
    tooltip.style.top = `${nearY}px`;
    
    // 动态调整 tooltip 位置，避免被遮挡或超出边界
    const containerWidth = canvas.width;
    const tooltipWidth = 140; // 估算 tooltip 宽度
    
    // 默认水平居中 (-50%)，如果靠近右边缘，则向左偏移更多
    let translateX = '-50%';
    if (pointX + tooltipWidth / 2 > containerWidth - 20) {
      translateX = '-100%';
      tooltip.style.left = `${pointX - 10}px`;
    } else if (pointX - tooltipWidth / 2 < 20) {
      translateX = '0%';
      tooltip.style.left = `${pointX + 10}px`;
    }
    
    // 如果靠近顶部，tooltip 显示在下方
    let translateY = '-110%';
    if (nearY < 60) {
      translateY = '20%';
    }
    
    tooltip.style.transform = `translate(${translateX}, ${translateY})`;
    tooltip.classList.remove('hidden');
    tooltip.classList.add('chart-tooltip');
  };
  
  canvas.onmouseleave = () => {
    tooltip.classList.add('hidden');
    dataPoint.classList.add('hidden');
  };
}

function setupAccuracyChartInteraction() {
  const canvas = $('accuracyChart');
  const tooltip = $('accChartTooltip');
  if (!canvas || !tooltip) return;
  
  // 创建数据点指示器
  let dataPoint = document.getElementById('accChartDataPoint');
  if (!dataPoint) {
    dataPoint = document.createElement('div');
    dataPoint.id = 'accChartDataPoint';
    dataPoint.className = 'chart-data-point hidden';
    canvas.parentElement.style.position = 'relative';
    canvas.parentElement.appendChild(dataPoint);
  }
  
  canvas.style.cursor = 'crosshair';
  
  canvas.onmousemove = (e) => {
    const data = canvas.chartData;
    if (!data || !data.epochs || data.epochs.length === 0) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (x < data.padding.left || x > canvas.width - data.padding.right ||
        y < data.padding.top || y > canvas.height - data.padding.bottom) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    const dataLen = data.epochs.length;
    const dataIndex = Math.round((x - data.padding.left) / data.chartWidth * (dataLen - 1));
    if (dataIndex < 0 || dataIndex >= dataLen) {
      tooltip.classList.add('hidden');
      dataPoint.classList.add('hidden');
      return;
    }
    
    const trainVal = data.trainAcc[dataIndex];
    const validVal = data.validAcc ? data.validAcc[dataIndex] : null;
    
    const pointX = data.padding.left + (data.chartWidth / (dataLen - 1)) * dataIndex;
    const trainY = data.padding.top + data.chartHeight - ((trainVal - data.minAcc) / (data.maxAcc - data.minAcc)) * data.chartHeight;
    const validY = validVal !== null 
      ? data.padding.top + data.chartHeight - ((validVal - data.minAcc) / (data.maxAcc - data.minAcc)) * data.chartHeight
      : trainY;
    
    const nearTrain = validVal === null || Math.abs(y - trainY) < Math.abs(y - validY);
    const nearY = nearTrain ? trainY : validY;
    
    const trainColor = 'rgb(16, 185, 129)';
    const validColor = 'rgb(245, 158, 11)';
    const nearColor = nearTrain ? trainColor : validColor;
    
    dataPoint.style.left = `${pointX}px`;
    dataPoint.style.top = `${nearY}px`;
    dataPoint.style.background = nearColor;
    dataPoint.classList.remove('hidden');
    
    let tooltipContent = `
      <div class="font-semibold mb-1" style="color: rgb(var(--on-surface))">Epoch ${data.epochs[dataIndex]}</div>
      <div class="flex items-center gap-2 ${nearTrain ? 'font-bold' : ''}">
        <span class="w-3 h-3 rounded-full" style="background: ${trainColor}"></span>
        <span style="color: rgb(var(--on-surface))">训练: ${(trainVal * 100).toFixed(2)}%</span>
      </div>
    `;
    
    if (validVal !== null) {
      tooltipContent += `
        <div class="flex items-center gap-2 ${!nearTrain ? 'font-bold' : ''}">
          <span class="w-3 h-3 rounded-full" style="background: ${validColor}"></span>
          <span style="color: rgb(var(--on-surface))">验证: ${(validVal * 100).toFixed(2)}%</span>
        </div>
      `;
    }
    
    tooltip.innerHTML = tooltipContent;
    tooltip.style.left = `${pointX}px`;
    tooltip.style.top = `${nearY}px`;
    
    // 动态调整 tooltip 位置，避免被遮挡或超出边界
    // 计算 tooltip 相对于容器的位置
    const containerWidth = canvas.width;
    const tooltipWidth = 140; // 估算 tooltip 宽度
    
    // 默认水平居中 (-50%)，如果靠近右边缘，则向左偏移更多
    let translateX = '-50%';
    if (pointX + tooltipWidth / 2 > containerWidth - 20) {
      // 靠近右边缘，向左偏移
      translateX = '-100%';
      tooltip.style.left = `${pointX - 10}px`;
    } else if (pointX - tooltipWidth / 2 < 20) {
      // 靠近左边缘，向右偏移
      translateX = '0%';
      tooltip.style.left = `${pointX + 10}px`;
    }
    
    // 如果靠近顶部，tooltip 显示在下方
    let translateY = '-110%';
    if (nearY < 60) {
      translateY = '20%';
    }
    
    tooltip.style.transform = `translate(${translateX}, ${translateY})`;
    tooltip.classList.remove('hidden');
    tooltip.classList.add('chart-tooltip');
  };
  
  canvas.onmouseleave = () => {
    tooltip.classList.add('hidden');
    dataPoint.classList.add('hidden');
  };
}

// ==================== Additional Compare Selectors ====================
function populateErrorCompareSelects() {
  const leftSelect = $('errorLeftSelect');
  const rightSelect = $('errorRightSelect');
  if (!leftSelect || !rightSelect) return;
  
  const cnnErrors = [];
  const svmErrors = [];
  
  // 过滤出误差图
  (state.artifacts?.visualizations || []).forEach(v => {
    const name = (v.name || v.path || '').toLowerCase();
    if (name.includes('error') || name.includes('错误')) {
      const url = v.url ? `${API_BASE}${v.url}` : `${API_BASE}/files/${encodeURIComponent(v.path)}`;
      cnnErrors.push({ label: v.name, url });
    }
  });
  
  (state.svmArtifacts?.visualizations || []).forEach(v => {
    const name = (v.name || v.path || '').toLowerCase();
    if (name.includes('error') || name.includes('错误')) {
      const url = v.url ? `${API_BASE}${v.url}` : `${API_BASE}/files/${encodeURIComponent(v.path)}`;
      svmErrors.push({ label: v.name, url });
    }
  });
  
  [leftSelect, rightSelect].forEach((select, idx) => {
    select.innerHTML = '<option value="">选择误差图...</option>';
    const images = idx === 0 ? cnnErrors : svmErrors;
    const prefix = idx === 0 ? 'CNN' : 'SVM';
    images.forEach(img => {
      const opt = document.createElement('option');
      opt.value = img.url;
      opt.textContent = `${prefix}: ${img.label}`;
      select.appendChild(opt);
    });
  });
}

function populateGtPredCompareSelects() {
  const gtSelect = $('gtSelect');
  const predSelect = $('predSelect');
  if (!gtSelect || !predSelect) return;
  
  const gtImages = [];
  const predImages = [];
  
  const allVisuals = [
    ...(state.artifacts?.visualizations || []).map(v => ({...v, model: 'CNN'})),
    ...(state.svmArtifacts?.visualizations || []).map(v => ({...v, model: 'SVM'}))
  ];
  
  allVisuals.forEach(v => {
    const name = (v.name || v.path || '').toLowerCase();
    const url = v.url ? `${API_BASE}${v.url}` : `${API_BASE}/files/${encodeURIComponent(v.path)}`;
    
    if (name.includes('groundtruth') || name.includes('gt') || name.includes('truth')) {
      gtImages.push({ label: `${v.model}: ${v.name}`, url });
    }
    if (name.includes('prediction') || name.includes('pred') || name.includes('预测')) {
      predImages.push({ label: `${v.model}: ${v.name}`, url });
    }
  });
  
  gtSelect.innerHTML = '<option value="">选择Ground Truth...</option>';
  gtImages.forEach(img => {
    const opt = document.createElement('option');
    opt.value = img.url;
    opt.textContent = img.label;
    gtSelect.appendChild(opt);
  });
  
  predSelect.innerHTML = '<option value="">选择预测图...</option>';
  predImages.forEach(img => {
    const opt = document.createElement('option');
    opt.value = img.url;
    opt.textContent = img.label;
    predSelect.appendChild(opt);
  });
}

function updateErrorComparison() {
  const leftUrl = $('errorLeftSelect')?.value;
  const rightUrl = $('errorRightSelect')?.value;
  const container = $('errorCompareContainer');
  
  if (!container) return;
  
  if (!leftUrl || !rightUrl) {
    container.innerHTML = `
      <div class="image-compare-placeholder">
        <span class="material-symbols-outlined text-5xl text-on-surface-variant/30">error</span>
        <p class="text-on-surface-variant mt-2">选择CNN和SVM的误差图开始对比</p>
      </div>
    `;
    return;
  }
  
  container.innerHTML = `
    <div class="image-compare-container" id="errorSliderContainer">
      <div class="compare-image-bottom" id="errorBottom">
        <img src="${rightUrl}" alt="SVM Error" />
      </div>
      <div class="compare-image-top" id="errorTop">
        <img src="${leftUrl}" alt="CNN Error" />
      </div>
      <span class="compare-label left">CNN</span>
      <span class="compare-label right">SVM</span>
      <div class="compare-slider" id="errorSlider"></div>
    </div>
  `;
  
  initGenericOverlaySlider('errorSliderContainer', 'errorSlider', 'errorTop');
}

function updateGtPredComparison() {
  const gtUrl = $('gtSelect')?.value;
  const predUrl = $('predSelect')?.value;
  const container = $('gtPredCompareContainer');
  
  if (!container) return;
  
  if (!gtUrl || !predUrl) {
    container.innerHTML = `
      <div class="image-compare-placeholder">
        <span class="material-symbols-outlined text-5xl text-on-surface-variant/30">difference</span>
        <p class="text-on-surface-variant mt-2">选择Ground Truth和预测图开始对比</p>
      </div>
    `;
    return;
  }
  
  container.innerHTML = `
    <div class="image-compare-container" id="gtPredSliderContainer">
      <div class="compare-image-bottom" id="gtPredBottom">
        <img src="${predUrl}" alt="Prediction" />
      </div>
      <div class="compare-image-top" id="gtPredTop">
        <img src="${gtUrl}" alt="Ground Truth" />
      </div>
      <span class="compare-label left">Truth</span>
      <span class="compare-label right">Pred</span>
      <div class="compare-slider" id="gtPredSlider"></div>
    </div>
  `;
  
  initGenericOverlaySlider('gtPredSliderContainer', 'gtPredSlider', 'gtPredTop');
}

// 通用重叠式滑块初始化
function initGenericOverlaySlider(containerId, sliderId, topLayerId) {
  const container = $(containerId);
  const slider = $(sliderId);
  const topLayer = $(topLayerId);
  
  if (!container || !slider || !topLayer) return;
  
  let isDragging = false;
  
  function updateSlider(clientX) {
    const rect = container.getBoundingClientRect();
    let position = ((clientX - rect.left) / rect.width) * 100;
    position = Math.max(2, Math.min(98, position));
    
    slider.style.left = `${position}%`;
    topLayer.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
  }
  
  slider.addEventListener('mousedown', (e) => { isDragging = true; e.preventDefault(); });
  container.addEventListener('mousedown', (e) => { isDragging = true; updateSlider(e.clientX); });
  document.addEventListener('mousemove', (e) => { if (isDragging) updateSlider(e.clientX); });
  document.addEventListener('mouseup', () => { isDragging = false; });
  
  // Touch support
  slider.addEventListener('touchstart', (e) => { isDragging = true; e.preventDefault(); });
  container.addEventListener('touchstart', (e) => { isDragging = true; updateSlider(e.touches[0].clientX); });
  document.addEventListener('touchmove', (e) => { if (isDragging) updateSlider(e.touches[0].clientX); });
  document.addEventListener('touchend', () => { isDragging = false; });
  
  // Initialize at 50%
  updateSlider(container.getBoundingClientRect().left + container.getBoundingClientRect().width / 2);
}

// ==================== Dataset Path Selection ====================
function initDatasetPathSelection() {
  const select = $('datasetPathSelect');
  const manual = $('datasetPathManual');
  const fillBtn = $('btnFillDatasetPath');
  
  if (!select || !manual || !fillBtn) return;
  
  // 填充数据集选项
  select.innerHTML = '<option value="">选择数据集路径...</option>';
  state.datasets.forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds.data_path.replace(/[^/\\]+$/, '');
    opt.textContent = `${ds.name} (${ds.id})`;
    select.appendChild(opt);
  });
  
  select.addEventListener('change', () => {
    if (select.value) {
      manual.value = select.value;
    }
  });
  
  fillBtn.addEventListener('click', () => {
    const firstReady = state.datasets.find(d => d.ready);
    if (firstReady) {
      const path = firstReady.data_path.replace(/[^/\\]+$/, '');
      manual.value = path;
      log(`已填入数据集路径: ${path}`);
    }
  });
}

async function init() {
  bindEvents();
  initParticles();
  
  // 应用保存的主题
  const savedTheme = localStorage.getItem("specsure-theme") || "modern";
  updateThemeEffects(savedTheme);
  
  updateProgressUI("cnn", 0, "等待执行");
  updateProgressUI("svm", 0, "等待执行");
  await loadDefaults();
  await loadArtifacts();
  setSvmDefaultParams();
  applyDefaultModelHints();
  await loadEvaluations();
  
  // 填充图片对比选择器
  populateCompareSelects();
  populateErrorCompareSelects();
  populateGtPredCompareSelects();
  
  // 初始化数据集路径选择
  initDatasetPathSelection();
  
  // 初始化 Loss 曲线图、Accuracy 曲线图和雷达图
  setTimeout(async () => {
    await initLossChart();
    await initAccuracyChart();
    setupChartInteraction();
    initRadarChart();
  }, 500);
  
  // 更新指标面板
  updateMetricsPanel();
  
  log("前端就绪，按顺序进行数据→训练→查看产物");
}

init();
