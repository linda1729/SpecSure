const DEFAULT_API_BASE = "http://8.140.214.49:8000";

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
  artifacts: { models: [], reports: [], visualizations: [] },
  evaluations: [],
  classMap: {},
  lastRun: null,
  currentJobId: null,
  pollTimer: null,
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

function updateProgressUI(progress = 0, statusText = "等待执行") {
  const percent = Math.max(0, Math.min(100, Number(progress) || 0));
  const fill = $("progressFill");
  const num = $("progressNumber");
  const label = $("progressLabel");
  if (fill) fill.style.width = `${percent}%`;
  if (num) num.textContent = `${percent.toFixed(1)}%`;
  if (label) label.textContent = statusText;
}

function stopPolling() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  state.currentJobId = null;
}

async function pollJob(jobId) {
  if (!jobId) return;
  state.currentJobId = jobId;
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/train/${jobId}`);
    if (state.currentJobId !== jobId) return;
    state.lastRun = data;
    renderRunResult(data);
    const statusText = readableStatus(data);
    updateProgressUI(data.progress ?? 0, statusText);
    if (data.status === "running" || data.status === "pending") {
      state.pollTimer = setTimeout(() => pollJob(jobId), 1500);
    } else {
      stopPolling();
      await loadArtifacts();
      await loadEvaluations();
      log(data.message || statusText || "任务完成");
    }
  } catch (err) {
    if (err.status === 404) {
      stopPolling();
      return log("未找到任务，可能已被清理");
    }
    log(`查询进度失败: ${err.message}`);
    updateProgressUI(0, "查询进度失败，稍后重试");
    state.pollTimer = setTimeout(() => pollJob(jobId), 2000);
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
  if ($("uploadDataset")) $("uploadDataset").value = id;
  if ($("trainDataset")) $("trainDataset").value = id;
  renderClassLegend(getClassNames(id));
  log(`已选择数据集 ${id}，可直接训练/推理`);
}

function renderDatasets() {
  const box = $("datasetCards");
  if (!box) return;
  box.innerHTML = "";
  if (!state.datasets.length) {
    box.innerHTML = '<div class="muted">暂无数据，请先上传或检查 models/cnn/data/ 目录。</div>';
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
        <span class="muted tiny">${ds.ready ? "可直接训练/推理" : "文件缺失需上传"}</span>
      </div>
    `;
    box.appendChild(card);
  });
  fillDatasetSelects();
  qsa(".btn-use-dataset").forEach((btn) => btn.addEventListener("click", () => chooseDataset(btn.dataset.id)));
}

function fillDatasetSelects() {
  const selects = [$("uploadDataset"), $("trainDataset")];
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
    if ($("uploadDataset")) $("uploadDataset").value = firstReady.id;
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

function defaultModelPath() {
  const dsId = $("trainDataset")?.value || "SA";
  const ds = state.datasets.find((d) => d.id === dsId);
  const folder = ds?.folder || dsId;
  const w = $("windowSize")?.value || state.defaults?.window_size || 25;
  const lr = $("learningRate")?.value || state.defaults?.lr || 0.001;
  const k =
    dsId === "IP"
      ? $("pcaIP")?.value || state.defaults?.pca_components_ip || 30
      : $("pcaOther")?.value || state.defaults?.pca_components_other || 15;
  const e = $("epochs")?.value || state.defaults?.epochs || 100;
  return `models/cnn/trained_models/HybridSN/${folder}_model_pca=${k}_window=${w}_lr=${lr}_epochs=${e}.pth`;
}

async function loadDefaults() {
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
    log("默认参数已加载");
  } catch (err) {
    log(`加载默认参数失败: ${err.message}`);
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

async function uploadDataset() {
  const dataset = $("uploadDataset").value;
  const hsi = $("hsiFile").files[0];
  const gt = $("gtFile").files[0];
  if (!dataset || !hsi || !gt) {
    return log("请选择数据集并上传 hsi / gt 文件");
  }
  const fd = new FormData();
  fd.append("dataset", dataset);
  fd.append("hsi_file", hsi);
  fd.append("gt_file", gt);
  try {
    await fetchJSON(`${API_BASE}/api/cnn/datasets/upload`, { method: "POST", body: fd });
    log(`已上传 ${dataset} 数据`);
    await refreshDatasets();
  } catch (err) {
    log(`上传失败: ${err.message}`);
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

async function loadArtifacts() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/artifacts`);
    state.artifacts = data;
    renderArtifacts();
    fillModelSelect();
  } catch (err) {
    log(`加载产物失败: ${err.message}`);
  }
}

async function loadEvaluations() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/evaluations`);
    state.evaluations = data || [];
    renderEvaluations();
  } catch (err) {
    log(`加载评估结果失败: ${err.message}`);
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
}

function renderEvaluations() {
  const box = $("evaluationList");
  if (!box) return;
  if (!state.evaluations.length) {
    box.innerHTML = '<div class="muted">暂无评估报告，先运行一次训练即可生成。</div>';
    return;
  }
  box.innerHTML = "";
  state.evaluations.forEach((item) => {
    const metrics = item.metrics || {};
    const art = item.artifacts || {};
    const urls = art.urls || {};
    const visuals = [
      ["预测", urls.prediction || art.prediction_path],
      ["GT", urls.groundtruth || art.groundtruth_path],
      ["混淆矩阵", urls.confusion || art.confusion_path],
      ["推理混淆", urls.inference_confusion || art.inference_confusion_path],
      ["伪彩色", urls.pseudocolor || art.pseudocolor_path],
      ["分类图", urls.classification || art.classification_path],
      ["对比图", urls.comparison || art.comparison_path],
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
    const card = document.createElement("div");
    card.className = "eval-card";
    card.innerHTML = `
      <div class="eval-head">
        <div>
          <p class="eyebrow">评估</p>
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
    `;
    box.appendChild(card);
  });
}

function renderRunResult(data) {
  if (!data) return;
  renderClassLegend(data.class_names || getClassNames(data.dataset));
  const setImg = (id, url) => {
    const img = $(id);
    if (!img) return;
    if (url) {
      img.src = toUrl(url);
      img.parentElement?.classList?.remove("muted");
    } else {
      img.removeAttribute("src");
      img.parentElement?.classList?.add("muted");
    }
  };
  const metricsBox = $("metricsBox");
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
  if ($("runMessage")) $("runMessage").textContent = data.message || statusText || "";
  const art = data.artifacts || {};
  const urls = art.urls || {};
  setImg("imgPrediction", urls.prediction || art.prediction_path);
  setImg("imgConfusion", urls.confusion || art.confusion_path);
  setImg("imgGT", urls.groundtruth || art.groundtruth_path);
  setImg("imgPseudo", urls.pseudocolor || art.pseudocolor_path);
  setImg("imgClassify", urls.classification || art.classification_path);
  setImg("imgCompare", urls.comparison || art.comparison_path);
  setImg("imgInferConfusion", urls.inference_confusion || art.inference_confusion_path);
  const linkBox = $("artifactLinks");
  if (linkBox) {
    linkBox.innerHTML = "";
    const pairs = [
      ["模型", urls.model || art.model_path],
      ["PCA", urls.pca || art.pca_path],
      ["报告", urls.report || art.report_path],
      ["混淆矩阵", urls.confusion || art.confusion_path],
      ["预测", urls.prediction || art.prediction_path],
      ["GT", urls.groundtruth || art.groundtruth_path],
      ["推理混淆", urls.inference_confusion || art.inference_confusion_path],
      ["伪彩色", urls.pseudocolor || art.pseudocolor_path],
      ["分类图", urls.classification || art.classification_path],
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
  const logsBox = $("runLogs");
  if (logsBox) logsBox.textContent = (data.logs_tail || []).join("\n");
}

async function runHybrid(mode) {
  stopPolling();
  updateProgressUI(2, "准备启动");
  const isInfer = mode === "infer";
  const payload = gatherParams();
  payload.inference_only = isInfer;
  if (payload.inference_only) {
    const manual = $("modelPathManual").value.trim();
    payload.input_model_path = $("modelPathSelect").value || manual || defaultModelPath();
    if (!payload.input_model_path) return log("推理模式需要选择或填写模型路径");
  }
  try {
    const data = await fetchJSON(`${API_BASE}/api/cnn/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastRun = data;
    renderRunResult(data);
    updateProgressUI(data.progress ?? 5, readableStatus(data));
    if (data.job_id) {
      log(`已提交 HybridSN ${isInfer ? "推理" : "训练"} 任务，Job ID: ${data.job_id}`);
      pollJob(data.job_id);
    } else {
      await loadArtifacts();
      await loadEvaluations();
      log(data.message || `HybridSN ${isInfer ? "推理" : "训练"}完成`);
    }
  } catch (err) {
    stopPolling();
    updateProgressUI(0, "执行失败");
    log(`执行失败: ${err.message}`);
  }
}

function bindEvents() {
  qsa(".nav-item").forEach((btn) => {
    if (btn.dataset.view) {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    }
  });
  if ($("btnUpload")) $("btnUpload").addEventListener("click", uploadDataset);
  if ($("btnUseDefaults"))
    $("btnUseDefaults").addEventListener("click", () => {
      const firstReady = state.datasets.find((d) => d.ready);
      if (firstReady) {
        chooseDataset(firstReady.id);
        log("已选默认数据，无需上传");
      } else {
        log("默认数据缺失，请上传对应 .mat");
      }
    });
  if ($("btnRefreshDatasets")) $("btnRefreshDatasets").addEventListener("click", refreshDatasets);
  if ($("btnTrain")) $("btnTrain").addEventListener("click", () => runHybrid("train"));
  if ($("btnInfer")) $("btnInfer").addEventListener("click", () => runHybrid("infer"));
  if ($("btnResetParams"))
    $("btnResetParams").addEventListener("click", () => {
      setDefaultParams();
      log("已恢复默认超参");
    });
  if ($("trainDataset"))
    $("trainDataset").addEventListener("change", (e) => {
      renderClassLegend(getClassNames(e.target.value));
    });
  if ($("btnFillDefaultModel"))
    $("btnFillDefaultModel").addEventListener("click", () => {
      const path = defaultModelPath();
      const input = $("modelPathManual");
      if (input) input.value = path;
      log(`已填入默认模型路径: ${path}`);
    });
  if ($("btnDocs")) $("btnDocs").addEventListener("click", () => window.open(`${API_BASE}/docs`, "_blank"));
}

async function init() {
  bindEvents();
  updateProgressUI(0, "等待执行");
  await loadDefaults();
  await loadArtifacts();
  await loadEvaluations();
  log("前端就绪，按顺序进行数据→训练→查看产物");
}

init();
