function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api");
  if (fromQuery) return fromQuery.replace(/\/$/, "");
  if (window.API_BASE) return window.API_BASE.replace(/\/$/, "");
  if (location.origin.includes(":5500")) return location.origin.replace(":5500", ":8000");
  return "http://localhost:8000";
}

const API_BASE = resolveApiBase();

const state = {
  datasets: [],
  labels: [],
  predictions: [],
  legend: [],
  currentRawUrl: "",
  evalResult: null,
  cnnStatus: null,
};

const $ = (id) => document.getElementById(id);
const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const formatTime = () => new Date().toLocaleTimeString();

function log(message) {
  const box = $("logBox");
  if (!box) return;
  box.textContent = `[${formatTime()}] ${message}\n` + box.textContent;
  const status = $("statusText");
  if (status) status.textContent = message;
}

function setView(view) {
  qsa(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  qsa(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
}

function updateMetrics() {
  const table = $("datasetTable");
  if (!table) return;
  table.innerHTML = "";
  state.datasets.forEach((d) => {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = `<span>${d.name}</span><span>Ready</span><span>${d.rows}×${d.cols}×${d.bands}</span>`;
    table.appendChild(row);
  });
}

function fillSelect(id, items, valueKey = "id", labelKey = "name", placeholder = "请选择") {
  const el = $(id);
  if (!el) return;
  el.innerHTML = "";
  if (!items.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = placeholder;
    el.appendChild(opt);
    return;
  }
  items.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item[valueKey];
    opt.textContent = item[labelKey] || item[valueKey];
    el.appendChild(opt);
  });
}

async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(txt || resp.statusText);
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) return resp.json();
  return resp.text();
}

function pickLabelForDataset(datasetId) {
  if (!state.labels.length) return "";
  const match = state.labels.find((l) => l.dataset_id === datasetId);
  return match ? match.id : state.labels[0].id;
}

function syncLabelForDataset(datasetId) {
  const labelId = pickLabelForDataset(datasetId);
  if (!labelId) return;
  const labelSelect = $("labelSelect");
  const resultLabel = $("resultLabelSelect");
  if (labelSelect) labelSelect.value = labelId;
  if (resultLabel) resultLabel.value = labelId;
  loadLegend(labelId);
}

async function loadDatasets() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/datasets`);
    state.datasets = data;
    fillSelect("datasetSelect", data);
    fillSelect("trainDatasetSelect", data);
    updateMetrics();
    if (data.length) {
      $("datasetSelect").value = data[0].id;
      $("trainDatasetSelect").value = data[0].id;
      await generatePreview();
      if (state.labels.length) syncLabelForDataset(data[0].id);
    }
    log("已加载数据集列表");
  } catch (err) {
    log(`加载数据集失败: ${err.message}`);
  }
}

async function loadDemo() {
  try {
    await fetchJSON(`${API_BASE}/api/datasets/demo?force=true`, { method: "POST" });
    await loadDatasets();
    await loadLabels();
    log("已加载示例数据");
  } catch (err) {
    log(`加载示例数据失败: ${err.message}`);
  }
}

async function loadLabels() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/labels`);
    state.labels = data;
    fillSelect("labelSelect", data);
    fillSelect("resultLabelSelect", data);
    if (data.length) {
      const dsId = $("trainDatasetSelect").value || data[0].dataset_id;
      const labelId = pickLabelForDataset(dsId);
      $("labelSelect").value = labelId;
      $("resultLabelSelect").value = labelId;
      await loadLegend(labelId);
    }
    log("已加载标注列表");
  } catch (err) {
    log(`加载标注失败: ${err.message}`);
  }
}

async function loadPredictions() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/predictions`);
    state.predictions = data.map((p) => ({
      id: p.id,
      name: p.model_id,
      url: normalizeImageUrl(p.preview_image_path),
    }));
    fillSelect("resultPredSelect", state.predictions, "id", "name", "选择预测结果");
    if (state.predictions[0]) $("circleA").src = state.predictions[0].url;
    if (state.predictions[1]) $("circleB").src = state.predictions[1].url;
    log("已加载历史预测结果");
  } catch (err) {
    log(`加载预测失败: ${err.message}`);
  }
}

async function loadLegend(labelId) {
  if (!labelId) return;
  try {
    const res = await fetchJSON(`${API_BASE}/api/labels/${labelId}/legend`);
    state.legend = res.legend || [];
    renderLegend();
  } catch (err) {
    log(`加载图例失败: ${err.message}`);
  }
}

function renderLegend() {
  const box = $("legendBox");
  if (!box) return;
  box.innerHTML = "";
  state.legend.forEach((item) => {
    const div = document.createElement("div");
    div.className = "legend-item";
    const [r, g, b] = item.color;
    div.innerHTML = `<span class="legend-color" style="background: rgb(${r},${g},${b})"></span><span>${item.class_name}</span>`;
    box.appendChild(div);
  });
}

function normalizeImageUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  const filename = path.split("/").pop();
  return `${API_BASE}/static/previews/${filename}`;
}

async function uploadDataset() {
  const fileInput = $("uploadFile");
  if (!fileInput.files.length) return log("请选择文件后再上传");
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  const name = $("uploadName").value.trim();
  if (name) fd.append("name", name);
  try {
    await fetchJSON(`${API_BASE}/api/datasets/upload`, { method: "POST", body: fd });
    log("上传成功，刷新列表");
    await loadDatasets();
  } catch (err) {
    log(`上传失败: ${err.message}`);
  }
}

async function generatePreview() {
  const ds = $("datasetSelect").value;
  if (!ds) return;
  const r = Number($("bandR").value);
  const g = Number($("bandG").value);
  const b = Number($("bandB").value);
  const down = Number($("downsample").value);
  try {
    const data = await fetchJSON(`${API_BASE}/api/datasets/${ds}/preview-rgb?r=${r}&g=${g}&b=${b}&downsample=${down}`);
    const url = `${API_BASE}${data.image_url}`;
    $("imgPreview").src = url;
    state.currentRawUrl = url;
    $("circleRaw").src = url;
    log("已生成伪彩色预览");
  } catch (err) {
    log(`预览生成失败: ${err.message}`);
  }
}

async function querySpectrum() {
  const ds = $("datasetSelect").value;
  const row = $("specRow").value;
  const col = $("specCol").value;
  if (!ds || row === "" || col === "") return log("请输入 row/col");
  try {
    const data = await fetchJSON(`${API_BASE}/api/datasets/${ds}/spectrum?row=${row}&col=${col}`);
    $("spectrumBox").textContent = JSON.stringify(data, null, 2);
    log("光谱查询成功");
  } catch (err) {
    log(`光谱查询失败: ${err.message}`);
  }
}

function parseRanges(text) {
  return text
    .split(",")
    .map((seg) => seg.trim())
    .filter(Boolean)
    .map((seg) => {
      const [a, b] = seg.split("-").map((v) => Number(v));
      if (Number.isFinite(a) && Number.isFinite(b)) return [a, b];
      return null;
    })
    .filter(Boolean);
}

async function runPreprocess() {
  const ds = $("datasetSelect").value;
  if (!ds) return log("请选择数据集");
  const config = {
    dataset_id: ds,
    noise_reduction: {
      enabled: $("nrEnabled").checked,
      method: $("nrMethod").value,
      kernel_size: Number($("nrKernel").value),
    },
    band_selection: {
      enabled: $("bsEnabled").checked,
      method: $("bsMethod").value,
      manual_ranges: parseRanges($("bsRanges").value),
      n_components: Number($("bsPca").value),
    },
    normalization: {
      enabled: $("normEnabled").checked,
      method: $("normMethod").value,
    },
  };
  try {
    const data = await fetchJSON(`${API_BASE}/api/preprocess/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    $("preprocessResult").textContent = `输出数据集: ${data.output_dataset.id} (${data.output_dataset.rows}×${data.output_dataset.cols}×${data.output_dataset.bands})`;
    await loadDatasets();
    $("trainDatasetSelect").value = data.output_dataset.id;
    log("预处理完成");
  } catch (err) {
    log(`预处理失败: ${err.message}`);
  }
}

async function loadBandImportance() {
  const ds = $("datasetSelect").value;
  if (!ds) return log("请选择数据集");
  try {
    const data = await fetchJSON(`${API_BASE}/api/preprocess/band-importance?dataset_id=${ds}`);
    renderImportance(data.bands || []);
    log("已获取波段评分");
  } catch (err) {
    log(`波段评分失败: ${err.message}`);
  }
}

function renderImportance(list) {
  const box = $("bandImportance");
  if (!box) return;
  box.innerHTML = "";
  const top = list.slice().sort((a, b) => b.score - a.score).slice(0, 10);
  top.forEach((item) => {
    const div = document.createElement("div");
    div.className = "importance-bar";
    const width = Math.round(item.score * 100);
    div.innerHTML = `<span>Band ${item.index}</span><div class="bar" style="width:${width}%"></div><span>${item.score.toFixed(2)}</span>`;
    box.appendChild(div);
  });
}

function collectModels(trainRatio) {
  const models = [];
  if ($("modelAEnabled").checked) {
    models.push({
      name: "ModelA",
      type: $("modelAType").value,
      enabled: true,
      train_ratio: Number(trainRatio),
      params: {
        C: Number($("modelAC").value || 1),
        gamma: $("modelAGamma").value || "scale",
        kernel: "rbf",
      },
    });
  }
  if ($("modelBEnabled").checked) {
    const patchSize = Number($("modelBPatch").value || 11);
    models.push({
      name: "ModelB",
      type: $("modelBType").value,
      enabled: true,
      train_ratio: Number(trainRatio),
      params: {
        n_estimators: 80,
        epochs: Number($("modelBEpochs").value || 50),
        batch_size: Number($("modelBBatch").value || 32),
        optimizer: $("modelBOpt").value || "adam",
        patch_size: patchSize,
      },
    });
  }
  return models;
}

function appendPredictions(runs) {
  runs.forEach((run, idx) => {
    const pred = run.prediction;
    const imageUrl = normalizeImageUrl(pred.preview_image_path);
    state.predictions = state.predictions.filter((p) => p.id !== pred.id);
    state.predictions.push({ id: pred.id, name: run.model_run.id, url: imageUrl });
    if (idx === 0) $("circleA").src = imageUrl;
    if (idx === 1) $("circleB").src = imageUrl;
  });
  fillSelect("resultPredSelect", state.predictions, "id", "name", "选择预测结果");
}

async function runTraining() {
  const datasetId = $("trainDatasetSelect").value;
  const labelId = $("labelSelect").value;
  if (!datasetId || !labelId) return log("请选择训练数据和标注");
  const trainRatio = $("trainRatio").value || 0.7;
  const models = collectModels(trainRatio);
  if (!models.length) return log("请至少启用一个模型");
  const payload = {
    dataset_id: datasetId,
    label_id: labelId,
    random_seed: Number($("randomSeed").value || 42),
    models,
  };
  const btn = $("btnTrain");
  const resultBox = $("trainResult");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Training...";
  }
  if (resultBox) {
    resultBox.textContent = "训练中... 数据较多时可能需要几分钟，请勿刷新页面。";
  }
  log("开始训练，等待后端返回（可在后端终端查看实时日志）");
  try {
    const data = await fetchJSON(`${API_BASE}/api/train-and-predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("trainResult").textContent = `完成 ${data.runs.length} 个模型，预测 ID: ${data.runs.map((r) => r.prediction.id).join(", ")}`;
    appendPredictions(data.runs);
    log("训练与预测完成");
  } catch (err) {
    log(`训练失败: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Start Training";
    }
  }
}

async function runEvaluate() {
  const predId = $("resultPredSelect").value;
  const labelId = $("resultLabelSelect").value;
  if (!predId || !labelId) return log("请选择预测与标注");
  try {
    const data = await fetchJSON(`${API_BASE}/api/evaluate?prediction_id=${predId}&label_id=${labelId}`, { method: "POST" });
    state.evalResult = data;
    renderEval(data);
    log("评估完成");
  } catch (err) {
    log(`评估失败: ${err.message}`);
  }
}

function renderEval(result) {
  $("metricOA").textContent = result.overall_accuracy.toFixed(4);
  $("metricKappa").textContent = result.kappa.toFixed(4);
  renderCM(result.confusion_matrix);
  renderClassTable(result.per_class);
}

function renderCM(cm) {
  const box = $("cmTable");
  if (!box) return;
  const labels = cm.labels;
  const table = document.createElement("table");
  let html = "<thead><tr><th>GT\\Pred</th>";
  labels.forEach((l) => (html += `<th>${l}</th>`));
  html += "</tr></thead><tbody>";
  cm.matrix.forEach((row, i) => {
    html += `<tr><th>${labels[i]}</th>`;
    row.forEach((val) => {
      const shade = Math.min(1, val / (Math.max(...row) + 1e-6));
      const bg = `rgba(77, 225, 193, ${0.15 + shade * 0.5})`;
      html += `<td style="background:${bg}">${val}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  table.innerHTML = html;
  box.innerHTML = "";
  box.appendChild(table);
}

function renderClassTable(list) {
  const box = $("classTable");
  if (!box) return;
  const table = document.createElement("table");
  let html = "<thead><tr><th>Class</th><th>Producer</th><th>User</th></tr></thead><tbody>";
  list.forEach((c) => {
    html += `<tr><td>${c.class_name}</td><td>${c.producer_accuracy.toFixed(3)}</td><td>${c.user_accuracy.toFixed(3)}</td></tr>`;
  });
  html += "</tbody>";
  table.innerHTML = html;
  box.innerHTML = "";
  box.appendChild(table);
}

function bindNav() {
  qsa(".nav-item").forEach((btn) => {
    if (btn.dataset.view) {
      btn.addEventListener("click", () => setView(btn.dataset.view));
    }
  });
}

function bindEvents() {
  bindNav();
  $("btnDocs").addEventListener("click", () => window.open(`${API_BASE}/docs`, "_blank"));
  $("btnUpload").addEventListener("click", uploadDataset);
  $("btnDemo").addEventListener("click", loadDemo);
  $("btnPreview").addEventListener("click", generatePreview);
  $("btnSpectrum").addEventListener("click", querySpectrum);
  $("btnPreprocess").addEventListener("click", runPreprocess);
  $("btnBandImportance").addEventListener("click", loadBandImportance);
  $("btnTrain").addEventListener("click", runTraining);
  $("btnEvaluate").addEventListener("click", runEvaluate);
  $("trainRatio").addEventListener("input", (e) => ($("trainRatioVal").textContent = e.target.value));
  $("modelAC").addEventListener("input", (e) => ($("modelACVal").textContent = e.target.value));
  $("modelBPatch").addEventListener("input", (e) => ($("modelBPatchVal").textContent = e.target.value));
  $("nrKernel").addEventListener("input", (e) => ($("nrKernelVal").textContent = e.target.value));
  $("resultLabelSelect").addEventListener("change", (e) => loadLegend(e.target.value));
  $("trainDatasetSelect").addEventListener("change", (e) => syncLabelForDataset(e.target.value));
}

async function loadCnnStatus() {
  try {
    const data = await fetchJSON(`${API_BASE}/api/models/cnn/status`);
    state.cnnStatus = data;
    renderCnnStatus();
    log(`CNN 服务模式: ${data.mode}`);
  } catch (err) {
    state.cnnStatus = { mode: "unknown", message: err.message };
    renderCnnStatus();
    log(`CNN 状态获取失败: ${err.message}`);
  }
}

function renderCnnStatus() {
  const status = state.cnnStatus || {};
  const badge = $("cnnStatusBadge");
  const endpoint = $("cnnEndpointBadge");
  if (!badge || !endpoint) return;
  badge.textContent = status.mode || "unknown";
  badge.className = `badge ${status.mode === "remote" ? "online" : "offline"}`;
  endpoint.textContent = status.endpoint || "本地占位";
  $("cnnStatusDesc").textContent = status.message || "等待检查";
  $("cnnMode").textContent = status.mode || "-";
  $("cnnTimeout").textContent = status.timeout ? `${status.timeout}s` : "-";
  $("cnnPath").textContent = status.predict_path || "/predict";
}

async function init() {
  bindEvents();
  await loadCnnStatus();
  await loadDatasets();
  if (!state.datasets.length) {
    await loadDemo();
  }
  await loadLabels();
  await loadPredictions();
  log("前端就绪，按侧边栏步骤操作");
}

init();
