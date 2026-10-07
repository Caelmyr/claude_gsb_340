/* Intervention attribution: counterfactual / Shapley analysis UI. */

let runId = null;
let runMeta = null;
let doc = null;
let chart = null;
let pollTimer = null;

const EFFECT_LABEL = {
  strong: ["强有效", "finished"],
  effective: ["有效", "running"],
  weak: ["弱效果", "paused"],
  none: ["基本无效", "ready"],
  adverse: ["反效果", "error"],
};

const KIND_LABEL = {
  synergy: ["效果放大（协同）", "finished"],
  antagonism: ["重叠/抵消", "stopped"],
  additive: ["近似叠加", "ready"],
};

function itvById(id) {
  return doc.instances.find((i) => i.id === id);
}

function itvName(id) {
  const it = itvById(id);
  if (!it) return id;
  const params = it.param_text ? `（${it.param_text}）` : "";
  return `${it.label}${params} · 第${it.step}步`;
}

function signed(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return v > 0 ? `+${fmt(v)}` : fmt(v);
}

function sdCell(mean, sd) {
  return `${fmt(mean)} <span class="muted small">± ${fmt(sd)}</span>`;
}

/* ------------------------------------------------------------------ */
/* Render: validation / summary / method                              */
/* ------------------------------------------------------------------ */
function renderValidation() {
  const v = doc.validation;
  const cls = v.verified ? "finished" : "error";
  el("validationCard").innerHTML = `
    <div class="row between">
      <div>
        <strong>重放校验</strong>
        <span class="badge ${cls}" style="margin-left:8px">
          ${v.verified ? "逐点一致 ✓" : "存在偏差"}</span>
      </div>
      <span class="muted small">${esc(doc.name)} · ${DOMAIN_LABEL[doc.domain]} · ${MODEL_LABEL[doc.model]} · ${doc.params.horizon} 步</span>
    </div>
    <p class="muted small" style="margin:8px 0 0">${esc(v.note)}</p>`;
}

function renderSummary() {
  el("summary").innerHTML = doc.summary.map((s) =>
    `<li style="margin:6px 0">${esc(s)}</li>`).join("");
}

function renderMethod() {
  const p = doc.params;
  const rule = p.onset_rule;
  el("methodInfo").innerHTML = `
    <div>· 归因方法：${p.method === "exact" ? "全子集精确枚举 + Shapley 值" : "排列抽样 Shapley 估计"}；模拟联盟组合 ${p.coalitions} 个 × ${p.replicates} 组随机种子 = 共 ${p.replays} 次反事实重放${p.method === "sampled" ? `（${p.permutations} 个随机排列，单干预/两两/缺一组合为精确值）` : ""}。</div>
    <div>· 公共随机数（CRN）：每个联盟在相同种子序列下配对重放，差值直接归因，避免随机性混淆。</div>
    <div>· Shapley 值 = 该干预在所有出场顺序下的平均边际贡献；各干预 Shapley 值之和 = 全部干预的合计改善（抽样模式下有小残差，见对照表）。</div>
    <div>· 起效判定口径：配对差值按 ${rule.window} 步因果滑动平均平滑，阈值 = max(${rule.noise_z}×干预前 ${rule.pre_window} 步的步间波动 σ, 基线均值的 ${rule.rel_floor * 100}%)，连续 ${rule.hold} 步超过阈值才记为起效；否则记「观察窗内未检出」。</div>
    <div>· 稳定性：中位数 ± 总体标准差，「方向一致率」= 该种子下贡献为有利方向的比例；排名在所有种子下不变时标记「排名稳定」。</div>
    <div>· 种子序列：${doc.seeds.join(", ")}</div>`;
}

/* ------------------------------------------------------------------ */
/* Render: chart                                                      */
/* ------------------------------------------------------------------ */
function renderChart() {
  const t = doc.trajectories;
  if (!chart) chart = echarts.init(el("chart"), "dark");
  const series = [
    { name: "无干预基线（反事实）", type: "line", showSymbol: false,
      lineStyle: { type: "dashed", width: 2, color: "#7f8c8d" },
      data: t.baseline },
    { name: "全部干预叠加（反事实）", type: "line", showSymbol: false,
      lineStyle: { width: 2.5, color: "#2ecc71" }, data: t.full },
  ];
  const colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#1abc9c",
                  "#e67e22", "#95a5a6", "#d35400", "#27ae60", "#2980b9",
                  "#8e44ad", "#c0392b"];
  t.singletons.forEach((s, k) => {
    const it = itvById(s.id);
    series.push({
      name: `仅「${it ? it.label : s.id}」`, type: "line", showSymbol: false,
      lineStyle: { width: 1.2, color: colors[k % colors.length] },
      data: s.values,
    });
  });
  const markLines = t.intervention_steps.map((step) => ({ xAxis: step }));
  series[1].markLine = {
    symbol: "none", silent: true,
    label: { formatter: "干预", color: "#8b98a5", fontSize: 10 },
    lineStyle: { color: "#566573", type: "dotted" },
    data: markLines,
  };
  chart.setOption({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: "#8b98a5", fontSize: 11 }, top: 0,
              type: "scroll" },
    grid: { left: 60, right: 24, top: 56, bottom: 40 },
    xAxis: { type: "category", data: t.steps, name: "时间步",
             axisLine: { lineStyle: { color: "#26313f" } } },
    yAxis: { type: "value", axisLine: { lineStyle: { color: "#26313f" } },
             splitLine: { lineStyle: { color: "#1b2430" } } },
    series,
  }, true);
  chart.resize();
  const metricLabel = (doc.metrics.find((m) => m.key === t.metric) || {}).label
      || t.metric;
  el("chartHint").textContent =
    `纵轴：${metricLabel}。虚线为「什么都不做」的反事实基线，绿线为全部干预同时生效，`
    + `彩色线为只保留单项干预的轨迹；竖线为干预施加时刻。`;
}

window.addEventListener("resize", () => chart && chart.resize());

/* ------------------------------------------------------------------ */
/* Render: attribution table                                          */
/* ------------------------------------------------------------------ */
function initMetricSelect() {
  const sel = el("metricSelect");
  sel.innerHTML = doc.metrics
    .filter((m) => m.ranking)
    .map((m) => `<option value="${esc(m.key)}">${esc(m.label)}${m.primary ? " ★" : ""}</option>`)
    .join("");
  const primaryKey = (doc.metrics.find((m) => m.primary && m.ranking)
                      || doc.metrics.find((m) => m.ranking) || {}).key;
  if (primaryKey) sel.value = primaryKey;
  sel.onchange = renderTable;
}

function renderTable() {
  const block = doc.metrics.find((m) => m.key === el("metricSelect").value);
  if (!block) return;

  if (!block.ranking) {
    el("attribTable").innerHTML = "";
    el("tableHint").textContent = "该指标为中性参照量（无好坏方向），不参与贡献排序。";
    return;
  }
  const dirTxt = block.direction === "down" ? "下降" : "提升";
  const rows = block.ranking.map((r, pos) => {
    const it = itvById(r.id);
    const [effTxt, effCls] = EFFECT_LABEL[r.effect_class] || ["", ""];
    const pctTotal = r.pct_of_total == null ? "—" : `${fmt(r.pct_of_total, 1)}%`;
    return `<tr>
      <td><strong>${pos + 1}</strong></td>
      <td>${esc(it ? it.label : r.id)}
        <div class="muted small">第 ${it ? it.step : "?"} 步 · ${it ? (it.scheduled ? "定时" : "手动") : ""}</div></td>
      <td class="num">${sdCell(r.shapley_mean, r.shapley_sd)}</td>
      <td class="num">${pctTotal}</td>
      <td class="num">${fmt(r.pct_of_baseline, 1)}%</td>
      <td class="num">${sdCell(r.singleton_mean, r.singleton_sd)}</td>
      <td class="num">${signed(r.loo_mean)}</td>
      <td class="num">${fmt(r.benefit_rate * 100, 0)}%</td>
      <td class="num">${r.rank_mode}${r.rank_stable ? " ✓" : ""}</td>
      <td><span class="badge ${effCls}">${effTxt}</span></td>
    </tr>`;
  }).join("");

  el("attribTable").innerHTML = `
    <thead><tr>
      <th>排名</th><th>干预</th>
      <th class="num">Shapley 贡献<br><span class="muted small">中位±SD</span></th>
      <th class="num">占合计<br>改善</th>
      <th class="num">占基线<br>${dirTxt}</th>
      <th class="num">单独实施<br>效果</th>
      <th class="num">缺一影响<br>(LOO)</th>
      <th class="num">方向<br>一致率</th>
      <th class="num">多种子<br>排名</th>
      <th>判定</th>
    </tr></thead><tbody>${rows}</tbody>`;

  const residual = block.decomposition_residual;
  el("tableHint").innerHTML =
    `无干预基线 <strong>${fmt(block.baseline_mean)} ± ${fmt(block.baseline_sd)}</strong> → `
    + `全干预叠加 <strong>${fmt(block.full_mean)} ± ${fmt(block.full_sd)}</strong>，`
    + `合计${dirTxt} <strong>${signed(block.total_benefit_mean)}</strong>`
    + (Math.abs(residual) > 0.5
       ? `（各干预 Shapley 之和与合计相差 ${fmt(residual)}，为抽样估计残差，精确模式下为 0）`
       : "（Shapley 之和与合计完全吻合）")
    + `。<br>「单独实施效果」= 只有该干预时相对基线的改善；「缺一影响」= 从全部干预中拿掉它后总效果损失，衡量其在叠加中的不可替代性；判定阈值见页面底部口径说明。`;
}

/* ------------------------------------------------------------------ */
/* Render: onset / interactions                                       */
/* ------------------------------------------------------------------ */
function renderOnset() {
  const rows = doc.onset || [];
  const primary = rows.filter((r) => r.primary);
  const shown = primary.length ? primary : rows;
  el("onsetTable").innerHTML = `
    <thead><tr><th>干预</th><th class="num">起效步</th><th class="num">滞后</th>
      <th class="num">检出率</th><th class="num">持续占比</th><th class="num">IQR</th></tr></thead>
    <tbody>${shown.map((r) => {
      const it = itvById(r.id);
      const detected = r.detection_rate > 0;
      const iqr = detected
        ? `[${r.lag_q1_q3[0] == null ? "—" : fmt(r.lag_q1_q3[0])}, ${r.lag_q1_q3[1] == null ? "—" : fmt(r.lag_q1_q3[1])}]`
        : "—";
      return `<tr>
        <td>${esc(it ? it.label : r.id)}<div class="muted small">第 ${it ? it.step : "?"} 步施加 · 指标 ${esc(r.metric_label)}</div></td>
        <td class="num">${detected ? r.onset_step_median : '<span class="muted">未检出</span>'}</td>
        <td class="num">${detected ? `${r.lag_median} 步` : "—"}</td>
        <td class="num">${fmt(r.detection_rate * 100, 0)}%</td>
        <td class="num">${fmt(r.sustained_rate * 100, 0)}%</td>
        <td class="num muted small">${iqr}</td>
      </tr>`;
    }).join("")}</tbody>`;

  const rule = doc.params.onset_rule;
  el("onsetRule").textContent =
    `判定口径：与无干预基线配对（相同种子），方向化差值做 ${rule.window} 步滑动平均，`
    + `超过 max(${rule.noise_z}σ, ${rule.rel_floor * 100}% 基线) 且连续 ${rule.hold} 步记为起效。`
    + `滞后 = 起效步 − 施加步；IQR = 滞后在多种子下的四分位距。`;
}

function renderInteractions() {
  const blocks = doc.interactions || [];
  const primary = blocks.find((b) => b.primary) || blocks[0];
  if (!primary) {
    el("interTable").innerHTML = '<tbody><tr><td class="muted">无可计算的相互作用。</td></tr></tbody>';
    return;
  }
  el("interTable").innerHTML = `
    <thead><tr><th>干预 A</th><th>干预 B</th><th class="num">协同项</th>
      <th class="num">Shapley 交互指数</th><th>判定</th></tr></thead>
    <tbody>${primary.pairs.map((p) => {
      const [kTxt, kCls] = KIND_LABEL[p.kind] || ["", ""];
      return `<tr>
        <td>${esc((itvById(p.a) || {}).label || p.a)}</td>
        <td>${esc((itvById(p.b) || {}).label || p.b)}</td>
        <td class="num">${signed(p.synergy)}</td>
        <td class="num">${p.interaction_index == null ? '<span class="muted small">抽样模式未计</span>' : signed(p.interaction_index)}</td>
        <td><span class="badge ${kCls}">${kTxt}</span></td>
      </tr>`;
    }).join("")}</tbody>`;
}

/* ------------------------------------------------------------------ */
/* Data loading                                                       */
/* ------------------------------------------------------------------ */
function renderAll() {
  el("result").style.display = "block";
  el("exportBtn").disabled = false;
  renderValidation();
  renderSummary();
  renderChart();
  initMetricSelect();
  renderTable();
  renderOnset();
  renderInteractions();
  renderMethod();
}

async function loadDoc() {
  const data = await get(`/api/runs/${runId}/attribution`);
  if (data.status === "running") {
    const j = data.job;
    el("jobInfo").textContent =
      `归因计算中… ${fmt((j.progress || 0) * 100, 0)}% — ${j.message || ""}`;
    pollTimer = setTimeout(loadDoc, 1500);
    return;
  }
  el("jobInfo").textContent =
    `分析完成于 ${data.generated_at || ""}`;
  doc = data;
  renderAll();
}

async function startAnalysis() {
  if (!runId) { alert("请先选择运行"); return; }
  const replicates = Math.max(1, Math.min(12,
      parseInt(el("replicates").value || "6", 10)));
  el("runBtn").disabled = true;
  el("jobInfo").textContent = "归因分析已提交，正在进行反事实重模拟…";
  try {
    await post(`/api/runs/${runId}/attribution`,
               { replicates, background: true });
    pollTimer = setTimeout(loadDoc, 800);
  } catch (e) {
    el("jobInfo").textContent = "分析失败：" + e.message;
  } finally {
    el("runBtn").disabled = false;
  }
}

async function onRunChange() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  doc = null;
  el("result").style.display = "none";
  el("exportBtn").disabled = true;
  runMeta = await get(`/api/runs/${runId}`);
  const nItv = (runMeta.interventions || []).filter((i) => i.applied).length;
  el("runInfo").textContent =
    `${runMeta.name} · ${DOMAIN_LABEL[runMeta.domain]} · ${MODEL_LABEL[runMeta.model]} · 第 ${runMeta.current_step} 步 · ${runMeta.status} · 定时干预 ${nItv} 项（手动干预同样计入归因）`;
  try {
    await loadDoc();
  } catch (e) {
    el("jobInfo").textContent = "尚未进行归因分析，点击上方按钮开始。";
  }
}

async function init() {
  el("runBtn").onclick = startAnalysis;
  el("refreshBtn").onclick = () => runId && onRunChange();
  el("exportBtn").onclick = () => {
    if (runId) window.location = `/api/export/${runId}?format=attribution`;
  };
  el("runSelect").onchange = (e) => {
    if (e.target.value) { runId = e.target.value; onRunChange(); }
  };
  await fillRunSelect(el("runSelect"));
  const q = new URLSearchParams(window.location.search).get("run");
  if (q && el("runSelect").querySelector(`option[value="${q}"]`)) {
    el("runSelect").value = q;
    runId = q;
    await onRunChange();
  } else if (el("runSelect").options.length > 1) {
    el("runSelect").selectedIndex = 1;
    runId = el("runSelect").value;
    await onRunChange();
  }
}

init().catch((e) => console.error(e));
