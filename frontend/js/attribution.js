/* Intervention attribution: counterfactual + Shapley UI. */

let DOC = null;
let TARGETS_META = null;
let pollTimer = null;
let curveChart = null;
let showSolo = new Set();

const STRENGTH_LABEL = { strong: "显著", weak: "微弱", insignificant: "基本无效", uncertain: "—" };
const STAB_LABEL = { robust: "稳健", sensitive: "敏感", fragile: "脆弱", uncertain: "数据不足" };
const PAIR_LABEL = {
  strong_synergy: "强协同（放大）", weak_synergy: "弱协同（放大）",
  strong_overlap: "强重叠（抵消）", weak_overlap: "弱重叠（抵消）",
  additive: "近似独立", unknown: "—",
};

function pct(v, digits = 1) {
  if (v == null) return "—";
  return (v * 100).toFixed(digits) + "%";
}
function num(v, digits = 2) {
  if (v == null) return "—";
  return fmt(v, digits);
}
function signCls(v) {
  if (v == null) return "muted";
  return v > 0 ? "pos" : (v < 0 ? "neg" : "muted");
}
function strengthBadge(s) {
  const cls = { strong: "finished", weak: "paused", insignificant: "ready", uncertain: "ready" }[s] || "ready";
  return `<span class="badge ${cls}">${STRENGTH_LABEL[s] || s}</span>`;
}
function stabBadge(s) {
  const cls = { robust: "finished", sensitive: "paused", fragile: "error", uncertain: "ready" }[s] || "ready";
  return `<span class="badge ${cls}">${STAB_LABEL[s] || s}</span>`;
}
function lagCell(med, nullFrac) {
  if (med == null) return '<span class="muted">未起效</span>';
  return `第 ${med} 步 <span class="muted small">(${pct(1 - (nullFrac || 0), 0)}种子)</span>`;
}

async function startAnalysis() {
  const runId = el("runSelect").value;
  if (!runId) { alert("请先选择运行"); return; }
  const seeds = parseInt(el("seedSel").value, 10);
  el("runBtn").disabled = true;
  el("abortBtn").style.display = "";
  el("progressWrap").style.display = "";
  try {
    await post(`/api/runs/${runId}/attribution`, { seeds });
    poll();
  } catch (e) {
    el("jobStatus").textContent = "启动失败：" + e.message;
    resetButtons();
  }
}

function resetButtons() {
  el("runBtn").disabled = false;
  el("abortBtn").style.display = "none";
}

async function poll() {
  const runId = el("runSelect").value;
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const st = await get(`/api/runs/${runId}/attribution/status`);
      const p = st.progress || {};
      if (p.total) {
        el("progressBar").style.width = Math.round((p.done / p.total) * 100) + "%";
        el("progressText").textContent =
          `反事实重放 ${p.done}/${p.total}（种子 ${p.seeds || 0}/${p.total_seeds || "?"}）`;
      }
      if (st.status === "finished") {
        clearInterval(pollTimer);
        resetButtons();
        el("jobStatus").textContent = "完成 ✓ " + (st.finished_at || "");
        await loadResult();
      } else if (st.status === "error") {
        clearInterval(pollTimer);
        resetButtons();
        el("jobStatus").textContent = "失败：" + (st.error || "");
        el("progressText").textContent = "";
      } else if (st.status === "aborted") {
        clearInterval(pollTimer);
        resetButtons();
        el("jobStatus").textContent = "已取消";
      } else {
        el("jobStatus").textContent = "计算中…";
      }
    } catch (e) { /* keep polling */ }
  }, 700);
}

async function abortAnalysis() {
  const runId = el("runSelect").value;
  await api(`/api/runs/${runId}/attribution`, { method: "DELETE" });
}

async function loadResult() {
  const runId = el("runSelect").value;
  try {
    DOC = await get(`/api/runs/${runId}/attribution`);
  } catch (e) {
    el("body").style.display = "none";
    el("emptyBox").style.display = "";
    return;
  }
  renderAll();
}

function renderAll() {
  el("body").style.display = "";
  el("emptyBox").style.display = "none";

  // Target selector
  const keys = Object.keys(DOC.results);
  el("targetSel").innerHTML = keys.map((k) => {
    const t = DOC.results[k].target;
    return `<option value="${esc(k)}">${esc(t.label)}</option>`;
  }).join("");
  if (keys.includes(DOC.primary_target)) el("targetSel").value = DOC.primary_target;
  el("targetSel").onchange = renderResult;

  // Notes
  if (DOC.notes && DOC.notes.length) {
    el("notesBox").style.display = "";
    el("notesBox").className = "notice warn";
    el("notesBox").innerHTML = DOC.notes.map((n) => `• ${esc(n)}`).join("<br>");
  } else {
    el("notesBox").style.display = "none";
  }
  renderResult();
}

function renderResult() {
  const key = el("targetSel").value;
  const r = DOC.results[key];
  const t = r.target;
  const dirTxt = t.direction === "lower" ? "下降" : "提升";

  // ---- KPI cards ----
  const rel = r.relative_benefit;
  const base = r.baseline, bun = r.bundle;
  el("kpis").innerHTML = `
    <div class="card stat"><div class="k">无干预基线 · ${esc(t.label)}</div>
      <div class="v">${num(base.mean)}</div><div class="d">±${num(base.sd)}（${DOC.n_seeds} 种子）</div></div>
    <div class="card stat"><div class="k">实际干预包 · ${esc(t.label)}</div>
      <div class="v">${num(bun.mean)}</div><div class="d">±${num(bun.sd)}</div></div>
    <div class="card stat"><div class="k">干预包总效果（${dirTxt}比例）</div>
      <div class="v ${signCls(rel)}">${pct(rel)}</div>
      <div class="d">方向一致概率 ${pct(r.prob_beneficial, 0)} · 排名一致性 ρ=${num(r.rank_concordance, 2)}</div></div>
    <div class="card stat"><div class="k">起效时间（干预包）</div>
      <div class="v">${DOC.bundle_lag_median == null ? "—" : "滞后 " + DOC.bundle_lag_median + " 步"}</div>
      <div class="d">${DOC.bundle_onset_null_fraction === 1 ? '<span class="neg">未检测到起效</span>'
        : `${pct(1 - DOC.bundle_onset_null_fraction, 0)} 种子起效`}${DOC.bundle_adverse_lag_median != null ? ' · <span class="neg">有害起效滞后 ' + DOC.bundle_adverse_lag_median + ' 步</span>' : ""}</div></div>`;

  renderRankTable(r);
  renderPairs(r);
  renderStability(r);
  renderAux();
  renderCurve(r);
}

function renderRankTable(r) {
  const rows = r.interventions.slice().sort((a, b) => a.rank - b.rank);
  el("rankTable").innerHTML = `<thead><tr>
    <th>排名</th><th>干预</th><th>触发步</th>
    <th class="num">Shapley 贡献</th><th class="num">相对基线</th>
    <th class="num">单独效果</th><th class="num">有益种子</th>
    <th>单独起效滞后</th><th>叠加边际起效</th><th>效果判定</th><th>稳定性</th>
    </tr></thead><tbody>` + rows.map((x) => {
    const shap = x.shapley || {};
    return `<tr>
      <td class="num"><strong>#${x.rank}</strong></td>
      <td>${esc(x.label)} <span class="muted small">${x.scheduled ? "· 定时" : "· 手动"}</span></td>
      <td class="num">${x.at_step}</td>
      <td class="num ${signCls(shap.mean)}">${num(shap.mean)} <span class="muted small">±${num(shap.sd)}</span></td>
      <td class="num ${signCls(x.shapley_relative)}">${pct(x.shapley_relative)}</td>
      <td class="num ${signCls(x.solo_relative)}">${pct(x.solo_relative)}</td>
      <td class="num">${pct(x.prob_beneficial, 0)}</td>
      <td>${lagCell(x.solo_lag_median, x.solo_lag_null_fraction)}${
        x.solo_adverse_lag_median != null ? `<br><span class="neg small">有害 ${x.solo_adverse_lag_median} 步</span>` : ""}</td>
      <td>${lagCell(x.sequence_lag_median, x.sequence_lag_null_fraction)}${
        x.sequence_adverse_lag_median != null ? `<br><span class="neg small">有害 ${x.sequence_adverse_lag_median} 步</span>` : ""}</td>
      <td>${strengthBadge(x.strength)}</td>
      <td>${stabBadge(x.stability)}</td>
    </tr>`;
  }).join("") + "</tbody>";
}

function renderPairs(r) {
  const pairs = r.interactions || [];
  if (!pairs.length) {
    el("pairTable").innerHTML = '<p class="muted small">单个干预，无交互项。</p>';
    return;
  }
  el("pairTable").innerHTML = `<thead><tr>
    <th>干预 A</th><th>干预 B</th><th>关系</th><th class="num">交互指数</th>
    <th class="num">相对基线</th><th class="num">协同种子占比</th></tr></thead><tbody>` +
    pairs.map((p) => {
      const cls = p.classification.includes("synergy") ? "pos"
        : p.classification.includes("overlap") ? "neg" : "muted";
      const badge = p.classification.includes("synergy") ? "finished"
        : p.classification.includes("overlap") ? "error" : "ready";
      return `<tr>
        <td>${esc(p.label_i)}</td><td>${esc(p.label_j)}</td>
        <td><span class="badge ${badge}">${PAIR_LABEL[p.classification] || p.classification}</span></td>
        <td class="num ${cls}">${num(p.index && p.index.mean)}</td>
        <td class="num ${cls}">${pct(p.relative_mean)}</td>
        <td class="num">${pct(p.prob_synergy, 0)}</td></tr>`;
    }).join("") + "</tbody>";
  el("pairNote").textContent = r.decomposition_residual
    ? `Shapley 分解校验：Σ贡献 − 总效果 = ${num(r.decomposition_residual.mean)}（精确法应为 0；MC 法为抽样误差）。`
    : "";
}

function renderStability(r) {
  const thr = DOC.verdict_thresholds || {};
  const rows = r.interventions.slice().sort((a, b) => a.rank - b.rank);
  const rankBars = rows.map((x) => {
    const dist = x.rank_distribution || [];
    const k = dist.length;
    return `<div style="margin:6px 0">
      <div class="small muted">${esc(x.label)} · 平均名次 ${num(x.rank_mean, 1)}</div>
      <div class="row gap-6" style="align-items:flex-end">
        ${dist.map((c, i) => `<div title="第${i + 1}名 ×${c}" style="flex:1;height:${4 + c * 14}px;min-height:4px;
          background:${i + 1 === x.rank ? "var(--accent)" : "var(--panel-2)"};border-radius:2px"></div>`).join("")}
      </div></div>`;
  }).join("");
  el("stabilityBox").innerHTML = `
    <p class="small" style="line-height:1.7">
      排名一致性（平均 Spearman ρ）：<strong>${num(r.rank_concordance, 2)}</strong>
      <span class="muted">（≥0.8 排名稳定，&lt;0.5 说明不同随机性下“谁最有效”会变）</span><br>
      判定口径：|相对贡献| ≥ ${pct(thr.strong_relative, 0)} 为「显著」，≥ ${pct(thr.weak_relative, 0)} 为「微弱」，否则「基本无效」；
      变异系数 ≤ ${pct(thr.robust_cv, 0)} 且 ≥${pct(thr.robust_prob, 0)} 种子同号为「稳健」，
      ≥ ${pct(thr.fragile_cv, 0)} 为「脆弱」。
    </p>
    <p class="small muted" style="line-height:1.7">${esc(DOC.onset_rule.description)}</p>
    <p class="small muted" style="line-height:1.7">单独起效：${esc(DOC.onset_rule.solo)}<br>叠加边际起效：${esc(DOC.onset_rule.sequence)}</p>
    <hr style="border-color:var(--border)">
    <div class="small muted" style="margin-bottom:4px">各干预跨种子排名分布（${DOC.n_seeds} 个种子）：</div>
    ${rankBars}
    <p class="small muted">分析方法：${DOC.method === "exact" ? "精确 Shapley（全部 2^k 子集）"
      : `蒙特卡洛 Shapley（${DOC.n_permutations} 条排列 + 对偶抽样）`}；
      反事实模拟 ${DOC.n_simulations} 次；保真度（全集重放 vs 实际运行）：
      ${Object.entries(DOC.fidelity || {}).map(([k, v]) => `${esc(k)} ${pct(v, 2)}`).join("，") || "—"}。</p>`;
}

function renderAux() {
  const keys = Object.keys(DOC.aux || {});
  if (!keys.length) { el("auxTable").innerHTML = '<p class="muted small">无辅助指标。</p>'; return; }
  el("auxTable").innerHTML = `<thead><tr><th>指标</th><th class="num">基线均值</th>
    <th class="num">干预包均值</th><th class="num">变化（均值）</th></tr></thead><tbody>` +
    keys.map((ak) => {
      const a = DOC.aux[ak];
      const d = a.delta || {};
      return `<tr><td>${esc(a.spec.label)}</td>
        <td class="num">${num(a.baseline.mean)}</td>
        <td class="num">${num(a.bundle.mean)}</td>
        <td class="num ${signCls(d.mean)}">${d.mean > 0 ? "+" : ""}${num(d.mean)}</td></tr>`;
    }).join("") + "</tbody>";
}

// --------------------------------------------------------------------------- //
function medianSeries(seriesList) {
  const n = Math.min(...seriesList.map((s) => s.length));
  const out = [];
  for (let i = 0; i < n; i++) {
    const vals = seriesList.map((s) => s[i]).sort((a, b) => a - b);
    out.push(vals[Math.floor(vals.length / 2)]);
  }
  return out;
}

function renderCurve(r) {
  if (!curveChart) curveChart = echarts.init(el("curveChart"), "dark");
  const curves = DOC.curves || [];
  const steps = curves[0] ? curves[0].steps : [];
  const baseMed = medianSeries(curves.map((c) => c.baseline));
  const bundleMed = medianSeries(curves.map((c) => c.bundle));
  const soloMeds = (DOC.interventions || []).map((iv) =>
    medianSeries(curves.map((c) => (c.solo[iv.index] || []).slice())));

  // Mark intervention steps.
  const markLines = DOC.interventions.map((iv) => ({
    xAxis: iv.at_step, label: { formatter: iv.label.length > 8 ? iv.label.slice(0, 8) + "…" : iv.label,
      color: "#8b98a5", fontSize: 10 },
    lineStyle: { color: "#5b6b7d", type: "dashed" },
  }));

  const series = [
    { name: "无干预基线", type: "line", data: baseMed, showSymbol: false,
      smooth: true, lineStyle: { width: 2, type: "dashed" }, itemStyle: { color: "#e74c3c" } },
    { name: "实际干预包", type: "line", data: bundleMed, showSymbol: false,
      smooth: true, lineStyle: { width: 2.5 }, itemStyle: { color: "#2ecc71" },
      markLine: { symbol: "none", data: markLines, silent: true } },
  ];
  // Solo curves toggled by checkboxes.
  DOC.interventions.forEach((iv, i) => {
    if (showSolo.has(i)) {
      series.push({ name: "仅：" + iv.label, type: "line", data: soloMeds[i],
        showSymbol: false, smooth: true, lineStyle: { width: 1, type: "dotted", opacity: 0.75 } });
    }
  });

  curveChart.setOption({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: "#8b98a5" }, top: 0, type: "scroll" },
    grid: { left: 56, right: 20, top: 44, bottom: 36 },
    xAxis: { type: "category", data: steps, name: "时间步",
             axisLine: { lineStyle: { color: "#26313f" } } },
    yAxis: { type: "value", name: r.target.label,
             axisLine: { lineStyle: { color: "#26313f" } },
             splitLine: { lineStyle: { color: "#1b2430" } } },
    series,
  }, true);
  curveChart.resize();

  el("curveControls").innerHTML = '<span class="muted small">叠加单干预曲线：</span>' +
    DOC.interventions.map((iv) =>
      `<label class="check small"><input type="checkbox" class="soloChk" value="${iv.index}"
        ${showSolo.has(iv.index) ? "checked" : ""}> ${esc(iv.label)}</label>`).join("");
  el("curveControls").querySelectorAll(".soloChk").forEach((c) => {
    c.onchange = () => {
      const i = parseInt(c.value, 10);
      c.checked ? showSolo.add(i) : showSolo.delete(i);
      renderCurve(r);
    };
  });
}

window.addEventListener("resize", () => curveChart && curveChart.resize());

// --------------------------------------------------------------------------- //
async function selectRun() {
  const runId = el("runSelect").value;
  clearInterval(pollTimer);
  if (!runId) { el("body").style.display = "none"; return; }
  const meta = await get(`/api/runs/${runId}`);
  el("runInfo").textContent =
    `${meta.name} · ${DOMAIN_LABEL[meta.domain]} · 第 ${meta.current_step} 步 · ${meta.status} · ` +
    `${(meta.interventions || []).length} 个场景干预`;
  el("jobStatus").textContent = "";
  // Existing job?
  const st = await get(`/api/runs/${runId}/attribution/status`);
  if (st.status === "running") {
    el("runBtn").disabled = true;
    el("abortBtn").style.display = "";
    el("progressWrap").style.display = "";
    poll();
  } else {
    resetButtons();
    el("progressWrap").style.display = "none";
    if (st.has_result) { await loadResult(); el("jobStatus").textContent = "已有结果（" + st.result_summary.generated_at + "），可重新计算"; }
    else { el("body").style.display = "none"; el("emptyBox").style.display = ""; }
  }
}

async function init() {
  el("runBtn").onclick = startAnalysis;
  el("abortBtn").onclick = abortAnalysis;
  el("runSelect").onchange = selectRun;
  await fillRunSelect(el("runSelect"));
  if (el("runSelect").options.length > 1) {
    el("runSelect").selectedIndex = 1;
    selectRun();
  }
}

init().catch((e) => console.error(e));
