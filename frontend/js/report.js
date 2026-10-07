/* Report generation: summary + metrics table + config + events. */

async function generate() {
  const runId = el("runSelect").value;
  if (!runId) { alert("请选择运行"); return; }
  el("genStatus").textContent = "生成中…";
  try {
    const rpt = await post(`/api/reports/${runId}`);
    render(rpt);
    el("genStatus").textContent = "已生成 ✓";
  } catch (e) { el("genStatus").textContent = "生成失败：" + e.message; }
}

function render(rpt) {
  const metrics = Object.values(rpt.metrics || {});
  const metricRows = metrics.map((m) => `
    <tr><td>${esc(m.label)}</td><td class="num">${fmt(m.initial)}</td>
    <td class="num">${fmt(m.final)}</td><td class="num">${fmt(m.min)}</td>
    <td class="num">${fmt(m.max)}</td><td class="num">${m.peak_step}</td></tr>`).join("");

  const events = (rpt.events || []).slice().reverse().map((e) => `
    <tr><td>${e.step}</td><td>${esc(e.type)}</td><td class="muted small">${esc((e.result && e.result.reason) || "")}</td></tr>`).join("");

  const attr = rpt.attribution;
  const EFF = { strong: "强有效", effective: "有效", weak: "弱效果",
                none: "基本无效", adverse: "反效果" };
  const attrCard = attr ? `
    <div class="card">
      <div class="row between">
        <h2 style="margin:0">干预归因（${esc(attr.primary_metric || "")}）</h2>
        <a class="btn small" href="/attribution.html?run=${encodeURIComponent(rpt.run_id)}">查看完整归因分析 →</a>
      </div>
      <p class="muted small" style="margin:8px 0">
        无干预基线 ${fmt(attr.baseline_mean)} → 全干预叠加 ${fmt(attr.full_mean)}，
        合计改善 ${fmt(attr.total_benefit_mean)}；
        ${attr.method === "exact" ? "全子集精确 Shapley" : "排列抽样 Shapley"}，
        ${attr.replicates} 组随机种子。
      </p>
      <div style="overflow-x:auto"><table>
        <thead><tr><th>排名</th><th>干预</th><th class="num">施加步</th>
          <th class="num">Shapley 贡献</th><th class="num">占基线</th>
          <th class="num">方向一致率</th><th>判定</th></tr></thead>
        <tbody>${attr.ranking.map((r, k) => `
          <tr><td>${k + 1}</td><td>${esc(r.label)}</td>
            <td class="num">${r.step ?? "—"}</td>
            <td class="num">${fmt(r.shapley_mean)}</td>
            <td class="num">${fmt(r.pct_of_baseline, 1)}%</td>
            <td class="num">${fmt(r.benefit_rate * 100, 0)}%</td>
            <td>${esc(EFF[r.effect_class] || r.effect_class)}</td></tr>`).join("")}
        </tbody>
      </table></div>
    </div>` : `
    <div class="card">
      <h2>干预归因</h2>
      <p class="muted small">尚未进行归因分析。
        <a href="/attribution.html?run=${encodeURIComponent(rpt.run_id)}">前往「干预归因」页</a>
        通过反事实重模拟量化每项干预的贡献、起效时刻与相互作用。</p>
    </div>`;

  el("reportBody").innerHTML = `
    <div class="card">
      <div class="row between">
        <h2 style="margin:0">${esc(rpt.name)}</h2>
        <div class="row gap-6">
          <span class="badge">${DOMAIN_LABEL[rpt.domain] || rpt.domain}</span>
          <span class="badge">${MODEL_LABEL[rpt.model] || rpt.model}</span>
          <span class="badge ready">${rpt.steps} 步</span>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>结论摘要</h2>
      <ul style="margin:0;padding-left:18px">${(rpt.summary || []).map((s) => `<li style="margin:6px 0">${esc(s)}</li>`).join("")}</ul>
    </div>

    ${attrCard}

    <div class="card">
      <h2>指标统计</h2>
      <div style="overflow-x:auto"><table>
        <thead><tr><th>指标</th><th class="num">初始</th><th class="num">最终</th><th class="num">最小</th><th class="num">最大</th><th class="num">峰值步</th></tr></thead>
        <tbody>${metricRows || '<tr><td colspan="6" class="muted">无指标</td></tr>'}</tbody>
      </table></div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h2>干预日志（${(rpt.events || []).length}）</h2>
        <div style="overflow-x:auto"><table>
          <thead><tr><th>步</th><th>类型</th><th>结果</th></tr></thead>
          <tbody>${events || '<tr><td colspan="3" class="muted">无干预</td></tr>'}</tbody>
        </table></div>
      </div>
      <div class="card">
        <h2>配置参数</h2>
        <pre class="mono small" style="margin:0;white-space:pre-wrap;color:var(--accent-2)">${esc(JSON.stringify(rpt.config, null, 2))}</pre>
      </div>
    </div>`;
}

async function init() {
  el("genBtn").onclick = generate;
  await fillRunSelect(el("runSelect"));
  const q = new URLSearchParams(window.location.search).get("run");
  if (q && el("runSelect").querySelector(`option[value="${q}"]`)) {
    el("runSelect").value = q;
    await generate();
  } else if (el("runSelect").options.length > 1) {
    el("runSelect").selectedIndex = 1;
    await generate();
  }
}

init().catch((e) => console.error(e));
