"""Intervention attribution: quantify how much *each* intervention mattered.

Given one finished run with one or more interventions, this module answers:

* 干预让关键指标改变了多少（相对「无干预反事实」的基线）？
* 多个干预同时 / 先后叠加时，各自贡献多少（Shapley 值公平分摊）？
* 谁和谁相互放大（协同）、谁和谁相互抵消（重叠/冗余）？
* 干预何时开始起效（明确的判定口径）？
* 换随机种子后结论是否稳定（多种子分布 + 排名一致性）？

Method
------
1. **Counterfactual replays.** 对每个随机种子，用完全相同的初始条件重放
   全部干预子集（空集 = 无干预基线，全集 = 实际干预包）。因为基线与干预场景
   共享种子与初始状态，前后差异可归因于干预而非随机波动；同时用「全集重放 vs
   实际运行」的指标差做保真度校验。
2. **Shapley values.** 把「干预包的总收益」视作合作博弈的联盟收益，用精确
   Shapley 值（干预 ≤ 6 个）或随机排列蒙特卡洛估计把总效果公平分摊到每个
   干预——自动处理同时、先后、相互重叠的干预。
3. **Interaction index.** 二阶 Shapley 交互指数（精确模式）或空集锚定协同项
   （蒙特卡洛模式）量化成对放大 / 抵消。
4. **Onset.** 起效时间有明确口径（见 :func:`detect_onset`），并同时给出
   「单独起效」与「按实际顺序叠加时的边际起效」。
5. **Stability.** 默认多个种子重复，报告均值 / 标准差 / 变异系数 / 有益方向
   概率 / 排名的 Spearman 一致性。

Heavy simulations run in-memory (never touch :mod:`backend.storage` run dirs);
only the final attribution document is persisted by the caller.
"""

from __future__ import annotations

import math
import random
import statistics
from itertools import combinations, permutations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import catalog, storage, util
from .engine import make_engine
from .engine.base import Engine

# --------------------------------------------------------------------------- #
# Onset rule — single, explicit, documented criterion
# --------------------------------------------------------------------------- #
# 起效判定：在干预触发之后，对「基线曲线 - 干预曲线」的收益序列做
# SMOOTH_WINDOW 步后向滑动平均，从首个达到 |基线值|*REL_THRESHOLD（且基线
# 量级高于 EPS_REF）且方向有益的点开始，若连续 PERSIST_STEPS 步保持有益且
# 达阈值，则该窗口起点记为起效步；否则视为未起效（null）。
SMOOTH_WINDOW = 3
REL_THRESHOLD = 0.05
PERSIST_STEPS = 3
EPS_REF = 1e-9

# Exact Shapley is cheap up to 6 players (64 subsets per seed).
EXACT_MAX_PLAYERS = 6
MAX_PLAYERS = 8

# Verdict thresholds (on the primary target, relative benefit).
EFFECT_STRONG = 0.15      # |相对收益| ≥ 15% → 显著
EFFECT_WEAK = 0.03        # |相对收益| ≥ 3%  → 微弱
STABILITY_ROBUST_CV = 0.35
STABILITY_FRAGILE_CV = 0.75
STABILITY_PROB = 0.8      # ≥80% 的种子方向一致才算稳健
INTERACTION_STRONG = 0.10
INTERACTION_WEAK = 0.03


# --------------------------------------------------------------------------- #
# Counterfactual simulation
# --------------------------------------------------------------------------- #
def simulate(domain: str, model: str, config: Dict[str, Any], seed: int,
             steps: int, interventions: Sequence[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
    """Replay a run in-memory; return its per-step stats series.

    ``interventions`` carry ``at_step`` and are applied in
    (at_step, original order).  An intervention at step 0 acts on the initial
    state; an intervention at step ``t>0`` applies immediately after step
    ``t`` completes and before step ``t+1`` — exactly the run-manager
    semantics, so the replay mirrors a scheduled run (the full-bundle replay
    doubles as the fidelity check against the recorded series).
    """
    engine = make_engine(domain, model, config=config, seed=seed)
    ordered = sorted(enumerate(interventions),
                     key=lambda p: (int(p[1].get("at_step", 0)), p[0]))
    for _, itv in ordered:
        if int(itv.get("at_step", 0)) == 0:
            engine.apply_intervention(itv)
    series = [{"step": 0, **engine.stats()}]
    for _ in range(int(steps)):
        t = engine.step_count
        engine.step()
        for _, itv in ordered:
            if int(itv.get("at_step", 0)) == t + 1:
                engine.apply_intervention(itv)
        series.append({"step": engine.step_count, **engine.stats()})
    return series


# --------------------------------------------------------------------------- #
# Series features
# --------------------------------------------------------------------------- #
def _values(series: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(r[key]) for r in series if key in r and r[key] is not None]


def feature_value(series: List[Dict[str, Any]], key: str, feat: str
                  ) -> Optional[float]:
    """Aggregate one series column into a scalar feature."""
    vals = _values(series, key)
    if not vals:
        return None
    if feat == "max":
        return max(vals)
    if feat == "min":
        return min(vals)
    if feat == "mean":
        return sum(vals) / len(vals)
    if feat == "sum":
        return sum(vals)
    if feat == "last":
        return vals[-1]
    if feat == "argmax":
        steps = [int(r["step"]) for r in series if key in r and r[key] is not None]
        return float(steps[vals.index(max(vals))])
    raise ValueError(f"unknown feature: {feat}")


def benefit(base: Optional[float], treat: Optional[float], direction: str
            ) -> Optional[float]:
    """Signed intervention benefit: positive = good (for either direction)."""
    if base is None or treat is None:
        return None
    if direction == "higher":
        return treat - base
    return base - treat


def relative_benefit(ben: Optional[float], base: Optional[float]) -> Optional[float]:
    if ben is None or base is None or abs(base) < EPS_REF:
        return None
    return ben / abs(base)


# --------------------------------------------------------------------------- #
# Onset detection
# --------------------------------------------------------------------------- #
def _trailing_mean(vals: List[float], i: int, w: int) -> float:
    lo = max(0, i - w + 1)
    seg = vals[lo:i + 1]
    return sum(seg) / len(seg)


def detect_onset(base_curve: List[float], treat_curve: List[float],
                 steps: List[int], after: int, direction: str,
                 rel_threshold: float = REL_THRESHOLD,
                 smooth_window: int = SMOOTH_WINDOW,
                 persist: int = PERSIST_STEPS,
                 adverse: bool = False) -> Optional[int]:
    """First step at which the treated curve differs meaningfully from baseline.

    Criterion (documented in the UI and the attribution JSON): strictly after
    ``after``, the trailing-mean benefit must reach ``rel_threshold`` of the
    baseline magnitude and keep the same sign for ``persist`` consecutive
    steps; the first step of that run is the onset.  By default a *beneficial*
    deviation is sought (``adverse=False``); pass ``adverse=True`` to locate
    when a harmful deviation established instead — this lets the report
    distinguish "never mattered" from "backfired".  Returns ``None`` when the
    effect never establishes or the baseline level is essentially zero (no
    measurable gap to compare against).
    """
    n = min(len(base_curve), len(treat_curve), len(steps))
    if n == 0:
        return None
    for i in range(n):
        if steps[i] <= after:
            continue
        run = 0
        for j in range(i, n):
            mb = _trailing_mean(base_curve, j, smooth_window)
            mt = _trailing_mean(treat_curve, j, smooth_window)
            ben = benefit(mb, mt, direction)
            if ben is None or abs(mb) < EPS_REF:
                break
            if adverse:
                if ben >= 0:
                    break
            elif ben <= 0:
                break
            if abs(ben) < rel_threshold * abs(mb):
                break
            run += 1
            if run >= persist:
                return steps[i]
    return None


# --------------------------------------------------------------------------- #
# Intervention extraction
# --------------------------------------------------------------------------- #
def _itv_label(domain: str, itv_type: str, params: Dict[str, Any]) -> str:
    for spec in catalog.interventions(domain):
        if spec["type"] == itv_type:
            base = spec["label"]
            break
    else:
        base = itv_type
    if params:
        base += "(" + ",".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items())) + ")"
    return base


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_interventions(meta: Dict[str, Any], events: List[Dict[str, Any]]
                        ) -> Tuple[List[Dict[str, Any]], str]:
    """Extract the attribution intervention list and the analysis mode.

    Prefers interventions actually applied (from the event log); falls back to
    the scene's scheduled interventions when none have fired yet (prospective /
    pre-run analysis).  Repeated actions of the same type stay separate
    players so each firing is attributed independently.
    """
    applied = [e for e in events
               if e.get("type") and (e.get("result") or {}).get("applied", True)]
    if applied:
        out: List[Dict[str, Any]] = []
        seen: Dict[Tuple[Any, Any], int] = {}
        for e in applied:
            params = e.get("params", {}) or {}
            key = (e["step"], e["type"])
            seen[key] = seen.get(key, 0) + 1
            out.append({
                "type": e["type"],
                "params": params,
                "at_step": int(e["step"]),
                "scheduled": bool(e.get("scheduled", False)),
                "label": _itv_label(meta["domain"], e["type"], params),
                "dup": seen[key],
            })
        for itv in out:
            if itv["dup"] > 1:
                itv["label"] += f"#{itv['dup']}"
        return out, "applied"

    scheduled = [dict(i) for i in meta.get("interventions", []) if i.get("type")]
    for i in scheduled:
        i["scheduled"] = True
        i["params"] = i.get("params", {})
        i["at_step"] = int(i.get("at_step", 0))
        i["label"] = i.get("label") or _itv_label(
            meta["domain"], i["type"], i.get("params", {}))
    return scheduled, "scheduled"


# --------------------------------------------------------------------------- #
# Stats aggregation helpers
# --------------------------------------------------------------------------- #
def _dist(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    mean = sum(vals) / len(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    rel = sd / abs(mean) if abs(mean) > EPS_REF else None
    return {
        "n": len(vals),
        "mean": _r(mean), "sd": _r(sd),
        "cv": _r(rel) if rel is not None else None,
        "min": _r(min(vals)), "max": _r(max(vals)),
        "p10": _r(_quantile(vals, 0.10)),
        "p90": _r(_quantile(vals, 0.90)),
        "median": _r(statistics.median(vals)),
    }


def _quantile(vals: List[float], q: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _r(v: Optional[float], nd: int = 4) -> Optional[float]:
    if v is None:
        return None
    av = abs(v)
    if av != 0 and av < 1e-4:
        return round(v, 7)
    return round(v, nd)


def _rank(x: List[float]) -> List[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va < EPS_REF or vb < EPS_REF:
        return None
    return cov / math.sqrt(va * vb)


def _strength(rel_mean: Optional[float]) -> str:
    if rel_mean is None:
        return "insignificant"
    a = abs(rel_mean)
    if a >= EFFECT_STRONG:
        return "strong"
    if a >= EFFECT_WEAK:
        return "weak"
    return "insignificant"


def _stability(cv: Optional[float], prob: float) -> str:
    if cv is None:
        return "uncertain"
    if cv <= STABILITY_ROBUST_CV and prob >= STABILITY_PROB:
        return "robust"
    if cv >= STABILITY_FRAGILE_CV or prob < 0.5:
        return "fragile"
    return "sensitive"


def _interaction_label(idx: Optional[float], rel: Optional[float]) -> str:
    if idx is None or rel is None:
        return "unknown"
    a = abs(rel)
    if a < INTERACTION_WEAK:
        return "additive"
    word = "synergy" if idx > 0 else "overlap"
    if a >= INTERACTION_STRONG:
        return "strong_" + word
    return "weak_" + word


# --------------------------------------------------------------------------- #
# Mask bookkeeping
# --------------------------------------------------------------------------- #
def _masks_exact(k: int) -> List[int]:
    return list(range(1 << k))


def _plan(k: int, perm_count: int, rng: random.Random
          ) -> Tuple[List[int], List[Tuple[int, ...]], str]:
    """Choose which subset masks to simulate and the MC permutations.

    Returns (masks, permutations, method).  Pair masks are always included so
    pair-wise interactions can be reported even under Monte Carlo.
    """
    if k <= EXACT_MAX_PLAYERS:
        return _masks_exact(k), [], "exact"
    all_perms = list(permutations(range(k)))
    rng.shuffle(all_perms)
    # Antithetic pairs: every sampled permutation is paired with its reverse,
    # which sharply cuts the Shapley variance for position-dependent effects.
    perms: List[Tuple[int, ...]] = []
    for p in all_perms:
        perms.append(p)
        perms.append(tuple(reversed(p)))
        if len(perms) >= perm_count:
            break
    need = {0, (1 << k) - 1}
    need.update(1 << i for i in range(k))
    for p in perms:
        m = 0
        for i in p:
            need.add(m)
            m |= 1 << i
        need.add(m)
    for i, j in combinations(range(k), 2):
        need.add((1 << i) | (1 << j))
    return sorted(need), perms, "monte_carlo"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def attribute_run(run_id: str, seeds: Optional[Sequence[int]] = None,
                  steps: Optional[int] = None,
                  target_keys: Optional[Sequence[str]] = None,
                  progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                  is_aborted: Optional[Callable[[], bool]] = None
                  ) -> Dict[str, Any]:
    """Compute the full intervention-attribution document for a finished run.

    ``secrets`` defaults to the run's own seed + deterministic offsets (5
    seeds).  ``steps`` defaults to the run's current length.  Heavy work runs
    in-memory; this function only *returns* the document (caller persists).
    """
    meta = storage.load_run_meta(run_id)
    if meta is None:
        raise KeyError(f"run not found: {run_id}")
    events = storage.load_events(run_id)
    actual_series = storage.load_series(run_id)

    domain = meta["domain"]
    model = meta["model"]
    config = dict(meta["config"])
    actual_steps = max(0, len(actual_series) - 1)
    n_steps = int(steps) if steps is not None else actual_steps
    if n_steps <= 0:
        raise ValueError("运行尚无任何步进结果，无法归因（请先运行至少 1 步）")

    interventions, mode = build_interventions(meta, events)
    if not interventions:
        raise ValueError("该运行没有任何干预措施（手动施加或定时触发均可），无法归因")
    k = len(interventions)
    if k > MAX_PLAYERS:
        raise ValueError(f"最多支持 {MAX_PLAYERS} 个干预的归因，当前为 {k} 个")

    if seeds is None:
        base_seed = int(meta.get("seed", 0))
        seed_list = [base_seed + 2654435761 * i for i in range(5)]
    else:
        seed_list = [int(s) for s in seeds]

    target_specs = catalog.attribution_targets(domain)
    primary_key = catalog.attribution_primary(domain)
    if target_keys:
        wanted = set(target_keys)
        target_specs = [t for t in target_specs if t["key"] in wanted]
    if not target_specs:
        raise ValueError(f"领域 {domain} 未配置归因指标")
    aux_specs = catalog.attribution_aux(domain)

    plan_rng = random.Random(92821)
    # More permutations when players are few-ish, fewer when very many.
    perm_budget = {7: 300, 8: 160}.get(k, 400)
    masks, perms, method = _plan(k, perm_budget, plan_rng)
    full_mask = (1 << k) - 1

    # Chronological prefix masks are needed for the sequence-onset analysis;
    # the MC permutation sample is not guaranteed to visit them.
    chrono = sorted(range(k),
                    key=lambda i: (int(interventions[i]["at_step"]), i))
    seq_masks = set()
    m = 0
    for i in chrono:
        m |= 1 << i
        seq_masks.add(m)
    extra = seq_masks - set(masks)
    if extra:
        masks = sorted(set(masks) | extra)

    notes: List[str] = []
    if mode == "scheduled":
        notes.append("运行中尚无已施加的干预事件，已按场景的定时干预做前瞻式归因"
                     "（结论为模型预测，非事后观测）。")
    if n_steps < actual_steps:
        notes.append(f"归因仅使用前 {n_steps} 步（实际运行已有 {actual_steps} 步）。")
    if method == "monte_carlo":
        notes.append(f"干预数 {k}>{EXACT_MAX_PLAYERS}，Shapley 值采用 {len(perms)} 条"
                     "随机排列的蒙特卡洛估计（交互项为空集锚定近似）。")

    # ------------------------------------------------------------------ #
    # Per-seed counterfactual sweep
    # ------------------------------------------------------------------ #
    per_seed: List[Dict[str, Any]] = []
    total_sims = len(seed_list) * len(masks)
    done_sims = 0

    def report_progress(phase: str) -> None:
        if progress:
            progress({"phase": phase, "done": done_sims, "total": total_sims,
                      "seeds": len(per_seed), "total_seeds": len(seed_list)})

    for seed in seed_list:
        if is_aborted and is_aborted():
            raise RuntimeError("归因任务已取消")

        # Cache per subset: scalar features for every target plus curves.
        feats: Dict[int, Dict[str, Optional[float]]] = {}
        aux_feats: Dict[int, Dict[str, Optional[float]]] = {}
        curves: Dict[int, Dict[str, List[float]]] = {}

        for mask in masks:
            chosen = [interventions[i] for i in range(k) if mask & (1 << i)]
            series = simulate(domain, model, config, seed, n_steps, chosen)
            feats[mask] = {
                t["key"]: feature_value(series, t["series"], t["feature"])
                for t in target_specs}
            aux_feats[mask] = {
                a["key"]: feature_value(series, a["series"], a["feature"])
                for a in aux_specs}
            curves[mask] = {t["key"]: _values(series, t["series"])
                            for t in target_specs}
            done_sims += 1
        report_progress("counterfactuals")

        seed_rec: Dict[str, Any] = {"seed": seed, "targets": {},
                                   "fidelity": {}, "interventions": []}

        for t in target_specs:
            tk, direction = t["key"], t["direction"]
            b, f = feats[0][tk], feats[full_mask][tk]
            total = benefit(b, f, direction)
            rel = relative_benefit(total, b)

            # Exact Shapley via incremental value over every permutation;
            # MC via the sampled permutation prefixes (precomputed in feats).
            shap = [0.0] * k
            if method == "exact":
                for i in range(k):
                    acc = 0.0
                    for mask in masks:
                        if mask & (1 << i):
                            continue
                        s = mask.bit_count()
                        w = (math.factorial(s) * math.factorial(k - s - 1)
                             / math.factorial(k))
                        # Marginal V(S∪i) − V(S); for "lower is better"
                        # benefit() keeps this base-first order.
                        v_with = benefit(feats[mask][tk],
                                         feats[mask | (1 << i)][tk], direction)
                        acc += w * (v_with or 0.0)
                    shap[i] = acc
            else:
                for p in perms:
                    mask = 0
                    for i in p:
                        # φ_i = E[marginal of i in a random ordering];
                        # the marginal itself is V(S∪i) − V(S).  Summing the
                        # *differences of successive marginals would be wrong
                        # (and no longer telescopes to the total benefit).
                        v_now = benefit(feats[mask][tk],
                                        feats[mask | (1 << i)][tk], direction) or 0.0
                        shap[i] += v_now
                        mask |= 1 << i
                shap = [v / len(perms) for v in shap]

            # Pairwise interaction.
            interactions: Dict[str, Dict[str, Any]] = {}
            for i, j in combinations(range(k), 2):
                vi = benefit(b, feats[1 << i][tk], direction) or 0.0
                vj = benefit(b, feats[1 << j][tk], direction) or 0.0
                vij = benefit(b, feats[(1 << i) | (1 << j)][tk], direction) or 0.0
                if method == "exact":
                    # Shapley interaction index (all subsets of the other
                    # players): how i's marginal changes when j is present.
                    idx_val = 0.0
                    rest = [x for x in range(k) if x not in (i, j)]
                    for q in range(len(rest) + 1):
                        for sub in combinations(rest, q):
                            m = 0
                            for x in sub:
                                m |= 1 << x
                            # Δ_i with j absent vs present:
                            #   v(S∪i)−v(S)  vs  v(S∪{i,j})−v(S∪j)
                            v00 = benefit(feats[m][tk],
                                          feats[m | (1 << i)][tk], direction) or 0.0
                            v01 = benefit(feats[m | (1 << j)][tk],
                                          feats[m | (1 << i) | (1 << j)][tk],
                                          direction) or 0.0
                            w = (math.factorial(q) * math.factorial(k - q - 2)
                                 / math.factorial(k - 1))
                            idx_val += w * (v01 - v00)
                else:
                    idx_val = vij - vi - vj  # empty-set anchored synergy
                interactions[f"{i},{j}"] = {
                    "index": idx_val,
                    "relative": (idx_val / abs(b)) if abs(b or 0.0) > EPS_REF else None,
                    "solo_i": vi, "solo_j": vj, "pair": vij,
                }

            # Solo benefit (intervention alone vs baseline).
            solo = [benefit(b, feats[1 << i][tk], direction) for i in range(k)]

            seed_rec["targets"][tk] = {
                "baseline": b, "bundle": f,
                "total_benefit": total,
                "relative_benefit": rel,
                "shapley": shap, "solo": solo,
                "interactions": interactions,
            }

        # Auxiliary feature baseline → bundle delta (reported, not attributed).
        seed_rec["aux"] = {
            a["key"]: {"baseline": aux_feats[0][a["key"]],
                       "bundle": aux_feats[full_mask][a["key"]]}
            for a in aux_specs}

        # Onsets on the primary target's series.
        pkey = primary_key
        pspec = next(t for t in target_specs if t["key"] == pkey)
        series_col = pspec["series"]
        pdir = pspec["direction"]
        step_axis = list(range(n_steps + 1))

        solo_onsets: List[Optional[int]] = []
        solo_lags: List[Optional[int]] = []
        solo_adv_onsets: List[Optional[int]] = []
        solo_adv_lags: List[Optional[int]] = []
        for i in range(k):
            at = int(interventions[i]["at_step"])
            on = detect_onset(curves[0][pkey], curves[1 << i][pkey],
                              step_axis, at, pdir)
            on_bad = detect_onset(curves[0][pkey], curves[1 << i][pkey],
                                  step_axis, at, pdir, adverse=True)
            solo_onsets.append(on)
            solo_lags.append(None if on is None else on - at)
            solo_adv_onsets.append(on_bad)
            solo_adv_lags.append(None if on_bad is None else on_bad - at)

        # Marginal onset along the actual chronological order: the curve gains
        # one more intervention at each firing, so i's marginal onset is
        # measured against the bundle of the interventions fired before it.
        order = chrono
        seq_onsets: List[Optional[int]] = [None] * k
        seq_lags: List[Optional[int]] = [None] * k
        seq_adv_lags: List[Optional[int]] = [None] * k
        prev_mask = 0
        for pos, i in enumerate(order):
            at = int(interventions[i]["at_step"])
            new_mask = prev_mask | (1 << i)
            compare_after = at if pos == 0 else max(
                at, int(interventions[order[pos - 1]]["at_step"]))
            on = detect_onset(curves[prev_mask][pkey], curves[new_mask][pkey],
                              step_axis, compare_after, pdir)
            on_bad = detect_onset(curves[prev_mask][pkey], curves[new_mask][pkey],
                                  step_axis, compare_after, pdir, adverse=True)
            seq_onsets[i] = on
            seq_lags[i] = None if on is None else on - at
            seq_adv_lags[i] = None if on_bad is None else on_bad - at
            prev_mask = new_mask

        first_at = min(int(x["at_step"]) for x in interventions)
        bundle_onset = detect_onset(curves[0][pkey], curves[full_mask][pkey],
                                    step_axis, first_at, pdir)
        bundle_adv_onset = detect_onset(curves[0][pkey],
                                        curves[full_mask][pkey],
                                        step_axis, first_at, pdir,
                                        adverse=True)
        seed_rec["onsets"] = {
            "solo_onset": solo_onsets, "solo_lag": solo_lags,
            "solo_adverse_onset": solo_adv_onsets,
            "solo_adverse_lag": solo_adv_lags,
            "sequence_onset": seq_onsets, "sequence_lag": seq_lags,
            "sequence_adverse_lag": seq_adv_lags,
            "bundle_onset": bundle_onset,
            "bundle_adverse_onset": bundle_adv_onset,
        }

        # Curves retained only for the primary target charts.
        seed_rec["curves"] = {
            "steps": step_axis,
            "baseline": curves[0][pkey],
            "bundle": curves[full_mask][pkey],
            "solo": [curves[1 << i][pkey] for i in range(k)],
        }

        # Fidelity: full-bundle replay vs the recorded run on seed #0.
        if seed == seed_list[0] and actual_series:
            fid = {}
            for t in target_specs:
                actual = feature_value(actual_series[:n_steps + 1],
                                       t["series"], t["feature"])
                replay = feats[full_mask][t["key"]]
                if actual is not None and replay is not None and abs(actual) > EPS_REF:
                    fid[t["key"]] = abs(actual - replay) / abs(actual)
                else:
                    fid[t["key"]] = 0.0
            seed_rec["fidelity"] = fid
        per_seed.append(seed_rec)
        report_progress("analysis")

    if is_aborted and is_aborted():
        raise RuntimeError("归因任务已取消")

    # ------------------------------------------------------------------ #
    # Cross-seed aggregation
    # ------------------------------------------------------------------ #
    results: Dict[str, Any] = {}
    for t in target_specs:
        tk = t["key"]
        rs = [s["targets"][tk] for s in per_seed]

        base_dist = _dist([r["baseline"] for r in rs])
        bundle_dist = _dist([r["bundle"] for r in rs])
        total_dist = _dist([r["total_benefit"] for r in rs])
        rel_vals = [r["relative_benefit"] for r in rs
                    if r["relative_benefit"] is not None]
        prob_good = (sum(1 for v in (r["total_benefit"] for r in rs)
                         if v is not None and v > 0) / len(rs))

        per_itv: List[Dict[str, Any]] = []
        for i in range(k):
            shap_vals = [r["shapley"][i] for r in rs]
            solo_vals = [r["solo"][i] for r in rs]
            shap_dist = _dist(shap_vals)
            solo_dist = _dist(solo_vals)
            base_mean = base_dist.get("mean")
            shap_rel = (shap_dist["mean"] / abs(base_mean)
                        if base_mean and abs(base_mean) > EPS_REF else None)
            solo_rel = (solo_dist["mean"] / abs(base_mean)
                        if base_mean and abs(base_mean) > EPS_REF else None)
            p_good = sum(1 for v in shap_vals if v is not None and v > 0) / len(rs)

            ranks_per_seed = []
            for r in rs:
                vals = [(x if x is not None else 0.0) for x in r["shapley"]]
                order = sorted(range(k), key=lambda x: vals[x], reverse=True)
                rank = [0] * k
                for pos, idx in enumerate(order):
                    rank[idx] = pos + 1
                ranks_per_seed.append(rank[i])
            at = int(interventions[i]["at_step"])
            sl = [s["onsets"]["solo_lag"][i] for s in per_seed]
            so = [s["onsets"]["solo_onset"][i] for s in per_seed]
            sla = [s["onsets"]["solo_adverse_lag"][i] for s in per_seed]
            ql = [s["onsets"]["sequence_lag"][i] for s in per_seed]
            qo = [s["onsets"]["sequence_onset"][i] for s in per_seed]
            qla = [s["onsets"]["sequence_adverse_lag"][i] for s in per_seed]

            per_itv.append({
                "index": i,
                "type": interventions[i]["type"],
                "label": interventions[i]["label"],
                "at_step": at,
                "scheduled": interventions[i].get("scheduled", False),
                "shapley": shap_dist,
                "shapley_relative": _r(shap_rel),
                "solo": solo_dist,
                "solo_relative": _r(solo_rel),
                "prob_beneficial": _r(p_good, 3),
                "rank_mean": _r(sum(ranks_per_seed) / len(ranks_per_seed), 2),
                "rank_distribution": [ranks_per_seed.count(rk) for rk in
                                      range(1, k + 1)],
                "solo_lag_median": _median_or_none(sl),
                "solo_lag_null_fraction": _null_fraction(sl),
                "solo_onset_median": _median_or_none(so),
                "solo_adverse_lag_median": _median_or_none(sla),
                "solo_adverse_null_fraction": _null_fraction(sla),
                "sequence_lag_median": _median_or_none(ql),
                "sequence_lag_null_fraction": _null_fraction(ql),
                "sequence_onset_median": _median_or_none(qo),
                "sequence_adverse_lag_median": _median_or_none(qla),
                "strength": _strength(shap_rel),
                "stability": _stability(shap_dist.get("cv"), p_good),
            })

        # Ranking: Shapley mean on the primary target, beneficial first.
        order_idx = sorted(range(k),
                           key=lambda i: (per_itv[i]["shapley"].get("mean")
                                          if per_itv[i]["shapley"].get("mean") is not None
                                          else 0.0),
                           reverse=True)
        for pos, i in enumerate(order_idx):
            per_itv[i]["rank"] = pos + 1

        # Pair interactions aggregated across seeds.
        pairs: List[Dict[str, Any]] = []
        if k >= 2:
            for i, j in combinations(range(k), 2):
                key = f"{i},{j}"
                vals = [r["interactions"][key]["index"] for r in rs]
                rels = [r["interactions"][key]["relative"] for r in rs
                        if r["interactions"][key]["relative"] is not None]
                dist = _dist(vals)
                rel_mean = (sum(rels) / len(rels)) if rels else None
                base_mean = base_dist.get("mean")
                pairs.append({
                    "i": i, "j": j,
                    "label_i": interventions[i]["label"],
                    "label_j": interventions[j]["label"],
                    "index": dist,
                    "relative_mean": _r(rel_mean),
                    "classification": _interaction_label(
                        dist.get("mean"), rel_mean),
                    "prob_synergy": _r(
                        sum(1 for v in vals if v > EPS_REF) / len(vals), 3),
                })
            pairs.sort(key=lambda p: -(abs(p["index"].get("mean") or 0.0)))

        # Rank concordance: mean pairwise Spearman ρ across seeds.
        rhos = []
        for a, b in combinations(range(len(rs)), 2):
            rho = _spearman(rs[a]["shapley"], rs[b]["shapley"])
            if rho is not None:
                rhos.append(rho)
        rank_concordance = sum(rhos) / len(rhos) if rhos else None

        # Shapley additivity check: Σ φ vs total benefit (exact must be ~0).
        resid = [sum(r["shapley"]) - (r["total_benefit"] or 0.0) for r in rs]
        decomp_residual = _dist(resid)

        results[tk] = {
            "target": t,
            "baseline": base_dist,
            "bundle": bundle_dist,
            "total_benefit": total_dist,
            "relative_benefit": _r(sum(rel_vals) / len(rel_vals)) if rel_vals else None,
            "relative_benefit_sd": _r(statistics.pstdev(rel_vals)) if len(rel_vals) > 1 else 0.0,
            "prob_beneficial": _r(prob_good, 3),
            "interventions": per_itv,
            "interactions": pairs,
            "rank_concordance": _r(rank_concordance, 3),
            "decomposition_residual": decomp_residual,
        }

    # Auxiliary deltas.
    aux_out = {}
    for a in aux_specs:
        ak = a["key"]
        bd = _dist([s["aux"][ak]["baseline"] for s in per_seed])
        fd = _dist([s["aux"][ak]["bundle"] for s in per_seed])
        delta = _dist([(s["aux"][ak]["bundle"] or 0.0) - (s["aux"][ak]["baseline"] or 0.0)
                       for s in per_seed])
        aux_out[ak] = {"spec": a, "baseline": bd, "bundle": fd, "delta": delta}

    # Bundle onset (primary target) aggregated.
    bundle_lags = []
    bundle_onsets = []
    bundle_adv_lags = []
    first_at = min(int(x["at_step"]) for x in interventions)
    for s, seed_rec in zip(seed_list, per_seed):
        on = seed_rec["onsets"]["bundle_onset"]
        bundle_onsets.append(on)
        if on is not None:
            bundle_lags.append(on - first_at)
        on_bad = seed_rec["onsets"]["bundle_adverse_onset"]
        if on_bad is not None:
            bundle_adv_lags.append(on_bad - first_at)

    fid0 = per_seed[0].get("fidelity", {})
    doc = {
        "run_id": run_id,
        "name": meta["name"],
        "domain": domain,
        "model": model,
        "generated_at": util.now_iso(),
        "mode": mode,
        "steps": n_steps,
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "n_interventions": k,
        "method": method,
        "n_subsets": len(masks),
        "n_permutations": len(perms),
        "n_simulations": total_sims,
        "primary_target": primary_key,
        "interventions": [
            {"index": i, "type": x["type"], "label": x["label"],
             "at_step": int(x["at_step"]), "scheduled": x.get("scheduled", False)}
            for i, x in enumerate(interventions)],
        "results": results,
        "aux": aux_out,
        "bundle_onset_median": _median_or_none(bundle_onsets),
        "bundle_lag_median": _median_or_none(bundle_lags),
        "bundle_onset_null_fraction": _null_fraction(bundle_onsets),
        "bundle_adverse_lag_median": _median_or_none(bundle_adv_lags),
        "fidelity": {k: _r(v, 4) for k, v in fid0.items()},
        "onset_rule": {
            "description": "在对照起点之后，收益（基线−干预，方向归一为越大越好）的 "
                           f"{SMOOTH_WINDOW} 步滑动均值首次达到基线量级的 "
                           f"{REL_THRESHOLD:.0%}，且连续 {PERSIST_STEPS} 步保持"
                           "同号且达阈值，该窗口起点记为起效步；始终未达到则为 null。",
            "smooth_window": SMOOTH_WINDOW,
            "relative_threshold": REL_THRESHOLD,
            "persist_steps": PERSIST_STEPS,
            "beneficial": "有益起效：偏离方向与目标期望一致（如峰值下降）。",
            "adverse": "有害起效：偏离方向与目标期望相反（如峰值反而上升）；"
                       "有益起效为 null 且有害起效非 null 表示干预起了反作用。",
            "solo": "只施加该干预 vs 无干预基线（回答“它单独何时起效”）",
            "sequence": "按实际触发顺序，加入该干预前后两条曲线对比"
                        "（回答“在叠加现场中它何时带来新增效果”）",
        },
        "verdict_thresholds": {
            "strong_relative": EFFECT_STRONG, "weak_relative": EFFECT_WEAK,
            "robust_cv": STABILITY_ROBUST_CV, "fragile_cv": STABILITY_FRAGILE_CV,
            "robust_prob": STABILITY_PROB,
        },
        "curves": [
            {"seed": s["seed"], **s["curves"]} for s in per_seed
        ],
        "notes": notes,
    }
    return doc


def _median_or_none(vals: Iterable[Optional[int]]) -> Optional[float]:
    real = [v for v in vals if v is not None]
    if not real:
        return None
    return _r(float(statistics.median(real)), 2)


def _null_fraction(vals: Iterable[Optional[int]]) -> float:
    vs = list(vals)
    if not vs:
        return 1.0
    return round(sum(1 for v in vs if v is None) / len(vs), 3)
