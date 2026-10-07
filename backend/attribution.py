"""Intervention attribution via counterfactual re-simulation.

Given a finished run with one or more applied interventions, this module
quantifies *how much each intervention actually caused* the observed change in
the domain's key metrics.  A naive "before vs after" comparison cannot separate
two interventions that overlap in time, so the analysis is built on three
explicit pieces:

1. **Counterfactual replays.** The run is re-simulated from its saved config
   and seed for every subset ("coalition") of its interventions.  Interventions
   not in the subset simply never fire, everything else is byte-for-byte the
   same (common random numbers), so a paired comparison isolates causation
   instead of mixing it with randomness.

2. **Shapley-value decomposition.** Intervention *i*'s contribution is its
   average marginal effect over every possible ordering of the interventions::

       φ_i = Σ_{S⊆N\i}  |S|!(n-1-|S|)! / n!  · ( B(S∪{i}) − B(S) )

   where ``B(S)`` is the benefit (signed so positive always means "better") of
   the coalition ``S`` relative to a no-intervention baseline.  Shapley values
   are fair (they sum exactly to the total joint benefit), order-independent and
   work whether interventions fire simultaneously or sequentially.

3. **Pairwise interaction index.** The second-order Shapley interaction
   ``B(S∪ij) − B(S∪i) − B(S∪j) + B(S)`` (averaged over all contexts ``S``) is
   positive when two interventions amplify each other and negative when they
   cancel / overlap.

Stability across stochastic runs is assessed by repeating every counterfactual
with several seeds (CRN paired): each contribution carries a mean, standard
deviation, "beneficial rate" (fraction of seeds with the right sign) and a rank,
so conclusions come with evidence instead of two point estimates.

Cost control: with ``n ≤ EXACT_LIMIT`` interventions all ``2**n`` coalitions are
enumerated exactly; beyond that a fixed number of random permutations is sampled
(singletons, pairs and leave-one-out coalitions are always included so the
onset, interaction and LOO columns stay exact).
"""

from __future__ import annotations

import math
import random
import statistics
import threading
from itertools import combinations
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from . import catalog, storage, util
from .engine import make_engine

# --------------------------------------------------------------------------- #
# Analysis constants (part of the documented "onset rule")
# --------------------------------------------------------------------------- #
EXACT_LIMIT = 5            # exact coalition enumeration up to this many itvs
SAMPLED_PERMUTATIONS = 40  # permutation draws when sampling is required
MAX_INSTANCES = 12
DEFAULT_REPLICATES = 6
ONSET_WINDOW = 5           # trailing moving-average window (steps)
ONSET_HOLD = 3             # consecutive above-threshold steps required
ONSET_PRE_WINDOW = 20      # baseline-noise estimation window before the itv
ONSET_NOISE_Z = 2.0        # threshold: z × baseline step-to-step σ
ONSET_REL_FLOOR = 0.01     # …plus at least 1 % of the pre-intervention level
TRAJ_POINTS = 160          # downsampled chart points per trajectory

# In-memory job registry for background attribution runs.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Metric used for the time-series chart / replay validation, per domain/model.
CHART_METRIC = {
    "epidemic/ca": "infected",
    "epidemic/abm": "infected",
    "traffic/ca": "mean_speed",
    "traffic/abm": "mean_speed",
    "ecology/ca": "rabbits",
    "ecology/abm": "boids",
}


# --------------------------------------------------------------------------- #
# Scalar summaries derived from a replay's time series
# --------------------------------------------------------------------------- #
def _series_summaries(domain: str, model: str,
                      series: List[Dict[str, Any]]) -> Dict[str, float]:
    """Collapse a full series into the scalar headline quantities."""
    def col(key: str) -> List[float]:
        return [float(r[key]) for r in series if key in r]

    out: Dict[str, float] = {}
    if domain == "epidemic":
        infected = col("infected")
        new = col("new_infections")
        if infected:
            out["peak_infected"] = max(infected)
            out["auc_infected"] = float(sum(infected))
            out["end_infected"] = infected[-1]
            out["peak_step"] = float(infected.index(max(infected)))
        if new:
            initial = float(series[0].get("infected", 0))
            out["total_infected"] = initial + float(sum(new))
    elif domain == "traffic":
        speed, stopped, flow = col("mean_speed"), col("stopped"), col("flow")
        if speed:
            out["mean_speed"] = sum(speed) / len(speed)
        if stopped:
            out["stopped"] = sum(stopped) / len(stopped)
        if flow:
            out["flow"] = float(sum(flow))
    elif domain == "ecology" and model == "ca":
        rabbits, foxes, grass = col("rabbits"), col("foxes"), col("grass_coverage")
        if rabbits:
            out["rabbits_mean"] = sum(rabbits) / len(rabbits)
            out["rabbits_min"] = min(rabbits)
        if foxes:
            out["foxes_mean"] = sum(foxes) / len(foxes)
        if grass:
            out["grass_coverage"] = grass[-1]
    elif domain == "ecology" and model == "abm":
        boids, predators, eaten = col("boids"), col("predators"), col("eaten")
        if boids:
            out["boids_mean"] = sum(boids) / len(boids)
            out["boids_min"] = min(boids)
        if predators:
            out["predators_mean"] = sum(predators) / len(predators)
        if eaten:
            out["total_eaten"] = max(eaten)
    return out


def _benefit(direction: str, v_empty: float, v_coalition: float) -> float:
    """Signed benefit of a coalition value vs the no-intervention value."""
    if direction == "down":
        return v_empty - v_coalition
    if direction == "up":
        return v_coalition - v_empty
    return float("nan")  # info / neutral metrics have no benefit direction


# --------------------------------------------------------------------------- #
# Counterfactual replay
# --------------------------------------------------------------------------- #
def _replay(meta: Dict[str, Any], seed: int, horizon: int,
            instances: Sequence[Dict[str, Any]], included: FrozenSet[int]
            ) -> List[Dict[str, Any]]:
    """Re-simulate the run firing only interventions whose index is ``included``.

    The apply timing mirrors :meth:`RunManager._apply_due`: an intervention due
    at step ``t`` is applied to the engine while ``step_count == t`` and
    therefore first affects the transition into step ``t+1``.
    """
    engine = make_engine(meta["domain"], meta["model"],
                         config=meta["config"], seed=seed)
    series: List[Dict[str, Any]] = [{"step": 0, **engine.stats()}]
    pending = sorted((idx for idx in included),
                     key=lambda i: (instances[i]["step"], i))
    cursor = 0
    for _ in range(horizon):
        at = engine.step_count
        while cursor < len(pending) and instances[pending[cursor]]["step"] <= at:
            idx = pending[cursor]
            inst = instances[idx]
            engine.apply_intervention(
                {"type": inst["type"], "params": inst["params"]})
            cursor += 1
        engine.step()
        series.append({"step": engine.step_count, **engine.stats()})
    return series


def _plan_coalitions(n: int) -> Tuple[List[FrozenSet[int]], str,
                                      List[Tuple[int, ...]]]:
    """Return ``(coalitions, method, permutations)`` to simulate.

    Exact mode enumerates all ``2**n`` subsets; sampled mode always includes
    the empty/full set, singletons, pairs and leave-one-out coalitions (so the
    onset, pairwise-synergy and LOO columns remain exact) and tops them up with
    the prefix coalitions of a fixed set of random permutations for the
    Shapley estimator.
    """
    full = frozenset(range(n))
    if n <= EXACT_LIMIT:
        coalitions: List[FrozenSet[int]] = [frozenset()]
        for size in range(1, n + 1):
            for comb in combinations(range(n), size):
                coalitions.append(frozenset(comb))
        return coalitions, "exact", []

    rng = random.Random(0xC0FFEE ^ n)
    planned = {frozenset(), full}
    planned.update(frozenset({i}) for i in range(n))
    planned.update(frozenset(p) for p in combinations(range(n), 2))
    planned.update(full - {i} for i in range(n))
    perms: List[Tuple[int, ...]] = []
    for _ in range(SAMPLED_PERMUTATIONS):
        perm = list(range(n))
        rng.shuffle(perm)
        perms.append(tuple(perm))
        prefix: set = set()
        for i in perms[-1]:
            prefix.add(i)
            planned.add(frozenset(prefix))
    ordered = sorted(planned, key=lambda s: (len(s), sorted(s)))
    return ordered, "sampled", perms


def _derive_seed(base_seed: int, replicate: int) -> int:
    """Deterministic, well-spread CRN seeds; replicate 0 == the run's seed."""
    if replicate == 0:
        return base_seed
    return (base_seed * 1_000_003 + 7919 * replicate + 17) % (2 ** 31 - 1)


# --------------------------------------------------------------------------- #
# Onset ("when does it start working") — explicit criterion
# --------------------------------------------------------------------------- #
def _onset_for(base: Sequence[float], treated: Sequence[float],
               direction: str, at_step: int) -> Dict[str, Any]:
    """Detect the first step an intervention produces a detectable effect.

    Criterion (identical for every metric/domain):

    1. pair the treated and intervention-free series at the same CRN seed and
       take the direction-aligned gap ``g_t`` (positive = improvement);
    2. smooth ``g`` with a causal trailing mean of :data:`ONSET_WINDOW` steps;
    3. the detection threshold is ``max(z·σ, floor·|m̄|)`` where ``σ`` is the
       standard deviation of the *baseline series'* step-to-step differences in
       the :data:`ONSET_PRE_WINDOW` steps before firing and ``m̄`` is that
       window's mean — i.e. the effect must beat both natural fluctuation (2σ)
       and a 1 % level floor;
    4. the first step from which the smoothed gap stays above the threshold for
       :data:`ONSET_HOLD` consecutive steps is the onset step; if this never
       happens within the horizon the intervention is reported as "not detected
       within the observation window".
    """
    horizon = min(len(base), len(treated)) - 1
    lo = max(1, at_step - ONSET_PRE_WINDOW)
    pre = [float(base[t]) for t in range(lo, max(lo + 1, at_step))]
    if len(pre) >= 2:
        sigma = statistics.pstdev(pre[k] - pre[k - 1] for k in range(1, len(pre)))
        level = abs(sum(pre) / len(pre))
    else:
        sigma, level = 0.0, abs(float(base[min(at_step, len(base) - 1)]))
    threshold = max(ONSET_NOISE_Z * sigma, ONSET_REL_FLOOR * max(level, 1e-9))

    raw: List[float] = []
    for t in range(horizon + 1):
        b, a = float(base[t]), float(treated[t])
        raw.append(b - a if direction == "down" else a - b)

    smoothed: List[float] = []
    for t in range(horizon + 1):
        w = raw[max(0, t - ONSET_WINDOW + 1):t + 1]
        smoothed.append(sum(w) / len(w))

    onset: Optional[int] = None
    run_above = 0
    for t in range(at_step, horizon + 1):
        if smoothed[t] > threshold:
            run_above += 1
            if run_above >= ONSET_HOLD and onset is None:
                onset = t - ONSET_HOLD + 1
        else:
            run_above = 0

    above_after = [smoothed[t] > threshold for t in range(at_step, horizon + 1)]
    sustained = bool(above_after) and sum(above_after) / len(above_after) >= 0.6
    return {
        "detected": onset is not None,
        "onset_step": onset,
        "lag": (onset - at_step) if onset is not None else None,
        "threshold": round(threshold, 4),
        "sustained": sustained,
        "at_step": at_step,
    }


# --------------------------------------------------------------------------- #
# Shapley values and interactions
# --------------------------------------------------------------------------- #
def _shapley_exact(benefits: Dict[FrozenSet[int], float], n: int
                   ) -> Dict[int, float]:
    values: Dict[int, float] = {i: 0.0 for i in range(n)}
    fact: List[int] = [math.factorial(k) for k in range(n + 1)]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = fact[size] * fact[n - size - 1] / fact[n]
            for comb in combinations(others, size):
                s = frozenset(comb)
                values[i] += weight * (benefits[s | {i}] - benefits[s])
    return values


def _shapley_sampled(benefits: Dict[FrozenSet[int], float], n: int,
                     perms: Sequence[Sequence[int]]) -> Dict[int, float]:
    acc: Dict[int, float] = {i: 0.0 for i in range(n)}
    for perm in perms:
        s: set = set()
        for i in perm:
            acc[i] += benefits[frozenset(s | {i})] - benefits[frozenset(s)]
            s.add(i)
    return {i: acc[i] / len(perms) for i in range(n)}


def _interaction_index(benefits: Dict[FrozenSet[int], float], n: int,
                       i: int, j: int) -> float:
    """Exact second-order Shapley interaction (>0 amplifies, <0 cancels)."""
    fact = [math.factorial(k) for k in range(max(n - 1, 0) + 1)]
    others = [k for k in range(n) if k not in (i, j)]
    total = 0.0
    for size in range(n - 1):
        weight = fact[size] * fact[n - size - 2] / fact[n - 1]
        for comb in combinations(others, size):
            s = frozenset(comb)
            total += weight * (benefits[s | {i, j}] - benefits[s | {i}]
                               - benefits[s | {j}] + benefits[s])
    return total


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _quartiles(xs: Sequence[float]) -> List[float]:
    if not xs:
        return [float("nan"), float("nan")]
    ordered = sorted(xs)
    def pct(p: float) -> float:
        k = (len(ordered) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
    return [pct(0.25), pct(0.75)]


# --------------------------------------------------------------------------- #
# Instances / labels
# --------------------------------------------------------------------------- #
def _build_instances(meta: Dict[str, Any],
                     events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = {s["type"]: s for s in catalog.interventions(meta["domain"])}
    instances: List[Dict[str, Any]] = []
    for ev in events:
        result = ev.get("result") or {}
        if result.get("applied") is False:
            continue  # rejected interventions changed nothing
        params = ev.get("params") or {}
        spec = specs.get(ev["type"], {})
        param_txt = "，".join(f"{p}={util_fmt(v)}" for p, v in
                             sorted(params.items()))
        instances.append({
            "index": len(instances),
            "id": f"itv{len(instances) + 1}",
            "type": ev["type"],
            "label": spec.get("label", ev["type"]),
            "param_text": param_txt,
            "params": params,
            "step": int(ev["step"]),
            "scheduled": bool(ev.get("scheduled", False)),
        })
    return instances


def util_fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_attribution(run_id: str, replicates: int = DEFAULT_REPLICATES,
                    progress: Optional[Callable[[float, str], None]] = None
                    ) -> Dict[str, Any]:
    """Compute and persist the full intervention-attribution document."""
    meta = storage.load_run_meta(run_id)
    if meta is None:
        raise KeyError(f"run not found: {run_id}")
    actual = storage.load_series(run_id)
    events = storage.load_events(run_id)
    if len(actual) < 2:
        raise ValueError("运行步数不足，无法进行归因（至少需要 1 步数据）")
    instances = _build_instances(meta, events)
    if not instances:
        raise ValueError("该运行没有已生效的干预记录，无法进行干预归因")
    if len(instances) > MAX_INSTANCES:
        raise ValueError(f"干预次数过多（{len(instances)} > {MAX_INSTANCES}），"
                         "请基于干预更少的运行进行归因")

    n = len(instances)
    replicates = max(1, min(int(replicates), 12))
    horizon = len(actual) - 1
    base_seed = int(meta.get("seed", 0))
    seeds = [_derive_seed(base_seed, r) for r in range(replicates)]
    metric_specs = catalog.attribution_metrics(meta["domain"], meta["model"])
    onset_specs = [s for s in metric_specs if s.get("onset")]
    chart_key = CHART_METRIC[f"{meta['domain']}/{meta['model']}"]

    coalitions, method, perms = _plan_coalitions(n)
    n_perms = len(perms)
    empty_set, full_set = frozenset(), frozenset(range(n))

    # Replay every (seed × coalition).  Scalar summaries for all replays plus,
    # for the chart metric, full trajectories of the empty/full/singleton
    # coalitions (needed for the onset test and the comparison chart).
    sims: List[Dict[FrozenSet[int], Dict[str, float]]] = []
    traj_chart: List[Dict[FrozenSet[int], List[float]]] = []
    onset_keys = {s["key"] for s in onset_specs}
    onset_cache: List[Dict[int, Dict[str, List[float]]]] = []
    singleton_sets = {frozenset({i}) for i in range(n)}
    total_work = len(coalitions) * replicates
    done = 0
    for r, seed in enumerate(seeds):
        sim_r: Dict[FrozenSet[int], Dict[str, float]] = {}
        traj_r: Dict[FrozenSet[int], List[float]] = {}
        onset_r: Dict[int, Dict[str, List[float]]] = {}
        for coalition in coalitions:
            series = _replay(meta, seed, horizon, instances, coalition)
            sim_r[coalition] = _series_summaries(meta["domain"],
                                                 meta["model"], series)
            if coalition == empty_set or coalition in singleton_sets:
                idx = -1 if coalition == empty_set else next(iter(coalition))
                onset_r[idx] = {k: [float(row[k]) for row in series]
                                for k in onset_keys}
                if r == 0:
                    traj_r[coalition] = onset_r[idx][chart_key] \
                        if chart_key in onset_keys else \
                        [float(row[chart_key]) for row in series]
            if r == 0 and coalition == full_set:
                traj_r[coalition] = [float(row[chart_key]) for row in series]
            done += 1
        sims.append(sim_r)
        traj_chart.append(traj_r)
        onset_cache.append(onset_r)
        if progress:
            progress(done / total_work, f"已完成第 {r + 1}/{replicates} 组随机种子")

    onset_rows = _compute_onsets(instances, seeds, onset_specs,
                                 chart_key, onset_cache)
    validation = _validate_replay(meta, actual, chart_key, horizon,
                                  instances, seeds[0])

    # ---------------------------------------------------------------- #
    # Per-metric Shapley tables
    # ---------------------------------------------------------------- #
    metric_blocks: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    for spec in metric_specs:
        key, direction = spec["key"], spec["direction"]
        if key not in sims[0][empty_set]:
            continue
        empty_vals = [sim_r[empty_set][key] for sim_r in sims]
        full_vals = [sim_r[full_set][key] for sim_r in sims]

        block: Dict[str, Any] = {
            "key": key,
            "label": spec["label"],
            "direction": direction,
            "primary": bool(spec.get("primary", False)),
            "baseline_mean": round(_median(empty_vals), 4),
            "baseline_sd": round(statistics.pstdev(empty_vals), 4)
            if replicates > 1 else 0.0,
            "full_mean": round(_median(full_vals), 4),
            "full_sd": round(statistics.pstdev(full_vals), 4)
            if replicates > 1 else 0.0,
        }

        if direction in ("up", "down"):
            per_seed_phi: List[Dict[int, float]] = []
            total_benefit_seeds: List[float] = []
            base_scale = max(abs(_median(empty_vals)), 1e-9)
            for sim_r in sims:
                benefits = {c: _benefit(direction, sim_r[empty_set][key],
                                        v[key]) for c, v in sim_r.items()}
                phi = (_shapley_exact(benefits, n) if method == "exact"
                       else _shapley_sampled(benefits, n, perms))
                per_seed_phi.append(phi)
                total_benefit_seeds.append(benefits[full_set])

            ranks_per_seed = []
            for r in range(replicates):
                order = sorted(range(n), key=lambda i: per_seed_phi[r][i],
                               reverse=True)
                ranks_per_seed.append({i: pos + 1
                                       for pos, i in enumerate(order)})

            total_benefit_median = _median(total_benefit_seeds)
            rows = []
            for i in range(n):
                seed_phis = [per_seed_phi[r][i] for r in range(replicates)]
                mean_phi = _median(seed_phis)
                sd_phi = statistics.pstdev(seed_phis) if replicates > 1 else 0.0
                seed_ranks = [ranks_per_seed[r][i] for r in range(replicates)]
                rank_mode = statistics.mode(seed_ranks)
                beneficial = sum(1 for v in seed_phis
                                 if v > 0.005 * base_scale) / replicates
                rel = mean_phi / base_scale
                if mean_phi < -0.005 * base_scale:
                    effect = "adverse"
                elif rel >= 0.10 and beneficial >= 0.8:
                    effect = "strong"
                elif rel >= 0.02 and beneficial >= 0.6:
                    effect = "effective"
                elif rel >= 0.005 and beneficial >= 0.5:
                    effect = "weak"
                else:
                    effect = "none"

                single_seeds = [
                    _benefit(direction, sim_r[empty_set][key],
                             sim_r[frozenset({i})][key]) for sim_r in sims]
                loo_seeds = []
                for sim_r in sims:
                    b_full = _benefit(direction, sim_r[empty_set][key],
                                      sim_r[full_set][key])
                    b_without = _benefit(direction, sim_r[empty_set][key],
                                         sim_r[full_set - {i}][key])
                    loo_seeds.append(b_full - b_without)

                rows.append({
                    "id": instances[i]["id"],
                    "index": i,
                    "shapley_mean": round(mean_phi, 4),
                    "shapley_sd": round(sd_phi, 4),
                    "shapley_per_seed": [round(v, 4) for v in seed_phis],
                    "pct_of_total": round(100 * mean_phi / total_benefit_median, 1)
                    if abs(total_benefit_median) > 1e-9 else None,
                    "pct_of_baseline": round(100 * rel, 1),
                    "singleton_mean": round(_median(single_seeds), 4),
                    "singleton_sd": round(statistics.pstdev(single_seeds), 4)
                    if replicates > 1 else 0.0,
                    "loo_mean": round(_median(loo_seeds), 4),
                    "benefit_rate": round(beneficial, 2),
                    "rank_mode": rank_mode,
                    "rank_stable": len(set(seed_ranks)) == 1,
                    "effect_class": effect,
                })
            rows.sort(key=lambda row: row["shapley_mean"], reverse=True)
            # Decomposition residual per seed (exactly zero under exact
            # enumeration); only its cross-seed median is reported, since the
            # sum of per-intervention *medians* need not equal the median
            # total in sampled/stochastic settings.
            residuals = [sum(per_seed_phi[r][i] for i in range(n))
                         - total_benefit_seeds[r]
                         for r in range(replicates)]
            block["total_benefit_mean"] = round(total_benefit_median, 4)
            block["total_benefit_sd"] = round(
                statistics.pstdev(total_benefit_seeds), 4) \
                if replicates > 1 else 0.0
            block["decomposition_residual"] = round(_median(residuals), 4)
            block["ranking"] = rows

            # Pairwise interaction: simple synergy and (exact mode) the
            # second-order Shapley interaction index, median across seeds.
            pair_rows = []
            total_scale = max(abs(total_benefit_median), 1e-9)
            for i, j in combinations(range(n), 2):
                simple_vals, idx_vals = [], []
                for sim_r in sims:
                    benefits = {c: _benefit(direction, sim_r[empty_set][key],
                                            v[key]) for c, v in sim_r.items()}
                    simple_vals.append(
                        benefits[frozenset({i, j})]
                        - benefits[frozenset({i})]
                        - benefits[frozenset({j})])
                    if method == "exact":
                        idx_vals.append(_interaction_index(benefits, n, i, j))
                simple = _median(simple_vals)
                idx_val = _median(idx_vals) if idx_vals else None
                thr = 0.05 * total_scale
                if simple > thr or (idx_val is not None and idx_val > thr):
                    kind = "synergy"
                elif simple < -thr or (idx_val is not None and idx_val < -thr):
                    kind = "antagonism"
                else:
                    kind = "additive"
                pair_rows.append({
                    "a": instances[i]["id"], "b": instances[j]["id"],
                    "synergy": round(simple, 4),
                    "interaction_index": round(idx_val, 4)
                    if idx_val is not None else None,
                    "kind": kind,
                })
            interactions.append({"metric": key, "label": spec["label"],
                                  "primary": bool(spec.get("primary")),
                                  "pairs": pair_rows})
        else:
            block["raw_shift_mean"] = round(
                _median([v2 - v1 for v1, v2 in zip(empty_vals, full_vals)]), 4)
        metric_blocks.append(block)

    # ---------------------------------------------------------------- #
    # Trajectories (seed 0, downsampled) for the comparison chart
    # ---------------------------------------------------------------- #
    steps_full = list(range(horizon + 1))
    pick = _downsample(steps_full, TRAJ_POINTS)
    traj0 = traj_chart[0]
    actual_chart = [float(row[chart_key]) for row in actual[:horizon + 1]]
    payload_traj = {
        "metric": chart_key,
        "steps": [steps_full[k] for k in pick],
        "actual": [actual_chart[k] for k in pick],
        "baseline": [traj0[empty_set][k] for k in pick],
        "full": [traj0[full_set][k] for k in pick],
        "singletons": [
            {"id": instances[i]["id"],
             "values": [traj0[frozenset({i})][k] for k in pick]}
            for i in range(n)],
        "intervention_steps": [inst["step"] for inst in instances],
    }

    document = {
        "run_id": run_id,
        "name": meta["name"],
        "domain": meta["domain"],
        "model": meta["model"],
        "status": "finished",
        "generated_at": util.now_iso(),
        "params": {
            "replicates": replicates,
            "horizon": horizon,
            "method": method,
            "permutations": n_perms if method == "sampled" else None,
            "coalitions": len(coalitions),
            "replays": len(coalitions) * replicates,
            "onset_rule": {
                "window": ONSET_WINDOW,
                "hold": ONSET_HOLD,
                "noise_z": ONSET_NOISE_Z,
                "pre_window": ONSET_PRE_WINDOW,
                "rel_floor": ONSET_REL_FLOOR,
            },
        },
        "seeds": seeds,
        "validation": validation,
        "instances": [{k: inst[k] for k in
                       ("id", "type", "label", "param_text", "step",
                        "scheduled", "index")}
                      for inst in instances],
        "metrics": metric_blocks,
        "onset": onset_rows,
        "interactions": interactions,
        "trajectories": payload_traj,
        "summary": _narrative(instances, metric_blocks, onset_rows,
                              interactions, validation, method),
    }
    storage.save_attribution(run_id, document)
    if progress:
        progress(1.0, "归因分析完成")
    return document


def _compute_onsets(instances: List[Dict[str, Any]], seeds: List[int],
                    onset_specs: List[Dict[str, Any]], chart_key: str,
                    onset_cache: List[Dict[int, Dict[str, List[float]]]]
                    ) -> List[Dict[str, Any]]:
    """Per intervention × onset-metric: paired CRN onset over all seeds."""
    primary_onset = chart_key if any(s["key"] == chart_key
                                     for s in onset_specs) else \
        (next(s["key"] for s in onset_specs if s.get("primary"))
         if onset_specs else None)
    rows: List[Dict[str, Any]] = []
    for spec in onset_specs:
        key, direction = spec["key"], spec["direction"]
        for inst in instances:
            i = inst["index"]
            per_seed = []
            for r in range(len(seeds)):
                res = _onset_for(onset_cache[r][-1][key],
                                 onset_cache[r][i][key],
                                 direction, inst["step"])
                per_seed.append(res)
            detected = [r for r in per_seed if r["detected"]]
            lags = [r["lag"] for r in detected]
            steps_detected = [r["onset_step"] for r in detected]
            rows.append({
                "id": inst["id"],
                "metric": key,
                "metric_label": spec["label"],
                "primary": key == primary_onset,
                "detection_rate": round(len(detected) / len(seeds), 2),
                "onset_step_median": _median(steps_detected)
                    if steps_detected else None,
                "lag_median": _median(lags) if lags else None,
                "lag_q1_q3": _quartiles(lags) if lags else [None, None],
                "sustained_rate": round(
                    sum(1 for r in detected if r["sustained"])
                    / max(len(detected), 1), 2) if detected else 0.0,
                "threshold_median": _median([r["threshold"] for r in per_seed]),
                "per_seed": [{"detected": r["detected"],
                              "onset_step": r["onset_step"], "lag": r["lag"]}
                             for r in per_seed],
            })
    return rows


def _validate_replay(meta: Dict[str, Any], actual: List[Dict[str, Any]],
                     key: str, horizon: int,
                     instances: List[Dict[str, Any]],
                     seed: int) -> Dict[str, Any]:
    """The full-coalition replay at the run's own seed must reproduce it."""
    full = _replay(meta, seed, horizon, instances,
                   frozenset(range(len(instances))))
    diffs = [abs(float(a.get(key, 0)) - float(b.get(key, 0)))
             for a, b in zip(actual[:horizon + 1], full)]
    max_diff = max(diffs) if diffs else 0.0
    scale = max(abs(float(r.get(key, 0))) for r in actual[:horizon + 1]) or 1.0
    rel = max_diff / scale
    verified = max_diff == 0.0
    note = ("全干预反事实重放与实际运行逐点一致，归因结果可信"
            if verified else
            f"全干预重放与实际运行存在偏差（最大相对差 {rel:.2%}），"
            "归因仍可参考但请检查运行是否被手动续跑或修改")
    return {"verified": verified, "metric": key,
            "max_abs_diff": round(max_diff, 4),
            "max_rel_diff": round(rel, 4), "note": note}


def _downsample(xs: Sequence[int], limit: int) -> List[int]:
    if len(xs) <= limit:
        return list(range(len(xs)))
    step = len(xs) / limit
    return sorted({min(len(xs) - 1, int(k * step)) for k in range(limit)})


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #
def _narrative(instances: List[Dict[str, Any]],
               blocks: List[Dict[str, Any]], onset: List[Dict[str, Any]],
               interactions: List[Dict[str, Any]],
               validation: Dict[str, Any], method: str) -> List[str]:
    lines = [validation["note"]]
    method_txt = "全子集精确 Shapley 分解" if method == "exact" \
        else "排列抽样 Shapley 估计（单干预/两两/缺一组合为精确值）"
    lines.append(f"方法：{method_txt}，每个联盟组合用相同随机种子配对重放，"
                 f"共 {len(instances)} 项干预。")

    primary = next((b for b in blocks if b.get("primary")), None)
    if primary and primary.get("ranking"):
        lines.append(
            f"关键指标「{primary['label']}」：无干预基线 "
            f"{primary['baseline_mean']:g}，全部干预叠加后 "
            f"{primary['full_mean']:g}，合计改善 "
            f"{primary['total_benefit_mean']:g}。")
        top = primary["ranking"][0]
        top_inst = next(i for i in instances if i["id"] == top["id"])
        lines.append(
            f"贡献最大：{top_inst['label']}（第 {top_inst['step']} 步），"
            f"Shapley 贡献 {top['shapley_mean']:g}"
            f"（占基线 {top['pct_of_baseline']}%，"
            f"{top['benefit_rate'] * 100:.0f}% 的随机种子下方向一致）。")
        weakest = next((r for r in reversed(primary["ranking"])
                        if r["effect_class"] in ("none", "adverse")), None)
        if weakest is not None:
            weak_inst = next(i for i in instances if i["id"] == weakest["id"])
            if weakest["effect_class"] == "adverse":
                lines.append(
                    f"存在反效果：{weak_inst['label']}（第 {weak_inst['step']} 步），"
                    f"Shapley 贡献 {weakest['shapley_mean']:g}。")
            else:
                lines.append(
                    f"基本无效：{weak_inst['label']}（第 {weak_inst['step']} 步），"
                    f"Shapley 贡献 {weakest['shapley_mean']:g}，"
                    f"仅 {weakest['benefit_rate'] * 100:.0f}% 种子下观测到有利效果。")

    primary_onset = [r for r in onset if r.get("primary")]
    for r in primary_onset:
        inst = next(i for i in instances if i["id"] == r["id"])
        if r["detection_rate"] == 0:
            lines.append(f"{inst['label']}：观察窗内未检测到对"
                         f"「{r['metric_label']}」的显著起效。")
        else:
            lines.append(
                f"{inst['label']}：中位起效时刻第 {r['onset_step_median']:g} 步"
                f"（滞后 {r['lag_median']:g} 步，"
                f"{r['detection_rate'] * 100:.0f}% 种子可检出）。")

    primary_block = next((b for b in interactions if b.get("primary")),
                         interactions[0] if interactions else None)
    if primary_block:
        for pair in primary_block["pairs"]:
            a = next(i for i in instances if i["id"] == pair["a"])
            b = next(i for i in instances if i["id"] == pair["b"])
            if pair["kind"] == "synergy":
                lines.append(f"{a['label']} 与 {b['label']} 存在效果放大"
                             f"（协同项 {pair['synergy']:g}）。")
            elif pair["kind"] == "antagonism":
                lines.append(f"{a['label']} 与 {b['label']} 存在效果重叠/抵消"
                             f"（交互项 {pair['synergy']:g}），"
                             "叠加并未带来等比例收益。")
    return lines


# --------------------------------------------------------------------------- #
# Background job control
# --------------------------------------------------------------------------- #
def start_attribution(run_id: str, replicates: int = DEFAULT_REPLICATES) -> Dict[str, Any]:
    """Launch (or reuse) a background attribution job for a run."""
    with _JOBS_LOCK:
        existing = _JOBS.get(run_id)
        if existing and existing["status"] == "running":
            return existing
        job = {"run_id": run_id, "status": "running", "progress": 0.0,
               "message": "排队中", "started_at": util.now_iso()}
        _JOBS[run_id] = job

        def worker() -> None:
            try:
                def cb(frac: float, msg: str) -> None:
                    job["progress"] = round(frac, 3)
                    job["message"] = msg
                run_attribution(run_id, replicates=replicates, progress=cb)
                job["status"] = "finished"
                job["progress"] = 1.0
            except Exception as exc:  # noqa: BLE001
                job["status"] = "error"
                job["error"] = str(exc)

        threading.Thread(target=worker, daemon=True).start()
        return job


def job_status(run_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        return dict(job) if job else None
