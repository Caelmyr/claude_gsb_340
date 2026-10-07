"""Headless command-line batch runner.

Run a scene from the terminal without the web UI — useful for quick parameter
sweeps and for verifying the simulation + storage layer independently:

    python3 cli.py --list
    python3 cli.py --scene scene_epidemic_abm --steps 200 --report
    python3 cli.py --scene scene_traffic_ca --steps 500 --snapshot-interval 5
"""

from __future__ import annotations

import argparse
import sys

from backend import attribution, models, report, run_manager, storage, util


def list_scenes() -> None:
    for s in storage.list_scenes():
        print(f"{s['id']:24s} {s['domain']:8s}/{s['model']:3s}  {s['name']}")


def run_one(scene_id: str, steps: int, snapshot_interval: int,
            make_report: bool) -> int:
    scene = storage.load_scene(scene_id)
    if scene is None:
        print(f"error: scene not found: {scene_id}", file=sys.stderr)
        return 1
    scene_obj = models.Scene.from_dict(scene)
    meta = run_manager.manager.create_run(
        scene_obj, snapshot_interval=snapshot_interval)
    print(f"run {meta['id']}: {meta['name']} "
          f"({meta['domain']}/{meta['model']}) seed={meta['seed']}")

    result = run_manager.manager.run_batch(meta["id"], steps, keep_engine=True)
    print(f"finished at step {result['step']}")
    for k, v in result["stats"].items():
        print(f"  {k:16s} {v}")

    if make_report:
        rpt = report.generate_report(meta["id"])
        print("\nsummary:")
        for line in rpt["summary"]:
            print(f"  - {line}")
    return 0


def attribute_one(run_id: str, n_seeds: int, steps: int | None = None) -> int:
    meta = storage.load_run_meta(run_id)
    if meta is None:
        print(f"error: run not found: {run_id}", file=sys.stderr)
        return 1
    base_seed = int(meta.get("seed", 0))
    seeds = [base_seed + 2654435761 * i for i in range(max(1, n_seeds))]

    last = [0]

    def show(p):
        if p["done"] != last[0]:
            last[0] = p["done"]
            print(f"\r  counterfactuals {p['done']}/{p['total']}", end="",
                  file=sys.stderr)

    doc = attribution.attribute_run(run_id, seeds=seeds, steps=steps,
                                    progress=show)
    storage.save_attribution(run_id, doc)
    print(file=sys.stderr)
    print(f"attribution {run_id}: method={doc['method']} "
          f"seeds={doc['n_seeds']} simulations={doc['n_simulations']}")
    primary = doc["primary_target"]
    res = doc["results"][primary]
    t_label = res["target"]["label"]
    print(f"\n[{t_label}] 基线均值 {res['baseline'].get('mean')} → "
          f"干预包 {res['bundle'].get('mean')}，"
          f"相对收益 {pct(res['relative_benefit'])} "
          f"(方向一致概率 {pct(res['prob_beneficial'])}, "
          f"排名一致性 ρ={res['rank_concordance']})")
    print(f"{'排名':<4}{'干预':<28}{'Shapley':>10}{'相对份额':>10}"
          f"{'单独效果':>10}{'起效滞后':>8}  判定")
    for row in sorted(res["interventions"], key=lambda r: r["rank"]):
        lag = row["solo_lag_median"]
        lag_s = "—" if lag is None else f"{lag:g}步"
        print(f"{row['rank']:<4}{row['label'][:26]:<28}"
              f"{row['shapley'].get('mean', 0):>10.2f}"
              f"{pct(row['shapley_relative']):>10}"
              f"{pct(row['solo_relative']):>10}{lag_s:>8}"
              f"  {row['strength']}/{row['stability']}")
    for p in res["interactions"]:
        if p["classification"] != "additive":
            print(f"  交互 [{p['label_i']} × {p['label_j']}]: "
                  f"{p['classification']} (指数 {p['index'].get('mean')}, "
                  f"相对 {pct(p['relative_mean'])})")
    for note in doc["notes"]:
        print(f"  note: {note}")
    return 0


def pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description="Headless simulation batch runner")
    p.add_argument("--list", action="store_true", help="list available scenes")
    p.add_argument("--scene", help="scene id to run")
    p.add_argument("--steps", type=int, default=200, help="steps to run")
    p.add_argument("--snapshot-interval", type=int, default=1,
                   help="persist a full snapshot every N steps")
    p.add_argument("--report", action="store_true", help="generate a report")
    p.add_argument("--attribute", action="store_true",
                   help="run intervention attribution after the batch run")
    p.add_argument("--attribute-run",
                   help="compute attribution for an existing run id")
    p.add_argument("--seeds", type=int, default=5,
                   help="number of random seeds for attribution (1-9)")
    args = p.parse_args()

    storage.ensure_dirs()
    if args.list:
        list_scenes()
        return 0
    if args.attribute_run:
        return attribute_one(args.attribute_run, args.seeds)
    if not args.scene:
        p.print_help()
        return 1
    rc = run_one(args.scene, args.steps, args.snapshot_interval, args.report)
    if rc == 0 and args.attribute:
        runs = storage.list_runs()
        if runs:
            rc = attribute_one(runs[0]["id"], args.seeds)
    return rc


if __name__ == "__main__":
    sys.exit(main())
