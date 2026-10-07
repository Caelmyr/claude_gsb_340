"""Smoke tests for the simulation engines, storage and run lifecycle.

Run directly::

    python3 tests/smoke.py

Each check is independent and prints PASS / FAIL; the script exits non-zero on
the first failure so it can be wired into CI or a pre-commit hook.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import attribution, models, report, storage  # noqa: E402
from backend.engine import make_engine  # noqa: E402
from backend.run_manager import manager  # noqa: E402

_ENGINES = ["traffic/ca", "traffic/abm", "ecology/ca", "ecology/abm",
            "epidemic/ca", "epidemic/abm"]


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as exc:
        print(f"FAIL  {name}: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {name}: {exc}")
        sys.exit(1)


def engines_step() -> None:
    for key in _ENGINES:
        d, m = key.split("/")
        eng = make_engine(d, m, seed=42)
        for _ in range(5):
            eng.step()
        assert eng.step_count == 5, key
        assert len(eng.individuals()) > 0, f"{key} has no individuals"
        stats = eng.stats()
        assert stats, f"{key} produced empty stats"
        snap = eng.snapshot()
        assert snap["step"] == 5
        assert snap["stats"] == stats


def interventions_apply() -> None:
    eng = make_engine("epidemic", "abm", seed=1)
    before = eng.stats()["susceptible"]
    res = eng.apply_intervention({"type": "vaccinate", "params": {"fraction": 1.0}})
    assert res["applied"], res
    assert eng.stats()["susceptible"] == 0
    assert before > 0


def storage_atomic_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        # Redirect the module's DATA_DIR for this isolated check.
        old = storage.DATA_DIR
        storage.DATA_DIR = td
        try:
            storage.ensure_dirs()
            storage.save_scene({"id": "x", "name": "t", "updated_at": "z"})
            assert storage.load_scene("x")["name"] == "t"
            storage.save_step("r1", 0, {"step": 0, "v": 1})
            storage.save_step("r1", 7, {"step": 7, "v": 2})
            assert storage.load_step("r1", 7)["v"] == 2
            assert storage.list_steps("r1") == [0, 7]
        finally:
            storage.DATA_DIR = old


def run_lifecycle() -> None:
    scene = models.Scene(domain="epidemic", model="abm",
                         config={"n": 200, "width": 300, "height": 300,
                                 "initial_infected": 5})
    meta = manager.create_run(scene, seed=1, snapshot_interval=2)
    rid = meta["id"]
    try:
        r = manager.step(rid, 6)
        assert r["step"] == 6
        series = manager.get_series(rid)
        assert series[0]["step"] == 0 and series[-1]["step"] == 6
        assert len(manager.get_individuals(rid, 6)) == 200
        # snapshot_interval=2 -> full snapshots persisted at 0,2,4,6
        steps = storage.list_steps(rid)
        assert steps == [0, 2, 4, 6], steps
        rpt = report.generate_report(rid)
        assert rpt["steps"] == 7
    finally:
        manager.delete_run(rid)


def intervention_attribution() -> None:
    # Two sequential interventions on a sustained ABM epidemic.
    scene = models.Scene(
        domain="epidemic", model="abm",
        config={"n": 500, "width": 450, "height": 450, "beta": 0.35,
                "gamma": 0.05, "speed": 2.0, "radius": 7.0,
                "initial_infected": 12},
        interventions=[
            {"type": "lockdown", "params": {"scale": 0.7}, "at_step": 20},
            {"type": "mask", "params": {"scale": 0.5}, "at_step": 40},
        ])
    meta = manager.create_run(scene, seed=11, snapshot_interval=200)
    rid = meta["id"]
    try:
        manager.run_batch(rid, 150, keep_engine=False)
        doc = attribution.attribute_run(rid, seeds=[11, 21], progress=lambda p: None)
        assert doc["method"] == "exact"
        assert doc["n_interventions"] == 2
        res = doc["results"]["peak_infected"]
        # Shapley values must partition the total benefit exactly.
        total = res["total_benefit"]["mean"]
        shap_sum = sum(x["shapley"]["mean"] for x in res["interventions"])
        assert abs(shap_sum - total) < 1e-6, (shap_sum, total)
        # Ranks are a 1..k permutation.
        ranks = sorted(x["rank"] for x in res["interventions"])
        assert ranks == [1, 2]
        # Full-bundle replay reproduces the recorded run feature exactly.
        assert all(v == 0.0 for v in doc["fidelity"].values()), doc["fidelity"]
        # Onset rule block is documented.
        assert doc["onset_rule"]["persist_steps"] >= 1

        # Error: a run without interventions cannot be attributed.
        empty = models.Scene(domain="epidemic", model="abm",
                             config={"n": 100, "initial_infected": 3},
                             interventions=[])
        m2 = manager.create_run(empty, seed=1)
        try:
            manager.run_batch(m2["id"], 5, keep_engine=False)
            try:
                attribution.attribute_run(m2["id"], seeds=[1])
                raise AssertionError("expected ValueError for no interventions")
            except ValueError:
                pass
        finally:
            manager.delete_run(m2["id"])
    finally:
        manager.delete_run(rid)


def main() -> None:
    check("six engines step and snapshot", engines_step)
    check("interventions apply", interventions_apply)
    check("atomic sharded storage", storage_atomic_roundtrip)
    check("run lifecycle + report", run_lifecycle)
    check("intervention attribution (shapley + onset + fidelity)",
          intervention_attribution)
    print("\nall smoke tests passed")


if __name__ == "__main__":
    main()
