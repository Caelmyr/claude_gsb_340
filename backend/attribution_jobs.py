"""Background job management for intervention attribution.

An attribution analysis replays many counterfactual simulations (up to
``seeds × 2^k`` for ``k`` interventions), which can take seconds to minutes.
Jobs therefore run on daemon threads, publish coarse progress into an in-memory
registry, and persist their final document via :func:`storage.save_attribution`.
The API layer polls the registry while a job runs and falls back to the saved
``attribution.json`` afterwards.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Sequence

from . import attribution, storage, util


class AttributionJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    def start(self, run_id: str, seeds: Optional[Sequence[int]] = None,
              steps: Optional[int] = None,
              target_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Launch an attribution job for ``run_id`` (replacing any prior job)."""
        meta = storage.load_run_meta(run_id)
        if meta is None:
            raise KeyError(f"run not found: {run_id}")
        job_id = util.new_id("attr")
        job: Dict[str, Any] = {
            "id": job_id,
            "run_id": run_id,
            "status": "running",
            "progress": {"phase": "queued", "done": 0, "total": 0},
            "error": "",
            "started_at": util.now_iso(),
            "finished_at": "",
        }
        with self._lock:
            self._jobs[run_id] = job

        def worker() -> None:
            try:
                doc = attribution.attribute_run(
                    run_id, seeds=seeds, steps=steps,
                    target_keys=target_keys,
                    progress=lambda p: self._update(run_id, {"progress": p}),
                    is_aborted=lambda: self._get(run_id).get("status") == "aborted")
                storage.save_attribution(run_id, doc)
                self._update(run_id, {"status": "finished",
                                      "finished_at": util.now_iso(),
                                      "progress": {"phase": "done", "done": 1,
                                                   "total": 1}})
            except Exception as exc:  # noqa: BLE001
                self._update(run_id, {"status": "error", "error": str(exc),
                                      "finished_at": util.now_iso()})

        threading.Thread(target=worker, daemon=True).start()
        return job

    # ------------------------------------------------------------------ #
    def _get(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._jobs.get(run_id, {})

    def _update(self, run_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is not None:
                job.update(patch)

    def status(self, run_id: str) -> Dict[str, Any]:
        """Return job state augmented by the persisted document if available."""
        job = dict(self._get(run_id))
        doc = storage.load_attribution(run_id)
        if doc:
            job["has_result"] = True
            job["result_summary"] = {
                "generated_at": doc.get("generated_at"),
                "n_seeds": doc.get("n_seeds"),
                "n_interventions": doc.get("n_interventions"),
                "method": doc.get("method"),
                "primary_target": doc.get("primary_target"),
            }
        else:
            job["has_result"] = False
        return job

    def abort(self, run_id: str) -> Dict[str, Any]:
        job = self._get(run_id)
        if job and job.get("status") == "running":
            job["status"] = "aborted"
            job["finished_at"] = util.now_iso()
        return job


attr_jobs = AttributionJobManager()
