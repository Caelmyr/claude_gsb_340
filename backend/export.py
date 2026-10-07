"""Data export: CSV time series and full / per-step JSON dumps.

Exports are written into ``data/exports/`` through the atomic-write helper and
the returned file path is served back to the browser as a download.  Three
formats are offered:

* ``csv``         — the aggregate series (one row per step), openable in Excel.
* ``json``        — a full dump of meta + series + events + snapshots (bounded).
* ``individuals`` — the individual states at a chosen step, as a JSON array.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict, List, Optional

from . import storage, util


def _export_path(run_id: str, fmt: str) -> str:
    ext = "csv" if fmt in ("csv", "attribution") else "json"
    return os.path.join(storage.exports_dir(), f"{run_id}_{fmt}.{ext}")


def _series_csv(series: List[Dict[str, Any]]) -> str:
    if not series:
        return "step\n"
    keys: List[str] = ["step"]
    for row in series:
        for k in row.keys():
            if k != "step" and k not in keys:
                keys.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in series:
        writer.writerow(row)
    return buf.getvalue()


def export_series_csv(run_id: str) -> str:
    series = storage.load_series(run_id)
    path = _export_path(run_id, "csv")
    storage.atomic_write_bytes(path, _series_csv(series).encode("utf-8"))
    return path


def export_run_json(run_id: str) -> str:
    """Full run dump (meta + series + events), a self-contained JSON document."""
    meta = storage.load_run_meta(run_id)
    series = storage.load_series(run_id)
    events = storage.load_events(run_id)
    doc = {"meta": meta, "series": series, "events": events,
           "exported_at": util.now_iso()}
    path = _export_path(run_id, "json")
    storage.atomic_write_bytes(path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    return path


def export_individuals_json(run_id: str, step: Optional[int] = None) -> str:
    """Export the individual states at one step as a JSON array."""
    from .run_manager import manager
    snap = manager.get_snapshot(run_id, step)
    doc = {
        "run_id": run_id,
        "step": snap.get("step"),
        "bounds": snap.get("bounds"),
        "stats": snap.get("stats"),
        "individuals": snap.get("individuals", []),
        "exported_at": util.now_iso(),
    }
    path = _export_path(run_id, "individuals")
    storage.atomic_write_bytes(path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    return path


def export_attribution_csv(run_id: str) -> str:
    """Attribution table as a multi-section CSV (one block per metric)."""
    doc = storage.load_attribution(run_id)
    if doc is None:
        raise KeyError(f"attribution not found: {run_id}")
    instances = {i["id"]: i for i in doc.get("instances", [])}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# 干预归因结果", doc.get("name", ""),
                f"replicates={doc.get('params', {}).get('replicates')}",
                f"method={doc.get('params', {}).get('method')}"])
    w.writerow([])
    for block in doc.get("metrics", []):
        ranking = block.get("ranking")
        if not ranking:
            continue
        w.writerow([f"## 指标: {block['label']} ({block['key']})",
                    f"基线中位={block.get('baseline_mean')}",
                    f"全干预中位={block.get('full_mean')}",
                    f"合计改善={block.get('total_benefit_mean')}",
                    f"分解残差={block.get('decomposition_residual')}"])
        w.writerow(["排名", "干预", "类型", "施加步",
                    "Shapley贡献中位", "Shapley贡献SD",
                    "占合计改善%", "占基线%",
                    "单独实施效果中位", "缺一影响(LOO)",
                    "方向一致率", "多种子排名", "排名稳定", "判定"])
        for pos, r in enumerate(ranking, start=1):
            inst = instances.get(r["id"], {})
            w.writerow([pos, inst.get("label", r["id"]),
                        inst.get("type", ""), inst.get("step", ""),
                        r["shapley_mean"], r["shapley_sd"],
                        r.get("pct_of_total"), r["pct_of_baseline"],
                        r["singleton_mean"], r["loo_mean"],
                        r["benefit_rate"], r["rank_mode"],
                        r["rank_stable"], r["effect_class"]])
        w.writerow([])

    w.writerow(["## 何时开始起效"])
    w.writerow(["干预", "指标", "检出率", "起效步(中位)", "滞后(中位)",
                "滞后Q1", "滞后Q3", "持续占比"])
    for r in doc.get("onset", []):
        if not r.get("primary"):
            continue
        inst = instances.get(r["id"], {})
        q1, q3 = r.get("lag_q1_q3") or [None, None]
        w.writerow([inst.get("label", r["id"]), r.get("metric_label"),
                    r["detection_rate"], r.get("onset_step_median"),
                    r.get("lag_median"), q1, q3, r.get("sustained_rate")])
    w.writerow([])

    w.writerow(["## 干预间相互作用"])
    w.writerow(["指标", "干预A", "干预B", "协同项",
                "Shapley交互指数", "判定"])
    for ib in doc.get("interactions", []):
        for p in ib.get("pairs", []):
            a = instances.get(p["a"], {}).get("label", p["a"])
            b = instances.get(p["b"], {}).get("label", p["b"])
            w.writerow([ib.get("label", ib.get("metric")), a, b,
                        p["synergy"], p.get("interaction_index"),
                        p["kind"]])

    path = _export_path(run_id, "attribution")
    storage.atomic_write_bytes(path, buf.getvalue().encode("utf-8-sig"))
    return path


def export_run(run_id: str, fmt: str = "csv",
               step: Optional[int] = None) -> str:
    """Dispatch an export by format; return the absolute file path."""
    if fmt == "csv":
        return export_series_csv(run_id)
    if fmt == "json":
        return export_run_json(run_id)
    if fmt == "individuals":
        return export_individuals_json(run_id, step)
    if fmt == "attribution":
        return export_attribution_csv(run_id)
    raise ValueError(f"unknown format: {fmt}")


def formats() -> List[Dict[str, str]]:
    """Describe available formats for the export page."""
    return [
        {"key": "csv", "label": "统计序列 CSV", "desc": "每个时间步一行的聚合统计，可用 Excel 打开。"},
        {"key": "json", "label": "完整运行 JSON", "desc": "配置 + 统计序列 + 干预日志的完整转储。"},
        {"key": "individuals", "label": "个体状态 JSON", "desc": "某一步所有个体的状态数组（车辆/动物/人）。"},
        {"key": "attribution", "label": "干预归因 CSV", "desc": "Shapley 贡献排序、起效时刻与干预间相互作用对照表。"},
    ]
