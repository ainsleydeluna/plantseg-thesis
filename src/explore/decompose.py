"""N4: free decompositions from existing records (audit EXPLORE-AUDIT-1 §2 N4; lane L-AM24, build item E1).

(a) `term_magnitudes`: per-term loss magnitudes over the last `last` iterations (10,000 by default) of a KD run's
    telemetry (E3 seed 42 of record for A1''), read from the run folder given by path: <run>/<stage>_telemetry.jsonl
    and <run>/<stage>_run_meta.jsonl, as src/training/train_distill.py writes them. A train row carries the UNRAMPED
    value of each instantiated term under its term name (logit_kd, cwd_feat, cwd_logit). The window is the rows with
    iteration max_iter - last + 1 .. max_iter, where max_iter is the run_end row's iter; exactly one row per
    iteration is required. A term that is absent from the rows, null (strict JSON writes a non-finite value as null)
    or non-finite in any window row is reported as such and no mean is formed for it: nothing is ever estimated.
    The A1'' weight is alpha = lambda * mean(L_logitKD) / mean(L_feat) (audit §2 A1''), printed with every input.

(b) `qat_table`: for each QAT model, four VAL all-class mIoU values from existing records, one table (DL-176's form):
    parent (its best.json), fake-quant at the selected epoch (the QAT telemetry's epoch_end row of that epoch),
    converted (the selection record's winner), fake-quant-off (the valfq artifact's summary.json). The table also
    gives the three differences parent - fake-quant, fake-quant - converted and fake-quant-off - converted, in pp.
    A record that is not given or not readable leaves its cell null, with the reason; nothing is estimated.

Every file read is hashed; the output records each hash. No model, no dataset, no TEST.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import guard as G

TERMS = ("logit_kd", "cwd_feat", "cwd_logit")
WINDOW = 10_000
FORMAT_TERMS = "x_n4_terms/1"
FORMAT_TABLE = "x_n4_qat_table/1"


class DecomposeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _strict_rows(path: Path) -> list[dict]:
    """JSONL rows, strictly: a non-JSON line other than a torn last line is refused; NaN/Infinity literals refused."""
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line, parse_constant=bad))
        except ValueError as e:
            if i == len(lines) - 1:
                raise DecomposeError("telemetry_torn", f"{path.name}: the last line is torn ({e})") from e
            raise DecomposeError("telemetry_not_strict", f"{path.name} line {i + 1}: {e}") from e
    return rows


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def term_magnitudes(run_dir, *, stage_key: str = "e3", last: int = WINDOW, terms=TERMS,
                    lambda_key: str = "lambda_logit") -> dict:
    """(a): the window means of each term and the A1'' alpha, or the reason a value cannot be formed."""
    d = Path(run_dir)
    tel = d / f"{stage_key}_telemetry.jsonl"
    meta_p = d / f"{stage_key}_run_meta.jsonl"
    for p in (tel, meta_p):
        if G.names_test(p):
            raise DecomposeError("path_test", f"{p} holds 'test' (SL-1)")
        if not p.is_file():
            raise DecomposeError("record_missing", f"{p} is missing")
    if isinstance(last, bool) or not isinstance(last, int) or last < 1:
        raise DecomposeError("window", f"the window must be a positive integer, got {last!r}")
    metas = [r for r in _strict_rows(meta_p) if r.get("event") == "run_meta"]
    if len(metas) != 1:
        raise DecomposeError("run_meta_rows", f"{meta_p.name} holds {len(metas)} run_meta rows; exactly 1 is required")
    meta = metas[0]
    rows = _strict_rows(tel)
    ends = [r for r in rows if r.get("event") == "run_end"]
    aborts = [r for r in rows if r.get("event") == "run_abort"]
    if aborts:
        raise DecomposeError("run_aborted", f"{tel.name} holds a run_abort row: not a completed run")
    if len(ends) != 1:
        raise DecomposeError("run_end_rows", f"{tel.name} holds {len(ends)} run_end rows; exactly 1 is required")
    max_iter = ends[0].get("iter")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < last:
        raise DecomposeError("window_exceeds_run", f"run_end iter {max_iter!r} is shorter than the window {last}")
    lo = max_iter - last + 1
    window = [r for r in rows if r.get("event") == "train" and isinstance(r.get("iter"), int)
              and lo <= r["iter"] <= max_iter]
    iters = [r["iter"] for r in window]
    if len(iters) != last or sorted(set(iters)) != list(range(lo, max_iter + 1)):
        raise DecomposeError("window_rows", f"the window {lo}..{max_iter} holds {len(iters)} train rows over "
                                            f"{len(set(iters))} distinct iterations; exactly one row per iteration "
                                            "is required (a relaunch interleaved into one file is refused)")
    out_terms = {}
    for t in terms:
        present = sum(1 for r in window if t in r)
        nulls = sum(1 for r in window if t in r and r[t] is None)
        nonfinite = sum(1 for r in window if t in r and r[t] is not None and not _finite_number(r[t]))
        if present == 0:
            out_terms[t] = {"status": "absent", "reason": f"no train row in the window carries {t!r}: the term is "
                                                          "not in the telemetry (not instantiated); no value formed"}
            continue
        if present != last or nulls or nonfinite:
            out_terms[t] = {"status": "incomplete", "rows_with_term": present, "null_rows": nulls,
                            "nonfinite_rows": nonfinite,
                            "reason": "the term is missing, null or non-finite in some window row; no mean formed"}
            continue
        vals = [float(r[t]) for r in window]
        out_terms[t] = {"status": "ok", "rows": last, "mean": math.fsum(vals) / last, "sum": math.fsum(vals),
                        "min": min(vals), "max": max(vals)}
    lam = meta.get(lambda_key)
    weights = {"lambda_logit": meta.get("lambda_logit"), "alpha_cwd": meta.get("alpha_cwd"),
               "beta_cwd": meta.get("beta_cwd")}
    alpha = {"status": "not formed"}
    kd, feat = out_terms.get("logit_kd", {}), out_terms.get("cwd_feat", {})
    if not _finite_number(lam):
        alpha["reason"] = f"run_meta {lambda_key} is {lam!r}, not a finite number"
    elif kd.get("status") != "ok" or feat.get("status") != "ok":
        alpha["reason"] = f"logit_kd {kd.get('status')}, cwd_feat {feat.get('status')}: both window means are needed"
    elif feat["mean"] <= 0.0:
        alpha["reason"] = f"mean(L_feat) is {feat['mean']!r}, not positive"
    else:
        value = float(lam) * kd["mean"] / feat["mean"]
        alpha = {"status": "ok", "value": value, "formula": "alpha = lambda * mean(L_logitKD) / mean(L_feat)",
                 "lambda": float(lam), "mean_L_logitKD": kd["mean"], "mean_L_feat": feat["mean"],
                 "repr": repr(value)}
    contrib = {}
    for t, wk in (("logit_kd", "lambda_logit"), ("cwd_feat", "alpha_cwd"), ("cwd_logit", "beta_cwd")):
        if out_terms.get(t, {}).get("status") == "ok" and _finite_number(weights[wk]):
            contrib[t] = {"weight": weights[wk], "weighted_mean": weights[wk] * out_terms[t]["mean"]}
    return {"format": FORMAT_TERMS, "run_dir": str(d.resolve()), "stage_key": stage_key,
            "telemetry": {"file": tel.name, "sha256": G.sha256_file(tel), "rows": len(rows)},
            "run_meta": {"file": meta_p.name, "sha256": G.sha256_file(meta_p), "stage": meta.get("stage"),
                         "seed": meta.get("seed"), "mode": meta.get("mode"), "terms": meta.get("terms")},
            "run_weights": weights, "run_end": {"iter": max_iter, "checks_passed": ends[0].get("checks_passed")},
            "window": {"last": last, "first_iter": lo, "last_iter": max_iter, "rows": len(window)},
            "terms": out_terms, "weighted_contributions": contrib, "alpha_a1pp": alpha}


# ------------------------------------------------------------------ (b)
def _read_json(path: Path) -> tuple[dict | None, str | None, str | None]:
    if G.names_test(path):
        return None, None, f"{path} holds 'test' (SL-1); not read"
    if not path.is_file():
        return None, None, f"{path} is missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), G.sha256_file(path), None
    except (OSError, ValueError) as e:
        return None, None, f"{path} is not readable JSON ({type(e).__name__}: {e})"


def _cell(value, source: dict) -> dict:
    if _finite_number(value):
        return {"value": float(value), **source}
    return {"value": None, **source, "reason": source.get("reason") or f"no finite value ({value!r})"}


def qat_row(spec: dict) -> dict:
    """One model's row. spec: label, and any of parent_best_json, qat_run_dir, selection, fqoff_summary."""
    row = {"label": spec.get("label")}
    # parent
    if spec.get("parent_best_json"):
        doc, sha, err = _read_json(Path(spec["parent_best_json"]))
        row["parent"] = _cell(None if doc is None else doc.get("best_val_miou_all_class"),
                              {"file": spec["parent_best_json"], "sha256": sha, "reason": err})
    else:
        row["parent"] = {"value": None, "reason": "no parent best.json given"}
    # selection: converted VAL and the selected epoch
    epoch = None
    if spec.get("selection"):
        doc, sha, err = _read_json(Path(spec["selection"]))
        win = (doc or {}).get("winner") or {}
        epoch = win.get("epoch") if isinstance(win.get("epoch"), int) else None
        row["converted"] = _cell(win.get("value"), {"file": spec["selection"], "sha256": sha,
                                                     "format": (doc or {}).get("format"), "reason": err})
        row["seed"] = (doc or {}).get("seed")
        row["run_id"] = (doc or {}).get("run_id")
    else:
        row["converted"] = {"value": None, "reason": "no qat_selection.json given"}
    row["selected_epoch"] = epoch
    # fake-quant VAL at the selected epoch, from the QAT telemetry's epoch_end row
    if spec.get("qat_run_dir") and epoch is not None:
        tel = Path(spec["qat_run_dir"]) / "qat_telemetry.jsonl"
        if G.names_test(tel) or not tel.is_file():
            row["fake_quant"] = {"value": None, "reason": f"{tel} is missing or refused"}
        else:
            try:
                ends = [r for r in _strict_rows(tel) if r.get("event") == "epoch_end" and r.get("epoch") == epoch]
            except DecomposeError as e:
                ends, why = [], str(e)
            else:
                why = None
            v = ends[0].get("fake_quant_val_all_class_miou") if len(ends) == 1 else None
            row["fake_quant"] = _cell(v, {"file": str(tel), "sha256": G.sha256_file(tel), "epoch": epoch,
                                          "reason": why or (None if len(ends) == 1 else
                                                            f"{len(ends)} epoch_end rows for epoch {epoch}")})
    else:
        row["fake_quant"] = {"value": None, "reason": "no QAT run dir given, or no selected epoch"}
    # fake-quant-off VAL
    if spec.get("fqoff_summary"):
        doc, sha, err = _read_json(Path(spec["fqoff_summary"]))
        split = ((doc or {}).get("dataset") or {}).get("split")
        v = ((doc or {}).get("dataset_level") or {}).get("all_class_miou")
        reason = err or (None if split == "val" else f"the summary's split is {split!r}, not 'val'; not used")
        row["fake_quant_off"] = _cell(v if split == "val" else None, {"file": spec["fqoff_summary"], "sha256": sha,
                                                                       "reason": reason})
    else:
        row["fake_quant_off"] = {"value": None, "reason": "no fake-quant-off summary.json given"}

    def diff(a, b):
        va, vb = row[a]["value"], row[b]["value"]
        return None if va is None or vb is None else 100.0 * (va - vb)
    row["pp"] = {"parent_minus_fake_quant": diff("parent", "fake_quant"),
                 "fake_quant_minus_converted": diff("fake_quant", "converted"),
                 "fake_quant_off_minus_converted": diff("fake_quant_off", "converted"),
                 "parent_minus_converted": diff("parent", "converted")}
    return row


def qat_table(specs: list[dict]) -> dict:
    if not isinstance(specs, list) or not specs:
        raise DecomposeError("spec_empty", "the model spec must be a non-empty JSON list")
    return {"format": FORMAT_TABLE, "rows": [qat_row(s) for s in specs],
            "columns": ["parent", "fake_quant (selected epoch)", "converted (selected)", "fake_quant_off"],
            "note": ("every converted value of record is the best of 15 converted epochs on VAL (DL-139/146); the "
                     "fake-quant value is the QAT run's in-training VAL (GPU) of the same epoch")}


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def table_text(tab: dict) -> str:
    head = (f"{'model':<14} {'ep':>3} {'parent':>8} {'fq':>8} {'conv':>8} {'fqoff':>8} "
            f"{'par-fq':>7} {'fq-cv':>7} {'off-cv':>7}")
    lines = [head, "-" * len(head)]
    for r in tab["rows"]:
        lines.append(f"{str(r['label']):<14} {str(r['selected_epoch'] or '—'):>3} {_fmt(r['parent']['value']):>8} "
                     f"{_fmt(r['fake_quant']['value']):>8} {_fmt(r['converted']['value']):>8} "
                     f"{_fmt(r['fake_quant_off']['value']):>8} {_fmt(r['pp']['parent_minus_fake_quant'], 2):>7} "
                     f"{_fmt(r['pp']['fake_quant_minus_converted'], 2):>7} "
                     f"{_fmt(r['pp']['fake_quant_off_minus_converted'], 2):>7}")
    return "\n".join(lines)


def terms_text(res: dict) -> str:
    lines = [f"run {res['run_dir']} ({res['run_meta']['stage']}, seed {res['run_meta']['seed']}); window "
             f"{res['window']['first_iter']}..{res['window']['last_iter']} ({res['window']['rows']} rows); telemetry "
             f"sha256 {res['telemetry']['sha256']}"]
    for t, v in res["terms"].items():
        lines.append(f"  {t}: " + (f"mean {v['mean']!r} (min {v['min']!r}, max {v['max']!r})" if v["status"] == "ok"
                                   else f"{v['status'].upper()}: {v['reason']}"))
    a = res["alpha_a1pp"]
    lines.append("  A1'' alpha: " + (f"{a['repr']} = {a['lambda']!r} * {a['mean_L_logitKD']!r} / {a['mean_L_feat']!r}"
                                     if a["status"] == "ok" else f"NOT FORMED: {a['reason']}"))
    return "\n".join(lines)
