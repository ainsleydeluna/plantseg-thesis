#!/usr/bin/env python3
"""Smoke: D3, the teacher's per-class TRAIN and VAL IoU by stratum (scripts/teacher_d3_perclass_strata.py;
lane L-TEACHER-DIAG).

Provisional synthetic inputs, as S3's smokes build them: a strata file (scripts/teacher_diag_fixtures.
write_strata), a TRAIN scores JSON whose per-class supports reproduce its dataset-level mIoU under the
evaluator's float32 arithmetic, and S3's VAL table built by src.stats.perclass.perclass_rows itself (class
69 without VAL ground truth). Checked: rows (stratum, rare, both IoUs, TRAIN - VAL), not-evaluable classes
kept out of every summary, the group summaries against an independent fsum mean and median (P20), the CSV
and its sha256; refusals of a smoke TRAIN input (P2), another teacher, another strata file or split list,
--generated-utc (C2), a commit other than HEAD (P26), an in-repo out dir and a second output (P27); a
TRAIN table that does not reproduce its mIoU is a STOP. Git answers are faked (fake_git), as the D0 OK
allows. No model, no PlantSeg data, no GPU.

    python -B scripts/smoke_teacher_d3.py
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
HEAD = "c" * 40


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def per_class(tp, gt, pred) -> dict:
    un = gt + pred - tp
    return {"class_ids": list(range(116)), "gt_support": gt.tolist(), "pred_support": pred.tolist(),
            "intersection": tp.tolist(), "union": un.tolist(),
            "iou": [(int(a) / int(u)) if u > 0 else None for a, u in zip(tp, un)],
            "iou_status": ["ok" if u > 0 else "undefined_absent_from_gt_and_pred" for u in un]}


def dataset_level(tp, gt, pred) -> dict:
    from src.stats.eligibility import UNION_PRESENT, ClassTotals, dataset_miou
    tot = ClassTotals(tp=tp, gt=gt, pred=pred)
    a = dataset_miou(tot, range(116), UNION_PRESENT)
    d = dataset_miou(tot, range(1, 116), UNION_PRESENT)
    return {"all_class_miou": a.value, "all_class_miou_n_eligible": a.n_classes,
            "disease_only_miou": d.value, "disease_only_miou_n_eligible": d.n_classes}


def fixtures(tmp: Path) -> dict:
    from scripts import teacher_diag_fixtures as fx
    from src.eval import teacher_diag as td
    from src.stats.perclass import load_strata, perclass_rows, strata_index
    stems = [f"plant_leaf_{i:04d}" for i in range(td.TRAIN_ROWS)]
    strata_p = fx.write_strata(tmp / "train_strata_v1.json", stems)
    strata_doc, strata_sha = load_strata(strata_p)
    rng = np.random.default_rng(3)
    gt = rng.integers(2000, 90000, size=116).astype(np.int64)          # every class present on TRAIN
    tp = (gt * rng.uniform(0.05, 0.9, size=116)).astype(np.int64)
    pred = tp + rng.integers(0, 40000, size=116).astype(np.int64)
    train = {"lane": td.LANE, "script": "scripts/score_teacher_train.py", "schema": "plantseg-teacher-train-scores/1.0.0",
             "artifact_status": "provisional", "mode": "real", "status": "written",
             "teacher": {"role": "record", "checkpoint": {"sha256_verified": td.RECORD_SHA256},
                         "loaded_state_sha256": "a" * 64},
             "inputs": {"strata": {"sha256": strata_sha, "split_list_sha256": strata_doc["split_list_sha256"]}},
             "pass": {"split": "train", "rows": td.TRAIN_ROWS},
             "per_class": per_class(tp, gt, pred), "dataset_level": dataset_level(tp, gt, pred),
             "artifact": {"files_sha256": {"summary.json": "b" * 64}}}
    vgt = rng.integers(100, 9000, size=116).astype(np.int64)
    vgt[69] = 0                                                           # class 69: no VAL ground truth
    vtp_t = (vgt * rng.uniform(0.0, 0.8, size=116)).astype(np.int64)
    vpr_t = vtp_t + rng.integers(0, 3000, size=116).astype(np.int64)
    vtp_e = (vgt * rng.uniform(0.0, 0.7, size=116)).astype(np.int64)
    vpr_e = vtp_e + rng.integers(0, 3000, size=116).astype(np.int64)
    strata = strata_index(strata_doc, range(1, 116))
    rows, background = perclass_rows(per_class(vtp_t, vgt, vpr_t), per_class(vtp_e, vgt, vpr_e), strata)
    val = {"lane": "L-AM17-PERCLASS", "artifact_status": "provisional", "split": "val",
           "inputs": {"teacher": {"stage": "teacher", "checkpoint_sha256": td.RECORD_SHA256},
                      "strata": {"sha256": strata_sha}},
           "rows": rows, "background": background, "csv_sha256": "c" * 64}
    tp_p, val_p = tmp / "teacher_train_scores_20261002T000000Z.json", tmp / "perclass_gap_val_20261002T000000Z.json"
    tp_p.write_text(json.dumps(train, indent=2))
    val_p.write_text(json.dumps(val, indent=2))
    return {"strata": strata_p, "train": tp_p, "val": val_p, "train_doc": train, "val_doc": val,
            "strata_doc": strata_doc}


def cases(tmp: Path) -> None:
    from scripts import teacher_d3_perclass_strata as d3
    from scripts import teacher_diag_fixtures as fx
    from src.eval import teacher_diag as td
    f = fixtures(tmp)
    present = tuple(r for r in td.CODE_FILES if (td.REPO / r).is_file())

    def call(out, **kw):
        flags = dict(out_dir=out, train_scores=f["train"], perclass_val=f["val"], strata=f["strata"],
                     script_commit=HEAD, script_commit_dl_id="DL-64")
        flags.update(kw)
        with fx.patched(td, CODE_FILES=present), fx.fake_git(head=HEAD):
            return fx.call_run(d3, factory=None, **flags)

    out = tmp / "out"
    code, err = call(out)
    files = fx.output_files(out)
    names = [p.name for p in files]
    doc = json.loads([p for p in files if p.suffix == ".json"][0].read_text()) if files else {}
    check("D3 exit 0: teacher_d3_perclass_<UTC>.json and .csv, status written",
          code == 0 and len(files) == 2 and all(n.startswith("teacher_d3_perclass_") for n in names)
          and doc.get("status") == "written", err)
    rows = {r["class_id"]: r for r in doc.get("rows", [])}
    pc, vrows = f["train_doc"]["per_class"], {r["class_id"]: r for r in f["val_doc"]["rows"]}
    ok = sorted(rows) == list(range(1, 116)) and all(
        rows[c]["iou_train"] == pc["iou"][c] and rows[c]["iou_val"] == (vrows[c]["iou_teacher"] if c != 69 else None)
        and rows[c]["stratum"] == vrows[c]["stratum"] and rows[c]["rare"] == vrows[c]["rare"]
        and rows[c]["train_gt_support"] == pc["gt_support"][c] and rows[c]["val_gt_support"] == vrows[c]["gt_support"]
        for c in rows)
    check("D3 rows: one per disease class with stratum, rare, both supports and both IoUs", ok)
    check("D3 TRAIN - VAL = iou_train - iou_val; class 69 (no VAL ground truth) is not evaluable on VAL, "
          "difference null",
          all(rows[c]["train_minus_val"] == rows[c]["iou_train"] - rows[c]["iou_val"] for c in rows if c != 69)
          and rows[69]["val_status"] == "not_evaluable" and rows[69]["train_minus_val"] is None
          and rows[69]["train_status"] == "ok")
    s = doc.get("summaries", {})
    ok = True
    for g, sel in (("T1", lambda r: r["stratum"] == "T1"), ("rare", lambda r: r["rare"] is True),
                   ("overall", lambda r: True)):
        mem = [r for r in rows.values() if sel(r)]
        for split, key, filt in (("train", "iou_train", lambda r: r["train_status"] == "ok"),
                                 ("val", "iou_val", lambda r: r["val_status"] == "ok"),
                                 ("train_minus_val", "train_minus_val", lambda r: r["train_minus_val"] is not None)):
            vals = [r[key] for r in mem if filt(r)]
            got = s.get(g, {}).get(split, {})
            ok &= (got.get("n") == len(vals) and got.get("mean") == math.fsum(vals) / len(vals)
                   and got.get("median") == float(np.median(vals)))
    check("P20 group summaries (T1, rare, overall): n, fsum mean and median on TRAIN, on VAL and of TRAIN - VAL "
          "equal an independent computation", ok)
    check("P20 class 69 enters no VAL summary and no difference summary, and is listed as not evaluable on VAL",
          69 not in s.get("overall", {}).get("val", {}).get("classes", [69])
          and 69 in s.get("overall", {}).get("train", {}).get("classes", [])
          and s.get("overall", {}).get("not_evaluable_val") == [69] and s.get("overall", {}).get("val", {}).get("n") == 114)
    csv_p = [p for p in files if p.suffix == ".csv"]
    csv_ok = bool(csv_p) and hashlib.sha256(csv_p[0].read_bytes()).hexdigest() == doc.get("csv", {}).get("sha256")
    if csv_ok:
        lines = list(csv.reader(io.StringIO(csv_p[0].read_text())))
        csv_ok = lines[0] == list(d3.COLUMNS) and len(lines) == 1 + 115 + 1 and lines[-1][1] == "background"
    check("D3 the CSV: columns, 115 rows plus the background line, sha256 recorded in the JSON", csv_ok)
    check("D3 TRAIN integrity: the supports reproduce the dataset-level mIoU bitwise; input hashes recorded",
          doc.get("train_integrity", {}).get("all_class", {}).get("bitwise_equal") is True
          and doc.get("inputs", {}).get("train_scores", {}).get("sha256") == td.file_sha256(f["train"])
          and doc.get("inputs", {}).get("perclass_val", {}).get("sha256") == td.file_sha256(f["val"]))

    # refusals and the STOP
    def variant(name, doc_key, mutate):
        d = copy.deepcopy(f[doc_key])
        mutate(d)
        p = tmp / name
        p.write_text(json.dumps(d))
        return p
    smoke_t = variant("t_smoke.json", "train_doc", lambda d: d.update(artifact_status="smoke"))
    other_t = variant("t_other.json", "train_doc", lambda d: d["teacher"]["checkpoint"].update(sha256_verified="1" * 64))
    other_split = variant("t_split.json", "train_doc", lambda d: d["inputs"]["strata"].update(split_list_sha256="2" * 64))
    other_v = variant("v_strata.json", "val_doc", lambda d: d["inputs"]["strata"].update(sha256="3" * 64))
    other_vt = variant("v_teacher.json", "val_doc", lambda d: d["inputs"]["teacher"].update(checkpoint_sha256="4" * 64))
    bad_dl = variant("t_dl.json", "train_doc", lambda d: d["dataset_level"].update(
        all_class_miou=d["dataset_level"]["all_class_miou"] + 1e-6))
    r = [call(tmp / "o1", train_scores=smoke_t)[0], call(tmp / "o2", train_scores=other_t)[0],
         call(tmp / "o3", train_scores=other_split)[0], call(tmp / "o4", perclass_val=other_v)[0],
         call(tmp / "o5", perclass_val=other_vt)[0]]
    check("P2/D3 refused (exit 2): a smoke TRAIN input, another teacher on either table, another strata file, "
          "another TRAIN split list", r == [2] * 5, str(r))
    code, _ = call(tmp / "o6", train_scores=bad_dl)
    check("D3 a TRAIN table that does not reproduce its mIoU is a STOP (exit 1), nothing written",
          code == 1 and not fx.output_files(tmp / "o6"))
    with fx.patched(td, CODE_FILES=present), fx.fake_git(head="d" * 40):
        r_head = fx.call_run(d3, factory=None, out_dir=tmp / "o7", train_scores=f["train"], perclass_val=f["val"],
                             strata=f["strata"], script_commit=HEAD, script_commit_dl_id="DL-64")[0]
    r_utc = call(tmp / "o8", generated_utc="2026-10-02T00:00:00Z")[0]
    r_nocommit = call(tmp / "o9", script_commit=None)[0]
    r_repo = call(td.REPO / "d3_out_in_repo")[0]
    r_second = call(out)[0]
    json_out = [p for p in files if p.suffix == ".json"][0] if files else out / "missing.json"
    with fx.patched(td, utc_now=lambda: "2099-01-01T00:00:00Z"):         # a later clock: a new output name
        r_repeat = call(out, repeat_of=td.file_sha256(json_out) if json_out.is_file() else "0" * 64, repeat_case="i",
                        repeat_dl_id="DL-65")[0]
    check("P26/C2/P27 refused: HEAD other than --script-commit, --generated-utc, no --script-commit, an in-repo "
          "out dir, a second output; --repeat-of naming the first output is accepted",
          (r_head, r_utc, r_nocommit, r_repo, r_second, r_repeat) == (2, 2, 2, 2, 2, 0)
          and not (td.REPO / "d3_out_in_repo").exists(), str((r_head, r_utc, r_nocommit, r_repo, r_second, r_repeat)))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m_usage = d3.main([])
    check("P29 a usage error exits 2", m_usage == 2)


def main() -> int:
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    tmp = safe_tmpdir("diag_d3_")
    try:
        cases(tmp)
    except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
        check("cases raised", False, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
