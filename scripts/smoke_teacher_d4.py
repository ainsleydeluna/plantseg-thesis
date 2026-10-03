#!/usr/bin/env python3
"""Smoke: D4, the duplicate-free VAL (scripts/hash_split_files.py, scripts/dedup_val_scores.py; lane
L-TEACHER-DIAG; acceptance (f) as amended by P21-P25 and P32(f)).

Hasher, on a synthetic 5,367/846 root with planted copies: SHA-256 and SHA-1 equal hashlib on the bytes;
two TRAIN-VAL groups (three images with identical masks; two images with different masks), one VAL-only
group (reported, not removed), a re-encoded copy not grouped; the VAL manifest hash equals the
evaluator's; the TRAIN split list equals the strata file's; it lists exactly the four folders; an as-given
"test" path is refused with zero lstat/stat/scandir/listdir/open calls; names containing "test" are
refused by count and sha256; --names-check prints counts only and a count above 0 is a STOP (R4); only
sha256s and counts are printed.

Reducer, on VAL re-scores written by the real evaluation core and writer at a throwaway git pin (S3's
pattern; PAIRING pointed at synthetic identities): a class whose only ground truth is in a duplicate image
changes the eligible set and flips the margin's sign; the subset mIoU equals an independent recomputation;
an empty duplicate list reproduces the full-VAL scores exactly (both rules); margin 0 gives le_0 true; a
sign disagreement (unit and end to end) exits 1; a second teacher artifact that passes check_pairing is
refused; the P23 refusals; a full set that does not reproduce S3's point is a STOP; --single (P25).
Synthetic inputs only; no PlantSeg data, no checkpoint, no GPU.

    python -B scripts/smoke_teacher_d4.py
"""
from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
HEAD = "c" * 40
X = 50                                             # the class whose only VAL ground truth is a duplicate


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def run_quiet(module, **flags):
    from scripts import teacher_diag_fixtures as fx
    with fx.fake_git(head=HEAD):
        return fx.call_run(module, factory=None, **flags)


# ---------------------------------------------------------------------------------------------------
def plant_copies(root: Path, stems: dict) -> dict:
    from PIL import Image
    tr, va = stems["train"], stems["val"]
    img, ann = root / "images", root / "annotations"
    for v in (va[3], va[4]):                                   # group A: t10, v3, v4; identical masks
        shutil.copyfile(img / "train" / f"{tr[10]}.jpg", img / "val" / f"{v}.jpg")
        shutil.copyfile(ann / "train" / f"{tr[10]}.png", ann / "val" / f"{v}.png")
    shutil.copyfile(img / "train" / f"{tr[20]}.jpg", img / "val" / f"{va[5]}.jpg")   # group B: t20, v5
    m = np.asarray(Image.open(ann / "train" / f"{tr[20]}.png")).copy()
    m[m > 0] = 1 + (int(m.max()) % 114)                       # a different mask for v5
    Image.fromarray(m.astype(np.uint8), "L").save(ann / "val" / f"{va[5]}.png", format="PNG")
    shutil.copyfile(img / "val" / f"{va[7]}.jpg", img / "val" / f"{va[8]}.jpg")      # VAL-only: v7, v8
    with Image.open(img / "train" / f"{tr[30]}.jpg") as im:                             # re-encoded: t30 -> v9
        im.convert("RGB").save(img / "val" / f"{va[9]}.jpg", format="JPEG", quality=95)
    return {"A": (tr[10], [va[3], va[4]]), "B": (tr[20], [va[5]]), "val_only": [va[7], va[8]], "reenc": (tr[30], va[9])}


def hasher_cases(tmp: Path, root: Path, stems: dict, strata: Path) -> Path:
    from scripts import hash_split_files as hs
    from scripts import teacher_diag_fixtures as fx
    from src.eval.adapters import build_expected_manifest_for
    from src.eval.artifacts import am5_excluded_ids_sha256, hash_split_manifest
    planted = plant_copies(root, stems)
    flags = dict(data_root=str(root), strata=str(strata), script_commit=HEAD, script_commit_dl_id="DL-66")
    out = tmp / "hashes"
    buf = io.StringIO()
    with fx.fake_git(head=HEAD), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = hs.run(hs.build_parser().parse_args(fx.argv(out=out, **flags)))
    printed = buf.getvalue()
    jp, tp = out / hs.JSON_NAME, out / hs.TSV_NAME
    doc = json.loads(jp.read_text()) if jp.is_file() else {}
    check("f hasher: exit 0, the TSV and the JSON, status written, provisional",
          code == 0 and jp.is_file() and tp.is_file() and doc.get("status") == "written"
          and doc.get("artifact_status") == "provisional", printed)
    check("P22 only sha256s and counts are printed (no stem, no path)",
          "plant_" not in printed and str(root) not in printed and doc.get("hash_list", {}).get("sha256", "x") in printed)
    rows = [ln.split("\t") for ln in tp.read_text().splitlines()[1:]] if tp.is_file() else []
    ok = len(rows) == 5367 + 846
    for split, stem, name, nbytes, sha, sha1 in rows[:3] + rows[-3:] + [r for r in rows if r[1] == planted["A"][1][0]]:
        data = (root / "images" / split / name).read_bytes()
        ok &= hashlib.sha256(data).hexdigest() == sha and hashlib.sha1(data).hexdigest() == sha1 and int(nbytes) == len(data)
    check("f SHA-256 and SHA-1 equal hashlib on the file bytes; one row per image; the TSV sha256 is recorded",
          ok and doc.get("hash_list", {}).get("sha256") == hashlib.sha256(tp.read_bytes()).hexdigest() if tp.is_file() else False)
    groups = {tuple(g["val"]): g for g in doc.get("groups", [])}
    ga, gb = groups.get(tuple(planted["A"][1])), groups.get(tuple(planted["B"][1]))
    check("f byte-identical copies are grouped: (t10, v3, v4) and (t20, v5); sizes recorded",
          ga is not None and gb is not None and ga["train"] == [planted["A"][0]] and ga["size"] == 3
          and gb["train"] == [planted["B"][0]] and doc.get("group_sizes") == {"3": 1, "2": 1})
    check("f mask identity per group: identical for (t10, v3, v4), different for (t20, v5)",
          ga is not None and gb is not None and ga["masks_identical"] is True and gb["masks_identical"] is False)
    check("f a re-encoded copy (t30 -> v9) is not grouped; the output says byte-identical only",
          all(planted["reenc"][1] not in g["val"] for g in doc.get("groups", []))
          and "re-encoded" in doc.get("detection", ""))
    vo = doc.get("val_only_groups", [])
    check("P22/P32(f) the VAL-only group (v7, v8) is reported and not removed",
          [g["val"] for g in vo] == [planted["val_only"]] and not set(planted["val_only"]) & set(doc.get("duplicate_val_ids", [])))
    dup = sorted(planted["A"][1] + planted["B"][1])
    check("f duplicate VAL ids and their sha256 (AM-5 convention); issue #11's counts reported beside, not a STOP",
          doc.get("duplicate_val_ids") == dup and doc.get("duplicate_val_ids_sha256") == am5_excluded_ids_sha256(dup)
          and doc.get("issue_11", {}).get("this_run") == {"train_val_groups": 2, "three_way_or_more": 1}
          and doc.get("issue_11", {}).get("differs") is True)
    check("P22 the VAL manifest hash equals the evaluator's; the TRAIN split list equals the strata file's",
          doc.get("val_manifest", {}).get("hash_split_manifest") == hash_split_manifest(build_expected_manifest_for("val", range(846)))
          and doc.get("train_split_list_equals_strata") is True)

    # only the four folders are listed
    listed = []
    real_scandir, real_listdir = os.scandir, os.listdir

    def rec_scandir(p=".", *a, **k):
        listed.append(os.fspath(p))
        return real_scandir(p, *a, **k)

    def rec_listdir(p=".", *a, **k):
        listed.append(os.fspath(p))
        return real_listdir(p, *a, **k)
    with fx.patched(os, scandir=rec_scandir, listdir=rec_listdir):
        code, _ = run_quiet(hs, out=tmp / "hashes2", **flags)
    want = sorted(os.path.join(str(root), f) for f in hs.FOLDERS)
    check("f the hasher lists exactly the four folders it is given (never the root or another folder)",
          code == 0 and sorted(set(listed)) == want, str(sorted(set(listed)))[:300])

    # an as-given "test" path: zero filesystem calls
    calls = []

    def counting(fn, name):
        def inner(*a, **k):
            calls.append(name)
            return fn(*a, **k)
        return inner
    bad = str(tmp / ("x" + "TeSt" + "x"))
    patches = dict(lstat=counting(os.lstat, "lstat"), stat=counting(os.stat, "stat"),
                   scandir=counting(os.scandir, "scandir"), listdir=counting(os.listdir, "listdir"))
    with fx.patched(os, **patches), fx.patched(builtins, open=counting(builtins.open, "open")), \
            fx.patched(io, open=counting(io.open, "io.open")):
        r1, err1 = run_quiet(hs, out=tmp / "h3", **dict(flags, data_root=bad))
        r2, err2 = run_quiet(hs, data_root=bad, names_check=True)
        r3, err3 = run_quiet(hs, out=str(tmp / ("o" + "test")), **flags)
    check("P21/P32(f) an as-given 'test' path (data root, names-check root, --out) is refused with zero lstat, "
          "stat, scandir, listdir and open calls; the message has a count and sha256s, not the name",
          (r1, r2, r3) == (2, 2, 2) and not calls and "TeSt" not in err1 + err2 and "offending entr" in err1,
          f"{(r1, r2, r3)} calls={calls[:5]}")

    # names containing "test": the names-check (R4) and the hashing mode
    real_list = hs.list_folder

    def with_name(folder):
        names = real_list(folder)
        return names + ["x_" + "te" + "st_1.jpg"] if folder.endswith(os.path.join("images", "val")) else names
    buf = io.StringIO()
    with fx.patched(hs, list_folder=real_list), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        clean = hs.run(hs.build_parser().parse_args(fx.argv(data_root=str(root), names_check=True)))
    clean_out = buf.getvalue()
    buf = io.StringIO()
    with fx.patched(hs, list_folder=with_name), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        dirty = hs.run(hs.build_parser().parse_args(fx.argv(data_root=str(root), names_check=True)))
    dirty_out = buf.getvalue()
    with fx.patched(hs, list_folder=with_name):
        r_hash, err_hash = run_quiet(hs, out=tmp / "h4", **flags)
    check("R4 --names-check prints only the TRAIN and VAL counts (0, 0 -> exit 0; a VAL name with 'test' -> "
          "count 1, STOP exit 1) and writes nothing",
          clean == 0 and "TRAIN names containing 'test': 0; VAL names containing 'test': 0" in clean_out
          and dirty == 1 and "VAL names containing 'test': 1" in dirty_out and "x_" not in dirty_out)
    check("P21 the hashing mode refuses a listed name containing 'test' by count and sha256",
          r_hash == 2 and "1 offending entry" in err_hash and "x_" not in err_hash, err_hash)
    other_strata = tmp / "strata_other.json"
    fx.write_strata(other_strata, stems["train"][:-1] + ["plant_leaf_zzzz"])
    dropped = lambda folder: real_list(folder)[:-1] if folder.endswith(os.path.join("images", "val")) else real_list(folder)  # noqa: E731
    with fx.patched(hs, list_folder=dropped):
        r_count = run_quiet(hs, out=tmp / "h5", **flags)[0]
    r = [r_count, run_quiet(hs, out=tmp / "h6", **dict(flags, strata=str(other_strata)))[0],
         run_quiet(hs, out=out, **flags)[0], run_quiet(hs, out=tmp / "h7", generated_utc="2026-10-02T00:00:00Z", **flags)[0],
         run_quiet(hs, out=tmp / "h8", **dict(flags, script_commit=None))[0],
         run_quiet(hs, out=REPO / "reports" / "d4_hashes", **flags)[0]]
    check("P21/P22/P27/C2 refused: 845 VAL images, another strata split list, an --out holding the outputs, "
          "--generated-utc, no --script-commit, an --out inside the repository (reports/)",
          r == [2] * 6 and not (REPO / "reports" / "d4_hashes").exists(), str(r))
    return jp


# ---------------------------------------------------------------------------------------------------
def label_maps(n: int = 846):
    dis = [c for c in range(1, 116) if c != X]
    gts = torch.zeros(n, 4, 4, dtype=torch.long)
    for i in range(n):
        gts[i, 1:3, 1:3] = X if i == 3 else dis[i % len(dis)]
    t, e = gts.clone(), gts.clone()
    for i in (100, 200, 300):
        t[i, 1, 1] = 0                                            # the teacher misses three pixels
    e[3, 1:3, 1:3] = 0                                            # E1 misses class X in image 3
    return gts, t, e


def miou64(gts, preds, keep) -> float:
    """Independent float64 union-present all-class mIoU over the kept images."""
    g, p = gts[keep].reshape(-1), preds[keep].reshape(-1)
    terms = []
    for c in range(116):
        tp = int(((g == c) & (p == c)).sum())
        un = int((g == c).sum()) + int((p == c).sum()) - tp
        if un > 0:
            terms.append(tp / un)
    return math.fsum(terms) / len(terms)


def reducer_cases(tmp: Path, stems: dict, dup_json: Path) -> None:
    from scripts import dedup_val_scores as d4
    from scripts import teacher_diag_fixtures as fx
    from src.eval.artifacts import am5_excluded_ids_sha256
    from src.eval import teacher_diag as td
    from src.stats.val_artifacts import load_val_artifact
    import scripts.gap_bootstrap_val as gap

    ids = list(stems["val"])
    pin = fx.clean_git_pin(tmp / "pin_repo")
    gts, pt, pe = label_maps()
    rt, man = fx.core_result(ids, gts, pt)
    re_, _ = fx.core_result(ids, gts, pe)
    a_t = fx.write_rescore(tmp / "teacher_cpu", rt, man, stage="teacher", role="teacher", sha=fx.SYN_SHA_T,
                           digest=fx.SYN_DIG_T, pin=pin)
    a_e = fx.write_rescore(tmp / "e1_cpu", re_, man, stage="E1", role="student", sha=fx.SYN_SHA_E,
                           digest=fx.SYN_DIG_E, pin=pin)
    a_e0 = fx.write_rescore(tmp / "e1_same", rt, man, stage="E1", role="student", sha=fx.SYN_SHA_E,
                            digest=fx.SYN_DIG_E, pin=pin)
    a_t2 = fx.write_rescore(tmp / "teacher_cpu_again", rt, man, stage="teacher", role="teacher", sha=fx.SYN_SHA_T,
                            digest=fx.SYN_DIG_T, pin=pin)
    v_t = json.loads((a_t / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    v_e = json.loads((a_e / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    dup_sha = td.file_sha256(dup_json)
    dup_ids = json.loads(dup_json.read_text())["duplicate_val_ids"]

    with fx.synthetic_pairing(v_t, v_e):
        art_t, art_e = load_val_artifact(a_t, label="t"), load_val_artifact(a_e, label="e")
        g1 = fx.gap_output(tmp / "gap_val_1.json", art_t, art_e)
        base = dict(teacher=a_t, e1=a_e, gap_output=g1, duplicates=dup_json, duplicates_sha256=dup_sha,
                    script_commit=HEAD, script_commit_dl_id="DL-67")
        code, err = run_quiet(d4, out_dir=tmp / "r_main", **base)
        files = fx.output_files(tmp / "r_main")
        doc = json.loads(files[0].read_text()) if files else {}
        check("f reducer: exit 0, teacher_d4_dedup_<UTC>.json, status written",
              code == 0 and len(files) == 1 and files[0].name.startswith("teacher_d4_dedup_")
              and doc.get("status") == "written", err)
        keep = torch.ones(846, dtype=torch.bool)
        keep[[ids.index(i) for i in dup_ids]] = False
        full_m = miou64(gts, pt, torch.ones(846, dtype=torch.bool)) - miou64(gts, pe, torch.ones(846, dtype=torch.bool))
        sub_m = miou64(gts, pt, keep) - miou64(gts, pe, keep)
        check("f margin_dedup equals an independent float64 recomputation on the subset; the full-set margin "
              "equals S3's point",
              doc.get("margin_dedup") == sub_m and doc.get("full_set", {}).get("float64_teacher_minus_e1") == full_m
              and doc.get("full_set", {}).get("abs_diff", 1) <= 1e-12, f"{doc.get('margin_dedup')!r} vs {sub_m!r}")
        check("P32(f) a class whose only ground truth is in a duplicate image leaves the eligible set and flips "
              "the margin's sign (full > 0, subset <= 0)",
              full_m > 0 and doc.get("margin_dedup", 1) < 0 and doc.get("margin_dedup_le_0") is True
              and doc.get("teacher", {}).get("eligibility_changed", {}).get("union_present", {}).get("dropped") == [X]
              and doc.get("e1", {}).get("eligibility_changed", {}).get("gt_present", {}).get("dropped") == [X])
        check("P24 the three signs agree; n_total, n_dup, n_kept; the full float32 values equal summary.json",
              doc.get("signs_agree") is True and set(doc.get("signs", {}).values()) == {-1}
              and (doc.get("teacher", {}).get("n_total"), doc.get("teacher", {}).get("n_dup"),
                   doc.get("teacher", {}).get("n_kept")) == (846, 3, 843)
              and doc.get("teacher", {}).get("full_float32_equals_summary") is True)
        check("P24 the exact margin (Fraction) agrees with the float64 margin within 1e-15",
              abs(float(Fraction(doc.get("exact_margin", "1/1"))) - doc.get("margin_dedup", 0)) <= 1e-15)
        check("P24 the GT-present margin is written", isinstance(doc.get("gt_present_margin"), float))

        # an empty duplicate list reproduces the full-VAL scores exactly
        empty = json.loads(dup_json.read_text())
        empty["duplicate_val_ids"] = []
        empty["duplicate_val_ids_sha256"] = am5_excluded_ids_sha256([])
        ep = tmp / "dups_empty.json"
        ep.write_text(json.dumps(empty))
        code, err = run_quiet(d4, out_dir=tmp / "r_empty", **dict(base, duplicates=ep, duplicates_sha256=td.file_sha256(ep)))
        files = fx.output_files(tmp / "r_empty")
        ed = json.loads(files[0].read_text()) if files else {}
        check("f an empty duplicate list reproduces the full-VAL scores exactly (union-present and GT-present)",
              code == 0 and ed.get("teacher", {}).get("subset") == ed.get("teacher", {}).get("full")
              and ed.get("e1", {}).get("subset") == ed.get("e1", {}).get("full")
              and ed.get("margin_dedup") == ed.get("full_set", {}).get("float64_teacher_minus_e1"), err)

        # sign disagreement: unit and end to end
        unit = d4.decide(1e-9, 0.0, Fraction(1, 10 ** 9))
        with fx.patched(d4, float32_margin=lambda t, e: 1.0):
            code, _ = run_quiet(d4, out_dir=tmp / "r_sign", **base)
        files = fx.output_files(tmp / "r_sign")
        sd = json.loads(files[0].read_text()) if files else {}
        check("P24 sign disagreement: decide() flags it; end to end the status is 'sign disagreement', exit 1",
              unit["margin_status"] == "sign disagreement" and unit["signs_agree"] is False
              and code == 1 and sd.get("status") == "sign disagreement")

        # a second teacher artifact that passes check_pairing is refused
        art_t2 = load_val_artifact(a_t2, label="t2")
        gap.check_pairing(art_t2, art_e)                          # it passes S3's pairing...
        check("P32(f) a second teacher artifact that passes check_pairing is refused (not the gap output's)",
              run_quiet(d4, out_dir=tmp / "r_t2", **dict(base, teacher=a_t2))[0] == 2)

        # P23 refusals
        def dup_variant(name, fn):
            d = json.loads(dup_json.read_text())
            fn(d)
            p = tmp / name
            p.write_text(json.dumps(d))
            return dict(duplicates=p, duplicates_sha256=td.file_sha256(p))
        r = [run_quiet(d4, out_dir=tmp / "p1", **dict(base, duplicates_sha256="0" * 64))[0],
             run_quiet(d4, out_dir=tmp / "p2", **dict(base, **dup_variant("d_smoke.json",
                                                                           lambda d: d.update(artifact_status="smoke"))))[0],
             run_quiet(d4, out_dir=tmp / "p3", **dict(base, **dup_variant(
                 "d_manifest.json", lambda d: d["val_manifest"].update(hash_split_manifest="0" * 64))))[0],
             run_quiet(d4, out_dir=tmp / "p4", **dict(base, **dup_variant(
                 "d_unknown.json", lambda d: d.update(duplicate_val_ids=sorted(d["duplicate_val_ids"] + ["zz_unknown"]),
                                                      duplicate_val_ids_sha256=am5_excluded_ids_sha256(
                                                          sorted(d["duplicate_val_ids"] + ["zz_unknown"]))))))[0]]
        g_lane = json.loads(g1.read_text())
        g_lane["lane"] = "L-OTHER"
        gl = tmp / "gap_lane.json"
        gl.write_text(json.dumps(g_lane))
        r.append(run_quiet(d4, out_dir=tmp / "p5", **dict(base, gap_output=gl))[0])
        with fx.patched(gap, PAIRING={**gap.PAIRING, "e1": {**gap.PAIRING["e1"], "reference": v_e + 1e-6}}):
            r.append(run_quiet(d4, out_dir=tmp / "p6", **base)[0])
        r.append(run_quiet(d4, out_dir=tmp / "p7", **dict(base, generated_utc="2026-10-02T00:00:00Z"))[0])
        r.append(run_quiet(d4, out_dir=tmp / "p8", **dict(base, single=str(a_t)))[0])
        check("P23 refused (exit 2): a wrong --duplicates-sha256, a smoke duplicates file, another VAL manifest, "
              "a duplicate id outside manifest_ids, another lane's gap output, validity() outside tolerance, "
              "--generated-utc, --single mixed with pair flags", r == [2] * 8, str(r))
        g_off = json.loads(g1.read_text())
        g_off["rules"]["union_present"]["point"] += 1e-9
        go = tmp / "gap_off.json"
        go.write_text(json.dumps(g_off))
        code, _ = run_quiet(d4, out_dir=tmp / "p9", **dict(base, gap_output=go))
        check("P24 a full set that does not reproduce S3's point (1e-9 > 1e-12) is a STOP, nothing written",
              code == 1 and not fx.output_files(tmp / "p9"))

        # a summary.json value the sufficient statistics do not reproduce
        a_te = tmp / "teacher_cpu_edited"
        shutil.copytree(a_t, a_te)
        sj = json.loads((a_te / "summary.json").read_text())
        sj["dataset_level"]["all_class_miou"] = float(np.nextafter(np.float32(v_t), np.float32(1.0)))
        (a_te / "summary.json").write_text(json.dumps(sj, indent=2) + "\n")
        fx.rehash_artifact(a_te)
        try:
            art_te = load_val_artifact(a_te, label="te")
            ge = fx.gap_output(tmp / "gap_edited.json", art_te, art_e)
            code, err = run_quiet(d4, out_dir=tmp / "p10", **dict(base, teacher=a_te, gap_output=ge))
        except Exception as e:  # noqa: BLE001 -- the ingest itself may refuse the edited summary
            code, err = None, f"{type(e).__name__}: {e}"
        check("P24 a full-set float32 value that differs from summary.json (one float32 step) is a STOP, "
              "nothing written", code == 1 and not fx.output_files(tmp / "p10"), f"{code} {err[-200:]}")

        # --single (P25)
        code, err = run_quiet(d4, out_dir=tmp / "r_single", single=a_t, duplicates=dup_json, duplicates_sha256=dup_sha,
                              script_commit=HEAD, script_commit_dl_id="DL-67")
        files = fx.output_files(tmp / "r_single")
        sg = json.loads(files[0].read_text()) if files else {}
        check("P25 --single: val_dedup_single_<UTC>.json, the same subset scores as the pair's teacher, gates nothing",
              code == 0 and files and files[0].name.startswith("val_dedup_single_")
              and sg.get("model", {}).get("subset") == doc.get("teacher", {}).get("subset")
              and "nothing" in sg.get("gates", ""), err)

    # margin = 0 gives le_0 true
    v_e0 = json.loads((a_e0 / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    with fx.synthetic_pairing(v_t, v_e0):
        art_t, art_e0 = load_val_artifact(a_t, label="t"), load_val_artifact(a_e0, label="e0")
        g0 = fx.gap_output(tmp / "gap_val_0.json", art_t, art_e0)
        code, err = run_quiet(d4, out_dir=tmp / "r_zero", **dict(base, e1=a_e0, gap_output=g0))
        files = fx.output_files(tmp / "r_zero")
        zd = json.loads(files[0].read_text()) if files else {}
    check("P32(f) margin = 0 gives margin_dedup_le_0 true (all three signs 0)",
          code == 0 and zd.get("margin_dedup") == 0.0 and zd.get("margin_dedup_le_0") is True
          and set(zd.get("signs", {}).values()) == {0}, err)


def main() -> int:
    from scripts import teacher_diag_fixtures as fx
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    tmp = safe_tmpdir("diag_d4_")
    root = None
    try:
        root, stems = fx.make_data_root("diag_d4_data_")
        fx.set_data_root(root)
        strata = fx.write_strata(tmp / "train_strata_v1.json", stems["train"])
        dup_json = hasher_cases(tmp, root, stems, strata)
        reducer_cases(tmp, stems, dup_json)
    except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
        import traceback
        check("cases raised", False, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if root is not None:
            shutil.rmtree(Path(root).parent, ignore_errors=True)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
