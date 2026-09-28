#!/usr/bin/env python3
"""L-AM5 zero-disease exclusion smoke (docs/lane_specs/part1.md lane 1 (d) d1-d3; NONOFFICIAL).

  D1  evaluator: 5 synthetic images through the real evaluation core and artifact writer, two with no
      disease ground truth -> am5_excluded true for exactly those two (appended last), summary am5
      {rule no_disease_gt, excluded_count 2, included_count 3, excluded_ids_sha256 sha256("id2\\nid4")},
      artifact_schema_version set, schema_version and config_sha256 unchanged.
  D2  ingest: OFFICIAL and REHEARSAL drop the two rows (n = 3); an undefined unflagged row aborts with
      its id in the message; two runs with different excluded sets abort as a pair; a pre-lane row set
      (field absent) derives flags identical to D1; inconsistent fields abort; full-size REHEARSAL (846
      rows) and OFFICIAL (1,561 rows) wiring, per run and per pair.
  D3  robustness: 3 images x 15 cells with image 2 excluded -> n = 2 and per-image mIoU-C equal to the
      flat 15-cell mean; a cell undefined for image 1 raises; a differing flagged set raises; the same
      at full REHEARSAL size.
  T   the local d4/d5 tool (scripts/am5_real_checks.py) dry-run on synthetic artifacts.

No dataset, checkpoint, GPU or TEST data: every artifact is synthetic, labelled as such, and written to
a temporary directory that is removed at the end. The OFFICIAL fixture is written against a temp-only
clean Git repository, never the project repository.

Run:  python -B scripts/smoke_am5_exclusion.py
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from src.eval.artifacts import (AM5_RULE, ARTIFACT_FILES, ARTIFACT_SCHEMA_VERSION,  # noqa: E402
                                EVAL_RUNTIME_VERSION, MANIFEST_NAME, SCHEMA_VERSION,
                                ArtifactWriteError, DatasetMeta, RunMeta, build_config_payload,
                                canonical_json_bytes, prepare_artifact_request,
                                validate_artifact_request, validate_summary, verify_artifact,
                                write_artifact)
from src.eval.evaluate import (Condition, EvalBatch, EvalResult, ManifestEntry,  # noqa: E402
                               PerImageRow, evaluate_model)
from src.stats import (AlignmentError, CorruptionGrid, IngestError, Policy,  # noqa: E402
                       PooledStages, RobustnessError, align_miou_c, align_runs, am5_pair,
                       apply_am5_policy, assemble_miou_c, load_run)

C = 116
CLEAN = Condition()
FROZEN_ROW_KEYS = ["image_id", "clean_image_id", "manifest_index", "condition", "all_class_miou",
                   "all_class_miou_status", "disease_only_miou", "disease_only_miou_status",
                   "n_eligible_all_class", "n_eligible_disease_only", "gt_disease_classes"]
SYN_CLASS_MAP = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
                [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, C)]
GRID = CorruptionGrid(names=("synA", "synB", "synC", "synD", "synE"),
                      provenance="SYNTHETIC nonofficial fixture vocabulary")
CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def expect(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"wrong exception {type(e).__name__}: {e}") from e
    raise AssertionError("expected an exception, none raised")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------
# synthetic artifacts
# --------------------------------------------------------------------------------------------------
def core_result(ids, gts, preds, condition=CLEAN):
    """The REAL evaluation core. The "model" returns one-hot logits of `preds`, so the core scores
    exactly these predictions against `gts` with the production metric code."""
    n = len(ids)
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    logits = torch.nn.functional.one_hot(preds, C).permute(0, 3, 1, 2).float()
    batches = [EvalBatch(images=logits[i:i + 2], targets=gts[i:i + 2], image_ids=ids[i:i + 2],
                         clean_image_ids=ids[i:i + 2], manifest_indices=list(range(i, min(i + 2, n))))
               for i in range(0, n, 2)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=condition,
                         num_classes=C, background_index=0, ignore_index=255,
                         forward=lambda _m, x: x)
    return res, manifest


def direct_result(ids, disease_vals, *, condition=CLEAN, excluded=(), undefined=()):
    """An EvalResult with EXACT per-image values (as in smoke_stats_paired). `excluded` rows have no
    disease ground truth (AM-5 flags them); `undefined` rows are undefined but NOT flagged."""
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    rows, s = [], {k: [] for k in ("img", "cls", "tp", "gt", "pr")}
    for i, iid in enumerate(ids):
        ex, und = iid in excluded, iid in undefined
        rows.append(PerImageRow(
            image_id=iid, clean_image_id=iid, manifest_index=i, condition=condition,
            all_class_miou=0.8, all_class_miou_status="ok",
            disease_only_miou=None if (ex or und) else float(disease_vals[i]),
            disease_only_miou_status=("undefined_no_eligible_class" if ex else
                                      "evaluation_error" if und else "ok"),
            n_eligible_all_class=1 if ex else 2, n_eligible_disease_only=0 if ex else 1,
            gt_disease_classes=[] if ex else [1]))
        for cid, (tp, gt, pr) in ((0, (8, 10, 9)),) + (() if ex else ((1, (2, 3, 4)),)):
            for k, v in zip(("img", "cls", "tp", "gt", "pr"), (i, cid, tp, gt, pr)):
                s[k].append(v)
    arr = {k: np.array(v, np.int64) for k, v in s.items()}
    tot = {}
    for k in ("tp", "gt", "pr"):
        tot[k] = np.zeros(C, np.int64)
        np.add.at(tot[k], arr["cls"], arr[k])
    un = tot["gt"] + tot["pr"] - tot["tp"]
    el = un > 0
    iou = np.where(el, tot["tp"] / np.maximum(un, 1), np.nan)
    res = EvalResult(
        rows=rows, manifest_ids=list(ids), sparse_image_index=arr["img"], sparse_class_id=arr["cls"],
        sparse_tp=arr["tp"], sparse_gt=arr["gt"], sparse_pred=arr["pr"],
        dataset_tp=tot["tp"], dataset_gt=tot["gt"], dataset_pred=tot["pr"],
        dataset_level={"all_class_miou": float(np.nanmean(iou[el])),
                       "all_class_miou_n_eligible": int(el.sum()),
                       "all_class_macro_dice": 0.5, "all_class_dice_n_eligible": int(el.sum()),
                       "all_class_macc": 0.5, "all_class_macc_n_eligible": int((tot["gt"] > 0).sum()),
                       "disease_only_miou": float(iou[1]), "disease_only_miou_n_eligible": 1,
                       "aacc_diagnostic": 0.5},
        per_class={"class_ids": list(range(C)), "gt_support": tot["gt"].tolist(),
                   "pred_support": tot["pr"].tolist(), "intersection": tot["tp"].tolist(),
                   "union": un.tolist(),
                   "iou": [None if not el[c] else float(iou[c]) for c in range(C)],
                   "dice": [None if not el[c] else 0.5 for c in range(C)],
                   "acc": [None if not tot["gt"][c] else 0.5 for c in range(C)],
                   "iou_status": ["ok" if el[c] else "undefined_absent_from_gt_and_pred"
                                  for c in range(C)]},
        integrity={"row_count_ok": True, "ids_unique": True, "ids_match_manifest": True,
                   "order_canonical": True, "duplicate_ids": [], "missing_ids": [],
                   "invalid_pred_labels": 0,
                   "undefined_per_image_scores": len(excluded) + len(undefined),
                   "overwrite_policy": "refuse_existing"},
        num_classes=C, background_index=0, ignore_index=255, condition=condition)
    return res, manifest


_seq = [0]


def write(out_dir, res, manifest, *, split="val", status="smoke", stage="E1", repo_root=REPO):
    """Write through the unmodified production writer. Non-smoke fixtures carry a synthetic
    checkpoint identity (random_init=False); official ones also carry an eval_runtime record."""
    _seq[0] += 1
    smoke = status == "smoke"
    cond = res.condition
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status=status,
        run_id=f"syn_am5_{stage}_{cond.name or 'clean'}_{cond.severity}_{_seq[0]}",
        run=RunMeta(stage=stage, model_role="student", precision="fp32", random_init=smoke,
                    checkpoint_path=None if smoke else "synthetic_checkpoint.pt",
                    checkpoint_sha256=None if smoke else "0" * 64),
        dataset=DatasetMeta(name="SYNTHETIC am5 fixture (NOT PlantSeg data)",
                            doi="10.5281/zenodo.17719108", split=split, condition=cond,
                            preprocess_protocol="synthetic-am5/1.0.0", expected_rows=len(manifest)),
        expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=repo_root)
    prov = validate_artifact_request(req)
    rt = {"eval_runtime_version": EVAL_RUNTIME_VERSION} if status == "official" else None
    return write_artifact(res, req, prov, timestamp_utc="2026-09-28T00:00:00Z", eval_runtime=rt)


def rehash(d: Path) -> None:
    """Re-write MANIFEST.sha256 after a deliberate fixture edit (sha256sum format, sorted names)."""
    (d / MANIFEST_NAME).write_text(
        "".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ARTIFACT_FILES),
        encoding="utf-8", newline="\n")


def edit_rows(src: Path, dst: Path, fn) -> Path:
    shutil.copytree(src, dst)
    rows = [json.loads(ln) for ln in (dst / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = fn(rows)
    (dst / "per_image.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8", newline="\n")
    rehash(dst)
    return dst


def edit_summary(d: Path, fn) -> Path:
    s = fn(json.loads((d / "summary.json").read_text(encoding="utf-8")))
    (d / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                    encoding="utf-8", newline="\n")
    rehash(d)
    return d


def to_pre_lane(src: Path, dst: Path) -> Path:
    """What the pre-lane writer emits for the same result: no am5_excluded, no am5, no version."""
    edit_rows(src, dst, lambda rows: [{k: v for k, v in r.items() if k != "am5_excluded"}
                                      for r in rows])
    return edit_summary(dst, lambda s: {k: v for k, v in s.items()
                                        if k not in ("am5", "artifact_schema_version")})


def clean_git_fixture(root: Path) -> Path:
    """A temp-only repository with one commit and nothing dirty, so the production pre-inference gate
    admits an 'official' synthetic fixture. Never the project repository."""
    root.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=am5-fixture", "-c", "user.email=am5@invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True,
                       capture_output=True)
    git("init", "-q")
    (root / "seed.txt").write_text("am5 clean fixture\n", encoding="utf-8")
    git("add", "--", "seed.txt")
    git("commit", "-q", "-m", "fixture baseline")
    return root


def real_checks_main(argv):
    spec = importlib.util.spec_from_file_location("am5_real_checks", REPO / "scripts" /
                                                  "am5_real_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main(argv)
    return rc, buf.getvalue()


def label_maps():
    """5 images, 4x4. id2 and id4 hold no disease pixel in the GROUND TRUTH (id2 is nevertheless
    predicted with a disease pixel: the flag depends on ground truth only)."""
    gts = torch.zeros(5, 4, 4, dtype=torch.long)
    gts[0, :2, :2] = 7
    gts[1, :2, :2] = 12
    gts[3, 2:, 2:] = 7
    preds = gts.clone()
    preds[1, 0, 0] = 0                    # one missed disease pixel
    preds[2, 3, 3] = 30                   # false positive on a zero-disease image
    preds[3, 2:, 2:] = 12                 # wrong disease
    return [f"id{i}" for i in range(5)], gts, preds


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("L-AM5 -- zero-disease exclusion smoke  [NONOFFICIAL: synthetic data only]")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="am5_"))
    try:
        # ================================ D1 evaluator ================================
        ids, gts, preds = label_maps()
        res, manifest = core_result(ids, gts, preds)
        d1 = write(work / "d1", res, manifest)
        summary = verify_artifact(d1)
        rows = [json.loads(ln) for ln in (d1 / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
        flags = [r["am5_excluded"] for r in rows]
        check("D1-1 am5_excluded true for exactly the two zero-disease images (id2, id4)",
              flags == [False, False, True, False, True], flags)
        check("D1-2 am5_excluded appended LAST; the frozen 11 keys keep their order",
              all(list(r) == FROZEN_ROW_KEYS + ["am5_excluded"] for r in rows))
        check("D1-3 flag == (n_eligible_disease_only == 0) == (gt_disease_classes == [])",
              all(r["am5_excluded"] is (r["n_eligible_disease_only"] == 0) is
                  (r["gt_disease_classes"] == []) for r in rows))
        check("D1-4 disease_only_miou unchanged: null + undefined_no_eligible_class on id2/id4",
              all(r["disease_only_miou"] is None
                  and r["disease_only_miou_status"] == "undefined_no_eligible_class"
                  for r in rows if r["am5_excluded"])
              and all(isinstance(r["disease_only_miou"], float) for r in rows if not r["am5_excluded"]))
        want = {"rule": "no_disease_gt", "excluded_count": 2, "included_count": 3,
                "excluded_ids_sha256": sha("id2\nid4")}
        check("D1-5 summary.am5 == {no_disease_gt, 2, 3, sha256('id2\\nid4')}",
              summary.get("am5") == want and AM5_RULE == "no_disease_gt", summary.get("am5"))
        check("D1-6 artifact_schema_version set; schema_version unchanged",
              summary.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
              == "plantseg-eval-artifact/1.1.0" and summary["schema_version"] == SCHEMA_VERSION
              == "plantseg-eval/1.0.0")
        z = np.load(d1 / "sufficient_stats.npz", allow_pickle=False)
        fp = (z["image_index"] == 2) & (z["class_id"] == 30)
        check("D1-7 a predicted disease on a zero-disease image leaves it excluded (GT only)",
              bool(fp.any()) and int(z["pred"][fp][0]) == 1 and int(z["gt"][fp][0]) == 0
              and flags[2] is True)
        req_cfg = prepare_artifact_request(
            out_dir=work / "unused", artifact_status="smoke", run_id="cfg",
            run=RunMeta(stage="E1", model_role="student", precision="fp32", random_init=True),
            dataset=DatasetMeta(name="SYNTHETIC am5 fixture (NOT PlantSeg data)",
                                doi="10.5281/zenodo.17719108", split="val", condition=CLEAN,
                                preprocess_protocol="synthetic-am5/1.0.0", expected_rows=5),
            expected_manifest=manifest, class_map=SYN_CLASS_MAP, repo_root=REPO)
        payload = build_config_payload(req_cfg)
        check("D1-8 config_sha256 is untouched by the lane (no AM-5 key in the config payload)",
              hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == summary["run"]["config_sha256"]
              and not {"am5", "artifact_schema_version"} & set(payload))
        res0, man0 = core_result(["n0", "n1"], gts[[0, 1]], preds[[0, 1]])
        s0 = verify_artifact(write(work / "d1_none", res0, man0))
        check("D1-9 no zero-disease image -> excluded_count 0, sha256 of the empty string",
              s0["am5"] == {"rule": "no_disease_gt", "excluded_count": 0, "included_count": 2,
                            "excluded_ids_sha256": sha("")}, s0["am5"])
        bad = json.loads(json.dumps(summary))
        bad["am5"]["excluded_count"] = 1
        e = expect(ArtifactWriteError, validate_summary, bad, req_cfg, res)
        check("D1-10 the writer refuses a summary.am5 that disagrees with the rows",
              "am5" in str(e), str(e)[:120])
        check("D1-11 PerImageRow.am5_excluded is derived (n_eligible_disease_only == 0)",
              [r.am5_excluded for r in res.rows] == flags)

        # ================================ D2 ingest ================================
        run_s = load_run(d1, Policy.NONOFFICIAL_SMOKE)
        check("D2-1 NONOFFICIAL_SMOKE keeps all 5 rows; per-run n_excluded_am5=2, n_included=3",
              len(run_s.records) == 5 and run_s.n_excluded_am5 == 2 and run_s.n_included == 3
              and run_s.am5.source == "artifact" and run_s.am5.applied is False
              and run_s.am5.excluded_ids_sha256 == want["excluded_ids_sha256"])
        kept = {p: [r.image_id for r in apply_am5_policy(run_s.records, p)]
                for p in (Policy.REHEARSAL, Policy.OFFICIAL, Policy.NONOFFICIAL_SMOKE)}
        check("D2-2 OFFICIAL and REHEARSAL drop the 2 flagged rows -> n = 3",
              kept[Policy.REHEARSAL] == kept[Policy.OFFICIAL] == ["id0", "id1", "id3"]
              and len(kept[Policy.NONOFFICIAL_SMOKE]) == 5, kept)
        r_und, m_und = direct_result([f"u{i}" for i in range(5)], [0.1, 0.2, 0.3, 0.4, 0.5],
                                     excluded=("u2", "u4"), undefined=("u1",))
        run_und = load_run(write(work / "d2_und", r_und, m_und), Policy.NONOFFICIAL_SMOKE)
        msgs = [str(expect(IngestError, apply_am5_policy, run_und.records, p))
                for p in (Policy.REHEARSAL, Policy.OFFICIAL)]
        check("D2-3 an undefined row AM-5 does not flag aborts, naming the row id",
              all("'u1'" in m and "u2" not in m for m in msgs), msgs[0][:120])
        r_b, m_b = direct_result(ids, [0.1] * 5, excluded=("id2",))
        run_b = load_run(write(work / "d2_other", r_b, m_b), Policy.NONOFFICIAL_SMOKE)
        e = expect(IngestError, am5_pair, run_s, run_b)
        check("D2-4 two runs with different excluded sets abort as a pair (manifest mismatch)",
              "excluded id sets differ" in str(e) and "manifest mismatch" in str(e), str(e)[:140])
        pre = to_pre_lane(d1, work / "d1_pre_lane")
        run_pre = load_run(pre, Policy.NONOFFICIAL_SMOKE)
        check("D2-5 pre-lane rows (field absent) -> derived flags identical to D1",
              run_pre.am5.source == "derived"
              and [r.am5_excluded for r in run_pre.records] == flags
              and run_pre.am5.excluded_ids == run_s.am5.excluded_ids
              and run_pre.am5.excluded_ids_sha256 == run_s.am5.excluded_ids_sha256)
        flip = edit_rows(d1, work / "d2_flip", lambda rs: [
            {**r, "am5_excluded": (not r["am5_excluded"]) if r["image_id"] == "id0" else
             r["am5_excluded"]} for r in rs])
        check("D2-6 a present flag that disagrees with n_eligible_disease_only aborts",
              "id0" in str(expect(IngestError, load_run, flip, Policy.NONOFFICIAL_SMOKE)))
        mixed = edit_rows(d1, work / "d2_mixed", lambda rs: [
            {k: v for k, v in r.items() if not (k == "am5_excluded" and r["image_id"] == "id3")}
            for r in rs])
        check("D2-7 rows mixing present and absent flags abort",
              "mixes" in str(expect(IngestError, load_run, mixed, Policy.NONOFFICIAL_SMOKE)))
        blk = edit_summary(shutil.copytree(d1, work / "d2_block"),
                           lambda s: {**s, "am5": {**s["am5"], "excluded_count": 3}})
        check("D2-8 a summary.am5 block that disagrees with the rows aborts",
              "summary.am5" in str(expect(IngestError, load_run, blk, Policy.NONOFFICIAL_SMOKE)))
        orphan = edit_summary(shutil.copytree(pre, work / "d2_orphan"),
                              lambda s: {**s, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION})
        check("D2-9 a declared lane layout without row flags aborts",
              "lack am5_excluded" in str(expect(IngestError, load_run, orphan,
                                                Policy.NONOFFICIAL_SMOKE)))

        # full-size REHEARSAL (846 rows, 2 flagged) and OFFICIAL (1,561 rows, 2 flagged)
        vids = [f"v{i:04d}" for i in range(846)]
        vals = [0.2 + 0.0005 * i for i in range(846)]
        rv, mv = direct_result(vids, vals, excluded=("v0010", "v0500"))
        ra = load_run(write(work / "reh_a", rv, mv, status="provisional"), Policy.REHEARSAL)
        rv2, mv2 = direct_result(vids, [v + 0.01 for v in vals], excluded=("v0010", "v0500"))
        rb = load_run(write(work / "reh_b", rv2, mv2, status="provisional", stage="E2"),
                      Policy.REHEARSAL)
        check("D2-10 REHEARSAL (846 rows) drops the 2 flagged rows: records 844, am5 applied",
              len(ra.records) == 844 and ra.n_excluded_am5 == 2 and ra.n_included == 844
              and ra.am5.applied and len(ra.stats.manifest_ids) == 846)
        pr = am5_pair(ra, rb)
        pv = align_runs(ra, rb, policy=Policy.REHEARSAL)
        check("D2-11 per pair: n_excluded_am5=2, n_included=844; the paired vector has n=844",
              pr.n_excluded_am5 == 2 and pr.n_included == 844 and pv.n == 844
              and np.allclose(pv.delta, 0.01) and pr.excluded_ids_sha256 == sha("v0010\nv0500")
              and pv.am5 == pr)
        ps = PooledStages.from_runs(ra, rb)
        check("D2-12 dataset-level (pooled) stats still see all 846 images",
              ps.n_images == 846)
        check("D2-13 REHEARSAL refuses a smoke/random-init artifact",
              "real-run artifact" in str(expect(IngestError, load_run, d1, Policy.REHEARSAL)))
        r5, m5 = direct_result(vids[:5], vals[:5])
        check("D2-14 REHEARSAL refuses anything but 846 VAL rows",
              "846 rows" in str(expect(IngestError, load_run,
                                       write(work / "reh_5", r5, m5, status="provisional"),
                                       Policy.REHEARSAL)))
        rt_, mt_ = direct_result(vids, vals, excluded=("v0010",))
        check("D2-15 REHEARSAL never reads a TEST-split artifact",
              "VAL artifacts only" in str(expect(IngestError, load_run,
                                                 write(work / "reh_test", rt_, mt_, split="test",
                                                       status="provisional"), Policy.REHEARSAL)))
        fixture = clean_git_fixture(work / "_clean_fixture_repo")
        tids = [f"t{i:04d}" for i in range(1561)]
        ro, mo = direct_result(tids, [0.3] * 1561, excluded=("t0001", "t1500"))
        off = write(work / "off", ro, mo, split="test", status="official", repo_root=fixture)
        run_off = load_run(off, Policy.OFFICIAL)
        check("D2-16 OFFICIAL (1,561 rows) drops the 2 flagged rows: records 1,559",
              len(run_off.records) == 1559 and run_off.n_excluded_am5 == 2
              and run_off.n_included == 1559 and len(run_off.stats.manifest_ids) == 1561)
        ro2, mo2 = direct_result(tids, [0.3] * 1561, excluded=("t0001",), undefined=("t0700",))
        off2 = write(work / "off2", ro2, mo2, split="test", status="official", repo_root=fixture)
        check("D2-17 OFFICIAL still aborts on an undefined row AM-5 does not flag, naming it",
              "'t0700'" in str(expect(IngestError, load_run, off2, Policy.OFFICIAL)))
        check("D2-18 the frozen OFFICIAL gate still refuses a VAL artifact",
              "requires split='test'" in str(expect(IngestError, load_run,
                                                    write(work / "off_val", rv, mv, status="official",
                                                          repo_root=fixture), Policy.OFFICIAL)))
        met = edit_summary(shutil.copytree(work / "reh_b", work / "reh_b_metric"), lambda s: {
            **s, "run": {**s["run"], "metric_impl_sha256": "f" * 64}})
        e = expect(AlignmentError, align_runs, ra, load_run(met, Policy.REHEARSAL),
                   policy=Policy.REHEARSAL)
        lenient = align_runs(load_run(work / "reh_a", Policy.NONOFFICIAL_SMOKE),
                             load_run(met, Policy.NONOFFICIAL_SMOKE), policy=Policy.NONOFFICIAL_SMOKE)
        check("D2-19 REHEARSAL alignment is as strict as OFFICIAL: a metric_impl mismatch is fatal "
              "(NONOFFICIAL_SMOKE unchanged: no error)",
              "REHEARSAL analysis forbids this" in str(e) and lenient.policy is Policy.NONOFFICIAL_SMOKE,
              str(e)[:120])
        rv3, mv3 = direct_result(vids, vals, excluded=("v0010",))
        reh_c = write(work / "reh_c", rv3, mv3, status="provisional")
        e = expect(AlignmentError, align_runs, ra, load_run(reh_c, Policy.REHEARSAL),
                   policy=Policy.REHEARSAL)
        check("D2-20 align_runs (REHEARSAL/OFFICIAL) aborts on differing AM-5 sets with the AM-5 message",
              "excluded id sets differ" in str(e) and "manifest mismatch" in str(e), str(e)[:120])
        v12 = edit_summary(shutil.copytree(d1, work / "v12"), lambda s: {
            **s, "artifact_schema_version": "plantseg-eval-artifact/1.2.0"})
        v2 = edit_summary(shutil.copytree(d1, work / "v2"), lambda s: {
            **s, "artifact_schema_version": "plantseg-eval-artifact/2.0.0"})
        vbad = edit_summary(shutil.copytree(d1, work / "vbad"), lambda s: {
            **s, "artifact_schema_version": "1.1"})
        check("D2-21 a later additive layout (1.2.0) is read; another major or a malformed one aborts",
              load_run(v12, Policy.NONOFFICIAL_SMOKE).am5.source == "artifact"
              and "unsupported artifact_schema_version" in str(expect(
                  IngestError, load_run, v2, Policy.NONOFFICIAL_SMOKE))
              and "unsupported artifact_schema_version" in str(expect(
                  IngestError, load_run, vbad, Policy.NONOFFICIAL_SMOKE)))

        # ================================ D3 robustness ================================
        cids = ["c0", "c1", "c2"]                    # c2: no disease ground truth (excluded)
        base = {"synA": (0.10, 0.20, 0.30), "synB": (0.40, 0.40, 0.40), "synC": (0.00, 0.30, 0.60),
                "synD": (0.50, 0.60, 0.70), "synE": (0.10, 0.10, 0.40)}
        tag = [0]

        def grid(stage, *, n_ids=None, bump=0.0, excluded=("c2",), policy=Policy.NONOFFICIAL_SMOKE,
                 status="smoke", undefined_cell=None, extra_flag_cell=None):
            tag[0] += 1
            ids_ = n_ids or cids
            runs = []
            for cname, vs in base.items():
                for k, sev in enumerate((1, 2, 3)):
                    v = [vs[k] + bump + 0.01 * (j % 10) for j in range(len(ids_))]
                    und = (ids_[1],) if undefined_cell == (cname, sev) else ()
                    exc = excluded + ((ids_[1],) if extra_flag_cell == (cname, sev) else ())
                    r, m = direct_result(ids_, v, condition=Condition("corruption", cname, sev),
                                         excluded=exc, undefined=und)
                    runs.append(load_run(write(work / f"g{tag[0]}_{stage}_{cname}_{sev}", r, m,
                                               status=status, stage=stage), policy))
            r, m = direct_result(ids_, [0.5] * len(ids_), excluded=excluded)
            clean = load_run(write(work / f"g{tag[0]}_{stage}_clean", r, m, status=status,
                                   stage=stage), policy)
            return runs, clean

        runs, clean = grid("E1")
        v = assemble_miou_c(runs, GRID, policy=Policy.NONOFFICIAL_SMOKE, stage="E1", clean=clean)
        flat = [float(np.mean([vs[k] + 0.01 * j for vs in base.values() for k in range(3)]))
                for j in range(2)]
        check("D3-1 image c2 excluded -> n = 2 (c0, c1) and n equals the clean n",
              v.clean_image_ids == ("c0", "c1") and clean.n_included == 2, v.clean_image_ids)
        check("D3-2 per-image mIoU-C == the flat 15-cell mean (c0 = 0.34 by hand, c1 = 0.35)",
              all(abs(float(v.values[j]) - flat[j]) < 1e-12 for j in range(2))
              and abs(flat[0] - 0.34) < 1e-12 and abs(flat[1] - 0.35) < 1e-12,
              f"{[float(x) for x in v.values]} vs {flat}")
        runs_u, clean_u = grid("E1", undefined_cell=("synC", 2))
        check("D3-3 a cell undefined for included image c1 still raises",
              "undefined disease-only score" in str(expect(
                  RobustnessError, assemble_miou_c, runs_u, GRID,
                  policy=Policy.NONOFFICIAL_SMOKE, stage="E1", clean=clean_u)))
        runs_f, clean_f = grid("E1", extra_flag_cell=("synD", 3))
        check("D3-4 a cell whose flagged set differs from the clean artifact's raises",
              "manifest mismatch" in str(expect(RobustnessError, assemble_miou_c, runs_f, GRID,
                                                policy=Policy.NONOFFICIAL_SMOKE, clean=clean_f)))
        check("D3-5 without `clean` (NONOFFICIAL legacy path) a flagged image still raises",
              "undefined disease-only score" in str(expect(RobustnessError, assemble_miou_c, runs,
                                                           GRID, policy=Policy.NONOFFICIAL_SMOKE)))
        check("D3-6 OFFICIAL/REHEARSAL mIoU-C refuse to run without the clean artifact",
              all("requires the stage's clean artifact" in str(expect(
                  RobustnessError, assemble_miou_c, runs, GRID, policy=p))
                  for p in (Policy.OFFICIAL, Policy.REHEARSAL)))
        r_c, m_c = direct_result(cids, [0.5] * 3, excluded=())
        clean_none = load_run(write(work / "d3_clean_none", r_c, m_c), Policy.NONOFFICIAL_SMOKE)
        check("D3-7 a clean artifact with a different excluded set raises",
              "manifest mismatch" in str(expect(RobustnessError, assemble_miou_c, runs, GRID,
                                                policy=Policy.NONOFFICIAL_SMOKE, clean=clean_none)))
        runs6, _ = grid("E6")
        check("D3-8 a clean artifact of another stage raises (stage given, and stage=None)",
              all("`clean` run is stage 'E1', expected 'E6'" in str(expect(
                  RobustnessError, assemble_miou_c, runs6, GRID, policy=Policy.NONOFFICIAL_SMOKE,
                  stage=st, clean=clean)) for st in ("E6", None)))
        runs_r, clean_r = grid("E1", n_ids=vids, excluded=("v0010", "v0500"),
                               policy=Policy.REHEARSAL, status="provisional")
        runs_r6, clean_r6 = grid("E6", n_ids=vids, excluded=("v0010", "v0500"), bump=0.05,
                                 policy=Policy.REHEARSAL, status="provisional")
        e1 = assemble_miou_c(runs_r, GRID, policy=Policy.REHEARSAL, stage="E1", clean=clean_r)
        e6 = assemble_miou_c(runs_r6, GRID, policy=Policy.REHEARSAL, stage="E6", clean=clean_r6)
        pmc = align_miou_c(e1, e6, policy=Policy.REHEARSAL)
        check("D3-9 REHEARSAL at full size: 15 x 846-row cells -> robustness n = 844 = clean n",
              len(e1.values) == 844 == clean_r.n_included and pmc.n == 844
              and np.allclose(pmc.delta, 0.05))

        # ================================ T: the d4/d5 tool ================================
        rc, out = real_checks_main(["d4", str(pre), str(d1)])
        check("T1 d4: pre-lane vs lane artifact of one result -> PASS", rc == 0
              and "RESULT: D4 PASS" in out and "bitwise equal: True" in out, out[-160:])
        tam = edit_rows(pre, work / "t_pre_tampered", lambda rs: [
            {**r, "disease_only_miou": 0.123} if r["image_id"] == "id0" else r for r in rs])
        rc, out = real_checks_main(["d4", str(tam), str(d1)])
        check("T2 d4: a changed disease_only_miou -> FAIL (exit 1)", rc == 1
              and "RESULT: D4 FAIL" in out, out[-160:])
        npz_t = shutil.copytree(pre, work / "t_pre_npz")
        with np.load(npz_t / "sufficient_stats.npz", allow_pickle=False) as zz:
            arrays = {k: zz[k].copy() for k in zz.files}
        arrays["tp"][0] += 1
        np.savez_compressed(npz_t / "sufficient_stats.npz", **arrays)
        rehash(npz_t)
        rc, out = real_checks_main(["d4", str(npz_t), str(d1)])
        check("T3 d4: a changed npz array -> FAIL (exit 1)", rc == 1 and "NPZ tp differs" in out,
              out[-160:])
        rc, out = real_checks_main(["d5", str(work / "reh_a"), str(work / "reh_b")])
        check("T4 d5: two VAL artifacts with one excluded set -> PASS, K_val = 2",
              rc == 0 and "K_val = 2" in out and "RESULT: D5 PASS" in out, out[-160:])
        rc, out = real_checks_main(["d5", str(work / "reh_a"), str(reh_c)])
        check("T5 d5: different excluded sets -> FAIL (exit 1)", rc == 1
              and "excluded id sets differ" in out, out[-160:])
        rc, out = real_checks_main(["d5", str(d1), str(work / "reh_a")])
        check("T6 d5: a smoke artifact is refused by the REHEARSAL gate (exit 2)", rc == 2
              and "RESULT: ERROR" in out, out[-160:])
        moved = edit_summary(shutil.copytree(pre, work / "t_pre_path"), lambda s: {
            **s, "run": {**s["run"], "checkpoint_path": "elsewhere/copy_of_checkpoint.pt"}})
        rc, out = real_checks_main(["d4", str(moved), str(d1)])
        check("T7 d4: a different checkpoint path string is provenance (NOTE), not a failure",
              rc == 0 and "NOTE: run.checkpoint_path" in out, out[-160:])
        cfg = edit_summary(shutil.copytree(pre, work / "t_pre_cfg"), lambda s: {
            **s, "run": {**s["run"], "config_sha256": "0" * 64}})
        rc, out = real_checks_main(["d4", str(cfg), str(d1)])
        check("T8 d4: any other summary difference fails and is named by its dotted key",
              rc == 1 and "run.config_sha256" in out, out[-160:])

    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check("Z1 temporary synthetic artifacts removed", not work.exists())

    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if det:
            print(f"         {det[:165]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM5 OK' if allok else 'L-AM5 BLOCKED'} ({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; the real checks d4/d5 run locally (am5_real_checks.py).")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
