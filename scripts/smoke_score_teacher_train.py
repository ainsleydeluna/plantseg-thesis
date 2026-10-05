#!/usr/bin/env python3
"""Smoke: the teacher TRAIN entry point (scripts/score_teacher_train.py; lane L-TEACHER-DIAG; acceptance (e)).

On the stub teacher and a synthetic 5,367/846 data root: the per-class IoU and the all-class mIoU equal
the evaluator's metric functions (metrics.confusion_matrix, miou_from_confusion) on an independent pass;
the TRAIN canvas wrapper is the VAL branch bit for bit and never augments; order_equals_sorted_stems is
recorded; a data root containing "test" is refused by its string before any realpath of it; the strict
VAL readers (load_val_artifact, ingest.load_run, the --val-reference check) refuse the TRAIN artifact;
the arm role, a real run not pinned to HEAD and the other gates refuse before any load.
scripts/evaluate_model.py is not edited; the report shows `git diff <base> -- scripts/evaluate_model.py`.
Synthetic inputs only; no PlantSeg data, no checkpoint of record, no GPU.

    python -B scripts/smoke_score_teacher_train.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
UTC = "2026-10-02T00:00:00Z"


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def cases(tmp: Path, root: Path, stems: dict) -> None:
    from PIL import Image

    from scripts import score_teacher_train as st
    from scripts import teacher_diag_fixtures as fx
    from src.data.dataset import PlantSegDataset
    from src.data.transforms import core_preprocess, finalize
    from src.eval import teacher_diag as td
    from src.eval.adapters import deterministic_subset
    from src.eval.metrics import confusion_matrix, miou_from_confusion

    ckpt, sha = fx.write_stub_ckpt(tmp / "stub_teacher.pth")
    strata = fx.write_strata(tmp / "train_strata_v1.json", stems["train"])
    teacher = dict(teacher_ckpt=str(ckpt), teacher_ckpt_sha256=sha, teacher_config=str(td.REPO / td.TEACHER_CONFIG_REL),
                   teacher_role="record")
    base = dict(strata=str(strata), max_samples=4, generated_utc=UTC, **teacher)

    out, art = tmp / "out", tmp / "art"
    code, err = fx.call_run(st, out_dir=out, artifact_dir=art, **base)
    files = fx.output_files(out)
    doc = json.loads(files[0].read_text()) if files else {}
    check("e TRAIN pass: exit 0, teacher_train_scores_<UTC>.json, status written, smoke",
          code == 0 and [p.name for p in files] == ["teacher_train_scores_20261002T000000Z.json"]
          and doc.get("status") == "written" and doc.get("artifact_status") == "smoke", err)
    art_dir = Path(doc.get("artifact", {}).get("dir", tmp / "missing"))
    summary = json.loads((art_dir / "summary.json").read_text()) if art_dir.is_dir() else {}
    check("P19 the TRAIN artifact: schema plantseg-teacher-train-diagnostic/1.0.0, split train, "
          "order_equals_sorted_stems recorded, four files hashed",
          summary.get("schema_version") == st.TRAIN_ARTIFACT_SCHEMA and summary.get("dataset", {}).get("split") == "train"
          and summary.get("diagnostic", {}).get("order_equals_sorted_stems") is True
          and doc.get("pass", {}).get("order_equals_sorted_stems") is True
          and len(doc.get("artifact", {}).get("files_sha256", {})) == 4)
    c = doc.get("checks", {})
    check("e M4-V stream: one draw per image; the caller's RNG unchanged across the pass",
          c.get("nmf_stream", {}).get("draws") == 4 and c.get("rng_state_unchanged") is True)

    # per-class IoU equals the evaluator's metric functions on an independent pass
    args = types.SimpleNamespace(**dict(teacher, arm_id=None, arm_dl_id=None, teacher_config_sha256=None))
    inputs = td.verify_teacher_inputs(args, stub=True)
    loaded = td.load_teacher("evaluator", inputs, model_factory=fx.stub_factory)
    idx = deterministic_subset(td.TRAIN_ROWS, 4)
    ds = td.train_canvas_dataset(idx)
    cm = torch.zeros(116, 116, dtype=torch.long)
    for k in range(len(ds)):
        item = ds[k]
        pred = loaded.eval_model(item["image"][None]).argmax(1)[0]
        cm += confusion_matrix(pred, item["target"], 116, 255)
    tp, gt, pr = cm.diag().numpy(), cm.sum(1).numpy(), cm.sum(0).numpy()
    un = gt + pr - tp
    iou = doc.get("per_class", {}).get("iou", [])
    ok = len(iou) == 116 and all((iou[i] == tp[i] / un[i]) if un[i] > 0 else iou[i] is None for i in range(116))
    want = miou_from_confusion(cm)
    check("e per-class IoU equals TP/union of metrics.confusion_matrix on an independent pass", ok,
          str([(i, iou[i], tp[i], un[i]) for i in range(116) if un[i] > 0][:3]))
    check("e all-class mIoU equals miou_from_confusion (union-present) exactly",
          doc.get("dataset_level", {}).get("all_class_miou") == (None if math.isnan(want) else want))
    check("e per-class supports equal the confusion matrix's GT and prediction sums",
          doc.get("per_class", {}).get("gt_support") == gt.tolist()
          and doc.get("per_class", {}).get("pred_support") == pr.tolist())

    # the canvas wrapper is the VAL branch and never augments
    def boom(*a, **k):
        raise AssertionError("augmentation RNG drawn")
    ok = True
    with fx.patched(np.random, randint=boom):
        for k in range(len(ds)):
            item = ds[k]
            img_path, mask_path = ds.base.pairs[idx[k]]
            with Image.open(img_path) as im, Image.open(mask_path) as mk:
                ref_img, ref_mask = finalize(*core_preprocess(im, mk))
            ok &= torch.equal(item["image"], ref_img) and torch.equal(item["target"], ref_mask)
    aug = PlantSegDataset("train")[idx[1]][0]
    check("e the TRAIN canvas item equals core_preprocess + finalize bit for bit, with the augmentation draw "
          "forbidden; the augmented TRAIN item differs (the check is not vacuous)",
          ok and not torch.equal(aug, ds[1]["image"]))

    # the strict readers refuse the TRAIN artifact
    from src.stats.ingest import IngestError, load_run
    from src.stats.val_artifacts import POLICY, ValArtifactError, load_val_artifact
    check("P32(e) the strict readers refuse the TRAIN artifact: load_val_artifact, ingest.load_run, "
          "the --val-reference check",
          raises(lambda: load_val_artifact(art_dir, label="train"), ValArtifactError)
          and raises(lambda: load_run(art_dir, POLICY), IngestError)
          and raises(lambda: td.check_val_reference(art_dir, stub=True, expected_rows=4), td.Refused))

    # a 'test' data root is refused by its string before any realpath of it
    real_rp = os.path.realpath
    seen = []

    def counting(p, *a, **k):
        if "test" in str(p).lower():
            seen.append(1)
        return real_rp(p, *a, **k)
    bad_root = tmp / ("x" + "TeSt" + "x")
    with fx.count_loads() as calls:
        fx.set_data_root(bad_root)
        try:
            with fx.patched(os.path, realpath=counting):
                code, err = fx.call_run(st, out_dir=tmp / "o_t", artifact_dir=tmp / "a_t", **base)
        finally:
            fx.set_data_root(root)
    check("P21 a data root containing 'test' is refused by its string: no realpath of it, no load, the "
          "message names a count and a sha256, not the path",
          code == 2 and not seen and sum(calls.values()) == 0 and "TeSt" not in err and "1 offending entry" in err, err)

    # other refusals, before any load
    bad_strata = fx.write_strata(tmp / "strata_other.json", stems["train"][:-1] + ["plant_leaf_zzzz"])
    arm_cfg = Path(shutil.copy(teacher["teacher_config"], tmp / "arm_cfg.py"))
    arm = dict(teacher_role="arm", arm_id="R1", arm_dl_id="DL-70", teacher_config=str(arm_cfg),
               teacher_config_sha256=td.file_sha256(arm_cfg))
    real = dict(base, max_samples=None, generated_utc=None, script_commit="c" * 40, script_commit_dl_id="DL-61")
    with fx.count_loads() as calls:
        r = [fx.call_run(st, out_dir=tmp / "o1", artifact_dir=tmp / "a1", **dict(base, **arm))[0],
             fx.call_run(st, out_dir=tmp / "o2", artifact_dir=tmp / "a2", **dict(base, strata=str(bad_strata)))[0],
             fx.call_run(st, out_dir=tmp / "o3", artifact_dir=td.REPO / "train_art_in_repo", **base)[0],
             fx.call_run(st, out_dir=out, artifact_dir=tmp / "a4", **base)[0],
             fx.call_run(st, factory=None, out_dir=tmp / "o5", artifact_dir=tmp / "a5", **real)[0],
             fx.call_run(st, factory=None, out_dir=tmp / "o6", artifact_dir=tmp / "a6", **dict(real, max_samples=4))[0]]
    check("P3/P27/P8 refused before any load: the arm role, another strata split list, an in-repo artifact "
          "dir, an existing output name, a real run not pinned to HEAD, --max-samples in a real run",
          r == [2] * 6 and sum(calls.values()) == 0 and not (td.REPO / "train_art_in_repo").exists(), str(r))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m_usage = st.main([])
        m_real = st.main(fx.argv(out_dir=tmp / "o7", artifact_dir=tmp / "a7", **dict(real)))
    check("P29 the CLI: a usage error exits 2; real mode refuses a --script-commit that is not HEAD (exit 2)",
          m_usage == 2 and m_real == 2)
    code, err = fx.call_run(st, factory=None, out_dir=tmp / "o8", artifact_dir=tmp / "a8", **dict(real, max_samples=4))
    check("P2 --max-samples in a real run is refused by its own gate", code == 2 and "--max-samples is a stub-mode flag" in err,
          err)

    # Q10: a TRAIN listing whose name order is not the stem order runs, and the fact is recorded False
    from scripts.build_train_strata import train_split_list
    s0, s1 = stems["train"][0], stems["train"][1]
    for sub, ext in (("images", ".jpg"), ("annotations", ".png")):
        (root / sub / "train" / f"{s1}{ext}").rename(root / sub / "train" / f"{s0} (1){ext}")
    names = [x for x, _ in train_split_list(root)]
    strata2 = fx.write_strata(tmp / "train_strata_renamed.json", names)
    code, err = fx.call_run(st, out_dir=tmp / "o9", artifact_dir=tmp / "a9", **dict(base, strata=str(strata2)))
    files = fx.output_files(tmp / "o9")
    d9 = json.loads(files[0].read_text()) if files else {}
    check("P19/Q10 'x (1).jpg' before 'x.jpg': the run is not gated, order_equals_sorted_stems is recorded False",
          code == 0 and names[:2] == [f"{s0} (1)", s0] and d9.get("pass", {}).get("order_equals_sorted_stems") is False,
          err)


def main() -> int:
    from scripts import teacher_diag_fixtures as fx
    from scripts.synthetic_ptq_fixtures import safe_tmpdir
    tmp = safe_tmpdir("diag_train_")
    root = None
    try:
        root, stems = fx.make_data_root("diag_train_data_")
        fx.set_data_root(root)
        cases(tmp, root, stems)
    except Exception as e:  # noqa: BLE001 -- a crash is a failed case, never a pass
        check("cases raised", False, f"{type(e).__name__}: {e}")
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
