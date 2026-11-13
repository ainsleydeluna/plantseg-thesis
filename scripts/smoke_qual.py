#!/usr/bin/env python3
"""Smoke for the QUAL-1 kit (scripts/qual_select.py, qual_predict.py, qual_render.py), synthetic data only.

No dataset, no model file, no record and no TEST or VAL file is read: every input is made here, in a
temporary folder. Checks:
  S0  every repo module the three steps import loads in this checkout (a module the sparse clone leaves out
      stops the smoke, not a run)
  S1  DL-182's rule: eligibility from the rows (a row without it stops), nearest ranks, ties to the lower
      manifest index under both tie readings, a duplicate percentile pick stops, six random picks outside
      the percentile picks in shuffle order; the tie-readings word (ruling QUAL-A-1 Q2)
  S2  the version guard refuses another interpreter, version or image digest, and no random pick is drawn
      without it; random.Random(42).sample's known answer (measured under CPython 3.11.15 and 3.13)
  S3  the check of AM-27 item 4: the evaluator's own evaluate_model on a synthetic batch gives the recorded
      value back (gap 0) and a one-pixel change of the prediction fails the 1e-6 tolerance; the confusion
      counts follow the prediction and are compared by class (Q6); the class-level error map's TP pixels
      equal the record's tp sum and padding is none of TP, FN or FP (Q4)
  S4  the teacher's stream advance (QUAL-1 Q1): a subset with skipped rows draws exactly the bases a full
      pass draws at those rows
  S5  every figure renders from synthetic arrays (11 PNG files at 300 dpi, 1,950 px wide, and a manifest),
      a re-render from the same arrays gives the same colours (Q3), the pair legend (Q5), the value label
      (Q8), the padding key (L1), a changed array is refused, on TEST a failed tile stops the step before
      anything is drawn, on TEST an unexpected error prints its type only (its message, which may name an
      image, goes to the run's log), and a re-render opens only from a render-only commit (Q10)
  S6  the class palette is the measured list and the categorical keys are colour-blind-safe; the padding grey
      stays apart from the map underlay

  python -B scripts/smoke_qual.py --font PATH_TO_A_TTF      (inside the KD image: torch for S0, S3 and S4)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import qual_render as R  # noqa: E402
from scripts import qual_select as Q  # noqa: E402

KAT_SAMPLE_30 = [20, 3, 0, 23, 8, 7, 24, 4, 28, 17, 2, 18, 13, 1, 27, 16, 21, 29, 25, 9, 15, 11, 12, 5, 6,
                 22, 26, 10, 14, 19]     # random.Random(42).sample(list(range(30)), 30), CPython 3.11.15 and 3.13
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}", flush=True)


def stops(code: str, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except Q.QualStop as e:
        return e.code == code
    return False


def fake_official(values, excluded=(), drop_flag=None):
    rows = {}
    for m, v in enumerate(values):
        ex = m in excluded
        r = {"manifest_index": m, "image_id": f"img{m:03d}", "clean_image_id": f"img{m:03d}",
             "am5_excluded": ex, "n_eligible_disease_only": 0 if ex else 1,
             "disease_only_miou": None if ex else float(v), "disease_only_miou_status": "undefined" if ex else "ok"}
        if m == drop_flag:
            del r["am5_excluded"]
        rows[m] = r
    return Q.Official(label="synthetic", directory=None, summary={}, rows=rows, npz=None, files_sha256={},
                      source={})


# ------------------------------------------------------------------------------------------------- S1, S2
def s1_s2():
    vals = [0.0] * 12 + [round(0.05 * i, 4) for i in range(1, 29)]          # 40 rows, 12 tied at 0
    rnd = random.Random(7)
    rnd.shuffle(vals)
    off = fake_official(vals, excluded={5, 17})
    d = Q.eligible(off, 40)
    check("S1 eligibility read from the rows (2 excluded)", d == [m for m in range(40) if m not in (5, 17)])
    check("S1 a row without am5_excluded stops", stops("eligibility_absent", Q.eligible,
                                                       fake_official(vals, drop_flag=3), 40))
    pairs = Q.scored(off, d)
    ranks = Q.nearest_ranks(len(pairs))
    check("S1 nearest ranks of 38 values are 4, 19, 35", ranks == {10: 4, 50: 19, 90: 35}, str(ranks))
    check("S1 nearest ranks of 1561 values are 157, 781, 1405", Q.nearest_ranks(1561) == {10: 157, 50: 781, 90: 1405})
    picks, _ = Q.percentile_picks(pairs, "rank")
    order = sorted(pairs, key=lambda t: (t[1], t[0]))
    zeros = sorted(m for m, v in pairs if v == 0.0)
    check("S1 rank reading: the elements at the ranks, ties ordered by manifest index",
          picks == [order[3][0], order[18][0], order[34][0]] and picks[0] == zeros[3])
    picks_b, _ = Q.percentile_picks(pairs, "lowest_index_at_value")
    check("S1 lowest-index reading: the lowest manifest index holding the percentile value",
          picks_b[0] == zeros[0] and picks_b[1:] == picks[1:])
    tied = fake_official([0.0] * 30 + [0.5] * 10)
    check("S1 a duplicate percentile pick stops (lowest-index reading, 75 % ties)",
          stops("percentile_duplicate", Q.percentile_picks, Q.scored(tied, Q.eligible(tied, 40)),
                "lowest_index_at_value"))
    guard = Q.version_guard(implementation="CPython", version=Q.PYTHON, digest=Q.KD_DIGEST)
    sel = Q.select_dl182(off, 40, guard=guard, reading="rank")
    shuffled = random.Random(42).sample(sorted(d), len(d))
    want_r = [m for m in shuffled if m not in picks][:6]
    check("S1 S = three percentile picks, then the first six shuffle entries outside them",
          sel["indices"] == picks + want_r and len(set(sel["indices"])) == 9
          and sel["roles"] == ["p10", "p50", "p90", "r1", "r2", "r3", "r4", "r5", "r6"])
    distinct = fake_official([round(0.01 * i, 4) for i in range(40)])
    check("S1 the tie-readings word (QUAL-A-1 Q2): same, different, literal_undefined; it selects nothing",
          Q.tie_readings_word(Q.scored(distinct, Q.eligible(distinct, 40))) == "same"
          and Q.tie_readings_word(pairs) == "different" and sel["tie_readings"] == "different"
          and Q.tie_readings_word(Q.scored(tied, Q.eligible(tied, 40))) == "literal_undefined"
          and sel["indices"][:3] == picks)
    check("S2 the guard refuses CPython 3.11.15", stops("version_guard", Q.version_guard, implementation="CPython",
                                                        version="3.11.15", digest=Q.KD_DIGEST))
    check("S2 the guard refuses another implementation", stops("version_guard", Q.version_guard,
                                                               implementation="PyPy", version=Q.PYTHON,
                                                               digest=Q.KD_DIGEST))
    check("S2 the guard refuses another image", stops("version_guard", Q.version_guard, implementation="CPython",
                                                      version=Q.PYTHON, digest="sha256:" + "0" * 64))
    check("S2 no random pick without the guard", stops("version_guard", Q.random_picks, d, picks, guard=None))
    check("S2 random.Random(42).sample known answer", random.Random(42).sample(list(range(30)), 30) == KAT_SAMPLE_30)


# ----------------------------------------------------------------------------------------------------- S3
def s3():
    import numpy as np
    import torch
    from src.eval import Condition, evaluate_model
    from src.eval.evaluate import EvalBatch, ManifestEntry
    from scripts.qual_predict import result_counts
    g = torch.Generator().manual_seed(0)
    target = torch.zeros(1, 64, 64, dtype=torch.int64)
    target[0, 10:30, 10:30] = 7
    target[0, 40:50, 40:60] = 12
    target[0, :, :4] = 255                                            # padding
    logits = torch.rand(1, 116, 64, 64, generator=g)
    logits[0, 7, 12:30, 10:28] += 2.0
    logits[0, 9, 40:50, 40:50] += 2.0
    entry = [ManifestEntry(0, "img000", "img000")]

    def run(lg):
        batch = EvalBatch(images=torch.zeros(1, 3, 64, 64), targets=target, image_ids=["img000"],
                          clean_image_ids=["img000"], manifest_indices=[0])
        return evaluate_model(None, [batch], expected_manifest=entry, condition=Condition("clean"),
                              num_classes=116, background_index=0, ignore_index=255, forward=lambda m, x: lg)

    ref = run(logits)
    again = run(logits.clone())
    v_ref = ref.rows[0].disease_only_miou
    gap = abs(again.rows[0].disease_only_miou - v_ref)
    check("S3 the evaluator gives the recorded value back (gap 0)", gap == 0.0 and gap <= 1e-6)
    changed = logits.clone()
    p = changed[0].argmax(0)
    y, x = [int(v) for v in torch.nonzero((target[0] == 7) & (p == 7))[0]]
    changed[0, :, y, x] = 0.0
    changed[0, 3, y, x] = 10.0                                          # one pixel: class 7 -> class 3
    moved = run(changed)
    gap1 = abs(moved.rows[0].disease_only_miou - v_ref)
    check("S3 a one-pixel change fails the 1e-6 tolerance", gap1 > 1e-6, f"gap {gap1}")

    class Off:
        npz = {"manifest_ids": np.array(ref.manifest_ids, dtype=np.str_), "image_index": ref.sparse_image_index,
               "class_id": ref.sparse_class_id, "tp": ref.sparse_tp, "gt": ref.sparse_gt, "pred": ref.sparse_pred}
    off = Q.Official(label="s3", directory=None, summary={}, rows={}, npz=Off.npz, files_sha256={}, source={})
    check("S3 the confusion counts equal the record's for the same prediction",
          result_counts(again, "img000") == off.counts("img000"))
    check("S3 the confusion counts differ after the change", result_counts(moved, "img000") != off.counts("img000"))
    perm = np.arange(len(ref.sparse_class_id))[::-1]
    off_r = Q.Official(label="s3r", directory=None, summary={}, rows={}, npz={
        "manifest_ids": Off.npz["manifest_ids"], "image_index": ref.sparse_image_index[perm],
        "class_id": ref.sparse_class_id[perm], "tp": ref.sparse_tp[perm], "gt": ref.sparse_gt[perm],
        "pred": ref.sparse_pred[perm]}, files_sha256={}, source={})
    check("S3 the counts are compared as a mapping by class, not as an ordered list (QUAL-A-1 Q6)",
          result_counts(again, "img000") == off_r.counts("img000"))
    gt_np, pr_np = target[0].numpy(), logits[0].argmax(0).numpy()
    lay = R.error_layers(gt_np, pr_np)
    sel = ref.sparse_image_index == 0
    tp_disease = int(sum(int(t) for c, t in zip(ref.sparse_class_id[sel], ref.sparse_tp[sel]) if int(c) >= 1))
    check("S3 the class-level error map's TP pixels equal the record's tp sum (QUAL-A-1 Q4)",
          int(lay["tp"].sum()) == tp_disease and tp_disease > 0, f"{int(lay['tp'].sum())} vs {tp_disease}")
    em = R.error_map(np.zeros((64, 64, 3), dtype=np.uint8), gt_np, pr_np)
    check("S3 padding is grey and none of TP, FN or FP; FN holds the wrong-class pixels (QUAL-A-1 Q4)",
          (em[gt_np == 255] == R.IGNORE).all() and not (lay["tp"] | lay["fn"] | lay["fp"])[gt_np == 255].any()
          and bool(lay["wrong"].any()) and not (lay["wrong"] & ~lay["fn"]).any())
    st = Q.repo_state(None)
    check("S3 the checkout is read with the evaluator's git helpers (HEAD and the governed paths)",
          len(st["commit"]) == 40 and isinstance(st["governed_paths_clean"], bool), str(st)[:120])
    check("S3 another expected commit stops", stops("repo_state", Q.repo_state, "f" * 40))


# ----------------------------------------------------------------------------------------------------- S4
def s4():
    import torch
    from src.distill.nmf_stream import NMFStream
    from scripts.qual_predict import StreamSkip

    class FakeNMF(torch.nn.Module):
        SUPPORTS_NMF_STREAM = True

        def __init__(self):
            super().__init__()
            self.nmf_stream = None

        def _build_bases(self, B, S, D, R, device=None):
            return self.nmf_stream.draw(lambda: torch.rand((B * S, D, R)))

        def forward(self, x):                                           # one draw per forward, as NMF2D
            return self._build_bases(x.shape[0], 1, 8, 4, device=x.device).sum() + x.sum() * 0

    def fresh():
        net = torch.nn.Sequential(FakeNMF())
        net[0].nmf_stream = NMFStream(42, "M4-V")
        return net

    full, outs = fresh(), []
    for _ in range(12):
        outs.append(full(torch.zeros(1, 3)).item())
    net = fresh()
    skip = StreamSkip(net)
    subset = [0, 3, 4, 11]
    got = []
    for m in subset:
        skip.before(m)
        got.append(net(torch.zeros(1, 3)).item())
        skip.after(m)
    check("S4 the advanced stream draws the full pass's bases at the subset's rows",
          got == [outs[m] for m in subset] and net[0].nmf_stream.draws == 12 and skip.skipped == 8)
    check("S4 a subset out of order stops", stops("nmf_order", skip.before, 5))
    caller = torch.get_rng_state()
    net2 = fresh()
    StreamSkip(net2).before(0)
    net2(torch.zeros(1, 3))
    check("S4 the caller's RNG is never consumed", torch.equal(caller, torch.get_rng_state()))


# ----------------------------------------------------------------------------------------------------- S5
FAKE_REPO = {"commit": "0" * 40, "governed_paths_clean": True, "governed_violations": []}   # the smoke's own checkout


def synthetic_run(root: Path, split: str, *, fail_tile=False) -> str:
    import numpy as np
    from PIL import Image

    def png(arr):
        b = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(arr)).save(b, format="PNG")
        return b.getvalue()

    rng = np.random.default_rng(3)
    picks = [{"position": i + 1, "role": r, "manifest_index": 10 * i, "image_id": f"syn{i}"}
             for i, r in enumerate(["p10", "p50", "p90", "r1", "r2", "r3", "r4", "r5", "r6"])]
    s_doc = {"format": Q.SELECT_FORMAT, "split": split, "rule": "synthetic", "tie_reading": "rank", "n_rows": 90,
             "n_eligible": 90, "ranks": None, "shuffle_sha256": None, "picks": picks}
    s_sha = Q.write_new(root / "select" / "S.json", Q.canonical_bytes(s_doc))
    Q.write_new(root / "select" / "record.json", Q.canonical_bytes({"inputs": {"synthetic": True}, "repo": FAKE_REPO,
                                                                    "outputs": {"select/S.json": s_sha}}))

    def canvas(seed):
        r = np.random.default_rng(seed)
        c = np.empty((512, 512, 3), dtype=np.uint8)
        c[:] = (124, 116, 104)
        c[64:448] = r.integers(0, 255, (384, 512, 3), dtype=np.uint8)
        m = np.full((512, 512), 255, dtype=np.uint8)
        m[64:448] = 0
        m[150:300, 100:260] = 1 + seed % 115
        return c, m

    shared = {}
    for p in range(1, 10):
        c, m = canvas(p)
        shared[f"arrays/input/clean/{p}.png"] = png(c)
        shared[f"arrays/mask/{p}.png"] = png(m)
        if p <= 3:
            for cell in Q.CELLS:
                shared[f"arrays/input/{cell}/{p}.png"] = png(np.clip(c.astype(int) + 20, 0, 255).astype(np.uint8))
    big = rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8)
    bigm = np.zeros((900, 1200), dtype=np.uint8)
    bigm[300:600, 200:700] = 1 + 2 % 115
    shared["arrays/original/2_image.png"] = png(big)
    shared["arrays/original/2_mask.png"] = png(bigm)
    for rel, data in shared.items():
        Q.write_new(root / rel, data)
    for model in Q.MODEL_ORDER:
        arrays, checks, values = {}, [], {}
        for cond in Q.CONDITIONS[model]:
            for p in range(1, Q.k_of(cond) + 1):
                _, m = canvas(p)
                pred = np.where(m == 255, 0, m).astype(np.uint8)
                pred[200:230, 150:200] = 40 + Q.MODEL_ORDER.index(model)          # a wrong disease class
                pred[260:280, 120:140] = 60 + p                                    # a second wrong class
                pred[150:170, 100:150] = 0                                         # missed
                pred[100:120, 400:480] = 1 + p % 115                               # a false alarm
                rel = f"arrays/pred/{model}/{cond}/{p}.png"
                arrays[rel] = Q.write_new(root / rel, png(pred))
                ok = not (fail_tile and model == "E3" and cond == Q.CLEAN and p == 4)
                checks.append({"condition": cond, "position": p, "manifest_index": 10 * (p - 1),
                               "image_id": f"syn{p - 1}", "recomputed": 0.5, "official": 0.5 if ok else 0.4,
                               "gap": 0.0 if ok else 0.1, "tolerance": 1e-6, "same_device_class": True,
                               "binding": True, "pass": ok, "counts_equal": ok, "am5_excluded": False})
                values.setdefault(cond, {})[str(p)] = 0.5 + 0.01 * p
        for rel in shared:
            arrays[rel] = Q.sha256_file(root / rel)
        rec = {"format": Q.RECORD_FORMAT, "step": "predict", "model": model, "split": split, "repo": FAKE_REPO,
               "select_sha256": s_sha, "streamcheck": None, "inputs": {"synthetic": True},
               "versions": {"synthetic": True},
               "outputs": {"checks": Q.write_new(root / "checks" / f"{model}.json", Q.canonical_bytes(checks)),
                           "values": Q.write_new(root / "values" / f"{model}.json", Q.canonical_bytes(values)),
                           "arrays": arrays},
               "summary": {"all_pass": all(c["pass"] for c in checks)}}
        Q.write_new(root / "records" / f"predict_{model}.json", Q.canonical_bytes(rec))
    return s_sha


def s5(font: str):
    import numpy as np
    from PIL import Image
    names = [""] + [f"Plant {i // 5} disease {i}" if i % 7 else f"A much longer plant name {i} with a long disease name"
                    for i in range(1, 116)]
    tmp = Path(tempfile.mkdtemp(prefix="qual1_smoke_"))
    real_repo_state, real_roc = R.repo_state, R.render_only_change
    R.repo_state = lambda expect: dict(FAKE_REPO)        # synthetic records name a synthetic checkout
    try:
        run = tmp / "run_val"
        s_sha = synthetic_run(run, "val")
        rc = R.main(["--run", str(run), "--split", "val", "--select-sha256", s_sha, "--font", font,
                     "--synthetic"], class_names=names)
        figs = sorted(p.name for p in (run / "figures").glob("*.png"))
        want = ["fig4_2.png", "fig4_3a.png", "fig4_3b.png", "fig4_3c.png", "fig4_4a.png", "fig4_4b.png",
                "fig4_4c.png", "fig4_5a.png", "fig4_5b.png", "fig4_5c.png", "fig4_6.png"]
        check("S5 every figure renders (11 files and a manifest)", rc == 0 and figs == want
              and (run / "figures" / "MANIFEST.json").is_file(), f"rc {rc}, {figs}")
        sizes = []
        for f in want:
            with Image.open(run / "figures" / f) as im:
                sizes.append((im.size[0], round(im.info.get("dpi", (0, 0))[0])))
        check("S5 every figure is 1,950 px wide at 300 dpi", all(s == (1950, 300) for s in sizes), str(sizes))
        check("S5 a second rendering into the same folder is refused",
              R.main(["--run", str(run), "--split", "val", "--select-sha256", s_sha, "--font", font,
                      "--synthetic"], class_names=names) == 1)
        rc_b = R.main(["--run", str(run), "--split", "val", "--select-sha256", s_sha, "--font", font,
                       "--synthetic", "--out", "figures_b"], class_names=names)
        man_a = json.loads((run / "figures" / "MANIFEST.json").read_text(encoding="utf-8"))
        man_b = json.loads((run / "figures_b" / "MANIFEST.json").read_text(encoding="utf-8")) if rc_b == 0 else {}
        check("S5 a re-render from the same arrays gives the same colours and the same figures (QUAL-A-1 Q3)",
              rc_b == 0 and man_a["palette"] == man_b.get("palette") and man_a["figures"] == man_b.get("figures"))
        fonts, _ = R.load_fonts(font, True)
        cv = R.Canvas(fonts)
        _, side = R.columns(4)
        pl = R.pair_lines(cv, names, [(1, 9, 50), (2, 8, 40), (1, 3, 40), (5, 6, 7), (7, 5, 7)], 4000)
        tied = R.confused_pairs(np.array([[3, 3, 1, 1]], dtype=np.uint8), np.array([[5, 4, 6, 2]], dtype=np.uint8))
        check("S5 pairs: three by pixel count, ties to the lower (ground truth, predicted), then 'and n further "
              "pairs' (QUAL-A-1 Q5)", pl[-1] == "and 2 further pairs" and len(pl) == 4 and R.further(1) ==
              "and 1 further pair" and [t[:2] for t in tied] == [(1, 2), (1, 6), (3, 4), (3, 5)], str(pl))
        check("S5 the value label is the record's value as a percentage with two decimals (QUAL-A-1 Q8)",
              " ".join(R.value_lines(cv, 0.5, "test", side)) == "Per-image disease-only mIoU: 50.00%"
              and stops("value_missing", R.value_lines, cv, None, "test", side))
        check("S5 the padding key appears only where a panel shows padding (L1)",
              R.has_pad(np.array([[0, 255]])) and not R.has_pad(np.array([[0, 7]]), np.array([[1, 2]])))
        run2 = tmp / "run_changed"
        s_sha2 = synthetic_run(run2, "val")
        p = run2 / "arrays" / "pred" / "E1" / "clean" / "5.png"
        a = np.asarray(Image.open(p)).copy()
        a[0, 0] = 3
        p.unlink()
        Image.fromarray(a).save(p)
        check("S5 a changed array is refused", stops("array_changed", R.Inputs, run2, "val", s_sha2))
        run3 = tmp / "run_gate"
        s_sha3 = synthetic_run(run3, "test", fail_tile=True)
        check("S5 on TEST a failed tile stops the step before anything is drawn",
              stops("check_incomplete", R.Inputs, run3, "test", s_sha3))
        run4 = tmp / "run_gate_ok"
        s_sha4 = synthetic_run(run4, "test")
        try:
            R.Inputs(run4, "test", s_sha4)
            ok4 = True
        except Q.QualStop:
            ok4 = False
        check("S5 on TEST a run whose every tile passed opens", ok4)
        check("S5 a path naming 'test' is refused", stops("test_path", Q.no_test, tmp / "a_TeSt_folder"))
        run5 = tmp / "run_quiet"
        run5.mkdir()
        secret = "syn_image_id_0042"

        def boom(_args):
            raise ValueError(f"a library message naming {secret}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc5 = Q.run_step("render", argparse.Namespace(run=str(run5), split="test"), boom)
        log = (run5 / "logs" / "render.txt").read_text(encoding="utf-8")
        check("S5 on TEST another error prints its type only; its message goes to the run's log",
              rc5 == 1 and secret not in buf.getvalue() and "unexpected_error: ValueError" in buf.getvalue()
              and secret in log, buf.getvalue().strip())
        R.repo_state = lambda expect: dict(FAKE_REPO, commit="1" * 40)
        run6 = tmp / "run_other_commit"
        s_sha6 = synthetic_run(run6, "val")
        check("S5 a run whose steps ran from another checkout is refused",
              R.main(["--run", str(run6), "--split", "val", "--select-sha256", s_sha6, "--font", font,
                      "--synthetic"], class_names=names) == 1 and not (run6 / "figures").exists())
        # QUAL-A-1 Q10: a re-render from a commit that differs from the predict commit in qual_render.py alone
        R.render_only_change = lambda repo, base, head: ["scripts/qual_render.py"]
        base = ["--run", str(run6), "--split", "val", "--select-sha256", s_sha6, "--font", font, "--synthetic"]
        rc_r = R.main(base + ["--rerender", "L1", "--out", "figures_r2"], class_names=names)
        man_r = json.loads((run6 / "figures_r2" / "MANIFEST.json").read_text(encoding="utf-8")) if rc_r == 0 else {}
        R.render_only_change = lambda repo, base, head: ["scripts/qual_predict.py", "scripts/qual_render.py"]
        rc_x = R.main(base + ["--rerender", "L1", "--out", "figures_r3"], class_names=names)
        check("S5 a re-render (QUAL-A-1 Q10) opens when only qual_render.py changed, records both commits, and is "
              "refused when anything else changed", rc_r == 0 and (man_r.get("rerender") or {}).get("predict_commit")
              == FAKE_REPO["commit"] and man_r.get("repo", {}).get("commit") == "1" * 40 and rc_x == 1
              and not (run6 / "figures_r3").exists())
        R.render_only_change = lambda repo, base, head: ["scripts/qual_render.py"]
        tbase = ["--run", str(run4), "--split", "test", "--select-sha256", s_sha4, "--font", font, "--out", "fr",
                 "--expect-commit", "1" * 40]
        check("S5 on TEST a re-render needs its decision-log row and the audited commit named (QUAL-A-1 Q10)",
              stops("rerender_refused", R.rerender_check, R.build_parser().parse_args(tbase + ["--rerender", "L1"]),
                    {"commit": "1" * 40}, {"x": FAKE_REPO})
              and stops("rerender_refused", R.rerender_check, R.build_parser().parse_args(
                  tbase + ["--rerender", "DL-9", "--predict-commit", "2" * 40]), {"commit": "1" * 40},
                  {"x": FAKE_REPO})
              and R.rerender_check(R.build_parser().parse_args(tbase + ["--rerender", "DL-9", "--predict-commit",
                                                                        FAKE_REPO["commit"]]),
                                   {"commit": "1" * 40}, {"x": FAKE_REPO})["changed"] == ["scripts/qual_render.py"])
        R.render_only_change = real_roc
        check("S5 the render-only check reads git: an ancestor whose diff is qual_render.py alone, else refused",
              git_rerender_case(tmp))
    finally:
        R.repo_state = real_repo_state
        R.render_only_change = real_roc
        shutil.rmtree(tmp, ignore_errors=True)


def git_rerender_case(tmp: Path) -> bool:
    """R.render_only_change on a throwaway git repository: base, a render-only child, and a child that also
    changes another file."""
    import subprocess
    g = tmp / "gitcase"
    (g / "scripts").mkdir(parents=True)

    def git(*a):
        return subprocess.run(["git", "-C", str(g), "-c", "user.name=smoke", "-c", "user.email=smoke@example.invalid",
                               *a], capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    for name in ("qual_render.py", "qual_predict.py"):
        (g / "scripts" / name).write_text("v1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    b = git("rev-parse", "HEAD")
    (g / "scripts" / "qual_render.py").write_text("v2\n", encoding="utf-8")
    git("commit", "-q", "-am", "render only")
    h1 = git("rev-parse", "HEAD")
    (g / "scripts" / "qual_predict.py").write_text("v2\n", encoding="utf-8")
    git("commit", "-q", "-am", "predict too")
    h2 = git("rev-parse", "HEAD")
    return (R.render_only_change(g, b, h1) == ["scripts/qual_render.py"]
            and R.render_only_change(g, b, h2) == ["scripts/qual_predict.py", "scripts/qual_render.py"]
            and stops("rerender_refused", R.render_only_change, g, h2, b))


# ----------------------------------------------------------------------------------------------------- S6
MACHADO = {   # Machado, Oliveira and Fernandes (2009), severity 1.0, applied in linear sRGB
    "normal": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)),
}


def lab(rgb, kind):
    import numpy as np
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ np.asarray(MACHADO[kind]).T
    xyz = np.clip(lin, 0, 1) @ np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                                         [0.0193339, 0.1191920, 0.9503041]]).T
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de2000(l1, l2):
    import numpy as np
    L1, a1, b1 = l1[..., 0], l1[..., 1], l1[..., 2]
    L2, a2, b2 = l2[..., 0], l2[..., 1], l2[..., 2]
    cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cb ** 7 / (cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    dh = np.where(c1p * c2p == 0, 0, dh)
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    lbp, cbp, hs = (L1 + L2) / 2, (c1p + c2p) / 2, h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, (hs + 360) / 2, hs / 2)
    hbp = np.where(c1p * c2p == 0, hs, hbp)
    t = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    rt = -np.sin(np.radians(60 * np.exp(-((hbp - 275) / 25) ** 2))) * 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    return np.sqrt(((L2 - L1) / sl) ** 2 + ((c2p - c1p) / sc) ** 2 + (dH / sh) ** 2 + rt * ((c2p - c1p) / sc) * (dH / sh))


def worst(a, b) -> float:
    return float(min(de2000(lab(a, k), lab(b, k)) for k in MACHADO))


def palette_generate() -> list:
    """CLASS_COLOURS from scratch (about two minutes): the greedy max-min list qual_render documents."""
    import numpy as np
    lv = np.arange(0, 256, 17)
    cand = np.array([[r, g, b] for r in lv for g in lv for b in lv], dtype=np.float64)
    ln = lab(cand, "normal")
    cand = cand[(ln[:, 0] >= 45.0) & (np.hypot(ln[:, 1], ln[:, 2]) >= 15.0)]
    labs = {k: lab(cand, k) for k in MACHADO}
    mind = np.full(len(cand), np.inf)
    for s in ((0, 0, 0), (128, 128, 128)):
        mind = np.minimum(mind, np.min([de2000(labs[k], lab(np.array([s], dtype=np.float64), k)) for k in MACHADO],
                                       axis=0))
    out = []
    for _ in range(115):
        i = int(np.argmax(mind))
        out.append("#%02X%02X%02X" % tuple(int(v) for v in cand[i]))
        mind = np.minimum(mind, np.min([de2000(labs[k], labs[k][i][None, :]) for k in MACHADO], axis=0))
    return out


def s6(regenerate: bool):
    import hashlib
    import itertools
    check("S6 CLASS_COLOURS is the measured list (115 colours, sha256 pinned)",
          len(R.CLASS_COLOURS) == 115 and len(set(R.CLASS_COLOURS)) == 115 and
          hashlib.sha256(json.dumps(list(R.CLASS_COLOURS)).encode()).hexdigest() == R.CLASS_COLOURS_SHA256)
    keys = list(R.CATEGORY.values())
    pair = min(worst(a, b) for a, b in itertools.combinations(keys, 2))
    grey = min(worst(k, (v, v, v)) for k in keys for v in range(110, 256, 5))
    check("S6 the four categorical colours stay apart for colour-blind readers (worst CIEDE2000 >= 15 between "
          "them and from the greys)", pair >= 15 and grey >= 15, f"pairs {pair:.1f}, greys {grey:.1f}")
    first8 = [R.hex_rgb(h) for h in R.CLASS_COLOURS[:8]] + [R.BACKGROUND, R.IGNORE]
    m8 = min(worst(a, b) for a, b in itertools.combinations(first8, 2))
    check("S6 the first eight class colours, black and the padding grey stay apart (worst CIEDE2000 >= 14)",
          m8 >= 14, f"{m8:.1f}")
    under = worst(R.IGNORE, (R.UNDERLAY_MIN,) * 3)
    check("S6 the padding grey stays apart from the lightest-to-darkest map underlay (worst CIEDE2000 >= 10)",
          under >= 10, f"{under:.1f}")
    if regenerate:
        check("S6 CLASS_COLOURS regenerates exactly", palette_generate() == list(R.CLASS_COLOURS))


# every repo module the three steps import (select, predict on clean and corrupted conditions, render), TEST and
# VAL paths alike; S0 loads them all in this checkout, so a module the clone leaves out stops the smoke, not a run
STEP_MODULES = ("scripts.qual_select", "scripts.qual_predict", "scripts.qual_render",
                "src.eval", "src.eval.artifacts", "src.eval.protocols", "src.eval.adapters", "src.eval.evaluate",
                "src.eval.eval_runtime", "src.eval.model_loading", "src.eval.stage_artifacts", "src.eval.corrupted",
                "src.eval.robustness", "src.data.dataset", "src.data.transforms", "src.distill.nmf_stream",
                "src.corruption_cache", "scripts.evaluate_model", "scripts.evaluate_corruptions",
                "src.stats.heldout_inputs", "src.stats.ingest", "src.stats.driver", "src.tman.validate",
                "src.tman.schema", "src.tman.binding")


def s0():
    import importlib
    failed = []
    for name in STEP_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:                               # noqa: BLE001 -- reported by name
            failed.append(f"{name}: {type(e).__name__}: {e}")
    check(f"S0 every module the three steps import loads in this checkout ({len(STEP_MODULES)} modules)",
          not failed, "; ".join(failed)[:400])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--font", required=True, help="any TrueType font file (the smoke's figures are synthetic)")
    ap.add_argument("--no-torch", action="store_true", help="skip S3 and S4 (only where torch is absent)")
    ap.add_argument("--palette", action="store_true", help="also regenerate CLASS_COLOURS (about two minutes)")
    args = ap.parse_args(argv)
    s1_s2()
    if args.no_torch:
        print("SKIP S0 S3 S4: --no-torch (not a PASS of the full smoke)")
    else:
        s0()
        s3()
        s4()
    s5(args.font)
    s6(args.palette)
    n, ok = len(RESULTS), sum(RESULTS)
    full = not args.no_torch and ok == n
    print(f"RESULT: QUAL SMOKE {'OK' if full else 'PARTIAL' if ok == n else 'FAILED'} ({ok}/{n})")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
