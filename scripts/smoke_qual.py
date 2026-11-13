#!/usr/bin/env python3
"""Smoke for the QUAL-1 kit (scripts/qual_select.py, qual_predict.py, qual_render.py), synthetic data only.

No dataset, no model file, no record and no TEST or VAL file is read: every input is made here, in a
temporary folder (configs/plantseg_class_map.json, a committed config, gives S9 its 115 names). Checks:
  S0  every repo module the three steps import loads in this checkout (a module the sparse clone leaves out
      stops the smoke, not a run)
  S1  DL-182's rule: eligibility from the rows (a row without it stops), nearest ranks, ties to the lower
      manifest index under both tie readings, a duplicate percentile pick stops, six random picks outside
      the percentile picks in shuffle order; the tie-readings word (ruling QUAL-A-1 Q2); S.json's nonce (N-01)
  S2  the version guard refuses another interpreter, version or image digest, and no random pick is drawn
      without it; random.Random(42).sample's known answer (measured under CPython 3.11.15 and 3.13)
  S3  the check of AM-27 item 4: the evaluator's own evaluate_model on a synthetic batch gives the recorded
      value back (gap 0) and a one-pixel change of the prediction fails the 1e-6 tolerance; the confusion
      counts follow the prediction and are compared by class (Q6); the class-level error map's TP pixels
      equal the record's tp sum and padding is none of TP, FN or FP (Q4); the counts recomputed from the saved
      arrays equal the pass's (N-14); core_preprocess of an original pair gives its canvas (N-09)
  S4  the teacher's stream advance (QUAL-1 Q1): a subset with skipped rows draws exactly the bases a full
      pass draws at those rows
  S5  every figure renders from synthetic arrays (11 PNG files at 300 dpi, 1,950 px wide; the manifest beside
      the figures folder, N-06), a re-render from the same arrays gives the same colours (Q3), the pair legend
      (Q5, F-05), the value label (Q8), the padding key (L1), a changed array is refused, on TEST the render
      gate refuses a failed tile's `pass` alone and a record's `all_pass` alone before anything is drawn
      (T7 (2)), on TEST an unexpected error prints its type only (its message, which may name an image, goes
      to the run's log), a re-render opens only from a render-only commit (Q10), outputs that exist already
      and a run folder inside the repository are refused before any read (N-03, N-04)
  S6  the class palette is the measured list and the categorical keys are colour-blind-safe; the padding grey
      stays apart from the map underlay
  S7  the pass rule in one function (T7 (1)): the 1e-6 and 0.5 pp tolerances at their edges, the device rule
      (the record's model device), the counts on the same device class; the execution context (F-03)
  S8  wiring (T7 (3)): every tile of synthetic arrays encodes its position, model and condition; Figures 4.2 to
      4.6 are read back at the tile centres and show the right tile in the right place; class maps and the
      categorical layers hold no colour outside their legend (N-10)
  S9  the 115 class names of configs/plantseg_class_map.json, each as ground truth and as prediction, wrap at a
      space within the Figure 4.6 column, never cut (F-07); every class legend keys black "Background" (F-08)
  S10 the TEST-only functions on made-up files (T7 (4)): LREP's open_inventory reads a made-up export and binding
      through LREP's own stub reader standing in for TMAN's; load_test_official, check_record_identity and
      manifest_order on made-up artifacts (LREP's load_bound stood in: the binding record and the four files'
      hashes are real); open_cell's official branch (the evaluator's two cache gates stood in, the pins real);
      the preflight P1 passes, and refuses a record's context, a second repo_commit, a changed file, another
      listing and a cache refusal (exit 3, REFUSED, nothing per-image named); select's TEST path end to end

  python -B scripts/smoke_qual.py --font PATH_TO_A_TTF      (inside the KD image: torch for S0, S3, S4, S10)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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


def tmpdir(prefix: str) -> Path:
    """A new temporary folder whose path never names 'test' (the qual steps refuse such paths)."""
    while True:
        p = Path(tempfile.mkdtemp(prefix=prefix))
        if "test" not in str(p).casefold():
            return p
        p.rmdir()


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
    doc_a = Q.s_document("val", sel, [f"img{m:03d}" for m in range(40)], rule="r", reading="rank")
    doc_b = Q.s_document("val", sel, [f"img{m:03d}" for m in range(40)], rule="r", reading="rank")
    check("S1 S.json carries a fresh 32-hex nonce that changes its sha256 and nothing else (QUAL-AUD-1 N-01)",
          re.fullmatch(r"[0-9a-f]{32}", doc_a["nonce"]) is not None and doc_a["nonce"] != doc_b["nonce"]
          and {k: v for k, v in doc_a.items() if k != "nonce"} == {k: v for k, v in doc_b.items() if k != "nonce"})
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
    from scripts.qual_predict import counts_of_arrays, original_matches
    pr_moved = changed[0].argmax(0).numpy().astype(np.uint8)
    check("S3 the counts recomputed from the saved arrays equal the pass's, and follow a changed prediction "
          "(QUAL-AUD-1 N-14)", counts_of_arrays(pr_np.astype(np.uint8), gt_np.astype(np.uint8))
          == result_counts(ref, "img000")
          and counts_of_arrays(pr_moved, gt_np.astype(np.uint8)) == result_counts(moved, "img000"))
    from PIL import Image
    from src.data.transforms import core_preprocess
    rgb = np.random.default_rng(5).integers(0, 255, (300, 437, 3), dtype=np.uint8)
    omask = np.zeros((300, 437), dtype=np.uint8)
    omask[100:200, 50:300] = 9
    cv_, mc_ = core_preprocess(Image.fromarray(rgb), Image.fromarray(omask))
    bad = mc_.copy()
    bad[256, 256] = 3
    check("S3 core_preprocess of the original pair gives the saved canvas and mask, and a changed mask is caught "
          "(QUAL-AUD-1 N-09)", original_matches(rgb, omask, cv_, mc_) and not original_matches(rgb, omask, cv_, bad))
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
FAIL_MODES = (None, "both", "pass_only", "all_pass_only")


def png_of(arr) -> bytes:
    import numpy as np
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, format="PNG")
    return b.getvalue()


def write_run(root: Path, split: str, *, canvas, mask, pred, original, fail_tile=None) -> str:
    """A synthetic run folder as the select and predict steps leave it: S.json, the select record, the arrays
    (canvas(condition, position), mask(position), pred(model, condition, position), original = (image, mask) of
    position 2), each model's checks, values and record. `fail_tile` (FAIL_MODES): "both" fails E3's clean tile 4
    and its record says so; "pass_only" fails that tile while the record's all_pass stays true; "all_pass_only"
    passes every tile while E3's record says all_pass false."""
    if fail_tile not in FAIL_MODES:
        raise ValueError(fail_tile)
    picks = [{"position": i + 1, "role": r, "manifest_index": 10 * i, "image_id": f"syn{i}"}
             for i, r in enumerate(["p10", "p50", "p90", "r1", "r2", "r3", "r4", "r5", "r6"])]
    s_doc = {"format": Q.SELECT_FORMAT, "split": split, "rule": "synthetic", "tie_reading": "rank", "n_rows": 90,
             "n_eligible": 90, "ranks": None, "shuffle_sha256": None, "picks": picks, "nonce": "0" * 32}
    s_sha = Q.write_new(root / "select" / "S.json", Q.canonical_bytes(s_doc))
    Q.write_new(root / "select" / "record.json", Q.canonical_bytes({"inputs": {"synthetic": True}, "repo": FAKE_REPO,
                                                                    "outputs": {"select/S.json": s_sha}}))
    shared = {}
    for p in range(1, 10):
        shared[f"arrays/input/clean/{p}.png"] = png_of(canvas(Q.CLEAN, p))
        shared[f"arrays/mask/{p}.png"] = png_of(mask(p))
        if p <= 3:
            for cell in Q.CELLS:
                shared[f"arrays/input/{cell}/{p}.png"] = png_of(canvas(cell, p))
    shared["arrays/original/2_image.png"] = png_of(original[0])
    shared["arrays/original/2_mask.png"] = png_of(original[1])
    for rel, data in shared.items():
        Q.write_new(root / rel, data)
    for model in Q.MODEL_ORDER:
        arrays, checks, values = {}, [], {}
        for cond in Q.CONDITIONS[model]:
            for p in range(1, Q.k_of(cond) + 1):
                rel = f"arrays/pred/{model}/{cond}/{p}.png"
                arrays[rel] = Q.write_new(root / rel, png_of(pred(model, cond, p)))
                ok = not (fail_tile in ("both", "pass_only") and model == "E3" and cond == Q.CLEAN and p == 4)
                checks.append({"condition": cond, "position": p, "manifest_index": 10 * (p - 1),
                               "image_id": f"syn{p - 1}", "recomputed": 0.5, "official": 0.5 if ok else 0.4,
                               "gap": 0.0 if ok else 0.1, "tolerance": 1e-6, "same_device_class": True,
                               "binding": True, "pass": ok, "counts_equal": ok, "am5_excluded": False})
                values.setdefault(cond, {})[str(p)] = 0.5 + 0.01 * p
        for rel in shared:
            arrays[rel] = Q.sha256_file(root / rel)
        all_pass = all(c["pass"] for c in checks)
        if model == "E3" and fail_tile == "pass_only":
            all_pass = True
        if model == "E3" and fail_tile == "all_pass_only":
            all_pass = False
        rec = {"format": Q.RECORD_FORMAT, "step": "predict", "model": model, "split": split, "repo": FAKE_REPO,
               "select_sha256": s_sha, "streamcheck": None, "inputs": {"synthetic": True},
               "versions": {"synthetic": True}, "conditions": {},
               "outputs": {"checks": Q.write_new(root / "checks" / f"{model}.json", Q.canonical_bytes(checks)),
                           "values": Q.write_new(root / "values" / f"{model}.json", Q.canonical_bytes(values)),
                           "arrays": arrays},
               "summary": {"all_pass": all_pass}}
        Q.write_new(root / "records" / f"predict_{model}.json", Q.canonical_bytes(rec))
    return s_sha


def synthetic_run(root: Path, split: str, *, fail_tile=None) -> str:
    import numpy as np
    rng = np.random.default_rng(3)

    def canvas_mask(seed):
        r = np.random.default_rng(seed)
        c = np.empty((512, 512, 3), dtype=np.uint8)
        c[:] = (124, 116, 104)
        c[64:448] = r.integers(0, 255, (384, 512, 3), dtype=np.uint8)
        m = np.full((512, 512), 255, dtype=np.uint8)
        m[64:448] = 0
        m[150:300, 100:260] = 1 + seed % 115
        return c, m

    def canvas(cond, p):
        c = canvas_mask(p)[0]
        return c if cond == Q.CLEAN else np.clip(c.astype(int) + 20, 0, 255).astype(np.uint8)

    def pred(model, cond, p):
        m = canvas_mask(p)[1]
        a = np.where(m == 255, 0, m).astype(np.uint8)
        a[200:230, 150:200] = 40 + Q.MODEL_ORDER.index(model)          # a wrong disease class
        a[260:280, 120:140] = 60 + p                                    # a second wrong class
        a[150:170, 100:150] = 0                                         # missed
        a[100:120, 400:480] = 1 + p % 115                               # a false alarm
        return a

    big = rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8)
    bigm = np.zeros((900, 1200), dtype=np.uint8)
    bigm[300:600, 200:700] = 1 + 2 % 115
    return write_run(root, split, canvas=canvas, mask=lambda p: canvas_mask(p)[1], pred=pred, original=(big, bigm),
                     fail_tile=fail_tile)


def s5(font: str):
    import numpy as np
    from PIL import Image
    names = [""] + [f"Plant {i // 5} disease {i}" if i % 7 else f"A much longer plant name {i} with a long disease name"
                    for i in range(1, 116)]
    tmp = tmpdir("qual1_smoke_")
    real_repo_state, real_roc = R.repo_state, R.render_only_change
    R.repo_state = lambda expect: dict(FAKE_REPO)        # synthetic records name a synthetic checkout
    try:
        run = tmp / "run_val"
        s_sha = synthetic_run(run, "val")
        rc = R.main(["--run", str(run), "--split", "val", "--select-sha256", s_sha, "--font", font,
                     "--synthetic"], class_names=names)
        figs = sorted(p.name for p in (run / "figures").iterdir())
        want = ["fig4_2.png", "fig4_3a.png", "fig4_3b.png", "fig4_3c.png", "fig4_4a.png", "fig4_4b.png",
                "fig4_4c.png", "fig4_5a.png", "fig4_5b.png", "fig4_5c.png", "fig4_6.png"]
        check("S5 every figure renders (11 PNG files, nothing else, in the figures folder; the manifest beside it: "
              "QUAL-AUD-1 N-06)", rc == 0 and figs == want and (run / "manifest_figures.json").is_file(),
              f"rc {rc}, {figs}")
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
        man_a = json.loads((run / "manifest_figures.json").read_text(encoding="utf-8"))
        man_b = json.loads((run / "manifest_figures_b.json").read_text(encoding="utf-8")) if rc_b == 0 else {}
        check("S5 a re-render from the same arrays gives the same colours and the same figures (QUAL-A-1 Q3)",
              rc_b == 0 and man_a["palette"] == man_b.get("palette") and man_a["figures"] == man_b.get("figures"))
        fonts, _ = R.load_fonts(font, True)
        cv = R.Canvas(fonts)
        _, side = R.columns(4)
        pl = R.pair_lines(cv, names, [(1, 9, 50), (2, 8, 40), (1, 3, 40), (5, 6, 7), (7, 5, 7)], 4000)
        tied = R.confused_pairs(np.array([[3, 3, 1, 1]], dtype=np.uint8), np.array([[5, 4, 6, 2]], dtype=np.uint8))
        check("S5 pairs: three by pixel count, ties to the lower (ground truth, predicted), then 'and further pairs' "
              "with no number (QUAL-A-1 Q5; QUAL-AUD-1 F-05)", pl[-1] == "and further pairs" and len(pl) == 4
              and not any(ch.isdigit() for ch in pl[-1]) and R.pair_lines(cv, names, [(1, 9, 5)], 4000)[-1]
              != "and further pairs" and [t[:2] for t in tied] == [(1, 2), (1, 6), (3, 4), (3, 5)], str(pl))
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
        gate = {}
        for mode in ("both", "pass_only", "all_pass_only"):
            r3 = tmp / f"run_gate_{mode}"
            gate[mode] = stops("check_incomplete", R.Inputs, r3, "test", synthetic_run(r3, "test", fail_tile=mode))
        check("S5 on TEST the gate stops before anything is drawn when a tile failed and its record says so, when a "
              "tile's pass is false alone, and when a record's all_pass is false alone (QUAL-AUD-1 T7 (2))",
              all(gate.values()), str(gate))
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
        # QUAL-AUD-1 N-03 and N-04: outputs that exist already, and a run folder inside the repository
        r7 = tmp / "run_exists"
        (r7 / "select").mkdir(parents=True)
        (r7 / "select" / "S.json").write_text("{}", encoding="utf-8")
        (r7 / "manifest_figures.json").write_text("{}", encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc7 = Q.main(["--run", str(r7), "--split", "val", "--val-artifact", str(tmp / "absent")])
            rc8 = R.main(["--run", str(r7), "--split", "val", "--select-sha256", "0" * 64, "--font", font,
                          "--synthetic"], class_names=names)
        check("S5 a step whose outputs exist already stops before any read: select (refused, exit 3), render (exit 1) "
              "(QUAL-AUD-1 N-03)", rc7 == 3 and rc8 == 1 and buf.getvalue().count("output_exists") == 2
              and not (r7 / "figures").exists(), buf.getvalue().strip()[-300:])
        check("S5 a run folder inside the repository is refused (QUAL-AUD-1 N-04)",
              stops("run_in_repo", Q.run_dir, Q.REPO / "scripts"))
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
        man_r = json.loads((run6 / "manifest_figures_r2.json").read_text(encoding="utf-8")) if rc_r == 0 else {}
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


# ----------------------------------------------------------------------------------------------------- S7
def s7():
    from scripts.qual_predict import OTHER_DEVICE_TOL, SAME_DEVICE_TOL, tile_verdict

    def v(a, b, eq=True, same=True):
        return tile_verdict(a, b, counts_equal=eq, same_device=same)

    check("S7 the pass rule (T7 (1)): on the record's device class a gap of exactly 1e-6 passes, a larger one fails",
          v(0.0, 1e-6)[0] and v(0.0, 1e-6)[2] == SAME_DEVICE_TOL == 1e-6 and not v(0.0, 1.0000001e-6)[0]
          and v(0.25, 0.25) == (True, 0.0, 1e-6))
    check("S7 the pass rule: on the record's device class unequal confusion counts fail, even at gap 0",
          not v(0.5, 0.5, eq=False)[0])
    check("S7 the pass rule: on another device class the tolerance is 0.5 pp and the counts do not bind",
          v(0.5, 0.504, eq=False, same=False)[0] and not v(0.5, 0.506, same=False)[0]
          and v(0.5, 0.5, same=False)[2] == OTHER_DEVICE_TOL == 0.005)
    check("S7 the pass rule: a missing, non-finite or non-float value fails",
          not any(v(a, b)[0] for a, b in ((0.5, None), (None, 0.5), (0.5, float("nan")), (float("inf"), 0.5),
                                         (0.5, 1), (True, 1.0))) and v(0.5, None)[1] is None)

    def summ(**er):
        base = {"torch_num_threads": 1, "batch_size": 1, "image_digest": Q.KD_DIGEST, "model_device": "cpu"}
        return {"run": {"eval_runtime": dict(base, **er)}}

    check("S7 the device rule: the record's model device as the evaluator recorded it, 'cpu' only (F-03)",
          Q.same_device_class(summ()) and not any(Q.same_device_class(summ(model_device=d))
                                                  for d in ("cuda:0", "CPU", "cpu:0", None))
          and not Q.same_device_class({}))
    guard = {"image_digest": Q.KD_DIGEST}
    check("S7 the execution context (QUAL-AUD-1 F-03): T 1, batch 1, the guard's image, model device cpu; each "
          "departure is named", Q.record_context_problems(summ(), guard) == []
          and Q.record_context_problems(summ(torch_num_threads=8), guard) == ["torch_num_threads"]
          and Q.record_context_problems(summ(torch_num_threads=True), guard) == ["torch_num_threads"]
          and Q.record_context_problems(summ(batch_size=2), guard) == ["batch_size"]
          and Q.record_context_problems(summ(image_digest="sha256:" + "b" * 64), guard) == ["image_digest"]
          and Q.record_context_problems(summ(model_device="cuda:0"), guard) == ["model_device"]
          and Q.record_context_problems(summ(), {"image_digest": None}) == ["image_digest"]
          and len(Q.record_context_problems({}, guard)) == 4)


# ----------------------------------------------------------------------------------------------------- S8
W_COMBOS = [(m, c, p) for m in Q.MODEL_ORDER for c in Q.CONDITIONS[m] for p in range(1, Q.k_of(c) + 1)]
W_CONDS = (Q.CLEAN, *Q.CELLS)
W_EDGES = (0, 171, 341, 512)
W_RING = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
W_CENTRES = (85, 256, 426)
W_ORIG_CLASS, W_ORIG_RGB = 110, (7, 77, 177)


def w_code(m, c, p) -> int:          # a prediction's centre class: 1..63, one per (model, condition, position)
    return 1 + W_COMBOS.index((m, c, p))


def w_gt(p) -> int:                  # the ground truth's disease class of a position: 101..109
    return 100 + p


def w_bits(m, c, p) -> int:          # eight bits in the ring of cells: TP where a bit is 1, missed where it is 0
    return (Q.MODEL_ORDER.index(m) * 4 + W_CONDS.index(c)) * 16 + p


def w_rgb(c, p) -> tuple:            # a canvas' one colour, one per (condition, position)
    return (25 * p, 50 * W_CONDS.index(c) + 20, 128)


def wiring_run(root: Path) -> str:
    import numpy as np

    def canvas(c, p):
        a = np.empty((512, 512, 3), dtype=np.uint8)
        a[:] = w_rgb(c, p)
        return a

    def mask(p):
        m = np.full((512, 512), 255, dtype=np.uint8)
        m[64:448] = w_gt(p)
        return m

    def pred(m, c, p):
        a = np.zeros((512, 512), dtype=np.uint8)
        e = W_EDGES
        a[e[1]:e[2], e[1]:e[2]] = w_code(m, c, p)
        for k, (i, j) in enumerate(W_RING):
            if w_bits(m, c, p) >> k & 1:
                a[e[i]:e[i + 1], e[j]:e[j + 1]] = w_gt(p)
        return a

    orig = np.empty((300, 400, 3), dtype=np.uint8)
    orig[:] = W_ORIG_RGB
    return write_run(root, "val", canvas=canvas, mask=mask, pred=pred,
                     original=(orig, np.full((300, 400), W_ORIG_CLASS, dtype=np.uint8)))


def rows_of(placed) -> list:
    """The panels of a figure as rows (by their centre's height) of panels left to right."""
    rows = []
    for t in sorted(placed, key=lambda t: (t[1] + t[3] / 2, t[0])):
        cy = t[1] + t[3] / 2
        if rows and abs(rows[-1][0] - cy) <= 3:
            rows[-1][1].append(t)
        else:
            rows.append((cy, [t]))
    return [sorted(r, key=lambda t: t[0]) for _, r in rows]


def s8(font: str):
    import numpy as np
    from PIL import Image
    tmp = tmpdir("qual1_smoke_wiring_")
    try:
        run = tmp / "run"
        inp = R.Inputs(run, "val", wiring_run(run))
        names = [""] + [f"class {i}" for i in range(1, 116)]
        present = R.present_classes(inp)
        lut = R.class_lut(present)
        pset = set(present)
        rev = {tuple(int(x) for x in lut[c]): c for c in present}
        bits_of = {w_bits(*k): k for k in W_COMBOS}
        code_of = {w_code(*k): k for k in W_COMBOS}
        fonts, _ = R.load_fonts(font, True)
        tp, fn, wr, ig = (tuple(R.CATEGORY["tp"]), tuple(R.CATEGORY["fn"]), tuple(R.CATEGORY["wrong"]),
                          tuple(R.IGNORE))

        def px(img, t, sy, sx):
            ox, oy, nw, nh = t
            return tuple(int(v) for v in img[oy + int(sy * nh / 512), ox + int(sx * nw / 512)])

        def colours(img, t):
            ox, oy, nw, nh = t
            return {tuple(int(v) for v in c) for c in np.unique(img[oy:oy + nh, ox:ox + nw].reshape(-1, 3), axis=0)}

        def probe(img, t, kind):
            """What a panel shows: ('photo', (condition, position)) / ('photo', 'original'), ('class', id), or
            (kind, (model, condition, position)) decoded from the ring of an error map or overlay."""
            ox, oy, nw, nh = t
            centre = tuple(int(v) for v in img[oy + nh // 2, ox + nw // 2])
            if kind == "photo":
                if centre == W_ORIG_RGB:
                    return ("photo", "original")
                hits = [(c, p) for c in W_CONDS for p in range(1, 10) if w_rgb(c, p) == centre]
                return ("photo", hits[0] if len(hits) == 1 else None)
            if kind == "class":
                return ("class", rev.get(centre))
            one, zero = tp, fn
            b = 0
            for k, (i, j) in enumerate(W_RING):
                c = px(img, t, W_CENTRES[i], W_CENTRES[j])
                if c not in (one, zero):
                    return (kind, None)
                b |= (c == one) << k
            return (kind, bits_of.get(b), centre)

        def pred_cell(m, c, p):
            return ("class", w_code(m, c, p))

        def err(m, p):
            return ("error", (m, Q.CLEAN, p), fn)

        def conf(m, p):
            return ("conf", (m, Q.CLEAN, p), wr)

        want = {
            "fig4_2": [[("photo", "original"), ("class", W_ORIG_CLASS), ("photo", (Q.CLEAN, 2)), err("E6", 2)],
                       [pred_cell(m, Q.CLEAN, 2) for m in Q.MODEL_ORDER]],
            "fig4_6": [[("photo", (Q.CLEAN, p))] + [conf(m, p) for m in Q.MODEL_ORDER] for p in (1, 2, 3)]}
        for part, ps in zip("abc", ((1, 2, 3), (4, 5, 6), (7, 8, 9))):
            want[f"fig4_3{part}"] = [[("photo", (Q.CLEAN, p)), ("class", w_gt(p))]
                                     + [pred_cell(m, Q.CLEAN, p) for m in Q.MODEL_ORDER] for p in ps]
            want[f"fig4_4{part}"] = [[("photo", (Q.CLEAN, p))] + [err(m, p) for m in Q.MODEL_ORDER] for p in ps]
        for part, pos in zip("abc", (1, 2, 3)):
            want[f"fig4_5{part}"] = [[("photo", (c, pos)), ("class", w_gt(pos))]
                                     + [pred_cell(m, c, pos) for m in ("E1", "E3", "E6")] for c in W_CONDS]
        draw = {"fig4_2": lambda cv: R.fig4_2(inp, cv, lut, pset, names),
                "fig4_6": lambda cv: R.fig4_6(inp, cv, names, [1, 2, 3])}
        for part, ps in zip("abc", ([1, 2, 3], [4, 5, 6], [7, 8, 9])):
            draw[f"fig4_3{part}"] = lambda cv, ps=ps: R.fig4_3(inp, cv, lut, pset, names, ps)
            draw[f"fig4_4{part}"] = lambda cv, ps=ps: R.fig4_4(inp, cv, ps)
        for part, pos in zip("abc", (1, 2, 3)):
            draw[f"fig4_5{part}"] = lambda cv, pos=pos: R.fig4_5(inp, cv, lut, pset, names, pos)
        wrong_place, stray = [], []
        for name in sorted(want):
            cv = R.Canvas(fonts)
            img = np.asarray(Image.open(io.BytesIO(draw[name](cv))).convert("RGB"))
            grid = rows_of(cv.placed)
            shape = [len(r) for r in grid]
            if shape != [len(r) for r in want[name]]:
                wrong_place.append(f"{name}: panels {shape}")
                continue
            for r, (row, wrow) in enumerate(zip(grid, want[name])):
                for c, (t, w) in enumerate(zip(row, wrow)):
                    got = probe(img, t, w[0])
                    if got != w:
                        wrong_place.append(f"{name} row {r + 1} column {c + 1}: {got} for {w}")
                    if w[0] == "class":                                      # N-10: no colour outside the legend
                        k = code_of.get(w[1])
                        allowed = ({tuple(int(x) for x in lut[w[1]]), (0, 0, 0)} if w[1] == W_ORIG_CLASS else
                                   {tuple(int(x) for x in lut[w[1]]), ig} if k is None else
                                   {tuple(int(x) for x in lut[w[1]]), tuple(int(x) for x in lut[w_gt(k[2])]),
                                    (0, 0, 0)})
                    elif w[0] in ("error", "conf"):
                        allowed = {tp, fn, ig} if w[0] == "error" else {tp, wr, fn, ig}
                    else:
                        continue
                    extra = colours(img, t) - allowed
                    if extra:
                        stray.append(f"{name} row {r + 1} column {c + 1}: {len(extra)} colours outside the legend")
        check("S8 wiring (T7 (3)): every panel of Figures 4.2 to 4.6 shows its own position, model and condition "
              "in its own place", not wrong_place, "; ".join(wrong_place[:4]))
        blend = Image.fromarray(lut[inp.pred("E1", Q.CLEAN, 1)]).resize((379, 379), Image.BOX)
        check("S8 class maps, error maps and overlays hold no colour outside their legend (nearest; the photographs "
              "box-filtered), where a box filter would blend (QUAL-AUD-1 N-10)",
              not stray and len(np.unique(np.asarray(blend).reshape(-1, 3), axis=0)) > 3, "; ".join(stray[:4]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------- S9
def s9(font: str):
    doc = json.loads((Q.REPO / "configs" / "plantseg_class_map.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in doc["entries"]]
    fonts, info = R.load_fonts(font, True)
    cv = R.Canvas(fonts)
    _, side = R.columns(5)                                   # Figure 4.6's column
    widest = max(range(1, 116), key=lambda c: cv.text_size(R.name_of(names, c), R.TEXT_PT)[0])
    bad = []
    for c in range(1, 116):
        for a, b in ((c, widest), (widest, c)):
            try:
                lines = R.pair_lines(cv, names, [(a, b, 9)], side)
            except Q.QualStop as e:
                bad.append(f"{a}->{b}: {e.code}")
                continue
            joined = " ".join(lines)
            if (joined != f"{R.name_of(names, a)} → {R.name_of(names, b)}" or any("…" in t for t in lines)
                    or not all(cv.fits(t, R.TEXT_PT, side) for t in lines)):
                bad.append(f"{a}->{b}")
    check(f"S9 the 115 class names of configs/plantseg_class_map.json, as ground truth and as prediction beside the "
          f"widest, wrap at a space within the {side} px column of Figure 4.6, never cut (QUAL-AUD-1 F-07; font "
          f"{info['family']})", len(names) == 116 and not bad, "; ".join(bad[:5]))

    class Keys:
        def __init__(self):
            self.entries = None

        def legend(self, entries, size=R.TEXT_PT):
            self.entries = list(entries)

    k1, k2 = Keys(), Keys()
    lut = R.class_lut([3, 9])
    R.class_legend(k1, {3, 9}, lut, names, padding=True)
    R.class_legend(k2, set(), lut, names, padding=False)
    check("S9 every class legend keys black first as 'Background' (QUAL-AUD-1 F-08), then the classes, then the "
          "padding where shown", k1.entries[0] == ((0, 0, 0), "Background") and len(k1.entries) == 4
          and k1.entries[-1] == R.PAD_KEY and k2.entries == [((0, 0, 0), "Background")])


# ---------------------------------------------------------------------------------------------------- S10
def s10():
    """T7 (4): the TEST-only functions and the preflight P1 on made-up files. Real: LREP's open_inventory (its
    binding and freeze rules) reading a made-up export and binding through LREP's own stub reader (TMAN's reader
    validates against the committed catalogue and the 37 stats identities, so a made-up export cannot pass it),
    LREP's binding_record and input_file_sha256, the qual steps' own code and the evaluator's check_cache_sources.
    Stood in: LREP's load_bound (it needs an artifact the evaluator wrote in official mode) and the evaluator's two
    cache gates (they need an official cache). The listing is made up (no TEST folder is read)."""
    import hashlib
    import subprocess
    import numpy as np
    import scripts.evaluate_corruptions as EC
    import src.eval.adapters as AD
    import src.stats.heldout_inputs as HI
    from src.corruption_cache import MANIFEST_NAME
    from src.eval.artifacts import hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    from src.eval.protocols import CANVAS_PROTOCOL_ID
    from src.stats.driver import input_file_sha256
    n = Q.N_SPLIT["test"]
    entries = [ManifestEntry(i, f"syn{i:04d}", f"syn{i:04d}") for i in range(n)]
    lsha = hash_split_manifest(entries)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Q.REPO), capture_output=True, text=True).stdout.strip()
    model_sha = {m: Q.record_model_sha256(m) for m in Q.MODEL_ORDER}
    guard = {"implementation": "CPython", "python": Q.PYTHON, "image_digest": Q.KD_DIGEST}
    tmp = tmpdir("qual1_smoke_t74_")
    calls = {"resolve": [], "record": [], "load_bound": []}
    keep = {(HI, "tman_reader"): HI.tman_reader, (HI, "load_bound"): HI.load_bound,
            (EC, "resolve_corruption_cache"): EC.resolve_corruption_cache,
            (EC, "check_verification_record"): EC.check_verification_record,
            (AD, "build_expected_manifest_for"): AD.build_expected_manifest_for, (Q, "split_listing"): Q.split_listing,
            (Q, "version_guard"): Q.version_guard, (Q, "repo_state"): Q.repo_state}
    cache_err = {}

    def setup(name, *, threads_of=None, commit_of=None, tamper=None):
        """A session checkout with the thirteen made-up artifacts, the export and binding beside a derived folder,
        a cache; returns the select command line for it."""
        root = tmp / name
        session, derived, cache = root / "session", root / "derived", root / "cache"
        recs = []
        rng = np.random.default_rng(11)
        for m, c in Q.RECORD_KEYS:
            rel = f"reports/tday/{m}/{c}"
            d = session / rel
            d.mkdir(parents=True)
            want = {"type": "clean", "name": None, "severity": None} if c == Q.CLEAN else {
                "type": "corruption", "name": c.rpartition("_s")[0], "severity": int(c.rpartition("_s")[2])}
            commit = "e" * 40 if commit_of == (m, c) else head
            summary = {"dataset": {"split": "test", "preprocess_protocol": CANVAS_PROTOCOL_ID, "condition": want,
                                   "actual_rows": n, "split_manifest_sha256": lsha},
                       "run": {"checkpoint_sha256": model_sha[m], "repo_commit": commit, "env": {"device": "cpu"},
                               "eval_runtime": {"torch_num_threads": 8 if threads_of == (m, c) else 1,
                                                "batch_size": 1, "image_digest": Q.KD_DIGEST, "model_device": "cpu"}}}
            vals = rng.random(n)
            rows = "".join(json.dumps({"manifest_index": i, "image_id": e.image_id, "clean_image_id": e.image_id,
                                       "am5_excluded": False, "n_eligible_disease_only": 1,
                                       "disease_only_miou": float(vals[i]), "disease_only_miou_status": "ok"}) + "\n"
                           for i, e in enumerate(entries))
            buf = io.BytesIO()
            np.savez(buf, image_index=np.arange(n), class_id=np.full(n, 7), tp=np.ones(n, dtype=np.int64),
                     gt=np.ones(n, dtype=np.int64), pred=np.ones(n, dtype=np.int64),
                     manifest_ids=np.array([e.image_id for e in entries]))
            payload = {"per_image.jsonl": rows.encode(), "sufficient_stats.npz": buf.getvalue(),
                       "summary.json": Q.canonical_bytes(summary)}
            for k, data in payload.items():
                (d / k).write_bytes(data)
            (d / "MANIFEST.sha256").write_text("".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n"
                                                       for k, v in sorted(payload.items())), encoding="utf-8")
            slot, prec, scope, cell = Q.binding_key(m, c)
            recs.append({"slot": slot, "precision": prec, "scope": scope, "cell": cell, "protocol": "canvas",
                         "repo_relative_path": rel, "files_sha256": input_file_sha256(d),
                         "checkpoint_sha256": model_sha[m], "repo_commit": commit, "split_manifest_sha256": lsha})
            if tamper == (m, c):
                (d / "per_image.jsonl").write_bytes(rows.encode() + b"\n")
        derived.mkdir(parents=True)
        bdata = Q.canonical_bytes({"schema": HI.BINDING_SCHEMA, "artifacts": recs})
        (derived / "binding.json").write_bytes(bdata)
        doc = {"schema": HI.INVENTORY_SCHEMA, "evaluator_commit": head, "manifest_sha256": "a" * 64,
               "execution_context": {}, "dedup": None, "freeze": {"manifest_sha256": "a" * 64},
               "models": [{"slot": Q.MODELS[m]["slot"], "precision": Q.MODELS[m]["precision"]} for m in Q.MODEL_ORDER],
               "entries": [{"slot": Q.MODELS[m]["slot"], "role": ["family"]} for m in Q.MODEL_ORDER],
               "binding": {"state": "bound", "path": "binding.json", "sha256": hashlib.sha256(bdata).hexdigest()}}
        edata = Q.canonical_bytes(doc)
        (derived / "inventory.json").write_bytes(edata)
        (cache / "verification").mkdir(parents=True)
        cman = {"cells": [{"cell_id": c} for c in Q.CELLS], "entries": [],
                "sources": [{"manifest_index": e.manifest_index, "image_id": e.image_id} for e in entries]}
        (cache / MANIFEST_NAME).write_bytes(Q.canonical_bytes(cman))
        (cache / "verification" / "record.json").write_text("{}", encoding="utf-8")
        cache_err["manifest"] = cman
        (root / "run").mkdir()
        return root, ["--run", str(root / "run"), "--split", "test", "--export", str(derived / "inventory.json"),
                      "--export-sha256", hashlib.sha256(edata).hexdigest(), "--session-root", str(session),
                      "--expect-commit", head, "--corruption-cache", str(cache), "--cache-canonical-sha256", "c" * 64,
                      "--cache-manifest-file-sha256", Q.sha256_file(cache / MANIFEST_NAME),
                      "--verification-record", str(cache / "verification" / "record.json"),
                      "--verification-record-sha256", Q.sha256_file(cache / "verification" / "record.json")]

    def fake_resolve(root, corruption, severity, *, split, official, artifact_status=None, expected_sha256=None):
        calls["resolve"].append((corruption, severity, split, official, artifact_status, expected_sha256))
        if cache_err.get("raise"):
            e = RuntimeError("[cache_canonical_sha256_mismatch] a made-up refusal naming syn0001")
            e.code = "cache_canonical_sha256_mismatch"
            raise e
        return cache_err["manifest"]

    def fake_record(root, manifest, record, *, artifact_status):
        calls["record"].append((Path(record).name, artifact_status))
        return Q.sha256_file(record)

    def fake_load_bound(inv, *, slot, stage, precision, scope, cell=None, input_root, policy):
        calls["load_bound"].append((slot, stage, precision, scope, cell, policy.name))
        rec = HI.binding_record(inv, slot, precision, scope, cell)
        d = Path(input_root) / rec["repo_relative_path"]
        if input_file_sha256(d) != rec["files_sha256"]:
            raise HI.refusal("artifact_record_mismatch", f"{d}: files differ from the binding record")
        return SimpleNamespace(record=rec, loaded=SimpleNamespace(directory=d))

    listing = {"entries": entries}
    try:
        HI.tman_reader = lambda: HI.read_inventory_stub
        HI.load_bound = fake_load_bound
        EC.resolve_corruption_cache = fake_resolve
        EC.check_verification_record = fake_record
        AD.build_expected_manifest_for = lambda split, idx: [listing["entries"][i] for i in sorted(idx)]
        Q.split_listing = lambda split: (list(listing["entries"]), [e.image_id for e in entries], n)
        root, argv = setup("pass")
        args = Q.build_parser().parse_args(argv)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pre = Q.preflight(args, guard, {"commit": head})
        trees = pre["evaluator_trees"].get(head, {})
        check("S10 P1 passes on made-up files: 13 records and 52 files against the binding (LREP's open_inventory), "
              "one repo_commit whose src/ and configs/ trees are HEAD's, the listing's sha256, 3 cells opened "
              "official with their pins; nothing per-image parsed, no id printed",
              len(pre["records"]) == 13 and pre["repo_commits"] == [head] and trees.get("compared") is True
              and trees["src"]["equal"] and trees["configs"]["equal"]
              and pre["listing"]["split_manifest_sha256"] == lsha
              and [(x[0], x[3], x[4], x[5]) for x in calls["resolve"]] == [
                  (c.rpartition("_s")[0], True, "official", "c" * 64) for c in Q.CELLS]
              and calls["record"] == [("record.json", "official")] * 3 and calls["load_bound"] == []
              and "syn0" not in buf.getvalue(), buf.getvalue().strip()[-300:])
        inv = Q.open_test_inventory(args.export, args.export_sha256)
        offs = {}
        for m, c in Q.RECORD_KEYS:
            offs[(m, c)] = Q.load_test_official(inv, m, c, args.session_root)
            Q.check_record_identity(offs[(m, c)], split="test", condition=c, checkpoint_sha256=model_sha[m])
        ids = Q.manifest_order("test", offs[("E6", Q.CLEAN)])
        want_calls = [(Q.MODELS[m]["slot"], Q.MODELS[m]["stage"], *Q.binding_key(m, c)[1:], "OFFICIAL")
                      for m, c in Q.RECORD_KEYS]
        check("S10 open_test_inventory, load_test_official (each model and condition's binding key), "
              "check_record_identity and manifest_order on the made-up export, binding and artifacts (T7 (4))",
              calls["load_bound"] == want_calls and ids == [e.image_id for e in entries]
              and all(o.files_sha256 == pre["records"][Q.record_label(*k)]["files_sha256"] for k, o in offs.items())
              and stops("record_identity", Q.check_record_identity, offs[("E1", "fog_s2")], split="test",
                        condition="motion_blur_s2", checkpoint_sha256=model_sha["E1"])
              and stops("record_identity", Q.check_record_identity, offs[("E1", Q.CLEAN)], split="test",
                        condition=Q.CLEAN, checkpoint_sha256=model_sha["E3"]), str(calls["load_bound"][:2]))
        refused, kept = {}, []

        def refusal_of(name, a):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    Q.preflight(a, guard, {"commit": head})
                refused[name] = None
            except Q.PreflightRefusal as e:
                refused[name] = e.code
                kept.append(e.private)

        for name, kw in (("context", {"threads_of": ("E3", "gaussian_noise_s2")}),
                         ("commit", {"commit_of": ("teacher", Q.CLEAN)}), ("tamper", {"tamper": ("E6", "fog_s2")})):
            refusal_of(name, Q.build_parser().parse_args(setup(name, **kw)[1]))
        listing["entries"] = entries[:5] + [ManifestEntry(5, "syn9999", "syn9999")] + entries[6:]
        refusal_of("listing", args)
        listing["entries"] = entries
        cache_err["raise"] = True
        refusal_of("cache", args)
        cache_err["raise"] = False
        check("S10 P1 refuses a record made with T 8, a second repo_commit, a file changed after the binding, "
              "another listing and a cache refusal (QUAL-AUD-1 A.3, F-01, F-03, F-04); on TEST no refusal keeps a "
              "library message, even for the run's log (A.2)",
              refused == {"context": "record_context", "commit": "repo_commit", "tamper": "record_files",
                          "listing": "manifest_mismatch", "cache": "cache_refused"} and kept == [None] * 5,
              f"{refused} {kept}")
        # select's TEST path end to end (its interpreter and checkout guards stood in)
        Q.version_guard = lambda **kw: dict(guard)
        Q.repo_state = lambda expect: {"commit": head, "governed_paths_clean": True, "governed_violations": []}
        calls["load_bound"].clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = Q.main(argv)
        out = buf.getvalue()
        rec = json.loads((root / "run" / "select" / "record.json").read_text(encoding="utf-8")) if rc == 0 else {}
        sdoc = json.loads((root / "run" / "select" / "S.json").read_text(encoding="utf-8")) if rc == 0 else {}
        check("S10 select on TEST: the preflight first, then E6's record and the other twelve (F-02), all thirteen "
              "files_sha256 kept, S drawn; the console names no image",
              rc == 0 and calls["load_bound"][0][:2] == (Q.MODELS["E6"]["slot"], "E6")
              and len(calls["load_bound"]) == 13
              and out.index("P1 records") < out.index("records: 13 loaded")
              and sorted(rec.get("records", {})) == sorted(Q.record_label(*k) for k in Q.RECORD_KEYS)
              and rec["records"] == {k: v["files_sha256"] for k, v in pre["records"].items()}
              and len(sdoc.get("picks", [])) == 9 and len(sdoc.get("nonce", "")) == 32 and "syn" not in out, out[-400:])
        root_x, argv_x = setup("context_select", threads_of=("E1", Q.CLEAN))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_x = Q.main(argv_x)
        out = buf.getvalue()
        check("S10 a refusal in select's preflight prints REFUSED, exits 3 and writes nothing of S",
              rc_x == 3 and "REFUSED record_context" in out and not (root_x / "run" / "select").exists()
              and "syn" not in out, out[-300:])
    finally:
        for (mod, attr), v in keep.items():
            setattr(mod, attr, v)
        shutil.rmtree(tmp, ignore_errors=True)


# every repo module the three steps import (select, predict on clean and corrupted conditions, render), TEST and
# VAL paths alike; S0 loads them all in this checkout, so a module the clone leaves out stops the smoke, not a run
STEP_MODULES = ("scripts.qual_select", "scripts.qual_predict", "scripts.qual_render",
                "src.eval", "src.eval.artifacts", "src.eval.protocols", "src.eval.adapters", "src.eval.evaluate",
                "src.eval.eval_runtime", "src.eval.model_loading", "src.eval.stage_artifacts", "src.eval.corrupted",
                "src.eval.robustness", "src.data.dataset", "src.data.transforms", "src.distill.nmf_stream",
                "src.corruption_cache", "scripts.evaluate_model", "scripts.evaluate_corruptions",
                "src.stats.heldout_inputs", "src.stats.ingest", "src.stats.driver", "src.tman.validate",
                "src.tman.schema", "src.tman.binding", "configs.data")


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
    ap.add_argument("--no-torch", action="store_true", help="skip S0, S3, S4 and S10 (only where torch is absent)")
    ap.add_argument("--palette", action="store_true", help="also regenerate CLASS_COLOURS (about two minutes)")
    args = ap.parse_args(argv)
    s1_s2()
    if args.no_torch:
        print("SKIP S0 S3 S4 S10: --no-torch (not a PASS of the full smoke)")
    else:
        s0()
        s3()
        s4()
        s10()
    s5(args.font)
    s6(args.palette)
    s7()
    s8(args.font)
    s9(args.font)
    n, ok = len(RESULTS), sum(RESULTS)
    full = not args.no_torch and ok == n
    print(f"RESULT: QUAL SMOKE {'OK' if full else 'PARTIAL' if ok == n else 'FAILED'} ({ok}/{n})")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
