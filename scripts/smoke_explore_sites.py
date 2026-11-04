#!/usr/bin/env python3
"""Smoke for the ten D-1 sites (src/explore/sites.py), N2's core claim, the TF32 switch and B1's passes and
reconciliation (src/explore/b1_split.py, scripts/explore_b1.py). Synthetic students and inputs only; no dataset.

N2's core claim, shown two ways:
  (A) per site, exactly: the converted QNNPACK model's own quantized input to each of the ten operations, dequantized
      and passed through the float operation, then fake-quantized on the producer's grid (the site-matched graph),
      reproduces the converted operation's output in at least 99.9% of elements; without the site fake-quant (the
      registered graph) it does not. The producer's grid is the converted input's grid (scale and zero point equal),
      and the converted output carries its input's grid.
  (B) end to end, on random inputs, on a student whose SE fc1 and context weights are scaled x300 (src/explore/
      synthetic.py says why): stated tolerance, per image: the all-ten-matched graph's mean |logit - converted| is at
      most 0.6 x the registered graph's, its argmax agreement with the converted model is at least the registered
      graph's, and its max |difference| is at most 1 LSB of head.ff_add (+1e-4).
Also: the sites resolve in order with their producers (object identity at run time); hooks add no state and are
removed; training through the sites is finite and the site output lies on the producer's current grid; set_tf32 sets
both backend flags; B1's pass plan, its passes on synthetic batches and its reconciliation (hand-computed); the B1
CLI refuses TEST, a non-x- run ID, and a data root that fails the TRAIN/VAL-only check. Mutations: the site fake-quant
as identity, a wrong producer, and a TF32 switch that sets one flag each fail a case.

    python -I scripts/smoke_explore_sites.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from src.explore import b1_split as B                                # noqa: E402
from src.explore import sites as S                                   # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402
from src.explore.synthetic import calibrated_prepared, converted_of, random_student  # noqa: E402
from src.explore.valscore import score_batches, synthetic_batches  # noqa: E402

_CACHE: dict = {}


def models():
    if "plain" not in _CACHE:
        p = calibrated_prepared(random_student(seed=0), size=128)
        _CACHE["plain"] = (p, converted_of(p))
        pa = calibrated_prepared(random_student(seed=0, amplify=300.0), size=256)
        _CACHE["amp"] = (pa, converted_of(pa))
    return _CACHE["plain"], _CACHE["amp"]


def per_site_exactness(p, c, x) -> dict:
    """(A): the converted model's quantized site inputs and outputs, captured; the float op + site fake-quant vs not."""
    cap, hs = {}, []
    cm = dict(c.named_modules())
    for n in S.SITE_NAMES[:-1]:
        path = S.resolve_sites(p)[n]["module_path"]
        hs.append(cm[path].register_forward_hook(lambda m, i, o, n=n: cap.__setitem__(n, (i[0], o))))
    hs.append(cm["head.high_proj"].register_forward_hook(lambda m, i, o: cap.__setitem__("a", o)))
    hs.append(cm["head.context"].register_forward_hook(lambda m, i, o: cap.__setitem__("s", o)))
    hs.append(cm["head.high_logits"].register_forward_pre_hook(lambda m, i: cap.__setitem__("hl_in", i[0])))
    hs.append(cm["features.6"].register_forward_hook(lambda m, i, o: cap.__setitem__("low", o)))
    with torch.no_grad():
        c(x)
    for h in hs:
        h.remove()
    sites = S.resolve_sites(p)
    out = {}
    for n in S.SITE_NAMES:
        prod = sites[n]["producer"]
        if n == "head_resize":
            hf = c.head.ff_mul.mul(cap["a"], cap["s"])                         # the converted head's own multiply
            q_in, q_out = hf, cap["hl_in"]
            f_in = hf.dequantize()
            ref = F.interpolate(f_in, size=cap["low"].shape[-2:], mode="bilinear", align_corners=False)
        else:
            q_in, q_out = cap[n]
            ref = F.adaptive_avg_pool2d(q_in.dequantize(), 1)
        target = q_out.dequantize()
        matched = S.fake_quant_like(ref, prod)
        step = float(prod.scale)
        out[n] = {"grid_in_equal": (math.isclose(q_in.q_scale(), float(prod.scale), rel_tol=1e-6)
                                    and q_in.q_zero_point() == int(prod.zero_point)),
                  "grid_out_is_input": (q_out.q_scale() == q_in.q_scale()
                                        and q_out.q_zero_point() == q_in.q_zero_point()),
                  "matched_exact": float(((matched - target).abs() <= 1e-6 * max(1.0, step)).float().mean()),
                  "matched_within_1lsb": float(((matched - target).abs() <= step * 1.0001).float().mean()),
                  "registered_exact": float(((ref - target).abs() <= 1e-6 * max(1.0, step)).float().mean()),
                  "n": int(target.numel())}
    return out


def op_level(p, seed: int = 11) -> dict:
    """(A'): each site's operation on a random quantized tensor drawn on that site's own grid (the producer's scale and
    zero point): the quantized operation (the converted model's kernel, QNNPACK engine) against the float operation
    with and without the site fake-quant."""
    from src.quant.qconfig import select_qnnpack_backend
    select_qnnpack_backend()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, s in S.resolve_sites(p).items():
        prod = s["producer"]
        scale, zp = float(prod.scale), int(prod.zero_point)
        shape = (1, 16, 32, 32) if n != "head_resize" else (1, 16, 16, 16)
        ints = torch.randint(int(prod.quant_min), int(prod.quant_max) + 1, shape, generator=g)
        xq = torch._make_per_tensor_quantized_tensor(ints.to(torch.uint8), scale, zp)
        if n == "head_resize":
            yq = F.interpolate(xq, size=(32, 32), mode="bilinear", align_corners=False)
            ref = F.interpolate(xq.dequantize(), size=(32, 32), mode="bilinear", align_corners=False)
        else:
            yq = F.adaptive_avg_pool2d(xq, 1)
            ref = F.adaptive_avg_pool2d(xq.dequantize(), 1)
        target = yq.dequantize()
        matched = S.fake_quant_like(ref, prod)
        tol = 1e-6 * max(1.0, scale)
        out[n] = {"matched_exact": float(((matched - target).abs() <= tol).float().mean()),
                  "matched_within_1lsb": float(((matched - target).abs() <= scale * 1.0001).float().mean()),
                  "registered_exact": float(((ref - target).abs() <= tol).float().mean()),
                  "output_grid_is_input": yq.q_scale() == scale and yq.q_zero_point() == zp}
    return out


def end_to_end(p, c, n_img=3, size=256, seed=7):
    lsb = float(p.head.ff_add.activation_post_process.scale)
    g = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        for _ in range(n_img):
            x = torch.randn(1, 3, size, size, generator=g)
            yc, yr = c(x), p(x)
            with S.SiteMatcher(p):
                ym = p(x)
            dr, dm = (yr - yc).abs() / lsb, (ym - yc).abs() / lsb
            rows.append({"reg_mean": float(dr.mean()), "mat_mean": float(dm.mean()), "reg_max": float(dr.max()),
                         "mat_max": float(dm.max()),
                         "reg_agree": float((yr.argmax(1) == yc.argmax(1)).float().mean()),
                         "mat_agree": float((ym.argmax(1) == yc.argmax(1)).float().mean())})
    return rows


def run_cases(verbose: bool = False) -> Cases:
    c = Cases("SMOKE_EXPLORE_SITES", verbose=verbose)
    (p, conv), (pa, conva) = models()
    sites = S.resolve_sites(p)
    c.check("ten sites resolve in order, SE pools first", tuple(sites) == S.SITE_NAMES and S.SITE_NAMES[:8] == S.SE_SITES)
    want_prod = {**{f"se{i}": f"features.{i}.block.1.0.activation_post_process" for i in (4, 5, 6)},
                 **{f"se{i}": f"features.{i}.block.1.2.activation_post_process" for i in (11, 12, 13, 14, 15)},
                 "head_pool": "features.15.skip_add.activation_post_process",
                 "head_resize": "head.ff_mul.activation_post_process"}
    c.check("producers: ReLU blocks block.1.0, Hardswish blocks block.1.2, head pool skip_add, resize ff_mul",
            {n: s["producer_path"] for n, s in sites.items()} == want_prod)
    # run-time identity of the producer's output and the site's input (9 pool sites)
    seen, hs = {}, []
    mods = dict(p.named_modules())
    for n, s in sites.items():
        if s["kind"] == "head_resize":
            continue
        hs.append(s["producer"].register_forward_hook(lambda m, i, o, n=n: seen.__setitem__(("p", n), o)))
        hs.append(mods[s["module_path"]].register_forward_pre_hook(lambda m, i, n=n: seen.__setitem__(("s", n), i[0])))
    with torch.no_grad():
        p(torch.randn(1, 3, 128, 128))
    for h in hs:
        h.remove()
    c.check("each pool site's input IS its producer's output (object identity, one forward)",
            all(seen[("s", n)] is seen[("p", n)] for n in sites if sites[n]["kind"] != "head_resize"))
    # (A)
    ex = per_site_exactness(p, conv, torch.randn(1, 3, 128, 128, generator=torch.Generator().manual_seed(3)))
    c.check("(A) the producer's grid is the converted input's grid, at all ten sites",
            all(v["grid_in_equal"] for v in ex.values()), str({k: v["grid_in_equal"] for k, v in ex.items()}))
    c.check("(A) the converted output carries its input's grid, at all ten sites",
            all(v["grid_out_is_input"] for v in ex.values()))
    c.check("(A) site-matched reproduces the converted op: >= 99.9% exact, 100% within 1 LSB, all ten",
            all(v["matched_exact"] >= 0.999 and v["matched_within_1lsb"] == 1.0 for v in ex.values()),
            str({k: round(v["matched_exact"], 4) for k, v in ex.items()}))
    c.check("(A) in the random student, where the registered graph is off the converted op, the matched one is on",
            all(v["matched_exact"] >= v["registered_exact"] for v in ex.values()),
            "registered exact fractions " + str({k: round(v["registered_exact"], 4) for k, v in ex.items()}))
    op = op_level(p)
    c.check("(A') op level, random quantized inputs on each site's own grid: matched >= 99.9% exact at the nine "
            "pools, >= 98% at the resize, 100% within 1 LSB at all ten",
            all(v["matched_exact"] >= (0.98 if k == "head_resize" else 0.999) and v["matched_within_1lsb"] == 1.0
                for k, v in op.items()),
            str({k: (round(v["matched_exact"], 4), round(v["matched_within_1lsb"], 4)) for k, v in op.items()}))
    c.check("(A') op level: the quantized operations emit on their input's grid",
            all(v["output_grid_is_input"] for v in op.values()))
    c.check("(A') op level: the registered graph is exact in under half the elements at every site",
            all(v["registered_exact"] < 0.5 for v in op.values()),
            str({k: round(v["registered_exact"], 4) for k, v in op.items()}))
    # (B)
    rows = end_to_end(pa, conva)
    c.check("(B) all-ten-matched mean |diff| <= 0.6 x registered, every image",
            all(r["mat_mean"] <= 0.6 * r["reg_mean"] for r in rows),
            str([(round(r["mat_mean"], 4), round(r["reg_mean"], 4)) for r in rows]))
    c.check("(B) matched argmax agreement >= registered, every image",
            all(r["mat_agree"] >= r["reg_agree"] for r in rows),
            str([(round(r["mat_agree"], 4), round(r["reg_agree"], 4)) for r in rows]))
    c.check("(B) matched max |diff| <= 1 LSB (+1e-4), every image", all(r["mat_max"] <= 1.0001 for r in rows))
    # hooks: no state, removable
    keys = list(p.state_dict())
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y0 = p(x)
        m = S.SiteMatcher(p)
        k2 = list(p.state_dict())
        p(x)
        m.remove()
        y1 = p(x)
    c.check("the site hooks add nothing to the state_dict", keys == k2)
    c.check("remove() restores the registered graph bit for bit", torch.equal(y0, y1))
    c.check("every site hook fired once per forward", all(v == 1 for v in m.calls.values()), str(m.calls))
    # training through the sites
    pt = calibrated_prepared(random_student(seed=2), steps=1, size=96)
    from src.quant.prepare import enable_observers
    enable_observers(pt)
    pt.train()
    cap = {}
    sm = S.SiteMatcher(pt)
    h = dict(pt.named_modules())["features.4.block.2.avgpool"].register_forward_hook(
        lambda mm, i, o: cap.__setitem__("o", o))
    xin = torch.randn(2, 3, 96, 96)
    y = pt(xin)
    loss = F.cross_entropy(y, torch.randint(0, 116, (2, 96, 96)))
    loss.backward()
    h.remove()
    prod = S.resolve_sites(pt)["se4"]["producer"]
    q = cap["o"].detach() / float(prod.scale) + float(prod.zero_point)
    sm.remove()
    c.check("training through the sites: finite loss and gradients",
            bool(torch.isfinite(loss)) and all(torch.isfinite(pp.grad).all() for pp in pt.parameters()
                                               if pp.grad is not None))
    c.check("a site's output lies on its producer's current grid (train mode, observers on)",
            float((q - q.round()).abs().max()) < 1e-3)
    # TF32
    before = S.tf32_flags()
    off = S.set_tf32(False)
    on = S.set_tf32(True)
    torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
    c.check("set_tf32(False) sets both backend flags off", off["cudnn_allow_tf32"] is False
            and off["cuda_matmul_allow_tf32"] is False)
    c.check("set_tf32(True) sets both on", on["cudnn_allow_tf32"] is True and on["cuda_matmul_allow_tf32"] is True)
    # B1: plan, passes on synthetic batches, reconciliation
    c.check("B1 CPU plan: 13 passes, no TF32 pass", B.pass_plan("cpu") == [q for q in B.PASS_ORDER
                                                                           if q not in B.CUDA_ONLY])
    c.check("B1 CUDA plan: all 15 passes", B.pass_plan("cuda") == list(B.PASS_ORDER))
    batches, manifest = synthetic_batches(3, size=(64, 64))
    scores = B.run_passes(p, conv, lambda mm, d: score_batches(mm, batches, manifest)[0], device="cpu",
                          log=lambda *a: None)
    c.check("B1 passes on synthetic batches: 13 summaries with a value each",
            list(scores) == B.pass_plan("cpu") and all(isinstance(s["all_class_miou"], float) for s in scores.values()))
    c.check("B1 site passes record their hook calls", scores["site:se4"]["site_calls"] == {"se4": 3}
            and scores["all10"]["site_calls"] == {n: 3 for n in S.SITE_NAMES})
    syn = {"baseline": 0.3600, "site:se4": 0.3580, "site:se5": 0.3590, **{f"site:{n}": 0.3600 for n in S.SITE_NAMES
                                                                          if n not in ("se4", "se5")},
           "all10": 0.3560, "tf32off": 0.3598, "all10_tf32off": 0.3555, "converted": 0.3550}
    syn = {k: {"all_class_miou": v} for k, v in syn.items()}
    rc = B.reconcile(syn, converted_of_record=0.3550, device="cuda")
    close = lambda a, b: a is not None and abs(a - b) < 1e-9           # noqa: E731
    c.check("reconcile: gap 0.5 pp, se4 0.2, se5 0.1 (hand)", close(rc["gap_baseline_minus_converted_pp"], 0.5)
            and close(rc["site_shares_pp"]["se4"], 0.2) and close(rc["site_shares_pp"]["se5"], 0.1))
    c.check("reconcile: sum of shares 0.3, all ten 0.4, interaction +0.1 (hand)",
            close(rc["sum_of_ten_shares_pp"], 0.3) and close(rc["all10_share_pp"], 0.4)
            and close(rc["interaction_pp"], 0.1))
    c.check("reconcile: TF32 share 0.02, residual +0.05, explained 0.9 (hand)", close(rc["tf32_share_pp"], 0.02)
            and close(rc["residual_full_corrected_minus_converted_pp"], 0.05) and close(rc["explained_fraction"], 0.9))
    rc_cpu = B.reconcile(syn, device="cpu")
    c.check("reconcile on a CPU: no TF32 share, residual from all10, said so", rc_cpu["tf32_share_pp"] is None
            and close(rc_cpu["residual_full_corrected_minus_converted_pp"], 0.1) and rc_cpu["tf32_note"])
    txt = B.table_text(syn, rc)
    c.check("B1 table lists the SE-pool sites first", txt.index("site:se4") < txt.index("site:head_pool")
            and txt.index("SE-pool shares") < txt.index("sum of ten"))
    # B1 CLI refusals (this session has no data root: the TRAIN/VAL-only check must refuse)
    base = scratch_dir("x_b1_")
    inp = base / "inp"
    inp.mkdir()
    py, cli = sys.executable, str(REPO / "scripts" / "explore_b1.py")
    common = ["--qat-run-dir", str(inp / "run"), "--selection", str(inp / "sel.json"), "--selection-sha256", "0" * 64,
              "--artifact", str(inp / "a.pt"), "--stage", "E6-E2"]
    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    r = subprocess.run([py, "-I", cli, *common, "--split", "TE" + "ST", "--run-id", "x-b1-s",
                        "--out-dir", str(base / "o" / "x-b1-s")], capture_output=True, text=True, env=env)
    c.check("B1 CLI refuses --split TEST (exit 2)", r.returncode == 2 and "split_test" in r.stdout)
    r = subprocess.run([py, "-I", cli, *common, "--run-id", "b1-plain", "--out-dir", str(base / "o" / "b1-plain")],
                       capture_output=True, text=True, env=env)
    c.check("B1 CLI refuses a run ID without x- (exit 2)", r.returncode == 2 and "run_id_prefix" in r.stdout)
    r = subprocess.run([py, "-I", cli, *common, "--run-id", "x-b1-root", "--out-dir", str(base / "o" / "x-b1-root")],
                       capture_output=True, text=True, env=env)
    c.check("B1 CLI refuses a data root that fails the TRAIN/VAL-only check (exit 2)",
            r.returncode == 2 and "data_root_not_trainval_only" in r.stdout and not (base / "o" / "x-b1-root").exists(),
            r.stdout.strip()[-200:])
    return c


def main() -> int:
    print("[smoke_explore_sites] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_sites] mutations")
    for name, mut in {
        "site fake-quant as identity": lambda: patched(S, "fake_quant_like", lambda x, producer: x),
        "every site reads head.ff_add's grid": lambda: patched(
            S, "resolve_sites", _wrong_producer(S.resolve_sites)),
        "TF32 switch sets cuDNN only": lambda: patched(S, "set_tf32", _one_flag),
    }.items():
        killed, failed = mutation_killed(lambda: run_cases(False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
    return c.result()


def _wrong_producer(orig):
    def resolve(prepared):
        s = orig(prepared)
        wrong = dict(prepared.named_modules())["head.ff_add.activation_post_process"]
        return {k: {**v, "producer": wrong, "producer_path": "head.ff_add.activation_post_process"} for k, v in s.items()}
    return resolve


def _one_flag(enabled: bool) -> dict:
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    return S.tf32_flags()


if __name__ == "__main__":
    raise SystemExit(main())
