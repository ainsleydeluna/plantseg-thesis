#!/usr/bin/env python3
"""Smoke: the mutation harness of lane L-TEACHER-DIAG (acceptance (h); P33; Q8).

Each mutation rewrites one statistic's or check's implementation from its own source (an exact,
unique text replacement, compiled into the module's namespace and restored afterwards) and runs the
named case that must catch it. A case is first run unmutated and must pass; under the mutation it must
fail (or raise). P33's list is included: a float F rule on the two big-integer pairs, a float32 softmax
in the ECE, side="right", pbar as the softmax of mean logits, a T^2 factor, an any-valid mask and a
|V_i|-weighted KL_cwd. The rest are the plan's: one per statistic and per gate. Every case mirrors a
case of the lane's smokes (named in the table); the fixtures are the same synthetic ones.

    python -B scripts/smoke_teacher_diag_mutations.py
"""
from __future__ import annotations

import __future__ as _future
import builtins
import contextlib
import inspect
import io
import json
import math
import os
import shutil
import sys
import textwrap
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
UTC = "2026-10-02T00:00:00Z"
HEAD = "c" * 40


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


@contextlib.contextmanager
def mutate(owner, name: str, *pairs):
    """Replace `owner.name` (a module function or a class method) by its own source with each (old, new)
    replaced exactly once, compiled into its module's namespace; restore it afterwards."""
    original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    module = sys.modules[original.__module__]
    src = textwrap.dedent(inspect.getsource(original))
    for old, new in pairs:
        if src.count(old) != 1:
            raise AssertionError(f"mutation anchor occurs {src.count(old)} times in {name}: {old!r}")
        src = src.replace(old, new)
    ns = module.__dict__
    had, prev = name in ns, ns.get(name)
    exec(compile(src, f"<mutant {module.__name__}.{name}>", "exec", flags=_future.annotations.compiler_flag,
                 dont_inherit=True), ns)
    mutant = ns[name]
    if isinstance(owner, type):
        if had:
            ns[name] = prev
        else:
            del ns[name]
    setattr(owner, name, mutant)
    try:
        yield
    finally:
        setattr(owner, name, original)


# ===================================================================================================
# fixtures
# ===================================================================================================
class Env:
    pass


def build_env() -> Env:
    from scripts import dedup_val_scores as d4
    from scripts import hash_split_files as hs
    from scripts import smoke_teacher_d1 as s1
    from scripts import smoke_teacher_d3 as s3
    from scripts import smoke_teacher_d4 as s4
    from scripts import teacher_d1_nmf_sensitivity as d1
    from scripts import teacher_d2_calibration as d2
    from scripts import teacher_d3_perclass_strata as d3
    from scripts import teacher_diag_fixtures as fx
    from src.stats.val_artifacts import load_val_artifact
    e = Env()
    from scripts import score_teacher_train as st
    e.s1, e.s3, e.s4, e.fx, e.d1, e.d2, e.d3, e.d4, e.hs, e.st = s1, s3, s4, fx, d1, d2, d3, d4, hs, st
    e.fx_, e.td, e.tmp, e.ckpt, e.sha, e.args = s1._seam_env()
    e.n = 0
    e.root, e.stems = fx.make_data_root("diag_mut_data_")
    fx.set_data_root(e.root)
    e.strata = fx.write_strata(e.tmp / "train_strata_v1.json", e.stems["train"])
    e.planted = s4.plant_copies(e.root, e.stems)
    e.ref = fx.reference_artifact(e.tmp / "ref_artifact", e.ckpt, e.sha, n=3)
    e.cfg = str(e.td.REPO / e.td.TEACHER_CONFIG_REL)
    e.teacher = dict(teacher_ckpt=str(e.ckpt), teacher_ckpt_sha256=e.sha, teacher_config=e.cfg)
    fx.call_run(d2, out_dir=e.tmp / "d2_item1", artifact_dir=e.tmp / "d2_item1_art", teacher_role="record",
                purpose="item1", val_reference=str(e.ref), max_samples=3, generated_utc=UTC, **e.teacher)
    e.d2_json = fx.output_files(e.tmp / "d2_item1")[0]
    cfg_copy = Path(shutil.copy(e.cfg, e.tmp / "arm_cfg.py"))
    e.arm = dict(teacher_role="arm", arm_id="R1", arm_dl_id="DL-70", teacher_config=str(cfg_copy),
                 teacher_config_sha256=e.td.file_sha256(cfg_copy))
    e.d3f = s3.fixtures(_mk(e.tmp / "d3"))
    e.hs_flags = dict(data_root=str(e.root), strata=str(e.strata), script_commit=HEAD, script_commit_dl_id="DL-66")
    code, _ = hs_run(e, out=e.tmp / "hashes")
    if code != 0:
        raise RuntimeError("fixture: the hasher did not run")
    e.dup_json = e.tmp / "hashes" / hs.JSON_NAME
    e.dup_sha = e.td.file_sha256(e.dup_json)
    ids = list(e.stems["val"])
    e.ids = ids
    pin = fx.clean_git_pin(e.tmp / "pin_repo")
    e.gts, pt, pe = s4.label_maps()
    e.pt, e.pe = pt, pe
    rt, man = fx.core_result(ids, e.gts, pt)
    re_, _ = fx.core_result(ids, e.gts, pe)
    w = lambda name, res, stage, role, sha, dig: fx.write_rescore(e.tmp / name, res, man, stage=stage, role=role,  # noqa: E731
                                                               sha=sha, digest=dig, pin=pin)
    e.a_t = w("teacher_cpu", rt, "teacher", "teacher", fx.SYN_SHA_T, fx.SYN_DIG_T)
    e.a_e = w("e1_cpu", re_, "E1", "student", fx.SYN_SHA_E, fx.SYN_DIG_E)
    e.a_e0 = w("e1_same", rt, "E1", "student", fx.SYN_SHA_E, fx.SYN_DIG_E)
    e.a_t2 = w("teacher_again", rt, "teacher", "teacher", fx.SYN_SHA_T, fx.SYN_DIG_T)
    e.v_t = json.loads((e.a_t / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    e.v_e = json.loads((e.a_e / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    e.a_te = e.tmp / "teacher_edited"
    shutil.copytree(e.a_t, e.a_te)
    sj = json.loads((e.a_te / "summary.json").read_text())
    sj["dataset_level"]["all_class_miou"] = float(np.nextafter(np.float32(e.v_t), np.float32(1.0)))
    (e.a_te / "summary.json").write_text(json.dumps(sj, indent=2) + "\n")
    fx.rehash_artifact(e.a_te)
    with fx.synthetic_pairing(e.v_t, e.v_e):
        art_t, art_e = load_val_artifact(e.a_t, label="t"), load_val_artifact(e.a_e, label="e")
        e.g1 = fx.gap_output(e.tmp / "gap_1.json", art_t, art_e)
        e.ge = fx.gap_output(e.tmp / "gap_edited.json", load_val_artifact(e.a_te, label="te"), art_e)
        e.g0 = fx.gap_output(e.tmp / "gap_0.json", art_t, load_val_artifact(e.a_e0, label="e0"))
    g = json.loads(e.g1.read_text())
    g["rules"]["union_present"]["point"] += 1e-9
    e.goff = e.tmp / "gap_off.json"
    e.goff.write_text(json.dumps(g))
    keep = torch.ones(846, dtype=torch.bool)
    keep[[ids.index(i) for i in json.loads(e.dup_json.read_text())["duplicate_val_ids"]]] = False
    e.sub_m = s4.miou64(e.gts, pt, keep) - s4.miou64(e.gts, pe, keep)
    e.keep = keep
    with fx.synthetic_pairing(e.v_t, e.v_e):
        e.g_role = fx.gap_output(e.tmp / "gap_role.json", load_val_artifact(e.a_t, label="t"),
                                 load_val_artifact(e.a_t2, label="t2"))
    gts2 = e.gts.clone()
    gts2[7, 1, 1] = 0
    re2, man2 = fx.core_result(ids, gts2, pe)
    e.a_eg = fx.write_rescore(e.tmp / "e1_other_gt", re2, man2, stage="E1", role="student", sha=fx.SYN_SHA_E,
                              digest=fx.SYN_DIG_E, pin=pin)
    with fx.synthetic_pairing(e.v_t, e.v_e):
        e.g_gt = fx.gap_output(e.tmp / "gap_gt.json", load_val_artifact(e.a_t, label="t"),
                               load_val_artifact(e.a_eg, label="eg"))
    e.link = e.tmp / "plain_root_link"
    os.symlink(e.tmp / ("x" + "TeSt" + "x"), e.link)
    return e


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def fresh(e, tag: str) -> Path:
    e.n += 1
    return e.tmp / f"{tag}_{e.n:03d}"


def hs_run(e, **flags):
    with e.fx.fake_git(head=HEAD):
        return e.fx.call_run(e.hs, factory=None, **{**e.hs_flags, **flags})


def d4_run(e, *, zero=False, **flags):
    base = dict(teacher=e.a_t, e1=e.a_e, gap_output=e.g1, duplicates=e.dup_json, duplicates_sha256=e.dup_sha,
                script_commit=HEAD, script_commit_dl_id="DL-67")
    base.update(flags)
    out = fresh(e, "d4")
    v_e = json.loads((Path(base["e1"]) / "summary.json").read_text())["dataset_level"]["all_class_miou"]
    with e.fx.synthetic_pairing(e.v_t, v_e), e.fx.fake_git(head=HEAD):
        code, err = e.fx.call_run(e.d4, factory=None, out_dir=out, **base)
    files = e.fx.output_files(out)
    e.last_err = err
    return code, (json.loads(files[0].read_text()) if files else {}), files


# ===================================================================================================
# killing cases (each mirrors a case of the lane's smokes)
# ===================================================================================================
def killers(e) -> dict:
    from src.distill.nmf_stream import NMFStream
    from src.eval import calibration as cal
    from src.eval import nmf_sensitivity as ns
    s1, fx, td = e.s1, e.fx, e.td
    k = {}

    # ---- D1 statistics (smoke_teacher_d1 stats_cases)
    k["b1 flips: 0000 1111 -> 16 pairs; eight labels -> 28"] = lambda: (
        ns.pairwise_flips(np.array([[0], [0], [0], [0], [1], [1], [1], [1]])) == 16
        and ns.pairwise_flips(np.arange(8).reshape(8, 1)) == 28)

    def grid():
        z, v = s1.grid_with_deviants(25, 3)
        cs = ns.crop_statistics(z, v)
        return (cs.N, cs.D) == (21, 700)
    k["b1 N = 21, D = 700 on a 25-cell grid"] = grid
    k["P33 big-integer pairs: integers decide, a float rule would not"] = lambda: (
        ns.branch(210000000000000005, 7000000000000000168)["below_0_03"] is True
        and ns.branch(1400000000000000011, 14000000000000000112)["at_least_0_10"] is False)

    def lesion():
        labels = np.zeros((8, 2, 2), dtype=int)
        labels[:, 0, 0] = 1
        labels[:, 0, 1] = 2
        labels[0, 0, 1] = 1
        labels[0, 1, 0] = 3
        cs = ns.crop_statistics(*s1.crafted(labels, np.ones((2, 2), bool)))
        return (cs.N_L, cs.D_L, cs.n_lesion, cs.N) == (7, 56, 2, 14)
    k["b2 F_lesion counts only cells peaking at a disease class"] = lesion

    def halves():
        labels = np.ones((8, 2, 2), dtype=int)
        labels[4:, 0, 0] = 2
        labels[7, 1, 1] = 3
        return ns.crop_statistics(*s1.crafted(labels, np.ones((2, 2), bool))).H == 1
    k["b3 F_halves = 1 flip over 4 cells"] = halves

    def kl_case():
        rng = np.random.default_rng(3)
        z = rng.normal(0, 2.0, size=(8, 6, 4, 5)).astype(np.float32)
        v = rng.random((4, 5)) > 0.3
        z2 = rng.normal(0, 3.0, size=(8, 6, 4, 5)).astype(np.float32)
        v2 = np.zeros((4, 5), dtype=bool)
        v2[0, :3] = True
        return z, v, z2, v2

    def kl_logit():
        z, v, _, _ = kl_case()
        rl = s1.ref_kl_logit(z, v)
        return abs(ns.crop_statistics(z, v).kl_logit_sum - rl) <= 1e-12 * max(1.0, abs(rl))
    k["b4 KL_logit sum equals the numpy reference to 1e-12"] = kl_logit

    def pooled():
        z, v, z2, v2 = kl_case()
        p = ns.summarize([ns.crop_statistics(z, v), ns.crop_statistics(z2, v2)])
        want_cwd = (s1.ref_kl_cwd(z, v) + s1.ref_kl_cwd(z2, v2)) / (8 * 6 * 2)
        want_logit = (s1.ref_kl_logit(z, v) + s1.ref_kl_logit(z2, v2)) / (8 * (int(v.sum()) + 3))
        return (abs(p["KL_cwd"]["value"] - want_cwd) <= 1e-12 * max(1.0, want_cwd)
                and abs(p["KL_logit"]["value"] - want_logit) <= 1e-12 * max(1.0, want_logit))
    k["b5 KL_cwd and KL_logit pooled over two crops of different |V|"] = pooled

    def p32():
        n = 256
        vv = [0 if i % 29 == 0 else 1 + (37 * i) % 101 for i in range(n)]
        d_i = [28 * x for x in vv]
        n_i = [(d_i[i] * ((11 * i) % 17)) // 100 for i in range(n)]
        return n_i, d_i
    k["b6 P32 bootstrap CI exactly [0.07283787157730934, 0.08666608984644292]"] = lambda: (
        (lambda bs: (bs["low"], bs["high"]) == s1.P32_CI)(ns.bootstrap_f(*p32())))
    k["b6 P15 index sha256 73f7b7cb... (numpy 1.26.4)"] = lambda: (
        ns.bootstrap_f(*p32())["index_sha256"] == s1.P15_INDEX_SHA)
    k["b6 a replicate with sum D == 0 makes the CI null"] = lambda: (
        (lambda z: z["low"] is None and z["replicates_with_zero_D"] > 0)(ns.bootstrap_f([0] * 255 + [3], [0] * 255 + [28])))

    # ---- ECE (smoke_teacher_calibration unit_cases, hook_cases)
    from scripts import smoke_teacher_calibration as sc
    a9, a3 = math.log(27.0), math.log(9.0 / 7.0)
    k["d2 two-bin hand case: ECE = 0.55"] = lambda: abs(
        sc.acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [a3, 0, 0, 0], [a3, 0, 0, 0]], [0, 1, 0, 0], 4)["ece"] - 0.55) <= 1e-12
    k["d3 conf == m/15 lands in bin m (m = 1..14)"] = lambda: all(
        int(cal.bin_index(np.array([m / 15.0]))[0]) == m for m in range(1, 15))

    def ignore255():
        rows = [[a9, 0, 0, 0], [a3, 0, 1.0, 0], [0.2, 0.1, 0, 0]]
        more = sc.acc_for(rows + [[50.0, -3, 2, 0], [0, 0, 0, 9.0]], [0, 2, 1, 255, 255], 4)
        return more["n"] == 3
    k["d4 pixels labelled 255 change nothing"] = ignore255
    k["d6 disease-pixel ECE over target in 1..C-1 only"] = lambda: (
        lambda r: r["n_disease"] == 2 and abs(r["ece_disease"] - 0.2) <= 1e-12)(
        sc.acc_for([[a9, 0, 0, 0], [a9, 0, 0, 0], [0, 0, a3, 0], [0, 0, a3, 0]], [0, 0, 2, 1], 4))

    def f32_vector():
        z32 = torch.tensor([[1.3862942457199097, 0.0], [1.3862942457199097, 0.0],
                            [1.1526795625686646, 0.0], [1.1526795625686646, 0.0]], dtype=torch.float32)
        acc = cal.CalibrationAccumulator(num_classes=2, disease_classes=range(1, 2))
        acc.update(z32.T[None, :, None, :].contiguous(), torch.tensor([[[0, 0, 1, 1]]]))
        return acc.result()["ece"] == 0.2799999955678827
    k["d7 P32(d) float32 logits: ECE == 0.2799999955678827"] = f32_vector

    def hook_pass():
        import gc

        from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model
        rng = np.random.default_rng(11)
        ids = [f"syn_{i}" for i in range(3)]
        t = torch.tensor(rng.integers(0, 116, size=(3, 6, 8)))
        z = torch.tensor(rng.normal(0, 2, size=(3, 116, 6, 8)).astype(np.float32))
        hook = cal.CalibratingBatchForward(lambda m, x: x.clone(), cal.CalibrationAccumulator())
        batches = [EvalBatch(images=z[i:i + 1], targets=t[i:i + 1], image_ids=[ids[i]], clean_image_ids=[ids[i]],
                             manifest_indices=[i]) for i in range(3)]
        evaluate_model(None, batches, expected_manifest=[ManifestEntry(i, ids[i], ids[i]) for i in range(3)],
                       condition=Condition(), num_classes=116, background_index=0, ignore_index=255, batch_forward=hook)
        gc.collect()
        return hook, batches
    k["d5 the hook keeps no tensor (weakref dead after the core's del)"] = lambda: (
        (lambda hb: hb[0].last_logits_ref is not None and hb[0].last_logits_ref() is None)(hook_pass()))

    class Mutating(cal.CalibrationAccumulator):
        def update(self, logits, target):
            logits.add_(0.0)
            cal.CalibrationAccumulator.update(self, logits, target)
    k["d5 an in-place change of the core's tensor stops"] = lambda: raises(
        lambda: cal.CalibratingBatchForward(lambda m, x: x.clone(), Mutating())(None, hook_pass()[1][0]), cal.CalibrationStop)

    # ---- the seam (smoke_teacher_d1 seam_cases)
    x = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(9))

    def loaded(factory=None):
        return s1.gated_load(td, e.args(), factory=factory or fx.stub_factory)[1]

    def split_of(ld):
        sp = td.SplitTeacher(ld)
        f = sp.features(x)
        return sp, f, td.feature_sha256(f)
    def wrong_stream():
        ld = loaded()
        sp, f, h = split_of(ld)
        try:
            sp.head(f, NMFStream(42, "M4-KD"), feat_hash=h)
        except td.Stop as ex:
            return "not the expected stream object" in str(ex)
        return False
    k["P9 a head call with a stream that is not the attached object stops"] = wrong_stream
    k["C1 handing the head check the begin call's dict stops"] = lambda: (
        lambda ld: (lambda sp, f, h: raises(lambda: sp.head(f, ld.stream_description, feat_hash=h), td.Stop))(*split_of(ld)))(loaded())

    def stops(factory, why):
        def run_case():
            ld = loaded(factory)
            sp, f, h = split_of(ld)
            try:
                sp.head(f, ld.stream, feat_hash=h)
            except td.Stop as ex:
                return why in str(ex)
            return False
        return run_case
    k["P9 a head drawing twice per call stops"] = stops(fx.double_draw_factory, "by 2 draws")
    k["P9 an in-place-mutating head stops"] = stops(fx.mutating_factory, "features changed")
    k["P9 a NaN head stops"] = stops(fx.nan_factory, "non-finite")
    k["P9 a head drawing from the global CPU RNG stops"] = stops(fx.global_draw_factory, "RNG")
    k["P5 the format check alone refuses empty, uppercase and 63-character sha256"] = lambda: all(
        raises(lambda b=b: td.check_teacher_flags(e.args(teacher_ckpt_sha256=b)), td.Refused)
        for b in ("", e.sha.upper(), e.sha[:63]))

    def refused_before_load(fn):
        with fx.count_loads() as calls:
            r = raises(fn, td.Refused)
        return r and sum(calls.values()) == 0
    k["P5 a well-formed wrong sha256 is refused with the load counter at 0"] = lambda: refused_before_load(
        lambda: s1.gated_load(td, e.args(teacher_ckpt_sha256="0" * 64), factory=fx.stub_factory))
    big, big_sha = fx.write_stub_ckpt(e.tmp / "big_stub.pth", pad_bytes=td.STUB_MAX_CKPT_BYTES)
    k["P2 stub refuses a checkpoint over 16 MiB before any load"] = lambda: refused_before_load(
        lambda: s1.gated_load(td, e.args(teacher_ckpt=str(big), teacher_ckpt_sha256=big_sha), factory=fx.stub_factory))

    def registered():
        with fx.patched(td, REGISTERED_TEACHER_SHA256=frozenset({e.sha})):
            return refused_before_load(lambda: s1.gated_load(td, e.args(), factory=fx.stub_factory))
    k["P2 stub refuses a registered teacher sha256 before any load"] = registered

    def mmseg_build():
        pre = "mmseg" in sys.modules
        try:
            return not pre and raises(lambda: s1.gated_load(td, e.args(), factory=fx.mmseg_factory), td.Refused)
        finally:
            if not pre:
                sys.modules.pop("mmseg", None)
    k["P2 stub refuses a build that leaves mmseg imported"] = mmseg_build
    k["P2 stub refuses a model over 1e6 parameters (own big_factory checkpoint; the message names the count)"] = (
        lambda: s1.case_big_model(td, fx, e.tmp, e.args)[0])
    k["h1 the stub checkpoint holds exactly the factory's state_dict; the loaded stub equals the factory's state"] = (
        lambda: s1.case_stub_ckpt_exact(td, fx, e.tmp, s1.gated_load(td, e.args(), factory=fx.stub_factory)[1]))
    k["h3 the reference builder passes --teacher-ckpt-sha256 (built and recorded; a wrong sha refused)"] = (
        lambda: s1.case_reference_sha(fx, e.tmp, e.ckpt, e.sha)[0])
    k["a both load lines pass the verified sha (expected_sha256 recorded in the kd and evaluator form)"] = (
        lambda: s1.case_expected_sha(s1.gated_load(td, e.args(), factory=fx.stub_factory)[1],
                                     s1.gated_load(td, e.args(), mode="evaluator", factory=fx.stub_factory)[1], e.sha))
    k["h4 a checkpoint changed after validation is refused with exit 2 (TeacherChecksumMismatch)"] = (
        lambda: s1.case_h4_checksum(td, fx, e.tmp, e.args)[0])
    k["h4 a checkpoint the strict load refuses is refused with exit 2 (TeacherStateDictMismatch)"] = (
        lambda: s1.case_h4_state_dict(td, fx, e.tmp, e.args)[0])
    k["P8 the field count on stand-ins: 11 fields refused, 12 not (holds before and after K-part)"] = lambda: s1.case_p8_count(td)

    def binding(head=HEAD, tree_status=None):
        with fx.fake_git(head=head, tree_status=tree_status):
            return raises(lambda: td.require_commit_binding(HEAD, "DL-61"), td.Refused)
    k["P26 a HEAD other than --script-commit is refused"] = lambda: binding(head="d" * 40)
    k["P26 a dirty src/configs/scripts tree is refused"] = lambda: binding(tree_status="?? scripts/a.py\n")

    def single_output():
        out = _mk(fresh(e, "single"))
        td.write_json_exclusive(out / "teacher_d2_20261002T000100Z.json", {"artifact_status": "provisional"})
        a = types.SimpleNamespace(repeat_of=None, repeat_case=None, repeat_dl_id=None)
        return raises(lambda: td.require_single_output(out, "teacher_d2", a), td.Refused)
    k["P27 a second non-smoke output is refused"] = single_output
    k["C2 real mode refuses --generated-utc"] = lambda: raises(lambda: td.check_common_flags(types.SimpleNamespace(
        generated_utc=UTC, script_commit=HEAD, script_commit_dl_id="DL-61", repeat_of=None, repeat_case=None,
        repeat_dl_id=None), real=True), td.Refused)

    def string_first():
        seen = []
        real_rp = os.path.realpath
        with fx.patched(os.path, realpath=lambda p, *a, **kw: seen.append(1) or real_rp(p, *a, **kw)):
            r = raises(lambda: td.refuse_test_path(e.tmp / ("a" + "TEST" + "b"), "x"), td.Refused)
        return r and not seen
    k["P21 a 'test' path is refused by its string, before any realpath"] = string_first

    # ---- D1 script
    def domain():
        m = torch.zeros(512, 512, dtype=torch.long)
        m[0, 0] = 255
        v = e.d1.valid_cells(m)
        return not v[0, 0] and int(v.sum()) == 64 * 64 - 1
    k["a2 the domain is the all-valid min-pool: one ignore pixel drops its cell only"] = domain

    def mean_prob():
        mp = e.d1.MeanProbability()
        for d in [5.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]:
            mp.add(torch.tensor([[[[0.0]], [[d]]]]))
        return int(mp.prediction().reshape(-1)[0]) == 0
    k["c3 the mean-probability prediction averages probabilities, not logits"] = mean_prob

    # ---- TRAIN canvas (smoke_score_teacher_train)
    def canvas():
        from PIL import Image

        from src.data.transforms import core_preprocess, finalize
        ds = td.train_canvas_dataset([0])
        img_path, mask_path = ds.base.pairs[0]
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            ref_img, ref_mask = finalize(*core_preprocess(im, mk))
        item = ds[0]
        return torch.equal(item["image"], ref_img) and torch.equal(item["target"], ref_mask)
    k["e the TRAIN canvas item equals core_preprocess + finalize bit for bit"] = canvas

    # ---- D2 gate (smoke_teacher_calibration d2_cases)
    def gate():
        out, art = fresh(e, "gate_out"), fresh(e, "gate_art")
        with fx.patched(td, R3_TOLERANCE=-1.0):
            code, _ = fx.call_run(e.d2, out_dir=out, artifact_dir=art, teacher_ckpt=str(e.ckpt), teacher_ckpt_sha256=e.sha,
                                  teacher_config=str(td.REPO / td.TEACHER_CONFIG_REL), teacher_role="record",
                                  purpose="item1", val_reference=str(e.ref), max_samples=3, generated_utc=UTC)
        return code == 1 and not art.exists()
    k["P18 a failed gate: exit 1, no evaluator artifact"] = gate

    # ---- D3 (smoke_teacher_d3)
    def d3_doc():
        out = fresh(e, "d3_out")
        present = tuple(r for r in td.CODE_FILES if (td.REPO / r).is_file())
        with fx.patched(td, CODE_FILES=present), fx.fake_git(head=HEAD):
            code, _ = fx.call_run(e.d3, factory=None, out_dir=out, train_scores=e.d3f["train"],
                                  perclass_val=e.d3f["val"], strata=e.d3f["strata"], script_commit=HEAD,
                                  script_commit_dl_id="DL-64")
        files = [p for p in fx.output_files(out) if p.suffix == ".json"]
        return code, (json.loads(files[0].read_text()) if files else {})

    def d3_median():
        code, doc = d3_doc()
        vals = [r["iou_train"] for r in doc.get("rows", []) if r["train_status"] == "ok"]
        return code == 0 and doc["summaries"]["overall"]["train"]["median"] == float(np.median(vals))
    k["P20 D3 group medians equal an independent median"] = d3_median

    def d3_69():
        code, doc = d3_doc()
        return code == 0 and {r["class_id"]: r for r in doc["rows"]}[69]["val_status"] == "not_evaluable"
    k["D3 class 69 (no VAL ground truth) is not evaluable on VAL"] = d3_69

    # ---- D4 hasher (smoke_teacher_d4 hasher_cases)
    def zero_fs():
        calls = []

        def counting(fn, name):
            return lambda *a, **kw: calls.append(name) or fn(*a, **kw)
        pat = dict(lstat=counting(os.lstat, "lstat"), stat=counting(os.stat, "stat"),
                   scandir=counting(os.scandir, "scandir"), listdir=counting(os.listdir, "listdir"))
        with fx.patched(os, **pat), fx.patched(builtins, open=counting(builtins.open, "open")), \
                fx.patched(io, open=counting(io.open, "io.open")):
            code, _ = hs_run(e, out=fresh(e, "h"), data_root=str(e.tmp / ("x" + "TeSt" + "x")))
        return code == 2 and not calls
    k["P32(f) an as-given 'test' path is refused with zero filesystem calls"] = zero_fs

    def hs_doc():
        out = fresh(e, "h")
        code, _ = hs_run(e, out=out)
        jp = out / e.hs.JSON_NAME
        return code, (json.loads(jp.read_text()) if jp.is_file() else {}), out

    def listing():
        listed = []
        rs, rl = os.scandir, os.listdir
        with fx.patched(os, scandir=lambda p=".", *a, **kw: listed.append(os.fspath(p)) or rs(p, *a, **kw),
                        listdir=lambda p=".", *a, **kw: listed.append(os.fspath(p)) or rl(p, *a, **kw)):
            code, _, _ = hs_doc()
        return code == 0 and sorted(set(listed)) == sorted(os.path.join(str(e.root), f) for f in e.hs.FOLDERS)
    k["f the hasher lists exactly the four folders"] = listing

    def sha1():
        import hashlib
        code, _, out = hs_doc()
        rows = [ln.split("\t") for ln in (out / e.hs.TSV_NAME).read_text().splitlines()[1:4]] if code == 0 else []
        return bool(rows) and all(hashlib.sha1((e.root / "images" / r[0] / r[2]).read_bytes()).hexdigest() == r[5] for r in rows)
    k["f SHA-1 equals hashlib on the file bytes"] = sha1

    def masks():
        code, doc, _ = hs_doc()
        g = {tuple(x["val"]): x["masks_identical"] for x in doc.get("groups", [])}
        return code == 0 and g.get(tuple(e.planted["A"][1])) is True and g.get(tuple(e.planted["B"][1])) is False
    k["f mask identity per group (identical / different)"] = masks

    def val_only():
        code, doc, _ = hs_doc()
        return code == 0 and [x["val"] for x in doc.get("val_only_groups", [])] == [e.planted["val_only"]] \
            and not set(e.planted["val_only"]) & set(doc.get("duplicate_val_ids", []))
    k["P22 the VAL-only group is reported and not removed"] = val_only

    # ---- D4 reducer (smoke_teacher_d4 reducer_cases)
    k["f margin_dedup equals an independent float64 recomputation"] = lambda: (
        lambda r: r[0] == 0 and r[1].get("margin_dedup") == e.sub_m)(d4_run(e))
    k["f the reducer drops exactly the listed rows (n_dup 3, n_kept 843)"] = lambda: (
        lambda r: r[0] == 0 and r[1].get("teacher", {}).get("n_dup") == 3
        and r[1].get("teacher", {}).get("n_kept_images_with_rows") == 843)(d4_run(e))
    k["P24 a full-set value that differs from summary.json is a STOP"] = lambda: (
        lambda r: r[0] == 1 and not r[2])(d4_run(e, teacher=e.a_te, gap_output=e.ge))
    k["P24 a full set that does not reproduce S3's point is a STOP"] = lambda: (
        lambda r: r[0] == 1 and not r[2])(d4_run(e, gap_output=e.goff))

    def sign():
        with fx.patched(e.d4, float32_margin=lambda t, v: 1.0):
            code, doc, _ = d4_run(e)
        return code == 1 and doc.get("status") == "sign disagreement"
    k["P24 a sign disagreement exits 1"] = sign
    k["P32(f) margin = 0 gives margin_dedup_le_0 true"] = lambda: (
        lambda r: r[0] == 0 and r[1].get("margin_dedup") == 0.0 and r[1].get("margin_dedup_le_0") is True)(
        d4_run(e, e1=e.a_e0, gap_output=e.g0))
    k["P32(f) a second teacher artifact that passes check_pairing is refused"] = lambda: d4_run(e, teacher=e.a_t2)[0] == 2

    # ---- added after the DL-24 workflow (each mirrors a new smoke case)
    k["b1 one tied draw (5, 5, 0) among seven (5, 4, 0): first index, N == 0"] = s1.case_tie_per_draw
    k["b2 F_lesion's cells come from the mean probability, not the mean logit"] = s1.case_lesion_mean_probability
    k["b3 F_halves compares half means of probabilities, not of logits"] = s1.case_halves_mean_probability
    k["b7 D1's share > 0.99 is strict"] = s1.case_strict_099
    k["P16 the ECE share > 0.99 is strict"] = sc.case_strict_099
    k["d6 the disease subset follows the ground truth, not the prediction"] = sc.case_disease_by_ground_truth

    def fresh_load():
        return s1.gated_load(td, e.args(), factory=fx.stub_factory)
    k["P8 a real run refuses None or empty provenance fields, naming them"] = lambda: (
        lambda il: s1.case_p8_nonempty(td, il[1], il[0]))(fresh_load())
    k["P8 the written record must hold every field"] = lambda: (
        lambda il: s1.case_p8_record(td, il[1], il[0], il[2]))(fresh_load())
    k["P7 a frozen blob id other than the table's is refused"] = lambda: s1.case_frozen_blob(td)
    k["P7 a segmentor in training mode stops"] = lambda: s1.case_training_mode(td, fx, e.args)
    k["P9 a second isolated NMF module stops the after-load check"] = lambda: s1.case_extra_isolated(td, fx, e.args)
    k["C1 a begin description that differs from stream.describe() stops"] = lambda: s1.case_describe_mismatch(td, fx, e.args)
    k["C1 the stream is the adapter's live object after a fresh load"] = lambda: s1.case_c1_object(td, fx, e.args)
    k["P29 exit codes: STOP classes 1, refusal classes 2, anything else 4"] = lambda: s1.case_exit_codes(td)

    def d1_flags(**kw):
        base = dict(part="crops", correction_state="declined", correction_dl_id="DL-62", strata="s", d2_val_output="d",
                    val_reference=None, max_samples=None, n_crops=None)
        base.update(kw)
        try:
            e.d1.check_d1_flags(types.SimpleNamespace(**base), real=True)
            return ""
        except td.Refused as ex:
            return str(ex)
    k["P28 a real run requires the correction flags"] = lambda: "--correction-state" in d1_flags(
        correction_state=None, correction_dl_id=None)
    k["P2 --n-crops is refused in a real run"] = lambda: "--n-crops" in d1_flags(n_crops=3)

    def d2_max_samples():
        try:
            e.d2.check_d2_flags(types.SimpleNamespace(teacher_role="record", purpose="item1", of_record_output=None,
                                                      max_samples=3), real=True)
            return False
        except td.Refused as ex:
            return "--max-samples" in str(ex)
    k["P2 D2's --max-samples is refused in a real run"] = d2_max_samples

    def train_max_samples():
        from scripts import score_teacher_train as st
        code, err = fx.call_run(st, factory=None, out_dir=fresh(e, "tr"), artifact_dir=fresh(e, "tra"), strata=str(e.strata),
                                max_samples=4, script_commit=HEAD, script_commit_dl_id="DL-61", teacher_role="record",
                                **e.teacher)
        return code == 2 and "--max-samples is a stub-mode flag" in err
    k["P2 TRAIN's --max-samples is refused in a real run"] = train_max_samples

    corr = dict(correction_state="declined", correction_dl_id="DL-62")

    def crops_run(**kw):
        out = fresh(e, "crops")
        code, err = fx.call_run(e.d1, part="crops", out_dir=out, strata=str(e.strata), d2_val_output=str(e.d2_json),
                                n_crops=2, generated_utc=UTC, **corr, **{**dict(teacher_role="record", **e.teacher), **kw})
        files = fx.output_files(out)
        return code, err, (json.loads(files[0].read_text()) if files else {})

    def crop_wiring():
        code, _, doc = crops_run()
        return code == 0 and [p["feature_sha256"] for p in doc.get("per_crop", [])] == s1.crop_feature_hashes(e, 2)
    k["a2 run-level crop wiring: the run's crops are RandomState(1801 + i)"] = crop_wiring

    def arm_mismatch():
        code, err, _ = crops_run(**e.arm)
        return code == 2 and "arm R1" in err
    k["P11 an arm run refuses a D2 output that is not its own arm's"] = arm_mismatch

    def val_run():
        out = fresh(e, "d1val")
        code, err = fx.call_run(e.d1, part="val", out_dir=out, teacher_role="record", val_reference=str(e.ref),
                                max_samples=3, generated_utc=UTC, **corr, **e.teacher)
        files = fx.output_files(out)
        return code, err, (json.loads(files[0].read_text()) if files else {}), out

    def mean_prob_wiring():
        code, _, doc, _ = val_run()
        return code == 0 and doc.get("mean_probability", {}).get("all_class_miou") == s1.mean_probability_miou(e)
    k["c3 run-level wiring: the mean-probability mIoU equals an independent eight-stream recomputation"] = mean_prob_wiring

    def core_confusion():
        import src.eval.metrics as em
        real_cm = em.confusion_matrix

        def skewed(pred, target, num_classes, ignore_index=255):
            cm = real_cm(pred, target, num_classes, ignore_index)
            cm[0, 0] += 1
            return cm
        with fx.patched(em, confusion_matrix=skewed):
            code, err, _, out = val_run()
        return code == 1 and not fx.output_files(out)
    k["c2 a stream-42 confusion that differs from the evaluator core's stops"] = core_confusion
    want_ref = {"of record": False, "another checkpoint": True, "a smoke status": True, "an mIoU 2e-5 from R3": True,
                "stage E1": True}
    k["P18 real --val-reference: record passes; another checkpoint, smoke, 2e-5 off and stage E1 are refused"] = lambda: (
        sc.val_reference_real_outcomes(e.ref, _mk(fresh(e, "refreal"))) == want_ref)

    def control_other():
        doc = json.loads(e.d2_json.read_text())
        doc["teacher"]["loaded_state_sha256"] = "0" * 64
        other = fresh(e, "item1_other")
        other.write_text(json.dumps(doc))
        out = fresh(e, "ctl")
        code, err = fx.call_run(e.d2, out_dir=out, artifact_dir=fresh(e, "ctl_art"), teacher_role="record",
                                purpose="control", of_record_output=str(other), val_reference=str(e.ref), max_samples=3,
                                generated_utc=UTC, **e.teacher)
        return code == 2 and not fx.output_files(out)
    k["P4 a control re-score of another teacher is refused"] = control_other

    def d3_77():
        code, doc = d3_doc()
        r = {x["class_id"]: x for x in doc.get("rows", [])}
        return code == 0 and r[77]["train_status"] == "not_evaluable" and r[77]["train_minus_val"] is None
    k["D3 a class without TRAIN ground truth is not evaluable on TRAIN"] = d3_77

    def gt_present():
        code, doc, _ = d4_run(e)
        allimg = torch.ones(846, dtype=torch.bool)
        got = doc.get("teacher", {}).get("full", {}).get("gt_present", {}).get("all_class", {}).get("value_float64")
        sub = doc.get("teacher", {}).get("subset", {}).get("gt_present", {}).get("all_class", {}).get("value_float64")
        return (code == 0 and got == e.s4.miou64(e.gts, e.pt, allimg, "gt")
                and sub == e.s4.miou64(e.gts, e.pt, e.keep, "gt"))
    k["STATISTICS-1 the GT-present values equal an independent recomputation"] = gt_present
    k["P23 a pair failing S3's check_pairing is refused"] = lambda: (
        d4_run(e, e1=e.a_t2, gap_output=e.g_role)[0] == 2 and "role e1" in e.last_err)
    k["P23 a pair scored against other ground truth is refused"] = lambda: (
        d4_run(e, e1=e.a_eg, gap_output=e.g_gt)[0] == 2 and "ground-truth" in e.last_err)
    k["P21 the hasher refuses a data root resolving to a 'test' location"] = lambda: (
        lambda r: r[0] == 2 and "resolving" in r[1])(hs_run(e, out=fresh(e, "h"), data_root=str(e.link)))

    def missing_mask():
        real_list = e.hs.list_folder
        drop = f"{e.stems['val'][20]}.png"
        with fx.patched(e.hs, list_folder=lambda f: [n for n in real_list(f) if n != drop]):
            code, err = hs_run(e, out=fresh(e, "h"))
        return code == 2 and "without a .png mask" in err
    k["P21 a VAL image without its mask is refused"] = missing_mask
    return k


# ===================================================================================================
# the mutations
# ===================================================================================================
def mutations(e) -> list:
    from src.eval import calibration as cal
    from src.eval import nmf_sensitivity as ns
    td, d1, d2, d3, d4, hs, fx = e.td, e.d1, e.d2, e.d3, e.d4, e.hs, e.fx
    F = "if False:"
    return [
        # P33
        ("P33 a float F rule (below 0.03)", ns, "branch", [("below = 100 * N < 3 * D", "below = N / D < 0.03")],
         "P33 big-integer pairs: integers decide, a float rule would not"),
        ("P33 a float F rule (at least 0.10)", ns, "branch", [("at_least = 10 * N >= D", "at_least = N / D >= 0.10")],
         "P33 big-integer pairs: integers decide, a float rule would not"),
        ("P33 a float32 softmax in the ECE", cal.CalibrationAccumulator, "update",
         [(".cpu()[:, idx].to(torch.float64)", ".cpu()[:, idx].to(torch.float32)")],
         "d7 P32(d) float32 logits: ECE == 0.2799999955678827"),
        ("P33 side='right' in the ECE bins", cal, "bin_index", [('side="left"', 'side="right"')],
         "d3 conf == m/15 lands in bin m (m = 1..14)"),
        ("P33 pbar as the softmax of mean logits", ns, "crop_statistics",
         [("log_pbar = logsumexp(lsm4, axis=0) - LN_K", "log_pbar = log_softmax(z64.mean(axis=0) / T_KD, axis=0)")],
         "b4 KL_logit sum equals the numpy reference to 1e-12"),
        ("P33 a T^2 factor on KL_logit", ns, "crop_statistics",
         [("out.kl_logit_sum = math.fsum(", "out.kl_logit_sum = T_KD ** 2 * math.fsum(")],
         "b4 KL_logit sum equals the numpy reference to 1e-12"),
        ("P33 an any-valid mask", d1, "valid_cells",
         [("return downsample_validity(mask[None], GRID, ignore_index=IGNORE_INDEX)[0].numpy()",
           "return (__import__('torch').nn.functional.adaptive_max_pool2d((mask[None] != IGNORE_INDEX).float()"
           ".unsqueeze(1), GRID).squeeze(1) > 0)[0].numpy()")],
         "a2 the domain is the all-valid min-pool: one ignore pixel drops its cell only"),
        ("P33 a |V_i|-weighted KL_cwd", ns, "summarize",
         [("(math.fsum(c.kl_cwd_sum for c in crops) / n_cwd)",
           "(math.fsum(c.kl_cwd_sum * c.n_valid for c in crops) / math.fsum(c.n_cwd * c.n_valid for c in crops))")],
         "b5 KL_cwd and KL_logit pooled over two crops of different |V|"),
        # D1 statistics
        ("flips over ordered pairs (56, not 28)", ns, "pairwise_flips",
         [("for b in range(a + 1, k))", "for b in range(k) if b != a)")], "b1 flips: 0000 1111 -> 16 pairs; eight labels -> 28"),
        ("D = 8 |V| instead of 28 |V|", ns, "crop_statistics",
         [("out = CropStats(n_valid=nv, D=N_PAIRS * nv)", "out = CropStats(n_valid=nv, D=K_DRAWS * nv)")],
         "b1 N = 21, D = 700 on a 25-cell grid"),
        ("F_lesion admits cells peaking at the background", ns, "crop_statistics",
         [("lesion = (peak >= disease_min) & (peak <= disease_max)", "lesion = peak <= disease_max")],
         "b2 F_lesion counts only cells peaking at a disease class"),
        ("F_halves from draw 1 vs draw 8 instead of the half means", ns, "crop_statistics",
         [("half_a = np.argmax(p1[:4].sum(axis=0) / 4.0, axis=0)", "half_a = labels[0]"),
          ("half_b = np.argmax(p1[4:].sum(axis=0) / 4.0, axis=0)", "half_b = labels[-1]")],
         "b3 F_halves = 1 flip over 4 cells"),
        ("bootstrap percentile method 'nearest'", ns, "bootstrap_f",
         [('list(BOOTSTRAP_LEVEL), method="linear")', 'list(BOOTSTRAP_LEVEL), method="nearest")')],
         "b6 P32 bootstrap CI exactly [0.07283787157730934, 0.08666608984644292]"),
        ("bootstrap seed 1802", ns, "bootstrap_f", [("rng = np.random.default_rng(seed)", "rng = np.random.default_rng(seed + 1)")],
         "b6 P15 index sha256 73f7b7cb... (numpy 1.26.4)"),
        ("a replicate with sum D == 0 kept", ns, "bootstrap_f", [("    if zero_d:\n", "    if False:\n")],
         "b6 a replicate with sum D == 0 makes the CI null"),
        # ECE
        ("ECE without the n_m / n weights", cal, "ece_from_bins", [("(int(n_m[m]) / n) * abs(", "abs(")],
         "d2 two-bin hand case: ECE = 0.55"),
        ("ECE counts pixels labelled 255", cal.CalibrationAccumulator, "update",
         [("idx = torch.nonzero(t != self.ignore_index, as_tuple=False).reshape(-1)", "idx = torch.arange(t.numel())")],
         "d4 pixels labelled 255 change nothing"),
        ("disease ECE includes the background", cal.CalibrationAccumulator, "update",
         [("dmask = (tvn >= self.disease_lo) & (tvn <= self.disease_hi)", "dmask = tvn <= self.disease_hi")],
         "d6 disease-pixel ECE over target in 1..C-1 only"),
        ("the hook keeps the core's tensor", cal.CalibratingBatchForward, "__call__",
         [("self.last_logits_ref = weakref.ref(logits)", "self.last_logits_ref = weakref.ref(logits); self._kept = logits")],
         "d5 the hook keeps no tensor (weakref dead after the core's del)"),
        ("the hook skips the _version check", cal.CalibratingBatchForward, "__call__",
         [("if logits._version != version:", F)], "d5 an in-place change of the core's tensor stops"),
        # the seam
        ("P9 stream identity check removed", td.SplitTeacher, "head", [("if mods[0].nmf_stream is not stream:", F)],
         "P9 a head call with a stream that is not the attached object stops"),
        ("P9 identity check removed: the begin call's dict is taken as a stream", td.SplitTeacher, "head", [("if mods[0].nmf_stream is not stream:", F)],
         "C1 handing the head check the begin call's dict stops"),
        ("P9 one-draw check removed", td.SplitTeacher, "head", [("if stream.draws != before + 1:", F)],
         "P9 a head drawing twice per call stops"),
        ("P9 feature-hash check removed", td.SplitTeacher, "head", [("if feature_sha256(feats) != feat_hash:", F)],
         "P9 an in-place-mutating head stops"),
        ("P9 finite check removed", td.SplitTeacher, "head", [("if not bool(torch.isfinite(z).all()):", F)],
         "P9 a NaN head stops"),
        ("P9 caller-RNG check removed", td.SplitTeacher, "head", [("if rng_state_sha256() != rng:", F)],
         "P9 a head drawing from the global CPU RNG stops"),
        ("P5 sha256 format regex removed", td, "check_teacher_flags",
         [("if sha is None or not SHA256_RE.fullmatch(sha):", "if sha is None:")],
         "P5 the format check alone refuses empty, uppercase and 63-character sha256"),
        ("P5 own sha256 == flag removed", td, "verify_teacher_inputs", [("if sha != args.teacher_ckpt_sha256:", F)],
         "P5 a well-formed wrong sha256 is refused with the load counter at 0"),
        ("P2 16 MiB stat guard removed", td, "verify_teacher_inputs", [("if stub and size > STUB_MAX_CKPT_BYTES:", F)],
         "P2 stub refuses a checkpoint over 16 MiB before any load"),
        ("P2 registered-sha guard removed", td, "verify_teacher_inputs",
         [("if stub and sha in REGISTERED_TEACHER_SHA256:", F)], "P2 stub refuses a registered teacher sha256 before any load"),
        ("P2 mmseg guard removed", td, "after_load_checks", [('if "mmseg" in sys.modules:', F)],
         "P2 stub refuses a build that leaves mmseg imported"),
        ("P2 parameter guard removed", td, "after_load_checks", [("if n_params > STUB_MAX_PARAMETERS:", F)],
         "P2 stub refuses a model over 1e6 parameters (own big_factory checkpoint; the message names the count)"),
        ("P8 provenance field count removed", td, "require_provenance_field_count",
         [("if real and len(names) != EXPECTED_PROVENANCE_FIELDS:", F)],
         "P8 the field count on stand-ins: 11 fields refused, 12 not (holds before and after K-part)"),
        ("P26 HEAD == --script-commit removed", td, "require_commit_binding", [("if head != script_commit:", F)],
         "P26 a HEAD other than --script-commit is refused"),
        ("P26 clean-tree check removed", td, "require_commit_binding", [("if tree is None or tree.strip():", F)],
         "P26 a dirty src/configs/scripts tree is refused"),
        ("P27 one output per kind removed", td, "require_single_output", [("if existing and repeat_of is None:", F)],
         "P27 a second non-smoke output is refused"),
        ("C2 --generated-utc accepted in real mode", td, "check_common_flags",
         [('if real:\n            raise Refused("--generated-utc', 'if False:\n            raise Refused("--generated-utc')],
         "C2 real mode refuses --generated-utc"),
        ("P21 the string check removed (realpath only)", td, "refuse_test_path", [('if "test" in s.lower():', F)],
         "P21 a 'test' path is refused by its string, before any realpath"),
        ("TRAIN canvas mirrored (not the VAL branch)", td, "train_canvas_dataset",
         [("img_np, mask_np = core_preprocess(im, mk)", "img_np, mask_np = core_preprocess(im.transpose(0), mk.transpose(0))")],
         "e the TRAIN canvas item equals core_preprocess + finalize bit for bit"),
        # D1 script, D2, D3
        ("mean-probability prediction from logits", d1.MeanProbability, "add",
         [("p = torch.softmax(logits.to(torch.float64), dim=1)", "p = logits.to(torch.float64)")],
         "c3 the mean-probability prediction averages probabilities, not logits"),
        ("P18 the reproduction gate removed", d2, "_run",
         [("passed = delta is not None and delta <= td.R3_TOLERANCE", "passed = True")],
         "P18 a failed gate: exit 1, no evaluator artifact"),
        ("D3 median replaced by the mean", d3, "_stats",
         [("float(np.median(np.asarray(values, dtype=np.float64)))", "float(np.mean(np.asarray(values, dtype=np.float64)))")],
         "P20 D3 group medians equal an independent median"),
        ("D3 a class without VAL ground truth counted as evaluable", d3, "build_rows",
         [('v_ok = vrow.get("status") == OK if c != BACKGROUND else', "v_ok = True if c != BACKGROUND else")],
         "D3 class 69 (no VAL ground truth) is not evaluable on VAL"),
        # D4
        ("P21 hasher string guard removed", hs, "guard_strings",
         [('bad = [s for s in named if "test" in str(s).lower()]', "bad = []")],
         "P32(f) an as-given 'test' path is refused with zero filesystem calls"),
        ("the hasher also lists the data root", hs, "listing", [("    out = {}\n", "    os.listdir(root)\n    out = {}\n")],
         "f the hasher lists exactly the four folders"),
        ("SHA-1 of the first byte only", hs, "digests", [("h1.update(block)", "h1.update(block[:1])")],
         "f SHA-1 equals hashlib on the file bytes"),
        ("mask identity always true", hs, "masks_identical", [("return all(a.shape", "return True or all(a.shape")],
         "f mask identity per group (identical / different)"),
        ("VAL-only groups treated as duplicates", hs, "group",
         [("val_only.append((sha, members))", "cross.append((sha, members))")],
         "P22 the VAL-only group is reported and not removed"),
        ("rows dropped by keeping only the duplicates", d4, "subset_totals", [("keep = ~np.isin(", "keep = np.isin(")],
         "f the reducer drops exactly the listed rows (n_dup 3, n_kept 843)"),
        ("margin_dedup from the float32 values", d4, "margin_block",
         [('m64 = t_u["value_float64"] - e_u["value_float64"]', 'm64 = t_u["value"] - e_u["value"]')],
         "f margin_dedup equals an independent float64 recomputation"),
        ("full-set summary reproduction removed", d4, "model_block",
         [('if full["union_present"][scope]["value"] != dl.get(key):', F)],
         "P24 a full-set value that differs from summary.json is a STOP"),
        ("S3 point reproduction removed", d4, "_run",
         [("if not isinstance(point, float) or abs(full_diff - point) > POINT_TOLERANCE:", F)],
         "P24 a full set that does not reproduce S3's point is a STOP"),
        ("sign agreement assumed", d4, "decide", [("agree = len(set(signs.values())) == 1", "agree = True")],
         "P24 a sign disagreement exits 1"),
        ("margin_dedup_le_0 as a strict < 0", d4, "decide", [('"margin_dedup_le_0": m64 <= 0', '"margin_dedup_le_0": m64 < 0')],
         "P32(f) margin = 0 gives margin_dedup_le_0 true"),
        ("P23 artifact binding to the gap output removed", d4, "_run",
         [('if ((gdoc.get("inputs") or {}).get(role) or {}).get("artifact_sha256s") != art.file_sha256s:', F)],
         "P32(f) a second teacher artifact that passes check_pairing is refused"),
        # added after the DL-24 workflow
        ("argmax takes the last of tied classes", ns, "crop_statistics",
         [("labels = np.argmax(zv, axis=1)", "labels = zv.shape[1] - 1 - np.argmax(zv[:, ::-1], axis=1)")],
         "b1 one tied draw (5, 5, 0) among seven (5, 4, 0): first index, N == 0"),
        ("F_lesion's peak from the mean logit", ns, "crop_statistics",
         [("mean_all = p1.sum(axis=0) / K_DRAWS", "mean_all = z64.sum(axis=0) / K_DRAWS")],
         "b2 F_lesion's cells come from the mean probability, not the mean logit"),
        ("F_halves from half means of logits", ns, "crop_statistics",
         [("half_a = np.argmax(p1[:4].sum(axis=0) / 4.0, axis=0)", "half_a = np.argmax(z64[:4].sum(axis=0), axis=0)"),
          ("half_b = np.argmax(p1[4:].sum(axis=0) / 4.0, axis=0)", "half_b = np.argmax(z64[4:].sum(axis=0), axis=0)")],
         "b3 F_halves compares half means of probabilities, not of logits"),
        ("D1 share >= 0.99 instead of > 0.99", ns, "crop_statistics",
         [("int(np.count_nonzero(max1 > HIGH_CONFIDENCE))", "int(np.count_nonzero(max1 >= HIGH_CONFIDENCE))")],
         "b7 D1's share > 0.99 is strict"),
        ("ECE share >= 0.99 instead of > 0.99", cal.CalibrationAccumulator, "update",
         [("int(np.count_nonzero(max1 > HIGH_CONFIDENCE))", "int(np.count_nonzero(max1 >= HIGH_CONFIDENCE))")],
         "P16 the ECE share > 0.99 is strict"),
        ("disease subset chosen by the prediction", cal.CalibrationAccumulator, "update",
         [("dmask = (tvn >= self.disease_lo) & (tvn <= self.disease_hi)",
           "pvn = pv.numpy(); dmask = (pvn >= self.disease_lo) & (pvn <= self.disease_hi)")],
         "d6 the disease subset follows the ground truth, not the prediction"),
        ("P8 empty-field check removed", td, "after_load_checks",
         [('empty = [f.name for f in fields if getattr(prov, f.name) in (None, "")]', "empty = []")],
         "P8 a real run refuses None or empty provenance fields, naming them"),
        ("P8 record check removed", td, "teacher_record",
         [('bad = [n for n in provenance_fields() if provenance.get(n) in (None, "")]', "bad = []")],
         "P8 the written record must hold every field"),
        ("P7 frozen blob comparison removed", td, "frozen_blob_record", [("if got != blob:", F)],
         "P7 a frozen blob id other than the table's is refused"),
        ("P7 training-mode check removed", td, "after_load_checks", [("if loaded.segmentor.training is not False:", F)],
         "P7 a segmentor in training mode stops"),
        ("P9 one-isolated-module check removed after the load", td, "after_load_checks",
         [("if len(mods) != 1 or mods[0].nmf_stream is not loaded.stream:", F)],
         "P9 a second isolated NMF module stops the after-load check"),
        ("C1 describe() == description check removed", td, "load_teacher", [("if stream.describe() != description:", F)],
         "C1 a begin description that differs from stream.describe() stops"),
        ("C1 the begin call's dict used as the stream", td, "load_teacher",
         [('stream = getattr(adapter, "nmf_stream", None)', "stream = description")],
         "C1 the stream is the adapter's live object after a fresh load"),
        ("P29 imported STOP classes not mapped to 1", td, "exit_code_for",
         [("if isinstance(exc, Stop) or isinstance(exc, _loaded_classes(_STOP_CLASSES)):", "if isinstance(exc, Stop):")],
         "P29 exit codes: STOP classes 1, refusal classes 2, anything else 4"),
        ("P29 an unexpected exception exits 2", td, "cli_main", [("return EXIT_ERROR", "return EXIT_REFUSED")],
         "P29 exit codes: STOP classes 1, refusal classes 2, anything else 4"),
        ("P28 correction flags optional in a real run", d1, "check_d1_flags",
         [("if real or args.correction_state is not None or args.correction_dl_id is not None:",
           "if args.correction_state is not None or args.correction_dl_id is not None:")],
         "P28 a real run requires the correction flags"),
        ("P2 --n-crops accepted in a real run", d1, "check_d1_flags",
         [("if args.n_crops is not None and (real or not 1 <= args.n_crops <= SAMPLE_COUNT):",
           "if args.n_crops is not None and (not 1 <= args.n_crops <= SAMPLE_COUNT):")],
         "P2 --n-crops is refused in a real run"),
        ("P2 D2 --max-samples accepted in a real run", d2, "check_d2_flags",
         [('if real:\n            raise td.Refused("--max-samples', 'if False:\n            raise td.Refused("--max-samples')],
         "P2 D2's --max-samples is refused in a real run"),
        ("P2 TRAIN --max-samples accepted in a real run", e.st, "_run",
         [("(real or not 1 <= args.max_samples <= td.TRAIN_ROWS)", "(not 1 <= args.max_samples <= td.TRAIN_ROWS)")],
         "P2 TRAIN's --max-samples is refused in a real run"),
        ("every crop drawn with RandomState(1801)", d1, "crop",
         [("np.random.RandomState(CROP_SEED_BASE + i)", "np.random.RandomState(CROP_SEED_BASE)")],
         "a2 run-level crop wiring: the run's crops are RandomState(1801 + i)"),
        ("the mean probability fed by stream 42 only", d1.EightStreamForward, "_forward_all",
         [("mean.add(z)", "mean.add(z) if k == 0 else None")],
         "c3 run-level wiring: the mean-probability mIoU equals an independent eight-stream recomputation"),
        ("core confusion == stream 42 check removed", d1, "_run_val", [("if not core_equal:", F)],
         "c2 a stream-42 confusion that differs from the evaluator core's stops"),
        ("--val-reference record-sha check removed", td, "check_val_reference",
         [('if run.get("checkpoint_sha256") != RECORD_SHA256:', F)],
         "P18 real --val-reference: record passes; another checkpoint, smoke, 2e-5 off and stage E1 are refused"),
        ("--val-reference real-status check removed", td, "check_val_reference",
         [('if run.get("artifact_status") not in ("provisional", "official"):', F)],
         "P18 real --val-reference: record passes; another checkpoint, smoke, 2e-5 off and stage E1 are refused"),
        ("--val-reference R3 tolerance check removed", td, "check_val_reference",
         [("if abs(miou - R3_VAL_MIOU) > R3_TOLERANCE:", F)],
         "P18 real --val-reference: record passes; another checkpoint, smoke, 2e-5 off and stage E1 are refused"),
        ("--val-reference stage/role/precision check removed", td, "check_val_reference",
         [('if (run.get("stage"), run.get("model_role"), run.get("precision")) != ("teacher", "teacher", "fp32"):', F)],
         "P18 real --val-reference: record passes; another checkpoint, smoke, 2e-5 off and stage E1 are refused"),
        ("P4 control teacher identity removed", d2, "_run",
         [('diffs = td.same_teacher(teacher, of_record["doc"].get("teacher") or {})', "diffs = []")],
         "P4 a control re-score of another teacher is refused"),
        ("P11 arm-id check of --d2-val-output removed", d1, "_run_crops",
         [('if role == "arm" and (d2t.get("role") != "arm" or d2t.get("arm_id") != args.arm_id):', F)],
         "P11 an arm run refuses a D2 output that is not its own arm's"),
        ("D3 every class evaluable on TRAIN", d3, "build_rows", [("t_ok = tgt > 0", "t_ok = True")],
         "D3 a class without TRAIN ground truth is not evaluable on TRAIN"),
        ("D4 GT-present computed with the union-present rule", d4, "scores",
         [("dm = dataset_miou(tot, classes, rule)", "dm = dataset_miou(tot, classes, UNION_PRESENT)")],
         "STATISTICS-1 the GT-present values equal an independent recomputation"),
        ("P23 check_pairing removed", d4, "_run", [("gap.check_pairing(art_t, art_e)", "pass")],
         "P23 a pair failing S3's check_pairing is refused"),
        ("P23 identical ground truth check removed", d4, "_run", [("paired_totals_identical_gt(stages)", "pass")],
         "P23 a pair scored against other ground truth is refused"),
        ("P21 hasher resolved-path guard removed", hs, "guard_resolved",
         [('bad = [r for r in real if "test" in r.lower()]', "bad = []")],
         "P21 the hasher refuses a data root resolving to a 'test' location"),
        ("hasher accepts an image without its mask", hs, "_run",
         [('missing = [s for s in stems if s + MASK_SUFFIX not in lst[split]["masks"]]', "missing = []")],
         "P21 a VAL image without its mask is refused"),
        # P41 follow-up (GO-2 of 2026-10-04): appended, so M01-M89 keep their numbers
        ("h1 the stub checkpoint written as zeros (not the factory's state)", fx, "write_stub_ckpt",
         [("state = {k: v.detach().clone() for k, v in (factory or stub_factory)().state_dict().items()}",
           "state = {k: torch.zeros_like(v) for k, v in (factory or stub_factory)().state_dict().items()}")],
         "h1 the stub checkpoint holds exactly the factory's state_dict; the loaded stub equals the factory's state"),
        ("h3 the reference builder omits --teacher-ckpt-sha256", fx, "reference_artifact",
         [('"--teacher-ckpt-sha256", sha256,', "")],
         "h3 the reference builder passes --teacher-ckpt-sha256 (built and recorded; a wrong sha refused)"),
        ("a the kd load line drops expected_sha256", td, "load_teacher",
         [("load_frozen_teacher(str(inputs.ckpt), config_path=str(inputs.config), expected_sha256=sha, **kw)",
           "load_frozen_teacher(str(inputs.ckpt), config_path=str(inputs.config), **kw)")],
         "a both load lines pass the verified sha (expected_sha256 recorded in the kd and evaluator form)"),
        ("a the evaluator load line drops expected_sha256", td, "load_teacher",
         [("load_teacher_model(resolved, config_path=str(inputs.config), expected_sha256=sha, **kw)",
           "load_teacher_model(resolved, config_path=str(inputs.config), **kw)")],
         "a both load lines pass the verified sha (expected_sha256 recorded in the kd and evaluator form)"),
        ("h4 TeacherChecksumMismatch dropped from the refusal classes", td, "exit_code_for",
         [("_loaded_classes(_REFUSAL_CLASSES)", "_loaded_classes(_REFUSAL_CLASSES[:4] + _REFUSAL_CLASSES[5:])")],
         "h4 a checkpoint changed after validation is refused with exit 2 (TeacherChecksumMismatch)"),
        ("h4 TeacherStateDictMismatch dropped from the refusal classes", td, "exit_code_for",
         [("_loaded_classes(_REFUSAL_CLASSES)", "_loaded_classes(_REFUSAL_CLASSES[:5])")],
         "h4 a checkpoint the strict load refuses is refused with exit 2 (TeacherStateDictMismatch)"),
    ]


def run_killer(fn) -> tuple[bool, str]:
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            return bool(fn()), ""
    except Exception as ex:  # noqa: BLE001 -- a raise under a mutation is a failed case
        return False, f"{type(ex).__name__}: {ex}"


def main() -> int:
    e = None
    try:
        e = build_env()
        ks = killers(e)
        baseline = {}
        for name, fn in ks.items():
            ok, why = run_killer(fn)
            baseline[name] = ok
            check(f"baseline (unmutated): {name}", ok, why)
        ms = mutations(e)
        used = set()
        for i, (desc, owner, fname, pairs, killer) in enumerate(ms, 1):
            used.add(killer)
            if killer not in ks:
                check(f"M{i:02d} {desc}", False, f"no case named {killer!r}")
                continue
            try:
                with mutate(owner, fname, *pairs):
                    ok, why = run_killer(ks[killer])
            except AssertionError as ex:
                check(f"M{i:02d} {desc}", False, f"mutation not applied: {ex}")
                continue
            check(f"M{i:02d} KILLED {desc}  <- {killer}", baseline.get(killer) and not ok,
                  "the case still passes under the mutation" if ok else "")
        check("every killing case is used by at least one mutation", set(ks) <= used, str(sorted(set(ks) - used)))
    except Exception as ex:  # noqa: BLE001 -- a crash is a failed case, never a pass
        import traceback
        check("harness raised", False, f"{type(ex).__name__}: {ex} {traceback.format_exc()[-800:]}")
    finally:
        if e is not None:
            shutil.rmtree(e.tmp, ignore_errors=True)
            shutil.rmtree(Path(e.root).parent, ignore_errors=True)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    good = sum(ok for _, ok, _ in RESULTS)
    n_mut = sum(1 for n, _, _ in RESULTS if n.startswith("M"))
    killed = sum(1 for n, ok, _ in RESULTS if n.startswith("M") and ok)
    print(f"mutations killed: {killed}/{n_mut}")
    print(f"RESULT: {'PASS' if good == len(RESULTS) else 'FAIL'} ({good}/{len(RESULTS)})")
    return 0 if good == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
