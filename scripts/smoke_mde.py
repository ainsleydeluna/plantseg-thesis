#!/usr/bin/env python3
"""Synthetic smoke for the AM-17 item 3 MDE entry and the DL-27 band (lane L-AM17-MDE, S4).

SYNTHETIC ONLY. The differences of groups U and D are generated here; the VAL artifacts of groups C, R and S
are written by scripts/stats_fixtures.py through the unmodified evaluator writer into a temporary fixture
repository, then dressed (stats_fixtures.edit_summary) with the DL-17 B66 pins seed 42's artifact carries:
eval_runtime (cuda:0, NVIDIA A40, batch 16, 53 batches, image b80b645d..., iteration 80000), the checkpoint
sha256 cf0879f7..., dataset_level.all_class_miou 0.36314016580581665 and per_class.gt_support summing to
159,279,104. No PlantSeg image, mask, checkpoint or TEST file is read; nothing is written into the repository.
Every created path lies under the temporary root. Its parent is TMPDIR, whose string and realpath are both
checked to contain no "test" before anything is created; its own name is a fixed prefix plus hex digits, which
cannot spell "test". The TEST-path refusals use paths that are never created. R6 (an out-dir inside the
repository) and R14 (a path resolving to a test location) run with td.write_files_exclusive replaced by a
tripwire; every other check writes only under the temporary root. A child process (R16a-b) shows only the type
and message of an uncaught error, so no traceback line is read.

  U  units of src/stats/mde.py: U0 the constants; n_planning; the grid; the 3-point guard on integer counts;
     the max over pairs; the caveat boundary; p < alpha strict; a spy on every pre-registered call; a spy on
     the index draw; U8b the draw equals default_rng(42).choice(d + delta) on this numpy; the analytic rule;
     U10 the refused d (exact zeros only, non-finite, too short)
  D  lane 6 (d) d1-d3 on exactly symmetric synthetic d (the erratum; see the lane report): d1 30% zeros, d2 no
     zeros with SD 0.10; D1e the tie counts; D2b non-centring (MDE_W(d2 + 0.003) = MDE_W(d2) - 0.003)
  B  the band round trip through its reader src/training/sweep_select.py dl27_band
  C  the CLI on synthetic VAL artifacts: exit 0, the two files, the DL line, the entry's keys and every headline
     number recomputed from the entry with literals, the report validator, the report layer's "pending" for an
     uncommitted entry, byte-identical reruns, the pair orientation; C6 the same numbers from an in-process run
  R  refusals (exit 2, nothing written): a test path in each of the seven path flags, a path resolving to a
     test location, existing outputs, missing inputs, provenance mismatches (each gate), the lane 6 (f) AM-5
     STOP (same K, other ids; and another K), out-dir inside the repository, mode flags in both directions,
     the run-of-record binding (HEAD, the code files, a clean tree), the same run twice (directories, run_ids,
     checkpoints), --expect-k-val, best.json content, the rule file's band, the environment, the MANIFEST
     pin, two runs that agree image for image
  S  computed STOPs (exit 1, nothing written): the re-derivation of item 3(a)'s field (value; class count); no
     qualifying delta; the report validator; the band read back through its reader
  M  mutations of one rule each, every one killed by the named check

Run (TMPDIR must be set; its realpath must contain no "test"):
    env TMPDIR=/tmp/s4scratch PYTHONPATH=<repo> python -B scripts/smoke_mde.py
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
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import stats_fixtures as F  # noqa: E402
from scripts import compare_eval_artifacts as CEA  # noqa: E402
from scripts import mde_entry as ME  # noqa: E402
from src.eval.artifacts import EVAL_RUNTIME_VERSION, MANIFEST_NAME  # noqa: E402
from src.eval.evaluate import Condition  # noqa: E402
from src.stats import mde as M  # noqa: E402
from src.stats import report as RP  # noqa: E402
from src.stats.val_artifacts import load_val_artifact  # noqa: E402
from src.training import sweep_select as SS  # noqa: E402

N, K = 846, 6
SEEDS = ("42", "43", "44")
GEN_UTC = "2026-10-09T00:00:00Z"
STAMP = "20261009T000000Z"
ENTRY = f"synthetic_mde_entry_{STAMP}.json"
BAND = "synthetic_dl27_band.json"
BEST = {"42": CEA.DL17_REFERENCE_MIOU, "43": 0.34126096963882446, "44": 0.3548465073108673}
PLAN_S, PLAN_BAND = 0.011045744504733232, 0.015621041685101825        # FACTS 3, smoke fixture values only
KWARGS = {"zero_method": "pratt", "alternative": "greater", "correction": True, "method": "approx"}
LANE6C_KEYS = ("artifacts", "n_included", "K_excluded", "n_planning", "pairs", "mde_w", "sd_delta", "dz_mde",
               "mde_t", "mde_t_range", "tau_p", "power_caveat", "scipy_version", "seed", "git_commit",
               "decision_log_line")
PAIR_KEYS = ("n", "mean", "sd", "share_ties", "power_curve", "mde_w", "m", "n_zero")
PAIR_ORDER = ("43-42", "44-42", "44-43")
#: AM-17 item 3(e), verbatim (docs/PREREGISTRATION_AMENDMENTS.md:410-412): a literal, not mde_entry.LABEL
LABEL_3E = ("planning proxy — seed-pair differences approximate noise, not the spread of between-recipe "
            "differences, so the true MDE may be larger. Reported with the results; it changes no test or "
            "decision.")
RESULTS: list = []
NORMAL: dict = {}


class Ctx:
    """Shared fixtures and cached results."""


CTX = Ctx()


def record(cid: str, desc: str, ok: bool, detail="") -> None:
    RESULTS.append((cid, desc, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} {desc}" + ("" if ok else f"  -- {str(detail)[:900]}"), flush=True)


@contextlib.contextmanager
def patched(*triples):
    """Set (object, attribute, value) for the duration; restore afterwards."""
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in triples]
    try:
        for obj, name, value in triples:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


@contextlib.contextmanager
def no_filesystem():
    """Every filesystem probe raises (and is recorded): a refusal must come from the path string alone."""
    calls = []

    def make(name):
        def blocked(*a, **k):
            calls.append((name, str(a[0]) if a else ""))
            raise AssertionError(f"filesystem call {name} before the test-path refusal")
        return blocked
    targets = [(os, "stat"), (os, "lstat"), (os, "scandir"), (os, "listdir"), (os, "mkdir"), (os.path, "realpath"),
               (builtins, "open"), (io, "open"), (os, "open"), (os, "access")]
    with patched(*[(obj, name, make(name)) for obj, name in targets]):
        yield calls


def tripwire(*a, **k):
    """Replaces td.write_files_exclusive where a regressed gate could otherwise write."""
    raise AssertionError("write attempted")


def run_cli(argv) -> tuple[int, str, str]:
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        code = ME.main([str(x) for x in argv])
    return code, so.getvalue(), se.getvalue()


def run_subprocess(argv) -> tuple[int, str, str]:
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "mde_entry.py"), *map(str, argv)],
                       cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode, p.stdout, p.stderr


def fresh_out(tag: str) -> Path:
    CTX.counter = getattr(CTX, "counter", 0) + 1
    return CTX.root / "outs" / f"{tag}_{CTX.counter:03d}"


def listing(d: Path) -> list:
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------
def runtime_block(s: str) -> dict:
    return {"eval_runtime_version": EVAL_RUNTIME_VERSION, "model_device": "cuda:0",
            "model_tensor_devices": ["cuda:0"], "input_devices": ["cuda:0"], "forward_batches": 53,
            "batch_size": 16, "num_workers": 0, "determinism_policy_applied": True,
            "cuda_initialized_before_policy": False, "inherited_cublas_workspace_config": None,
            "gpu_name": "NVIDIA A40", "gpu_capability": "8.6", "cudnn_version": 8902, "torch_cuda": "12.1",
            "torch_num_threads": 1, "pillow": "12.3.0", "checkpoint_iteration": 80000,
            "checkpoint_best_val_miou_all_class": BEST[s], "image_digest": CEA.DL17_IMAGE_DIGEST,
            "eval_warnings": [], "eval_warnings_truncated": False, "nondeterministic_alert_count": 0}


def dress(j: dict, s: str) -> dict:
    """The DL-17 B66 pins and an eval_runtime block, as the real re-scores carry them."""
    j["run"]["env"]["device"] = "cuda"
    j["run"]["checkpoint_path"] = f"/workspace/e1_ckpts_s{s}/e1_student_best_iter80000.pt"
    j["run"]["eval_runtime"] = runtime_block(s)
    if s == "42":
        j["dataset_level"]["all_class_miou"] = CEA.DL17_REFERENCE_MIOU
        gt = j["per_class"]["gt_support"]
        gt[0] += CEA.DL17_GT_SUPPORT - sum(gt)
    return j


def write_seed(repo: Path, dest: Path, truth, s: str, quality: float, tag: str) -> Path:
    scored = F.score(truth, F.predict(truth, quality=quality, seed=F.int_seed(tag, s)), Condition())
    ck = CEA.DL17_CHECKPOINT_SHA256 if s == "42" else hashlib.sha256(f"{tag} e1 seed {s}".encode()).hexdigest()
    d = F.write_fixture(dest, scored, N, stage="E1", condition=Condition(), split="val", status="provisional",
                        repo_root=repo, run_id=f"{tag}-e1-s{s}", checkpoint_sha256=ck)
    return F.edit_summary(d, lambda j: dress(j, s))


def build_set(name: str, qualities) -> tuple[dict, Path]:
    repo = F.clean_git_fixture(CTX.root / name)
    truth = F.make_truth(N, K, F.int_seed(name, "truth"))
    arts = {s: write_seed(repo, repo / f"runs/val/e1_s{s}", truth, s, q, name) for s, q in zip(SEEDS, qualities)}
    return arts, repo


def write_best(dest: Path, s: str, value=None, extra: dict | None = None) -> Path:
    doc = {"best_ckpt": f"/workspace/e1_ckpts_s{s}/e1_student_best_iter80000.pt",
           "best_val_miou_all_class": BEST[s] if value is None else value}
    doc.update(extra or {})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return dest


def manifest_sha(d: Path) -> str:
    return hashlib.sha256((Path(d) / MANIFEST_NAME).read_bytes()).hexdigest()


def variant(src: Path, tag: str, fn) -> Path:
    """A copy of an artifact with its summary.json edited and the manifest rehashed."""
    dest = CTX.root / "variants" / tag
    shutil.copytree(src, dest)
    return F.edit_summary(dest, fn)


def argv_for(out: Path, *, arts=None, best=None, synthetic=True, extra=(), drop=()) -> list:
    arts = arts or CTX.A
    best = best or CTX.best
    a = {"--s42": arts["42"], "--s43": arts["43"], "--s44": arts["44"], "--best-json-42": best["42"],
         "--best-json-43": best["43"], "--best-json-44": best["44"], "--out-dir": out}
    argv = []
    for k, v in a.items():
        if k not in drop:
            argv += [k, v]
    if synthetic:
        argv += ["--synthetic-inputs", "--generated-utc", GEN_UTC]
    return argv + list(extra)


def setup_fixtures() -> None:
    CTX.A, CTX.repo_A = build_set("fxA", (0.70, 0.70, 0.70))
    CTX.Bset, _ = build_set("fxB", (0.70, 0.76, 0.70))
    CTX.best = {s: write_best(CTX.root / "best" / f"s{s}" / "best.json", s) for s in SEEDS}
    CTX.pins = {s: manifest_sha(CTX.A[s]) for s in SEEDS}
    alt_truth = F.make_truth(N, K, F.int_seed("fxA-alt", "truth"))              # the same K, other image ids
    CTX.alt44 = write_seed(CTX.repo_A, CTX.repo_A / "runs/val/e1_s44_alt", alt_truth, "44", 0.70, "fxA-alt")
    alt2_truth = F.make_truth(N, K + 2, F.int_seed("fxA-alt2", "truth"))        # another K
    CTX.alt44b = write_seed(CTX.repo_A, CTX.repo_A / "runs/val/e1_s44_alt2", alt2_truth, "44", 0.70, "fxA-alt2")


# --------------------------------------------------------------------------------------------------
# U: units of src/stats/mde.py
# --------------------------------------------------------------------------------------------------
def small_d(m=13, seed=5) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.01, 0.1, m)


def U0():
    want = {"B": 2000, "RNG_SEED": 42, "ALPHA": 0.00625, "GUARD_POINTS": 3, "POWER_MIN": 0.80,
            "Z_ALPHA": 2.4977, "Z_POWER": 0.8416, "EFFICIENCY": (1.023, 1.076), "N_TEST": 1561, "N_VAL": 846,
            "WILCOXON_KWARGS": KWARGS}
    got = {k: getattr(M, k) for k in want}
    grid = len(M.GRID) == 51 and all(M.GRID[k] == k / 1000 for k in range(51))
    ok = got == want and grid and M.TAU_P == RP.TAU_P == 0.010 and M.min_count(2000) == 1600
    return ok, {"got": got, "grid": grid, "tau_p": M.TAU_P, "min_count(2000)": M.min_count(2000)}


def U1a():
    return M.n_planning(846) == 1561, M.n_planning(846)


def U1b():
    return M.n_planning(800) == 1476, M.n_planning(800)


def U1c():
    return M.n_planning(423) == 780, M.n_planning(423)


def U1d():
    try:
        M.n_planning(0)
    except M.MdeError:
        return True, "refused"
    return False, "0 accepted"


def U2():
    ok = len(M.GRID) == 51 and M.GRID[0] == 0.0 and M.GRID[50] == 0.05 and M.GRID[9] == 0.009
    return ok, M.GRID[:3] + M.GRID[-2:]


def U3a():
    c = [0] * 51
    c[5] = c[6] = c[7] = 1600
    return M.guard_index(c, M.min_count(2000)) == 5, M.guard_index(c, M.min_count(2000))


def U3b():
    c = [0] * 51
    c[5], c[6] = 1600, 1599
    c[7] = c[8] = c[9] = 2000
    return M.guard_index(c, M.min_count(2000)) == 7, M.guard_index(c, M.min_count(2000))


def U3c():
    c = [1599] * 51
    return M.guard_index(c, M.min_count(2000)) is None, M.guard_index(c, M.min_count(2000))


def U3d():
    c = [0] * 48 + [2000] * 3
    c2 = [0] * 49 + [2000] * 2
    a, b = M.guard_index(c, 1600), M.guard_index(c2, 1600)
    return a == 48 and b is None, (a, b)


def U4():
    pairs = {"43-42": {"mde_w": 0.01}, "44-42": {"mde_w": 0.02}, "44-43": {"mde_w": 0.02}}
    got = M.overall(pairs)
    return got == (0.02, "44-42"), got


def U5a():
    doc = {"mde_w": 0.01, "tau_p": 0.010, "power_caveat": M.power_caveat(0.01)}
    return M.power_caveat(0.010) is False and RP.validate_mde_entry(doc) == (), doc


def U5b():
    doc = {"mde_w": 0.011, "tau_p": 0.010, "power_caveat": M.power_caveat(0.011)}
    return M.power_caveat(0.011) is True and RP.validate_mde_entry(doc) == (), doc


def U6():
    return M.rejects(0.00625) is False and M.rejects(0.006249999999) is True, (M.rejects(0.00625),)


def U7():
    import scipy.stats as st
    real, calls = st.wilcoxon, []

    def spy(x, *a, **k):
        calls.append((np.asarray(x).ndim, np.asarray(x).dtype, len(x), a, dict(k)))
        return real(x, *a, **k)
    with patched((st, "wilcoxon", spy)):
        M.pair_summary(small_d(), 11, b=4, grid=(0.0, 0.01))
    ok = len(calls) == 8 and all(c[0] == 1 and c[1] == np.float64 and c[2] == 11 and c[3] == () and c[4] == KWARGS
                                 for c in calls)
    return ok, calls[:2] + [len(calls)]


class _GenSpy:
    def __init__(self, gen, log):
        self._gen, self._log = gen, log

    def integers(self, *a, **k):
        self._log.append(("integers", a, k))
        return self._gen.integers(*a, **k)


def U8():
    real, log = np.random.default_rng, []

    def spy(seed=None):
        log.append(("default_rng", seed))
        return _GenSpy(real(seed), log)
    d = small_d()
    with patched((np.random, "default_rng", spy)):
        M.pair_summary(d, 11, b=4, grid=(0.0, 0.01, 0.02))
    ok = log == [("default_rng", 42), ("integers", (0, 13), {"size": (4, 11), "dtype": np.int64})]
    return ok, log


def U8b():
    d = small_d()
    out = []
    for delta in (0.0, 0.001, 0.05):
        a = np.random.default_rng(42).choice(d + delta, size=(7, 11), replace=True)
        b = d[M.resample_indices(13, 11, b=7)] + delta
        out.append(np.array_equal(a, b) and a.tobytes() == b.tobytes())
    return all(out) and np.__version__ == "1.26.4", {"equal": out, "numpy": np.__version__}


def U10():
    got = {}
    for name, d in (("zeros", np.zeros(5)), ("nan", np.array([0.1, np.nan, 0.2])), ("one", np.array([0.1]))):
        try:
            M.check_d(d)
            got[name] = "accepted"
        except M.MdeError as e:
            got[name] = str(e)
    ok = ("exact zero" in got["zeros"] and "non-finite" in got["nan"] and "at least two" in got["one"]
          and M.check_d([0.0, 0.1]).dtype == np.float64)
    return ok, got


def U9():
    d = small_d()
    summ = M.pair_summary(d, 11, b=4, grid=(0.0,))
    sd = float(np.std(d, ddof=1))
    ana = M.analytic([0.1, 0.2, 0.15], 1561)
    dz = (2.4977 + 0.8416) / math.sqrt(1561)
    ok = (summ["sd"] == sd and ana["sd_delta"] == 0.2 and ana["dz_mde"] == dz and ana["mde_t"] == dz * 0.2
          and ana["mde_t_range"] == [dz * 0.2 * 1.023, dz * 0.2 * 1.076] and summ["m"] == 13 and summ["n"] == 11)
    return ok, {"sd": summ["sd"], "want": sd, "analytic": ana}


# --------------------------------------------------------------------------------------------------
# D: lane 6 (d) on exactly symmetric synthetic d
# --------------------------------------------------------------------------------------------------
def make_d1() -> np.ndarray:
    half = np.random.default_rng(20261008).normal(0.0, 0.12, 296)
    return np.concatenate([np.zeros(254), half, -half])              # 846: 254 exact zeros, 592 = +-x


def make_d2() -> np.ndarray:
    half = np.random.default_rng(20261009).normal(0.0, 0.10, 423)
    d = np.concatenate([half, -half])
    return d * (0.10 / float(np.std(d, ddof=1)))                     # SD (ddof 1) normalised to 0.10


def setup_d() -> None:
    t0 = time.time()
    CTX.d1, CTX.d2 = make_d1(), make_d2()
    CTX.s1 = M.pair_summary(CTX.d1, 1561)
    CTX.s2 = M.pair_summary(CTX.d2, 1561)
    print(f"  (D: two full curves in {time.time() - t0:.0f} s)", flush=True)


def D1a():
    dz_smoke = (2.4977 + 0.8416) / math.sqrt(1561)
    sd = float(np.std(CTX.d1, ddof=1))
    ana = M.analytic([CTX.s1["sd"]], 1561)
    ok = abs(ana["mde_t"] - dz_smoke * sd) <= 1e-9 and abs(ana["dz_mde"] - 0.0845) < 5e-5
    return ok, {"mde_t": ana["mde_t"], "dz_smoke*sd": dz_smoke * sd, "dz_mde": ana["dz_mde"]}


def D1b():
    count = M.rejection_counts(CTX.d1, M.resample_indices(N, 1561), (0.0,))[0]
    power = count / M.B
    return 0.001 <= power <= 0.015, {"power_at_0": power, "count": count}


def D1c():
    k, c = CTX.s1["mde_w_index"], CTX.s1["rejections"]
    ok = k is not None and all(c[j + 1] >= c[j] for j in range(k, len(c) - 1))
    return ok, {"mde_w_index": k, "rejections": c}


def D1d():
    k = CTX.s1["mde_w_index"]
    return k is not None and CTX.s1["rejections"][k] >= 1600, {"mde_w": CTX.s1["mde_w"],
                                                               "count": CTX.s1["rejections"][k] if k is not None else None}


def D1e():
    return CTX.s1["n_zero"] == 254 and CTX.s1["share_ties"] == 254 / 846, (CTX.s1["n_zero"], CTX.s1["share_ties"])


def D2():
    ana = M.analytic([CTX.s2["sd"]], 1561)
    ratio = CTX.s2["mde_w"] / ana["mde_t"] if CTX.s2["mde_w"] is not None else None
    powers = {f"{M.GRID[k]:.3f}": CTX.s2["power_curve"][k][1] for k in (8, 9, 10)}
    CTX.d2_report = {"mde_w": CTX.s2["mde_w"], "mde_t": ana["mde_t"], "ratio": ratio, "power": powers,
                     "sd": CTX.s2["sd"], "expected": "MDE_W 0.009, ratio 1.065 (Annex A row 31)"}
    print(f"  D2 realised: {CTX.d2_report}", flush=True)
    return ratio is not None and 0.95 <= ratio <= 1.15, CTX.d2_report


def D2b():
    shifted = M.pair_summary(CTX.d2 + 0.003, 1561)
    ok = (shifted["mde_w"] is not None and CTX.s2["mde_w"] is not None
          and abs(shifted["mde_w"] - (CTX.s2["mde_w"] - 0.003)) <= 0.001 + 1e-12)
    return ok, {"mde_w(d2+0.003)": shifted["mde_w"], "mde_w(d2)": CTX.s2["mde_w"]}


def D3a():
    again = M.pair_summary(CTX.d2, 1561)
    a, b = json.dumps(again, sort_keys=True).encode(), json.dumps(CTX.s2, sort_keys=True).encode()
    return a == b, hashlib.sha256(a).hexdigest()


def D3b():
    other = M.pair_summary(CTX.d2, 1561, seed=43)
    ok = other["mde_w"] is not None and abs(other["mde_w"] - CTX.s2["mde_w"]) <= 0.001 + 1e-12
    return ok, {"seed43": other["mde_w"], "seed42": CTX.s2["mde_w"]}


# --------------------------------------------------------------------------------------------------
# B: the band through its reader
# --------------------------------------------------------------------------------------------------
def floor() -> float:
    return ME.band_floor()


def B1():
    doc, data = ME.build_band(BEST, floor())
    s, band, _ = SS.dl27_band(json.loads(data), floor())
    return repr(s) == repr(doc["s"]) and repr(band) == repr(doc["band"]), (s, band, doc)


def _refused_code(doc) -> str | None:
    try:
        SS.dl27_band(doc, floor())
    except SS.SelectionRefused as e:
        return e.code
    return None


def B2():
    doc, data = ME.build_band(BEST, floor())
    d = json.loads(data)
    d["s"] += 1e-9
    code = _refused_code(d)
    return code == "band_s_mismatch", code


def B3():
    doc, data = ME.build_band(BEST, floor())
    d = json.loads(data)
    d["band"] += 1e-9
    code = _refused_code(d)
    return code == "band_mismatch", code


def B4():
    doc, _ = ME.build_band(BEST, floor())
    return doc["s"] == PLAN_S and doc["band"] == PLAN_BAND, doc


def B5():
    near = {"42": 0.36, "43": 0.3601, "44": 0.3602}
    doc, data = ME.build_band(near, floor())
    _, band, trace = SS.dl27_band(json.loads(data), floor())
    return doc["band"] == 0.005 and band == 0.005, (doc, trace)


def B6():
    doc, data = ME.build_band(BEST, floor())
    back = json.loads(data)
    ok = (sorted(back) == ["band", "e1_best_val", "s"] and sorted(back["e1_best_val"]) == list(SEEDS)
          and data.endswith(b"\n") and back == doc)
    return ok, sorted(back)


# --------------------------------------------------------------------------------------------------
# C: the CLI on synthetic VAL artifacts
# --------------------------------------------------------------------------------------------------
def pin_args() -> list:
    out = []
    for s in SEEDS:
        out += [f"--expect-manifest-sha256-{s}", CTX.pins[s]]
    return out


def setup_c() -> None:
    t0 = time.time()
    CTX.c1_out = fresh_out("c1")
    CTX.c1 = run_subprocess(argv_for(CTX.c1_out, extra=pin_args()))
    print(f"  (C1: subprocess CLI in {time.time() - t0:.0f} s, exit {CTX.c1[0]})", flush=True)
    print("  " + CTX.c1[1].replace("\n", "\n  ").rstrip(), flush=True)


def dl_line(so: str):
    """The text of the CLI's printed 'DL line: ...' (None if absent)."""
    return next((x[len("DL line: "):] for x in so.splitlines() if x.startswith("DL line: ")), None)


def C1():
    code, so, se = CTX.c1
    files = listing(CTX.c1_out)
    if code != 0 or files != sorted([ENTRY, BAND]):
        return False, {"exit": code, "files": files, "stderr": se[-600:]}
    shas = {n: hashlib.sha256((CTX.c1_out / n).read_bytes()).hexdigest() for n in files}
    entry = json.loads((CTX.c1_out / ENTRY).read_text(encoding="utf-8"))
    dl, arts = dl_line(so), entry["artifacts"]
    need = [f"{ENTRY} sha256 {shas[ENTRY]}", f"{BAND} sha256 {shas[BAND]}", "share of exact-zero differences",
            "n_zero", "so the three guard points share rows", "power_caveat", "label: planning proxy (item 3(e))",
            "lane 6 (f): exact ties deflate MDE_W under the constant-shift null"]
    need += [arts[s][k] for s in SEEDS for k in ("run_id", "checkpoint_sha256")]
    need += [arts[s]["sha256s"][MANIFEST_NAME] for s in SEEDS]
    need += dl_fields(entry, ENTRY, shas[ENTRY], f"{BAND} sha256 {shas[BAND]}", " (None).")
    missing = [x for x in need if dl is None or x not in dl]
    stored = dl is not None and entry["decision_log_line"] == dl.replace(f" sha256 {shas[ENTRY]}", "", 1)
    written = all(f"written: {n} sha256 {h}" in so for n, h in shas.items())
    return not missing and stored and written, {"missing": missing, "stored_equals_printed": stored,
                                                "written_lines": written, "dl": dl}


def guard_k(counts, need=1600):
    """The smoke's own 3-point guard on integer counts (not mde.guard_index)."""
    for k in range(len(counts) - 2):
        if counts[k] >= need and counts[k + 1] >= need and counts[k + 2] >= need:
            return k
    return None


def entry_values(entry: dict) -> list:
    """Every headline number of the entry against its recomputation from the entry itself, with literals."""
    pairs, n, bad = entry["pairs"], entry["n_planning"], []
    for name in PAIR_ORDER:
        p = pairs[name]
        k = guard_k(p["rejections"])
        if not (p["n"] == n == entry["resampling"]["size"] and p["m"] == entry["n_included"]):
            bad.append([name, "n and m", p["n"], p["m"]])
        if not (len(p["rejections"]) == len(p["power_curve"]) == 51 and p["power_min_count"] == 1600):
            bad.append([name, "51 grid points, power_min_count 1600"])
        if p["power_curve"] != [[j / 1000, c / 2000] for j, c in enumerate(p["rejections"])]:
            bad.append([name, "power_curve"])
        if k is None or p["mde_w"] != k / 1000 or p["mde_w_index"] != k:
            bad.append([name, "mde_w", p["mde_w"], k])
    best = PAIR_ORDER[0]
    for name in PAIR_ORDER[1:]:
        if pairs[name]["mde_w"] > pairs[best]["mde_w"]:
            best = name                                                    # the first pair wins a tie
    sd = max(pairs[name]["sd"] for name in PAIR_ORDER)
    dz = (2.4977 + 0.8416) / math.sqrt(n)
    want = {"mde_w": pairs[best]["mde_w"], "mde_w_pair": best, "sd_delta": sd, "dz_mde": dz, "mde_t": dz * sd,
            "mde_t_range": [dz * sd * 1.023, dz * sd * 1.076], "power_caveat": pairs[best]["mde_w"] > 0.010,
            "n_planning": round(Fraction(1561 * entry["n_included"], 846)), "label": LABEL_3E}
    bad += [[key, entry[key], value] for key, value in want.items() if entry[key] != value]
    if entry["resampling"]["B"] != 2000:
        bad.append(["resampling.B", entry["resampling"]["B"]])
    return bad


def C2():
    entry = json.loads((CTX.c1_out / ENTRY).read_text(encoding="utf-8"))
    band_bytes = (CTX.c1_out / BAND).read_bytes()
    pairs_ok = sorted(entry["pairs"]) == ["43-42", "44-42", "44-43"] and all(
        all(k in entry["pairs"][p] for k in PAIR_KEYS) for p in entry["pairs"])
    arts_ok = sorted(entry["artifacts"]) == list(SEEDS) and all(
        "run_id" in entry["artifacts"][s] and "sha256s" in entry["artifacts"][s] for s in SEEDS)
    pins = [c for c in entry["provenance_checks"] if "MANIFEST.sha256" in c[0]]
    res = entry["resampling"]
    values = entry_values(entry)
    ok = (all(k in entry for k in LANE6C_KEYS) and pairs_ok and arts_ok and RP.validate_mde_entry(entry) == ()
          and all(c[1] is True for c in entry["provenance_checks"]) and len(pins) == 3 and not values
          and entry["tau_p"] == 0.010 and entry["dl27_band"]["sha256"] == hashlib.sha256(band_bytes).hexdigest()
          and res["drawn_once_per_pair"] is True and res["guard_points_share_rows"] is True
          and res["numpy_version"] == np.__version__ and entry["artifacts"]["42"]["dl17_delta"] == 0.0
          and all(entry["artifacts"][s]["checkpoint_iteration"] == 80000
                  and entry["artifacts"][s]["checkpoint_path_basename"] == "e1_student_best_iter80000.pt"
                  and "governed_paths_clean" in entry["artifacts"][s] for s in SEEDS)
          and entry["K_excluded"] == K and entry["seed"] == 42 and entry["artifact_status"] == "smoke")
    return ok, {"values": values, **{k: entry.get(k) for k in ("mde_w", "mde_w_pair", "n_included", "n_planning")}}


def C6():
    out = fresh_out("c6")
    with patched((M, "rejection_counts", fast_counts)):
        code, so, se = run_cli(argv_for(out))
    if code != 0 or not (out / ENTRY).exists():
        return False, {"exit": code, "stderr": se[-500:]}
    bad = entry_values(json.loads((out / ENTRY).read_text(encoding="utf-8")))
    return not bad, bad


def C3():
    doc, status, problems = RP.load_mde_entry(CTX.c1_out / ENTRY)
    mde = dict(status, **RP.mde_fields(doc), problems=list(problems))
    got = RP.committed_caveat(mde)
    return got == ("pending", None) and status["inside_repository"] is False, got


def setup_c4() -> None:
    t0 = time.time()
    CTX.c4_out = fresh_out("c4")
    CTX.c4 = run_cli(argv_for(CTX.c4_out, extra=pin_args()))
    print(f"  (C4: in-process CLI in {time.time() - t0:.0f} s, exit {CTX.c4[0]})", flush=True)


def C4a():
    a, b = (CTX.c1_out / ENTRY).read_bytes(), (CTX.c4_out / ENTRY).read_bytes() if (CTX.c4_out / ENTRY).exists() else b""
    return CTX.c4[0] == 0 and a == b, {"exit": CTX.c4[0], "sha_c1": hashlib.sha256(a).hexdigest(),
                                       "sha_c4": hashlib.sha256(b).hexdigest(), "stderr": CTX.c4[2][-300:]}


def C4b():
    a, b = (CTX.c1_out / BAND).read_bytes(), (CTX.c4_out / BAND).read_bytes() if (CTX.c4_out / BAND).exists() else b""
    return CTX.c4[0] == 0 and a == b, (hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())


def C5():
    args = ME.build_parser().parse_args([str(x) for x in argv_for(fresh_out("c5"), arts=CTX.Bset)])
    arts = ME.load_artifacts(args, True)
    pv = ME.pair_vectors(arts)
    means = {k: float(np.mean(v.delta)) for k, v in pv.items()}
    return means["43-42"] > 0 and means["44-43"] < 0, means


# --------------------------------------------------------------------------------------------------
# R and S: refusals (exit 2) and computed STOPs (exit 1), nothing written
# --------------------------------------------------------------------------------------------------
def expect(argv, code_want: int, needle: str, out: Path, keep=()) -> tuple[bool, dict]:
    code, so, se = run_cli(argv)
    left = listing(out)
    ok = code == code_want and needle in se and left == sorted(keep)
    return ok, {"exit": code, "stderr": se.strip()[-500:], "left": left}


def make_r1(flag: str):
    def check():
        bad = CTX.root / "never" / f"never_TeSt_{flag}"
        out = fresh_out("r1")
        argv = argv_for(bad if flag == "out_dir" else out)
        if flag != "out_dir":
            key = "--" + flag.replace("_", "-")
            i = argv.index(key)
            argv[i + 1] = bad
        with no_filesystem() as calls:
            code, so, se = run_cli(argv)
        ok = code == 2 and "test" in se.lower() and not calls and not os.path.lexists(bad) and not out.exists()
        return ok, {"exit": code, "calls": calls, "stderr": se.strip()[-300:]}
    return check


def R2a():
    out = fresh_out("r2a")
    out.mkdir(parents=True)
    (out / BAND).write_bytes(b"pre-existing\n")
    ok, det = expect(argv_for(out), 2, "already holds", out, keep=[BAND])
    return ok and (out / BAND).read_bytes() == b"pre-existing\n", det


def R2b():
    out = fresh_out("r2b")
    out.mkdir(parents=True)
    (out / ENTRY).write_bytes(b"pre-existing\n")
    ok, det = expect(argv_for(out), 2, "already holds", out, keep=[ENTRY])
    return ok and (out / ENTRY).read_bytes() == b"pre-existing\n", det


def R3a():
    out = fresh_out("r3a")
    arts = dict(CTX.A, **{"43": CTX.root / "absent" / "e1_s43"})
    return expect(argv_for(out, arts=arts), 2, "does not exist", out)


def R3b():
    out = fresh_out("r3b")
    best = dict(CTX.best, **{"44": CTX.root / "absent" / "best.json"})
    return expect(argv_for(out, best=best), 2, "does not exist", out)


def _with(s: str, path: Path) -> dict:
    return dict(CTX.A, **{s: path})


def R4a():
    out = fresh_out("r4a")
    return expect(argv_for(out, arts=_with("43", CTX.v_commit)), 2, "one evaluate_model.py commit", out)


def R4b():
    out = fresh_out("r4b")
    return expect(argv_for(out, arts=_with("44", CTX.v_gpu)), 2, "one device class", out)


def R4c():
    out = fresh_out("r4c")
    return expect(argv_for(out, arts=_with("44", CTX.v_digest)), 2, "one non-null image digest", out)


def R4d():
    out = fresh_out("r4d")
    return expect(argv_for(out, arts=_with("42", CTX.v_batch)), 2, "valid DL-17 B66 re-score", out)


def R4e():
    out = fresh_out("r4e")
    return expect(argv_for(out, arts=_with("42", CTX.v_miou)), 2, "DL17_BAND", out)


def R4f():
    out = fresh_out("r4f")
    best = dict(CTX.best, **{"43": write_best(CTX.root / "best_bad" / "s43" / "best.json", "43", value=0.34)})
    return expect(argv_for(out, best=best), 2, "checkpoint_best_val_miou_all_class", out)


def R5():
    out = fresh_out("r5")
    a, b = (load_val_artifact(Path(p), label=t).run.am5 for p, t in ((CTX.A["44"], "a"), (CTX.alt44, "b")))
    same_k, other_ids = a.excluded_count == b.excluded_count, a.excluded_ids != b.excluded_ids
    ok, det = expect(argv_for(out, arts=_with("44", CTX.alt44)), 2, "STOP (lane 6 (f))", out)
    return ok and same_k and other_ids, dict(det, same_k=same_k, other_ids=other_ids)


def R5b():
    out = fresh_out("r5b")
    return expect(argv_for(out, arts=_with("44", CTX.alt44b)), 2, "STOP (lane 6 (f))", out)


def R6():
    inside = REPO / "s4_smoke_out_inside_never"
    with patched((ME.td, "write_files_exclusive", tripwire), (M, "rejection_counts", fast_counts)):
        ok, det = expect(argv_for(inside), 2, "outside the repository", inside)
    return ok and not inside.exists(), det


FLAG_GATE = "a real run requires --script-commit and --script-commit-dl-id"


def R7a():
    out = fresh_out("r7a")
    return expect(argv_for(out, synthetic=False), 2, FLAG_GATE, out)


def R7d():
    out = fresh_out("r7d")
    return expect(argv_for(out, synthetic=False, extra=["--script-commit", "0" * 40]), 2, FLAG_GATE, out)


def R7e():
    import src.eval.adapters as AD
    out = fresh_out("r7e")
    with patched((AD, "DATASET_NAME", F.FIXTURE_DATASET_NAME), (M, "rejection_counts", fast_counts)):
        return expect(argv_for(out), 2, "--synthetic-inputs is True", out)


def R7b():
    out = fresh_out("r7b")
    extra = ["--script-commit", "0" * 40, "--script-commit-dl-id", "DL-99", "--generated-utc", GEN_UTC]
    return expect(argv_for(out, synthetic=False, extra=extra), 2, "--generated-utc is refused", out)


def _real_extra() -> list:
    return ["--script-commit", "0" * 40, "--script-commit-dl-id", "DL-99"]


def R7c():
    out = fresh_out("r7c")
    with patched((ME, "commit_binding", lambda args: {"head": "0" * 40}),
                 (ME, "environment_check", lambda synthetic: {"matches_pinned": True})):
        return expect(argv_for(out, synthetic=False, extra=_real_extra()), 2, "--synthetic-inputs is False", out)


def R8():
    out = fresh_out("r8")
    return expect(argv_for(out, arts=_with("43", CTX.A["42"])), 2, "three runs", out)


def R8b():
    out = fresh_out("r8b")
    return expect(argv_for(out, arts=_with("43", CTX.v_copy42)), 2, "their run_ids", out)


def R8c():
    out = fresh_out("r8c")
    return expect(argv_for(out, arts=_with("43", CTX.v_copy42_id)), 2, "their checkpoints", out)


def fast_expect(argv, code_want: int, needle: str, out: Path) -> tuple[bool, dict]:
    """expect() with fast counts, so a regressed gate never runs the full MDE."""
    with patched((M, "rejection_counts", fast_counts)):
        return expect(argv, code_want, needle, out)


def R4g():
    out = fresh_out("r4g")
    return fast_expect(argv_for(out, arts=_with("44", CTX.v_stage)), 2, "expected stage/role/precision", out)


def R4h():
    out = fresh_out("r4h")
    return fast_expect(argv_for(out, arts=_with("44", CTX.v_precision)), 2, "expected stage/role/precision", out)


def R4i():
    out = fresh_out("r4i")
    return fast_expect(argv_for(out, arts=_with("43", CTX.v_inputs)), 2, "inputs on the model device", out)


def R4j():
    out = fresh_out("r4j")
    return fast_expect(argv_for(out, arts=_with("44", CTX.v_policy)), 2, "the determinism policy applied", out)


def R9():
    out = fresh_out("r9")
    return expect(argv_for(out, extra=["--expect-k-val", str(K + 1)]), 2, "K_val is", out)


def R10():
    out = fresh_out("r10")
    best = dict(CTX.best, **{"42": write_best(CTX.root / "best_extra" / "s42" / "best.json", "42",
                                              extra={"note": "extra"})})
    return expect(argv_for(out, best=best), 2, "keys", out)


def R10b():
    out = fresh_out("r10b")
    best = dict(CTX.best, **{"42": write_best(CTX.root / "best_int" / "s42" / "best.json", "42",
                                              extra={"best_ckpt": 7})})
    return fast_expect(argv_for(out, best=best), 2, "is not a path", out)


def R11():
    out = fresh_out("r11")
    with patched((ME, "commit_binding", lambda args: {"head": "0" * 40}),
                 (ME, "software_environment_block", lambda: {"matches_pinned": False, "observed": {}, "pinned": {}})):
        return expect(argv_for(out, synthetic=False, extra=_real_extra()), 2, "is not the pinned", out)


def R12():
    out = fresh_out("r12")
    extra = pin_args()
    extra[extra.index("--expect-manifest-sha256-43") + 1] = "0" * 64
    code, so, se = run_cli(argv_for(out, extra=extra))
    ok = code == 2 and "MANIFEST.sha256" in se and CTX.pins["43"] in se and listing(out) == []
    return ok, {"exit": code, "observed_hash_in_message": CTX.pins["43"] in se, "stderr": se.strip()[-500:]}


def fake_git(head, status=""):
    """td._git for the binding cases: no git process runs."""
    def _git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return None if head is None else head + "\n"
        return status if args[:1] == ("status",) else None
    return _git


def binding_case(tag: str, head, status: str, needle: str, extra_patches=()) -> tuple[bool, dict]:
    out = fresh_out(tag)
    with patched((ME.td, "_git", fake_git(head, status)), *extra_patches):
        return expect(argv_for(out, synthetic=False, extra=_real_extra()), 2, needle, out)


def R13a():
    return binding_case("r13a", None, "", "git HEAD could not be read")


def R13b():
    return binding_case("r13b", "1" * 40, "", f"HEAD {'1' * 40} != --script-commit {'0' * 40}")


def R13c():
    return binding_case("r13c", "0" * 40, "", "code file(s) missing at HEAD",
                        ((ME, "CODE_FILES", ME.CODE_FILES + ("scripts/s4_absent_never.py",)),))


def R13d():
    return binding_case("r13d", "0" * 40, " M src/x.py\n", "needs them clean at HEAD")


def R14():
    out = fresh_out("r14")
    alias = CTX.root / "alias_s43"                                    # never created
    target = str(CTX.root / "never" / "never_TeSt_alias")             # never created
    real = os.path.realpath

    def fake(p, *a, **k):
        return target if os.fspath(p) == str(alias) else real(p, *a, **k)
    with patched((os.path, "realpath", fake), (ME.td, "write_files_exclusive", tripwire),
                 (M, "rejection_counts", fast_counts)):
        ok, det = expect(argv_for(out, arts=_with("43", alias)), 2, "resolving to a 'test' location", out)
    return ok and not os.path.lexists(alias) and not os.path.lexists(target), det


def rules_with(**band):
    real = SS.load_rules

    def fake(*a, **k):
        rules = copy.deepcopy(real(*a, **k))
        rules["alpha_cwd"]["band"].update(band)
        return rules
    return fake


def R15a():
    out = fresh_out("r15a")
    with patched((SS, "load_rules", rules_with(floor=0.01)), (M, "rejection_counts", fast_counts)):
        return expect(argv_for(out), 2, "is not the DL-27 rule", out)


def R15b():
    out = fresh_out("r15b")
    with patched((SS, "load_rules", rules_with(kind="fixed")), (M, "rejection_counts", fast_counts)):
        return expect(argv_for(out), 2, "is not the DL-27 rule", out)


#: set by mutation M40 only: the child runs mde_entry.py with its import-time stack guard call removed
STACK_GUARD_REMOVED = False
STACK_CHILD = """import sys
sys.excepthook = lambda t, v, tb: print("UNCAUGHT " + t.__name__ + ": " + str(v), file=sys.stderr)
sys.modules[{module!r}] = None
script = {script!r}
source = open(script, encoding="utf-8").read()
if {removed!r}:
    source = source.replace("\\nrequire_stats_stack()\\n", "\\n", 1)
sys.argv = [script] + {argv!r}
exec(compile(source, script, "exec"), {{"__name__": "__main__", "__file__": script}})
"""


def stack_case(tag: str, module: str) -> tuple[bool, dict]:
    """A child process in which `module` cannot be imported (sys.modules[module] = None). Its excepthook prints
    only the type and message of an uncaught error, so no traceback line of any module is ever read or shown."""
    out = fresh_out(tag)
    child = STACK_CHILD.format(module=module, script=str(REPO / "scripts" / "mde_entry.py"),
                               removed=STACK_GUARD_REMOVED, argv=[str(x) for x in argv_for(out)])
    p = subprocess.run([sys.executable, "-B", "-c", child], cwd=str(REPO), capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    ok = p.returncode == 2 and f"the running stack lacks {module}" in p.stderr and listing(out) == []
    return ok, {"exit": p.returncode, "stderr": p.stderr.strip()[-300:], "left": listing(out)}


def R16a():
    return stack_case("r16a", "statsmodels")


def R16b():
    return stack_case("r16b", "scipy")


def C7():
    """A completed real-mode run (git faked to a clean HEAD equal to --script-commit; the fixtures' dataset name
    taken as the evaluator's; fast counts): the record-path names, the DL id and every field of the DL line."""
    import src.eval.adapters as AD
    out = fresh_out("c7")
    with patched((ME.td, "_git", fake_git("0" * 40, "")), (AD, "DATASET_NAME", F.FIXTURE_DATASET_NAME),
                 (M, "rejection_counts", fast_counts)):
        code, so, se = run_cli(argv_for(out, synthetic=False, extra=_real_extra()))
    entry_name = next((x.split()[1] for x in so.splitlines() if x.startswith("written: mde_entry_")), None)
    if code != 0 or entry_name is None or not (out / entry_name).exists():
        return False, {"exit": code, "stderr": se[-500:], "files": listing(out)}
    entry = json.loads((out / entry_name).read_text(encoding="utf-8"))
    dl, sha = dl_line(so), hashlib.sha256((out / entry_name).read_bytes()).hexdigest()
    band_sha = hashlib.sha256((out / "dl27_band.json").read_bytes()).hexdigest()
    missing = [x for x in dl_fields(entry, f"reports/derived/{entry_name}", sha,
                                    f"reports/derived/dl27_band.json sha256 {band_sha}", " (DL-99).") if x not in dl]
    stored = entry["decision_log_line"] == dl.replace(f" sha256 {sha}", "", 1)
    return (not missing and stored and entry["artifact_status"] == "provisional"
            and entry["script_commit_dl_id"] == "DL-99"), {"missing": missing, "stored_equals_printed": stored}


def dl_fields(entry: dict, entry_ref: str, entry_sha: str, band_ref: str, tail: str) -> list:
    """The DL line's expected fields, built from the entry's values with the smoke's own formatting."""
    def g(x):
        return f"{x:.6g}"
    p, a, b = entry["pairs"], entry["artifacts"], entry["dl27_band"]
    lo, hi = entry["mde_t_range"]
    return ["AM-17 item 3 MDE entry and AM-16 item 2 / DL-27 tie band (one entry, AM-17 item 3). "
            f"MDE: {entry_ref} sha256 {entry_sha}; E1 s42/s43/s44 VAL artifacts s42 {a['42']['run_id']} ",
            f"at evaluate_model.py commit {a['42']['repo_commit']}, NVIDIA A40, image {CEA.DL17_IMAGE_DIGEST}; ",
            f"n_included {entry['n_included']} (K {entry['K_excluded']}); n_planning {entry['n_planning']}; ",
            f"shifted-null MDE_W {g(entry['mde_w'])} (43−42 {g(p['43-42']['mde_w'])}, 44−42 {g(p['44-42']['mde_w'])}, "
            f"44−43 {g(p['44-43']['mde_w'])}; ",
            "share of exact-zero differences " + "/".join(g(p[k]["share_ties"]) for k in PAIR_ORDER)
            + ", n_zero " + "/".join(str(p[k]["n_zero"]) for k in PAIR_ORDER) + "; ",
            f"SD_Δ {g(entry['sd_delta'])}, dz_MDE {g(entry['dz_mde'])}, MDE_t {g(entry['mde_t'])} [{g(lo)}, {g(hi)}]; ",
            f"power_caveat {entry['power_caveat']}; ",
            f"Band: {band_ref}; E1 best VAL 42/43/44 {BEST['42']!r}/{BEST['43']!r}/{BEST['44']!r} ",
            f"s {PLAN_S!r}; band = max(0.005, √2·s) = {PLAN_BAND!r}.", tail]


def R17():
    out = fresh_out("r17")
    with patched((M, "rejection_counts", fast_counts)):
        return expect(argv_for(out, arts=_with("43", CTX.v_zero)), 2, "every difference is an exact zero", out)


def S1():
    out = fresh_out("s1")
    return expect(argv_for(out, arts=_with("44", CTX.v_value)), 1, "re-derivation", out)


def S3():
    out = fresh_out("s3")
    with patched((M, "power_caveat", lambda w: not (w > M.TAU_P)), (M, "rejection_counts", fast_counts)):
        return expect(argv_for(out), 1, "fails the report layer's validator", out)


def S4():
    out = fresh_out("s4")
    with patched((M, "rejection_counts", fast_counts)):
        return expect(argv_for(out, arts=_with("44", CTX.v_count)), 1, "class count", out)


def S5():
    out = fresh_out("s5")
    with patched((ME, "band_sd", statistics.pstdev), (M, "rejection_counts", fast_counts)):
        return expect(argv_for(out), 1, "refused by its reader", out)


def fast_counts(d, idx, grid):
    return [int(idx.shape[0])] * len(grid)


def S2():
    out = fresh_out("s2")
    with patched((M, "min_count", lambda b: b + 1), (M, "rejection_counts", fast_counts)):
        code, so, se = run_cli(argv_for(out))
    ok = code == 1 and "not determined" in se and so.count("rejection counts") == 3 and not out.exists()
    return ok, {"exit": code, "stdout": so[-400:], "stderr": se[-300:]}


def setup_variants() -> None:
    def setr(path, value):
        def fn(j):
            obj = j
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = value
            return j
        return fn
    CTX.v_commit = variant(CTX.A["43"], "s43_commit", setr(("run", "repo_commit"), "1" * 40))
    CTX.v_gpu = variant(CTX.A["44"], "s44_gpu", setr(("run", "eval_runtime", "gpu_name"), "NVIDIA A100"))
    CTX.v_digest = variant(CTX.A["44"], "s44_digest", setr(("run", "eval_runtime", "image_digest"), None))
    CTX.v_batch = variant(CTX.A["42"], "s42_batch", setr(("run", "eval_runtime", "batch_size"), 8))
    CTX.v_miou = variant(CTX.A["42"], "s42_miou",
                         setr(("dataset_level", "all_class_miou"), CEA.DL17_REFERENCE_MIOU + 2e-4))
    CTX.v_stage = variant(CTX.A["44"], "s44_stage", setr(("run", "stage"), "E2"))
    CTX.v_precision = variant(CTX.A["44"], "s44_precision", setr(("run", "precision"), "int8"))
    CTX.v_inputs = variant(CTX.A["43"], "s43_inputs", setr(("run", "eval_runtime", "input_devices"), ["cpu"]))
    CTX.v_policy = variant(CTX.A["44"], "s44_policy",
                           setr(("run", "eval_runtime", "determinism_policy_applied"), False))
    CTX.v_copy42 = CTX.root / "variants" / "s42_copy"                 # the same run in another directory
    shutil.copytree(CTX.A["42"], CTX.v_copy42)
    CTX.v_copy42_id = variant(CTX.A["42"], "s42_copy_id", setr(("run", "run_id"), "fxA-e1-s42-copy"))

    def as_s43(j):                                                    # s42's scores as a distinct seed-43 run
        j["run"]["run_id"] = "fxA-e1-s43-same-scores"
        j["run"]["checkpoint_sha256"] = hashlib.sha256(b"fxA same scores as s42").hexdigest()
        j["run"]["eval_runtime"]["checkpoint_best_val_miou_all_class"] = BEST["43"]
        return j
    CTX.v_zero = variant(CTX.A["42"], "s42_scores_as_s43", as_s43)

    def bump_value(rows):
        i = next(j for j, r in enumerate(rows) if r["n_eligible_disease_only"] > 0 and r["disease_only_miou"] <= 0.99)
        rows[i]["disease_only_miou"] += 1e-3

    def bump_count(rows):
        i = next(j for j, r in enumerate(rows) if r["n_eligible_disease_only"] > 0)
        rows[i]["n_eligible_disease_only"] += 1
    CTX.v_value = per_image_variant(CTX.A["44"], "s44_value", bump_value)
    CTX.v_count = per_image_variant(CTX.A["44"], "s44_count", bump_count)


def per_image_variant(src: Path, tag: str, fn) -> Path:
    """A copy of an artifact with per_image.jsonl edited by fn(rows) and the manifest rehashed."""
    dest = CTX.root / "variants" / tag
    shutil.copytree(src, dest)
    p = dest / "per_image.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    fn(rows)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    F.rehash(dest)
    return dest


# --------------------------------------------------------------------------------------------------
# M: mutations, each killed by a named check
# --------------------------------------------------------------------------------------------------
def kw(**over) -> dict:
    return dict(KWARGS, **over)


def fresh_per_delta(d, idx, grid):
    counts = []
    for k, delta in enumerate(grid):
        idx_k = M.resample_indices(d.size, idx.shape[1], seed=M.RNG_SEED + 1 + k, b=idx.shape[0])
        x = d[idx_k] + delta
        counts.append(sum(1 for r in range(x.shape[0]) if M.rejects(M.wilcoxon_p(x[r]))))
    return counts


def min_overall(pairs):
    name = min(pairs, key=lambda key: pairs[key]["mde_w"])
    return pairs[name]["mde_w"], name


_REAL_PAIR_SUMMARY = M.pair_summary
_REAL_DL_LINE = ME.decision_log_line


def resample_at_m(d, n, **kw):
    """Mutation: each pair resampled at m = n_included instead of n_planning (the size the entry records)."""
    return _REAL_PAIR_SUMMARY(d, len(d), **kw)


def binding_without(rule: str):
    """Mutation: commit_binding without one rule ('tree': the clean-tree check; 'files': the code-files check)."""
    def binding(args) -> dict:
        head = ME.td.git_head()
        if head is None:
            raise ME.td.Refused("git HEAD could not be read")
        if head != args.script_commit:
            raise ME.td.Refused(f"HEAD {head} != --script-commit {args.script_commit}")
        missing = [p for p in ME.CODE_FILES if not (ME.REPO / p).is_file()]
        if rule != "files" and missing:
            raise ME.td.Refused(f"{len(missing)} code file(s) missing at HEAD: {missing}")
        tree = ME.td.git_status(("src", "configs", "scripts"))
        if rule != "tree" and (tree is None or tree.strip()):
            raise ME.td.Refused("git status lists entries; a real run needs them clean at HEAD")
        return {"head": head, "script_commit": args.script_commit, "script_commit_dl_id": args.script_commit_dl_id}
    return binding


def guard_names_only(paths: dict) -> None:
    """Mutation: guard_paths without its resolved-path stage."""
    ME.td.refuse_test_names([str(v) for v in paths.values()], "path arguments")


def mode_one_way(names, synthetic) -> None:
    """Mutation: the mode gate refuses synthetic names in a real run only."""
    from src.eval.adapters import DATASET_NAME
    if any(n != DATASET_NAME for n in names) and not synthetic:
        raise ME.td.Refused("--synthetic-inputs is False but the names are synthetic")


def names_only(checks, what) -> None:
    """Mutation: require_passed names the failed checks without their details."""
    bad = [c[0] for c in checks if c[1] is not True]
    if bad:
        raise ME.td.Refused(f"{what}: {len(bad)} check(s) failed: {bad}")


def check_d_with_zeros(d):
    """Mutation: check_d accepts a d of exact zeros only."""
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 1 or d.size < 2 or not np.isfinite(d).all():
        raise M.MdeError("refused")
    return d


MUTATIONS = [
    ("M1", "zero_method 'wilcox'", lambda: [(M, "WILCOXON_KWARGS", kw(zero_method="wilcox"))], "U7"),
    ("M2", "correction False", lambda: [(M, "WILCOXON_KWARGS", kw(correction=False))], "U7"),
    ("M3", "alternative 'two-sided'", lambda: [(M, "WILCOXON_KWARGS", kw(alternative="two-sided"))], "U7"),
    ("M4", "ALPHA 0.05", lambda: [(M, "ALPHA", 0.05)], "D1b"),
    ("M5", "p <= ALPHA", lambda: [(M, "rejects", lambda p: p <= M.ALPHA)], "U6"),
    ("M6", "fresh indices per delta", lambda: [(M, "rejection_counts", fresh_per_delta)], "U8"),
    ("M7", "RNG seed 43", lambda: [(M, "RNG_SEED", 43)], "U8"),
    ("M8", "GUARD_POINTS 1", lambda: [(M, "GUARD_POINTS", 1)], "U3b"),
    ("M9", "count > need", lambda: [(M, "qualifies", lambda c, need: c > need)], "U3a"),
    ("M10", "min over pairs", lambda: [(M, "overall", min_overall)], "U4"),
    ("M11", "SD with ddof 0", lambda: [(M, "sample_sd", lambda d: float(np.std(np.asarray(d), ddof=0)))], "U9"),
    ("M12", "n_planning = included", lambda: [(M, "n_planning", lambda included: included)], "U1a"),
    ("M13", "caveat mde_w >= tau_p", lambda: [(M, "power_caveat", lambda w: w >= M.TAU_P)], "U5a"),
    ("M14", "pair orientation flipped", lambda: [(ME, "PAIRS", tuple((n, c, b) for n, b, c in ME.PAIRS))], "C5"),
    ("M15", "DL-17 check skipped",
     lambda: [(ME, "check_dl17", lambda arts: ["skipped", True, []]), (M, "rejection_counts", fast_counts)], "R4d"),
    ("M16", "commit check skipped",
     lambda: [(ME, "check_commit", lambda arts: ["skipped", True, {}]), (M, "rejection_counts", fast_counts)], "R4a"),
    ("M17", "band s with pstdev", lambda: [(ME, "band_sd", statistics.pstdev)], "B1"),
    ("M18", "band without its floor", lambda: [(ME, "band_value", lambda s, fl: math.sqrt(2.0) * s)], "B5"),
    ("M28", "pairs resampled at m = n_included", lambda: [(M, "pair_summary", resample_at_m)], "C6"),
    ("M29", "analytic block at m = n_included",
     lambda: [(ME, "analytic_block", lambda pairs, n: M.analytic([pairs[k]["sd"] for k in PAIR_ORDER],
                                                                 pairs["43-42"]["m"]))], "C6"),
    ("M30", "binding without its clean-tree check", lambda: [(ME, "commit_binding", binding_without("tree"))], "R13d"),
    ("M31", "path guard without its resolved-path stage", lambda: [(ME, "guard_paths", guard_names_only)], "R14"),
    ("M32", "require_role skipped", lambda: [(ME, "require_role", lambda art, **k: None)], "R4g"),
    ("M33", "mode gate in one direction only", lambda: [(ME, "check_mode", mode_one_way)], "R7e"),
    ("M34", "distinctness without the checkpoints", lambda: [(ME, "DISTINCT", ME.DISTINCT[:2])], "R8c"),
    ("M35", "AM-5 sets compared by count only",
     lambda: [(ME, "same_excluded", lambda a5: len({a5[s].excluded_count for s in SEEDS}) == 1)], "R5"),
    ("M36", "report validator skipped", lambda: [(ME, "validate_mde_entry", lambda doc: ())], "S3"),
    ("M37", "class-count rule removed", lambda: [(ME, "count_ok", lambda n_stats, n_record: True)], "S4"),
    ("M38", "band read-back skipped", lambda: [(ME, "band_self_check", lambda *a, **k: [])], "S5"),
    ("M39", "rule file's band not validated",
     lambda: [(ME, "band_floor", lambda: float(SS.load_rules()["alpha_cwd"]["band"]["floor"]))], "R15a"),
    ("M40", "the import-time stack guard removed (in the child's source)",
     lambda: [(sys.modules[__name__], "STACK_GUARD_REMOVED", True)], "R16a"),
    ("M41", "failed checks named without their details", lambda: [(ME, "require_passed", names_only)], "R12"),
    ("M42", "a d of exact zeros only accepted", lambda: [(M, "check_d", check_d_with_zeros)], "R17"),
    ("M51", "the DL line prints dz_MDE as MDE_t",
     lambda: [(ME, "decision_log_line", lambda entry, *a: _REAL_DL_LINE(dict(entry, mde_t=entry["dz_mde"]), *a))],
     "C7"),
    ("M52", "binding without its code-files check", lambda: [(ME, "commit_binding", binding_without("files"))],
     "R13c"),
]


# --------------------------------------------------------------------------------------------------
def checks() -> list:
    out = [("U0", "constants: B 2000, seed 42, alpha 0.00625, 51 grid points, 3 guard points, the kwargs, Z, "
                  "EFFICIENCY, TAU_P == report.TAU_P, min_count(2000) == 1600", U0),
           ("U1a", "n_planning(846) == 1561", U1a), ("U1b", "n_planning(800) == 1476", U1b),
           ("U1c", "n_planning(423) == 780 (the one tie, to even)", U1c), ("U1d", "n_planning(0) refused", U1d),
           ("U2", "grid: 51 points, k/1000", U2),
           ("U3a", "guard: 1600 at k..k+2 qualifies", U3a), ("U3b", "guard: a 1599 in the window skips k", U3b),
           ("U3c", "guard: never >= 1600 gives None", U3c), ("U3d", "guard: k = 48 is the last candidate", U3d),
           ("U4", "MDE_W is the max over pairs, first pair on a tie", U4),
           ("U5a", "caveat: 0.010 -> False, validator ok", U5a), ("U5b", "caveat: 0.011 -> True, validator ok", U5b),
           ("U6", "p == 0.00625 is not a rejection", U6),
           ("U7", "call spy: the pre-registered kwargs, 1-D float64 of length n, len(grid) x b calls", U7),
           ("U8", "index spy: one default_rng(42) and one integers(0, m, size=(b, n), int64) per pair", U8),
           ("U8b", "default_rng(42).choice(d + delta) == d[idx] + delta for delta 0, 0.001, 0.05 (numpy 1.26.4)", U8b),
           ("U9", "analytic: ddof 1, max SD, dz from literals, MDE_t and its range", U9),
           ("U10", "check_d refuses exact zeros only, a non-finite value and a single value", U10),
           ("B1", "band: the writer's file passes dl27_band, (s, band) bitwise equal", B1),
           ("B2", "band: s + 1e-9 -> band_s_mismatch", B2), ("B3", "band: band + 1e-9 -> band_mismatch", B3),
           ("B4", "band: FACTS 3 values give s and band exactly", B4),
           ("B5", "band: near-equal values give the 0.005 floor", B5),
           ("B6", "band: exactly the reader's keys, LF-terminated", B6)]
    out += [(f"R1{chr(ord('a') + i)}", f"--{f.replace('_', '-')} with a never-created test path -> exit 2 before "
             "any filesystem call", make_r1(f)) for i, f in enumerate(ME.PATH_FLAGS)]
    out += [("R2a", "existing dl27_band.json -> exit 2, bytes unchanged", R2a),
            ("R2b", "existing mde_entry name -> exit 2, bytes unchanged", R2b),
            ("R3a", "missing --s43 directory -> exit 2", R3a), ("R3b", "missing --best-json-44 -> exit 2", R3b),
            ("R4a", "s43 with another repo_commit -> exit 2", R4a), ("R4b", "s44 with another gpu_name -> exit 2", R4b),
            ("R4c", "s44 with a null image_digest -> exit 2", R4c), ("R4d", "s42 at batch 8 (not DL-17) -> exit 2", R4d),
            ("R4e", "s42 all_class_miou off by 2e-4 -> exit 2", R4e),
            ("R4f", "best.json 43 != its artifact's checkpoint value -> exit 2", R4f),
            ("R4g", "s44 at stage E2 -> exit 2 (require_role)", R4g),
            ("R4h", "s44 at precision int8 -> exit 2 (require_role)", R4h),
            ("R4i", "s43 with its inputs on the CPU -> exit 2", R4i),
            ("R4j", "s44 without the determinism policy -> exit 2", R4j),
            ("R5", "s44 with another AM-5 set of the same size -> exit 2 STOP (lane 6 (f))", R5),
            ("R5b", "s44 with an AM-5 set of another size -> exit 2 STOP (lane 6 (f))", R5b),
            ("R6", "--out-dir inside the repository -> exit 2 (writes tripwired)", R6),
            ("R7a", "real mode without the binding flags -> exit 2 (the flag gate's own words)", R7a),
            ("R7b", "--generated-utc in real mode -> exit 2", R7b),
            ("R7c", "synthetic inputs without --synthetic-inputs (binding patched) -> exit 2", R7c),
            ("R7d", "real mode with --script-commit but no --script-commit-dl-id -> exit 2", R7d),
            ("R7e", "the evaluator's dataset name with --synthetic-inputs -> exit 2", R7e),
            ("R8", "--s43 given the s42 directory -> exit 2", R8),
            ("R8b", "--s43 given a copy of s42 (same run_id) -> exit 2", R8b),
            ("R8c", "--s43 given s42 under another run_id (same checkpoint) -> exit 2", R8c),
            ("R9", "--expect-k-val K+1 -> exit 2", R9),
            ("R10", "best.json with an extra key -> exit 2", R10),
            ("R10b", "best.json with a non-string best_ckpt -> exit 2", R10b),
            ("R11", "real mode with matches_pinned False -> exit 2", R11),
            ("R12", "a wrong --expect-manifest-sha256-43 -> exit 2, the observed hash in the message", R12),
            ("R13a", "binding: HEAD unreadable -> exit 2", R13a),
            ("R13b", "binding: HEAD != --script-commit -> exit 2", R13b),
            ("R13c", "binding: a code file missing -> exit 2", R13c),
            ("R13d", "binding: src/configs/scripts not clean -> exit 2", R13d),
            ("R14", "a --s43 resolving to a test location -> exit 2, nothing created", R14),
            ("R15a", "the rule file's band floor 0.01 -> exit 2", R15a),
            ("R15b", "the rule file's band kind 'fixed' -> exit 2", R15b),
            ("R16a", "a child process without statsmodels -> exit 2 (the import-time guard), nothing written", R16a),
            ("R16b", "a child process without scipy -> exit 2 (the import-time guard), nothing written", R16b),
            ("R17", "two runs that agree image for image -> exit 2", R17),
            ("S1", "a per-image value off by 1e-3 -> exit 1 (re-derivation), nothing written", S1),
            ("S2", "no qualifying delta -> exit 1, nothing written, three curves printed", S2),
            ("S3", "an entry the report validator rejects -> exit 1, nothing written", S3),
            ("S4", "a per-image class count off by one -> exit 1, nothing written", S4),
            ("S5", "a band its reader refuses -> exit 1, nothing written", S5),
            ("C6", "in-process CLI (fast counts): every headline number recomputed from the entry", C6),
            ("C7", "a completed real-mode run (git faked, fast counts): record paths, DL-99 and every DL-line field",
             C7)]
    return out


def checks_slow() -> list:
    return [("C1", "subprocess CLI: exit 0, exactly the two files, the DL line with both sha256", C1),
            ("C2", "entry: lane 6 (c) keys, pairs, validator (), checks passed with the three pins", C2),
            ("C3", "report layer: an uncommitted entry reads ('pending', None)", C3),
            ("C4a", "rerun with the same --generated-utc: entry byte-identical", C4a),
            ("C4b", "rerun: band byte-identical", C4b),
            ("C5", "orientation: 43-42 mean > 0 and 44-43 mean < 0 when seed 43 is better", C5),
            ("D1a", "d1: |MDE_t - dz_smoke * SD| <= 1e-9 and |dz - 0.0845| < 5e-5", D1a),
            ("D1b", "d1: power at delta 0 in [0.001, 0.015]", D1b),
            ("D1c", "d1: power non-decreasing from MDE_W on", D1c),
            ("D1d", "d1: power at MDE_W >= 0.80", D1d),
            ("D1e", "d1: n_zero == 254 and share_ties == 254/846", D1e),
            ("D2", "d2: MDE_W / MDE_t in [0.95, 1.15]", D2),
            ("D2b", "d2: MDE_W(d2 + 0.003) == MDE_W(d2) - 0.003 within one step", D2b),
            ("D3a", "d3: same seed -> identical pair-block bytes", D3a),
            ("D3b", "d3: seed 43 -> MDE_W within one grid step", D3b)]


def run_check(cid, desc, fn) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 -- a crash is a failed check, reported
        ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    record(cid, desc, ok, detail)
    NORMAL[cid] = (bool(ok), fn)
    return ok


def make_root(prefix: str):
    """The temporary root, created only after TMPDIR's realpath is known to hold no 'test'. Its name is the
    prefix plus hex digits, which contain neither 's' nor 't', so it cannot spell 'test' either."""
    base = os.environ.get("TMPDIR")
    if not base:
        return None, "TMPDIR is not set; run with TMPDIR=/tmp/s4scratch"
    if "test" in base.lower():                                        # the string first, before any filesystem call
        return None, "TMPDIR contains 'test'"
    real = os.path.realpath(base)
    if "test" in real.lower() or not os.path.isdir(real):
        return None, f"TMPDIR resolves to {real!r}, which contains 'test' or is not a directory"
    root = Path(real) / f"{prefix}{secrets.token_hex(4)}"
    root.mkdir(exist_ok=False)
    return root, None


def main() -> int:
    t_all = time.time()
    root, why = make_root("s4_smoke_mde_")
    if root is None:
        print(f"ABORT: {why}", file=sys.stderr)
        return 2
    real = os.path.realpath(root)
    if "test" in real.lower():
        print(f"ABORT: the temporary root {real} contains 'test'", file=sys.stderr)
        return 2
    CTX.root = root
    print(f"smoke_mde: temporary root {real} (left in place); numpy {np.__version__}", flush=True)
    t0 = time.time()
    setup_fixtures()
    setup_variants()
    print(f"  (fixtures in {time.time() - t0:.0f} s)", flush=True)
    for cid, desc, fn in checks():
        run_check(cid, desc, fn)
    setup_c()
    setup_c4()
    t0 = time.time()
    setup_d()
    for cid, desc, fn in checks_slow():
        run_check(cid, desc, fn)
    print(f"  (C and D checks in {time.time() - t0:.0f} s after setup)", flush=True)
    for mid, desc, make, killer in MUTATIONS:
        passed, fn = NORMAL[killer]
        try:
            with patched(*make()):
                ok, detail = fn()
        except Exception as e:  # noqa: BLE001 -- a crash under the mutation fails the killer: killed
            ok, detail = False, f"{type(e).__name__}: {e}"
        killed = passed and not ok
        record(mid, f"mutation '{desc}' killed by {killer}", killed,
               detail if not killed else "")
    n_ok = sum(1 for r in RESULTS if r[2])
    n_mut = sum(1 for r in RESULTS if r[0].startswith("M") and r[2])
    print(f"\nPASS {n_ok}/{len(RESULTS)} (mutations killed {n_mut}/{len(MUTATIONS)}); {time.time() - t_all:.0f} s",
          flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
