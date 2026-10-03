#!/usr/bin/env python3
"""Statistics artifact writer gate and thirteen integrity checks (lane L-STATS-OFFICIAL; synthetic).

Exercises src/stats/artifact.py end to end on synthetic evaluation artifacts written through the
unmodified evaluator writer into temporary clean fixture repositories outside the working tree
(scripts/stats_fixtures.py). No dataset, checkpoint, model or GPU; nothing is written into this
repository, and no statistics artifact with status `official` is written anywhere.

  A  a valid nonofficial artifact: written, all 13 checks established with the inputs and the
     contract; the artifact-only re-establishment reports the six input/contract checks
  C  acceptance (c): one break case per integrity check, each with its injection mechanism; each is
     refused naming that check, and nothing is left behind (final or temporary directory). A check
     the gate establishes before anything is created is exercised with a recording tamper hook that
     must never run, so a refusal that slipped from the gate to the final verifier pass is caught
  W  the verifier alone: the valid artifact edited consistently (family.json and bootstrap.npz,
     manifest regenerated) and verified artifact-only, for every check whose writer-side refusal is
     layered (P37: a single dropped rule cannot hide behind another layer)
  F  acceptance (f): one refused case per writer-gate clause, the official-condition clauses by name,
     each clause of check_input_provenance (the function the gate runs on the inputs it re-reads), O3
     code provenance (one test per refusal), P6 in both directions, the unpinned stack (section 10.1
     forces nonofficial), the driver's P5 snapshot required
  B  acceptance (b): count warnings -- every task's B and jackknife count against its own expected
     value (1561 dataset-level, 1561 - k per-image), a per-image count != n_paired is fatal
  K  OK-2: the section 10.1 pins parsed from the contract, and the refusals of a reworded sentence
  V  the verifier: a post-write edit, duplicate keys, an artifact read as official while unbound
     (relabelled in memory after the strict parse: no smoke writes an official status, P2)

"Official request" cases patch the running stack to the pinned one (as smoke_stats_bootstrap does)
and use cheap placeholder bootstrap tasks with B = 10,000: the gate refuses them before anything is
created, so no draw is needed.

Run:  python -B scripts/smoke_stats_integrity.py
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                       # noqa: E402

import stats_fixtures as F                                               # noqa: E402
from src.eval.adapters import DATASET_NAME                               # noqa: E402
from src.stats import artifact as A                                      # noqa: E402
from src.stats import bootstrap as BS                                    # noqa: E402
from src.stats import driver as D                                        # noqa: E402
from src.stats.ingest import Policy                                      # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
B_SMALL = 24
N_SMALL = 30
N_OFF, K_OFF = 1561, 3
CREATED = "2026-10-03T00:00:00Z"
PINNED_STACK = A.observed_environment() == A.PINNED_ENVIRONMENT


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def outcome(fn):
    """('ok', value) or (exception class name, exception)."""
    try:
        return "ok", fn()
    except Exception as e:                                               # noqa: BLE001
        return type(e).__name__, e


def nothing_left(out: Path) -> bool:
    return not out.exists() and not A.stale_temp_dirs(out)


def integrity_case(name, out: Path, expect_check: str, fn, note=""):
    kind, e = outcome(fn)
    ok = kind == "IntegrityError" and e.check == expect_check and nothing_left(out)
    check(f"{name} -> refused: {expect_check}; nothing left{note}", ok, f"{kind}: {e}")


def gate_integrity_case(name, out: Path, expect_check: str, make_fn, note=""):
    """An integrity check the gate (writer step 3) establishes before anything is created. The request
    carries a recording tamper hook; a refusal from the writer's final verifier pass would have called
    it, so an empty record shows the gate refused."""
    calls: list[str] = []
    kind, e = outcome(make_fn(lambda tmp, stage: calls.append(stage)))
    ok = (kind == "IntegrityError" and e.check == expect_check and nothing_left(out)
          and calls == [])
    check(f"{name} -> refused by the gate before anything is created: {expect_check}; nothing "
          f"left{note}", ok, f"{kind}: {e}; tamper hook calls {calls}")


def gate_case(name, out: Path, expect_names, fn, *, exact=False):
    kind, e = outcome(fn)
    names = set(e.names) if kind == "WriterGateRefused" else set()
    ok = (kind == "WriterGateRefused" and (names == set(expect_names) if exact
                                           else set(expect_names) <= names) and nothing_left(out))
    check(f"{name} -> refused by the gate naming {sorted(expect_names)}; nothing left", ok,
          f"{kind}: {e}")


def git(root: Path, *args):
    return subprocess.run(["git", "-c", "user.name=stats-fixture", "-c", "user.email=stats@invalid",
                           "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True,
                          capture_output=True, text=True)


def rewrite_family(tmp: Path, fn) -> None:
    fam = json.loads((tmp / A.FAMILY_JSON).read_text(encoding="utf-8"))
    (tmp / A.FAMILY_JSON).write_text(A.canonical_json(fn(fam)), encoding="utf-8", newline="\n")


def tamper_at(stage_wanted, fn):
    def hook(tmp, stage):
        if stage == stage_wanted:
            fn(tmp)
    return hook


def placeholder_tasks(obs, analysis_id, B):
    """Cheap tasks with real seeds and counts (constant replicates, P3 fallback): the gate and
    build_family read counts, never replicate values."""
    out = []
    for spec in BS.FROZEN_TASK_MATRIX:
        v = D.observed_value(obs, spec)
        n = (obs.pooled[spec.comparison_id].n_images if spec.statistic_name == BS.DATASET_MIOU_DELTA
             else obs.paired[spec.comparison_id].n)
        reps, jack = np.full(B, v), np.full(n, v)
        out.append(BS.BootstrapTaskResult(
            spec, analysis_id, BS.derive_seed(analysis_id, spec.comparison_id, spec.statistic_name),
            v, reps, jack, BS.bca_interval(reps, jack, v, spec.interval_type)))
    return tuple(out)


class Env:
    """One loaded inventory, its observed values and tasks, and ArtifactInputs factories."""

    def __init__(self, root: Path, *, n, k, split, status, policy, analysis_id, B, name, prefix,
                 placeholder=False):
        self.fs = F.build_inventory(root, n=n, k=k, split=split, status=status, dataset_name=name,
                                    run_prefix=prefix)
        self.entries = D.load_input_list(F.write_input_list(root.parent / f"{prefix}.json", self.fs))
        self.policy, self.analysis_id = policy, analysis_id
        self.inputs = D.load_inputs(self.entries, self.fs.input_root, policy)
        self.obs = D.assemble_observed(self.inputs, policy)
        self.tasks = (placeholder_tasks(self.obs, analysis_id, B) if placeholder
                      else D.run_tasks(self.obs, analysis_id, B))
        self.csha, self.craw = A.contract_sha256()

    def inp(self, run_id, synthetic, **over):
        base = D.artifact_inputs(self.obs, self.inputs, self.tasks, analysis_id=self.analysis_id,
                                 run_id=run_id, created_at_utc=CREATED, contract_sha256=self.csha,
                                 synthetic=synthetic, input_root=self.fs.input_root)
        return dataclasses.replace(base, **over)


class verifier_reads_official:
    """The verifier reads family.json as status official with no warnings: relabelled in memory right
    after the strict parse (check 10), so no smoke writes an official status to disk (P2)."""

    def __enter__(self):
        self.real = real = A._check_finite_values

        def relabel(ctx):
            real(ctx)
            ctx.fam = dict(ctx.fam, artifact_status="official", warnings=[])
        A._check_finite_values = relabel

    def __exit__(self, *exc):
        A._check_finite_values = self.real


class pinned_stack:
    """Run as if on the section 10.1 stack (the existing smoke's monkeypatch of the running stack)."""

    def __enter__(self):
        self.real = A.observed_environment
        A.observed_environment = lambda: dict(A.PINNED_ENVIRONMENT)

    def __exit__(self, *exc):
        A.observed_environment = self.real


# --------------------------------------------------------------------------------------------------
def part_a(env: Env, outs: Path) -> None:
    out = outs / "a-0001"
    path = A.write_statistics_artifact(out, env.inp("a-0001", True))
    check("A1 a nonofficial artifact is written with exactly three files and no temporary sibling",
          sorted(p.name for p in path.iterdir())
          == sorted([A.FAMILY_JSON, A.BOOTSTRAP_NPZ, A.MANIFEST_NAME])
          and not A.stale_temp_dirs(out))
    full = A.verify_statistics_artifact(path, input_root=env.fs.input_root, contract_bytes=env.craw)
    check("A2 with the inputs and the contract all 13 checks are re-established",
          full.established == A.INTEGRITY_CHECKS and not full.not_reestablished)
    only = A.verify_statistics_artifact(path)
    want_skip = {A.C_PROVENANCE, A.C_CONTRACT, A.C_A3A, A.C_MATRIX, A.C_SOURCE, A.C_POLICY}
    check("A3 artifact-only: checks 5, 7, 8, 9, 10, 12, 13 re-established; 1, 2, 3, 4, 6, 11 "
          "reported as not re-established", set(only.not_reestablished) == want_skip
          and set(only.established) == set(A.INTEGRITY_CHECKS) - want_skip, only.not_reestablished)
    c_only = A.verify_statistics_artifact(path, contract_bytes=env.craw)
    i_only = A.verify_statistics_artifact(path, input_root=env.fs.input_root)
    check("A4 contract only -> 1, 3, 4, 6, 11 not re-established; inputs only -> 2, 11",
          set(c_only.not_reestablished) == want_skip - {A.C_CONTRACT}
          and set(i_only.not_reestablished) == {A.C_CONTRACT, A.C_POLICY})
    fam = full.family
    want = [A.W_ANALYSIS_ID] + ([] if PINNED_STACK else [A.W_SOFTWARE]) + [
        A.W_INPUT, A.W_SYNTHETIC, A.W_B, A.W_JACKKNIFE]
    check("A5 status nonofficial; warnings re-derived from the running stack, inputs and counts",
          fam["artifact_status"] == "nonofficial" and fam["warnings"] == want, fam["warnings"])
    check("A6 the integrity block is the all-true finalized value",
          fam["integrity"] == {"status": "passed", "checks": {c: True for c in A.INTEGRITY_CHECKS}})
    check("A7 the software block is the running stack's",
          fam["software_environment"] == A.software_environment_block())


def part_w(env: Env, outs: Path) -> None:
    """The verifier alone: the valid artifact a-0001 copied, edited consistently (family.json and,
    where needed, bootstrap.npz), its manifest regenerated, then verified artifact-only (W4b: with the
    inputs, for the input part of check 4). The writer's gate and its read-back comparison are not
    involved, so the named check is the only layer that can refuse the edit."""
    src = outs / "a-0001"

    def edited(tag, fam_fn=None, npz_fn=None):
        d = outs / f"w-{tag}" / src.name
        shutil.copytree(src, d)
        if fam_fn is not None:
            rewrite_family(d, fam_fn)
        if npz_fn is not None:
            with np.load(d / A.BOOTSTRAP_NPZ, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            npz_fn(arrays)
            np.savez_compressed(d / A.BOOTSTRAP_NPZ, **arrays)
        (d / A.MANIFEST_NAME).write_text(A._manifest_text(d), encoding="utf-8", newline="\n")
        return d

    def case(name, d, expect_check, **kw):
        kind, e = outcome(lambda: A.verify_statistics_artifact(d, **kw))
        how = "verification with the inputs" if kw else "artifact-only verification"
        check(f"{name} -> {how} refuses: {expect_check}",
              kind == "IntegrityError" and e.check == expect_check, f"{kind}: {e}")

    def upper_hex(f):
        r = f["input_artifacts"][0]
        r["class_map_sha256"] = r["class_map_sha256"].upper()
        return f
    case("W1 an input record's class_map_sha256 in uppercase hex", edited("01", upper_hex),
         A.C_PROVENANCE)

    def lf_hex(f):
        r = f["input_artifacts"][0]
        r["config_sha256"] = r["config_sha256"] + "\n"
        return f
    case("W1b an input record's config_sha256 with a trailing newline (65 characters)",
         edited("01b", lf_hex), A.C_PROVENANCE)

    def bad_contract(f):
        f["statistical_contract_sha256"] = f["statistical_contract_sha256"][:-1] + "z"
        return f
    case("W2 a statistical_contract_sha256 that is not 64 lowercase hex", edited("02", bad_contract),
         A.C_CONTRACT)

    def flip_reject(f):
        m = f["a3a_family"]["holm"]["members"][2]
        m["reject"] = not m["reject"]
        return f
    case("W3 a stored Holm member's reject flipped", edited("03", flip_reject), A.C_A3A)

    def holm_failed(f):
        f["a3a_family"]["holm"]["status"] = "failed"
        return f
    case("W3b a3a_family.holm.status 'failed' (a frozen literal: check 10 owns the key sets, types "
         "and frozen values of every block, the accepted P0 design)", edited("03b", holm_failed),
         A.C_FINITE)

    def swapped_keys(f):
        c = f["a3a_family"]["comparisons"][0]
        keys = list(c)
        keys[0], keys[1] = keys[1], keys[0]
        f["a3a_family"]["comparisons"][0] = {k: c[k] for k in keys}
        return f
    case("W3c the first two keys of a3a_family.comparisons[0] swapped (key order: check 10)",
         edited("03c", swapped_keys), A.C_FINITE)
    i = BS.FROZEN_TASK_IDS.index("accuracy_e2_e3__mean_delta")

    def short_json(f):
        f["bootstrap_tasks"][i]["jackknife_count"] -= 1
        return f

    def short_npz(a):
        key = f"{BS.FROZEN_TASK_IDS[i]}__jackknife"
        a[key] = a[key][:-1]
    case("W4 a per-image task's jackknife count n_paired - 1 in family.json and bootstrap.npz alike",
         edited("04", short_json, short_npz), A.C_MATRIX)
    ds = [t for t in BS.FROZEN_TASK_IDS if t.endswith(f"__{BS.DATASET_MIOU_DELTA}")]

    def long_json(f):
        for t in f["bootstrap_tasks"]:
            if t["task_id"] in ds:
                t["jackknife_count"] += 1
        return f

    def long_npz(a):
        for tid in ds:
            key = f"{tid}__jackknife"
            a[key] = np.append(a[key], a[key][-1])
    case("W4b every dataset-level task's jackknife count N + 1 in family.json and bootstrap.npz "
         "alike (one N, still >= n)", edited("04b", long_json, long_npz), A.C_MATRIX,
         input_root=env.fs.input_root)
    pi = [t for t in BS.FROZEN_TASK_IDS if t not in ds]

    def all_short_json(f):
        for t in f["bootstrap_tasks"]:
            if t["task_id"] in pi:
                t["jackknife_count"] -= 1
        return f

    def all_short_npz(a):
        for tid in pi:
            key = f"{tid}__jackknife"
            a[key] = a[key][:-1]
    case(f"W4c every per-image task's ({len(pi)}) jackknife count n_paired - 1 in family.json and "
         "bootstrap.npz alike (one n, still <= N)", edited("04c", all_short_json, all_short_npz),
         A.C_MATRIX)
    j = BS.FROZEN_TASK_IDS.index("accuracy_e4_e5__hodges_lehmann_shift")
    bumped = {}

    def observed_json(f):
        t = f["bootstrap_tasks"][j]
        t["observed"] = bumped["v"] = float(np.nextafter(t["observed"], 1.0))
        return f

    def observed_npz(a):
        a[f"{BS.FROZEN_TASK_IDS[j]}__observed"] = np.array([bumped["v"]], dtype=np.float64)
    case("W6 a Source-A observed value one ulp from its comparison record, family.json and "
         "bootstrap.npz alike", edited("06", observed_json, observed_npz), A.C_SOURCE)
    jc = BS.FROZEN_TASK_IDS.index(f"accuracy_e1_e2__{BS.DATASET_MIOU_DELTA}")
    pooled = {}

    def source_c_json(f):
        t = f["bootstrap_tasks"][jc]
        t["observed"] = pooled["v"] = float(np.nextafter(t["observed"], 1.0))
        return f

    def source_c_npz(a):
        a[f"{BS.FROZEN_TASK_IDS[jc]}__observed"] = np.array([pooled["v"]], dtype=np.float64)
    case("W6c a Source-C (pooled) observed value one ulp off, family.json and bootstrap.npz alike "
         "(no artifact-level counterpart; O4)", edited("06c", source_c_json, source_c_npz),
         A.C_SOURCE, input_root=env.fs.input_root)

    def entropy_decimal(f):
        t = f["bootstrap_tasks"][11]
        t["seed_entropy_decimal"] = str(int(t["seed_entropy_decimal"]) + 1)
        return f
    case("W7 one task's seed_entropy_decimal off by one (root seed, digest and NPZ bytes intact)",
         edited("07", entropy_decimal), A.C_SEED)

    def display(f):
        t = f["bootstrap_tasks"][12]
        t["seed_input_display"] = t["seed_input_display"].replace("\\0", "|", 1)
        return f
    case("W7b one task's seed_input_display with a changed separator", edited("07b", display),
         A.C_SEED)

    def z0_ulp(f):
        t = f["bootstrap_tasks"][4]
        t["z0"] = float(np.nextafter(t["z0"], 1.0))
        return f
    case("W8 one task's z0 one ulp off in family.json only", edited("08", z0_ulp), A.C_JSON_NPZ)

    def drop_b(f):
        f["warnings"] = [w for w in f["warnings"] if w != A.W_B]
        return f
    case("W11 bootstrap_replicates_nonproduction removed while every task has B = 24 (status still "
         "nonofficial)", edited("11", drop_b), A.C_POLICY)


def part_c(env: Env, outs: Path) -> None:
    """Acceptance (c): one break case per integrity check. '(afd2d33 accepts)' marks a case the
    pre-lane writer and verifier accepted."""
    inp = env.inp

    def run(run_id, synthetic=True, hook=None, **over):
        return lambda: A.write_statistics_artifact(outs / run_id, inp(run_id, synthetic, **over),
                                                   _tamper=hook)
    recs = [dict(r) for r in inp("x", True).input_artifacts]
    recs[3] = dict(recs[3], config_sha256="0" * 64)
    gate_integrity_case("C1 provenance: one record's config_sha256 replaced in ArtifactInputs "
                        "(afd2d33 accepts)", outs / "c-01", A.C_PROVENANCE,
                        lambda h: run("c-01", hook=h, input_artifacts=recs))
    gate_integrity_case("C2 contract: contract_sha256 recorded at run start is another contract's "
                        "(afd2d33 accepts)", outs / "c-02", A.C_CONTRACT,
                        lambda h: run("c-02", hook=h, contract_sha256=hashlib.sha256(
                            b"another contract").hexdigest()))
    holm = env.obs.holm
    m0 = holm.members[0]
    bad_holm = dataclasses.replace(holm, members=(dataclasses.replace(m0, reject=not m0.reject),)
                                   + holm.members[1:])
    integrity_case("C3 a3a family: a hand-built finalized HolmFamily with member 0's reject "
                   "flipped (afd2d33 accepts)", outs / "c-03", A.C_A3A,
                   run("c-03", holm_family=bad_holm))
    tasks = list(env.tasks)
    i9 = 9
    tasks[i9] = dataclasses.replace(tasks[i9], jackknife=tasks[i9].jackknife[:-1])
    integrity_case(f"C4 matrix: task {i9} ({env.tasks[i9].task_id}) jackknife count = n_paired - 1 "
                   "(afd2d33 accepts)", outs / "c-04", A.C_MATRIX, run("c-04", task_results=tuple(
                       tasks)))

    def swap(fam):
        t = fam["bootstrap_tasks"]
        t[3], t[4] = t[4], t[3]
        return fam
    integrity_case("C5 order: tamper hook swaps task records 3 and 4 in family.json (afd2d33 "
                   "rejects too)", outs / "c-05", A.C_ORDER,
                   run("c-05", hook=tamper_at("payloads", lambda d: rewrite_family(d, swap))))
    tasks = list(env.tasks)
    j = BS.FROZEN_TASK_IDS.index("accuracy_e4_e5__median_delta")
    tasks[j] = dataclasses.replace(tasks[j], observed=float(np.nextafter(tasks[j].observed, 1.0)))
    integrity_case("C6 observed source: a Source-A observed value one ulp from its comparison "
                   "record, JSON and NPZ consistent (afd2d33 accepts)", outs / "c-06", A.C_SOURCE,
                   run("c-06", task_results=tuple(tasks)))
    tasks = list(env.tasks)
    j = BS.FROZEN_TASK_IDS.index("accuracy_e7_e6__hodges_lehmann_shift")
    spec, pv = BS.FROZEN_TASK_MATRIX[j], env.obs.paired["accuracy_e7_e6"]
    tasks[j] = BS.run_bootstrap_task(spec, env.analysis_id, pv.n, B_SMALL, tasks[j].observed,
                                     *BS.scalar_callables(spec.statistic_name, pv.delta),
                                     root_seed=43)
    integrity_case("C7 seed: one task drawn with root seed 43, its seed bytes self-consistent "
                   "(afd2d33 accepts)", outs / "c-07", A.C_SEED, run("c-07", task_results=tuple(
                       tasks)))

    def z0_ulp(fam):
        t = fam["bootstrap_tasks"][2]
        t["z0"] = float(np.nextafter(t["z0"], 1.0))
        return fam
    integrity_case("C8 json/npz: tamper hook moves one task's z0 one ulp in family.json only "
                   "(afd2d33 rejects too)", outs / "c-08", A.C_JSON_NPZ,
                   run("c-08", hook=tamper_at("payloads", lambda d: rewrite_family(d, z0_ulp))))

    def reshape(tmp):
        with np.load(tmp / A.BOOTSTRAP_NPZ, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        key = f"{BS.FROZEN_TASK_IDS[1]}__bootstrap"
        arrays[key] = arrays[key].reshape(-1, 1)
        np.savez_compressed(tmp / A.BOOTSTRAP_NPZ, **arrays)
    integrity_case("C9 npz schema: tamper hook stores one task's replicates as (B, 1) (afd2d33 "
                   "accepts)", outs / "c-09", A.C_NPZ, run("c-09", hook=tamper_at("payloads",
                                                                                  reshape)))

    def overflow(tmp):
        p = tmp / A.FAMILY_JSON
        p.write_text(p.read_text(encoding="utf-8").replace('"alpha": 0.05', '"alpha": 1e999', 1),
                     encoding="utf-8", newline="\n")
    integrity_case("C10 finite values: tamper hook writes the literal 1e999 for alpha (afd2d33 "
                   "accepts)", outs / "c-10", A.C_FINITE, run("c-10", hook=tamper_at("payloads",
                                                                                     overflow)))

    def flip(fam):
        sw = fam["software_environment"]
        sw["matches_pinned"] = not sw["matches_pinned"]
        return fam
    integrity_case("C11 officiality: tamper hook flips software_environment.matches_pinned "
                   "(afd2d33 accepts)", outs / "c-11", A.C_POLICY,
                   run("c-11", hook=tamper_at("payloads", lambda d: rewrite_family(d, flip))))
    integrity_case("C12 file set: tamper hook adds a fourth file after the manifest (afd2d33 "
                   "rejects too)", outs / "c-12", A.C_FILES,
                   run("c-12", hook=tamper_at("manifest", lambda d: (d / "notes.txt").write_text(
                       "x", encoding="utf-8"))))

    def symlink_family(tmp):
        target = outs.parent / "c12b_family.json"
        shutil.copyfile(tmp / A.FAMILY_JSON, target)
        (tmp / A.FAMILY_JSON).unlink()
        (tmp / A.FAMILY_JSON).symlink_to(target)
    integrity_case("C12b file set: tamper hook replaces family.json by a symlink to an identical copy "
                   "(afd2d33 accepts: it compared names only)", outs / "c-12b", A.C_FILES,
                   run("c-12b", hook=tamper_at("manifest", symlink_family)))
    (outs.parent / "c12b_family.json").unlink()

    def drop_line(tmp):
        lines = (tmp / A.MANIFEST_NAME).read_text(encoding="utf-8").splitlines(keepends=True)
        (tmp / A.MANIFEST_NAME).write_text(lines[1], encoding="utf-8", newline="\n")

    def third_line(tmp):
        text = (tmp / A.MANIFEST_NAME).read_text(encoding="utf-8")
        (tmp / A.MANIFEST_NAME).write_text(text + text.splitlines(keepends=True)[1],
                                           encoding="utf-8", newline="\n")
    integrity_case("C13a manifest: tamper hook removes the bootstrap.npz line (afd2d33 accepts)",
                   outs / "c-13a", A.C_MANIFEST,
                   run("c-13a", hook=tamper_at("manifest", drop_line)))
    integrity_case("C13b manifest: tamper hook appends a third (repeated) line (afd2d33 accepts)",
                   outs / "c-13b", A.C_MANIFEST,
                   run("c-13b", hook=tamper_at("manifest", third_line)))


def part_f(env: Env, plant: Env, off: Env, work: Path, outs: Path) -> None:
    """Acceptance (f): one refused case per writer-gate clause."""
    def write(e, run_id, synthetic, **kw):
        hooks = {k: kw.pop(k) for k in ("_inject_failure", "_tamper") if k in kw}
        return lambda: A.write_statistics_artifact(outs / run_id, e.inp(run_id, synthetic, **kw),
                                                   **hooks)
    reh = tuple(dataclasses.replace(r, policy=Policy.REHEARSAL) for r in env.obs.results)
    gate_case("F1 policy: the eight comparisons under REHEARSAL", outs / "f-01",
              ["policy_not_allowed"], write(env, "f-01", True, comparison_results=reh,
                                            descriptive=dataclasses.replace(
                                                env.obs.descriptive, policy="rehearsal")))
    mixed = (dataclasses.replace(env.obs.results[0], policy=Policy.OFFICIAL),) + env.obs.results[1:]
    gate_case("F2 policy: one comparison official, the rest nonofficial_smoke", outs / "f-02",
              ["mixed_policies"], write(env, "f-02", True, comparison_results=mixed))
    reworded = env.craw.replace(b"(`requirements.lock`: Python",
                                b"(pinned in requirements.lock: Python", 1)
    real = A.contract_sha256
    A.contract_sha256 = lambda repo_root=None: (hashlib.sha256(reworded).hexdigest(), reworded)
    try:
        gate_case("F3 contract pins: the section 10.1 sentence reworded (contract bytes swapped in "
                  "memory)", outs / "f-03", ["contract_pins"],
                  write(env, "f-03", True, contract_sha256=hashlib.sha256(reworded).hexdigest()))
    finally:
        A.contract_sha256 = real
    sha = {k: dict(v) for k, v in env.inp("x", True).input_file_sha256.items()}
    rel = sorted(sha)[5]
    sha[rel]["summary.json"] = "0" * 64
    gate_integrity_case("F4 P5: the driver's snapshot of one input differs from the files the gate "
                        "re-reads", outs / "f-04", A.C_PROVENANCE,
                        lambda h: write(env, "f-04", True, input_file_sha256=sha, _tamper=h))
    gate_integrity_case("F4b P5: a request without the driver's snapshot (input_file_sha256=None)",
                        outs / "f-04b", A.C_PROVENANCE,
                        lambda h: write(env, "f-04b", True, input_file_sha256=None, _tamper=h))
    short = {k: v for k, v in sha.items() if k != rel}
    gate_integrity_case("F4c P5: a driver snapshot that omits one input", outs / "f-04c",
                        A.C_PROVENANCE,
                        lambda h: write(env, "f-04c", True, input_file_sha256=short, _tamper=h))
    copy_root = work / "moved"
    shutil.copytree(env.fs.input_root, copy_root)
    moved = env.inp("f-05", True, input_root=copy_root)
    target = copy_root / sorted(sha)[7]
    F.edit_summary(target, lambda s: (s["run"].__setitem__("timestamp_utc",
                                                           "2026-10-03T01:02:03Z"), s)[1])
    gate_integrity_case("F5 P5: an input edited (manifest kept consistent) after the driver read it",
                        outs / "f-05", A.C_PROVENANCE,
                        lambda h: lambda: A.write_statistics_artifact(outs / "f-05", moved,
                                                                      _tamper=h))
    gate_integrity_case("F6 P7: declared k = 1 while the inputs exclude 0 images", outs / "f-06",
                        A.C_PROVENANCE,
                        lambda h: write(env, "f-06", True, am5_excluded_count=1, _tamper=h))
    gate_case("F7 P6: synthetic inputs declared not synthetic", outs / "f-07",
              ["synthetic_declaration"], write(env, "f-07", False))
    gate_case(f"F8 P6: inputs named {DATASET_NAME!r} declared synthetic", outs / "f-08",
              ["synthetic_declaration"], write(plant, "f-08", True))
    path = A.write_statistics_artifact(outs / "f-08b", plant.inp("f-08b", False))
    fam = json.loads((path / A.FAMILY_JSON).read_text(encoding="utf-8"))
    check(f"F9 P6: inputs named {DATASET_NAME!r} declared not synthetic -> written without "
          "synthetic_input_data", A.W_SYNTHETIC not in fam["warnings"]
          and fam["artifact_status"] == "nonofficial")

    # ---- official requests (running stack patched to the pinned one) ----
    o3_real = [n for n, _ in A.code_provenance_unmet()]
    numpy_drift = np.__version__ != A.PINNED_ENVIRONMENT["numpy"]
    unpinned_extra = ["numpy_version_differs"] if numpy_drift else []
    with pinned_stack():
        inp_off = off.inp("f-10", False)
        fam_off = A.build_family(inp_off)
        check("F10 an official-grade request derives status official (no warning)",
              fam_off["artifact_status"] == "official" and fam_off["warnings"] == [])
        expect = set(unpinned_extra + o3_real + [A.BINDING_CONDITION])
        gate_case("F11 a fully consistent official request (binding only; plus O3 entries on an "
                  "uncommitted tree" + ("; numpy_version_differs: the running numpy is not the "
                                        "pinned one" if numpy_drift else "") + ")",
                  outs / "f-10", expect,
                  lambda: A.write_statistics_artifact(outs / "f-10", inp_off), exact=True)
        calls = []
        gate_case("F12 an official request carrying test hooks (OK-3: refused in the gate, before the "
                  "temp directory exists; the hook never runs)", outs / "f-12",
                  {"test_hooks_on_official_request", A.BINDING_CONDITION},
                  lambda: A.write_statistics_artifact(
                      outs / "f-12", off.inp("f-12", False), _inject_failure=True,
                      _tamper=lambda d, s: calls.append(s)))
        check("F12b the tamper hook of a refused official request was never called", calls == [])
        gate_case("F12c an official request carrying only _inject_failure (OK-3)", outs / "f-12c",
                  {"test_hooks_on_official_request", A.BINDING_CONDITION},
                  lambda: A.write_statistics_artifact(outs / "f-12c", off.inp("f-12c", False),
                                                      _inject_failure=True))
        only_tamper = []
        gate_case("F12d an official request carrying only a (recording) _tamper hook (OK-3)",
                  outs / "f-12d", {"test_hooks_on_official_request", A.BINDING_CONDITION},
                  lambda: A.write_statistics_artifact(outs / "f-12d", off.inp("f-12d", False),
                                                      _tamper=lambda d, st: only_tamper.append(st)))
        check("F12e that hook was never called", only_tamper == [])
        views = A.load_input_views(fam_off["input_artifacts"], off.fs.input_root, Policy.OFFICIAL)
    clean_repo = work / "o3_clean"
    o3_fixture(clean_repo)
    conds = A.official_conditions(fam_off, views, off.craw, code_status_repo=clean_repo)
    base = {n for n, _ in conds}
    check("F13 official_conditions on that request with a clean code repository: exactly the "
          "binding" + (" (and numpy_version_differs: the running numpy is not the pinned one)"
                       if numpy_drift else ""),
          base == set(unpinned_extra + [A.BINDING_CONDITION]), sorted(base))
    official_clause_cases(fam_off, views, off, clean_repo, base)
    provenance_clause_cases(env)
    o3_cases(work)
    real_obs = A.observed_environment
    A.observed_environment = lambda: dict(real_obs(), numpy=real_obs()["numpy"] + "+drifted")
    try:
        fam_d = A.build_family(off.inp("f-20", False))
        path = A.write_statistics_artifact(outs / "f-20", off.inp("f-20", False))
    finally:
        A.observed_environment = real_obs
    on_disk = json.loads((path / A.FAMILY_JSON).read_text(encoding="utf-8"))
    check("F20 an unpinned running stack: the caller cannot claim the pin; the official-grade request "
          "becomes nonofficial with software_environment_unpinned only and is written as such "
          "(step 5 defects R1-R3 closed)",
          fam_d["warnings"] == [A.W_SOFTWARE] and fam_d["software_environment"]["matches_pinned"]
          is False and on_disk["artifact_status"] == "nonofficial"
          and on_disk["warnings"] == [A.W_SOFTWARE])
    kind, e = outcome(write(env, "f-21", True, _inject_failure=True))
    check("F21 test hook on a nonofficial request: the injected failure raises; nothing left",
          kind == "ArtifactError" and nothing_left(outs / "f-21"), f"{kind}: {e}")
    (outs / ".f-22.tmp-deadbeef0000").mkdir(parents=True)
    kind, e = outcome(write(env, "f-22", True))
    check("F22 a stale temporary sibling from an interrupted write -> refused (P10)",
          kind == "ArtifactError" and "stale" in str(e) and not (outs / "f-22").exists(),
          f"{kind}: {e}")
    kind, e = outcome(lambda: A.write_statistics_artifact(outs / "a-0001", env.inp("a-0001", True)))
    check("F23 an existing target -> refused", kind == "ArtifactError" and "overwrite" in str(e),
          f"{kind}: {e}")
    kind, e = outcome(lambda: A.write_statistics_artifact(outs / "f-24-other",
                                                          env.inp("f-24", True)))
    check("F24 run_id != directory basename -> refused", kind == "ArtifactError"
          and not (outs / "f-24-other").exists(), f"{kind}: {e}")
    kind, e = outcome(lambda: A.write_statistics_artifact(outs / "Bad_Run", env.inp("Bad_Run", True)))
    check("F25 a run_id outside the frozen pattern -> refused", kind == "ArtifactError",
          f"{kind}: {e}")
    lf = "f-25b\n"
    kind, e = outcome(lambda: A.write_statistics_artifact(outs / lf, env.inp(lf, True)))
    check("F25b a run_id with a trailing newline (section 12.4.2: no trailing whitespace) -> "
          "refused; nothing created", kind == "ArtifactError" and not (outs / lf).exists()
          and not A.stale_temp_dirs(outs / lf), f"{kind}: {e}")
    check("F26 the TEST-manifest binding constant is 'unbound'", A.TEST_MANIFEST_BINDING == "unbound")


def o3_fixture(root: Path) -> Path:
    for rel in A.OFFICIAL_CODE_PATHS[1:]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"{rel}\n", encoding="utf-8")
    (root / "src" / "stats" / "__init__.py").write_text("\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", "--", *A.OFFICIAL_CODE_PATHS[1:], "src/stats/__init__.py")
    git(root, "commit", "-q", "-m", "o3 fixture")
    return root


def o3_cases(work: Path) -> None:
    """O3: one test per refusal, each naming its path."""
    for i, rel in enumerate(A.OFFICIAL_CODE_PATHS):
        root = o3_fixture(work / f"o3_{i}")
        target = root / ("src/stats/extra.py" if rel == "src/stats" else rel)
        target.write_text("changed\n", encoding="utf-8")
        got = A.code_provenance_unmet(root)
        named = [d for n, d in got if n == "code_not_clean_at_head" and d.startswith(f"{rel}:")]
        check(f"F30.{i} O3: {rel} not clean at HEAD -> refused naming it", bool(named), got)
    root = o3_fixture(work / "o3_untracked")
    git(root, "rm", "-q", "--cached", "--", "scripts/run_stats.py")
    got = A.code_provenance_unmet(root)
    check("F31 O3: scripts/run_stats.py not committed -> code_not_committed naming it",
          ("code_not_committed", f"scripts/run_stats.py is not tracked in {root}") in got, got)
    plain = work / "o3_no_git"
    plain.mkdir()
    got = A.code_provenance_unmet(plain)
    check("F32 O3: no git repository -> code_provenance_unprovable",
          [n for n, _ in got] == ["code_provenance_unprovable"], got)
    check("F33 O3: a clean committed code repository -> no condition",
          A.code_provenance_unmet(work / "o3_clean") == [])


def provenance_clause_cases(env: Env) -> None:
    """P34 and P4: each clause of check_input_provenance -- the function the gate runs on the inputs it
    re-reads -- refuses by name. The loaded views are edited in memory and the records rebuilt from
    them (driver.provenance_record), so only the targeted clause can fire."""
    from src.eval.artifacts import hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    recs = list(env.inp("x", True).input_artifacts)
    order = [r["repo_relative_path"] for r in recs]
    views = A.load_input_views(recs, env.fs.input_root, env.policy)

    def edit(key, summary_fn=None, rows_fn=None):
        loaded, rows = dict(views.loaded), dict(views.rows)
        li = loaded[key]
        s = copy.deepcopy(li.summary)
        r = [dict(o) for o in rows[li.rel_path]]
        if rows_fn is not None:
            rows_fn(r, s)
        if summary_fn is not None:
            summary_fn(s)
        loaded[key] = dataclasses.replace(li, summary=s)
        rows[li.rel_path] = tuple(r)
        v = dataclasses.replace(views, loaded=loaded, rows=rows)
        by = v.by_path()
        return [D.provenance_record(by[p]) for p in order], v

    def rehash(r, s):
        s["dataset"]["split_manifest_sha256"] = hash_split_manifest(
            [ManifestEntry(o["manifest_index"], o["image_id"], o["clean_image_id"]) for o in r])

    def first(r, **kv):
        r[0] = dict(r[0], **kv)

    clean = {st: (st, D.CLEAN) for st in D.STAGES}
    cell = next(k for k in views.loaded if k[0] == "E1" and k[1] != D.CLEAN)
    e1_run_id = views.loaded[clean["E1"]].summary["run"]["run_id"]
    control = edit(clean["E3"])
    check("F17.0 control: the records rebuilt from unedited views pass check_input_provenance",
          outcome(lambda: A.check_input_provenance(*control))[0] == "ok")

    def stems(r, s):
        first(r, clean_image_id=r[0]["image_id"] + "_c")
        rehash(r, s)

    def other_manifest(r, s):
        first(r, image_id="zz_" + r[0]["image_id"], clean_image_id="zz_" + r[0]["image_id"])
        rehash(r, s)
    cases = [
        ("split manifest recomputed from the rows", "split manifest recomputed",
         edit(clean["E3"], rows_fn=lambda r, s: first(r, manifest_index=r[0]["manifest_index"]
                                                      + 100_000))),
        ("each row's condition", "carry another condition",
         edit(clean["E3"], rows_fn=lambda r, s: first(r, condition={"kind": "other"}))),
        ("bare stems (O1)", "bare stems", edit(clean["E3"], rows_fn=stems)),
        ("a per-image value that is a JSON int", "not a JSON float",
         edit(clean["E5"], rows_fn=lambda r, s: first(r, all_class_miou=1))),
        ("a per-image value above 1", "not a JSON float",
         edit(clean["E5"], rows_fn=lambda r, s: first(r, disease_only_miou=1.5))),
        ("student role and stage precision (Q11)", "role/precision",
         edit(clean["E4"], summary_fn=lambda s: s["run"].__setitem__("precision", "fp32"))),
        ("37 distinct run_ids", "run_ids repeat",
         edit(clean["E2"], summary_fn=lambda s: s["run"].__setitem__("run_id", e1_run_id))),
        ("one split manifest across the 37 (O1)", "split manifests; O1 requires one",
         edit(clean["E7"], rows_fn=other_manifest)),
        ("every E1/E6 cell carries its stage's clean checkpoint", "carry a checkpoint other",
         edit(cell, summary_fn=lambda s: s["run"].__setitem__("checkpoint_sha256", "f" * 64))),
    ]
    for i, (name, needle, (records, v)) in enumerate(cases, 1):
        kind, e = outcome(lambda records=records, v=v: A.check_input_provenance(records, v))
        check(f"F17.{i} gate clause: {name} -> refused: input_artifact_provenance_verified, naming "
              "it", kind == "IntegrityError" and e.check == A.C_PROVENANCE and needle in str(e),
              f"{kind}: {e}")


def official_clause_cases(fam, views, off: Env, clean_repo: Path, base: set) -> None:
    """Every official-request clause of P3/P4, by name, on a copy of one request's family or of one
    input's summary (official_conditions is the function the gate calls)."""
    def names(f=fam, v=views, contract=off.craw, hooks=False):
        conds = A.official_conditions(f, v, contract, hooks=hooks, code_status_repo=clean_repo)
        return {n for n, _ in conds}

    def fam_with(fn):
        f = copy.deepcopy(fam)
        fn(f)
        return f

    def views_with(rel_index, fn):
        rel = sorted(views.by_path())[rel_index]
        loaded = dict(views.loaded)
        for key, li in loaded.items():
            if li.rel_path == rel:
                s = copy.deepcopy(li.summary)
                fn(s)
                loaded[key] = dataclasses.replace(li, summary=s)
        return dataclasses.replace(views, loaded=loaded)

    cases = [
        ("policy_not_official", names(f=fam_with(lambda f: [c.__setitem__(
            "policy", "nonofficial_smoke") for c in f["a3a_family"]["comparisons"]]))),
        ("n_paired_plus_k_not_1561", names(f=fam_with(lambda f: f["a3a_family"]["comparisons"][
            2].__setitem__("n_paired", N_OFF)))),
        ("non_canvas_input", names(v=views_with(4, lambda s: s.__setitem__(
            "protocol", {"name": "upstream"})))),
        ("class_map_digest_differs", names(v=views_with(4, lambda s: s["dataset"].__setitem__(
            "class_map_sha256", "1" * 64)))),
        ("split_manifest_digest_differs", names(v=views_with(4, lambda s: s["dataset"].__setitem__(
            "split_manifest_sha256", "1" * 64)))),
        ("numpy_version_differs", names(f=fam_with(lambda f: f["bootstrap_tasks"][6].__setitem__(
            "numpy_version", "0.0.0")))),
        ("contract_pins_differ", names(contract=off.craw.replace(b"numpy 1.26.4,", b"numpy 1.26.5,",
                                                                 1))),
        ("contract_unavailable", names(contract=None)),
        ("pinned_block_differs", names(f=fam_with(lambda f: f["software_environment"][
            "pinned"].__setitem__("numpy", "0.0.0")))),
        ("governed_paths_not_clean", names(v=views_with(1, lambda s: s["run"].__setitem__(
            "governed_paths_clean", False)))),
        ("eval_runtime_missing", names(v=views_with(1, lambda s: s["run"].pop("eval_runtime")))),
        ("checkpoint_not_64_hex", names(v=views_with(1, lambda s: s["run"].__setitem__(
            "checkpoint_sha256", "ABC")))),
        ("repo_commit_not_40_hex", names(v=views_with(1, lambda s: s["run"].__setitem__(
            "repo_commit", "abc1234")))),
        ("split_not_test", names(v=views_with(1, lambda s: s["dataset"].__setitem__("split",
                                                                                    "val")))),
        ("rows_not_1561", names(v=views_with(1, lambda s: s["dataset"].__setitem__(
            "expected_rows", 1560)))),
        ("random_init", names(v=views_with(1, lambda s: s["run"].__setitem__("random_init",
                                                                             True)))),
        ("test_hooks_on_official_request", names(hooks=True)),
    ]
    for w in A.OFFICIALITY_WARNINGS:
        cases.append((w, names(f=fam_with(lambda f, w=w: f.__setitem__("warnings", [w])))))
    for clause, key, n in (("checkpoint_not_64_hex", "checkpoint_sha256", 64),
                           ("repo_commit_not_40_hex", "repo_commit", 40)):
        got = names(v=views_with(1, lambda s, key=key, n=n: s["run"].__setitem__(key,
                                                                                 "c" * n + "\n")))
        check(f"F14b official clause {clause}: {n} hex characters plus a trailing newline -> named",
              got == base | {clause}, sorted(got - base))
    metric = names(v=views_with(1, lambda s: s["run"].__setitem__("metric_impl_sha256", "2" * 64)))
    for clause, got in cases:
        check(f"F14 official clause {clause} -> named", got == base | {clause},
              sorted(got - base))
    check("F15 official clauses metric_impl_not_this_checkout and metric_impl_digest_differs -> "
          "named", metric == base | {"metric_impl_not_this_checkout", "metric_impl_digest_differs"},
          sorted(metric - base))
    check("F16 the binding is the last condition of every official request",
          A.official_conditions(fam, views, off.craw, code_status_repo=clean_repo)[-1][0]
          == A.BINDING_CONDITION)


def part_b(env: Env, off: Env, outs: Path) -> None:
    """Acceptance (b): count warnings."""
    with pinned_stack():
        tasks = list(off.tasks)
        i = 7
        tasks[i] = dataclasses.replace(tasks[i], replicates=tasks[i].replicates[:-1])
        fam = A.build_family(off.inp("b-01", False, task_results=tuple(tasks)))
        check(f"B1 task {i} ({tasks[i].task_id}) alone at B = 9,999 -> "
              "bootstrap_replicates_nonproduction (every task compared, not task 0)",
              fam["warnings"] == [A.W_B], fam["warnings"])
        fam = A.build_family(off.inp("b-02", False))
        t0 = fam["bootstrap_tasks"][0]
        check(f"B2 k = {K_OFF}: per-image counts 1561 - k = {N_OFF - K_OFF} and dataset-level counts "
              "1561 carry no count warning (the afd2d33 rule, task 0 against 1561, would have)",
              fam["warnings"] == [] and t0["jackknife_count"] == N_OFF - K_OFF
              and {t["jackknife_count"] for t in fam["bootstrap_tasks"]} == {N_OFF, N_OFF - K_OFF})
        fam = A.build_family(off.inp("b-03", False, am5_excluded_count=K_OFF - 1))
        check("B3 n_paired != 1561 - k (k declared one lower) -> jackknife_count_nonproduction",
              fam["warnings"] == [A.W_JACKKNIFE], fam["warnings"])
        gate_integrity_case("B4 ... and the writer refuses that k against the inputs' AM-5 count",
                            outs / "b-04", A.C_PROVENANCE,
                            lambda h: lambda: A.write_statistics_artifact(outs / "b-04", off.inp(
                                "b-04", False, am5_excluded_count=K_OFF - 1), _tamper=h))
    tasks = list(env.tasks)
    tasks[20] = dataclasses.replace(tasks[20], jackknife=tasks[20].jackknife[:-2])
    kind, e = outcome(lambda: A.build_family(env.inp("b-05", True, task_results=tuple(tasks))))
    check(f"B5 a per-image count != its comparison's n_paired (task 20, {tasks[20].task_id}) is "
          "fatal, never a warning", kind == "IntegrityError" and e.check == A.C_MATRIX,
          f"{kind}: {e}")
    bad = [outcome(lambda v=v: A.require_k(v))[0] for v in (True, -1, N_OFF, 3.0)]
    check("B6 k must be an int (not a bool) with 0 <= k < 1561", bad == ["ArtifactError"] * 4
          and A.require_k(0) == 0 and A.require_k(N_OFF - 1) == N_OFF - 1, bad)


def part_k(env: Env) -> None:
    raw = env.craw
    check("K1 the section 10.1 pins parsed from the contract equal PINNED_ENVIRONMENT",
          A.contract_pins(raw) == A.PINNED_ENVIRONMENT, A.contract_pins(raw))
    cases = [
        ("K2 the pinned-stack parenthetical reworded", raw.replace(
            b"(`requirements.lock`: Python", b"(pinned in requirements.lock: Python", 1),
         "parenthetical"),
        ("K3 numpy stated twice inside the parenthetical", raw.replace(
            b"numpy 1.26.4,", b"numpy 1.26.4, numpy 1.26.4,", 1), "numpy pin occurs 2"),
        ("K4 the scipy pin removed", raw.replace(b"scipy 1.11.4, statsmodels", b"statsmodels", 1),
         "scipy pin occurs 0"),
        ("K5 the section 10.1 heading duplicated", raw.replace(
            b"### 10.1 Software version policy", b"### 10.1 Software version policy x\n\n"
                                                 b"### 10.1 Software version policy", 1),
         "headings"),
    ]
    for name, data, needle in cases:
        kind, e = outcome(lambda d=data: A.contract_pins(d))
        check(f"{name} -> refused by name (no fallback)", kind == "ArtifactError"
              and needle in str(e), f"{kind}: {e}")
    kind, e = outcome(lambda: A.require_contract_pins(raw.replace(b"numpy 1.26.4,",
                                                                  b"numpy 1.26.5,", 1)))
    check("K6 a pin that differs from PINNED_ENVIRONMENT -> refused (double entry)",
          kind == "ArtifactError" and "differs" in str(e), f"{kind}: {e}")


def part_v(env: Env, outs: Path) -> None:
    src = outs / "a-0001"
    edited = outs / "v-01"
    shutil.copytree(src, edited)
    p = edited / A.FAMILY_JSON
    p.write_text(p.read_text(encoding="utf-8").replace('"alpha": 0.05', '"alpha": 0.050', 1),
                 encoding="utf-8", newline="\n")
    kind, e = outcome(lambda: A.verify_statistics_artifact(edited, final_name="a-0001"))
    check("V1 a post-write edit of family.json -> manifest_verified fails",
          kind == "IntegrityError" and e.check == A.C_MANIFEST, f"{kind}: {e}")

    def dup(tmp):
        q = tmp / A.FAMILY_JSON
        q.write_text(q.read_text(encoding="utf-8").replace(
            '"alpha": 0.05,', '"alpha": 0.05,\n  "alpha": 0.05,', 1), encoding="utf-8",
                     newline="\n")
    integrity_case("V2 duplicate keys in family.json", outs / "v-02", A.C_FINITE,
                   lambda: A.write_statistics_artifact(outs / "v-02", env.inp("v-02", True),
                                                       _tamper=tamper_at("payloads", dup)))

    with verifier_reads_official():
        got = [outcome(lambda kw=kw: A.verify_statistics_artifact(src, **kw)) for kw in (
            {"input_root": env.fs.input_root, "contract_bytes": env.craw}, {})]
    check("V3 an artifact the verifier reads as official with no warnings (relabelled in memory; "
          "nothing official on disk, P2) -> refused while the binding is unbound "
          "(officiality_policy_verified), with the inputs and the contract and artifact-only",
          all(k == "IntegrityError" and e.check == A.C_POLICY and "unbound" in str(e)
              for k, e in got), [f"{k}: {e}" for k, e in got])
    renamed = outs / "v-04"
    shutil.copytree(src, renamed)
    kind, e = outcome(lambda: A.verify_statistics_artifact(renamed))
    check("V4 an artifact whose directory name differs from its run_id -> refused",
          kind == "IntegrityError" and e.check == A.C_POLICY, f"{kind}: {e}")
    statuses = []
    for d in outs.iterdir():
        if (d / A.FAMILY_JSON).is_file():
            statuses.append(json.loads((d / A.FAMILY_JSON).read_text(encoding="utf-8"))[
                "artifact_status"])
    lf = outs / "v-06" / "a-0001"
    shutil.copytree(src, lf)
    rewrite_family(lf, lambda f: dict(f, run_id="a-0001\n"))
    (lf / A.MANIFEST_NAME).write_text(A._manifest_text(lf), encoding="utf-8", newline="\n")
    kind, e = outcome(lambda: A.verify_statistics_artifact(lf, final_name="a-0001\n"))
    check("V6 an artifact whose run_id (and directory name) end in a newline -> refused "
          "(officiality_policy_verified; section 12.4.2)",
          kind == "IntegrityError" and e.check == A.C_POLICY, f"{kind}: {e}")
    check("V5 no statistics artifact written by this smoke has status official",
          statuses and set(statuses) == {"nonofficial"}, statuses)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="smoke_stats_integrity_"))
    outs = work / "out"
    try:
        smoke = Env(work / "smoke", n=N_SMALL, k=0, split="val", status="smoke",
                    policy=Policy.NONOFFICIAL_SMOKE, analysis_id=BS.ANALYSIS_ID_SMOKE, B=B_SMALL,
                    name=F.FIXTURE_DATASET_NAME, prefix="smoke")
        plant = Env(work / "plant", n=N_SMALL, k=0, split="val", status="smoke",
                    policy=Policy.NONOFFICIAL_SMOKE, analysis_id=BS.ANALYSIS_ID_SMOKE, B=B_SMALL,
                    name=DATASET_NAME, prefix="plant")
        off = Env(work / "off", n=N_OFF, k=K_OFF, split="test", status="official",
                  policy=Policy.OFFICIAL, analysis_id=BS.ANALYSIS_ID_OFFICIAL, B=BS.PRODUCTION_B,
                  name=DATASET_NAME, prefix="off", placeholder=True)
        part_a(smoke, outs)
        part_w(smoke, outs)
        part_c(smoke, outs)
        part_f(smoke, plant, off, work, outs)
        part_b(smoke, off, outs)
        part_k(smoke)
        part_v(smoke, outs)
    except Exception:                                                    # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check("Z1 temporary fixtures and artifacts removed", not work.exists())

    print()
    for name, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if det and not ok:
            print(f"         {det[:400]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'STATS INTEGRITY OK' if allok else 'STATS INTEGRITY FAILED'} "
          f"({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic fixtures only; "
          + ("pinned statistics stack." if PINNED_STACK else "running stack differs from the pin."))
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
