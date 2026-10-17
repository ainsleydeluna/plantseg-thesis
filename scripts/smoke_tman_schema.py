#!/usr/bin/env python3
"""Smoke for the TEST-day manifest's schema, names, canonical JSON, catalogue and score-blind readers (lane TMAN,
B-1, C1). Also the fixture library of the TMAN smokes (scratch repositories and synthetic records): the lane's file
list has no separate fixture module, so the other smokes import this one.

Synthetic inputs only: no dataset, no checkpoint of record, no GPU, no network. Scratch folders lie under the system
temp dir, named tman<kind>_<pid>_<n> (no random suffix can spell "test"), and are removed at the end; scratch
repositories are git repositories there, committed with fixed dates and no user or system configuration. Nothing is
written in the repository; the committed catalogue and selection files are read, identity fields only.

Run:  python -B scripts/smoke_tman_schema.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GIT_DATE = "2026-10-16T08:00:00+00:00"
_SCRATCH = itertools.count(1)
SCRATCH_DIRS: list[Path] = []
RESULTS: list[tuple[str, bool, str]] = []
LANE_MODULES = ("src/tman/__init__.py", "src/tman/schema.py", "src/tman/records.py", "src/tman/completion.py",
                "src/tman/schedule.py", "src/tman/entries.py", "src/tman/manifest.py", "src/tman/validate.py",
                "scripts/tday_manifest.py")


# ----------------------------------------------------------------------------------------- check machinery
def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def refused_code(fn, *a, **kw):
    """The TmanRefused code `fn` raises, or None when it returns."""
    from src.tman.schema import TmanRefused
    try:
        fn(*a, **kw)
    except TmanRefused as e:
        return e.code
    return None


def expect_refused(name: str, want: str, fn, *a, **kw) -> None:
    got = refused_code(fn, *a, **kw)
    check(name, got == want, f"got {got!r}, expected {want!r}")


def run_cases(title: str, cases) -> int:
    """Run each case (a case that raises fails by name), remove the scratch folders, print the checks and the
    RESULT line; returns the exit code."""
    print("=" * 78)
    print(f"{title} -- synthetic inputs only; TEST is never read")
    print("=" * 78)
    for fn in cases:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- a case that raises fails by name
            check(f"{fn.__name__}_raised", False, f"{type(e).__name__}: {e}")
    check("scratch_folders_removed", remove_scratch(), str([str(p) for p in SCRATCH_DIRS if p.exists()]))
    print("\n[CHECKS]")
    for name, ok, detail in RESULTS:
        print(f"  {name:62}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nRESULT: {'PASS' if passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if passed == len(RESULTS) else 1


# ------------------------------------------------------------------------------------------ scratch folders
def scratch_dir(prefix: str = "tman") -> Path:
    """A new folder <tmp>/<prefix>_<pid>_<n>, outside the repository; its path never contains "test"."""
    while True:
        p = Path(tempfile.gettempdir()) / f"{prefix}_{os.getpid()}_{next(_SCRATCH)}"
        if "test" in str(p).lower():
            raise SystemExit(f"refusing to create {p}: its path contains 'test'; set TMPDIR elsewhere")
        if REPO == p.resolve() or REPO in p.resolve().parents:
            raise SystemExit(f"refusing to create {p} inside the repository")
        try:
            p.mkdir()
        except FileExistsError:
            continue
        SCRATCH_DIRS.append(p)
        return p


def remove_scratch() -> bool:
    """Remove the folders scratch_dir made in this process (each directly under the temp dir, named
    tman<kind>_<pid>_<n>); True when none is left."""
    tmp = Path(tempfile.gettempdir()).resolve()
    for p in SCRATCH_DIRS:
        if p.resolve().parent == tmp and re.fullmatch(rf"tman[a-z]*_{os.getpid()}_\d+", p.name):
            shutil.rmtree(p, ignore_errors=True)
    return not any(p.exists() for p in SCRATCH_DIRS)


def git(repo: Path, *args: str, date: str | None = None) -> str:
    """git in a scratch repository only: -C <scratch>, fixed dates, no user or system configuration."""
    repo = Path(repo).resolve()
    assert repo != REPO and REPO not in repo.parents, repo
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_DATE=date or GIT_DATE,
               GIT_COMMITTER_DATE=date or GIT_DATE)
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=tman-smoke", "-c", "user.email=tman@localhost",
                        "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=master", *args],
                       capture_output=True, env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout.decode("utf-8", "replace").strip()


def new_repo() -> Path:
    repo = scratch_dir("tmanrepo")
    git(repo, "init", "-q")
    return repo


def commit_files(repo: Path, files: dict, msg: str = "records", *, date: str | None = None) -> str:
    """Write {path: bytes|str|dict} into `repo`, stage those paths by name, commit; returns the commit. A dict is
    written as a record file (schema.json_file_bytes)."""
    from src.tman.schema import json_file_bytes
    for rel, data in files.items():
        f = Path(repo) / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, dict):
            data = json_file_bytes(data)
        f.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    if files:
        git(repo, "add", "--", *files)
    git(repo, "commit", "-q", "--allow-empty", "-m", msg, date=date)
    return git(repo, "rev-parse", "HEAD")


def dl_table(rows: dict) -> str:
    """A decision log with one row per {DL id: (status, text)}."""
    lines = ["# Decision log (synthetic; TMAN smokes)", "", "| ID | Status | Decision | Where | Ch4 |",
             "|---|---|---|---|---|"]
    lines += [f"| {dl} | {st} | {text} | - | no |" for dl, (st, text) in rows.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- score sentinel (vi)
SCORE_KEY_RE = re.compile(r"(best_val|miou|^value$|^values$|fake_quant|^loss$|grad_norm|rule_trace|^tied|band|"
                          r"^ece|val_x|ece_x|ece_b|all_class|disease_only|per_class|^ce$|^dice$|accuracy|^conditions$)",
                          re.I)


class SentinelDict(dict):
    """A parsed JSON object that raises when a reader touches a score key's value (plan decision (vi) 3). Keys may
    be listed; a value is reached only through __getitem__, get, items or values (iteration is overridden, so a
    copy by dict() or ** goes through __getitem__ too)."""

    def __getitem__(self, k):
        if isinstance(k, str) and SCORE_KEY_RE.search(k):
            raise AssertionError(f"a reader read the score key {k!r}")
        return dict.__getitem__(self, k)

    def __iter__(self):
        return iter(list(dict.keys(self)))

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]


def with_sentinel(fn):
    """Run `fn` with the readers' JSON objects as SentinelDict; returns fn's result."""
    from src.tman import records as R
    saved = R.JSON_OBJECT
    R.JSON_OBJECT = SentinelDict
    try:
        return fn()
    finally:
        R.JSON_OBJECT = saved


# --------------------------------------------------------------------------------------- run-directory fixtures
def utc(s: str) -> float:
    """An ISO instant 'YYYY-MM-DDTHH:MM:SS[.f]Z' as Unix time."""
    from datetime import datetime, timezone
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in s else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()


def write_jsonl(path: Path, rows: list, torn: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(json.dumps(r, allow_nan=True).encode("utf-8") + b"\n" for r in rows) + torn)
    return path


def kd_run(root: Path, run_id: str, *, stage_key: str = "e2", end: str | None = "2026-10-20T10:00:00Z",
           checks: bool = True, abort: dict | None = None, iters: int = 3) -> Path:
    """A KD run directory: run_meta and telemetry with train rows (loss, iter_seconds), VAL rows with scores and,
    when `end`, a run_end record whose best_val is a score; or an abort record."""
    d = Path(root) / run_id
    t0 = utc("2026-10-10T00:00:00Z")
    rows = [{"event": "train", "iter": i, "loss": 2.0 / i, "grad_norm": 1.5, "iter_seconds": 0.7 + 0.01 * i,
             "wall_clock": t0 + i} for i in range(1, iters + 1)]
    rows.append({"event": "val", "iter": iters, "all_class_miou": 0.31, "wall_clock": t0 + iters + 0.5})
    if abort:
        abort = dict(abort)
        rows.append({"event": "run_abort", "iter": iters, "detail": {"loss": float("nan")},
                     "wall_clock": utc(abort.pop("at", "2026-10-15T00:00:00Z")), **abort})
    elif end:
        rows.append({"event": "run_end", "iter": iters, "best_val_miou_all_class": 0.31, "best_ckpt": "iter_3.pth",
                     "checks_passed": checks, "wall_clock_start": t0, "wall_clock_end": utc(end),
                     "wall_seconds": 9.0, "train_seconds": 8.0, "gpu_hours": 0.01})
    write_jsonl(d / f"{stage_key}_telemetry.jsonl", rows)
    write_jsonl(d / f"{stage_key}_run_meta.jsonl", [{"event": "run_meta", "stage": stage_key.upper(), "mode": "real",
                                                    "seed": 42, "wall_clock": t0, "max_iters": iters,
                                                    "lambda_logit": 0.5, "image_digest": "sha256:" + "c" * 64}])
    return d


def e1_run(root: Path, run_id: str, *, end: str | None = "2026-09-20T10:00:00Z", result: str | None = "PASS",
           max_iters: int = 4) -> Path:
    """An E1 run directory: e1_telemetry.jsonl (run_meta, train rows, a VAL row with scores, at max_iters when
    `end`) and e1_stdout.log with its RESULT line when `result`."""
    d = Path(root) / run_id
    t0 = utc("2026-09-18T00:00:00Z")
    rows = [{"event": "run_meta", "wall_clock": t0, "mode": "real", "seed": 42, "max_iters": max_iters,
             "git_head": "0" * 40, "image_digest": "sha256:" + "b" * 64}]
    rows += [{"event": "train", "iter": i, "loss": 1.0, "iter_seconds": 0.5, "wall_clock": t0 + i}
             for i in range(1, max_iters + 1)]
    last = max_iters if end else max_iters - 2
    rows.append({"event": "val", "iter": last, "all_class_miou": 0.36, "per_class_iou": [0.1] * 3,
                 "wall_clock": utc(end) if end else t0 + 100})
    write_jsonl(d / "e1_telemetry.jsonl", rows)
    if result is not None:
        (d / "e1_stdout.log").write_text(f"[summary] best_all_class_val_miou=0.36314\n\nRESULT: {result} (6/6 checks "
                                         "exercised, 0 skipped)\n", encoding="utf-8")
    return d


def qat_run(root: Path, run_id: str, *, epochs: int = 15, end: str = "2026-10-25T12:00:00Z", stop: bool = False,
            nonfinite_from: int | None = None, complete_flag: bool = True, stage: str = "E6",
            torn: bool = False) -> Path:
    """A QAT run directory: qat_telemetry.jsonl with run_meta, train and fake-quant VAL rows (scores), epoch_end
    rows and, when asked, a non-finite state, a run_stop or a torn last line."""
    d = Path(root) / run_id
    t0 = utc("2026-10-24T00:00:00Z")
    rows = [{"event": "run_meta", "wall_clock": t0, "mode": "real", "stage": stage, "source_stage": "E3",
             "seed": 42, "run_id": run_id, "clip_norm": 5.0, "clip_source": "clip_selection",
             "source_checkpoint_sha256": "c" * 64, "image_digest": "sha256:" + "b" * 64, "git_head": "0" * 40}]
    since = None
    for e in range(1, epochs + 1):
        rows.append({"event": "train", "epoch": e, "step": e * 10, "loss": 0.5, "grad_norm": 2.0,
                     "iter_seconds": 0.3, "wall_clock": t0 + e * 100})
        if nonfinite_from == e:
            since = e * 10
            rows.append({"event": "state_nonfinite", "wall_clock": t0 + e * 100 + 1, "epoch": e, "step": since,
                         "nonfinite_since_step": since, "failing": ["x"]})
        rows.append({"event": "val", "epoch": e, "all_class_miou": 0.3, "kind": "fake_quant",
                     "wall_clock": t0 + e * 100 + 2})
        last = e == epochs
        rows.append({"event": "epoch_end", "wall_clock": utc(end) if (last and epochs == 15) else t0 + e * 100 + 3,
                     "epoch": e, "step": e * 10, "checkpoint": f"epoch_ckpts/e{e:02d}.pt",
                     "checkpoint_sha256": f"{e:064x}", "checkpoint_bytes": 10, "state_sha256": f"{e + 100:064x}",
                     "fake_quant_val_all_class_miou": 0.3, "max_pre_clip_norm": 3.0,
                     "state_finite": since is None, "nonfinite_since_step": since, "state_nonfinite": since is not None,
                     "run_complete": complete_flag and e == 15})
    if stop:
        rows.append({"event": "run_stop", "wall_clock": t0 + epochs * 100 + 5, "epoch": epochs, "step": epochs * 10,
                     "code": "x", "message": "stopped"})
    write_jsonl(d / "qat_telemetry.jsonl", rows, b'{"event": "tr' if torn else b"")
    return d


def teacher_run(root: Path, work: str, *, final: str | None = "10/13 05:26:26", with_year: bool = False,
                folder: str = "20261012_010203") -> Path:
    """A teacher work dir: an mmengine log with a train line and, when `final`, the M12 "(final)" line (a score on
    it, which no reader reads)."""
    d = Path(root) / work / folder
    d.mkdir(parents=True, exist_ok=True)
    lines = ["10/12 01:02:03 - mmengine - INFO - Iter(train) [   50/40000]  lr: 1.0e-05  time: 1.1  loss: 2.5"]
    if final:
        stamp = ("2026/" + final) if with_year else final
        lines.append(f"{stamp} - mmengine - INFO - M12 selection (final): iteration 20000 "
                     "mIoU_full=0.38849183917045593 sha256=" + "1535c289" + "0" * 56)
    (d / f"{folder}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(root) / work


def selection_doc(key: str, winner: float, values: list) -> dict:
    """A lambda/alpha selection file with its score fields (which no reader may touch)."""
    field = "lambda" if key == "lambda_logit" else "alpha"
    cands = [{field: v, "run_id": f"run_{field}{v:g}", "best_val": 0.3 + 0.01 * i, "ckpt_sha256": f"{i + 1:064x}",
              "checkpoint_present": True} for i, v in enumerate(values)]
    w = next(c for c in cands if c[field] == winner)
    return {"format": f"{field}_selection/1", "winner": dict(w), "candidates": cands,
            "rule_trace": ["best_val 0.31 > 0.30"], "band": 0.005}


def ptq_record_doc(stage: str = "E4", seed: int = 42, role: str = "of_record") -> dict:
    """A ptq_record/1-shaped record (lane LQP's keys) whose parent block holds a VAL score."""
    return {"format": "ptq_record/1", "stage": stage, "seed": seed, "role": role, "descriptive": role != "of_record",
            "replicate_of": None, "authority": "AM-10", "parent": {"run_id": "e1_s42", "checkpoint_sha256": "1" * 64,
                                                                    "best_val_miou_all_class": 0.36},
            "calibration": {"list_file_sha256": "2" * 64, "checksum_sha256": "3" * 64, "seed": 42,
                            "role": "am10_list_of_record"},
            "ptq_run": {"run_meta": "e4_ptq_run_meta.json", "sha256": "4" * 64},
            "artifacts": {k: {"path": f"e4_{k}.bin", "sha256": c * 64, "bytes": 1}
                          for k, c in (("torchscript", "5"), ("state_dict", "6"), ("x86_copy", "7"))},
            "qconfig_fingerprint": "abc", "observers": {}, "engine": "qnnpack", "environment": {},
            "isolation": {"result": "ok"}, "writer": {"commit": "8" * 40}, "generated_utc": "2026-11-04T00:00:00Z"}


def ptq_reader_stub(doc: dict, *, sentinel: bool = True):
    """A stand-in for lane LQP's read_record(path, expect_sha256) -> (record, sha256) until lane/lqp merges: it
    returns `doc` (built of SentinelDict objects when `sentinel`) with the sha256 of its record-file bytes."""
    from src.tman import schema as S
    data = S.json_file_bytes(doc)

    def read(path, expect_sha256=None):
        sha = S.sha256_bytes(data)
        if expect_sha256 is not None and sha != expect_sha256:
            raise ValueError(f"{path} has sha256 {sha}, expected {expect_sha256}")
        return (S.strict_json(data, str(path), code="ptq_record_invalid",
                              object_type=SentinelDict if sentinel else dict), sha)
    return read, S.sha256_bytes(data)


# ======================================================================================================= cases
def case_formats():
    from src.tman import schema as S
    check("formats_named", (S.MANIFEST_FORMAT, S.ENTRY_FORMAT, S.LANES_FORMAT, S.CATALOGUE_FORMAT,
                            S.INVENTORY_FORMAT, S.BINDING_FORMAT, S.ISOLATION_FORMAT) ==
          ("tday_manifest/1", "tday_entry/1", "tday_lanes/1", "tday_catalogue/1", "tday-inventory/1", "tday_binding/1",
           "tday_isolation/1"))
    check("binding_reserves_mde_entry_TB11_and_artifacts_A1",
          S.BINDING_RESERVED == {"references": ("mde_entry",), "artifacts": (
              "cell", "checkpoint_sha256", "files_sha256", "precision", "protocol", "repo_commit",
              "repo_relative_path", "scope", "slot", "split_manifest_sha256")})
    check("binding_scopes_exclude_dedup_A1", "dedup" not in S.BINDING_SCOPES and set(S.BINDING_SCOPES) < set(S.SCOPES))
    check("expected_rows_1561_TB14", S.EXPECTED_ROWS == 1561)
    check("arm_latency_dropped_TB10", "arm_latency" not in S.SCOPES and "fq_off" in S.SCOPES, str(S.SCOPES))
    check("quant_descriptive_and_e6kd_roles", "quant_descriptive" in S.ROLES and "e6kd" in S.ROLES)
    check("kinds_by_class_cover_entry_kinds",
          sorted({k for v in S.KINDS_BY_CLASS.values() for k in v}) == sorted(S.ENTRY_KINDS))
    check("entry_specs_cover_kinds", sorted(S.ENTRY_FIELDS) == sorted(S.ENTRY_KINDS))
    check("not_made_kind_A1", "not_made" in S.ENTRY_KINDS and "not_made" in S.KINDS_BY_CLASS["derived"])
    check("non_finite_on_the_run_class_A1", "non_finite" in S.KINDS_BY_CLASS["run"]
          and "non_finite" not in S.KINDS_BY_CLASS["derived"])
    check("terminal_kinds_TB10", set(S.TERMINAL_KINDS) == {"cut", "not_run", "non_finite", "superseded", "diverged",
                                                           "aborted", "not_made"})
    check("cut_rules_b1", S.CUT_RULES == ["freeze", "decision_date", "named_rule", "arithmetic"])
    check("inventory_precisions_TB5", S.INVENTORY_PRECISIONS == ("fp32", "int8_ptq", "int8_qat",
                                                                "qat_fakequant_off"))
    check("image_digests_of_record_TB1", S.KD_IMAGE == "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b1168"
                                                       "7bb9eb23dc32c2b" and S.KD_IMAGE in S.IMAGE_DIGESTS_OF_RECORD)
    check("ptq_role_of_record_A2", S.PTQ_ROLE_OF_RECORD == "of_record"
          and set(S.PTQ_RECORD_ROLES) == {"of_record", "calibration_sensitivity", "d3_replicate"})
    check("cutoff_moving_gates_subset", set(S.CUTOFF_MOVING_GATES) < set(S.GATE_RECORDS))
    check("exit_codes", (S.EXIT_PASS, S.EXIT_REFUSED, S.EXIT_MISSING, S.EXIT_ERROR) == (0, 2, 3, 4))
    check("refusal_line_form", S.TmanRefused("x_code", "why").line() == "REFUSED [x_code] -- why")


def _entry(base: str, **kw) -> dict:
    """A well-formed manifest entry of kind `base` (IF§11.2), with `kw` replacing fields."""
    sha, ref = "a" * 64, {"path": "reports/derived/x.json", "sha256": "b" * 64}
    run = {"run_id": "e2_s42_lambda0.5_80k_a1", "attempts": [{"run_id": "e2_s42_lambda0.5_80k_a1", "attempt": 1}],
           "completion": {"record": "kd_run_end", "at_utc": "2026-10-20T10:00:00Z", "sha256": sha, "line": 7}}
    ckpt = {"name": "best.pth", "sha256": sha, "bytes": 10,
            "sources": [{"kind": "run_dir", "path": "e2/best.pth", "dl": "DL-109", "sha256": sha}]}
    scored = {**run, "checkpoint": ckpt, "evaluator_stage": "E2", "teacher_ckpt_sha256": None}
    own = {"scored_run": scored, "default_candidate": {**scored, "selection": ref},
           "arm_teacher": {**scored, "arm_record": ref, "launch_provenance": ref,
                           "go_rule": {"path": "x.json", "sha256": sha, "verdict": "go"}},
           "listed_run": {**run, "checkpoint": None, "reason": "not the winner"},
           "derived": {"method": "ptq", "parent": "E1:s42:80k", "inputs": {"list": sha}, "hashed": None,
                       "evaluator_stage": "E4", "ptq_record": None},
           "not_run": {"reason": "parent cut", "parent": "E3-R1:s42:lambda0.5:alpha25:80k", "parent_record": ref},
           "non_finite": {**run, "telemetry_sha256": sha, "refusal_text": "non-finite: no model"},
           "cut": {"entry": ref, "cut_date": "2026-11-03", "rule": "freeze", "comparator": None},
           "superseded": {"run_id": "x", "superseding_dl": "DL-110"},
           "diverged": {"run_id": "x", "rule": "AM-7(b)", "cause": "non-finite loss", "iteration": 1200,
                        "epoch": None, "telemetry_sha256": sha},
           "aborted": {"run_ids": ["x"], "rule": "AM-7(a)", "cause": "plateau"},
           "measurement": {"output": ref, "agreement": ref, "completion": run["completion"]},
           "not_made": {"reason": "code not merged", "entry": ref}}[base]
    doc = {"slot": "E2:s42:lambda0.5:80k", "kind": base, "class": "run", "stage": "E2", "seed": 42,
           "catalogue": {"never_cut": True, "cut_place": None, "parent": None}, "dl_rows": ["DL-109"],
           "scope": ["clean", "upstream", "dedup"], "role": ["family"], **own}
    doc.update(kw)
    return doc


def case_entry_schema():
    from src.tman import schema as S
    bad = {k: S.entry_errors(_entry(k)) for k in S.ENTRY_KINDS}
    check("entry_kinds_well_formed_IF11_2", not any(bad.values()), str({k: v for k, v in bad.items() if v}))
    muts = {
        "unknown_kind": _entry("cut", kind="launch"),
        "unexpected_key": {**_entry("cut"), "value": 0.3},
        "missing_common": {k: v for k, v in _entry("cut").items() if k != "catalogue"},
        "cut_place_not_int": _entry("cut", catalogue={"never_cut": False, "cut_place": "1", "parent": None}),
        "catalogue_extra_key": _entry("cut", catalogue={"never_cut": False, "cut_place": 1, "parent": None,
                                                        "arm_cut_place": 1}),
        "comparator_not_slot": _entry("cut", comparator="E2"),
        "cut_date_not_day": _entry("cut", cut_date="2026-11-3"),
        "scope_unknown": _entry("cut", scope=["clean", "arm_latency"]),
        "role_unknown": _entry("cut", role=["winner"]),
        "dl_row_malformed": _entry("cut", dl_rows=["DL109"]),
        "diverged_both_iteration_and_epoch": _entry("diverged", epoch=3),
        "diverged_neither": _entry("diverged", iteration=None),
        "not_made_other_reason": _entry("not_made", reason="later"),
        "non_finite_other_text": _entry("non_finite", refusal_text="nan"),
        "go_rule_verdict_unknown": _entry("arm_teacher", go_rule={"path": "x", "sha256": "a" * 64, "verdict": "ok"}),
        "attempt_not_int": _entry("scored_run", attempts=[{"run_id": "x", "attempt": True}]),
        "never_cut_not_bool": _entry("cut", catalogue={"never_cut": "no", "cut_place": None, "parent": None}),
        "sha256_uppercase": _entry("diverged", telemetry_sha256="A" * 64),
        "class_unknown": _entry("cut", **{"class": "arm"}),
    }
    for name, doc in muts.items():
        check(f"entry_schema_refuses_{name}", bool(S.entry_errors(doc)))
    check("entry_seed_null_allowed_measurement", not S.entry_errors(_entry("measurement", slot="ARM:latency",
                                                                           seed=None, **{"class": "measurement"})))
    table = {k: dict(v) for k, v in S.ENTRY_FIELDS.items()}
    S.register_kind(table, "cut", {"cutoff_rule": "str"})
    S.register_kind(table, "freeze_check", {"verdict": "str"})
    check("field_specs_additive_T19", table["cut"]["cutoff_rule"] == "str" and "rule" in table["cut"]
          and table["freeze_check"] == {"verdict": "str"} and "cutoff_rule" not in S.ENTRY_FIELDS["cut"])


# ---------------------------------------------------------------------- IF§9, transcribed (the table of record)
# who -> (clean, upstream, dedup, corrupted, corrupted_sev45, efficiency, fq_off); "Y" required, "N" forbidden
IF9 = {
    "E1:s42:80k": "YYYYYYN", "E2:s42:lambda0.5:80k": "YYYYYYN", "E3:s42:lambda0.5:alpha25:80k": "YYYYYYN",
    "E4:s42:list42": "YYYYYYN", "E5:s42:sel": "YYYYYYY", "E6:s42:sel": "YYYYYYY", "E7:s42:list42": "YYYYYYN",
    "teacher:s42:40k": "YYYYYYN",
    "E2:s43:lambda0.5:80k": "YYYNNNN", "E3:s44:lambda0.5:alpha25:80k": "YYYNNNN", "E4:s43:list42": "YYYNNNN",
    "E5:s44:sel": "YYYNNNY", "E7:s43:list42": "YYYNNNN",
    "E2:s42:lambda1:80k": "YYYNNNN", "E3:s42:lambda0.5:alpha50:80k": "YYYNNNN", "E1:s42:160k": "YYYNNNN",
    "A:s42:alpha25:80k": "YYYNNNN", "G:s42:80k": "YYYNNNN", "E3-R1:s42:lambda0.5:alpha25:80k": "YYYNNNN",
    "E7-R2:s42:list42": "YYYNNNN", "E3-flip:s42:lambda0.5:alpha25:80k": "YYYNNNN", "E7-flip:s42:list42": "YYYNNNN",
    "E6-KD:s42:sel": "YYYNNNY",
    "E6-E2:s42:sel": "YYYNNNY", "E7-E2:s42:list42": "YYYNNNN", "E6-R1:s42:sel": "YYYNNNY",
    "E6-R2:s42:sel": "YYYNNNY",
    "E4:s42:list43": "YNYNNNN", "E7:s42:list45": "YNYNNNN",
    "teacher-R1:s42:40k": "YNYNNNN", "teacher-R2:s42:40k": "YNYNNNN",
    "E5:s42:clip1:15ep": "NNNNNNN", "E2:s42:lambda4:80k": "NNNNNNN", "teacher-L:s42:40k": "NNNNNNN",
    "ARM:latency": "NNNNNNN",
}
# E1 and E6 at seeds 43/44: corrupted only when both runs of the seed have a model (TB2)
IF9_PAIR = {("E1:s43:80k", True): "YYYYNNN", ("E1:s43:80k", False): "YYYNNNN",
            ("E6:s44:sel", True): "YYYYNNY", ("E6:s44:sel", False): "YYYNNNY"}


def _catalogue():
    from src.tman import schema as S
    return S.load_catalogue((REPO / S.CATALOGUE_REL).read_bytes())


def case_scope_table():
    from src.tman import schema as S
    cat = _catalogue()
    bad = []
    for slot, row in IF9.items():
        req, forb = S.scope_rule(cat.slots[slot])
        for sc, flag in zip(S.SCOPES, row):
            if (flag == "Y") != (sc in req) or (flag == "N") != (sc in forb):
                bad.append(f"{slot}:{sc}")
    check("scope_rule_reproduces_IF9_rows", not bad, f"differs at {bad}")
    bad = []
    for (slot, both), row in IF9_PAIR.items():
        req, forb = S.scope_rule(cat.slots[slot], pair_has_models=both)
        for sc, flag in zip(S.SCOPES, row):
            if (flag == "Y") != (sc in req) or (flag == "N") != (sc in forb):
                bad.append(f"{slot}/{both}:{sc}")
    check("scope_rule_pair_corrupted_only_with_both_models_TB2", not bad, f"differs at {bad}")
    check("scope_rule_every_scope_decided", all(S.scope_rule(s, b)[0] | S.scope_rule(s, b)[1] == S.ALL_SCOPES
                                                and not S.scope_rule(s, b)[0] & S.scope_rule(s, b)[1]
                                                for s in cat.slots.values() for b in (True, False)))
    check("dedup_required_wherever_clean_TB4", all(("dedup" in S.scope_rule(s, b)[0]) == ("clean" in
                                                                                         S.scope_rule(s, b)[0])
                                                   for s in cat.slots.values() for b in (True, False)))
    family = [s for s in cat.slots.values() if s["scope_class"] in ("family_s42", "teacher_of_record")]
    check("robust_scopes_required_for_family_and_teacher_TB4",
          len(family) == 8 and all({"corrupted", "corrupted_sev45", "efficiency"} <= S.scope_rule(s)[0]
                                   for s in family), str(sorted(s["slot"] for s in family)))
    students = [s for s in cat.slots.values() if s["model_role"] == "student" and s["scope_class"] not in
                ("none", "not_built", "calibration_sensitivity")]
    check("upstream_required_for_every_student_TB3", all("upstream" in S.scope_rule(s, True)[0] for s in students),
          str([s["slot"] for s in students if "upstream" not in S.scope_rule(s, True)[0]]))
    check("upstream_forbidden_arm_teachers_and_sensitivity_TB3",
          all("upstream" in S.scope_rule(s)[1] for s in cat.slots.values()
              if s["scope_class"] in ("arm_teacher", "calibration_sensitivity"))
          and "upstream" in S.scope_rule(cat.slots["teacher:s42:40k"])[0])
    qat = sorted(s["slot"] for s in cat.slots.values() if "fq_off" in S.scope_rule(s, True)[0])
    check("fq_off_required_for_every_selected_qat_model_TB5", qat == sorted(
        k for k, s in cat.slots.items() if s["precision"] == "int8_qat"), str(qat))
    check("e6kd_never_corrupted_TB2", {"corrupted", "corrupted_sev45"} <= S.scope_rule(cat.slots["E6-KD:s42:sel"])[1])
    check("pilot_loser_has_no_scope_TB9", all(S.scope_rule(cat.slots[r])[0] == frozenset()
                                              for r in cat.slots["E5:s42:sel"]["pilot"]["runs"]))


def case_catalogue_cross_checks():
    from src.eval.artifacts import STAGES
    from src.tman import records as R
    from src.tman import schema as S
    from src.training import kd_arms
    cat = _catalogue()
    rel = []
    for spec in kd_arms.load_arms().values():
        slot = next((s for s in cat.slots.values() if s["stage"] == spec.run_stage), None)
        rel.append((spec.cut_place, cat.position(slot["cut_group"]) if slot else None,
                    cat.arm_places.get(spec.cut_place) == (slot or {}).get("cut_group")))
    rel.sort()
    check("registry_cut_places_in_catalogue_relative_order_TB8",
          len(rel) == 3 and all(b is not None and same for _, b, same in rel)
          and [b for _, b, _ in rel] == sorted(b for _, b, _ in rel), str(rel))
    check("cut_order_relative_TB8", cat.groups == [
        "E6-R2", "E6-R1", "E6-E2", "R2", "W", "K8", "R1", "avg", "L", "G", "F", "A", "E3-160k", "ARM", "E1-160k",
        "E2-160k", "alpha-sweep", "E2-s4344", "E5E6-s4344", "E3-s4344", "E1-rerun-s4344"], str(cat.groups))
    check("cut_place_is_group_position_IF11_2",
          [cat.entry_block(k)["cut_place"] for k in ("E6-R2:s42:15ep", "E6-E2:s42:sel", "teacher-R2:s42:40k",
                                                     "A:s42:alpha25:80k", "ARM:latency", "E1:s42:80k")]
          == [1, 3, 4, 12, 14, None])
    rules = json.loads((REPO / "configs/sweep_rules.json").read_text(encoding="utf-8"))
    ok = True
    for key in S.SWEEP_KEYS:
        sl = cat.sweep_slots(key)
        ok &= sorted(float(s["sweep"]["value"]) for s in sl) == sorted(float(v) for v in rules[key]["grid"])
        ok &= [float(s["sweep"]["value"]) for s in sl if s["sweep"]["default_candidate"]] == \
            [float(rules[key]["default_candidate"])]
    check("sweep_slots_equal_sweep_rules_grids_and_defaults", ok)
    rec = R.Records.at_head(REPO)
    winners = {}
    for key, rel_path, field in (("lambda_logit", "reports/derived/lambda_selection.json", "lambda"),
                                 ("alpha_cwd", "reports/derived/alpha_selection.json", "alpha")):
        doc, _ = with_sentinel(lambda: rec.json(rel_path, R.SELECTION, code="reference_mismatch"))
        winners[key] = (float(doc["winner"][field]),
                        [float(s["sweep"]["value"]) for s in cat.sweep_slots(key) if s["sweep"]["of_record"]])
    check("sweep_of_record_equals_committed_selection_winners", all([w] == r for w, r in winners.values()),
          str(winners))
    pend = sorted({s["evaluator_stage"] for s in cat.slots.values()
                   if s["evaluator_stage"] and s["evaluator_stage"] not in STAGES})
    gated = all(any(g["record"] == "code_merged" for g in (s["gates"] + (cat.slots[s["parent"]]["gates"]
                                                                        if s["parent"] else [])))
                for s in cat.slots.values() if s["evaluator_stage"] in pend)
    check("evaluator_stages_from_artifacts_STAGES_or_pending_code_TB8", gated and pend == [
        "E6-E2", "E6-KD", "E6-R1", "E6-R2", "E7-E2"], f"pending {pend}")


def case_names():
    from src.tman import schema as S
    check("slot_key_regex_accepts_plan_examples",
          all(S.SLOT_KEY_RE.fullmatch(k) for k in ("E2:s42:lambda1:80k", "E3-R1:s42:lambda0.5:alpha25:80k",
                                                   "teacher-R2:s42:40k", "E5:s42:clip5:15ep", "E7:s43:list42",
                                                   "E4:s42:list44", "ARM:latency", "E6:s42:sel")))
    check("slot_key_regex_refuses_bad_keys",
          not any(S.SLOT_KEY_RE.fullmatch(k) for k in ("E2", "E2::80k", ":s42", "E2:s42:", "E2 s42", "E2:s4/2")))
    check("file_id_good", S.file_id_error("E3-R1_s42_lambda0.5_alpha25_80k") is None)
    check("file_id_of_slot_TB7", S.file_id_of("E3-R1:s42:lambda0.5:alpha25:80k") == "E3-R1_s42_lambda0.5_alpha25_80k")
    check("file_id_refuses_double_underscore", "'__'" in (S.file_id_error("E2__s42") or ""))
    check("file_id_refuses_test_any_case", "test" in (S.file_id_error("E2_TeSt_1") or "").lower())
    check("file_id_refuses_colon", S.file_id_error("E2:s42") is not None)
    name = S.entry_file_name("cut", "A_s42_alpha25_80k", "2026-11-03", 1)
    check("entry_file_name_form_T20", name == "cut__A_s42_alpha25_80k__20261103__01.json", name)
    m = S.ENTRY_FILE_RE.fullmatch(name)
    check("entry_file_name_parses_back", bool(m) and m.group("kind") == "cut" and m.group("nn") == "01"
          and m.group("file_id") == "A_s42_alpha25_80k" and m.group("date") == "20261103")
    check("entry_file_rel", S.entry_file_rel(name) == "reports/derived/tday_entries/" + name)
    expect_refused("entry_file_name_refuses_unknown_kind", "entry_field", S.entry_file_name, "launch",
                   "A_s42_alpha25_80k", "2026-11-03", 1)
    expect_refused("entry_file_name_refuses_nn_0", "entry_field", S.entry_file_name, "cut", "A_s42_alpha25_80k",
                   "2026-11-03", 0)
    expect_refused("entry_file_name_refuses_nn_100", "entry_field", S.entry_file_name, "cut", "A_s42_alpha25_80k",
                   "2026-11-03", 100)
    expect_refused("entry_file_name_refuses_bad_day", "entry_field", S.entry_file_name, "cut", "A_s42_alpha25_80k",
                   "20261103", 1)
    expect_refused("entry_file_name_refuses_bad_file_id", "entry_field", S.entry_file_name, "cut", "A:s42",
                   "2026-11-03", 1)
    for bad in ("a:b.json", "x<y", "dir/a|b", "q?.json", 'say"x', "star*", "TeSt.json", "a\\b:c", "x>y"):
        check(f"created_name_refuses_{re.sub('[^a-z]', '', bad.lower()) or 'sym'}", S.created_name_error(bad)
              is not None)
    check("created_name_accepts_lane_paths", all(S.created_name_error(p) is None for p in (
        S.MANIFEST_REL, S.LANES_REL, S.ENTRIES_DIR_REL, S.CATALOGUE_REL, name,
        "/tmp/tman_1_2/tday_clean_E1_s42_80k")))


def case_canonical():
    import hashlib
    from src.eval.artifacts import canonical_json_bytes
    from src.tman import schema as S
    doc = {"b": [1, {"y": "é", "x": 2}], "a": None}
    check("canonical_sha256_is_artifacts_canonical_json_TB14",
          S.canonical_sha256(doc) == hashlib.sha256(canonical_json_bytes(doc)).hexdigest()
          and S.canonical_json_bytes is canonical_json_bytes)
    check("canonical_ignores_key_order", S.canonical_sha256({"a": 1, "b": 2}) == S.canonical_sha256({"b": 2, "a": 1}))
    check("canonical_bytes_compact_sorted_utf8", canonical_json_bytes(doc) == '{"a":null,"b":[1,{"x":2,"y":"é"}]}'
          .encode("utf-8"))
    try:
        S.canonical_sha256({"x": float("nan")})
        nan_ok = False
    except ValueError:
        nan_ok = True
    check("canonical_refuses_nonfinite", nan_ok)
    b = S.json_file_bytes(doc)
    check("record_file_bytes_sorted_indent_lf", b.endswith(b"\n") and not b.endswith(b"\n\n")
          and b == S.json_file_bytes(json.loads(b)) and b.index(b'"a"') < b.index(b'"b"'))
    expect_refused("strict_json_refuses_duplicate_key", "manifest_unreadable", S.strict_json, b'{"a":1,"a":2}', "x",
                   code="manifest_unreadable")
    expect_refused("strict_json_refuses_nan", "manifest_unreadable", S.strict_json, b'{"a":NaN}', "x",
                   code="manifest_unreadable")
    expect_refused("strict_json_refuses_infinity", "manifest_unreadable", S.strict_json, b'[-Infinity]', "x",
                   code="manifest_unreadable")
    expect_refused("strict_json_refuses_non_utf8", "manifest_unreadable", S.strict_json, b'"\xff"', "x",
                   code="manifest_unreadable")
    expect_refused("strict_json_refuses_trailing_garbage", "manifest_unreadable", S.strict_json, b'{} x', "x",
                   code="manifest_unreadable")
    found = [rel for rel in LANE_MODULES if (REPO / rel).is_file()
             and re.search(r"^\s*def canonical_json", (REPO / rel).read_text(encoding="utf-8"), re.M)]
    check("canonical_defined_once_TB14", not found and "src.eval.artifacts import canonical_json_bytes" in
          (REPO / "src/tman/schema.py").read_text(encoding="utf-8"), str(found))


def case_readers():
    from src.tman import records as R
    from src.tman import schema as S
    doc = selection_doc("lambda_logit", 0.5, [0.25, 0.5, 1, 2, 4])
    data = S.json_file_bytes(doc)
    sel = with_sentinel(lambda: R.project(R.strict_json(data, "sel", code="record_uncommitted"), R.SELECTION))
    check("selection_projection_drops_scores", "best_val" not in json.dumps(sel) and "rule_trace" not in sel
          and sel["winner"]["lambda"] == 0.5 and len(sel["candidates"]) == 5)
    for name, touch in (("getitem", lambda d: d["band"]), ("get", lambda d: d.get("rule_trace")),
                        ("dict_copy", lambda d: dict(d)), ("unpack", lambda d: {**d}),
                        ("items", lambda d: d.items()), ("values", lambda d: list(d.values())),
                        ("json_dumps", lambda d: json.dumps(d))):
        try:
            with_sentinel(lambda: touch(R.strict_json(data, "sel", code="record_uncommitted")))
            leak = False
        except AssertionError:
            leak = True
        check(f"sentinel_raises_on_score_key_{name}", leak)
    allow_keys = []

    def walk(a):
        if isinstance(a, dict):
            for k, v in a.items():
                allow_keys.append(k)
                walk(v)
        elif isinstance(a, list):
            walk(a[0])
    for a in (R.KD_ROWS, R.KD_RUN_META, R.E1_ROWS, R.QAT_ROWS, R.QAT_CHECKPOINT, R.QAT_SELECTION, R.CLIP_SELECTION,
              R.ARM_TEACHER_RECORD, R.GO_RULE_RECORD, R.E7_ARM_RECORD, R.LAUNCH_PROVENANCE, R.PTQ_PROVENANCE,
              R.EVAL_RUNTIME, R.SELECTION, R.PTQ_RECORD):
        walk(a)
    check("allowlists_hold_no_score_key", not [k for k in allow_keys if SCORE_KEY_RE.search(k)],
          str([k for k in allow_keys if SCORE_KEY_RE.search(k)]))
    d = scratch_dir()
    kd = kd_run(d, "e2_s42_lambda0.5_80k_a1")
    got, torn, sha = with_sentinel(lambda: R.read_jsonl(kd / "e2_telemetry.jsonl", R.KD_ROWS,
                                                        code="completion_record_invalid"))
    end = got[-1]
    check("kd_telemetry_projection_keeps_time_and_identity",
          not torn and len(sha) == 64 and end == {"event": "run_end", "iter": 3, "checks_passed": True,
                                                  "wall_clock_end": utc("2026-10-20T10:00:00Z"),
                                                  "wall_clock_start": utc("2026-10-10T00:00:00Z"),
                                                  "best_ckpt": "iter_3.pth"}
          and got[0] == {"event": "train", "iter": 1, "wall_clock": utc("2026-10-10T00:00:01Z"),
                         "iter_seconds": 0.7 + 0.01} and got[3] == {"event": "val", "iter": 3,
                                                              "wall_clock": utc("2026-10-10T00:00:03.5Z")}, str(got))
    meta, _, _ = with_sentinel(lambda: R.read_jsonl(kd / "e2_run_meta.jsonl", {"run_meta": R.KD_RUN_META},
                                                    code="completion_record_invalid"))
    check("kd_run_meta_projection", meta[0].get("max_iters") == 3 and "lambda_logit" not in meta[0], str(meta))
    e1 = e1_run(d, "e1_s42_80k")
    rows, _, _ = with_sentinel(lambda: R.read_jsonl(e1 / "e1_telemetry.jsonl", R.E1_ROWS,
                                                    code="completion_record_invalid"))
    check("e1_telemetry_projection_drops_val_scores", rows[-1] == {"event": "val", "iter": 4,
                                                                   "wall_clock": utc("2026-09-20T10:00:00Z")})
    q = qat_run(d, "e6_s42_15ep", nonfinite_from=14)
    rows, _, _ = with_sentinel(lambda: R.read_jsonl(q / "qat_telemetry.jsonl", R.QAT_ROWS,
                                                    code="completion_record_invalid"))
    last = rows[-1]
    check("qat_telemetry_projection_keeps_flags_drops_fake_quant",
          last["event"] == "epoch_end" and last["run_complete"] is True and last["state_nonfinite"] is True
          and last["nonfinite_since_step"] == 140 and "fake_quant_val_all_class_miou" not in last
          and {"event": "val"} in rows and {"event": "train"} in rows, str(last))
    (d / "torn.jsonl").write_bytes((kd / "e2_telemetry.jsonl").read_bytes() + b'{"event": "tra')
    rows2, torn2, _ = R.read_jsonl(d / "torn.jsonl", R.KD_ROWS, code="completion_record_invalid")
    check("telemetry_torn_last_line_reported_not_parsed", torn2 and len(rows2) == len(got))
    (d / "bad.jsonl").write_bytes(b'{"event": "train", "loss": NaN}\n')
    expect_refused("telemetry_nonfinite_constant_refused", "completion_record_invalid", R.read_jsonl,
                   d / "bad.jsonl", R.KD_ROWS, code="completion_record_invalid")
    (d / "dup.jsonl").write_bytes(b'{"event": "train", "event": "run_end"}\n')
    expect_refused("telemetry_duplicate_key_refused", "completion_record_invalid", R.read_jsonl,
                   d / "dup.jsonl", R.KD_ROWS, code="completion_record_invalid")
    (d / "sel.json").write_bytes(data)
    got2, sha2 = with_sentinel(lambda: R.read_json_file(d / "sel.json", R.SELECTION, code="reference_mismatch"))
    check("json_file_reader_projects_and_hashes", got2 == sel and sha2 == S.sha256_bytes(data))


def case_ptq_record():
    from src.tman import records as R
    path = "/nonexistent_tman_fx/e4_ptq_record.json"
    read, sha = ptq_reader_stub(ptq_record_doc())
    got = R.read_ptq_record(path, sha, reader=read)
    check("ptq_record_projection_A2", got == {
        "stage": "E4", "seed": 42, "role": "of_record", "descriptive": False,
        "parent": {"checkpoint_sha256": "1" * 64}, "calibration": {"checksum_sha256": "3" * 64},
        "artifacts": {"torchscript": {"sha256": "5" * 64}, "state_dict": {"sha256": "6" * 64},
                      "x86_copy": {"sha256": "7" * 64}}}, str(got))
    expect_refused("ptq_record_sha256_mismatch_refused", "ptq_record_invalid", R.read_ptq_record, path, "0" * 64,
                   reader=read)

    def raising(p, e=None):
        raise ValueError("not an intact ptq_record/1")
    expect_refused("ptq_record_reader_refusal_mapped", "ptq_record_invalid", R.read_ptq_record, path, sha,
                   reader=raising)
    doc = ptq_record_doc()
    del doc["artifacts"]["x86_copy"]
    read3, sha3 = ptq_reader_stub(doc)
    expect_refused("ptq_record_incomplete_projection_refused", "ptq_record_invalid", R.read_ptq_record, path, sha3,
                   reader=read3)
    read4, sha4 = ptq_reader_stub(ptq_record_doc(role="d3_replicate"))
    got4 = R.read_ptq_record(path, sha4, reader=read4)
    check("ptq_record_role_projected_as_written", got4["role"] == "d3_replicate" and got4["descriptive"] is True)
    saved = R.PTQ_RECORD_MODULE
    try:
        R.PTQ_RECORD_MODULE = "src.quant.tman_absent_module"
        expect_refused("ptq_record_reader_absent_refused", "ptq_record_invalid", R.read_ptq_record, path, sha)
    finally:
        R.PTQ_RECORD_MODULE = saved
    expect_refused("ptq_record_test_path_refused", "test_path", R.read_ptq_record,
                   "/nonexistent_tman_fx/Test/e4_ptq_record.json", sha, reader=read)
    src = (REPO / "src/tman/records.py").read_text(encoding="utf-8")
    check("ptq_record_reader_is_lqp_read_record_static_A2", R.PTQ_RECORD_MODULE == "src.quant.ptq_record"
          and "importlib.import_module(PTQ_RECORD_MODULE).read_record" in src)


def case_decision_log():
    from src.tman import records as R
    sha = "ab" * 32
    text = dl_table({"DL-1": ("RECORDED (x)", f"entry {sha}"), "DL-2": ("DECIDED", f"also {sha}"),
                     "DL-3": ("IMPLEMENTED 1234567", "code"), "DL-10b": ("DROPPED", "gone")})
    dl = R.DecisionLog(text)
    check("dl_rows_parsed", sorted(dl.rows) == ["DL-1", "DL-10b", "DL-2", "DL-3"] and dl.status("DL-3")
          .startswith("IMPLEMENTED"))
    check("dl_recorded_rows_with_sha_ignore_decided", dl.recorded_rows_with(sha) == ["DL-1"])
    check("dl_header_not_a_row", not dl.has("ID") and dl.status("DL-9") is None and dl.row_text("DL-9") is None)


def case_records_source():
    from src.tman import records as R
    from src.tman import schema as S
    repo = new_repo()
    commit_files(repo, {"docs/DECISION_LOG.md": dl_table({"DL-1": ("RECORDED", "x")}),
                        "reports/derived/lambda_selection.json": selection_doc("lambda_logit", 0.5, [0.5, 1])})
    rec = R.Records.at_head(repo)
    doc, sha = with_sentinel(lambda: rec.json("reports/derived/lambda_selection.json", R.SELECTION,
                                              code="record_uncommitted"))
    check("records_read_committed_projected", doc["winner"]["lambda"] == 0.5 and len(sha) == 64)
    check("records_has", rec.has("reports/derived/lambda_selection.json") and not rec.has("reports/x.json"))
    check("records_written_once", rec.written_once_error("reports/derived/lambda_selection.json") is None
          and len(rec.adding_commits("reports/derived/lambda_selection.json")) == 1)
    (repo / "reports/derived/lambda_selection.json").write_bytes(b"{}\n")
    expect_refused("records_refuse_modified_file", "record_uncommitted", rec.read,
                   "reports/derived/lambda_selection.json")
    expect_refused("records_refuse_absent_file", "record_uncommitted", rec.read, "reports/derived/none.json")
    expect_refused("records_refuse_absent_file_with_its_code", "dl_row_missing", rec.read, "reports/derived/none.json",
                   missing="dl_row_missing")
    expect_refused("records_refuse_test_named_path", "test_path", rec.read, "reports/derived/test/x.json")
    check("decision_log_through_records_at_head", R.Records.at_head(repo).decision_log().has("DL-1"))
    (repo / "docs/DECISION_LOG.md").write_text(dl_table({"DL-2": ("RECORDED", "y")}), encoding="utf-8")
    check("decision_log_read_at_the_records_ref", R.Records.at_head(repo).decision_log().has("DL-1"))
    expect_refused("launch_log_absent_record_uncommitted", "record_uncommitted",
                   R.Records.at_head(repo).launch_log, code="completion_record_invalid")
    empty = new_repo()
    commit_files(empty, {"reports/x.json": {"format": "x"}})
    expect_refused("decision_log_absent_refused", "dl_row_missing", R.Records.at_head(empty).decision_log)
    check("scratch_names_free_of_test", all("test" not in str(p).lower() for p in SCRATCH_DIRS))
    check("records_ref_head", rec.ref == "HEAD" and S.DECISION_LOG_REL == "docs/DECISION_LOG.md")


def case_catalogue():
    import copy
    from src.tman import schema as S
    data = (REPO / S.CATALOGUE_REL).read_bytes()
    cat = S.load_catalogue(data)
    check("catalogue_loads_69_slots", len(cat.slots) == 69 and cat.sha256 == S.sha256_bytes(data),
          str(len(cat.slots)))
    check("catalogue_file_bytes_canonical_record_form", data == S.json_file_bytes(json.loads(data)))
    check("catalogue_file_ids_unique_and_ntfs_safe_TB7",
          len(cat.by_file_id) == len(cat.slots) and all(S.file_id_error(f) is None for f in cat.by_file_id))
    check("catalogue_am23_slots_TB8", all(k in cat.slots for k in ("E6-E2:s42:15ep", "E6-E2:s42:sel",
                                                                    "E7-E2:s42:list42", "E6-R1:s42:15ep",
                                                                    "E6-R2:s42:15ep", "E6-R1:s42:sel",
                                                                    "E6-R2:s42:sel")))
    check("catalogue_am23_parents_TB8", cat.slots["E6-E2:s42:15ep"]["parent"] == "E2:s42:lambda0.5:80k"
          and cat.slots["E7-E2:s42:list42"]["parent"] == "E2:s42:lambda0.5:80k"
          and cat.slots["E6-R1:s42:15ep"]["parent"] == "E3-R1:s42:lambda0.5:alpha25:80k"
          and cat.slots["E6-R2:s42:15ep"]["parent"] == "E3-R2:s42:lambda0.5:alpha25:80k")
    check("catalogue_classes_TB8", {s["class"] for s in cat.slots.values()} == {"run", "derived", "measurement"}
          and cat.slots["ARM:latency"]["class"] == "measurement"
          and all(s["record_kind"] in S.RECORD_KINDS[s["class"]] for s in cat.slots.values()))
    sel = [k for k in cat.slots if k.endswith(":sel")]
    check("catalogue_qat_selection_slots_TB9", len(sel) == 10 and all(
        cat.slots[k]["class"] == "derived" and cat.slots[k]["record_kind"] == "qat_selection"
        and cat.slots[k]["precision"] == "int8_qat" for k in sel), str(sel))
    runs = [k for k, s in cat.slots.items() if s["record_kind"] == "qat_epoch15"]
    check("catalogue_qat_runs_carry_no_scope_TB9", len(runs) == 11 and all(
        cat.slots[k]["scope_class"] == "none" and cat.slots[k]["evaluator_stage"] is None for k in runs))
    pilot = cat.slots["E5:s42:sel"]["pilot"]
    check("catalogue_pilot_maps_to_one_e5_slot_TB9", pilot == {"runs": ["E5:s42:clip1:15ep", "E5:s42:clip5:15ep"]}
          and all(cat.slots[r]["scope_class"] == "none" and cat.slots[r]["pilot"] == {"of": "E5:s42:sel"}
                  for r in pilot["runs"]))
    check("catalogue_gates_present_TB8", any(g["record"] == "condition_entry" and not g["moves_cutoff"]
                                             for g in cat.slots["E6-R1:s42:15ep"]["gates"])
          and any(g["record"] == "code_merged" for g in cat.slots["E7-E2:s42:list42"]["gates"]))
    check("catalogue_never_cut_set", sorted(k for k, s in cat.slots.items() if s["never_cut"]) == sorted([
        "teacher:s42:40k", "E1:s42:80k", "E2:s42:lambda0.5:80k", "E2:s42:lambda1:80k",
        "E3:s42:lambda0.5:alpha25:80k", "E3:s42:lambda0.5:alpha50:80k", "E5:s42:clip1:15ep", "E5:s42:clip5:15ep",
        "E5:s42:sel", "E6:s42:15ep", "E6:s42:sel", "E6-KD:s42:15ep", "E6-KD:s42:sel",
        "E4:s42:list42", "E4:s43:list42", "E4:s44:list42", "E7:s42:list42", "E7:s43:list42", "E7:s44:list42",
        "E4:s42:list43", "E4:s42:list44", "E4:s42:list45", "E7:s42:list43", "E7:s42:list44", "E7:s42:list45"]))
    check("catalogue_children_index", sorted(cat.children["E2:s42:lambda0.5:80k"]) == ["E6-E2:s42:15ep",
                                                                                      "E7-E2:s42:list42"])
    check("catalogue_group_slots", sorted(cat.group_slots("R2")) == ["E3-R2:s42:lambda0.5:alpha25:80k",
                                                                    "E7-R2:s42:list42", "teacher-R2:s42:40k"]
          and sorted(cat.group_slots("E6-R2")) == ["E6-R2:s42:15ep", "E6-R2:s42:sel"], str(cat.group_slots("R2")))
    doc = json.loads(data)

    def broken(mut):
        d = copy.deepcopy(doc)
        mut(d)
        return S.json_file_bytes(d)

    def slot(d, k):
        return next(s for s in d["slots"] if s["slot"] == k)
    muts = {
        "format": lambda d: d.update(format="tday_catalogue/2"),
        "top_level_key": lambda d: d.update(note="x"),
        "duplicate_slot": lambda d: d["slots"].append(dict(d["slots"][0])),
        "duplicate_group": lambda d: d["cut_order"].append(dict(d["cut_order"][0])),
        "unknown_group": lambda d: slot(d, "G:s42:80k").update(cut_group="Z"),
        "never_cut_with_group": lambda d: slot(d, "E1:s42:80k").update(cut_group="A"),
        "file_id_not_slot": lambda d: slot(d, "G:s42:80k").update(file_id="G-s42-80k"),
        "gate_moves_cutoff_wrong": lambda d: slot(d, "E6-R1:s42:15ep")["gates"][2].update(moves_cutoff=True),
        "arm_places_out_of_order": lambda d: d["cut_order"][3].update(arm_cut_place=9),
        "unknown_parent": lambda d: slot(d, "E7:s43:list42").update(parent="E3:s43:80k"),
        "record_kind_of_other_class": lambda d: slot(d, "E4:s42:list42").update(record_kind="kd_run_end"),
        "scored_without_precision": lambda d: slot(d, "A:s42:alpha25:80k").update(precision=None),
        "two_sweep_winners": lambda d: slot(d, "E2:s42:lambda1:80k")["sweep"].update(of_record=True),
        "unknown_key": lambda d: slot(d, "G:s42:80k").update(note="x"),
        "params_key_unknown": lambda d: slot(d, "G:s42:80k")["params"].update(beta=1),
        "pilot_run_not_naming_selection": lambda d: slot(d, "E5:s42:clip1:15ep").update(
            pilot={"of": "E5:s42:clip5:15ep"}),
        "scope_class_unknown": lambda d: slot(d, "G:s42:80k").update(scope_class="arm_latency"),
    }
    for name, mut in muts.items():
        expect_refused(f"catalogue_invalid_{name}", "catalogue_invalid", S.load_catalogue, broken(mut))
    expect_refused("catalogue_invalid_duplicate_json_key", "catalogue_invalid", S.load_catalogue,
                   b'{"format": "tday_catalogue/1", "format": "x", "cut_order": [], "slots": []}')


if __name__ == "__main__":
    raise SystemExit(run_cases("TMAN SCHEMA SMOKE (lane TMAN, B-1, C1)",
                               (case_formats, case_names, case_canonical, case_entry_schema, case_catalogue,
                                case_scope_table, case_catalogue_cross_checks, case_readers, case_ptq_record,
                                case_decision_log, case_records_source)))
