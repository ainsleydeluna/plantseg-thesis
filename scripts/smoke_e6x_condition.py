#!/usr/bin/env python3
"""Smoke: L-AM23Q B-1, the E6-X condition tool (src/quant/e6x_condition.py, scripts/e6x_condition.py).

Synthetic only: no PlantSeg data, no real checkpoint, no GPU, no TEST path. A records repository (git, in the work
dir: the λ and α selections and the arm teacher records copied byte for byte from this checkout, and a synthetic KD
launch log with E3-R1's and E3-R2's launch and launched lines) and synthetic E3-R1/E3-R2 run directories (run_meta,
telemetry, best.json and a small torch payload) stand in for the real inputs. e6x_condition.RECORDS_ROOT points at the
records repository and src.quant.ptq.code_provenance answers a fixed identity (a clean checkout at a known commit), in
this process, restored after each case; the AQ5 cases run the real code_provenance on a code repository built from
this checkout's TOOL_CODE_PATHS files (ptq.REPO pointed at it, restored). Every refusal is reached with every earlier
gate passing and checked as: exit, code (and field), nothing created. No created file or folder name contains "test"
or ":"; a forbidden-name case uses a never-created path; an audit hook records any open of a path under the work dir
or the repository whose name contains "test". The repository's HEAD and scoped git status (the project's form over
".", ignored files included) are the same after the run as before its fixtures; the run writes no bytecode and works
from a folder of its work dir, so a relative path never names the repository.

  w_  written records: the verdict at, above and below the threshold, the record's fields, determinism, the
      decision-log text (AQ3), the deadline (AQ7), the readers accepting it
  r_  refusals of `record` and `print-dl`, gate by gate (G0-G10; AQ4 value_invalid; the write)
  i_  E3-X incomplete (exit 3)        s_  the write STOP (exit 1)
  aq5_ the tool's identity on a real code repository (AQ5)
  d_  the readers' refusals (print-dl's read_record_file; B-2's read_condition)
  h_  hygiene: no forbidden name created or opened, repository untouched, the threshold of record, every code a case

    python -B scripts/smoke_e6x_condition.py [--work-dir DIR] [--only CASE,CASE]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True    # no __pycache__ in the repository, with or without -B

import torch  # noqa: E402

from src.quant import e6x_condition as C  # noqa: E402
from src.quant import ptq as P  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import runner as R  # noqa: E402
from src.training import kd_arms as K  # noqa: E402
from src.training import sweep_select as SS  # noqa: E402

CLI = importlib.import_module("scripts.e6x_condition")

T = float(C.THRESHOLD)
FAKE_HEAD = "6" * 40              # the identity the in-process cases answer: a clean checkout at this commit
PIN = "a" * 40                    # the trainer's git_head and the launch line's code_pin
RECORDS_COMMIT = "b" * 40         # run_meta's records_commit and the launched line's
CKPT_NAME = "e3_student_best_iter64000.pt"
VALUE_YES, VALUE_NO, VALUE_R2 = 0.3712345678901234, 0.35, 0.3801
LAUNCH_T = {"R1": 1792500000.25, "R2": 1792500100.5}
END_T = {"R1": 1792650000.75, "R2": 1792650100.25}
MANILA = timezone(timedelta(hours=8))
ARMS = ("R1", "R2")
REAL_RECORDS = (C.LAMBDA_SELECTION_REL, C.ALPHA_SELECTION_REL, C.TEACHER_RECORD_REL.format(arm="R1"),
                C.TEACHER_RECORD_REL.format(arm="R2"))
COND_REL = C.RECORD_REL.format(arm="R1")
IDENT = {"commit": FAKE_HEAD, "clean": True}
FX: dict = {"n": 0}
CASES: dict = {}
RESULTS: list = []
OPENED: list = []                 # opens of a path under the work dir or the repository whose name contains "test"
AUDIT_BASES: list = []


def case(name: str, code: str | None = None):
    """Register a case; `code` is the refusal code it reaches (h_every_code_has_a_case)."""
    if "test" in name.lower():
        raise AssertionError(f"case name {name!r} contains 'test'")

    def deco(fn):
        CASES[name] = (fn, code)
        return fn
    return deco


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jl(rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


def new_dir(kind: str) -> Path:
    """A fresh folder of the work dir with a deterministic name (<kind>_<n>, never a random suffix)."""
    FX["n"] += 1
    d = FX["work"] / f"{kind}_{FX['n']:04d}"
    d.mkdir()
    return d


def new_out(arm: str = "R1") -> Path:
    return new_dir("out") / f"e6x_condition_{arm}.json"


def git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update({"GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
                "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
                "GIT_AUTHOR_DATE": "2026-10-17T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-17T00:00:00Z",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout


def repo_status() -> tuple:
    """The repository: HEAD and its scoped status (the project's form over ".", ignored files included)."""
    return (git(REPO, "rev-parse", "HEAD").strip(),
            git(REPO, "status", "--porcelain=v1", "--ignored", "--", ".", ":(exclude)docs/reference/reference.pdf",
                ":(exclude)docs/reference", ":(exclude,icase)*test*"))


# ------------------------------------------------------------------ fixtures
def meta_row(arm: str, /, **over) -> dict:
    """An arm run's run_meta row (src/training/train_distill.py :1009-1049 with launch_fields :703-731)."""
    spec, tsha = FX["spec"][arm], FX["teacher_sha"][arm]
    row = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": 42,
           "terms": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}, "projection_params": 1024,
           "lambda_logit": 0.5, "alpha_cwd": 25.0, "max_iters": 80000, "num_classes": 116,
           "wall_clock": LAUNCH_T[arm], "git_head": PIN, "poly_horizon": 80000, "grad_clip_norm": None,
           "teacher_provenance": {"builder": "synthetic", "ckpt_sha256": tsha, "expected_sha256": tsha},
           "teacher_mock": False, "arm": spec.run_stage, "descriptive": True, "parent_of_e4_e7": False,
           "selection_sha256": {"lambda_logit": FX["lam_sha"], "alpha_cwd": FX["alpha_sha"]},
           "selection_files": {"lambda_logit": C.LAMBDA_SELECTION_REL, "alpha_cwd": C.ALPHA_SELECTION_REL},
           "records_commit": RECORDS_COMMIT, "target_construction": spec.target_construction}
    row.update(over)
    return row


def end_row(arm: str, value, /, **over) -> dict:
    """The trainer's run_end record (train_distill.py :1305-1311)."""
    row = {"event": "run_end", "iter": 80000, "best_val_miou_all_class": value, "best_ckpt": CKPT_NAME,
           "checks_passed": True, "wall_clock_start": LAUNCH_T[arm], "wall_clock_end": END_T[arm],
           "wall_seconds": END_T[arm] - LAUNCH_T[arm], "train_seconds": 140000.0, "gpu_hours": 41.6}
    row.update(over)
    return row


def tel_rows(arm: str, value, end: dict | None = None) -> list:
    return [{"event": "train", "iter": 50, "loss": 2.5}, {"event": "val", "iter": 40000, "all_class_miou": 0.30},
            {"event": "val", "iter": 64000, "all_class_miou": value},
            end if end is not None else end_row(arm, value)]


def payload(arm: str, value, /, **over):
    """A distillation checkpoint payload of an arm run (train_distill.py :395-418), with a tiny state."""
    spec = FX["spec"][arm]
    p = {"stage": "E3", "iter": 64000, "model_state_dict": {"head.weight": torch.zeros(2, 2)},
         "best_val_miou_all_class": value, "num_classes": 116,
         "teacher_provenance": {"ckpt_sha256": FX["teacher_sha"][arm]},
         "terms": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}, "arm": spec.run_stage,
         "target_construction": spec.target_construction}
    p.update(over)
    return p


def make_run(arm: str = "R1", value=VALUE_YES, *, name=None, meta=None, meta_bytes=None, tel=None, tel_bytes=None,
             best=None, best_bytes=None, pay=None, ckpt_bytes=None, skip=()) -> Path:
    """A synthetic E3-X run directory: e3_run_meta.jsonl, e3_telemetry.jsonl, best.json and the checkpoint."""
    rid = FX["run_id"][arm]
    d = new_dir("run") / (name or rid)
    d.mkdir()
    files = {C.META_NAME: meta_bytes if meta_bytes is not None else jl([meta if meta is not None else meta_row(arm)]),
             C.TELEMETRY_NAME: tel_bytes if tel_bytes is not None else jl(tel if tel is not None
                                                                           else tel_rows(arm, value)),
             C.BEST_JSON_NAME: best_bytes if best_bytes is not None else
             json.dumps(best if best is not None else {"best_ckpt": f"/workspace/kd_runs/{rid}/{CKPT_NAME}",
                                                       "best_val_miou_all_class": value}, indent=2).encode("utf-8")}
    for n, data in files.items():
        if n not in skip:
            write(d / n, data)
    if CKPT_NAME not in skip:
        if ckpt_bytes is not None:
            write(d / CKPT_NAME, ckpt_bytes)
        else:
            torch.save(pay if pay is not None else payload(arm, value), d / CKPT_NAME)
    return d


def launch_line(arm: str, **over) -> dict:
    """An arm's launch line (scripts/preflight_arms.py arm_launch_line :189-197)."""
    spec, rid = FX["spec"][arm], FX["run_id"][arm]
    line = {"format": SS.LAUNCH_LOG_FORMAT, "event": "launch", "run_id": rid, "stage": spec.run_stage, "seed": 42,
            "horizon": 80000, "sweep": None, "value": None, "lambda_logit": 0.5, "alpha_cwd": 25.0, "attempt": 1,
            "schedule": "T", "code_pin": PIN, "ckpt_dir": f"/workspace/kd_runs/{rid}", "am8a_report": None}
    line.update(over)
    return line


def launched_line(arm: str, **over) -> dict:
    """Its launched line (preflight_arms.py check_run_meta :1084-1086: from the run_meta row)."""
    line = {"event": "launched", "run_id": FX["run_id"][arm], "run_meta_sha256": FX["meta_sha"][arm],
            "launch_time_utc": LAUNCH_T[arm], "git_head": PIN, "records_commit": RECORDS_COMMIT,
            "decision_date": None, "launch_order_position": None}
    line.update(over)
    return line


def stopped_line(arm: str) -> dict:
    return {"event": "stopped", "run_id": FX["run_id"][arm], "report": {"path": "reports/stop.md", "sha256": "0" * 64}}


def base_log(r1_launch=None, r1_launched=None, extra=()) -> list:
    return [launch_line("R1", **(r1_launch or {})), launch_line("R2"),
            launched_line("R1", **(r1_launched or {})), launched_line("R2"), *extra]


def make_records(*, files=None, drop=(), after=None, untracked=None) -> Path:
    """A records repository: the real selections and teacher records, the synthetic launch log; `files` replaces
    or adds files before the one commit, `drop` leaves files out, `after` rewrites committed files after it and
    `untracked` adds files after it without committing them."""
    root = new_dir("records")
    content = {rel: (REPO / rel).read_bytes() for rel in REAL_RECORDS}
    content[SS.LAUNCH_LOG_REL] = jl(base_log())
    content.update(files or {})
    for rel in drop:
        content.pop(rel, None)
    for rel, data in content.items():
        write(root / rel, data)
    git(root, "init", "-q")
    git(root, "add", "--", *sorted(content))
    git(root, "commit", "-q", "-m", "synthetic records")
    for rel, data in {**(after or {}), **(untracked or {})}.items():
        write(root / rel, data)
    return root


def code_repo(*, after=None, untracked=None) -> Path:
    """A code repository holding this checkout's TOOL_CODE_PATHS files, committed once (AQ5)."""
    root = new_dir("code")
    files = [p for p in C.TOOL_CODE_PATHS if (REPO / p).is_file()]
    for p in files:
        write(root / p, (REPO / p).read_bytes())
    git(root, "init", "-q")
    git(root, "add", "--", *files)
    git(root, "commit", "-q", "-m", "tool code")
    for rel, data in {**(after or {}), **(untracked or {})}.items():
        write(root / rel, data)
    return root


def setup(work: Path) -> None:
    """The shared fixtures: the arm specs, the records repository and the three run directories."""
    FX["status0"] = repo_status()
    FX["work"] = work
    arms = K.load_arms()
    FX["spec"] = {x: next(s for s in arms.values() if s.teacher == "arm" and s.arm_teacher == x) for x in ARMS}
    FX["run_id"] = {x: f"{FX['spec'][x].name}_s42_lambda0.5_alpha25_80k_a1" for x in ARMS}
    FX["teacher_sha"] = {x: json.loads((REPO / C.TEACHER_RECORD_REL.format(arm=x)).read_text(encoding="utf-8"))
                         ["selection"]["checkpoint"]["sha256"] for x in ARMS}
    FX["lam_sha"] = sha((REPO / C.LAMBDA_SELECTION_REL).read_bytes())
    FX["alpha_sha"] = sha((REPO / C.ALPHA_SELECTION_REL).read_bytes())
    FX["meta_sha"] = {x: sha(jl([meta_row(x)])) for x in ARMS}
    FX["file_sha"] = {p: sha((REPO / p).read_bytes()) for p in C.TOOL_CODE_PATHS if (REPO / p).is_file()}
    FX["records"] = make_records()
    FX["run_yes"] = make_run("R1", VALUE_YES)
    FX["run_no"] = make_run("R1", VALUE_NO)
    FX["run_r2"] = make_run("R2", VALUE_R2)


# ------------------------------------------------------------------ running the tool in-process
def fake_identity(paths=C.TOOL_CODE_PATHS) -> dict:
    return {"commit": IDENT["commit"], "clean_at_commit": IDENT["clean"],
            "files_sha256": {p: FX["file_sha"][p] for p in paths if p in FX["file_sha"]}}


@contextlib.contextmanager
def env(records=None, *, identity: bool = True):
    saved = (C.RECORDS_ROOT, P.code_provenance)
    C.RECORDS_ROOT = records if records is not None else FX["records"]
    if identity:
        P.code_provenance = fake_identity
    try:
        yield
    finally:
        C.RECORDS_ROOT, P.code_provenance = saved
        IDENT.update(commit=FAKE_HEAD, clean=True)


def run_cli(argv, records=None, *, identity: bool = True) -> tuple[int, str]:
    buf = io.StringIO()
    with env(records, identity=identity), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            rc = CLI.main([str(a) for a in argv])
        except SystemExit as e:              # argparse
            rc = e.code if isinstance(e.code, int) else 2
    return rc, buf.getvalue()


def result_of(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("RESULT:")]
    return lines[-1] if lines else text[-300:]


def dl_of(text: str) -> str:
    lines = text.splitlines()
    return lines[lines.index(CLI.DL_HEADER) + 1] if CLI.DL_HEADER in lines else ""


def record_argv(run, out, *, arm: str = "R1", flags=None) -> list:
    opts = {"--arm": arm, "--run-dir": run, "--lambda-selection": C.LAMBDA_SELECTION_REL,
            "--lambda-selection-sha256": FX["lam_sha"], "--alpha-selection": C.ALPHA_SELECTION_REL,
            "--alpha-selection-sha256": FX["alpha_sha"], "--expect-head": FAKE_HEAD, "--out": out}
    opts.update(flags or {})
    argv = ["record"]
    for k, v in opts.items():
        if v is not None:
            argv += [k, str(v)]
    return argv


def record_ok(run=None, *, arm: str = "R1", records=None, flags=None):
    out = new_out(arm)
    rc, text = run_cli(record_argv(run if run is not None else FX["run_yes"], out, arm=arm, flags=flags), records)
    return rc, text, out, (out.read_bytes() if out.is_file() else b"")


def nothing_created(out: Path) -> bool:
    p = out if out.is_absolute() else Path.cwd() / out
    folder_clean = not (p.parent.is_dir() and str(p.parent).startswith(str(FX["work"])) and any(p.parent.iterdir()))
    return not os.path.lexists(p) and folder_clean


def refusal(code: str, run=None, *, arm: str = "R1", flags=None, records=None, out=None, field: str | None = None,
            status: str = "REFUSED", exit_code: int = 2) -> tuple[bool, str]:
    """`record` with one input changed: the exit, the code (and the field), nothing created."""
    out = out if out is not None else new_out(arm)
    rc, text = run_cli(record_argv(run if run is not None else FX["run_yes"], out, arm=arm, flags=flags), records)
    res = result_of(text)
    ok = (rc == exit_code and res.startswith(f"RESULT: {status} [{code}]") and nothing_created(out)
          and (field is None or f"[{code}] -- {field}" in res))
    return ok, f"rc={rc} {res[:400]}"


def print_dl(record, sha256) -> tuple[int, str]:
    return run_cli(["print-dl", "--record", record, "--record-sha256", sha256])


def api_refusal(code: str, fn, *a, **k) -> tuple[bool, str]:
    try:
        fn(*a, **k)
    except C.ConditionRefused as e:
        return e.code == code, f"[{e.code}] {e}"
    return False, "no refusal"


def run_with_end(value=VALUE_YES, **end_over) -> Path:
    return make_run("R1", value, tel=tel_rows("R1", value, end=end_row("R1", value, **end_over)))


def alpha_doc(**winner_over) -> bytes:
    doc = json.loads((REPO / C.ALPHA_SELECTION_REL).read_text(encoding="utf-8"))
    doc["winner"].update(winner_over)
    return json.dumps(doc, indent=2).encode("utf-8")


def lambda_doc(**winner_over) -> bytes:
    doc = json.loads((REPO / C.LAMBDA_SELECTION_REL).read_text(encoding="utf-8"))
    doc["winner"].update(winner_over)
    return json.dumps(doc, indent=2).encode("utf-8")


def teacher_doc(arm: str = "R1", /, ckpt=None, **over) -> bytes:
    doc = json.loads((REPO / C.TEACHER_RECORD_REL.format(arm=arm)).read_text(encoding="utf-8"))
    doc.update(over)
    doc["selection"]["checkpoint"].update(ckpt or {})
    return json.dumps(doc, indent=2).encode("utf-8")


def written_record(run=None) -> tuple[bytes, dict, str]:
    rc, text, out, data = record_ok(run)
    if rc != 0:
        raise RuntimeError(f"fixture record not written: {result_of(text)}")
    return data, json.loads(data), sha(data)


def record_file(rec: dict) -> tuple[Path, str]:
    """A record file (sorted keys, indent 2, NaN allowed for the format cases) and its sha256."""
    data = (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write(new_dir("rec") / "e6x_condition_R1.json", data), sha(data)


# ================================================================== w_: written records
@case("w_yes_written")
def _w_yes():
    rc, text, out, data = record_ok()
    rec = json.loads(data) if data else {}
    want = f"RESULT: E6-R1 CONDITION yes ({VALUE_YES!r} > {C.THRESHOLD})"
    return rc == 0 and result_of(text) == want and rec.get("verdict") == "yes" and C.parse_record(data) == rec, \
        result_of(text)


@case("w_no_written")
def _w_no():
    rc, text, out, data = record_ok(FX["run_no"])
    want = f"RESULT: E6-R1 CONDITION no ({VALUE_NO!r} <= {C.THRESHOLD})"
    return rc == 0 and result_of(text) == want and json.loads(data)["verdict"] == "no", result_of(text)


@case("w_equal_is_no")
def _w_equal():
    rc, text, out, data = record_ok(make_run("R1", T))
    want = f"RESULT: E6-R1 CONDITION no ({C.THRESHOLD} <= {C.THRESHOLD})"
    return rc == 0 and result_of(text) == want and json.loads(data)["verdict"] == "no", result_of(text)


@case("w_one_ulp_above_is_yes")
def _w_above():
    v = math.nextafter(T, 1.0)
    rc, text, out, data = record_ok(make_run("R1", v))
    return rc == 0 and json.loads(data)["verdict"] == "yes" and repr(v) != C.THRESHOLD, result_of(text)


@case("w_one_ulp_below_is_no")
def _w_below():
    v = math.nextafter(T, 0.0)
    rc, text, out, data = record_ok(make_run("R1", v))
    return rc == 0 and json.loads(data)["verdict"] == "no" and repr(v) != C.THRESHOLD, result_of(text)


@case("w_r2_written")
def _w_r2():
    rc, text, out, data = record_ok(FX["run_r2"], arm="R2")
    rec = json.loads(data) if data else {}
    ok = (rc == 0 and result_of(text) == f"RESULT: E6-R2 CONDITION yes ({VALUE_R2!r} > {C.THRESHOLD})"
          and rec.get("stage") == "E6-R2" and rec.get("parent_run_stage") == "E3-R2"
          and rec.get("run_id") == FX["run_id"]["R2"] and out.name == "e6x_condition_R2.json")
    return ok, result_of(text)


@case("w_record_fields")
def _w_fields():
    d = FX["run_yes"]
    rc, text, out, data = record_ok()
    rec = json.loads(data)
    files = {"best_json": C.BEST_JSON_NAME, "run_meta": C.META_NAME, "telemetry": C.TELEMETRY_NAME,
             "checkpoint": CKPT_NAME}
    end_day = datetime.fromtimestamp(END_T["R1"], MANILA).date()
    teacher_rel = C.TEACHER_RECORD_REL.format(arm="R1")
    pairs = [
        ("rc", rc == 0),
        ("keys", sorted(rec) == sorted(C.RECORD_KEYS)),
        ("names", (rec["format"], rec["amendment"], rec["arm"], rec["stage"], rec["parent_run_stage"], rec["seed"])
         == (C.FORMAT, "AM-23 item 3(b)", "R1", "E6-R1", "E3-R1", 42)),
        ("run_id", rec["run_id"] == FX["run_id"]["R1"]),
        ("value", rec["value"] == {"best_val_miou_all_class": VALUE_YES, "repr": repr(VALUE_YES), "source": "best.json"}),
        ("threshold", rec["threshold"]["repr"] == C.THRESHOLD and "DL-121" in rec["threshold"]["source"]),
        ("files", all(rec[k]["file"] == n and rec[k]["sha256"] == sha((d / n).read_bytes()) for k, n in files.items())),
        ("bytes", rec["checkpoint"]["bytes"] == (d / CKPT_NAME).stat().st_size),
        ("run_end", rec["run_end"] == {k: end_row("R1", VALUE_YES)[k] for k in
                                      ("iter", "checks_passed", "best_val_miou_all_class", "best_ckpt",
                                       "wall_clock_end")}),
        ("deadline", (rec["dl_deadline"]["run_end_manila_date"], rec["dl_deadline"]["enter_by_manila_date"])
         == (end_day.isoformat(), (end_day + timedelta(days=1)).isoformat())),
        ("selections", (rec["selections"]["lambda"]["path"], rec["selections"]["lambda"]["sha256"],
                        rec["selections"]["alpha"]["path"], rec["selections"]["alpha"]["sha256"],
                        rec["selections"]["alpha"]["winner"]["best_val"])
         == (C.LAMBDA_SELECTION_REL, FX["lam_sha"], C.ALPHA_SELECTION_REL, FX["alpha_sha"], T)),
        ("teacher", rec["teacher_record"] == {"path": teacher_rel, "sha256": sha((REPO / teacher_rel).read_bytes()),
                                              "checkpoint_name": "iter_20000.pth",
                                              "checkpoint_sha256": FX["teacher_sha"]["R1"]}),
        ("launch_log", rec["launch_log"] == {"path": SS.LAUNCH_LOG_REL, "sha256": sha(jl(base_log())),
                                            "launch": launch_line("R1"), "launched": launched_line("R1")}),
        ("tool", rec["tool"] == {"code_paths": list(C.TOOL_CODE_PATHS), **fake_identity()}),
    ]
    bad = [n for n, ok in pairs if not ok]
    return not bad, f"differ: {bad}"


@case("w_deterministic")
def _w_det():
    rc1, _, out1, d1 = record_ok()
    rc2, _, out2, d2 = record_ok()
    ok = (rc1 == rc2 == 0 and out1 != out2 and d1 == d2 and str(FX["work"]).encode() not in d1
          and str(REPO).encode() not in d1)
    return ok, f"rc {rc1} {rc2}; identical {d1 == d2}"


@case("w_print_dl_matches")
def _w_print():
    rc, text, out, data = record_ok()
    s = sha(data)
    rc2, text2 = print_dl(out, s)
    ok = (rc == 0 and rc2 == 0 and dl_of(text2) == dl_of(text) != ""
          and result_of(text2) == f"RESULT: DL TEXT (E6-R1, verdict yes, record {s[:12]})")
    return ok, result_of(text2)


@case("w_dl_text_one_line_aq3")
def _w_dl():
    rc, text, out, data = record_ok()
    rec, s = json.loads(data), sha(data)
    dl = dl_of(text)
    tokens = [s, "E6-R1 condition (AM-23 item 3(b)): yes", repr(VALUE_YES), rec["best_json"]["sha256"],
              rec["checkpoint"]["sha256"], f"record {COND_REL} sha256 {s}"]
    ok = (dl == C.dl_text(rec, s) and dl.isascii() and "|" not in dl and "\n" not in dl and "\r" not in dl
          and all(t in dl for t in tokens))
    return ok, dl[:300]


@case("w_dl_text_no_verdict")
def _w_dl_no():
    rc, text, out, data = record_ok(FX["run_no"])
    dl = dl_of(text)
    ok = (rc == 0 and "E6-R1 condition (AM-23 item 3(b)): no." in dl and f"{VALUE_NO!r} (best.json) <= {C.THRESHOLD}"
          in dl and "E6-R1 is ruled out" in dl and "|" not in dl)
    return ok, dl[:300]


@case("w_deadline_before_manila_midnight")
def _w_dl_before():
    t = datetime(2026, 10, 19, 15, 59, 59, tzinfo=timezone.utc).timestamp()
    rc, text, out, data = record_ok(run_with_end(wall_clock_end=t))
    dd = json.loads(data)["dl_deadline"]
    ok = (rc == 0 and (dd["run_end_utc"], dd["run_end_manila_date"], dd["enter_by_manila_date"], dd["enter_by_end_utc"])
          == ("2026-10-19T15:59:59Z", "2026-10-19", "2026-10-20", "2026-10-20T16:00:00Z")
          and "by the end of 2026-10-20 (Asia/Manila)" in dl_of(text))
    return ok, str(dd)


@case("w_deadline_after_manila_midnight")
def _w_dl_after():
    t = datetime(2026, 10, 19, 16, 0, 0, tzinfo=timezone.utc).timestamp()
    rc, text, out, data = record_ok(run_with_end(wall_clock_end=t))
    dd = json.loads(data)["dl_deadline"]
    ok = (rc == 0 and (dd["run_end_utc"], dd["run_end_manila_date"], dd["enter_by_manila_date"], dd["enter_by_end_utc"])
          == ("2026-10-19T16:00:00Z", "2026-10-20", "2026-10-21", "2026-10-21T16:00:00Z"))
    return ok, str(dd)


def _deadline_before_midnight(t: float):
    def fn():
        rc, text, out, data = record_ok(run_with_end(wall_clock_end=t))
        dd = json.loads(data)["dl_deadline"] if data else {}
        ok = (rc == 0 and dd["run_end_wall_clock_end"] == t
              and (dd["run_end_utc"], dd["run_end_manila_date"], dd["enter_by_manila_date"], dd["enter_by_end_utc"])
              == ("2026-10-19T15:59:59Z", "2026-10-19", "2026-10-20", "2026-10-20T16:00:00Z")
              and "by the end of 2026-10-20 (Asia/Manila)" in dl_of(text))
        return ok, str(dd)
    return fn


MIDNIGHT = datetime(2026, 10, 19, 16, 0, 0, tzinfo=timezone.utc).timestamp()       # 2026-10-20 00:00 Manila
case("w_deadline_fraction_before_manila_midnight")(_deadline_before_midnight(MIDNIGHT - 1 + 0.999))
case("w_deadline_last_float_before_manila_midnight")(_deadline_before_midnight(math.nextafter(MIDNIGHT, 0.0)))


@case("w_read_condition_accepts")
def _w_read():
    data, rec, s = written_record()
    root = make_records(files={COND_REL: data})
    got, s2 = C.read_condition(COND_REL, s, repo_root=root)
    return got == rec and s2 == s, s2


@case("w_cli_subprocess_print_dl")
def _w_sub():
    rc, text, out, data = record_ok()
    s = sha(data)
    p = subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "e6x_condition.py"), "print-dl", "--record",
                        str(out), "--record-sha256", s], capture_output=True, text=True, cwd=str(REPO),
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p.returncode == 0 and dl_of(p.stdout) == dl_of(text) != "", result_of(p.stdout)


# ================================================================== r_: refusals of record and print-dl
# ---- G0, G1
@case("r_missing_flags", "missing_flags")
def _r_missing():
    return refusal("missing_flags", flags={"--out": None}, field="--out:")


@case("r_missing_flags_print_dl", "missing_flags")
def _r_missing_pdl():
    rc, text = run_cli(["print-dl", "--record", FX["work"] / "absent.json"])
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [missing_flags]"), result_of(text)


@case("r_no_command", "missing_flags")
def _r_no_cmd():
    rc, text = run_cli([])
    return (rc == 2 and result_of(text).startswith("RESULT: REFUSED [missing_flags] -- a command is required")), \
        result_of(text)


@case("r_forbidden_path_run_dir", "forbidden_path")
def _r_forbid_run():
    run = FX["work"] / "latest_runs" / FX["run_id"]["R1"]
    ok, detail = refusal("forbidden_path", run, field="--run-dir")
    return ok and not os.path.lexists(run.parent), detail


@case("r_forbidden_path_out", "forbidden_path")
def _r_forbid_out():
    out = FX["work"] / "latest_outs" / "e6x_condition_R1.json"
    ok, detail = refusal("forbidden_path", out=out, field="--out")
    return ok and not os.path.lexists(out.parent), detail


@case("r_forbidden_path_selection", "forbidden_path")
def _r_forbid_sel():
    return refusal("forbidden_path", flags={"--lambda-selection": "reports/derived/latest_lambda_selection.json"},
                   field="--lambda-selection")


@case("r_forbidden_path_selection_alpha", "forbidden_path")
def _r_forbid_sel_alpha():
    return refusal("forbidden_path", flags={"--alpha-selection": "reports/derived/latest_alpha_selection.json"},
                   field="--alpha-selection")


@case("r_forbidden_path_print_dl", "forbidden_path")
def _r_forbid_pdl():
    rec = FX["work"] / "latest_records" / "e6x_condition_R1.json"
    rc, text = print_dl(rec, "0" * 64)
    return (rc == 2 and result_of(text).startswith("RESULT: REFUSED [forbidden_path] -- --record")
            and not os.path.lexists(rec.parent)), result_of(text)


# ---- G2
@case("r_out_relative", "out_relative")
def _r_out_rel():
    return refusal("out_relative", out=Path("e6x_condition_R1.json"))


@case("r_out_inside_repo", "out_inside_repo")
def _r_out_repo():
    return refusal("out_inside_repo", out=REPO / "never_created_e6x" / "e6x_condition_R1.json")


@case("r_out_name", "out_name")
def _r_out_name():
    return refusal("out_name", out=new_dir("out") / "e6x_condition_R2.json")


@case("r_out_name_arm_form", "out_name")
def _r_out_arm():
    return refusal("out_name", arm="r1", out=new_dir("out") / "e6x_condition_r1.json")


@case("r_out_exists", "out_exists")
def _r_out_exists():
    out = write(new_out(), b"kept\n")
    rc, text = run_cli(record_argv(FX["run_yes"], out))
    return (rc == 2 and result_of(text).startswith(f"RESULT: REFUSED [out_exists] -- {out} exists: a condition record")
            and out.read_bytes() == b"kept\n" and len(list(out.parent.iterdir())) == 1), result_of(text)


@case("r_out_exists_incomplete_run", "out_exists")
def _r_out_exists_incomplete():
    out = write(new_out(), b"kept\n")              # G2 refuses before G8 would find E3-X incomplete (exit 3)
    rc, text = run_cli(record_argv(make_run(tel=tel_rows("R1", VALUE_YES)[:-1]), out))
    return (rc == 2 and result_of(text).startswith("RESULT: REFUSED [out_exists]") and out.read_bytes() == b"kept\n"), \
        result_of(text)


@case("r_out_parent_missing", "out_parent_missing")
def _r_out_parent():
    out = FX["work"] / "never_made_folder" / "e6x_condition_R1.json"
    ok, detail = refusal("out_parent_missing", out=out)
    return ok and not os.path.lexists(out.parent), detail


# ---- G3
@case("r_expect_head_format", "expect_head_format")
def _r_head_format():
    return refusal("expect_head_format", flags={"--expect-head": "abc"})


@case("r_code_identity_unavailable", "code_identity_unavailable")
def _r_ident():
    IDENT["commit"] = None
    return refusal("code_identity_unavailable")


@case("r_expect_head_mismatch", "expect_head_mismatch")
def _r_head_mismatch():
    return refusal("expect_head_mismatch", flags={"--expect-head": "7" * 40})


@case("r_code_not_clean", "code_not_clean")
def _r_not_clean():
    IDENT["clean"] = False
    return refusal("code_not_clean")


# ---- G4
@case("r_arm_spec", "arm_spec")
def _r_arm_spec():
    bad = new_dir("arms")
    write(bad / "e3_r1.json", b"{\n")
    saved = K.ARM_DIR
    K.ARM_DIR = bad
    try:
        return refusal("arm_spec")
    finally:
        K.ARM_DIR = saved


@case("r_arm_unknown", "arm_unknown")
def _r_arm_unknown():
    return refusal("arm_unknown", arm="R3", out=new_out("R3"))


# ---- G5
@case("r_selection_not_of_record", "selection_not_of_record")
def _r_sel_record():
    copy = "reports/derived/alpha_selection_copy.json"        # a committed byte copy: only the path is not of record
    root = make_records(files={copy: (REPO / C.ALPHA_SELECTION_REL).read_bytes()})
    return refusal("selection_not_of_record", records=root, flags={"--alpha-selection": copy})


@case("r_selection_not_of_record_lambda", "selection_not_of_record")
def _r_sel_record_lambda():
    copy = "reports/derived/lambda_selection_copy.json"
    root = make_records(files={copy: (REPO / C.LAMBDA_SELECTION_REL).read_bytes()})
    return refusal("selection_not_of_record", records=root, flags={"--lambda-selection": copy})


@case("r_selection_sha256_mismatch")
def _r_sel_sha():
    return refusal("selection_sha256_mismatch", flags={"--lambda-selection-sha256": "0" * 64})


@case("r_selection_not_tracked")
def _r_sel_untracked():
    root = make_records(drop=(C.ALPHA_SELECTION_REL,),
                        untracked={C.ALPHA_SELECTION_REL: (REPO / C.ALPHA_SELECTION_REL).read_bytes()})
    return refusal("selection_not_tracked", records=root)


@case("r_selection_changed_since_head")
def _r_sel_changed():
    root = make_records(after={C.LAMBDA_SELECTION_REL: (REPO / C.LAMBDA_SELECTION_REL).read_bytes() + b"\n"})
    return refusal("selection_changed_since_head", records=root)


@case("r_selection_format_winner", "selection_format")
def _r_sel_winner():
    doc = alpha_doc(alpha=None)
    root = make_records(files={C.ALPHA_SELECTION_REL: doc})
    return refusal("selection_format", records=root, flags={"--alpha-selection-sha256": sha(doc)})


@case("r_selection_format_winner_lambda", "selection_format")
def _r_sel_winner_lambda():
    doc = lambda_doc(**{"lambda": None})
    root = make_records(files={C.LAMBDA_SELECTION_REL: doc})
    return refusal("selection_format", records=root, flags={"--lambda-selection-sha256": sha(doc)})


@case("r_threshold_mismatch", "threshold_mismatch")
def _r_threshold():
    doc = alpha_doc(best_val=0.36)
    root = make_records(files={C.ALPHA_SELECTION_REL: doc})
    return refusal("threshold_mismatch", records=root, flags={"--alpha-selection-sha256": sha(doc)})


# ---- G6
@case("r_teacher_record_missing", "teacher_record_missing")
def _r_teacher_missing():
    return refusal("teacher_record_missing", records=make_records(drop=(C.TEACHER_RECORD_REL.format(arm="R1"),)))


@case("r_teacher_record_uncommitted", "teacher_record_uncommitted")
def _r_teacher_uncommitted():
    rel = C.TEACHER_RECORD_REL.format(arm="R1")
    return refusal("teacher_record_uncommitted", records=make_records(after={rel: (REPO / rel).read_bytes() + b"\n"}))


@case("r_teacher_record_invalid", "teacher_record_invalid")
def _r_teacher_invalid():
    rel = C.TEACHER_RECORD_REL.format(arm="R1")
    return refusal("teacher_record_invalid", records=make_records(files={rel: teacher_doc(status="running")}))


for _key, _over in (("format", {"format": "arm_teacher_record/2"}), ("arm", {"arm": "R2"}),
                    ("sha256", {"ckpt": {"sha256": "abc"}})):
    case(f"r_teacher_record_invalid_{_key}", "teacher_record_invalid")(
        lambda _over=_over: refusal("teacher_record_invalid", records=make_records(
            files={C.TEACHER_RECORD_REL.format(arm="R1"): teacher_doc("R1", **_over)})))


# ---- G7
@case("r_run_dir_relative", "run_dir_relative")
def _r_run_rel():
    return refusal("run_dir_relative", Path(FX["run_id"]["R1"]))


@case("r_run_dir_inside_repo", "run_dir_inside_repo")
def _r_run_repo():
    return refusal("run_dir_inside_repo", REPO / "never_created_e6x" / FX["run_id"]["R1"])


@case("r_run_dir_name", "run_dir_name")
def _r_run_name():
    return refusal("run_dir_name", make_run("R1", name=FX["run_id"]["R2"]))


@case("r_run_records_missing", "run_records_missing")
def _r_run_missing():
    return refusal("run_records_missing", make_run(skip=(C.TELEMETRY_NAME,)))


@case("r_run_meta_rows", "run_meta_rows")
def _r_meta_rows():
    return refusal("run_meta_rows", make_run(meta_bytes=jl([meta_row("R1"), meta_row("R1")])))


@case("r_run_meta_rows_torn", "run_meta_rows")
def _r_meta_rows_torn():
    return refusal("run_meta_rows", make_run(meta_bytes=jl([meta_row("R1")]) + b'{"event": "run_me'))


@case("r_run_meta_rows_event", "run_meta_rows")
def _r_meta_rows_event():
    return refusal("run_meta_rows", make_run(meta=meta_row("R1", event="run_end")))


RUN_META_BAD = {"mode": "dry", "stage": "E2", "seed": 43, "arm": "E3-R2", "descriptive": False, "parent_of_e4_e7": True,
                "max_iters": 40000, "poly_horizon": 40000, "target_construction": {"K": 1, "views": 2},
                "lambda_logit": 1.0, "alpha_cwd": 50.0, "teacher_mock": True}


def _run_meta_case(key: str, bad):
    def fn():
        over = {key: bad} if key != "teacher_provenance.ckpt_sha256" else \
            {"teacher_provenance": {"builder": "synthetic", "ckpt_sha256": "0" * 64, "expected_sha256": "0" * 64}}
        return refusal("run_meta_mismatch", make_run(meta=meta_row("R1", **over)), field=f"run_meta {key} is")
    return fn


for _key, _bad in [*RUN_META_BAD.items(), ("teacher_provenance.ckpt_sha256", None)]:
    case(f"r_run_meta_{_key.replace('.', '_')}", "run_meta_mismatch")(_run_meta_case(_key, _bad))


# ---- G8
@case("r_telemetry_format_line", "telemetry_format")
def _r_tel_line():
    rows = tel_rows("R1", VALUE_YES)
    return refusal("telemetry_format", make_run(tel_bytes=jl(rows[:1]) + b"not json\n" + jl(rows[1:])))


@case("r_telemetry_format_row", "telemetry_format")
def _r_tel_row():
    rows = tel_rows("R1", VALUE_YES)
    return refusal("telemetry_format", make_run(tel=[*rows[:-1], [1, 2], rows[-1]]))


@case("r_telemetry_torn", "telemetry_torn")
def _r_tel_torn():
    return refusal("telemetry_torn", make_run(tel_bytes=jl(tel_rows("R1", VALUE_YES)) + b'{"event": "val", "it'))


@case("r_parent_aborted", "parent_aborted")
def _r_aborted():
    rows = tel_rows("R1", VALUE_YES)[:-1] + [{"event": "run_abort", "iter": 50000, "rule": "AM-7(b)"}]
    return refusal("parent_aborted", make_run(tel=rows))


@case("i_run_end_missing", "run_end_missing")
def _i_no_end():
    return refusal("run_end_missing", make_run(tel=tel_rows("R1", VALUE_YES)[:-1]), status="INCOMPLETE", exit_code=3)


@case("r_run_end_not_last", "run_end_not_last")
def _r_end_not_last():
    rows = tel_rows("R1", VALUE_YES) + [{"event": "val", "iter": 80001, "all_class_miou": 0.3}]
    return refusal("run_end_not_last", make_run(tel=rows))


@case("r_run_end_twice", "run_end_not_last")
def _r_end_twice():
    rows = tel_rows("R1", VALUE_YES)
    return refusal("run_end_not_last", make_run(tel=rows + [rows[-1]]))


@case("r_parent_checks_failed", "parent_checks_failed")
def _r_checks():
    return refusal("parent_checks_failed", run_with_end(checks_passed=False))


@case("r_run_end_format_iter", "run_end_format")
def _r_end_iter():
    return refusal("run_end_format", run_with_end(iter=79999))


@case("r_run_end_format_iter_float", "run_end_format")
def _r_end_iter_float():
    return refusal("run_end_format", run_with_end(iter=80000.0))


@case("r_run_end_format_clock", "run_end_format")
def _r_end_clock():
    return refusal("run_end_format", run_with_end(wall_clock_end="2026-10-19T08:00:00Z"))


# ---- G9
@case("r_best_json_format_parse", "best_json_format")
def _r_best_parse():
    return refusal("best_json_format", make_run(best_bytes=b'{"best_ckpt": '))


@case("r_best_json_format_keys", "best_json_format")
def _r_best_keys():
    best = {"best_ckpt": f"/workspace/kd_runs/x/{CKPT_NAME}", "best_val_miou_all_class": VALUE_YES, "extra": 1}
    return refusal("best_json_format", make_run(best=best))


@case("r_best_json_format_name", "best_json_format")
def _r_best_name():
    best = {"best_ckpt": "/workspace/kd_runs/x/model.pt", "best_val_miou_all_class": VALUE_YES}
    return refusal("best_json_format", make_run(best=best))


def _best(value) -> dict:
    return {"best_ckpt": f"/workspace/kd_runs/x/{CKPT_NAME}", "best_val_miou_all_class": value}


@case("r_value_invalid_nan_best_json", "value_invalid")
def _r_v_nan():
    return refusal("value_invalid", make_run(best=_best(float("nan"))), field="best.json's")


@case("r_value_invalid_infinity_run_end", "value_invalid")
def _r_v_inf():
    return refusal("value_invalid", run_with_end(best_val_miou_all_class=float("inf")), field="run_end's")


@case("r_value_invalid_int_payload", "value_invalid")
def _r_v_int():
    return refusal("value_invalid", make_run(pay=payload("R1", 1)), field="the payload's")


@case("r_value_invalid_string_best_json", "value_invalid")
def _r_v_str():
    return refusal("value_invalid", make_run(best=_best("0.3712")), field="best.json's")


@case("r_value_invalid_above_one_best_json", "value_invalid")
def _r_v_above():
    return refusal("value_invalid", make_run(best=_best(1.5)), field="best.json's")


@case("r_value_invalid_bool_run_end", "value_invalid")
def _r_v_bool():
    return refusal("value_invalid", run_with_end(best_val_miou_all_class=True), field="run_end's")


@case("r_value_invalid_negative_payload", "value_invalid")
def _r_v_neg():
    return refusal("value_invalid", make_run(pay=payload("R1", -0.1)), field="the payload's")


@case("r_checkpoint_missing", "checkpoint_missing")
def _r_ckpt_missing():
    return refusal("checkpoint_missing", make_run(skip=(CKPT_NAME,)))


@case("r_payload_unreadable", "payload_mismatch")
def _r_pay_unreadable():
    return refusal("payload_mismatch", make_run(ckpt_bytes=b"not a checkpoint\n"))


@case("r_payload_not_dict", "payload_mismatch")
def _r_pay_list():
    return refusal("payload_mismatch", make_run(pay=[1, 2]))


for _key, _bad in (("stage", "E2"), ("arm", "E3-R2"), ("target_construction", {"K": 1, "views": 2}),
                   ("best_val_miou_all_class", VALUE_YES + 1e-6)):
    case(f"r_payload_{_key}", "payload_mismatch")(
        lambda _key=_key, _bad=_bad: refusal("payload_mismatch", make_run(pay=payload("R1", VALUE_YES, **{_key: _bad})),
                                             field=f"{CKPT_NAME}'s payload {_key} is"))


@case("r_run_end_best", "run_end_mismatch")
def _r_end_best():
    return refusal("run_end_mismatch", run_with_end(best_val_miou_all_class=VALUE_YES + 1e-6),
                   field="run_end best_val_miou_all_class is")


@case("r_run_end_ckpt", "run_end_mismatch")
def _r_end_ckpt():
    return refusal("run_end_mismatch", run_with_end(best_ckpt="e3_student_best_iter60000.pt"), field="run_end best_ckpt is")


# ---- G10
@case("r_launch_log_missing")
def _r_log_missing():
    return refusal("launch_log_missing", records=make_records(drop=(SS.LAUNCH_LOG_REL,)))


@case("r_launch_log_uncommitted")
def _r_log_uncommitted():
    return refusal("launch_log_uncommitted", records=make_records(after={SS.LAUNCH_LOG_REL: jl(base_log()) + b"\n"}))


@case("r_launch_log_format")
def _r_log_format():
    return refusal("launch_log_format", records=make_records(files={SS.LAUNCH_LOG_REL: jl(base_log())[:-1]}))


@case("r_launch_log_launch", "launch_log")
def _r_log_launch():
    root = make_records(files={SS.LAUNCH_LOG_REL: jl([launch_line("R2"), launched_line("R2")])})
    return refusal("launch_log", records=root, field="launch:")


@case("r_launch_log_launched", "launch_log")
def _r_log_launched():
    root = make_records(files={SS.LAUNCH_LOG_REL: jl([launch_line("R1"), launch_line("R2"), launched_line("R2")])})
    return refusal("launch_log", records=root, field="launched:")


@case("r_launch_log_stopped", "launch_log")
def _r_log_stopped():
    root = make_records(files={SS.LAUNCH_LOG_REL: jl(base_log(extra=[stopped_line("R1")]))})
    return refusal("launch_log", records=root, field="stopped:")


@case("r_launch_log_attempt", "launch_log")
def _r_log_attempt():
    rid2 = FX["run_id"]["R1"][:-1] + "2"                     # the same grid point's attempt 2, launched after it
    again = launch_line("R1", run_id=rid2, attempt=2, ckpt_dir=f"/workspace/kd_runs/{rid2}",
                        am8a_report={"path": "reports/am8a_e3-r1.md", "sha256": "0" * 64})
    root = make_records(files={SS.LAUNCH_LOG_REL: jl(base_log(extra=[again]))})
    return refusal("launch_log", records=root, field="attempt:")


LOG_FIELD_CASES = {
    "run_meta_sha256": ({}, {"run_meta_sha256": "0" * 64}),
    "git_head": ({"code_pin": "c" * 40}, {"git_head": "c" * 40}),       # code_pin agrees: git_head alone differs
    "code_pin": ({"code_pin": "c" * 40}, {}),
    "records_commit": ({}, {"records_commit": "d" * 40}),
    "stage": ({"stage": "E3-R9"}, {}),
    "seed": ({"seed": 43}, {}),
    "horizon": ({"horizon": 40000}, {}),
    "sweep": ({"sweep": "alpha_cwd", "value": 25.0}, {}),
    "lambda_logit": ({"lambda_logit": 1.0}, {}),
    "alpha_cwd": ({"alpha_cwd": 50.0}, {}),
    "launch_time_utc": ({}, {"launch_time_utc": LAUNCH_T["R1"] + 1.0}),
}


def _log_field_case(key: str, launch_over: dict, launched_over: dict):
    def fn():
        root = make_records(files={SS.LAUNCH_LOG_REL: jl(base_log(launch_over, launched_over))})
        return refusal("launch_log", records=root, field=f"{key}:")
    return fn


for _key, (_lo, _ldo) in LOG_FIELD_CASES.items():
    case(f"r_launch_log_{_key}", "launch_log")(_log_field_case(_key, _lo, _ldo))


# ---- the write
@case("r_out_created_before_write", "out_exists")
def _r_out_race():
    out = new_out()
    saved = C.launch_binding

    def racing(*a, **k):
        write(out, b"kept\n")
        return saved(*a, **k)
    C.launch_binding = racing
    try:
        rc, text = run_cli(record_argv(FX["run_yes"], out))
    finally:
        C.launch_binding = saved
    return (rc == 2 and result_of(text).startswith("RESULT: REFUSED [out_exists]")
            and out.read_bytes() == b"kept\n"), result_of(text)


@case("s_write_failed", "write_failed")
def _s_write_failed():
    out = new_out()

    def full_disk(path, mode="r", *a, **k):
        fh = open(path, mode, *a, **k)                        # the exclusive create succeeds, the write fails

        class Full:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fh.close()
                return False

            def write(self, data):
                raise OSError(28, "No space left on device")
        return Full()
    C.open = full_disk                                        # the module's open (a global), restored below
    try:
        rc, text = run_cli(record_argv(FX["run_yes"], out))
    finally:
        del C.open
    return (rc == 1 and result_of(text).startswith(f"RESULT: STOP [write_failed] -- {out} was created but not completed")
            and out.is_file() and out.stat().st_size == 0), result_of(text)


@case("s_write_mismatch", "write_mismatch")
def _s_write():
    out = new_out()
    saved = C._read_back
    C._read_back = lambda path: b"another record\n"
    try:
        rc, text = run_cli(record_argv(FX["run_yes"], out))
    finally:
        C._read_back = saved
    return rc == 1 and result_of(text).startswith("RESULT: STOP [write_mismatch]") and out.is_file(), result_of(text)


# ================================================================== aq5_: the tool's identity on a code repository
def real_identity(code_root: Path) -> tuple[int, str, Path, str]:
    head = git(code_root, "rev-parse", "HEAD").strip()
    out = new_out()
    saved = P.REPO
    P.REPO = code_root
    try:
        rc, text = run_cli(record_argv(FX["run_yes"], out, flags={"--expect-head": head}), identity=False)
    finally:
        P.REPO = saved
    return rc, text, out, head


@case("aq5_identity_clean_record")
def _aq5_clean():
    rc, text, out, head = real_identity(code_repo())
    tool = json.loads(out.read_bytes())["tool"] if out.is_file() else {}
    want = {"code_paths": list(C.TOOL_CODE_PATHS), "commit": head, "clean_at_commit": True,
            "files_sha256": {p: FX["file_sha"][p] for p in C.TOOL_CODE_PATHS if p in FX["file_sha"]}}
    return rc == 0 and tool == want, result_of(text)


@case("aq5_untracked_arm_file_not_clean", "code_not_clean")
def _aq5_untracked():
    rc, text, out, _ = real_identity(code_repo(untracked={"configs/kd_arms/e3_r9.json": b"{}\n"}))
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [code_not_clean]") and nothing_created(out), \
        result_of(text)


@case("aq5_edited_arm_file_not_clean", "code_not_clean")
def _aq5_edited_arm():
    rel = "configs/kd_arms/e3_r1.json"
    rc, text, out, _ = real_identity(code_repo(after={rel: (REPO / rel).read_bytes() + b"\n"}))
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [code_not_clean]") and nothing_created(out), \
        result_of(text)


@case("aq5_edited_code_file_not_clean", "code_not_clean")
def _aq5_edited_code():
    rel = "src/training/sweep_select.py"
    rc, text, out, _ = real_identity(code_repo(after={rel: (REPO / rel).read_bytes() + b"# edited\n"}))
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [code_not_clean]") and nothing_created(out), \
        result_of(text)


@case("aq5_tool_code_paths_cover")
def _aq5_cover():
    registry = git(REPO, "ls-tree", "-r", "--name-only", "HEAD", "--", "configs/kd_arms").split()
    need = {"src/quant/e6x_condition.py", "scripts/e6x_condition.py", "src/quant/qat.py", "src/quant/runner.py",
            "src/training/train_e1.py", "src/quant/ptq.py", "src/training/sweep_select.py", "src/training/kd_arms.py",
            "configs/kd_arms", *registry}
    missing = sorted(need - set(C.TOOL_CODE_PATHS))
    return len(registry) >= 2 and not missing, f"registry {registry}; missing {missing}"


# ================================================================== d_: the readers' refusals
@case("d_condition_sha256_format", "condition_sha256_format")
def _d_sha_format():
    data, rec, s = written_record()
    path, _ = record_file(rec)
    rc, text = print_dl(path, "xyz")
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_sha256_format]"), result_of(text)


@case("d_condition_missing_file", "condition_missing_file")
def _d_missing():
    rc, text = print_dl(new_dir("rec") / "absent.json", "0" * 64)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_missing_file]"), result_of(text)


@case("d_condition_sha256_mismatch", "condition_sha256_mismatch")
def _d_sha_mismatch():
    data, rec, s = written_record()
    path, _ = record_file(rec)
    rc, text = print_dl(path, "0" * 64)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_sha256_mismatch]"), result_of(text)


@case("d_condition_format_keys", "condition_format")
def _d_format_keys():
    data, rec, s = written_record()
    path, s2 = record_file({**rec, "extra": 1})
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_format]"), result_of(text)


@case("d_condition_format_nan", "condition_format")
def _d_format_nan():
    data, rec, s = written_record()
    rec["run_end"]["best_val_miou_all_class"] = float("nan")       # outside the value block: the strict parse
    path, s2 = record_file(rec)
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_format]") and b"NaN" in path.read_bytes(), \
        result_of(text)


@case("d_condition_format_value", "condition_format")
def _d_format_value():
    data, rec, s = written_record()
    rec["value"].update(best_val_miou_all_class=1.5, repr="1.5")    # strict JSON, outside [0, 1] (AQ4)
    path, s2 = record_file(rec)
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_format]"), result_of(text)


@case("d_condition_format_threshold", "condition_format")
def _d_format_threshold():
    data, rec, s = written_record()
    rec["threshold"]["repr"] = "0.36"
    path, s2 = record_file(rec)
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_format]"), result_of(text)


@case("d_condition_void_yes", "condition_void")
def _d_void_yes():
    data, rec, s = written_record(make_run("R1", T))
    rec["verdict"] = "yes"
    path, s2 = record_file(rec)
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_void]"), result_of(text)


@case("d_condition_void_no", "condition_void")
def _d_void_no():
    data, rec, s = written_record(make_run("R1", math.nextafter(T, 1.0)))
    rec["verdict"] = "no"
    path, s2 = record_file(rec)
    rc, text = print_dl(path, s2)
    return rc == 2 and result_of(text).startswith("RESULT: REFUSED [condition_void]"), result_of(text)


@case("d_read_record_file_forbidden_path", "condition_forbidden_path")
def _d_rf_forbid():
    return api_refusal("condition_forbidden_path", C.read_record_file, FX["work"] / "latest_records" / "x.json",
                       "0" * 64)


@case("d_read_condition_sha256_format", "condition_sha256_format")
def _d_rc_sha_format():
    return api_refusal("condition_sha256_format", C.read_condition, COND_REL, "xyz", repo_root=FX["records"])


@case("d_read_condition_not_repo_relative", "condition_path_not_repo_relative")
def _d_rc_abs():
    return api_refusal("condition_path_not_repo_relative", C.read_condition, str(FX["work"] / "x.json"), "0" * 64,
                       repo_root=FX["records"])


@case("d_read_condition_forbidden_path", "condition_forbidden_path")
def _d_rc_forbid():
    return api_refusal("condition_forbidden_path", C.read_condition, "reports/arms/latest_e6x_condition.json",
                       "0" * 64, repo_root=FX["records"])


@case("d_read_condition_missing_file", "condition_missing_file")
def _d_rc_missing():
    return api_refusal("condition_missing_file", C.read_condition, "reports/arms/e6x_condition_R9.json", "0" * 64,
                       repo_root=FX["records"])


@case("d_read_condition_not_tracked", "condition_not_tracked")
def _d_rc_untracked():
    data, rec, s = written_record()
    root = make_records(untracked={COND_REL: data})
    return api_refusal("condition_not_tracked", C.read_condition, COND_REL, s, repo_root=root)


@case("d_read_condition_changed_since_head", "condition_changed_since_head")
def _d_rc_changed():
    data, rec, s = written_record()
    root = make_records(files={COND_REL: data}, after={COND_REL: data + b"\n"})
    return api_refusal("condition_changed_since_head", C.read_condition, COND_REL, s, repo_root=root)


@case("d_read_condition_sha256_mismatch", "condition_sha256_mismatch")
def _d_rc_sha():
    data, rec, s = written_record()
    root = make_records(files={COND_REL: data})
    return api_refusal("condition_sha256_mismatch", C.read_condition, COND_REL, "0" * 64, repo_root=root)


# ================================================================== h_: hygiene (run last)
def module_codes() -> set:
    """Every refusal, incomplete and STOP code the tool names (its two source files)."""
    src = ((REPO / "src" / "quant" / "e6x_condition.py").read_text(encoding="utf-8")
           + (REPO / "scripts" / "e6x_condition.py").read_text(encoding="utf-8"))
    codes = set(re.findall(r'_refuse\(\s*"([a-z0-9_]+)"', src))
    codes |= set(re.findall(r'(?:missing|uncommitted)="([a-z0-9_]+)"', src))
    codes |= set(re.findall(r'Condition(?:Incomplete|Stop)\(\s*"([a-z0-9_]+)"', src))
    codes |= set(re.findall(r"RESULT: REFUSED \[([a-z0-9_]+)\]", src))
    return codes


@case("h_every_code_has_a_case")
def _h_codes():
    have = {code for _, code in CASES.values() if code}
    missing = sorted(module_codes() - have)
    return not missing and "forbidden_path" in module_codes(), f"codes without a case: {missing}"


@case("h_threshold_is_alpha_winner")
def _h_threshold():
    winner = json.loads((REPO / C.ALPHA_SELECTION_REL).read_text(encoding="utf-8"))["winner"]
    return repr(winner["best_val"]) == C.THRESHOLD and winner["alpha"] == 25.0, repr(winner["best_val"])


@case("h_no_forbidden_named_path_created")
def _h_names():
    bad = []
    for root, dirs, files in os.walk(FX["work"]):
        bad += [os.path.join(root, n) for n in dirs + files if "test" in n.lower() or ":" in n]
    return not bad, f"{bad[:5]}"


@case("h_no_forbidden_path_opened")
def _h_opened():
    return not OPENED, f"{OPENED[:5]}"


@case("h_repository_untouched")
def _h_repo():
    now = repo_status()
    return now == FX["status0"], f"before {FX['status0']!r} after {now!r}"


# ------------------------------------------------------------------ main
def audit(event: str, args) -> None:
    if event not in ("open", "os.listdir", "os.scandir") or not args:
        return
    try:
        p = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    for base in AUDIT_BASES:
        if p.startswith(base) and "test" in p[len(base):].lower():
            OPENED.append(p)


def run_case(name: str) -> tuple[bool, str]:
    fn, _ = CASES[name]
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 -- a case that raises fails with this detail
        ok, detail = False, f"{type(e).__name__}: {e}"
    return bool(ok), str(detail)


def choose_work_dir(value: str | None) -> Path:
    if value is not None:
        if Q.names_test(value):
            raise SystemExit(f"--work-dir {value!r} contains 'test'")
        work = R.check_output_dir(value)
        if work.exists():
            raise SystemExit(f"--work-dir {work} exists; a work dir is never reused")
        return work
    for var in ("TMPDIR", "TEMP", "TMP"):
        if os.environ.get(var) and Q.names_test(os.environ[var]):
            raise SystemExit(f"${var} contains 'test'; set it elsewhere")
    n = 0
    while True:
        n += 1
        work = R.check_output_dir(str(Path(tempfile.gettempdir()) / f"e6x_condition_smoke_{os.getpid()}_{n}"))
        if not work.exists():
            return work


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--only", default=None, help="comma-separated case names")
    a = ap.parse_args(argv)
    work = choose_work_dir(a.work_dir)
    work.mkdir(parents=True)
    (work / "scratch").mkdir()
    (work / "relative_base").mkdir()
    tempfile.tempdir = str(work / "scratch")
    os.chdir(work / "relative_base")             # a relative --out or --run-dir resolves here, never in the repository
    AUDIT_BASES[:] = [str(work), str(REPO)]
    sys.addaudithook(audit)
    t0 = time.time()
    print("=" * 78)
    print("E6-X CONDITION SMOKE (L-AM23Q B-1) -- CPU, synthetic records repository and run directories")
    print(f"work dir: {work}")
    print("=" * 78, flush=True)
    setup(work)
    names = list(CASES) if not a.only else [n.strip() for n in a.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"RESULT: ERROR unknown cases {unknown}")
        return 2
    for name in [n for n in names if not n.startswith("h_")] + [n for n in names if n.startswith("h_")]:
        ok, detail = run_case(name)
        RESULTS.append((name, ok, detail))
    print("\n[CHECKS]")
    for name, ok, detail in RESULTS:
        print(f"  {name:44}: {'PASS' if ok else 'FAIL'}  {detail[:400] if not ok else ''}".rstrip())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n({round(time.time() - t0, 1)} s)")
    print(f"RESULT: {'PASS' if RESULTS and passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if RESULTS and passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
