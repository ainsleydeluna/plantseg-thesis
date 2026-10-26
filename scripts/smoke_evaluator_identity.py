#!/usr/bin/env python3
"""Smoke: IF-T1's three flags in scripts/evaluate_model.py (lane LEFF B-1b; TL-6(b), TL-1).

  --expected-checkpoint-sha256 H   the sha256 of the handed file (--checkpoint, or --provenance for an INT8
                                   stage): hashed before its first parse (P1) and again after validation (P2)
  --expected-converted-sha256 H2   an INT8 stage's converted artifact: compared after the official gate (P2)
                                   and re-hashed immediately before the load (P3)
  --threads T                      torch.set_num_threads(T) before anything is loaded or built, read back, and
                                   checked against run.eval_runtime.torch_num_threads before the write

Each flag is required under R (--split test with --artifact-status official), optional elsewhere, refused
where it does not apply, and checked after every existing guard; without the flags nothing is hashed,
compared or set. The checks run on scripts/evaluate_model.py; then each mutation edits a copy of it outside
the repository, loaded under its own module name, and must make its named check FAIL.

Synthetic only (DL-41): build_fp32_student checkpoints; the stub teacher of smoke_eval_teacher's builder
pattern, through scripts/teacher_diag_fixtures.py; INT8 state_dict artifacts by smoke_eval_int8's generator
pattern; a make_tree VAL root in safe_tmpdir. Run checks use --split val, where a given flag is always
compared; the only --split test calls are refused before any dataset, and every refused run's never-created
out-dir is asserted absent. The thread count is restored after every check. The artifact writer's git
status and HEAD are stubbed, as smoke_run_ptq does (a cloud lane never lists docs/reference/, DL-37).

    python -B scripts/smoke_evaluator_identity.py [--no-mutations]

Ends with one RESULT line, RESULT: PASS (n/n; m/m edits killed); exit 0 only when every check passes and
every edit is killed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import make_tree, repo_state, safe_tmpdir, state_verdict  # noqa: E402

STATE_BEFORE = repo_state(REPO)          # the repository as the run found it (repo_state_unchanged)
BASE = safe_tmpdir("evaluator_identity_")
ROOT = BASE / "plantseg"
make_tree(ROOT, n_train=4, n_val=846, seed=5)          # images/test and annotations/test: never created
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)          # before configs/data.py is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402

import scripts.evaluate_model as EM_REAL  # noqa: E402
import scripts.teacher_diag_fixtures as TDF  # noqa: E402
import src.eval.adapters as ADP  # noqa: E402
import src.eval.artifacts as ART  # noqa: E402
import src.eval.eval_runtime as ERT  # noqa: E402
import src.eval.model_loading as ML  # noqa: E402
import src.eval.stage_artifacts as SA  # noqa: E402
from src.distill.segnext_teacher import segnext_builder  # noqa: E402
from src.distill.teacher import sha256_format_error  # noqa: E402

FIX, OUTS, NEVER = BASE / "fixtures", BASE / "outs", BASE / "never_created"
TEACHER_CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
TEACHER_BUILDER = segnext_builder(model_factory=TDF.stub_factory)
H_ANY, H_OTHER = "a" * 64, "b" * 64          # well-formed sha256 values that name no fixture file
SEQ = itertools.count(1)
F: dict = {}                                 # the shared fixtures, built once (build_fixtures); never edited
TOKEN = "".join(("t", "e", "s", "t"))        # the name no created path, check or case may hold (SL-1, LB36)

# LF4's texts, verbatim (L-CORR-B2's evaluate_corruptions.py at da8469c, not_applicable aside)
TEMPLATES = {
    "expected_checkpoint_sha256_required":
        "--split test with --artifact-status official requires --expected-checkpoint-sha256, the sha256 of the "
        "file this run is handed (--checkpoint, or --provenance for an INT8 stage)",
    "expected_checkpoint_sha256_format": "--expected-checkpoint-sha256 {fmt}",
    "expected_checkpoint_sha256_not_applicable":
        "--expected-checkpoint-sha256 names the file a run is handed; --random-init is handed none",
    "expected_checkpoint_sha256_teacher_disagrees":
        "--expected-checkpoint-sha256 {e} and --teacher-ckpt-sha256 {t} differ; for the teacher both name "
        "--checkpoint and must be equal",
    "expected_checkpoint_sha256_mismatch": "{path} has sha256 {actual}, expected {e}; the file was not parsed",
    "source_changed_during_validation": "{path} had sha256 {e} before validation and {actual} after it",
    "expected_converted_sha256_required":
        "--split test with --artifact-status official requires --expected-converted-sha256 for an INT8 stage, "
        "the sha256 of the converted artifact its provenance names",
    "expected_converted_sha256_format": "--expected-converted-sha256 {fmt}",
    "expected_converted_sha256_not_int8":
        "--expected-converted-sha256 applies to an INT8 stage handed --provenance; this run "
        "(precision={p!r}, random_init={r}) takes none",
    "expected_converted_sha256_mismatch":
        "the converted artifact {path} has sha256 {actual}, expected {e2}; it was not loaded",
    "converted_changed_before_load":
        "the converted artifact {path} had sha256 {e2} at validation and {actual} before the load; it was not "
        "loaded",
    "threads_invalid": "--threads must be a positive integer, got {t!r}",
    "threads_required":
        "--split test with --artifact-status official requires --threads, the torch thread count of the "
        "execution context of record",
    "threads_not_applied": "torch.set_num_threads({t}) left torch.get_num_threads() at {n}",
    "threads_drifted": "run.eval_runtime.torch_num_threads is {n}, --threads is {t}; no artifact was written",
}


def expected_text(code: str, **values) -> str:
    return f"[{code}] " + TEMPLATES[code].format(**values)


# ---------------------------------------------------------------- fixtures
def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_fix(tag: str, suffix: str) -> Path:
    return FIX / f"{tag}_{next(SEQ):03d}{suffix}"


def new_out(tag: str) -> Path:
    return OUTS / f"{tag}_{next(SEQ):03d}"


def never(tag: str) -> Path:
    return NEVER / f"{tag}_{next(SEQ):03d}"


def fp32_ckpt(path: Path, seed: int) -> Path:
    """An E1 checkpoint (no `stage`) of build_fp32_student's architecture, in train_e1.py's schema."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ML.build_fp32_student()
    torch.save({"iter": 80000, "model_state_dict": model.state_dict(), "optimizer_state_dict": {},
                "best_val_miou_all_class": 0.0, "num_classes": 116}, path)
    return path


def int8_artifact(path: Path, stage: str, method: str, seed: int) -> Path:
    """smoke_eval_int8's generator pattern: the runner schema {stage, quantization, num_classes, model}."""
    from src.models.student import build_student
    from src.quant.prepare import convert_model, prepare_ptq, prepare_qat_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        student = build_student(pretrained=False)
        prepared = (prepare_ptq(student, select_backend=False) if method == "ptq"
                    else prepare_qat_model(student, select_backend=False))
        if method == "qat":
            prepared.train()
            with torch.no_grad():
                prepared(torch.randn(1, 3, 64, 64))     # size the per-channel fake-quant qparams
        converted = convert_model(prepared)
    torch.save({"stage": stage, "quantization": method, "num_classes": 116, "model": converted.state_dict()}, path)
    return path


def int8_prov(path: Path, stage: str, art: Path) -> Path:
    source, method = {"E4": ("E1", "ptq"), "E6": ("E3", "qat")}[stage]
    payload = {"stage": stage, "source_stage": source, "quantization_method": method, "random_init": False,
               "num_classes": 116, "backend": "qnnpack", "converted_artifact": str(art),
               "converted_artifact_sha256": sha(art)}
    if stage == "E6":
        payload["cwd_projection_loaded"] = False
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def copy_fix(key: str, tag: str) -> Path:
    p = new_fix(tag, F[key].suffix)
    shutil.copyfile(F[key], p)
    return p


def int8_pair(tag: str, stage: str = "E4") -> tuple[Path, Path]:
    """An own copy of a converted artifact and a provenance naming it, for a check that changes them."""
    art = copy_fix(f"{stage.lower()}_art", f"{tag}_art")
    return art, int8_prov(new_fix(f"{tag}_prov", ".json"), stage, art)


def build_fixtures() -> None:
    FIX.mkdir()
    OUTS.mkdir()
    F["fp32"] = fp32_ckpt(FIX / "fp32_e1.pt", 7)
    F["fp32_other"] = fp32_ckpt(FIX / "fp32_e1_other.pt", 8)
    F["teacher"], F["teacher_sha"] = TDF.write_stub_ckpt(FIX / "teacher_stub.pth", tag="stub")
    F["teacher_other"], _ = TDF.write_stub_ckpt(FIX / "teacher_stub_other.pth", tag="other")
    F["e4_art"] = int8_artifact(FIX / "e4_int8.pt", "E4", "ptq", 42)
    F["e4_art_other"] = int8_artifact(FIX / "e4_int8_other.pt", "E4", "ptq", 43)
    F["e4_prov"] = int8_prov(FIX / "e4_prov.json", "E4", F["e4_art"])
    F["e6_art"] = int8_artifact(FIX / "e6_int8.pt", "E6", "qat", 44)
    F["e6_prov"] = int8_prov(FIX / "e6_prov.json", "E6", F["e6_art"])


# ---------------------------------------------------------------- calling the evaluator
def fp32_argv(ckpt, out, *extra) -> list[str]:
    return ["--stage", "E1", "--checkpoint", str(ckpt), "--split", "val", "--max-samples", "2",
            "--batch-size", "1", "--artifact-status", "smoke", "--device", "cpu", "--out-dir", str(out),
            "--run-id", "evaluator_identity", *extra]


def teacher_argv(ckpt, teacher_sha, out, *extra) -> list[str]:
    return ["--stage", "teacher", "--model-role", "teacher", "--checkpoint", str(ckpt),
            "--teacher-ckpt-sha256", teacher_sha, "--teacher-config", str(TEACHER_CONFIG), "--split", "val",
            "--max-samples", "2", "--batch-size", "1", "--artifact-status", "smoke", "--device", "cpu",
            "--out-dir", str(out), "--run-id", "evaluator_identity", *extra]


def int8_argv(prov, out, *extra, stage="E4", precision="int8_ptq", status="smoke") -> list[str]:
    return ["--stage", stage, "--precision", precision, "--provenance", str(prov), "--split", "val",
            "--max-samples", "2", "--batch-size", "1", "--artifact-status", status, "--device", "cpu",
            "--out-dir", str(out), "--run-id", "evaluator_identity", *extra]


def run_em(EM, argv, *, teacher=False):
    """run() with construction counters; (artifact dir or None, exception or None, counters)."""
    args = EM.build_parser().parse_args(argv)
    ctr = EM.Counters(dataset=[], model=[])
    out = err = None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            out = EM.run(args, counters=ctr, teacher_builder=TEACHER_BUILDER if teacher else None)
        except Exception as e:  # noqa: BLE001 -- the outcome is the check's to judge
            err = e
    return out, err, ctr


def code_of(err) -> str | None:
    if err is None:
        return None
    m = re.match(r"\[([a-z0-9_]+)\] ", str(err))
    return m.group(1) if m else f"{type(err).__name__}: {str(err)[:120]}"


def summary(out) -> dict:
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def scored(result) -> bool:
    out, err, _ = result
    return err is None and out is not None and (Path(out) / "summary.json").is_file()


@contextlib.contextmanager
def patched(obj, **attrs):
    """Set attributes of a module or object for one case; restored in a finally block."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def counted(fn, box: dict, key: str, probe=None):
    """Wrap fn: each call appends probe() (or None) to box[key]."""
    def inner(*a, **k):
        box.setdefault(key, []).append(probe() if probe else None)
        return fn(*a, **k)
    return inner


def other_threads() -> int:
    return 2 if torch.get_num_threads() != 2 else 3


# ---------------------------------------------------------------- CLI arguments
def argv_of(kw: dict) -> list[str]:
    argv = []
    for k, v in kw.items():
        if v is True:
            argv.append("--" + k.replace("_", "-"))
        elif v is not None and v is not False:
            argv += ["--" + k.replace("_", "-"), str(v)]
    return argv


def cli_kw(**kw) -> dict:
    base = dict(stage="E1", precision="fp32", split="val", out_dir=str(never("cli_out")), artifact_status="smoke",
                device="cpu", batch_size=1, checkpoint=str(never("e1_ckpt.pt")))
    return {**base, **kw}


def cli(EM, **kw):
    with contextlib.redirect_stderr(io.StringIO()):
        return EM.build_parser().parse_args(argv_of(cli_kw(**kw)))


def preset(kind: str, **kw) -> dict:
    """R (--split test with --artifact-status official) with every flag that kind takes, all well formed."""
    r = dict(split="test", artifact_status="official", confirm_test_split=True,
             expected_checkpoint_sha256=H_ANY, threads="1")
    base = {"fp32": dict(stage="E1"),
            "teacher": dict(stage="teacher", model_role="teacher", checkpoint=str(never("teacher.pth")),
                            teacher_config=str(TEACHER_CONFIG), teacher_ckpt_sha256=H_ANY),
            "int8": dict(stage="E4", precision="int8_ptq", checkpoint=None,
                         provenance=str(never("e4_prov.json")), expected_converted_sha256=H_OTHER),
            "int8_qat": dict(stage="E5", precision="int8_qat", checkpoint=None,
                             provenance=str(never("e5_prov.json")), expected_converted_sha256=H_OTHER)}[kind]
    return {**base, **r, **kw}


def offr(kind: str, **kw) -> dict:
    """Outside R: --split val with smoke status and no new flag, unless kw adds one."""
    return {**preset(kind, split="val", artifact_status="smoke", confirm_test_split=None,
                     expected_checkpoint_sha256=None, expected_converted_sha256=None, threads=None), **kw}


def cli_outcome(EM, args) -> str:
    try:
        EM.validate_cli_args(args)
        return "accepted"
    except EM.CliError as e:
        m = re.match(r"\[([a-z0-9_]+)\] ", str(e))
        return m.group(1) if m else f"refused: {e}"


def cli_text_outcome(EM, kw: dict) -> str:
    """As cli_outcome, through the parser, which may refuse first (exit 2)."""
    try:
        args = cli(EM, **kw)
    except SystemExit as e:
        return f"argparse exit {e.code}"
    return cli_outcome(EM, args)


CHECKS: list[tuple[str, str, object]] = []


def check(name: str, group: str):
    def deco(fn):
        CHECKS.append((name, group, fn))
        return fn
    return deco


# ---------------------------------------------------------------- CLI (14)
@check("ecs_required_under_r", "cli")
def _ecs_required(EM):
    got = {k: cli_outcome(EM, cli(EM, **preset(k, expected_checkpoint_sha256=None)))
           for k in ("fp32", "teacher", "int8", "int8_qat")}
    return all(v == "expected_checkpoint_sha256_required" for v in got.values()), str(got)


@check("ecs_optional_off_r", "cli")
def _ecs_optional(EM):
    cases = {"fp32_val": offr("fp32"),
             "fp32_val_flags_given": offr("fp32", expected_checkpoint_sha256=H_ANY, threads="2"),
             "fp32_heldout_split_provisional": offr("fp32", split="test", artifact_status="provisional",
                                                    confirm_test_split=True),
             "teacher_val": offr("teacher"),
             "int8_val": offr("int8"),
             "int8_val_flags_given": offr("int8", expected_checkpoint_sha256=H_ANY,
                                          expected_converted_sha256=H_OTHER),
             "int8_val_official": offr("int8", artifact_status="official")}
    got = {k: cli_outcome(EM, cli(EM, **v)) for k, v in cases.items()}
    return all(v == "accepted" for v in got.values()), str(got)


@check("ecs_format_uppercase_refused", "cli")
def _ecs_upper(EM):
    up = H_ANY.upper()
    got = [cli_outcome(EM, cli(EM, **preset("fp32", expected_checkpoint_sha256=up))),
           cli_outcome(EM, cli(EM, **offr("fp32", expected_checkpoint_sha256=up)))]
    return all(v == "expected_checkpoint_sha256_format" for v in got), str(got)


@check("ecs_format_empty_refused", "cli")
def _ecs_empty(EM):
    got = [cli_outcome(EM, cli(EM, **preset("fp32", expected_checkpoint_sha256=""))),
           cli_outcome(EM, cli(EM, **offr("int8", expected_checkpoint_sha256="")))]
    return all(v == "expected_checkpoint_sha256_format" for v in got), str(got)


@check("ecs_refused_for_random_init", "cli")
def _ecs_random_init(EM):
    got = [cli_outcome(EM, cli(EM, **offr("fp32", checkpoint=None, random_init=True,
                                          expected_checkpoint_sha256=H_ANY))),
           cli_outcome(EM, cli(EM, **offr("fp32", checkpoint=None, random_init=True,
                                          expected_checkpoint_sha256="")))]     # before the format check
    return all(v == "expected_checkpoint_sha256_not_applicable" for v in got), str(got)


@check("ecs_teacher_flags_disagree_refused", "cli")
def _ecs_teacher_disagree(EM):
    got = [cli_outcome(EM, cli(EM, **preset("teacher", expected_checkpoint_sha256=H_OTHER))),
           cli_outcome(EM, cli(EM, **offr("teacher", expected_checkpoint_sha256=H_OTHER)))]
    return all(v == "expected_checkpoint_sha256_teacher_disagrees" for v in got), str(got)


@check("ecs_teacher_flags_equal_accepted", "cli")
def _ecs_teacher_equal(EM):
    got = [cli_outcome(EM, cli(EM, **preset("teacher"))),
           cli_outcome(EM, cli(EM, **offr("teacher", expected_checkpoint_sha256=H_ANY)))]
    return all(v == "accepted" for v in got), str(got)


@check("ecv_required_under_r_int8", "cli")
def _ecv_required(EM):
    got = [cli_outcome(EM, cli(EM, **preset(k, expected_converted_sha256=None))) for k in ("int8", "int8_qat")]
    return all(v == "expected_converted_sha256_required" for v in got), str(got)


@check("ecv_refused_off_int8", "cli")
def _ecv_off_int8(EM):
    got = [cli_outcome(EM, cli(EM, **offr("fp32", expected_converted_sha256=H_OTHER))),
           cli_outcome(EM, cli(EM, **offr("teacher", expected_converted_sha256=H_OTHER))),
           cli_outcome(EM, cli(EM, **offr("fp32", checkpoint=None, random_init=True,
                                          expected_converted_sha256=H_OTHER))),
           cli_outcome(EM, cli(EM, **preset("fp32", expected_converted_sha256=H_OTHER)))]
    return all(v == "expected_converted_sha256_not_int8" for v in got), str(got)


@check("ecv_format_refused", "cli")
def _ecv_format(EM):
    got = [cli_outcome(EM, cli(EM, **preset("int8", expected_converted_sha256=H_OTHER.upper()))),
           cli_outcome(EM, cli(EM, **preset("int8", expected_converted_sha256=""))),
           cli_outcome(EM, cli(EM, **offr("int8", expected_converted_sha256="0x" + H_OTHER[2:])))]
    return all(v == "expected_converted_sha256_format" for v in got), str(got)


@check("threads_required_under_r", "cli")
def _threads_required(EM):
    got = [cli_outcome(EM, cli(EM, **preset(k, threads=None))) for k in ("fp32", "teacher", "int8")]
    return all(v == "threads_required" for v in got), str(got)


@check("threads_invalid_refused", "cli")
def _threads_invalid(EM):
    got = [cli_outcome(EM, cli(EM, **preset("fp32", threads=v))) for v in ("0", "-1")]
    return all(v == "threads_invalid" for v in got), str(got)


@check("existing_guards_keep_precedence", "cli")
def _existing_first(EM):
    cases = {
        "confirmation": (preset("fp32", confirm_test_split=None, expected_checkpoint_sha256=None, threads=None),
                         "requires --confirm-test-split"),
        "sample_cap": (preset("fp32", max_samples=4, expected_checkpoint_sha256=None, threads=None),
                       "forbids any sample cap"),
        "smoke_status": (preset("fp32", artifact_status="smoke", expected_checkpoint_sha256=None, threads=None),
                         "refuses artifact_status=smoke"),
        "int8_two_sources": (preset("int8", checkpoint=str(never("e4_ckpt.pt")), expected_checkpoint_sha256=None,
                                    expected_converted_sha256=None, threads=None), "both identify a model source"),
        "teacher_sha_required": (preset("teacher", teacher_ckpt_sha256=None, expected_checkpoint_sha256=None,
                                        threads=None), "[teacher_ckpt_sha256_required]"),
        "teacher_sha_not_teacher": (preset("fp32", teacher_ckpt_sha256=H_ANY, expected_converted_sha256=H_OTHER),
                                    "[teacher_ckpt_sha256_not_teacher]"),
    }
    got = {}
    for k, (kw, _) in cases.items():
        try:
            EM.validate_cli_args(cli(EM, **kw))
            got[k] = "accepted"
        except EM.CliError as e:
            got[k] = str(e)
    ok = all(needle in got[k] and "[expected_" not in got[k] and "[threads_" not in got[k]
             for k, (_, needle) in cases.items())
    return ok, str({k: v[:60] for k, v in got.items()})


@check("threads_non_integer_refused", "cli")
def _threads_non_integer(EM):
    got = {v: cli_text_outcome(EM, offr("fp32", threads=v)) for v in ("abc", "1.5", "+1", "\u0663", "0", "-1")}
    prog = cli(EM, **offr("fp32"))
    prog.threads = True
    got["True (programmatic)"] = cli_outcome(EM, prog)
    good = {"2": cli_text_outcome(EM, offr("fp32", threads="2"))}
    prog = cli(EM, **offr("fp32"))
    prog.threads = 2
    good["2 (programmatic)"] = cli_outcome(EM, prog)
    ok = all(v == "threads_invalid" for v in got.values()) and all(v == "accepted" for v in good.values())
    return ok, str({**got, **good})


# ---------------------------------------------------------------- run (14)
@check("fp32_mismatch_before_any_parse", "run")
def _fp32_mismatch(EM):
    junk = new_fix("fp32_unparsable", ".pt")
    junk.write_bytes(b"no torch archive in this file\n")
    box, out = {}, never("fp32_mismatch")
    with patched(torch, load=counted(torch.load, box, "torch.load")):
        _, err, ctr = run_em(EM, fp32_argv(junk, out, "--expected-checkpoint-sha256", sha(F["fp32"])))
    ok = (code_of(err) == "expected_checkpoint_sha256_mismatch" and not box and ctr.dataset == []
          and ctr.model == [] and not out.exists())
    return ok, f"{code_of(err)}; torch.load calls {len(box.get('torch.load', []))}; dataset {ctr.dataset}"


@check("fp32_match_scores_and_records_identity", "run")
def _fp32_match(EM):
    ck = copy_fix("fp32", "fp32_match")
    res = run_em(EM, fp32_argv(ck, new_out("fp32_match"), "--expected-checkpoint-sha256", sha(ck)))
    s = summary(res[0]) if scored(res) else {}
    ok = scored(res) and s["run"]["checkpoint_sha256"] == sha(ck) and s["dataset"]["actual_rows"] == 2
    return ok, f"{code_of(res[1])}; run.checkpoint_sha256 {s.get('run', {}).get('checkpoint_sha256', '-')[:16]}"


@check("fp32_changed_during_validation_refused", "run")
def _fp32_changed(EM):
    ck = copy_fix("fp32", "fp32_changing")
    h, other = sha(ck), F["fp32_other"].read_bytes()
    real = SA.validate_fp32_artifact

    def changing(stage, path):
        res = real(stage, path)
        Path(path).write_bytes(other)          # a valid E1 checkpoint of other weights, written during validation
        return res
    out = never("fp32_changing")
    with patched(SA, validate_fp32_artifact=changing):
        _, err, ctr = run_em(EM, fp32_argv(ck, out, "--expected-checkpoint-sha256", h))
    ok = code_of(err) == "source_changed_during_validation" and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}; dataset {ctr.dataset}"


@check("teacher_mismatch_before_any_parse", "run")
def _teacher_mismatch(EM):
    junk = new_fix("teacher_unparsable", ".pth")
    junk.write_bytes(b"no teacher checkpoint in this file\n")
    h = F["teacher_sha"]                       # H must equal --teacher-ckpt-sha256; neither names these bytes
    box, out = {}, never("teacher_mismatch")
    with patched(torch, load=counted(torch.load, box, "torch.load")), \
            patched(SA, validate_teacher_artifact=counted(SA.validate_teacher_artifact, box, "validate")):
        _, err, ctr = run_em(EM, teacher_argv(junk, h, out, "--expected-checkpoint-sha256", h), teacher=True)
    ok = code_of(err) == "expected_checkpoint_sha256_mismatch" and not box and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}; calls {sorted((k, len(v)) for k, v in box.items())}"


@check("teacher_match_scores", "run")
def _teacher_match(EM):
    ck = copy_fix("teacher", "teacher_match")
    h = sha(ck)
    res = run_em(EM, teacher_argv(ck, h, new_out("teacher_match"), "--expected-checkpoint-sha256", h), teacher=True)
    s = summary(res[0]) if scored(res) else {}
    ok = scored(res) and s["run"]["checkpoint_sha256"] == h and s["dataset"]["actual_rows"] == 2
    return ok, f"{code_of(res[1])}"


@check("int8_provenance_mismatch_before_parse", "run")
def _int8_prov_mismatch(EM):
    junk = new_fix("e4_prov_unparsable", ".json")
    junk.write_text("{ this is not JSON", encoding="utf-8")
    box, out = {}, never("int8_prov_mismatch")
    with patched(SA, validate_int8_artifact=counted(SA.validate_int8_artifact, box, "validate")):
        _, err, ctr = run_em(EM, int8_argv(junk, out, "--expected-checkpoint-sha256", sha(F["e4_prov"])))
    ok = code_of(err) == "expected_checkpoint_sha256_mismatch" and not box and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}; validator calls {len(box.get('validate', []))}"


@check("int8_provenance_changed_during_validation_refused", "run")
def _int8_prov_changed(EM):
    prov = copy_fix("e4_prov", "e4_prov_changing")
    h = sha(prov)
    real = SA.resolve_evaluation_source

    def changing(*a, **k):
        res = real(*a, **k)
        with open(prov, "a", encoding="utf-8") as f:
            f.write("\n")                      # the same JSON in other bytes, written during validation
        return res
    out = never("int8_prov_changing")
    with patched(SA, resolve_evaluation_source=changing):
        _, err, ctr = run_em(EM, int8_argv(prov, out, "--expected-checkpoint-sha256", h))
    ok = code_of(err) == "source_changed_during_validation" and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}"


@check("int8_converted_mismatch_before_load", "run")
def _int8_conv_mismatch(EM):
    box, out = {}, never("int8_conv_mismatch")
    with patched(ML, load_int8_student=counted(ML.load_int8_student, box, "load")):
        _, err, ctr = run_em(EM, int8_argv(F["e4_prov"], out, "--expected-checkpoint-sha256", sha(F["e4_prov"]),
                                           "--expected-converted-sha256", H_OTHER))
    ok = (code_of(err) == "expected_converted_sha256_mismatch" and not box and ctr.dataset == []
          and ctr.model == [] and not out.exists())
    return ok, f"{code_of(err)}; loads {len(box.get('load', []))}"


@check("int8_converted_changed_before_load_refused", "run")
def _int8_conv_changed(EM):
    art, prov = int8_pair("e4_changing")
    h, h2, other = sha(prov), sha(art), F["e4_art_other"].read_bytes()
    real = ADP.build_protocol_adapter

    def changing(*a, **k):
        art.write_bytes(other)                 # after P2, before the load
        return real(*a, **k)
    box, out = {}, never("int8_conv_changing")
    with patched(ADP, build_protocol_adapter=changing), \
            patched(ML, load_int8_student=counted(ML.load_int8_student, box, "load")):
        _, err, _ = run_em(EM, int8_argv(prov, out, "--expected-checkpoint-sha256", h,
                                         "--expected-converted-sha256", h2))
    ok = (code_of(err) == "converted_changed_before_load" and isinstance(err, ERT.EvalRuntimeError)
          and not box and not out.exists())
    return ok, f"{code_of(err)}; loads {len(box.get('load', []))}"


@check("int8_match_scores_and_records_converted", "run")
def _int8_match(EM):
    res = run_em(EM, int8_argv(F["e4_prov"], new_out("int8_match"), "--expected-checkpoint-sha256",
                               sha(F["e4_prov"]), "--expected-converted-sha256", sha(F["e4_art"])))
    s = summary(res[0]) if scored(res) else {}
    ok = (scored(res) and s["run"]["checkpoint_sha256"] == sha(F["e4_art"]) and s["run"]["precision"] == "int8_ptq"
          and s["dataset"]["actual_rows"] == 2)
    return ok, f"{code_of(res[1])}"


@check("int8_official_order", "run")
def _int8_order(EM):
    gate, outs = {}, [never("order_h"), never("order_gate"), never("order_h2")]
    with patched(SA, int8_official_error=counted(SA.int8_official_error, gate, "gate")):
        # a wrong H is refused before the official gate is consulted
        _, e1, c1 = run_em(EM, int8_argv(F["e4_prov"], outs[0], "--expected-checkpoint-sha256", H_OTHER,
                                         "--expected-converted-sha256", sha(F["e4_art"]), status="official"))
        n1 = len(gate.get("gate", []))
        # a right H and a wrong H2: the gate refuses this legacy state_dict PTQ artifact before H2 is compared
        _, e2, c2 = run_em(EM, int8_argv(F["e4_prov"], outs[1], "--expected-checkpoint-sha256", sha(F["e4_prov"]),
                                         "--expected-converted-sha256", H_OTHER, status="official"))
        n2 = len(gate.get("gate", []))
        # a QAT artifact passes the gate; then the wrong H2 is refused
        _, e3, c3 = run_em(EM, int8_argv(F["e6_prov"], outs[2], "--expected-checkpoint-sha256", sha(F["e6_prov"]),
                                         "--expected-converted-sha256", H_OTHER, stage="E6",
                                         precision="int8_qat", status="official"))
        n3 = len(gate.get("gate", []))
    ok = (code_of(e1) == "expected_checkpoint_sha256_mismatch" and n1 == 0
          and e2 is not None and "[legacy_ptq_artifact]" in str(e2) and "[expected_converted" not in str(e2)
          and n2 == 1 and code_of(e3) == "expected_converted_sha256_mismatch" and n3 == 2
          and c1.dataset == c2.dataset == c3.dataset == [] and not any(p.exists() for p in outs))
    return ok, f"H: {code_of(e1)} (gate {n1}); H2 on E4: {str(e2)[:60]} (gate {n2}); H2 on E6: {code_of(e3)}"


def score_view(out) -> dict:
    s = summary(out)
    with np.load(Path(out) / "sufficient_stats.npz", allow_pickle=False) as z:
        stats = {k: z[k].tolist() for k in sorted(z.files)}
    return {"per_image": (Path(out) / "per_image.jsonl").read_bytes(), "stats": stats,
            "dataset_level": s.get("dataset_level"), "am5": s.get("am5"),
            "checkpoint_sha256": s["run"]["checkpoint_sha256"], "eval_runtime": s["run"]["eval_runtime"]}


@check("flags_change_no_score", "run")
def _flags_no_score(EM):
    t = str(torch.get_num_threads())          # the same context: the flags alone differ
    pairs = {
        "fp32": (run_em(EM, fp32_argv(F["fp32"], new_out("fp32_plain"))),
                 run_em(EM, fp32_argv(F["fp32"], new_out("fp32_flagged"), "--expected-checkpoint-sha256",
                                      sha(F["fp32"]), "--threads", t))),
        "int8": (run_em(EM, int8_argv(F["e4_prov"], new_out("int8_plain"))),
                 run_em(EM, int8_argv(F["e4_prov"], new_out("int8_flagged"), "--expected-checkpoint-sha256",
                                      sha(F["e4_prov"]), "--expected-converted-sha256", sha(F["e4_art"]),
                                      "--threads", t))),
        "teacher": (run_em(EM, teacher_argv(F["teacher"], F["teacher_sha"], new_out("teacher_plain")), teacher=True),
                    run_em(EM, teacher_argv(F["teacher"], F["teacher_sha"], new_out("teacher_flagged"),
                                            "--expected-checkpoint-sha256", F["teacher_sha"], "--threads", t),
                           teacher=True)),
    }
    same = {k: scored(a) and scored(b) and score_view(a[0]) == score_view(b[0]) for k, (a, b) in pairs.items()}
    return all(same.values()), str(same)


@check("teacher_changed_during_validation_refused", "run")
def _teacher_changed(EM):
    ck = copy_fix("teacher", "teacher_changing")
    h, other = sha(ck), F["teacher_other"].read_bytes()
    real = SA.validate_teacher_artifact

    def changing(path, **k):
        res = real(path, **k)
        Path(path).write_bytes(other)          # after its pre-parse hash and its parse: the recorded hash is stale
        return res
    out = never("teacher_changing")
    with patched(SA, validate_teacher_artifact=changing):
        _, err, ctr = run_em(EM, teacher_argv(ck, h, out, "--expected-checkpoint-sha256", h), teacher=True)
    ok = code_of(err) == "source_changed_during_validation" and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}; dataset {ctr.dataset[:1]}"


@check("flags_absent_nothing_hashed", "run")
def _absent_nothing_hashed(EM):
    box, calls, runs = {}, {}, {}
    with patched(EM, identity_sha256=counted(EM.identity_sha256, box, "hash")):
        for kind in ("teacher", "fp32", "int8"):
            before = len(box.get("hash", []))
            if kind == "teacher":
                runs[kind] = run_em(EM, teacher_argv(F["teacher"], F["teacher_sha"], new_out("absent_teacher")),
                                    teacher=True)
            elif kind == "fp32":
                runs[kind] = run_em(EM, fp32_argv(F["fp32"], new_out("absent_fp32")))
            else:
                runs[kind] = run_em(EM, int8_argv(F["e4_prov"], new_out("absent_int8")))
            calls[kind] = len(box.get("hash", [])) - before
    ok = all(scored(r) for r in runs.values()) and all(n == 0 for n in calls.values())
    return ok, f"helper calls {calls}; scored {[k for k, r in runs.items() if scored(r)]}"


# ---------------------------------------------------------------- threads (5)
@check("threads_applied_before_any_model", "threads")
def _threads_before_build(EM):
    t, probes = other_threads(), {}

    def probe(fn, key):
        return counted(fn, probes, key, probe=torch.get_num_threads)
    with patched(ADP, load_class_map=probe(ADP.load_class_map, "load_class_map"),
                 build_protocol_adapter=probe(ADP.build_protocol_adapter, "adapter")), \
            patched(SA, validate_fp32_artifact=probe(SA.validate_fp32_artifact, "validate_fp32_artifact"),
                    validate_teacher_artifact=probe(SA.validate_teacher_artifact, "validate_teacher_artifact"),
                    resolve_evaluation_source=probe(SA.resolve_evaluation_source, "resolve_evaluation_source")), \
            patched(ML, load_student_checkpoint=probe(ML.load_student_checkpoint, "load_student_checkpoint"),
                    load_teacher_model=probe(ML.load_teacher_model, "load_teacher_model"),
                    load_int8_student=probe(ML.load_int8_student, "load_int8_student")):
        runs = [run_em(EM, fp32_argv(F["fp32"], new_out("threads_fp32"), "--threads", str(t))),
                run_em(EM, int8_argv(F["e4_prov"], new_out("threads_int8"), "--threads", str(t))),
                run_em(EM, teacher_argv(F["teacher"], F["teacher_sha"], new_out("threads_teacher"), "--threads",
                                        str(t)), teacher=True)]
    sites = ("load_class_map", "adapter", "validate_fp32_artifact", "validate_teacher_artifact",
             "resolve_evaluation_source", "load_student_checkpoint", "load_teacher_model", "load_int8_student")
    ok = (all(scored(r) for r in runs) and set(probes) == set(sites)
          and all(all(n == t for n in v) for v in probes.values()))
    return ok, f"T={t}; seen {sorted((k, sorted(set(v))) for k, v in probes.items())}"


@check("threads_not_applied_by_injection", "threads")
def _threads_not_applied(EM):
    t, box, out = other_threads(), {}, never("threads_not_applied")
    with patched(torch, set_num_threads=lambda n: None), \
            patched(ADP, load_class_map=counted(ADP.load_class_map, box, "load_class_map")), \
            patched(SA, validate_fp32_artifact=counted(SA.validate_fp32_artifact, box, "validate")):
        _, err, ctr = run_em(EM, fp32_argv(F["fp32"], out, "--threads", str(t)))
    ok = code_of(err) == "threads_not_applied" and not box and ctr.dataset == [] and not out.exists()
    return ok, f"{code_of(err)}; calls {sorted((k, len(v)) for k, v in box.items())}"


@check("threads_drift_refused_nothing_written", "threads")
def _threads_drift(EM):
    t, box, out = other_threads(), {}, never("threads_drift")
    real = ERT.build_eval_runtime_record

    def drifting(*a, **k):
        torch.set_num_threads(t + 1)           # the pool moved after --threads set it
        return real(*a, **k)
    with patched(ERT, build_eval_runtime_record=drifting), \
            patched(ART, write_artifact=counted(ART.write_artifact, box, "write")):
        _, err, _ = run_em(EM, fp32_argv(F["fp32"], out, "--threads", str(t)))
    ok = (code_of(err) == "threads_drifted" and isinstance(err, ERT.EvalRuntimeError) and not box
          and not out.exists())
    return ok, f"{code_of(err)}; write_artifact calls {len(box.get('write', []))}"


@check("threads_recorded_in_eval_runtime", "threads")
def _threads_recorded(EM):
    t = other_threads()
    runs = [run_em(EM, fp32_argv(F["fp32"], new_out("recorded_fp32"), "--threads", str(t))),
            run_em(EM, int8_argv(F["e4_prov"], new_out("recorded_int8"), "--threads", str(t)))]
    got = [summary(r[0])["run"]["eval_runtime"]["torch_num_threads"] if scored(r) else code_of(r[1]) for r in runs]
    return all(g == t for g in got), f"T={t}; recorded {got}"


@check("threads_absent_behaviour_unchanged", "threads")
def _threads_absent(EM):
    before, box = torch.get_num_threads(), {}
    with patched(torch, set_num_threads=counted(torch.set_num_threads, box, "set")):
        runs = [run_em(EM, fp32_argv(F["fp32"], new_out("unset_fp32"))),
                run_em(EM, int8_argv(F["e4_prov"], new_out("unset_int8"))),
                run_em(EM, teacher_argv(F["teacher"], F["teacher_sha"], new_out("unset_teacher")), teacher=True)]
    got = [summary(r[0])["run"]["eval_runtime"]["torch_num_threads"] if scored(r) else code_of(r[1]) for r in runs]
    ok = not box and torch.get_num_threads() == before and all(g == before for g in got)
    return ok, f"set_num_threads calls {len(box.get('set', []))}; inherited {before}; recorded {got}"


# ---------------------------------------------------------------- layout and repo (5)
def shape(s: dict) -> tuple:
    return (s.get("artifact_schema_version"), sorted(s), sorted(s["run"]), tuple(s["run"]["eval_runtime"]))


@check("no_new_summary_or_runtime_field", "layout")
def _no_new_field(EM):
    t = str(torch.get_num_threads())
    runs = {
        "fp32": (fp32_argv(F["fp32"], new_out("layout_fp32_plain")),
                 fp32_argv(F["fp32"], new_out("layout_fp32"), "--expected-checkpoint-sha256", sha(F["fp32"]),
                           "--threads", t), False),
        "int8": (int8_argv(F["e4_prov"], new_out("layout_int8_plain")),
                 int8_argv(F["e4_prov"], new_out("layout_int8"), "--expected-checkpoint-sha256", sha(F["e4_prov"]),
                           "--expected-converted-sha256", sha(F["e4_art"]), "--threads", t), False),
        "teacher": (teacher_argv(F["teacher"], F["teacher_sha"], new_out("layout_teacher_plain")),
                    teacher_argv(F["teacher"], F["teacher_sha"], new_out("layout_teacher"),
                                 "--expected-checkpoint-sha256", F["teacher_sha"], "--threads", t), True),
    }
    verdict = {}
    for kind, (plain, flagged, is_teacher) in runs.items():
        a, b = run_em(EM, plain, teacher=is_teacher), run_em(EM, flagged, teacher=is_teacher)
        verdict[kind] = (scored(a) and scored(b) and shape(summary(a[0])) == shape(summary(b[0]))
                         and summary(b[0])["artifact_schema_version"] == "plantseg-eval-artifact/1.2.0"
                         and tuple(summary(b[0])["run"]["eval_runtime"]) == ERT.RECORD_KEYS)
    ok = all(verdict.values()) and len(ERT.RECORD_KEYS) == 26 and ART.ARTIFACT_SCHEMA_VERSION.endswith("/1.2.0")
    return ok, f"{verdict}; RECORD_KEYS {len(ERT.RECORD_KEYS)}"


@check("repo_state_unchanged", "layout")
def _repo_unchanged(EM):
    verdict = state_verdict(STATE_BEFORE, repo_state(REPO))
    return verdict == "unchanged", verdict


@check("no_forbidden_named_path_created", "layout")
def _no_forbidden_names(EM):
    bad = [os.path.join(d, n) for d, dirs, files in os.walk(BASE) for n in dirs + files
           if TOKEN in n.lower() or ":" in n]
    return not bad and not NEVER.exists(), f"{len(bad)} forbidden names; never-created root exists: {NEVER.exists()}"


def scenarios(EM) -> list[dict]:
    """The 15 refusals, each reached with every earlier gate passing: code, class, argv, its never-created
    out-dir, whether it comes before any dataset, patches and LF4's text. The run-time ones get fresh fixture
    files, so every call can change its own."""
    rows = []

    def add(code, cls, argv, out, before_dataset=True, patches=contextlib.nullcontext, **values):
        rows.append({"code": code, "cls": cls, "argv": argv, "out": out, "before_dataset": before_dataset,
                     "patches": patches, "text": expected_text(code, **values)})

    def cli_argv(kw):
        out = never("msg_cli")
        return argv_of(cli_kw(**{**kw, "out_dir": str(out)})), out
    t, n = other_threads(), torch.get_num_threads()
    add("expected_checkpoint_sha256_required", "CliError",
        *cli_argv(preset("fp32", expected_checkpoint_sha256=None)))
    add("expected_checkpoint_sha256_format", "CliError",
        *cli_argv(offr("fp32", expected_checkpoint_sha256=H_ANY.upper())), fmt=sha256_format_error(H_ANY.upper()))
    add("expected_checkpoint_sha256_not_applicable", "CliError",
        *cli_argv(offr("fp32", checkpoint=None, random_init=True, expected_checkpoint_sha256=H_ANY)))
    add("expected_checkpoint_sha256_teacher_disagrees", "CliError",
        *cli_argv(offr("teacher", expected_checkpoint_sha256=H_OTHER)), e=H_OTHER, t=H_ANY)
    junk = new_fix("fp32_unparsable", ".pt")
    junk.write_bytes(b"no torch archive in this file\n")
    out = never("msg_mismatch")
    add("expected_checkpoint_sha256_mismatch", "CliError",
        fp32_argv(junk, out, "--expected-checkpoint-sha256", sha(F["fp32"])), out,
        path=junk, actual=sha(junk), e=sha(F["fp32"]))
    ck = copy_fix("fp32", "fp32_changing")
    h, real_fp32 = sha(ck), SA.validate_fp32_artifact

    def fp32_changing(stage, path):
        res = real_fp32(stage, path)
        Path(path).write_bytes(F["fp32_other"].read_bytes())
        return res
    out = never("msg_changed")
    add("source_changed_during_validation", "CliError", fp32_argv(ck, out, "--expected-checkpoint-sha256", h), out,
        patches=lambda: patched(SA, validate_fp32_artifact=fp32_changing), path=ck, e=h, actual=sha(F["fp32_other"]))
    add("expected_converted_sha256_required", "CliError",
        *cli_argv(preset("int8", expected_converted_sha256=None)))
    add("expected_converted_sha256_format", "CliError",
        *cli_argv(offr("int8", expected_converted_sha256=H_OTHER.upper())), fmt=sha256_format_error(H_OTHER.upper()))
    add("expected_converted_sha256_not_int8", "CliError",
        *cli_argv(offr("fp32", expected_converted_sha256=H_OTHER)), p="fp32", r=False)
    out = never("msg_conv_mismatch")
    add("expected_converted_sha256_mismatch", "CliError",
        int8_argv(F["e4_prov"], out, "--expected-checkpoint-sha256", sha(F["e4_prov"]),
                  "--expected-converted-sha256", H_OTHER), out, path=F["e4_art"], actual=sha(F["e4_art"]), e2=H_OTHER)
    art, prov = int8_pair("e4_changing")
    h2, real_adapter = sha(art), ADP.build_protocol_adapter

    def adapter_changing(*a, **k):
        art.write_bytes(F["e4_art_other"].read_bytes())
        return real_adapter(*a, **k)
    out = never("msg_conv_changed")
    add("converted_changed_before_load", "EvalRuntimeError",
        int8_argv(prov, out, "--expected-checkpoint-sha256", sha(prov), "--expected-converted-sha256", h2), out,
        before_dataset=False, patches=lambda: patched(ADP, build_protocol_adapter=adapter_changing), path=art, e2=h2,
        actual=sha(F["e4_art_other"]))
    add("threads_invalid", "CliError", *cli_argv(offr("fp32", threads="0")), t="0")
    add("threads_required", "CliError", *cli_argv(preset("fp32", threads=None)))
    out = never("msg_not_applied")
    add("threads_not_applied", "CliError", fp32_argv(F["fp32"], out, "--threads", str(t)), out,
        patches=lambda: patched(torch, set_num_threads=lambda _n: None), t=t, n=n)
    real_record = ERT.build_eval_runtime_record

    def drifting(*a, **k):
        torch.set_num_threads(t + 1)
        return real_record(*a, **k)
    out = never("msg_drift")
    add("threads_drifted", "EvalRuntimeError", fp32_argv(F["fp32"], out, "--threads", str(t)), out,
        before_dataset=False, patches=lambda: patched(ERT, build_eval_runtime_record=drifting), n=t + 1, t=t)
    return rows


@check("main_exits_1_on_each_refusal", "layout")
def _main_exits_1(EM):
    rows = scenarios(EM)
    got = {}
    for row in rows:
        err = io.StringIO()
        saved = torch.get_num_threads()
        try:
            with row["patches"](), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = EM.main(row["argv"])
        finally:
            torch.set_num_threads(saved)
        got[row["code"]] = (rc == 1 and err.getvalue() == f"{row['cls']}: {row['text']}\n"
                            and not row["out"].exists())
    ok = len(rows) == len(TEMPLATES) and set(got) == set(TEMPLATES) and all(got.values())
    return ok, f"{sum(got.values())}/{len(TEMPLATES)}; failing {[k for k, v in got.items() if not v]}"


@check("messages_verbatim", "layout")
def _messages_verbatim(EM):
    rows = scenarios(EM)
    got = {}
    for row in rows:
        saved = torch.get_num_threads()
        try:
            with row["patches"]():
                _, err, ctr = run_em(EM, row["argv"])
        finally:
            torch.set_num_threads(saved)
        got[row["code"]] = (err is not None and type(err).__name__ == row["cls"] and str(err) == row["text"]
                            and not row["out"].exists() and (ctr.dataset == [] or not row["before_dataset"]))
    ok = len(rows) == len(TEMPLATES) and set(got) == set(TEMPLATES) and all(got.values())
    return ok, f"{sum(got.values())}/{len(TEMPLATES)}; failing {[k for k, v in got.items() if not v]}"


# ---------------------------------------------------------------- mutations (each on a copy outside the repository)
REAL_SRC = (REPO / "scripts" / "evaluate_model.py").read_text(encoding="utf-8")
REPO_LINE = "REPO = Path(__file__).resolve().parents[1]\n"
P1_LINE = "    handed = hash_before_parse(args)     # IF-T1 P1: the handed file, before any validator parses it\n"
T_LINE = "    threads = apply_threads(args)        # IF-T1 (TL-1): before the class map, the source checks and any build\n"
RESOLVE_LINE = "        resolved = resolve_evaluation_source(args.stage, provenance=args.provenance)\n"
H2_LINE = '    if expected_converted is not None and resolved["artifact_sha256"] != expected_converted:\n'
DRIFT_LINE = '    if threads is not None and runtime["torch_num_threads"] != threads:        # IF-T1 (TL-1)\n'
WRITE_LINE = "    return write_artifact(result, request, provenance, eval_runtime=runtime)\n"
H2_LATE = ('        if expected_converted is not None and resolved["artifact_sha256"] != expected_converted:\n'
           '            raise CliError("[expected_converted_sha256_mismatch] (mutant)")\n')


def cli_block() -> str:
    """validate_cli_args' IF-T1 block, as the module holds it (M14 copies it before the test-split guards)."""
    start = REAL_SRC.index("    # --- IF-T1 (TL-6(b), TL-1; lane LEFF B-1b)")
    return REAL_SRC[start:REAL_SRC.index("\n\n\ndef parse_threads", start) + 1]


def mutations() -> list[tuple[str, str, list[tuple[str, str]], str]]:
    return [
        ("M1", "P1 after validate_fp32_artifact",
         [(P1_LINE, "    handed = None\n"),
          ("        ckpt_sha = sha256_file(Path(args.checkpoint))\n",
           "        ckpt_sha = sha256_file(Path(args.checkpoint))\n    handed = hash_before_parse(args)\n")],
         "fp32_mismatch_before_any_parse"),
        ("M2", "no P2", [("        if after != expected:\n", "        if False:\n")],
         "fp32_changed_during_validation_refused"),
        ("M3", "no required rule", [("    if expected is None and in_r:\n", "    if False:\n")],
         "ecs_required_under_r"),
        ("M4", "uppercase accepted", [("        sha_error = sha256_format_error(expected)\n",
                                       "        sha_error = sha256_format_error(expected.lower())\n")],
         "ecs_format_uppercase_refused"),
        ("M5", "H with --random-init", [("    if expected is not None and args.random_init:\n", "    if False:\n")],
         "ecs_refused_for_random_init"),
        ("M6", "no teacher equality",
         [('    if args.stage == "teacher" and expected is not None and expected != teacher_sha:\n',
           "    if False:\n")], "ecs_teacher_flags_disagree_refused"),
        ("M7", "INT8 H after resolve",
         [(P1_LINE, "    handed = None\n"), (RESOLVE_LINE, RESOLVE_LINE + "        handed = hash_before_parse(args)\n")],
         "int8_provenance_mismatch_before_parse"),
        ("M8", "H2 after the load",
         [(H2_LINE, "    if False:\n"),
          ("        model = load_int8_student(resolved, require_qnnpack=True)[0]\n",
           "        model = load_int8_student(resolved, require_qnnpack=True)[0]\n" + H2_LATE)],
         "int8_converted_mismatch_before_load"),
        ("M9", "no P3", [("            if actual != expected_converted:\n", "            if False:\n")],
         "int8_converted_changed_before_load_refused"),
        ("M10", "threads after the build",
         [(T_LINE, ""), ("    model_device = resolve_model_device(args.device, cpu_only=cpu_only)\n",
                         "    threads = apply_threads(args)\n"
                         "    model_device = resolve_model_device(args.device, cpu_only=cpu_only)\n")],
         "threads_applied_before_any_model"),
        ("M11", "no read-back", [("    if got != threads:\n", "    if False:\n")], "threads_not_applied_by_injection"),
        ("M12", "drift check after the write",
         [(DRIFT_LINE, "    out = write_artifact(result, request, provenance, eval_runtime=runtime)\n" + DRIFT_LINE),
          (WRITE_LINE, "    return out\n")], "threads_drift_refused_nothing_written"),
        ("M13", "a run.eval_runtime field",
         [(WRITE_LINE, '    runtime = {**runtime, "threads_flag": threads}\n' + WRITE_LINE)],
         "no_new_summary_or_runtime_field"),
        ("M14", "the IF-T1 checks before the split guards (:190)",
         [("    # --- test-split guards (contract section 7) ---\n",
           cli_block() + "\n    # --- test-split guards (contract section 7) ---\n")],
         "existing_guards_keep_precedence"),
        ("M15", "H2 accepted on FP32",
         [("    if converted is not None and (args.precision not in INT8_PRECISIONS or args.random_init):\n",
           "    if converted is not None and args.random_init:\n")], "ecv_refused_off_int8"),
        ("M16", "--threads parsed with type=int",
         [('    p.add_argument("--threads", default=None,\n', '    p.add_argument("--threads", default=None, type=int,\n')],
         "threads_non_integer_refused"),
        ("M17", "the teacher's P2 through resolved['checkpoint_sha256']",
         [("        after = ckpt_sha if resolved is None else identity_sha256(handed)\n",
           '        after = (resolved["checkpoint_sha256"] if args.model_role == "teacher"\n'
           "                 else ckpt_sha if resolved is None else identity_sha256(handed))\n")],
         "teacher_changed_during_validation_refused"),
        ("M18", "the helper called without H",
         [("    if expected is None:\n        return None\n    path = handed_file(args)\n",
           "    if expected is None:\n        identity_sha256(handed_file(args))\n        return None\n"
           "    path = handed_file(args)\n")], "flags_absent_nothing_hashed"),
        ("M19", "H2 compared before the official gate",
         [(H2_LINE, "    if False:\n"),
          (RESOLVE_LINE, RESOLVE_LINE + '        expected_converted = getattr(args, "expected_converted_sha256", None)\n'
           + H2_LATE)], "int8_official_order"),
        ("M20", "one message's text changed",
         [('f"{expected}; the file was not parsed")', 'f"{expected}; the file was never parsed")')],
         "messages_verbatim"),
    ]


def load_mutant(mid: str, pairs: list[tuple[str, str]]):
    text = REAL_SRC
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"{mid}: the edit's old text occurs {n} times, not once")
        text = text.replace(old, new)
    if text.count(REPO_LINE) != 1:
        raise RuntimeError(f"{mid}: the REPO line is not unique")
    text = text.replace(REPO_LINE, f"REPO = Path({str(REPO)!r})\n")     # the copy evaluates this checkout
    d = BASE / "mutants"
    d.mkdir(exist_ok=True)
    name = f"evaluate_model_{mid.lower()}"
    path = d / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                 # a dataclass resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


def run_check(fn, EM) -> tuple[bool, str]:
    saved = torch.get_num_threads()
    try:
        ok, detail = fn(EM)
        return bool(ok), detail
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- a check that cannot finish FAILS, with its reason
        return False, f"raised {type(e).__name__}: {str(e)[:160]}"
    finally:
        torch.set_num_threads(saved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mutations", action="store_true", help="run the checks only (development)")
    opts = ap.parse_args()
    names = [n for n, _, _ in CHECKS]
    if len(set(names)) != len(names) or any(TOKEN in n.lower() for n in names):
        raise SystemExit("check names must be unique and free of the SL-1 token")
    print("=" * 78)
    print("EVALUATOR IDENTITY AND THREAD FLAGS SMOKE (IF-T1; lane LEFF B-1b) -- synthetic; no GPU, no PlantSeg")
    print(f"temp: {BASE}")
    print("=" * 78)
    real_git = (ART.git_porcelain_bytes, ART.git_commit)
    ART.git_porcelain_bytes = lambda repo: b""
    ART.git_commit = lambda repo: "0" * 40
    results, killed, edits, end_state = [], 0, [], None
    try:
        build_fixtures()
        for name, group, fn in CHECKS:
            ok, detail = run_check(fn, EM_REAL)
            results.append((name, ok))
            print(f"  [{group:7}] {name:52}: {'PASS' if ok else 'FAIL'}  {detail}")
        if not opts.no_mutations:
            by_name = {n: fn for n, _, fn in CHECKS}
            print("\n[MUTATIONS] each on a copy outside the repository, loaded under its own module name")
            edits = mutations()
            for mid, what, pairs, target in edits:
                try:
                    module = load_mutant(mid, pairs)
                    ok, detail = run_check(by_name[target], module)
                    sys.modules.pop(module.__name__, None)
                except Exception as e:  # noqa: BLE001 -- an edit that cannot be applied is not killed
                    ok, detail = True, f"not applied: {e}"
                killed += not ok
                print(f"  {mid:4} {what:52}: {'KILLED' if not ok else 'SURVIVED'} by {target}  ({detail[:120]})")
            end_state = state_verdict(STATE_BEFORE, repo_state(REPO))
            print(f"\n  repository after the mutations: {end_state}")
    finally:
        ART.git_porcelain_bytes, ART.git_commit = real_git
    passed, n = sum(ok for _, ok in results), len(results)
    if opts.no_mutations:
        ok_all = passed == n
        print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'} ({passed}/{n})")
    else:
        ok_all = passed == n and killed == len(edits) and end_state == "unchanged"
        print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'} ({passed}/{n}; {killed}/{len(edits)} edits killed)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
