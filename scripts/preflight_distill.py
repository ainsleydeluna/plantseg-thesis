#!/usr/bin/env python3
"""KD launch gate for E2, E3, A, F and G (lane 4(b), K2; DL-87, DL-88, DL-89; AM-19 with AM-19a).

Usage (every command runs with PYTHONPATH=<checkout>; the gate refuses anything else):
  python -B scripts/preflight_distill.py entry --profile P --stage S --seed N --ckpt-dir D --code-pin PIN
         [--lambda-logit L] [--alpha A] [--am8a-report REL]
      The launch line of a launch (PL-27), built from these arguments and printed for the launch log, with its
      decision date and launch-order position (the launch note of the E1 runbook's 9.13). Runs where the clone
      is (the KD image, the clone mounted read-only; PL-32); the attempt is 1 + the earlier launch lines of the
      same stage, seed, horizon, lambda and alpha in the committed launch log.
  python -B scripts/preflight_distill.py records --records-commit H --out F
      Writes the records files of records commit H into the folder F, outside the repository (PL-28): the launch
      log, the selection files, the decision records and the corrected ones with the void records and fault
      reports they name, the band file, the push evidence and its list, the AM-7a clipping value, the decision
      log and every report the launch log names. Runs on the pod before the remote is invalidated. HEAD (the
      pin) must be an ancestor of H.
  python -B scripts/preflight_distill.py gate --profile P --stage S --seed N --ckpt-dir D --evidence E
         --expect-head PIN --records-commit H --records F --teacher-ckpt T --clock-offset-seconds O --record R
         [--lambda-logit L] [--alpha A] [--attempt K] [--am8a-report REL] [--previous-run DIR ...]
         [--decision-record REL ...] [--rehearsal]
      The gate; GO prints the launch block (PL-29) after writing the record R. A sweep's corrected decision record
      (AM-19a reading 19) is read only through --decision-record, and must be when H holds one (CHECK ITEM 5).
  python -B scripts/preflight_distill.py check-run-meta --gate-record R --ckpt-dir D
  python -B scripts/preflight_distill.py check-run-meta --allow-smoke --ckpt-dir D --stage S --seed N --profile P
         --expect-head PIN [--lambda-logit L] [--alpha A]
      The run's run_meta row against one rule per key (PL-30); prints the launched line on PASS and on FAIL.
  python -B scripts/preflight_distill.py push-evidence --response FILE --out DIR [--remote NAME]
      The push day's evidence for the lambda selection (AM-19a reading 10; CHECK ITEM 3): FILE is GitHub's activity
      response, saved verbatim by the runbook's `gh api` line; each push after T_lo is fetched from the remote
      and its answer recorded; DIR receives the response (verbatim) and the list, to be committed together.
Exit: 0 GO / PASS / written; 1 NO-GO / FAIL (a rehearsal ALWAYS exits 1); 2 usage or refused.

The E1 gate (scripts/preflight_e1_trainval.py) is loaded from the checkout by file location and never run: its
stages arguments, data_isolation, repo_state, module_provenance, class_weights, smoke_loss, cuda,
imagenet_backbone, smoke_loader_seed, smoke_dataloader, seed_sequence_R5 and repo_unchanged are reused as the
same objects, in E1's order (PL-25), with the KD stages between them:
  arguments (E1's, plus the evidence folder, the run_id, the records folder, the teacher file and the clock
  offset, |O| <= 60 s: AM-19a reading 21) -> data_isolation -> repo_state (E1's, unchanged: HEAD = --expect-head
  = the pin; PL-28) -> module_provenance -> records (H holds the pin; the records folder by blob id; AM-7a item 5
  on record; the schedule) -> selection_inputs (the lambda/alpha sources: the plan's table, E-40, PL-31) ->
  launch_order (PL-27) -> class_weights -> smoke_loss -> kd_image (the KD digest and the fingerprint of record,
  PL-26) -> teacher_identity -> cuda -> imagenet_backbone -> smoke_loader_seed -> smoke_dataloader ->
  seed_sequence_R5 -> kd_dry_run (PL-25) -> repo_unchanged.
The first FAIL stops the gate; a rehearsal (--rehearsal) never prints a launch block and SKIPs only what needs
the pod or the teacher file.
"""
import sys

sys.dont_write_bytecode = True                  # before the first non-stdlib import (as the E1 gate)

import os  # noqa: E402

BYTECODE_ENV_AT_ENTRY = os.environ.get("PYTHONDONTWRITEBYTECODE")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"      # children inherit it

import argparse          # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import platform          # noqa: E402
import re                # noqa: E402
import shlex             # noqa: E402
import shutil            # noqa: E402
import subprocess        # noqa: E402
import tempfile          # noqa: E402
import time              # noqa: E402
import urllib.parse      # noqa: E402
from pathlib import Path, PurePosixPath  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The E1 gate, loaded by file location (its own pattern for preflight_e1, :52-54); only its definitions run.
_E1G_SPEC = importlib.util.spec_from_file_location("preflight_e1_trainval",
                                                   str(REPO / "scripts" / "preflight_e1_trainval.py"))
E1G = importlib.util.module_from_spec(_E1G_SPEC)
_E1G_SPEC.loader.exec_module(E1G)

GateError = E1G.GateError                       # one error class: E1's stages raise it too
_under, sha256_file, _ok, _skip = E1G._under, E1G.sha256_file, E1G._ok, E1G._skip

# ------------------------------------------------------------------------------------- values of record
KD_IMAGE_DIGEST = "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b"   # DL-87
# PL-26: the fingerprint of record, from Block D's pip list of the KD image (DL-87): the Python version and the
# versions of these distributions; no nvidia-* distribution. The distributions that provide a third-party module
# imported by FINGERPRINT_FILES must be among these (each has its value of record here).
FINGERPRINT_OF_RECORD = {"python": "3.11.16", "torch": "2.1.0+cu121", "torchvision": "0.16.0+cu121",
                         "numpy": "1.26.4", "pillow": "12.3.0", "mmcv": "2.1.0", "mmengine": "0.10.7",
                         "mmsegmentation": "1.2.2", "ftfy": "6.3.0", "nvidia": []}
FINGERPRINT_DISTS = ("torch", "torchvision", "numpy", "pillow", "mmcv", "mmengine", "mmsegmentation", "ftfy")
FINGERPRINT_FILES = ("src/data/transforms.py", "src/data/dataset.py", "src/models/student.py",
                     "src/training/losses.py")
# DL-88: the teacher of record and its values (read from the row, never a summary)
TEACHER_FILE = "iter_24000.pth"
TEACHER_SHA256 = "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e"
TEACHER_BYTES = 335_949_080
TEACHER_BUILDER = "segnext_mscan_b_builder"
TEACHER_VALUES = {
    "config_sha256": "1b94aa32a7d22be647d8e60a42daa1dd8dd756158b586feec6465385e299ae4a",
    "teacher_components_sha256": "1b46680c6d03982c84387cc3c3ea78eb14de7ec643ac3b55b1937ef7cbf63025",
    "architecture_signature": "aa3a73b9a3a34c60c05f0b89dc9ddd08b6280779f5a946992c5614f716429e76",
    "model_cfg_sha256": "d0bfa1215ef575e9f7e16354c8b06a684acf47c4e136af30cfc5c1daa716507b",
    "ham_kwargs": {"MD_S": 1, "MD_R": 16, "train_steps": 6, "eval_steps": 7, "inv_t": 100, "rand_init": True},
    "reused_module_hashes": {
        "teacher_components": "1b46680c6d03982c84387cc3c3ea78eb14de7ec643ac3b55b1937ef7cbf63025",
        "src/data/transforms.py": "df920496cf8710442ae6311b7538c8d3f5d67152a06d0ec1f1c05afc3307193b",
        "configs/augment.py": "8c450f30aedb20b213916d90c054c155906bf4ae49302b5c8d0aedfaf0b1dded",
        "src/eval/metrics.py": "9898d6dc0ec68f15c8c90cebb5889914c0a2de2e3bd409c3fba9ba14e1be86d9",
        "src/distill/nmf_stream.py": "053e53cc441d58a2477affbd281331d7321dd2023e9f09c9f75010b9b36a64d4",
        "src/data/isolation.py": "74a063313cc04c1e327dce22b0faa4ae3a9334a01e6cf7c58eb1bee13c2e5014"}}
TEACHER_PARAMS = 27_618_868                     # P5 stage 9
TEACHER_STATE_ENTRIES = 854                     # E-31; DL-88 (strict load 854 of 854)
CLASS_WEIGHTS_SHA256 = E1G.CLASS_WEIGHTS_SHA256  # DL-88 (5325e32 blob)
NUM_WORKERS = E1G.E1_NUM_WORKERS                # 12 (DL-88)
LOG_EVERY = E1G.LOG_EVERY                       # 50
STUDENT_PARAMS = 2_933_688                      # AM-17 item 10(b)
PROJECTION_PARAMS = 51_200                      # the cwd_feat projection, 160 x 320
RAMP_ITERS = 335                                # one epoch at batch 16 (the 5,367-image TRAIN set)
EXPECTED_GPU_NAME = E1G.EXPECTED_GPU_NAME       # a warning only (P5)
# MG-1: AM-7a, with item 5's clipping value, is on record for every KD launch
PREREG_REL = "docs/PREREGISTRATION_AMENDMENTS.md"
AM7A_HEADING = "## AM-7a — "
AM7A_ITEM5_TEXT = "max_norm = 100"
AM7_CLIP_REL = "reports/derived/am7_clip_value.json"
AM7_CLIP_SHA256 = "38267e82f562c6ed640b76e5bf7c770403923fde35e32947477661af7bb4e7ed"
AM7A_DL_ROW = "DL-56"
PROFILES = {"distill_80k": {"horizon": 80000, "extra_args": ()},
            "distill_160k": {"horizon": 160000, "extra_args": ("--iterations", "160000")}}
STAGE_NAMES = {"e2": "E2", "e3": "E3", "a": "A", "f": "F", "g": "G"}
TERMS = {"e2": (True, False, False), "e3": (True, True, True), "a": (False, True, True),
         "f": (False, True, False), "g": (False, False, True)}     # (logit_kd, cwd_feat, cwd_logit)
SEEDS = {"e2": (42, 43, 44), "e3": (42, 43, 44), "a": (42,), "f": (42,), "g": (42,)}
# The launch block never passes these (P5 and PL-29); --dry-run is the dry stage's only.
NEVER_PASSED = ("--lambda", "--grad-clip-norm", "--resume", "--max-iters", "--batch-size", "--val-interval",
                "--max-val-batches", "--device", "--teacher-config", "--lambda-semantics",
                "--allow-semantics-mismatch", "--allow-offgrid", "--dry-run")
NEXT_LINE = ("printf 'Next: check-run-meta within 5 minutes; commit and push its launched line; then press Sync "
             "now on both GitHub sources\\n'")
CLOCK_OFFSET_MAX = 60.0                         # seconds (AM-19a reading 21; E1 runbook 9.13, RB:458)
DEFAULT_LAUNCH_WARN = 6 * 3600                  # the 6-hour note after the default's launch (P5)
RECORDS_MANIFEST = "records_manifest.json"
RECORDS_FORMAT = "kd_records/1"
NOT_SERVED_PATTERNS = ("not our ref", "unadvertised object", "no such remote ref")
STAGE_NAMES_ORDER = ("arguments", "data_isolation", "repo_state", "module_provenance", "records", "selection_inputs",
                     "launch_order", "class_weights", "smoke_loss", "kd_image", "teacher_identity", "cuda",
                     "imagenet_backbone", "smoke_loader_seed", "smoke_dataloader", "seed_sequence_R5", "kd_dry_run",
                     "repo_unchanged")


# ---------------------------------------------------------------------------------------------- helpers
def _lib():
    """sweep_select and train_distill from this checkout (import_origin_foreign otherwise)."""
    import src.training.sweep_select as ss
    import src.training.train_distill as td
    for m in (ss, td):
        if not _under(Path(m.__file__), REPO):
            raise GateError("import_origin_foreign", f"{m.__name__} loaded from {m.__file__}, outside {REPO}")
    return ss, td


def _refused(e) -> GateError:
    """A selection library refusal as a gate failure with the same code."""
    return GateError(e.code, str(e))


def fmt_value(v) -> str:
    return f"{float(v):g}"


def launch_identity(stage: str, seed: int, profile: str, lam, alpha) -> dict:
    """The sweep a launch belongs to (E2 / E3 at seed 42 on the 80,000-iteration horizon: the lambda / alpha
    sweep, AM-2, AM-16 item 2), its swept value, and its horizon."""
    horizon = PROFILES[profile]["horizon"]
    sweep = None
    if seed == 42 and horizon == 80000:
        sweep = {"e2": "lambda_logit", "e3": "alpha_cwd"}.get(stage)
    value = None if sweep is None else (lam if sweep == "lambda_logit" else alpha)
    return {"sweep": sweep, "value": None if value is None else float(value), "horizon": horizon}


def run_id_of(stage: str, seed: int, horizon: int, lam, alpha, attempt: int) -> str:
    """<stage>_s<seed>[_lambda<v>][_alpha<a>]_<80|160>k_a<n>: exactly one alpha token where the stage has the
    feature-map term (the trainer's ckpt_dir_alpha_error), none elsewhere."""
    parts = [stage, f"s{seed}"]
    if lam is not None:
        parts.append(f"lambda{fmt_value(lam)}")
    if alpha is not None:
        parts.append(f"alpha{fmt_value(alpha)}")
    return "_".join(parts + [f"{horizon // 1000}k", f"a{attempt}"])


def launch_line(ss, *, stage, seed, profile, lam, alpha, attempt, schedule, code_pin, ckpt_dir, am8a_report) -> dict:
    """The launch line (PL-13, PL-27) in LAUNCH_KEYS order."""
    ident = launch_identity(stage, seed, profile, lam, alpha)
    line = {"format": ss.LAUNCH_LOG_FORMAT, "event": "launch",
            "run_id": run_id_of(stage, seed, ident["horizon"], lam, alpha, attempt), "stage": STAGE_NAMES[stage],
            "seed": seed, "horizon": ident["horizon"], "sweep": ident["sweep"], "value": ident["value"],
            "lambda_logit": None if lam is None else float(lam), "alpha_cwd": None if alpha is None else float(alpha),
            "attempt": attempt, "schedule": schedule, "code_pin": code_pin, "ckpt_dir": str(ckpt_dir),
            "am8a_report": am8a_report}
    assert tuple(line) == ss.LAUNCH_KEYS
    return line


def line_bytes(line: dict) -> bytes:
    """A launch-log line as the log holds it (one JSON object, no newline): entry prints it, the gate compares it."""
    return json.dumps(line).encode("utf-8")


def group_of(line: dict) -> tuple:
    """The attempt group (LaunchLog): stage, seed, horizon, lambda_logit and alpha_cwd."""
    num = (lambda x: None if x is None else float(x))  # noqa: E731
    return (line["stage"], line["seed"], line["horizon"], num(line["lambda_logit"]), num(line["alpha_cwd"]))


def stage_args_error(stage: str, seed: int, profile: str, lam, alpha) -> str | None:
    """The launch's own arguments: a registered seed of the stage; 160,000 iterations for E2 and E3 at seed 42
    only (AM-16 item 3); lambda exactly for a stage with Logit KD and alpha exactly for one with the feature-map
    term (explicit, so the launch line and the run directory carry them)."""
    lk, cf, _ = TERMS[stage]
    if seed not in SEEDS[stage]:
        return f"[seed] --seed {seed} is not a registered seed of {STAGE_NAMES[stage]}: {SEEDS[stage]}"
    if PROFILES[profile]["horizon"] != 80000 and (stage not in ("e2", "e3") or seed != 42):
        return f"[iterations] --profile {profile} is the AM-16 item 3 control: E2 and E3 at seed 42 only"
    if lk != (lam is not None):
        return (f"[lambda_argument] --lambda-logit {'is required' if lk else 'is refused'} for "
                f"{STAGE_NAMES[stage]}")
    if cf != (alpha is not None):
        return f"[alpha_argument] --alpha {'is required' if cf else 'is refused'} for {STAGE_NAMES[stage]}"
    return None


def sweep_dates(ss, src, rules: dict, schedule: dict, log, ident: dict, alpha_record=None) -> dict:
    """A sweep launch's decision date, its end C (cutoff_utc) and its launch-order position (AM-19 items 1(e) and
    2(a)); Nones for a launch outside the sweeps. The alpha date is derived as the selection derives it
    (alpha_cutoff: the lambda selection's push day, AM-19a reading 10), from the same committed records;
    `alpha_record` is the alpha sweep's record in force (alpha_inputs' alpha_cut_conflict check)."""
    if ident["sweep"] is None:
        return {"decision_date": None, "cutoff_utc": None, "position": None, "basis": None}
    key = ident["sweep"]
    sweep = rules[key]
    order = [float(v) for v in ss.LAUNCH_ORDER[key]]
    if ident["value"] not in order:
        raise GateError("grid_mismatch", f"{key} {ident['value']:g} is not on the registered grid {order}")
    position = order.index(ident["value"]) + 1
    calendar = sweep["decision_dates"][schedule["in_force"]]
    if key == "lambda_logit":
        return {"decision_date": calendar, "cutoff_utc": ss.cutoff_instant(calendar), "position": position,
                "basis": None}
    atts = ss.sweep_attempts(log, sweep, key, schedule)
    inputs = ss.alpha_inputs(src, sweep, atts, decision_record=alpha_record)
    cut = ss.alpha_cutoff(src, calendar, lambda_doc=inputs["lambda_doc"],
                          default_attempts=atts[float(sweep["default_candidate"])])
    return {"decision_date": cut["decision_date"], "cutoff_utc": cut["cutoff_utc"], "position": position,
            "basis": cut["basis"]}


def read_record_doc(ss, src, rel: str, dl_text: str) -> dict:
    """A decision record of the records commit (AM-19 item 2(h); AM-19a reading 19; CHECK ITEM 5): a record of a
    sweep at its place (record_place_error: the sweep's record at its path corrects nothing; its corrected record
    at the corrected path corrects it), its sha256 in that commit's decision log, written once
    (decision_record_rewritten otherwise); for a corrected record, the void record kept unedited and the references
    (correction_refs: the void record and the AM-8a fault report committed with the sha256 values it names, both in
    the decision log). Whether the void record is void is the selection's to derive: that needs the run
    directories."""
    rel, data = src.read(rel, missing="decision_record_missing", uncommitted="decision_record_uncommitted",
                         record=True)
    sha = ss.sha256_bytes(data)
    doc = ss.parse_json_bytes(data, rel, code="decision_record_format")
    cor = doc.get("corrects") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != ss.RECORD_FORMAT \
            or doc.get("sweep") not in tuple(ss.DECISION_RECORD_REL) or not ss.corrects_ok(cor):
        raise GateError("decision_record_format", f"{rel} is not a {ss.RECORD_FORMAT} record of a sweep with a "
                                                  "well-formed corrects")
    err = ss.record_place_error(doc["sweep"], rel, cor)
    if err is not None:
        raise GateError("decision_record_correction_invalid", err)
    if sha not in dl_text:
        raise GateError("decision_record_not_in_decision_log", f"{rel}: sha256 {sha} is not in the decision log at "
                                                               "the records commit (AM-19 item 2(h))")
    err = ss.written_once_error(src, rel)
    if err is not None:
        raise GateError("decision_record_rewritten", err)
    if cor is not None:
        err = ss.written_once_error(src, cor["void_record"]["path"])
        if err is not None:
            raise GateError("decision_record_correction_invalid", err)
        ss.correction_refs(src, dl_text, cor, rel)
    return {"path": rel, "sha256": sha, "doc": doc}


def sweep_records(ss, src, dl_text: str, named: list) -> dict:
    """{sweep: its record in force} at the records commit (AM-19a reading 19; CHECK ITEM 5): a --decision-record is
    its sweep's record in force; a sweep whose corrected record the records commit holds must name it so; otherwise
    the sweep's record at its path, when the records commit holds one."""
    out = {}
    places = (*ss.DECISION_RECORD_REL.values(), *ss.DECISION_RECORD_CORRECTED_REL.values())
    for rel in named:
        if rel not in places:
            raise GateError("decision_record_correction_invalid", f"--decision-record {rel}: a sweep's decision record "
                                                                  f"is read at one of {list(places)} (AM-19a reading 19)")
        rec = read_record_doc(ss, src, rel, dl_text)
        sweep = rec["doc"]["sweep"]
        if sweep in out:
            raise GateError("decision_record_format", f"two --decision-record files for the {sweep} sweep")
        out[sweep] = rec
    for sweep, rel in ss.DECISION_RECORD_REL.items():
        fixed = ss.DECISION_RECORD_CORRECTED_REL[sweep]
        for place in (rel, fixed):          # a record deleted after its commit: a STOP, as at the selection
            err = ss.deleted_record_error(src, place)
            if err is not None:
                raise GateError("decision_record_rewritten", err)
        if src.has(fixed) and (sweep not in out or out[sweep]["path"] != fixed):
            raise GateError("decision_record_correction_invalid", f"the records commit holds {fixed}: the {sweep} "
                                                                  "sweep's record in force is its corrected record; "
                                                                  f"pass --decision-record {fixed} (AM-19a reading "
                                                                  "19; CHECK ITEM 5)")
        if sweep not in out and src.has(rel):
            out[sweep] = read_record_doc(ss, src, rel, dl_text)
    return out


def record_cuts(rec: dict | None, value: float) -> bool:
    """A record's entry for `value` reads cut (AM-19 item 2(h))."""
    if rec is None:
        return False
    return any(isinstance(v, dict) and float(v.get("value", math.nan)) == float(value) and v.get("status") == "cut"
               for v in rec["doc"].get("values") or [])


def selection_doc(ss, src, rel: str, fmt: str, field: str) -> dict:
    """A committed selection file at the records commit and its winner."""
    rel, data = src.read(rel, missing="selection_missing", uncommitted="selection_uncommitted", record=True)
    doc = ss.parse_json_bytes(data, rel, code="selection_format")
    winner = doc.get("winner") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != fmt or not isinstance(winner, dict) \
            or not ss._finite_number(winner.get(field)):
        raise GateError("selection_format", f"{rel} is not a {fmt} file with a winner")
    return {"path": rel, "sha256": ss.sha256_bytes(data), "winner": float(winner[field]), "doc": doc}


def alpha_cut_record_ok(ss, src, rec: dict, rules: dict, schedule: dict, log, now: float) -> dict:
    """PL-31: a launch that takes alpha = 50 from a record: the record of the records commit (its sha256 in its
    decision log: read_record_doc), of the alpha sweep, alpha_sweep_cut true, its schedule and cutoff equal to the
    derived ones, now at or after the cutoff, and no non-default alpha launched line before the cutoff
    (alpha_selection.json beside it is refused up front by selection_inputs: alpha_cut_conflict)."""
    doc = rec["doc"]
    sweep = rules["alpha_cwd"]
    if doc.get("sweep") != "alpha_cwd" or doc.get("alpha_sweep_cut") is not True:
        raise GateError("alpha_cut_record_invalid", f"{rec['path']} does not cut the alpha sweep (alpha_sweep_cut)")
    want_sched = {"in_force": schedule["in_force"], "decision_log_entry": schedule["decision_log_entry"],
                  "file_sha256": schedule["file_sha256"]}
    dates = sweep_dates(ss, src, rules, schedule, log, {"sweep": "alpha_cwd", "value": 50.0}, rec["path"])
    if not ss._same(doc.get("schedule"), want_sched) or doc.get("decision_date") != dates["decision_date"] \
            or not ss._same(doc.get("cutoff_utc"), dates["cutoff_utc"]):
        raise GateError("alpha_cut_record_invalid", f"{rec['path']}: schedule {doc.get('schedule')}, decision date "
                                                    f"{doc.get('decision_date')} and cutoff {doc.get('cutoff_utc')} "
                                                    f"differ from the derived {want_sched}, {dates['decision_date']}, "
                                                    f"{dates['cutoff_utc']} (PL-31)")
    if now < dates["cutoff_utc"]:
        raise GateError("alpha_cut_record_invalid", f"{rec['path']}: the alpha decision date {dates['decision_date']} "
                                                    "has not ended (PL-31)")
    atts = ss.sweep_attempts(log, sweep, "alpha_cwd", schedule)
    early = [a.run_id for v, lst in atts.items() if v != float(sweep["default_candidate"]) for a in lst
             if a.launched is not None and a.launch_ts < dates["cutoff_utc"]]
    if early:
        raise GateError("alpha_cut_record_invalid", f"{rec['path']} cuts the alpha sweep, and the non-default "
                                                    f"alpha attempts {early} launched before its cutoff (PL-31)")
    return dates


# ------------------------------------------------------------------------------------------------- stages
def stage_arguments(ctx):
    """E1's arguments stage (the data root, PYTHONPATH = the checkout, a fresh --ckpt-dir outside the repository
    and the data root, the record path), then the KD arguments: the launch's own (stage_args_error), the run_id
    (basename of --ckpt-dir), the evidence folder (absolute, an existing directory outside the repository, the
    ckpt dir and the data root, without <run_id>.stdout.log or .pid: E-17), the records folder (absolute, outside
    the repository), the teacher file (absolute, iter_24000.pth: E-31) and the clock offset (|O| <= 60 s: AM-19a
    reading 21, the E1 runbook's 9.13)."""
    status, detail, rec = E1G.stage_arguments(ctx)
    error = stage_args_error(ctx["stage"], ctx["seed"], ctx["profile"], ctx["lambda_logit"], ctx["alpha"])
    if error is not None:
        code = error[1:error.index("]")]
        raise GateError(code, error[error.index("]") + 2:])
    if not (isinstance(ctx["attempt"], int) and ctx["attempt"] >= 1):
        raise GateError("attempt", f"--attempt {ctx['attempt']!r}: a whole number >= 1")
    ident = launch_identity(ctx["stage"], ctx["seed"], ctx["profile"], ctx["lambda_logit"], ctx["alpha"])
    rid = run_id_of(ctx["stage"], ctx["seed"], ident["horizon"], ctx["lambda_logit"], ctx["alpha"], ctx["attempt"])
    ck, root = Path(ctx["ckpt_dir"]), ctx["root"]
    if ck.name != rid:
        raise GateError("run_id_mismatch", f"basename(--ckpt-dir) {ck.name!r} != the run_id {rid!r} of this launch "
                                           "(PL-27)")
    ev = ctx["evidence"]
    if not os.path.isabs(ev):
        raise GateError("evidence_relative", f"--evidence must be absolute: {ev!r}")
    ev = Path(ev)
    if not ev.is_dir():
        raise GateError("evidence_missing", f"--evidence {ev} is not an existing directory")
    if _under(ev, REPO) or _under(ev, ck) or _under(ev, root) or _under(ck, ev):
        raise GateError("evidence_misplaced", f"--evidence {ev} overlaps the repository, the ckpt dir or the data "
                                              "root (E-17: the stdout and pid files live outside the ckpt dir)")
    present = [n for n in (f"{rid}.stdout.log", f"{rid}.pid") if (ev / n).exists()]
    if present:
        raise GateError("evidence_exists", f"{present} already exist in {ev}: a launch writes them once")
    folder = ctx["records"]
    if not os.path.isabs(folder) or _under(Path(folder), REPO) or not Path(folder).is_dir():
        raise GateError("records_folder", f"--records {folder!r} must be an existing absolute folder outside the "
                                          "repository (PL-28)")
    t = ctx["teacher_ckpt"]
    if not os.path.isabs(t) or _under(Path(t), REPO) or Path(t).name != TEACHER_FILE:
        raise GateError("teacher_ckpt_path", f"--teacher-ckpt {t!r} must be an absolute path outside the "
                                             f"repository named {TEACHER_FILE} (E-31)")
    off = ctx["clock_offset"]
    if not (isinstance(off, float) and math.isfinite(off)) or abs(off) > CLOCK_OFFSET_MAX:
        raise GateError("clock_offset", f"--clock-offset-seconds {off!r}: the pod's offset from a network time "
                                        f"source must be within {CLOCK_OFFSET_MAX:g} s (AM-19a reading 21; E1 runbook "
                                        "9.13: more is a STOP)")
    ctx.update(run_id=rid, ident=ident)
    return _ok(f"{detail}; run_id {rid}; evidence {ev}; records {folder}; clock offset {off:g} s",
               **rec, run_id=rid, clock_offset_seconds=off)


def stage_records(ctx):
    """PL-28 and MG-1: records commit H holds the pin (HEAD, which repo_state checked against --expect-head);
    every file of the records folder's manifest has H's blob id; AM-7a is on record with item 5's value
    (its heading and "max_norm = 100" with the clipping file's sha256 at the pin, the file at H, its DL-56 row in
    H's decision log); the schedule is valid (load_schedule); the rules and the launch log load. PL-12, as at the
    selection: the clone is not shallow (repository_shallow; a partial clone is not shallow), and the schedule file
    at every launched line's git_head equals the pin's (schedule_changed_after_first_launch,
    launch_log_head_unavailable)."""
    ss, _ = _lib()
    h = ctx["records_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", h or ""):
        raise GateError("records_commit_malformed", f"--records-commit must be 40 lowercase hex: {h!r}")
    src = ss.RecordsSource(ctx["records"], h, root=ctx["git_root"])
    try:
        src.require_not_shallow()                   # PL-12: before any history is read (a partial clone is not shallow)
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    if not src.commit_exists(h):
        raise GateError("records_commit_unavailable", f"records commit {h} is not in this clone (fetch it before "
                                                      "the remote is invalidated)")
    pin = src.head()
    if not src.is_ancestor(pin, h):
        raise GateError("pin_not_ancestor", f"the pin {pin} is not an ancestor of the records commit {h} (PL-28)")
    folder = Path(ctx["records"])
    try:
        man = json.loads((folder / RECORDS_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GateError("records_mismatch", f"{folder / RECORDS_MANIFEST} unreadable ({type(e).__name__}: {e}); "
                                            "write the folder with the records command (PL-28)") from e
    if not (isinstance(man, dict) and man.get("format") == RECORDS_FORMAT and man.get("records_commit") == h
            and isinstance(man.get("files"), dict)):
        raise GateError("records_mismatch", f"{folder / RECORDS_MANIFEST} is not the {RECORDS_FORMAT} manifest of {h}")
    blobs = {}
    try:
        for rel in sorted(man["files"]):
            _, data = src.read(rel, missing="records_mismatch", uncommitted="records_mismatch", record=True)
            blobs[rel] = ss.git_blob_id(data, src.object_format())
        prereg = src.show("HEAD", PREREG_REL)
        text = "" if prereg is None else prereg.decode("utf-8", "replace")
        start = text.find("\n" + AM7A_HEADING)
        section = "" if start < 0 else text[start + 1:text.find("\n## ", start + 1)]
        flat = " ".join(section.split())
        if not section or AM7A_ITEM5_TEXT not in flat or AM7_CLIP_SHA256 not in flat:
            raise GateError("am7a_not_on_record", f"{PREREG_REL} at the pin has no AM-7a section with item 5's "
                                                  f"'{AM7A_ITEM5_TEXT}' and {AM7_CLIP_SHA256[:12]}… (MG-1)")
        _, clip = src.read(AM7_CLIP_REL, missing="am7a_not_on_record", uncommitted="am7a_not_on_record", record=True)
        if ss.sha256_bytes(clip) != AM7_CLIP_SHA256:
            raise GateError("am7a_not_on_record", f"{AM7_CLIP_REL} at {h[:12]}: sha256 {ss.sha256_bytes(clip)} != "
                                                  f"{AM7_CLIP_SHA256} (AM-7a item 5)")
        dl_text = src.read_at_head(ss.DECISION_LOG_REL, missing="decision_log_missing").decode("utf-8", "replace")
        row = next((ln for ln in dl_text.splitlines() if re.match(rf"^\|\s*{AM7A_DL_ROW}\s*\|", ln)), "")
        if "AM-7a" not in row or AM7A_ITEM5_TEXT not in row:
            raise GateError("am7a_not_on_record", f"the decision log at {h[:12]} has no {AM7A_DL_ROW} row recording "
                                                  f"AM-7a with item 5's '{AM7A_ITEM5_TEXT}' (MG-1)")
        schedule = ss.load_schedule(src)
        _, rules = ss.load_committed_rules(src)
        log = ss.load_launch_log(src)
        ss.check_schedule_at_launches(src, schedule, log)  # PL-12: the schedule at every launched line's head
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    ctx.update(src=src, pin=pin, dl_text=dl_text, schedule=schedule, rules=rules, log=log, records_blobs=blobs)
    return _ok(f"records commit {h[:12]} holds the pin {pin[:12]}; {len(blobs)} records files by blob id; AM-7a item 5 "
               f"on record; Schedule {schedule['in_force']} ({schedule['decision_log_entry']}); launch log "
               f"{log.n_lines} lines", records_commit=h, pin=pin, records_blobs=blobs,
               schedule=schedule["in_force"])


def stage_selection_inputs(ctx):
    """The lambda/alpha sources (the plan's table; E-40; PL-31). Every selection file and decision record comes
    from the records commit; every record's sha256 is in its decision log. ctx['selection_files'] names the
    records-folder copies the launch block hands the trainer (--lambda-selection, --alpha-selection)."""
    ss, _ = _lib()
    src, rules, schedule, log = ctx["src"], ctx["rules"], ctx["schedule"], ctx["log"]
    stage, ident, lam, alpha = ctx["stage"], ctx["ident"], ctx["lambda_logit"], ctx["alpha"]
    files = {"lambda_logit": None, "alpha_cwd": None}
    try:
        recs = sweep_records(ss, src, ctx["dl_text"], ctx["decision_records"])
        lam_sel = (selection_doc(ss, src, ss.LAMBDA_SELECTION_REL, "lambda_selection/1", "lambda")
                   if src.has(ss.LAMBDA_SELECTION_REL) else None)
        alp_sel = (selection_doc(ss, src, ss.ALPHA_SELECTION_REL, "alpha_selection/1", "alpha")
                   if src.has(ss.ALPHA_SELECTION_REL) else None)
        cut_rec = recs.get("alpha_cwd")
        if alp_sel is not None and cut_rec is not None and cut_rec["doc"].get("alpha_sweep_cut") is True:
            raise GateError("alpha_cut_conflict", f"{ss.ALPHA_SELECTION_REL} and the alpha-cut record {cut_rec['path']} "
                                                  "are both committed at the records commit: one of them is wrong "
                                                  "(PL-20(c), PL-31)")
        note = []
        if lam is not None:
            grid = [float(v) for v in rules["lambda_logit"]["grid"]]
            if float(lam) not in grid:
                raise GateError("grid_mismatch", f"--lambda-logit {lam:g} is not on the AM-2 grid {grid}")
            if ident["sweep"] == "lambda_logit":
                if lam_sel is not None and float(lam) != lam_sel["winner"]:
                    raise GateError("lambda_selection_exists", f"{lam_sel['path']} selects lambda "
                                                               f"{lam_sel['winner']:g}: a lambda-sweep launch now "
                                                               f"takes only the winner, not {lam:g}")
                if record_cuts(recs.get("lambda_logit"), lam):
                    raise GateError("lambda_cut_by_record", f"{recs['lambda_logit']['path']} cuts lambda {lam:g} "
                                                            "(AM-19 item 2; PL-27)")
                note.append(f"lambda {lam:g} explicit (sweep)")
            else:
                if lam_sel is None:
                    raise GateError("lambda_selection_missing", f"{ss.LAMBDA_SELECTION_REL} is not committed at the "
                                                                "records commit: this launch takes lambda from it")
                if float(lam) != lam_sel["winner"]:
                    raise GateError("lambda_not_selected", f"--lambda-logit {lam:g} != the selected lambda "
                                                           f"{lam_sel['winner']:g} ({lam_sel['path']})")
                files["lambda_logit"] = lam_sel
                note.append(f"lambda {lam:g} = {lam_sel['path']}'s winner")
        if alpha is not None:
            grid = [float(v) for v in rules["alpha_cwd"]["grid"]]
            if float(alpha) not in grid:
                raise GateError("grid_mismatch", f"--alpha {alpha:g} is not on the AM-16 item 2 grid {grid}")
            arec = recs.get("alpha_cwd")
            if ident["sweep"] == "alpha_cwd":
                if lam_sel is None:
                    raise GateError("lambda_selection_missing", f"{ss.LAMBDA_SELECTION_REL} is not committed: the "
                                                                "alpha sweep runs at the selected lambda")
                if float(lam) != lam_sel["winner"]:
                    raise GateError("lambda_not_selected", f"--lambda-logit {lam:g} != the selected lambda "
                                                           f"{lam_sel['winner']:g}")
                files["lambda_logit"] = lam_sel
                atts = ss.sweep_attempts(log, rules["alpha_cwd"], "alpha_cwd", schedule)
                ss.alpha_inputs(src, rules["alpha_cwd"], atts,          # the band entry; alpha_cut_conflict
                                decision_record=None if arec is None else arec["path"])
                if alp_sel is not None and float(alpha) != alp_sel["winner"]:
                    raise GateError("alpha_selection_exists", f"{alp_sel['path']} selects alpha {alp_sel['winner']:g}: "
                                                              f"an alpha-sweep launch takes only the winner, not "
                                                              f"{alpha:g}")
                # E-40: refused when a committed record cuts this value. A record that cuts the sweep never cuts
                # its default (its entry is the default's status), so E3 s42 alpha = 50 and its AM-8a repeats
                # still launch (AM-19 item 2(g); AM-19a readings 16 and 19)
                if record_cuts(arec, alpha):
                    raise GateError("alpha_cut_by_record", f"{arec['path']} cuts alpha {alpha:g} (E-40)")
                note.append(f"alpha {alpha:g} explicit (sweep)")
            else:
                if stage == "e3":
                    if lam_sel is None:
                        raise GateError("lambda_selection_missing", f"{ss.LAMBDA_SELECTION_REL} is not committed: "
                                                                    "this launch takes lambda from it")
                    if float(lam) != lam_sel["winner"]:
                        raise GateError("lambda_not_selected", f"--lambda-logit {lam:g} != the selected lambda "
                                                               f"{lam_sel['winner']:g}")
                    files["lambda_logit"] = lam_sel
                if alp_sel is not None:
                    if float(alpha) != alp_sel["winner"]:
                        raise GateError("alpha_not_selected", f"--alpha {alpha:g} != the selected alpha "
                                                              f"{alp_sel['winner']:g} ({alp_sel['path']})")
                    files["alpha_cwd"] = alp_sel
                    note.append(f"alpha {alpha:g} = {alp_sel['path']}'s winner")
                elif arec is not None and arec["doc"].get("alpha_sweep_cut") is True:
                    alpha_cut_record_ok(ss, src, arec, rules, schedule, log, ctx["now"])
                    if float(alpha) != float(rules["alpha_cwd"]["default_candidate"]):
                        raise GateError("alpha_not_selected", f"--alpha {alpha:g}: the alpha-cut record "
                                                              f"{arec['path']} gives alpha "
                                                              f"{rules['alpha_cwd']['default_candidate']:g} (AM-19 "
                                                              "item 2(g); PL-31)")
                    files["alpha_cwd"] = {"path": arec["path"], "sha256": arec["sha256"], "winner": float(alpha)}
                    note.append(f"alpha {alpha:g} from the alpha-cut record {arec['path']}")
                else:
                    raise GateError("alpha_source_missing", f"no {ss.ALPHA_SELECTION_REL} and no alpha-cut record at "
                                                            "the records commit: this launch takes alpha from one")
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    folder = Path(ctx["records"]).resolve()             # resolved, as the trainer records it (run_meta selection_files)
    ctx["selection_files"] = {k: None if v is None else {"rel": v["path"], "path": str(folder / v["path"]),
                                                         "sha256": v["sha256"]} for k, v in files.items()}
    ctx["sweep_records"] = recs
    return _ok("; ".join(note) or "no lambda or alpha source (G)", selection_files=ctx["selection_files"])


def previous_runs_ok(ss, ctx, line: dict, cutoff: float) -> str:
    """PL-27 / PL-9(e): a non-default attempt after C needs --previous-run directories, one for every stopped
    attempt of the value (each bound to its launched line by its run_meta sha256), and passes only when the
    last launched attempt before it is s, the value's first stop that was on course (_on_course_stops, the
    selection's own function): not_launched starts between them are skipped, as the selection's r is the attempt
    launched next after s (PL-9(e)), and s has no later launched attempt. Returns s's run_id
    (repeat_not_on_course otherwise)."""
    log, rules, schedule = ctx["log"], ctx["rules"], ctx["schedule"]
    key = ctx["ident"]["sweep"]
    sweep = rules[key]
    atts = ss.sweep_attempts(log, sweep, key, schedule)[float(line["value"])]
    mine = next((i for i, a in enumerate(atts) if a.run_id == line["run_id"]), None)
    before = atts[:mine]
    by_name = {}
    for d in ctx["previous_runs"]:
        d = Path(d)
        a = next((x for x in before if x.run_id == d.name), None)
        if a is None or a.launched is None:
            raise GateError("repeat_not_on_course", f"--previous-run {d}: not a launched earlier attempt of this value")
        meta_p = d / f"{STAGE_NAMES[ctx['stage']].lower()}_run_meta.jsonl"
        if not meta_p.is_file() or ss.sha256_file(meta_p) != a.launched["run_meta_sha256"]:
            raise GateError("repeat_not_on_course", f"--previous-run {d}: its run_meta file does not hash to the "
                                                    "launched line's run_meta_sha256")
        a.run = ss.read_run(d, sweep, key)
        by_name[a.run_id] = a
    missing = [a.run_id for a in before if a.stopped is not None and a.run_id not in by_name]
    if missing:
        raise GateError("repeat_not_on_course", f"the stopped attempts {missing} need their --previous-run directories "
                                                "(the on-course exception is the value's first on-course stop)")
    _, s = ss._on_course_stops(before, cutoff, sweep["iterations"])
    launched_before = [a for a in before if a.launched is not None]
    if s is None or not launched_before or s is not launched_before[-1] or s.stopped is None:
        raise GateError("repeat_not_on_course", f"{line['run_id']}: its previous attempt is not the value's first stop "
                                                "that was on course, so no attempt follows after the end of the "
                                                "decision date (AM-19 item 2(b); AM-19a reading 7; PL-27)")
    return s.run_id


def stage_launch_order(ctx):
    """PL-27: the launch's own line is in the records commit's launch log and equals, byte for byte, the line
    built from the gate's arguments (the run_id unique, the attempt numbered, basename(ckpt_dir) = run_id:
    LaunchLog); a repeat of a launched attempt names its AM-8a report, committed with its sha256; HEAD (the pin)
    is the line's code_pin and the git_head of the sweep's earlier launched lines (DL-89). A non-default sweep
    candidate needs the default's launched line and its predecessors' launch lines; attempt 1 is refused once
    the decision date has ended (decision_date_ended), a later attempt then needs --previous-run (s of PL-9(e)).
    Prints the decision date and the launch-order position; warns 6 h after the default's launch."""
    ss, _ = _lib()
    src, log, rules, schedule = ctx["src"], ctx["log"], ctx["rules"], ctx["schedule"]
    report = None
    try:
        if ctx["am8a_report"] is not None:
            rel, data = src.read(ctx["am8a_report"], missing="repeat_report_missing",
                                 uncommitted="repeat_report_mismatch", record=True)
            report = {"path": rel, "sha256": ss.sha256_bytes(data)}
        line = launch_line(ss, stage=ctx["stage"], seed=ctx["seed"], profile=ctx["profile"],
                           lam=ctx["lambda_logit"], alpha=ctx["alpha"], attempt=ctx["attempt"],
                           schedule=schedule["in_force"], code_pin=ctx["pin"], ckpt_dir=ctx["ckpt_dir"],
                           am8a_report=report)
        rid, raw = line["run_id"], line_bytes(line)
        if rid not in log.launch:
            raise GateError("launch_line_missing", f"{rid}: no launch line in {ss.LAUNCH_LOG_REL} at the records commit "
                                                   "(print it with entry, commit and push it before the gate; PL-27)")
        if log.raw[rid] != raw:
            raise GateError("launch_line_mismatch", f"{rid}: the launch log's line differs from the line built from the "
                                                    f"gate's arguments:\n  log:  {log.raw[rid].decode()}\n  gate: "
                                                    f"{raw.decode()}")
        group = [x for x in log.order[:log.order.index(rid)] if group_of(log.launch[x]) == group_of(line)]
        if group:
            prev = group[-1]
            if prev not in log.launched and prev not in log.not_launched:
                raise GateError("previous_attempt_unresolved", f"{prev}, the attempt before {rid}, has neither a "
                                                               "launched nor a not_launched line")
            launched_before = [x for x in group if x in log.launched]
            if launched_before and report is None:
                raise GateError("repeat_report_missing", f"{rid} follows the launched attempt {launched_before[-1]}: "
                                                         "its launch line names the AM-8a report (AM-19 item 3(a); "
                                                         "PL-9(c): only an attempt that follows not_launched lines "
                                                         "alone needs none)")
        arec = (ctx.get("sweep_records") or {}).get("alpha_cwd")
        dates = sweep_dates(ss, src, rules, schedule, log, ctx["ident"], None if arec is None else arec["path"])
        ident, notes = ctx["ident"], []
        if ident["sweep"] is not None:
            key = ident["sweep"]
            sweep = rules[key]
            atts = ss.sweep_attempts(log, sweep, key, schedule)
            pins = sorted({a.line["code_pin"] for v in atts for a in atts[v] if a.run_id != rid})
            if pins and pins != [ctx["pin"]]:
                raise GateError("code_pin_mismatch", f"the {key} sweep's earlier launch lines carry the code pins "
                                                     f"{pins}; this launch is at {ctx['pin']} (DL-89: one pin per "
                                                     "sweep; PL-14: every launch line, a not_launched start's too)")
            earlier = [a for v in atts for a in atts[v] if a.launched is not None and a.run_id != rid]
            heads = sorted({a.launched["git_head"] for a in earlier})
            if heads and heads != [ctx["pin"]]:
                raise GateError("code_pin_mismatch", f"the {key} sweep's launched lines ran at {heads}; this launch "
                                                     f"is at {ctx['pin']} (DL-89: one pin per sweep)")
            default = float(sweep["default_candidate"])
            if ident["value"] != default:
                d_launched = [a for a in atts[default] if a.launched is not None]
                if not d_launched:
                    raise GateError("default_not_launched", f"the {key} default {default:g} has no launched line: a "
                                                            "non-default candidate launches after it (AM-19 item 1(e))")
                order = [float(v) for v in ss.LAUNCH_ORDER[key]]
                missing = [v for v in order[:order.index(ident["value"])] if not atts[v]]
                if missing:
                    raise GateError("launch_order", f"{key} {ident['value']:g} launches after {missing}, which have no "
                                                    f"launch line (the launch order {order}; AM-19 item 1(e))")
                if ctx["now"] >= dates["cutoff_utc"]:
                    if ctx["attempt"] == 1:
                        raise GateError("decision_date_ended", f"{rid}: the {key} decision date {dates['decision_date']} "
                                                               "has ended; a non-default candidate's first attempt "
                                                               "no longer launches (AM-19 item 2(b); PL-27)")
                    notes.append(f"the on-course repeat of {previous_runs_ok(ss, ctx, line, dates['cutoff_utc'])}")
                since = ctx["now"] - d_launched[0].launch_ts
                if since > DEFAULT_LAUNCH_WARN:
                    ctx["warnings"].append(f"{since / 3600:.1f} h since the {key} default's first launch "
                                           f"({d_launched[0].run_id}): more than 6 h (the launch note; P5)")
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    # PL-13: the gate record stores the sha256 of its launch line: the log's own bytes for the line, without the
    # newline that ends it (equal to the bytes built from the gate's arguments, checked above)
    ctx.update(launch_line=line, launch_line_sha256=ss.sha256_bytes(log.raw[rid]), decision_date=dates["decision_date"],
               cutoff_utc=dates["cutoff_utc"], launch_order_position=dates["position"])
    note = (f"decision date {dates['decision_date']} (C = {ss.iso_utc(dates['cutoff_utc'])}), launch-order position "
            f"{dates['position']}" if ident["sweep"] is not None else "outside the sweeps: no decision date")
    for w in ctx["warnings"]:
        print(f"WARN: {w}", flush=True)
    return _ok(f"own launch line equals the log's ({rid}); {note}" + (f"; {'; '.join(notes)}" if notes else ""),
               launch_line=line, decision_date=dates["decision_date"], launch_order_position=dates["position"])


def third_party_imports(paths) -> list:
    """The top-level third-party modules the files import (absolute imports that are neither stdlib nor this
    repository's src/configs)."""
    import ast
    mods = set()
    for rel in paths:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
    return sorted(m for m in mods if m not in sys.stdlib_module_names and m not in ("src", "configs", "__future__"))


def fingerprint() -> dict:
    """PL-26, in the image that runs it (a child process): the Python version; the installed versions of
    FINGERPRINT_DISTS; every nvidia-* distribution; the distributions providing the third-party modules that
    FINGERPRINT_FILES import, with their versions; and whether mmseg.apis and ftfy import (the teacher stack)."""
    import importlib.metadata as md
    fp = {"python": platform.python_version()}
    for dist in FINGERPRINT_DISTS:
        try:
            fp[dist] = md.version(dist)
        except md.PackageNotFoundError:
            fp[dist] = None
    fp["nvidia"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
                          if str(d.metadata["Name"] or "").lower().startswith("nvidia-"))
    pd = md.packages_distributions()
    fp["imported"] = {m: sorted(pd.get(m, [])) for m in third_party_imports(FINGERPRINT_FILES)}
    for mod in ("mmseg.apis", "ftfy"):
        try:
            importlib.import_module(mod)
            fp[f"import {mod}"] = True
        except Exception as e:                             # noqa: BLE001 -- recorded, judged by the gate
            fp[f"import {mod}"] = f"{type(e).__name__}: {e}"
    return fp


def fingerprint_problems(fp: dict) -> list:
    """The differences of a fingerprint from the fingerprint of record (PL-26); [] when it matches."""
    bad = [f"{k}: {fp.get(k)!r} != {v!r}" for k, v in FINGERPRINT_OF_RECORD.items() if fp.get(k) != v]
    for mod, dists in sorted((fp.get("imported") or {}).items()):
        for d in dists or [None]:
            if d is None or d.lower() not in FINGERPRINT_DISTS:
                bad.append(f"module {mod} comes from {d!r}, which has no value of record")
    if not fp.get("imported"):
        bad.append("no third-party import was found in the fingerprint files")
    for mod in ("mmseg.apis", "ftfy"):
        if fp.get(f"import {mod}") is not True:
            bad.append(f"import {mod}: {fp.get(f'import {mod}')!r}")
    return bad


def _child_json(argv, tag: str, timeout: int, env=None) -> tuple:
    """A child interpreter (-B, cwd = checkout); returns (returncode, its last `<tag>:` JSON or None, output)."""
    r = subprocess.run([sys.executable, "-B", *[str(a) for a in argv]], cwd=str(REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       env=dict(os.environ if env is None else env, PYTHONIOENCODING="utf-8"))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(f"{tag}:")]
    doc = json.loads(lines[-1][len(tag) + 1:]) if lines else None
    return r.returncode, doc, (r.stdout + r.stderr)[-600:]


def stage_kd_image(ctx):
    """PL-26 (DL-87, E-53): PLANTSEG_IMAGE_DIGEST is the KD image of record, and a child process's fingerprint
    equals the fingerprint of record."""
    digest = os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip()
    if not digest:
        raise GateError("image_digest_missing", "PLANTSEG_IMAGE_DIGEST is not set (DL-87)")
    if digest != KD_IMAGE_DIGEST:
        raise GateError("image_digest_mismatch", f"PLANTSEG_IMAGE_DIGEST {digest} != the KD image of record "
                                                 f"{KD_IMAGE_DIGEST} (DL-87)")
    rc, fp, out = ctx["child_json"]([Path(__file__).resolve(), "_fingerprint"], "FINGERPRINT_JSON", 600)
    if rc != 0 or fp is None:
        raise GateError("fingerprint_unavailable", f"the fingerprint child failed: rc={rc} {out}")
    bad = fingerprint_problems(fp)
    if bad:
        raise GateError("fingerprint_mismatch", f"the image differs from the fingerprint of record (PL-26): {bad[:6]}")
    return _ok(f"digest == DL-87; fingerprint == record (python {fp['python']}, torch {fp['torch']}, numpy "
               f"{fp['numpy']}, mmcv {fp['mmcv']}, mmsegmentation {fp['mmsegmentation']}, no nvidia-*)",
               image_digest=digest, fingerprint=fp)


def teacher_identity_child(ckpt: str) -> dict:
    """The teacher of record, loaded on the CPU in a child process (P5 stage 9; E-31; DL-88)."""
    rec = {"ok": False}
    try:
        _, td = _lib()
        import torch
        rec["sha256"] = sha256_file(ckpt)
        rec["bytes"] = os.path.getsize(ckpt)
        frozen = td.load_frozen_teacher(ckpt, config_path=str(td.DEFAULT_TEACHER_CONFIG),
                                        expected_sha256=TEACHER_SHA256)
        model = frozen.teacher
        rec["provenance"] = frozen.provenance.as_dict()
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        rec["trainable"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        rec["training_modules"] = sorted(n for n, m in model.named_modules() if m.training)[:10]
        rec["state_entries"] = len(model.state_dict())
        rec["cuda_initialized"] = bool(torch.cuda.is_initialized())
        rec["ok"] = True
    except Exception as e:                                 # noqa: BLE001 -- reported, judged by the gate
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def teacher_identity_problems(rec: dict, ckpt: str) -> list:
    """The differences of the child's record from the teacher of record; [] when it matches."""
    if not rec.get("ok"):
        return [f"the teacher did not load: {rec.get('error')}"]
    prov = rec.get("provenance") or {}
    want = {"sha256": TEACHER_SHA256, "bytes": TEACHER_BYTES, "params": TEACHER_PARAMS, "trainable": 0,
            "training_modules": [], "state_entries": TEACHER_STATE_ENTRIES}
    bad = [f"{k}: {rec.get(k)!r} != {v!r}" for k, v in want.items() if rec.get(k) != v]
    pwant = {"builder": TEACHER_BUILDER, "ckpt_sha256": TEACHER_SHA256, "expected_sha256": TEACHER_SHA256,
             "ckpt_bytes": TEACHER_BYTES, "ckpt_path": str(ckpt), **TEACHER_VALUES}
    bad += [f"provenance {k}: {prov.get(k)!r} != {v!r}" for k, v in pwant.items() if prov.get(k) != v]
    cfg = prov.get("config_path")
    if not (isinstance(cfg, str) and _under(Path(cfg), REPO)):
        bad.append(f"provenance config_path {cfg!r} is not under the checkout")
    return bad


def stage_teacher_identity(ctx):
    """P5 stage 9: a CPU child (CUDA_VISIBLE_DEVICES empty) hashes the teacher file, loads it strictly with the
    expected sha256, and the record equals the teacher of record: sha256, bytes, provenance (DL-88's values),
    27,618,868 parameters, none trainable, no module in training mode, 854 state entries (E-31)."""
    t = ctx["teacher_ckpt"]
    if ctx["rehearsal"] and not Path(t).is_file():
        return _skip(f"SKIPPED (rehearsal: no teacher file at {t})")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    rc, rec, out = ctx["child_json"]([Path(__file__).resolve(), "_teacher-identity", "--teacher-ckpt", t],
                                     "TEACHER_JSON", 1800, env)
    if rec is None:
        raise GateError("teacher_identity_failed", f"the teacher child failed: rc={rc} {out}")
    bad = teacher_identity_problems(rec, t)
    if bad:
        raise GateError("teacher_identity_mismatch", f"the teacher differs from the teacher of record (DL-88): "
                                                     f"{bad[:6]}")
    return _ok(f"{TEACHER_FILE} sha256 {TEACHER_SHA256[:12]}…, strict load, DL-88 values, {TEACHER_PARAMS:,} params, "
               "0 trainable, eval mode", teacher=rec)


def dry_run_argv(ctx, tmp: Path) -> list:
    """PL-25: the launch's stage, seed, lambda and alpha, the teacher of record with its sha256, the profile's
    horizon; a dry run on the CPU into `tmp`."""
    argv = [REPO / "src" / "training" / "train_distill.py", "--dry-run", "--stage", ctx["stage"], "--seed",
            str(ctx["seed"])]
    if ctx["lambda_logit"] is not None:
        argv += ["--lambda-logit", fmt_value(ctx["lambda_logit"])]
    if ctx["alpha"] is not None:
        argv += ["--alpha", fmt_value(ctx["alpha"])]
    return argv + ["--teacher-ckpt", ctx["teacher_ckpt"], "--teacher-ckpt-sha256", TEACHER_SHA256,
                   "--ckpt-dir", str(tmp), *PROFILES[ctx["profile"]]["extra_args"]]


def scratch_dir(prefix: str) -> Path:
    """A new directory under the system temp dir, named <prefix>_<pid>_<n>: no random suffix (SL-1)."""
    n = 0
    while True:
        n += 1
        p = Path(tempfile.gettempdir()).resolve() / f"{prefix}_{os.getpid()}_{n}"
        if "test" in str(p).lower():
            raise GateError("tmp_misplaced", f"{p}: its path contains 'test' (SL-1); set TMPDIR elsewhere")
        try:
            p.mkdir()
        except FileExistsError:
            continue
        return p


def stage_kd_dry_run(ctx):
    """PL-25: a child runs train_distill.py --dry-run with the launch's values and the teacher of record, into a
    temporary directory outside the repository and the data root (removed); it must print RESULT: PASS and name
    that directory."""
    if ctx["rehearsal"] and not Path(ctx["teacher_ckpt"]).is_file():
        return _skip(f"SKIPPED (rehearsal: no teacher file at {ctx['teacher_ckpt']})")
    tmp = scratch_dir("kd_dryrun")
    try:
        if _under(tmp, REPO) or _under(tmp, ctx["root"]):
            raise GateError("dry_run_tmp_misplaced", f"{tmp} is inside the repository or the data root")
        r = ctx["child_run"](dry_run_argv(ctx, tmp), 3600)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("RESULT:")), "")
        ck = re.search(r"^\[ckpt\] dir=(.+?) \(verified OUTSIDE repo\)$", r.stdout, re.M)
        if r.returncode != 0 or line != "RESULT: PASS" or ck is None or Path(ck.group(1)).resolve() != tmp:
            raise GateError("kd_dry_run_failed", f"rc={r.returncode} {line!r} ckpt_dir={ck.group(1) if ck else None} "
                                                 f"(expected {tmp}) {r.stderr[-300:]}")
        return _ok(f"{line} into {tmp} (removed)", dry_run=line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_module_provenance(ctx):
    """E1's module provenance, with sweep_select, train_distill and both gates' files."""
    _lib()
    status, detail, rec = E1G.stage_module_provenance(ctx)
    foreign = E1G.module_provenance(extra=(("preflight_e1_trainval", E1G.__file__),
                                           ("preflight_distill", __file__)))
    if foreign:
        raise GateError("import_origin_foreign", f"modules loaded from outside {REPO}: {foreign[:5]}")
    return status, f"{detail}; sweep_select, train_distill and both gates under {REPO}", rec


# E1's stages are the same objects (PL-25), in E1's order, with the KD stages between them
STAGES = [
    ("arguments", stage_arguments),
    ("data_isolation", E1G.stage_data_isolation),
    ("repo_state", E1G.stage_repo_state),
    ("module_provenance", stage_module_provenance),
    ("records", stage_records),
    ("selection_inputs", stage_selection_inputs),
    ("launch_order", stage_launch_order),
    ("class_weights", E1G.stage_class_weights),
    ("smoke_loss", E1G.stage_smoke_loss),
    ("kd_image", stage_kd_image),
    ("teacher_identity", stage_teacher_identity),
    ("cuda", E1G.stage_cuda),
    ("imagenet_backbone", E1G.stage_imagenet_backbone),
    ("smoke_loader_seed", E1G.stage_smoke_loader_seed),
    ("smoke_dataloader", E1G.stage_smoke_dataloader),
    ("seed_sequence_R5", E1G.stage_seed_sequence),
    ("kd_dry_run", stage_kd_dry_run),
    ("repo_unchanged", E1G.stage_repo_unchanged),
]
assert tuple(n for n, _ in STAGES) == STAGE_NAMES_ORDER


# ------------------------------------------------------------------------------------------ the launch
def build_launch_block(ctx) -> str:
    """PL-29 and P5's launch block: the ckpt dir absent or empty and the stdout file absent, nothing written into
    the ckpt dir before the trainer (E-17: stdout and pid go to the evidence folder), the launch's values, the
    records folder's copies of the selection files, the records commit; its last line names the next steps."""
    q = shlex.quote
    d, ev, rid = q(str(ctx["ckpt_dir"])), Path(ctx["evidence"]), ctx["run_id"]
    out, pid = q(str(ev / f"{rid}.stdout.log")), q(str(ev / f"{rid}.pid"))
    args = ["--real-run", "--confirm-real-run", "--init", "imagenet", "--stage", ctx["stage"], "--seed", str(ctx["seed"])]
    sel = ctx["selection_files"]
    if ctx["lambda_logit"] is not None:
        args += ["--lambda-logit", fmt_value(ctx["lambda_logit"])]
        if sel["lambda_logit"] is not None:
            args += ["--lambda-selection", sel["lambda_logit"]["path"]]
    if ctx["alpha"] is not None:
        args += ["--alpha", fmt_value(ctx["alpha"])]
        if sel["alpha_cwd"] is not None:
            args += ["--alpha-selection", sel["alpha_cwd"]["path"]]
    args += [*PROFILES[ctx["profile"]]["extra_args"], "--teacher-ckpt", ctx["teacher_ckpt"], "--teacher-ckpt-sha256",
             TEACHER_SHA256, "--num-workers", str(NUM_WORKERS), "--log-every", str(LOG_EVERY), "--records-commit",
             ctx["records_commit"], "--ckpt-dir", str(ctx["ckpt_dir"])]
    env = [f"PYTHONPATH={q(str(REPO))}", f"PLANTSEG_DATA_ROOT={q(str(ctx['root']))}",
           f"PLANTSEG_IMAGE_DIGEST={q(KD_IMAGE_DIGEST)}"]
    if ctx.get("hub_dir"):
        env.append(f"TORCH_HOME={q(str(Path(ctx['hub_dir']).parent))}")
    cmd = (f"env -u PLANTSEG_GIT_COMMIT {' '.join(env)} nohup {q(sys.executable)} -B src/training/train_distill.py "
           + " ".join(q(a) for a in args) + f" > {out} 2>&1 &")
    return "\n".join(["set -eu", f"cd {q(str(REPO))}",
                      f"[ -z \"$(ls -A {d} 2>/dev/null)\" ] || {{ printf 'STOP: %s is not empty\\n' {d}; exit 1; }}",
                      f"[ ! -e {out} ] || {{ printf 'STOP: %s exists\\n' {out}; exit 1; }}",
                      cmd, f"echo $! > {pid}", NEXT_LINE])


def gate_record(ctx, results: list, verdict: str, block=None) -> dict:
    """The gate's evidence record (PL-28: the pin, the records commit and each records file's blob id); written
    before GO is printed; check-run-meta reads it."""
    keys = ("profile", "stage", "seed", "run_id", "ckpt_dir", "evidence", "records", "records_commit", "pin",
            "records_blobs", "teacher_ckpt", "lambda_logit", "alpha", "attempt", "clock_offset", "selection_files",
            "launch_line", "launch_line_sha256", "decision_date", "cutoff_utc", "launch_order_position", "warnings")
    return {"format": "kd_gate_record/1", "verdict": verdict, "repo": str(REPO), "python": sys.executable,
            **{k: ctx.get(k) for k in keys}, "stages": results, "launch_block": block}


def _write_record(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def gate_ctx(args) -> dict:
    return {"profile": args.profile, "stage": args.stage, "seed": args.seed, "ckpt_dir": args.ckpt_dir,
            "evidence": args.evidence, "expect_head": args.expect_head, "records_commit": args.records_commit,
            "records": args.records, "teacher_ckpt": args.teacher_ckpt, "clock_offset": args.clock_offset_seconds,
            "record": args.record, "rehearsal": args.rehearsal, "lambda_logit": args.lambda_logit,
            "alpha": args.alpha, "attempt": args.attempt, "am8a_report": args.am8a_report,
            "previous_runs": list(args.previous_run or []), "decision_records": list(args.decision_record or []),
            "git_root": None, "now": time.time(), "warnings": [], "child_json": _child_json,
            "child_run": lambda argv, timeout: E1G._child(argv, timeout=timeout)}


def run_gate(args, *, stages=None, ctx=None) -> int:
    """E1's gate loop (preflight_e1_trainval.run_gate) over the KD stages: the first FAIL stops; GO only when every
    stage PASSes outside a rehearsal; the record is written before GO is printed."""
    if not args.rehearsal and args.record is None:
        print("usage error: a real gate run requires --record PATH (or pass --rehearsal)", file=sys.stderr)
        return 2
    ctx = gate_ctx(args) if ctx is None else ctx
    stages = STAGES if stages is None else stages
    print("=" * 78)
    print("KD LAUNCH GATE (lane 4(b), K2) -- the E1 gate is loaded, never run")
    print(f"repo={REPO}")
    print(f"python={sys.executable} {platform.python_version()} | {platform.platform()}")
    print(f"profile={args.profile} | stage={args.stage} | seed={args.seed} | lambda={args.lambda_logit} | "
          f"alpha={args.alpha} | attempt={args.attempt} | records commit={args.records_commit}")
    if args.rehearsal:
        print("MODE: --rehearsal -- exercises the gate; NEVER a launch authorization.")
    print("=" * 78, flush=True)
    results, first = [], None
    for name, fn in stages:
        t0 = time.time()
        try:
            status, detail, rec = fn(ctx)
            code = None
        except GateError as e:
            status, detail, rec, code = "FAIL", str(e), {}, e.code
        except Exception as e:                             # noqa: BLE001 -- a crash is a FAIL, never a GO
            status, detail, rec, code = "FAIL", f"[stage_error] {type(e).__name__}: {e}", {}, "stage_error"
        results.append({"stage": name, "status": status, "detail": detail, "code": code,
                        "secs": round(time.time() - t0, 1), "record": rec})
        print(f"\n>>> {name}: {status}  {detail}", flush=True)
        if status == "FAIL":
            first = (name, code)
            break
    print("\n" + "=" * 78)
    print("GATE SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"  {r['stage']:<20}{r['status']:<10}{r['secs']:>7.1f}  {r['detail'][:160]}")
    for name, _fn in stages[len(results):]:
        print(f"  {name:<20}{'SKIPPED':<10}{'-':>7}  not reached")
    go = (not args.rehearsal and len(results) == len(stages) and all(r["status"] == "PASS" for r in results))
    record_ok = args.record is not None and bool(results) and results[0]["status"] == "PASS"
    print()
    if go:
        block = build_launch_block(ctx)
        try:
            _write_record(Path(args.record), gate_record(ctx, results, "GO", block))
        except OSError as e:
            print(f"VERDICT: NO-GO -- the evidence record could not be written ({type(e).__name__}: {e})")
            return 1
        print(f"record written: {args.record}")
        print("VERDICT: GO")
        if ctx.get("decision_date") is not None:
            print(f"launch note: decision date {ctx['decision_date']}, launch-order position "
                  f"{ctx['launch_order_position']} (the E1 runbook's 9.13)")
        for w in ctx.get("warnings") or []:
            print(f"WARN: {w}")
        print(f"Save the block below as {Path(args.evidence) / (ctx['run_id'] + '.sh')} and run it with bash from THIS "
              "shell:")
        print("----- launch block -----")
        print(block)
        print("----- end launch block -----")
        return 0
    verdict = "NO-GO (rehearsal)" if args.rehearsal else "NO-GO"
    if first is None:
        print(f"VERDICT: {verdict} -- " + ("not a launch authorization" if args.rehearsal
                                           else "a stage did not PASS (SKIPPED is never a GO)"))
    else:
        print(f"VERDICT: {verdict} -- first failing stage: {first[0]} [{first[1]}]")
    if args.rehearsal:
        print("(launch block withheld)")
    if record_ok:
        try:
            _write_record(Path(args.record), gate_record(ctx, results, verdict))
        except OSError as e:            # the verdict stands: a NO-GO whose record cannot be written is still a NO-GO
            print(f"(the evidence record could not be written: {type(e).__name__}: {e})")
    return 1


# ------------------------------------------------------------------------------------- check-run-meta
LOGIT_KD_GRID = "os8 64x64 (head-native, no upsample)"
CWD_FEAT_GRID = "stride-16 32x32"
CWD_LOGIT_GRID = "os8 64x64 (validity mask shared with Logit-KD when on)"
SUPERVISED_GRID = "full 512x512"
ARMS = ("a", "f", "g")
PROVENANCE_KEYS = ("builder", "ckpt_path", "ckpt_sha256", "ckpt_bytes", "expected_sha256", "config_path",
                   "config_sha256", "ham_kwargs", "architecture_signature", "teacher_components_sha256",
                   "reused_module_hashes", "model_cfg_sha256")


def same(a, b) -> bool:
    """Equal values of equal JSON types (a bool is never a number; an int is never a float)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return sorted(a) == sorted(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def run_meta_keys(stage: str) -> list:
    """The run_meta keys train_distill.run() writes for `stage`, in order (lane 4(a): the launch fields last)."""
    lk, cf, cl = TERMS[stage]
    keys = ["event", "stage", "mode", "seed", "terms", "projection_params"]
    if lk:
        keys += ["lambda_logit", "logit_kd_semantics", "logit_kd_semantics_declared", "logit_kd_semantics_override_used",
                 "logit_kd_grid", "T_logit", "lambda_sweep_grid"]
    if cf:
        keys += ["alpha_cwd", "alpha_offgrid", "alpha_grid", "cwd_feat_grid", "T_cwd", "cwd_C"]
    if cl:
        keys += ["beta_cwd"] + ([] if cf else ["T_cwd"]) + ["cwd_logit_grid"]
    return keys + ["supervised_grid", "batch_size", "max_iters", "num_classes", "teacher_nmf", "wall_clock",
                   "git_head", "git_head_source", "image_digest", "torch", "numpy", "device", "cuda_available",
                   "gpu_name", "num_workers", "persistent_workers", "val_interval", "max_val_batches", "learning_rate",
                   "momentum", "weight_decay", "lr_power", "poly_horizon", "grad_clip_norm", "used_pretrained", "params",
                   "ignore_index", "ramp_iters", "class_weights_sha256", "tf32", "teacher_provenance", "teacher_mock",
                   "arm", "descriptive", "parent_of_e4_e7", "selection_sha256", "selection_files", "records_commit"]


def provenance_rules(exp: dict) -> dict:
    """teacher_provenance, key by key (E-13 widened; PL-30; DL-88)."""
    path_ok = (lambda p: isinstance(p, str) and Path(p).name == TEACHER_FILE  # noqa: E731
               and (exp["teacher_ckpt"] is None or p == exp["teacher_ckpt"]))
    return {"builder": lambda v: same(v, TEACHER_BUILDER), "ckpt_path": path_ok,
            "ckpt_sha256": lambda v: same(v, TEACHER_SHA256), "ckpt_bytes": lambda v: same(v, TEACHER_BYTES),
            "expected_sha256": lambda v: same(v, TEACHER_SHA256),
            "config_path": lambda v: isinstance(v, str) and _under(Path(v), REPO),
            **{k: (lambda v, w=w: same(v, w)) for k, w in TEACHER_VALUES.items()}}


def meta_rules(stage: str, exp: dict, td) -> dict:
    """PL-30: exactly one rule per run_meta key: (kind, check), kind one of value, record (a value of record),
    stage (a per-stage value), exempt, warning; check(value) -> bool."""
    lk, cf, cl = TERMS[stage]
    hor = exp["horizon"]

    def eq(w):
        return lambda v: same(v, w)
    r = {"event": ("value", eq("run_meta")), "stage": ("stage", eq(STAGE_NAMES[stage])), "mode": ("value", eq(exp["mode"])),
         "seed": ("stage", eq(exp["seed"])),
         "terms": ("stage", eq({"logit_kd": lk, "cwd_feat": cf, "cwd_logit": cl})),
         "projection_params": ("stage", eq(PROJECTION_PARAMS if cf else 0))}
    if lk:
        r.update({"lambda_logit": ("stage", eq(exp["lambda_logit"])),
                  "logit_kd_semantics": ("record", eq(td.LOGIT_KD_SEMANTICS)),
                  "logit_kd_semantics_declared": ("value", eq(None)),
                  "logit_kd_semantics_override_used": ("value", eq(False)),
                  "logit_kd_grid": ("record", eq(LOGIT_KD_GRID)), "T_logit": ("record", eq(td.T_LOGIT)),
                  "lambda_sweep_grid": ("record", eq(list(td.LAMBDA_SWEEP)))})
    if cf:
        r.update({"alpha_cwd": ("stage", eq(exp["alpha"])), "alpha_offgrid": ("value", eq(False)),
                  "alpha_grid": ("record", eq(list(td.ALPHA_GRID))), "cwd_feat_grid": ("record", eq(CWD_FEAT_GRID)),
                  "cwd_C": ("record", eq(td.CWD_C_FEAT))})
    if cf or cl:
        r["T_cwd"] = ("record", eq(td.T_CWD))
    if cl:
        r.update({"beta_cwd": ("record", eq(td.BETA_CWD_LOGIT)), "cwd_logit_grid": ("record", eq(CWD_LOGIT_GRID))})
    prov = provenance_rules(exp)
    r.update({
        "supervised_grid": ("record", eq(SUPERVISED_GRID)), "batch_size": ("value", eq(16)),
        "max_iters": ("value", eq(exp["max_iters"])), "num_classes": ("value", eq(116)),
        "teacher_nmf": ("value", lambda v: isinstance(v, dict) and bool(v)),
        "wall_clock": ("exempt", lambda v: isinstance(v, float) and math.isfinite(v)),
        "git_head": ("value", eq(exp["pin"])), "git_head_source": ("value", eq("git_checkout")),
        "image_digest": ("record", eq(KD_IMAGE_DIGEST)), "torch": ("record", eq(FINGERPRINT_OF_RECORD["torch"])),
        "numpy": ("record", eq(FINGERPRINT_OF_RECORD["numpy"])), "device": ("value", eq("cuda")),
        "cuda_available": ("value", eq(True)), "gpu_name": ("warning", eq(EXPECTED_GPU_NAME)),
        "num_workers": ("record", eq(NUM_WORKERS)), "persistent_workers": ("value", eq(True)),
        "val_interval": ("value", eq(4000)), "max_val_batches": ("value", eq(None)),
        **{k: ("record", eq(E1G.RUN_META_EXPECT[k])) for k in ("learning_rate", "momentum", "weight_decay", "lr_power")},
        "poly_horizon": ("value", eq(hor)), "grad_clip_norm": ("value", eq(None)),
        "used_pretrained": ("value", eq(exp["used_pretrained"])), "params": ("record", eq(STUDENT_PARAMS)),
        "ignore_index": ("value", eq(255)), "ramp_iters": ("record", eq(RAMP_ITERS)),
        "class_weights_sha256": ("record", eq(CLASS_WEIGHTS_SHA256)), "tf32": ("record", eq(dict(td.TF32_DEFAULTS))),
        "teacher_provenance": ("record", lambda v: isinstance(v, dict) and sorted(v) == sorted(PROVENANCE_KEYS)
                               and all(fn(v[k]) for k, fn in prov.items())),
        "teacher_mock": ("value", eq(False)),
        "arm": ("stage", eq(STAGE_NAMES[stage] if stage in ARMS else None)),
        "descriptive": ("stage", eq(stage in ARMS or hor != 80000)),
        "parent_of_e4_e7": ("stage", eq(stage == "e3" and hor == 80000)),
        "selection_sha256": ("value", eq(exp["selection_sha256"])),
        "selection_files": ("value", eq(exp["selection_files"])),
        "records_commit": ("value", eq(exp["records_commit"]))})
    return r


def meta_problems(row: dict, stage: str, exp: dict, td) -> tuple:
    """(problems, warnings) of one run_meta row: the exact key set of the stage, then every key's rule."""
    rules = meta_rules(stage, exp, td)
    want = run_meta_keys(stage)
    problems, warns = [], []
    if sorted(rules) != sorted(want):
        problems.append(f"rule table and key list differ: {sorted(set(rules) ^ set(want))}")
    if sorted(row) != sorted(want):
        problems.append(f"keys: missing {sorted(set(want) - set(row))}, extra {sorted(set(row) - set(want))}")
    for k in want:
        if k not in row or k not in rules:
            continue
        kind, ok = rules[k]
        if not ok(row[k]):
            if kind == "warning":
                warns.append(f"{k} {row[k]!r} (recorded, not gated)")
            elif kind != "exempt" or row[k] is not None:
                detail = row[k]
                if k == "teacher_provenance" and isinstance(row[k], dict):
                    prov = provenance_rules(exp)
                    detail = {pk: row[k].get(pk) for pk, fn in prov.items() if not fn(row[k].get(pk))}
                problems.append(f"{k} ({kind}): {detail!r}")
    return problems, warns


def run_meta_expectations(args) -> dict:
    """From the gate's GO record, or (--allow-smoke) the pod smoke's dry row: mode dry, max_iters 25, no
    pretrained weights, no selection files and no records commit."""
    if args.allow_smoke:
        hor = PROFILES[args.profile]["horizon"]
        nulls = {"lambda_logit": None, "alpha_cwd": None}
        return {"stage": args.stage, "mode": "dry", "seed": args.seed, "max_iters": 25, "horizon": hor,
                "used_pretrained": False, "pin": args.expect_head, "records_commit": None,
                "lambda_logit": None if args.lambda_logit is None else float(args.lambda_logit),
                "alpha": None if args.alpha is None else float(args.alpha), "selection_sha256": nulls,
                "selection_files": nulls, "teacher_ckpt": None, "decision_date": None, "position": None,
                "run_id": Path(args.ckpt_dir).name}
    try:                                # an unreadable or partial record is a usage refusal (exit 2), never a FAIL
        rec = json.loads(Path(args.gate_record).read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or rec.get("format") != "kd_gate_record/1" or rec.get("verdict") != "GO":
            raise GateError("gate_record", f"{args.gate_record} is not a GO record of this gate")
        if not (isinstance(rec["ckpt_dir"], str) and Path(rec["ckpt_dir"]).is_absolute()):
            raise GateError("gate_record", f"{args.gate_record}: ckpt_dir {rec['ckpt_dir']!r} is not an absolute path")
        if rec["stage"] not in STAGE_NAMES:
            raise GateError("gate_record", f"{args.gate_record}: stage {rec['stage']!r} is not a stage of this gate")
        sel = rec["selection_files"]
        hor = PROFILES[rec["profile"]]["horizon"]
        return {"stage": rec["stage"], "mode": "real", "seed": rec["seed"], "max_iters": hor, "horizon": hor,
                "used_pretrained": True, "pin": rec["pin"], "records_commit": rec["records_commit"],
                "lambda_logit": rec["lambda_logit"], "alpha": rec["alpha"],
                "selection_sha256": {k: None if v is None else v["sha256"] for k, v in sel.items()},
                "selection_files": {k: None if v is None else v["path"] for k, v in sel.items()},
                "teacher_ckpt": rec["teacher_ckpt"], "decision_date": rec["decision_date"],
                "position": rec["launch_order_position"], "run_id": rec["run_id"], "ckpt_dir": rec["ckpt_dir"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise GateError("gate_record", f"{args.gate_record}: unreadable, or not a complete GO record of this gate "
                                       f"({type(e).__name__}: {e})") from e


def check_run_meta(args) -> int:
    """PL-30: the run's run_meta file holds exactly one row, and every key passes its one rule. The launched line
    (PL-13) is printed on PASS and on FAIL: commit and push it; after a FAIL the run is stopped under the runbook
    (a stopped line and its report). A --ckpt-dir that is not the gate record's run is refused (exit 2) before any
    check: a mistyped directory is no FAIL of the run, and no launched line is printed for it."""
    _, td = _lib()
    exp = run_meta_expectations(args)
    d = Path(args.ckpt_dir)
    stage = exp["stage"]
    meta_p = d / f"{STAGE_NAMES[stage].lower()}_run_meta.jsonl"
    problems, warns, rows = [], [], []
    # paths, not strings: a ckpt_dir typed with a trailing slash, an inner '//' or '/./' is the same directory
    # (a leading '//' is not: pathlib keeps it, and the refusal then names the spelling that passes)
    wrong_dir = d.name != exp["run_id"] or (exp.get("ckpt_dir") is not None and d != Path(exp["ckpt_dir"]))
    if wrong_dir:                       # a mistyped directory is a usage refusal: no FAIL, no launched line
        raise GateError("ckpt_dir_not_the_runs", f"--ckpt-dir {d} is not the run of this check "
                                                 f"({exp.get('ckpt_dir') or exp['run_id']}): pass this run's own "
                                                 "--ckpt-dir and its own gate record; nothing is printed for the "
                                                 "launch log")
    if not meta_p.is_file():
        problems.append(f"run_meta file not found: {meta_p}")
    else:
        for n, ln in enumerate(meta_p.read_text(encoding="utf-8").splitlines(), 1):
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                problems.append(f"run_meta line {n} is not JSON ({e})")
        if len(rows) != 1:
            problems.append(f"run_meta rows: {len(rows)} (exactly 1; more means an appended relaunch)")
    if len(rows) == 1 and isinstance(rows[0], dict):
        p, w = meta_problems(rows[0], stage, exp, td)
        problems += p
        warns += w
    for w in warns:
        print(f"WARN: {w}")
    for p in problems:
        print(f"MISMATCH: {p}")
    row = rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else None
    if row is not None:
        launched = {"event": "launched", "run_id": d.name, "run_meta_sha256": sha256_file(meta_p),
                    "launch_time_utc": row.get("wall_clock"), "git_head": row.get("git_head"),
                    "records_commit": row.get("records_commit"), "decision_date": exp["decision_date"],
                    "launch_order_position": exp["position"]}
        print("LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):")
        print(json.dumps(launched))
    else:
        print("LAUNCHED LINE: none (no single run_meta row): if the start wrote no run_meta, append a not_launched "
              "line with its evidence (AM-19a reading 6)")
    ok = not problems
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- run_meta vs the KD recipe, one rule per key (PL-30)"
          + ("" if ok else "; stop the run under the runbook: a stopped line and its AM-8a report"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------------- entry and records
def cmd_entry(args) -> int:
    """PL-27 / PL-32: the launch line of a launch, from the committed launch log at HEAD of this clone."""
    ss, _ = _lib()
    error = stage_args_error(args.stage, args.seed, args.profile, args.lambda_logit, args.alpha)
    if error is not None:
        raise GateError(error[1:error.index("]")], error[error.index("]") + 2:])
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_pin or ""):
        raise GateError("code_pin_malformed", f"--code-pin must be 40 lowercase hex: {args.code_pin!r}")
    src = ss.HeadSource()
    if not src.commit_exists(args.code_pin):
        raise GateError("code_pin_unavailable", f"--code-pin {args.code_pin} is not in this clone")
    try:
        schedule = ss.load_schedule(src)
        _, rules = ss.load_committed_rules(src)
        log = ss.load_launch_log(src)
        probe = launch_line(ss, stage=args.stage, seed=args.seed, profile=args.profile, lam=args.lambda_logit,
                            alpha=args.alpha, attempt=1, schedule=schedule["in_force"], code_pin=args.code_pin,
                            ckpt_dir=args.ckpt_dir, am8a_report=None)
        earlier = [x for x in log.order if group_of(log.launch[x]) == group_of(probe)]
        if earlier:
            prev = earlier[-1]
            if prev not in log.launched and prev not in log.not_launched:
                raise GateError("previous_attempt_unresolved", f"{prev} has neither a launched nor a not_launched line: "
                                                               "record its outcome first")
            launched_before = [x for x in earlier if x in log.launched]
            if launched_before and args.am8a_report is None:
                raise GateError("repeat_report_missing", f"this launch follows the launched attempt "
                                                         f"{launched_before[-1]}: pass --am8a-report <its committed "
                                                         "AM-8a report> (AM-19 item 3(a); PL-9(c))")
        ident = launch_identity(args.stage, args.seed, args.profile, args.lambda_logit, args.alpha)
        if ident["sweep"] is not None:                  # DL-89 / PL-14: one code pin per sweep, at every launch line
            key = ident["sweep"]
            pins = sorted({a.line["code_pin"] for v, lst in ss.sweep_attempts(log, rules[key], key, schedule).items()
                           for a in lst})
            if pins and pins != [args.code_pin]:
                raise GateError("code_pin_mismatch", f"the {key} sweep's launch lines carry the code pins {pins}; "
                                                     f"--code-pin {args.code_pin} would add another (DL-89: one pin "
                                                     "per sweep, a STOP)")
        report = None
        if args.am8a_report is not None:
            rel, data = src.read(args.am8a_report, missing="repeat_report_missing", uncommitted="repeat_report_mismatch",
                                 record=True)
            report = {"path": rel, "sha256": ss.sha256_bytes(data)}
        line = launch_line(ss, stage=args.stage, seed=args.seed, profile=args.profile, lam=args.lambda_logit,
                           alpha=args.alpha, attempt=len(earlier) + 1, schedule=schedule["in_force"],
                           code_pin=args.code_pin, ckpt_dir=args.ckpt_dir, am8a_report=report)
        if Path(args.ckpt_dir).name != line["run_id"] or not os.path.isabs(args.ckpt_dir):
            raise GateError("run_id_mismatch", f"--ckpt-dir must be an absolute path named {line['run_id']} (attempt "
                                               f"{line['attempt']}; PL-27)")
        fixed = ss.DECISION_RECORD_CORRECTED_REL["alpha_cwd"]
        dates = sweep_dates(ss, src, rules, schedule, log, ident, fixed if src.has(fixed) else None)
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    print("LAUNCH LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it before the gate):")
    print(line_bytes(line).decode("utf-8"))
    if ident["sweep"] is None:
        print("launch note: outside the sweeps (no decision date, no launch-order position)")
    else:
        print(f"launch note: {ident['sweep']} {ident['value']:g}: decision date {dates['decision_date']} (C = "
              f"{ss.iso_utc(dates['cutoff_utc'])}), launch-order position {dates['position']} of "
              f"{ss.LAUNCH_ORDER[ident['sweep']]}")
    return 0


def cmd_records(args) -> int:
    """PL-28: the records files of records commit H, written into a folder outside the repository."""
    ss, _ = _lib()
    h = args.records_commit
    if not re.fullmatch(r"[0-9a-f]{40}", h or ""):
        raise GateError("records_commit_malformed", f"--records-commit must be 40 lowercase hex: {h!r}")
    src = ss.HeadSource()
    if not src.commit_exists(h):
        raise GateError("records_commit_unavailable", f"records commit {h} is not in this clone")
    pin = src.head()
    if not src.is_ancestor(pin, h):
        raise GateError("pin_not_ancestor", f"HEAD (the pin) {pin} is not an ancestor of {h} (PL-28)")
    out = Path(args.out)
    if not out.is_absolute() or _under(out, REPO) or "test" in str(out).lower():
        raise GateError("records_folder", f"--out {out} must be absolute, outside the repository, without 'test'")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise GateError("records_folder", f"--out {out} exists and is not an empty directory")
    records = [*ss.DECISION_RECORD_REL.values(), *ss.DECISION_RECORD_CORRECTED_REL.values()]
    rels = [ss.LAUNCH_LOG_REL, ss.LAMBDA_SELECTION_REL, ss.ALPHA_SELECTION_REL, *records, ss.BAND_REL,
            ss.PUSH_EVIDENCE_REL, ss.PUSH_LIST_REL, AM7_CLIP_REL, ss.DECISION_LOG_REL]
    try:
        log = ss.LaunchLog(src.show(h, ss.LAUNCH_LOG_REL) or b"")
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    for line, _raw in log.entries:
        for k in ("am8a_report", "evidence", "report"):
            if isinstance(line.get(k), dict):
                rels.append(line[k]["path"])
    for rel in records:
        raw = src.show(h, rel)
        try:
            cor = json.loads(raw.decode("utf-8")).get("corrects") if raw else None
        except (ValueError, AttributeError):
            cor = None
        if isinstance(cor, dict):
            rels += [cor[k]["path"] for k in ("void_record", "fault_report") if isinstance(cor.get(k), dict)]
    files, absent = {}, []
    for rel in dict.fromkeys(rels):
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or "test" in rel.lower():
            raise GateError("records_mismatch", f"{rel!r} is not a repository-relative records path (SL-1)")
        oid = src.blob_at(h, rel)
        if oid is None:
            absent.append(rel)
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.blob_bytes(oid))
        files[rel] = oid
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORDS_MANIFEST).write_text(json.dumps({"format": RECORDS_FORMAT, "records_commit": h, "pin": pin,
                                                    "files": files, "absent": absent}, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"records of {h} written to {out}: {len(files)} files; absent at {h[:12]}: {absent}")
    return 0


def remote_serves(ss, url: str, commit: str) -> tuple:
    """Whether the remote at `url` serves `commit` now (AM-19a reading 10): fetched into a new, empty bare repository
    outside the checkout (removed afterwards), so that no object the clone already holds can answer for the remote;
    (returncode, the remote's answer)."""
    probe = scratch_dir("push_probe")
    try:
        git = ss.HeadSource(probe).git
        init = git("init", "-q", "--bare")
        if init.returncode != 0:
            raise GateError("push_evidence_remote", f"git init in {probe} failed: "
                                                    f"{init.stderr.decode('utf-8', 'replace')[-300:]}")
        r = git("fetch", "--no-tags", "--no-write-fetch-head", "--depth=1", "--filter=blob:none", url, commit)
        return r.returncode, without_credentials((r.stdout + r.stderr).decode("utf-8", "replace").strip(), url)
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def public_url(url: str) -> str:
    """The remote's URL as the push list records it: without user information, so that a token in
    https://<token>@host/... never reaches the committed list (the fetch itself uses the URL as given)."""
    parts = urllib.parse.urlsplit(url)
    if not parts.netloc or "@" not in parts.netloc:
        return url
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[1], parts.path, parts.query,
                                    parts.fragment))


def without_credentials(text: str, url: str) -> str:
    """The remote's answer as the push list records it: the URL and any scheme://user@ in it without user
    information."""
    return re.sub(r"(\b[a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1", text.replace(url, public_url(url)))


def cmd_push_evidence(args) -> int:
    """CHECK ITEM 3 (AM-19a reading 10): from the activity response saved verbatim (the runbook's gh api line), ask
    the remote for each push after T_lo whether it serves the pushed commit now (remote_serves: never the clone's
    own objects), record its answer, take carries_file from the commit's history (fetched into the clone when the
    clone lacks it), and write the response and the list, to be committed together in one commit. Nothing is
    written when any answer is neither served nor not served, and a write that fails leaves neither file (a file
    the cleanup cannot remove is named, for a rerun into a fresh --out)."""
    ss, _ = _lib()
    src = ss.HeadSource()
    out = Path(args.out)
    if _under(out, REPO) or "test" in str(out.resolve()).lower():
        raise GateError("push_evidence_out", f"--out {out} must lie outside the repository, without 'test'")
    targets = [out / PurePosixPath(ss.PUSH_EVIDENCE_REL).name, out / PurePosixPath(ss.PUSH_LIST_REL).name]
    if any(t.exists() for t in targets):
        raise GateError("push_evidence_out", f"{[str(t) for t in targets if t.exists()]} already exist: the evidence "
                                             "is settled once (AM-19a reading 10)")
    try:
        response = Path(args.response).read_bytes()
    except OSError as e:
        raise GateError("push_evidence_response", f"--response {args.response} unreadable ({type(e).__name__}: "
                                                  f"{e})") from e
    try:
        src.require_not_shallow()       # PL-12: a shallow clone sees a wrong adding commit and wrong ancestry
        entries = ss.activity_entries(response, args.response)
        _, ldata = ss.require_committed(ss.LAMBDA_SELECTION_REL, src=src, missing="lambda_selection_missing",
                                        uncommitted="lambda_selection_uncommitted")
        ldoc = ss.parse_json_bytes(ldata, ss.LAMBDA_SELECTION_REL, code="lambda_selection_format")
        t_in = ldoc.get("inputs_last_timestamp_utc") if isinstance(ldoc, dict) else None
        if not ss._finite_number(t_in):
            raise GateError("lambda_selection_format", f"{ss.LAMBDA_SELECTION_REL} has no finite "
                                                       "inputs_last_timestamp_utc")
        t_lo = float(t_in) - ss.PUSH_MARGIN_SECONDS
        commit = ss.adding_commit(src, ss.LAMBDA_SELECTION_REL)
        if commit is None:
            raise GateError("lambda_push_evidence_mismatch", f"no commit adds {ss.LAMBDA_SELECTION_REL} with its HEAD "
                                                             "blob")
        if not any(t < t_lo for _, t in ss._activity_times(entries, what=args.response)):
            raise GateError("lambda_push_evidence_mismatch", f"{args.response}: no entry is older than T_lo "
                                                             f"{ss.iso_utc(t_lo)}: fetch more of the activity record")
        url = src._out("remote", "get-url", args.remote)
        if not url:
            raise GateError("push_evidence_remote", f"no remote {args.remote!r} in this clone")
        answers = {}
        for e in ss.pushes_after(entries, t_lo, what=args.response):
            after = e.get("after")
            if not (isinstance(after, str) and re.fullmatch(r"[0-9a-f]{40}", after)):
                raise GateError("lambda_push_evidence_mismatch", f"push {e.get('id')!r}: after {after!r} is no commit id")
            rc, text = remote_serves(ss, url, after)
            if rc == 0:
                if not src.commit_exists(after):
                    r = src.git("fetch", "--no-tags", "--no-write-fetch-head", args.remote, after)
                    if r.returncode != 0 or not src.commit_exists(after):
                        raise GateError("push_evidence_remote", f"push {e['id']} ({after}): the remote serves it but "
                                                                "it could not be fetched into this clone; run the step "
                                                                "again")
                answers[e["id"]] = {"served": True, "carries_file": src.is_ancestor(commit, after),
                                    "remote_answer": text[-300:] or "fetched"}
            elif any(pat in text.lower() for pat in NOT_SERVED_PATTERNS):
                answers[e["id"]] = {"served": False, "carries_file": True, "remote_answer": text[-300:]}
            else:
                raise GateError("push_evidence_remote", f"push {e['id']} ({after}): the remote gave no answer "
                                                        f"(rc={rc}: {text[-300:]}); run the step again")
        doc = ss.push_list_doc(response, ldata, commit, t_lo, public_url(url), answers)
        if not any(x["carries_file"] for x in doc["pushes"]):
            raise GateError("lambda_push_evidence_mismatch", f"no push after T_lo {ss.iso_utc(t_lo)} in {args.response} "
                                                             f"carries {commit} yet: the pair would be settled with "
                                                             "no push to date (push_evidence refuses it for good); "
                                                             "fetch the activity again later, nothing is written")
    except ss.SelectionRefused as e:
        raise _refused(e) from e
    out.mkdir(parents=True, exist_ok=True)
    tmps = [t.with_name(t.name + ".tmp") for t in targets]
    try:                                # both files or neither: a refusal writes nothing, an io_error included
        tmps[0].write_bytes(response)
        tmps[1].write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        for tmp, t in zip(tmps, targets):
            tmp.replace(t)
    except OSError:
        left = []
        for p in (*tmps, *targets):     # the targets did not exist before this call (checked above)
            try:
                p.unlink(missing_ok=True)
            except OSError:             # a path whose removal failed is named unless lstat reports it absent
                try:
                    os.lstat(p)
                except (FileNotFoundError, NotADirectoryError):
                    continue            # absent: nothing remains to name
                except OSError:
                    pass                # lstat cannot tell: named, since a fresh --out is always safe
                left.append(str(p))
        if left:                        # a path that may remain in --out: the rerun needs a fresh --out
            print(f"push-evidence could not remove {left}: run the step again into a fresh --out", file=sys.stderr)
        raise
    unserved = [x["id"] for x in doc["pushes"] if not x["served"]]
    print(f"push evidence written to {out}: {len(doc['pushes'])} pushes after T_lo {ss.iso_utc(t_lo)}; not served "
          f"(each counts as carrying the file, AM-19a reading 10): {unserved}")
    print(f"Copy both files to {ss.PUSH_EVIDENCE_REL} and {ss.PUSH_LIST_REL} and commit them together in ONE commit "
          "(settled once); never re-run this step for the same push day.")
    return 0


# ------------------------------------------------------------------------------------------ the parser
class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(2)


def _launch_args(p, *, profile_default=True):
    p.add_argument("--profile", required=not profile_default, default="distill_80k" if profile_default else None,
                   choices=sorted(PROFILES))
    p.add_argument("--stage", required=True, choices=sorted(STAGE_NAMES))
    p.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    p.add_argument("--lambda-logit", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)


def build_parser():
    p = _Parser(description="KD launch gate (lane 4(b), K2).", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    e = sub.add_parser("entry", allow_abbrev=False)
    _launch_args(e)
    e.add_argument("--ckpt-dir", required=True)
    e.add_argument("--code-pin", required=True)
    e.add_argument("--am8a-report", default=None)
    r = sub.add_parser("records", allow_abbrev=False)
    r.add_argument("--records-commit", required=True)
    r.add_argument("--out", required=True)
    g = sub.add_parser("gate", allow_abbrev=False)
    _launch_args(g)
    for name in ("--ckpt-dir", "--evidence", "--expect-head", "--records-commit", "--records", "--teacher-ckpt"):
        g.add_argument(name, required=True)
    g.add_argument("--clock-offset-seconds", type=float, required=True)
    g.add_argument("--record", default=None)
    g.add_argument("--attempt", type=int, default=1)
    g.add_argument("--am8a-report", default=None)
    g.add_argument("--previous-run", action="append", default=None)
    g.add_argument("--decision-record", action="append", default=None)
    g.add_argument("--rehearsal", action="store_true")
    c = sub.add_parser("check-run-meta", allow_abbrev=False)
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument("--gate-record", default=None)
    c.add_argument("--allow-smoke", action="store_true")
    c.add_argument("--stage", choices=sorted(STAGE_NAMES), default=None)
    c.add_argument("--seed", type=int, choices=(42, 43, 44), default=None)
    c.add_argument("--profile", choices=sorted(PROFILES), default="distill_80k")
    c.add_argument("--expect-head", default=None)
    c.add_argument("--lambda-logit", type=float, default=None)
    c.add_argument("--alpha", type=float, default=None)
    pe = sub.add_parser("push-evidence", allow_abbrev=False)
    pe.add_argument("--response", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--remote", default="origin")
    sub.add_parser("_fingerprint", allow_abbrev=False)
    t = sub.add_parser("_teacher-identity", allow_abbrev=False)
    t.add_argument("--teacher-ckpt", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "_fingerprint":
        print("FINGERPRINT_JSON:" + json.dumps(fingerprint(), sort_keys=True), flush=True)
        return 0
    if args.cmd == "_teacher-identity":
        print("TEACHER_JSON:" + json.dumps(teacher_identity_child(args.teacher_ckpt), sort_keys=True, default=str),
              flush=True)
        return 0
    if args.cmd == "check-run-meta":
        if args.allow_smoke == (args.gate_record is not None) or (args.allow_smoke and (
                args.stage is None or args.seed is None or args.expect_head is None)):
            print("usage error: check-run-meta takes --gate-record R, or --allow-smoke with --stage, --seed and "
                  "--expect-head", file=sys.stderr)
            return 2
    try:
        _lib()
    except ImportError as e:
        print(f"usage error: run with PYTHONPATH={REPO} ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    try:
        if args.cmd == "gate":
            return run_gate(args)
        if args.cmd == "check-run-meta":
            return check_run_meta(args)
        return {"entry": cmd_entry, "records": cmd_records, "push-evidence": cmd_push_evidence}[args.cmd](args)
    except GateError as e:
        print(f"REFUSED [{e.code}]: {e}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code})")
        return 2
    except OSError as e:                # a file that cannot be read or written is a refusal, never a FAIL (exit 1)
        print(f"REFUSED [io_error]: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: REFUSED (io_error)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
