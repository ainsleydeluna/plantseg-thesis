#!/usr/bin/env python3
"""KD arm launch gate for AM-18 item 3's arms (lane L-TEACHER-ARMS, commit C2; ARMS plan decision (v), audit F7,
F23, F25, F26, F34, F39, F40; Phase B-2 rulings R3, R4(a), R5, R7 to R10).

Usage (every command runs with PYTHONPATH=<checkout>, as scripts/preflight_distill.py):
  python -B scripts/preflight_arms.py entry --arm NAME --lambda-logit L --alpha A --ckpt-dir D --code-pin PIN
         --arms-pin-dl DL-n --clock-offset-seconds O [--am8a-report REL] [--previous-run DIR ...]
      The arm's launch line, from the committed launch log at HEAD of this clone, after the would-be log (the log
      plus the line) parses with sweep_select.LaunchLog. The attempt is 1 + the earlier launch lines of the same
      run stage, seed, horizon, lambda and alpha. Prints the launch note: the arm's cut-order place and this
      machine's clock offset (AM-19a reading 21; |O| <= 60 s).
  python -B scripts/preflight_arms.py records --records-commit H --out F
      preflight_distill's records folder of H, plus every reports/arms/*.json at H (the arm teacher records and
      the go-rule records).
  python -B scripts/preflight_arms.py gate --arm NAME --lambda-logit L --alpha A --ckpt-dir D --evidence E
         --expect-head PIN --records-commit H --records F --teacher-ckpt T --clock-offset-seconds O --record R
         --approval-dl DL-n --arms-pin-dl DL-n [--go-rule-dl DL-n] [--reference-telemetry PATH]
         [--reference-smoke PATH ...] [--attempt K] [--am8a-report REL] [--previous-run DIR ...]
         [--decision-record REL ...] [--rehearsal]
      The gate; GO prints the launch block after writing the record R (format kd_arm_gate_record/1).
  python -B scripts/preflight_arms.py check-run-meta --ckpt-dir D --gate-record R
  python -B scripts/preflight_arms.py check-run-meta --ckpt-dir D --allow-smoke --arm NAME --expect-head PIN
         --lambda-logit L --alpha A [--teacher-sha256 HEX]
      The arm run's run_meta row against one rule per key; prints the launched line on PASS and on FAIL.
  python -B scripts/preflight_arms.py not-launched --arm NAME --run-id RID --reason TEXT --evidence REL
         [--run-dir DIR]
      The not_launched line of an arm launch line that has no launched line (AM-19a reading 6: a start that wrote
      no run_meta row with mode real, or a launch the gate refused after its line was pushed), from the committed
      launch log at HEAD of this clone; REL is the committed evidence.
  python -B scripts/preflight_arms.py stopped --arm NAME --run-id RID --report REL [--run-dir DIR]
      The stopped line of a launched arm run that ended without a trainer record (AM-19a reading 6; audit F34), from
      the committed launch log at HEAD of this clone; REL is the committed stop report.
Exit: 0 GO / PASS / written; 1 NO-GO / FAIL (a rehearsal ALWAYS exits 1); 2 usage or refused.

The arm gate extends scripts/preflight_distill.py ("pd"), imported unmodified. pd's 18 stages run in pd's order:
the same stage objects where a stage does not depend on the arm, an arm version where it does (arguments,
module_provenance, records, launch_order, teacher_identity, kd_dry_run), and one new stage, arm_cut_arithmetic,
before repo_unchanged:
  arguments          E1's arguments stage, then the arm's own: a registered arm (configs/kd_arms/), lambda and
                     alpha, the run_id <arm>_s<seed>_lambda<L>_alpha<A>_<80>k_a<n> = basename(--ckpt-dir), pd's
                     evidence, records and clock-offset rules, the teacher file outside the repository (the
                     teacher of record's name for a record-teacher arm), the decision-log ids, --go-rule-dl exactly
                     for an arm teacher
  records            pd's records stage (records commit H holds the pin; the records folder by blob id; AM-7a
                     item 5 on record; the schedule, rules and launch log), then the arm records, every row read
                     from H's decision log (the records folder, verified by blob id against H):
                       approval   --approval-dl: RECORDED, carrying "AM-18 written approval" (AM-20 item 3)
                       arms pin   --arms-pin-dl: RECORDED, carrying the literal "AM-20 item 3: open" (R5; the Tier F
                                  re-check of DL-101) and exactly one "P_K = <40 hex>", which equals HEAD, the pin
                                  (AM-18 item 7(a); DL-89 mirrored for the arm family)
                       teacher    an arm teacher only (teacher "arm"): reports/arms/arm_teacher_record_<X>.json at H,
                                  arm_teacher_record/1, arm X, status complete, its selected checkpoint the
                                  --teacher-ckpt file by name (and by sha256 and size when the file is here), never
                                  the teacher of record (IC:156; DL-30)
                       go rule    an arm teacher only: --go-rule-dl RECORDED, carrying the sha256 of
                                  reports/arms/arm_go_rule_<X>.json and of the teacher record; that file
                                  arm_go_rule/1, arm X, verdict go, for the record's checkpoint (AM-18 item 5)
                     A record-teacher arm (teacher "record") has no teacher record and no go-rule row; it still
                     needs the approval and arms-pin rows, and its teacher is the teacher of record (DL-88).
  selection_inputs   pd's own stage, as for an E3 launch outside the sweeps: lambda from lambda_selection.json and
                     alpha from alpha_selection.json, or from the committed alpha decision record when it cuts the
                     alpha sweep (pd's PL-31 path; audit F25, ruling R8); each must equal --lambda-logit / --alpha
                     (AM-18 item 3: lambda and alpha are inherited from E3 seed 42's selections)
  launch_order       the arm's launch line is in H's launch log byte for byte; the previous attempt is resolved; a
                     repeat after a launched attempt names its committed AM-8a report (pd's rule); a repeat after a
                     launched attempt also needs that attempt's --previous-run directory (bound to its launched line
                     by its run_meta sha256), and is refused when its telemetry ends in a valid AM-7a divergence
                     record (ruling R4(a); AM-18 item 3(e); DL-04: no rerun) [arm_rerun_after_divergence]; every
                     arm launch line carries the code pin P_K and every arm launched line ran at it
  teacher_identity   a CPU child loads the teacher strictly with the record config and the expected sha256 (the
                     arm teacher's, from its record; the teacher of record's for a record-teacher arm): 27,618,868
                     parameters, none trainable, eval mode, 854 state entries, DL-88's provenance values except
                     the checkpoint fields, which are the teacher file's (plan decision (iv))
  kd_dry_run         pd's dry run with --arm NAME and the arm's teacher sha256
  arm_cut_arithmetic AM-19 item 3(b): the arm's iterations times the larger of the named references' median
                     seconds per iteration, against the time left to the original freeze date of the schedule in
                     force (AM-19 item 1(c)); no reference: "no measurement: not cut" (rulings R7, F40)
The first FAIL stops the gate; a rehearsal (--rehearsal) never prints a launch block and SKIPs only what needs the
pod or the teacher file, so a PASS of every other stage covers every non-GPU gate.

Nothing here is specific to one arm: every rule reads the arm's registry entry (kd_arms.ArmSpec: its run stage,
teacher kind, arm teacher, go rule, K, views and cut place), so an arm registered later runs through the same
code (ruling R10).
"""
import sys

sys.dont_write_bytecode = True                  # before the first non-stdlib import (as pd)

import os  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"      # children inherit it

import argparse          # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import platform          # noqa: E402
import re                # noqa: E402
import shlex             # noqa: E402
import shutil            # noqa: E402
import statistics        # noqa: E402
import time              # noqa: E402
from pathlib import Path, PurePosixPath  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.preflight_distill as pd  # noqa: E402

GateError = pd.GateError
E1G = pd.E1G
_ok, _skip, _under, sha256_file = pd._ok, pd._skip, pd._under, pd.sha256_file

# ------------------------------------------------------------------------------------------ values of record
ARM_DIR = None                     # the registry folder (None: configs/kd_arms/); a smoke points it at a scratch copy
PROFILE = "distill_80k"            # AM-18 item 3(a): 80,000 iterations; the arm's horizon is checked against it
GATE_RECORD_FORMAT = "kd_arm_gate_record/1"    # pd's kd_gate_record/1 keys plus the arm's; pd refuses this format
ARMS_REL = "reports/arms"
TEACHER_RECORD_REL = "reports/arms/arm_teacher_record_{arm}.json"   # written by scripts/arm_teacher_record.py
GO_RULE_REL = "reports/arms/arm_go_rule_{arm}.json"                 # written by scripts/arm_go_rule.py
TEACHER_RECORD_FORMAT = "arm_teacher_record/1"
GO_RULE_FORMAT = "arm_go_rule/1"
APPROVAL_MARKER = "AM-18 written approval"     # DL-101's text (AM-20 item 3)
RECHECK_TOKEN = "AM-20 item 3: open"           # the Tier F re-check of DL-101's reading (ruling R5; audit part 4)
PIN_RE = re.compile(r"P_K\s*[=:]\s*([0-9a-f]{40})(?![0-9a-f])")
DL_ID_RE = re.compile(r"DL-\d+[a-z]?")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# AM-19 item 1(a): the TEST manifest freeze of each schedule; item 1(c): in every cut rule "the freeze" is the
# original freeze date of the schedule in force, at the end of that date (sweep_select.cutoff_instant).
FREEZE_DATES = {"T": "2026-11-03", "R": "2026-12-18"}
E3_SMOKE_FORMAT = "kd_smoke/1"                 # scripts/smoke_kd_step.py's JSON: throughput["e3"] is E3's figure
E3_SMOKE_KEY = "e3"
TELEMETRY_FIELD = "iter_seconds"               # the trainer's train rows (train_distill.py write_train_row)
VERDICTS = ("go", "no_go", "not_evaluable")
STAGE_NAMES_ORDER = (*pd.STAGE_NAMES_ORDER[:-1], "arm_cut_arithmetic", pd.STAGE_NAMES_ORDER[-1])
# The two rules a smoke mutates by name (the median of a reference, the larger of the references)
MEDIAN = statistics.median
APPLIED = max


# ---------------------------------------------------------------------------------------------- the arm
def arm_spec(name: str):
    """The registered arm `name` (kd_arms.load_arms; the builder is resolved by the trainer, not here)."""
    from src.training import kd_arms
    if not _under(Path(kd_arms.__file__), REPO):
        raise GateError("import_origin_foreign", f"kd_arms loaded from {kd_arms.__file__}, outside {REPO}")
    try:
        arms = kd_arms.load_arms(ARM_DIR)
    except kd_arms.ArmSpecError as e:
        raise GateError("arm_spec", str(e)) from e
    if name not in arms:
        raise GateError("arm_spec", f"--arm {name!r} is not a registered arm; registered: {sorted(arms) or 'none'}")
    spec = arms[name]
    if spec.horizon != pd.PROFILES[PROFILE]["horizon"] or spec.stage not in pd.STAGE_NAMES:
        raise GateError("arm_spec", f"{spec.name}: stage {spec.stage!r} and horizon {spec.horizon} are not a KD "
                                    f"launch of this gate ({PROFILE})")
    return spec


def arm_run_stages() -> set:
    """Every registered arm's run stage: the launch lines that belong to the arm family."""
    from src.training import kd_arms
    try:
        return {s.run_stage for s in kd_arms.load_arms(ARM_DIR).values()}
    except kd_arms.ArmSpecError as e:
        raise GateError("arm_spec", str(e)) from e


def arm_fields(spec) -> dict:
    """The arm's registry entry as the gate record and the launch note name it."""
    return {"name": spec.name, "run_stage": spec.run_stage, "e7_stage": spec.e7_stage, "stage": spec.stage,
            "seed": spec.seed, "horizon": spec.horizon, "teacher": spec.teacher, "arm_teacher": spec.arm_teacher,
            "go_rule": spec.go_rule, "target_construction": spec.target_construction, "cut_place": spec.cut_place,
            "registry_file": Path(spec.path).name, "registry_sha256": spec.sha256}


def stage_key(spec) -> str:
    """The trainer's stage name of the arm's runs (run_meta's stage; the arm's own name is in arm)."""
    return pd.STAGE_NAMES[spec.stage]


def run_id_of(spec, lam, alpha, attempt: int) -> str:
    """<arm>_s<seed>_lambda<L>_alpha<A>_<80>k_a<n>: exactly one alpha token (the trainer's ckpt_dir_alpha_error)."""
    return "_".join([spec.name, f"s{spec.seed}", f"lambda{pd.fmt_value(lam)}", f"alpha{pd.fmt_value(alpha)}",
                     f"{spec.horizon // 1000}k", f"a{attempt}"])


def arm_launch_line(ss, spec, *, lam, alpha, attempt, schedule, code_pin, ckpt_dir, am8a_report) -> dict:
    """The arm's launch line (PL-13) in LAUNCH_KEYS order: its run stage, seed and horizon, outside the sweeps
    (sweep and value null), so sweep_select never counts it as a lambda or alpha candidate."""
    line = {"format": ss.LAUNCH_LOG_FORMAT, "event": "launch", "run_id": run_id_of(spec, lam, alpha, attempt),
            "stage": spec.run_stage, "seed": spec.seed, "horizon": spec.horizon, "sweep": None, "value": None,
            "lambda_logit": float(lam), "alpha_cwd": float(alpha), "attempt": attempt, "schedule": schedule,
            "code_pin": code_pin, "ckpt_dir": str(ckpt_dir), "am8a_report": am8a_report}
    assert tuple(line) == ss.LAUNCH_KEYS
    return line


def lam_alpha_error(spec, lam, alpha) -> str | None:
    """The arm's stage takes lambda and alpha explicitly (E3 instantiates Logit KD and the feature-map term)."""
    lk, cf, _ = pd.TERMS[spec.stage]
    for flag, value, needed in (("--lambda-logit", lam, lk), ("--alpha", alpha, cf)):
        if needed and (value is None or not math.isfinite(float(value))):
            return f"{flag} is required for {spec.run_stage} (a finite value; E3's recipe of record)"
        if not needed and value is not None:
            return f"{flag} is refused for {spec.run_stage}"
    return None


def clock_offset_error(off) -> str | None:
    if not (isinstance(off, float) and math.isfinite(off)) or abs(off) > pd.CLOCK_OFFSET_MAX:
        return (f"--clock-offset-seconds {off!r}: this machine's offset from a network time source must be within "
                f"{pd.CLOCK_OFFSET_MAX:g} s (AM-19a reading 21: more is a STOP)")
    return None


# ------------------------------------------------------------------------------------------ decision-log rows
def dl_row(dl_text: str, dl_id: str) -> str | None:
    """The decision log's table row whose first cell is `dl_id`, or None."""
    for ln in dl_text.splitlines():
        if not ln.lstrip().startswith("|"):
            continue
        cells = ln.strip().strip("|").split("|")
        if cells and cells[0].strip() == dl_id:
            return ln
    return None


def row_status(row: str) -> str:
    """The status word that opens a row's second cell (RECORDED, DECIDED, ...)."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    words = cells[1].split() if len(cells) > 1 else []
    return words[0] if words else ""


def recorded_row(dl_text: str, dl_id, *, what: str, code: str, need=()) -> str:
    """A committed decision-log row with status RECORDED that contains every string in `need`."""
    if not isinstance(dl_id, str) or not DL_ID_RE.fullmatch(dl_id):
        raise GateError(code, f"{what}: {dl_id!r} is not a decision-log id DL-<n>")
    row = dl_row(dl_text, dl_id)
    if row is None:
        raise GateError(code, f"{what}: {dl_id} is not a row of the committed decision log")
    if row_status(row) != "RECORDED":
        raise GateError(code, f"{what}: {dl_id} has status {row_status(row)!r}, not RECORDED")
    missing = [n for n in need if n not in row]
    if missing:
        raise GateError(code, f"{what}: {dl_id} does not contain {missing}")
    return row


def approval_row(dl_text: str, dl_id) -> dict:
    """AM-20 item 3: the written approval of AM-18 is in the decision log (DL-101)."""
    recorded_row(dl_text, dl_id, what="--approval-dl (AM-20 item 3: the written approval of AM-18)",
                 code="approval_missing", need=(APPROVAL_MARKER,))
    return {"dl": dl_id, "marker": APPROVAL_MARKER}


def arms_pin_of(dl_text: str, dl_id) -> dict:
    """The arms-pin row (AM-18 item 7(a)): RECORDED, the re-check token (ruling R5) and exactly one P_K."""
    row = recorded_row(dl_text, dl_id, what="--arms-pin-dl (AM-18 item 7(a): the arms pin P_K)",
                       code="arms_pin_row_invalid")
    if RECHECK_TOKEN not in row:
        raise GateError("approval_recheck_missing", f"--arms-pin-dl {dl_id} does not contain the literal "
                                                    f"{RECHECK_TOKEN!r}: the Tier F re-check of DL-101's reading "
                                                    "opens the arms (AM-20 item 3; ruling R5)")
    pins = sorted(set(PIN_RE.findall(row)))
    if len(pins) != 1:
        raise GateError("arms_pin_row_invalid", f"--arms-pin-dl {dl_id} names {len(pins)} pins as 'P_K = <40 hex>'; "
                                                "it names exactly one")
    return {"dl": dl_id, "pin": pins[0], "token": RECHECK_TOKEN}


def pin_problem(arms_pin: dict, pin: str) -> str | None:
    """AM-18 item 7(a): the checkout's HEAD (the pin) is P_K, the arms pin the row names."""
    if arms_pin["pin"] != pin:
        return f"HEAD (the pin) {pin} is not P_K {arms_pin['pin']} named in {arms_pin['dl']} (AM-18 item 7(a))"
    return None


def arm_pins_problem(log, pin: str, run_stages) -> str | None:
    """DL-89 mirrored for the arm family (AM-18 item 7(a): the code of all item 3 arms is pinned in one commit):
    every arm launch line carries the code pin P_K and every arm launched line ran at it."""
    pins = sorted({log.launch[r]["code_pin"] for r in log.order if log.launch[r]["stage"] in run_stages})
    heads = sorted({log.launched[r]["git_head"] for r in log.launched if log.launch[r]["stage"] in run_stages})
    bad = [p for p in dict.fromkeys(pins + heads) if p != pin]
    if bad:
        return (f"the arm launch lines carry the code pins {pins} and the arm launched lines ran at {heads}; the "
                f"arms pin P_K is {pin} (AM-18 item 7(a); DL-89 for the arm family)")
    return None


# ------------------------------------------------------------------------------------------ teacher records
def is_record_teacher(sha256) -> bool:
    """The teacher of record's checkpoint (DL-88): never an arm teacher (IC:156; DL-30)."""
    return sha256 == pd.TEACHER_SHA256


def record_teacher_rule(spec, name, sha256=None) -> str | None:
    """A record-teacher arm (teacher "record", ruling R10) distils from the teacher of record only: its file name
    (E-31) and, once known, its sha256 (DL-88). None for an arm-teacher arm, and when the rule holds."""
    if spec.teacher != "record":
        return None
    if name != pd.TEACHER_FILE or (sha256 is not None and not is_record_teacher(sha256)):
        return (f"{spec.run_stage} distils from the teacher of record ({pd.TEACHER_FILE}, sha256 "
                f"{pd.TEACHER_SHA256[:12]}…), not {name} {'' if sha256 is None else sha256[:12] + '…'}".rstrip())
    return None


def read_json_record(ss, src, rel: str, *, missing: str, fmt: str) -> dict:
    """A committed reports/arms record at the records commit: (path, sha256, doc), its format checked."""
    rel, data = src.read(rel, missing=missing, uncommitted=missing, record=True)
    doc = ss.parse_json_bytes(data, rel, code=missing)
    if not isinstance(doc, dict) or doc.get("format") != fmt:
        raise GateError(missing, f"{rel} is not a {fmt} record")
    return {"path": rel, "sha256": ss.sha256_bytes(data), "doc": doc}


def teacher_record_check(ss, src, spec, teacher_ckpt: str) -> dict:
    """Plan decision (iv) checks 1 and 2: the arm teacher's record is complete, of this arm teacher, and its
    selected checkpoint is the --teacher-ckpt file (by name; by sha256 and size when the file is here)."""
    rec = read_json_record(ss, src, TEACHER_RECORD_REL.format(arm=spec.arm_teacher), missing="teacher_record_missing",
                           fmt=TEACHER_RECORD_FORMAT)
    doc = rec["doc"]
    ck = (doc.get("selection") or {}).get("checkpoint") if isinstance(doc.get("selection"), dict) else None
    if doc.get("arm") != spec.arm_teacher:
        raise GateError("teacher_record_invalid", f"{rec['path']}: arm {doc.get('arm')!r}, the arm teacher of "
                                                  f"{spec.run_stage} is {spec.arm_teacher}")
    if doc.get("status") != "complete":
        raise GateError("teacher_record_invalid", f"{rec['path']}: status {doc.get('status')!r}: E3-X distils from a "
                                                  "teacher whose fine-tune completed (AM-18 items 2 and 7(a))")
    if not (isinstance(ck, dict) and isinstance(ck.get("name"), str) and ck["name"]
            and SHA256_RE.fullmatch(str(ck.get("sha256"))) and isinstance(ck.get("bytes"), int)
            and not isinstance(ck.get("bytes"), bool)):
        raise GateError("teacher_record_invalid", f"{rec['path']}: no selection.checkpoint {{name, sha256, bytes}}")
    if is_record_teacher(ck["sha256"]):
        raise GateError("teacher_is_record", f"{rec['path']} selects the teacher of record's checkpoint: an arm "
                                             "teacher is never the teacher of record (IC:156; DL-30)")
    t = Path(teacher_ckpt)
    if t.name != ck["name"]:
        raise GateError("teacher_ckpt_mismatch", f"--teacher-ckpt {teacher_ckpt}: the record selects {ck['name']}")
    if t.is_file():
        got = {"sha256": sha256_file(t), "bytes": t.stat().st_size}
        if got != {"sha256": ck["sha256"], "bytes": ck["bytes"]}:
            raise GateError("teacher_ckpt_mismatch", f"--teacher-ckpt {teacher_ckpt}: sha256 {got['sha256']} and "
                                                     f"{got['bytes']} bytes, the record selects {ck['sha256']} and "
                                                     f"{ck['bytes']} bytes (plan decision (iv), check 2)")
    return {"path": rec["path"], "sha256": rec["sha256"], "checkpoint": {k: ck[k] for k in ("name", "sha256", "bytes")}}


def go_rule_check(ss, src, dl_text: str, spec, dl_id, teacher_record: dict) -> dict:
    """AM-18 item 5, as the gate reads it: the go-rule row (RECORDED, carrying the go-rule record's sha256 and the
    teacher record's) and the go-rule record (arm X, verdict go, for the teacher record's checkpoint)."""
    rec = read_json_record(ss, src, GO_RULE_REL.format(arm=spec.arm_teacher), missing="go_rule_record_missing",
                           fmt=GO_RULE_FORMAT)
    recorded_row(dl_text, dl_id, what="--go-rule-dl (AM-18 item 5: the outcome, with the artifacts' sha256)",
                 code="go_rule_record_missing", need=(rec["sha256"], teacher_record["sha256"]))
    doc = rec["doc"]
    if doc.get("arm") != spec.arm_teacher or doc.get("verdict") not in VERDICTS:
        raise GateError("go_rule_record_invalid", f"{rec['path']}: arm {doc.get('arm')!r} verdict "
                                                  f"{doc.get('verdict')!r}")
    if doc.get("verdict") != "go":
        raise GateError("go_rule_not_go", f"{rec['path']}: verdict {doc['verdict']}: E3-{spec.arm_teacher} does not "
                                          "launch (AM-18 item 5)")
    ck = (doc.get("checkpoint") or {}).get("sha256") if isinstance(doc.get("checkpoint"), dict) else None
    tr = (doc.get("teacher_record") or {}).get("sha256") if isinstance(doc.get("teacher_record"), dict) else None
    if ck != teacher_record["checkpoint"]["sha256"] or tr != teacher_record["sha256"]:
        raise GateError("go_rule_record_invalid", f"{rec['path']}: checkpoint {ck} and teacher record {tr}; the gate's "
                                                  f"are {teacher_record['checkpoint']['sha256']} and "
                                                  f"{teacher_record['sha256']}")
    return {"dl": dl_id, "path": rec["path"], "sha256": rec["sha256"], "verdict": doc["verdict"],
            "checkpoint_sha256": ck}


def teacher_expectations(spec, teacher_record: dict | None) -> dict:
    """The teacher the run distils from: the arm teacher's selected checkpoint, or the teacher of record."""
    if spec.teacher == "arm":
        return dict(teacher_record["checkpoint"])
    return {"name": pd.TEACHER_FILE, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES}


def teacher_identity_child(ckpt: str, expect_sha256: str) -> dict:
    """The teacher, loaded on the CPU in a child process with the record config and the expected sha256 (pd's child,
    with the arm teacher's sha256 in place of the teacher of record's)."""
    rec = {"ok": False}
    try:
        _, td = pd._lib()
        import torch
        rec["sha256"] = sha256_file(ckpt)
        rec["bytes"] = os.path.getsize(ckpt)
        frozen = td.load_frozen_teacher(ckpt, config_path=str(td.DEFAULT_TEACHER_CONFIG), expected_sha256=expect_sha256)
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


def teacher_identity_problems(rec: dict, ckpt: str, spec, exp: dict) -> list:
    """The differences of the child's record from the teacher the arm names; [] when it matches (plan decision
    (iv), check 3: DL-88's values except the checkpoint fields, which are the teacher file's)."""
    if not rec.get("ok"):
        return [f"the teacher did not load: {rec.get('error')}"]
    prov = rec.get("provenance") or {}
    bad = []
    if spec.teacher == "arm" and (is_record_teacher(rec.get("sha256")) or is_record_teacher(exp["sha256"])):
        bad.append("the arm teacher is the teacher of record (IC:156; DL-30)")
    rule = record_teacher_rule(spec, Path(ckpt).name, rec.get("sha256"))
    if rule is not None:
        bad.append(rule)
    want = {"sha256": exp["sha256"], "bytes": exp["bytes"], "params": pd.TEACHER_PARAMS, "trainable": 0,
            "training_modules": [], "state_entries": pd.TEACHER_STATE_ENTRIES}
    bad += [f"{k}: {rec.get(k)!r} != {v!r}" for k, v in want.items() if rec.get(k) != v]
    pwant = {"builder": pd.TEACHER_BUILDER, "ckpt_sha256": exp["sha256"], "expected_sha256": exp["sha256"],
             "ckpt_bytes": exp["bytes"], "ckpt_path": str(ckpt), **pd.TEACHER_VALUES}
    bad += [f"provenance {k}: {prov.get(k)!r} != {v!r}" for k, v in pwant.items() if prov.get(k) != v]
    cfg = prov.get("config_path")
    if not (isinstance(cfg, str) and _under(Path(cfg), REPO)):
        bad.append(f"provenance config_path {cfg!r} is not under the checkout")
    return bad


# --------------------------------------------------------------------------------------- previous attempts
def divergence_view(spec) -> dict:
    """The run shape sweep_select.read_run reads a run directory by: the trainer's stage name, seed, iterations."""
    return {"stage": stage_key(spec), "seed": spec.seed, "iterations": spec.horizon}


def is_divergence(run) -> bool:
    """The run's telemetry ends in a valid AM-7a divergence record: the trainer's run_abort row for a student
    divergence under AM-7 (a) or (b), validated by sweep_select's abort-ending reader (ss._abort_ending)."""
    return isinstance(run.ending, dict) and run.ending.get("kind") == "diverged"


def check_previous_runs(ss, log, earlier: list, spec, previous_runs) -> list:
    """Ruling R4(a): every launched earlier attempt of the launch group is given by --previous-run (bound to its
    launched line by its run_meta sha256) and read with sweep_select.read_run; a repeat after an attempt whose
    telemetry ends in a valid AM-7a divergence record is refused (AM-18 item 3(e): AM-7a item 3's arm rule; DL-04)."""
    launched = [rid for rid in earlier if rid in log.launched]
    given = {}
    for d in previous_runs:
        p = Path(d)
        if p.name not in launched or p.name in given:
            raise GateError("previous_run_invalid", f"--previous-run {d}: not a launched earlier attempt of this "
                                                    "launch group, or named twice")
        meta_p = p / f"{spec.stage}_run_meta.jsonl"
        if not meta_p.is_file() or ss.sha256_file(meta_p) != log.launched[p.name]["run_meta_sha256"]:
            raise GateError("previous_run_invalid", f"--previous-run {d}: its run_meta file does not hash to the "
                                                    "launched line's run_meta_sha256")
        given[p.name] = ss.read_run(p, divergence_view(spec), "alpha_cwd")
    missing = [rid for rid in launched if rid not in given]
    if missing:
        raise GateError("previous_run_missing", f"the launched earlier attempts {missing} need their --previous-run "
                                                "directories: a repeat is refused after a divergence (ruling R4(a))")
    out = []
    for rid in launched:
        run = given[rid]
        fault = run.tel_fault or run.end_fault
        if fault is not None:
            raise GateError("previous_run_invalid", f"{rid}: its telemetry cannot be read or its end record is "
                                                    f"invalid ({fault}): STOP and investigate")
        if is_divergence(run):
            ab = run.ending.get("abort") or {}
            raise GateError("arm_rerun_after_divergence",
                            f"{rid} ended in a student divergence (AM-7a: rule {ab.get('rule')}, iteration "
                            f"{ab.get('iter')}): the arm is recorded as diverged and is never rerun (AM-18 item 3(e); "
                            "DL-04; ruling R4(a))")
        out.append({"run_id": rid, "ending": None if run.ending is None else run.ending.get("kind")})
    return out


# ------------------------------------------------------------------------------------------- arithmetic
def pod_smoke_format(spec) -> str:
    """The format of the arm's own KD pod smoke (lane rulings K6): kd_<suffix>_pod_smoke/1, whose throughput holds
    the arm's name as its stage key."""
    return f"kd_{spec.run_stage[len('E3-'):].lower()}_pod_smoke/1"


def smoke_figure(path, key: str) -> float:
    """One reader for every reference smoke (ruling R7): throughput[<key>]["iter_seconds"]["median"]."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    v = (((doc.get("throughput") or {}).get(key) or {}).get("iter_seconds") or {}).get("median")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise GateError("reference_unusable", f"{path}: throughput[{key!r}]['iter_seconds']['median'] is {v!r}")
    return float(v)


def reference_smoke(path, spec) -> dict:
    """A KD pod smoke's JSON: smoke_kd_step's kd_smoke/1 gives E3's figure (throughput["e3"]); the arm's own pod
    smoke (pod_smoke_format) gives the arm's (throughput[<arm name>])."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GateError("reference_unusable", f"--reference-smoke {path}: unreadable ({type(e).__name__}: {e})") from e
    fmt = doc.get("format") if isinstance(doc, dict) else None
    key = {E3_SMOKE_FORMAT: E3_SMOKE_KEY, pod_smoke_format(spec): spec.name}.get(fmt)
    if key is None:
        raise GateError("reference_format", f"--reference-smoke {path}: format {fmt!r}; {spec.run_stage} reads "
                                            f"{E3_SMOKE_FORMAT!r} (E3's figure) or {pod_smoke_format(spec)!r} (its own)")
    return {"source": "smoke", "path": str(p), "sha256": sha256_file(p), "format": fmt, "key": key,
            "field": f"throughput[{key!r}]['iter_seconds']['median']", "median": smoke_figure(p, key)}


def reference_telemetry(path, spec) -> dict:
    """The plan's definition (AM-19 item 3(b): the most recently finished run of its stage): an <stage>_telemetry.jsonl
    whose last row is the trainer's run_end; the figure is the median of every train row's iter_seconds."""
    p = Path(path)
    want = f"{spec.stage}_telemetry.jsonl"
    if p.name != want:
        raise GateError("reference_unusable", f"--reference-telemetry {path}: a finished run's {want}")
    rows = []
    try:
        for n, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if ln.strip():
                rows.append(json.loads(ln))
    except (OSError, ValueError) as e:
        raise GateError("reference_unusable", f"--reference-telemetry {path}: unreadable ({type(e).__name__}: {e})") from e
    if not rows or not isinstance(rows[-1], dict) or rows[-1].get("event") != "run_end":
        raise GateError("reference_unusable", f"--reference-telemetry {path}: its last row is not run_end (not a "
                                              "finished run)")
    vals = [r[TELEMETRY_FIELD] for r in rows if isinstance(r, dict) and r.get("event") == "train"
            and isinstance(r.get(TELEMETRY_FIELD), (int, float)) and not isinstance(r.get(TELEMETRY_FIELD), bool)
            and math.isfinite(r[TELEMETRY_FIELD]) and r[TELEMETRY_FIELD] > 0]
    if not vals:
        raise GateError("reference_unusable", f"--reference-telemetry {path}: no train row with a finite iter_seconds")
    return {"source": "telemetry", "path": str(p), "sha256": sha256_file(p), "rows": len(vals),
            "field": f"every train row's {TELEMETRY_FIELD}", "median": float(MEDIAN(vals))}


def cut_arithmetic(ss, iterations: int, figures: list, schedule_in_force: str, now: float) -> dict:
    """AM-19 item 3(b) for one arm: iterations x the larger of the references' figures (F40; the stricter reading)
    against the time left to the original freeze date; no figure: no measurement, not cut."""
    day = FREEZE_DATES[schedule_in_force]
    freeze = float(ss.cutoff_instant(day))
    left = freeze - float(now)
    base = {"iterations": iterations, "freeze_date": day, "freeze_utc": ss.iso_utc(freeze), "now_utc": ss.iso_utc(now),
            "seconds_left": left, "references": figures}
    if not figures:
        return {**base, "measured": False, "applied_s_per_iter": None, "seconds_needed": None, "cut": False,
                "text": "no measurement: not cut (AM-19 item 3(b): a stage with no measurement is not cut before its "
                        "launch)"}
    applied = float(APPLIED(f["median"] for f in figures))
    needed = iterations * applied
    cut = needed > left
    text = (f"{iterations} iterations x {applied!r} s/iter (the larger of {len(figures)} reference(s): "
            + ", ".join(f"{f['source']} {f['median']!r}" for f in figures)
            + f") = {needed:.0f} s; time left to the original freeze date {day} (end: {ss.iso_utc(freeze)}) = "
            f"{left:.0f} s: " + ("it does not fit: the run is cut and is not launched (AM-19 item 3(b); for an "
                                 "AM-18 arm this entry is its not-launched record under AM-18 item 7(a))"
                                 if cut else "it fits: not cut"))
    return {**base, "measured": True, "applied_s_per_iter": applied, "seconds_needed": needed, "cut": cut, "text": text}


def launch_note(spec) -> str:
    return (f"launch note: {spec.run_stage}: an AM-18 item 3 arm, descriptive, never a parent of a run of record; "
            f"cut-order place {spec.cut_place} (AM-18 item 7(b)); cuttable runs launch in the reverse of the cut "
            "order (AM-19 item 1(e)); outside the sweeps (no decision date, no launch-order position)")


# ------------------------------------------------------------------------------------------------- stages
def stage_arguments(ctx):
    """E1's arguments stage (the data root, PYTHONPATH = the checkout, a fresh --ckpt-dir outside the repository and
    the data root, the record path), then the arm's launch arguments (see the module docstring)."""
    status, detail, rec = E1G.stage_arguments(ctx)
    spec = ctx["spec"] = arm_spec(ctx["arm"])
    error = lam_alpha_error(spec, ctx["lambda_logit"], ctx["alpha"])
    if error is not None:
        raise GateError("arm_arguments", error)
    if not (isinstance(ctx["attempt"], int) and not isinstance(ctx["attempt"], bool) and ctx["attempt"] >= 1):
        raise GateError("attempt", f"--attempt {ctx['attempt']!r}: a whole number >= 1")
    rid = run_id_of(spec, ctx["lambda_logit"], ctx["alpha"], ctx["attempt"])
    ck, root = Path(ctx["ckpt_dir"]), ctx["root"]
    if ck.name != rid:
        raise GateError("run_id_mismatch", f"basename(--ckpt-dir) {ck.name!r} != the run_id {rid!r} of this launch")
    ev = ctx["evidence"]
    if not os.path.isabs(ev):
        raise GateError("evidence_relative", f"--evidence must be absolute: {ev!r}")
    ev = Path(ev)
    if not ev.is_dir():
        raise GateError("evidence_missing", f"--evidence {ev} is not an existing directory")
    if _under(ev, REPO) or _under(ev, ck) or _under(ev, root) or _under(ck, ev):
        raise GateError("evidence_misplaced", f"--evidence {ev} overlaps the repository, the ckpt dir or the data root")
    present = [n for n in (f"{rid}.stdout.log", f"{rid}.pid") if (ev / n).exists()]
    if present:
        raise GateError("evidence_exists", f"{present} already exist in {ev}: a launch writes them once")
    folder = ctx["records"]
    if not os.path.isabs(folder) or _under(Path(folder), REPO) or not Path(folder).is_dir():
        raise GateError("records_folder", f"--records {folder!r} must be an existing absolute folder outside the "
                                          "repository (PL-28)")
    t = ctx["teacher_ckpt"]
    if not os.path.isabs(t) or _under(Path(t), REPO):
        raise GateError("teacher_ckpt_path", f"--teacher-ckpt {t!r} must be an absolute path outside the repository")
    rule = record_teacher_rule(spec, Path(t).name)
    if rule is not None:
        raise GateError("teacher_ckpt_path", f"--teacher-ckpt {t}: {rule} (E-31)")
    error = clock_offset_error(ctx["clock_offset"])
    if error is not None:
        raise GateError("clock_offset", error)
    for flag, value in (("--approval-dl", ctx["approval_dl"]), ("--arms-pin-dl", ctx["arms_pin_dl"])):
        if not (isinstance(value, str) and DL_ID_RE.fullmatch(value)):
            raise GateError("dl_id", f"{flag} {value!r} is not a decision-log id DL-<n>")
    if spec.go_rule and not (isinstance(ctx["go_rule_dl"], str) and DL_ID_RE.fullmatch(ctx["go_rule_dl"])):
        raise GateError("go_rule_record_missing", f"{spec.run_stage}: --go-rule-dl DL-<n> is required (AM-18 item 5)")
    if not spec.go_rule and ctx["go_rule_dl"] is not None:
        raise GateError("go_rule_not_applicable", f"{spec.run_stage} distils from the teacher of record: it has no go "
                                                  "rule, and --go-rule-dl is refused")
    ctx.update(run_id=rid, stage=spec.stage, seed=spec.seed, profile=PROFILE,
               ident={"sweep": None, "value": None, "horizon": spec.horizon})
    return _ok(f"{detail}; arm {spec.name} ({spec.run_stage}, teacher {spec.teacher}"
               + (f" {spec.arm_teacher}" if spec.arm_teacher else "") + f"); run_id {rid}; evidence {ev}; records "
               f"{folder}; clock offset {ctx['clock_offset']:g} s", **rec, run_id=rid, arm=arm_fields(spec),
               clock_offset_seconds=ctx["clock_offset"])


def stage_module_provenance(ctx):
    """pd's module provenance, with this gate and the arm registry."""
    status, detail, rec = pd.stage_module_provenance(ctx)
    from src.training import kd_arms
    foreign = E1G.module_provenance(extra=(("preflight_arms", __file__), ("kd_arms", kd_arms.__file__)))
    if foreign:
        raise GateError("import_origin_foreign", f"modules loaded from outside {REPO}: {foreign[:5]}")
    return status, f"{detail}; preflight_arms and kd_arms under {REPO}", rec


def arm_records(ctx) -> dict:
    """The arm records at H (see the module docstring's records stage)."""
    ss, _ = pd._lib()
    spec, src, dl_text = ctx["spec"], ctx["src"], ctx["dl_text"]
    approval = approval_row(dl_text, ctx["approval_dl"])
    arms_pin = arms_pin_of(dl_text, ctx["arms_pin_dl"])
    problem = pin_problem(arms_pin, ctx["pin"])
    if problem is not None:
        raise GateError("arms_pin_mismatch", problem)
    teacher_record = go_rule = None
    try:
        if spec.teacher == "arm":
            teacher_record = teacher_record_check(ss, src, spec, ctx["teacher_ckpt"])
            go_rule = go_rule_check(ss, src, dl_text, spec, ctx["go_rule_dl"], teacher_record)
    except ss.SelectionRefused as e:
        raise pd._refused(e) from e
    exp = teacher_expectations(spec, teacher_record)
    ctx.update(approval=approval, arms_pin=arms_pin, teacher_record=teacher_record, go_rule=go_rule, teacher_expect=exp)
    return {"approval": approval, "arms_pin": arms_pin, "teacher_record": teacher_record, "go_rule": go_rule,
            "teacher": exp}


def stage_records(ctx):
    """pd's records stage, then the arm records (approval, arms pin and token, teacher record, go rule)."""
    status, detail, rec = pd.stage_records(ctx)
    arm = arm_records(ctx)
    spec = ctx["spec"]
    note = (f"approval {arm['approval']['dl']}; P_K {arm['arms_pin']['pin'][:12]} ({arm['arms_pin']['dl']}, "
            f"re-check token); " + (f"{spec.arm_teacher} record {arm['teacher_record']['sha256'][:12]}…, go rule "
                                    f"{arm['go_rule']['dl']} go for {arm['teacher']['name']}"
                                    if spec.teacher == "arm" else "the teacher of record (no go rule)"))
    return status, f"{detail}; {note}", {**rec, "arm_records": arm}


def stage_launch_order(ctx):
    """PL-27 for the arm's line (see the module docstring's launch_order stage)."""
    ss, _ = pd._lib()
    src, log, spec = ctx["src"], ctx["log"], ctx["spec"]
    report = None
    try:
        if ctx["am8a_report"] is not None:
            rel, data = src.read(ctx["am8a_report"], missing="repeat_report_missing", uncommitted="repeat_report_mismatch",
                                 record=True)
            report = {"path": rel, "sha256": ss.sha256_bytes(data)}
        line = arm_launch_line(ss, spec, lam=ctx["lambda_logit"], alpha=ctx["alpha"], attempt=ctx["attempt"],
                               schedule=ctx["schedule"]["in_force"], code_pin=ctx["pin"], ckpt_dir=ctx["ckpt_dir"],
                               am8a_report=report)
        rid, raw = line["run_id"], pd.line_bytes(line)
        if rid not in log.launch:
            raise GateError("launch_line_missing", f"{rid}: no launch line in {ss.LAUNCH_LOG_REL} at the records commit "
                                                   "(print it with entry, commit and push it before the gate)")
        if log.raw[rid] != raw:
            raise GateError("launch_line_mismatch", f"{rid}: the launch log's line differs from the line built from the "
                                                    f"gate's arguments:\n  log:  {log.raw[rid].decode()}\n  gate: "
                                                    f"{raw.decode()}")
        group = [x for x in log.order[:log.order.index(rid)] if pd.group_of(log.launch[x]) == pd.group_of(line)]
        if group:
            prev = group[-1]
            if prev not in log.launched and prev not in log.not_launched:
                raise GateError("previous_attempt_unresolved", f"{prev}, the attempt before {rid}, has neither a "
                                                               "launched nor a not_launched line")
            launched_before = [x for x in group if x in log.launched]
            if launched_before and report is None:
                raise GateError("repeat_report_missing", f"{rid} follows the launched attempt {launched_before[-1]}: "
                                                         "its launch line names the AM-8a report (AM-19 item 3(a))")
        prior = check_previous_runs(ss, log, group, spec, ctx["previous_runs"])
        problem = arm_pins_problem(log, ctx["pin"], arm_run_stages())
        if problem is not None:
            raise GateError("code_pin_mismatch", problem)
    except ss.SelectionRefused as e:
        raise pd._refused(e) from e
    ctx.update(launch_line=line, launch_line_sha256=ss.sha256_bytes(log.raw[rid]), decision_date=None, cutoff_utc=None,
               launch_order_position=None, previous=prior)
    return _ok(f"own launch line equals the log's ({rid}); attempt {line['attempt']}"
               + (f" after {[p['run_id'] for p in prior]} (no divergence)" if prior else "")
               + f"; every arm line at P_K {ctx['pin'][:12]}", launch_line=line, previous_runs=prior)


def stage_teacher_identity(ctx):
    """Plan decision (iv), check 3: a CPU child loads the teacher strictly; its record equals the arm's teacher."""
    t = ctx["teacher_ckpt"]
    if ctx["rehearsal"] and not Path(t).is_file():
        return _skip(f"SKIPPED (rehearsal: no teacher file at {t})")
    exp, spec = ctx["teacher_expect"], ctx["spec"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    rc, rec, out = ctx["child_json"]([Path(__file__).resolve(), "_teacher-identity", "--teacher-ckpt", t,
                                      "--expect-sha256", exp["sha256"]], "TEACHER_JSON", 1800, env)
    if rec is None:
        raise GateError("teacher_identity_failed", f"the teacher child failed: rc={rc} {out}")
    bad = teacher_identity_problems(rec, t, spec, exp)
    if bad:
        raise GateError("teacher_identity_mismatch", f"the teacher differs from {spec.run_stage}'s: {bad[:6]}")
    return _ok(f"{exp['name']} sha256 {exp['sha256'][:12]}…, strict load, DL-88's values with its checkpoint fields, "
               f"{pd.TEACHER_PARAMS:,} params, 0 trainable, eval mode", teacher=rec)


def dry_run_argv(ctx, tmp: Path) -> list:
    """pd's dry run (PL-25) with --arm NAME and the arm's teacher and its sha256."""
    spec, exp = ctx["spec"], ctx["teacher_expect"]
    return [REPO / "src" / "training" / "train_distill.py", "--dry-run", "--stage", spec.stage, "--seed", str(spec.seed),
            "--lambda-logit", pd.fmt_value(ctx["lambda_logit"]), "--alpha", pd.fmt_value(ctx["alpha"]),
            "--teacher-ckpt", ctx["teacher_ckpt"], "--teacher-ckpt-sha256", exp["sha256"], "--ckpt-dir", str(tmp),
            "--arm", spec.name]


def stage_kd_dry_run(ctx):
    """pd's kd_dry_run with the arm's argv: the trainer's step-1 checks (teacher_logits_shape, teacher_feat_shape,
    teacher_same_augmented_input; readiness rule 4, audit F8) on the arm teacher."""
    if ctx["rehearsal"] and not Path(ctx["teacher_ckpt"]).is_file():
        return _skip(f"SKIPPED (rehearsal: no teacher file at {ctx['teacher_ckpt']})")
    tmp = pd.scratch_dir("kd_arm_dryrun")
    try:
        if _under(tmp, REPO) or _under(tmp, ctx["root"]):
            raise GateError("dry_run_tmp_misplaced", f"{tmp} is inside the repository or the data root")
        r = ctx["child_run"](dry_run_argv(ctx, tmp), 3600)
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("RESULT:")), "")
        ck = re.search(r"^\[ckpt\] dir=(.+?) \(verified OUTSIDE repo\)$", r.stdout, re.M)
        arm = re.search(r"^\[arm\] (\S+) ", r.stdout, re.M)
        if r.returncode != 0 or line != "RESULT: PASS" or ck is None or Path(ck.group(1)).resolve() != tmp \
                or arm is None or arm.group(1) != ctx["spec"].run_stage:
            raise GateError("kd_dry_run_failed", f"rc={r.returncode} {line!r} ckpt_dir={ck.group(1) if ck else None} "
                                                 f"arm={arm.group(1) if arm else None} (expected {tmp}, "
                                                 f"{ctx['spec'].run_stage}) {r.stderr[-300:]}")
        return _ok(f"{line} as {ctx['spec'].run_stage} into {tmp} (removed)", dry_run=line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_arm_cut_arithmetic(ctx):
    """AM-19 item 3(b) before the launch (see cut_arithmetic); a cut is a FAIL with the arithmetic as its record."""
    ss, _ = pd._lib()
    spec = ctx["spec"]
    figures = []
    if ctx["reference_telemetry"] is not None:
        figures.append(reference_telemetry(ctx["reference_telemetry"], spec))
    for p in ctx["reference_smokes"]:
        figures.append(reference_smoke(p, spec))
    ar = cut_arithmetic(ss, spec.horizon, figures, ctx["schedule"]["in_force"], ctx["now"])
    ctx["arithmetic"] = ar
    print(f"AM-19 item 3(b) arithmetic for {spec.run_stage}: {ar['text']}", flush=True)
    if ar["cut"]:
        raise GateError("arm_cut_by_arithmetic", ar["text"])
    return _ok(ar["text"], arithmetic=ar)


STAGES = [
    ("arguments", stage_arguments),
    ("data_isolation", E1G.stage_data_isolation),
    ("repo_state", E1G.stage_repo_state),
    ("module_provenance", stage_module_provenance),
    ("records", stage_records),
    ("selection_inputs", pd.stage_selection_inputs),
    ("launch_order", stage_launch_order),
    ("class_weights", E1G.stage_class_weights),
    ("smoke_loss", E1G.stage_smoke_loss),
    ("kd_image", pd.stage_kd_image),
    ("teacher_identity", stage_teacher_identity),
    ("cuda", E1G.stage_cuda),
    ("imagenet_backbone", E1G.stage_imagenet_backbone),
    ("smoke_loader_seed", E1G.stage_smoke_loader_seed),
    ("smoke_dataloader", E1G.stage_smoke_dataloader),
    ("seed_sequence_R5", E1G.stage_seed_sequence),
    ("kd_dry_run", stage_kd_dry_run),
    ("arm_cut_arithmetic", stage_arm_cut_arithmetic),
    ("repo_unchanged", E1G.stage_repo_unchanged),
]
assert tuple(n for n, _ in STAGES) == STAGE_NAMES_ORDER


# ------------------------------------------------------------------------------------------ the launch
def build_launch_block(ctx) -> str:
    """pd's launch block (PL-29) for the arm: --arm NAME after the stage and seed, the arm's teacher and its sha256;
    never --teacher-config or any other of pd's NEVER_PASSED flags."""
    q = shlex.quote
    spec, exp = ctx["spec"], ctx["teacher_expect"]
    d, ev, rid = q(str(ctx["ckpt_dir"])), Path(ctx["evidence"]), ctx["run_id"]
    out, pid = q(str(ev / f"{rid}.stdout.log")), q(str(ev / f"{rid}.pid"))
    args = ["--real-run", "--confirm-real-run", "--init", "imagenet", "--stage", spec.stage, "--seed", str(spec.seed),
            "--arm", spec.name]
    sel = ctx["selection_files"]
    args += ["--lambda-logit", pd.fmt_value(ctx["lambda_logit"])]
    if sel["lambda_logit"] is not None:
        args += ["--lambda-selection", sel["lambda_logit"]["path"]]
    args += ["--alpha", pd.fmt_value(ctx["alpha"])]
    if sel["alpha_cwd"] is not None:
        args += ["--alpha-selection", sel["alpha_cwd"]["path"]]
    args += ["--teacher-ckpt", ctx["teacher_ckpt"], "--teacher-ckpt-sha256", exp["sha256"], "--num-workers",
             str(pd.NUM_WORKERS), "--log-every", str(pd.LOG_EVERY), "--records-commit", ctx["records_commit"],
             "--ckpt-dir", str(ctx["ckpt_dir"])]
    assert not any(a in pd.NEVER_PASSED for a in args)
    env = [f"PYTHONPATH={q(str(REPO))}", f"PLANTSEG_DATA_ROOT={q(str(ctx['root']))}",
           f"PLANTSEG_IMAGE_DIGEST={q(pd.KD_IMAGE_DIGEST)}"]
    if ctx.get("hub_dir"):
        env.append(f"TORCH_HOME={q(str(Path(ctx['hub_dir']).parent))}")
    cmd = (f"env -u PLANTSEG_GIT_COMMIT {' '.join(env)} nohup {q(sys.executable)} -B src/training/train_distill.py "
           + " ".join(q(a) for a in args) + f" > {out} 2>&1 &")
    return "\n".join(["set -eu", f"cd {q(str(REPO))}",
                      f"[ -z \"$(ls -A {d} 2>/dev/null)\" ] || {{ printf 'STOP: %s is not empty\\n' {d}; exit 1; }}",
                      f"[ ! -e {out} ] || {{ printf 'STOP: %s exists\\n' {out}; exit 1; }}",
                      cmd, f"echo $! > {pid}", pd.NEXT_LINE])


def gate_record(ctx, results: list, verdict: str, block=None) -> dict:
    """pd's gate record keys (kd_gate_record/1) plus the arm's; format kd_arm_gate_record/1, so pd's check-run-meta
    refuses it and pa's reads it."""
    rec = pd.gate_record(ctx, results, verdict, block)
    rec["format"] = GATE_RECORD_FORMAT
    spec = ctx.get("spec")
    rec.update(arm=None if spec is None else arm_fields(spec), approval=ctx.get("approval"),
               arms_pin=ctx.get("arms_pin"), teacher_record=ctx.get("teacher_record"), go_rule=ctx.get("go_rule"),
               teacher=ctx.get("teacher_expect"), arithmetic=ctx.get("arithmetic"), previous_runs=ctx.get("previous"))
    return rec


def gate_ctx(args) -> dict:
    return {"arm": args.arm, "profile": PROFILE, "stage": None, "seed": None, "ckpt_dir": args.ckpt_dir,
            "evidence": args.evidence, "expect_head": args.expect_head, "records_commit": args.records_commit,
            "records": args.records, "teacher_ckpt": args.teacher_ckpt, "clock_offset": args.clock_offset_seconds,
            "record": args.record, "rehearsal": args.rehearsal, "lambda_logit": args.lambda_logit, "alpha": args.alpha,
            "attempt": args.attempt, "am8a_report": args.am8a_report, "previous_runs": list(args.previous_run or []),
            "decision_records": list(args.decision_record or []), "approval_dl": args.approval_dl,
            "arms_pin_dl": args.arms_pin_dl, "go_rule_dl": args.go_rule_dl,
            "reference_telemetry": args.reference_telemetry, "reference_smokes": list(args.reference_smoke or []),
            "git_root": None, "now": time.time(), "warnings": [], "child_json": pd._child_json,
            "child_run": lambda argv, timeout: E1G._child(argv, timeout=timeout)}


def run_gate(args, *, stages=None, ctx=None) -> int:
    """pd's gate loop over the arm stages: the first FAIL stops; GO only when every stage PASSes outside a
    rehearsal; the record is written before GO is printed."""
    if not args.rehearsal and args.record is None:
        print("usage error: a real gate run requires --record PATH (or pass --rehearsal)", file=sys.stderr)
        return 2
    ctx = gate_ctx(args) if ctx is None else ctx
    stages = STAGES if stages is None else stages
    print("=" * 78)
    print("KD ARM LAUNCH GATE (L-TEACHER-ARMS C2) -- preflight_distill is imported, its stages reused")
    print(f"repo={REPO}")
    print(f"python={sys.executable} {platform.python_version()} | {platform.platform()}")
    print(f"arm={args.arm} | lambda={args.lambda_logit} | alpha={args.alpha} | attempt={args.attempt} | records "
          f"commit={args.records_commit}")
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
            pd._write_record(Path(args.record), gate_record(ctx, results, "GO", block))
        except OSError as e:
            print(f"VERDICT: NO-GO -- the evidence record could not be written ({type(e).__name__}: {e})")
            return 1
        print(f"record written: {args.record}")
        print("VERDICT: GO")
        print(launch_note(ctx["spec"]))
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
            pd._write_record(Path(args.record), gate_record(ctx, results, verdict))
        except OSError as e:            # the verdict stands
            print(f"(the evidence record could not be written: {type(e).__name__}: {e})")
    return 1


# ------------------------------------------------------------------------------------- check-run-meta
def arm_run_meta_keys(spec) -> list:
    """The run_meta keys the trainer writes for an arm run: its stage's keys, then target_construction last."""
    return pd.run_meta_keys(spec.stage) + ["target_construction"]


def arm_provenance_rules(exp: dict) -> dict:
    """teacher_provenance, key by key: DL-88's values, with the checkpoint fields of the arm's teacher."""
    t = exp["teacher"]

    def path_ok(p):
        return (isinstance(p, str) and (t["name"] is None or Path(p).name == t["name"])
                and (exp["teacher_ckpt"] is None or p == exp["teacher_ckpt"]))

    def bytes_ok(v):
        return pd.same(v, t["bytes"]) if t["bytes"] is not None else (isinstance(v, int) and not isinstance(v, bool)
                                                                        and v > 0)
    return {"builder": lambda v: pd.same(v, pd.TEACHER_BUILDER), "ckpt_path": path_ok,
            "ckpt_sha256": lambda v: pd.same(v, t["sha256"]), "ckpt_bytes": bytes_ok,
            "expected_sha256": lambda v: pd.same(v, t["sha256"]),
            "config_path": lambda v: isinstance(v, str) and _under(Path(v), REPO),
            **{k: (lambda v, w=w: pd.same(v, w)) for k, w in pd.TEACHER_VALUES.items()}}


def arm_meta_rules(spec, exp: dict, td) -> dict:
    """pd's rules for the arm's stage, with the arm's: arm = its run stage, descriptive true, parent_of_e4_e7 false,
    teacher_provenance of its teacher, and target_construction = its {K, views} (AM-18 item 3's header)."""
    def eq(w):
        return lambda v: pd.same(v, w)
    r = pd.meta_rules(spec.stage, exp, td)
    prov = arm_provenance_rules(exp)
    r.update({"arm": ("stage", eq(spec.run_stage)), "descriptive": ("stage", eq(True)),
              "parent_of_e4_e7": ("stage", eq(False)),
              "teacher_provenance": ("record", lambda v: isinstance(v, dict) and sorted(v) == sorted(pd.PROVENANCE_KEYS)
                                     and all(fn(v[k]) for k, fn in prov.items())),
              "target_construction": ("stage", eq(spec.target_construction))})
    return r


def arm_meta_problems(row: dict, spec, exp: dict, td) -> tuple:
    """(problems, warnings) of one arm run_meta row: the exact key set with target_construction last, then every
    key's rule."""
    rules = arm_meta_rules(spec, exp, td)
    want = arm_run_meta_keys(spec)
    problems, warns = [], []
    if sorted(rules) != sorted(want):
        problems.append(f"rule table and key list differ: {sorted(set(rules) ^ set(want))}")
    if sorted(row) != sorted(want):
        problems.append(f"keys: missing {sorted(set(want) - set(row))}, extra {sorted(set(row) - set(want))}")
    elif list(row)[-1] != "target_construction":
        problems.append(f"keys: target_construction is not the row's last key ({list(row)[-1]!r} is)")
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
                    prov = arm_provenance_rules(exp)
                    detail = {pk: row[k].get(pk) for pk, fn in prov.items() if not fn(row[k].get(pk))}
                problems.append(f"{k} ({kind}): {detail!r}")
    return problems, warns


def run_meta_expectations(args) -> tuple:
    """(spec, expectations) from the arm gate's GO record, or (--allow-smoke) a dry row of the arm: mode dry, 25
    iterations, no pretrained weights, no selection files and no records commit."""
    if args.allow_smoke:
        spec = arm_spec(args.arm)
        error = lam_alpha_error(spec, args.lambda_logit, args.alpha)
        if error is not None:
            raise GateError("arm_arguments", error)
        if spec.teacher == "arm" and not (isinstance(args.teacher_sha256, str) and SHA256_RE.fullmatch(args.teacher_sha256)):
            raise GateError("teacher_sha256", f"{spec.run_stage}: --teacher-sha256 <the arm teacher's sha256> is "
                                              "required with --allow-smoke")
        teacher = ({"name": None, "sha256": args.teacher_sha256, "bytes": None} if spec.teacher == "arm"
                   else {"name": pd.TEACHER_FILE, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES})
        nulls = {"lambda_logit": None, "alpha_cwd": None}
        return spec, {"stage": spec.stage, "mode": "dry", "seed": spec.seed, "max_iters": 25, "horizon": spec.horizon,
                      "used_pretrained": False, "pin": args.expect_head, "records_commit": None,
                      "lambda_logit": float(args.lambda_logit), "alpha": float(args.alpha), "selection_sha256": nulls,
                      "selection_files": nulls, "teacher_ckpt": None, "teacher": teacher,
                      "run_id": Path(args.ckpt_dir).name}
    try:                                # an unreadable or partial record is a usage refusal (exit 2), never a FAIL
        rec = json.loads(Path(args.gate_record).read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or rec.get("format") != GATE_RECORD_FORMAT or rec.get("verdict") != "GO":
            raise GateError("gate_record", f"{args.gate_record} is not a GO record of this gate")
        if not (isinstance(rec["ckpt_dir"], str) and Path(rec["ckpt_dir"]).is_absolute()):
            raise GateError("gate_record", f"{args.gate_record}: ckpt_dir {rec['ckpt_dir']!r} is not an absolute path")
        spec = arm_spec(rec["arm"]["name"])
        if spec.sha256 != rec["arm"]["registry_sha256"]:
            raise GateError("gate_record", f"{args.gate_record}: the registry entry of {spec.name} changed since the "
                                           "gate ran")
        t = rec["teacher"]
        if not (isinstance(t, dict) and SHA256_RE.fullmatch(str(t.get("sha256")))):
            raise GateError("gate_record", f"{args.gate_record}: no teacher sha256")
        sel = rec["selection_files"]
        return spec, {"stage": spec.stage, "mode": "real", "seed": spec.seed, "max_iters": spec.horizon,
                      "horizon": spec.horizon, "used_pretrained": True, "pin": rec["pin"],
                      "records_commit": rec["records_commit"], "lambda_logit": rec["lambda_logit"],
                      "alpha": rec["alpha"],
                      "selection_sha256": {k: None if v is None else v["sha256"] for k, v in sel.items()},
                      "selection_files": {k: None if v is None else v["path"] for k, v in sel.items()},
                      "teacher_ckpt": rec["teacher_ckpt"], "teacher": {k: t.get(k) for k in ("name", "sha256", "bytes")},
                      "run_id": rec["run_id"], "ckpt_dir": rec["ckpt_dir"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise GateError("gate_record", f"{args.gate_record}: unreadable, or not a complete GO record of this gate "
                                       f"({type(e).__name__}: {e})") from e


def check_run_meta(args) -> int:
    """PL-30 for an arm run: its run_meta file holds exactly one row and every key passes its one rule. The launched
    line (PL-13) is printed on PASS and on FAIL (ruling R9); a --ckpt-dir that is not the gate record's run is
    refused (exit 2) before any check."""
    _, td = pd._lib()
    spec, exp = run_meta_expectations(args)
    d = Path(args.ckpt_dir)
    meta_p = d / f"{spec.stage}_run_meta.jsonl"
    problems, warns, rows = [], [], []
    if d.name != exp["run_id"] or (exp.get("ckpt_dir") is not None and d != Path(exp["ckpt_dir"])):
        raise GateError("ckpt_dir_not_the_runs", f"--ckpt-dir {d} is not the run of this check "
                                                 f"({exp.get('ckpt_dir') or exp['run_id']}); nothing is printed for "
                                                 "the launch log")
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
        p, w = arm_meta_problems(rows[0], spec, exp, td)
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
                    "records_commit": row.get("records_commit"), "decision_date": None, "launch_order_position": None}
        print("LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):")
        print(json.dumps(launched))
    else:
        print("LAUNCHED LINE: none (no single run_meta row): if the start wrote no run_meta row with mode real, write "
              "its not_launched line with `preflight_arms.py not-launched` and its evidence (AM-19a reading 6)")
    ok = not problems
    print(f"RESULT: {'PASS' if ok else 'FAIL'} -- {spec.run_stage} run_meta vs E3's recipe with the arm's rules, one "
          "rule per key (PL-30)" + ("" if ok else "; stop the run under the runbook: a stopped line and its AM-8a report"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------- entry, records, not-launched
def cmd_entry(args) -> int:
    """PL-27 / PL-32 for an arm: its launch line, from the committed launch log at HEAD of this clone."""
    ss, _ = pd._lib()
    spec = arm_spec(args.arm)
    error = lam_alpha_error(spec, args.lambda_logit, args.alpha)
    if error is not None:
        raise GateError("arm_arguments", error)
    if not COMMIT_RE.fullmatch(args.code_pin or ""):
        raise GateError("code_pin_malformed", f"--code-pin must be 40 lowercase hex: {args.code_pin!r}")
    error = clock_offset_error(args.clock_offset_seconds)
    if error is not None:
        raise GateError("clock_offset", error)
    src = ss.HeadSource()
    if not src.commit_exists(args.code_pin):
        raise GateError("code_pin_unavailable", f"--code-pin {args.code_pin} is not in this clone")
    try:
        schedule = ss.load_schedule(src)
        log = ss.load_launch_log(src)
        dl_text = src.read_at_head(ss.DECISION_LOG_REL, missing="decision_log_missing").decode("utf-8", "replace")
        arms_pin = arms_pin_of(dl_text, args.arms_pin_dl)
        problem = pin_problem(arms_pin, args.code_pin)
        if problem is not None:
            raise GateError("code_pin_mismatch", f"--code-pin: {problem}: every item 3 arm runs at P_K")
        probe = arm_launch_line(ss, spec, lam=args.lambda_logit, alpha=args.alpha, attempt=1,
                                schedule=schedule["in_force"], code_pin=args.code_pin, ckpt_dir=args.ckpt_dir,
                                am8a_report=None)
        earlier = [x for x in log.order if pd.group_of(log.launch[x]) == pd.group_of(probe)]
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
        prior = check_previous_runs(ss, log, earlier, spec, args.previous_run or [])
        problem = arm_pins_problem(log, args.code_pin, arm_run_stages())
        if problem is not None:
            raise GateError("code_pin_mismatch", problem)
        report = None
        if args.am8a_report is not None:
            rel, data = src.read(args.am8a_report, missing="repeat_report_missing", uncommitted="repeat_report_mismatch",
                                 record=True)
            report = {"path": rel, "sha256": ss.sha256_bytes(data)}
        line = arm_launch_line(ss, spec, lam=args.lambda_logit, alpha=args.alpha, attempt=len(earlier) + 1,
                               schedule=schedule["in_force"], code_pin=args.code_pin, ckpt_dir=args.ckpt_dir,
                               am8a_report=report)
        if Path(args.ckpt_dir).name != line["run_id"] or not os.path.isabs(args.ckpt_dir):
            raise GateError("run_id_mismatch", f"--ckpt-dir must be an absolute path named {line['run_id']} (attempt "
                                               f"{line['attempt']}; PL-27)")
        raw = pd.line_bytes(line)
        would = ss.LaunchLog(log.data + raw + b"\n", ss.LAUNCH_LOG_REL)    # audit F39: the would-be log parses
        if would.raw.get(line["run_id"]) != raw or would.launch[line["run_id"]]["attempt"] != line["attempt"]:
            raise GateError("launch_log_format", f"{line['run_id']}: the would-be log does not hold the line as built")
    except ss.SelectionRefused as e:
        raise pd._refused(e) from e
    print("LAUNCH LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it before the gate):")
    print(raw.decode("utf-8"))
    print(launch_note(spec))
    print(f"clock offset {args.clock_offset_seconds:+g} s (this machine's offset from a network time source; AM-19a "
          f"reading 21, |O| <= {pd.CLOCK_OFFSET_MAX:g} s)")
    if prior:
        print(f"previous attempts: {prior} (none diverged: ruling R4(a))")
    return 0


def cmd_records(args) -> int:
    """PL-28: preflight_distill's records folder of H, plus every reports/arms/*.json at H, in its manifest."""
    rc = pd.cmd_records(args)
    if rc != 0:
        return rc
    ss, _ = pd._lib()
    src = ss.HeadSource()
    h, out = args.records_commit, Path(args.out)
    r = src.git("ls-tree", "-z", "--name-only", h, "--", ARMS_REL + "/")
    if r.returncode != 0:
        raise GateError("records_mismatch", f"git ls-tree {h} {ARMS_REL}/ failed: "
                                            f"{r.stderr.decode('utf-8', 'replace')[-200:]}")
    names = sorted(n for n in r.stdout.decode("utf-8").split("\0") if n)
    man_p = out / pd.RECORDS_MANIFEST
    man = json.loads(man_p.read_text(encoding="utf-8"))
    added = []
    for rel in names:
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts or "test" in rel.lower() or p.parent.as_posix() != ARMS_REL:
            raise GateError("records_mismatch", f"{rel!r} is not a reports/arms record path (SL-1)")
        if p.suffix != ".json":
            continue
        oid = src.blob_at(h, rel)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.blob_bytes(oid))
        man["files"][rel] = oid
        added.append(rel)
    tmp = man_p.with_name(man_p.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, man_p)
    print(f"arm records of {h} added to {out}: {added}")
    return 0


def cmd_not_launched(args) -> int:
    """AM-19a reading 6 for an arm launch line without a launched line: its not_launched line, after the would-be log
    parses. REL is the committed evidence (a report, or the gate's NO-GO record committed as one); --run-dir, when
    given, must hold no run_meta row with mode real."""
    ss, _ = pd._lib()
    spec = arm_spec(args.arm)
    src = ss.HeadSource()
    try:
        log = ss.load_launch_log(src)
        rid = args.run_id
        if rid not in log.launch or log.launch[rid]["stage"] != spec.run_stage:
            raise GateError("not_launched_unknown_run", f"{rid} is not a launch line of {spec.run_stage} in the "
                                                        "committed launch log")
        if rid in log.launched or rid in log.not_launched:
            raise GateError("not_launched_resolved", f"{rid} already has a launched or not_launched line")
        if args.run_dir is not None:
            d = Path(args.run_dir)
            if d.name != rid:
                raise GateError("not_launched_unknown_run", f"--run-dir {d} is not the run {rid}")
            meta_p = d / f"{spec.stage}_run_meta.jsonl"
            if meta_p.is_file():
                for ln in meta_p.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(ln) if ln.strip() else None
                    except json.JSONDecodeError:
                        row = None
                    if isinstance(row, dict) and row.get("mode") == "real":
                        raise GateError("not_launched_but_started", f"{meta_p} holds a run_meta row with mode real: "
                                                                    "the run launched (check-run-meta writes its "
                                                                    "launched line)")
        rel, data = src.read(args.evidence, missing="launch_log_report_missing", uncommitted="launch_log_report_mismatch",
                             record=True)
        line = {"event": "not_launched", "run_id": rid, "reason": args.reason,
                "evidence": {"path": rel, "sha256": ss.sha256_bytes(data)}}
        raw = json.dumps(line).encode("utf-8")
        would = ss.LaunchLog(log.data + raw + b"\n", ss.LAUNCH_LOG_REL)
        if rid not in would.not_launched:
            raise GateError("launch_log_format", f"{rid}: the would-be log does not hold the not_launched line")
    except ss.SelectionRefused as e:
        raise pd._refused(e) from e
    print("NOT_LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):")
    print(raw.decode("utf-8"))
    return 0


def cmd_stopped(args) -> int:
    """AM-19a reading 6 (audit F34) for a launched arm run that ended without a trainer record (an operator, pod or
    host stop): its stopped line naming the committed stop report, after the would-be log parses. --run-dir, when
    given, must hold no trainer end record (run_end or run_abort): a run the trainer ended has no stopped line."""
    ss, _ = pd._lib()
    spec = arm_spec(args.arm)
    src = ss.HeadSource()
    try:
        log = ss.load_launch_log(src)
        rid = args.run_id
        if rid not in log.launch or log.launch[rid]["stage"] != spec.run_stage:
            raise GateError("stopped_unknown_run", f"{rid} is not a launch line of {spec.run_stage} in the committed "
                                                   "launch log")
        if rid not in log.launched:
            raise GateError("stopped_not_launched", f"{rid} has no launched line: a start without a real run_meta row "
                                                    "is not launched (not-launched)")
        if rid in log.stopped:
            raise GateError("stopped_resolved", f"{rid} already has a stopped line")
        if args.run_dir is not None:
            d = Path(args.run_dir)
            if d.name != rid:
                raise GateError("stopped_unknown_run", f"--run-dir {d} is not the run {rid}")
            tel = d / f"{spec.stage}_telemetry.jsonl"
            ended = []
            if tel.is_file():
                for ln in tel.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(ln) if ln.strip() else None
                    except json.JSONDecodeError:
                        row = None
                    if isinstance(row, dict) and row.get("event") in ("run_end", "run_abort"):
                        ended.append(row["event"])
            if ended:
                raise GateError("stopped_but_recorded", f"{tel} holds the trainer's {ended[-1]} record: the trainer "
                                                        "ended the run, so it has no stopped line")
        rel, data = src.read(args.report, missing="launch_log_report_missing", uncommitted="launch_log_report_mismatch",
                             record=True)
        line = {"event": "stopped", "run_id": rid, "report": {"path": rel, "sha256": ss.sha256_bytes(data)}}
        raw = json.dumps(line).encode("utf-8")
        would = ss.LaunchLog(log.data + raw + b"\n", ss.LAUNCH_LOG_REL)
        if rid not in would.stopped:
            raise GateError("launch_log_format", f"{rid}: the would-be log does not hold the stopped line")
    except ss.SelectionRefused as e:
        raise pd._refused(e) from e
    print("STOPPED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it with its report):")
    print(raw.decode("utf-8"))
    return 0


# ------------------------------------------------------------------------------------------ the parser
def path_args(args) -> list:
    """(value, flag) of every path argument: each is refused as a string before any filesystem call (SL-1)."""
    names = {"entry": ("ckpt_dir", "am8a_report", "previous_run"), "records": ("out",),
             "gate": ("ckpt_dir", "evidence", "records", "teacher_ckpt", "record", "reference_telemetry",
                      "reference_smoke", "am8a_report", "previous_run", "decision_record"),
             "check-run-meta": ("ckpt_dir", "gate_record"), "not-launched": ("evidence", "run_dir"),
             "stopped": ("report", "run_dir"), "_teacher-identity": ("teacher_ckpt",)}.get(args.cmd, ())
    out = []
    for n in names:
        v = getattr(args, n, None)
        for x in (v if isinstance(v, list) else [v]):
            if x is not None:
                out.append((x, "--" + n.replace("_", "-")))
    return out


def build_parser():
    p = pd._Parser(description="KD arm launch gate (L-TEACHER-ARMS C2).", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=pd._Parser)
    e = sub.add_parser("entry", allow_abbrev=False)
    e.add_argument("--arm", required=True)
    e.add_argument("--lambda-logit", type=float, required=True)
    e.add_argument("--alpha", type=float, required=True)
    for name in ("--ckpt-dir", "--code-pin", "--arms-pin-dl"):
        e.add_argument(name, required=True)
    e.add_argument("--clock-offset-seconds", type=float, required=True)
    e.add_argument("--am8a-report", default=None)
    e.add_argument("--previous-run", action="append", default=None)
    r = sub.add_parser("records", allow_abbrev=False)
    r.add_argument("--records-commit", required=True)
    r.add_argument("--out", required=True)
    g = sub.add_parser("gate", allow_abbrev=False)
    g.add_argument("--arm", required=True)
    g.add_argument("--lambda-logit", type=float, required=True)
    g.add_argument("--alpha", type=float, required=True)
    for name in ("--ckpt-dir", "--evidence", "--expect-head", "--records-commit", "--records", "--teacher-ckpt",
                 "--approval-dl", "--arms-pin-dl"):
        g.add_argument(name, required=True)
    g.add_argument("--clock-offset-seconds", type=float, required=True)
    g.add_argument("--record", default=None)
    g.add_argument("--go-rule-dl", default=None)
    g.add_argument("--reference-telemetry", default=None)
    g.add_argument("--reference-smoke", action="append", default=None)
    g.add_argument("--attempt", type=int, default=1)
    g.add_argument("--am8a-report", default=None)
    g.add_argument("--previous-run", action="append", default=None)
    g.add_argument("--decision-record", action="append", default=None)
    g.add_argument("--rehearsal", action="store_true")
    c = sub.add_parser("check-run-meta", allow_abbrev=False)
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument("--gate-record", default=None)
    c.add_argument("--allow-smoke", action="store_true")
    c.add_argument("--arm", default=None)
    c.add_argument("--expect-head", default=None)
    c.add_argument("--lambda-logit", type=float, default=None)
    c.add_argument("--alpha", type=float, default=None)
    c.add_argument("--teacher-sha256", default=None)
    n = sub.add_parser("not-launched", allow_abbrev=False)
    for name in ("--arm", "--run-id", "--reason", "--evidence"):
        n.add_argument(name, required=True)
    n.add_argument("--run-dir", default=None)
    s = sub.add_parser("stopped", allow_abbrev=False)
    for name in ("--arm", "--run-id", "--report"):
        s.add_argument(name, required=True)
    s.add_argument("--run-dir", default=None)
    t = sub.add_parser("_teacher-identity", allow_abbrev=False)
    t.add_argument("--teacher-ckpt", required=True)
    t.add_argument("--expect-sha256", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:                                # SL-1 (plan 1.0): the DIAG guard, as-given string first, then the realpath
        from src.eval.teacher_diag import Refused, refuse_test_path
    except ImportError as e:
        print(f"usage error: run with PYTHONPATH={REPO} ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    try:
        for value, what in path_args(args):
            refuse_test_path(value, what)
    except Refused as e:
        print(f"REFUSED [test_path]: {e}", file=sys.stderr)
        print("RESULT: REFUSED (test_path)")
        return 2
    if args.cmd == "_teacher-identity":
        print("TEACHER_JSON:" + json.dumps(teacher_identity_child(args.teacher_ckpt, args.expect_sha256),
                                           sort_keys=True, default=str), flush=True)
        return 0
    if args.cmd == "check-run-meta":
        if args.allow_smoke == (args.gate_record is not None) or (args.allow_smoke and (
                args.arm is None or args.expect_head is None or args.lambda_logit is None or args.alpha is None)):
            print("usage error: check-run-meta takes --gate-record R, or --allow-smoke with --arm, --expect-head, "
                  "--lambda-logit and --alpha", file=sys.stderr)
            return 2
    try:
        pd._lib()
    except ImportError as e:
        print(f"usage error: run with PYTHONPATH={REPO} ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    try:
        if args.cmd == "gate":
            return run_gate(args)
        if args.cmd == "check-run-meta":
            return check_run_meta(args)
        return {"entry": cmd_entry, "records": cmd_records, "not-launched": cmd_not_launched,
                "stopped": cmd_stopped}[args.cmd](args)
    except GateError as e:
        print(f"REFUSED [{e.code}]: {e}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code})")
        return 2
    except OSError as e:                # a file that cannot be read or written is a refusal, never a FAIL
        print(f"REFUSED [io_error]: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: REFUSED (io_error)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
