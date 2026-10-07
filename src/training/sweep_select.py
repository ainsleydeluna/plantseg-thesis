"""Sweep selection for lambda_logit (AM-2 / DL-06) and alpha_cwd (AM-16 item 2 / DL-27), with AM-7a and
AM-19 item 2 (the sweep decision dates).

Shared by scripts/select_lambda.py and scripts/select_alpha.py (lane L-AM16-ALPHA; hardened in lane
L-KD-HARDEN; decision dates in lane K2). The cross-lane selection contract (docs/lane_specs/part2.md, with
errata E-39 and E-40): refuse partial inputs unless a committed decision record (AM-19 item 2(h)) lists the
missing candidates as cut at the decision date and the code derives the same statuses; read the band and
rule from committed JSON (configs/sweep_rules.json; alpha's band from reports/derived/dl27_band.json);
write a JSON with a `rule_trace`; never read TEST; the winner is always an existing run.

Committed inputs (PL-16) are read through one accessor with two sources: HeadSource, the working tree at
HEAD of the repository at GIT_ROOT (the selection), and RecordsSource, a records folder whose files are
verified by blob id against a records commit (the gate, PL-28). A committed file is under GIT_ROOT, present
at HEAD and byte-equal to its HEAD blob; no index or mode comparison is made. The code the selection runs
(this module, the select script and configs/distill.py) is committed too, and its blobs, the rules file's
and the schedule file's equal those at the sweep's code pin (code_pin_mismatch).

Launch log (reports/kd_launch_log.jsonl, PL-13): one line per event of every KD launch: `launch` (printed by
the gate and committed before the pod is cloned), `launched` (printed by check-run-meta when the run_meta
row exists), `not_launched` (a start that wrote no run_meta) and `stopped` (a launched run that ended
without a trainer record). A grid value's attempts are its launch lines in log order (PL-9). Each supplied
run directory has exactly one launch line and one launched line and agrees with them (PL-14).

Push evidence (AM-19a reading 10; PL-17(c)): reports/derived/lambda_selection_push.json, GitHub's activity
response saved verbatim on the day the lambda selection file was pushed, and reports/derived/
lambda_selection_push_list.json, committed with it: for each push after T_lo, whether its commit carries the
file, settled then by fetching it, with the remote's answer. Both are settled once: one commit adds them, once,
and neither changes afterwards. Branch deletions are not counted as pushes; a push the list marks as not
served counts as carrying the file and is named in the alpha basis; a push after T_lo the list lacks refuses;
wherever the clone holds a pushed commit, served or not, the list is checked against it.

Decision record (AM-19 item 2(h); AM-19a reading 19): committed at reports/derived/<sweep>_decision_record.json,
written from a launch-log prefix (its lines and sha256) and checked against those lines: a later launch line of
the default does not void it, a later launch line of a non-default value refuses. Its decision date and cutoff
must equal the derived ones (the alpha basis is not compared). A record the checks refuse for another reason is
void, kept and never edited; one corrected record replaces it, written once to
reports/derived/<sweep>_decision_record_corrected.json after the AM-8a fault report, naming both, with the
sha256 of the void record, of the fault report and of the corrected record in the decision log. A record at any
other path and a void record whose later non-default launch lines refuse are decision_record_correction_invalid;
the writer refuses a second record at either path (decision_record_correction_invalid). Written once means one
commit adds the file and no other commit changes or deletes it: any later change of a committed record, a
deletion or an identical restore included, makes it decision_record_rewritten (a second correction written over
the first too), and no correction lifts that refusal (the void record must be written once as well).

A candidate directory: only these files are opened:
  <stage>_run_meta.jsonl     exactly one row: mode real, the sweep's seed, stage and terms, max_iters
                             equal to the sweep's iterations, the swept value, the current Logit-KD
                             semantics without an override, the recipe of record (RECIPE_EXPECT), the
                             carriers every candidate must share (RECIPE_IDENTICAL) and ramp_iters; its
                             wall_clock is the launch time, and git_head and records_commit agree with the
                             launch log
  <stage>_telemetry.jsonl    its rows, each with a finite timestamp (wall_clock; wall_clock_end in run_end)
  best.json                  train_e1's schema {best_ckpt, best_val_miou_all_class}: read when present,
                             for a finished run (and for a diverged one: best_val_partial)
  the checkpoint             best.json names it (by basename when the pod path is absent), or run_end's
                             best_ckpt without best.json; a finished run's, when present
Any input path with a component named `test` is refused. A file that cannot be parsed is refused by
name (never a traceback), except a torn last telemetry line, which marks a run that did not finish.

Status of a grid value (AM-19 item 2, with the readings of AM-19a, DL-86; DL-53 as amended). C is the end
of the sweep's decision date (24:00 Asia/Manila); a timestamp meets the date when it is earlier.
  finished   a run_end record with its checks passed (the last train row at max_iters; iter =
             max_iters; best_val_miou_all_class finite in [0, 1]), the run passing every check of the
             selection; for a non-default candidate after C, its wall_clock_end meets the date. Finished
             is final (PL-6): its score of record is run_end's best value; best.json and the checkpoint,
             when absent, leave it finished with null hashes; when present they must agree (exit 2 at any
             time, never a cut).
  diverged   AM-7a: the trainer's own run_abort record for a STUDENT divergence under AM-7 (a) or (b): a
             rule in DIVERGENCE_RULES, cause DIVERGENCE_CAUSE, finite input and teacher outputs, at the last
             train iteration; for a non-default candidate after C, it meets the date. The run is excluded
             from the selection and never relaunched; its directory is a required input (not a shortfall).
  cut        after C, a non-default candidate that is neither, with its reason: never_launched,
             still_running, stopped_no_finished_repeat, finished_after_date, aborted_after_date,
             run_aborted_other:<rule>/<cause>, run_checks_failed, aborted_twice:<rule>/<cause>,
             on_course_repeat_stopped, or a candidate-level refusal code (CANDIDATE_LEVEL_CODES). One
             exception: a stop without a trainer record of a candidate that was on course (on_course())
             waits once for its repeat (PL-9(e)). A selection with a cut needs a committed decision record.
  waiting    exit 3 (SHORTFALL_CODES): before C, a candidate that has not finished or diverged; at any
             time, the default candidate, which is never cut (item 2(d)).
Refusal codes (exit 2 in the select scripts), besides the format, consistency and launch-log refusals:
  default_candidate_diverged  the sweep's default candidate (sweep_rules default_candidate: lambda 1,
                              alpha 50) diverged: AM-7 applies in full (stop the stage, apply the AM-7a
                              item 5 clipping value, rerun the FP32 stages); no selection is made. Only
                              checked once the candidates agree on RECIPE_IDENTICAL (num_workers and the
                              teacher's checkpoint, config and resolved model-config hashes;
                              recipe_mismatch_across_candidates first) and, for alpha, once every supplied
                              run shares the selection's lambda (K8-2(a)): a diverged default run at another
                              lambda reads as lambda_mismatch.
  default_aborted_twice       the default candidate's latest attempt ended the same way as an earlier one
                              (AM-19 item 3(a); AM-19a reading 9): a STOP that a new amendment settles.
  run_aborted_other           a run_abort that is not a student divergence (another rule, a non-finite
                              input or teacher output), or a train row flagged `nonfinite` (or a val row
                              whose all-class mIoU is) with no run_abort after it (the abort record is
                              missing or torn). Before C, and for the default: STOP: investigate; AM-8a
                              governs a repeat; never excluded, never a shortfall. After C a non-default
                              candidate in that state is cut. A non-default value whose latest attempt ended
                              the same way as any earlier attempt of the value (AM-19a reading 9: the attempts
                              between them, a stopped one included, change nothing) is not repeated again:
                              it waits before C (exit 3; AM-19a reading 17) and is cut after it. Before C it
                              also waits, not refused, while an earlier attempt of the value is incomplete and
                              could still end the same way (AM-19a reading 18; _twice_undecided).
  abort_record_invalid        a run_abort record that is malformed, is not the last telemetry row, is
                              accompanied by a run_end, is not at the last train iteration, or is an
                              AM-7 record the trainer cannot have written:
                              AM-7 (b) unless iter > ramp_iters + AM7B_WINDOW, window_mean and
                                running_min are finite numbers with running_min >= 0, the trainer's own
                                test window_mean > AM7B_FACTOR x running_min holds on the recorded values,
                                threshold == AM7B_FACTOR x running_min, and ratio == window_mean /
                                running_min ("inf" when running_min is 0 or the quotient overflows);
                              AM-7 (a) unless iter >= 2 (a non-finite value at iteration 1 is recorded
                                as step1_checks, R8-1) and either detail.loss is "nan", "inf" or "-inf"
                                with detail.grad_norm null and the train row at the abort iteration
                                holding the key grad_norm with value null (the loss stop precedes
                                backward, K8-2(c)), or detail.loss is finite and detail.grad_norm is
                                non-finite; in both cases that key is named in the `nonfinite` map of
                                the train row at the abort iteration.
  recipe_mismatch             a run_meta that violates RECIPE_EXPECT or lacks a RECIPE_IDENTICAL value
                              (candidate-level); recipe_mismatch_across_candidates: candidates whose
                              RECIPE_IDENTICAL values differ (sweep-level, before and after C; AM-19a
                              reading 5).

Tie arithmetic. "Within the band of the best" is inclusive (best - value <= band) and is evaluated in
decimal on the recorded values (Decimal(repr(x))), so an exact 0.5 pp gap between two recorded values
counts as tied, as it does by hand. Only finished candidates are compared; a sole finished candidate
wins with no band and no tie (AM-7a). A winner is a boundary result at a grid end, or at an edge of the
finished set next to a diverged or cut value (boundary_kind grid_end / edge_of_finished_set; edge_removed
names each such value with the rule that removed it, AM-7a or AM-19).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
RULES_PATH = REPO / "configs" / "sweep_rules.json"
# AM-19 (lane K2): committed inputs are read from the repository at GIT_ROOT, at its HEAD. The smoke repoints
# GIT_ROOT and now_utc() in-process; no CLI flag redirects either.
GIT_ROOT = REPO
RULES_REL = "configs/sweep_rules.json"
SCHEDULE_REL = "configs/kd_schedule.json"
LAUNCH_LOG_REL = "reports/kd_launch_log.jsonl"
DECISION_LOG_REL = "docs/DECISION_LOG.md"
BAND_REL = "reports/derived/dl27_band.json"
LAMBDA_SELECTION_REL = "reports/derived/lambda_selection.json"
ALPHA_SELECTION_REL = "reports/derived/alpha_selection.json"
PUSH_EVIDENCE_REL = "reports/derived/lambda_selection_push.json"
PUSH_LIST_REL = "reports/derived/lambda_selection_push_list.json"      # AM-19a reading 10: committed with it
DECISION_RECORD_REL = {"lambda_logit": "reports/derived/lambda_decision_record.json",
                       "alpha_cwd": "reports/derived/alpha_decision_record.json"}
# AM-19a reading 19: the one corrected record of a sweep, written once to its own new file
DECISION_RECORD_CORRECTED_REL = {"lambda_logit": "reports/derived/lambda_decision_record_corrected.json",
                                 "alpha_cwd": "reports/derived/alpha_decision_record_corrected.json"}
SWEEP_SELECT_REL = "src/training/sweep_select.py"
DISTILL_REL = "configs/distill.py"
SELECT_SCRIPT_REL = {"lambda_logit": "scripts/select_lambda.py", "alpha_cwd": "scripts/select_alpha.py"}
BEST_JSON = "best.json"
BEST_JSON_KEYS = ["best_ckpt", "best_val_miou_all_class"]
STAGE_TERMS = {"E2": {"logit_kd": True, "cwd_feat": False, "cwd_logit": False},
               "E3": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}}
# PL-5: every refusal code sits in exactly one of three closed classes (the smoke checks the three modules).
# A SHORTFALL waits (exit 3): a run that is absent, has not written its files yet, or has not finished;
# shortfall_am19_item2 is the select scripts' aggregate. From C a non-default value in that state is cut,
# except candidate_missing for a launched line (launch_log_mismatch, PL-14).
SHORTFALL_CODES = ("candidate_missing", "candidate_incomplete", "run_unfinished", "shortfall_am19_item2")
# A cut reason for a non-default candidate once now >= C; a refusal otherwise (and for the default).
CANDIDATE_LEVEL_CODES = ("run_meta_rows", "run_meta_mismatch", "offgrid", "semantics", "recipe_mismatch",
                         "unreadable_input", "telemetry_clock_format", "telemetry_clock_straddle",
                         "run_end_format", "abort_record_invalid", "run_aborted_other", "run_checks_failed")
# Exit 2 at any time, never a cut.
SWEEP_LEVEL_CODES = (
    "test_path", "duplicate_candidate", "grid_mismatch", "partial_input", "output_exists", "bad_value",
    "rules_missing", "rules_uncommitted", "rules_format", "rules_mismatch",
    "band_missing", "band_uncommitted", "band_format", "band_s_mismatch", "band_mismatch",
    "band_changed_after_launch",
    "lambda_selection_missing", "lambda_selection_uncommitted", "lambda_selection_format",
    "lambda_selection_changed", "lambda_mismatch",
    "schedule_missing", "schedule_uncommitted", "schedule_format", "schedule_changed_after_first_launch",
    "decision_log_missing",
    "launch_log_missing", "launch_log_uncommitted", "launch_log_format", "launch_log_mismatch",
    "launch_log_rewritten", "launch_log_incomplete", "launch_log_head_unavailable",
    "launch_log_report_missing", "launch_log_report_mismatch",
    "decision_record_missing", "decision_record_uncommitted", "decision_record_format",
    "decision_record_not_in_decision_log", "decision_record_schedule_mismatch",
    "decision_record_status_mismatch", "decision_record_hash_mismatch", "decision_record_launch_log_mismatch",
    "decision_record_runs_unended", "decision_record_correction_invalid", "decision_record_rewritten",
    "decision_date_not_ended",
    "repeat_report_missing", "repeat_report_mismatch", "repeat_after_end", "repeat_after_aborted_twice",
    "repeat_overlap",
    "clock_inconsistent", "lambda_push_evidence_missing", "lambda_push_evidence_mismatch",
    "default_candidate_diverged", "default_aborted_twice", "alpha_cut_conflict",
    "code_pin_mismatch", "code_uncommitted", "repository_shallow", "records_mismatch",
    "recipe_mismatch_across_candidates",
    "best_json_format", "run_end_mismatch", "checkpoint_mismatch", "checkpoint_unreadable")
UNEXPECTED_ERROR_EXIT = 4        # the select scripts: exit 0 selected, 2 refused, 3 shortfall, 4 error
SWEEP_CUT_EXIT = 5               # select_alpha: the alpha sweep is cut (AM-16 item 2; AM-19 item 2(g))
# AM-7a: only the trainer's own run_abort record decides that a run diverged.
DIVERGENCE_RULES = ("AM-7(a)", "AM-7(b)")
DIVERGENCE_CAUSE = "student_divergence"
AM7B_WINDOW = 100                # configs/distill.py AM7_DIVERGENCE; load_rules requires agreement
AM7B_FACTOR = 5.0
ABORT_DETAIL_KEYS = ("loss", "grad_norm", "window_mean", "running_min", "ratio", "threshold")
# L-KD-HARDEN item 10b: the recipe of record of every candidate, and the carriers that must be
# identical across the candidates of one sweep.
RECIPE_EXPECT = {"grad_clip_norm": None, "batch_size": 16, "val_interval": 4000, "max_val_batches": None,
                 "used_pretrained": True, "poly_horizon": 80000}
# The teacher's config and resolved model-config hashes join its checkpoint hash (L-CKPT-GUARD): the NMF
# settings, eval_steps among them, come from the config and the installed mmseg base config.
RECIPE_IDENTICAL = ("num_workers", "teacher_provenance.ckpt_sha256", "teacher_provenance.config_sha256",
                    "teacher_provenance.model_cfg_sha256")
TEACHER_HASH_FIELDS = RECIPE_IDENTICAL[1:]
# AM-19 (DL-70). sweep_rules.json must carry the same values (load_rules), so neither file changes alone.
AM19_DL_ENTRY = "DL-70"
SCHEDULES = ("T", "R")
AM19_DECISION_DATES = {"lambda_logit": {"T": "2026-10-19", "R": "2026-12-07"},
                       "alpha_cwd": {"T": "2026-10-22", "R": "2026-12-10"}}
LAUNCH_ORDER = {"lambda_logit": [1, 0.5, 2, 0.25, 4], "alpha_cwd": [50, 25, 100]}       # item 1(e); PL-21
ALPHA_DECISION_DATE_RULE = {"lambda_selection_days": 3}                                 # item 2(a); PL-17
INSTANT = {"timezone": "Asia/Manila", "utc_offset": "+08:00", "end_of_date_utc": "16:00:00",
           "meets": "strictly earlier"}
MANILA = timezone(timedelta(hours=8))        # Asia/Manila keeps no DST: a fixed +08:00, no zoneinfo
STOPPED_EARLY = {"min_seconds": 1200, "val_seconds_factor": 2}                          # item 2(h); PL-8
PUSH_MARGIN_SECONDS = 60                                                                # PL-17 T_lo, T_hi
# PL-16: the values the selection rule depends on, pinned in the code as well as in the rules file. A key a
# sweep does not use is pinned as None (absent), so adding it to the rules file is refused too.
RULE_PINS = {
    "lambda_logit": {"stage": "E2", "seed": 42, "tie": "smallest", "default": None, "boundary": [0.25, 4],
                     "band.kind": "fixed", "band.value": 0.005, "band.floor": None, "band.file": None,
                     "winner_is": "E2 seed 42"},
    "alpha_cwd": {"stage": "E3", "seed": 42, "tie": "default_if_tied_else_smallest", "default": 50,
                  "boundary": [25, 100], "band.kind": "dl27", "band.value": None, "band.floor": 0.005,
                  "band.file": BAND_REL, "winner_is": "E3 seed 42"},
}
LAUNCH_LOG_FORMAT = "kd_launch_log/1"
SCHEDULE_FORMAT = "kd_schedule/1"
RECORD_FORMAT = "sweep_decision_record/1"
# AM-19a reading 10: the push list committed with the activity response (scripts/preflight_distill.py writes it)
PUSH_LIST_FORMAT = "lambda_push_list/1"
PUSH_LIST_KEYS = ("format", "response", "lambda_selection", "adding_commit", "t_lo_utc", "remote", "pushes")
PUSH_ENTRY_KEYS = ("id", "ref", "before", "after", "timestamp", "activity_type", "served", "carries_file",
                   "remote_answer")
BRANCH_DELETION = "branch_deletion"      # GitHub's activity_type for a deleted branch: it pushes no commit
LAUNCH_KEYS = ("format", "event", "run_id", "stage", "seed", "horizon", "sweep", "value", "lambda_logit",
               "alpha_cwd", "attempt", "schedule", "code_pin", "ckpt_dir", "am8a_report")
EVENT_KEYS = {"launched": ("event", "run_id", "run_meta_sha256", "launch_time_utc", "git_head",
                           "records_commit", "decision_date", "launch_order_position"),
              "not_launched": ("event", "run_id", "reason", "evidence"),
              "stopped": ("event", "run_id", "report")}
RUNNING_DEFAULT = "running (item 2(d))"
RUNNING_ON_COURSE = "running (on-course repeat, item 2(b))"
CUT_LABEL = "cut at the decision date (AM-19)"
_COMMIT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class SelectionRefused(RuntimeError):
    """A named refusal; `code` makes every guard testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SweepCut(Exception):
    """AM-16 item 2 with AM-19 item 2(g): no non-default alpha candidate was launched before the end of the
    alpha decision date; the committed decision record states the cut. Not a refusal (exit 5)."""

    def __init__(self, record_sha256: str, message: str):
        super().__init__(message)
        self.record_sha256 = record_sha256


def dec(x) -> Decimal:
    return Decimal(repr(float(x)))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refuse_test_path(p: Path) -> None:
    parts = [part.lower() for part in Path(p).resolve().parts]
    if "test" in parts:
        raise SelectionRefused("test_path", f"refusing {p}: a selection never reads anything under a "
                                            "directory named 'test'")


def read_json(path: Path, *, code: str):
    """json.loads of a file, refused by name when it is not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SelectionRefused(code, f"{path} is not valid JSON ({e})") from e


def parse_json_bytes(data: bytes, what: str, *, code: str):
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SelectionRefused(code, f"{what} is not valid JSON ({e})") from e


def _finite_unit(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) \
            or not 0.0 <= float(x) <= 1.0:
        raise SelectionRefused("bad_value", f"{what} must be a finite number in [0, 1], got {x!r}")
    return float(x)


NONFINITE_TAGS = ("nan", "inf", "-inf")    # the trainer's strict-JSON encoding of a non-finite float


def _finite_number(x) -> bool:
    """A finite int or float as a run_abort record writes one (never a bool or a string). A JSON integer
    too large for a float is not one (math.isfinite raises on it)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _same(got, want) -> bool:
    """Equality for rule pins and launch-log fields: numbers by value (never a bool), lists element-wise,
    objects key by key, anything else by type and ==."""
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(_same(g, w) for g, w in zip(got, want))
    if isinstance(want, dict):
        return isinstance(got, dict) and sorted(got) == sorted(want) and all(_same(got[k], want[k]) for k in want)
    if _finite_number(want):
        return _finite_number(got) and float(got) == float(want)
    return type(got) is type(want) and got == want


def _is_iso_day(x) -> bool:
    if not isinstance(x, str) or _ISO_DAY.fullmatch(x) is None:
        return False
    try:
        date.fromisoformat(x)
    except ValueError:
        return False
    return True


def _am7b_record_holds(it: int, detail: dict, ramp_iters: int) -> bool:
    """Q2: could train_distill's AM7bMonitor have fired with these window fields at iteration `it`? The
    first window closes at ramp_iters + AM7B_WINDOW and only seeds the running minimum, so the earliest
    firing is one iteration later; the trainer fires on window_mean > AM7B_FACTOR x running_min and
    records threshold = AM7B_FACTOR x running_min and ratio = window_mean / running_min ("inf" when
    running_min is 0, and the same encoding when the quotient overflows)."""
    wm, rm, th, ratio = (detail.get(k) for k in ("window_mean", "running_min", "threshold", "ratio"))
    if not (it > ramp_iters + AM7B_WINDOW and _finite_number(wm) and _finite_number(rm) and rm >= 0):
        return False
    if not (wm > AM7B_FACTOR * rm and _finite_number(th) and th == AM7B_FACTOR * rm):
        return False
    if rm == 0:
        return ratio == "inf"
    expected = wm / rm
    return ratio == "inf" if not math.isfinite(expected) else (_finite_number(ratio) and ratio == expected)


def _am7a_record_key(it: int, detail: dict, train_row: dict) -> str | None:
    """Q4: the non-finite quantity of an AM-7 (a) detail ("loss" or "grad_norm"), or None when the record
    fails a check every train_distill record passes: iteration 2 or later (R8-1); a non-finite loss (the
    loss case), or a finite loss and a non-finite grad_norm (the norm case); and the aborting iteration's
    train row naming that key in its `nonfinite` map. K8-2(c): the loss stop precedes backward, so in the
    loss case the record's detail.grad_norm is null and the train row holds the key grad_norm with value
    null, as the trainer writes them."""
    loss, grad_norm = detail.get("loss"), detail.get("grad_norm")
    if loss in NONFINITE_TAGS:
        row_norm_null = isinstance(train_row, dict) and "grad_norm" in train_row \
            and train_row["grad_norm"] is None
        if grad_norm is not None or not row_norm_null:
            return None
        key = "loss"
    elif _finite_number(loss) and grad_norm in NONFINITE_TAGS:
        key = "grad_norm"
    else:
        return None
    named = train_row.get("nonfinite") if isinstance(train_row, dict) else None
    return key if it >= 2 and isinstance(named, dict) and key in named else None


def _dotted(m: dict, dotted: str):
    head, _, rest = dotted.partition(".")
    v = m.get(head)
    if rest:
        return v.get(rest) if isinstance(v, dict) else None
    return v


# --------------------------------------------------------------------------------------- time
def now_utc() -> float:
    """The selection's clock: Unix time, no time zone (the smoke patches it in-process)."""
    return time.time()


def cutoff_instant(day: str) -> int:
    """The end of `day` (YYYY-MM-DD): 24:00 Asia/Manila = 16:00:00 UTC, as Unix time (AM-19 Definitions)."""
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=timezone.utc).timestamp())


def meets(ts, cutoff) -> bool:
    """A timestamp meets a date when it is strictly earlier than the date's end (AM-19 Definitions)."""
    return ts < cutoff


def manila_day(ts) -> date:
    return datetime.fromtimestamp(float(ts), tz=MANILA).date()


def iso_utc(ts) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_activity_time(s) -> float:
    """GitHub's activity timestamp, e.g. 2026-10-05T08:07:42Z, as Unix time."""
    if not isinstance(s, str):
        raise ValueError(f"timestamp {s!r} is not a string")
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------- committed inputs
def _git_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def git_blob_id(data: bytes, object_format: str = "sha1") -> str:
    """The id git gives a blob with these bytes (git hash-object without filters)."""
    return hashlib.new(object_format, b"blob %d\0" % len(data) + data).hexdigest()


class HeadSource:
    """The selection's committed inputs: files of the working tree at HEAD of the repository at `root`
    (default GIT_ROOT). A file is accepted only when it is under the repository, present at HEAD and
    byte-equal to its HEAD blob (PL-16: no index or mode comparison). Every git call is read-only:
    `git -c safe.directory=<root> --no-optional-locks -C <root>`, with GIT_DIR, GIT_WORK_TREE and
    GIT_INDEX_FILE removed from the environment."""
    kind = "head"
    records_ref = "HEAD"        # the commit the records (reports/, the decision log) are read at (PL-28)

    def __init__(self, root=None):
        self.root = Path(GIT_ROOT if root is None else root).resolve()
        self._object_format = None

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-c", f"safe.directory={self.root.as_posix()}", "--no-optional-locks",
                               "-C", str(self.root), *args], capture_output=True, env=_git_env(), check=False)

    def _out(self, *args: str) -> str | None:
        r = self.git(*args)
        return r.stdout.decode("utf-8", "replace").strip() if r.returncode == 0 else None

    def require_not_shallow(self) -> None:
        out = self._out("rev-parse", "--is-shallow-repository")
        if out not in ("true", "false") or self._out("rev-parse", "--verify", "--quiet", "HEAD^{commit}") is None:
            raise SelectionRefused("code_uncommitted", f"no git repository with a HEAD commit at {self.root}: "
                                                       "the selection runs only from a committed checkout")
        if out == "true":
            raise SelectionRefused("repository_shallow", f"the repository at {self.root} is shallow: the "
                                                         "launch log's history and the code pin need the "
                                                         "full history (git fetch --unshallow)")

    def head(self) -> str:
        return self._out("rev-parse", "--verify", "--quiet", "HEAD^{commit}") or ""

    def object_format(self) -> str:
        if self._object_format is None:
            self._object_format = self._out("rev-parse", "--show-object-format") or "sha1"
        return self._object_format

    def rel(self, path) -> str | None:
        """`path` (absolute, or relative to the repository) as a repository-relative POSIX path; None
        when it is not under the repository."""
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return None

    def blob_at(self, commit: str, rel: str) -> str | None:
        return self._out("rev-parse", "--verify", "--quiet", f"{commit}:{rel}")

    def commit_exists(self, sha) -> bool:
        return isinstance(sha, str) and self.git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0

    def is_ancestor(self, ancestor: str, commit: str) -> bool:
        return self.git("merge-base", "--is-ancestor", ancestor, commit).returncode == 0

    def blob_bytes(self, oid: str) -> bytes:
        r = self.git("cat-file", "blob", oid)
        if r.returncode != 0:
            raise RuntimeError(f"git cat-file blob {oid} failed in {self.root}")
        return r.stdout

    def show(self, commit: str, rel: str) -> bytes | None:
        oid = self.blob_at(commit, rel)
        return None if oid is None else self.blob_bytes(oid)

    def has(self, rel: str) -> bool:
        """Whether a committed input exists at `rel` (here: in the working tree)."""
        return (self.root / rel).is_file()

    def read(self, path, *, missing: str, uncommitted: str, record: bool = False) -> tuple[str, bytes]:
        """(rel, bytes) of a committed input; `missing` when absent, `uncommitted` when outside the
        repository, absent at HEAD or not byte-equal to its HEAD blob. `record` matters to RecordsSource
        only (a report the launch log names is a records file wherever it lies)."""
        rel = self.rel(path)
        if rel is None:
            raise SelectionRefused(uncommitted, f"{path} is not under the repository {self.root}: committed "
                                                "inputs are read from the repository only (PL-16)")
        f = self.root / rel
        refuse_test_path(f)
        if not f.is_file():
            raise SelectionRefused(missing, f"{rel} not found in {self.root}")
        data = f.read_bytes()
        oid = self.blob_at("HEAD", rel)
        if oid is None:
            raise SelectionRefused(uncommitted, f"{rel} is not committed (absent at HEAD {self.head()[:12]})")
        if self.blob_bytes(oid) != data:
            raise SelectionRefused(uncommitted, f"{rel} differs from its blob at HEAD {self.head()[:12]}: "
                                                "commit it first (PL-16)")
        return rel, data

    def read_at_head(self, rel: str, *, missing: str) -> bytes:
        """`git show HEAD:<rel>` (the decision log)."""
        data = self.show("HEAD", rel)
        if data is None:
            raise SelectionRefused(missing, f"{rel} does not exist at HEAD {self.head()[:12]}")
        return data


class RecordsSource(HeadSource):
    """The gate's committed inputs (PL-28): the records files (reports/ and docs/DECISION_LOG.md) come from a
    folder written from records commit H, each accepted only when its git blob id, computed from its bytes,
    equals `git rev-parse H:<path>` (records_mismatch); "exists" and "committed" for a record mean at H.
    Code and configs are read at HEAD (the pin), as HeadSource reads them."""
    kind = "records"

    def __init__(self, folder, records_commit: str, root=None):
        super().__init__(root)
        self.folder = Path(folder).resolve()
        self.records_commit = records_commit

    @property
    def records_ref(self) -> str:
        return self.records_commit

    @staticmethod
    def is_record(rel: str) -> bool:
        return rel == DECISION_LOG_REL or rel.startswith("reports/")

    def has(self, rel: str) -> bool:
        if self.is_record(rel):
            return self.blob_at(self.records_commit, rel) is not None
        return super().has(rel)

    def rel(self, path) -> str | None:
        p = Path(path)
        if p.is_absolute():
            try:
                return p.resolve().relative_to(self.folder).as_posix()
            except ValueError:
                pass
        return super().rel(path)

    def _record(self, rel: str, *, missing: str) -> bytes:
        f = self.folder / rel
        refuse_test_path(f)
        at_h = self.blob_at(self.records_commit, rel)
        if not f.is_file():
            if at_h is None:
                raise SelectionRefused(missing, f"{rel} does not exist at the records commit "
                                                f"{self.records_commit[:12]}")
            raise SelectionRefused("records_mismatch", f"{rel} exists at the records commit "
                                                       f"{self.records_commit[:12]} but not in the records "
                                                       f"folder {self.folder}")
        data = f.read_bytes()
        if at_h is None or git_blob_id(data, self.object_format()) != at_h:
            raise SelectionRefused("records_mismatch", f"{f}: its blob id differs from "
                                                       f"{self.records_commit[:12]}:{rel} ({at_h}); the records "
                                                       "folder is not that commit's (PL-28)")
        return data

    def read(self, path, *, missing: str, uncommitted: str, record: bool = False) -> tuple[str, bytes]:
        rel = self.rel(path)
        if rel is not None and (record or self.is_record(rel)):
            return rel, self._record(rel, missing=missing)
        return super().read(path, missing=missing, uncommitted=uncommitted)

    def read_at_head(self, rel: str, *, missing: str) -> bytes:
        if self.is_record(rel):
            return self._record(rel, missing=missing)
        return super().read_at_head(rel, missing=missing)


def require_committed(path, *, missing: str, uncommitted: str, src: HeadSource | None = None,
                      record: bool = False) -> tuple[str, bytes]:
    """PL-16's one helper: `path` is under GIT_ROOT and present at HEAD, and its bytes equal `git show
    HEAD:<path>` (no index or mode comparison); returns (repository path, bytes). It reads through the
    two-source accessor: `src` is HeadSource (the selection; the default) or RecordsSource (the gate,
    PL-28), whose records files come from the folder verified by blob id against the records commit;
    record=True reads `path` as a records file wherever it lies (a report the launch log names)."""
    return (HeadSource() if src is None else src).read(path, missing=missing, uncommitted=uncommitted,
                                                         record=record)


def check_code_committed(src: HeadSource, files) -> None:
    """PL-16: the code the selection runs equals its blob at HEAD (code_uncommitted). `files` are
    (running path, repository path) pairs."""
    for path, rel in files:
        oid = src.blob_at("HEAD", rel)
        if oid is None or src.blob_bytes(oid) != Path(path).read_bytes():
            raise SelectionRefused("code_uncommitted", f"{rel}: the running file {path} is not the blob at "
                                                       f"HEAD {src.head()[:12]}; commit the code first (PL-16)")


def selection_code_files(key: str, script_path) -> list:
    import configs.distill as distill_cfg
    return [(Path(__file__), SWEEP_SELECT_REL), (Path(script_path), SELECT_SCRIPT_REL[key]),
            (Path(distill_cfg.__file__), DISTILL_REL)]


# ------------------------------------------------------------------------------------ rules
def rules_from_bytes(data: bytes, path) -> dict:
    """Parse the rule file and require it to agree with configs/distill.py and this module's AM-19
    constants and rule pins."""
    from configs.distill import AM7_DIVERGENCE, DISTILL

    doc = parse_json_bytes(data, str(path), code="rules_format")
    if not isinstance(doc, dict) or doc.get("format") != "sweep_rules/1":
        got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise SelectionRefused("rules_format", f"{path}: format {got!r} != 'sweep_rules/1'")
    lam, alp = doc.get("lambda_logit"), doc.get("alpha_cwd")
    if not isinstance(lam, dict) or not isinstance(alp, dict):
        raise SelectionRefused("rules_format", f"{path}: needs 'lambda_logit' and 'alpha_cwd' blocks")
    for name, sweep in (("lambda_logit", lam), ("alpha_cwd", alp)):
        for key in ("stage", "seed", "iterations", "grid", "band", "tie", "boundary", "winner_is",
                    "default_candidate"):
            if key not in sweep:
                raise SelectionRefused("rules_format", f"{path}: '{name}' lacks '{key}'")
        if sweep["stage"] not in STAGE_TERMS:
            raise SelectionRefused("rules_format", f"{path}: '{name}' stage {sweep['stage']!r}")
        dc = sweep["default_candidate"]
        if isinstance(dc, bool) or not isinstance(dc, (int, float)) \
                or float(dc) not in [float(v) for v in sweep["grid"]]:
            raise SelectionRefused("rules_format", f"{path}: '{name}' default_candidate {dc!r} is not a "
                                                   "value on its grid")
    lk, cwd = DISTILL["logit_kd"], DISTILL["cwd"]
    expect = [
        ("lambda grid", [float(v) for v in lam.get("grid", [])],
         [float(v) for v in lk["lambda_logit_sweep_grid"]]),
        ("lambda band", lam.get("band", {}).get("value"), lk["lambda_sweep_tie_band_pp"] / 100.0),
        ("lambda iterations", lam.get("iterations"), lk["lambda_sweep_iters_per_candidate"]),
        ("lambda seed", lam.get("seed"), lk["sweep_seed"]),
        ("lambda default candidate", lam.get("default_candidate"), lk["lambda_default_candidate"]),
        ("alpha grid", [float(v) for v in alp.get("grid", [])], [float(v) for v in cwd["alpha_cwd_grid"]]),
        ("alpha default", alp.get("default"), cwd["alpha_cwd_feature_map"]),
        ("alpha default candidate", alp.get("default_candidate"), cwd["alpha_cwd_feature_map"]),
        ("alpha iterations", alp.get("iterations"), cwd["alpha_sweep_iters_per_candidate"]),
    ]
    bad = [(what, got, want) for what, got, want in expect if got != want]
    if bad:
        raise SelectionRefused("rules_mismatch", f"{path} disagrees with configs/distill.py: {bad}")
    am7b = {"window": (AM7B_WINDOW, AM7_DIVERGENCE["window"]),
            "factor": (AM7B_FACTOR, AM7_DIVERGENCE["factor"])}
    if any(mine != cfg for mine, cfg in am7b.values()):
        raise SelectionRefused("rules_mismatch", f"sweep_select's AM-7 (b) constants disagree with "
                                                 f"configs/distill.py AM7_DIVERGENCE: {am7b} (here, config)")
    _check_am19_rules(doc, path)
    return doc


def _check_am19_rules(doc: dict, path) -> None:
    """AM-19 (lane K2): the decision dates of both schedules, the launch order (PL-21: the literal
    constants), the alpha date rule, the date instant and the stopped-early constants, each equal to this
    module's constants; and the rule pins (PL-16)."""
    for top in ("instant", "stopped_early"):
        if not isinstance(doc.get(top), dict):
            raise SelectionRefused("rules_format", f"{path}: needs an '{top}' object (AM-19)")
    for name in ("lambda_logit", "alpha_cwd"):
        sweep = doc[name]
        dates = sweep.get("decision_dates")
        if not isinstance(dates, dict) or sorted(dates) != sorted(SCHEDULES) \
                or not all(_is_iso_day(dates[s]) for s in SCHEDULES):
            raise SelectionRefused("rules_format", f"{path}: '{name}' needs decision_dates {{\"T\": YYYY-MM-DD, "
                                                   "\"R\": YYYY-MM-DD}} (AM-19 item 1(a))")
        order = sweep.get("launch_order")
        if not isinstance(order, list) or not all(_finite_number(v) for v in order):
            raise SelectionRefused("rules_format", f"{path}: '{name}' needs a launch_order list of numbers "
                                                   "(AM-19 item 1(e))")
    if not isinstance(doc["alpha_cwd"].get("decision_date_rule"), dict):
        raise SelectionRefused("rules_format", f"{path}: 'alpha_cwd' needs a decision_date_rule object "
                                               "(AM-19 item 2(a))")
    bad = []
    for name in ("lambda_logit", "alpha_cwd"):
        sweep = doc[name]
        if sweep["decision_dates"] != AM19_DECISION_DATES[name]:
            bad.append((f"{name} decision_dates", sweep["decision_dates"], AM19_DECISION_DATES[name]))
        if not _same(sweep["launch_order"], LAUNCH_ORDER[name]):
            bad.append((f"{name} launch_order", sweep["launch_order"], LAUNCH_ORDER[name]))
        for pin, want in RULE_PINS[name].items():
            got = _dotted(sweep, pin)
            if not _same(got, want):
                bad.append((f"{name} {pin}", got, want))
    if not _same(doc["alpha_cwd"]["decision_date_rule"], ALPHA_DECISION_DATE_RULE):
        bad.append(("alpha_cwd decision_date_rule", doc["alpha_cwd"]["decision_date_rule"],
                    ALPHA_DECISION_DATE_RULE))
    for top, want in (("instant", INSTANT), ("stopped_early", STOPPED_EARLY)):
        if not _same(doc[top], want):
            bad.append((top, doc[top], want))
    if bad:
        raise SelectionRefused("rules_mismatch", f"{path} disagrees with the AM-19 constants or the rule pins "
                                                 f"of src/training/sweep_select.py: {bad} (got, want)")


def load_rules(path: Path = RULES_PATH) -> dict:
    """Read a rule file and require it to agree with configs/distill.py (no git: rule units)."""
    path = Path(path)
    if not path.is_file():
        raise SelectionRefused("rules_missing", f"rule file not found: {path}")
    return rules_from_bytes(path.read_bytes(), path)


def load_committed_rules(src: HeadSource, rules=None) -> tuple[str, dict]:
    """The committed rule file (--rules accepts only a committed path under the repository; PL-16)."""
    rel, data = require_committed(RULES_REL if rules is None else rules, src=src, missing="rules_missing",
                                  uncommitted="rules_uncommitted")
    return rel, rules_from_bytes(data, rel)


def dl27_band(doc: dict, floor: float) -> tuple[float, float, list[str]]:
    """(s, band, trace) from the DL-27 band entry: band = max(floor, sqrt(2) * s), s the n=3 sample
    SD of E1's best VAL all-class mIoU over seeds 42, 43, 44 (AM-16 item 2).

    Required keys: e1_best_val {"42", "43", "44"} and s. s is recomputed from the three values and must
    agree to 1e-12; a recorded band must agree with the recomputed one to 1e-12."""
    if not isinstance(doc, dict):
        raise SelectionRefused("band_format", f"the band file must hold a JSON object, got "
                                              f"{type(doc).__name__}")
    vals = doc.get("e1_best_val")
    if not isinstance(vals, dict) or sorted(vals) != ["42", "43", "44"]:
        raise SelectionRefused("band_format", "the band file needs e1_best_val for seeds '42', '43', "
                                              f"'44'; got {sorted(vals) if isinstance(vals, dict) else vals!r}")
    xs = [_finite_unit(vals[k], f"e1_best_val[{k}]") for k in ("42", "43", "44")]
    s_file = doc.get("s")
    if isinstance(s_file, bool) or not isinstance(s_file, (int, float)) or not math.isfinite(s_file) \
            or s_file < 0:
        raise SelectionRefused("band_format", f"the band file's s must be finite and >= 0, got {s_file!r}")
    s_calc = statistics.stdev(xs)
    if abs(s_calc - float(s_file)) > 1e-12:
        raise SelectionRefused("band_s_mismatch", f"recorded s {s_file!r} != the sample SD of "
                                                  f"{xs} = {s_calc!r}")
    s = float(s_file)
    band = max(float(floor), math.sqrt(2.0) * s)
    if "band" in doc and abs(float(doc["band"]) - band) > 1e-12:
        raise SelectionRefused("band_mismatch", f"recorded band {doc['band']!r} != max({floor}, "
                                                f"sqrt(2)*{s!r}) = {band!r}")
    trace = [f"band: s = {s!r} (sample SD, n=3, E1 seeds 42-44 = {xs}); sqrt(2)*s = "
             f"{math.sqrt(2.0) * s!r}; band = max({floor}, sqrt(2)*s) = {band!r}"
             + (" (the 0.5 pp floor applies)" if band == float(floor) else "")]
    return s, band, trace


# --------------------------------------------------------------------------------- schedule
def _dl_has_row(text: str, entry: str) -> bool:
    return re.search(rf"(?m)^\|\s*{re.escape(entry)}\s*\|", text) is not None


def load_schedule(src: HeadSource) -> dict:
    """PL-12: configs/kd_schedule.json, committed, with a valid format and in_force, naming a row of the
    committed decision log (for Schedule R, not AM-19's own row)."""
    rel, data = require_committed(SCHEDULE_REL, src=src, missing="schedule_missing",
                                  uncommitted="schedule_uncommitted")
    doc = parse_json_bytes(data, rel, code="schedule_format")
    entry = doc.get("decision_log_entry") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != SCHEDULE_FORMAT or doc.get("in_force") not in SCHEDULES \
            or not (isinstance(entry, str) and re.fullmatch(r"DL-\d+", entry)):
        raise SelectionRefused("schedule_format", f"{rel}: needs format {SCHEDULE_FORMAT!r}, in_force 'T' or "
                                                  "'R' and a decision_log_entry 'DL-<n>' (AM-19 item 1(b))")
    dl = src.read_at_head(DECISION_LOG_REL, missing="decision_log_missing").decode("utf-8", "replace")
    if not _dl_has_row(dl, entry):
        raise SelectionRefused("schedule_format", f"{rel}: {entry} is not a row of the committed "
                                                  f"{DECISION_LOG_REL}")
    if doc["in_force"] == "R" and entry == AM19_DL_ENTRY:
        raise SelectionRefused("schedule_format", f"{rel}: Schedule R is in force only by its own "
                                                  f"decision-log entry (AM-19 item 1(b)); {entry} is AM-19's "
                                                  "own row")
    return {"in_force": doc["in_force"], "decision_log_entry": entry, "file": rel,
            "file_sha256": sha256_bytes(data), "blob": src.blob_at("HEAD", rel), "decision_log": dl}


def check_schedule_at_launches(src: HeadSource, schedule: dict, log: "LaunchLog") -> None:
    """PL-12: for every launched line, the schedule file's blob at its git_head equals HEAD's (the schedule
    does not change after the first KD launch); a head the repository does not hold refuses."""
    for line, _raw in log.entries:
        if line.get("event") != "launched":
            continue
        head = line["git_head"]
        if not src.commit_exists(head):
            raise SelectionRefused("launch_log_head_unavailable", f"{LAUNCH_LOG_REL}: launched line of "
                                                                  f"{line['run_id']} names git_head {head}, "
                                                                  "which this repository does not hold")
        if src.blob_at(head, SCHEDULE_REL) != schedule["blob"]:
            raise SelectionRefused("schedule_changed_after_first_launch",
                                   f"{SCHEDULE_REL} at {head[:12]} (the launch of {line['run_id']}) differs "
                                   "from HEAD's: the schedule in force changes only by a new amendment once "
                                   "the first KD run has launched (AM-19 item 1(b), PL-12)")


# ------------------------------------------------------------------------------- launch log
def _is_report_ref(x) -> bool:
    return isinstance(x, dict) and sorted(x) == ["path", "sha256"] and isinstance(x["path"], str) \
        and bool(x["path"]) and isinstance(x["sha256"], str) and _SHA256.fullmatch(x["sha256"]) is not None


class LaunchLog:
    """reports/kd_launch_log.jsonl (PL-13), parsed and checked for structure (launch_log_format): one JSON
    object per line; launch lines with unique run_ids, attempt = 1 + the earlier launch lines with the same
    (stage, seed, horizon, lambda_logit, alpha_cwd) and basename(ckpt_dir) = run_id; every other event
    follows its run's launch line; at most one launched line, never beside a not_launched line; a stopped
    line follows a launched line."""

    def __init__(self, data: bytes, rel: str = LAUNCH_LOG_REL):
        self.data, self.rel = data, rel
        self.sha256 = sha256_bytes(data)
        self.entries: list[tuple[dict, bytes]] = []
        self.launch: dict[str, dict] = {}
        self.raw: dict[str, bytes] = {}
        self.order: list[str] = []
        self.launched: dict[str, dict] = {}
        self.not_launched: dict[str, dict] = {}
        self.stopped: dict[str, dict] = {}
        self._attempts: dict[tuple, int] = {}
        self._parse()

    @property
    def n_lines(self) -> int:
        return len(self.entries)

    def _bad(self, msg: str):
        raise SelectionRefused("launch_log_format", f"{self.rel}: {msg}")

    def _parse(self) -> None:
        if not self.data:
            return
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError as e:
            self._bad(f"not UTF-8 ({e})")
        if not self.data.endswith(b"\n"):
            self._bad("the last line is torn (the file does not end with a newline)")
        for n, raw in enumerate(self.data.split(b"\n")[:-1], 1):
            try:
                line = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as e:
                self._bad(f"line {n} is not JSON ({e})")
            if not isinstance(line, dict):
                self._bad(f"line {n} is not a JSON object")
            event = line.get("event")
            if event == "launch":
                self._launch_line(n, line, raw)
            elif isinstance(event, str) and event in EVENT_KEYS:
                self._event_line(n, event, line)
            else:
                self._bad(f"line {n}: unknown event {event!r}")
            self.entries.append((line, raw))

    def _launch_line(self, n: int, line: dict, raw: bytes) -> None:
        if sorted(line) != sorted(LAUNCH_KEYS):
            self._bad(f"line {n}: a launch line has exactly the keys {list(LAUNCH_KEYS)}")
        rid = line["run_id"]
        if line["format"] != LAUNCH_LOG_FORMAT:
            self._bad(f"line {n}: format {line['format']!r} != {LAUNCH_LOG_FORMAT!r}")
        if not isinstance(rid, str) or not rid or rid in (".", "..") or "/" in rid or "\\" in rid:
            self._bad(f"line {n}: run_id {rid!r}")
        if rid in self.launch:
            self._bad(f"line {n}: a second launch line for run_id {rid}")
        ok = (isinstance(line["stage"], str) and line["stage"] and _is_int(line["seed"])
              and _is_int(line["horizon"]) and line["horizon"] > 0
              and line["sweep"] in (None, "lambda_logit", "alpha_cwd")
              and ((line["sweep"] is None and line["value"] is None)
                   or (line["sweep"] is not None and _finite_number(line["value"])))
              and all(line[k] is None or _finite_number(line[k]) for k in ("lambda_logit", "alpha_cwd"))
              and _is_int(line["attempt"]) and line["attempt"] >= 1 and line["schedule"] in SCHEDULES
              and isinstance(line["code_pin"], str) and _COMMIT_ID.fullmatch(line["code_pin"]) is not None
              and isinstance(line["ckpt_dir"], str)
              and (line["am8a_report"] is None or _is_report_ref(line["am8a_report"])))
        if not ok:
            self._bad(f"line {n}: a launch line field has the wrong type or value: {raw[:240]!r}")
        if PurePosixPath(line["ckpt_dir"]).name != rid:
            self._bad(f"line {n}: basename(ckpt_dir) {PurePosixPath(line['ckpt_dir']).name!r} != run_id {rid!r}")
        num = (lambda x: None if x is None else float(x))  # noqa: E731
        group = (line["stage"], line["seed"], line["horizon"], num(line["lambda_logit"]), num(line["alpha_cwd"]))
        expected = self._attempts.get(group, 0) + 1
        if line["attempt"] != expected:
            self._bad(f"line {n}: {rid} attempt {line['attempt']} != {expected} (1 + the earlier launch lines "
                      "with the same stage, seed, horizon, lambda_logit and alpha_cwd)")
        self._attempts[group] = expected
        self.launch[rid], self.raw[rid] = line, raw
        self.order.append(rid)

    def _event_line(self, n: int, event: str, line: dict) -> None:
        if sorted(line) != sorted(EVENT_KEYS[event]):
            self._bad(f"line {n}: a {event} line has exactly the keys {list(EVENT_KEYS[event])}")
        rid = line["run_id"]
        if not isinstance(rid, str) or not rid:
            self._bad(f"line {n}: a {event} line's run_id {rid!r} is not a run_id")
        if rid not in self.launch:
            self._bad(f"line {n}: {event} line for {rid!r} follows no launch line")
        if event == "launched":
            if rid in self.launched or rid in self.not_launched:
                self._bad(f"line {n}: a second launched or not_launched line for {rid}")
            ok = (isinstance(line["run_meta_sha256"], str) and _SHA256.fullmatch(line["run_meta_sha256"])
                  and _finite_number(line["launch_time_utc"])
                  and all(isinstance(line[k], str) and _COMMIT_ID.fullmatch(line[k])
                          for k in ("git_head", "records_commit"))
                  and (line["decision_date"] is None or _is_iso_day(line["decision_date"]))
                  and (line["launch_order_position"] is None
                       or (_is_int(line["launch_order_position"]) and line["launch_order_position"] >= 1)))
            if not ok:
                self._bad(f"line {n}: a launched line field has the wrong type or value")
            self.launched[rid] = line
        elif event == "not_launched":
            if rid in self.launched or rid in self.not_launched:
                self._bad(f"line {n}: a second launched or not_launched line for {rid}")
            if not isinstance(line["reason"], str) or not _is_report_ref(line["evidence"]):
                self._bad(f"line {n}: a not_launched line needs a reason and evidence {{path, sha256}}")
            self.not_launched[rid] = line
        else:
            if rid not in self.launched:
                self._bad(f"line {n}: a stopped line for {rid} follows no launched line")
            if rid in self.stopped or not _is_report_ref(line["report"]):
                self._bad(f"line {n}: a second stopped line for {rid}, or no report {{path, sha256}}")
            self.stopped[rid] = line


def load_launch_log(src: HeadSource) -> "LaunchLog":
    rel, data = require_committed(LAUNCH_LOG_REL, src=src, missing="launch_log_missing",
                                  uncommitted="launch_log_uncommitted")
    return LaunchLog(data, rel)


class Attempt:
    """One launch line of the sweep: a grid value's attempt (PL-9)."""

    def __init__(self, log: LaunchLog, run_id: str):
        self.line, self.raw = log.launch[run_id], log.raw[run_id]
        self.run_id = run_id
        self.value = float(self.line["value"])
        self.attempt = self.line["attempt"]
        self.launched = log.launched.get(run_id)
        self.not_launched = log.not_launched.get(run_id)
        self.stopped = log.stopped.get(run_id)
        self.pos = log.order.index(run_id)
        self.run: Run | None = None

    @property
    def launch_ts(self) -> float | None:
        return None if self.launched is None else float(self.launched["launch_time_utc"])


def sweep_attempts(log: LaunchLog, sweep: dict, key: str, schedule: dict) -> dict:
    """{grid value: [Attempt in log order]}. Every launch of the sweep's stage, seed and horizon is a
    candidate of the sweep and is marked so; its value is on the grid (grid_mismatch) and equals its
    lambda_logit (lambda) or alpha_cwd (alpha); it ran under the schedule in force."""
    grid = [float(v) for v in sweep["grid"]]
    out = {v: [] for v in grid}
    canon = (sweep["stage"], sweep["seed"], sweep["iterations"])
    for rid in log.order:
        line = log.launch[rid]
        member = (line["stage"], line["seed"], line["horizon"]) == canon
        if line["sweep"] != key and not member:
            continue
        if line["sweep"] != key or not member:
            raise SelectionRefused("launch_log_format", f"{LAUNCH_LOG_REL}: {rid}: a launch of {canon} is a "
                                                        f"candidate of the {key} sweep and carries sweep "
                                                        f"{key!r}, and only such a launch does")
        if key == "lambda_logit":
            ok = line["alpha_cwd"] is None and _same(line["value"], line["lambda_logit"])
        else:
            ok = line["lambda_logit"] is not None and _same(line["value"], line["alpha_cwd"])
        if not ok:
            raise SelectionRefused("launch_log_format", f"{LAUNCH_LOG_REL}: {rid}: value {line['value']!r} "
                                                        f"does not match its {key} field")
        if float(line["value"]) not in grid:
            raise SelectionRefused("grid_mismatch", f"{rid}: {key} {line['value']!r} not in the registered "
                                                    f"grid {grid}")
        if line["schedule"] != schedule["in_force"]:
            raise SelectionRefused("launch_log_mismatch", f"{rid} was launched under Schedule "
                                                          f"{line['schedule']}; Schedule {schedule['in_force']} "
                                                          "is in force")
        out[float(line["value"])].append(Attempt(log, rid))
    return out


def _check_report(src: HeadSource, ref: dict, *, missing: str, mismatch: str) -> None:
    p = PurePosixPath(ref["path"])
    if p.is_absolute() or ".." in p.parts:
        raise SelectionRefused(mismatch, f"report path {ref['path']!r} is not repository-relative")
    rel, data = require_committed(ref["path"], src=src, missing=missing, uncommitted=mismatch, record=True)
    if sha256_bytes(data) != ref["sha256"]:
        raise SelectionRefused(mismatch, f"{rel}: sha256 {sha256_bytes(data)} != {ref['sha256']} in the "
                                         "launch log")


def check_reports(src: HeadSource, atts: dict) -> None:
    """PL-16: every report the sweep's lines name is committed with its sha256: the AM-8a report of a repeat
    (repeat_report_*), a not_launched line's evidence and a stopped line's report (launch_log_report_*)."""
    for a in _in_log_order(atts):
        if a.line["am8a_report"] is not None:
            _check_report(src, a.line["am8a_report"], missing="repeat_report_missing",
                          mismatch="repeat_report_mismatch")
        if a.not_launched is not None:
            _check_report(src, a.not_launched["evidence"], missing="launch_log_report_missing",
                          mismatch="launch_log_report_mismatch")
        if a.stopped is not None:
            _check_report(src, a.stopped["report"], missing="launch_log_report_missing",
                          mismatch="launch_log_report_mismatch")


def _in_log_order(atts: dict) -> list:
    return sorted((a for v in atts for a in atts[v]), key=lambda a: a.pos)


def check_code_pin(src: HeadSource, atts: dict, rules_rel: str, key: str) -> str | None:
    """PL-14 and PL-16: one code_pin over the sweep's launch lines and one git_head over its launched lines,
    the same commit; the blobs of the selection's code, the rules file and the schedule file at the pin
    equal HEAD's (master's other code may move: no whole-tree comparison)."""
    every = [a for v in atts for a in atts[v]]
    pins = {a.line["code_pin"] for a in every}
    heads = {a.launched["git_head"] for a in every if a.launched is not None}
    if len(pins) > 1 or len(heads) > 1 or (heads and heads != pins):
        raise SelectionRefused("code_pin_mismatch", f"the {key} sweep's launches carry code pins {sorted(pins)} "
                                                    f"and git heads {sorted(heads)}: a sweep runs from one pin")
    if not pins:
        return None
    pin = pins.pop()
    if not src.commit_exists(pin):
        raise SelectionRefused("launch_log_head_unavailable", f"the {key} sweep's code pin {pin} is not in "
                                                              "this repository")
    for rel in (SWEEP_SELECT_REL, SELECT_SCRIPT_REL[key], DISTILL_REL, rules_rel, SCHEDULE_REL):
        if src.blob_at(pin, rel) != src.blob_at("HEAD", rel):
            raise SelectionRefused("code_pin_mismatch", f"{rel} at HEAD {src.head()[:12]} differs from the "
                                                        f"sweep's code pin {pin[:12]} (PL-16)")
    return pin


def check_log_prefix(src: HeadSource, log: LaunchLog, cache: dict) -> None:
    """PL-14: for every launched line, the launch log at its records_commit is a byte prefix of HEAD's
    (the log is append-only); a records commit without a launch log refuses too (PL-13: it holds the run's
    launch line). Selection only: a pod's partial clone holds no old blobs."""
    for line, _raw in log.entries:
        if line.get("event") != "launched":
            continue
        h = line["records_commit"]
        if not src.commit_exists(h):
            raise SelectionRefused("launch_log_head_unavailable", f"{line['run_id']}: records_commit {h} is not "
                                                                  "in this repository")
        if h not in cache:
            cache[h] = src.show(h, LAUNCH_LOG_REL)
        if cache[h] is None:
            raise SelectionRefused("launch_log_rewritten", f"{line['run_id']}: its records commit {h[:12]} holds no "
                                                           f"{LAUNCH_LOG_REL} (PL-13: it holds the run's launch line)")
        if not log.data.startswith(cache[h]):
            raise SelectionRefused("launch_log_rewritten", f"{LAUNCH_LOG_REL} at records commit {h[:12]} "
                                                           f"({line['run_id']}) is not a prefix of HEAD's: the "
                                                           "launch log is append-only")


# ------------------------------------------------------------------------------- candidates
def _parse_run_meta(meta_p: Path) -> dict:
    rows = []
    for n, ln in enumerate(meta_p.read_text(encoding="utf-8").splitlines(), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError as e:
                raise SelectionRefused("unreadable_input", f"{meta_p}: line {n} is not JSON ({e})") from e
    if len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("event") != "run_meta":
        raise SelectionRefused("run_meta_rows", f"{meta_p}: {len(rows)} run_meta rows; exactly one is "
                                                "required (more means an appended relaunch)")
    return rows[0]


def _check_run_meta(run_dir: Path, m: dict, sweep: dict, key: str) -> None:
    stage = sweep["stage"]
    want = {"mode": "real", "seed": sweep["seed"], "stage": stage, "max_iters": sweep["iterations"],
            "terms": STAGE_TERMS[stage]}
    wrong = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if wrong:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta {wrong} (got, want)")
    if key not in m:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta has no {key!r}")
    if key == "alpha_cwd" and m.get("alpha_offgrid"):
        raise SelectionRefused("offgrid", f"{run_dir.name}: trained with --allow-offgrid (tests only)")
    if STAGE_TERMS[stage]["logit_kd"]:
        # lambda_logit is only comparable under one Logit-KD semantics (B32c-2), in either sweep
        from configs.distill import LOGIT_KD_SEMANTICS
        if m.get("logit_kd_semantics") != LOGIT_KD_SEMANTICS or m.get("logit_kd_semantics_override_used"):
            raise SelectionRefused("semantics", f"{run_dir.name}: Logit-KD semantics "
                                                f"{m.get('logit_kd_semantics')!r} / override "
                                                f"{m.get('logit_kd_semantics_override_used')!r}")
    bad = {k: (m.get(k, "<absent>"), v) for k, v in RECIPE_EXPECT.items()
           if k not in m or m[k] != v or type(m[k]) is not type(v)}
    if bad:
        raise SelectionRefused("recipe_mismatch", f"{run_dir.name}: run_meta departs from the recipe of "
                                                  f"record {bad} (got, want)")
    shared = {f: _dotted(m, f) for f in RECIPE_IDENTICAL}
    if not (isinstance(shared["num_workers"], int) and not isinstance(shared["num_workers"], bool)
            and all(isinstance(shared[f], str) and shared[f] for f in TEACHER_HASH_FIELDS)):
        raise SelectionRefused("recipe_mismatch", f"{run_dir.name}: run_meta lacks a comparable "
                                                  f"{list(RECIPE_IDENTICAL)}: {shared}")
    ramp = m.get("ramp_iters")
    if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 1:
        raise SelectionRefused("run_meta_mismatch", f"{run_dir.name}: run_meta ramp_iters {ramp!r}")


def _scan_telemetry(tel_p: Path) -> tuple[list[dict], bool]:
    """(rows, torn_last). A torn LAST line marks a run cut mid-write; a torn line elsewhere is a
    damaged file and is refused."""
    lines = tel_p.read_text(encoding="utf-8").splitlines()
    rows, torn_last = [], False
    for n, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            if n == len(lines):
                torn_last = True
                break
            raise SelectionRefused("unreadable_input", f"{tel_p}: line {n} is not JSON ({e})") from e
        rows.append(row if isinstance(row, dict) else {})
    return rows, torn_last


def _row_ts(row: dict) -> float | None:
    """A telemetry row's timestamp (AM-19 Definitions): wall_clock, or wall_clock_end in run_end."""
    v = row.get("wall_clock_end" if row.get("event") == "run_end" else "wall_clock")
    return float(v) if _finite_number(v) else None


def _abort_ending(run_dir: Path, rows: list[dict], torn_last: bool, ramp_iters) -> tuple:
    """Validate the run's run_abort record (AM-7a): ("diverged", record, None) for a student divergence,
    ("aborted_other", record, its run_aborted_other refusal) for another abort; abort_record_invalid is
    raised for a record the trainer cannot have written."""
    events = [r.get("event") for r in rows]
    aborts = [i for i, e in enumerate(events) if e == "run_abort"]
    train_iters = [r.get("iter") for r in rows if r.get("event") == "train"]
    ab = rows[aborts[-1]]
    if len(aborts) != 1 or aborts[0] != len(rows) - 1 or torn_last or "run_end" in events:
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: a run_abort record must be the "
                                                       "single last telemetry row, never beside a run_end")
    detail = ab.get("detail")
    if not (isinstance(ab.get("iter"), int) and isinstance(ab.get("rule"), str)
            and isinstance(ab.get("cause"), str) and isinstance(detail, dict)
            and all(k in detail for k in ABORT_DETAIL_KEYS)):
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: malformed run_abort record "
                                                       f"{json.dumps(ab)[:200]}")
    if not train_iters or ab["iter"] != train_iters[-1]:
        raise SelectionRefused("abort_record_invalid", f"{run_dir.name}: run_abort at iter {ab['iter']} "
                                                       f"but the last train row is iter "
                                                       f"{train_iters[-1] if train_iters else None}")
    if ab["rule"] == "AM-7(b)" and (ramp_iters is None or not _am7b_record_holds(ab["iter"], detail, ramp_iters)):
        raise SelectionRefused("abort_record_invalid",
                               f"{run_dir.name}: an AM-7 (b) abort at iter {ab['iter']} with detail "
                               f"{json.dumps(detail)[:240]} cannot hold: it needs iter > ramp_iters + "
                               f"{AM7B_WINDOW} = {None if ramp_iters is None else ramp_iters + AM7B_WINDOW}, a "
                               f"finite window_mean above {AM7B_FACTOR} x a finite running_min >= 0, threshold = "
                               f"{AM7B_FACTOR} x running_min and ratio = window_mean / running_min (\"inf\" at 0 "
                               "or on overflow)")
    if ab["rule"] == "AM-7(a)":
        train_rows = [r for r in rows if r.get("event") == "train"]
        key = _am7a_record_key(ab["iter"], detail, train_rows[-1])
        if key is None:
            raise SelectionRefused("abort_record_invalid",
                                   f"{run_dir.name}: an AM-7 (a) abort at iter {ab['iter']} with detail "
                                   f"{json.dumps(detail)[:240]} cannot hold: it needs iter >= 2 (iteration "
                                   "1 is step1_checks) and either a non-finite loss with grad_norm null in "
                                   "the detail and in that train row (K8-2(c)), or a finite loss with a "
                                   "non-finite grad_norm; that key named in the train row's `nonfinite` map")
    if ab["rule"] not in DIVERGENCE_RULES or ab["cause"] != DIVERGENCE_CAUSE \
            or ab.get("input_finite") is not True or ab.get("teacher_finite") is not True:
        return "aborted_other", ab, SelectionRefused(
            "run_aborted_other",
            f"{run_dir.name}: the run aborted at iter {ab['iter']} under rule {ab['rule']!r} with cause "
            f"{ab['cause']!r} (input_finite={ab.get('input_finite')!r}, teacher_finite="
            f"{ab.get('teacher_finite')!r}). This is not a student divergence under AM-7 (a)/(b). STOP: "
            "investigate; AM-8a governs a repeat; never excluded, never a shortfall.")
    return "diverged", ab, None


def _run_end_ending(run_dir: Path, rows: list[dict], max_iters: int) -> tuple[str, dict]:
    """PL-6: a run_end record is the single last telemetry row, after the last train row at max_iters, with
    iter = max_iters and a boolean checks_passed; a finished run's best_val_miou_all_class (the score of
    record) is finite in [0, 1] and its best_ckpt a file name or null (run_end_format otherwise)."""
    ends = [i for i, r in enumerate(rows) if r.get("event") == "run_end"]
    train_iters = [r.get("iter") for r in rows if r.get("event") == "train"]
    re_ = rows[ends[-1]]
    if len(ends) != 1 or ends[0] != len(rows) - 1:
        raise SelectionRefused("run_end_format", f"{run_dir.name}: a run_end record must be the single last "
                                                 "telemetry row")
    if not train_iters or train_iters[-1] != max_iters or re_.get("iter") != max_iters:
        raise SelectionRefused("run_end_format", f"{run_dir.name}: run_end at iter {re_.get('iter')!r} with "
                                                 f"the last train row at {train_iters[-1] if train_iters else None}"
                                                 f"; a finished run reaches {max_iters}")
    checks = re_.get("checks_passed")
    if not isinstance(checks, bool):
        raise SelectionRefused("run_end_format", f"{run_dir.name}: run_end checks_passed {checks!r}")
    if not checks:
        return "checks_failed", re_
    best, ck = re_.get("best_val_miou_all_class"), re_.get("best_ckpt")
    if not (_finite_number(best) and 0.0 <= float(best) <= 1.0) or not (ck is None or (isinstance(ck, str) and ck)):
        raise SelectionRefused("run_end_format", f"{run_dir.name}: run_end best_val_miou_all_class {best!r} / "
                                                 f"best_ckpt {ck!r}: the score of record is finite in [0, 1]")
    return "finished", re_


class Run:
    """One run directory read without a cutoff (read_run). Candidate-level faults are kept, not raised:
    before the decision date, and for the default candidate, they refuse; after it they cut a non-default
    candidate (PL-5)."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.run_id = self.dir.name
        self.missing = None              # a SHORTFALL code: candidate_missing / candidate_incomplete
        self.meta = None                 # the run_meta row, when the file holds exactly one
        self.meta_fault = self.tel_fault = self.end_fault = None
        self.ending = None               # {"kind": finished|checks_failed|diverged|aborted_other, "ts", ...}
        self.end_refusal = None          # the refusal of an aborted_other or checks_failed ending
        self.rows: list[dict] = []
        self.row_ts: list = []
        self.torn_last = False
        self.launch_ts = None
        self.run_meta_sha256 = self.telemetry_sha256 = None
        self.clock_backsteps, self.max_backstep_seconds = 0, 0.0
        self.artifacts = None
        self.artifact_fault = None

    @property
    def raw_end(self) -> bool:
        """The telemetry holds a trainer end record (run_end or run_abort), valid or not."""
        return any(r.get("event") in ("run_end", "run_abort") for r in self.rows)

    def timestamps(self) -> list:
        """The run's timestamps in file order: the run_meta's wall_clock, then each row's."""
        return ([] if self.launch_ts is None else [self.launch_ts]) + [t for t in self.row_ts if t is not None]

    def last_row_ts(self, before=None):
        """The timestamp of the run's last row in file order (any event, trainer records included), among
        the rows earlier than `before` when given; the launch time when there is none."""
        ts = [t for t in self.row_ts if t is not None and (before is None or t < before)]
        return ts[-1] if ts else self.launch_ts

    def last_train_iter(self, before=None):
        its = [r.get("iter") for r, t in zip(self.rows, self.row_ts)
               if r.get("event") == "train" and (before is None or (t is not None and t < before))]
        return its[-1] if its else None


def read_run(run_dir: Path, sweep: dict, key: str) -> Run:
    """Read a run directory (no cutoff): run_meta, telemetry rows with their timestamps, the trainer's end
    record. Nothing raises except test_path; faults are kept on the Run."""
    run = Run(run_dir)
    refuse_test_path(run.dir)
    if not run.dir.is_dir():
        run.missing = "candidate_missing"
        return run
    sk = sweep["stage"].lower()
    meta_p, tel_p = run.dir / f"{sk}_run_meta.jsonl", run.dir / f"{sk}_telemetry.jsonl"
    if meta_p.is_file():
        run.run_meta_sha256 = sha256_file(meta_p)
        try:
            run.meta = _parse_run_meta(meta_p)
            _check_run_meta(run.dir, run.meta, sweep, key)
        except SelectionRefused as e:
            run.meta_fault = e
        wc = run.meta.get("wall_clock") if isinstance(run.meta, dict) else None
        if _finite_number(wc):
            run.launch_ts = float(wc)
        elif run.meta_fault is None:
            run.meta_fault = SelectionRefused("telemetry_clock_format", f"{run.run_id}: run_meta wall_clock "
                                                                        f"{wc!r} (the launch time) is not a "
                                                                        "finite timestamp")
    else:
        run.missing = "candidate_incomplete"
    if not tel_p.is_file():
        run.missing = run.missing or "candidate_incomplete"
        return run
    run.telemetry_sha256 = sha256_file(tel_p)
    try:
        run.rows, run.torn_last = _scan_telemetry(tel_p)
    except SelectionRefused as e:
        run.tel_fault = e
        return run
    run.row_ts = [_row_ts(r) for r in run.rows]
    bad = [(r.get("event"), r.get("iter")) for r, t in zip(run.rows, run.row_ts) if t is None]
    if bad:
        run.tel_fault = SelectionRefused("telemetry_clock_format", f"{run.run_id}: row(s) {bad[:5]} lack a finite "
                                                                   "timestamp (wall_clock; wall_clock_end in "
                                                                   "run_end)")
        return run
    seq = run.timestamps()
    steps = [a - b for a, b in zip(seq, seq[1:]) if b < a]
    run.clock_backsteps, run.max_backstep_seconds = len(steps), (max(steps) if steps else 0.0)
    _interpret_ending(run, sweep)
    return run


def _interpret_ending(run: Run, sweep: dict) -> None:
    rows = run.rows
    ramp = run.meta.get("ramp_iters") if isinstance(run.meta, dict) else None
    ramp = ramp if (_is_int(ramp) and ramp >= 1) else None
    if any(r.get("event") == "run_abort" for r in rows):
        try:
            kind, ab, refusal = _abort_ending(run.dir, rows, run.torn_last, ramp)
        except SelectionRefused as e:
            run.end_fault = e
            return
        run.ending = {"kind": kind, "ts": _row_ts(ab), "abort": ab, "rule": ab["rule"], "cause": ab["cause"]}
        run.end_refusal = refusal
        return
    # A row the trainer writes just before an abort (a train row with a `nonfinite` map, or a val row whose
    # all-class mIoU is non-finite) with no run_abort record after it: the record is missing or torn.
    bad = [(r.get("event"), r.get("iter"), t) for r, t in zip(rows, run.row_ts)
           if "nonfinite" in r and (r.get("event") == "train"
                                    or (r.get("event") == "val" and "all_class_miou" in r["nonfinite"]))]
    if bad:
        run.ending = {"kind": "aborted_other", "ts": bad[-1][2], "rule": "nonfinite_row",
                      "cause": "abort_record_missing", "abort": None}
        run.end_refusal = SelectionRefused(
            "run_aborted_other",
            f"{run.run_id}: row(s) {[b[:2] for b in bad]} carry the non-finite values of an abort, with no "
            "run_abort record after them (the abort record is missing or torn). STOP: investigate; AM-8a "
            "governs a repeat; never excluded, never a shortfall.")
        return
    if run.torn_last:
        return
    if any(r.get("event") == "run_end" for r in rows):
        try:
            kind, re_ = _run_end_ending(run.dir, rows, sweep["iterations"])
        except SelectionRefused as e:
            run.end_fault = e
            return
        run.ending = {"kind": kind, "ts": _row_ts(re_), "record": re_}
        if kind == "checks_failed":
            run.end_refusal = SelectionRefused("run_checks_failed", f"{run.run_id}: the run's own checks did "
                                                                    "not pass (run_end checks_passed=False)")


def straddle_fault(run: Run, cutoff):
    """PL-10, AM-19a reading 4: for a non-default attempt, a timestamp earlier than C that follows, in file
    order, one at or after C. Other backsteps are only counted (clock_backsteps, max_backstep_seconds)."""
    if cutoff is None:
        return None
    seen = False
    for t in run.timestamps():
        if t >= cutoff:
            seen = True
        elif seen:
            return SelectionRefused("telemetry_clock_straddle",
                                    f"{run.run_id}: a timestamp before the end of the decision date follows one "
                                    "at or after it (file order): its clock cannot place the run at the date "
                                    "(AM-19 item 2(b); PL-10)")
    return None


def run_fault(run: Run, cutoff=None):
    """The run's first candidate-level fault: run_meta, telemetry, the straddle across `cutoff` (a
    non-default attempt), then its end record."""
    return run.meta_fault or run.tel_fault or straddle_fault(run, cutoff) or run.end_fault


def read_artifacts(run: Run, sweep: dict, key: str) -> dict:
    """PL-6: best.json and the checkpoint of a finished run (best.json only, for best_val_partial, of a
    diverged one). Absent: null hashes, checkpoint_present false. Present: they must parse and agree with
    run_end (best_json_format, bad_value, run_end_mismatch, checkpoint_unreadable, checkpoint_mismatch)."""
    out = {"best_json_sha256": None, "ckpt_sha256": None, "checkpoint": None, "checkpoint_present": False,
           "best_val_partial": None}
    best_p = run.dir / BEST_JSON
    best = None
    if best_p.is_file():
        best = read_json(best_p, code="best_json_format")
        if not isinstance(best, dict) or sorted(best) != BEST_JSON_KEYS:
            got = sorted(best) if isinstance(best, dict) else best
            raise SelectionRefused("best_json_format", f"{best_p}: keys {got!r} != {BEST_JSON_KEYS}")
        if not isinstance(best["best_ckpt"], str) or not best["best_ckpt"]:
            raise SelectionRefused("best_json_format", f"{best_p}: best_ckpt {best['best_ckpt']!r}")
        bv = _finite_unit(best["best_val_miou_all_class"], f"{best_p} best_val_miou_all_class")
        out["best_json_sha256"] = sha256_file(best_p)
        if run.ending["kind"] == "diverged":
            out["best_val_partial"] = bv
    if run.ending["kind"] != "finished":
        return out
    re_ = run.ending["record"]
    best_val, ck_name = float(re_["best_val_miou_all_class"]), re_.get("best_ckpt")
    if best is not None:
        if best["best_val_miou_all_class"] != re_["best_val_miou_all_class"] \
                or Path(best["best_ckpt"]).name != ck_name:
            raise SelectionRefused("run_end_mismatch", f"{run.run_id}: run_end records best "
                                                       f"{re_['best_val_miou_all_class']!r} / {ck_name!r}, "
                                                       f"best.json {best['best_val_miou_all_class']!r} / "
                                                       f"{Path(best['best_ckpt']).name!r}")
        ck_path = Path(best["best_ckpt"])
        if not ck_path.is_file():
            ck_path = run.dir / ck_path.name
    else:
        ck_path = None if ck_name is None else run.dir / Path(ck_name).name
    if ck_path is None:
        return out
    refuse_test_path(ck_path)
    if not ck_path.is_file():
        return out
    import torch
    try:
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 — any unreadable checkpoint is refused by name
        raise SelectionRefused("checkpoint_unreadable", f"{run.run_id}: {ck_path.name} could not be "
                                                        f"read ({type(e).__name__}: {e})") from e
    stage = sweep["stage"]
    if not isinstance(ck, dict) or ck.get("stage") != stage or ck.get("best_val_miou_all_class") != best_val:
        raise SelectionRefused("checkpoint_mismatch", f"{run.run_id}: checkpoint stage "
                                                      f"{ck.get('stage') if isinstance(ck, dict) else '?'!r} / best "
                                                      f"{ck.get('best_val_miou_all_class') if isinstance(ck, dict) else '?'!r}"
                                                      f" do not match {stage} / {best_val!r}")
    value = float(run.meta[key])
    if key in ck and float(ck[key]) != value:
        raise SelectionRefused("checkpoint_mismatch", f"{run.run_id}: checkpoint {key} {ck[key]!r} "
                                                      f"!= run_meta {value!r}")
    out.update(ckpt_sha256=sha256_file(ck_path), checkpoint=ck_path.name, checkpoint_present=True)
    return out


def candidate_record(run: Run, key: str) -> dict:
    """The finished or diverged candidate as apply_rule and the selection file use it."""
    m = run.meta
    art = run.artifacts or {}
    common = {"value": float(m[key]), "run_id": run.run_id, "run_dir": str(run.dir),
              "run_meta_sha256": run.run_meta_sha256, "telemetry_sha256": run.telemetry_sha256,
              "lambda_logit": m.get("lambda_logit"), "num_workers": m["num_workers"],
              "teacher_ckpt_sha256": m["teacher_provenance"]["ckpt_sha256"],
              "teacher_config_sha256": m["teacher_provenance"]["config_sha256"],
              "teacher_model_cfg_sha256": m["teacher_provenance"]["model_cfg_sha256"],
              "ramp_iters": m["ramp_iters"], "best_json_sha256": art.get("best_json_sha256")}
    if run.ending["kind"] == "diverged":
        ab = run.ending["abort"]
        return {"status": "diverged", **common,
                "abort": {"iter": ab["iter"], "rule": ab["rule"], "cause": ab["cause"], "detail": ab["detail"]},
                "n_val": ab.get("n_val"), "best_val_partial": art.get("best_val_partial")}
    return {"status": "finished", **common, "best_val": float(run.ending["record"]["best_val_miou_all_class"]),
            "ckpt_sha256": art.get("ckpt_sha256"), "checkpoint": art.get("checkpoint"),
            "checkpoint_present": bool(art.get("checkpoint_present"))}


def load_candidate(run_dir: Path, sweep: dict, key: str, *, cutoff=None) -> dict:
    """One run directory on its own (rule units): {status: finished|diverged, ...}, else its refusal: a
    candidate-level fault, a shortfall code (candidate_missing, candidate_incomplete, run_unfinished), its
    abort or failed check, or a best.json or checkpoint fault. With `cutoff` (a non-default candidate after
    its decision date) only an end record that meets it counts, and the straddle rule applies."""
    run = read_run(Path(run_dir), sweep, key)
    if run.missing == "candidate_missing":
        raise SelectionRefused("candidate_missing", f"candidate directory not found: {run_dir}")
    fault = run_fault(run, cutoff)
    if fault is not None:
        raise fault
    if run.missing is not None:
        raise SelectionRefused("candidate_incomplete", f"{run.run_id}: its run_meta or telemetry is missing")
    e = run.ending
    if e is None or (cutoff is not None and e["ts"] >= cutoff):
        raise SelectionRefused("run_unfinished", f"{run.run_id}: no run_end or run_abort record"
                                                 + ("" if cutoff is None else " before the cutoff"))
    if run.end_refusal is not None:
        raise run.end_refusal
    run.artifacts = read_artifacts(run, sweep, key)
    return candidate_record(run, key)


def missing_grid_values(cands: list[dict], sweep: dict) -> list[float]:
    have = {c["value"] for c in cands}
    return [float(v) for v in sweep["grid"] if float(v) not in have]


# ------------------------------------------------------------------------- AM-19 item 2 statuses
def on_course(run: Run, cutoff, max_iters: int) -> tuple[bool, dict]:
    """PL-7, AM-19a reading 3: a stopped attempt was on course iff t_last + remaining iterations x
    median(iter_seconds) + remaining validations x max(val_seconds) < C, in Decimal(repr()), from its own
    rows only (all earlier than C, no run_end, no run_abort). t_last is its last row's timestamp; the
    remaining iterations are max_iters minus its last train iteration; the remaining validations are the
    validation points (every multiple of the VAL interval and max_iters, each once) with no VAL row. No train
    row or no VAL row: not on course; nor is a run whose last row has no finite timestamp or a VAL row whose
    iter is no integer."""
    trains = [r for r in run.rows if r.get("event") == "train"]
    vals = [r for r in run.rows if r.get("event") == "val"]
    if not trains or not vals:
        return False, {"on_course": False, "reason": "no train row" if not trains else "no VAL row"}
    iter_s = [r.get("iter_seconds") for r in trains]
    val_s = [r.get("val_seconds") for r in vals]
    if not all(_finite_number(x) for x in iter_s + val_s) or not _is_int(trains[-1].get("iter")):
        return False, {"on_course": False, "reason": "an iter_seconds, val_seconds or iter value is not a "
                                                     "finite number"}
    t_last = run.row_ts[-1] if len(run.row_ts) == len(run.rows) else None
    if not _finite_number(t_last) or not all(_is_int(r.get("iter")) for r in vals):
        return False, {"on_course": False, "reason": "the last row has no finite timestamp, or a VAL row's iter "
                                                     "is not an integer"}
    last_iter = trains[-1]["iter"]
    interval = RECIPE_EXPECT["val_interval"]
    points = sorted(set(range(interval, max_iters + 1, interval)) | {max_iters})
    have = {r.get("iter") for r in vals}
    rem_iters = max_iters - last_iter
    rem_vals = sum(1 for p in points if p not in have)
    med = statistics.median([dec(x) for x in iter_s])
    vmax = max(dec(x) for x in val_s)
    proj = dec(t_last) + rem_iters * med + rem_vals * vmax
    ok = proj < dec(cutoff)
    return ok, {"on_course": ok, "t_last": t_last, "last_train_iter": last_iter, "remaining_iterations": rem_iters,
                "median_iter_seconds": str(med), "remaining_validations": rem_vals, "max_val_seconds": str(vmax),
                "projection": str(proj), "cutoff_utc": cutoff}


def stopped_early(att: Attempt, cutoff, later: list) -> tuple[bool, dict]:
    """PL-8: a cut non-default value's latest attempt launched before C is marked "stopped early,
    unexplained" when its last row (any event, trainer records included) is more than max(1200, 2 x its
    largest val_seconds) seconds before C (1200 without a VAL row; strictly more), no stopped line names
    it, and no later attempt exists (so no AM-8a report names it)."""
    run = att.run
    vals = [r.get("val_seconds") for r in run.rows if r.get("event") == "val" and _finite_number(r.get("val_seconds"))]
    threshold = max(dec(STOPPED_EARLY["min_seconds"]),
                    dec(STOPPED_EARLY["val_seconds_factor"]) * max((dec(v) for v in vals), default=dec(0)))
    t_last = run.last_row_ts()
    gap = None if t_last is None else dec(cutoff) - dec(t_last)
    marked = gap is not None and gap > threshold and att.stopped is None and not later
    return marked, {"run_id": att.run_id, "last_row_utc": t_last, "seconds_before_cutoff": None if gap is None
                    else str(gap), "threshold_seconds": str(threshold), "stopped_line": att.stopped is not None,
                    "later_attempt": bool(later)}


def _same_ending(p: Run | None, q: Run | None) -> bool:
    """AM-19a reading 9: the same (rule, cause) of an abort that is no student divergence, or a failed check
    twice."""
    if p is None or q is None or p.ending is None or q.ending is None:
        return False
    a, b = p.ending, q.ending
    if a["kind"] == b["kind"] == "checks_failed":
        return True
    return a["kind"] == b["kind"] == "aborted_other" and (a["rule"], a["cause"]) == (b["rule"], b["cause"])


def _aborted_twice(atts: list, a: Attempt) -> bool:
    """PL-9(b) with AM-19a reading 9: `a` ends the same way as any earlier launched attempt of its value; the
    attempts between them, a stopped one included, change nothing."""
    launched = [x for x in atts if x.launched is not None]
    i = launched.index(a)
    return any(_same_ending(p.run, a.run) for p in launched[:i])


def _twice_undecided(atts: list, a: Attempt) -> Attempt | None:
    """AM-19a reading 18 with readings 9 and 17: `a`, a non-default value's latest attempt, ended in an abort or
    a failed check that no complete earlier attempt shares; before C the value waits, not refused, while an
    earlier attempt p (one without a not_launched line) is incomplete (no launched line yet, its run directory
    not supplied, or no telemetry yet) and some completion of it could make `a`'s ending its second same one.
    Returns the first such p in launch order, None otherwise. None when `a`'s launch line names no AM-8a report
    (p launched would be repeat_report_missing; p not launched changes nothing). A p with no launched line is
    passed over when an attempt after it, up to and including `a`, names no AM-8a report: p launched would make
    that attempt repeat_report_missing, so no completion of p waits (AM-19a reading 18). A p supplied with a
    stopped line is complete (a stopped run writes no further row); a p with a stopped line cannot end as `a`
    did when `a` ended in a trainer record (a stopped run has none)."""
    if a.line["am8a_report"] is None:
        return None
    i = atts.index(a)
    for k, p in enumerate(atts[:i]):
        if p.not_launched is not None:
            continue
        if p.run is not None and (p.run.missing is None or p.stopped is not None):
            continue
        if p.stopped is not None and a.run.raw_end:
            continue
        if p.launched is None and any(x.line["am8a_report"] is None for x in atts[k + 1:i + 1]):
            continue
        return p
    return None


def _twice_label(run: Run) -> str:
    e = run.ending
    return "aborted_twice:run_end/checks_failed" if e["kind"] == "checks_failed" \
        else f"aborted_twice:{e['rule']}/{e['cause']}"


def _ending_reason(run: Run) -> str:
    e = run.ending
    return "run_checks_failed" if e["kind"] == "checks_failed" else f"run_aborted_other:{e['rule']}/{e['cause']}"


def _status(status: str, *, value: float, attempt: Attempt | None = None, reason: str | None = None,
            code: str | None = None, refusal=None, on_course_inputs: list | None = None,
            running: Attempt | None = None, repeat_of: Attempt | None = None) -> dict:
    """A grid value's status. on_course: every on-course evaluation made for the value after C (PL-7,
    recorded with its inputs; None before C and for the default); running: the on-course repeat that is
    still running (write_decision_record excepts it, AM-19a reading 7); repeat_of: the on-course stop s whose repeat
    decided the status (AM-19 item 2(b): the repeat's VAL rows are reported beside the stopped run's)."""
    return {"status": status, "value": value, "attempt": attempt, "reason": reason, "code": code,
            "refusal": refusal, "on_course": on_course_inputs, "running": running, "repeat_of": repeat_of}


def resolve_latest(value: float, atts: list, *, default: bool, cutoff) -> dict:
    """Before C, and for the default candidate at any time (never cut, item 2(d)): the status of the value's
    latest attempt, read without a cutoff (item 2(c): the status of record is that of the latest repeat)."""
    if not atts:
        return _status("waiting", value=value, code="candidate_missing", reason="no launch line")
    a = atts[-1]
    if a.launched is None:
        why = "was not launched (no run_meta)" if a.not_launched is not None else "has no launched line yet"
        return _status("waiting", value=value, attempt=a, code="candidate_missing",
                       reason=f"attempt {a.attempt} ({a.run_id}) {why}")
    run = a.run
    if run is None:
        return _status("waiting", value=value, attempt=a, code="candidate_missing",
                       reason=f"{a.run_id}: run directory not supplied")
    fault = run_fault(run, None if default else cutoff)
    if fault is not None:
        return _status("refused", value=value, attempt=a, refusal=fault)
    if run.missing is not None:
        return _status("waiting", value=value, attempt=a, code=run.missing,
                       reason=f"{a.run_id}: its telemetry is not written yet")
    e = run.ending
    if e is None:
        return _status("waiting", value=value, attempt=a, code="run_unfinished",
                       reason=f"{a.run_id}: no run_end and no run_abort record"
                              + (" (the last telemetry line is torn)" if run.torn_last else ""))
    if e["kind"] in ("finished", "diverged"):
        return _status(e["kind"], value=value, attempt=a)
    if _aborted_twice(atts, a):
        label = _twice_label(run)
        if default:
            return _status("aborted_twice", value=value, attempt=a, reason=label, refusal=SelectionRefused(
                "default_aborted_twice", f"the default candidate = {value:g} ({a.run_id}) ended the same way as "
                                         f"an earlier attempt ({label}): a STOP that a new amendment settles "
                                         "(AM-19 item 3(a); AM-19a reading 9); no selection is made"))
        # AM-19a reading 17: not a refusal before C (a refusal invites a repeat PL-9(b) refuses); it waits
        return _status("waiting", value=value, attempt=a, reason=f"{a.run_id}: {label}: not repeated again; cut "
                                                                 "once the decision date has ended (AM-19 item 3(a); "
                                                                 "AM-19a readings 9 and 17)")
    p = None if default else _twice_undecided(atts, a)
    if p is not None:
        # AM-19a readings 17 and 18: a refusal only when the complete log would refuse too; with an earlier
        # attempt incomplete, this ending may yet be its second same one, which waits
        why = "has no launched line yet" if p.launched is None else ("is not supplied" if p.run is None
                                                                      else "has no telemetry yet")
        return _status("waiting", value=value, attempt=a, code="candidate_missing" if p.run is None else p.run.missing,
                       reason=f"{a.run_id}: {_ending_reason(run)}; an earlier attempt ({p.run_id}) {why}, so "
                              "whether it ended the same way is not known yet (AM-19a readings 9, 17 and 18)")
    return _status("refused", value=value, attempt=a, refusal=run.end_refusal)


def _on_course_stops(atts: list, cutoff, max_iters: int) -> tuple[list, Attempt | None]:
    """PL-7, AM-19a reading 7: (the on-course evaluations of a value's stops in launch order, up to the first
    one on course; that stop s, or None). A stop is an attempt launched before C whose run is supplied, has no
    candidate-level fault, is written, and has no trainer record (no run_end, no run_abort, no row an abort
    writes) and no row at or after C."""
    evaluated: list = []
    for s in atts:
        rs = s.run
        if s.launched is None or s.launch_ts >= cutoff or rs is None or run_fault(rs, cutoff) is not None \
                or rs.missing is not None or rs.raw_end or rs.ending is not None or any(t >= cutoff for t in rs.row_ts):
            continue
        ok, inputs = on_course(rs, cutoff, max_iters)
        evaluated.append({"run_id": s.run_id, **inputs})
        if ok:
            return evaluated, s
    return evaluated, None


def resolve_after_cutoff(value: float, atts: list, cutoff, max_iters: int) -> dict:
    """PL-9(e): a non-default value's status of record once the decision date has ended. (1) finished, if
    an attempt launched before C is finished at C; (2) diverged, if the latest attempt launched before C
    has a valid divergence record that meets C; (3) the on-course exception, once per value (AM-19a reading
    7): s is the first attempt launched before C that ended before C with no trainer record (no run_end, no
    run_abort, no row an abort writes, no row at or after C) while on course; its repeat r, read without a
    cutoff, decides (finished, diverged, cut for a stop, a later attempt or its own ending, else waiting while
    it has no ending, its telemetry not yet written included), and when the value's latest attempt, after r, ended
    before C the same way as an earlier attempt (AM-19a readings 9 and 17), each cut of (3) takes that ending twice
    as its reason; (4) otherwise cut, with the reason of the latest attempt launched before C. Every on-course
    evaluation is kept with its inputs (PL-7)."""
    evaluated: list = []
    before = [a for a in atts if a.launched is not None and a.launch_ts < cutoff]
    if not before:
        return _status("cut", value=value, reason="never_launched", on_course_inputs=evaluated)
    for a in before:
        r = a.run
        if run_fault(r, cutoff) is None and r.missing is None and r.ending is not None \
                and r.ending["kind"] == "finished" and meets(r.ending["ts"], cutoff):
            return _status("finished", value=value, attempt=a, on_course_inputs=evaluated)
    last = before[-1]
    r = last.run
    if run_fault(r, cutoff) is None and r.missing is None and r.ending is not None \
            and r.ending["kind"] == "diverged" and meets(r.ending["ts"], cutoff):
        return _status("diverged", value=value, attempt=last, on_course_inputs=evaluated)
    evaluated, s = _on_course_stops(atts, cutoff, max_iters)
    if s is not None:
        later = atts[atts.index(s) + 1:]
        rep = next((x for x in later if x.launched is not None), None)
        if rep is None:
            return _status("waiting", value=value, attempt=s, reason=f"{s.run_id} stopped on course: its repeat "
                                                                     "is waited for once (AM-19 item 2(b))",
                           on_course_inputs=evaluated)
        rr = rep.run
        # AM-19a reading 17: when the value's latest attempt, after r, ended before C the same way as an earlier one,
        # every cut of (3) takes that ending twice as its reason (r is read without a cutoff, reading 7; the attempts
        # after it are not)
        twice = _latest_twice(atts, rep, cutoff)
        fault = run_fault(rr, None)
        if fault is not None:
            return _status("cut", value=value, attempt=rep, reason=twice or fault.code, on_course_inputs=evaluated,
                           repeat_of=s)
        if rr.missing is None and rr.ending is not None and rr.ending["kind"] in ("finished", "diverged"):
            return _status(rr.ending["kind"], value=value, attempt=rep, on_course_inputs=evaluated, repeat_of=s)
        if rr.missing is None and rr.ending is not None:      # AM-19a reading 9: against every earlier attempt
            reason = _twice_label(rr) if _aborted_twice(atts, rep) else _ending_reason(rr)
            return _status("cut", value=value, attempt=rep, reason=twice or reason, on_course_inputs=evaluated,
                           repeat_of=s)
        if rep.stopped is not None or later.index(rep) < len(later) - 1:
            return _status("cut", value=value, attempt=rep, reason=twice or "on_course_repeat_stopped",
                           on_course_inputs=evaluated, repeat_of=s)
        return _status("waiting", value=value, attempt=rep, reason=f"{rep.run_id}: the on-course repeat of "
                                                                   f"{s.run_id} is still running (AM-19 item "
                                                                   "2(b))", on_course_inputs=evaluated,
                       running=rep, repeat_of=s)
    return _status("cut", value=value, attempt=last, reason=_cut_reason(atts, last, cutoff),
                   on_course_inputs=evaluated)


def _latest_twice(atts: list, rep: Attempt, cutoff) -> str | None:
    """AM-19a reading 17 in the on-course path (3): the value's latest launched attempt, when it is not the
    on-course repeat `rep` and it ended before C (a trainer record that meets C, with no fault under the cutoff, so
    it also launched before C) the same way as an earlier attempt (reading 9): that ending twice. None otherwise:
    an ending at or after C changes nothing (AM-19 item 2(b); reading 14)."""
    lastl = [x for x in atts if x.launched is not None][-1]
    lr = lastl.run
    if lastl is rep or lr is None or run_fault(lr, cutoff) is not None or lr.missing is not None \
            or lr.ending is None or lr.ending["kind"] in ("finished", "diverged") \
            or not meets(lr.ending["ts"], cutoff) or not _aborted_twice(atts, lastl):
        return None
    return _twice_label(lr)


def _cut_reason(atts: list, last: Attempt, cutoff) -> str:
    r = last.run
    fault = run_fault(r, cutoff)
    if fault is not None:
        return fault.code
    if r.missing is not None:
        return r.missing
    e = r.ending
    if e is not None:
        if not meets(e["ts"], cutoff):
            return "finished_after_date" if e["kind"] == "finished" else "aborted_after_date"
        return _twice_label(r) if _aborted_twice(atts, last) else _ending_reason(r)
    return "still_running" if any(t >= cutoff for t in r.row_ts) else "stopped_no_finished_repeat"


# ------------------------------------------------------------------------ launch-log agreement
def check_run_against_log(src: HeadSource, a: Attempt, key: str, log_at: dict) -> None:
    """PL-14: a supplied run directory has a launched line; its run_meta file's sha256 equals the line's;
    run_meta.git_head equals the line's git_head and the launch line's code_pin; run_meta.records_commit
    equals the line's records_commit; stage, seed, horizon, value, lambda_logit, alpha_cwd and the launch
    time equal run_meta's; the log at that records commit holds its launch line; basename(ckpt_dir) is the
    directory's name; a stopped line names only a run without a trainer end record (launch_log_mismatch)."""
    rid, line, ld, run = a.run_id, a.line, a.launched, a.run
    if ld is None:
        raise SelectionRefused("launch_log_mismatch", f"{rid}: a supplied run directory needs a launched line in "
                                                      f"{LAUNCH_LOG_REL} (PL-14); this attempt has "
                                                      + ("a not_launched line" if a.not_launched else "none"))
    if run.run_meta_sha256 != ld["run_meta_sha256"]:
        raise SelectionRefused("launch_log_mismatch", f"{rid}: run_meta sha256 {run.run_meta_sha256} != "
                                                      f"{ld['run_meta_sha256']} in its launched line")
    m = run.meta
    if isinstance(m, dict):
        fields = [("git_head", m.get("git_head"), ld["git_head"]), ("code_pin", ld["git_head"], line["code_pin"]),
                  ("records_commit", m.get("records_commit"), ld["records_commit"]),
                  ("stage", m.get("stage"), line["stage"]), ("seed", m.get("seed"), line["seed"]),
                  ("horizon", m.get("poly_horizon"), line["horizon"]), (key, m.get(key), line["value"]),
                  ("lambda_logit", m.get("lambda_logit"), line["lambda_logit"]),
                  ("alpha_cwd", m.get("alpha_cwd"), line["alpha_cwd"]),
                  ("launch_time_utc", m.get("wall_clock"), ld["launch_time_utc"])]
        bad = [(f, got, want) for f, got, want in fields if not _same(got, want)]
        if bad:
            raise SelectionRefused("launch_log_mismatch", f"{rid}: run_meta disagrees with its launch-log lines: "
                                                          f"{bad} (run_meta, log)")
    h = ld["records_commit"]
    if h not in log_at:
        log_at[h] = src.show(h, LAUNCH_LOG_REL)
    if a.raw not in (log_at[h] or b"").split(b"\n"):
        raise SelectionRefused("launch_log_mismatch", f"{rid}: its launch line is not in {LAUNCH_LOG_REL} at its "
                                                      f"records commit {h[:12]}")
    if PurePosixPath(line["ckpt_dir"]).name != run.dir.name:
        raise SelectionRefused("launch_log_mismatch", f"{rid}: basename(ckpt_dir) "
                                                      f"{PurePosixPath(line['ckpt_dir']).name!r} != the directory "
                                                      f"{run.dir.name!r}")
    if a.stopped is not None and run.raw_end:
        raise SelectionRefused("launch_log_mismatch", f"{rid}: a stopped line names a run whose telemetry ends in a "
                                                      "trainer record (PL-13: stopped is for a run that ended "
                                                      "without one)")


def check_repeats(atts: dict, key: str) -> None:
    """PL-9, sweep-level at any time (never a cut): two final attempts (finished or validly diverged) of one
    value (duplicate_candidate); an attempt after a final one (repeat_after_end: a run_end with its checks
    passed, or a valid divergence; AM-19a reading 1); an attempt after a launched attempt without its AM-8a
    report (repeat_report_missing); an attempt launched not later than the previous launched attempt's largest
    timestamp (repeat_overlap; AM-19a reading 12); an attempt after one that ended the same way as any earlier launched
    attempt of its value (repeat_after_aborted_twice; AM-19a reading 9: the attempts between them change
    nothing)."""
    def final(a):
        return a.run is not None and a.run.ending is not None and a.run.ending["kind"] in ("finished", "diverged")

    for v, lst in atts.items():
        fin = [a.run_id for a in lst if final(a)]
        if len(fin) > 1:
            raise SelectionRefused("duplicate_candidate", f"more than one finished or diverged run for {key} "
                                                          f"[{v:g}]: {fin}")
    for v, lst in atts.items():
        for i, a in enumerate(lst[:-1]):
            if final(a):
                what = "a run_end with its checks passed" if a.run.ending["kind"] == "finished" \
                    else "a valid divergence record"
                raise SelectionRefused("repeat_after_end", f"{a.run_id} ({key} = {v:g}) ended in {what}, and a later "
                                                           f"attempt {lst[i + 1].run_id} has a launch line: a "
                                                           "finished or diverged candidate is final and is never "
                                                           "repeated (AM-19 items 2(b) and 2(h); AM-19a reading 1)")
    for v, lst in atts.items():
        prior = False
        for a in lst:
            if prior and a.line["am8a_report"] is None:
                raise SelectionRefused("repeat_report_missing", f"{a.run_id} ({key} = {v:g}, attempt {a.attempt}) "
                                                                "follows a launched attempt, and its launch line "
                                                                "names no AM-8a report (AM-19 item 3(a); PL-9(c))")
            prior = prior or a.launched is not None
    for v, lst in atts.items():
        launched = [a for a in lst if a.launched is not None]
        for p, q in zip(launched, launched[1:]):
            # AM-19a reading 12: the largest timestamp of p. A withheld p: its launched line gives its launch time, one
            # of its timestamps, so its largest timestamp is no earlier (reading 18: every completion refuses when q
            # launched no later)
            last = p.launch_ts if p.run is None else max(p.run.timestamps(), default=p.launch_ts)
            if q.launch_ts <= last:
                what = ("the largest timestamp" if p.run is not None
                        else "the launch time (its directory is not supplied)")
                raise SelectionRefused("repeat_overlap", f"{q.run_id} launched at {iso_utc(q.launch_ts)}, not after "
                                                         f"{what} of {p.run_id} ({iso_utc(last)}): a repeat starts "
                                                         "after the attempt it repeats has stopped (AM-19a reading "
                                                         "12; PL-9(d))")
    for v, lst in atts.items():
        launched = [a for a in lst if a.launched is not None]
        for j, q in enumerate(launched):
            p = next((x for x in launched[:j] if _same_ending(x.run, q.run)), None)
            i = lst.index(q)
            if p is not None and i < len(lst) - 1:
                raise SelectionRefused("repeat_after_aborted_twice", f"{lst[i + 1].run_id} follows {q.run_id}, which "
                                                                     f"ended the same way as {p.run_id} "
                                                                     f"({_twice_label(q.run)}): such a run is not "
                                                                     "repeated again (AM-19 item 3(a); PL-9(b); "
                                                                     "AM-19a reading 9)")


def check_recipe_across(cands: list[dict]) -> None:
    """Q3 and PL-11: the finished and diverged candidates' RECIPE_IDENTICAL values agree, before a
    divergence is read; a difference is never attributed to one candidate (AM-19a reading 5)."""
    for field in ("num_workers", "teacher_ckpt_sha256", "teacher_config_sha256", "teacher_model_cfg_sha256"):
        seen = {repr(c[field]) for c in cands}
        if len(seen) > 1:
            raise SelectionRefused("recipe_mismatch_across_candidates",
                                   f"the candidates differ in {field}: {sorted(seen)} (a sweep's runs share one "
                                   "value; the difference refuses the sweep and is never attributed to one "
                                   "candidate, AM-19a reading 5)")


def check_shared_lambda(atts: dict, lam: float) -> None:
    """PL-20(a): every alpha attempt, finished, diverged, cut or still running, carries the lambda
    selection's winner: run_meta's lambda_logit, or the launch line's when no run_meta row is readable."""
    got = []
    for a in _in_log_order(atts):
        m = a.run.meta if a.run is not None and isinstance(a.run.meta, dict) else None
        got.append(m.get("lambda_logit") if m is not None else a.line["lambda_logit"])
    if any(not _finite_number(x) or float(x) != lam for x in got):
        raise SelectionRefused("lambda_mismatch", f"the alpha runs must share one lambda_logit equal to the lambda "
                                                  f"selection's winner {lam!r}; got {sorted({repr(x) for x in got})}")


def refuse_diverged_default(diverged: list[dict], sweep: dict, key: str) -> None:
    """AM-7a: a diverged default candidate keeps AM-7 in full; no selection is made."""
    default = float(sweep["default_candidate"])
    for c in diverged:
        if c["value"] == default:
            raise SelectionRefused(
                "default_candidate_diverged",
                f"the default candidate {key} = {default:g} (run {c['run_id']}) diverged at iter "
                f"{c['abort']['iter']} under {c['abort']['rule']}: AM-7 applies in full: stop the stage, "
                "apply the AM-7a item 5 clipping value, rerun the FP32 stages; no selection is made")


# ------------------------------------------------------------------------------ alpha inputs
def alpha_inputs(src: HeadSource, sweep: dict, atts: dict, *, band=None, lambda_selection=None,
                 decision_record=None) -> dict:
    """The alpha selection's committed inputs (PL-16, PL-20(c)): the band file named by the rules (its blob
    unchanged since the alpha default's records commit), the lambda selection file (likewise), and no
    alpha_selection.json beside a decision record that cuts the alpha sweep. Records are compared at the
    source's records_ref: HEAD for the selection, the records commit for the gate (PL-28)."""
    ref = src.records_ref
    band_rel = sweep["band"]["file"]
    if band is not None and src.rel(band) != band_rel:
        raise SelectionRefused("band_format", f"--band {band} is not the rules' band file {band_rel} (PL-16)")
    rel, data = require_committed(band_rel, src=src, missing="band_missing", uncommitted="band_uncommitted")
    s, band_w, trace = dl27_band(parse_json_bytes(data, rel, code="band_format"), float(sweep["band"]["floor"]))
    first = next((a for a in atts[float(sweep["default_candidate"])] if a.launched is not None), None)
    if first is not None and src.blob_at(first.launched["records_commit"], band_rel) != src.blob_at(ref, band_rel):
        raise SelectionRefused("band_changed_after_launch", f"{band_rel} at {ref[:12]} differs from its blob at the "
                                                            f"alpha default's records commit "
                                                            f"{first.launched['records_commit'][:12]} (PL-16)")
    if lambda_selection is not None and src.rel(lambda_selection) != LAMBDA_SELECTION_REL:
        raise SelectionRefused("lambda_selection_format", f"--lambda-selection {lambda_selection} is not "
                                                          f"{LAMBDA_SELECTION_REL}, the file whose commit dates the "
                                                          "alpha decision (PL-17)")
    lrel, ldata = require_committed(LAMBDA_SELECTION_REL, src=src, missing="lambda_selection_missing",
                                    uncommitted="lambda_selection_uncommitted")
    ldoc = parse_json_bytes(ldata, lrel, code="lambda_selection_format")
    winner = ldoc.get("winner") if isinstance(ldoc, dict) else None
    if not isinstance(ldoc, dict) or ldoc.get("format") != "lambda_selection/1" or not isinstance(winner, dict) \
            or not _finite_number(winner.get("lambda")):
        raise SelectionRefused("lambda_selection_format", f"{lrel} is not a lambda_selection/1 file with a winner")
    if first is not None and src.blob_at(first.launched["records_commit"], LAMBDA_SELECTION_REL) \
            != src.blob_at(ref, LAMBDA_SELECTION_REL):
        raise SelectionRefused("lambda_selection_changed", f"{LAMBDA_SELECTION_REL} at {ref[:12]} differs from its "
                                                           f"blob at the alpha default's records commit "
                                                           f"{first.launched['records_commit'][:12]} (PL-17)")
    rec_rel = DECISION_RECORD_REL["alpha_cwd"] if decision_record is None else src.rel(decision_record)
    if src.blob_at(ref, ALPHA_SELECTION_REL) is not None and rec_rel is not None:
        raw = src.show(ref, rec_rel)
        try:
            rdoc = None if raw is None else json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rdoc = None
        if isinstance(rdoc, dict) and rdoc.get("alpha_sweep_cut") is True:
            raise SelectionRefused("alpha_cut_conflict", f"{ALPHA_SELECTION_REL} and a decision record that cuts the "
                                                         f"alpha sweep ({rec_rel}) are both committed: one of them "
                                                         "is wrong (PL-20(c))")
    return {"band_rel": rel, "band_sha256": sha256_bytes(data), "s": s, "band": band_w, "band_trace": trace,
            "lambda_rel": lrel, "lambda_sha256": sha256_bytes(ldata), "lambda": float(winner["lambda"]),
            "lambda_doc": ldoc}


def adding_commit(src: HeadSource, rel: str) -> str | None:
    """The commit that added `rel` with its blob at the source's records_ref (HEAD; the records commit at
    the gate): present there with that blob and absent in each parent, a merge commit included
    (_adding_commits); the newest such commit."""
    head_blob = src.blob_at(src.records_ref, rel)
    if head_blob is None:
        return None
    return next((c for c in _adding_commits(src, rel) if src.blob_at(c, rel) == head_blob), None)


def _path_history(src: HeadSource, rel: str) -> list:
    """[(commit, [parents])] of every commit `git log --full-history` lists for `rel` in the history of the source's
    records_ref, newest first: each one not TREESAME to some parent, merges included (no parent rewriting, so the
    parents are the commit's own)."""
    out = src._out("log", "--full-history", "--no-renames", "--format=%H %P", src.records_ref, "--", rel)
    return [(c, parents) for c, *parents in (ln.split() for ln in (out or "").splitlines() if ln.strip())]


def _adding_commits(src: HeadSource, rel: str) -> list:
    """Every commit in the history of the source's records_ref that added `rel`: present in the commit and absent
    in each parent (a merge that brings the file from a parent adds nothing; a merge that holds it while no parent
    does adds it). Newest first."""
    return [c for c, parents in _path_history(src, rel)
            if src.blob_at(c, rel) is not None and all(src.blob_at(p, rel) is None for p in parents)]


def _activity_times(entries: list, *, what: str) -> list:
    """[(entry, timestamp)] of an activity response, every entry."""
    try:
        return [(e, parse_activity_time(e.get("timestamp"))) for e in entries]
    except ValueError as err:
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{what}: an entry's timestamp ({err})") from err


def _activity_pushes(timed: list) -> list:
    """The entries that push a commit: every entry but a branch deletion (AM-19a reading 10: a deletion pushes
    no commit and is not counted)."""
    return [(e, t) for e, t in timed if e.get("activity_type") != BRANCH_DELETION]


def activity_entries(data: bytes, what: str) -> list:
    """A raw activity response: a non-empty JSON list of objects (lambda_push_evidence_mismatch otherwise)."""
    entries = parse_json_bytes(data, what, code="lambda_push_evidence_mismatch")
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{what} is not a raw activity response (a JSON list "
                                                                "of entries)")
    return entries


def pushes_after(entries: list, t_lo: float, *, what: str) -> list:
    """The entries the push list answers (AM-19a reading 10): every push (not a branch deletion) at or after
    T_lo, in the response's order."""
    return [e for e, t in _activity_pushes(_activity_times(entries, what=what)) if t >= t_lo]


def push_list_doc(response: bytes, lambda_selection: bytes, commit: str, t_lo: float, remote: str,
                  answers: dict) -> dict:
    """AM-19a reading 10: the list committed with the activity response (scripts/preflight_distill.py
    push-evidence writes it on the push day). One entry per push after T_lo (pushes_after), in the response's
    order: answers[id] = {"served": bool, "carries_file": bool, "remote_answer": str}, settled by fetching the
    pushed commit from the remote; a commit the remote does not serve counts as carrying the file. Checked here,
    so that the push day refuses what push_evidence would refuse in the response and the list alone (the pair
    is settled once and cannot be replaced): the arguments' types (T_lo a finite number, never a bool; the
    commit is push_evidence's to check); the response's form; that it reaches back before T_lo; each push after
    T_lo once (its id); an answer of the right form for each; every entry's form (lambda_push_evidence_mismatch).
    The checks that need the clone (the commits that added the files, the lambda selection's adding commit, a held
    commit against its answer, the earliest carrying push and its bounds) are push_evidence's."""
    code = "lambda_push_evidence_mismatch"
    if not (isinstance(response, bytes) and isinstance(lambda_selection, bytes) and isinstance(remote, str)
            and isinstance(answers, dict) and _finite_number(t_lo)):
        raise SelectionRefused(code, "the push list takes the response and the lambda selection as bytes, T_lo as a "
                                     "finite number, the remote as a string and the answers as a mapping")
    entries = activity_entries(response, "the activity response")
    if not any(t < t_lo for _, t in _activity_times(entries, what="the activity response")):
        raise SelectionRefused(code, f"the activity response: no entry is older than T_lo {iso_utc(t_lo)}: it does not "
                                     "reach back far enough; fetch more of the activity before the list is made "
                                     "(AM-19a reading 10)")
    ids = [json.dumps(e.get("id")) for e in pushes_after(entries, t_lo, what="the activity response")]
    if len(set(ids)) != len(ids):
        raise SelectionRefused(code, "the activity response names a push after T_lo twice (its id): each push is "
                                     "listed once")
    pushes = []
    for e in pushes_after(entries, t_lo, what="the activity response"):
        ans = answers.get(e.get("id")) if _is_int(e.get("id")) else None
        if not (isinstance(ans, dict) and isinstance(ans.get("served"), bool)
                and isinstance(ans.get("carries_file"), bool) and isinstance(ans.get("remote_answer"), str)):
            raise SelectionRefused(code, f"push {e.get('id')!r}: no answer {{served: bool, carries_file: bool, "
                                         f"remote_answer: str}} ({ans!r})")
        x = {**{k: e.get(k) for k in ("id", "ref", "before", "after", "timestamp", "activity_type")},
             "served": ans["served"], "carries_file": True if not ans["served"] else ans["carries_file"],
             "remote_answer": ans["remote_answer"]}
        if not _push_entry_ok(x):
            raise SelectionRefused(code, f"push {e.get('id')!r}: the entry {x} is not a list entry push_evidence "
                                         "accepts (an int id, strings, a commit id after)")
        pushes.append(x)
    return {"format": PUSH_LIST_FORMAT, "response": {"path": PUSH_EVIDENCE_REL, "sha256": sha256_bytes(response)},
            "lambda_selection": {"path": LAMBDA_SELECTION_REL, "sha256": sha256_bytes(lambda_selection)},
            "adding_commit": commit, "t_lo_utc": t_lo, "remote": remote, "pushes": pushes}


def _push_entry_ok(x) -> bool:
    return (isinstance(x, dict) and sorted(x) == sorted(PUSH_ENTRY_KEYS) and _is_int(x["id"])
            and all(isinstance(x[k], str) for k in ("ref", "before", "timestamp", "activity_type", "remote_answer"))
            and isinstance(x["after"], str) and _COMMIT_ID.fullmatch(x["after"]) is not None
            and isinstance(x["served"], bool) and isinstance(x["carries_file"], bool)
            and (x["served"] or x["carries_file"]))


def push_evidence(src: HeadSource, t_lo: float, t_hi) -> dict:
    """PL-17(c) with AM-19a reading 10: reports/derived/lambda_selection_push.json, GitHub's activity response
    saved verbatim on the push day (PL-32), and the list committed with it (PUSH_LIST_REL, push_list_doc). The
    commit that added the lambda selection file reached the remote at the earliest timestamp over every
    push, of any ref, that carries it (its commit is or descends from that commit). Branch deletions are not
    counted as pushes (they still show how far back the response reaches). The response reaches back before
    T_lo. The response and the list were settled once: one commit added both, each was added only once, and
    neither has changed since. Each push after T_lo takes its status from the list, which must hold every such
    push and nothing else: a push the remote did not serve when the list was made counts as carrying the file
    and is named in the result (`unserved`, written to alpha_cutoff_basis). Wherever the clone holds a pushed
    commit, served or not, the list's status is checked against it (the stricter reading: an unserved push
    whose commit the clone holds and finds without the file refuses). A push before T_lo counts when the
    clone holds it and it carries the file. pushed_at lies in [T_lo, T_hi]. No committer time and no local
    clock is compared."""
    code = "lambda_push_evidence_mismatch"
    rel, data = require_committed(PUSH_EVIDENCE_REL, src=src, missing="lambda_push_evidence_missing",
                                  uncommitted=code)
    entries = activity_entries(data, rel)
    lrel, ldata = require_committed(PUSH_LIST_REL, src=src, missing="lambda_push_evidence_missing",
                                    uncommitted=code)
    lst = parse_json_bytes(ldata, lrel, code=code)
    adds = {r: _adding_commits(src, r) for r in (PUSH_EVIDENCE_REL, PUSH_LIST_REL)}
    if len(adds[PUSH_EVIDENCE_REL]) != 1 or adds[PUSH_LIST_REL] != adds[PUSH_EVIDENCE_REL] \
            or any(src.blob_at(adds[r][0], r) != src.blob_at(src.records_ref, r) for r in adds):
        raise SelectionRefused(code, f"{rel} and {lrel} were not added together in one commit, once, and left "
                                     f"unchanged since (commits that added them: {adds}): the push evidence is "
                                     "settled once, when the response is saved, and the list is committed with "
                                     "it (AM-19a reading 10)")
    commit = adding_commit(src, LAMBDA_SELECTION_REL)
    if commit is None:
        raise SelectionRefused(code, f"no commit adds {LAMBDA_SELECTION_REL} with its HEAD blob (it changed after "
                                     "it was added)")
    _, lsel = require_committed(LAMBDA_SELECTION_REL, src=src, missing="lambda_selection_missing",
                                uncommitted="lambda_selection_uncommitted")
    if not (isinstance(lst, dict) and sorted(lst) == sorted(PUSH_LIST_KEYS) and lst["format"] == PUSH_LIST_FORMAT
            and _same(lst["response"], {"path": PUSH_EVIDENCE_REL, "sha256": sha256_bytes(data)})
            and _same(lst["lambda_selection"], {"path": LAMBDA_SELECTION_REL, "sha256": sha256_bytes(lsel)})
            and lst["adding_commit"] == commit and _same(lst["t_lo_utc"], t_lo) and isinstance(lst["remote"], str)
            and isinstance(lst["pushes"], list) and all(_push_entry_ok(x) for x in lst["pushes"])):
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{lrel}: not the {PUSH_LIST_FORMAT} list of this "
                                                                f"response (its sha256), of {LAMBDA_SELECTION_REL}, "
                                                                f"of the commit {commit[:12]} that added it and of "
                                                                f"T_lo {iso_utc(t_lo)} (AM-19a reading 10)")
    timed = _activity_times(entries, what=rel)
    if not any(t < t_lo for _, t in timed):
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{rel}: no entry is older than T_lo "
                                                                f"{iso_utc(t_lo)}: the response does not reach "
                                                                "back far enough")
    pushes = _activity_pushes(timed)
    listed = {x["id"]: x for x in lst["pushes"]}
    after_lo = [e for e, t in pushes if t >= t_lo]
    for e in after_lo:
        x = listed.get(e.get("id")) if _is_int(e.get("id")) else None
        if x is None or any(x[k] != e.get(k) for k in ("ref", "before", "after", "timestamp", "activity_type")):
            raise SelectionRefused("lambda_push_evidence_mismatch", f"{lrel}: the push {e.get('id')!r} at "
                                                                    f"{e.get('timestamp')} (after T_lo) is not in the "
                                                                    "list made when the response was saved, or "
                                                                    "differs from it (AM-19a reading 10)")
    if len(listed) != len(lst["pushes"]) or sorted(listed) != sorted(e["id"] for e in after_lo):
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{lrel}: the list names pushes that are not the "
                                                                "response's pushes after T_lo, each once")
    hits, unserved = [], []
    for e, t in pushes:
        after = e.get("after")
        held = isinstance(after, str) and _COMMIT_ID.fullmatch(after) is not None and src.commit_exists(after)
        if t < t_lo:
            carries = held and src.is_ancestor(commit, after)
        elif not listed[e["id"]]["served"]:
            if held and not src.is_ancestor(commit, after):
                raise SelectionRefused(code, f"{lrel}: push {e['id']!r} ({after[:12]}) is listed as not served, so "
                                             "it would count as carrying the file, and the clone holds its commit, "
                                             "which does not carry it (AM-19a reading 10: the list is checked "
                                             "wherever the clone holds the pushed commit)")
            carries = True
            unserved.append({"id": e["id"], "ref": e.get("ref"), "after": after, "timestamp": e.get("timestamp"),
                             "remote_answer": listed[e["id"]]["remote_answer"]})
        else:
            carries = listed[e["id"]]["carries_file"]
            if held and src.is_ancestor(commit, after) != carries:
                raise SelectionRefused("lambda_push_evidence_mismatch", f"{lrel}: push {e['id']!r} ({after[:12]}): the "
                                                                        f"list says carries_file={carries}, and the "
                                                                        "clone's commit says otherwise")
        if carries:
            hits.append((t, e))
    if not hits:
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{rel}: no push carries {commit[:12]}")
    pushed_at, entry = min(hits, key=lambda h: h[0])
    if pushed_at < t_lo or (t_hi is not None and pushed_at > t_hi):
        raise SelectionRefused("lambda_push_evidence_mismatch", f"{rel}: the push at {iso_utc(pushed_at)} lies "
                                                                f"outside [T_lo {iso_utc(t_lo)}, T_hi {iso_utc(t_hi)}]")
    return {"path": rel, "sha256": sha256_bytes(data), "list_path": lrel, "list_sha256": sha256_bytes(ldata),
            "commit": commit, "pushed_at_utc": pushed_at, "entry_id": entry.get("id"), "ref": entry.get("ref"),
            "unserved": unserved}


def alpha_cutoff(src: HeadSource, calendar: str, *, lambda_doc: dict, default_attempts: list) -> dict:
    """PL-17, AM-19a reading 10: the alpha decision date is the later of its calendar date and D + 3, D the
    Asia/Manila day on which the commit that added reports/derived/lambda_selection.json reached the remote,
    on any ref. T_lo = the lambda selection's inputs_last_timestamp_utc - 60 s; T_hi = the launch time of the
    alpha default's first launched line + 60 s, valid only if the lambda selection's blob at its
    records_commit equals the one at the source's records_ref, HEAD for the selection and the records commit
    for the gate (absent while no alpha default has launched). Basis (a): day(T_hi) + 3 <= the calendar date;
    (b) otherwise day(T_lo) = day(T_hi), which is D; (c) otherwise the committed push evidence and its list
    (push_evidence). The gate uses the same function."""
    cal = date.fromisoformat(calendar)
    days = ALPHA_DECISION_DATE_RULE["lambda_selection_days"]
    t_in = lambda_doc.get("inputs_last_timestamp_utc") if isinstance(lambda_doc, dict) else None
    if not _finite_number(t_in):
        raise SelectionRefused("lambda_selection_format", f"{LAMBDA_SELECTION_REL} has no finite "
                                                          "inputs_last_timestamp_utc: the alpha decision date "
                                                          "cannot be derived (PL-17)")
    t_lo = float(t_in) - PUSH_MARGIN_SECONDS
    first = next((a for a in default_attempts if a.launched is not None), None)
    t_hi = None
    if first is not None:
        if src.blob_at(first.launched["records_commit"], LAMBDA_SELECTION_REL) \
                != src.blob_at(src.records_ref, LAMBDA_SELECTION_REL):
            raise SelectionRefused("lambda_selection_changed", f"{LAMBDA_SELECTION_REL} at {src.records_ref[:12]} "
                                                               "differs from its blob at the alpha default's records "
                                                               "commit (PL-17)")
        t_hi = first.launch_ts + PUSH_MARGIN_SECONDS
    basis = {"basis": None, "T_lo_utc": t_lo, "T_hi_utc": t_hi, "calendar_date": calendar,
             "lambda_selection_days": days, "D": None, "evidence": None}
    if t_hi is not None and manila_day(t_hi) + timedelta(days=days) <= cal:
        basis["basis"] = "a"
        decision = cal
    elif t_hi is not None and manila_day(t_lo) == manila_day(t_hi):
        d = manila_day(t_lo)
        basis.update(basis="b", D=d.isoformat())
        decision = max(cal, d + timedelta(days=days))
    else:
        ev = push_evidence(src, t_lo, t_hi)
        d = manila_day(ev["pushed_at_utc"])
        basis.update(basis="c", D=d.isoformat(), evidence=ev)
        decision = max(cal, d + timedelta(days=days))
    return {"decision_date": decision.isoformat(), "cutoff_utc": cutoff_instant(decision.isoformat()),
            "basis": basis}


# ----------------------------------------------------------------------------------- the sweep
def derive_sweep(key: str, runs, *, script_path, rules=None, band=None, lambda_selection=None,
                 decision_record=None) -> dict:
    """Everything the selection derives before it applies the rule, in its refusal order: (a) the test-path
    and duplicate-directory refusals, the repository, the code, the rules, the launch log, the schedule,
    the sweep's launch lines, their reports, the code pin, the log's prefixes, alpha's band and lambda
    files, and the decision date; (b) the supplied run directories; (c) their agreement with the launch log,
    and for every attempt of every value that a launched line has its directory and a launch line its
    outcome (PL-14: from C a refusal; before C the value waits, exit 3, once no refusal below applies); (d) the
    repeat rules and the finished runs' best.json and checkpoint; (e) each grid value's status; (f) the
    candidate-level refusals, in --runs order; then clock_inconsistent, recipe_mismatch_across_candidates,
    alpha's shared lambda, and the default candidate's divergence or second same ending."""
    now = now_utc()
    runs = [Path(r) for r in runs]
    for r in runs:
        refuse_test_path(r)
    names = [r.name for r in runs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SelectionRefused("duplicate_candidate", f"--runs names the run directory {dup} more than once")
    src = HeadSource()
    src.require_not_shallow()
    check_code_committed(src, selection_code_files(key, script_path))
    rules_rel, rules_doc = load_committed_rules(src, rules)
    sweep = rules_doc[key]
    log = load_launch_log(src)
    schedule = load_schedule(src)
    check_schedule_at_launches(src, schedule, log)
    atts = sweep_attempts(log, sweep, key, schedule)
    check_reports(src, atts)
    pin = check_code_pin(src, atts, rules_rel, key)
    log_at: dict = {}
    check_log_prefix(src, log, log_at)
    default = float(sweep["default_candidate"])
    alpha = alpha_inputs(src, sweep, atts, band=band, lambda_selection=lambda_selection,
                         decision_record=decision_record) if key == "alpha_cwd" else None
    calendar = sweep["decision_dates"][schedule["in_force"]]
    if key == "lambda_logit":
        cut = {"decision_date": calendar, "cutoff_utc": cutoff_instant(calendar), "basis": None}
    elif now < cutoff_instant(calendar):
        cut = {"decision_date": None, "cutoff_utc": None, "basis": None}  # the alpha date cannot have ended yet
    else:
        cut = alpha_cutoff(src, calendar, lambda_doc=alpha["lambda_doc"], default_attempts=atts[default])
    cutoff = cut["cutoff_utc"]
    after_c = cutoff is not None and now >= cutoff

    by_id = {a.run_id: a for lst in atts.values() for a in lst}
    supplied = []
    for d in runs:
        if not d.is_dir():
            continue                       # an absent directory is not supplied: its value waits (or is cut)
        a = by_id.get(d.name)
        if a is None:
            raise SelectionRefused("launch_log_mismatch", f"{d.name}: no launch line of the {key} sweep in "
                                                          f"{LAUNCH_LOG_REL} names this run directory (PL-14)")
        a.run = read_run(d, sweep, key)
        supplied.append(a)
    for a in supplied:
        check_run_against_log(src, a, key, log_at)
    pending = {}     # PL-14, every attempt of every value: from C a refusal; before C the value waits (exit 3)
    for a in _in_log_order(atts):
        if a.launched is None and a.not_launched is None:
            if after_c:
                raise SelectionRefused("launch_log_incomplete", f"{a.run_id}: its launch line has neither a launched "
                                                                "nor a not_launched line, and the decision date has "
                                                                "ended (PL-14)")
            pending.setdefault(a.value, (a, "its launch line has neither a launched nor a not_launched line"))
        elif a.launched is not None and a.run is None:
            if after_c:
                raise SelectionRefused("launch_log_mismatch", f"{a.run_id} was launched, and its run directory is not "
                                                              "supplied (PL-14: never a cut)")
            pending.setdefault(a.value, (a, "it was launched, and its run directory is not supplied"))
    check_repeats(atts, key)
    for a in supplied:
        r = a.run
        if run_fault(r) is None and r.missing is None and r.ending is not None \
                and r.ending["kind"] in ("finished", "diverged"):
            try:
                r.artifacts = read_artifacts(r, sweep, key)
            except SelectionRefused as e:
                r.artifact_fault = e
    for a in supplied:
        if a.run.artifact_fault is not None:
            raise a.run.artifact_fault

    statuses = {}
    for v, lst in atts.items():
        if v != default and after_c:
            statuses[v] = resolve_after_cutoff(v, lst, cutoff, sweep["iterations"])
        else:
            statuses[v] = resolve_latest(v, lst, default=(v == default), cutoff=cutoff)
    order = {a.run_id: i for i, a in enumerate(supplied)}
    refused = sorted((s for s in statuses.values() if s["status"] == "refused"),
                     key=lambda s: order.get(s["attempt"].run_id, len(order)))
    if refused:
        raise refused[0]["refusal"]
    late = [(a.run_id, t) for a in supplied for t in a.run.timestamps() if t > now] \
        + [(a.run_id, a.launch_ts) for a in _in_log_order(atts) if a.launched is not None and a.launch_ts > now]
    if late:
        raise SelectionRefused("clock_inconsistent", f"timestamps later than now ({iso_utc(now)}): "
                                                     f"{[(rid, iso_utc(t)) for rid, t in late[:5]]}: a clock is wrong "
                                                     "(the pod's, or this machine's)")
    of_record = [statuses[v]["attempt"] for v in sorted(statuses) if statuses[v]["status"] in ("finished", "diverged")]
    check_recipe_across([candidate_record(a.run, key) for a in of_record])
    if key == "alpha_cwd":
        check_shared_lambda(atts, alpha["lambda"])
    d = statuses[default]
    if d["status"] == "diverged":
        refuse_diverged_default([candidate_record(d["attempt"].run, key)], sweep, key)
    if d["status"] == "aborted_twice":
        raise d["refusal"]
    # PL-14 before C: a value with an attempt that is still incomplete waits (exit 3), after every refusal
    # above (exit 2), each of which the complete log would refuse too (AM-19a reading 18; a non-default
    # value's single abort or failed check waits instead while an earlier attempt is incomplete and could
    # still end the same way, _twice_undecided).
    for v, (a, why) in pending.items():
        statuses[v] = _status("waiting", value=v, attempt=a, code="candidate_missing",
                              reason=f"{a.run_id}: {why}; the value waits until it is (PL-14)")
    return {"key": key, "now": now, "src": src, "sweep": sweep, "rules_rel": rules_rel, "log": log,
            "schedule": schedule, "atts": atts, "pin": pin, "default": default, "alpha": alpha,
            "calendar_date": calendar, "decision_date": cut["decision_date"], "cutoff_utc": cutoff,
            "basis": cut["basis"], "after_c": after_c, "statuses": statuses, "supplied": supplied}


def alpha_sweep_cut(state: dict) -> bool:
    """AM-19 item 2(g): no non-default alpha candidate launched before the end of the alpha decision date."""
    if state["key"] != "alpha_cwd" or not state["after_c"]:
        return False
    cutoff, default = state["cutoff_utc"], state["default"]
    return not any(a.launched is not None and a.launch_ts < cutoff
                   for v, lst in state["atts"].items() if v != default for a in lst)


# ------------------------------------------------------------------------------ decision record
_RECORD_KEYS = ("format", "sweep", "schedule", "decision_date", "cutoff_utc", "alpha_cutoff_basis",
                "alpha_sweep_cut", "launch_log", "values", "never_launched", "corrects")
_DIR_KEYS = ("run_id", "attempt", "launched", "launch_time_utc", "last_iter_before_cutoff", "run_meta_sha256",
             "telemetry_sha256", "best_json_sha256", "best_ckpt_sha256", "am8a_report_sha256",
             "stop_report_sha256")
# AM-19a reading 19: a corrected record names the void record it replaces and the AM-8a fault report
_CORRECTS_KEYS = ("fault_report", "void_code", "void_record")


def corrects_ok(cor) -> bool:
    """The form of a record's `corrects` (AM-19a reading 19): null, or {void_record, fault_report, void_code}
    with two {path, sha256} references and a refusal code."""
    return cor is None or (isinstance(cor, dict) and sorted(cor) == sorted(_CORRECTS_KEYS)
                           and _is_report_ref(cor["void_record"]) and _is_report_ref(cor["fault_report"])
                           and isinstance(cor["void_code"], str))


def record_place_error(key: str, rel: str, cor) -> str | None:
    """AM-19a reading 19: where a decision record of the sweep `key` is read. The sweep's record is
    DECISION_RECORD_REL[key] and corrects nothing; its one corrected record is DECISION_RECORD_CORRECTED_REL[key]
    and corrects that record. None when `rel` and `cor` (null or of corrects_ok's form) agree; the reason
    otherwise (the selection and the launch gate refuse it as decision_record_correction_invalid)."""
    first, fixed = DECISION_RECORD_REL[key], DECISION_RECORD_CORRECTED_REL[key]
    if rel == first:
        return None if cor is None else (f"{rel} is the sweep's decision record and corrects nothing: its corrected "
                                         f"record is written to {fixed} (AM-19a reading 19)")
    if rel == fixed:
        if cor is None:
            return f"{rel} is the sweep's corrected record: it names the void record it corrects (AM-19a reading 19)"
        if cor["void_record"]["path"] != first:
            return (f"{rel} corrects {cor['void_record']['path']!r}: the corrected record corrects the sweep's "
                    f"decision record {first} (AM-19a reading 19)")
        return None
    return (f"{rel}: a decision record of the {key} sweep is read at {first}, or, corrected once, at {fixed} "
            "(AM-19a reading 19)")


def _changing_commits(src: HeadSource, rel: str) -> list:
    """Every commit in the history of the source's records_ref whose blob at `rel` (or its absence) differs from
    that of each parent (a root commit that holds it included): the commits that add, change or delete it. A
    merge that takes one parent's version changes nothing."""
    found = []
    for c, parents in _path_history(src, rel):
        blob = src.blob_at(c, rel)
        if all(src.blob_at(p, rel) != blob for p in parents) and (parents or blob is not None):
            found.append(c)
    return found


def written_once_error(src: HeadSource, rel: str) -> str | None:
    """AM-19a reading 19: a void record is kept and never edited, and a corrected record is written once: in the
    history of the source's records_ref one commit adds `rel` and no other commit changes or deletes it. None
    when that holds; the reason otherwise."""
    changes = _changing_commits(src, rel)
    if len(changes) != 1 or _adding_commits(src, rel) != changes:
        return (f"{rel}: {len(changes)} commit(s) in the history of {src.records_ref[:12]} add, change or delete it: "
                "a void record is kept and never edited, and a corrected record is written once (AM-19a reading 19)")
    return None


def deleted_record_error(src: HeadSource, rel: str) -> str | None:
    """AM-19a reading 19: a committed decision record is kept and never edited. A record path with a history in the
    source's records_ref (_changing_commits) and no file there was deleted after its commit: the selection and the
    launch gate refuse it (decision_record_rewritten). No later input lifts that (the writer refuses a path with a
    history, and a correction needs the void record), so it is a STOP for a new amendment (AM-19a). None when `rel`
    is present at the records_ref or never held a file."""
    if src.blob_at(src.records_ref, rel) is not None or not _changing_commits(src, rel):
        return None
    return (f"{rel} was committed and is deleted at {src.records_ref[:12]}: a committed decision record is kept and "
            "never edited (AM-19a reading 19); no record replaces it, a STOP for a new amendment")


def _record_dirs(state: dict, value: float, prefix: dict | None = None) -> list[dict]:
    """AM-19 item 2(h): every attempt of a grid value in launch order: its launch time, its last iteration
    before the end of the date, and the sha256 of its run_meta and telemetry, of a finished run's best.json
    and best checkpoint, of the AM-8a report that precedes it and of its stop report. With `prefix` (a
    record's own launch-log lines, AM-19a reading 19: {"launch": run_ids, "launched": run_ids}), only the
    attempts whose launch line is in it, launched as those lines say."""
    out = []
    for a in state["atts"][value]:
        if prefix is not None and a.run_id not in prefix["launch"]:
            continue
        r = a.run
        art = (r.artifacts or {}) if r is not None else {}
        launched = a.launched is not None if prefix is None else a.run_id in prefix["launched"]
        out.append({"run_id": a.run_id, "attempt": a.attempt, "launched": launched,
                    "launch_time_utc": a.launch_ts,
                    "last_iter_before_cutoff": None if r is None else r.last_train_iter(state["cutoff_utc"]),
                    "run_meta_sha256": None if r is None else r.run_meta_sha256,
                    "telemetry_sha256": None if r is None else r.telemetry_sha256,
                    "best_json_sha256": art.get("best_json_sha256"), "best_ckpt_sha256": art.get("ckpt_sha256"),
                    "am8a_report_sha256": (a.line["am8a_report"] or {}).get("sha256"),
                    "stop_report_sha256": None if a.stopped is None else a.stopped["report"]["sha256"]})
    return out


def _record_status(state: dict, value: float) -> str:
    s = state["statuses"][value]
    if s["status"] == "waiting":
        return RUNNING_DEFAULT if value == state["default"] else RUNNING_ON_COURSE
    return s["status"]


def _schedule_fields(state: dict) -> dict:
    sch = state["schedule"]
    return {"in_force": sch["in_force"], "decision_log_entry": sch["decision_log_entry"],
            "file_sha256": sch["file_sha256"]}


def decision_record_doc(state: dict, corrects: dict | None = None) -> dict:
    """The decision record --write-decision-record derives (AM-19 item 2(h); PL-15); `corrects` names the void
    record a corrected one replaces (AM-19a reading 19; correction_refs)."""
    st, grid = state["statuses"], [float(v) for v in state["sweep"]["grid"]]
    return {"format": RECORD_FORMAT, "sweep": state["key"], "schedule": _schedule_fields(state),
            "decision_date": state["decision_date"], "cutoff_utc": state["cutoff_utc"],
            "alpha_cutoff_basis": state["basis"], "alpha_sweep_cut": alpha_sweep_cut(state),
            "launch_log": {"sha256": state["log"].sha256, "lines": state["log"].n_lines},
            "values": [{"value": v, "status": _record_status(state, v),
                        "reason": st[v]["reason"] if st[v]["status"] == "cut" else None,
                        "directories": _record_dirs(state, v), "on_course": st[v]["on_course"]} for v in grid],
            "never_launched": sorted(v for v in grid if st[v]["status"] == "cut" and st[v]["reason"] == "never_launched"),
            "corrects": corrects}


def _check_on_course_record(state: dict, v: float, e: dict, rel: str, stopped_then: set) -> None:
    """AM-19a reading 7: a value recorded "running (on-course repeat, item 2(b))" waited on the
    launched repeat r of its on-course stop s: s and r are its last two launched directories, and its recorded
    evaluations are those of its stops up to s, derived again from their rows (PL-7). Neither depends on r's
    rows, so this holds whichever later status the value reaches (PL-15), r's own ending under PL-9(e)(1) or
    (2) included (the directories are compared with the launch log's below). The writer records it only while
    r has no later attempt, no stopped line, no ending and no fault (PL-9(e)(3)): r is the value's last
    directory, the record's launch-log prefix holds no stopped line for r (stopped_then: the run_ids that
    prefix stops), and a telemetry file of r unchanged since the record shows no ending and no fault."""
    evaluated, s = _on_course_stops(state["atts"][v], state["cutoff_utc"], state["sweep"]["iterations"])
    launched = [x["run_id"] for x in e["directories"] if x["launched"]]
    rr = next((a.run for a in state["atts"][v] if launched and a.run_id == launched[-1]), None)
    tel = e["directories"][-1]["telemetry_sha256"] if e["directories"] else None
    unchanged = rr is not None and tel is not None and tel == rr.telemetry_sha256
    ok = (s is not None and len(launched) >= 2 and launched[-2] == s.run_id and _same(e["on_course"], evaluated)
          and e["directories"][-1]["run_id"] == launched[-1]
          and launched[-1] not in stopped_then
          and not (unchanged and (run_fault(rr, None) is not None or rr.ending is not None)))
    if not ok:
        raise SelectionRefused("decision_record_status_mismatch",
                               f"{rel}: {v:g}: {RUNNING_ON_COURSE!r} needs the value's on-course stop and its "
                               "launched repeat as its last two launched directories, the evaluations of its stops "
                               f"up to that one, derived again (recorded {e['on_course']!r}; derived "
                               f"{evaluated!r}), and the repeat as its last directory, without a stopped line in "
                               "the record's launch log and without an ending or fault in a telemetry file "
                               "unchanged since (AM-19a reading 7; PL-7; PL-9(e)(3))")


def verify_record(state: dict, doc, rel: str, sha: str) -> None:
    """AM-19 item 2(h), PL-15 and AM-19a reading 19: the selection does not trust the record. It derives every
    status again and refuses a record that is malformed (decision_record_format), whose sha256 is not in the
    committed decision log (decision_record_not_in_decision_log), before the end of the decision date
    (decision_date_not_ended), whose schedule, decision date or cutoff differ (decision_record_schedule_mismatch;
    the alpha basis is not compared, CHECK ITEM 4), whose launch log is not a prefix of HEAD's, whose directories
    differ from the attempts of the record's own launch-log lines, or to which a launch line of a non-default
    value was added later (decision_record_launch_log_mismatch; a later launch line of the default is derived
    and checked against the launch log, reading 19), whose statuses or on-course evaluations differ
    (decision_record_status_mismatch; a running status, which has no reason, accepts any later one, the
    default's only on the default and an on-course repeat's only with that repeat's shape) or whose hashes
    differ (decision_record_hash_mismatch). A running value's last directory is still being written: its
    telemetry and best files are not compared, its last iteration before the date may only grow (the
    telemetry is append-only), and a stop report may appear."""
    st, grid = state["statuses"], [float(v) for v in state["sweep"]["grid"]]
    if not isinstance(doc, dict) or sorted(doc) != sorted(_RECORD_KEYS) or doc["format"] != RECORD_FORMAT \
            or doc["sweep"] != state["key"]:
        raise SelectionRefused("decision_record_format", f"{rel}: not a {RECORD_FORMAT} record of the "
                                                         f"{state['key']} sweep with the keys {list(_RECORD_KEYS)}")
    ll, vals, nl, cor = doc["launch_log"], doc["values"], doc["never_launched"], doc["corrects"]
    ok = (isinstance(ll, dict) and sorted(ll) == ["lines", "sha256"] and _is_int(ll["lines"]) and ll["lines"] >= 0
          and isinstance(ll["sha256"], str) and isinstance(doc["alpha_sweep_cut"], bool)
          and isinstance(nl, list) and all(_finite_number(x) for x in nl) and isinstance(vals, list)
          and all(isinstance(e, dict) and sorted(e) == ["directories", "on_course", "reason", "status", "value"]
                  and _finite_number(e["value"]) and isinstance(e["directories"], list)
                  and all(isinstance(x, dict) and sorted(x) == sorted(_DIR_KEYS) for x in e["directories"])
                  for e in vals)
          and sorted(float(e["value"]) for e in vals) == sorted(grid) and corrects_ok(cor))
    if not ok:
        raise SelectionRefused("decision_record_format", f"{rel}: a field has the wrong form (launch_log "
                                                         "{sha256, lines}; one values entry per grid value with "
                                                         "value, status, reason, directories and on_course; corrects "
                                                         "null or {void_record, fault_report, void_code})")
    if sha not in state["schedule"]["decision_log"]:
        raise SelectionRefused("decision_record_not_in_decision_log", f"{rel}: its sha256 {sha} is not in the "
                                                                      f"committed {DECISION_LOG_REL}")
    if state["cutoff_utc"] is None or state["now"] < state["cutoff_utc"]:
        raise SelectionRefused("decision_date_not_ended", f"{rel}: the {state['key']} decision date has not ended")
    # AM-19a reading 19 and CHECK ITEM 4: the decision date and its instant must equal the derived ones; the way
    # the alpha date was derived (its basis) is not compared
    derived = {"schedule": _schedule_fields(state), "decision_date": state["decision_date"],
               "cutoff_utc": state["cutoff_utc"]}
    if not _same({k: doc[k] for k in derived}, derived):
        raise SelectionRefused("decision_record_schedule_mismatch", f"{rel}: schedule, decision date or cutoff "
                                                                    f"{[doc[k] for k in derived]} != derived "
                                                                    f"{list(derived.values())}")
    lines = state["log"].data.split(b"\n")[:-1] if state["log"].data else []
    if ll["lines"] > len(lines) or sha256_bytes(b"".join(x + b"\n" for x in lines[:ll["lines"]])) != ll["sha256"]:
        raise SelectionRefused("decision_record_launch_log_mismatch", f"{rel}: its launch log ({ll['lines']} lines, "
                                                                      f"{ll['sha256'][:12]}) is not a prefix of HEAD's")
    head = state["log"].entries[:ll["lines"]]
    stopped_then = {ln["run_id"] for ln, _ in head if ln.get("event") == "stopped"}
    prefix = {"launch": {ln["run_id"] for ln, _ in head if ln.get("event") == "launch"},
              "launched": {ln["run_id"] for ln, _ in head if ln.get("event") == "launched"}}
    for e in vals:
        v = float(e["value"])
        running = e["status"] in (RUNNING_DEFAULT, RUNNING_ON_COURSE)
        if running and (e["status"] == RUNNING_DEFAULT) != (v == state["default"]):
            raise SelectionRefused("decision_record_status_mismatch", f"{rel}: {v:g}: {e['status']!r} belongs to "
                                                                      + ("the default candidate" if e["status"] ==
                                                                         RUNNING_DEFAULT else "a non-default one"))
        if running and e["reason"] is not None:
            raise SelectionRefused("decision_record_status_mismatch", f"{rel}: {v:g}: {e['status']!r} with the reason "
                                                                      f"{e['reason']!r}: the writer gives a running "
                                                                      "status none")
        if not running and (e["status"] != st[v]["status"]
                            or e["reason"] != (st[v]["reason"] if st[v]["status"] == "cut" else None)):
            raise SelectionRefused("decision_record_status_mismatch", f"{rel}: {v:g}: recorded {e['status']!r} "
                                                                      f"({e['reason']!r}), derived "
                                                                      f"{st[v]['status']!r} ({st[v]['reason']!r})")
        if e["status"] == RUNNING_ON_COURSE:
            _check_on_course_record(state, v, e, rel, stopped_then)
        elif not _same(e["on_course"], st[v]["on_course"]):
            raise SelectionRefused("decision_record_status_mismatch", f"{rel}: {v:g}: recorded on-course "
                                                                      f"evaluations {e['on_course']!r} != derived "
                                                                      f"{st[v]['on_course']!r} (PL-7)")
        later = [a.run_id for a in state["atts"][v] if a.run_id not in prefix["launch"]]
        if later and v != state["default"]:
            raise SelectionRefused("decision_record_launch_log_mismatch", f"{rel}: {v:g}: launch line(s) {later} "
                                                                          "were added after the record's "
                                                                          f"{ll['lines']} launch-log lines: a later "
                                                                          "launch line of a non-default value of the "
                                                                          "sweep refuses the selection (AM-19a "
                                                                          "reading 19)")
        want = _record_dirs(state, v, prefix)
        if [(x["run_id"], x["attempt"], x["launched"]) for x in e["directories"]] \
                != [(x["run_id"], x["attempt"], x["launched"]) for x in want]:
            raise SelectionRefused("decision_record_launch_log_mismatch", f"{rel}: {v:g}: directories "
                                                                          f"{[x['run_id'] for x in e['directories']]}"
                                                                          " != the attempts of the record's launch-log "
                                                                          f"lines {[x['run_id'] for x in want]}")
        last = max((i for i, x in enumerate(want) if x["launched"]), default=None)
        for i, (got, exp) in enumerate(zip(e["directories"], want)):
            if not _same(got["launch_time_utc"], exp["launch_time_utc"]):
                raise SelectionRefused("decision_record_launch_log_mismatch", f"{rel}: {got['run_id']}: launch time "
                                                                              f"{got['launch_time_utc']!r} != the "
                                                                              f"log's {exp['launch_time_utc']!r}")
            live = running and i == last
            g_it, x_it = got["last_iter_before_cutoff"], exp["last_iter_before_cutoff"]
            if g_it != x_it and not (live and (g_it is None or (_is_int(g_it) and _is_int(x_it) and x_it >= g_it))):
                raise SelectionRefused("decision_record_status_mismatch", f"{rel}: {got['run_id']}: last iteration "
                                                                          "before the end of the date "
                                                                          f"{g_it!r} != {x_it!r}")
            skip = ("telemetry_sha256", "best_json_sha256", "best_ckpt_sha256") if live else ()
            bad = [f for f in _DIR_KEYS[5:] if f not in skip and got[f] != exp[f]
                   and not (live and f == "stop_report_sha256" and got[f] is None)]
            if bad:
                raise SelectionRefused("decision_record_hash_mismatch", f"{rel}: {got['run_id']}: {bad} differ from "
                                                                        "the run directory and the launch log")
    want_nl = sorted(v for v in grid if st[v]["status"] == "cut" and st[v]["reason"] == "never_launched")
    if sorted(float(x) for x in nl) != want_nl or doc["alpha_sweep_cut"] != alpha_sweep_cut(state):
        raise SelectionRefused("decision_record_status_mismatch", f"{rel}: never_launched {nl} / alpha_sweep_cut "
                                                                  f"{doc['alpha_sweep_cut']} != derived {want_nl} / "
                                                                  f"{alpha_sweep_cut(state)}")


def correction_refs(src: HeadSource, dl_text: str, cor: dict, rel: str | None) -> bytes:
    """AM-19a reading 19: the references a corrected record carries, checked without run directories (the
    launch gate checks them too): the void record it replaces is committed with that sha256, and the AM-8a
    fault report is committed with its sha256; both sha256 values are in the committed decision log (the
    corrected record's own is checked with the record). That the report was written before the correction is
    checked by the writer, which reads it as committed. Returns the void record's bytes.
    decision_record_correction_invalid otherwise."""
    code = "decision_record_correction_invalid"
    void, report = cor["void_record"], cor["fault_report"]
    p = PurePosixPath(void["path"])
    if p.is_absolute() or ".." in p.parts:
        raise SelectionRefused(code, f"{rel}: the void record's path {void['path']!r} is not repository-relative")
    vrel, vdata = require_committed(void["path"], src=src, missing=code, uncommitted=code, record=True)
    if sha256_bytes(vdata) != void["sha256"]:
        raise SelectionRefused(code, f"{vrel}: sha256 {sha256_bytes(vdata)} != {void['sha256']}, the void record's "
                                     f"sha256 in {rel}")
    _check_report(src, report, missing=code, mismatch=code)
    absent = [f"{what} {x['sha256']}" for what, x in (("the void record's", void), ("the fault report's", report))
              if x["sha256"] not in dl_text]
    if absent:
        raise SelectionRefused(code, f"{rel}: {' and '.join(absent)} sha256 not in the committed {DECISION_LOG_REL}: "
                                     "they are entered there before the corrected record is written (AM-19a "
                                     "reading 19)")
    return vdata


def _later_launches(state: dict, void_rel: str, vdata: bytes) -> list:
    """AM-19a reading 19: the run_ids of the non-default values' launch lines added after the void record: those
    beyond the launch log at the first commit that added it (a byte prefix of HEAD's log), or beyond the record's
    own launch-log lines when they are fewer and a prefix of HEAD's log. The record's own statement is never
    trusted beyond the log its commit saw. decision_record_correction_invalid when that log is no prefix of
    HEAD's."""
    src, log = state["src"], state["log"]
    adds = _adding_commits(src, void_rel)
    at_add = (src.show(adds[-1], LAUNCH_LOG_REL) or b"") if adds else None
    if at_add is None or not log.data.startswith(at_add):
        raise SelectionRefused("decision_record_correction_invalid",
                               f"{void_rel}: the launch log at the commit that added it is not a byte prefix of "
                               "HEAD's: the launch lines added after the void record cannot be told (AM-19a "
                               "reading 19)")
    n = at_add.count(b"\n")
    try:
        ll = json.loads(vdata.decode("utf-8")).get("launch_log")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        ll = None
    lines = log.data.split(b"\n")[:-1] if log.data else []
    if isinstance(ll, dict) and _is_int(ll.get("lines")) and 0 <= ll["lines"] < n \
            and sha256_bytes(b"".join(x + b"\n" for x in lines[:ll["lines"]])) == ll.get("sha256"):
        n = ll["lines"]
    seen = {ln["run_id"] for ln, _ in log.entries[:n] if ln.get("event") == "launch"}
    return [a.run_id for v, lst in state["atts"].items() if v != state["default"] for a in lst if a.run_id not in seen]


def void_code(state: dict, cor: dict, rel: str | None) -> str:
    """AM-19a reading 19: the record `cor` replaces is void: its references hold (correction_refs), it was kept
    and never edited (written_once_error), and the selection's checks refuse it (check_record), for a reason
    other than a launch line of a non-default value added after it (that refusal stands: the selection
    refuses, and no corrected record lifts it). Returns the refusal's code."""
    code = "decision_record_correction_invalid"
    void = cor["void_record"]
    vdata = correction_refs(state["src"], state["schedule"]["decision_log"], cor, rel)
    err = written_once_error(state["src"], void["path"])
    if err is not None:
        raise SelectionRefused(code, err)
    later = _later_launches(state, void["path"], vdata)
    if later:
        raise SelectionRefused(code, f"{void['path']}: the non-default launch lines {later} were added after it: a "
                                     "later launch line of a non-default value refuses the selection, and no "
                                     "corrected record replaces such a record (AM-19a reading 19)")
    try:
        check_record(state, void["path"], vdata)
    except SelectionRefused as e:
        return e.code
    raise SelectionRefused(code, f"{void['path']} passes the selection's checks: it is not void, and no corrected "
                                 "record replaces it (AM-19a reading 19)")


def check_record(state: dict, rel: str, data: bytes) -> dict:
    """The selection's checks of the committed decision record at `rel` (AM-19 item 2(h); AM-19a reading 19): its
    place (record_place_error), verify_record, written once (decision_record_rewritten: a record replaced in place,
    after a later launch line or otherwise, is an edit, never a new record), and for the corrected record the
    record it corrects void (void_code), with the void record's code as derived again. Returns {path, sha256,
    corrects}."""
    sha = sha256_bytes(data)
    doc = parse_json_bytes(data, rel, code="decision_record_format")
    cor = doc.get("corrects") if isinstance(doc, dict) else None
    if corrects_ok(cor):
        err = record_place_error(state["key"], rel, cor)
        if err is not None:
            raise SelectionRefused("decision_record_correction_invalid", err)
    verify_record(state, doc, rel, sha)
    err = written_once_error(state["src"], rel)
    if err is not None:
        raise SelectionRefused("decision_record_rewritten", err)
    if cor is not None:
        derived = void_code(state, cor, rel)
        if derived != cor["void_code"]:
            raise SelectionRefused("decision_record_correction_invalid",
                                   f"{rel}: the void record's code {cor['void_code']!r} != {derived!r}, derived again "
                                   "(AM-19 item 2(h): the selection does not trust the record)")
    return {"path": rel, "sha256": sha, "corrects": cor}


def load_record(state: dict, path=None) -> dict | None:
    """The committed decision record (--decision-record; default reports/derived/<sweep>_decision_record.json),
    checked (check_record). None when the default path holds no file: for the selection (HeadSource), neither in
    the working tree nor at HEAD; for the gate (RecordsSource, whose has() reads a record at the records commit,
    never in the folder), not at the records commit. A committed record missing from the working tree is
    decision_record_missing, a file absent at HEAD decision_record_uncommitted, and settle() has already refused
    one deleted after its commit. A corrected record (AM-19a reading 19) is read only at
    reports/derived/<sweep>_decision_record_corrected.json, through --decision-record."""
    src, key = state["src"], state["key"]
    target = DECISION_RECORD_REL[key] if path is None else path
    if path is None and not src.has(target) and src.blob_at(src.records_ref, target) is None:
        return None
    rel, data = require_committed(target, src=src, missing="decision_record_missing",
                                  uncommitted="decision_record_uncommitted")
    return check_record(state, rel, data)


def write_json_once(path: Path, doc: dict) -> str:
    """Write a JSON file once (never overwritten); returns its sha256."""
    path = Path(path)
    refuse_test_path(path.parent)
    if path.exists():
        raise SelectionRefused("output_exists", f"{path} already exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return sha256_bytes(data)


def ordinary_record_error(src: HeadSource, key: str, out_rel: str | None) -> str | None:
    """AM-19a reading 19, the writer of the sweep's record (no --corrects): it is written once and read at
    DECISION_RECORD_REL, so it is refused while that path has any history (a committed record, a void one
    included, is kept and never edited: it is replaced only by the corrected record, --corrects and
    --fault-report), and refused into the repository at any other path (record_place_error). `out_rel` is the
    output's repository-relative path, None outside the repository. None when it may be written."""
    first = DECISION_RECORD_REL[key]
    if _changing_commits(src, first):
        return (f"{first} is committed, or was: the sweep's decision record is written once and a committed record is "
                "kept and never edited; a void record is replaced only by the corrected record (--corrects "
                f"{first} --fault-report <the committed AM-8a fault report>; AM-19a reading 19)")
    return None if out_rel is None else record_place_error(key, out_rel, None)


def write_decision_record(state: dict, path: Path, corrects=None) -> tuple[str, dict]:
    """--write-decision-record (PL-15, PL-32): only after the end of the decision date, and only once every
    non-default launched run has ended (a trainer record) or was stopped (a stopped line), the on-course
    repeat that is still running excepted: it is recorded as "running (on-course repeat, item 2(b))" (AM-19a
    reading 7; OQ8), being waited for as a default candidate is (AM-19 item 2(b)). The record may be written
    while the default runs (AM-19a reading 19). A value whose on-course stop has no repeat yet refuses: a
    record written before the repeat launches would lack its directory (item 2(h)). Without `corrects`, the
    sweep's record (ordinary_record_error: never while DECISION_RECORD_REL has a history). With `corrects` (void
    record path, fault report path), the sweep's corrected record (AM-19a reading 19): written once (never while
    DECISION_RECORD_CORRECTED_REL has a history), to that path (checked when `path`, resolved against the working
    directory, is in the repository), after the committed AM-8a fault report, naming it and the sweep's void
    record at DECISION_RECORD_REL, which must be void now (void_code)."""
    if state["cutoff_utc"] is None or state["now"] < state["cutoff_utc"]:
        raise SelectionRefused("decision_date_not_ended", f"the {state['key']} decision date has not ended: no "
                                                          "decision record is written before it (AM-19 item 2(h))")
    st, unended = state["statuses"], []
    for v, lst in state["atts"].items():
        if v == state["default"]:
            continue
        if st[v]["status"] == "waiting" and st[v]["running"] is None:
            unended.append(f"{v:g} (its on-course repeat has not launched: {st[v]['reason']})")
        for a in lst:
            if a.launched is None or a.run is None or a.run.raw_end or a.stopped is not None:
                continue
            if st[v]["running"] is a:
                continue
            unended.append(a.run_id)
    if unended:
        raise SelectionRefused("decision_record_runs_unended", f"non-default runs or values {unended} have neither "
                                                               "ended nor a stopped line: stop them, commit their "
                                                               "stopped lines and reports, then write the record "
                                                               "(PL-32)")
    code = "decision_record_correction_invalid"
    cor, src = None, state["src"]
    out_rel = src.rel(Path(path).absolute())            # the path as written: relative to the working directory
    if corrects is None:
        err = ordinary_record_error(src, state["key"], out_rel)
        if err is not None:
            raise SelectionRefused(code, err)
    else:
        void_path, report_path = corrects
        fixed = DECISION_RECORD_CORRECTED_REL[state["key"]]
        if _changing_commits(src, fixed):
            raise SelectionRefused(code, f"{fixed} is committed, or was: the sweep's corrected record is written once "
                                         "(AM-19a reading 19)")
        vrel, vdata = require_committed(void_path, src=src, missing=code, uncommitted=code, record=True)
        rrel, rdata = require_committed(report_path, src=src, missing=code, uncommitted=code, record=True)
        cor = {"void_record": {"path": vrel, "sha256": sha256_bytes(vdata)},
               "fault_report": {"path": rrel, "sha256": sha256_bytes(rdata)}, "void_code": ""}
        err = record_place_error(state["key"], fixed if out_rel is None else out_rel, cor)
        if err is not None:
            raise SelectionRefused(code, err)
        cor["void_code"] = void_code(state, cor, out_rel)
    doc = decision_record_doc(state, cor)
    return write_json_once(path, doc), doc


# ----------------------------------------------------------------------------------- settle
def shortfall_message(state: dict, *, alpha_cut: bool = False) -> str:
    key, st = state["key"], state["statuses"]
    name = "lambda" if key == "lambda_logit" else "alpha"
    grid = [float(v) for v in state["sweep"]["grid"]]
    fin = [v for v in grid if st[v]["status"] == "finished"]
    div = [v for v in grid if st[v]["status"] == "diverged"]
    cut = [v for v in grid if st[v]["status"] == "cut"]
    missing = [v for v in grid if v not in fin + div]
    entries = []
    for v in grid:
        s = st[v]
        if s["status"] in ("waiting", "cut"):
            rid = s["attempt"].run_id if s["attempt"] is not None else "-"
            entries.append(f"{v:g}: {rid} ({s['code'] or s['status']}{': ' + s['reason'] if s['reason'] else ''})")
    date_txt = state["decision_date"] or f"{state['calendar_date']}, or later under item 2(a)"
    if alpha_cut:
        phase = (f"no non-default alpha candidate was launched before the end of the alpha decision date "
                 f"{date_txt}: the sweep is cut, alpha = {state['default']:g} (AM-16 item 2; AM-19 item 2(g)), and the "
                 "committed decision record that states the cut is required (item 2(h)).")
    elif not state["after_c"]:
        phase = (f"the {name} decision date {date_txt} (Schedule {state['schedule']['in_force']}) has not ended: the "
                 "selection waits until every candidate has finished or diverged; after the end of the date a "
                 "non-default candidate that has neither is cut, against a committed decision record (item 2(h)).")
    elif any(st[v]["status"] == "waiting" for v in grid):
        d = st[state["default"]]
        phase = ((f"the default candidate {name} = {state['default']:g} has not finished: the selection waits for it "
                  "(item 2(d)). " if d["status"] == "waiting" else "")
                 + ("An on-course stop's repeat is waited for once (item 2(b))." if any(
                     st[v]["status"] == "waiting" for v in grid if v != state["default"]) else ""))
    else:
        phase = (f"the decision date {date_txt} has ended and {cut} are cut: the selection needs the committed "
                 "decision record (item 2(h)): run --write-decision-record, commit and push it, and enter its "
                 "sha256 in the decision log.")
    tail = (" A cut alpha sweep means alpha = 50 (AM-16 item 2; AM-19 item 2(g)) and no selection file."
            if key == "alpha_cwd" and not alpha_cut else "")
    return (f"{name} shortfall: {len(fin)} of {len(grid)} candidates finished and {len(div)} diverged; grid "
            f"values without a finished or diverged run: {missing}; waiting or cut: {entries} (diverged "
            f"candidates are not shortfalls: pass their directories). AM-19 item 2: {phase}{tail} No selection "
            "was made.")


def settle(state: dict, decision_record=None) -> dict:
    """After derive_sweep: a sweep record deleted after its commit refuses first (decision_record_rewritten, a STOP
    that no wait lifts; AM-19a readings 18 and 19); then (k) alpha's AM-16 cut (exit 5 with a verified record;
    exit 3 without), (j) the shortfall (exit 3), then the decision record (verified when present; required with a
    cut), and the candidates of the rule (AM-19 items 2(e)-(h); PL-20(b))."""
    st, key = state["statuses"], state["key"]
    for rel in (DECISION_RECORD_REL[key], DECISION_RECORD_CORRECTED_REL[key]):
        err = deleted_record_error(state["src"], rel)
        if err is not None:
            raise SelectionRefused("decision_record_rewritten", err)
    if alpha_sweep_cut(state):
        record = load_record(state, decision_record)
        if record is None:
            raise SelectionRefused("shortfall_am19_item2", shortfall_message(state, alpha_cut=True))
        raise SweepCut(record["sha256"], f"no non-default alpha candidate was launched before the end of "
                                         f"{state['decision_date']}: alpha = {state['default']:g} (AM-16 item 2; "
                                         f"AM-19 item 2(g)); decision record {record['path']} {record['sha256']}")
    if any(s["status"] == "waiting" for s in st.values()):
        raise SelectionRefused("shortfall_am19_item2", shortfall_message(state))
    record = load_record(state, decision_record)
    grid = [float(v) for v in state["sweep"]["grid"]]
    cut = [st[v] for v in grid if st[v]["status"] == "cut"]
    if cut and record is None:
        raise SelectionRefused("shortfall_am19_item2", shortfall_message(state))
    return {"finished": [candidate_record(st[v]["attempt"].run, key) for v in grid if st[v]["status"] == "finished"],
            "diverged": [candidate_record(st[v]["attempt"].run, key) for v in grid if st[v]["status"] == "diverged"],
            "cut": [{"value": s["value"], "reason": s["reason"],
                     "run_id": None if s["attempt"] is None else s["attempt"].run_id} for s in cut],
            "record": record}


# -------------------------------------------------------------------------------------- rule
def boundary_kind(winner: float, sweep: dict, finished: list, removed: list) -> tuple[bool, str | None, list]:
    """AM-19 item 2(e), PL-19: grid_end when the winner is a registered grid end; edge_of_finished_set when it
    is the smallest finished value while a removed (diverged or cut) value is smaller, or the largest while
    one is larger. edge_removed names each such value with the rule that removed it (AM-7a or AM-19),
    whenever the edge condition holds, also under grid_end. `removed` is [(value, rule)]."""
    fin = sorted(finished)
    side = []
    if winner == fin[0]:
        side += [(v, rule) for v, rule in removed if v < winner]
    if winner == fin[-1]:
        side += [(v, rule) for v, rule in removed if v > winner]
    edge_removed = [{"value": v, "rule": rule} for v, rule in sorted(set(side))]
    grid_end = winner in [float(b) for b in sweep["boundary"]]
    kind = "grid_end" if grid_end else ("edge_of_finished_set" if edge_removed else None)
    return grid_end or bool(edge_removed), kind, edge_removed


def apply_rule(cands: list[dict], sweep: dict, band: float, name: str, *, diverged=(), cut=()) -> dict:
    """Highest best value among the FINISHED candidates wins; candidates within `band` of the best
    (inclusive, decimal) are tied; a tie goes to the registered default when the rule has one and it is
    tied, else to the smallest value; a sole finished candidate wins with no band and no tie (AM-7a).
    The finished, diverged and cut values together must be the full grid, each once. PL-18: every entry of
    `cands` is finished with a best_val, no diverged or cut entry carries a best_val (partial scores never
    enter a selection), and the default candidate is among the finished (partial_input otherwise). A winner
    at a grid end, or at an edge of the finished set next to a diverged or cut value, is a boundary result."""
    diverged, cut = list(diverged), list(cut)
    grid = [float(v) for v in sweep["grid"]]
    values = sorted([c["value"] for c in cands] + [c["value"] for c in diverged] + [c["value"] for c in cut])
    if values != sorted(grid):
        raise SelectionRefused("partial_input", f"{name} candidates {values} (finished, diverged and cut) != "
                                                f"the full grid {grid}")
    if not cands:
        raise SelectionRefused("partial_input", f"no finished {name} candidate")
    if any(c.get("status") != "finished" or not _finite_number(c.get("best_val")) for c in cands) \
            or any("best_val" in c for c in diverged + cut):
        raise SelectionRefused("partial_input", f"the {name} rule runs on finished candidates with a score of "
                                                "record only; a diverged or cut candidate carries no best_val "
                                                "(AM-7a item 2; AM-19 item 2(e))")
    if float(sweep["default_candidate"]) not in [c["value"] for c in cands]:
        raise SelectionRefused("partial_input", f"the default candidate {name} = {float(sweep['default_candidate']):g} "
                                                "is not among the finished: it is never cut, and its divergence "
                                                "refuses the selection")
    trace = []
    for c in sorted(diverged, key=lambda c: c["value"]):
        trace.append(f"{name}={c['value']:g} run={c['run_id']}: diverged (AM-7a) at iter "
                     f"{c['abort']['iter']} under {c['abort']['rule']}; excluded from the selection")
    for c in sorted(cut, key=lambda c: c["value"]):
        trace.append(f"{name}={c['value']:g}: {CUT_LABEL}, reason {c['reason']}; excluded from the selection")
    if len(cands) == 1:
        winner = cands[0]
        tied = [winner["value"]]
        best = dec(winner["best_val"])
        trace.append(f"sole finished candidate (AM-7a): no band, no tie -> {name} = {winner['value']:g} "
                     f"(run={winner['run_id']} best_val={winner['best_val']!r})")
    else:
        best = max(dec(c["best_val"]) for c in cands)
        band_d = dec(band)
        trace.append(f"best {name}-candidate value = {float(best)!r}; tie band = {band!r} (inclusive: "
                     f"best - value <= band, decimal on the recorded values)")
        tied_c = []
        for c in sorted(cands, key=lambda c: c["value"]):
            gap = best - dec(c["best_val"])
            inside = gap <= band_d
            trace.append(f"{name}={c['value']:g} run={c['run_id']} best_val={c['best_val']!r} gap={gap} "
                         f"-> {'tied' if inside else 'outside the band'}")
            if inside:
                tied_c.append(c)
        tied = [c["value"] for c in tied_c]
        rule = sweep["tie"]
        default = float(sweep["default"]) if sweep.get("default") is not None else None
        if rule == "default_if_tied_else_smallest" and default in tied:
            winner = next(c for c in tied_c if c["value"] == default)
            trace.append(f"the tied set {tied} contains the default {default:g} -> {name} = {default:g}")
        elif rule in ("smallest", "default_if_tied_else_smallest"):
            winner = tied_c[0]
            trace.append(f"tied set {tied} -> the smallest {name} = {winner['value']:g}"
                         if len(tied_c) > 1 else f"a single candidate is within the band -> {name} = "
                                                 f"{winner['value']:g}")
        else:
            raise SelectionRefused("rules_format", f"unknown tie rule {rule!r}")
    fin = sorted(c["value"] for c in cands)
    removed = [(c["value"], "AM-7a") for c in diverged] + [(c["value"], "AM-19") for c in cut]
    boundary, kind, edge_removed = boundary_kind(winner["value"], sweep, fin, removed)
    trace.append(f"boundary winner: {boundary} ({kind}; grid ends {sweep['boundary']}, finished set "
                 f"{fin}); the grid is not extended")
    trace.append(f"winner run {winner['run_id']} is {sweep['winner_is']} (an existing run; checkpoint "
                 f"sha256 {winner['ckpt_sha256']})")
    excluded = [{"value": c["value"], "run_id": c["run_id"],
                 "abort": {k: c["abort"][k] for k in ("iter", "rule", "cause", "detail")},
                 "n_val": c.get("n_val"), "best_val_partial": c.get("best_val_partial"),
                 "telemetry_sha256": c.get("telemetry_sha256"), "run_meta_sha256": c.get("run_meta_sha256")}
                for c in sorted(diverged, key=lambda c: c["value"])]
    return {"winner": winner, "tie": len(tied) > 1, "tied": tied, "boundary": boundary,
            "boundary_kind": kind, "best_val": float(best), "rule_trace": trace,
            "excluded_am7a": excluded, "n_finished": len(cands), "n_diverged": len(diverged),
            "n_cut": len(cut), "edge_removed": edge_removed}


# --------------------------------------------------------------------------- selection file
def inputs_last_timestamp(state: dict):
    """The latest timestamp the selection read (PL-17: the lambda selection's T_lo comes from it)."""
    ts = [t for a in state["supplied"] for t in a.run.timestamps()]
    return max(ts) if ts else None


def _val_rows(run: Run, before=None) -> list:
    """[iteration, all-class mIoU] of the run's VAL rows, those earlier than `before` when given."""
    return [[row.get("iter"), row.get("all_class_miou")] for row, t in zip(run.rows, run.row_ts)
            if row.get("event") == "val" and t is not None and (before is None or t < before)]


def _cut_entry(state: dict, cut: dict, winner_run: Run) -> dict:
    """AM-19 item 2(e) and PL-19: a cut candidate's row of the candidate table. Its last iteration is the
    largest over its directories before the end of the date (a repeat launched after it has none)."""
    v, cutoff = cut["value"], state["cutoff_utc"]
    atts = state["atts"][v]
    dirs, iters = [], set()
    for a in atts:
        if a.launched is None or a.run is None:
            continue
        r = a.run
        vals = _val_rows(r, cutoff)
        iters.update(it for it, _ in vals)
        to_cut, total = (a.launch_ts if t is None else t for t in (r.last_row_ts(cutoff), r.last_row_ts()))
        dirs.append({"run_id": a.run_id, "attempt": a.attempt, "launch_time_utc": a.launch_ts,
                     "last_iter_before_cutoff": r.last_train_iter(cutoff), "val_before_cutoff": vals,
                     "gpu_hours_to_cutoff": max(0.0, (to_cut - a.launch_ts) / 3600.0),
                     "gpu_hours_total": max(0.0, (total - a.launch_ts) / 3600.0)})
    wvals = {row.get("iter"): row.get("all_class_miou") for row in winner_run.rows if row.get("event") == "val"}
    before = [a for a in atts if a.launched is not None and a.launch_ts < cutoff]
    marked, inputs, item8 = False, None, False
    if before:
        last = before[-1]
        marked, inputs = stopped_early(last, cutoff, atts[atts.index(last) + 1:])
        e = last.run.ending
        item8 = e is not None and e["kind"] == "diverged" and not meets(e["ts"], cutoff)
    return {"value": v, "reason": cut["reason"], "label": CUT_LABEL, "directories": dirs,
            "last_iter": max((d["last_iter_before_cutoff"] for d in dirs if d["last_iter_before_cutoff"] is not None),
                             default=None),
            "winner_val_same_iters": {str(it): wvals.get(it) for it in sorted(i for i in iters if i is not None)},
            "gpu_hours_to_cutoff": sum(d["gpu_hours_to_cutoff"] for d in dirs),
            "gpu_hours_total": sum(d["gpu_hours_total"] for d in dirs),
            "stopped_early_unexplained": marked, "stopped_early": inputs, "am7a_item_8": item8,
            "on_course": state["statuses"][v]["on_course"]}


def _repeat_entry(state: dict, v: float) -> dict:
    """AM-19 item 2(b): an on-course stop's repeat, its VAL rows reported beside the stopped run's."""
    s = state["statuses"][v]
    stopped, rep = s["repeat_of"], s["attempt"]
    return {"stopped": stopped.run_id, "stopped_val": _val_rows(stopped.run), "repeat": rep.run_id,
            "repeat_val": [] if rep.run is None else _val_rows(rep.run), "status": s["status"],
            "reason": s["reason"]}


def selection_fields(state: dict, settled: dict, res: dict) -> dict:
    """PL-19's fields of a selection file (the format strings stay; every field is additive), with each
    value's on-course evaluations (PL-7) and, where an on-course repeat decided a status, both runs' VAL
    rows (AM-19 item 2(b))."""
    src, log, sch, atts, rec = state["src"], state["log"], state["schedule"], state["atts"], settled["record"]
    winner = next(a.run for lst in atts.values() for a in lst if a.run_id == res["winner"]["run_id"])
    cut = [_cut_entry(state, c, winner) for c in settled["cut"]]
    grid = [float(v) for v in state["sweep"]["grid"]]
    disclosure = None
    if cut:
        disclosure = (f"{len(cut)} of {len(grid)} candidates are {CUT_LABEL}: "
                      + "; ".join(f"{c['value']:g} ({c['reason']})" for c in cut)
                      + f". The selection ran among the finished candidates after the end of {state['decision_date']}"
                      " (AM-19 item 2(e)).")
    return {"schedule": {"in_force": sch["in_force"], "file": sch["file"], "file_sha256": sch["file_sha256"],
                         "decision_log_entry": sch["decision_log_entry"], "calendar_date": state["calendar_date"],
                         "decision_date": state["decision_date"], "cutoff_utc": state["cutoff_utc"],
                         "alpha_cutoff_basis": state["basis"]},
            "launch_log": {"path": LAUNCH_LOG_REL, "sha256": log.sha256, "lines": log.n_lines},
            "decision_record": None if rec is None else {"path": rec["path"], "sha256": rec["sha256"]},
            "decision_record_corrects": None if rec is None else rec["corrects"], "n_cut": len(cut), "cut_am19": cut,
            "stopped_early_unexplained": [c["value"] for c in cut if c["stopped_early_unexplained"]],
            "edge_removed": res["edge_removed"],
            "directories": {f"{v:g}": [a.run_id for a in atts[v] if a.launched is not None] for v in grid},
            "code": {"head": src.head(), "pin": state["pin"], "sweep_select_sha256": sha256_file(Path(__file__))},
            "inputs_last_timestamp_utc": inputs_last_timestamp(state), "shortfall_disclosure": disclosure,
            "clock": {a.run_id: {"clock_backsteps": a.run.clock_backsteps,
                                 "max_backstep_seconds": a.run.max_backstep_seconds}
                      for a in _in_log_order(atts) if a.run is not None},
            "on_course": {f"{v:g}": state["statuses"][v]["on_course"] for v in grid},
            "on_course_repeat": {f"{v:g}": _repeat_entry(state, v) for v in grid
                                 if state["statuses"][v]["repeat_of"] is not None}}


def write_selection(path: Path, doc: dict) -> Path:
    """Write the selection once. An existing selection file is never overwritten."""
    write_json_once(path, doc)
    return Path(path)
