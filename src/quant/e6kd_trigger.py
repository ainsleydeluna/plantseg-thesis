"""Lane L-AM3, B-1 (docs/lane_specs/part2.md lane 7; AM-3; DL-07): the AM-3 trigger of E6-KD.

    score    every input checked (below), then the two literal evaluator commands, one after the other, in this
             session: E3 seed 42's FP32 checkpoint (the file best.json names in the α winner's run) and the QNNPACK
             TorchScript artifact of the epoch that E6 seed 42's selection names (block B), each scored by
             scripts/evaluate_model.py on VAL with the canvas protocol, on the CPU at batch size 1. A failed
             command writes trigger_STOP.json, which spends the trigger eval dir. trigger_eval.json records both.
    decide   every input checked again, then the two summaries: each verified from disk and checked against the
             evaluator values of record, on one code path and one device. E6's re-score must equal the
             selection's value: a difference is refused (e6_rescore_differs), a STOP for diagnosis and a ruling,
             never a choice between versions (AM-21 item 2(c)). The drop is computed in decimal on the recorded
             values (Decimal(repr(x)), the convention of IC:271), and the trigger fires only when it is strictly
             greater than 1.0 percentage point. e6kd_trigger.json (format e6kd_trigger/1) is written once; it is
             deterministic and carries no wall-clock time or date (the dates go into the decision-log row).
    read_trigger_record, trigger_binding
             B-2's launch gate: E6-KD launches only from a committed e6kd_trigger.json with triggered true (lane 7,
             STOP condition). E6-KD's pod checks out the commit that holds the committed trigger JSON, or a later
             master; --expect-head names it. read_trigger_record keeps comparing with git show HEAD:<rel>.

The inputs, in the order they are checked; each failure is a named refusal and nothing is written:
  1. the names (SL-1): every directory flag, and decide's --out, on the given string and then on its resolved form
     (src/quant/qat.py names_test), before any filesystem call;
  2. the five sha256 flags (64 lowercase hex) and the host label;
  3. the trigger eval dir: for score new or empty, outside the repository; for decide no STOP file and its
     trigger_eval.json present; and decide's --out, new and outside the repository;
  4. the λ and α selections: repo-relative, tracked and unchanged at HEAD, each with its pinned sha256
     (src/quant/qat.py read_tracked_selection);
  5. the E3 parent: best.json's checkpoint with the pinned sha256, one run_meta row (stage E3, mode real, seed 42),
     a run_end with checks_passed and no run_abort (resolve_parent); the run the λ and α selections name
     (e6_parent_binding); and its checkpoint sha256 equal to the α winner's ckpt_sha256;
  6. the E6 run: complete (src/quant/qat_artifacts.py read_run_record), its telemetry sha256 the pod's; run_meta
     stage E6, seed 42, mode real, its parent that E3 checkpoint, the λ and α selections of step 4, and the clip of
     record (clip_selection.json's winner);
  7. E6's selection: qat_selection.json with its pinned sha256, of the committed rules, recomputed
     (src/quant/qat_select.py epoch_selection) and equal key by key. With no selection, an E6 whose epochs are all
     excluded is refused as "non-finite: no model" (AM-21 item 2(b)), and any other missing selection is
     incomplete;
  8. the data root: TRAIN and VAL only (src/data/isolation.py).

The smokes alone pass allow_smoke_inputs: it accepts a smoke-mode E6 run and capped evaluator rows, and it is recorded
in every output; a trigger record that carries it is refused at launch (trigger_smoke_inputs).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from . import qat as Q                                                      # noqa: E402
from . import qat_artifacts as A                                            # noqa: E402
from . import qat_select as S                                               # noqa: E402
from .qat import QATRefused                                                 # noqa: E402
from .stages import resolve_quant_stage                                     # noqa: E402

QATIncomplete = A.QATIncomplete
LANE = "L-AM3 (docs/lane_specs/part2.md lane 7), B-1"
TRIGGER_FORMAT = "e6kd_trigger/1"
TRIGGER_EVAL_FORMAT = "e6kd_trigger_eval/1"
TRIGGER_EVAL_RECORD = "trigger_eval.json"
TRIGGER_STOP = "trigger_STOP.json"
TRIGGER_RECORD_REL = "reports/derived/e6kd_trigger.json"     # the committed copy, committed alone
SELECTION_RECORD = "qat_selection.json"                      # E6's block B (scripts/select_qat_epoch.py)
SCORES_DIR = A.SCORES_DIR
SCORE_DIRS = {"e3": f"{SCORES_DIR}/e3", "e6": f"{SCORES_DIR}/e6"}
SEED = 42
E3_STAGE, E6_STAGE = "E3", "E6"
EVALUATOR_SCRIPT = "scripts/evaluate_model.py"               # run from the repository root, verbatim
THRESHOLD_PP = Decimal("1.0")
THRESHOLD_UNIT = "percentage points"
DECISION_RULE = "strictly_greater"
RULE = ("AM-3 (DL-07): E6-KD runs iff the E3 -> E6 drop in dataset-level VAL all-class mIoU is greater than 1.0 "
        "percentage point at seed 42, E6 scored on the converted INT8 model; lane 7: both scored by "
        "scripts/evaluate_model.py on the CPU with the canvas protocol, on one code path and one device, never "
        "best.json values")
ARITHMETIC = ("decimal on the recorded values (IC:271's convention): drop = Decimal(repr(e3)) - Decimal(repr(e6)); "
              "drop_pp = 100 x drop; triggered iff drop_pp > Decimal('1.0')")
CONSEQUENCE = {True: ("E6-KD runs, at seed 42 only (AM-3; DL-07, DL-39, DL-51). Its launch gate reads this file, "
                      "committed alone (read_trigger_record). E6-KD is descriptive and outside the Holm family."),
               False: "E6-KD does not run (AM-3; DL-07)."}
# what the two summaries share: one evaluator commit, metric code, split, class map, torch, device, protocol and
# thread setting, in one image. config_sha256 is not among them: the evaluator hashes the stage, the precision and
# the backend into it (src/eval/artifacts.py build_config_payload), which E3 and E6 do not share.
SAME_CODE_PATH_KEYS = ("run.repo_commit", "run.metric_impl_sha256", "dataset.split_manifest_sha256",
                       "dataset.class_map_sha256", "run.env.torch", "run.env.device", "dataset.preprocess_protocol",
                       "run.eval_runtime.torch_num_threads", "run.eval_runtime.image_digest", "metric_protocol")
NON_NULL_KEYS = ("run.repo_commit", "run.eval_runtime.image_digest")
THREADS_NOTE = ("no --threads and no thread environment variable: torch_num_threads is the evaluator's own record "
                "(summary eval_runtime), as in block B (scripts/qat_epoch_eval.py)")
# the code whose sha256 every record carries (src/quant/ptq.py code_provenance)
TRIGGER_CODE_PATHS = (
    "src/quant/e6kd_trigger.py", "scripts/e6kd_trigger.py", "src/quant/qat.py", "src/quant/qat_artifacts.py",
    "src/quant/qat_select.py", "src/quant/stages.py", "src/quant/runner.py", "src/quant/ptq.py",
    "configs/qat_selection_rules.json", "configs/quant.py", "configs/data.py", "src/data/isolation.py",
    "scripts/evaluate_model.py", "src/eval/artifacts.py", "src/eval/stage_artifacts.py", "src/eval/model_loading.py",
    "src/eval/metrics.py",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _refuse(code: str, message: str):
    raise QATRefused(code, message)


@dataclass(frozen=True)
class TriggerInputs:
    """The CLI's flags (scripts/e6kd_trigger.py); `out` is decide's."""
    e3_run_dir: str
    e3_checkpoint_sha256: str
    e6_run_dir: str
    e6_eval_dir: str
    e6_telemetry_sha256: str
    e6_selection_sha256: str
    lambda_selection: str
    lambda_selection_sha256: str
    alpha_selection: str
    alpha_selection_sha256: str
    trigger_eval_dir: str
    host_label: str
    allow_smoke_inputs: bool = False
    out: str | None = None


def inputs_from_args(args) -> TriggerInputs:
    return TriggerInputs(e3_run_dir=args.e3_run_dir, e3_checkpoint_sha256=args.e3_checkpoint_sha256,
                         e6_run_dir=args.e6_run_dir, e6_eval_dir=args.e6_eval_dir,
                         e6_telemetry_sha256=args.e6_telemetry_sha256, e6_selection_sha256=args.e6_selection_sha256,
                         lambda_selection=args.lambda_selection, lambda_selection_sha256=args.lambda_selection_sha256,
                         alpha_selection=args.alpha_selection, alpha_selection_sha256=args.alpha_selection_sha256,
                         trigger_eval_dir=args.trigger_eval_dir, host_label=args.host_label,
                         allow_smoke_inputs=bool(args.allow_smoke_inputs), out=getattr(args, "out", None))


# ------------------------------------------------------------------ small helpers
def _get(d, dotted: str):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _strict_json(raw: bytes):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(raw.decode("utf-8"), parse_constant=bad)


def _is_score(v) -> bool:
    return isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0


def code_identity() -> dict:
    from .ptq import code_provenance
    prov = code_provenance(TRIGGER_CODE_PATHS)
    return {"git_head": prov["commit"], "code_clean_at_head": prov["clean_at_commit"],
            "code_files_sha256": prov["files_sha256"]}


def drop_decision(e3: float, e6: float) -> dict:
    """The rule on the two recorded values: decimal, strictly greater than 1.0 pp (LB5)."""
    with localcontext() as ctx:
        ctx.prec = 60
        d3, d6 = Decimal(repr(e3)), Decimal(repr(e6))
        drop = d3 - d6
        drop_pp = drop * 100
        triggered = drop_pp > THRESHOLD_PP
    return {"e3_val_decimal": str(d3), "e6_val_decimal": str(d6), "drop": str(drop), "drop_pp": str(drop_pp),
            "drop_pp_float": float(drop_pp), "triggered": bool(triggered)}


def evaluator_command(*, stage: str, precision: str, source_flag: str, source, out_dir, run_id: str) -> list[str]:
    """One literal evaluator command: VAL, canvas protocol, provisional, CPU, batch size 1 (lane 7; LB7)."""
    return [sys.executable, "-B", EVALUATOR_SCRIPT, "--stage", stage, "--model-role", "student",
            "--precision", precision, "--split", "val", "--protocol", "canvas", source_flag, str(source),
            "--artifact-status", "provisional", "--device", "cpu", "--batch-size", "1", "--out-dir", str(out_dir),
            "--run-id", run_id]


def eval_run_ids(e3_checkpoint_sha256: str, e6_epoch: int, e6_checkpoint_sha256: str) -> dict:
    return {"e3": f"e6kd_trigger_e3_{e3_checkpoint_sha256[:12]}",
            "e6": f"e6kd_trigger_e6_e{e6_epoch:02d}_{e6_checkpoint_sha256[:12]}"}


# ------------------------------------------------------------------ the inputs (steps 1-8)
def check_inputs(inp: TriggerInputs, *, purpose: str, repo_root: Path = REPO) -> dict:
    """Steps 1-8 of the module docstring, in order; returns what score and decide use. Nothing is written."""
    smoke = bool(inp.allow_smoke_inputs)
    # 1. the names (SL-1), before any filesystem call
    named = [("--e3-run-dir", inp.e3_run_dir), ("--e6-run-dir", inp.e6_run_dir), ("--e6-eval-dir", inp.e6_eval_dir),
             ("--trigger-eval-dir", inp.trigger_eval_dir)]
    if purpose == "decide":
        named.append(("--out", inp.out))
    for flag, value in named:
        if Q.names_test(value):
            _refuse("forbidden_path", f"{flag} {value!r}: its given or resolved form holds a name SL-1 forbids; "
                                      "refused before any filesystem call")
    # 2. the sha256 flags and the host label
    for flag, value in (("--e3-checkpoint-sha256", inp.e3_checkpoint_sha256),
                        ("--e6-telemetry-sha256", inp.e6_telemetry_sha256),
                        ("--e6-selection-sha256", inp.e6_selection_sha256),
                        ("--lambda-selection-sha256", inp.lambda_selection_sha256),
                        ("--alpha-selection-sha256", inp.alpha_selection_sha256)):
        if not isinstance(value, str) or not _HEX64.match(value):
            _refuse("sha256_format", f"{flag} must be 64 lowercase hex characters, got {value!r}")
    if not inp.host_label or not str(inp.host_label).strip():
        _refuse("host_label", "--host-label is required (the host that scores, recorded in every output)")
    # 3. the trigger eval dir and decide's --out
    from .runner import QuantRunError, check_output_dir
    if purpose == "score":
        try:
            tdir = check_output_dir(inp.trigger_eval_dir, create=False)
        except QuantRunError as e:
            _refuse("trigger_eval_dir_not_outside_repo", str(e))
        if tdir.exists() and (not tdir.is_dir() or any(tdir.iterdir())):
            _refuse("trigger_eval_dir_not_fresh", f"{tdir} exists and is not an empty directory; every scoring "
                                                  "starts in a new trigger eval dir")
        out = None
    else:
        tdir = Path(inp.trigger_eval_dir).resolve()
        if (tdir / TRIGGER_STOP).exists():
            _refuse("stop_present", f"{tdir} holds {TRIGGER_STOP}: it is spent and is never decided")
        if not (tdir / TRIGGER_EVAL_RECORD).is_file():
            raise QATIncomplete("trigger_eval_missing", f"{tdir / TRIGGER_EVAL_RECORD} is missing: run `score` first")
        try:
            out = check_output_dir(inp.out, create=False)
        except QuantRunError as e:
            _refuse("out_not_outside_repo", str(e))
        if out.exists() or out.is_symlink():
            _refuse("out_exists", f"{out} exists; the trigger record is written once")
    # 4. the λ and α selections
    lam = Q.read_tracked_selection(inp.lambda_selection, inp.lambda_selection_sha256, Q.LAMBDA_SELECTION_FORMAT,
                                   "lambda", repo_root)
    alpha = Q.read_tracked_selection(inp.alpha_selection, inp.alpha_selection_sha256, Q.ALPHA_SELECTION_FORMAT,
                                     "alpha", repo_root)
    # 5. the E3 parent: the α winner's run and checkpoint
    e6_stage = resolve_quant_stage("e6")
    parent = Q.resolve_parent(e6_stage, inp.e3_run_dir, inp.e3_checkpoint_sha256, SEED)
    Q.e6_parent_binding(parent, SEED, lam, alpha)
    if parent["checkpoint_sha256"] != alpha["winner"].get("ckpt_sha256"):
        _refuse("e3_checkpoint_not_alpha_winner", f"{parent['checkpoint']} has sha256 {parent['checkpoint_sha256']}; "
                                                  f"the α winner's ckpt_sha256 is "
                                                  f"{alpha['winner'].get('ckpt_sha256')!r}")
    # 6. the E6 run
    rec = A.read_run_record(inp.e6_run_dir)
    A.require_complete(rec)
    if rec["telemetry_sha256"] != inp.e6_telemetry_sha256:
        _refuse("e6_telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}, expected "
                                                f"{inp.e6_telemetry_sha256} (the sha256 recorded on the pod)")
    meta = rec["run_meta"]
    if meta.get("stage") != E6_STAGE:
        _refuse("e6_stage_not_e6", f"{rec['run_id']} is stage {meta.get('stage')!r}, not {E6_STAGE}")
    if meta.get("seed") != SEED or isinstance(meta.get("seed"), bool):
        _refuse("e6_seed_not_42", f"{rec['run_id']} ran seed {meta.get('seed')!r}; the trigger reads seed {SEED}")
    if meta.get("mode") != "real" and not smoke:
        _refuse("e6_mode_not_real", f"{rec['run_id']} has mode {meta.get('mode')!r}; only a real run is the E6 of "
                                    "record")
    mp = meta.get("parent") or {}
    if (mp.get("checkpoint_sha256") != parent["checkpoint_sha256"] or mp.get("run_id") != parent["run_id"]
            or meta.get("source_checkpoint_sha256") != parent["checkpoint_sha256"]):
        _refuse("e6_parent_not_e3_checkpoint", f"{rec['run_id']} records parent {mp.get('run_id')!r} "
                                               f"({mp.get('checkpoint_sha256')}) and source "
                                               f"{meta.get('source_checkpoint_sha256')}; the E3 parent is "
                                               f"{parent['run_id']!r} ({parent['checkpoint_sha256']})")
    sels = meta.get("e6_selections") or {}
    if ((sels.get("lambda_selection") or {}).get("sha256") != lam["sha256"]
            or (sels.get("alpha_selection") or {}).get("sha256") != alpha["sha256"]):
        _refuse("e6_selections_mismatch", f"{rec['run_id']} was launched with λ/α selections "
                                          f"{(sels.get('lambda_selection') or {}).get('sha256')} / "
                                          f"{(sels.get('alpha_selection') or {}).get('sha256')}; the trigger reads "
                                          f"{lam['sha256']} / {alpha['sha256']}")
    clip, cs = meta.get("clip_norm"), meta.get("clip_selection") or {}
    if not (meta.get("clip_source") == "clip_selection" and meta.get("u4_pilot") is False
            and type(clip) is float and clip in Q.CLIP_CANDIDATES and cs.get("winner") == clip
            and isinstance(cs.get("sha256"), str) and _HEX64.match(cs["sha256"])):
        _refuse("e6_clip_binding", f"{rec['run_id']}: clip {clip!r} from {meta.get('clip_source')!r} (u4_pilot "
                                   f"{meta.get('u4_pilot')!r}, clip_selection {cs.get('sha256')!r} winner "
                                   f"{cs.get('winner')!r}); E6 takes the U4 winner bound at launch (AM-4a item 3)")
    # 7. E6's selection (block B)
    rules, rules_sha = S.load_rules()
    sel_p = Path(inp.e6_eval_dir) / SELECTION_RECORD
    if not sel_p.is_file():
        try:
            S.epoch_selection(inp.e6_run_dir, inp.e6_eval_dir, expect_telemetry_sha256=inp.e6_telemetry_sha256,
                              rules=rules, rules_sha256=rules_sha, allow_smoke_inputs=smoke)
            why = "a selection can be made: run scripts/select_qat_epoch.py first"
        except QATRefused as e:
            if e.code == "no_convertible_epoch":
                _refuse("e6_no_model", f"{S.NO_MODEL_ENTRY} (AM-21 item 2(b)): every epoch of {rec['run_id']} is "
                                       "excluded; the AM-3 trigger is not computed until a new amendment settles it")
            why = f"recomputing it is refused [{e.code}]"
        except QATIncomplete as e:
            why = f"recomputing it is incomplete [{e.code}]"
        raise QATIncomplete("e6_selection_missing", f"{sel_p} is missing ({why})")
    raw = sel_p.read_bytes()
    if Q.sha256_bytes(raw) != inp.e6_selection_sha256:
        _refuse("e6_selection_sha256_mismatch", f"{sel_p} has sha256 {Q.sha256_bytes(raw)}, expected "
                                                f"{inp.e6_selection_sha256} (the sha256 in its decision-log row)")
    try:
        stored = _strict_json(raw)
    except ValueError as e:
        stored = {"unreadable": str(e)}
    if not (isinstance(stored, dict) and stored.get("format") == S.EPOCH_SELECTION_FORMAT
            and stored.get("rules_sha256") == rules_sha):
        _refuse("e6_selection_format", f"{sel_p} is not a {S.EPOCH_SELECTION_FORMAT} file of the committed rules "
                                       f"(rules sha256 {rules_sha})")
    recomputed = S.epoch_selection(inp.e6_run_dir, inp.e6_eval_dir, expect_telemetry_sha256=inp.e6_telemetry_sha256,
                                   rules=rules, rules_sha256=rules_sha, allow_smoke_inputs=smoke)
    diff = S.selection_differences(stored, recomputed)
    if diff:
        _refuse("e6_selection_differs", f"{sel_p}: recomputing the selection gives different {diff}")
    winner = recomputed["winner"]
    # 8. the data root: TRAIN and VAL only
    from configs.data import DATA, SPLIT_SIZES
    from src.data import isolation as ISO
    try:
        iso = ISO.assert_trainval_only_root(Path(DATA["root"]), {"train": SPLIT_SIZES["train"],
                                                                 "val": SPLIT_SIZES["val"]})
    except ISO.TrainValIsolationError as e:
        _refuse("data_root_not_trainval_only", f"[{e.code}] {e}")
    E = Path(inp.e6_eval_dir).resolve()
    run_ids = eval_run_ids(parent["checkpoint_sha256"], winner["epoch"], winner["checkpoint_sha256"])
    return {"smoke": smoke, "tdir": tdir, "out": out, "lam": lam, "alpha": alpha, "parent": parent,
            "e6": {"run_id": rec["run_id"], "run_dir": rec["run_dir"], "telemetry_sha256": rec["telemetry_sha256"],
                   "clip_norm": clip, "clip_selection_sha256": cs["sha256"], "eval_dir": str(E)},
            "selection": stored, "selection_sha256": inp.e6_selection_sha256, "rules_sha256": rules_sha,
            "rules": rules, "winner": winner, "provenance_path": E / winner["provenance"]["path"],
            "run_ids": run_ids, "isolation_counts": iso.get("counts"), "data_root": str(DATA["root"])}


def commands(ctx: dict) -> dict:
    """The two literal evaluator commands of this trigger eval dir."""
    T = ctx["tdir"]
    return {"e3": evaluator_command(stage=E3_STAGE, precision="fp32", source_flag="--checkpoint",
                                    source=ctx["parent"]["checkpoint_path"], out_dir=T / SCORE_DIRS["e3"],
                                    run_id=ctx["run_ids"]["e3"]),
            "e6": evaluator_command(stage=E6_STAGE, precision="int8_qat", source_flag="--provenance",
                                    source=ctx["provenance_path"], out_dir=T / SCORE_DIRS["e6"],
                                    run_id=ctx["run_ids"]["e6"])}


def recorded_inputs(ctx: dict) -> dict:
    """What the trigger eval dir was scored from: decide requires it unchanged."""
    w = ctx["winner"]
    return {"e3_run_dir": ctx["parent"]["run_dir"], "e3_run_id": ctx["parent"]["run_id"],
            "e3_checkpoint": ctx["parent"]["checkpoint"], "e3_checkpoint_sha256": ctx["parent"]["checkpoint_sha256"],
            "e6_run_dir": ctx["e6"]["run_dir"], "e6_run_id": ctx["e6"]["run_id"], "e6_eval_dir": ctx["e6"]["eval_dir"],
            "e6_telemetry_sha256": ctx["e6"]["telemetry_sha256"], "e6_selection_sha256": ctx["selection_sha256"],
            "e6_epoch": w["epoch"], "e6_checkpoint_sha256": w["checkpoint_sha256"],
            "e6_artifact_of_record": w["artifact_of_record"], "e6_provenance": w["provenance"],
            "lambda_selection": {"path": ctx["lam"]["path"], "sha256": ctx["lam"]["sha256"]},
            "alpha_selection": {"path": ctx["alpha"]["path"], "sha256": ctx["alpha"]["sha256"]},
            "smoke_inputs": ctx["smoke"]}


# ------------------------------------------------------------------ score
def score(inp: TriggerInputs, *, repo_root: Path = REPO, log=print) -> dict:
    """Check every input, run the two literal evaluator commands, write trigger_eval.json. Raises QATRefused before
    anything is written, or QATIncomplete after writing trigger_STOP.json."""
    from .ptq import write_exclusive
    ctx = check_inputs(inp, purpose="score", repo_root=repo_root)
    T = ctx["tdir"]
    cmds = commands(ctx)
    ident = code_identity()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    head = {"format": TRIGGER_EVAL_FORMAT, "lane": LANE, "inputs": recorded_inputs(ctx),
            "host_label": inp.host_label, "cpu_model": Q.cpu_model(), "git_head": ident["git_head"],
            "code_clean_at_head": ident["code_clean_at_head"], "code_files_sha256": ident["code_files_sha256"],
            "smoke_inputs": ctx["smoke"], "threads_note": THREADS_NOTE, "affinity_cpus": affinity,
            "data_root": ctx["data_root"], "isolation_counts": ctx["isolation_counts"]}
    T.mkdir(parents=True, exist_ok=True)
    (T / SCORES_DIR).mkdir()
    done = {}
    for which in ("e3", "e6"):
        argv = cmds[which]
        t0 = time.time()
        p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True)
        seconds = time.time() - t0
        summ = T / SCORE_DIRS[which] / "summary.json"
        if p.returncode != 0 or not summ.is_file():
            write_exclusive({T / TRIGGER_STOP: A.json_bytes({**head, "status": "STOP", "failed": which,
                                                             "evaluator_exit_code": p.returncode,
                                                             "stdout_tail": p.stdout[-1500:],
                                                             "stderr_tail": p.stderr[-2500:], "scored": done,
                                                             "created_wall_clock": time.time()})})
            raise QATIncomplete("evaluator_failed", f"the {which.upper()} evaluator command exited {p.returncode}; "
                                                    f"{T} is spent ({TRIGGER_STOP}); a new scoring uses a new "
                                                    "trigger eval dir")
        s = json.loads(summ.read_text(encoding="utf-8"))
        done[which] = {"evaluator_command": argv[1:], "evaluator_python": argv[0], "exit_code": p.returncode,
                       "seconds": seconds, "run_id": ctx["run_ids"][which], "score_dir": SCORE_DIRS[which],
                       "summary": {"path": f"{SCORE_DIRS[which]}/summary.json", "sha256": Q.sha256_file(summ)},
                       "all_class_miou": _get(s, "dataset_level.all_class_miou"),
                       "torch_num_threads": _get(s, "run.eval_runtime.torch_num_threads")}
        log(f"[score] {which.upper()}: all-class mIoU {done[which]['all_class_miou']!r} ({seconds:.1f}s)")
    doc = {**head, "status": "scored", "scores": done, "created_wall_clock": time.time()}
    write_exclusive({T / TRIGGER_EVAL_RECORD: A.json_bytes(doc)})
    return {"trigger_eval_dir": str(T), "scores": done, "smoke_inputs": ctx["smoke"]}


# ------------------------------------------------------------------ decide
def summary_problems(s: dict, *, stage: str, precision: str, backend, checkpoint_sha256: str, run_id: str, ev: dict,
                     smoke: bool) -> list[str]:
    """The evaluator values of record (configs/qat_selection_rules.json qat_epoch.evaluator, as block B), for this
    model."""
    want = {"dataset.split": ev["split"], "run.env.device": ev["device"],
            "dataset.preprocess_protocol": ev["preprocess_protocol"], "run.artifact_status": ev["artifact_status"],
            "run.governed_paths_clean": ev["governed_paths_clean"], "run.stage": stage, "run.model_role": "student",
            "run.precision": precision, "run.quant_backend": backend, "run.random_init": False,
            "run.checkpoint_sha256": checkpoint_sha256, "run.run_id": run_id,
            "run.eval_runtime.batch_size": ev["eval_runtime"]["batch_size"],
            "run.eval_runtime.num_workers": ev["eval_runtime"]["num_workers"]}
    bad = [f"{k} = {_get(s, k)!r}, expected {v!r}" for k, v in want.items()
           if _get(s, k) != v or type(_get(s, k)) is not type(v)]
    exp, act = _get(s, "dataset.expected_rows"), _get(s, "dataset.actual_rows")
    fwd = _get(s, "run.eval_runtime.forward_batches")
    if not smoke:
        rows = ev["expected_rows"]
        if not (exp == act == fwd == rows == ev["eval_runtime"]["forward_batches"]):
            bad.append(f"rows expected {exp} / actual {act} / forward batches {fwd}; the rules require {rows}")
        gt = _get(s, "per_class.gt_support")
        if not isinstance(gt, list) or sum(gt) != ev["gt_support_sum"]:
            bad.append(f"sum(per_class.gt_support) = {sum(gt) if isinstance(gt, list) else gt!r}, expected "
                       f"{ev['gt_support_sum']}")
    elif not (isinstance(exp, int) and exp >= 1 and exp == act == fwd):
        bad.append(f"rows expected {exp} / actual {act} / forward batches {fwd} differ")
    return bad


def decide(inp: TriggerInputs, *, repo_root: Path = REPO) -> dict:
    """Check every input and the trigger eval dir, then decide and write --out once. Raises QATRefused (nothing
    written) or QATIncomplete."""
    from src.eval.artifacts import verify_artifact
    from .ptq import write_exclusive
    ctx = check_inputs(inp, purpose="decide", repo_root=repo_root)
    T, out, smoke = ctx["tdir"], ctx["out"], ctx["smoke"]
    raw_ev = (T / TRIGGER_EVAL_RECORD).read_bytes()
    try:
        tev = _strict_json(raw_ev)
    except ValueError as e:
        tev = {"unreadable": str(e)}
    if not (isinstance(tev, dict) and tev.get("format") == TRIGGER_EVAL_FORMAT and tev.get("status") == "scored"
            and isinstance(tev.get("scores"), dict) and isinstance(tev.get("inputs"), dict)):
        _refuse("trigger_eval_format", f"{T / TRIGGER_EVAL_RECORD} is not a scored {TRIGGER_EVAL_FORMAT} record")
    if tev.get("smoke_inputs") is not smoke:
        _refuse("trigger_eval_smoke_inputs", f"{TRIGGER_EVAL_RECORD} was scored with smoke inputs "
                                             f"{tev.get('smoke_inputs')!r}; this decision has {smoke}")
    ident = code_identity()
    now = {"git_head": ident["git_head"], "host_label": inp.host_label, "cpu_model": Q.cpu_model(),
           "inputs": recorded_inputs(ctx)}
    differ = sorted(k for k, v in now.items() if json.dumps(tev.get(k), sort_keys=True) != json.dumps(v, sort_keys=True))
    if differ:
        _refuse("trigger_eval_mismatch", f"{TRIGGER_EVAL_RECORD} differs from this decision in {differ}: one commit, "
                                         "one host and the same inputs score and decide")
    cmds = commands(ctx)
    sc = tev["scores"]
    for which in ("e3", "e6"):
        if (sc.get(which) or {}).get("evaluator_command") != cmds[which][1:]:
            _refuse("evaluator_command_not_literal", f"{which.upper()} was scored by "
                                                     f"{(sc.get(which) or {}).get('evaluator_command')!r}, not the "
                                                     f"literal command {cmds[which][1:]!r}")
    sdir = T / SCORES_DIR
    present = sorted(p.name for p in sdir.iterdir()) if sdir.is_dir() else []
    if present != ["e3", "e6"]:
        _refuse("scores_dir_contents", f"{sdir} holds {present}; expected exactly ['e3', 'e6']")
    ev = ctx["rules"]["qat_epoch"]["evaluator"]
    w = ctx["winner"]
    spec = {"e3": dict(stage=E3_STAGE, precision="fp32", backend=None,
                       checkpoint_sha256=ctx["parent"]["checkpoint_sha256"]),
            "e6": dict(stage=E6_STAGE, precision="int8_qat", backend=ev["quant_backend"],
                       checkpoint_sha256=w["artifact_of_record"]["sha256"])}
    summaries = {}
    for which in ("e3", "e6"):
        d = T / SCORE_DIRS[which]
        try:
            s = verify_artifact(d)
        except Exception as ex:                               # noqa: BLE001 -- a refusal, named
            _refuse("summary_artifact_invalid", f"{d}: {type(ex).__name__}: {ex}")
        if Q.sha256_file(d / "summary.json") != (sc[which].get("summary") or {}).get("sha256"):
            _refuse("summary_sha256_mismatch", f"{d / 'summary.json'} is not the summary {TRIGGER_EVAL_RECORD} "
                                               "records")
        bad = summary_problems(s, run_id=ctx["run_ids"][which], ev=ev, smoke=smoke, **spec[which])
        if _get(s, "run.repo_commit") != tev.get("git_head"):
            bad.append(f"run.repo_commit = {_get(s, 'run.repo_commit')!r}; scored at {tev.get('git_head')!r}")
        if bad:
            _refuse("summary_values", f"{which.upper()}: {bad}")
        summaries[which] = s
    same = {k: _get(summaries["e3"], k) for k in SAME_CODE_PATH_KEYS}
    split = [k for k in SAME_CODE_PATH_KEYS
             if json.dumps(_get(summaries["e3"], k), sort_keys=True) != json.dumps(_get(summaries["e6"], k),
                                                                                 sort_keys=True)]
    split += [f"{k} null" for k in NON_NULL_KEYS if _get(summaries["e3"], k) is None]
    if split:
        _refuse("summaries_not_same_code_path", f"the E3 and E6 summaries differ in {split} (lane 7: same code path "
                                                "and device)")
    e3 = _get(summaries["e3"], "dataset_level.all_class_miou")
    e6 = _get(summaries["e6"], "dataset_level.all_class_miou")
    if not (_is_score(e3) and _is_score(e6)):
        _refuse("score_invalid", f"all_class_miou E3 {e3!r}, E6 {e6!r}: each must be a float in [0, 1]")
    if not (type(w["value"]) is float and e6 == w["value"]):
        _refuse("e6_rescore_differs", f"E6's re-score {e6!r} differs from block B's value {w['value']!r} for epoch "
                                      f"{w['epoch']:02d}: a STOP for diagnosis and a ruling, never a choice between "
                                      "versions (AM-21 item 2(c))")
    dec = drop_decision(e3, e6)
    trig = dec["triggered"]
    sha_ev = Q.sha256_bytes(raw_ev)
    lam, alpha, parent = ctx["lam"], ctx["alpha"], ctx["parent"]
    sum_e3 = {"path": f"{SCORE_DIRS['e3']}/summary.json", "sha256": sc["e3"]["summary"]["sha256"]}
    sum_e6 = {"path": f"{SCORE_DIRS['e6']}/summary.json", "sha256": sc["e6"]["summary"]["sha256"]}
    trace = [f"rule {RULE}",
             f"E3 {e3!r}: {parent['run_id']} {parent['checkpoint']} (sha256 {parent['checkpoint_sha256'][:12]}), "
             f"scored FP32 here (summary {sum_e3['sha256'][:12]}); best.json's value "
             f"{parent['best_val_miou_all_class']!r} is not used",
             f"E6 {e6!r}: {ctx['e6']['run_id']} epoch {w['epoch']:02d} (artifact "
             f"{w['artifact_of_record']['sha256'][:12]}), re-scored here (summary {sum_e6['sha256'][:12]}); equals "
             f"block B's value {w['value']!r} (qat_selection.json {ctx['selection_sha256'][:12]})",
             f"drop = {dec['e3_val_decimal']} - {dec['e6_val_decimal']} = {dec['drop']}; drop_pp = {dec['drop_pp']}",
             (f"{dec['drop_pp']} > {THRESHOLD_PP} -> TRIGGERED" if trig else
              f"{dec['drop_pp']} <= {THRESHOLD_PP} -> NOT TRIGGERED")]
    inputs = {"e3_checkpoint": parent["checkpoint_sha256"], "e3_best_json": parent["best_json_sha256"],
              "e3_run_meta": parent["run_meta_sha256"], "e3_telemetry": parent["telemetry_sha256"],
              "e6_telemetry": ctx["e6"]["telemetry_sha256"], "e6_clip_selection": ctx["e6"]["clip_selection_sha256"],
              "e6_qat_selection": ctx["selection_sha256"], "qat_selection_rules": ctx["rules_sha256"],
              "e6_epoch_checkpoint": w["checkpoint_sha256"], "e6_artifact_of_record": w["artifact_of_record"]["sha256"],
              "e6_provenance": w["provenance"]["sha256"], "e6_block_b_summary": w["summary"]["sha256"],
              "lambda_selection": lam["sha256"], "alpha_selection": alpha["sha256"],
              "e3_trigger_summary": sum_e3["sha256"], "e6_trigger_summary": sum_e6["sha256"], "trigger_eval": sha_ev}
    doc = {"format": TRIGGER_FORMAT, "lane": LANE, "rule": RULE, "seed": SEED,
           "e3_val": e3, "e6_val": e6, "e3_val_decimal": dec["e3_val_decimal"], "e6_val_decimal": dec["e6_val_decimal"],
           "drop": dec["drop"], "drop_pp": dec["drop_pp"], "drop_pp_float": dec["drop_pp_float"],
           "threshold": float(THRESHOLD_PP), "threshold_decimal": str(THRESHOLD_PP), "threshold_unit": THRESHOLD_UNIT,
           "decision_rule": DECISION_RULE, "arithmetic": ARITHMETIC, "triggered": trig, "consequence": CONSEQUENCE[trig],
           "descriptive": True, "holm_family": False,
           "e3": {"run_id": parent["run_id"], "checkpoint": parent["checkpoint"],
                  "checkpoint_sha256": parent["checkpoint_sha256"], "best_json_sha256": parent["best_json_sha256"],
                  "run_meta_sha256": parent["run_meta_sha256"], "telemetry_sha256": parent["telemetry_sha256"],
                  "best_json_value_not_used": parent["best_val_miou_all_class"], "summary": sum_e3,
                  "eval_run_id": ctx["run_ids"]["e3"], "evaluator_command": sc["e3"]["evaluator_command"]},
           "e6": {"run_id": ctx["e6"]["run_id"], "telemetry_sha256": ctx["e6"]["telemetry_sha256"],
                  "clip_norm": ctx["e6"]["clip_norm"], "clip_selection_sha256": ctx["e6"]["clip_selection_sha256"],
                  "qat_selection": {"path": SELECTION_RECORD, "sha256": ctx["selection_sha256"]},
                  "rules_sha256": ctx["rules_sha256"], "epoch": w["epoch"], "checkpoint_sha256": w["checkpoint_sha256"],
                  "artifact_of_record": w["artifact_of_record"], "provenance": w["provenance"],
                  "block_b_summary": w["summary"], "selection_value": w["value"],
                  "excluded_epochs": ctx["selection"].get("excluded_epochs"), "summary": sum_e6,
                  "eval_run_id": ctx["run_ids"]["e6"], "evaluator_command": sc["e6"]["evaluator_command"],
                  "rescore_equals_selection": True},
           "selections": {"lambda": {"path": lam["path"], "sha256": lam["sha256"], "winner": lam["winner"]},
                          "alpha": {"path": alpha["path"], "sha256": alpha["sha256"], "winner": alpha["winner"]}},
           "same_code_path": same, "input_sha256s": inputs,
           "trigger_eval": {"path": TRIGGER_EVAL_RECORD, "sha256": sha_ev},
           "git_head": tev.get("git_head"), "code_clean_at_head": tev.get("code_clean_at_head"),
           "code_files_sha256": tev.get("code_files_sha256"), "host_label": tev.get("host_label"),
           "cpu_model": tev.get("cpu_model"), "smoke_inputs": smoke, "rule_trace": trace}
    write_exclusive({out: A.json_bytes(doc)})
    return doc


def result_line(doc: dict) -> str:
    tail = f"(E3 {doc['e3_val']!r}, E6 {doc['e6_val']!r}; seed {doc['seed']})"
    if doc["triggered"]:
        return f"RESULT: TRIGGERED -- drop {doc['drop_pp']} pp > {doc['threshold_decimal']} pp {tail}"
    return f"RESULT: NOT TRIGGERED -- drop {doc['drop_pp']} pp <= {doc['threshold_decimal']} pp {tail}"


# ------------------------------------------------------------------ B-2's launch gate
def read_trigger_record(rel_path: str | None, expect_sha256: str | None, *, repo_root: Path = REPO) -> dict:
    """The committed e6kd_trigger.json: repo-relative, tracked and unchanged at HEAD, its sha256 pinned, its decision
    following from its own two values, triggered true. Raises QATRefused before anything is written."""
    if not rel_path:
        _refuse("trigger_missing", "E6-KD requires --trigger <repo-relative e6kd_trigger.json> and --trigger-sha256")
    if not isinstance(expect_sha256, str) or not _HEX64.match(expect_sha256):
        _refuse("trigger_sha256_format", "--trigger-sha256 must be 64 lowercase hex characters")
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        _refuse("trigger_path_not_repo_relative", f"--trigger {rel_path!r} must be repo-relative")
    full = Path(repo_root) / rel
    if Q.names_test(full):
        _refuse("trigger_forbidden_path", f"--trigger {rel_path!r} holds a name SL-1 forbids")
    if not full.is_file():
        _refuse("trigger_missing", f"{full} is not a file")
    data = full.read_bytes()
    at_head = Q._git_head_bytes(Path(repo_root), rel.as_posix())
    if at_head is None:
        _refuse("trigger_not_tracked", f"{rel.as_posix()} is not tracked at HEAD: E6-KD launches only from a "
                                       "committed trigger JSON")
    if at_head != data:
        _refuse("trigger_changed_since_head", f"{rel.as_posix()} differs from its bytes at HEAD")
    if Q.sha256_bytes(data) != expect_sha256:
        _refuse("trigger_sha256_mismatch", f"{rel.as_posix()} has sha256 {Q.sha256_bytes(data)}, expected "
                                           f"{expect_sha256}")
    try:
        doc = _strict_json(data)
    except ValueError as e:
        _refuse("trigger_format", f"{rel.as_posix()} is not strict JSON ({e})")
    if not isinstance(doc, dict) or doc.get("format") != TRIGGER_FORMAT:
        _refuse("trigger_format", f"{rel.as_posix()} is not a {TRIGGER_FORMAT} record")
    if doc.get("smoke_inputs") is not False:
        _refuse("trigger_smoke_inputs", f"{rel.as_posix()} records smoke_inputs {doc.get('smoke_inputs')!r}")
    if doc.get("seed") != SEED or isinstance(doc.get("seed"), bool):
        _refuse("trigger_seed", f"{rel.as_posix()} is the trigger of seed {doc.get('seed')!r}, not {SEED}")
    e3, e6 = doc.get("e3_val"), doc.get("e6_val")
    dec = drop_decision(e3, e6) if (_is_score(e3) and _is_score(e6)) else None
    if dec is None or dec["drop_pp"] != doc.get("drop_pp") or dec["triggered"] is not doc.get("triggered"):
        _refuse("trigger_inconsistent", f"{rel.as_posix()}: triggered {doc.get('triggered')!r} and drop_pp "
                                        f"{doc.get('drop_pp')!r} do not follow from e3_val {e3!r} and e6_val {e6!r}")
    if doc.get("triggered") is not True:
        _refuse("trigger_not_triggered", f"{rel.as_posix()} records triggered false (drop {doc.get('drop_pp')} pp): "
                                         "E6-KD does not run")
    return {"path": rel.as_posix(), "sha256": expect_sha256, "record": doc}


def trigger_binding(trigger: dict, *, parent: dict, clip_norm: float, clip_selection_sha256: str, lam: dict,
                    alpha: dict) -> dict:
    """E6-KD's run is the run the trigger read: the same E3 checkpoint and run, E6's clip of record, the same λ and α
    selections. `trigger` is read_trigger_record's result."""
    doc = trigger["record"]
    e3, e6, sel = doc.get("e3") or {}, doc.get("e6") or {}, doc.get("selections") or {}
    if e3.get("checkpoint_sha256") != parent.get("checkpoint_sha256") or e3.get("run_id") != parent.get("run_id"):
        _refuse("trigger_e3_mismatch", f"the trigger read E3 {e3.get('run_id')!r} ({e3.get('checkpoint_sha256')}); "
                                       f"E6-KD's parent is {parent.get('run_id')!r} "
                                       f"({parent.get('checkpoint_sha256')})")
    if e6.get("clip_norm") != clip_norm or e6.get("clip_selection_sha256") != clip_selection_sha256:
        _refuse("trigger_clip_mismatch", f"the trigger's E6 ran clip {e6.get('clip_norm')!r} "
                                         f"({e6.get('clip_selection_sha256')}); E6-KD runs clip {clip_norm!r} "
                                         f"({clip_selection_sha256})")
    if ((sel.get("lambda") or {}).get("sha256") != lam.get("sha256")
            or (sel.get("alpha") or {}).get("sha256") != alpha.get("sha256")):
        _refuse("trigger_selection_mismatch", "the trigger read other λ/α selections than E6-KD's")
    return {"trigger": trigger["path"], "trigger_sha256": trigger["sha256"], "e3_checkpoint_sha256":
            e3["checkpoint_sha256"], "clip_norm": clip_norm}


__all__ = ["ARITHMETIC", "DECISION_RULE", "EVALUATOR_SCRIPT", "SAME_CODE_PATH_KEYS", "SCORE_DIRS", "SEED",
           "THRESHOLD_PP", "TRIGGER_EVAL_FORMAT", "TRIGGER_EVAL_RECORD", "TRIGGER_FORMAT",
           "TRIGGER_RECORD_REL", "TRIGGER_STOP", "TriggerInputs", "check_inputs", "commands", "decide",
           "drop_decision", "eval_run_ids", "evaluator_command", "inputs_from_args", "read_trigger_record",
           "result_line", "score", "summary_problems", "trigger_binding"]
