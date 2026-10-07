"""Lane 6 (L-AM4 + L-AM1q): the QAT epoch and clip selections of record.

    epoch_selection   one run's epoch (AM-4a item 2): the highest converted QNNPACK VAL all-class mIoU,
                      strict >, a tie keeps the earlier epoch; an epoch whose checkpoint state is not finite
                      is excluded whatever score its record holds (AM-21 item 2(a): the checkpoint decides),
                      and an excluded epoch whose not-convertible record disagrees refuses the selection; a
                      run of record with no convertible epoch selects nothing, "non-finite: no model" (item
                      2(b)); a rejected U4 pilot run needs no selection (item 3(d)), and its refusal reports
                      any deviation of the trainer; the fake-quant VAL score never selects
    clip_selection    the U4 clip (AM-4a item 3): each candidate's telemetry is checked against the sha256
                      recorded on the pod; a candidate rejected under AM-21 item 3(a), read from its telemetry
                      and its checkpoints, loses, and its conversion and scoring are reported, never required
                      (item 3(d)); otherwise the higher selected value wins, and a tie within 0.1 pp, exact
                      (|Fraction(a) - Fraction(b)| <= 1/1000), goes to 5.0

The rules are configs/qat_selection_rules.json, checked against configs/quant.py. Every input is
re-verified from disk: the run's record (AM-19), every checkpoint's sha256 and state, the conversion and
score records, and for each scored epoch the chain checkpoint -> provenance -> TorchScript identity ->
evaluator summary. Every refusal is named; a selection is written once.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import torch

from . import qat as Q
from . import qat_artifacts as A
from .qat import QATRefused

REPO = Q.REPO
RULES_PATH = REPO / "configs" / "qat_selection_rules.json"
RULES_FORMAT = "qat_selection_rules/1"
EPOCH_SELECTION_FORMAT = "qat_selection/1"
CLIP_SELECTION_FORMAT = Q.CLIP_SELECTION_FORMAT
QATIncomplete = A.QATIncomplete
EVALUATOR_SCRIPT = "scripts/evaluate_model.py"
REJECTED_NONFINITE = "rejected: non-finite state (AM-21 item 3)"
NO_WINNER = "no winner (AM-21 item 3): a new amendment decides the clip value"
NO_MODEL_ENTRY = "non-finite: no model"                # AM-21 item 2(b): the decision-log entry, that day
# run_meta keys the two pilot runs may differ in (reported, never compared): P27
CLIP_EXEMPT_KEYS = ("clip_norm", "wall_clock", "out_dir", "run_id", "host.hostname", "host.pod_id")
# what one selection's evaluations share, and both pilot runs' evaluations share too (P23, P26)
EVAL_IDENTITY_KEYS = ("git_head", "host_label", "cpu_model", "purpose")
# a stored selection is compared with its recomputation key by key, except where it was mounted
SELECTION_PATH_KEYS = ("run_dir", "eval_dir")


def _get(d: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


# ------------------------------------------------------------------ rules
def load_rules(path=RULES_PATH) -> tuple[dict, str]:
    p = Path(path)
    raw = p.read_bytes()
    rules = json.loads(raw.decode("utf-8"))
    from configs.quant import QUANT
    pilot = QUANT["qat_grad_clip_pilot"]
    want = {("format",): RULES_FORMAT, ("qat_epoch", "epochs"): QUANT["qat"]["epochs"],
            ("qat_epoch", "stages"): ["E5", "E6"], ("qat_epoch", "compare"): "strict_greater",
            ("qat_epoch", "tie"): "earliest_epoch", ("qat_clip", "candidates"): list(pilot["candidates"]),
            ("qat_clip", "stage"): pilot["run_on"], ("qat_clip", "seed"): pilot["seed"], ("qat_clip", "tie"): 5.0,
            ("qat_clip", "band", "numerator"): 1, ("qat_clip", "band", "denominator"): 1000,
            ("qat_clip", "inherited_by"): list(pilot["inherited_by"]),
            ("qat_epoch", "equal_across_evaluations"): list(EVAL_IDENTITY_KEYS), ("qat_epoch", "purpose"): "record"}
    wrong = {".".join(k): _get(rules, ".".join(k)) for k, v in want.items() if _get(rules, ".".join(k)) != v}
    if pilot["tie_band"] != "1/1000":
        wrong["configs/quant.py qat_grad_clip_pilot.tie_band"] = pilot["tie_band"]
    if wrong:
        raise QATRefused("rules_mismatch", f"{p} disagrees with configs/quant.py or this lane's rules: {wrong}")
    return rules, Q.sha256_bytes(raw)


def band(rules: dict) -> Fraction:
    b = rules["qat_clip"]["band"]
    return Fraction(b["numerator"], b["denominator"])


def exact_tie(a: float, b: float, within: Fraction) -> tuple[bool, Fraction]:
    d = abs(Fraction(a) - Fraction(b))
    return d <= within, d


def selection_differences(stored: dict, recomputed: dict) -> list[str]:
    """The keys where a stored qat_selection.json and its recomputation differ (mount paths aside)."""
    keys = (set(stored) | set(recomputed)) - set(SELECTION_PATH_KEYS)
    return sorted(k for k in keys
                  if json.dumps(stored.get(k), sort_keys=True) != json.dumps(recomputed.get(k), sort_keys=True))


# ------------------------------------------------------------------ the run's own records (AM-21 items 2(a), 3(a))
def checkpoint_states(rec: dict, *, kinds: dict | None = None, never=None) -> dict:
    """Item 1(a) applied to every epoch checkpoint of the run: {epoch: (finite, failing, path, sha256)}."""
    if kinds is None:
        kinds = Q.state_kinds(A.prepared_skeleton())
    if never is None:
        never = rec["run_meta"].get("never_observed_modules") or []
    out = {}
    for e in range(1, Q.EPOCHS + 1):
        ck, ck_sha = A.verified_checkpoint(rec, e)
        state = torch.load(ck, map_location="cpu", weights_only=True)["model_state_dict"]
        finite, fails = Q.state_predicate(state, kinds, never)
        del state
        out[e] = (finite, fails, ck, ck_sha)
    return out


def _epoch_list_text(eps) -> str:
    return " ".join(f"e{e:02d}" for e in eps)


def run_rejected(rec: dict, *, checks: dict | None = None) -> dict:
    """AM-21 item 3(a) (CHECK ITEM 11), read from the run's own records: its telemetry (the file whose sha256 was
    taken on the pod) and every epoch checkpoint. Grounds: a row records nonfinite_since_step, or an epoch_end row
    has state_finite false; a train row logs a non-finite loss or pre-clip gradient norm, whatever the state; an
    epoch checkpoint fails item 1(a). A failing checkpoint whose epoch_end row carries no state flag rejects the run
    too, and is reported as a deviation of the trainer. Returns the verdict, its grounds and the deviations."""
    rows, ends = rec["rows"], rec["ends"]
    flagged = (any(r.get("nonfinite_since_step") is not None for r in rows)
               or any(r.get("state_finite") is False for r in ends.values()))
    first = min((r["nonfinite_since_step"] for r in rows if type(r.get("nonfinite_since_step")) is int), default=None)
    logged = sorted(r.get("step") for r in rows if r.get("event") == "train"
                    and {"loss", "grad_norm"} & set(r.get("nonfinite") or {}))
    if checks is None:
        checks = checkpoint_states(rec)
    failing = sorted(e for e, (finite, *_rest) in checks.items() if not finite)
    unflagged = [e for e in failing if (ends.get(e) or {}).get("state_finite") is not False]
    grounds = []
    if flagged:
        grounds.append("its telemetry records a non-finite state"
                       + (f", first found at step {first}" if first is not None else ""))
    if logged:
        grounds.append(f"its telemetry logs a non-finite loss or pre-clip gradient norm at step {logged[0]}"
                       + (f" and {len(logged) - 1} later step(s)" if len(logged) > 1 else ""))
    if failing:
        grounds.append(f"checkpoint(s) {_epoch_list_text(failing)} fail item 1(a)")
    deviations = [f"{_epoch_list_text(unflagged)}: checkpoint(s) failing item 1(a) with no state flag in the "
                  "telemetry (a deviation of the trainer)"] if unflagged else []
    return {"rejected": bool(grounds), "grounds": grounds, "deviations": deviations, "first_nonfinite_step": first,
            "logged_nonfinite_steps": logged, "failing_checkpoints": failing}


def _no_constant(c):
    raise ValueError(f"non-strict JSON constant {c}")


def _record_state(rec: dict, path: Path, stops: list[str], *, smoke: bool) -> tuple[str, str, dict | None]:
    """One record of a rejected run's eval directory, as item 3(d) reports it: (state, detail, the record)."""
    if stops:
        return "stopped", f"STOP file(s) {' '.join(stops)}", None
    if not path.is_file():
        return "missing", f"{path.name} is missing", None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)
    except (OSError, ValueError) as e:
        return "failed", f"{path.name} is unreadable ({e})", None
    if not isinstance(doc, dict):
        return "failed", f"{path.name} is not a JSON object", None
    if doc.get("telemetry_sha256") != rec["telemetry_sha256"] or doc.get("run_id") != rec["run_id"]:
        return "failed", (f"{path.name} names run {doc.get('run_id')!r}, telemetry {doc.get('telemetry_sha256')}; "
                          f"the run is {rec['run_id']!r}"), None
    if doc.get("purpose") != "record":
        return "failed", f"{path.name} was written for purpose {doc.get('purpose')!r}", None
    if bool(doc.get("smoke_inputs")) and not smoke:
        return "failed", f"{path.name} was written from smoke inputs", None
    return "present", "", doc


def _by_epoch(items) -> dict:
    return {r["epoch"]: r for r in (items if isinstance(items, list) else [])
            if isinstance(r, dict) and type(r.get("epoch")) is int}


def rejected_records(rec: dict, eval_dir, *, smoke: bool = False) -> dict:
    """AM-21 item 3(d): a rejected U4 pilot run is still converted and scored as item 2(a) reads, and its records
    are kept; its scores are reported and select nothing. None of them is required: a failed conversion or scoring
    of it (item 2(c), for that run alone) never makes the other run's win wait. Each record is reported present
    (this run's record, of purpose "record"), missing, stopped (a STOP file) or failed (unreadable or not strict
    JSON, not an object, another run's, not of record, from smoke inputs, not covering every epoch, or a scored epoch
    without a finite score); the scores as qat_epoch_eval.json records them."""
    E = Path(eval_dir)
    cdir = E / A.CONVERTED_DIR
    stops = [n for n in ("convert_STOP.json",) if (E / n).exists()]
    stops += sorted(p.name for p in cdir.glob("e*_STOP.json")) if cdir.is_dir() else []
    c_state, c_detail, conv = _record_state(rec, E / "qat_convert.json", stops, smoke=smoke)
    if conv is not None:
        outcomes = _by_epoch(conv.get("outcomes"))
        if sorted(outcomes) != list(range(1, Q.EPOCHS + 1)):
            c_state, c_detail = "failed", f"qat_convert.json covers epochs {sorted(outcomes)}"
        else:
            bad = [e for e in sorted(outcomes) if outcomes[e].get("status") != "converted"]
            c_detail = (f"converted {Q.EPOCHS - len(bad)}/{Q.EPOCHS}"
                        + (f"; not convertible: {_epoch_list_text(bad)}" if bad else ""))
    s_stops = [n for n in ("score_STOP.json",) if (E / n).exists()]
    s_state, s_detail, evd = _record_state(rec, E / "qat_epoch_eval.json", s_stops, smoke=smoke)
    scores = {}
    if evd is not None:
        rows = _by_epoch(evd.get("epochs"))
        if sorted(rows) != list(range(1, Q.EPOCHS + 1)):
            s_state, s_detail = "failed", f"qat_epoch_eval.json covers epochs {sorted(rows)}"
        else:
            scored = [e for e in sorted(rows) if rows[e].get("status") == "scored"]
            unfit = [e for e in scored if not (isinstance(rows[e].get("all_class_miou"), float)
                                               and math.isfinite(rows[e]["all_class_miou"]))]
            if unfit:
                s_state, s_detail = "failed", f"qat_epoch_eval.json holds no finite score for {_epoch_list_text(unfit)}"
            else:
                scores = {f"e{e:02d}": rows[e]["all_class_miou"] for e in scored}
                s_detail = ("scores reported, select nothing: " + " ".join(f"{k} {v!r}" for k, v in scores.items())
                            if scores else "no epoch scored")
    return {"conversion": {"state": c_state, "detail": c_detail},
            "scoring": {"state": s_state, "detail": s_detail, "scores": scores}}


def records_text(records: dict) -> str:
    return "; ".join(f"{k} {records[k]['state']}" + (f" ({records[k]['detail']})" if records[k]["detail"] else "")
                     for k in ("conversion", "scoring"))


def rejected_report(clip, rej: dict, records: dict) -> str:
    """select_clip's REPORT line for a rejected candidate: its grounds, any deviation, its records."""
    parts = [f"clip {clip} {REJECTED_NONFINITE} -- item 3(a): {'; '.join(rej['grounds'])}"]
    parts += [f"deviation: {d}" for d in rej["deviations"]]
    parts.append(f"its records (item 3(d), reported, select nothing): {records_text(records)}")
    return "; ".join(parts)


# ------------------------------------------------------------------ one run's epoch
def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise QATRefused("record_unreadable", f"{path}: {e}") from e


def _check_summary(s: dict, *, stage: str, ev: dict, smoke: bool) -> list[str]:
    """The expected evaluator summary values of the rules (configs/qat_selection_rules.json)."""
    bad = []
    rows = ev["expected_rows"]
    want = {"dataset.split": ev["split"], "run.precision": ev["precision"], "run.quant_backend": ev["quant_backend"],
            "run.env.device": ev["device"], "dataset.preprocess_protocol": ev["preprocess_protocol"],
            "run.artifact_status": ev["artifact_status"], "run.governed_paths_clean": ev["governed_paths_clean"],
            "run.stage": stage, "run.model_role": "student",
            "run.eval_runtime.batch_size": ev["eval_runtime"]["batch_size"],
            "run.eval_runtime.num_workers": ev["eval_runtime"]["num_workers"]}
    for k, v in want.items():
        if _get(s, k) != v:
            bad.append(f"{k} = {_get(s, k)!r}, expected {v!r}")
    exp, act = _get(s, "dataset.expected_rows"), _get(s, "dataset.actual_rows")
    fwd = _get(s, "run.eval_runtime.forward_batches")
    if not smoke:
        if not (exp == act == rows and fwd == ev["eval_runtime"]["forward_batches"]):
            bad.append(f"rows expected {exp} / actual {act} / forward batches {fwd}, the rules require {rows} / "
                       f"{ev['eval_runtime']['forward_batches']}")
        gt = _get(s, "per_class.gt_support")
        if not isinstance(gt, list) or sum(gt) != ev["gt_support_sum"]:
            bad.append(f"sum(per_class.gt_support) = {sum(gt) if isinstance(gt, list) else gt!r}, "
                       f"expected {ev['gt_support_sum']}")
    elif not (exp == act == fwd):
        bad.append(f"rows expected {exp} / actual {act} / forward batches {fwd} differ")
    return bad


def epoch_selection(run_dir, eval_dir, *, expect_telemetry_sha256: str | None, rules: dict, rules_sha256: str,
                    allow_smoke_inputs: bool = False) -> dict:
    """Select one run's epoch. Returns the selection; raises QATRefused or QATIncomplete."""
    ev_rules = rules["qat_epoch"]["evaluator"]
    smoke = bool(allow_smoke_inputs)
    A.refuse_test_path(eval_dir, "eval_dir")
    rec = A.read_run_record(run_dir)
    A.require_complete(rec)
    if not expect_telemetry_sha256 or rec["telemetry_sha256"] != expect_telemetry_sha256:
        raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}, "
                                                      f"expected {expect_telemetry_sha256!r}")
    meta = rec["run_meta"]
    if meta.get("mode") != "real" and not smoke:
        raise QATRefused("run_not_real", f"the run's mode is {meta.get('mode')!r}")
    stage = meta["stage"]
    if stage not in rules["qat_epoch"]["stages"]:
        raise QATRefused("stage", f"stage {stage!r} is not a QAT stage of the rules")
    E = Path(eval_dir)
    stops = sorted(p.name for p in (E / A.CONVERTED_DIR).glob("e*_STOP.json")) if (E / A.CONVERTED_DIR).is_dir() else []
    spent = [n for n in ("convert_STOP.json", "score_STOP.json") if (E / n).exists()]
    if spent or stops:
        raise QATRefused("stop_present", f"{E} holds a STOP file {spent + stops}; it is never selected")
    conv_p, ev_p = E / "qat_convert.json", E / "qat_epoch_eval.json"
    for p in (conv_p, ev_p):
        if not p.is_file():
            raise QATIncomplete("eval_record_missing", f"{p} is missing")
    conv, evd = _json(conv_p), _json(ev_p)
    for name, doc in (("qat_convert.json", conv), ("qat_epoch_eval.json", evd)):
        if doc.get("telemetry_sha256") != rec["telemetry_sha256"] or doc.get("run_id") != rec["run_id"]:
            raise QATRefused("eval_record_mismatch", f"{name} names run {doc.get('run_id')!r}, telemetry "
                                                     f"{doc.get('telemetry_sha256')}; the run is {rec['run_id']!r}")
        if doc.get("purpose") != "record":
            raise QATRefused("purpose_not_record", f"{name} was written for purpose {doc.get('purpose')!r}")
        if bool(doc.get("smoke_inputs")) and not smoke:
            raise QATRefused("smoke_inputs", f"{name} was written from smoke inputs")
    one = {k: {conv.get(k), evd.get(k)} for k in EVAL_IDENTITY_KEYS}
    split = {k: sorted(map(str, v)) for k, v in one.items() if len(v) != 1}
    if split:
        raise QATRefused("eval_dir_mixed", f"one selection uses one commit, one host and one eval dir: {split}")
    eval_identity = {k: evd.get(k) for k in EVAL_IDENTITY_KEYS}
    rows = {r["epoch"]: r for r in evd.get("epochs", [])}
    if sorted(rows) != list(range(1, Q.EPOCHS + 1)):
        raise QATIncomplete("eval_incomplete", f"qat_epoch_eval.json covers epochs {sorted(rows)}")
    skeleton = A.prepared_skeleton()
    kinds = Q.state_kinds(skeleton)
    never = meta.get("never_observed_modules") or []
    checks = checkpoint_states(rec, kinds=kinds, never=never)    # item 1(a) on every checkpoint: it decides
    if meta.get("u4_pilot") is True:
        rej = run_rejected(rec, checks=checks)
        if rej["rejected"]:
            raise QATRefused("rejected_pilot_run", f"{rec['run_id']} is a rejected U4 pilot run (AM-21 item 3(a): "
                                                   f"{'; '.join(rej['grounds'] + rej['deviations'])}): it needs no "
                                                   "selection, its scores are reported and select nothing (item "
                                                   "3(d)), and item 2(b) does not apply to it")
    scored, excluded, trace = {}, [], []
    excluded_despite_record = []
    summaries = {}
    for e in range(1, Q.EPOCHS + 1):
        row = rows[e]
        finite, _fails, _ck, ck_sha = checks[e]
        if row.get("checkpoint_sha256") != ck_sha:
            raise QATRefused("chain_checkpoint", f"e{e:02d}: qat_epoch_eval.json names checkpoint "
                                                 f"{row.get('checkpoint_sha256')}, the file has {ck_sha}")
        names = A.epoch_names(e)
        if finite and row.get("status") == "excluded":
            raise QATRefused("exclusion_inconsistent", f"e{e:02d} is excluded but its checkpoint state is finite")
        if not finite:                                   # AM-21 item 2(a): the checkpoint decides
            if row.get("status") == "excluded":
                nc = E / A.CONVERTED_DIR / names["not_convertible"]
                ncd = _json(nc) if nc.is_file() else {}
                run_ident = (rec["run_id"], rec["telemetry_sha256"])
                differs = [k for k, same in (
                    ("rule of the record", ncd.get("rule") == A.NOT_CONVERTIBLE_RULE),
                    ("rule of the row", row.get("rule") == A.NOT_CONVERTIBLE_RULE),
                    ("status", ncd.get("status") == "not convertible"),
                    ("checkpoint", ncd.get("qat_checkpoint_sha256") == ck_sha),
                    ("epoch", ncd.get("epoch") == e),
                    ("run", (ncd.get("run_id"), ncd.get("telemetry_sha256")) == run_ident),
                    *((k, ncd.get(k) == eval_identity[k]) for k in EVAL_IDENTITY_KEYS)) if not same]
                if differs:
                    raise QATRefused("exclusion_inconsistent", f"e{e:02d} is excluded but its not-convertible record "
                                                               f"disagrees: {differs}")
                trace.append(f"e{e:02d} {A.NOT_CONVERTIBLE_RULE}; not scanned")
            else:
                excluded_despite_record.append(e)
                trace.append(f"e{e:02d} {A.NOT_CONVERTIBLE_RULE}: its checkpoint fails item 1(a) though its record "
                             f"says {row.get('status')!r} (value {row.get('all_class_miou')!r}); the checkpoint "
                             "decides; not scanned")
            excluded.append(e)
            continue
        if row.get("status") != "scored":
            raise QATIncomplete("score_missing", f"e{e:02d} has status {row.get('status')!r}")
        prov_p = E / row["provenance"]["path"]
        if not prov_p.is_file():
            raise QATIncomplete("conversion_missing", f"e{e:02d}: {prov_p} is missing")
        if Q.sha256_file(prov_p) != row["provenance"]["sha256"]:
            raise QATRefused("chain_provenance", f"e{e:02d}: {prov_p} is not the scored provenance")
        prov = _json(prov_p)
        ts_p = prov_p.parent / str(prov.get("converted_artifact"))
        if not ts_p.is_file():
            raise QATIncomplete("conversion_missing", f"e{e:02d}: {ts_p} is missing")
        mixed = {k: prov.get(k) for k in EVAL_IDENTITY_KEYS if prov.get(k) != eval_identity[k]}
        if mixed:
            raise QATRefused("eval_dir_mixed", f"e{e:02d} was converted with {mixed}; the evaluations ran with "
                                               f"{eval_identity}")
        ts_sha = Q.sha256_file(ts_p)
        ident = A.read_ts_identity(ts_p)
        chain = {"provenance qat_checkpoint_sha256": prov.get("qat_checkpoint_sha256") == ck_sha,
                 "identity qat_checkpoint_sha256": ident.get("qat_checkpoint_sha256") == ck_sha,
                 "TorchScript file sha256": ts_sha == prov.get("converted_artifact_sha256")
                 == row.get("converted_artifact_sha256"),
                 "epoch": prov.get("epoch") == ident.get("epoch") == e,
                 "run_id": prov.get("run_id") == ident.get("run_id") == rec["run_id"],
                 "stage": prov.get("stage") == ident.get("stage") == stage,
                 "verification": all((prov.get("verification") or {}).get("checks", {}).values())
                 and bool((prov.get("verification") or {}).get("checks")),
                 "no STOP file": not (E / A.CONVERTED_DIR / names["stop"]).exists()}
        if not all(chain.values()):
            raise QATRefused("chain_broken", f"e{e:02d}: {[k for k, v in chain.items() if not v]}")
        sdir = E / row["score_dir"]
        if not (sdir / "summary.json").is_file():
            raise QATIncomplete("score_missing", f"e{e:02d}: {sdir / 'summary.json'} is missing")
        from src.eval.artifacts import verify_artifact
        try:
            s = verify_artifact(sdir)
        except Exception as ex:                               # noqa: BLE001 -- a refusal, named
            raise QATRefused("score_artifact_invalid", f"e{e:02d}: {sdir}: {ex}") from ex
        if Q.sha256_file(sdir / "summary.json") != row["summary"]["sha256"]:
            raise QATRefused("chain_summary", f"e{e:02d}: summary.json is not the scored one")
        if _get(s, "run.repo_commit") != eval_identity["git_head"]:
            raise QATRefused("eval_dir_mixed", f"e{e:02d} was scored at commit {_get(s, 'run.repo_commit')}; the "
                                               f"evaluations ran at {eval_identity['git_head']}")
        want_id = f"{rec['run_id']}_e{e:02d}_{ck_sha[:12]}"
        if _get(s, "run.checkpoint_sha256") != ts_sha or _get(s, "run.run_id") != want_id:
            raise QATRefused("chain_summary", f"e{e:02d}: summary checkpoint {_get(s, 'run.checkpoint_sha256')}, "
                                              f"run_id {_get(s, 'run.run_id')!r}; expected {ts_sha}, {want_id!r}")
        cmd = row.get("evaluator_command") or []
        if not smoke and (not cmd or cmd[0:2] != ["-B", EVALUATOR_SCRIPT]):
            raise QATRefused("evaluator_not_literal", f"e{e:02d} was scored by {cmd[:2]}, not {EVALUATOR_SCRIPT}")
        bad = _check_summary(s, stage=stage, ev=ev_rules, smoke=smoke)
        if bad:
            raise QATRefused("summary_values", f"e{e:02d}: {bad}")
        v = _get(s, "dataset_level.all_class_miou")
        if not isinstance(v, float) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise QATRefused("score_invalid", f"e{e:02d}: all_class_miou {v!r} is not a float in [0, 1]")
        summaries[e] = s
        scored[e] = {"value": v, "checkpoint_sha256": ck_sha, "provenance": row["provenance"],
                     "converted_artifact": {"path": f"{A.CONVERTED_DIR}/{prov['converted_artifact']}",
                                            "sha256": ts_sha},
                     "summary": row["summary"], "run_id": want_id}
    # P25: scores/ holds exactly the epochs the record scored
    sc = E / A.SCORES_DIR
    present = sorted(p.name for p in sc.iterdir()) if sc.is_dir() else []
    expected = [f"e{e:02d}" for e in sorted(rows) if rows[e].get("status") == "scored"]
    if present != expected:
        raise QATRefused("scores_dir_contents", f"{sc} holds {present}; expected exactly {expected}")
    if not scored:
        raise QATRefused("no_convertible_epoch", f"{NO_MODEL_ENTRY}: every epoch of {rec['run_id']} is excluded "
                                                 "(non-finite state, AM-21 item 2(a)); no epoch is selected and the "
                                                 "run stays complete (item 2(b)): enter it in the decision log "
                                                 f"today as \"{NO_MODEL_ENTRY}\"")
    # fields equal across the summaries; non-null ones
    eq = {}
    for k in rules["qat_epoch"]["equal_across_summaries"]:
        vals = {json.dumps(_get(s, k), sort_keys=True) for s in summaries.values()}
        eq[k] = len(vals) == 1
    for k in rules["qat_epoch"]["non_null"]:
        if any(_get(s, k) is None for s in summaries.values()):
            eq[f"{k} non-null"] = False
    if not all(eq.values()):
        raise QATRefused("summaries_differ", f"fields not equal across the scored summaries: "
                                             f"{[k for k, v in eq.items() if not v]}")
    first = summaries[min(summaries)]
    summary_fields = {k: _get(first, k) for k in rules["qat_epoch"]["equal_across_summaries"]}
    # the rule
    trace.insert(0, f"rule configs/qat_selection_rules.json qat_epoch (sha256 {rules_sha256}): highest converted "
                    "QNNPACK VAL all-class mIoU, strict >, tie -> the earlier epoch; fake-quant VAL never selects; "
                    "an epoch whose checkpoint fails item 1(a) is excluded (AM-21 item 2(a))")
    best = None
    ties = []
    for e in sorted(scored):
        v = scored[e]["value"]
        tag = (f"(artifact {scored[e]['converted_artifact']['sha256'][:12]}, "
               f"summary {scored[e]['summary']['sha256'][:12]})")
        if best is None:
            best = e
            ties = [e]
            trace.append(f"e{e:02d} converted {v!r} {tag} -> best e{e:02d}")
        elif v > scored[best]["value"]:
            trace.append(f"e{e:02d} converted {v!r} > best {scored[best]['value']!r} (e{best:02d}) {tag} "
                         f"-> best e{e:02d}")
            best = e
            ties = [e]
        elif v == scored[best]["value"]:
            ties.append(e)
            trace.append(f"e{e:02d} converted {v!r} == best (e{best:02d}) {tag} -> tie; the earlier epoch "
                         f"e{best:02d} is kept")
        else:
            trace.append(f"e{e:02d} converted {v!r} < best {scored[best]['value']!r} (e{best:02d}) {tag}")
    fq = {e: r.get("fake_quant_val_all_class_miou") for e, r in rec["ends"].items()
          if isinstance(r.get("fake_quant_val_all_class_miou"), float)}
    fq_argmax = max(sorted(fq), key=lambda e: fq[e]) if fq else None
    # the winner exists: re-hash it
    w = scored[best]
    ck, _ = A.verified_checkpoint(rec, best)
    if Q.sha256_file(E / w["converted_artifact"]["path"]) != w["converted_artifact"]["sha256"] or \
            Q.sha256_file(E / w["provenance"]["path"]) != w["provenance"]["sha256"]:
        raise QATRefused("winner_changed", f"e{best:02d}'s files changed during selection")
    trace.append(f"winner e{best:02d} (checkpoint sha256 {w['checkpoint_sha256']}); fake-quant VAL argmax "
                 f"{('e%02d' % fq_argmax) if fq_argmax else None} recorded, not used")
    return {"format": EPOCH_SELECTION_FORMAT, "rules_sha256": rules_sha256, "stage": stage, "run_dir": rec["run_dir"],
            "run_id": rec["run_id"], "seed": meta["seed"], "clip_norm": meta["clip_norm"],
            "telemetry_sha256": rec["telemetry_sha256"], "eval_dir": str(E.resolve()),
            "git_head": evd.get("git_head"), "host_label": evd.get("host_label"), "cpu_model": evd.get("cpu_model"),
            "winner": {"epoch": best, "value": w["value"], "checkpoint": f"{Q.EPOCH_DIR}/e{best:02d}.pt",
                       "checkpoint_sha256": w["checkpoint_sha256"], "artifact_of_record": w["converted_artifact"],
                       "provenance": w["provenance"], "summary": w["summary"], "run_id": w["run_id"]},
            "tied_epochs": ties, "excluded_epochs": excluded, "excluded_despite_record": excluded_despite_record,
            "values": {f"e{e:02d}": scored[e]["value"] for e in sorted(scored)},
            "fake_quant_val_argmax_epoch": fq_argmax, "fake_quant_val_used": False,
            "summary_fields": summary_fields, "eval_identity": eval_identity,
            "smoke_inputs": smoke, "rule_trace": trace}


# ------------------------------------------------------------------ the U4 clip
def _recipe(meta: dict) -> dict:
    """run_meta without the P27 exemptions (dotted keys drop one nested entry)."""
    out = json.loads(json.dumps(meta))
    for k in CLIP_EXEMPT_KEYS:
        node, parts = out, k.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return out


def step_fingerprints(rec: dict) -> list[str]:
    rows = sorted((r for r in rec["rows"] if r.get("event") in ("train", "step_error")), key=lambda r: r["step"])
    return [r["batch_sha256"] for r in rows]


def clip_stats(rec: dict) -> dict:
    ends = rec["ends"].values()
    norms = [r["max_pre_clip_norm"] for r in ends if isinstance(r.get("max_pre_clip_norm"), float)]
    return {"clipped_steps": sum(int(r.get("clipped_steps") or 0) for r in ends),
            "max_pre_clip_norm": max(norms) if norms else None}


def clip_selection(candidates: list[tuple[str, str]], *, expect_telemetry_sha256: list[str] | None, rules: dict,
                   rules_sha256: str, allow_smoke_inputs: bool = False, report: list | None = None) -> dict:
    """The U4 clip. `expect_telemetry_sha256` holds each candidate's telemetry sha256 as recorded on the pod, in
    the candidates' order (CHECK ITEM 12). `report`, when given, receives one line per rejected candidate once the
    two are compared, so that a refusal for no winner reports them too (AM-21 item 3(d))."""
    smoke = bool(allow_smoke_inputs)
    cr = rules["qat_clip"]
    if len(candidates) != 2:
        raise QATRefused("candidates", f"select_clip takes exactly two --candidate pairs, got {len(candidates)}")
    expect = list(expect_telemetry_sha256 or [])
    if len(expect) != len(candidates):
        raise QATRefused("expect_telemetry_sha256_required", "select_clip takes one --expect-telemetry-sha256 per "
                                                             "--candidate, in their order (the sha256 recorded on the "
                                                             f"pod): got {len(expect)} for {len(candidates)}")
    runs = []
    for (run_dir, eval_dir), want_sha in zip(candidates, expect):
        A.refuse_test_path(eval_dir, "eval_dir")
        rec = A.read_run_record(run_dir)
        A.require_complete(rec)
        if rec["telemetry_sha256"] != want_sha:
            raise QATRefused("telemetry_sha256_mismatch", f"{rec['telemetry']} has sha256 {rec['telemetry_sha256']}, "
                                                          f"expected {want_sha!r} (the sha256 recorded on the pod)")
        meta = rec["run_meta"]
        if not (meta.get("stage") == cr["stage"] and meta.get("seed") == cr["seed"] and meta.get("u4_pilot") is True
                and meta.get("clip_source") == "u4_pilot" and (meta.get("mode") == "real" or smoke)):
            raise QATRefused("not_a_pilot_run", f"{rec['run_id']}: stage {meta.get('stage')}, seed {meta.get('seed')}, "
                                                f"u4_pilot {meta.get('u4_pilot')}, "
                                                f"clip_source {meta.get('clip_source')}, mode {meta.get('mode')}")
        rej = run_rejected(rec)                              # AM-21 item 3(a): read from the run's own records
        rejected = rej["rejected"]
        sel_p = Path(eval_dir) / "qat_selection.json"
        stored, selection, sel_sha, records = None, None, None, None
        if rejected:
            records = rejected_records(rec, eval_dir, smoke=smoke)   # item 3(d): reported, never required
        else:
            if not sel_p.is_file():
                raise QATIncomplete("epoch_selection_missing", f"{sel_p} is missing: run select_qat_epoch first")
            stored = _json(sel_p)
            if stored.get("format") != EPOCH_SELECTION_FORMAT or stored.get("rules_sha256") != rules_sha256:
                raise QATRefused("epoch_selection_format", f"{sel_p} is not a {EPOCH_SELECTION_FORMAT} file of the "
                                                           f"committed rules ({stored.get('rules_sha256')})")
            selection = epoch_selection(run_dir, eval_dir, expect_telemetry_sha256=want_sha,
                                        rules=rules, rules_sha256=rules_sha256, allow_smoke_inputs=smoke)
            diff = selection_differences(stored, selection)
            if diff:
                raise QATRefused("selection_differs", f"{sel_p}: recomputing gives different {diff}")
            sel_sha = Q.sha256_file(sel_p)
        runs.append({"rec": rec, "meta": meta, "sel_path": str(sel_p) if stored else None, "sel_sha256": sel_sha,
                     "rejected": rejected, "rejection": rej, "records": records, "selection": selection})
    clips = sorted(r["meta"].get("clip_norm") for r in runs)
    if clips != sorted(cr["candidates"]):
        raise QATRefused("clip_values", f"the candidates ran clip {clips}; the pilot compares exactly "
                                        f"{cr['candidates']}")
    a, b = runs
    ra, rb = _recipe(a["meta"]), _recipe(b["meta"])
    diff = sorted(k for k in set(ra) | set(rb)
                  if json.dumps(ra.get(k), sort_keys=True) != json.dumps(rb.get(k), sort_keys=True))
    if diff:
        raise QATRefused("recipe_mismatch", f"the two pilot runs differ in {diff}")
    for k in ("git_head", "image_digest"):
        if a["meta"].get(k) is None:
            raise QATRefused("recipe_identity_null", f"run_meta {k} is null; the pilot needs it recorded")
    fa, fb = step_fingerprints(a["rec"]), step_fingerprints(b["rec"])
    if fa != fb or len(fa) != a["meta"]["total_steps"]:
        first = next((i + 1 for i, (x, y) in enumerate(zip(fa, fb)) if x != y), None)
        total = a["meta"]["total_steps"]
        raise QATRefused("batch_order_differs", f"the batch fingerprints differ (first at step {first}) or are "
                                                f"incomplete ({len(fa)} vs {len(fb)} of {total} steps)")
    live_runs = [r for r in runs if not r["rejected"]]
    if len(live_runs) == 2:
        sa, sb = (r["selection"] for r in live_runs)
        fields = {f"{k}.{f}": [sa[k].get(f), sb[k].get(f)] for k in ("summary_fields", "eval_identity")
                  for f in sorted(set(sa[k]) | set(sb[k])) if sa[k].get(f) != sb[k].get(f)}
        if fields:
            raise QATRefused("pilot_evals_differ", f"the two pilot runs were converted or scored differently: {fields}")
    by_clip = {r["meta"]["clip_norm"]: r for r in runs}
    lo, hi = sorted(by_clip)                                  # 1.0, 5.0
    trace = [f"rule configs/qat_selection_rules.json qat_clip (sha256 {rules_sha256}): reject a non-finite run "
             "(AM-21 item 3); else the higher selected converted VAL all-class mIoU; |Fraction(a) - Fraction(b)| "
             f"<= {band(rules)} -> {cr['tie']}"]
    lines = []
    for c in (lo, hi):
        r = by_clip[c]
        if r["rejected"]:
            trace.append(f"clip {c}: {REJECTED_NONFINITE}")
            trace.append(f"clip {c}: item 3(a): {'; '.join(r['rejection']['grounds'])}")
            trace += [f"clip {c}: deviation: {d}" for d in r["rejection"]["deviations"]]
            trace.append(f"clip {c}: its records (item 3(d), reported, select nothing): {records_text(r['records'])}")
            lines.append(rejected_report(c, r["rejection"], r["records"]))
        else:
            rw = r["selection"]["winner"]
            trace.append(f"clip {c}: e{rw['epoch']:02d} value {rw['value']!r}")
    if report is not None:
        report.extend(lines)
    live = [c for c in (lo, hi) if not by_clip[c]["rejected"]]
    if not live:
        raise QATRefused("no_winner", NO_WINNER)
    tie, d = False, None
    if len(live) == 1:
        win = live[0]
        trace.append(f"winner clip {win}: the other candidate is {REJECTED_NONFINITE.split(':')[0]}")
    else:
        va, vb = by_clip[lo]["selection"]["winner"]["value"], by_clip[hi]["selection"]["winner"]["value"]
        tie, d = exact_tie(va, vb, band(rules))
        if tie:
            win = cr["tie"]
            trace.append(f"|{va!r} - {vb!r}| = {d.numerator}/{d.denominator} <= {band(rules)} -> tie -> clip {win}")
        else:
            win = lo if va > vb else hi
            trace.append(f"|{va!r} - {vb!r}| = {d.numerator}/{d.denominator} > {band(rules)} -> the higher: clip {win}")
    loser = hi if win == lo else lo

    def summary(c):
        r = by_clip[c]
        sel = r["selection"]
        return {"clip_norm": c, "run_id": r["rec"]["run_id"], "run_dir": r["rec"]["run_dir"],
                "telemetry_sha256": r["rec"]["telemetry_sha256"],
                "qat_selection": None if r["rejected"] else {"path": r["sel_path"], "sha256": r["sel_sha256"]},
                "rejected": r["rejected"], "epoch": None if r["rejected"] else sel["winner"]["epoch"],
                "value": None if r["rejected"] else sel["winner"]["value"],
                "rejection": {k: r["rejection"][k] for k in ("grounds", "deviations", "first_nonfinite_step",
                                                             "logged_nonfinite_steps", "failing_checkpoints")},
                "records": r["records"],
                **clip_stats(r["rec"])}
    return {"format": CLIP_SELECTION_FORMAT, "rules_sha256": rules_sha256, "stage": cr["stage"], "seed": cr["seed"],
            "winner": summary(win), "loser": {**summary(loser), "retained": True},
            "tie": tie,
            "difference": None if d is None else {"fraction": f"{d.numerator}/{d.denominator}", "float": float(d)},
            "band": f"{band(rules).numerator}/{band(rules).denominator}",
            "host_differences": {k: [_get(a["meta"], k), _get(b["meta"], k)] for k in CLIP_EXEMPT_KEYS},
            "smoke_inputs": smoke, "report": lines, "rule_trace": trace}


__all__ = ["CLIP_EXEMPT_KEYS", "EPOCH_SELECTION_FORMAT", "EVAL_IDENTITY_KEYS", "NO_MODEL_ENTRY", "NO_WINNER",
           "REJECTED_NONFINITE", "RULES_PATH", "SELECTION_PATH_KEYS", "band", "checkpoint_states", "clip_selection",
           "epoch_selection", "exact_tie", "load_rules", "records_text", "rejected_records", "rejected_report",
           "run_rejected", "selection_differences", "step_fingerprints"]
