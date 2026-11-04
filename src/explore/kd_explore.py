"""The exploratory KD entry for N3, A1'' and A2 (audit EXPLORE-AUDIT-1 §6.1, §6.3; lane L-AM24).

Every run is E2 seed 42's recipe of record (A1'': E3 seed 42's) run by the pinned src/training/train_distill.py
main() in mode real, with its every gate, except the one named change:

  N3    stage e2 (logit KD, lambda 0.5, T 4) from arm teacher R2 at its M12 iteration 24000 (iter_24000.pth, sha256
        d5085961..., DL-122), loaded as E3-R2 loaded it: --teacher-ckpt with its sha256 and the record's teacher
        config (the trainer's default; E-63: arm teachers are distilled with the record configuration). No
        substitution: train_distill accepts any teacher checkpoint with its sha256.
  A1''  stage e3 (E3's three terms) with alpha = the N4 value: lambda * mean(L_logitKD) / mean(L_feat) over E3 seed
        42's last 10,000 iterations, read from N4's n4_terms.json, whose telemetry sha256 must be E3 seed 42's
        completion record in the TEST-day manifest (recorded, never chosen). train_distill refuses an alpha off the
        AM-16 grid in a real run, so for this one call its alpha_gate_error admits exactly that value
        (src/explore/patching.py; recorded); run_meta still says alpha_offgrid true.
  A2    stage e2 with DIST (src/explore/dist_loss.py) in place of lambda * KL only, weight 1 (passed as
        --lambda-logit 1): for this one call train_distill's distillation_losses is DIST's (recorded). E2's grid,
        validity mask, ramp, teacher, data, schedule, evaluation and checkpoint rule stay.

The futility stop (audit §6.3; optional, a flag): at iteration 40,000, if the run's VAL all-class mIoU is MORE than
2.0 pp below the comparator's VAL at 40,000 (E2 seed 42's telemetry for N3 and A2, E3 seed 42's for A1''; each
bound to its completion sha256 in the TEST-day manifest), the run stops after writing that VAL row and an
x_futility_stop row; it is recorded as "stopped: futility rule, 40k VAL v vs comparator u", its rows are evidence,
not a result, and it is never a comparator or a D parent (no run_end, so resolve_parent refuses it). It only
records and stops: nothing else changes.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

from . import guard as G

R2_TEACHER_SHA256 = "d5085961e1529d83b1fc13ff04fac83957bcc44c4f70a0635ff0d840083ffab9"   # DL-122
TEACHER_FILE = "iter_24000.pth"
ITEMS = ("N3", "A1pp", "A2")
SEED = 42
ITEM_STAGE = {"N3": "e2", "A1pp": "e3", "A2": "e2"}
ITEM_LAMBDA = {"N3": 0.5, "A1pp": 0.5, "A2": 1.0}
SLOT_E2S42, SLOT_E3S42 = "E2:s42:lambda0.5:80k", "E3:s42:lambda0.5:alpha25:80k"
COMPARATOR_SLOT = {"N3": SLOT_E2S42, "A2": SLOT_E2S42, "A1pp": SLOT_E3S42}
FUTILITY_ITER = 40_000
FUTILITY_MARGIN = Decimal("0.020")                 # "more than 2.0 pp below", in mIoU units
EXIT_FUTILITY = 5
MANIFEST_REL = "reports/derived/tday_manifest.json"
KD_CODE_PATHS = (
    "src/training/train_distill.py", "src/training/train_e1.py", "src/training/losses.py",
    "src/training/kd_arms.py", "src/training/teacher_components.py", "src/training/__init__.py",
    "configs/distill.py", "configs/e1_student.py", "configs/data.py", "configs/augment.py", "configs/model.py",
    "src/data/__init__.py", "src/data/dataset.py", "src/data/transforms.py", "src/data/isolation.py",
    "src/models/__init__.py", "src/models/student.py", "src/distill/__init__.py", "src/distill/teacher.py",
    "src/distill/segnext_teacher.py", "src/distill/nmf_stream.py", "src/distill/cwd_projection.py",
    "src/distill/features.py", "src/distill/export.py", "src/eval/metrics.py", "src/seeds.py",
    "reports/e1_class_weights.json", "configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
CHANGES = {"N3": "teacher: arm teacher R2 (iter_24000.pth, DL-122) in place of the teacher of record",
           "A1pp": "alpha: the N4 value (feature CWD scaled to the logit-KD term) in place of 25",
           "A2": "loss: DIST (beta = gamma = 1, tau = 1, weight 1) in place of lambda * KL"}
COMPARATORS = {"E2 s42": 0.38749080896377563, "E2 seeds 42-44": [0.38749080896377563, 0.3716566562652588,
                                                                    0.3703170716762543],
               "E3 s42": 0.36337557435035706, "arms A/G/F": [0.3509, 0.3659, 0.1747],
               "source": "audit §6.1 (DL-109, DL-120, DL-121, DL-165)"}


class ExploreKDError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class FutilityStop(RuntimeError):
    """Raised by the futility wrapper after the 40k VAL row and its x_futility_stop row are written."""

    def __init__(self, record: dict):
        super().__init__(f"stopped: futility rule, 40k VAL {record['v']!r} vs comparator {record['u']!r}")
        self.record = record


def manifest_entry(slot: str, repo_root: Path = G.REPO) -> dict:
    doc = json.loads((Path(repo_root) / MANIFEST_REL).read_text(encoding="utf-8"))
    hits = [e for e in doc["body"]["entries"] if isinstance(e, dict) and e.get("slot") == slot]
    if len(hits) != 1:
        raise ExploreKDError("manifest_slot", f"{MANIFEST_REL} holds {len(hits)} entries for slot {slot!r}")
    e = hits[0]
    return {"slot": slot, "run_id": e.get("run_id"), "telemetry_sha256": (e.get("completion") or {}).get("sha256"),
            "telemetry": (e.get("completion") or {}).get("record"),
            "checkpoint_sha256": (e.get("checkpoint") or {}).get("sha256")}


def teacher_expectation(item: str) -> tuple[str, str]:
    if item == "N3":
        return R2_TEACHER_SHA256, "arm teacher R2, M12 iteration 24000 (DL-122)"
    from src.quant.qat_kd import preflight_distill
    return preflight_distill().TEACHER_SHA256, "the teacher of record (DL-88)"


def a1pp_alpha(n4_path, n4_sha256: str, repo_root: Path = G.REPO) -> dict:
    """A1'''s alpha from N4's n4_terms.json: its sha256, its alpha formed, its run E3 seed 42 of record (the telemetry
    sha256 of the manifest's E3 s42 completion), its window the last 10,000 iterations."""
    p = Path(n4_path)
    sha = G.sha256_file(p)
    if sha != n4_sha256:
        raise ExploreKDError("alpha_source_sha256", f"{p} has sha256 {sha}, expected {n4_sha256}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    a = doc.get("alpha_a1pp") or {}
    want = manifest_entry(SLOT_E3S42, repo_root)
    tel = (doc.get("telemetry") or {}).get("sha256")
    rm = doc.get("run_meta") or {}
    problems = []
    if doc.get("format") != "x_n4_terms/1":
        problems.append(f"format {doc.get('format')!r}")
    if a.get("status") != "ok" or not isinstance(a.get("value"), float) or not math.isfinite(a["value"]) \
            or a["value"] <= 0:
        problems.append(f"alpha {a.get('status')!r} {a.get('value')!r}")
    if tel != want["telemetry_sha256"]:
        problems.append(f"telemetry {tel} is not E3 seed 42's of record {want['telemetry_sha256']}")
    if rm.get("stage") != "E3" or rm.get("seed") != SEED:
        problems.append(f"run stage {rm.get('stage')!r} seed {rm.get('seed')!r}")
    if (doc.get("window") or {}).get("last") != 10_000:
        problems.append(f"window {(doc.get('window') or {}).get('last')!r}, not 10,000")
    if problems:
        raise ExploreKDError("alpha_source", f"{p}: {'; '.join(problems)}")
    return {"value": a["value"], "repr": a["repr"], "source": str(p.resolve()), "source_sha256": sha,
            "inputs": {k: a[k] for k in ("lambda", "mean_L_logitKD", "mean_L_feat")}, "e3_s42": want}


def comparator_at(telemetry_path, sha256: str, item: str, *, at_iter: int = FUTILITY_ITER,
                  repo_root: Path = G.REPO) -> dict:
    """The comparator's VAL all-class mIoU at `at_iter`, from its telemetry bound to the manifest's completion sha."""
    want = manifest_entry(COMPARATOR_SLOT[item], repo_root)
    p = Path(telemetry_path)
    sha = G.sha256_file(p)
    if sha != sha256 or sha != want["telemetry_sha256"]:
        raise ExploreKDError("comparator_sha256", f"{p} has sha256 {sha}; given {sha256}; the manifest's "
                                                  f"{COMPARATOR_SLOT[item]} completion is {want['telemetry_sha256']}")
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if '"val"' in ln]
    hits = [r for r in rows if r.get("event") == "val" and r.get("iter") == at_iter]
    if len(hits) != 1 or not isinstance(hits[0].get("all_class_miou"), float):
        raise ExploreKDError("comparator_val_row", f"{p} holds {len(hits)} usable VAL rows at iteration {at_iter}")
    return {"u": hits[0]["all_class_miou"], "iter": at_iter, "telemetry": str(p.resolve()), "sha256": sha,
            "slot": COMPARATOR_SLOT[item], "run_id": want["run_id"]}


def futile(v, u, margin: Decimal = None) -> bool:
    """MORE than `margin` (2.0 pp) below: Decimal of the values' repr, strict."""
    margin = FUTILITY_MARGIN if margin is None else margin
    if not isinstance(v, float) or not math.isfinite(v):
        return False                                    # a non-finite VAL is AM-7's val_nonfinite abort, not futility
    return Decimal(repr(u)) - Decimal(repr(v)) > margin


def futility_jsonl(orig_jsonl, *, comparator: dict, item: str, at_iter: int | None = None):
    """train_e1._jsonl's replacement inside train_distill for one call: every row is written by the original; after
    the VAL row of iteration `at_iter`, if futile, an x_futility_stop row is written and FutilityStop raised."""
    at = FUTILITY_ITER if at_iter is None else at_iter

    def wrapper(path, row):
        orig_jsonl(path, row)
        if isinstance(row, dict) and row.get("event") == "val" and row.get("iter") == at:
            v, u = row.get("all_class_miou"), comparator["u"]
            if futile(v, u):
                rec = {"event": "x_futility_stop", "item": item, "iter": at, "v": v, "u": u,
                       "margin": str(FUTILITY_MARGIN), "rule": "audit §6.3: more than 2.0 pp below the comparator's VAL "
                                                               "at 40,000; evidence, not a result",
                       "comparator": comparator}
                orig_jsonl(path, rec)
                raise FutilityStop(rec)
    return wrapper


def alpha_gate_admitting(orig, alpha_value: float):
    """train_distill.alpha_gate_error's replacement for one A1'' call: exactly `alpha_value` is admitted (it comes
    from N4's record); every other value goes to the pinned gate."""
    def gate(alpha, *, allow_offgrid: bool, mode: str):
        if alpha is not None and float(alpha) == alpha_value:
            return None
        return orig(alpha, allow_offgrid=allow_offgrid, mode=mode)
    return gate


def trainer_argv(item: str, *, teacher_ckpt: str, teacher_sha256: str, num_workers: int, ckpt_dir: Path,
                 alpha: float | None = None) -> list[str]:
    argv = ["--stage", ITEM_STAGE[item], "--real-run", "--confirm-real-run", "--teacher-ckpt", teacher_ckpt,
            "--teacher-ckpt-sha256", teacher_sha256, "--lambda-logit", repr(ITEM_LAMBDA[item]),
            "--num-workers", str(num_workers), "--ckpt-dir", str(ckpt_dir), "--seed", str(SEED), "--log-every", "50"]
    if item == "A1pp":
        argv += ["--alpha", repr(alpha)]
    return argv


def trainer_dir_name(item: str, run_id: str, alpha: float | None = None) -> str:
    if item == "A1pp":
        from src.training.train_distill import alpha_token
        return f"{run_id}_{alpha_token(alpha)}"
    return run_id
