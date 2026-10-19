#!/usr/bin/env python3
"""Smoke: L-AM23Q B-2, AM-23's descriptive INT8 stages E6-E2, E7-E2, E6-R1 and E6-R2 (the Phase A plan §4.3, as the
B-2 GO's rulings AQ12, AQ14-AQ17, AQ23-AQ25 and AQ28 amend it).

Synthetic only: no PlantSeg data, no real checkpoint, no GPU, no TEST path. Small synthetic run directories (E2 seed
42, E3-R1, E3-R2 and the α winner E3 run) and records repositories (git, in the work dir: synthetic λ and α
selections, E6-X condition records written by B-1's writer, build_record -> record_bytes -> write_record, and a
decision log whose rows carry B-1's dl_text) stand in for the real inputs (AQ15). Every refusal is reached with every
earlier gate passing. The launch gates run with CUDA, the TRAIN/VAL isolation and the code identity patched in this
process (as scripts/smoke_qat_runner.py does) and restored. The QAT runs are src/quant/qat.py run_qat in smoke mode on
tiny loaders, each in a fresh process; the cross-commit run is O's export (`git archive` with the SL-1 exclude, AQ16).
Scoring runs the evaluator stand-in of scripts/synthetic_qat_fixtures.py in a child process whose environment alone
names the synthetic VAL tree of the work dir; this process never sets PLANTSEG_DATA_ROOT. No created file or folder
name contains "test" or ":" (AQ14); a forbidden-name case uses a never-created path; an audit hook records any open
of a path under the work dir or the repository whose name contains "test". The repository's HEAD and scoped git
status are the same after the run as before it.

  s1  the stage table and the source routing          s7   check-run-meta's AM-23 profile
  s2  the E2 source loader                            s8   Block B's rules files (AQ24)
  s3  resolve_parent's E2 branch                      s9   conversion, the selection and finalize (AQ12)
  s4  E6-E2's parent binding (AQ21)                   s10  the evaluation tables and refusals (AQ17, AQ25)
  s5  E6-X's parent binding (AQ15)                    s11  training invariance (AQ16)
  s6  the launch gates and main's extra (AQ14)        h_   hygiene

    python -B scripts/smoke_am23_quant.py [--work-dir DIR] [--only CASE,CASE]
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True    # no __pycache__ in the repository, with or without -B

import torch  # noqa: E402

import scripts.evaluate_model as EM  # noqa: E402
import scripts.qat_epoch_eval as QEE  # noqa: E402
from src.distill.export import CWD_PROJECTION_KEY  # noqa: E402
from src.eval import artifacts as ART  # noqa: E402
from src.eval import efficiency as EFF  # noqa: E402
from src.eval import stage_artifacts as SA  # noqa: E402
from src.quant import checkpoint as CK  # noqa: E402
from src.quant import e6x_condition as X  # noqa: E402
from src.quant import qat as Q  # noqa: E402
from src.quant import qat_artifacts as A  # noqa: E402
from src.quant import qat_select as S  # noqa: E402
from src.quant import stages as ST  # noqa: E402
from src.quant import x86_latency as X86  # noqa: E402

E2_RUN, E2_CKPT, E2_VALUE = "e2_s42_lambda0.5_80k_a1", "e2_student_best_iter64000.pt", 0.38749080896377563
E3_RUN, E3_CKPT = "e3_s42_lambda0.5_alpha25_80k_a1", "e3_student_best_iter64000.pt"
R_RUN = {"R1": "e3-r1_s42_lambda0.5_alpha25_80k_a1", "R2": "e3-r2_s42_lambda0.5_alpha25_80k_a1"}
R1_NO_RUN = "e3-r1_s42_lambda0.5_alpha25_80k_a2"
VALUE_YES, VALUE_NO, VALUE_R2 = 0.3712345678901234, 0.35, 0.3801
T = X.THRESHOLD
LAM_REL, ALPHA_REL, DL_REL = "reports/derived/lambda_selection.json", "reports/derived/alpha_selection.json", \
    "docs/DECISION_LOG.md"
COND_REL = {x: X.RECORD_REL.format(arm=x) for x in ("R1", "R2")}
HEAD = "6" * 40
CLIP = 5.0
END_T = {"R1": 1792650000.75, "R2": 1792650100.25, "R1no": 1792650200.5}
HOST = "cloud-smoke"
# O (381d73e): the run_qat source, QAT_CODE_PATHS and the evaluator's stages, as B-2 found them
RUN_QAT_SHA256_AT_O = "14155c5a4ef2a4698754d3d98d0b1435c8ceb78ace858120e7d619e57c53b509"
EM_RUN_SHA256 = "b031d3aea7545039dae0aaa98eae809e8ad2cb64af8963da8d94c334115fc4f0"       # P34 (teacher_diag_drift)
O_COMMIT = "381d73eb1ea67a2415344157a11a1e634b1c7bf8"
QAT_CODE_PATHS_AT_O = (
    "src/quant/qat.py", "src/quant/qat_artifacts.py", "src/quant/qat_select.py",
    "src/quant/prepare.py", "src/quant/qconfig.py", "src/quant/stages.py", "src/quant/checkpoint.py",
    "src/quant/calibration.py", "src/quant/x86_latency.py", "src/quant/ptq.py", "src/quant/runner.py",
    "src/quant/__init__.py", "src/__init__.py", "src/models/__init__.py", "src/models/student.py",
    "src/data/__init__.py", "src/data/dataset.py", "src/data/transforms.py", "src/data/isolation.py",
    "src/training/__init__.py", "src/training/train_e1.py", "src/training/losses.py", "src/seeds.py",
    "src/eval/__init__.py", "src/eval/metrics.py", "src/distill/__init__.py", "src/distill/export.py",
    "configs/quant.py", "configs/qat_selection_rules.json", "configs/data.py", "configs/augment.py",
    "configs/model.py", "configs/e1_student.py", "reports/e1_class_weights.json",
    "scripts/run_e5.py", "scripts/run_e6.py", "scripts/qat_epoch_eval.py",
    "scripts/select_qat_epoch.py", "scripts/select_clip.py",
)
AM23_CODE_PATHS = ("src/quant/e6x_condition.py", "configs/qat_selection_rules_am23.json", "scripts/run_e6_e2.py",
                   "scripts/run_e6_r1.py", "scripts/run_e6_r2.py")
ARTIFACT_STAGES_AT_O = ("teacher", "E1", "E2", "E3", "A", "F", "G", "E4", "E5", "E6", "E7")
AM23 = ("E6-E2", "E7-E2", "E6-R1", "E6-R2")
CORRUPTION_STAGES_AT_O = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
Q2_RULES_SHA256 = "76addc57d0367dda9a1163f749ec32c9c240b20736104dc12aa77488344f9568"
NEW_CODES = ("am23_seed", "parent_projection_present", "parent_not_selection_input", "selection_input_missing",
             "parent_checkpoint_not_selected", "parent_arm_mismatch", "parent_not_arm_run", "condition_flags_missing",
             "condition_path_not_of_record", "condition_arm_mismatch", "condition_parent_mismatch", "condition_no",
             "condition_dl_row")
FX: dict = {"n": 0}
CASES: dict = {}
RESULTS: list = []
OPENED: list = []                 # opens of a path under the work dir or the repository whose name contains "test"
AUDIT_BASES: list = []


def case(name: str, code: str | None = None):
    """Register a case; `code` is the refusal code it reaches (h_every_new_code_has_a_case)."""
    if "test" in name.lower():
        raise AssertionError(f"case name {name!r} contains 'test'")

    def deco(fn):
        CASES[name] = (fn, code)
        return fn
    return deco


# ------------------------------------------------------------------ small helpers
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(p) -> str:
    return sha(Path(p).read_bytes())


def jl(rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


def new_dir(kind: str) -> Path:
    """A fresh folder of the work dir with a deterministic name (<kind>_<n>)."""
    FX["n"] += 1
    d = FX["work"] / f"{kind}_{FX['n']:04d}"
    d.mkdir()
    return d


def git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env.update({"GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@example.invalid",
                "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@example.invalid",
                "GIT_AUTHOR_DATE": "2026-10-19T00:00:00Z", "GIT_COMMITTER_DATE": "2026-10-19T00:00:00Z",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout


def repo_status() -> tuple:
    return (git(REPO, "rev-parse", "HEAD").strip(),
            git(REPO, "status", "--porcelain=v1", "--ignored", "--", ".", ":(exclude)docs/reference/reference.pdf",
                ":(exclude)docs/reference", ":(exclude,icase)*test*"))


def code_of(fn, *a, **kw) -> str | None:
    """fn's refusal code; None when it returns; "raised:<type>" for any other exception."""
    try:
        fn(*a, **kw)
    except Q.QATRefused as e:
        return e.code
    except Exception as e:                                      # noqa: BLE001 -- reported in the case's detail
        return f"raised:{type(e).__name__}: {str(e)[:120]}"
    return None


def refused(code: str, fn, *a, **kw) -> tuple[bool, str]:
    got = code_of(fn, *a, **kw)
    return got == code, f"[{got}]"


def refused_msg(code: str, needle: str, fn, *a, **kw) -> tuple[bool, str]:
    """fn refuses with `code` and a message holding `needle` (sites that share a code are told apart)."""
    try:
        fn(*a, **kw)
    except Q.QATRefused as e:
        return e.code == code and needle in str(e), f"[{e.code}] {str(e)[:200]}"
    except Exception as e:                                      # noqa: BLE001
        return False, f"raised:{type(e).__name__}: {str(e)[:160]}"
    return False, "accepted"


def invalid(fn, *a, needle: str, **kw) -> tuple[bool, str]:
    """A SourceCheckpointInvalid whose message holds `needle`."""
    try:
        fn(*a, **kw)
    except CK.SourceCheckpointInvalid as e:
        return needle in str(e), str(e)[:200]
    except Exception as e:                                      # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return False, "accepted"


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ------------------------------------------------------------------ fixtures: run directories
def tiny_state() -> dict:
    return {"head.weight": torch.zeros(2, 2)}


def full_state() -> dict:
    if "full_state" not in FX:
        from src.models.student import build_student
        torch.manual_seed(0)
        FX["full_state"] = {k: v.detach().clone() for k, v in build_student(pretrained=False).state_dict().items()}
    return FX["full_state"]


def e2_payload(**over) -> dict:
    """An E2 distillation checkpoint payload (8487551:src/training/train_distill.py :383-394): stage, terms, λ."""
    p = {"stage": "E2", "iter": 64000, "model_state_dict": tiny_state(), "optimizer_state_dict": {},
         "scheduler_state_dict": {}, "scheduler": "poly", "best_val_miou_all_class": E2_VALUE, "num_classes": 116,
         "teacher_provenance": {"ckpt_sha256": "7" * 64}, "terms": dict(CK.E2_TERMS), "lambda_logit": 0.5}
    p.update(over)
    return p


def e2_meta(**over) -> dict:
    """E2 seed 42's run_meta row (8487551 train_distill.py :963-1003 with launch_fields :678-699)."""
    row = {"event": "run_meta", "stage": "E2", "mode": "real", "seed": 42,
           "terms": dict(CK.E2_TERMS), "projection_params": 0, "lambda_logit": 0.5, "max_iters": 80000,
           "num_classes": 116, "poly_horizon": 80000, "teacher_mock": False, "arm": None, "descriptive": False,
           "parent_of_e4_e7": False, "records_commit": "b" * 40}
    row.update(over)
    return row


def e3_meta(arm: str | None, seed: int = 42, **over) -> dict:
    """An E3 run_meta row: an arm run (arm E3-X, descriptive, never a parent) or E3 of record (arm None)."""
    row = {"event": "run_meta", "stage": "E3", "mode": "real", "seed": seed,
           "terms": {"logit_kd": True, "cwd_feat": True, "cwd_logit": True}, "lambda_logit": 0.5,
           "alpha_cwd": 25.0, "max_iters": 80000, "num_classes": 116, "poly_horizon": 80000, "teacher_mock": False,
           "arm": arm, "descriptive": arm is not None, "parent_of_e4_e7": arm is None, "records_commit": "b" * 40}
    if arm is not None:
        row["target_construction"] = {"K": 2, "views": 2}
    row.update(over)
    return row


def end_row(value, t: float = 1792650000.75, **over) -> dict:
    row = {"event": "run_end", "iter": 80000, "best_val_miou_all_class": value, "best_ckpt": E3_CKPT,
           "checks_passed": True, "wall_clock_end": t}
    row.update(over)
    return row


def make_run(kind: str, name: str, *, value, meta: dict, ckpt_name: str, payload: dict, tel=None, seed: int = 42,
             skip=(), extra=None) -> Path:
    """A run directory: <kind>_run_meta.jsonl, <kind>_telemetry.jsonl, best.json and the checkpoint."""
    d = new_dir("runs") / name
    d.mkdir()
    files = {f"{kind}_run_meta.jsonl": jl([meta]),
             f"{kind}_telemetry.jsonl": jl(tel if tel is not None else
                                           [{"event": "val", "iter": 64000, "all_class_miou": value},
                                            end_row(value, best_ckpt=ckpt_name)]),
             "best.json": json.dumps({"best_ckpt": f"/workspace/kd_runs/{name}/{ckpt_name}",
                                      "best_val_miou_all_class": value}, indent=2).encode("utf-8")}
    files.update(extra or {})
    for n, data in files.items():
        if n not in skip:
            write(d / n, data)
    if ckpt_name not in skip:
        torch.save(payload, d / ckpt_name)
    return d


def e2_run(name: str = E2_RUN, *, seed: int = 42, **kw) -> Path:
    kw.setdefault("meta", e2_meta(seed=seed))
    return make_run("e2", name, value=E2_VALUE, ckpt_name=E2_CKPT, payload=e2_payload(), seed=seed, **kw)


def e3x_run(arm: str, name: str | None = None, *, value=VALUE_YES, t=None, **kw) -> Path:
    kw.setdefault("meta", e3_meta(f"E3-{arm}"))
    payload = {"stage": "E3", "model_state_dict": tiny_state(), "num_classes": 116, "best_val_miou_all_class": value,
               "arm": f"E3-{arm}", "target_construction": {"K": 2, "views": 2}}
    tel = [{"event": "val", "iter": 64000, "all_class_miou": value},
           end_row(value, t if t is not None else END_T[arm])]
    return make_run("e3", name or R_RUN[arm], value=value, ckpt_name=E3_CKPT, payload=payload, tel=tel, **kw)


def resolve(stage_key: str, run_dir: Path, seed: int = 42) -> dict:
    st = Q.resolve_quant_stage(stage_key)
    return Q.resolve_parent(st, run_dir, sha_file(run_dir / (E2_CKPT if st["source_stage"] == "E2" else E3_CKPT)),
                            seed)


# ------------------------------------------------------------------ fixtures: selections, records, decision log
def lam_doc(e2_dir: Path, **over) -> dict:
    """A synthetic λ selection (lambda_selection/1): the winner E2_RUN and its `inputs` entry, as sweep_select
    writes."""
    inputs = [{"run_dir": "/runs/e2_s42_lambda0.25_80k_a1", "checkpoint": E2_CKPT, "best_json_sha256": "1" * 64,
               "run_meta_sha256": "2" * 64},
              {"run_dir": f"/runs/{E2_RUN}", "checkpoint": E2_CKPT,
               "best_json_sha256": sha_file(e2_dir / "best.json"),
               "run_meta_sha256": sha_file(e2_dir / "e2_run_meta.jsonl")}]
    doc = {"format": Q.LAMBDA_SELECTION_FORMAT, "inputs": inputs,
           "winner": {"lambda": 0.5, "run_id": E2_RUN, "best_val": E2_VALUE,
                      "ckpt_sha256": sha_file(e2_dir / E2_CKPT), "checkpoint_present": True}}
    doc.update(over)
    return doc


def alpha_doc(e3_dir: Path) -> dict:
    return {"format": Q.ALPHA_SELECTION_FORMAT, "lambda_logit": 0.5,
            "winner": {"alpha": 25.0, "run_id": E3_RUN, "best_val": float(T),
                       "ckpt_sha256": sha_file(e3_dir / E3_CKPT), "checkpoint_present": True}}


def sel(rel: str, doc_bytes: bytes) -> dict:
    """What read_tracked_selection returns for these bytes at `rel`."""
    return {"path": rel, "sha256": sha(doc_bytes), "format": json.loads(doc_bytes)["format"],
            "winner": json.loads(doc_bytes)["winner"]}


def bytes_of(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def condition(arm: str, run_dir: Path) -> dict:
    """E6-X's condition record, written by B-1's own writer (AQ15): build_record from the run directory's files."""
    meta_data = (run_dir / X.META_NAME).read_bytes()
    tel_data = (run_dir / X.TELEMETRY_NAME).read_bytes()
    best_doc = json.loads((run_dir / X.BEST_JSON_NAME).read_text(encoding="utf-8"))
    ck = run_dir / E3_CKPT
    best = {"value": best_doc["best_val_miou_all_class"], "best_json_sha256": sha_file(run_dir / X.BEST_JSON_NAME),
            "checkpoint": E3_CKPT, "checkpoint_sha256": sha_file(ck), "checkpoint_bytes": ck.stat().st_size}
    end = json.loads(tel_data.decode("utf-8").splitlines()[-1])
    return X.build_record(arm=arm, spec=SimpleNamespace(run_stage=f"E3-{arm}"), run_id=run_dir.name, best=best,
                          meta_data=meta_data, tel_data=tel_data, end=end, lam=FX["lam"], alpha=FX["alpha"],
                          teacher={"path": f"reports/arms/arm_teacher_record_{arm}.json", "sha256": "8" * 64,
                                   "checkpoint_sha256": "9" * 64},
                          launch={"path": "reports/kd_launch_log.jsonl", "sha256": "a" * 64},
                          tool={"script": "scripts/e6x_condition.py", "module": "src/quant/e6x_condition.py",
                                "commit": HEAD, "clean_at_commit": True})


def edited(rec: dict, **fields) -> dict:
    """A writer-built record with named fields changed ("a.b" reaches a nested one); re-serialized by the caller."""
    r = copy.deepcopy(rec)
    for key, value in fields.items():
        d, parts = r, key.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value
    return r


def dl_line(dl_id: str, status: str, text: str) -> str:
    return f"| {dl_id} | {status} | {text} |\n"


def records_repo(*, records=None, files=None, after=None, untracked=None, dl=True) -> Path:
    """A records repository: the λ and α selections, `records` (rel -> record dict, written with B-1's
    record_bytes and write_record), `files` (rel -> bytes) and the decision log, committed once; `after` rewrites
    committed files after the commit and `untracked` adds files after it."""
    root = new_dir("records")
    content = {LAM_REL: FX["lam_bytes"], ALPHA_REL: FX["alpha_bytes"]}
    if dl:
        content[DL_REL] = FX["dl_bytes"]
    content.update(files or {})
    for rel, data in content.items():
        write(root / rel, data)
    for rel, rec in (records if records is not None else FX["records"]).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        X.write_record(root / rel, X.record_bytes(rec))
    git(root, "init", "-q")
    git(root, "add", "--", *sorted(set(content) | set(records if records is not None else FX["records"])))
    git(root, "commit", "-q", "-m", "synthetic records")
    for rel, data in {**(after or {}), **(untracked or {})}.items():
        write(root / rel, data)
    return root


def clip_file() -> tuple[Path, str]:
    p = write(FX["work"] / "clip" / "clip_selection.json",
              json.dumps({"format": Q.CLIP_SELECTION_FORMAT, "winner": {"clip_norm": CLIP}}))
    return p, sha_file(p)


def setup(work: Path) -> None:
    FX["status0"] = repo_status()
    FX["work"] = work
    FX["e2"] = e2_run()
    FX["e2_s43"] = e2_run("e2_s43_lambda0.5_80k_a1", seed=43)
    FX["e2_160k"] = e2_run("e2_s42_lambda0.5_160k_a1")
    FX["e3"] = make_run("e3", E3_RUN, value=float(T), meta=e3_meta(None), ckpt_name=E3_CKPT,
                        payload={"stage": "E3", "model_state_dict": tiny_state(), "num_classes": 116,
                                 "best_val_miou_all_class": float(T)},
                        tel=[end_row(float(T))])
    FX["r1"] = e3x_run("R1")
    FX["r2"] = e3x_run("R2", value=VALUE_R2)
    FX["r1_no"] = e3x_run("R1", R1_NO_RUN, value=VALUE_NO, t=END_T["R1no"])
    FX["lam_bytes"] = bytes_of(lam_doc(FX["e2"]))
    FX["alpha_bytes"] = bytes_of(alpha_doc(FX["e3"]))
    FX["lam"], FX["alpha"] = sel(LAM_REL, FX["lam_bytes"]), sel(ALPHA_REL, FX["alpha_bytes"])
    FX["rec"] = {"R1": condition("R1", FX["r1"]), "R2": condition("R2", FX["r2"]),
                 "R1no": condition("R1", FX["r1_no"])}
    FX["rec_sha"] = {k: sha(X.record_bytes(v)) for k, v in FX["rec"].items()}
    FX["records"] = {COND_REL["R1"]: FX["rec"]["R1"], COND_REL["R2"]: FX["rec"]["R2"]}
    t1, t2 = X.dl_text(FX["rec"]["R1"], FX["rec_sha"]["R1"]), X.dl_text(FX["rec"]["R2"], FX["rec_sha"]["R2"])
    r1, v1 = FX["rec"]["R1"], FX["rec"]["R1"]["value"]["repr"]
    FX["dl_text"] = t1
    rows = [dl_line("DL-900", "RECORDED (synthetic)", t1), dl_line("DL-901", "DECIDED (synthetic)", t1),
            dl_line("DL-902", "RECORDED (synthetic)", t1.replace(FX["rec_sha"]["R1"], "<record>")),
            dl_line("DL-903", "RECORDED (synthetic)", t1.replace("(AM-23 item 3(b)): yes", "(AM-23 item 3(b)): no")),
            dl_line("DL-904", "RECORDED (synthetic)", t1.replace(v1, "<value>")),
            dl_line("DL-905", "RECORDED (synthetic)", t1.replace(r1["best_json"]["sha256"], "<best.json>")),
            dl_line("DL-906", "RECORDED (synthetic)", t1.replace(r1["checkpoint"]["sha256"], "<checkpoint>")),
            dl_line("DL-910", "RECORDED (synthetic)", t2)]
    FX["dl_bytes"] = ("# Decision log (synthetic)\n\n| ID | Status | Entry |\n|---|---|---|\n"
                      + "".join(rows)).encode("utf-8")
    FX["root"] = records_repo()
    FX["clip"], FX["clip_sha"] = clip_file()
    FX["sha"] = {k: sha_file(FX[k] / c) for k, c in (("e2", E2_CKPT), ("e2_s43", E2_CKPT), ("e2_160k", E2_CKPT),
                                                      ("e3", E3_CKPT), ("r1", E3_CKPT), ("r2", E3_CKPT),
                                                      ("r1_no", E3_CKPT))}


def e6x_args(arm: str = "R1", **over):
    a = {"seed": 42, "condition_record": COND_REL[arm], "condition_sha256": FX["rec_sha"][arm],
         "condition_dl": "DL-900" if arm == "R1" else "DL-910"}
    a.update(over)
    return SimpleNamespace(**a)


def e6x_bind(arm: str = "R1", *, parent=None, root=None, lam=None, alpha=None, **over):
    st = Q.resolve_quant_stage(f"e6-{arm.lower()}")
    p = parent if parent is not None else resolve(f"e6-{arm.lower()}", FX[arm.lower()])
    return Q.e6x_parent_binding(st, p, e6x_args(arm, **over), lam or FX["lam"], alpha or FX["alpha"],
                                repo_root=root or FX["root"])


# ================================================================== s1: the stage table and the source routing
@case("s1_new_entries")
def _s1_entries():
    want = {"e6-e2": ("E6-E2", "qat", "E2"), "e7-e2": ("E7-E2", "ptq", "E2"), "e6-r1": ("E6-R1", "qat", "E3"),
            "e6-r2": ("E6-R2", "qat", "E3")}
    got = {k: (v["name"], v["method"], v["source_stage"]) for k, v in ST.QUANT_STAGES.items() if k in want}
    return got == want and list(ST.QUANT_STAGES)[-4:] == list(want), f"{got} {list(ST.QUANT_STAGES)}"


@case("s1_old_entries_unchanged")
def _s1_old():
    got = {k: (ST.QUANT_STAGES[k]["name"], ST.QUANT_STAGES[k]["method"], ST.QUANT_STAGES[k]["source_stage"],
               "parent_arm" in ST.QUANT_STAGES[k]) for k in ("e4", "e5", "e6", "e7")}
    return got == {"e4": ("E4", "ptq", "E1", False), "e5": ("E5", "qat", "E1", False),
                   "e6": ("E6", "qat", "E3", False), "e7": ("E7", "ptq", "E3", False)}, str(got)


@case("s1_parent_arm_and_am23_stages")
def _s1_arm():
    arms = {v["name"]: v.get("parent_arm") for v in ST.QUANT_STAGES.values() if "parent_arm" in v}
    return (arms == {"E6-R1": "E3-R1", "E6-R2": "E3-R2"} and ST.E6X_PARENT_ARM == arms
            and ST.AM23_STAGES == AM23 and ST.SHARED_CALIBRATION_STAGES == ("E4", "E7")), f"{arms} {ST.AM23_STAGES}"


@case("s1_resolve_each")
def _s1_resolve():
    got = [(ST.resolve_quant_stage(s)["key"], ST.resolve_quant_stage(s.lower())["name"]) for s in AM23]
    return got == [("e6-e2", "E6-E2"), ("e7-e2", "E7-E2"), ("e6-r1", "E6-R1"), ("e6-r2", "E6-R2")], str(got)


def routes() -> dict:
    """Which loader load_source_for_stage calls for each stage (the three loaders recorded, not run)."""
    calls = []
    with contextlib.ExitStack() as stack:
        for name in ("load_student_from_e1", "load_student_from_e2", "load_student_from_e3"):
            stack.enter_context(patched(ST, name, lambda p, _n=name, **kw: calls.append(_n) or (None, {})))
        out = {}
        for key in ST.QUANT_STAGES:
            calls.clear()
            ST.load_source_for_stage(key, "unused.pt")
            out[key] = list(calls)
    return out


@case("s1_routes_e2_to_e2_loader")
def _s1_e2():
    r = routes()
    return r["e6-e2"] == r["e7-e2"] == ["load_student_from_e2"], str(r)


@case("s1_routes_e6x_to_e3_loader")
def _s1_e6x():
    r = routes()
    return r["e6-r1"] == r["e6-r2"] == ["load_student_from_e3"], str(r)


@case("s1_routes_e4_to_e7_unchanged")
def _s1_e4e7():
    r = routes()
    return (r["e4"] == r["e5"] == ["load_student_from_e1"] and r["e6"] == r["e7"] == ["load_student_from_e3"]), str(r)


@case("s1_unknown_source_refused")
def _s1_unknown():
    table = {**ST.QUANT_STAGES, "e9-x": {"name": "E9-X", "method": "qat", "source_stage": "E9", "objective": "x"}}
    with patched(ST, "QUANT_STAGES", table):
        try:
            ST.load_source_for_stage("e9-x", "unused.pt")
        except ST.QuantStageError as e:
            return "E9" in str(e), str(e)
    return False, "no QuantStageError"


# ================================================================== s2: the E2 source loader (AM-23 item 4)
def ckpt_file(name: str, payload) -> Path:
    p = new_dir("ckpt") / name
    torch.save(payload, p)
    return p


@case("s2_clean_e2_accepted")
def _s2_clean():
    p = ckpt_file("e2_clean.pt", e2_payload(model_state_dict=full_state()))
    model, ckpt = CK.load_student_from_e2(p)
    keys_ok = sorted(model.state_dict()) == sorted(full_state())
    return keys_ok and ckpt["stage"] == "E2", f"keys {keys_ok}"


@case("s2_refuses_missing_file")
def _s2_missing():
    return invalid(CK.load_e2_source_checkpoint, FX["work"] / "absent_e2.pt", needle="not found")


@case("s2_refuses_not_a_dict")
def _s2_not_dict():
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_list.pt", [1, 2]), needle="expected a dict")


@case("s2_refuses_stage_none")
def _s2_none():
    p = e2_payload()
    del p["stage"]
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_nostage.pt", p), needle="carries no 'stage'")


@case("s2_refuses_stage_e3")
def _s2_e3():
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_as_e3.pt", e2_payload(stage="E3")),
                   needle="is stage 'E3'; E6-E2/E7-E2 take the E2 checkpoint only")


@case("s2_refuses_stage_e1")
def _s2_e1():
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_as_e1.pt", e2_payload(stage="E1")),
                   needle="is stage 'E1'; E6-E2/E7-E2 take the E2 checkpoint only")


@case("s2_refuses_no_model_state_dict")
def _s2_no_msd():
    p = e2_payload()
    del p["model_state_dict"]
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_nomsd.pt", p), needle="no 'model_state_dict'")


@case("s2_refuses_cwd_projection_state_dict")
def _s2_cwd_key():
    p = e2_payload(**{CWD_PROJECTION_KEY: {"weight": torch.zeros(2, 2)}})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_cwdkey.pt", p), needle=CWD_PROJECTION_KEY)


@case("s2_refuses_projection_key_in_student")
def _s2_proj():
    p = e2_payload(model_state_dict={**tiny_state(), "cwd_projection.weight": torch.zeros(2, 2)})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_proj.pt", p), needle="cwd_projection.weight")


@case("s2_refuses_no_tensors")
def _s2_no_tensors():
    p = e2_payload(model_state_dict={"head.weight": 1})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_notensor.pt", p), needle="holds no tensors")


@case("s2_refuses_teacher_keys")
def _s2_teacher():
    p = e2_payload(model_state_dict={**tiny_state(), "teacher.w": torch.zeros(1)})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_teacher.pt", p), needle="teacher state")


@case("s2_refuses_quant_keys")
def _s2_quant():
    p = e2_payload(model_state_dict={**tiny_state(), "x.activation_post_process.scale": torch.zeros(1)})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_quant.pt", p), needle="quantization state")


@case("s2_refuses_quantized_dtype")
def _s2_qdtype():
    q = torch.quantize_per_tensor(torch.zeros(2), 1.0, 0, torch.qint8)
    p = e2_payload(model_state_dict={**tiny_state(), "head.q": q})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_qdtype.pt", p), needle="quantized tensors")


@case("s2_refuses_num_classes")
def _s2_classes():
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_nc.pt", e2_payload(num_classes=115)),
                   needle="num_classes=115")


@case("s2_refuses_terms_not_e2")
def _s2_terms():
    p = e2_payload(terms={"logit_kd": True, "cwd_feat": True, "cwd_logit": True})
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_terms.pt", p), needle="E2's are")


@case("s2_refuses_terms_missing")
def _s2_terms_missing():
    p = e2_payload()
    del p["terms"]
    return invalid(CK.load_e2_source_checkpoint, ckpt_file("e2_noterms.pt", p), needle="records terms None")


@case("s2_refuses_structure")
def _s2_structure():
    p = ckpt_file("e2_tiny.pt", e2_payload())
    return invalid(CK.load_student_from_e2, p, needle="E2 state_dict is not structurally compatible")


@case("s2_e3_loader_refuses_e2_payload")
def _s2_e3_loader():
    p = ckpt_file("e2_for_e3.pt", e2_payload())
    return invalid(CK.load_e3_source_checkpoint, p, needle="is stage 'E2'; E6/E7 take the E3 checkpoint only")


# ================================================================== s3: resolve_parent's E2 branch
@case("s3_e2_parent_accepted")
def _s3_ok():
    p = resolve("e6-e2", FX["e2"])
    return (p["stage"] == "E2" and p["run_meta_file"] == "e2_run_meta.jsonl" and p["telemetry"] == "e2_telemetry.jsonl"
            and p["alpha_cwd"] is None and p["lambda_logit"] == 0.5 and p["checkpoint"] == E2_CKPT
            and p["checkpoint_sha256"] == FX["sha"]["e2"] and p["run_id"] == E2_RUN), str(p)[:300]


@case("s3_e7_e2_takes_the_e2_branch")
def _s3_e7e2():
    p = resolve("e7-e2", FX["e2"])
    return p["run_meta_file"] == "e2_run_meta.jsonl" and p["stage"] == "E2", str(p)[:200]


@case("s3_e3_layout_refused", "parent_records_missing")
def _s3_e3_layout():
    d = make_run("e3", E2_RUN, value=E2_VALUE, meta=e2_meta(), ckpt_name=E2_CKPT, payload=e2_payload())
    return refused("parent_records_missing", Q.resolve_parent, Q.resolve_quant_stage("e6-e2"), d,
                   sha_file(d / E2_CKPT), 42)


@case("s3_stage_e3_row_refused", "parent_stage_mismatch")
def _s3_stage():
    d = e2_run(meta=e2_meta(stage="E3"))
    return refused("parent_stage_mismatch", resolve, "e6-e2", d)


@case("s3_run_abort_refused", "parent_incomplete")
def _s3_abort():
    d = e2_run(tel=[{"event": "run_abort", "iter": 100}, end_row(E2_VALUE, best_ckpt=E2_CKPT)])
    return refused("parent_incomplete", resolve, "e6-e2", d)


@case("s3_no_run_end_refused", "parent_incomplete")
def _s3_no_end():
    d = e2_run(tel=[{"event": "val", "iter": 64000, "all_class_miou": E2_VALUE}])
    return refused("parent_incomplete", resolve, "e6-e2", d)


@case("s3_checks_failed_refused", "parent_incomplete")
def _s3_checks():
    d = e2_run(tel=[end_row(E2_VALUE, best_ckpt=E2_CKPT, checks_passed=False)])
    return refused("parent_incomplete", resolve, "e6-e2", d)


@case("s3_e6_e3_branch_unchanged")
def _s3_e6():
    p = resolve("e6", FX["e3"])
    want = {"stage": "E3", "run_meta_file": "e3_run_meta.jsonl", "telemetry": "e3_telemetry.jsonl",
            "alpha_cwd": 25.0, "lambda_logit": 0.5, "seed": 42, "mode": "real", "run_id": E3_RUN,
            "checkpoint": E3_CKPT, "checkpoint_sha256": FX["sha"]["e3"]}
    got = {k: p.get(k) for k in want}
    return got == want and sorted(p) == sorted(
        ["run_dir", "run_id", "best_json_sha256", "best_val_miou_all_class", "checkpoint", "checkpoint_path",
         "checkpoint_sha256", "telemetry", "telemetry_sha256", "run_meta_file", "run_meta_sha256", "seed", "mode",
         "stage", "lambda_logit", "alpha_cwd"]), str(got)


# ================================================================== s4: E6-E2's binding (AM-23 items 1, 4; AQ21)
def e2_bind(parent=None, seed: int = 42, lam=None, root=None):
    return Q.e2_parent_binding(parent if parent is not None else resolve("e6-e2", FX["e2"]), seed, lam or FX["lam"],
                               root or FX["root"])


def lam_variant(**over) -> tuple[dict, Path]:
    """A λ selection made from lam_doc with changes, committed in its own records repository."""
    doc = lam_doc(FX["e2"])
    for k, v in over.items():
        doc[k] = v
    data = bytes_of(doc)
    root = records_repo(files={LAM_REL: data})
    return sel(LAM_REL, data), root


@case("s4_binding_accepts")
def _s4_ok():
    b = e2_bind()
    entry = json.loads(FX["lam_bytes"])["inputs"][1]
    return (b["run_id"] == E2_RUN and b["selection_input"] == entry and b["checkpoint_sha256"] == FX["sha"]["e2"]
            and b["lambda_selection"] == {k: FX["lam"][k] for k in ("path", "sha256", "winner")}), str(b)[:300]


@case("s4_refuses_stage", "parent_stage_mismatch")
def _s4_stage():
    return refused("parent_stage_mismatch", e2_bind, {**resolve("e6-e2", FX["e2"]), "stage": "E3"})


@case("s4_refuses_seed_43", "am23_seed")
def _s4_seed():
    return refused("am23_seed", e2_bind, None, 43)


@case("s4_refuses_s43_directory", "am23_seed")
def _s4_s43_dir():
    return refused("am23_seed", e2_bind, resolve("e6-e2", FX["e2_s43"], seed=43), 43)


@case("s4_refuses_s43_parent_dict_at_seed_42", "am23_seed")
def _s4_parent_seed():
    return refused("am23_seed", e2_bind, {**resolve("e6-e2", FX["e2"]), "seed": 43}, 42)


@case("s4_refuses_160k_directory", "parent_run_id_mismatch")
def _s4_160k():
    return refused("parent_run_id_mismatch", e2_bind, resolve("e6-e2", FX["e2_160k"]))


@case("s4_refuses_lambda", "parent_lambda_mismatch")
def _s4_lambda():
    return refused("parent_lambda_mismatch", e2_bind, {**resolve("e6-e2", FX["e2"]), "lambda_logit": 1.0})


@case("s4_refuses_checkpoint_not_selected", "parent_checkpoint_not_selected")
def _s4_ckpt():
    return refused("parent_checkpoint_not_selected", e2_bind, {**resolve("e6-e2", FX["e2"]),
                                                               "checkpoint_sha256": "c" * 64})


@case("s4_refuses_selection_changed_since_read", "selection_changed_since_head")
def _s4_changed():
    return refused("selection_changed_since_head", e2_bind, None, 42, {**FX["lam"], "sha256": "d" * 64})


@case("s4_refuses_input_missing", "selection_input_missing")
def _s4_input_missing():
    lam, root = lam_variant(inputs=[json.loads(FX["lam_bytes"])["inputs"][0]])
    return refused("selection_input_missing", e2_bind, None, 42, lam, root)


@case("s4_refuses_input_twice", "selection_input_missing")
def _s4_input_twice():
    ins = json.loads(FX["lam_bytes"])["inputs"]
    lam, root = lam_variant(inputs=ins + [ins[1]])
    return refused("selection_input_missing", e2_bind, None, 42, lam, root)


@case("s4_refuses_best_json_not_input", "parent_not_selection_input")
def _s4_best():
    return refused("parent_not_selection_input", e2_bind, {**resolve("e6-e2", FX["e2"]), "best_json_sha256": "e" * 64})


@case("s4_refuses_run_meta_not_input", "parent_not_selection_input")
def _s4_meta():
    return refused("parent_not_selection_input", e2_bind, {**resolve("e6-e2", FX["e2"]), "run_meta_sha256": "e" * 64})


@case("s4_refuses_checkpoint_name_not_input", "parent_not_selection_input")
def _s4_ckpt_name():
    return refused("parent_not_selection_input", e2_bind, {**resolve("e6-e2", FX["e2"]),
                                                           "checkpoint": "e2_student_best_iter60000.pt"})


@case("s4_refuses_projection_pt", "parent_projection_present")
def _s4_projection():
    d = e2_run(extra={"projection.pt": b"synthetic"})
    lam_b = bytes_of(lam_doc(d))
    root = records_repo(files={LAM_REL: lam_b})
    return refused("parent_projection_present", Q.e2_parent_binding, resolve("e6-e2", d), 42, sel(LAM_REL, lam_b), root)


@case("s4_real_selection_names_the_winner_input")
def _s4_real():
    data = (REPO / LAM_REL).read_bytes()
    lam = sel(LAM_REL, data)
    entry = Q.lambda_selection_input(lam, "e2_s42_lambda0.5_80k_a1", REPO)
    return (lam["sha256"] == "719eaa0ae2379633786a545aff087d385b16a6c503cfa445ba456784f2592145"
            and lam["winner"]["run_id"] == "e2_s42_lambda0.5_80k_a1" and lam["winner"]["lambda"] == 0.5
            and lam["winner"]["ckpt_sha256"] == "73dbc32dfc988124a9b81df24abd97c526f927ac2b094996411d174e89d32f06"
            and entry == {"run_dir": "/runs/e2_s42_lambda0.5_80k_a1", "checkpoint": "e2_student_best_iter64000.pt",
                          "best_json_sha256": "860f2a8ed23faf25bc9ea03ac78a26588713cbb7a9b13ab0056a3854a1c629bb",
                          "run_meta_sha256": "d3ec29ccaebf85d4d5c753abaaa72178fbf7fa2e84a7ee33a3a8e6667f08fac0"}), \
        str(entry)


# ================================================================== s5: E6-X's binding (AM-23 items 3(b), 4; AQ15)
def record_variant(rec: dict, *, rel: str | None = None) -> Path:
    """A records repository whose E6-R1 record (at `rel`, the path of record by default) is `rec`, re-serialized
    with record_bytes, plus the R2 record of the main repository."""
    return records_repo(records={rel or COND_REL["R1"]: rec, COND_REL["R2"]: FX["rec"]["R2"]})


def bind_with(rec: dict, *, rel: str | None = None, parent=None):
    root = record_variant(rec, rel=rel)
    return lambda: e6x_bind(parent=parent, root=root, condition_record=rel or COND_REL["R1"],
                            condition_sha256=sha(X.record_bytes(rec)))


@case("s5_binding_accepts")
def _s5_ok():
    b = e6x_bind()
    c = b["condition"]
    return (b["arm"] == "E3-R1" and b["run_id"] == R_RUN["R1"] and c["path"] == COND_REL["R1"]
            and c["sha256"] == FX["rec_sha"]["R1"] and c["verdict"] == "yes" and c["dl"] == "DL-900"
            and c["value"] == repr(VALUE_YES)), str(b)[:300]


@case("s5_binding_accepts_r2")
def _s5_ok_r2():
    b = e6x_bind("R2")
    return b["arm"] == "E3-R2" and b["condition"]["dl"] == "DL-910", str(b)[:200]


@case("s5_refuses_seed_43", "am23_seed")
def _s5_seed():
    return refused("am23_seed", e6x_bind, seed=43)


@case("s5_refuses_e3_r2_run_for_e6_r1", "parent_arm_mismatch")
def _s5_r2_for_r1():
    return refused("parent_arm_mismatch", e6x_bind, parent=resolve("e6-r1", FX["r2"]))


@case("s5_refuses_alpha_winner_for_e6_r1", "parent_arm_mismatch")
def _s5_alpha_winner():
    return refused("parent_arm_mismatch", e6x_bind, parent=resolve("e6-r1", FX["e3"]))


def r1_with_meta(**over) -> dict:
    return resolve("e6-r1", e3x_run("R1", meta=e3_meta("E3-R1", **over)))


@case("s5_refuses_not_descriptive", "parent_not_arm_run")
def _s5_desc():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(descriptive=False))


@case("s5_refuses_parent_of_e4_e7", "parent_not_arm_run")
def _s5_poe():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(parent_of_e4_e7=True))


@case("s5_refuses_max_iters", "parent_not_arm_run")
def _s5_iters():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(max_iters=160000))


@case("s5_refuses_poly_horizon", "parent_not_arm_run")
def _s5_horizon():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(poly_horizon=160000))


@case("s5_refuses_float_horizon", "parent_not_arm_run")
def _s5_float_horizon():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(max_iters=80000.0))


@case("s5_refuses_no_target_construction", "parent_not_arm_run")
def _s5_tc():
    return refused("parent_not_arm_run", e6x_bind, parent=r1_with_meta(target_construction=None))


@case("s5_refuses_lambda", "parent_lambda_mismatch")
def _s5_lambda():
    return refused("parent_lambda_mismatch", e6x_bind, parent={**resolve("e6-r1", FX["r1"]), "lambda_logit": 1.0})


@case("s5_refuses_alpha", "parent_alpha_mismatch")
def _s5_alpha():
    return refused("parent_alpha_mismatch", e6x_bind, parent={**resolve("e6-r1", FX["r1"]), "alpha_cwd": 50.0})


@case("s5_refuses_no_condition_record", "condition_flags_missing")
def _s5_flag_record():
    return refused("condition_flags_missing", e6x_bind, condition_record=None)


@case("s5_refuses_no_condition_sha256", "condition_flags_missing")
def _s5_flag_sha():
    return refused("condition_flags_missing", e6x_bind, condition_sha256=None)


@case("s5_refuses_no_condition_dl", "condition_flags_missing")
def _s5_flag_dl():
    return refused("condition_flags_missing", e6x_bind, condition_dl=None)


@case("s5_condition_sha256_format", "condition_sha256_format")
def _s5_sha_form():
    return refused("condition_sha256_format", e6x_bind, condition_sha256="abc")


@case("s5_condition_not_repo_relative", "condition_path_not_repo_relative")
def _s5_abs():
    return refused("condition_path_not_repo_relative", e6x_bind,
                   condition_record=str(FX["root"] / COND_REL["R1"]))


@case("s5_condition_forbidden_path", "condition_forbidden_path")
def _s5_forbidden():
    return refused("condition_forbidden_path", e6x_bind, condition_record="reports/arms/latest_condition_R1.json")


@case("s5_condition_record_missing", "condition_missing_file")
def _s5_missing():
    return refused("condition_missing_file", e6x_bind, root=records_repo(records={}))


@case("s5_condition_untracked", "condition_not_tracked")
def _s5_untracked():
    root = records_repo(records={}, untracked={COND_REL["R1"]: X.record_bytes(FX["rec"]["R1"])})
    return refused("condition_not_tracked", e6x_bind, root=root)


@case("s5_condition_changed_since_head", "condition_changed_since_head")
def _s5_changed():
    root = records_repo(after={COND_REL["R1"]: X.record_bytes(FX["rec"]["R1"]) + b" "})
    return refused("condition_changed_since_head", e6x_bind, root=root)


@case("s5_condition_sha256_mismatch", "condition_sha256_mismatch")
def _s5_sha():
    return refused("condition_sha256_mismatch", e6x_bind, condition_sha256="f" * 64)


@case("s5_condition_format_passes_through", "condition_format")
def _s5_format():
    return refused("condition_format", bind_with(edited(FX["rec"]["R1"], seed=43)))


@case("s5_condition_threshold_passes_through", "condition_format")
def _s5_threshold():
    return refused("condition_format", bind_with(edited(FX["rec"]["R1"], **{"threshold.repr": "0.3"})))


@case("s5_void_yes_below_passes_through", "condition_void")
def _s5_void_yes():
    return refused("condition_void", bind_with(edited(FX["rec"]["R1no"], verdict="yes"), parent=resolve(
        "e6-r1", FX["r1_no"])))


@case("s5_void_no_above_passes_through", "condition_void")
def _s5_void_no():
    return refused("condition_void", bind_with(edited(FX["rec"]["R1"], verdict="no")))


@case("s5_void_equal_value_yes_passes_through", "condition_void")
def _s5_void_equal():
    rec = edited(FX["rec"]["R1"], **{"value.best_val_miou_all_class": float(T), "value.repr": T, "verdict": "yes"})
    return refused("condition_void", bind_with(rec))


@case("s5_condition_path_not_of_record", "condition_path_not_of_record")
def _s5_path():
    rel = "reports/arms/e6x_condition_R1_copy.json"
    return refused("condition_path_not_of_record", bind_with(FX["rec"]["R1"], rel=rel))


@case("s5_condition_arm_mismatch", "condition_arm_mismatch")
def _s5_arm():
    return refused("condition_arm_mismatch", e6x_bind, condition_record=COND_REL["R2"],
                   condition_sha256=FX["rec_sha"]["R2"])


def parent_mismatch(**fields) -> tuple[bool, str]:
    return refused("condition_parent_mismatch", bind_with(edited(FX["rec"]["R1"], **fields)))


@case("s5_condition_other_run_id", "condition_parent_mismatch")
def _s5_pm_run():
    return parent_mismatch(run_id="e3-r1_s42_lambda0.5_alpha25_80k_a9")


@case("s5_condition_other_best_json", "condition_parent_mismatch")
def _s5_pm_best():
    return parent_mismatch(**{"best_json.sha256": "1" * 64})


@case("s5_condition_other_checkpoint", "condition_parent_mismatch")
def _s5_pm_ckpt():
    return parent_mismatch(**{"checkpoint.sha256": "1" * 64})


@case("s5_condition_other_checkpoint_name", "condition_parent_mismatch")
def _s5_pm_ckpt_name():
    return parent_mismatch(**{"checkpoint.file": "e3_student_best_iter60000.pt"})


@case("s5_condition_other_run_meta", "condition_parent_mismatch")
def _s5_pm_meta():
    return parent_mismatch(**{"run_meta.sha256": "1" * 64})


@case("s5_condition_other_telemetry", "condition_parent_mismatch")
def _s5_pm_tel():
    return parent_mismatch(**{"telemetry.sha256": "1" * 64})


@case("s5_condition_other_value", "condition_parent_mismatch")
def _s5_pm_value():
    return parent_mismatch(**{"value.best_val_miou_all_class": 0.3801, "value.repr": "0.3801"})


@case("s5_condition_no", "condition_no")
def _s5_no():
    root = records_repo(records={COND_REL["R1"]: FX["rec"]["R1no"]})
    return refused("condition_no", e6x_bind, parent=resolve("e6-r1", FX["r1_no"]), root=root,
                   condition_sha256=FX["rec_sha"]["R1no"])


@case("s5_dl_row_missing", "condition_dl_row")
def _s5_dl_missing():
    return refused_msg("condition_dl_row", "DL-999 is not a row of", e6x_bind, condition_dl="DL-999")


@case("s5_dl_id_malformed", "condition_dl_row")
def _s5_dl_form():
    return refused_msg("condition_dl_row", "is not a decision-log id", e6x_bind, condition_dl="DL-x")


@case("s5_dl_log_untracked", "condition_dl_row")
def _s5_dl_untracked():
    return refused_msg("condition_dl_row", "is not tracked at HEAD", e6x_bind,
                       root=records_repo(dl=False, untracked={DL_REL: FX["dl_bytes"]}))


@case("s5_dl_row_not_recorded", "condition_dl_row")
def _s5_dl_status():
    return refused_msg("condition_dl_row", "has status 'DECIDED'", e6x_bind, condition_dl="DL-901")


@case("s5_dl_row_without_record_sha256", "condition_dl_row")
def _s5_dl_sha():
    return refused_msg("condition_dl_row", FX["rec_sha"]["R1"], e6x_bind, condition_dl="DL-902")


@case("s5_dl_row_without_yes", "condition_dl_row")
def _s5_dl_yes():
    return refused_msg("condition_dl_row", "E6-R1 condition (AM-23 item 3(b)): yes", e6x_bind, condition_dl="DL-903")


@case("s5_dl_row_without_value", "condition_dl_row")
def _s5_dl_value():
    return refused_msg("condition_dl_row", f"['{FX['rec']['R1']['value']['repr']}']", e6x_bind, condition_dl="DL-904")


@case("s5_dl_row_without_best_json_sha256", "condition_dl_row")
def _s5_dl_best():
    return refused_msg("condition_dl_row", FX["rec"]["R1"]["best_json"]["sha256"], e6x_bind, condition_dl="DL-905")


@case("s5_dl_row_without_checkpoint_sha256", "condition_dl_row")
def _s5_dl_ckpt():
    return refused_msg("condition_dl_row", FX["rec"]["R1"]["checkpoint"]["sha256"], e6x_bind, condition_dl="DL-906")


@case("s5_dl_row_of_another_record", "condition_dl_row")
def _s5_dl_other():
    return refused_msg("condition_dl_row", FX["rec_sha"]["R1"], e6x_bind, condition_dl="DL-910")


@case("s5_e6_binding_refuses_e3_r1", "parent_run_id_mismatch")
def _s5_e6_r1():
    return refused("parent_run_id_mismatch", Q.e6_parent_binding, resolve("e6", FX["r1"]), 42, FX["lam"], FX["alpha"])


@case("s5_e6_binding_accepts_alpha_winner")
def _s5_e6_ok():
    return code_of(Q.e6_parent_binding, resolve("e6", FX["e3"]), 42, FX["lam"], FX["alpha"]) is None, "e6 binding"


# ================================================================== s6: the launch gates and main's extra (AQ14)
def gate_argv(stage: str, **over) -> list[str]:
    """A passing launch command for `stage` (each keyword names a flag, underscores for dashes; None drops it)."""
    src = {"e6-e2": "e2", "e6-r1": "r1", "e6-r2": "r2", "e6": "e3"}[stage]
    a = {"--expect-head": HEAD, "--seed": "42", "--num-workers": "12", "--grad-clip-norm": str(CLIP),
         "--clip-selection": str(FX["clip"]), "--clip-selection-sha256": FX["clip_sha"],
         "--lambda-selection": LAM_REL, "--lambda-selection-sha256": FX["lam"]["sha256"],
         "--source-run-dir": str(FX[src]), "--expect-source-sha256": FX["sha"][src],
         "--out-dir": str(FX["work"] / f"qat_out_new_{stage}")}
    if stage != "e6-e2":
        a.update({"--alpha-selection": ALPHA_REL, "--alpha-selection-sha256": FX["alpha"]["sha256"]})
    if stage in ("e6-r1", "e6-r2"):
        arm = stage[-2:].upper()
        a.update({"--condition-record": COND_REL[arm], "--condition-sha256": FX["rec_sha"][arm],
                  "--condition-dl": "DL-900" if arm == "R1" else "DL-910"})
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        if v is None:
            a.pop(flag, None)
        else:
            a[flag] = str(v)
    return ["--real-run", "--confirm-real-run"] + [x for kv in a.items() for x in kv]


@contextlib.contextmanager
def launch_env():
    """CUDA, the TRAIN/VAL isolation and a clean code identity patched (smoke_qat_runner.py's gate patches)."""
    import src.data.isolation as iso
    with contextlib.ExitStack() as stack:
        stack.enter_context(patched(torch.cuda, "is_available", lambda: True))
        stack.enter_context(patched(torch.cuda, "is_initialized", lambda: False))
        stack.enter_context(patched(iso, "assert_trainval_only_root",
                                    lambda root, expected=None: {"counts": {"train": 5367, "val": 846}}))
        stack.enter_context(patched(Q, "code_identity", lambda: {"git_head": HEAD, "code_clean_at_head": True,
                                                                 "code_files_sha256": {}}))
        yield


def gates(stage: str, argv: list[str]):
    st = Q.resolve_quant_stage(stage)
    args = Q.build_parser(st).parse_args(argv)
    with launch_env():
        return Q.real_run_gates(args, st, repo_root=FX["root"])


@case("s6_e6_e2_gates_pass")
def _s6_e2():
    g = gates("e6-e2", gate_argv("e6-e2"))
    return (g["e6_binding"] is None and g["am23_binding"]["run_id"] == E2_RUN
            and g["am23_binding"]["stage"] == "E2"), str(g.get("am23_binding"))[:200]


@case("s6_e6_r1_gates_pass")
def _s6_r1():
    g = gates("e6-r1", gate_argv("e6-r1"))
    return g["e6_binding"] is None and g["am23_binding"]["arm"] == "E3-R1", str(g.get("am23_binding"))[:200]


@case("s6_e6_unchanged")
def _s6_e6():
    g = gates("e6", gate_argv("e6"))
    return g["am23_binding"] is None and g["e6_binding"]["alpha_selection"] == FX["alpha"], str(g["e6_binding"])[:200]


@case("s6_e6_e2_refuses_alpha_flags", "selection_not_applicable")
def _s6_alpha():
    return refused("selection_not_applicable", gates, "e6-e2",
                   gate_argv("e6-e2", alpha_selection=ALPHA_REL, alpha_selection_sha256=FX["alpha"]["sha256"]))


@case("s6_e6_e2_refuses_missing_lambda", "lambda_selection_missing")
def _s6_lambda():
    return refused("lambda_selection_missing", gates, "e6-e2",
                   gate_argv("e6-e2", lambda_selection=None, lambda_selection_sha256=None))


@case("s6_e6_e2_parser_refuses_condition_flags")
def _s6_parser():
    st = Q.resolve_quant_stage("e6-e2")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            Q.build_parser(st).parse_args(gate_argv("e6-e2") + ["--condition-dl", "DL-900"])
    except SystemExit as e:
        return e.code == 2, f"exit {e.code}"
    return False, "parsed"


@case("s6_e5_e6_parsers_take_no_condition_flags")
def _s6_e5e6():
    dests = {k: {a.dest for a in Q.build_parser(Q.resolve_quant_stage(k))._actions}
             for k in ("e5", "e6", "e6-e2", "e6-r1", "e6-r2")}
    flags = {"condition_record", "condition_sha256", "condition_dl"}
    ok = (not dests["e5"] & flags and not dests["e6"] & flags and not dests["e6-e2"] & flags
          and flags <= dests["e6-r1"] and flags <= dests["e6-r2"])
    return ok, str({k: sorted(v & flags) for k, v in dests.items()})


@case("s6_e6x_refuses_missing_condition", "condition_flags_missing")
def _s6_condition():
    return refused("condition_flags_missing", gates, "e6-r1",
                   gate_argv("e6-r1", condition_record=None, condition_sha256=None, condition_dl=None))


@case("s6_condition_forbidden_path", "condition_forbidden_path")
def _s6_condition_forbidden():
    return refused("condition_forbidden_path", gates, "e6-r1",
                   gate_argv("e6-r1", condition_record="reports/arms/latest_condition_R1.json"))


@case("s6_refuses_u4_pilot", "u4_pilot_not_e5_s42")
def _s6_pilot():
    return refused("u4_pilot_not_e5_s42", gates, "e6-e2",
                   gate_argv("e6-e2", clip_selection=None, clip_selection_sha256=None) + ["--u4-pilot"])


@case("s6_refuses_no_clip_selection", "clip_selection_required")
def _s6_clip():
    return refused("clip_selection_required", gates, "e6-e2",
                   gate_argv("e6-e2", clip_selection=None, clip_selection_sha256=None))


@case("s6_refuses_seed_43", "am23_seed")
def _s6_seed():
    return refused("am23_seed", gates, "e6-e2", gate_argv("e6-e2", seed=43, source_run_dir=FX["e2_s43"],
                                                         expect_source_sha256=FX["sha"]["e2_s43"]))


@case("s6_out_dir_forbidden_path", "out_dir")
def _s6_out():
    never = FX["work"] / "latest_out_e6_e2"
    ok, detail = refused("out_dir", gates, "e6-e2", gate_argv("e6-e2", out_dir=never))
    return ok and not never.exists(), detail


@case("s6_load_source_refuses_e2_projection_keys", "source_projection_keys")
def _s6_load_source():
    from src.models.student import build_student
    from src.quant import runner as R
    leaky = build_student(pretrained=False)
    leaky.register_buffer("cwd_projection_probe", torch.zeros(1))
    parent = resolve("e6-e2", FX["e2"])
    with patched(R, "check_source", lambda stage, path: (leaky, {}, {"sha256": parent["checkpoint_sha256"]})):
        return refused("source_projection_keys", Q.load_source, Q.resolve_quant_stage("e6-e2"), parent)


def main_extra(stage: str) -> dict:
    """The extra_meta main() hands run_qat, with load_source and run_qat replaced (nothing trains)."""
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"stage": kw["stage"], "seed": kw["seed"], "clip_norm": kw["clip_norm"], "epochs_completed": 15,
                "complete": True, "nonfinite_since_step": None}
    with contextlib.ExitStack() as stack:
        stack.enter_context(launch_env())
        stack.enter_context(patched(Q, "load_source",
                                    lambda st, parent: (None, {"sha256": parent["checkpoint_sha256"]})))
        stack.enter_context(patched(Q, "run_qat", fake_run))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        real = Q.real_run_gates
        stack.enter_context(patched(Q, "real_run_gates", lambda args, st, **kw: real(args, st, repo_root=FX["root"])))
        rc = Q.main(gate_argv(stage), stage)
    return {"rc": rc, **seen}


@case("s6_main_extra_e6_e2")
def _s6_main_e2():
    kw = main_extra("e6-e2")
    extra = kw.get("extra_meta") or {}
    return (kw["rc"] == 0 and extra.get("am23_parent", {}).get("run_id") == E2_RUN
            and extra.get("cwd_projection_loaded") is False and extra.get("e6_selections") is None), \
        str({k: extra.get(k) for k in ("cwd_projection_loaded", "e6_selections")})


@case("s6_main_extra_e6_r1")
def _s6_main_r1():
    kw = main_extra("e6-r1")
    extra = kw.get("extra_meta") or {}
    return (kw["rc"] == 0 and extra.get("am23_parent", {}).get("arm") == "E3-R1"
            and "cwd_projection_loaded" not in extra and extra.get("e6_selections") is None), str(sorted(extra))


@case("s6_main_extra_e6_unchanged")
def _s6_main_e6():
    kw = main_extra("e6")
    extra = kw.get("extra_meta") or {}
    return kw["rc"] == 0 and "am23_parent" not in extra and extra.get("e6_selections") is not None, str(sorted(extra))


# ================================================================== s7: check-run-meta's AM-23 profile
def launched_row(stage: str, parent: dict, extra: dict) -> dict:
    """A launch's run_meta row: run_qat's row of a 2-step smoke run, with the launch values of record over it."""
    from scripts.synthetic_qat_fixtures import task_loaders
    from src.models.student import build_student
    out = new_dir("profile") / "run"
    torch.manual_seed(0)
    Q.run_qat(stage=stage, mode="smoke", model=build_student(pretrained=False), source_meta={"sha256": parent[
        "checkpoint_sha256"], "path": None, "bytes": None}, out_dir=out, seed=42, clip_norm=CLIP,
        clip_source="clip_selection", device="cpu", num_workers=0, loaders=task_loaders(42), parent=parent,
        extra_meta=extra, max_steps=2, max_val_batches=1, log=lambda *a: None)
    row = json.loads((out / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines()[0])
    row.update({"mode": "real", "git_head": HEAD, "code_clean_at_head": True, "image_digest": Q.IMAGE_DIGEST_OF_RECORD,
                "num_workers": 12, "persistent_workers": True, "steps_per_epoch": 335, "total_steps": 5025,
                "bn_freeze_after_step": 3350, "obs_freeze_after_step": 4020, "val_batches_cap": None,
                "torch": "2.1.0+cu121", "cuda_initialized_at_seed": False, "batch_size": 16, "drop_last": True})
    row["scheduler"]["T_max"] = 5025
    return row


def profile_args(stage: str, **over):
    st = {"e6-e2": "E6-E2", "e6-r1": "E6-R1"}[stage]
    src = {"e6-e2": "e2", "e6-r1": "r1"}[stage]
    argv = ["check-run-meta", "--run-dir", "unused", "--stage", st, "--seed", "42", "--grad-clip-norm", str(CLIP),
            "--expect-head", HEAD, "--expect-source-sha256", FX["sha"][src], "--clip-selection", str(FX["clip"]),
            "--clip-selection-sha256", FX["clip_sha"], "--lambda-selection", LAM_REL,
            "--lambda-selection-sha256", FX["lam"]["sha256"]]
    if stage == "e6-r1":
        argv += ["--alpha-selection", ALPHA_REL, "--alpha-selection-sha256", FX["alpha"]["sha256"],
                 "--condition-record", COND_REL["R1"], "--condition-sha256", FX["rec_sha"]["R1"],
                 "--condition-dl", "DL-900"]
    args = QEE.build_parser().parse_args(argv)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def profile_row(stage: str) -> dict:
    key = f"row_{stage}"
    if key not in FX:
        src = {"e6-e2": "e2", "e6-r1": "r1"}[stage]
        parent = resolve(stage, FX[src])
        st = Q.resolve_quant_stage(stage)
        args = profile_args(stage)
        sel_t = ((FX["lam"],) if src == "e2" else (FX["lam"], FX["alpha"]))
        binding = Q.am23_parent_binding(st, parent, args, sel_t, repo_root=FX["root"])
        extra = {"clip_selection": {"path": str(FX["clip"]), "sha256": FX["clip_sha"], "winner": CLIP},
                 "u4_pilot": False, "e6_selections": None, "am23_parent": json.loads(json.dumps(binding))}
        if src == "e2":
            extra["cwd_projection_loaded"] = False
        FX[key] = launched_row(stage, parent, extra)
    return json.loads(json.dumps(FX[key]))


def profile(stage: str, row: dict, **over) -> dict:
    return QEE.profile_mismatches(row, profile_args(stage, **over), repo_root=FX["root"])


@case("s7_e6_e2_profile_passes")
def _s7_e2():
    bad = profile("e6-e2", profile_row("e6-e2"))
    return bad == {}, str(bad)[:300]


@case("s7_e6_r1_profile_passes")
def _s7_r1():
    bad = profile("e6-r1", profile_row("e6-r1"))
    return bad == {}, str(bad)[:300]


@case("s7_stops_on_stage")
def _s7_stage():
    row = profile_row("e6-e2")
    row["stage"] = "E6"
    return "stage" in profile("e6-e2", row), "stage"


@case("s7_stops_on_am23_parent_missing")
def _s7_missing():
    row = profile_row("e6-e2")
    del row["am23_parent"]
    return "am23_parent" in profile("e6-e2", row), "am23_parent"


@case("s7_stops_on_am23_parent_other")
def _s7_other():
    row = profile_row("e6-e2")
    row["am23_parent"]["checkpoint_sha256"] = "0" * 64
    return "am23_parent" in profile("e6-e2", row), "am23_parent"


@case("s7_stops_on_lambda_sha256")
def _s7_lambda():
    bad = profile("e6-e2", profile_row("e6-e2"), lambda_selection_sha256="0" * 64)
    return bad.get("am23_binding", ("",))[0] == "selection_sha256_mismatch", str(bad)[:200]


@case("s7_stops_on_condition_sha256")
def _s7_condition():
    bad = profile("e6-r1", profile_row("e6-r1"), condition_sha256="0" * 64)
    return bad.get("am23_binding", ("",))[0] == "condition_sha256_mismatch", str(bad)[:200]


@case("s7_stops_on_cwd_flag")
def _s7_cwd():
    row = profile_row("e6-e2")
    del row["cwd_projection_loaded"]
    return "cwd_projection_loaded" in profile("e6-e2", row), "cwd"


@case("s7_stops_on_e6_selections")
def _s7_e6sel():
    row = profile_row("e6-e2")
    row["e6_selections"] = {"lambda_selection": FX["lam"]}
    return "e6_selections" in profile("e6-e2", row), "e6_selections"


@case("s7_cli_accepts_the_am23_stages")
def _s7_cli():
    sub = QEE.build_parser()._subparsers._group_actions[0].choices["check-run-meta"]
    choices = next(a.choices for a in sub._actions if a.dest == "stage")
    return [s for s in choices if s.upper() in AM23] == ["E6-E2", "E6-R1", "E6-R2", "e6-e2", "e6-r1", "e6-r2"], \
        str(choices)


# ================================================================== s8: Block B's rules files (AQ24)
@case("s8_rules_path_for_stage")
def _s8_paths():
    got = {s: S.rules_path_for_stage(s).name for s in ("E5", "E6", "e6", "E6-E2", "e6-r1", "E6-R2", "E7", None)}
    want = {"E5": "qat_selection_rules.json", "E6": "qat_selection_rules.json", "e6": "qat_selection_rules.json",
            "E6-E2": "qat_selection_rules_am23.json", "e6-r1": "qat_selection_rules_am23.json",
            "E6-R2": "qat_selection_rules_am23.json", "E7": "qat_selection_rules.json",
            None: "qat_selection_rules.json"}
    return got == want, str(got)


@case("s8_rules_path_for_run")
def _s8_run():
    d = new_dir("rules_run")
    write(d / Q.TELEMETRY_NAME, json.dumps({"event": "run_meta", "stage": "E6-E2"}) + "\n")
    e6 = new_dir("rules_run")
    write(e6 / Q.TELEMETRY_NAME, json.dumps({"event": "run_meta", "stage": "E6"}) + "\n")
    got = (S.rules_path_for_run(d).name, S.rules_path_for_run(e6).name,
           S.rules_path_for_run(FX["work"] / "absent_run").name,
           S.rules_path_for_run(FX["work"] / "latest_run").name, S.rules_path_for_run(None).name)
    return got == ("qat_selection_rules_am23.json",) + ("qat_selection_rules.json",) * 4, str(got)


@case("s8_q2_rules_unchanged")
def _s8_q2():
    return sha_file(S.RULES_PATH) == Q2_RULES_SHA256 and S.load_rules()[1] == Q2_RULES_SHA256, sha_file(S.RULES_PATH)


@case("s8_am23_rules_load")
def _s8_load():
    p = S.rules_path_for_stage("E6-E2")
    rules, rsha = S.load_rules(p)
    q2 = json.loads(S.RULES_PATH.read_text(encoding="utf-8"))["qat_epoch"]
    same = {k: v for k, v in rules["qat_epoch"].items() if k not in ("stages", "source")} == \
        {k: v for k, v in q2.items() if k not in ("stages", "source")}
    return (rules["qat_epoch"]["stages"] == ["E6-E2", "E6-R1", "E6-R2"] and "qat_clip" not in rules and same
            and rsha == sha_file(p) and S.rules_rel_of(rules) == "configs/qat_selection_rules_am23.json"
            and S.rules_rel_of(S.load_rules()[0]) == "configs/qat_selection_rules.json"), rsha


def rules_root(doc: dict) -> Path:
    """A folder standing in for the repository, holding an AM-23 rules file (S.REPO is pointed at it)."""
    root = new_dir("rules_root")
    write(root / "configs" / "qat_selection_rules_am23.json", json.dumps(doc, indent=2))
    write(root / "configs" / "qat_selection_rules.json", S.RULES_PATH.read_bytes())
    return root


@case("s8_am23_rules_altered_stages_refused", "rules_mismatch")
def _s8_altered():
    doc = json.loads(S.rules_path_for_stage("E6-E2").read_text(encoding="utf-8"))
    doc["qat_epoch"]["stages"] = ["E6-E2", "E6-R1", "E6-R2", "E6"]
    root = rules_root(doc)
    with patched(S, "REPO", root):
        return refused("rules_mismatch", S.load_rules, root / "configs" / "qat_selection_rules_am23.json")


@case("s8_am23_rules_with_qat_clip_refused", "rules_mismatch")
def _s8_clip():
    doc = json.loads(S.rules_path_for_stage("E6-E2").read_text(encoding="utf-8"))
    doc["qat_clip"] = json.loads(S.RULES_PATH.read_text(encoding="utf-8"))["qat_clip"]
    root = rules_root(doc)
    with patched(S, "REPO", root):
        return refused("rules_mismatch", S.load_rules, root / "configs" / "qat_selection_rules_am23.json")


@case("s8_unlisted_file_checked_as_q2s", "rules_mismatch")
def _s8_unlisted():
    doc = json.loads(S.rules_path_for_stage("E6-E2").read_text(encoding="utf-8"))
    p = write(new_dir("rules_other") / "rules.json", json.dumps(doc))
    return refused("rules_mismatch", S.load_rules, p)


# ================================================================== s9: conversion, selection and finalize (AQ12)
SCORE_DRIVER = r'''"""Smoke child: qat_epoch_eval.py score with the evaluator stand-in (its environment names the
synthetic tree)."""
import sys
sys.dont_write_bytecode = True
repo, driver = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import scripts.qat_epoch_eval as QEE
QEE.EVALUATOR_SCRIPT = driver
raise SystemExit(QEE.main(sys.argv[3:]))
'''

RUN_DRIVER = r'''"""Smoke child: one QAT run in smoke mode from fixed weights, with the code of <root> (sys.argv[1])."""
import json
import sys
sys.dont_write_bytecode = True
root, out, stage, weights, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
sys.path.insert(0, root)
import torch
from scripts.synthetic_qat_fixtures import make_qat_run
from src.models.student import build_student
model = build_student(pretrained=False)
model.load_state_dict(torch.load(weights, map_location="cpu"))
summary = make_qat_run(out, model=model.eval(), stage=stage, seed=42, clip=1.0,
                       extra_meta=json.loads(extra) or None)
print("RUN-DRIVER", json.dumps({"complete": summary["complete"], "stage": summary["stage"]}))
'''


def child(argv: list[str], *, env=None, cwd=None, timeout: int = 2400) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-B", *argv], cwd=str(cwd or FX["work"] / "relative_base"),
                       capture_output=True, text=True, timeout=timeout, env=env)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("RESULT:", "RUN-DRIVER"))]
    return p.returncode, (lines[-1] if lines else (p.stdout + p.stderr).strip()[-400:])


def weights(name: str, seed: int) -> Path:
    from src.models.student import build_student
    p = FX["work"] / "weights" / f"{name}.pt"
    if not p.is_file():
        p.parent.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        torch.save(build_student(pretrained=False).state_dict(), p)
    return p


def qat_run(tag: str, stage: str, *, root: Path = REPO, w: str = "w0", extra=None) -> tuple[Path, int, str]:
    """A 15-epoch smoke-mode QAT run in a fresh process with the code of `root`; its run_id is `run_<stage>`."""
    out = FX["work"] / "qat_runs" / tag / f"run_{stage}"
    if out.exists():
        return out, 0, "cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    drv = write(FX["work"] / "drivers" / "run_driver.py", RUN_DRIVER)
    rc, res = child([str(drv), str(root), str(out), stage, str(weights(w, {"w0": 0, "w1": 1}[w])),
                     json.dumps(extra or {})])
    return out, rc, res


def e6_e2_run() -> Path:
    out, rc, res = qat_run("tip", "e6-e2", extra={"cwd_projection_loaded": False})
    if rc != 0:
        raise RuntimeError(f"the E6-E2 run failed: {res}")
    return out


def convert_record(run: Path, ev: Path) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = QEE.main(["convert", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose", "record",
                       "--host-label", HOST, "--allow-smoke-inputs",
                       "--expect-telemetry-sha256", Q.sha256_file(run / Q.TELEMETRY_NAME)])
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("RESULT:")]
    return rc, (lines[-1] if lines else buf.getvalue()[-300:])


def finalized() -> dict:
    """AQ12: the E6-E2 run converted (record), scored (the evaluator stand-in, one VAL row), selected under the AM-23
    rules (select_qat_epoch.py) and finalized (qat_epoch_eval.py finalize), each CLI in a fresh process."""
    if "final" in FX:
        return FX["final"]
    from scripts.synthetic_qat_fixtures import evaluator_driver, make_tree
    run, ev = e6_e2_run(), FX["work"] / "eval_e6_e2"
    out = {"run": run, "ev": ev}
    out["convert"] = convert_record(run, ev)
    data = FX["work"] / "data"
    if not data.is_dir():
        make_tree(data, n_train=16, n_val=846, seed=3)              # no TEST split
    drv = evaluator_driver(FX["work"] / "drivers" / "evaluator_driver.py")
    sdrv = write(FX["work"] / "drivers" / "score_driver.py", SCORE_DRIVER)
    env = {**os.environ, "PLANTSEG_DATA_ROOT": str(data), "SMOKE_REPO": str(REPO), "SMOKE_MAX_SAMPLES": "1",
           "SMOKE_GIT_HEAD": Q.code_identity()["git_head"] or "0" * 40,
           "PLANTSEG_IMAGE_DIGEST": "sha256:" + "6" * 64}
    out["score"] = child([str(sdrv), str(REPO), str(drv), "score", "--run-dir", str(run), "--eval-dir", str(ev),
                          "--purpose", "record", "--host-label", HOST, "--allow-smoke-inputs"], env=env)
    tel_sha = Q.sha256_file(run / Q.TELEMETRY_NAME)
    out["select"] = child([str(REPO / "scripts" / "select_qat_epoch.py"), "--run-dir", str(run), "--eval-dir", str(ev),
                           "--expect-telemetry-sha256", tel_sha, "--allow-smoke-inputs"], cwd=REPO)
    out["sel"] = json.loads((ev / QEE.SELECTION_RECORD).read_text()) if (ev / QEE.SELECTION_RECORD).is_file() else {}
    out["finalize"] = child([str(REPO / "scripts" / "qat_epoch_eval.py"), "finalize", "--run-dir", str(run),
                             "--eval-dir", str(ev), "--allow-smoke-inputs"], cwd=REPO)
    out["fin"] = json.loads((ev / QEE.FINALIZE_RECORD).read_text()) if (ev / QEE.FINALIZE_RECORD).is_file() else {}
    FX["final"] = out
    return out


@case("s9_convert_e6_e2_record")
def _s9_convert():
    f = finalized()
    p = f["ev"] / A.CONVERTED_DIR / A.epoch_names(15)["run_meta"]
    prov = json.loads(p.read_text()) if p.is_file() else {}
    return (f["convert"] == (0, "RESULT: CONVERTED 15/15 (record)") and prov.get("stage") == "E6-E2"
            and prov.get("source_stage") == "E2" and prov.get("cwd_projection_loaded") is False), \
        f"{f['convert']} {prov.get('cwd_projection_loaded')!r}"


@case("s9_run_provenance_cwd_flag")
def _s9_prov_flag():
    from src.quant import runner as R

    def prov(key):
        return R.build_run_provenance(stage=Q.resolve_quant_stage(key), source_meta={"path": "x", "sha256": "0" * 64,
                                      "bytes": 1}, method="qat", qconfig_summary={}, calibration=None, training=None,
                                      converted_path=None, coverage=None, forward=None, backend="qnnpack")
    got = {k: prov(k).get("cwd_projection_loaded", "absent") for k in ("e6-e2", "e7-e2", "e6-r1", "e6", "e4", "e5")}
    want = {"e6-e2": False, "e7-e2": False, "e6-r1": False, "e6": False, "e4": "absent", "e5": "absent"}
    return got == want, str(got)


@case("s9_select_in_process_takes_am23_rules")
def _s9_select_in_process():
    import scripts.select_qat_epoch as SE
    f = finalized()
    out = new_dir("reselect") / "qat_selection.json"           # the eval dir's own selection stays as written
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = SE.main(["--run-dir", str(f["run"]), "--eval-dir", str(f["ev"]), "--expect-telemetry-sha256",
                      Q.sha256_file(f["run"] / Q.TELEMETRY_NAME), "--allow-smoke-inputs", "--out", str(out)])
    doc = json.loads(out.read_text()) if out.is_file() else {}
    return rc == 0 and doc.get("rules_sha256") == sha_file(S.rules_path_for_stage("E6-E2")), buf.getvalue()[-200:]


@case("s9_validate_int8_accepts_e6_e2")
def _s9_validate():
    p = finalized()["ev"] / A.CONVERTED_DIR / A.epoch_names(15)["run_meta"]
    return SA.validate_int8_artifact("E6-E2", p)["provenance"]["stage"] == "E6-E2", str(p.name)


@case("s9_e6_refuses_an_e6_e2_provenance", "stage_mismatch")
def _s9_e6():
    p = finalized()["ev"] / A.CONVERTED_DIR / A.epoch_names(15)["run_meta"]
    try:
        SA.validate_int8_artifact("E6", p)
    except SA.StageArtifactError as e:
        return e.code == "stage_mismatch", e.code
    return False, "accepted"


@case("s9_e6_e2_provenance_without_cwd_flag_refused", "projection_flag_missing")
def _s9_cwd():
    src = finalized()["ev"] / A.CONVERTED_DIR
    d = new_dir("prov_nocwd")
    names = A.epoch_names(15)
    prov = json.loads((src / names["run_meta"]).read_text())
    del prov["cwd_projection_loaded"]
    prov["converted_artifact"] = str(src / names["torchscript"])
    write(d / names["run_meta"], json.dumps(prov))
    try:
        SA.validate_int8_artifact("E6-E2", d / names["run_meta"])
    except SA.StageArtifactError as e:
        return e.code == "projection_flag_missing", e.code
    return False, "accepted"


@case("s9_convert_epoch_e6_r1")
def _s9_r1():
    run, rc, res = qat_run("tip", "e6-r1")
    ev = FX["work"] / "eval_e6_r1"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc2 = QEE.main(["convert", "--run-dir", str(run), "--eval-dir", str(ev), "--purpose", "timing", "--host-label",
                        HOST, "--allow-smoke-inputs", "--epochs", "15"])
    p = ev / A.CONVERTED_DIR / A.epoch_names(15)["run_meta"]
    prov = json.loads(p.read_text()) if p.is_file() else {}
    ok = rc == 0 and rc2 == 0 and prov.get("stage") == "E6-R1" and prov.get("source_stage") == "E3" \
        and prov.get("cwd_projection_loaded") is False
    return ok and SA.validate_int8_artifact("E6-R1", p)["provenance"]["stage"] == "E6-R1", f"{res} rc={rc2}"


@case("s9_e6_e2_selected_under_am23_rules")
def _s9_selected():
    f = finalized()
    am23 = S.rules_path_for_stage("E6-E2")
    rc, res = f["select"]
    trace = f["sel"].get("rule_trace", [""])[0]
    return (rc == 0 and res.startswith("RESULT: SELECTED epoch ") and "(E6-E2, seed 42, clip 1.0" in res
            and f["sel"].get("rules_sha256") == sha_file(am23)
            and trace.startswith("rule configs/qat_selection_rules_am23.json qat_epoch")), f"{res} | {trace[:80]}"


@case("finalize_e6_e2_real")
def _s9_finalize():
    f = finalized()
    rc, res = f["finalize"]
    w = f["sel"].get("winner") or {}
    x86 = f["fin"].get("x86_latency_copy") or {}
    return (rc == 0 and res == f"RESULT: FINALIZED (E6-E2, seed 42, epoch {w.get('epoch', 0):02d})"
            and (x86.get("provenance") or {}).get("stage") == "E6-E2" and x86.get("usable_for_accuracy") is False
            and f["fin"].get("stage") == "E6-E2"), f"{res} {(x86.get('provenance') or {}).get('stage')!r}"


@case("s8_finalize_refuses_a_q2_rules_selection", "epoch_selection_format")
def _s8_finalize_q2():
    f = finalized()
    ev = FX["work"] / "eval_e6_e2_q2sha"
    import shutil
    shutil.copytree(f["ev"], ev, ignore=shutil.ignore_patterns(QEE.FINALIZE_RECORD, QEE.FINAL_DIR))
    sel_doc = json.loads((ev / QEE.SELECTION_RECORD).read_text())
    sel_doc["rules_sha256"] = Q2_RULES_SHA256
    (ev / QEE.SELECTION_RECORD).write_bytes(A.json_bytes(sel_doc))
    rc, res = child([str(REPO / "scripts" / "qat_epoch_eval.py"), "finalize", "--run-dir", str(f["run"]),
                     "--eval-dir", str(ev), "--allow-smoke-inputs"], cwd=REPO)
    return rc == 2 and "[epoch_selection_format]" in res, res


@case("s8_q2_rules_refuse_an_e6_e2_run", "stage")
def _s8_q2_refuses():
    f = finalized()
    rules, rsha = S.load_rules()
    return refused("stage", S.epoch_selection, f["run"], f["ev"], expect_telemetry_sha256=Q.sha256_file(
        f["run"] / Q.TELEMETRY_NAME), rules=rules, rules_sha256=rsha, allow_smoke_inputs=True)


@case("s8_am23_rules_refuse_an_e6_run", "stage")
def _s8_am23_refuses():
    run, rc, res = qat_run("tip", "e6")
    rules, rsha = S.load_rules(S.rules_path_for_stage("E6-E2"))
    return refused("stage", S.epoch_selection, run, FX["work"] / "eval_unused", expect_telemetry_sha256=Q.sha256_file(
        run / Q.TELEMETRY_NAME), rules=rules, rules_sha256=rsha, allow_smoke_inputs=True)


@case("x86_lists_unchanged")
def _x86_lists():
    return X86.X86_QAT_STAGES == ("E5", "E6") and X86.X86_PTQ_STAGES == ("E4", "E7"), \
        f"{X86.X86_QAT_STAGES} {X86.X86_PTQ_STAGES}"


# ================================================================== s10: the evaluation tables and refusals
@case("s10_stages_appended")
def _s10_stages():
    return ART.STAGES == ARTIFACT_STAGES_AT_O + AM23, str(ART.STAGES)


@case("s10_stage_artifact_rows")
def _s10_rows():
    want = {"E6-E2": ("int8_artifact", "int8_qat", "E2", "qat"), "E7-E2": ("int8_artifact", "int8_ptq", "E2", "ptq"),
            "E6-R1": ("int8_artifact", "int8_qat", "E3", "qat"), "E6-R2": ("int8_artifact", "int8_qat", "E3", "qat")}
    got = {s: tuple(SA.STAGE_ARTIFACTS[s][k] for k in ("kind", "precision", "source_stage", "method")) for s in want}
    return got == want, str(got)


@case("s10_projection_free_tuple")
def _s10_pf():
    return SA.PROJECTION_FREE_STAGES == ("E6", "E7") + AM23, str(SA.PROJECTION_FREE_STAGES)


@case("s10_descriptive_only_tuple")
def _s10_desc():
    return (SA.DESCRIPTIVE_ONLY_STAGES == ("TEACHER", "A", "F", "G") + AM23
            and all(SA.is_descriptive_only(s) for s in AM23)
            and not any(SA.is_descriptive_only(s) for s in ("E4", "E5", "E6", "E7"))), str(SA.DESCRIPTIVE_ONLY_STAGES)


@case("s10_em_choices_are_the_stages")
def _s10_choices():
    choices = next(a.choices for a in EM.build_parser()._actions if a.dest == "stage")
    return list(choices) == list(ART.STAGES) == list(ARTIFACT_STAGES_AT_O + AM23), str(choices)


@case("s10_em_parses_each")
def _s10_parse():
    got = [EM.build_parser().parse_args(["--stage", s, "--out-dir", str(FX["work"] / "em_out")]).stage for s in AM23]
    return got == list(AM23), str(got)


@case("s10_p34_run_source_unchanged")
def _s10_p34():
    return sha(inspect.getsource(EM.run).encode("utf-8")) == EM_RUN_SHA256, "P34"


@case("s10_upstream_protocol_accepted")
def _s10_upstream():
    errs = []
    for s, prec in (("E6-E2", "int8_qat"), ("E7-E2", "int8_ptq"), ("E6-R1", "int8_qat"), ("E6-R2", "int8_qat")):
        # validate_cli_args is pure: the provenance path is named, never opened
        args = EM.build_parser().parse_args(["--stage", s, "--precision", prec, "--protocol", "upstream",
                                             "--batch-size", "1", "--device", "cpu",
                                             "--provenance", str(FX["work"] / "absent_run_meta.json"),
                                             "--out-dir", str(FX["work"] / "em_out")])
        try:
            EM.validate_cli_args(args)
        except Exception as e:                              # noqa: BLE001
            errs.append(f"{s}: {e}")
    return not errs, str(errs)[:300]


def stage_choices(module: str) -> list:
    """The --stage choices of a script's build_parser (argparse refuses any other stage, exit 2)."""
    mod = importlib.import_module(module)
    return list(next(a for a in mod.build_parser()._actions if a.dest == "stage").choices)


@case("s10_corruptions_refuses_each")
def _s10_corruptions():
    # Read by ast, not imported: the script imports the vendored imagecorruptions, which needs scikit-image, and
    # this venv has none (B-2 STEP 3). Its --stage choices are list(STAGES), so argparse refuses the four.
    tree = ast.parse((REPO / "scripts" / "evaluate_corruptions.py").read_text(encoding="utf-8"))
    stages = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and [getattr(t, "id", None) for t in n.targets] == ["STAGES"])
    choices = [ast.unparse(k.value) for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", None) == "add_argument" and n.args
               and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--stage"
               for k in n.keywords if k.arg == "choices"]
    return (stages == CORRUPTION_STAGES_AT_O and choices == ["list(STAGES)"]
            and not set(AM23) & set(stages)), f"{stages} {choices}"


@case("s10_profile_efficiency_refuses_each")
def _s10_profile():
    choices = stage_choices("scripts.profile_efficiency")
    return choices == list(EFF.STAGES) and not set(AM23) & set(choices), str(choices)


@case("s10_backend_parity_refuses_each")
def _s10_parity_cli():
    choices = stage_choices("scripts.evaluate_backend_parity")
    return choices == list(EFF.INT8_STAGES) and not set(AM23) & set(choices), str(choices)


def eff_code(fn, **kw) -> str | None:
    try:
        fn(**kw)
    except Exception as e:                                  # noqa: BLE001
        return getattr(e, "code", type(e).__name__)
    return None


@case("s10_efficiency_provenance_refuses_each")
def _s10_prov():
    got = [eff_code(EFF.provenance, stage=s, model_role="student", precision="int8_qat", artifact_role="x",
                    backend="qnnpack") for s in AM23]
    return got == ["unknown_stage"] * 4, str(got)


@case("s10_parity_request_refuses_each")
def _s10_parity():
    got = [eff_code(EFF.validate_parity_request, stage=s, qnnpack_artifact={"stage": s}, x86_artifact={"stage": s},
                    manifest_sha256="0" * 64, expected_rows=846) for s in AM23]
    return got == ["parity_not_int8_stage"] * 4, str(got)


def x86_gate_case(stage: str):
    def fn():
        got = eff_code(EFF.x86_latency_copy_gate, stage=stage)
        return got == "x86_latency_copy_not_int8_stage", str(got)
    return fn


for _s in AM23:
    case(f"s10_x86_gate_refuses_{_s.lower().replace('-', '_')}")(x86_gate_case(_s))


@case("s10_int8_official_error_none_for_qat")
def _s10_official():
    p = finalized()["ev"] / A.CONVERTED_DIR / A.epoch_names(15)["run_meta"]
    resolved = SA.validate_int8_artifact("E6-E2", p)
    return SA.int8_official_error(resolved) is None, "QAT passes the record gate"


@case("s10_robustness_pair_unchanged")
def _s10_robustness():
    # Read by ast, not imported: src.eval.robustness imports src.stats, whose package loads a module this smoke
    # never opens.
    tree = ast.parse((REPO / "src" / "eval" / "robustness.py").read_text(encoding="utf-8"))
    pair = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and [getattr(t, "id", None) for t in n.targets] == ["INFERENTIAL_ROBUSTNESS_PAIR"])
    return pair == ("E1", "E6"), str(pair)


# ================================================================== s11: training invariance (AQ16)
DROP_ALWAYS = ("wall_clock", "iter_seconds", "val_seconds")
DROP_META = ("git_head", "code_clean_at_head", "code_files_sha256", "out_dir", "run_id", "host", "source_checkpoint")
DROP_STAGE = ("stage", "source_stage", "cwd_projection_loaded", "checkpoint_sha256", "checkpoint_bytes")


def rows_of(run: Path, drop_stage: bool) -> list:
    rows = [json.loads(ln) for ln in (run / Q.TELEMETRY_NAME).read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in rows:
        drop = set(DROP_ALWAYS) | (set(DROP_META) if r.get("event") == "run_meta" else set())
        drop |= set(DROP_STAGE) if drop_stage else set()
        out.append({k: v for k, v in r.items() if k not in drop})
    return out


def payloads_of(run: Path, drop_stage: bool) -> list:
    out = []
    for e in range(1, Q.EPOCHS + 1):
        p = torch.load(run / Q.EPOCH_DIR / f"e{e:02d}.pt", map_location="cpu", weights_only=True)
        out.append({k: v for k, v in p.items() if k != "model_state_dict"     # its digest is in the rows
                    and not (drop_stage and k in ("stage", "source_stage", "run_id"))})
    return out


def export_o() -> Path:
    """O's export in the archive_old form (scripts/smoke_arms_invariance.py :382-393; AQ16): `git archive` with the
    SL-1 exclude and E-5's suffix in the pathspec; a member whose name still contains "test" refuses the export
    before anything is extracted."""
    dest = FX["work"] / "export_o"
    if dest.is_dir():
        return dest
    p = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", O_COMMIT, "--", "src", "configs", "scripts",
                        "reports/e1_class_weights.json", ":(exclude,icase)*test*",
                        ":(exclude)docs/reference/reference.pdf"], capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        bad = [n for n in tf.getnames() if "test" in n.lower()]
        if bad:
            raise RuntimeError(f"O's archive kept {len(bad)} member(s) whose name contains 'test'")
        dest.mkdir()
        tf.extractall(dest, filter="data")
    return dest


def same_rows(a: Path, b: Path, drop_stage: bool) -> tuple[bool, str]:
    ra, rb = rows_of(a, drop_stage), rows_of(b, drop_stage)
    diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
    pa, pb = payloads_of(a, drop_stage), payloads_of(b, drop_stage)
    pdiff = [i + 1 for i, (x, y) in enumerate(zip(pa, pb)) if x != y]
    return (len(ra) == len(rb) and not diff and not pdiff), f"rows {len(ra)}/{len(rb)} differ at {diff[:3]}; " \
                                                             f"payloads differ at {pdiff[:3]}"


def run_ok(tag: str, stage: str, **kw) -> Path:
    out, rc, res = qat_run(tag, stage, **kw)
    if rc != 0:
        raise RuntimeError(f"{tag} {stage}: {res}")
    return out


@case("s11_cross_commit_e6_identical")
def _s11_cross():
    old = run_ok("o", "e6", root=export_o())
    new = run_ok("tip", "e6")
    ok, detail = same_rows(old, new, False)
    mo = json.loads((old / Q.TELEMETRY_NAME).read_text().splitlines()[0])
    return ok and mo.get("stage") == "E6", detail


@case("s11_e6_e2_equals_e6")
def _s11_e2():
    ok, detail = same_rows(run_ok("tip", "e6-e2", extra={"cwd_projection_loaded": False}), run_ok("tip", "e6"), True)
    return ok, detail


@case("s11_e6_r1_equals_e6")
def _s11_r1():
    return same_rows(run_ok("tip", "e6-r1"), run_ok("tip", "e6"), True)


@case("s11_other_weights_differ")
def _s11_other():
    a, b = run_ok("tip_w1", "e6", w="w1"), run_ok("tip", "e6")
    sa_ = [r.get("state_sha256") for r in rows_of(a, False) if r.get("event") == "epoch_end"]
    sb_ = [r.get("state_sha256") for r in rows_of(b, False) if r.get("event") == "epoch_end"]
    return len(sa_) == len(sb_) == Q.EPOCHS and all(x != y for x, y in zip(sa_, sb_)), f"{sa_[:1]} {sb_[:1]}"


@case("s11_run_qat_source_unchanged")
def _s11_source():
    return sha(inspect.getsource(Q.run_qat).encode("utf-8")) == RUN_QAT_SHA256_AT_O, "run_qat"


@case("s11_qat_code_paths_appended")
def _s11_paths():
    return Q.QAT_CODE_PATHS == QAT_CODE_PATHS_AT_O + AM23_CODE_PATHS, str(Q.QAT_CODE_PATHS[-6:])


@case("s11_no_module_level_condition_import")
def _s11_lazy():
    src_q = (REPO / "src" / "quant" / "qat.py").read_text(encoding="utf-8")
    src_e = (REPO / "scripts" / "qat_epoch_eval.py").read_text(encoding="utf-8")
    top = [ln for ln in (src_q + src_e).splitlines() if ln.startswith(("import ", "from ")) and "e6x_condition" in ln]
    return not top, str(top)


# ================================================================== hygiene
@case("h_every_new_code_has_a_case")
def _h_codes():
    codes = {c for _, c in CASES.values() if c}
    missing = [c for c in NEW_CODES if c not in codes]
    return not missing, str(missing)


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
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- a case that raises (or an argparse exit) fails here
        ok, detail = False, f"{type(e).__name__}: {e}"
    return bool(ok), str(detail)


def choose_work_dir(value: str | None) -> Path:
    from src.quant import runner as R
    if value is not None:
        if Q.names_test(value):
            raise SystemExit(f"--work-dir {value!r} contains 'test'")
        work = R.check_output_dir(value)
        if work.exists():
            raise SystemExit(f"--work-dir {work} exists; a work dir is never reused")
        return work
    n = 0
    while True:
        n += 1
        work = R.check_output_dir(str(Path(tempfile.gettempdir()) / f"am23_quant_smoke_{os.getpid()}_{n}"))
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
    os.chdir(work / "relative_base")             # a relative path resolves here, never in the repository
    AUDIT_BASES[:] = [str(work), str(REPO)]
    sys.addaudithook(audit)
    t0 = time.time()
    print("=" * 78)
    print("AM-23 QUANT SMOKE (L-AM23Q B-2) -- CPU, synthetic run directories and records repositories")
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
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}", flush=True)
    print("\n[CHECKS]")
    for name, ok, detail in RESULTS:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}  {detail[:400] if not ok else ''}".rstrip())
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n({round(time.time() - t0, 1)} s)")
    print(f"RESULT: {'PASS' if RESULTS and passed == len(RESULTS) else 'FAIL'} ({passed}/{len(RESULTS)})")
    return 0 if RESULTS and passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
