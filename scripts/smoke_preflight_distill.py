#!/usr/bin/env python3
"""Smoke for scripts/preflight_distill.py, the KD launch gate (lane 4(b), K2; plan P5 and P8; PL-25 to PL-32;
PL-39 cases 30 to 36; CHECK ITEMS 3 and 5; AM-19a readings 10, 19 and 21).

CPU only. Synthetic inputs: scratch git repositories built with the selection smoke's AM-19 scenarios
(scripts/smoke_select_sweeps.py, imported as a library; PL-22), scratch folders with deterministic names (SL-1),
stubbed child processes. Nothing is written in the repository; no GPU, no network, no teacher file, no data root.
The gate's stages run in-process on a context like the one the gate builds.

  A  values of record and the reused E1 stages (PL-25, PL-26, DL-87, DL-88)
  B  the launch block: nine goldens, the must and never tokens, E-17's paths, the last line (PL-29)
  C  the lambda/alpha sources (P5's table, E-40, PL-31; the alpha default repeated under an alpha-cut record;
     a records folder named through a symlink; values off the grids) and the corrected decision record (CHECK
     ITEM 5; a record deleted after its commit)
  D  launch_order (PL-27; PL-39 cases 30 to 33): the AM-8a report after any launched attempt, the code pin of
     every launch line and the head of every launched line (DL-89), the on-course repeat after C (a not_launched
     start between; each --previous-run binding), the launch line's sha256 in the gate record (PL-13)
  E  records (PL-28; PL-39 case 34; PL-12: the schedule at every launched head, a shallow clone), AM-7a on record
     (MG-1), the records command's refusals (an output that cannot be written: exit 2) and entry (the report, the
     code pin, an unresolved attempt)
  F  kd_image and the fingerprint (PL-26; PL-39 case 35), the arguments stage (clock offset: AM-19a reading
     21), teacher_identity, kd_dry_run
  G  check-run-meta (PL-30; PL-39 case 36): the trainer's real key list, one rule per key, one failing mutation
     per rule, the launched line on PASS and on FAIL, another directory refused (exit 2) without one, the same
     directory spelled with a trailing slash or dots, an unreadable gate record, and a record whose ckpt_dir or
     stage is of another form (exit 2)
  H  push-evidence end to end against a scratch bare remote (CHECK ITEM 3; AM-19a reading 10), and its refusals:
     an unreadable response, no carrying push, a shallow clone, credentials kept out of the list, the lambda
     selection's timestamp and adding commit, a malformed after, a served commit that cannot be fetched, a write
     or a rename that fails (neither file is left; a path the cleanup cannot remove is named, unless lstat reports
     it absent)
  I  the verdict (run_gate: GO, NO-GO, a rehearsal, SKIP, a crash, the record, a record that cannot be written)
"""
from __future__ import annotations

import ast
import contextlib
import errno
import hashlib
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import scripts.preflight_distill as pd  # noqa: E402
import scripts.smoke_select_sweeps as sss  # noqa: E402
import src.training.sweep_select as ss  # noqa: E402
import src.training.train_distill as td  # noqa: E402
import src.training.train_e1 as te1  # noqa: E402
from src.distill import FrozenTeacher, MockTeacher  # noqa: E402

E1G = pd.E1G
results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail="") -> None:
    results.append((name, bool(ok), "" if ok else str(detail)[:700]))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------------- fixtures
TEACHER = "/workspace/teacher/iter_24000.pth"           # an absolute path, never created
H_FAKE = "ab" * 20                                       # a records commit for the goldens (never resolved)
CLIP = (REPO / pd.AM7_CLIP_REL).read_bytes()             # the AM-7a clipping file of record
PREREG_TEXT = ("# Preregistration amendments (synthetic; smoke_preflight_distill)\n\n"
               "## AM-7 — (synthetic)\n\nNo item 5 here.\n\n"
               "## AM-7a — Student divergence (synthetic)\n\n"
               f"5. Clipping value: max_norm = 100 ({pd.AM7_CLIP_REL}, sha256\n   {pd.AM7_CLIP_SHA256}).\n\n"
               "## AM-8a — (synthetic)\n")
DL56 = (f"| DL-56 | RECORDED | AM-7a (synthetic). [Item 5 filled: max_norm = 100 ({pd.AM7_CLIP_REL}, sha256 "
        f"{pd.AM7_CLIP_SHA256[:8]}...).] | - | yes |\n")
DIRS: dict = {}


def scratch(name: str) -> Path:
    """A scratch folder of the selection smoke's kind (<prefix>_<pid>_<n>, removed at the end)."""
    return sss.scratch_dir(f"k2pf_{name}")


@contextlib.contextmanager
def git_root(path: Path):
    """The repository a command's HeadSource() reads (ss.GIT_ROOT), restored afterwards."""
    saved = ss.GIT_ROOT
    ss.GIT_ROOT = Path(path)
    try:
        yield
    finally:
        ss.GIT_ROOT = saved


def quiet(fn, *a, **k):
    """(result, stdout) of a call; a GateError or SelectionRefused becomes its code."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = fn(*a, **k)
    except (pd.GateError, ss.SelectionRefused) as e:
        out = e.code
    return out, buf.getvalue()


def code_of(fn, *a, **k):
    """A stage's status, or the code of the GateError it raises."""
    try:
        r = fn(*a, **k)
    except pd.GateError as e:
        return e.code
    return r[0] if isinstance(r, tuple) else r


# ------------------------------------------------------------------------- A: values of record
def strict_load_constants() -> dict:
    """FREEZE, HAM_KWARGS, SANDBOX, FULL_ENTRIES and FULL_PARAMS of smoke_teacher_strict_load.py, read from its
    source (never imported: it makes a temporary folder at import)."""
    tree = ast.parse((REPO / "scripts" / "smoke_teacher_strict_load.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("FREEZE", "HAM_KWARGS", "SANDBOX"):
                    out[t.id] = ast.literal_eval(node.value)
                elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(t.elts, node.value.elts):
                        if isinstance(n, ast.Name) and n.id in ("FULL_ENTRIES", "FULL_PARAMS"):
                            out[n.id] = ast.literal_eval(v)
    return out


REUSED = ("data_isolation", "repo_state", "class_weights", "smoke_loss", "cuda", "imagenet_backbone",
          "smoke_loader_seed", "smoke_dataloader", "seed_sequence_R5", "repo_unchanged")


def test_constants() -> None:
    check("a_kd_image_digest_is_dl87s",
          pd.KD_IMAGE_DIGEST == "sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b",
          pd.KD_IMAGE_DIGEST)
    check("a_fingerprint_of_record_is_block_ds", pd.FINGERPRINT_OF_RECORD == {
        "python": "3.11.16", "torch": "2.1.0+cu121", "torchvision": "0.16.0+cu121", "numpy": "1.26.4",
        "pillow": "12.3.0", "mmcv": "2.1.0", "mmengine": "0.10.7", "mmsegmentation": "1.2.2", "ftfy": "6.3.0",
        "nvidia": []} and set(pd.FINGERPRINT_DISTS) == set(pd.FINGERPRINT_OF_RECORD) - {"python", "nvidia"},
        pd.FINGERPRINT_OF_RECORD)
    fz = strict_load_constants()
    tv = pd.TEACHER_VALUES
    check("a_teacher_values_equal_the_strict_load_freeze_and_the_image_values",
          tv["config_sha256"] == fz["FREEZE"]["config_sha256"]
          and tv["teacher_components_sha256"] == fz["FREEZE"]["teacher_components_sha256"]
          and tv["reused_module_hashes"] == fz["FREEZE"]["reused_module_hashes"] and tv["ham_kwargs"] == fz["HAM_KWARGS"]
          and tv["architecture_signature"] == fz["SANDBOX"]["architecture_signature"]
          and tv["model_cfg_sha256"] == fz["SANDBOX"]["model_cfg_sha256"], f"{tv} | {fz}")
    check("a_teacher_of_record", pd.TEACHER_SHA256 == "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e"
          and pd.TEACHER_BYTES == 335_949_080 and pd.TEACHER_FILE == "iter_24000.pth"
          and pd.TEACHER_PARAMS == fz["FULL_PARAMS"] == 27_618_868
          and pd.TEACHER_STATE_ENTRIES == fz["FULL_ENTRIES"] == 854, f"{pd.TEACHER_PARAMS} {fz}")
    check("a_profiles_are_the_registered_horizons",
          sorted(p["horizon"] for p in pd.PROFILES.values()) == sorted(te1.REGISTERED_POLY_HORIZONS) == [80000, 160000]
          and pd.PROFILES["distill_80k"]["extra_args"] == () and pd.PROFILES["distill_160k"]["extra_args"]
          == ("--iterations", "160000"), pd.PROFILES)
    plan = ("--lambda", "--grad-clip-norm", "--resume", "--max-iters", "--batch-size", "--val-interval",
            "--max-val-batches", "--device")
    pl29 = ("--teacher-config", "--lambda-semantics", "--allow-semantics-mismatch", "--allow-offgrid", "--dry-run")
    check("a_never_passed_holds_the_plans_list_and_pl29s", set(plan) | set(pl29) <= set(pd.NEVER_PASSED),
          pd.NEVER_PASSED)
    kd, e1 = dict(pd.STAGES), dict(E1G.STAGES)
    check("a_reused_e1_stages_are_the_same_objects", all(kd[n] is e1[n] for n in REUSED),
          [n for n in REUSED if kd.get(n) is not e1.get(n)])
    keep = set(REUSED) | {"arguments", "module_provenance"}
    check("a_reused_stages_keep_e1s_order", [n for n, _ in pd.STAGES if n in keep] == [n for n, _ in E1G.STAGES
                                                                                     if n in keep])
    names = [n for n, _ in pd.STAGES]
    check("a_kd_stages_replace_image_and_dry_run", "image" not in names and "dry_run" not in names
          and names == list(pd.STAGE_NAMES_ORDER) and len(names) == 18
          and names.index("records") < names.index("selection_inputs") < names.index("launch_order")
          < names.index("kd_image") < names.index("teacher_identity") < names.index("kd_dry_run"), names)
    check("a_e1_values_reused", pd.CLASS_WEIGHTS_SHA256 == E1G.CLASS_WEIGHTS_SHA256
          == "fd78ba13d0bc06c0e806a5ab0b211f0044b448a494915b6135753c9701892dcf"
          and pd.NUM_WORKERS == E1G.E1_NUM_WORKERS == 12 and pd.LOG_EVERY == E1G.LOG_EVERY == 50
          and pd.EXPECTED_GPU_NAME == E1G.EXPECTED_GPU_NAME)
    check("a_next_line_is_pl29s", pd.NEXT_LINE == ("printf 'Next: check-run-meta within 5 minutes; commit and push its "
                                                   "launched line; then press Sync now on both GitHub sources\\n'"),
          pd.NEXT_LINE)
    check("a_am7_clip_file_of_record", sha(CLIP) == pd.AM7_CLIP_SHA256 and pd.AM7A_ITEM5_TEXT == "max_norm = 100"
          and pd.AM7A_DL_ROW == "DL-56", sha(CLIP))
    check("a_clock_offset_limit_is_60_seconds", pd.CLOCK_OFFSET_MAX == 60.0 and pd.DEFAULT_LAUNCH_WARN == 6 * 3600)
    rid = pd.run_id_of
    check("a_run_id_form", rid("e3", 42, 80000, 1.0, 25.0, 1) == "e3_s42_lambda1_alpha25_80k_a1"
          and rid("g", 42, 80000, None, None, 1) == "g_s42_80k_a1"
          and rid("e2", 42, 160000, 0.5, None, 2) == "e2_s42_lambda0.5_160k_a2"
          and rid("a", 42, 80000, None, 50.0, 1) == "a_s42_alpha50_80k_a1"
          and td.ckpt_dir_alpha_error(f"/workspace/kd_ckpts/{rid('e3', 42, 80000, 1.0, 25.0, 1)}", 25.0) is None
          and td.ckpt_dir_alpha_error(f"/workspace/kd_ckpts/{rid('f', 42, 80000, None, 100.0, 3)}", 100.0) is None)
    line = pd.launch_line(ss, stage="e2", seed=42, profile="distill_80k", lam=1.0, alpha=None, attempt=1,
                          schedule="T", code_pin="c" * 40, ckpt_dir="/workspace/kd_ckpts/e2_s42_lambda1_80k_a1",
                          am8a_report=None)
    log = ss.LaunchLog(pd.line_bytes(line) + b"\n")
    check("a_launch_line_has_the_logs_keys_and_parses", tuple(line) == ss.LAUNCH_KEYS
          and log.raw["e2_s42_lambda1_80k_a1"] == pd.line_bytes(line) and line["sweep"] == "lambda_logit"
          and line["value"] == 1.0 and line["horizon"] == 80000, line)
    errs = {(s, seed, prof, lam, a): pd.stage_args_error(s, seed, prof, lam, a) for s, seed, prof, lam, a in (
        ("a", 42, "distill_80k", 1.0, 50.0), ("g", 42, "distill_80k", None, 50.0), ("e3", 42, "distill_80k", 1.0, None),
        ("f", 43, "distill_80k", None, 50.0), ("e2", 43, "distill_160k", 1.0, None), ("g", 42, "distill_160k", None, None),
        ("e3", 42, "distill_160k", 1.0, 50.0), ("g", 42, "distill_80k", None, None))}
    want = {("a", 42, "distill_80k", 1.0, 50.0): "[lambda_argument]", ("g", 42, "distill_80k", None, 50.0):
            "[alpha_argument]", ("e3", 42, "distill_80k", 1.0, None): "[alpha_argument]",
            ("f", 43, "distill_80k", None, 50.0): "[seed]", ("e2", 43, "distill_160k", 1.0, None): "[iterations]",
            ("g", 42, "distill_160k", None, None): "[iterations]", ("e3", 42, "distill_160k", 1.0, 50.0): None,
            ("g", 42, "distill_80k", None, None): None}
    check("a_launch_arguments_lambda_on_a_alpha_on_g_seeds_and_the_160k_control",
          all((errs[k] is None) if w is None else (errs[k] or "").startswith(w) for k, w in want.items()), errs)


# ----------------------------------------------------------------------- B: the launch block
def block_ctx(stage, seed, *, profile="distill_80k", lam=None, alpha=None, sel_lam=False, sel_alpha=None) -> dict:
    """A context for build_launch_block with fixed fake paths (no file is touched)."""
    ident = pd.launch_identity(stage, seed, profile, lam, alpha)
    rid = pd.run_id_of(stage, seed, ident["horizon"], lam, alpha, 1)
    folder = f"/workspace/kd_records/{H_FAKE[:12]}"
    files = {"lambda_logit": None, "alpha_cwd": None}
    if sel_lam:
        files["lambda_logit"] = {"rel": ss.LAMBDA_SELECTION_REL, "path": f"{folder}/{ss.LAMBDA_SELECTION_REL}",
                                 "sha256": "1" * 64}
    if sel_alpha is not None:
        files["alpha_cwd"] = {"rel": sel_alpha, "path": f"{folder}/{sel_alpha}", "sha256": "2" * 64}
    return {"stage": stage, "seed": seed, "profile": profile, "lambda_logit": lam, "alpha": alpha,
            "ckpt_dir": f"/workspace/kd_ckpts/{rid}", "evidence": "/workspace/kd_evidence", "run_id": rid,
            "selection_files": files, "teacher_ckpt": TEACHER, "records_commit": H_FAKE,
            "root": "/workspace/plantseg_data/plantseg"}


def normalized(block: str) -> str:
    return block.replace(shlex.quote(str(pd.REPO)), "<REPO>").replace(shlex.quote(sys.executable), "<PY>")


BLOCKS = {
    "e2_s42_lambda_sweep": dict(stage="e2", seed=42, lam=0.5),
    "e2_s43": dict(stage="e2", seed=43, lam=1.0, sel_lam=True),
    "e3_s42_alpha_sweep": dict(stage="e3", seed=42, lam=1.0, alpha=25.0, sel_lam=True),
    "e3_s44": dict(stage="e3", seed=44, lam=1.0, alpha=50.0, sel_lam=True, sel_alpha=ss.ALPHA_SELECTION_REL),
    "a": dict(stage="a", seed=42, alpha=50.0, sel_alpha=ss.DECISION_RECORD_REL["alpha_cwd"]),
    "f": dict(stage="f", seed=42, alpha=100.0, sel_alpha=ss.ALPHA_SELECTION_REL),
    "g": dict(stage="g", seed=42),
    "e2_160k": dict(stage="e2", seed=42, profile="distill_160k", lam=1.0, sel_lam=True),
    "e3_160k": dict(stage="e3", seed=42, profile="distill_160k", lam=1.0, alpha=50.0, sel_lam=True,
                    sel_alpha=ss.ALPHA_SELECTION_REL),
}
_PRE = ["set -eu", "cd <REPO>"]
_ENV = ("env -u PLANTSEG_GIT_COMMIT PYTHONPATH=<REPO> PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg "
        "PLANTSEG_IMAGE_DIGEST=sha256:cb413304e2445e5c8ac3786f7a370f2ed05a07d11843b11687bb9eb23dc32c2b nohup <PY> -B "
        "src/training/train_distill.py --real-run --confirm-real-run --init imagenet")
_TAIL = (" --teacher-ckpt /workspace/teacher/iter_24000.pth --teacher-ckpt-sha256 "
         "8c0e649a1457782c99e02a3c81867c3b5aa55d924b697cc859455e6022179c4e --num-workers 12 --log-every 50 "
         f"--records-commit {H_FAKE}")
_REC = f"/workspace/kd_records/{H_FAKE[:12]}"


def golden(rid: str, args: str) -> str:
    """The launch block P5 and PL-29 give for run `rid` with these trainer arguments (between --init and
    --teacher-ckpt), written out in full."""
    d, out, pid = f"/workspace/kd_ckpts/{rid}", f"/workspace/kd_evidence/{rid}.stdout.log", f"/workspace/kd_evidence/{rid}.pid"
    return "\n".join(_PRE + [f"[ -z \"$(ls -A {d} 2>/dev/null)\" ] || {{ printf 'STOP: %s is not empty\\n' {d}; exit 1; }}",
                             f"[ ! -e {out} ] || {{ printf 'STOP: %s exists\\n' {out}; exit 1; }}",
                             f"{_ENV} {args}{_TAIL} --ckpt-dir {d} > {out} 2>&1 &", f"echo $! > {pid}",
                             pd.NEXT_LINE])


GOLDENS = {
    "e2_s42_lambda_sweep": golden("e2_s42_lambda0.5_80k_a1", "--stage e2 --seed 42 --lambda-logit 0.5"),
    "e2_s43": golden("e2_s43_lambda1_80k_a1", f"--stage e2 --seed 43 --lambda-logit 1 --lambda-selection "
                                             f"{_REC}/reports/derived/lambda_selection.json"),
    "e3_s42_alpha_sweep": golden("e3_s42_lambda1_alpha25_80k_a1", f"--stage e3 --seed 42 --lambda-logit 1 "
                                 f"--lambda-selection {_REC}/reports/derived/lambda_selection.json --alpha 25"),
    "e3_s44": golden("e3_s44_lambda1_alpha50_80k_a1", f"--stage e3 --seed 44 --lambda-logit 1 --lambda-selection "
                     f"{_REC}/reports/derived/lambda_selection.json --alpha 50 --alpha-selection "
                     f"{_REC}/reports/derived/alpha_selection.json"),
    "a": golden("a_s42_alpha50_80k_a1", f"--stage a --seed 42 --alpha 50 --alpha-selection "
                f"{_REC}/reports/derived/alpha_decision_record.json"),
    "f": golden("f_s42_alpha100_80k_a1", f"--stage f --seed 42 --alpha 100 --alpha-selection "
                f"{_REC}/reports/derived/alpha_selection.json"),
    "g": golden("g_s42_80k_a1", "--stage g --seed 42"),
    "e2_160k": golden("e2_s42_lambda1_160k_a1", f"--stage e2 --seed 42 --lambda-logit 1 --lambda-selection "
                      f"{_REC}/reports/derived/lambda_selection.json --iterations 160000"),
    "e3_160k": golden("e3_s42_lambda1_alpha50_160k_a1", f"--stage e3 --seed 42 --lambda-logit 1 --lambda-selection "
                      f"{_REC}/reports/derived/lambda_selection.json --alpha 50 --alpha-selection "
                      f"{_REC}/reports/derived/alpha_selection.json --iterations 160000"),
}


def trainer_argv(block: str) -> list:
    """The trainer's argument list from a launch block (the nohup line, up to the redirection)."""
    line = next(ln for ln in block.splitlines() if " nohup " in ln)
    toks = shlex.split(line)
    return toks[toks.index("src/training/train_distill.py") + 1:toks.index(">")]


def test_launch_block() -> None:
    blocks = {k: pd.build_launch_block(block_ctx(**v)) for k, v in BLOCKS.items()}
    for k, b in blocks.items():
        check(f"b_golden_{k}", normalized(b) == GOLDENS[k], f"\n{normalized(b)}\n!=\n{GOLDENS[k]}")
    argvs = {k: trainer_argv(b) for k, b in blocks.items()}
    must = ["--real-run", "--confirm-real-run", "--init", "imagenet", "--teacher-ckpt-sha256", pd.TEACHER_SHA256,
            "--num-workers", "12", "--log-every", "50", "--records-commit", H_FAKE, "--teacher-ckpt", TEACHER]
    check("b_must_tokens_in_every_block", all(all(t in a for t in must) for a in argvs.values()),
          {k: [t for t in must if t not in a] for k, a in argvs.items()})
    check("b_never_tokens_in_no_block", not any(t in a for a in argvs.values() for t in pd.NEVER_PASSED),
          {k: [t for t in pd.NEVER_PASSED if t in a] for k, a in argvs.items()})
    ok17 = True
    for k, b in blocks.items():
        c = block_ctx(**BLOCKS[k])
        a = argvs[k]
        out_tok = shlex.split(next(ln for ln in b.splitlines() if " nohup " in ln))
        out = out_tok[out_tok.index(">") + 1]
        ok17 &= (out == f"{c['evidence']}/{c['run_id']}.stdout.log" and not out.startswith(c["ckpt_dir"])
                 and b.splitlines()[-2] == f"echo $! > {c['evidence']}/{c['run_id']}.pid"
                 and a[a.index("--ckpt-dir") + 1] == c["ckpt_dir"]
                 and b.index(f"ls -A {c['ckpt_dir']}") < b.index(" nohup "))
    check("b_e17_stdout_and_pid_in_the_evidence_folder_and_the_ckpt_dir_checked_empty_first", ok17)
    check("b_last_line_is_pl29s", all(b.splitlines()[-1] == pd.NEXT_LINE for b in blocks.values()))
    check("b_selection_copies_and_the_horizon_by_launch",
          "--lambda-selection" not in argvs["e2_s42_lambda_sweep"] and "--alpha-selection" not in argvs["e3_s42_alpha_sweep"]
          and "--lambda-logit" not in argvs["a"] and "--lambda-logit" not in argvs["g"] and "--alpha" not in argvs["g"]
          and all(a[a.index("--iterations") + 1] == "160000" and a.index("--iterations") + 2 == a.index("--teacher-ckpt")
                  for a in (argvs["e2_160k"], argvs["e3_160k"]))
          and not any("--iterations" in argvs[k] for k in BLOCKS if not k.endswith("160k")), argvs)


# ------------------------------------------------------------------- the gate's scenarios (C, D, E)
def gate_sweep(label: str, key: str = "lambda_logit", *, files=None, prereg=True, dl56=True, clip=CLIP,
               lam=1.0) -> "sss.Sweep":
    """An AM-19 scenario of the selection smoke whose pin also holds MG-1's records: the AM-7a section (read at the
    pin), the clipping file and the DL-56 row (read at the records commit)."""
    sw = sss.Sweep(label, key, lam=lam, files=files)
    pins = {pd.AM7_CLIP_REL: clip, ss.DECISION_LOG_REL: sss.DL_TEXT + (DL56 if dl56 else "")}
    if prereg:
        pins[pd.PREREG_REL] = PREREG_TEXT
    sw.pin_files(pins, "pin: the code, AM-7a and its records")
    return sw


def launch_lines(sw) -> list:
    return [x for x in sw.lines + sw.events if x.get("event") == "launch"]


def own_line(sw, *, stage, seed=42, profile="distill_80k", lam=None, alpha=None, report=None) -> dict:
    """The gate's own launch line, built as the gate builds it (attempt = 1 + the earlier lines of its group)."""
    ident = pd.launch_identity(stage, seed, profile, lam, alpha)
    probe = pd.launch_line(ss, stage=stage, seed=seed, profile=profile, lam=lam, alpha=alpha, attempt=1,
                           schedule=sw.schedule, code_pin=sw.pin, ckpt_dir="/x/y", am8a_report=None)
    attempt = 1 + sum(1 for x in launch_lines(sw) if pd.group_of(x) == pd.group_of(probe))
    rid = pd.run_id_of(stage, seed, ident["horizon"], lam, alpha, attempt)
    rep = None if report is None else sw.report(report, f"AM-8a report before {rid} (synthetic)\n")
    return pd.launch_line(ss, stage=stage, seed=seed, profile=profile, lam=lam, alpha=alpha, attempt=attempt,
                          schedule=sw.schedule, code_pin=sw.pin, ckpt_dir=str(DIRS["ck"] / rid), am8a_report=rep)


def gate_ctx(sw, line: dict, *, now: float, profile="distill_80k", logged="same", head="pin", decision_records=(),
             previous_runs=(), report=None, edit_folder=None) -> dict:
    """Commit the gate's own line (logged as built, edited by `logged`, or left out with logged=None) as the
    records commit H; check out `head` (the pin by default); write the records folder with `records`; return the
    gate's context. HEAD is detached in this scratch repository only (PL-22)."""
    if logged is not None:
        sw.events.append(dict(line) if logged == "same" else logged(dict(line)))
    h = sw.commit("the gate's launch line")
    sss.git(sw.repo, "checkout", "-q", "--detach", sw.pin)
    folder = scratch("records")
    with git_root(sw.repo):
        rc, out = quiet(pd.cmd_records, SimpleNamespace(records_commit=h, out=str(folder)))
    assert rc == 0, out
    if head == "side":                                   # a commit that is not an ancestor of H
        sss.commit_files(sw.repo, {"side.txt": "a side commit\n"}, "a side commit off the pin")
    elif head == "h":
        sss.git(sw.repo, "checkout", "-q", "--detach", h)
    if edit_folder is not None:
        edit_folder(folder)
    stage = next(k for k, v in pd.STAGE_NAMES.items() if v == line["stage"])
    lam, alpha = line["lambda_logit"], line["alpha_cwd"]
    return {"profile": profile, "stage": stage, "seed": line["seed"], "ckpt_dir": line["ckpt_dir"],
            "evidence": str(DIRS["ev"]), "expect_head": sw.pin, "records_commit": h, "records": str(folder),
            "teacher_ckpt": TEACHER, "clock_offset": 0.0, "record": None, "rehearsal": False, "lambda_logit": lam,
            "alpha": alpha, "attempt": line["attempt"], "am8a_report": report,
            "previous_runs": [str(p) for p in previous_runs], "decision_records": list(decision_records),
            "git_root": sw.repo, "now": now, "warnings": [], "child_json": None, "child_run": None,
            "root": DIRS["data"], "ident": pd.launch_identity(stage, line["seed"], profile, lam, alpha),
            "run_id": line["run_id"]}


KD_STAGES = ("records", "selection_inputs", "launch_order")


def run_kd(ctx: dict, stages=KD_STAGES) -> str:
    """The gate's KD stages on `ctx`, in order: PASS, or '<stage>:<code>' of the first that does not pass."""
    fns = dict(pd.STAGES)
    for name in stages:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status, _detail, _rec = fns[name](ctx)
        except pd.GateError as e:
            return f"{name}:{e.code}"
        except Exception as e:  # noqa: BLE001 - a crash is reported as such (the gate makes it stage_error)
            return f"{name}:crash:{type(e).__name__}: {e}"
        if status != "PASS":
            return f"{name}:{status}"
    return "PASS"


def run_kd_detail(ctx: dict, stages=KD_STAGES) -> tuple:
    """run_kd with the refusal's text: (verdict, message), for cases whose code several checks share."""
    fns = dict(pd.STAGES)
    for name in stages:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                status, _detail, _rec = fns[name](ctx)
        except pd.GateError as e:
            return f"{name}:{e.code}", str(e)
        except Exception as e:  # noqa: BLE001 - a crash is reported as such
            return f"{name}:crash:{type(e).__name__}: {e}", ""
        if status != "PASS":
            return f"{name}:{status}", ""
    return "PASS", ""


def lam_gate_sweep(label: str, *, values=(1,), files=None, **kw) -> "sss.Sweep":
    """A lambda sweep for the gate: the values given launched in the launch order from BASE_L and finished."""
    sw = gate_sweep(label, files=files, **kw)
    for i, v in enumerate(values):
        sw.add(v, sss.fin(sss.BASE_L + 600 * i, 0.42 - 0.01 * i))
    return sw.build()


def alpha_gate_sweep(label: str, *, values=(50,), files=None, **kw) -> "sss.Sweep":
    """An alpha sweep for the gate (stage E3 at the selected lambda 1): the values given launched from BASE_A."""
    sw = gate_sweep(label, "alpha_cwd", files=sss.alpha_files() if files is None else files, **kw)
    for i, a in enumerate(values):
        sw.add(a, sss.fin(sss.BASE_A + 600 * i, 0.43, stage="E3"))
    return sw.build()


def lambda_selection(winner: float = 1.0) -> str:
    return json.dumps({"format": "lambda_selection/1", "winner": {"lambda": winner, "run_id": "e2_s42_lambda1_a1"},
                       "inputs_last_timestamp_utc": sss.T_IN})


def alpha_selection(winner: float = 50.0) -> str:
    return json.dumps({"format": "alpha_selection/1", "winner": {"alpha": winner, "run_id": "e3_s42_alpha50_a1"}})


def selfile(ctx: dict, key: str) -> dict:
    """The records copy the launch block hands the trainer for `key` ({} when none)."""
    return ((ctx.get("selection_files") or {}).get(key)) or {}


def gate(sw, *, now: float, profile="distill_80k", decision_records=(), **line_kw) -> tuple:
    """The gate's KD stages for the scenario's own launch: (verdict, context)."""
    ctx = gate_ctx(sw, own_line(sw, profile=profile, **line_kw), now=now, profile=profile,
                   decision_records=decision_records)
    return run_kd(ctx), ctx


# ------------------------------------------------------------------- C: the lambda/alpha sources
def test_sources() -> None:
    now_l, now_a = sss.BASE_L + 3600.0, sss.BASE_A + 3600.0
    nulls = {"lambda_logit": None, "alpha_cwd": None}
    lsel_text = lambda_selection(1.0)
    lsel = {ss.LAMBDA_SELECTION_REL: lsel_text}
    both = {**lsel, ss.ALPHA_SELECTION_REL: alpha_selection(50.0)}

    got, ctx = gate(lam_gate_sweep("c_g"), stage="g", now=now_l)
    check("c_g_with_nothing_is_accepted", got == "PASS" and ctx.get("selection_files") == nulls,
          f"{got} {ctx.get('selection_files')}")
    got, _ = gate(lam_gate_sweep("c_e2s43_missing"), stage="e2", seed=43, lam=1.0, now=now_l)
    check("c_e2_s43_without_the_lambda_selection_is_refused", got == "selection_inputs:lambda_selection_missing", got)
    got, _ = gate(lam_gate_sweep("c_e2s43_other", files=lsel), stage="e2", seed=43, lam=0.5, now=now_l)
    check("c_e2_s43_lambda_other_than_the_winner_is_refused", got == "selection_inputs:lambda_not_selected", got)
    got, ctx = gate(lam_gate_sweep("c_e2s43_ok", files=lsel), stage="e2", seed=43, lam=1.0, now=now_l)
    sel = selfile(ctx, "lambda_logit")
    check("c_e2_s43_lambda_from_the_selection_hands_the_records_copy", got == "PASS"
          and sel.get("rel") == ss.LAMBDA_SELECTION_REL and sel.get("path") == str(Path(ctx["records"]).resolve()
                                                                                 / ss.LAMBDA_SELECTION_REL)
          and sel.get("sha256") == sha(lsel_text.encode("utf-8")) and selfile(ctx, "alpha_cwd") == {},
          f"{got} {ctx.get('selection_files')}")
    got, _ = gate(lam_gate_sweep("c_late_lambda", values=(1, 0.5, 2, 0.25), files=lsel), stage="e2", lam=4.0,
                  now=now_l)
    check("c_late_lambda_candidate_other_than_the_winner_is_refused",
          got == "selection_inputs:lambda_selection_exists", got)
    got, ctx = gate(alpha_gate_sweep("c_alpha_ok"), stage="e3", lam=1.0, alpha=25.0, now=now_a)
    check("c_e3_alpha_sweep_value_before_the_selection_is_accepted", got == "PASS"
          and selfile(ctx, "lambda_logit").get("rel") == ss.LAMBDA_SELECTION_REL and selfile(ctx, "alpha_cwd") == {}
          and ctx.get("decision_date") == "2026-10-22" and ctx.get("launch_order_position") == 2,
          f"{got} {ctx.get('decision_date')} {ctx.get('launch_order_position')}")
    no_band = {k: v for k, v in sss.alpha_files().items() if k != ss.BAND_REL}
    got, _ = gate(alpha_gate_sweep("c_alpha_no_band", files=no_band), stage="e3", lam=1.0, alpha=25.0, now=now_a)
    check("c_alpha_sweep_without_the_band_is_refused", got == "selection_inputs:band_missing", got)
    got, _ = gate(alpha_gate_sweep("c_alpha_after_sel", files={**sss.alpha_files(),
                                                               ss.ALPHA_SELECTION_REL: alpha_selection(50.0)}),
                  stage="e3", lam=1.0, alpha=25.0, now=now_a)
    check("c_e3_alpha_sweep_value_after_the_selection_is_refused", got == "selection_inputs:alpha_selection_exists",
          got)
    got, _ = gate(alpha_gate_sweep("c_alpha_winner", files={**sss.alpha_files(),
                                                            ss.ALPHA_SELECTION_REL: alpha_selection(25.0)}),
                  stage="e3", lam=1.0, alpha=25.0, now=now_a)
    check("c_e3_alpha_sweep_winner_after_the_selection_is_accepted", got == "PASS", got)
    # E-40: a value a committed record cuts (25 never launched; 100 launched, so the sweep is not cut)
    sw = alpha_gate_sweep("c_alpha_cut_value", values=(50, 100))
    rc, log, _ = sw.record(sss.NOW_A)
    got, _ = gate(sw, stage="e3", lam=1.0, alpha=25.0, now=sss.NOW_A)
    check("c_e3_alpha_value_a_record_cuts_is_refused", rc == 0 and got == "selection_inputs:alpha_cut_by_record",
          f"{sss.result_line(log)} {got}")
    # PL-31: alpha 50 from the alpha-cut record, with the record's sha256
    sw = alpha_gate_sweep("c_alpha_cut_record")
    rc, log, rsha = sw.record(sss.NOW_A)
    got, ctx = gate(sw, stage="a", alpha=50.0, now=sss.NOW_A)
    sel = selfile(ctx, "alpha_cwd")
    check("c_a_alpha_50_from_the_alpha_cut_record_with_its_sha256", rc == 0 and got == "PASS"
          and sel.get("rel") == ss.DECISION_RECORD_REL["alpha_cwd"] and sel.get("sha256") == rsha
          and selfile(ctx, "lambda_logit") == {}, f"{sss.result_line(log)} {got} {sel}")
    sw = alpha_gate_sweep("c_alpha_cut_other_value")
    sw.record(sss.NOW_A)
    got, _ = gate(sw, stage="f", alpha=25.0, now=sss.NOW_A)
    check("c_f_alpha_other_than_50_under_the_alpha_cut_record_is_refused", got == "selection_inputs:alpha_not_selected",
          got)
    sw = alpha_gate_sweep("c_alpha_cut_early")
    sw.record(sss.NOW_A)
    got, _ = gate(sw, stage="a", alpha=50.0, now=sss.ALPHA_C - 60.0)
    check("c_alpha_cut_record_before_the_alpha_date_ends_is_refused", got == "selection_inputs:alpha_cut_record_invalid",
          got)
    sw = alpha_gate_sweep("c_alpha_cut_conflict")
    sw.record(sss.NOW_A)
    sw.commit("an alpha selection beside the cut record", {ss.ALPHA_SELECTION_REL: alpha_selection(50.0)})
    got, _ = gate(sw, stage="a", alpha=50.0, now=sss.NOW_A)
    check("c_alpha_selection_beside_an_alpha_cut_record_is_refused", got == "selection_inputs:alpha_cut_conflict", got)
    # PL-39 case 33: an alpha-cut record contradicted by a non-default launched line
    sw = alpha_gate_sweep("c_alpha_cut_contradicted", values=(50, 25))
    rc, log, _ = sw.record(sss.NOW_A, edit=lambda d: d.update(alpha_sweep_cut=True))
    got, _ = gate(sw, stage="a", alpha=50.0, now=sss.NOW_A)
    check("pl39_33_alpha_cut_record_contradicted_by_a_non_default_launched_line_is_refused",
          rc == 0 and got == "selection_inputs:alpha_cut_record_invalid", f"{sss.result_line(log)} {got}")
    # PL-39 case 33: a lambda value a committed record cuts (0.25 never launched)
    sw = lam_gate_sweep("c_lambda_cut", values=(1, 0.5, 2))
    rc, log, _ = sw.record(sss.NOW_L)
    got, _ = gate(sw, stage="e2", lam=0.25, now=sss.NOW_L)
    check("pl39_33_lambda_value_a_record_cuts_is_refused", rc == 0 and got == "selection_inputs:lambda_cut_by_record",
          f"{sss.result_line(log)} {got}")
    # A and F take alpha from the selection or the cut record; E3 s44 and the 160k controls from the selections
    got, _ = gate(lam_gate_sweep("c_a_nothing", files=lsel), stage="a", alpha=50.0, now=now_l)
    check("c_a_without_a_selection_or_a_record_is_refused", got == "selection_inputs:alpha_source_missing", got)
    got, _ = gate(lam_gate_sweep("c_f_other", files=both), stage="f", alpha=25.0, now=now_l)
    check("c_f_alpha_other_than_the_selection_is_refused", got == "selection_inputs:alpha_not_selected", got)
    got, ctx = gate(lam_gate_sweep("c_f_ok", files=both), stage="f", alpha=50.0, now=now_l)
    check("c_f_alpha_from_the_selection_is_accepted", got == "PASS"
          and selfile(ctx, "alpha_cwd").get("rel") == ss.ALPHA_SELECTION_REL and selfile(ctx, "lambda_logit") == {},
          f"{got} {ctx.get('selection_files')}")
    got, ctx = gate(lam_gate_sweep("c_e3s44", files=both), stage="e3", seed=44, lam=1.0, alpha=50.0, now=now_l)
    check("c_e3_s44_lambda_and_alpha_from_the_selections", got == "PASS"
          and selfile(ctx, "lambda_logit").get("rel") == ss.LAMBDA_SELECTION_REL
          and selfile(ctx, "alpha_cwd").get("rel") == ss.ALPHA_SELECTION_REL, f"{got} {ctx.get('selection_files')}")
    got, ctx = gate(lam_gate_sweep("c_e2_160k", files=lsel), stage="e2", lam=1.0, profile="distill_160k", now=now_l)
    check("c_e2_160k_takes_lambda_from_the_selection_outside_the_sweep", got == "PASS"
          and selfile(ctx, "lambda_logit").get("rel") == ss.LAMBDA_SELECTION_REL and ctx["ident"]["sweep"] is None
          and ctx.get("decision_date") is None, f"{got} {ctx.get('selection_files')}")
    got, ctx = gate(lam_gate_sweep("c_e3_160k", files=both), stage="e3", lam=1.0, alpha=50.0, profile="distill_160k",
                    now=now_l)
    check("c_e3_160k_takes_lambda_and_alpha_from_the_selections", got == "PASS"
          and selfile(ctx, "alpha_cwd").get("rel") == ss.ALPHA_SELECTION_REL, f"{got} {ctx.get('selection_files')}")
    # C4 workflow c4-2 (E-40; AM-19 item 2(g); AM-19a readings 16 and 19): an alpha-cut record written while the
    # alpha default runs never cuts the default: its AM-8a repeat, after the first attempt's pod is lost, launches
    sw = gate_sweep("c_alpha_cut_default_repeat", "alpha_cwd", files=sss.alpha_files())
    sw.add(50, sss.course(sss.BASE_A, 3000, val_at=(), stage="E3"))
    sw.build()
    rc, log, _ = sw.record(sss.NOW_A)
    a1 = "e3_s42_alpha50_a1"
    sw.events.append({"event": "stopped", "run_id": a1,
                      "report": sw.report(f"reports/stops/{a1}.md", f"{a1}: the pod was lost (synthetic)\n")})
    sw.commit("the alpha default's first attempt stopped")
    line = own_line(sw, stage="e3", lam=1.0, alpha=50.0, report="reports/am8a/e3_s42_alpha50_80k_a2.md")
    got = run_kd(gate_ctx(sw, line, now=sss.NOW_A, report=line["am8a_report"]["path"]))
    check("c_c4_2_alpha_default_repeated_under_an_alpha_cut_record_is_accepted", rc == 0 and got == "PASS",
          f"{sss.result_line(log)} {got}")
    # c4-10: the records folder named through a symlink: the copies the launch block hands the trainer are named
    # resolved, as the trainer records them (run_meta selection_files)
    sw = lam_gate_sweep("c_records_link", files=lsel)
    ctx = gate_ctx(sw, own_line(sw, stage="e2", seed=43, lam=1.0), now=now_l)
    real = Path(ctx["records"])
    link = scratch("records_link") / "f"
    link.symlink_to(real, target_is_directory=True)
    ctx["records"] = str(link)
    got = run_kd(ctx)
    check("c_c4_10_records_folder_through_a_symlink_hands_the_resolved_copies", got == "PASS"
          and selfile(ctx, "lambda_logit").get("path") == str(real.resolve() / ss.LAMBDA_SELECTION_REL),
          f"{got} {ctx.get('selection_files')}")
    # c4-17: a lambda or alpha off its grid is refused by the sources stage itself
    sw = lam_gate_sweep("c_grid_lambda")
    ctx = gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now_l)
    got = run_kd(dict(ctx, lambda_logit=3.0), ("records", "selection_inputs"))
    check("c_c4_17_lambda_off_the_grid_is_refused", got == "selection_inputs:grid_mismatch", got)
    sw = alpha_gate_sweep("c_grid_alpha")
    ctx = gate_ctx(sw, own_line(sw, stage="e3", lam=1.0, alpha=25.0), now=now_a)
    got = run_kd(dict(ctx, alpha=30.0), ("records", "selection_inputs"))
    check("c_c4_17_alpha_off_the_grid_is_refused", got == "selection_inputs:grid_mismatch", got)


def test_corrected() -> None:
    """CHECK ITEM 5 at the gate (AM-19a reading 19): the corrected record through --decision-record; its place, its
    references and written once, read from the records commit (the selection derives whether the void is void)."""
    void_rel, fix_rel = ss.DECISION_RECORD_REL["lambda_logit"], ss.DECISION_RECORD_CORRECTED_REL["lambda_logit"]
    fault_rel = "reports/am8a/record_fault.md"

    def forged(d):
        next(e for e in d["values"] if e["value"] == 1.0)["directories"][0].update(run_meta_sha256="0" * 64)

    def dl_plus(sw, row: str) -> str:
        return (sw.repo / ss.DECISION_LOG_REL).read_text(encoding="utf-8") + row

    def corrected(label: str):
        sw = lam_gate_sweep(label, values=(1, 0.5, 2))
        sw.record(sss.NOW_L, edit=forged)
        text = "AM-8a report: the decision record was refused (synthetic)\n"
        sw.commit("fault report", {fault_rel: text, ss.DECISION_LOG_REL: dl_plus(
            sw, f"| DL-94 | RECORDED | fault report, sha256 {sha(text.encode('utf-8'))} (synthetic) | - | no |\n")})
        rc, log, csha = sw.record_at(sss.NOW_L, fix_rel, corrects=(void_rel, fault_rel))
        return sw, rc, log, csha

    sw, rc, log, csha = corrected("c_corr_ok")
    ctx = gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L, decision_records=[fix_rel])
    got = run_kd(ctx)
    man = json.loads((Path(ctx["records"]) / pd.RECORDS_MANIFEST).read_text(encoding="utf-8"))
    rec = (ctx.get("sweep_records") or {}).get("lambda_logit") or {}
    check("c_corrected_record_named_with_decision_record_is_the_record_in_force", rc == 0 and got == "PASS"
          and rec.get("path") == fix_rel and rec.get("sha256") == csha
          and {void_rel, fix_rel, fault_rel} <= set(man["files"]), f"{sss.result_line(log)} {got} {sorted(man['files'])}")
    sw, *_ = corrected("c_corr_unnamed")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L))
    check("c_corrected_record_at_the_records_commit_must_be_named", got == "selection_inputs:decision_record_correction_invalid",
          got)
    sw, *_ = corrected("c_corr_void_named")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L, decision_records=[void_rel]))
    check("c_naming_the_void_record_beside_its_correction_is_refused",
          got == "selection_inputs:decision_record_correction_invalid", got)
    sw = lam_gate_sweep("c_corr_elsewhere", values=(1, 0.5, 2))
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L,
                          decision_records=["reports/derived/lambda_decision_record_v2.json"]))
    check("c_decision_record_at_another_path_is_refused", got == "selection_inputs:decision_record_correction_invalid",
          got)
    sw, *_ = corrected("c_corr_twice")
    second = json.loads((sw.repo / fix_rel).read_text(encoding="utf-8"))
    text2 = "AM-8a report 2 (synthetic)\n"
    second["corrects"]["fault_report"] = {"path": "reports/am8a/record_fault_2.md", "sha256": sha(text2.encode("utf-8"))}
    data = json.dumps(second, indent=2) + "\n"
    sw.commit("a second correction written over the first", {
        "reports/am8a/record_fault_2.md": text2, fix_rel: data, ss.DECISION_LOG_REL: dl_plus(
            sw, f"| DL-95 | RECORDED | report 2 sha256 {sha(text2.encode('utf-8'))}; correction 2 sha256 "
                f"{sha(data.encode('utf-8'))} (synthetic) | - | no |\n")})
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L, decision_records=[fix_rel]))
    check("c_corrected_record_written_over_is_refused_at_the_gate", got == "selection_inputs:decision_record_rewritten",
          got)
    sw, *_ = corrected("c_corr_report_row")
    report_sha = sha(b"AM-8a report: the decision record was refused (synthetic)\n")
    dl_p = sw.repo / ss.DECISION_LOG_REL
    sw.commit("the fault report's row removed", {ss.DECISION_LOG_REL: "".join(
        ln for ln in dl_p.read_text(encoding="utf-8").splitlines(keepends=True) if report_sha not in ln)})
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L, decision_records=[fix_rel]))
    check("c_corrected_record_whose_fault_report_is_not_in_the_decision_log_is_refused",
          got == "selection_inputs:decision_record_correction_invalid", got)
    # C1b round 3 c1b3-3: a record deleted after its commit, at either place, is decision_record_rewritten at the gate,
    # as at the selection (ss.deleted_record_error), never a sweep without a record
    sw, *_ = corrected("c_corr_deleted")
    sss.git(sw.repo, "rm", "-q", "--", fix_rel)
    sss.git(sw.repo, "commit", "-q", "-m", "the corrected record deleted")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L))
    check("c_corrected_record_deleted_after_its_commit_is_refused_at_the_gate",
          got == "selection_inputs:decision_record_rewritten", got)
    sw = lam_gate_sweep("c_record_deleted", values=(1, 0.5, 2))
    rc, log, _ = sw.record(sss.NOW_L)
    sss.git(sw.repo, "rm", "-q", "--", void_rel)
    sss.git(sw.repo, "commit", "-q", "-m", "the decision record deleted")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=sss.NOW_L))
    check("c_record_deleted_after_its_commit_is_refused_at_the_gate",
          rc == 0 and got == "selection_inputs:decision_record_rewritten", f"{sss.result_line(log)} {got}")


# --------------------------------------------------------------------------- D: launch_order
def test_launch_order() -> None:
    now = sss.BASE_L + 3600.0
    got, ctx = gate(lam_gate_sweep("d_ok"), stage="e2", lam=0.5, now=now)
    check("d_non_default_after_the_default_launches_with_its_date_and_position", got == "PASS"
          and ctx.get("decision_date") == "2026-10-19" and ctx.get("cutoff_utc") == sss.LAMBDA_C
          and ctx.get("launch_order_position") == 2 and not ctx["warnings"]
          and ctx.get("launch_line", {}).get("code_pin") == ctx.get("pin"), got)
    # C4 workflow c4-11 (PL-13): the gate record stores the sha256 of its launch line (the log's bytes, no newline)
    grec = pd.gate_record(ctx, [], "GO")
    check("d_c4_11_gate_record_stores_the_sha256_of_its_launch_line", got == "PASS"
          and grec.get("launch_line_sha256") == sha(pd.line_bytes(ctx.get("launch_line") or {})),
          str(grec.get("launch_line_sha256")))
    sw = lam_gate_sweep("d_missing")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now, logged=None))
    check("pl39_30_own_launch_line_missing_is_refused", got == "launch_order:launch_line_missing", got)
    sw = lam_gate_sweep("d_one_byte")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now,
                          logged=lambda ln: dict(ln, ckpt_dir=ln["ckpt_dir"].replace("k2pf_ck", "k2pf_cK", 1))))
    check("pl39_30_own_launch_line_differing_by_one_byte_is_refused", got == "launch_order:launch_line_mismatch", got)
    sw = gate_sweep("d_no_default")
    sw.add(1, [], state="pending", launch=sss.BASE_L)
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("pl39_31_default_launched_line_missing_is_refused", got == "launch_order:default_not_launched", got)
    got, _ = gate(lam_gate_sweep("d_order"), stage="e2", lam=2.0, now=now)
    check("d_out_of_the_launch_order_is_refused", got == "launch_order:launch_order", got)
    got, _ = gate(lam_gate_sweep("d_date_ended", values=(1, 0.5)), stage="e2", lam=2.0, now=sss.NOW_L)
    check("pl39_32_first_attempt_after_the_decision_date_is_refused", got == "launch_order:decision_date_ended", got)
    for label, rows, want in (("not_on_course", sss.course(sss.BASE_L + 600, 3000, val_at=()),
                               "launch_order:repeat_not_on_course"),
                              ("on_course", sss.oc_stop_rows(), "PASS")):
        sw = gate_sweep(f"d_{label}")
        sw.add(1, sss.fin(sss.BASE_L, 0.42))
        sw.add(0.5, rows, stopped=True)
        sw.build()
        line = own_line(sw, stage="e2", lam=0.5, report="reports/am8a/e2_s42_lambda0.5_80k_a2.md")
        ctx = gate_ctx(sw, line, now=sss.NOW_L, report=line["am8a_report"]["path"],
                       previous_runs=[sw.dir("e2_s42_lambda0.5_a1")])
        got = run_kd(ctx)
        name = ("pl39_32_repeat_after_the_date_whose_previous_attempt_is_not_an_on_course_stop_is_refused"
                if want != "PASS" else "d_on_course_repeat_after_the_date_is_accepted")
        check(name, got == want, got)
    sw = gate_sweep("d_no_report")
    sw.add(1, sss.fin(sss.BASE_L, 0.42))
    sw.add(0.5, sss.course(sss.BASE_L + 600, 3000, val_at=()), stopped=True)
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_repeat_of_a_launched_attempt_without_its_am8a_report_is_refused", got == "launch_order:repeat_report_missing",
          got)
    sw = gate_sweep("d_pin")
    sw.add(1, sss.fin(sss.BASE_L, 0.42), code_pin=sss.template()[1])
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_dl89_sweep_launched_at_another_pin_is_code_pin_mismatch", got == "launch_order:code_pin_mismatch", got)
    # c4-17 (C4-M13's killer): every launch line carries the pin, but a launched line ran at another head
    sw = gate_sweep("d_head_other")
    sw.add(1, sss.fin(sss.BASE_L, 0.42), launched_over={"git_head": sss.template()[1]})
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_c4_17_launched_line_at_another_head_is_code_pin_mismatch", got == "launch_order:code_pin_mismatch", got)
    got, ctx = gate(lam_gate_sweep("d_warn"), stage="e2", lam=0.5, now=sss.BASE_L + 7 * 3600.0)
    check("d_more_than_6_hours_after_the_default_launch_warns", got == "PASS" and len(ctx["warnings"]) == 1
          and "more than 6 h" in ctx["warnings"][0], f"{got} {ctx['warnings']}")
    sw = lam_gate_sweep("d_head")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now, head="h"))
    check("pl39_34_head_not_the_pin_is_refused", got == "launch_order:launch_line_mismatch", got)
    # C4 workflow c4-1 (PL-9(c), PL-27): an attempt after a launched one names its AM-8a report, also when a
    # not_launched start lies between them; with its report it passes

    def after_not_launched(label):
        s = gate_sweep(label)
        s.add(1, sss.fin(sss.BASE_L, 0.42))
        s.add(0.5, sss.course(sss.BASE_L + 600, 3000, val_at=()), stopped=True)
        s.add(0.5, state="not_launched", launch=sss.BASE_L + 2000)
        return s.build()
    sw = after_not_launched("d_report_after_nl")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_c4_1_attempt_after_a_launched_one_and_a_not_launched_start_without_its_report_is_refused",
          got == "launch_order:repeat_report_missing", got)
    sw = after_not_launched("d_report_after_nl_ok")
    line = own_line(sw, stage="e2", lam=0.5, report="reports/am8a/e2_s42_lambda0.5_80k_a3.md")
    got = run_kd(gate_ctx(sw, line, now=now, report=line["am8a_report"]["path"]))
    check("d_c4_1_attempt_after_a_launched_one_and_a_not_launched_start_with_its_report_passes", got == "PASS", got)
    # c4-3 (DL-89; PL-14): a not_launched start at another code pin is the sweep's second pin
    sw = gate_sweep("d_pin_not_launched")
    sw.add(1, sss.fin(sss.BASE_L, 0.42))
    sw.add(0.5, state="not_launched", launch=sss.BASE_L + 2000, code_pin=sss.template()[1])
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_c4_3_not_launched_start_at_another_pin_is_code_pin_mismatch", got == "launch_order:code_pin_mismatch", got)
    # c4-6 (PL-9(e)(3): r is the attempt launched next after s): after C, the on-course repeat whose first start was
    # not launched is started again

    def on_course_value(label, *, retry=False):
        s = gate_sweep(label)
        s.add(1, sss.fin(sss.BASE_L, 0.42))
        s.add(0.5, sss.oc_stop_rows(), stopped=True)
        if retry:
            s.add(0.5, state="not_launched", launch=sss.NOW_L - 600)
        return s.build()
    sw = on_course_value("d_oc_retry", retry=True)
    line = own_line(sw, stage="e2", lam=0.5, report="reports/am8a/e2_s42_lambda0.5_80k_a3.md")
    got = run_kd(gate_ctx(sw, line, now=sss.NOW_L, report=line["am8a_report"]["path"],
                          previous_runs=[sw.dir("e2_s42_lambda0.5_a1")]))
    check("d_c4_6_on_course_repeat_started_again_after_a_not_launched_start_is_accepted", got == "PASS", got)
    # c4-17: a previous attempt without an outcome; each --previous-run binding (PL-27)
    sw = gate_sweep("d_unresolved")
    sw.add(1, sss.fin(sss.BASE_L, 0.42))
    sw.add(0.5, [], state="pending", launch=sss.BASE_L + 600)
    sw.build()
    got = run_kd(gate_ctx(sw, own_line(sw, stage="e2", lam=0.5), now=now))
    check("d_c4_17_previous_attempt_without_an_outcome_is_refused", got == "launch_order:previous_attempt_unresolved",
          got)
    for label, runs, text in (
            ("other_value", lambda s: [s.dir("e2_s42_lambda1_a1")], "not a launched earlier attempt of this value"),
            ("run_meta_edited", None, "does not hash to the launched line's run_meta_sha256"),
            ("stopped_attempt_withheld", lambda s: [], "need their --previous-run directories")):
        sw = on_course_value(f"d_prev_{label}")
        if runs is None:
            alt = scratch("prev_alt") / "e2_s42_lambda0.5_a1"
            shutil.copytree(sw.dir("e2_s42_lambda0.5_a1"), alt)
            meta = alt / "e2_run_meta.jsonl"
            meta.write_bytes(meta.read_bytes() + b"\n")
            prev = [alt]
        else:
            prev = runs(sw)
        line = own_line(sw, stage="e2", lam=0.5, report="reports/am8a/e2_s42_lambda0.5_80k_a2.md")
        got, msg = run_kd_detail(gate_ctx(sw, line, now=sss.NOW_L, report=line["am8a_report"]["path"],
                                          previous_runs=prev))
        check(f"d_c4_17_previous_run_{label}_is_refused", got == "launch_order:repeat_not_on_course" and text in msg,
              f"{got} {msg[:200]}")


# ------------------------------------------------------------------ E: records, AM-7a and entry
def test_records() -> None:
    now = sss.BASE_L + 3600.0
    sw = lam_gate_sweep("e_ok")
    ctx = gate_ctx(sw, own_line(sw, stage="g"), now=now)
    got = run_kd(ctx, ("records",))
    folder = Path(ctx["records"])
    man = json.loads((folder / pd.RECORDS_MANIFEST).read_text(encoding="utf-8"))
    blobs_ok = all(ss.git_blob_id((folder / rel).read_bytes()) == sss.git(sw.repo, "rev-parse",
                                                                        f"{ctx['records_commit']}:{rel}")
                   for rel in man["files"])
    check("e_records_folder_holds_the_records_commits_blobs_and_its_manifest", got == "PASS"
          and man["format"] == pd.RECORDS_FORMAT and man["records_commit"] == ctx["records_commit"]
          and man["pin"] == sw.pin and blobs_ok
          and {ss.LAUNCH_LOG_REL, ss.DECISION_LOG_REL, pd.AM7_CLIP_REL} <= set(man["files"])
          and ss.ALPHA_SELECTION_REL in man["absent"] and ctx.get("records_blobs") == man["files"], f"{got} {man}")
    sw = lam_gate_sweep("e_side")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=now, head="side"), ("records",))
    check("pl39_34_pin_not_an_ancestor_of_the_records_commit_is_refused", got == "records:pin_not_ancestor", got)

    def touch(folder: Path) -> None:
        p = folder / ss.DECISION_LOG_REL
        p.write_bytes(p.read_bytes() + b"\n")
    sw = lam_gate_sweep("e_blob")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=now, edit_folder=touch), ("records",))
    check("pl39_34_records_file_whose_blob_differs_is_records_mismatch", got == "records:records_mismatch", got)
    sw = lam_gate_sweep("e_manifest")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=now,
                          edit_folder=lambda f: (f / pd.RECORDS_MANIFEST).unlink()), ("records",))
    check("e_records_folder_without_its_manifest_is_refused", got == "records:records_mismatch", got)
    sw = lam_gate_sweep("e_commit")
    ctx = gate_ctx(sw, own_line(sw, stage="g"), now=now)
    bad = [run_kd(dict(ctx, records_commit=h), ("records",)) for h in ("ABC", "e" * 40)]
    check("e_records_commit_malformed_or_absent_is_refused",
          bad == ["records:records_commit_malformed", "records:records_commit_unavailable"], bad)
    for label, kw, name in (("e_no_prereg", dict(prereg=False), "e_am7a_section_absent_at_the_pin_is_refused"),
                            ("e_no_dl56", dict(dl56=False), "e_am7a_dl56_row_absent_is_refused"),
                            ("e_clip", dict(clip=CLIP + b" "), "e_am7a_clipping_file_other_than_of_record_is_refused")):
        sw = lam_gate_sweep(label, **kw)
        got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=now), ("records",))
        check(name, got == "records:am7a_not_on_record", got)
    # entry prints the line the gate requires byte for byte (PL-27), with the decision date and position
    sw = lam_gate_sweep("e_entry")
    want = own_line(sw, stage="e2", lam=0.5)
    entry_args = SimpleNamespace(stage="e2", seed=42, profile="distill_80k", lambda_logit=0.5, alpha=None,
                                 ckpt_dir=want["ckpt_dir"], code_pin=sw.pin, am8a_report=None)
    with git_root(sw.repo):
        rc, out = quiet(pd.cmd_entry, entry_args)
    printed = next((ln for ln in out.splitlines() if ln.startswith("{")), "")
    line = json.loads(printed) if printed else {}
    got = run_kd(gate_ctx(sw, line, now=now)) if line else "no line"
    check("e_entry_prints_the_line_the_gate_requires", rc == 0 and printed.encode("utf-8") == pd.line_bytes(want)
          and "decision date 2026-10-19" in out and "launch-order position 2" in out and got == "PASS",
          f"rc={rc} {got} {out[-300:]}")
    sw = lam_gate_sweep("e_entry_repeat", values=(1, 0.5))
    rid2 = pd.run_id_of("e2", 42, 80000, 0.5, None, 2)
    with git_root(sw.repo):
        rc1, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "ckpt_dir": str(DIRS["ck"] / rid2)}))
    check("e_entry_refuses_a_repeat_without_its_am8a_report", rc1 == "repeat_report_missing", rc1)
    sw = lam_gate_sweep("e_entry_name")
    with git_root(sw.repo):
        rc3, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "ckpt_dir": str(DIRS["ck"] / "wrong_name")}))
        rc4, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "code_pin": "f" * 40}))
    check("e_entry_refuses_a_ckpt_dir_not_named_by_the_run_id_and_an_unknown_pin",
          rc3 == "run_id_mismatch" and rc4 == "code_pin_unavailable", f"{rc3} {rc4}")
    # C4 workflow c4-1 at entry: an attempt after a launched one, a not_launched start between them, names its report
    sw = gate_sweep("e_entry_after_nl")
    sw.add(1, sss.fin(sss.BASE_L, 0.42))
    sw.add(0.5, sss.course(sss.BASE_L + 600, 3000, val_at=()), stopped=True)
    sw.add(0.5, state="not_launched", launch=sss.BASE_L + 2000)
    sw.build()
    rid3 = pd.run_id_of("e2", 42, 80000, 0.5, None, 3)
    with git_root(sw.repo):
        rc5, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "code_pin": sw.pin,
                                                        "ckpt_dir": str(DIRS["ck"] / rid3)}))
    check("e_c4_1_entry_refuses_an_attempt_after_a_launched_one_and_a_not_launched_start_without_its_report",
          rc5 == "repeat_report_missing", rc5)
    # c4-3 at entry: a second code pin in the sweep (DL-89)
    sw = lam_gate_sweep("e_entry_pin")
    with git_root(sw.repo):
        rc6, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "code_pin": sss.template()[1]}))
    check("e_c4_3_entry_refuses_a_second_code_pin_in_the_sweep", rc6 == "code_pin_mismatch", rc6)
    # c4-17 at entry: the previous attempt without an outcome
    sw = gate_sweep("e_entry_unresolved")
    sw.add(1, sss.fin(sss.BASE_L, 0.42))
    sw.add(0.5, [], state="pending", launch=sss.BASE_L + 600)
    sw.build()
    rid2b = pd.run_id_of("e2", 42, 80000, 0.5, None, 2)
    with git_root(sw.repo):
        rc7, _ = quiet(pd.cmd_entry, SimpleNamespace(**{**vars(entry_args), "code_pin": sw.pin,
                                                        "ckpt_dir": str(DIRS["ck"] / rid2b)}))
    check("e_c4_17_entry_refuses_while_the_previous_attempt_has_no_outcome", rc7 == "previous_attempt_unresolved", rc7)
    # C4 workflow c4-4 (PL-12 at the gate): the schedule file changed after the first launch; a shallow clone
    sw = lam_gate_sweep("e_schedule_changed")
    sched = json.loads((sw.repo / ss.SCHEDULE_REL).read_text(encoding="utf-8"))
    sw.pin = sss.commit_files(sw.repo, {ss.SCHEDULE_REL: json.dumps(sched, indent=4) + "\n"},
                              "the schedule file written again after the first launch")
    got = run_kd(gate_ctx(sw, own_line(sw, stage="g"), now=now), ("records",))
    check("e_c4_4_schedule_changed_after_the_first_launch_is_refused_at_the_gate",
          got == "records:schedule_changed_after_first_launch", got)
    sw = lam_gate_sweep("e_shallow")
    ctx = gate_ctx(sw, own_line(sw, stage="g"), now=now)
    shallow = scratch("shallow") / "c"
    sss.git(shallow.parent, "clone", "-q", "--depth", "1", "--no-local", f"file://{sw.repo}", str(shallow))
    got = run_kd(dict(ctx, git_root=shallow), ("records",))
    check("e_c4_4_shallow_clone_is_refused_at_the_gate", got == "records:repository_shallow", got)
    # c4-17: the records command's own refusals (PL-28)
    sw = lam_gate_sweep("e_records_cmd")
    h = sw.commit("a records commit")

    def records_cmd(**over):
        a = {"records_commit": h, "out": str(scratch("rec_out") / "f"), **over}
        with git_root(sw.repo):
            rc, _ = quiet(pd.cmd_records, SimpleNamespace(**a))
        return rc
    full = scratch("rec_full")
    (full / "x.txt").write_text("x\n", encoding="utf-8")
    for name, over, want in (("a_malformed_commit", {"records_commit": "ABC"}, "records_commit_malformed"),
                             ("an_unknown_commit", {"records_commit": "e" * 40}, "records_commit_unavailable"),
                             ("a_folder_that_is_not_empty", {"out": str(full)}, "records_folder")):
        got = records_cmd(**over)
        check(f"e_c4_17_records_command_refuses_{name}", got == want, got)
    # a relative folder, from a working directory outside the repository: only the absolute-path rule refuses it
    # (inside the checkout the folder would resolve under the repository and be refused by that rule as well)
    elsewhere = scratch("rec_cwd")
    saved_cwd = os.getcwd()
    os.chdir(elsewhere)
    try:
        got = records_cmd(out="relative/f")
    finally:
        os.chdir(saved_cwd)
    check("e_c4_17_records_command_refuses_a_relative_folder",
          got == "records_folder" and not (elsewhere / "relative").exists(), got)
    sss.git(sw.repo, "checkout", "-q", "-b", "side_off_the_pin", sw.pin)
    sss.commit_files(sw.repo, {"side.txt": "a side commit\n"}, "a side commit off the pin")
    got = records_cmd()
    check("e_c4_17_records_command_refuses_a_head_that_is_no_ancestor", got == "pin_not_ancestor", got)
    sw = gate_sweep("e_records_dotdot")
    sw.add(1, sss.fin(sss.BASE_L, 0.42), report={"path": "../outside.md", "sha256": "0" * 64})
    sw.build()
    h = sw.head
    got = records_cmd()
    check("e_c4_17_records_command_refuses_a_report_path_outside_the_repository", got == "records_mismatch", got)
    # c4-16 (its residual): an output folder that cannot be written is a refusal (exit 2), never a traceback
    sw = lam_gate_sweep("e_records_unwritable")
    h = sw.commit("a records commit")
    blocker = scratch("rec_blocker") / "a_file"
    blocker.write_text("not a folder\n", encoding="utf-8")
    try:
        with git_root(sw.repo), contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = pd.main(["records", "--records-commit", h, "--out", str(blocker / "f")])
        got = (rc, "RESULT: REFUSED (io_error)" in buf.getvalue())
    except Exception as e:  # noqa: BLE001 - a traceback instead of exit 2 fails the case
        got = f"raised {type(e).__name__}"
    check("e_c4_16_records_output_that_cannot_be_written_exits_2", got == (2, True), got)


# ----------------------------------- F: kd_image, the arguments stage, teacher_identity, kd_dry_run
@contextlib.contextmanager
def env_var(name: str, value):
    saved = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


FP_OK = {**pd.FINGERPRINT_OF_RECORD, "imported": {"numpy": ["numpy"], "torch": ["torch"], "PIL": ["pillow"],
                                                  "torchvision": ["torchvision"]},
         "import mmseg.apis": True, "import ftfy": True}


def test_image_and_arguments() -> None:
    def kd(digest, fp, rc=0):
        calls = []

        def child(argv, tag, timeout, env=None):
            calls.append(([str(a) for a in argv], tag))
            return rc, fp, "stub output"
        with env_var("PLANTSEG_IMAGE_DIGEST", digest):
            return code_of(pd.stage_kd_image, {"child_json": child}), calls

    got, calls = kd(pd.KD_IMAGE_DIGEST, FP_OK)
    check("f_kd_image_of_record_passes_through_a_fingerprint_child", got == "PASS" and len(calls) == 1
          and calls[0][1] == "FINGERPRINT_JSON" and calls[0][0][-1] == "_fingerprint", f"{got} {calls}")
    check("f_digest_unset_is_refused", kd(None, FP_OK)[0] == "image_digest_missing")
    check("f_another_digest_is_refused", kd("sha256:" + "0" * 64, FP_OK)[0] == "image_digest_mismatch")
    check("f_fingerprint_child_failing_is_refused", kd(pd.KD_IMAGE_DIGEST, None, rc=1)[0] == "fingerprint_unavailable")
    bad = {"torch": {**FP_OK, "torch": "2.2.0+cu121"}, "numpy": {**FP_OK, "numpy": "2.0.0"},
           "python": {**FP_OK, "python": "3.11.17"}, "mmcv": {**FP_OK, "mmcv": None},
           "nvidia": {**FP_OK, "nvidia": ["nvidia-cudnn-cu12==8.9.2.26"]},
           "unrecorded_module": {**FP_OK, "imported": {**FP_OK["imported"], "yaml": ["PyYAML"]}},
           "mmseg_apis": {**FP_OK, "import mmseg.apis": "ModuleNotFoundError: No module named 'mmseg'"},
           "no_imports": {**FP_OK, "imported": {}}}
    res = {k: kd(pd.KD_IMAGE_DIGEST, fp)[0] for k, fp in bad.items()}
    check("pl39_35_fingerprint_mismatch_is_refused", all(v == "fingerprint_mismatch" for v in res.values()), res)
    here = pd.fingerprint()
    probs = pd.fingerprint_problems(here)
    check("f_this_venvs_fingerprint_is_not_the_kd_images_and_its_imports_have_values_of_record",
          here.get("mmcv") is None and any(p.startswith("mmcv:") for p in probs)
          and not any(p.startswith("module ") for p in probs) and {"torch", "numpy"} <= set(here.get("imported") or {}),
          f"{probs} {here.get('imported')}")
    # the arguments stage's KD part (E1's own part is stubbed: it reads the real data root and PYTHONPATH)
    rid = "e2_s42_lambda1_80k_a1"
    ev, rec_dir = scratch("args_ev"), scratch("args_records")
    base = {"stage": "e2", "seed": 42, "profile": "distill_80k", "lambda_logit": 1.0, "alpha": None, "attempt": 1,
            "ckpt_dir": str(DIRS["ck"] / rid), "root": DIRS["data"], "evidence": str(ev), "records": str(rec_dir),
            "teacher_ckpt": TEACHER, "clock_offset": 0.0, "record": None}
    saved = E1G.stage_arguments
    E1G.stage_arguments = lambda ctx: ("PASS", "E1 arguments (stub)", {})
    try:
        def arg(**over):
            ctx = {**base, **over}
            return code_of(pd.stage_arguments, ctx), ctx
        got, ctx = arg()
        check("f_arguments_pass_and_name_the_run_id", got == "PASS" and ctx.get("run_id") == rid
              and (ctx.get("ident") or {}).get("sweep") == "lambda_logit", got)
        offsets = [(o, arg(clock_offset=o)[0]) for o in (60.0, -60.0, 61.0, -61.0, math.nan, math.inf)]
        check("pl39_35_clock_offset_above_60_seconds_is_refused", [c for _, c in offsets]
              == ["PASS", "PASS", "clock_offset", "clock_offset", "clock_offset", "clock_offset"], offsets)
        (ev / f"{rid}.stdout.log").write_text("an earlier launch\n", encoding="utf-8")
        exists = arg()[0]
        (ev / f"{rid}.stdout.log").unlink()
        cases = {"evidence_relative": arg(evidence="kd_evidence")[0] == "evidence_relative",
                 "evidence_missing": arg(evidence=str(ev / "absent"))[0] == "evidence_missing",
                 "evidence_holding_the_ckpt_dir": arg(evidence=str(DIRS["ck"]))[0] == "evidence_misplaced",
                 "evidence_in_the_data_root": arg(evidence=str(DIRS["data"]))[0] == "evidence_misplaced",
                 "stdout_file_exists": exists == "evidence_exists",
                 "records_relative": arg(records="kd_records")[0] == "records_folder",
                 "records_missing": arg(records=str(rec_dir / "absent"))[0] == "records_folder",
                 "teacher_other_name": arg(teacher_ckpt="/workspace/teacher/iter_20000.pth")[0] == "teacher_ckpt_path",
                 "teacher_relative": arg(teacher_ckpt="iter_24000.pth")[0] == "teacher_ckpt_path",
                 "ckpt_dir_not_the_run_id": arg(ckpt_dir=str(DIRS["ck"] / "e2_s42_lambda1_80k_a2"))[0]
                 == "run_id_mismatch",
                 "attempt_0": arg(attempt=0)[0] == "attempt",
                 "lambda_on_g": arg(stage="g", ckpt_dir=str(DIRS["ck"] / "g_s42_80k_a1"))[0] == "lambda_argument",
                 "seed_45": arg(seed=45)[0] == "seed"}
        check("f_arguments_refusals", all(cases.values()), [k for k, v in cases.items() if not v])
    finally:
        E1G.stage_arguments = saved
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            pd.build_parser().parse_args(["gate", "--stage", "g", "--seed", "42", "--ckpt-dir", "/a/b", "--evidence",
                                          "/e", "--expect-head", "c" * 40, "--records-commit", "d" * 40, "--records",
                                          "/r", "--teacher-ckpt", TEACHER])
            parsed = 0
        except SystemExit as e:
            parsed = e.code
    check("f_gate_requires_the_clock_offset_argument", parsed == 2, parsed)
    # teacher_identity: a CPU child; its record against the teacher of record
    prov = {"builder": pd.TEACHER_BUILDER, "ckpt_path": TEACHER, "ckpt_sha256": pd.TEACHER_SHA256,
            "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256,
            "config_path": str(REPO / "configs" / "teacher" / "segnext_mscan_b.py"), **pd.TEACHER_VALUES}
    good = {"ok": True, "sha256": pd.TEACHER_SHA256, "bytes": pd.TEACHER_BYTES, "provenance": prov,
            "params": pd.TEACHER_PARAMS, "trainable": 0, "training_modules": [], "state_entries": 854,
            "cuda_initialized": False}

    def ti(rec, *, rehearsal=False):
        envs = []

        def child(argv, tag, timeout, env=None):
            envs.append(env or {})
            return (0 if rec else 1), rec, "stub output"
        return code_of(pd.stage_teacher_identity, {"teacher_ckpt": TEACHER, "rehearsal": rehearsal,
                                                   "child_json": child}), envs
    got, envs = ti(good)
    check("f_teacher_of_record_passes_in_a_cpu_child", got == "PASS" and envs and envs[0].get("CUDA_VISIBLE_DEVICES") == "",
          got)
    muts = {"sha256": "0" * 64, "bytes": 1, "params": 1, "trainable": 5, "training_modules": ["decode_head"],
            "state_entries": 853}
    pmuts = {**{k: "x" for k in pd.TEACHER_VALUES}, "builder": "other", "ckpt_sha256": "0" * 64,
             "expected_sha256": "0" * 64, "ckpt_bytes": 1, "ckpt_path": "/elsewhere/iter_24000.pth",
             "config_path": "/etc/segnext.py"}
    res = {k: ti({**good, k: v})[0] for k, v in muts.items()}
    res.update({f"provenance.{k}": ti({**good, "provenance": {**prov, k: v}})[0] for k, v in pmuts.items()})
    res["not_loaded"] = ti({"ok": False, "error": "TeacherChecksumMismatch: synthetic"})[0]
    check("f_teacher_differing_from_the_teacher_of_record_is_refused",
          all(v == "teacher_identity_mismatch" for v in res.values()), res)
    check("f_teacher_child_without_a_record_is_refused", ti(None)[0] == "teacher_identity_failed")
    # kd_dry_run: the launch's values and the teacher of record, in a temporary folder removed afterwards

    def dr(rc=0, line="RESULT: PASS", ckpt=None, rehearsal=False):
        seen = {}

        def child(argv, timeout):
            a = [str(x) for x in argv]
            tmp = a[a.index("--ckpt-dir") + 1]
            seen.update(argv=a, tmp=tmp, existed=Path(tmp).is_dir())
            return SimpleNamespace(returncode=rc, stdout=f"[ckpt] dir={ckpt or tmp} (verified OUTSIDE repo)\n{line}\n",
                                   stderr="")
        ctx = {"rehearsal": rehearsal, "teacher_ckpt": TEACHER, "stage": "e3", "seed": 42, "lambda_logit": 1.0,
               "alpha": 25.0, "profile": "distill_160k", "root": DIRS["data"], "child_run": child}
        return code_of(pd.stage_kd_dry_run, ctx), seen
    got, seen = dr()
    a = seen.get("argv", [])

    def val(flag):
        return a[a.index(flag) + 1] if flag in a else None
    check("f_kd_dry_run_runs_the_launchs_values_with_the_teacher_of_record", got == "PASS" and "--dry-run" in a
          and (val("--stage"), val("--seed"), val("--lambda-logit"), val("--alpha"), val("--teacher-ckpt"),
               val("--teacher-ckpt-sha256"), val("--iterations")) == ("e3", "42", "1", "25", TEACHER, pd.TEACHER_SHA256,
                                                                      "160000")
          and seen.get("existed") and not Path(seen["tmp"]).exists() and "--real-run" not in a, f"{got} {a}")
    check("f_kd_dry_run_failing_child_is_refused", [dr(rc=1)[0], dr(line="RESULT: FAIL")[0], dr(ckpt="/elsewhere")[0]]
          == ["kd_dry_run_failed"] * 3)
    check("f_rehearsal_without_the_teacher_file_skips_teacher_identity_and_kd_dry_run",
          dr(rehearsal=True)[0] == "SKIPPED" and ti(good, rehearsal=True)[0] == "SKIPPED")


# ----------------------------------------------------------------------------- G: check-run-meta
class _Stop(Exception):
    pass


class _Tiny(torch.nn.Module):
    used_pretrained = False

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


class _Loader:
    dataset = [0] * 6

    def __len__(self):
        return 3


def trainer_row(stage: str):
    """The REAL train_distill.run() in dry mode up to its training loop (smoke_distill_realrun_gates' method): a tiny
    student, length-only loaders, a fixed git_head and a no-op set_seed; the loop's first call stops it. Its one
    run_meta row, or the reason it has none."""
    ck = scratch("meta") / f"{stage}_s42_meta"
    saved = (td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed)

    def stop(*_a, **_k):
        raise _Stop()
    td.build_student, td.build_dataloader = (lambda *a, **k: _Tiny()), (lambda *a, **k: _Loader())
    td.cycle, td._git_provenance, td.set_seed = stop, (lambda: ("0" * 40, "smoke_stub")), (lambda s: None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            td.run(stage=td.resolve_stage(stage), mode="dry", device="cpu", pretrained=False,
                   teacher=FrozenTeacher(MockTeacher(116)), lambda_logit=1.0 if pd.TERMS[stage][0] else None,
                   batch_size=2, max_iters=2, val_interval=2, max_val_batches=1, num_workers=0, ckpt_dir_arg=str(ck),
                   grad_clip_norm=None, log_every=1, seed=42)
        return "run() did not reach its training loop"
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001 - the reason is the check's detail
        return f"{type(e).__name__}: {e}"
    finally:
        td.build_student, td.build_dataloader, td.cycle, td._git_provenance, td.set_seed = saved
    lines = (ck / f"{stage}_run_meta.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]) if len(lines) == 1 else f"{len(lines)} run_meta lines"


PIN = "c" * 40
PINNED = ("event", "stage", "seed", "terms", "projection_params", "logit_kd_semantics", "logit_kd_semantics_declared",
          "logit_kd_semantics_override_used", "logit_kd_grid", "T_logit", "lambda_sweep_grid", "alpha_offgrid",
          "alpha_grid", "cwd_feat_grid", "T_cwd", "cwd_C", "beta_cwd", "cwd_logit_grid", "supervised_grid",
          "num_classes", "learning_rate", "momentum", "weight_decay", "lr_power", "poly_horizon", "grad_clip_norm",
          "ignore_index", "class_weights_sha256", "arm", "descriptive", "parent_of_e4_e7", "selection_sha256",
          "selection_files", "records_commit")


def go_record(stage: str, *, lam=None, alpha=None, profile="distill_80k", sel=None) -> dict:
    """A GO gate record (kd_gate_record/1) as check-run-meta reads it."""
    ident = pd.launch_identity(stage, 42, profile, lam, alpha)
    rid = pd.run_id_of(stage, 42, ident["horizon"], lam, alpha, 1)
    swept = ident["sweep"] is not None
    return {"format": "kd_gate_record/1", "verdict": "GO", "profile": profile, "stage": stage, "seed": 42,
            "run_id": rid, "ckpt_dir": str(DIRS["ck"] / rid), "pin": PIN, "records_commit": H_FAKE,
            "lambda_logit": lam, "alpha": alpha, "selection_files": sel or {"lambda_logit": None, "alpha_cwd": None},
            "teacher_ckpt": TEACHER, "decision_date": "2026-10-19" if swept else None,
            "launch_order_position": 2 if swept else None}


def expect(rec: dict) -> dict:
    p = scratch("gate_record") / "gate_record.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return pd.run_meta_expectations(SimpleNamespace(allow_smoke=False, gate_record=str(p)))


def good_row(stage: str, exp: dict) -> dict:
    """A run_meta row that meets every rule for `exp`, in the trainer's key order."""
    lk, cf, cl = pd.TERMS[stage]
    row = {"event": "run_meta", "stage": pd.STAGE_NAMES[stage], "mode": exp["mode"], "seed": exp["seed"],
           "terms": {"logit_kd": lk, "cwd_feat": cf, "cwd_logit": cl}, "projection_params": pd.PROJECTION_PARAMS if cf else 0}
    if lk:
        row.update(lambda_logit=exp["lambda_logit"], logit_kd_semantics=td.LOGIT_KD_SEMANTICS,
                   logit_kd_semantics_declared=None, logit_kd_semantics_override_used=False,
                   logit_kd_grid=pd.LOGIT_KD_GRID, T_logit=td.T_LOGIT, lambda_sweep_grid=list(td.LAMBDA_SWEEP))
    if cf:
        row.update(alpha_cwd=exp["alpha"], alpha_offgrid=False, alpha_grid=list(td.ALPHA_GRID),
                   cwd_feat_grid=pd.CWD_FEAT_GRID, cwd_C=td.CWD_C_FEAT)
    if cf or cl:
        row["T_cwd"] = td.T_CWD
    if cl:
        row.update(beta_cwd=td.BETA_CWD_LOGIT, cwd_logit_grid=pd.CWD_LOGIT_GRID)
    prov = {"builder": pd.TEACHER_BUILDER, "ckpt_path": exp["teacher_ckpt"] or TEACHER,
            "ckpt_sha256": pd.TEACHER_SHA256, "ckpt_bytes": pd.TEACHER_BYTES, "expected_sha256": pd.TEACHER_SHA256,
            "config_path": str(REPO / "configs" / "teacher" / "segnext_mscan_b.py"), **pd.TEACHER_VALUES}
    row.update({"supervised_grid": pd.SUPERVISED_GRID, "batch_size": 16, "max_iters": exp["max_iters"],
                "num_classes": 116, "teacher_nmf": {"MD_S": 1}, "wall_clock": 1791500000.25, "git_head": exp["pin"],
                "git_head_source": "git_checkout", "image_digest": pd.KD_IMAGE_DIGEST,
                "torch": pd.FINGERPRINT_OF_RECORD["torch"], "numpy": pd.FINGERPRINT_OF_RECORD["numpy"],
                "device": "cuda", "cuda_available": True, "gpu_name": pd.EXPECTED_GPU_NAME,
                "num_workers": pd.NUM_WORKERS, "persistent_workers": True, "val_interval": 4000, "max_val_batches": None,
                **{k: E1G.RUN_META_EXPECT[k] for k in ("learning_rate", "momentum", "weight_decay", "lr_power")},
                "poly_horizon": exp["horizon"], "grad_clip_norm": None, "used_pretrained": exp["used_pretrained"],
                "params": pd.STUDENT_PARAMS, "ignore_index": 255, "ramp_iters": pd.RAMP_ITERS,
                "class_weights_sha256": pd.CLASS_WEIGHTS_SHA256, "tf32": dict(td.TF32_DEFAULTS),
                "teacher_provenance": prov, "teacher_mock": False,
                "arm": pd.STAGE_NAMES[stage] if stage in pd.ARMS else None,
                "descriptive": stage in pd.ARMS or exp["horizon"] != 80000,
                "parent_of_e4_e7": stage == "e3" and exp["horizon"] == 80000,
                "selection_sha256": exp["selection_sha256"], "selection_files": exp["selection_files"],
                "records_commit": exp["records_commit"]})
    return {k: row[k] for k in pd.run_meta_keys(stage)}


def fails(stage: str, exp: dict, row: dict, key: str) -> bool:
    """Whether check-run-meta reports `key` (or the key set, for key "keys") as a problem of `row`."""
    p, _ = pd.meta_problems(row, stage, exp, td)
    return any(x.startswith("keys:") if key == "keys" else x.startswith(f"{key} (") for x in p)


def test_check_run_meta() -> None:
    folder = f"/workspace/kd_records/{H_FAKE[:12]}"
    lsel = {"path": f"{folder}/{ss.LAMBDA_SELECTION_REL}", "sha256": "1" * 64}
    asel = {"path": f"{folder}/{ss.ALPHA_SELECTION_REL}", "sha256": "2" * 64}
    acut = {"path": f"{folder}/{ss.DECISION_RECORD_REL['alpha_cwd']}", "sha256": "3" * 64}
    recs = {"e2": go_record("e2", lam=0.5),
            "e3": go_record("e3", lam=1.0, alpha=25.0, sel={"lambda_logit": lsel, "alpha_cwd": None}),
            "a": go_record("a", alpha=50.0, sel={"lambda_logit": None, "alpha_cwd": acut}),
            "f": go_record("f", alpha=100.0, sel={"lambda_logit": None, "alpha_cwd": asel}),
            "g": go_record("g"),
            "e3_160k": go_record("e3", lam=1.0, alpha=50.0, profile="distill_160k",
                                 sel={"lambda_logit": lsel, "alpha_cwd": asel})}
    # PL-30: the gate's key list per stage is the trainer's own, and every key has exactly one rule
    for stage in ("e2", "e3", "a", "f", "g"):
        row = trainer_row(stage)
        smoke_exp = pd.run_meta_expectations(SimpleNamespace(
            allow_smoke=True, profile="distill_80k", stage=stage, seed=42, expect_head="0" * 40,
            lambda_logit=1.0 if pd.TERMS[stage][0] else None, alpha=50.0 if pd.TERMS[stage][1] else None,
            ckpt_dir=f"/x/{stage}_s42_meta"))
        rules = pd.meta_rules(stage, smoke_exp, td)
        check(f"g_run_meta_keys_of_{stage}_are_the_trainers_and_each_has_one_rule",
              isinstance(row, dict) and list(row) == pd.run_meta_keys(stage) and sorted(rules) == sorted(row),
              row if not isinstance(row, dict) else sorted(set(row) ^ set(pd.run_meta_keys(stage))))
        bad = [k for k in PINNED if isinstance(row, dict) and k in row and not rules[k][1](row[k])]
        check(f"g_values_of_record_are_what_the_pinned_trainer_writes_{stage}", isinstance(row, dict) and not bad,
              {k: row[k] for k in bad} if isinstance(row, dict) else row)
    for name, rec in recs.items():
        stage = rec["stage"]
        exp = expect(rec)
        p, w = pd.meta_problems(good_row(stage, exp), stage, exp, td)
        check(f"g_good_{name}_row_passes", not p and not w, (p, w))
    # PL-39 case 36: one failing mutation per rule, for every key of E3 (all three terms) and of G
    for stage in ("e3", "g"):
        exp = expect(recs[stage])
        good, rules = good_row(stage, exp), pd.meta_rules(stage, exp, td)
        missed = []
        for k in pd.run_meta_keys(stage):
            p, w = pd.meta_problems({**good, k: f"__mutated_{k}__"}, stage, exp, td)
            hit = (any(x.startswith(f"{k} ") for x in w) and not p) if rules[k][0] == "warning" else \
                any(x.startswith(f"{k} (") for x in p)
            if not hit:
                missed.append(k)
        check(f"pl39_36_one_failing_mutation_per_rule_of_{stage}", not missed, missed)
    exp3 = expect(recs["e3"])
    good3 = good_row("e3", exp3)

    def prov(**kw):
        return {**good3, "teacher_provenance": {**good3["teacher_provenance"], **kw}}
    named = {"clip_100": ({**good3, "grad_clip_norm": 100.0}, "grad_clip_norm"),
             "persistent_workers_false": ({**good3, "persistent_workers": False}, "persistent_workers"),
             "num_workers_11": ({**good3, "num_workers": 11}, "num_workers"),
             "matmul_tf32_on": ({**good3, "tf32": {**good3["tf32"], "matmul_allow_tf32": True}}, "tf32"),
             "the_160k_horizon_under_80k": ({**good3, "poly_horizon": 160000}, "poly_horizon"),
             "max_iters_160k_under_80k": ({**good3, "max_iters": 160000}, "max_iters"),
             "expected_sha256_other_than_ckpt_sha256": (prov(expected_sha256="0" * 64), "teacher_provenance"),
             **{f"teacher_{k}": (prov(**{k: "x"}), "teacher_provenance") for k in pd.TEACHER_VALUES},
             "teacher_builder": (prov(builder="other"), "teacher_provenance"),
             "teacher_ckpt_path_basename": (prov(ckpt_path="/workspace/teacher/iter_20000.pth"), "teacher_provenance"),
             "teacher_ckpt_path_other_than_the_gates": (prov(ckpt_path="/elsewhere/iter_24000.pth"), "teacher_provenance"),
             "teacher_ckpt_bytes": (prov(ckpt_bytes=1), "teacher_provenance"),
             "teacher_config_path_outside_the_checkout": (prov(config_path="/etc/segnext.py"), "teacher_provenance"),
             "teacher_provenance_without_ham_kwargs": ({**good3, "teacher_provenance": {
                 k: v for k, v in good3["teacher_provenance"].items() if k != "ham_kwargs"}}, "teacher_provenance"),
             "image_digest": ({**good3, "image_digest": "sha256:" + "0" * 64}, "image_digest"),
             "git_head_other_than_the_pin": ({**good3, "git_head": "d" * 40}, "git_head"),
             "records_commit_other_than_the_gates": ({**good3, "records_commit": "e" * 40}, "records_commit"),
             "selection_sha256_other_than_the_records_copy": ({**good3, "selection_sha256": {
                 **good3["selection_sha256"], "lambda_logit": "9" * 64}}, "selection_sha256"),
             "selection_files_other": ({**good3, "selection_files": {**good3["selection_files"],
                                                                     "lambda_logit": "/elsewhere.json"}}, "selection_files"),
             "lambda_other_than_the_gates": ({**good3, "lambda_logit": 0.5}, "lambda_logit"),
             "lambda_as_an_int": ({**good3, "lambda_logit": 1}, "lambda_logit"),
             "alpha_other_than_the_gates": ({**good3, "alpha_cwd": 50.0}, "alpha_cwd"),
             "torch_version": ({**good3, "torch": "2.2.0+cu121"}, "torch"),
             "numpy_version": ({**good3, "numpy": "2.0.0"}, "numpy"),
             "mode_dry_under_a_go_record": ({**good3, "mode": "dry"}, "mode"),
             "used_pretrained_false": ({**good3, "used_pretrained": False}, "used_pretrained"),
             "teacher_mock_true": ({**good3, "teacher_mock": True}, "teacher_mock"),
             "semantics_declared": ({**good3, "logit_kd_semantics_declared": td.LOGIT_KD_SEMANTICS},
                                    "logit_kd_semantics_declared"),
             "an_extra_key": ({**good3, "extra": 1}, "keys"),
             "a_missing_key": ({k: v for k, v in good3.items() if k != "tf32"}, "keys")}
    missed = [k for k, (row, key) in named.items() if not fails("e3", exp3, row, key)]
    check("g_named_mutations_of_an_e3_row_fail", not missed, missed)
    expg, expa = expect(recs["g"]), expect(recs["a"])
    goodg, gooda = good_row("g", expg), good_row("a", expa)
    check("g_alpha_on_g_and_g_projection_params_fail", fails("g", expg, {**goodg, "alpha_cwd": 50.0}, "keys")
          and fails("g", expg, {**goodg, "projection_params": pd.PROJECTION_PARAMS}, "projection_params"))
    check("g_alpha_other_than_50_under_the_alpha_cut_record_fails", fails("a", expa, {**gooda, "alpha_cwd": 25.0},
                                                                         "alpha_cwd"))
    exp160 = expect(recs["e3_160k"])
    good160 = good_row("e3", exp160)
    check("g_e3_160k_is_descriptive_and_not_the_parent", good160["descriptive"] is True
          and good160["parent_of_e4_e7"] is False and good160["poly_horizon"] == good160["max_iters"] == 160000
          and fails("e3", exp160, {**good160, "parent_of_e4_e7": True}, "parent_of_e4_e7"))
    # the command: the launched line on PASS and on FAIL (PL-13, PL-30)
    rec = recs["e2"]
    gr = scratch("gr") / "gate_record.json"
    gr.write_text(json.dumps(rec), encoding="utf-8")
    d = Path(rec["ckpt_dir"])
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "e2_run_meta.jsonl"
    row = good_row("e2", expect(rec))
    meta.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cli = SimpleNamespace(allow_smoke=False, gate_record=str(gr), ckpt_dir=str(d))
    rc, out = quiet(pd.check_run_meta, cli)
    head = "LAUNCHED LINE (append to reports/kd_launch_log.jsonl as one line, commit and push it):\n"
    launched = json.loads(out.split(head)[1].splitlines()[0]) if head in out else {}
    own = pd.launch_line(ss, stage="e2", seed=42, profile="distill_80k", lam=0.5, alpha=None, attempt=1, schedule="T",
                         code_pin=PIN, ckpt_dir=rec["ckpt_dir"], am8a_report=None)
    try:
        parsed = ss.LaunchLog(pd.line_bytes(own) + b"\n" + json.dumps(launched).encode("utf-8") + b"\n")
        logged = d.name in parsed.launched
    except ss.SelectionRefused as e:
        logged = str(e)
    check("g_cli_pass_prints_the_launched_line_the_log_accepts", rc == 0 and "RESULT: PASS" in out and launched == {
        "event": "launched", "run_id": d.name, "run_meta_sha256": sha(meta.read_bytes()),
        "launch_time_utc": row["wall_clock"], "git_head": PIN, "records_commit": H_FAKE, "decision_date": "2026-10-19",
        "launch_order_position": 2} and logged is True, f"rc={rc} {launched} {logged} {out[-300:]}")
    meta.write_text(json.dumps({**row, "num_workers": 11}) + "\n", encoding="utf-8")
    rc, out = quiet(pd.check_run_meta, cli)
    check("g_cli_fail_still_prints_the_launched_line", rc == 1 and head in out and "RESULT: FAIL" in out
          and "MISMATCH: num_workers" in out, out[-300:])
    meta.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    rc, out = quiet(pd.check_run_meta, cli)
    check("g_cli_two_rows_fail_without_a_launched_line", rc == 1 and "run_meta rows: 2" in out
          and "LAUNCHED LINE: none" in out, out[-300:])
    smoke_args = SimpleNamespace(allow_smoke=True, gate_record=None, profile="distill_80k", stage="e2", seed=42,
                                 expect_head=PIN, lambda_logit=0.5, alpha=None, ckpt_dir=str(scratch("dry") / "dry_e2"))
    dry_row = good_row("e2", pd.run_meta_expectations(smoke_args))
    Path(smoke_args.ckpt_dir).mkdir()
    (Path(smoke_args.ckpt_dir) / "e2_run_meta.jsonl").write_text(json.dumps(dry_row) + "\n", encoding="utf-8")
    rc_smoke, _ = quiet(pd.check_run_meta, smoke_args)
    meta.write_text(json.dumps(dry_row) + "\n", encoding="utf-8")
    rc_go, out = quiet(pd.check_run_meta, cli)
    check("g_the_dry_row_passes_only_with_allow_smoke", dry_row["mode"] == "dry" and dry_row["max_iters"] == 25
          and dry_row["used_pretrained"] is False and rc_smoke == 0 and rc_go == 1, f"{rc_smoke} {rc_go}")
    with contextlib.redirect_stderr(io.StringIO()):
        both = pd.main(["check-run-meta", "--ckpt-dir", str(d), "--allow-smoke", "--gate-record", str(gr)])
        bare = pd.main(["check-run-meta", "--ckpt-dir", str(d), "--allow-smoke"])
    check("g_cli_needs_a_gate_record_or_allow_smoke_with_its_arguments", both == 2 and bare == 2, f"{both} {bare}")
    # C4 workflow c4-16: an unreadable or partial gate record is a refusal (exit 2), never a FAIL (exit 1)
    partial_p = gr.parent / "partial_gate_record.json"
    partial_p.write_text(json.dumps({k: v for k, v in rec.items() if k != "selection_files"}), encoding="utf-8")
    exits = []
    for p in (gr.parent / "absent.json", partial_p):
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                exits.append(pd.main(["check-run-meta", "--ckpt-dir", str(d), "--gate-record", str(p)]))
        except Exception as e:  # noqa: BLE001 - a traceback instead of exit 2 fails the case
            exits.append(f"raised {type(e).__name__}")
    check("g_c4_16_unreadable_or_partial_gate_record_exits_2", exits == [2, 2], exits)
    # round-5 c4r3-3: a GO record whose ckpt_dir is not an absolute path (a number, a bare name) or whose stage is
    # not one of this gate's is a refusal (exit 2) as gate_record, never a traceback; round-6 c4r4-1: the code is
    # asserted, since the older folder test refuses the bare name with exit 2 too (ckpt_dir_not_the_runs)
    exits = []
    for i, bad in enumerate(({**rec, "ckpt_dir": 5}, {**rec, "ckpt_dir": d.name}, {**rec, "stage": "zz"})):
        bad_p = gr.parent / f"bad_gate_record_{i}.json"
        bad_p.write_text(json.dumps(bad), encoding="utf-8")
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as out_b:
                rc_b = pd.main(["check-run-meta", "--ckpt-dir", str(d), "--gate-record", str(bad_p)])
            exits.append((rc_b, "RESULT: REFUSED (gate_record)" in out_b.getvalue()))
        except Exception as e:  # noqa: BLE001 - a traceback instead of exit 2 fails the case
            exits.append(f"raised {type(e).__name__}")
    check("g_c4r3_3_a_gate_record_with_a_ckpt_dir_or_stage_of_another_form_exits_2", exits == [(2, True)] * 3, exits)
    # c4-18: a --ckpt-dir other than the gate's (the same name in another folder, or another name) is a usage
    # refusal (exit 2) that prints no launched line
    meta.write_text(json.dumps(row) + "\n", encoding="utf-8")
    other = scratch("other_run") / d.name
    other.mkdir(parents=True)
    (other / "e2_run_meta.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rc, out = quiet(pd.check_run_meta, SimpleNamespace(allow_smoke=False, gate_record=str(gr), ckpt_dir=str(other)))
    rc_name, out_name = quiet(pd.check_run_meta, SimpleNamespace(allow_smoke=False, gate_record=str(gr),
                                                                 ckpt_dir=str(d.parent / "another_name")))
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as buf:
        exit_other = pd.main(["check-run-meta", "--ckpt-dir", str(other), "--gate-record", str(gr)])
    check("g_c4_18_a_ckpt_dir_other_than_the_gates_is_refused_without_a_launched_line",
          rc == "ckpt_dir_not_the_runs" and rc_name == "ckpt_dir_not_the_runs" and exit_other == 2
          and head not in out + out_name + buf.getvalue(), f"{rc} {rc_name} {exit_other} {(out + out_name)[-300:]}")
    # round-4 c4r2-1: a ckpt_dir typed with a trailing slash, an inner '//' or '/./' (at entry and the gate, so in
    # the gate record, or at check-run-meta) is the same directory: check-run-meta passes and prints the launched line
    outs = []
    for i, spelled in enumerate((rec["ckpt_dir"] + "/", f"{d.parent}//{d.name}", f"{d.parent}/./{d.name}")):
        gr_s = scratch(f"gr_spelled_{i}") / "gate_record.json"
        gr_s.write_text(json.dumps({**rec, "ckpt_dir": spelled}), encoding="utf-8")
        outs.append(quiet(pd.check_run_meta, SimpleNamespace(allow_smoke=False, gate_record=str(gr_s),
                                                             ckpt_dir=str(d))))
    outs.append(quiet(pd.check_run_meta, SimpleNamespace(allow_smoke=False, gate_record=str(gr),
                                                         ckpt_dir=str(d) + "/")))
    check("g_c4r2_1_a_ckpt_dir_spelled_with_a_trailing_slash_or_dots_is_the_same_directory",
          len(outs) == 4 and all(rc_s == 0 and head in out_s for rc_s, out_s in outs),
          [(rc_s, out_s[-160:]) for rc_s, out_s in outs])


# ---------------------------------------------------- H: push-evidence (CHECK ITEM 3; AM-19a reading 10)
def activity(after: str, ts: str, eid: int, kind: str = "push", before=None) -> dict:
    """An activity entry in the format of GitHub's response (Session 1's report, section 1)."""
    return {"id": eid, "node_id": f"PSH_{eid}", "before": before or "0" * 40, "after": after,
            "ref": "refs/heads/master", "timestamp": ts, "activity_type": kind,
            "actor": {"login": "k2-smoke", "id": 1, "type": "User"}}


def test_push_evidence() -> None:
    repo, pin = sss.new_repo(), sss.template()[1]
    lsel = json.dumps({"format": "lambda_selection/1", "winner": {"lambda": 1.0}, "inputs_last_timestamp_utc": sss.T_IN})
    added = sss.commit_files(repo, {ss.LAMBDA_SELECTION_REL: lsel}, "lambda selection")
    later = sss.commit_files(repo, {"reports/derived/later.json": "{}\n"}, "a later commit")
    remote = scratch("remote")
    sss.git(repo, "clone", "-q", "--bare", str(repo), str(remote))
    sss.git(repo, "remote", "add", "origin", str(remote))
    local_only = sss.commit_files(repo, {"reports/derived/local.json": "{}\n"}, "a commit the remote never received")
    other = scratch("other_clone")                           # another clone pushes a commit this clone lacks
    sss.git(other, "clone", "-q", str(remote), str(other / "c"))
    pushed = sss.commit_files(other / "c", {"reports/derived/pushed.json": "{}\n"}, "pushed from another clone")
    sss.git(other / "c", "push", "-q", "origin", "HEAD:refs/heads/side")
    t_lo = sss.T_IN - ss.PUSH_MARGIN_SECONDS                   # 2026-10-11T23:59:00Z
    entries = [activity(pin, "2026-10-11T00:00:00Z", 1),          # before T_lo: the response reaches back
               activity("e" * 40, "2026-10-12T02:00:00Z", 2),     # a commit the remote does not serve: counts
               activity("0" * 40, "2026-10-12T04:00:00Z", 3, "branch_deletion"),
               activity(added, "2026-10-12T05:00:00Z", 4),        # served, carries the file
               activity(pin, "2026-10-12T06:00:00Z", 5),          # served, does not carry it
               activity(later, "2026-10-12T07:00:00Z", 6),        # served, carries it
               activity(local_only, "2026-10-12T08:00:00Z", 7),   # held here, not served: counts
               activity(pushed, "2026-10-12T09:00:00Z", 8)]       # served, not held here: fetched, carries it
    response = json.dumps(entries).encode("utf-8")
    work = scratch("push")
    (work / "activity.json").write_bytes(response)

    def run(out: Path, remote_name="origin", data=response):
        (work / "response.json").write_bytes(data)
        with git_root(repo):
            return quiet(pd.cmd_push_evidence, SimpleNamespace(response=str(work / "response.json"), out=str(out),
                                                               remote=remote_name))
    out = work / "evidence"
    rc, text = run(out)
    resp_p, list_p = out / Path(ss.PUSH_EVIDENCE_REL).name, out / Path(ss.PUSH_LIST_REL).name
    doc = json.loads(list_p.read_text(encoding="utf-8")) if list_p.exists() else {}
    got = {x["id"]: (x["served"], x["carries_file"]) for x in doc.get("pushes", [])}
    answers = {x["id"]: x["remote_answer"] for x in doc.get("pushes", [])}
    check("h_push_evidence_writes_the_response_verbatim_and_the_list_with_the_remotes_answers",
          rc == 0 and resp_p.read_bytes() == response
          and got == {2: (False, True), 4: (True, True), 5: (True, False), 6: (True, True), 7: (False, True),
                      8: (True, True)}
          and "not our ref" in answers.get(2, "") and "not our ref" in answers.get(7, "")
          and doc.get("adding_commit") == added and doc.get("t_lo_utc") == t_lo and doc.get("remote") == str(remote)
          and "ONE commit" in text, f"rc={rc} {got} {text[-300:]}")
    sss.commit_files(repo, {ss.PUSH_EVIDENCE_REL: resp_p.read_bytes(), ss.PUSH_LIST_REL: list_p.read_bytes()},
                     "the push evidence and its list, together")
    try:
        ev = ss.push_evidence(ss.HeadSource(repo), t_lo, None)
    except ss.SelectionRefused as e:
        ev = e.code
    check("h_the_written_evidence_is_what_the_selection_accepts", isinstance(ev, dict)
          and ev["entry_id"] == 2 and ev["pushed_at_utc"] == sss.utc("2026-10-12T02:00:00Z")
          and [u["id"] for u in ev["unserved"]] == [2, 7], str(ev))
    rc, _ = run(out)
    check("h_push_evidence_is_written_once", rc == "push_evidence_out", rc)
    # round-4 c4r2-3: a write that fails leaves neither file (exit 2, io_error), so the step runs again; round-7
    # c4r5-2: the blocker, a stale temporary this call did not write and cannot remove, is named for a fresh --out
    half = work / "half_out"
    blocker = half / (Path(ss.PUSH_LIST_REL).name + ".tmp")
    blocker.mkdir(parents=True)                             # the list's temporary file cannot be written
    (work / "response.json").write_bytes(response)
    argv = ["push-evidence", "--response", str(work / "response.json"), "--out", str(half), "--remote", "origin"]
    try:
        with git_root(repo), contextlib.redirect_stderr(io.StringIO()) as err_h, \
                contextlib.redirect_stdout(io.StringIO()) as buf:
            rc_half = pd.main(argv)
        named = f"could not remove {[str(blocker)]}:" in err_h.getvalue()
        left = sorted(p.name for p in half.iterdir())
        blocker.rmdir()
        with git_root(repo), contextlib.redirect_stdout(io.StringIO()):
            rc_again = pd.main(argv)
        got = (rc_half, "RESULT: REFUSED (io_error)" in buf.getvalue(), named, left, rc_again,
               sorted(p.name for p in half.iterdir()))
    except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
        got = f"raised {type(e).__name__}: {e}"
    want = (2, True, True, [blocker.name], 0,
            sorted([Path(ss.PUSH_EVIDENCE_REL).name, Path(ss.PUSH_LIST_REL).name]))
    check("h_c4r2_3_a_write_that_fails_leaves_neither_file_and_the_step_runs_again", got == want, got)
    # round-5 c4r3-1: a rename that fails after the first one succeeded leaves neither file too (the cleanup removes
    # the target already renamed), and the step runs again into the same --out; c4r3-5: a file the cleanup cannot
    # remove is named, for a rerun into a fresh --out
    resp_name, list_name = Path(ss.PUSH_EVIDENCE_REL).name, Path(ss.PUSH_LIST_REL).name
    real_replace, real_unlink = Path.replace, Path.unlink

    def failing_replace(self, target):
        if Path(target).name == list_name:
            raise OSError(errno.EIO, "a synthetic rename failure")
        return real_replace(self, target)

    def failing_unlink(self, missing_ok=False):   # both targets: one remains, the other never was written
        if self.name in (resp_name, list_name):
            raise OSError(errno.EIO, "a synthetic unlink failure")
        return real_unlink(self, missing_ok=missing_ok)

    real_lstat = os.lstat

    def failing_lstat(path, *args, **kwargs):     # the response target in blind_out only (rmtree calls lstat too)
        if Path(path).name == resp_name and Path(path).parent.name == "blind_out":
            raise OSError(errno.EIO, "a synthetic lstat failure")
        return real_lstat(path, *args, **kwargs)

    def evidence_into(out_dir: Path, *, rename_fails=True, unlink_fails=False, lstat_fails=False) -> tuple:
        err, buf_e = io.StringIO(), io.StringIO()
        argv_e = ["push-evidence", "--response", str(work / "response.json"), "--out", str(out_dir), "--remote",
                  "origin"]
        if rename_fails:
            Path.replace = failing_replace
        if unlink_fails:
            Path.unlink = failing_unlink
        if lstat_fails:
            os.lstat = failing_lstat
        try:
            with git_root(repo), contextlib.redirect_stderr(err), contextlib.redirect_stdout(buf_e):
                rc_e = pd.main(argv_e)
        finally:
            Path.replace, Path.unlink, os.lstat = real_replace, real_unlink, real_lstat
        return rc_e, buf_e.getvalue(), err.getvalue(), sorted(p.name for p in out_dir.iterdir())
    (work / "response.json").write_bytes(response)
    try:
        rc_r, out_r, err_r, left_r = evidence_into(work / "rename_out")
        rc_r2, _, _, again_r = evidence_into(work / "rename_out", rename_fails=False)
        got = (rc_r, "RESULT: REFUSED (io_error)" in out_r, "could not remove" not in err_r, left_r, rc_r2, again_r)
    except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
        got = f"raised {type(e).__name__}: {e}"
    check("h_c4r3_1_a_rename_that_fails_leaves_neither_file_and_the_step_runs_again",
          got == (2, True, True, [], 0, sorted([resp_name, list_name])), got)
    # round-6 c4r4-3: the message names exactly what remains (the response renamed before the failure; not the
    # list, whose removal also failed but which was never written), and nothing when all is removed (above)
    try:
        rc_s, out_s, err_s, left_s = evidence_into(work / "stuck_out", unlink_fails=True)
        got = (rc_s, "RESULT: REFUSED (io_error)" in out_s,
               f"could not remove {[str(work / 'stuck_out' / resp_name)]}:" in err_s, "fresh --out" in err_s, left_s)
    except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
        got = f"raised {type(e).__name__}: {e}"
    check("h_c4r3_5_a_file_the_cleanup_cannot_remove_is_named_for_a_fresh_out",
          got == (2, True, True, True, [resp_name]), got)
    # round-7 c4r5-1: a path whose removal failed and whose lstat cannot tell whether it remains is named too (only a
    # path lstat reports absent is left out)
    try:
        rc_b, out_b, err_b, left_b = evidence_into(work / "blind_out", unlink_fails=True, lstat_fails=True)
        got = (rc_b, "RESULT: REFUSED (io_error)" in out_b,
               f"could not remove {[str(work / 'blind_out' / resp_name)]}:" in err_b, left_b)
    except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
        got = f"raised {type(e).__name__}: {e}"
    check("h_c4r5_1_a_path_whose_lstat_fails_is_still_named", got == (2, True, True, [resp_name]), got)
    rc, _ = run(work / "nowhere", remote_name="nowhere")
    check("h_push_evidence_without_the_remote_is_refused", rc == "push_evidence_remote", rc)
    sss.git(repo, "remote", "add", "broken", str(work / "no_such_remote.git"))
    rc, _ = run(work / "broken_out", remote_name="broken")
    check("h_a_remote_that_gives_no_answer_refuses_and_writes_nothing",
          rc == "push_evidence_remote" and not (work / "broken_out").exists(), rc)
    rc, _ = run(work / "short_out", data=json.dumps(entries[1:]).encode("utf-8"))
    check("h_a_response_not_reaching_back_to_t_lo_is_refused", rc == "lambda_push_evidence_mismatch"
          and not (work / "short_out").exists(), rc)

    def refusal(repo_root: Path, out: Path, data=response, remote_name="origin"):
        """(code, message) of push-evidence in `repo_root` ((0, '') when it writes)."""
        (work / "response.json").write_bytes(data)
        with git_root(repo_root), contextlib.redirect_stdout(io.StringIO()):
            try:
                return pd.cmd_push_evidence(SimpleNamespace(response=str(work / "response.json"), out=str(out),
                                                            remote=remote_name)), ""
            except pd.GateError as e:
                return e.code, str(e)
            except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
                return f"raised {type(e).__name__}", str(e)
    # C4 workflow c4-16: a --response that cannot be read is a refusal; nothing is written
    try:
        with git_root(repo):
            rc, _ = quiet(pd.cmd_push_evidence, SimpleNamespace(response=str(work / "missing_response.json"),
                                                                out=str(work / "out16"), remote="origin"))
    except Exception as e:  # noqa: BLE001 - an escape instead of a refusal fails the case
        rc = f"raised {type(e).__name__}"
    check("h_c4_16_an_unreadable_response_is_refused", rc == "push_evidence_response" and not (work / "out16").exists(),
          rc)
    # c4-7: no push after T_lo carries the lambda selection yet: refused, nothing written (the pair is settled once)
    rc, msg = refusal(repo, work / "out7", json.dumps([entries[0], activity(pin, "2026-10-12T06:00:00Z", 9)]).encode())
    check("h_c4_7_a_response_with_no_push_carrying_the_selection_is_refused", rc == "lambda_push_evidence_mismatch"
          and "carries" in msg and not (work / "out7").exists(), f"{rc} {msg[:200]}")
    # c4-9: a shallow clone is refused (its adding commit and ancestry would be wrong)
    shallow = scratch("push_shallow") / "c"
    sss.git(shallow.parent, "clone", "-q", "--depth", "1", "--no-local", f"file://{repo}", str(shallow))
    rc, msg = refusal(shallow, work / "out9")
    check("h_c4_9_a_shallow_clone_is_refused", rc == "repository_shallow" and not (work / "out9").exists(),
          f"{rc} {msg[:200]}")
    # c4-8: a remote URL with credentials: the list records it without them (the fetch uses it as given; the clone
    # holds every pushed commit here, so nothing is fetched); a remote's answer naming the URL keeps none either
    sss.git(repo, "remote", "add", "tokened", "https://user:secret-token@k2pf.invalid/plantseg.git")
    saved_serves = pd.remote_serves
    pd.remote_serves = lambda _ss, url, commit: (0, f"fetched {commit[:7]} (synthetic)")
    try:
        rc, msg = refusal(repo, work / "out8", json.dumps([entries[0], activity(added, "2026-10-12T05:00:00Z", 10),
                                                           activity(later, "2026-10-12T07:00:00Z", 11)]).encode(),
                          remote_name="tokened")
    finally:
        pd.remote_serves = saved_serves
    list8 = work / "out8" / Path(ss.PUSH_LIST_REL).name
    raw8 = list8.read_bytes() if list8.exists() else b"{}"
    check("h_c4_8_the_list_records_the_remote_without_its_credentials", rc == 0 and b"secret-token" not in raw8
          and json.loads(raw8).get("remote") == "https://k2pf.invalid/plantseg.git", f"{rc} {msg[:120]} {raw8[:160]}")
    saved_git = ss.HeadSource.git

    def fetch_says_the_url(self, *a):
        if a and a[0] == "fetch":
            return subprocess.CompletedProcess(list(a), 128, b"", f"fatal: unable to access '{a[-2]}/': 403\n".encode())
        return saved_git(self, *a)
    ss.HeadSource.git = fetch_says_the_url
    try:
        rc_rs, text_rs = pd.remote_serves(ss, "https://user:secret-token@k2pf.invalid/x.git", "e" * 40)
    finally:
        ss.HeadSource.git = saved_git
    check("h_c4_8_a_remote_answer_naming_the_url_keeps_no_credentials", rc_rs == 128 and "secret-token" not in text_rs
          and "k2pf.invalid" in text_rs, text_rs)
    # c4-17: push-evidence's other refusals
    no_ts = sss.new_repo()
    sss.commit_files(no_ts, {ss.LAMBDA_SELECTION_REL: json.dumps({"format": "lambda_selection/1",
                                                                   "winner": {"lambda": 1.0}})}, "no inputs timestamp")
    rc, msg = refusal(no_ts, work / "out17a")
    check("h_c4_17_a_lambda_selection_without_its_inputs_timestamp_is_refused", rc == "lambda_selection_format", rc)
    edited = sss.new_repo()
    sss.commit_files(edited, {ss.LAMBDA_SELECTION_REL: lsel}, "lambda selection")
    sss.commit_files(edited, {ss.LAMBDA_SELECTION_REL: lsel + "\n"}, "lambda selection edited after its commit")
    rc, msg = refusal(edited, work / "out17b")
    check("h_c4_17_no_commit_adds_the_lambda_selection_with_its_blob_is_refused",
          rc == "lambda_push_evidence_mismatch" and "no commit adds" in msg, f"{rc} {msg[:200]}")
    rc, msg = refusal(repo, work / "out17c", json.dumps([entries[0], activity("not-a-commit", "2026-10-12T05:00:00Z",
                                                                               12)]).encode())
    check("h_c4_17_a_push_whose_after_is_no_commit_id_is_refused",
          rc == "lambda_push_evidence_mismatch" and "is no commit id" in msg, f"{rc} {msg[:200]}")
    pd.remote_serves = lambda _ss, url, commit: (0, "")
    try:
        rc, msg = refusal(repo, work / "out17d", json.dumps([entries[0], activity("d" * 40, "2026-10-12T05:00:00Z",
                                                                                  13)]).encode())
    finally:
        pd.remote_serves = saved_serves
    check("h_c4_17_a_served_commit_that_cannot_be_fetched_is_refused",
          rc == "push_evidence_remote" and "could not be fetched" in msg, f"{rc} {msg[:200]}")


# ------------------------------------------------------------------------------- I: the verdict
def test_verdict() -> None:
    def args(**over):
        base = dict(rehearsal=False, record=str(scratch("verdict") / "gate_record.json"), profile="distill_80k",
                    stage="g", seed=42, lambda_logit=None, alpha=None, attempt=1, records_commit=H_FAKE,
                    evidence="/workspace/kd_evidence")
        return SimpleNamespace(**{**base, **over})

    def ctx():
        return {**block_ctx("g", 42), "warnings": [], "decision_date": None, "launch_order_position": None,
                "pin": PIN}

    def P(c):
        return "PASS", "ok", {}

    def F(c):
        raise pd.GateError("x_code", "a synthetic refusal")

    def S(c):
        return "SKIPPED", "skipped", {}

    def X(c):
        raise RuntimeError("a synthetic crash")

    def run(a, *fns):
        c = ctx()
        rc, out = quiet(pd.run_gate, a, stages=[(f"s{i}", fn) for i, fn in enumerate(fns)], ctx=c)
        rec = json.loads(Path(a.record).read_text(encoding="utf-8")) if a.record and Path(a.record).exists() else None
        return rc, out, rec, c

    a = args()
    rc, out, rec, c = run(a, P, P, P)
    block = out.split("----- launch block -----\n")[1].split("\n----- end launch block -----")[0] \
        if "----- launch block -----" in out else ""
    check("i_every_stage_passing_is_go_with_the_record_written_first_and_the_block", rc == 0 and "VERDICT: GO" in out
          and rec is not None and rec["verdict"] == "GO" and rec["launch_block"] == block == pd.build_launch_block(c)
          and out.index("record written") < out.index("VERDICT: GO"), out[-400:])
    rc, out, rec, _ = run(args(), P, F, P)
    check("i_a_failing_stage_is_no_go_naming_it_and_the_record_is_written", rc == 1
          and "first failing stage: s1 [x_code]" in out and rec is not None and rec["verdict"] == "NO-GO"
          and "----- launch block -----" not in out, out[-300:])
    rc, out, rec, _ = run(args(), F, P)
    check("i_a_failing_first_stage_writes_no_record", rc == 1 and rec is None, out[-200:])
    rc, out, rec, _ = run(args(rehearsal=True), P, P, P)
    check("i_a_rehearsal_is_never_a_go", rc == 1 and "NO-GO (rehearsal)" in out and "(launch block withheld)" in out
          and "----- launch block -----" not in out and rec is not None and rec["verdict"] == "NO-GO (rehearsal)",
          out[-300:])
    rc, out, _, _ = run(args(), P, S, P)
    check("i_a_skipped_stage_is_never_a_go", rc == 1 and "SKIPPED is never a GO" in out, out[-300:])
    rc, out, _, _ = run(args(), P, X)
    check("i_a_crash_is_a_fail", rc == 1 and "first failing stage: s1 [stage_error]" in out, out[-300:])
    with contextlib.redirect_stderr(io.StringIO()):
        rc, _, _, _ = run(args(record=None), P)
    check("i_a_real_gate_without_record_is_a_usage_error", rc == 2, rc)
    rc, out, _, _ = run(args(record=str(scratch("verdict") / "absent_folder" / "gate_record.json")), P, P)
    check("i_a_record_that_cannot_be_written_is_no_go", rc == 1 and "could not be written" in out, out[-300:])
    try:                                    # a NO-GO whose record cannot be written stays a NO-GO (exit 1)
        rc, out, _, _ = run(args(record=str(scratch("verdict") / "absent_folder" / "gate_record.json")), P, F)
    except Exception as e:  # noqa: BLE001 - an escape instead of exit 1 fails the case
        rc, out = f"raised {type(e).__name__}", ""
    check("i_a_no_go_whose_record_cannot_be_written_stays_a_no_go", rc == 1 and "VERDICT: NO-GO" in out
          and "could not be written" in out, f"{rc} {out[-300:]}")


# ----------------------------------------------------------------------------------------- main
SECTIONS = ("test_constants", "test_launch_block", "test_sources", "test_corrected", "test_launch_order",
            "test_records", "test_image_and_arguments", "test_check_run_meta", "test_push_evidence", "test_verdict")


def main(argv=None) -> int:
    """All sections, or those named with --section (repeatable; mutation runs). A section that raises fails as
    <section>_completed."""
    import traceback
    argv = sys.argv[1:] if argv is None else argv
    only = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--section"]
    print("=" * 78)
    print("KD LAUNCH GATE SMOKE (preflight_distill) -- CPU, synthetic scratch repositories, stubbed children")
    print(f"torch {torch.__version__}")
    print("=" * 78)
    DIRS.update(ck=scratch("ck"), ev=scratch("ev"), data=scratch("data"))
    try:
        for name in SECTIONS:
            if only and name not in only:
                continue
            try:
                globals()[name]()
            except (Exception, SystemExit) as e:  # noqa: BLE001 - reported as a failed check
                check(f"{name}_completed", False, f"{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}")
    finally:
        for p in sss._SCRATCH_DIRS:          # only directories this process created (scratch_dir)
            shutil.rmtree(p, ignore_errors=True)
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:72}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail and not ok else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
