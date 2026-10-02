#!/usr/bin/env python3
"""Lane 8 on synthetic data: scripts/run_ptq.py, src/quant/ptq.py and INT8 TorchScript loading.

No PlantSeg data, no real checkpoint, no GPU. A random-weight E1-schema student, a synthetic dataset in
a temp dir outside the repository (140 TRAIN, 846 VAL, 3 TEST tiny image/mask pairs) and calibration
lists built by scripts/build_calibration_lists.py stand in for the real inputs.

  1 gates        every refusal exits 2 before anything is written: flags, pins, output inside the
                 repository, VAL/TEST data paths, wrong source stage, a malformed list, a list id
                 outside TRAIN; AM-10: calibrate() and run_e4.py refuse a batch other than 1
  2 run          a full E4 run (its own process, as scripts/run_ptq.py always runs): the TorchScript
                 artifact of record, its state_dict and the x86 copy, six verification checks, the
                 calibration record; an audit hook shows VAL/TEST never opened or listed
  3 d3           a second run in another process and `run_ptq.py --compare`: bitwise-identical
                 artifacts; comparing a run with itself is refused
  4 evaluator    validate_int8_artifact + load_int8_student read the TorchScript artifact through its
                 relative path; scripts/evaluate_model.py scores it on synthetic VAL; tampering,
                 an unknown format, a wrong identity, a wrong engine and the x86 copy are refused; a
                 PTQ run in a process that already traced the student is refused (d3 bytes)
  5 official     only the registered AM-10 list, as committed in configs/calibration/, yields an
                 official E4/E7 score: synthetic and sensitivity lists are refused before any dataset
  6 STOP units   a per-tensor weight, a float detour and a parity mismatch are each detected, and an
                 end-to-end E7 run whose parity fails writes a STOP record and no run provenance
  7 VAL tools    scripts/ptq_val_checks.py parity, backend (capped) and lists on the synthetic VAL

    python -B scripts/smoke_run_ptq.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import (make_e1_checkpoint, make_e3_checkpoint,  # noqa: E402
                                            make_tree, safe_tmpdir)

BASE = safe_tmpdir("smoke_run_ptq_")
ROOT = BASE / "data"
SPLITS = make_tree(ROOT, n_train=140, n_val=846, n_test=3, seed=2)
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)          # before configs/data.py is imported

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scripts.build_calibration_lists import FILENAME, main as build_lists  # noqa: E402
from scripts.evaluate_model import build_parser as eval_parser, run as eval_run  # noqa: E402
from scripts.ptq_val_checks import main as val_checks  # noqa: E402
from scripts.run_e4 import main as run_e4  # noqa: E402
from scripts.run_ptq import main as run_ptq_cli  # noqa: E402
from src.eval.model_loading import (CheckpointError, load_int8_student,  # noqa: E402
                                    load_int8_torchscript, rebuild_int8_from_state_dict)
from src.eval.stage_artifacts import StageArtifactError, validate_int8_artifact  # noqa: E402
from src.quant import ptq  # noqa: E402
from src.quant.prepare import QuantPreparationError, calibrate, prepare_ptq  # noqa: E402
from src.quant.qconfig import select_qnnpack_backend, select_x86_backend  # noqa: E402

E1 = make_e1_checkpoint(BASE / "e1.pt")
E3 = make_e3_checkpoint(BASE / "e3.pt")
E1_SHA = hashlib.sha256(E1.read_bytes()).hexdigest()
E3_SHA = hashlib.sha256(E3.read_bytes()).hexdigest()
LISTS = BASE / "lists"
results: list[tuple[str, bool, str]] = []
AUDIT = {"on": False, "paths": []}


def _audit(event, args):
    if AUDIT["on"] and event in ("open", "os.listdir", "os.scandir") and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            AUDIT["paths"].append(os.fsdecode(p))


sys.addaudithook(_audit)


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# A full PTQ run always gets its own process, as scripts/run_ptq.py does: TorchScript renames a class
# traced twice in one process, so in-process runs could not reproduce each other's bytes (d3). The
# driver records the paths the run opens or lists (audit hook) and can inject a failing parity check.
DRIVER = r"""
import json, os, sys
sys.dont_write_bytecode = True
repo, audit_out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)
paths = []
def _audit(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        paths.append(os.fsdecode(args[0]))
sys.addaudithook(_audit)
from src.quant import ptq
if mode == "failing_parity":
    real = ptq.output_parity
    def failing(models, inputs, pairs):
        res = real(models, [next(iter(inputs))], pairs)
        res["all_equal"] = False
        return res
    ptq.output_parity = failing
from scripts.run_ptq import main
try:
    rc = main(sys.argv[4:])
finally:
    with open(audit_out, "w", encoding="utf-8") as fh:
        json.dump(paths, fh)
raise SystemExit(rc)
"""


def run_sub(argv, *, mode: str = "plain", label: str = "run") -> tuple[int, str, list]:
    audit_out = BASE / f"audit_{label}.json"
    p = subprocess.run([sys.executable, "-B", "-c", DRIVER, str(REPO), str(audit_out), mode, *argv],
                       capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    paths = json.loads(audit_out.read_text()) if audit_out.exists() else []
    return p.returncode, p.stdout + p.stderr, paths


def cli(fn, argv, audit: bool = False) -> tuple[int, str]:
    buf = io.StringIO()
    AUDIT["on"] = audit
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = fn(argv)
            except SystemExit as e:
                rc = int(e.code) if isinstance(e.code, int) else 2
    finally:
        AUDIT["on"] = False
    return rc, buf.getvalue()


def last(log: str) -> str:
    lines = [ln for ln in log.strip().splitlines() if ln.startswith("RESULT") or ln.startswith("REFUSING")]
    return (lines[-1] if lines else (log.strip().splitlines() or [""])[-1])[:110]


def eval_split_touched(paths) -> list[str]:
    bad = []
    for p in paths:
        parts = Path(os.path.realpath(p)).parts
        if any(parts[i] in ("images", "annotations") and parts[i + 1] in ("val", "test")
               for i in range(len(parts) - 1)):
            bad.append(p)
    return bad


def list_path(seed: int = 42) -> Path:
    return LISTS / FILENAME.format(seed=seed)


def checksum(seed: int = 42) -> str:
    return json.loads(list_path(seed).read_text())["checksum_sha256"]


def ptq_argv(out: Path, *, stage="E4", src=None, src_sha=None, lst=None, ck=None, root=None) -> list:
    return ["--real-run", "--confirm-real-run", "--stage", stage,
            "--source-ckpt", str(src or E1), "--expect-source-sha256", src_sha or E1_SHA,
            "--calibration-list", str(lst or list_path()), "--expect-list-checksum", ck or checksum(),
            "--data-root", str(root or ROOT), "--out-dir", str(out)]


def empty(out: Path) -> bool:
    return not out.exists() or not any(out.iterdir())


def write_list_variant(name: str, mutate) -> Path:
    doc = json.loads(list_path().read_text())
    mutate(doc)
    doc["checksum_sha256"] = ptq.ids_checksum(doc["selected_ids"])
    p = BASE / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 1. gates
def test_gates() -> None:
    rc, log = cli(build_lists, ["--data-root", str(ROOT), "--out-dir", str(LISTS),
                                "--expected-images", "140", "--generated-utc", "2026-10-01T00:00:00Z"])
    check("fixture_lists_built", rc == 0, last(log))
    out = BASE / "refused"
    for label, argv in (("no_flags", []), ("only_real_run", ["--real-run"]),
                        ("missing_pins", ["--real-run", "--confirm-real-run", "--stage", "E4",
                                          "--source-ckpt", str(E1), "--out-dir", str(out)])):
        rc, log = cli(run_ptq_cli, argv)
        check(f"refuses_{label}", rc == 2 and empty(out), last(log))
    cases = [
        ("out_dir_inside_repo", ptq_argv(REPO / "ptq_smoke_out")),
        ("wrong_source_sha256", ptq_argv(out, src_sha="0" * 64)),
        ("malformed_pin", ptq_argv(out, ck="abc")),
        ("wrong_list_checksum", ptq_argv(out, ck="1" * 64)),
        ("e7_with_an_e1_source", ptq_argv(out, stage="E7")),
        ("e4_with_an_e3_source", ptq_argv(out, src=E3, src_sha=E3_SHA)),
    ]
    under_val = BASE / "val" / "data"
    make_tree(under_val, n_train=130, seed=6)
    latest = BASE / "latest"
    make_tree(latest, n_train=130, seed=7)
    cases += [("data_root_under_val", ptq_argv(out, root=under_val)),
              ("data_root_naming_test", ptq_argv(out, root=latest))]
    short = write_list_variant("short.json", lambda d: d["selected_ids"].pop())
    as_val = write_list_variant("as_val.json", lambda d: d.__setitem__("split", "val"))
    foreign = write_list_variant("foreign.json",
                                 lambda d: d["selected_ids"].__setitem__(0, SPLITS["val"][0]))
    cases += [("list_of_127", ptq_argv(out, lst=short, ck=json.loads(short.read_text())["checksum_sha256"])),
              ("list_split_val", ptq_argv(out, lst=as_val, ck=json.loads(as_val.read_text())["checksum_sha256"])),
              ("list_id_outside_train", ptq_argv(out, lst=foreign,
                                                 ck=json.loads(foreign.read_text())["checksum_sha256"]))]
    for label, argv in cases:
        rc, log = cli(run_ptq_cli, argv, audit=True)
        check(f"refuses_{label}", rc == 2 and empty(out) and not (REPO / "ptq_smoke_out").exists(), last(log))
    rc, log = cli(run_ptq_cli, ["--real-run", "--confirm-real-run", "--stage", "E5"])
    check("refuses_a_qat_stage", rc == 2, "argparse choices E4/E7")
    check("refused_runs_never_touched_val_or_test", not eval_split_touched(AUDIT["paths"]))

    # AM-10: one image per calibration mini-batch, everywhere
    select_qnnpack_backend()
    from src.models.student import build_student
    prepared = prepare_ptq(build_student(pretrained=False), select_backend=False)
    try:
        calibrate(prepared, [torch.randn(2, 3, 64, 64)])
        check("calibrate_refuses_batch_of_2", False, "accepted")
    except QuantPreparationError as e:
        check("calibrate_refuses_batch_of_2", "AM-10" in str(e))
    check("calibrate_accepts_batch_of_1", calibrate(prepared, [torch.randn(1, 3, 64, 64)]) == 1)
    rc, log = cli(run_e4, ["--real-run", "--confirm-real-run", "--batch-size", "2", "--source-ckpt", str(E1),
                           "--out-dir", str(BASE / "e4_legacy"), "--calibration-index", str(list_path()),
                           "--data-root", str(ROOT)])
    check("run_e4_refuses_batch_of_2", rc == 2 and "AM-10" in log and not (BASE / "e4_legacy").exists(),
          last(log))


# ---------------------------------------------------------------- 2. a full E4 run
RUN1, RUN2 = BASE / "run1", BASE / "run2"


def test_run() -> dict:
    rc, log, touched = run_sub(ptq_argv(RUN1), label="run1")
    check("e4_run_exit_0", rc == 0, last(log))
    names = ptq.artifact_names("e4")
    present = sorted(p.name for p in RUN1.iterdir())
    check("e4_run_writes_four_files",
          present == sorted([names["torchscript"], names["state_dict"], names["x86_copy"], names["run_meta"]]),
          str(present))
    meta = json.loads((RUN1 / names["run_meta"]).read_text())
    check("run_meta_declares_torchscript_of_record",
          meta["artifact_format"] == "torchscript" and meta["converted_artifact"] == names["torchscript"]
          and meta["backend"] == meta["engine"] == "qnnpack" and meta["stage"] == "E4"
          and meta["source_stage"] == "E1" and meta["random_init"] is False)
    shas = {k: hashlib.sha256((RUN1 / names[k]).read_bytes()).hexdigest()
            for k in ("torchscript", "state_dict", "x86_copy")}
    check("recorded_sha256_match_files",
          shas["torchscript"] == meta["converted_artifact_sha256"]
          and shas["state_dict"] == meta["state_dict_artifact"]["sha256"]
          and shas["x86_copy"] == meta["x86_latency_copy"]["sha256"])
    cal = meta["calibration"]
    check("calibration_record_am10",
          cal["batch_size"] == 1 and cal["batches_seen"] == cal["n_calib"] == 128 and cal["seed"] == 42
          and cal["role"] == "am10_list_of_record" and cal["split"] == "train"
          and cal["checksum_sha256"] == checksum() and cal["val_or_test_read"] is False)
    check("source_record", meta["source_checkpoint_sha256"] == E1_SHA
          and meta["source_expected_sha256_matched"] is True)
    checks = meta["verification"]["checks"]
    check("six_verification_checks_pass", len(checks) == 6 and all(checks.values()), str(checks))
    census = meta["verification"]["census"]
    check("sigmoid_quantized_with_u6_values",
          census["sigmoid"]["quantized"]
          and census["sigmoid"]["u6_values"] == {"scale": 1.0 / 256, "zero_point": 0},
          str(census["sigmoid"]["u6_values"]))
    check("x86_copy_is_latency_only",
          meta["x86_latency_copy"]["engine"] in ("x86", "fbgemm")
          and meta["x86_latency_copy"]["usable_for_accuracy"] is False
          and meta["x86_latency_copy"]["reduce_range"] is True)
    check("observers_recorded",
          meta["observers"] == {"activation": "HistogramObserver", "weight": "PerChannelMinMaxObserver"})
    par = meta["verification"]["parity"]["torchscript_vs_state_dict"]
    check("parity_on_all_128_calibration_images", par["all_equal"] and par["inputs"] == 128 + 4,
          f"{par['pairs']['torchscript == state_dict']['equal']}/{par['inputs']} inputs bitwise equal")
    check("synthetic_list_run_role_is_smoke", meta["ptq_run_role"] == "smoke")
    check("run_never_touched_val_or_test", bool(touched) and not eval_split_touched(touched),
          f"{len(touched)} paths audited")
    rc, log = cli(run_ptq_cli, ptq_argv(RUN1))
    check("refuses_to_overwrite_a_run", rc == 2, last(log))
    return meta


# ---------------------------------------------------------------- 3. d3
def test_d3() -> None:
    rc, log, _ = run_sub(ptq_argv(RUN2), label="run2")
    check("second_run_exit_0", rc == 0, last(log))
    names = ptq.artifact_names("e4")
    rc, log = cli(run_ptq_cli, ["--compare", str(RUN1 / names["run_meta"]), str(RUN2 / names["run_meta"])])
    check("d3_two_runs_bitwise_identical", rc == 0 and "PTQ d3 PASS" in log, last(log))
    rc, log = cli(run_ptq_cli, ["--compare", str(RUN1 / names["run_meta"]), str(RUN1 / names["run_meta"])])
    check("d3_compare_refuses_one_run_twice", rc == 2, last(log))
    for k in ("torchscript", "state_dict", "x86_copy"):
        check(f"d3_{k}_bytes_identical",
              (RUN1 / names[k]).read_bytes() == (RUN2 / names[k]).read_bytes())
    tampered = BASE / "run2_tampered"
    shutil.copytree(RUN2, tampered)
    data = bytearray((tampered / names["x86_copy"]).read_bytes())
    data[-100] ^= 0xFF
    (tampered / names["x86_copy"]).write_bytes(bytes(data))
    rc, log = cli(run_ptq_cli, ["--compare", str(RUN1 / names["run_meta"]), str(tampered / names["run_meta"])])
    check("d3_compare_catches_a_changed_file", rc == 1, last(log))


# ---------------------------------------------------------------- 4. evaluator path
def test_evaluator(meta: dict) -> None:
    names = ptq.artifact_names("e4")
    prov = RUN1 / names["run_meta"]
    resolved = validate_int8_artifact("E4", prov)
    check("validate_resolves_the_relative_artifact",
          resolved["artifact_format"] == "torchscript"
          and resolved["artifact_path"] == (RUN1 / names["torchscript"]).resolve())
    model, info = load_int8_student(resolved)
    check("loader_returns_the_torchscript_module",
          isinstance(model, torch.jit.ScriptModule) and info.sha256 == meta["converted_artifact_sha256"]
          and torch.backends.quantized.engine == "qnnpack")
    x = torch.randn(1, 3, 512, 512, generator=torch.Generator().manual_seed(5))
    sd = rebuild_int8_from_state_dict(RUN1 / names["state_dict"], stage="E4", method="ptq")
    with torch.no_grad():
        y_ts, y_sd = model(x), sd(x)
    check("loaded_torchscript_equals_state_dict_rebuild",
          tuple(y_ts.shape) == (1, 116, 512, 512) and torch.equal(y_ts, y_sd))

    out = BASE / "eval_val"
    args = eval_parser().parse_args(["--stage", "E4", "--precision", "int8_ptq", "--provenance", str(prov),
                                     "--split", "val", "--max-samples", "4", "--batch-size", "1",
                                     "--artifact-status", "smoke", "--device", "cpu", "--out-dir", str(out),
                                     "--run-id", "smoke_e4_int8_torchscript"])
    # The artifact writer's provenance runs a whole-repository `git status` (DL-26's accepted gap). A
    # cloud lane session never lists docs/reference/ (DL-37), so this smoke-status artifact records a
    # stub commit and an empty porcelain instead; the provenance gate itself is not under test here.
    import src.eval.artifacts as artifacts
    real_git = (artifacts.git_porcelain_bytes, artifacts.git_commit)
    artifacts.git_porcelain_bytes = lambda repo: b""
    artifacts.git_commit = lambda repo: "0" * 40
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            art = eval_run(args)
    finally:
        artifacts.git_porcelain_bytes, artifacts.git_commit = real_git
    summary = json.loads((Path(art) / "summary.json").read_text())
    check("evaluate_model_scores_the_torchscript_artifact",
          summary["dataset"]["actual_rows"] == 4 and summary["run"]["precision"] == "int8_ptq"
          and summary["run"]["quant_backend"] == "qnnpack"
          and summary["run"]["checkpoint_sha256"] == meta["converted_artifact_sha256"],
          f"{summary['dataset']['actual_rows']} VAL rows")

    bad = BASE / "run1_tampered"
    shutil.copytree(RUN1, bad)
    blob = bytearray((bad / names["torchscript"]).read_bytes())
    blob[-100] ^= 0xFF
    (bad / names["torchscript"]).write_bytes(bytes(blob))
    try:
        validate_int8_artifact("E4", bad / names["run_meta"])
        check("tampered_torchscript_refused", False, "accepted")
    except StageArtifactError as e:
        check("tampered_torchscript_refused", e.code == "artifact_hash_mismatch", e.code)
    fmt = BASE / "run1_fmt"
    shutil.copytree(RUN1, fmt)
    doc = json.loads((fmt / names["run_meta"]).read_text())
    doc["artifact_format"] = "onnx"
    (fmt / names["run_meta"]).write_text(json.dumps(doc))
    try:
        validate_int8_artifact("E4", fmt / names["run_meta"])
        check("unknown_artifact_format_refused", False, "accepted")
    except StageArtifactError as e:
        check("unknown_artifact_format_refused", e.code == "artifact_format_unknown", e.code)
    doc = json.loads((RUN1 / names["run_meta"]).read_text())
    doc.update(stage="E7", source_stage="E3", cwd_projection_loaded=False)
    e7 = BASE / "run1_as_e7"
    shutil.copytree(RUN1, e7)
    (e7 / names["run_meta"]).write_text(json.dumps(doc))
    try:
        load_int8_student(validate_int8_artifact("E7", e7 / names["run_meta"]))
        check("embedded_identity_refuses_e4_as_e7", False, "accepted")
    except CheckpointError as e:
        check("embedded_identity_refuses_e4_as_e7", "embedded identity" in str(e))
    select_x86_backend()
    try:
        load_int8_torchscript(RUN1 / names["torchscript"], stage="E4", method="ptq", engine="qnnpack")
        check("torchscript_refused_under_another_engine", False, "accepted")
    except CheckpointError as e:
        check("torchscript_refused_under_another_engine", "select 'qnnpack'" in str(e))
    select_qnnpack_backend()
    try:
        load_int8_torchscript(RUN1 / names["x86_copy"], stage="E4", method="ptq", engine="qnnpack")
        check("x86_copy_never_loads_as_accuracy_artifact", False, "accepted")
    except CheckpointError as e:
        check("x86_copy_never_loads_as_accuracy_artifact", "embedded identity" in str(e))
    # once the student has been traced in this process, a PTQ run here is refused up front (a second
    # trace would serialize renamed classes); loading TorchScript files, as above, renames nothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # the expected TracerWarnings
        torch.jit.trace(sd, torch.zeros(1, 3, 64, 64))
    inproc = BASE / "run_inproc"
    rc, log = cli(run_ptq_cli, ptq_argv(inproc))
    check("ptq_run_refused_in_a_process_that_traced_the_student",
          rc == 2 and "fresh_process_required" in log and empty(inproc), last(log))


# ---------------------------------------------------------------- 5. official E4/E7: list of record
def test_official() -> None:
    from scripts.evaluate_model import CliError, Counters
    from src.eval.stage_artifacts import int8_official_calibration_error

    names = ptq.artifact_names("e4")
    prov = RUN1 / names["run_meta"]
    args = eval_parser().parse_args(["--stage", "E4", "--precision", "int8_ptq", "--provenance", str(prov),
                                     "--split", "test", "--confirm-test-split", "--artifact-status",
                                     "official", "--batch-size", "1", "--device", "cpu",
                                     "--out-dir", str(BASE / "official_refused")])
    counters = Counters(dataset=[], model=[])
    try:
        eval_run(args, counters=counters)
        check("official_score_refused_for_a_synthetic_list", False, "accepted")
    except CliError as e:
        check("official_score_refused_for_a_synthetic_list",
              "registered" in str(e) and counters.dataset == [] and counters.model == []
              and not (BASE / "official_refused").exists(), str(e)[:100])

    resolved = validate_int8_artifact("E4", prov)

    def variant(**cal):
        r = json.loads(json.dumps({"spec": resolved["spec"], "provenance": resolved["provenance"]}))
        r["provenance"]["calibration"].update(cal)
        return r

    def fake_repo(name: str, seed: int | None, status: str = "registered") -> Path:
        root = BASE / name
        d = root / "configs" / "calibration"
        d.mkdir(parents=True)
        if seed is not None:
            doc = json.loads(list_path(seed).read_text())
            doc["artifact_status"] = status
            (d / FILENAME.format(seed=42)).write_text(json.dumps(doc), encoding="utf-8")
        return root

    committed = fake_repo("repo_committed", 42)
    registered = variant(list_artifact_status="registered")
    check("official_accepts_the_committed_list_of_record",
          int8_official_calibration_error(registered, committed) is None)
    other = fake_repo("repo_other", 42)
    doc = json.loads((other / "configs" / "calibration" / FILENAME.format(seed=42)).read_text())
    doc["selected_ids"][0] = SPLITS["train"][-1] if SPLITS["train"][-1] not in doc["selected_ids"] else "x"
    doc["checksum_sha256"] = ptq.ids_checksum(doc["selected_ids"])
    (other / "configs" / "calibration" / FILENAME.format(seed=42)).write_text(json.dumps(doc))
    check("official_refuses_a_list_other_than_the_committed_one",
          "committed list of record is" in str(int8_official_calibration_error(registered, other)))
    check("official_refuses_a_committed_file_that_is_not_seed_42",
          "not the list of record" in str(int8_official_calibration_error(registered,
                                                                          fake_repo("repo_seed43", 43))))
    check("official_refuses_an_uncommitted_list_of_record",
          "not committed" in str(int8_official_calibration_error(registered, fake_repo("repo_none", None))))
    check("official_refuses_a_sensitivity_list",
          "seed 43" in str(int8_official_calibration_error(
              variant(role="am16_calibration_sensitivity", seed=43, list_artifact_status="registered"),
              committed)))
    check("official_refuses_a_smoke_committed_list",
          "not a registered build" in str(int8_official_calibration_error(
              registered, fake_repo("repo_smoke", 42, status="smoke"))))
    legacy = {"spec": resolved["spec"], "provenance": {k: v for k, v in resolved["provenance"].items()
                                                       if k != "artifact_format"}}
    check("official_gate_leaves_legacy_state_dict_artifacts_alone",
          int8_official_calibration_error(legacy, BASE / "repo_none") is None)


# ---------------------------------------------------------------- 6. STOP detection
def test_stops() -> None:
    import torch.ao.quantization as tq
    from torch.ao.nn.quantized import DeQuantize, Quantize
    from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver

    select_qnnpack_backend()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant, self.conv, self.dequant = tq.QuantStub(), nn.Conv2d(3, 4, 1), tq.DeQuantStub()

        def forward(self, x):
            return self.dequant(self.conv(self.quant(x)))

    tiny = Tiny().eval()
    tiny.qconfig = tq.QConfig(activation=HistogramObserver.with_args(dtype=torch.quint8),
                              weight=MinMaxObserver.with_args(dtype=torch.qint8,
                                                              qscheme=torch.per_tensor_symmetric))
    prepared = tq.prepare(tiny)
    prepared(torch.randn(1, 3, 8, 8))
    per_tensor = tq.convert(prepared)
    eager = ptq.weight_scheme_report(per_tensor)
    scripted = ptq.weight_scheme_report(torch.jit.trace(per_tensor, torch.randn(1, 3, 8, 8)))
    check("per_tensor_weight_detected_eager_and_torchscript",
          eager["violations"] == ["conv"] and not eager["ok"] and scripted["violations"] and not scripted["ok"],
          f"{eager['qschemes']}")

    names = ptq.artifact_names("e4")
    model = rebuild_int8_from_state_dict(RUN1 / names["state_dict"], stage="E4", method="ptq")
    model.head.context[2] = nn.Sequential(DeQuantize(), nn.Sigmoid(),
                                          Quantize(scale=1.0 / 256, zero_point=0, dtype=torch.quint8))
    x = torch.zeros(1, 3, 512, 512)
    census = ptq.runtime_census(model, x)
    graph = ptq.graph_census(torch.jit.trace(model, x))
    flagged = {f["module"] for f in census["float_modules"]}
    check("float_detour_detected_at_run_time",
          not census["ok"] and "head.context.2.1" in flagged and census["quantize_modules"] == 2,
          f"float modules {sorted(flagged)}")
    check("float_detour_detected_in_the_graph",
          not graph["ok"] and graph["quantize_per_tensor"] == 2 and graph["dequantize"] == 2)

    obj = torch.load(RUN1 / names["state_dict"], map_location="cpu", weights_only=False)
    obj["model"]["head.ff_add.scale"] = obj["model"]["head.ff_add.scale"] * 1.5
    torch.save(obj, BASE / "tampered_state_dict.pt")
    ts, _ = load_int8_torchscript(RUN1 / names["torchscript"], stage="E4", method="ptq")
    sd = rebuild_int8_from_state_dict(BASE / "tampered_state_dict.pt", stage="E4", method="ptq")
    par = ptq.output_parity({"torchscript": ts, "state_dict": sd}, ptq.synthetic_parity_inputs(),
                            [("torchscript", "state_dict")])
    check("parity_mismatch_detected", not par["all_equal"])

    # end to end, as E7 in its own process: a failing parity writes the STOP record and no run
    # provenance (exit 1)
    out = BASE / "run_e7_stop"
    rc, log, _ = run_sub(ptq_argv(out, stage="E7", src=E3, src_sha=E3_SHA), mode="failing_parity",
                         label="e7_stop")
    n7 = ptq.artifact_names("e7")
    stop = json.loads((out / n7["stop"]).read_text()) if (out / n7["stop"]).exists() else {}
    check("e7_stop_exit_1_without_run_provenance",
          rc == 1 and not (out / n7["run_meta"]).exists() and stop.get("code") == "parity_mismatch"
          and stop.get("stage") == "E7", last(log))


# ---------------------------------------------------------------- 7. local VAL tools
def test_val_tools() -> None:
    names = ptq.artifact_names("e4")
    prov = str(RUN1 / names["run_meta"])
    rc, log = cli(val_checks, ["parity", "--provenance", prov, "--n", "4"])
    check("val_parity_subset_bitwise_equal", rc == 0 and "PARITY PASS" in log, last(log))
    rc, log = cli(val_checks, ["backend", "--provenance", prov, "--out-dir", str(BASE / "d5"),
                               "--max-samples", "4"])
    rec_path = BASE / "d5" / "ptq_backend_parity_val_E4.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    check("backend_parity_record_written_capped",
          rc == 0 and rec.get("capped") is True and rec.get("is_the_d5_result") is False
          and rec.get("is_official_stage_accuracy") is False and rec.get("rows") == 4
          and set(rec.get("deltas", {})) == {"all_class_miou", "all_class_macro_dice", "all_class_macc",
                                             "disease_only_miou"}, last(log))
    rc, log = cli(val_checks, ["lists", "--list-dir", str(LISTS)])
    check("lists_disjoint_from_val", rc == 0, last(log))
    bad = BASE / "lists_touching_val"
    bad.mkdir()
    shutil.copy(BASE / "foreign.json", bad / "ptq_calibration_seed42.json")
    rc, log = cli(val_checks, ["lists", "--list-dir", str(bad)])
    check("a_list_touching_val_is_a_stop", rc == 1, last(log))
    check("repository_untouched", not (REPO / "ptq_smoke_out").exists()
          and not (REPO / "configs" / "calibration").exists())


def main() -> int:
    print("=" * 78)
    print("PTQ RUNNER SMOKE (lane 8) -- synthetic; no PlantSeg data, no GPU")
    print(f"torch {torch.__version__} | engines={list(torch.backends.quantized.supported_engines)}")
    print(f"temp: {BASE}")
    print("=" * 78)
    test_gates()
    meta = test_run()
    test_d3()
    test_evaluator(meta)
    test_official()
    test_stops()
    test_val_tools()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
