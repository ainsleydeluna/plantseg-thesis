#!/usr/bin/env python3
"""Corrupted-condition evaluation for the teacher and E1-E7: one (stage, corruption, severity) cell.

The flow is the clean runner's (scripts/evaluate_model.py, which stays byte-unchanged), through the
same core: model-source validation, cache validation, the expected manifest from the filename
listing, `prepare_artifact_request`, `validate_artifact_request` (nothing is built before it
passes), the section 10 policy, the dataset wrapper and the model, `evaluate_model` with
`ModelDeviceForward`, `check_post_eval`, `build_eval_runtime_record`, `write_artifact`, and then
the cell record `<out_dir>.corruption.json` (src/eval/corrupted.py; TL interface section 7). No
metric, identity, writer or runtime code is copied.

PATH B ONLY. Corrupted pixels are READ from the frozen shared cache through
`src.eval.robustness.load_cached_corruption` (B-2's `load_cached`, resolved through its `CacheIndex`),
hash-verified on every read and never regenerated: a missing or modified item is a refusal. Direct
deterministic generation (PATH A) belongs to the cache builder alone, and this module never imports
it, so the teacher and E1-E7 score byte-identical corrupted pixels. The cached payload is already
resized, corrupted, uint8 RGB; the wrapper pastes it into the clean canvas, whose padding and
ground truth come from the frozen core_preprocess, and normalises it as the clean path does.

NO RANDOM-INIT PATH (LE12). This runner has no `--random-init`: every cell scores a trained model
handed as `--checkpoint` (FP32 students, the teacher) or `--provenance` (INT8 E4-E7). The clean
runner's random-init rules (guard G4) therefore have nothing to act on here.

Identity and threads (LE2; rulings TL-6, IF-T1; the codes are LEFF's for evaluate_model.py):
  --expected-checkpoint-sha256 H   the sha256 of the file this run is handed (--checkpoint, or
                                   --provenance for an INT8 stage), compared before its first parse
                                   and again after validation
  --expected-converted-sha256 H2   INT8 only: the converted artifact's sha256, compared after the
                                   provenance validates and re-hashed just before the load
  --threads T                      torch.set_num_threads(T) right after the CLI checks, read back;
                                   run.eval_runtime.torch_num_threads must equal T before anything
                                   is written
Under R (--split test with --artifact-status official) H, T, and for INT8 H2, are required. The
teacher also keeps --teacher-ckpt-sha256 (DL-59), and under R H must equal it. A coded refusal is
printed on stderr exactly as `[code] message`; an uncoded one (the clean runner's copied texts) is
printed after its exception type.

Guards (plan section 2.5): G1 --split test needs --confirm-test-split, checked before any path is
read (the split defaults to val); G2 test refuses smoke; G3 test refuses --max-samples; G5 official:
split test, 1,561 rows (enforced by the split listing's locked count, the dataset's count guard and
the cache's full-source check; official_launch_error is handed the constant), an official pinned
cache of 1,561 sources whose canonical checksum equals --expected-cache-sha256, a matching
verification record, clean governed paths, eval_runtime; G6
provisional: a provisional or official pinned cache, --expected-cache-sha256 and a matching
verification record; G7 an existing out-dir or record is refused; G8 an out-dir inside the
repository is refused (inputs are never copied in, LE14); G9 the cache's split (from its pin) must
be the requested split; G10 the protocol file must be the one the cache was generated under; G11
the canvas protocol only. The out-dir follows TL section 2 (no "test" in any case, none of
:<>"\\|?* in any component), checked as a string before any filesystem call and again resolved.
Stages A, F and G, and the artifact schema's other stages outside STAGES (E6-E2, E7-E2, E6-R1 and
E6-R2 of AM-23; E6-KD of AM-3), which TL interface section 9 gives no corrupted scope, are refused
by name before any file is read (LE23), and precision qat_fakequant_off (AM-4's clean-only score,
lane L-FQOFF) is refused by name (LE15). An official INT8 stage must pass
src.eval.stage_artifacts.int8_official_error, the one function an official INT8 path calls (the
AM-10 list of record, then the PTQ record of record; LQP ruling R2), as scripts/evaluate_model.py
does (LE22). The teacher's NMF stream must be M4-V with seed 42 and draw once per row (LE11;
`nmf_stream_mismatch`, `nmf_draws_mismatch`).

Exit 0: the artifact and its record were written. Exit 2: refused. Nothing was written, except when
the cell record's write fails after the artifact: the artifact is then left without its record,
which the grid runner STOPs on (`cell_left_partial_output`) and a restart refuses
(`artifact_without_record`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import corruption_roles  # noqa: E402
from src.corruption_cache import (MANIFEST_NAME, CorruptionCacheError, _strict_json,  # noqa: E402
                                  load_manifest, protocol_block)
from src.distill.teacher import sha256_format_error  # noqa: E402
from src.eval.artifacts import STAGES as ARTIFACT_STAGES  # noqa: E402
from src.eval.corrupted import (CacheCell, CorruptedEvalError,  # noqa: E402
                                PlantSegCorruptedEvalDataset, build_cell_record,
                                created_path_problem, record_path, utc_iso, write_cell_record)
from src.eval.robustness import (RobustnessError, load_cached_corruption,  # noqa: E402
                                 registered_corruptions, validate_condition)
from src.eval.stage_artifacts import (OFFICIAL_ROWS, OFFICIAL_SPLIT,  # noqa: E402
                                      STAGE_ARTIFACTS, official_launch_error,
                                      resolve_evaluation_source)
from scripts.verify_corruption_cache import VERIFICATION_SCHEMA  # noqa: E402

STAGES = ("teacher", "E1", "E2", "E3", "E4", "E5", "E6", "E7")
NOT_IN_ROBUSTNESS = ("A", "F", "G")
#: LE23: the clause that makes each other stage of the artifact schema (src.eval.artifacts.STAGES)
#: descriptive; TL interface section 9 gives none of them a corrupted scope.
NO_CORRUPTED_SCOPE = {
    **dict.fromkeys(("E6-E2", "E7-E2", "E6-R1", "E6-R2"), "AM-23: a descriptive INT8 stage"),
    "E6-KD": "AM-3: the descriptive E6-KD",
}
INT8_PRECISIONS = ("int8_ptq", "int8_qat")
FQ_OFF_PRECISION = "qat_fakequant_off"
EXPECTED_SPLIT_ROWS = {"val": 846, "test": 1561}
PINNED_STATUSES = ("official", "provisional")
CANVAS = "canvas"
_HEX64 = re.compile(r"[0-9a-f]{64}")
#: The teacher's stream under M4-V (src/distill/nmf_stream.py: M4_NMF_SEED; LE11).
M4V_POLICY, M4V_SEED = "M4-V", 42


class CorruptionCliError(RuntimeError):
    """Argument/mode violation, raised before any model or dataset construction."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message if code is None else f"[{code}] {message}")
        self.code = code


def _refuse(code: str, message: str, kind=None):
    """Every coded refusal of this runner raises here; `kind` picks the exception class (the CLI's
    by default; CorruptionCacheError for the cache gates, CorruptedEvalError after inference)."""
    if kind is CorruptionCacheError:
        raise CorruptionCacheError(code, message)
    if kind is CorruptedEvalError:
        raise CorruptedEvalError(code, message)
    raise CorruptionCliError(message, code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Corrupted-condition PlantSeg evaluation runner.")
    p.add_argument("--stage", required=True,
                   help="teacher or E1-E7 (refused: A, F and G, not in the robustness analyses; "
                        "E6-E2, E7-E2, E6-R1 and E6-R2 (AM-23) and E6-KD (AM-3), which carry no "
                        "corrupted scope)")
    p.add_argument("--precision", default=None,
                   help="optional; must be the stage's precision (fp32, int8_ptq, int8_qat)")
    p.add_argument("--corruption", required=True)
    p.add_argument("--severity", required=True, type=int)
    p.add_argument("--out-dir", required=True)
    # The ONE cache input. Everything else about the cache (grid, seed policy, vendor identity,
    # per-item digests) is read from the committed manifest rather than duplicated as CLI flags.
    p.add_argument("--corruption-cache", required=True,
                   help="root of the frozen shared corruption cache (must contain the manifest)")
    p.add_argument("--expected-cache-sha256", default=None,
                   help="the cache's canonical_manifest_sha256 of record (official, provisional)")
    p.add_argument("--verification-record", default=None,
                   help="the record scripts/verify_corruption_cache.py wrote for this cache")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--teacher-config", default=None,
                   help="the thesis teacher config (required for the teacher stage; M4-V)")
    p.add_argument("--teacher-ckpt-sha256", default=None,
                   help="the teacher checkpoint's SHA-256 (required for --stage teacher, refused on "
                        "any other stage); compared before the file is parsed")
    p.add_argument("--expected-checkpoint-sha256", default=None,
                   help="the sha256 of the file this run is handed (--checkpoint, or --provenance "
                        "for an INT8 stage); compared before its first parse")
    p.add_argument("--expected-converted-sha256", default=None,
                   help="INT8 only: the sha256 of the converted artifact the provenance names")
    p.add_argument("--threads", default=None,
                   help="torch.set_num_threads(T) before any model is built")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--artifact-status", default="smoke",
                   choices=["official", "provisional", "smoke"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--protocol", default=CANVAS,
                   help="canvas only: an upstream score is never computed under corruption")
    p.add_argument("--run-id", default=None)
    return p


def stage_precision(stage: str) -> str:
    return STAGE_ARTIFACTS["TEACHER" if stage == "teacher" else stage]["precision"]


def parse_threads(value):
    """A positive integer, from the CLI's text or a programmatic int; None when not given."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and value.isascii() and int(value) > 0:
        return int(value)
    _refuse("threads_invalid", f"--threads must be a positive integer, got {value!r}")


def validate_args(args) -> dict:
    """Pure validation: no filesystem, no dataset, no model. Returns the condition payload."""
    stage = str(args.stage)
    if stage.upper() in NOT_IN_ROBUSTNESS:
        _refuse("stage_not_in_robustness",
                f"stage {stage!r} is not in the robustness analyses (AM-17 item 7; AM-17b item "
                "1(e)): the arms A, F and G are never scored under corruption")
    if stage in ARTIFACT_STAGES and stage not in STAGES:
        _refuse("stage_not_in_robustness",
                f"stage {stage!r} is not in the robustness analyses "
                f"({NO_CORRUPTED_SCOPE.get(stage, 'a stage of the artifact schema')}; TL interface "
                "section 9 gives it no corrupted scope): it is never scored under corruption")
    if stage not in STAGES:
        _refuse("stage_unknown", f"stage {stage!r} is not one of {list(STAGES)}")
    precision = stage_precision(stage)
    asked = getattr(args, "precision", None)
    if asked is not None:
        if asked == FQ_OFF_PRECISION:
            _refuse("qat_fakequant_off_clean_only",
                    f"precision {FQ_OFF_PRECISION} is AM-4's fake-quant-disabled score: clean and "
                    "canvas only, built by lane L-FQOFF (TL-12); it is never scored under "
                    "corruption")
        if asked != precision:
            _refuse("precision_stage_mismatch",
                    f"stage {stage} is scored at precision {precision!r}, not {asked!r}")
    official = args.artifact_status == "official"
    condition = validate_condition(args.corruption, args.severity, official=official)
    protocol = getattr(args, "protocol", CANVAS)
    if protocol != CANVAS:
        _refuse("protocol_canvas_only",
                f"--protocol {protocol!r} is refused: a corrupted condition is scored on the canvas "
                "protocol only (EVALUATION_CONTRACT section 11(a))")

    # ---- test-split guards (contract section 7): G1, G2, G3 ----
    if args.split == OFFICIAL_SPLIT:
        if not args.confirm_test_split:
            raise CorruptionCliError(
                "--split test requires --confirm-test-split (the test split is not touched casually)")
        if args.artifact_status == "smoke":
            raise CorruptionCliError("--split test refuses artifact_status=smoke")
        if args.max_samples is not None:
            raise CorruptionCliError(
                "--split test forbids any sample cap; the full official test split is required")
    if official:
        err = official_launch_error(split=args.split, random_init=False, stage=stage.upper(),
                                    expected_manifest_rows=OFFICIAL_ROWS,
                                    max_samples=args.max_samples)
        if err:
            raise CorruptionCliError(err)

    # ---- model source (the clean runner's rules, without its random-init path) ----
    checkpoint = getattr(args, "checkpoint", None)
    provenance = getattr(args, "provenance", None)
    if stage == "teacher":
        if not checkpoint:
            raise CorruptionCliError("the teacher stage requires --checkpoint")
        if not getattr(args, "teacher_config", None):
            raise CorruptionCliError(
                "the teacher stage requires --teacher-config (the thesis teacher config)")
        if args.batch_size != 1:      # M4-V (B61 section 4): one complete pass at batch size 1
            raise CorruptionCliError(
                f"M4-V evaluates the teacher with --batch-size 1; got {args.batch_size}")
    if precision in INT8_PRECISIONS:
        if not provenance:
            raise CorruptionCliError(
                f"precision={precision!r} requires --provenance (the E4-E7 run-provenance JSON). A "
                "raw checkpoint cannot identify a quantized artifact.")
        if checkpoint:
            raise CorruptionCliError(
                "--checkpoint and --provenance both identify a model source; for INT8 stages pass "
                "--provenance only")
        if args.device != "cpu":
            raise CorruptionCliError(
                f"converted eager INT8 models are CPU-only; got --device {args.device!r}")
    else:
        if provenance:
            raise CorruptionCliError(
                f"--provenance applies to INT8 stages only; precision={precision!r} takes "
                "--checkpoint")
        if not checkpoint:
            _refuse("checkpoint_required",
                    f"stage {stage} is scored from its --checkpoint (this runner has no random-init "
                    "path)")
    if args.batch_size < 1:
        raise CorruptionCliError("--batch-size must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise CorruptionCliError("--max-samples must be >= 1")
    expected_cache = getattr(args, "expected_cache_sha256", None)
    if expected_cache is not None and not _HEX64.fullmatch(str(expected_cache)):
        _refuse("expected_cache_sha256_format",
                f"--expected-cache-sha256 {sha256_format_error(expected_cache)}")

    # ---- teacher checkpoint identity (DL-59; E-35 split to this lane) ----
    teacher_sha = getattr(args, "teacher_ckpt_sha256", None)
    if stage == "teacher":
        if teacher_sha is None:
            _refuse("teacher_ckpt_sha256_required",
                    "the teacher stage requires --teacher-ckpt-sha256, the checkpoint's SHA-256, "
                    "compared before the file is parsed")
        sha_error = sha256_format_error(teacher_sha)
        if sha_error is not None:
            _refuse("teacher_ckpt_sha256_format", f"--teacher-ckpt-sha256 {sha_error}")
    elif teacher_sha is not None:
        _refuse("teacher_ckpt_sha256_not_teacher",
                f"--teacher-ckpt-sha256 applies to the teacher stage only; stage={stage!r} takes "
                "none")

    # ---- LE2: identity of the handed file and of the converted artifact; the thread count ----
    in_r = args.split == OFFICIAL_SPLIT and official
    expected = getattr(args, "expected_checkpoint_sha256", None)
    if expected is None and in_r:
        _refuse("expected_checkpoint_sha256_required",
                "--split test with --artifact-status official requires "
                "--expected-checkpoint-sha256, the sha256 of the file this run is handed "
                "(--checkpoint, or --provenance for an INT8 stage)")
    if expected is not None:
        sha_error = sha256_format_error(expected)
        if sha_error is not None:
            _refuse("expected_checkpoint_sha256_format",
                    f"--expected-checkpoint-sha256 {sha_error}")
        if stage == "teacher" and expected != teacher_sha:
            _refuse("expected_checkpoint_sha256_teacher_disagrees",
                    f"--expected-checkpoint-sha256 {expected} and --teacher-ckpt-sha256 "
                    f"{teacher_sha} differ; for the teacher both name --checkpoint and must be "
                    "equal")
    converted = getattr(args, "expected_converted_sha256", None)
    if converted is not None and precision not in INT8_PRECISIONS:
        _refuse("expected_converted_sha256_not_int8",
                f"--expected-converted-sha256 applies to an INT8 stage handed --provenance; this "
                f"run (precision={precision!r}, random_init={False}) takes none")
    if precision in INT8_PRECISIONS:
        if converted is None and in_r:
            _refuse("expected_converted_sha256_required",
                    "--split test with --artifact-status official requires "
                    "--expected-converted-sha256 for an INT8 stage, the sha256 of the converted "
                    "artifact its provenance names")
        if converted is not None:
            sha_error = sha256_format_error(converted)
            if sha_error is not None:
                _refuse("expected_converted_sha256_format",
                        f"--expected-converted-sha256 {sha_error}")
    threads = parse_threads(getattr(args, "threads", None))
    if threads is None and in_r:
        _refuse("threads_required",
                "--split test with --artifact-status official requires --threads, the torch "
                "thread count of the execution context of record")
    return condition


def apply_threads(args):
    """torch.set_num_threads(T) with a read-back, before any model is built. None: not set."""
    import torch
    threads = parse_threads(getattr(args, "threads", None))
    if threads is None:
        return None
    torch.set_num_threads(threads)
    got = torch.get_num_threads()
    if got != threads:
        _refuse("threads_not_applied",
                f"torch.set_num_threads({threads}) left torch.get_num_threads() at {got}")
    return threads


def check_outputs(out_dir_arg):
    """G7, G8 and the TL section 2 name rule, before any input is read. Returns (out_dir, record)."""
    raw = os.path.abspath(str(out_dir_arg))
    problem = created_path_problem(out_dir_arg) or (None if os.path.basename(raw) else
                                                    f"{raw!r} names no directory")
    if problem:
        _refuse("out_dir_name_invalid",
                f"the out-dir {problem}; TL interface section 2 forbids 'test' in any case and "
                ":<>\"\\|?* in any component of a created path")
    out_dir = Path(raw).resolve()
    if REPO == out_dir or REPO in out_dir.parents:
        _refuse("out_dir_inside_repo", "artifacts are never written inside the repository")
    rec = record_path(out_dir)
    if out_dir.exists():
        _refuse("out_dir_exists", f"refusing to overwrite an existing artifact directory: "
                                  f"{out_dir}")
    if rec.exists():
        _refuse("record_exists", f"refusing to overwrite an existing cell record: {rec}")
    return out_dir, rec


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def handed_file(args) -> Path:
    """The file this run is handed: --provenance for an INT8 stage, else --checkpoint."""
    return Path(args.provenance if stage_precision(str(args.stage)) in INT8_PRECISIONS
                else args.checkpoint)


def hash_before_parse(args):
    """LE2: the handed file's sha256 before its first parse, compared with
    --expected-checkpoint-sha256. A missing file is left to the validator's own refusal."""
    path = handed_file(args)
    if not path.is_file():
        return path, None
    actual = _sha256_file(path)
    expected = getattr(args, "expected_checkpoint_sha256", None)
    if expected is not None and actual != expected:
        _refuse("expected_checkpoint_sha256_mismatch",
                f"{path} has sha256 {actual}, expected {expected}; the file was not parsed")
    return path, actual


def require_unchanged(path: Path, before: str | None) -> None:
    """LE2: the handed file after validation is the file that was hashed before it."""
    if before is None:
        return
    after = _sha256_file(path)
    if after != before:
        _refuse("source_changed_during_validation",
                f"{path} had sha256 {before} before validation and {after} after it")


def cache_split(manifest: dict):
    """LE3: the split the cache was generated for, read from its embedded generator pin's
    parameters. A pin-less cache (the in-memory generator's smoke caches) reports
    dataset.split, and serves a smoke artifact only (cache_pin_missing)."""
    pin = manifest.get("pin")
    if isinstance(pin, dict):
        return (pin.get("parameters") or {}).get("split")
    return (manifest.get("dataset") or {}).get("split")


def resolve_corruption_cache(cache_root, corruption: str, severity: int, *, split: str,
                             official: bool, artifact_status: str | None = None,
                             expected_sha256: str | None = None) -> dict:
    """Validate the shared cache BEFORE any corrupted-image inference. Never regenerates.

    `load_manifest` already enforces the manifest schema, the canonical manifest checksum, the seed
    policy identity, the patched vendor-runtime SHA and the item paths by index
    (item_path_mismatch); anything it rejects raises here. On top of that this adds the
    request-level checks the runner is responsible for: the cache must be the split being
    evaluated (G9, from its pin), must hold the requested cell (coverage per cell, `cells[]`),
    must be complete for an official run, must be pinned and of a sufficient status for an official
    or provisional run, must carry the expected canonical checksum (G5, G6) and must have been
    generated under the current protocol file (G10).
    """
    status = artifact_status or ("official" if official else "smoke")
    root = Path(cache_root).resolve()
    if not root.is_dir():
        _refuse("cache_root_missing", f"corruption cache not found: {root}", CorruptionCacheError)

    manifest = load_manifest(root)   # schema, checksum, seed policy, vendor SHA, item paths
    check_cache_request(manifest, [(corruption, severity)], split=split, status=status,
                        expected_sha256=expected_sha256)
    return manifest


def check_cache_request(manifest: dict, cells, *, split: str, status: str,
                        expected_sha256: str | None = None) -> None:
    """The request-level gates on a loaded manifest, for every (corruption, severity) in `cells`,
    in this order: G9 the split (from the pin); coverage per cell; G5 completeness for an official
    run; G5 and G6 the status, the pin and the expected checksum; G10 the protocol file. The grid
    runner calls it once at start for every planned cell (scripts/run_corruption_grid.py)."""
    cache = CorruptionCacheError
    held = cache_split(manifest)
    if held != split:
        _refuse("cache_split_mismatch", f"cache was built for split {held!r}, evaluating "
                                        f"{split!r}", cache)
    have = [c.get("cell_id") for c in manifest.get("cells") or []]
    for corruption, severity in cells:
        wanted = None
        try:
            wanted = corruption_roles.cell_id(corruption, severity)
        except corruption_roles.CorruptionRolesError as e:
            _refuse("cache_cell_absent", str(e), cache)
        if wanted not in have:
            _refuse("cache_cell_absent", f"the cache holds no cell {wanted!r}; its cells are "
                                         f"{have}", cache)
    if status == "official" and manifest.get("source_image_count") != OFFICIAL_ROWS:
        _refuse("cache_incomplete_for_official",
                f"an official robustness run requires the full cached test split "
                f"({OFFICIAL_ROWS} source images), the cache holds "
                f"{manifest.get('source_image_count')}", cache)
    if status in PINNED_STATUSES:
        allowed = ("official",) if status == "official" else PINNED_STATUSES
        if manifest.get("status") not in allowed:
            _refuse("cache_status_insufficient",
                    f"a {status} artifact is scored from a cache of status {list(allowed)}, this "
                    f"cache is {manifest.get('status')!r}", cache)
        if not isinstance(manifest.get("pin"), dict):
            _refuse("cache_pin_missing",
                    f"a {status} artifact needs a cache that embeds its generator pin", cache)
        if expected_sha256 is None:
            _refuse("expected_cache_sha256_required",
                    f"a {status} artifact requires --expected-cache-sha256, the cache's "
                    "canonical_manifest_sha256 of record", cache)
    if expected_sha256 is not None and manifest["canonical_manifest_sha256"] != expected_sha256:
        _refuse("cache_checksum_mismatch",
                f"the cache's canonical_manifest_sha256 is {manifest['canonical_manifest_sha256']}, "
                f"--expected-cache-sha256 is {expected_sha256}", cache)
    current = protocol_block(REPO)
    if manifest.get("protocol") != current:
        _refuse("protocol_changed_since_generation",
                f"the cache was generated under protocol {manifest.get('protocol')}, the current "
                f"file is {current}", cache)


def check_verification_record(cache_root, manifest: dict, record, *, artifact_status: str):
    """G5, G6: the verifier's record of THIS cache. Returns the record file's sha256 (None when
    no record is given to a smoke run). Read once: the digest and the parse use the same bytes."""
    cache = CorruptionCacheError
    if record is None:
        if artifact_status in PINNED_STATUSES:
            _refuse("verification_record_required",
                    f"a {artifact_status} artifact requires --verification-record, the record "
                    "scripts/verify_corruption_cache.py wrote for this cache", cache)
        return None
    path = Path(record)
    raw, rec = b"", None
    try:
        raw = path.read_bytes()
        rec = _strict_json(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        _refuse("verification_record_mismatch",
                f"{path} is not a readable verification record: {e}", cache)
    if not isinstance(rec, dict):
        _refuse("verification_record_mismatch", f"{path} is not an object", cache)
        rec = {}
    counts = rec.get("counts") if isinstance(rec.get("counts"), dict) else {}
    want = {
        "schema": VERIFICATION_SCHEMA,
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "inferential_cells_sha256": manifest["inferential_cells_sha256"],
        "manifest_file_sha256": _sha256_file(Path(cache_root) / MANIFEST_NAME),
        "split": cache_split(manifest),
        "status": manifest.get("status"),
        "pin_sha256": (manifest.get("pin") or {}).get("pin_sha256"),
    }
    bad = [k for k, v in want.items() if rec.get(k) != v]
    for key, value in (("sources_verified", manifest.get("source_image_count")),
                       ("items_verified", len(manifest.get("entries") or [])),
                       ("cells", len(manifest.get("cells") or []))):
        if counts.get(key) != value:
            bad.append(f"counts.{key}")
    if bad:
        _refuse("verification_record_mismatch",
                f"{path} does not record a verification of this cache: {bad}", cache)
    return hashlib.sha256(raw).hexdigest()


def check_cache_sources(manifest: dict, expected_manifest, *, full: bool) -> None:
    """C1: every expected row is a source of the cache at the same manifest_index, and a full run's
    cache holds exactly the split. The expected manifest comes from the filename listing, never
    from the cache."""
    by_index = {s.get("manifest_index"): s.get("image_id") for s in manifest.get("sources") or []}
    bad = [(e.manifest_index, e.image_id) for e in expected_manifest
           if by_index.get(e.manifest_index) != e.image_id]
    if full and len(by_index) != len(expected_manifest):
        bad.append(f"{len(by_index)} cached sources for {len(expected_manifest)} rows")
    if bad:
        _refuse("cache_sources_mismatch",
                f"the split listing and the cache's sources differ: {bad[:3]}",
                CorruptionCacheError)


def cached_image_loader(cache_root, manifest: dict, corruption: str, severity: int):
    """PATH B accessor: `image_id -> uint8 RGB`, hash-verified on every read.

    This is the ONLY way this runner obtains a corrupted image, and it is identical for the teacher
    and E1-E7, so every stage scores the same frozen realisation. It takes no ground truth: that
    never enters the corruption/cache path.
    """
    root = Path(cache_root).resolve()

    def _load(image_id: str):
        return load_cached_corruption(root, manifest, image_id, corruption, severity)

    return _load


def refusal_line(e: Exception) -> str:
    """LE2: a coded refusal prints exactly `[code] message`; an uncoded one keeps its type."""
    code, text = getattr(e, "code", None), str(e)
    if isinstance(code, str):
        return text if text.startswith(f"[{code}] ") else f"[{code}] {text}"
    return f"{type(e).__name__}: {text}"


def main(argv=None) -> int:
    started = utc_iso()
    args = build_parser().parse_args(argv)
    try:
        out = run(args, started_utc=started)
    except Exception as e:                                # noqa: BLE001 -- reported, not swallowed
        print(refusal_line(e), file=sys.stderr)
        return 2
    summary = json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))
    am5 = summary.get("am5") or {}
    level = summary.get("dataset_level") or {}
    print(f"artifact written: {out}")
    print(f"cell record written: {record_path(out)}")
    print(f"am5 ({am5.get('rule')}): excluded_count={am5.get('excluded_count')} "
          f"included_count={am5.get('included_count')}")
    print(f"condition {summary['dataset']['condition']}: all_class_miou="
          f"{level.get('all_class_miou')!r} (union-present, "
          f"n_eligible={level.get('all_class_miou_n_eligible')})")
    return 0


def run(args, *, teacher_builder=None, counters=None, started_utc=None) -> Path:
    """One cell, in the clean runner's order. Returns the finalised artifact directory.

    `counters` (an object with `dataset` and `model` lists) records construction, so a smoke can
    prove that a refusal happened before anything was built; `teacher_builder` injects a synthetic
    teacher.
    """
    from src.eval import Condition, DatasetMeta, RunMeta, evaluate_model
    from src.eval.adapters import (DATASET_DOI, DATASET_NAME, PREPROCESS_PROTOCOL,
                                   build_eval_loader, build_expected_manifest_for,
                                   deterministic_subset, load_class_map)
    from src.eval.artifacts import (prepare_artifact_request, validate_artifact_request,
                                    write_artifact)
    from src.eval.eval_runtime import (EVAL_NUM_WORKERS, ModelDeviceForward,
                                       apply_eval_determinism, build_eval_runtime_record,
                                       check_post_eval, determinism_required,
                                       evaluate_capturing_warnings, resolve_model_device)
    from src.eval.model_loading import load_student_checkpoint

    started = started_utc or utc_iso()
    condition = validate_args(args)
    threads = apply_threads(args)
    stage = str(args.stage)
    precision = stage_precision(stage)
    is_teacher = stage == "teacher"
    model_role = "teacher" if is_teacher else "student"
    out_dir, rec_path = check_outputs(args.out_dir)
    class_map = load_class_map()

    # ---- model SOURCE validation, before any cache or dataset work (LE2: hashed before parse) ----
    handed, handed_sha = hash_before_parse(args)
    resolved = resolve_evaluation_source(
        stage, checkpoint=args.checkpoint, provenance=args.provenance,
        expected_sha256=args.teacher_ckpt_sha256 if is_teacher else None)
    require_unchanged(handed, handed_sha)
    quant_backend = None
    if precision in INT8_PRECISIONS:
        from src.eval.model_loading import select_int8_backend
        from src.eval.stage_artifacts import int8_official_error
        if args.artifact_status == "official":
            # LQP ruling R2: the one function an official INT8 path calls (the committed AM-10 list
            # of record, then the PTQ record of record), its text unchanged (evaluate_model.py)
            official_error = int8_official_error(resolved, repo_root=REPO)
            if official_error:
                raise CorruptionCliError(official_error)
        expected_converted = getattr(args, "expected_converted_sha256", None)
        if expected_converted is not None and resolved["artifact_sha256"] != expected_converted:
            _refuse("expected_converted_sha256_mismatch",
                    f"the converted artifact {resolved['artifact_path']} has sha256 "
                    f"{resolved['artifact_sha256']}, expected {expected_converted}; it was not "
                    "loaded")
        quant_backend = select_int8_backend(require_qnnpack=True)
        ckpt_path, ckpt_sha = str(resolved["artifact_path"]), resolved["artifact_sha256"]
    elif is_teacher:
        ckpt_path, ckpt_sha = str(resolved["checkpoint_path"]), resolved["checkpoint_sha256"]
    else:
        ckpt_path, ckpt_sha = str(args.checkpoint), resolved["checkpoint_sha256"]

    # ---- PATH B: the shared cache is proven valid BEFORE any corrupted-image inference ----
    manifest = resolve_corruption_cache(
        args.corruption_cache, args.corruption, args.severity, split=args.split,
        official=args.artifact_status == "official", artifact_status=args.artifact_status,
        expected_sha256=getattr(args, "expected_cache_sha256", None))
    record_sha = check_verification_record(args.corruption_cache, manifest,
                                           getattr(args, "verification_record", None),
                                           artifact_status=args.artifact_status)
    cell = CacheCell(Path(args.corruption_cache).resolve(), manifest, args.corruption,
                     int(args.severity),
                     load=cached_image_loader(args.corruption_cache, manifest, args.corruption,
                                              args.severity))
    cell_entry = cell.entry()

    n_total = EXPECTED_SPLIT_ROWS[args.split]
    source_indices = (deterministic_subset(n_total, args.max_samples)
                      if args.max_samples is not None else list(range(n_total)))
    # Expected manifest from a FILENAME LISTING only -- never from the cache.
    expected_manifest = build_expected_manifest_for(args.split, source_indices)
    check_cache_sources(manifest, expected_manifest, full=args.max_samples is None)

    stamp = started[:19].replace("-", "").replace(":", "") + "Z"
    run_id = args.run_id or f"{stage}_{args.split}_{cell.cell_id}_{stamp}"
    cond = Condition("corruption", args.corruption, int(args.severity))
    request = prepare_artifact_request(
        out_dir=out_dir, artifact_status=args.artifact_status, run_id=run_id,
        run=RunMeta(stage=stage, model_role=model_role, precision=precision,
                    quant_backend=quant_backend, checkpoint_path=ckpt_path,
                    checkpoint_sha256=ckpt_sha, random_init=False, device=args.device),
        dataset=DatasetMeta(name=DATASET_NAME, doi=DATASET_DOI, split=args.split,
                            condition=cond, preprocess_protocol=PREPROCESS_PROTOCOL,
                            expected_rows=len(source_indices)),
        expected_manifest=expected_manifest, class_map=list(class_map.entries), repo_root=REPO)
    provenance = validate_artifact_request(request)     # <-- refusal happens HERE

    # ---- evaluation determinism policy (contract section 10): an FP32 student on CUDA only ----
    cpu_only = is_teacher or precision in INT8_PRECISIONS
    policy_applied = determinism_required(model_role, precision, args.device)
    policy_info = apply_eval_determinism() if policy_applied else None
    ck_info = None

    # ---- only now may the dataset wrapper and the model be constructed ----
    if counters is not None:
        counters.dataset.append(("corrupted", cell.cell_id, tuple(source_indices)))
    dataset = PlantSegCorruptedEvalDataset(args.split, source_indices, cell)
    if counters is not None:
        counters.model.append(("model", stage))
    if is_teacher:
        from src.eval.model_loading import load_teacher_model
        model = load_teacher_model(resolved, builder=teacher_builder,
                                   config_path=args.teacher_config,
                                   expected_sha256=args.teacher_ckpt_sha256)[0]
    elif precision in INT8_PRECISIONS:
        from src.eval.model_loading import load_int8_student
        actual = _sha256_file(resolved["artifact_path"])
        if actual != resolved["artifact_sha256"]:
            _refuse("converted_changed_before_load",
                    f"the converted artifact {resolved['artifact_path']} had sha256 "
                    f"{resolved['artifact_sha256']} at validation and {actual} before the load; it "
                    "was not loaded")
        model = load_int8_student(resolved, require_qnnpack=True)[0]
    else:
        model, ck_info = load_student_checkpoint(args.checkpoint, map_location=args.device)
        if ck_info.sha256 != ckpt_sha:
            _refuse("checkpoint_changed_after_validation",
                    f"checkpoint bytes changed between validation ({ckpt_sha}) and load "
                    f"({ck_info.sha256})", CorruptedEvalError)

    model_device = resolve_model_device(args.device, cpu_only=cpu_only)
    fwd = ModelDeviceForward(model_device)
    loader = build_eval_loader(dataset, args.batch_size, num_workers=EVAL_NUM_WORKERS)
    result, warn_summary = evaluate_capturing_warnings(
        evaluate_model, model, loader, expected_manifest=expected_manifest,
        condition=request.dataset.condition, num_classes=116,
        background_index=0, ignore_index=255, forward=fwd)
    check_post_eval(model=model, model_device=model_device, fwd=fwd,
                    policy_applied=policy_applied)
    runtime = build_eval_runtime_record(
        model=model, model_device=model_device, fwd=fwd, batch_size=args.batch_size,
        forward_batches=result.forward_batches, policy_applied=policy_applied,
        policy_info=policy_info, ckpt_info=ck_info, warn_summary=warn_summary)

    rows = len(result.rows)
    if not dataset.items_verified == dataset.source_digests_verified == rows:
        _refuse("verification_count_mismatch",
                f"{dataset.items_verified} items and {dataset.source_digests_verified} source "
                f"digests verified for {rows} rows; nothing was written", CorruptedEvalError)
    if is_teacher:
        nmf = model.teacher.nmf_stream_state()
        if nmf is not None:
            print(f"NMF stream {nmf['policy']}: {nmf['draws']} draws for {rows} rows "
                  f"(seed {nmf['seed']}, state {nmf['state_sha256'][:16]})")
            if (nmf["policy"], nmf["seed"]) != (M4V_POLICY, M4V_SEED):
                _refuse("nmf_stream_mismatch",
                        f"the teacher's NMF stream is {nmf['policy']} with seed {nmf['seed']}; "
                        f"scoring is {M4V_POLICY} with seed {M4V_SEED}; nothing was written",
                        CorruptedEvalError)
            if nmf["draws"] != rows:
                _refuse("nmf_draws_mismatch",
                        f"the teacher drew {nmf['draws']} NMF bases for {rows} rows; M4-V draws "
                        "once per image at batch size 1; nothing was written", CorruptedEvalError)
    if threads is not None and runtime["torch_num_threads"] != threads:
        _refuse("threads_drifted",
                f"run.eval_runtime.torch_num_threads is {runtime['torch_num_threads']}, --threads "
                f"is {threads}; no artifact was written")
    out = write_artifact(result, request, provenance, eval_runtime=runtime)

    record = build_cell_record(
        out_dir=out, run_id=run_id, stage=stage, condition=cond.as_dict(),
        role=condition["role"], manifest=manifest, cell_sha256=cell_entry["cell_sha256"],
        verification_record_sha256=record_sha, rows=rows, items_verified=dataset.items_verified,
        source_digests_verified=dataset.source_digests_verified,
        evaluator_commit=provenance.repo_commit, started_utc=started, split=args.split)
    write_cell_record(rec_path, record)
    return out


__all__ = ["CorruptionCliError", "M4V_POLICY", "M4V_SEED", "STAGES", "build_parser", "cache_split",
           "cached_image_loader", "check_cache_request", "check_cache_sources", "check_outputs",
           "check_verification_record", "main", "refusal_line", "registered_corruptions",
           "resolve_corruption_cache", "run", "validate_args"]


if __name__ == "__main__":
    raise SystemExit(main())
