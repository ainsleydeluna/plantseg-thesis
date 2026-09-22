#!/usr/bin/env python3
"""Synthetic verification of the teacher fine-tune pre-flight gate. No GPU, no real dataset.

Every fixture is a temp tree OUTSIDE the repository: empty files named to match the PlantSeg layout,
so TRAIN/VAL counting is exercised without a byte of image data. M11 (B60 §5): the staged root holds
TRAIN and VAL only; TEST surfaces (`images/test`, `annotations/test`, `annotation_test.json`) are
checked for EXISTENCE ONLY. The negative fixtures plant them with a trap file inside, and this smoke
asserts that nothing under them is ever listed or opened.

Each guard is checked through its distinct `PreflightError.code`, so a CPU box can prove every branch
deterministically even though the CUDA gate can never pass here. Runs on the host (no mmseg) and in the
teacher image (mmseg installed): version provenance is checked against what is ACTUALLY installed.

The ADE20K SHA-256 gate is always exercised unpatched with a synthetic checkpoint (it must refuse).
Acceptance of the real checkpoint is proven when SMOKE_ADE20K_CKPT points at it (read-only).
Positive-path provenance checks on synthetic fixtures patch the expected SHA in-process, test-only.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.launch_teacher_finetune as L  # noqa: E402
from scripts.launch_teacher_finetune import (PreflightError, check_data_root,  # noqa: E402
                                             check_init_checkpoint, check_splits, check_work_dir,
                                             main as launch_main, preflight)

TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_launch_"))
REAL_CKPT = os.environ.get("SMOKE_ADE20K_CKPT")
TEST_SURFACES = ("images/test", "annotations/test", "annotation_test.json")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def expect_code(name: str, code: str, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
        check(name, False, "no PreflightError raised")
    except PreflightError as e:
        check(name, e.code == code, f"{e.code}" + ("" if e.code == code else f" != {code}"))
    except Exception as e:  # noqa: BLE001
        check(name, False, f"wrong exception {type(e).__name__}: {e}")


def make_dataset(root: Path, counts: dict[str, int] | None = None, skip_dir: str | None = None) -> Path:
    """A TRAIN/VAL-only staged root (M11)."""
    counts = counts or L.SPLIT_COUNTS
    for split in ("train", "val"):
        for kind, suffix in (("images", ".jpg"), ("annotations", ".png")):
            d = root / kind / split
            if skip_dir == f"{kind}/{split}":
                continue
            d.mkdir(parents=True, exist_ok=True)
            for i in range(counts[split]):
                (d / f"{split}_{i:05d}{suffix}").touch()
    return root


def plant(root: Path, surface: str) -> Path:
    target = root.joinpath(*surface.split("/"))
    if surface.endswith(".json"):
        target.touch()
    else:
        target.mkdir(parents=True)
        (target / "trap_00000.jpg").touch()
    return target


def make_ckpt(path: Path, payload: bytes = b"synthetic-ade20k-checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _Args:
    def __init__(self, **kw):
        defaults = dict(config=None, data_root=None, init_ckpt=None, work_dir=None,
                        create_work_dir=False, skip_cuda_probe=False)
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class patched_expected_sha:
    """Test-only: accept a synthetic checkpoint by patching the launcher's expected SHA in-process."""

    def __init__(self, sha: str):
        self.sha = sha

    def __enter__(self):
        self.saved = L.EXPECTED_ADE20K_SHA256
        L.EXPECTED_ADE20K_SHA256 = self.sha

    def __exit__(self, *exc):
        L.EXPECTED_ADE20K_SHA256 = self.saved


# ---------------------------------------------------------------- 1. authorization
def test_authorization() -> None:
    check("no_flags_exit_2", launch_main([]) == 2)
    check("real_run_alone_exit_2", launch_main(["--real-run"]) == 2)
    check("confirm_alone_exit_2", launch_main(["--confirm-real-run"]) == 2)
    check("mmseg_not_imported_on_blocked_path",
          "mmseg" not in sys.modules and "mmcv" not in sys.modules)


# ---------------------------------------------------------------- 2. sentinels / data root
def test_data_root() -> None:
    expect_code("data_root_unset", "data_root_unset", check_data_root, "")
    expect_code("data_root_sentinel", "data_root_sentinel", check_data_root,
                "NEED_TO_CONFIRM__SET_PLANTSEG_DATA_ROOT")
    expect_code("data_root_missing", "data_root_missing", check_data_root, str(TMP / "absent"))
    good = make_dataset(TMP / "ds_ok")
    check("data_root_resolves", check_data_root(str(good)) == good.resolve())


# ---------------------------------------------------------------- 3. M11 TRAIN/VAL-only isolation
def test_splits() -> None:
    iso = check_splits(TMP / "ds_ok")
    check("split_counts_verified_trainval_only",
          iso["counts"] == {s: {"images": n, "masks": n} for s, n in L.SPLIT_COUNTS.items()}
          and set(iso["counts"]) == {"train", "val"},
          f"train={iso['counts']['train']['images']} val={iso['counts']['val']['images']}")
    check("split_counts_has_no_test_entry", "test" not in L.SPLIT_COUNTS)
    check("test_surfaces_recorded_absent",
          iso["test_surfaces"] == {s: "absent" for s in TEST_SURFACES}, str(iso["test_surfaces"]))
    missing = make_dataset(TMP / "ds_missing_dir", skip_dir="annotations/val")
    expect_code("split_dir_missing", "split_dir_missing", check_splits, missing)
    wrong = make_dataset(TMP / "ds_wrong_count", counts={"train": 5367, "val": 845})
    expect_code("split_count_mismatch", "split_count_mismatch", check_splits, wrong)


def test_test_surfaces_fail_closed_without_enumeration() -> None:
    """Each exact TEST surface fails closed; nothing under it is listed or opened."""
    root = TMP / "ds_ok"
    listed: list[str] = []
    opened: list[str] = []
    real_scandir, real_listdir, real_open = os.scandir, os.listdir, builtins.open

    def spy_scandir(p="."):
        listed.append(str(p))
        return real_scandir(p)

    def spy_listdir(p="."):
        listed.append(str(p))
        return real_listdir(p)

    def spy_open(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    for surface in TEST_SURFACES:
        target = plant(root, surface)
        os.scandir, os.listdir, builtins.open = spy_scandir, spy_listdir, spy_open
        try:
            expect_code(f"m11_{surface.replace('/', '_').replace('.', '_')}_fails_closed",
                        "test_split_present", check_splits, root)
        finally:
            os.scandir, os.listdir, builtins.open = real_scandir, real_listdir, real_open
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    touched = [p for p in listed + opened if "test" in Path(p).name]
    check("m11_test_contents_never_listed_or_opened", touched == [], str(touched))

    os.scandir, os.listdir, builtins.open = spy_scandir, spy_listdir, spy_open
    try:
        check_splits(root)
    finally:
        os.scandir, os.listdir, builtins.open = real_scandir, real_listdir, real_open
    check("no_dataset_file_opened_at_all", opened == [],
          f"{len(opened)} dataset files opened — counting is scandir-only")


# ---------------------------------------------------------------- 4. init checkpoint
def test_init_checkpoint() -> None:
    import src.training.train_e1 as e1
    expect_code("init_ckpt_unset", "init_ckpt_unset", check_init_checkpoint, "")
    expect_code("init_ckpt_sentinel", "init_ckpt_sentinel", check_init_checkpoint,
                "NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT")
    expect_code("init_ckpt_missing", "init_ckpt_missing", check_init_checkpoint, str(TMP / "nope.pth"))
    check("expected_sha_is_readiness_value",
          L.EXPECTED_ADE20K_SHA256 == "647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1")
    ck = make_ckpt(TMP / "weights" / "ade20k.pth")
    expect_code("wrong_sha_checkpoint_rejected", "init_ckpt_sha_mismatch", check_init_checkpoint, str(ck))

    # A checkpoint INSIDE the repository must be refused. The E1 guard's REPO is pointed, in-process,
    # at a temp fake repo so a real in-repo file exists without writing into the actual repository.
    fake_repo = TMP / "fake_repo"
    inside = make_ckpt(fake_repo / "weights" / "ade20k.pth")
    saved = e1.REPO
    e1.REPO = fake_repo.resolve()
    try:
        expect_code("in_repo_checkpoint_rejected", "init_ckpt_inside_repo",
                    check_init_checkpoint, str(inside))
    finally:
        e1.REPO = saved

    synthetic_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    with patched_expected_sha(synthetic_sha):
        prov = check_init_checkpoint(str(ck))
    check("init_ckpt_sha256_verified_and_recorded",
          prov["sha256"] == synthetic_sha and prov["sha256_verified"] is True, prov["sha256"][:16] + "…")
    check("init_ckpt_size_recorded", prov["bytes"] == ck.stat().st_size, f"{prov['bytes']} B")
    check("init_identity_recorded",
          prov["expected_filename"] == L.ADE20K_INIT_FILENAME
          and prov["stock_config"]["name"] == L.ADE20K_INIT_CONFIG
          and "conv_seg" in prov["classifier_only_rule"])

    if REAL_CKPT:
        real = check_init_checkpoint(REAL_CKPT)          # unpatched: the real readiness SHA
        check("real_external_checkpoint_accepted",
              real["sha256"] == L.EXPECTED_ADE20K_SHA256 and real["sha256_verified"] is True,
              real["sha256"][:16] + "…")
        stock = real["stock_config"]
        mmseg_present = importlib.util.find_spec("mmseg") is not None
        check("stock_config_identity_environment_correct",
              (stock["sha256"] == "33dcb71bcbf41bbdd6e2da2cfa399f84587b4edffb8de39c147d3e58178bc74b")
              if mmseg_present else (stock["path"] is None and stock["sha256"] is None),
              f"mmseg {'present' if mmseg_present else 'absent'}: {stock['sha256']}")


# ---------------------------------------------------------------- 5. work dir
def test_work_dir() -> None:
    expect_code("work_dir_unset", "work_dir_unset", check_work_dir, "")
    expect_code("work_dir_sentinel", "work_dir_sentinel", check_work_dir,
                "NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR")
    expect_code("work_dir_inside_repo", "work_dir_inside_repo", check_work_dir, str(REPO / "teacher_runs"))
    expect_code("work_dir_is_repo_root", "work_dir_inside_repo", check_work_dir, str(REPO))
    external = TMP / "work"
    check("work_dir_external_accepted", check_work_dir(str(external)) == external.resolve())
    check("work_dir_not_created_without_flag", not external.exists(),
          "creation only happens after every earlier gate passes")


# ---------------------------------------------------------------- 6. governed-clean launch gate
def test_governed_gate() -> None:
    pdf = b" M docs/reference/reference.pdf\x00"
    check("governed_gate_ignores_protected_pdf", L.governed_violations(pdf) == [])
    check("governed_gate_accepts_clean", L.check_governed_clean(pdf)["governed_paths_clean"] is True)
    for dirty in (b" M src/training/teacher_components.py\x00", b" M configs/teacher/x.py\x00",
                  b"?? scripts/new_file.py\x00", b" M requirements-e1.txt\x00",
                  b" M docs/EVALUATION_CONTRACT.md\x00"):
        name = dirty.decode().strip("\x00 ?M").replace("/", "_").replace(".", "_")[:40]
        expect_code(f"governed_gate_blocks_{name}", "governed_paths_dirty", L.check_governed_clean,
                    pdf + dirty)
    check("governed_gate_ignores_nongoverned_docs",
          L.governed_violations(pdf + b" M docs/open_questions.md\x00") == [])


# ---------------------------------------------------------------- 7. full ordering + provenance
def test_ordering_and_provenance() -> None:
    ok = TMP / "ds_ok"
    ck = TMP / "weights" / "ade20k.pth"
    work = TMP / "work_full"
    synthetic_sha = hashlib.sha256(ck.read_bytes()).hexdigest()

    with patched_expected_sha(synthetic_sha):
        # A fully valid setup on this CPU box must fail ONLY at the final CUDA gate — which proves
        # every earlier gate cleared, in order.
        expect_code("valid_setup_reaches_cuda_gate_last", "cuda_unavailable", preflight,
                    _Args(data_root=str(ok), init_ckpt=str(ck), work_dir=str(work)))
        prov = preflight(_Args(data_root=str(ok), init_ckpt=str(ck), work_dir=str(work),
                               skip_cuda_probe=True, create_work_dir=True))
        rc = launch_main(["--real-run", "--confirm-real-run", "--data-root", str(ok),
                          "--init-ckpt", str(ck), "--work-dir", str(work),
                          "--skip-cuda-probe", "--create-work-dir"])
        cuda_enforced = launch_main(["--real-run", "--confirm-real-run", "--data-root", str(ok),
                                     "--init-ckpt", str(ck), "--work-dir", str(work)]) == 2
    for key in ("stage", "architecture", "protocol_classification", "public_plantseg_source_commit",
                "config_path", "config_sha256", "plantseg_root", "split_counts", "test_isolation",
                "ade20k_init_checkpoint", "work_dir", "seed", "optimizer", "max_iters", "val_interval",
                "checkpoint_interval", "locked_policy", "versions", "cuda"):
        check(f"provenance_has_{key}", key in prov)
    check("provenance_stage_teacher", prov["stage"] == "teacher")
    check("provenance_classification", prov["protocol_classification"] == "thesis-derived")
    check("provenance_source_commit",
          prov["public_plantseg_source_commit"] == "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0")
    check("provenance_counts_trainval_only",
          prov["split_counts"] == {"train": {"images": 5367, "masks": 5367},
                                   "val": {"images": 846, "masks": 846}})
    check("provenance_m5_cadence", prov["val_interval"] == 4000 and prov["checkpoint_interval"] == 4000
          and prov["max_iters"] == 40000)
    check("provenance_locked_policy_complete",
          sorted(prov["locked_policy"]) == sorted(["M2_augmentation", "M3_scale", "M4_nmf",
                                                    "M5_schedule", "M11_isolation", "M12_selection",
                                                    "M13_loss"]))
    import torch
    for pkg in ("mmsegmentation", "mmcv", "mmengine"):
        try:
            installed = dist_version(pkg)
        except PackageNotFoundError:
            installed = None
        check(f"provenance_version_{pkg}_environment_correct", prov["versions"][pkg] == installed,
              f"recorded {prov['versions'][pkg]!r}, installed {installed!r}")
    check("provenance_version_torch_real", prov["versions"]["torch"] == torch.__version__)
    check("provenance_cuda_none_when_skipped", prov["cuda"] is None)
    check("work_dir_created_after_gates", work.exists() and work.is_dir())
    check("authorized_run_returns_0_without_training", rc == 0)
    check("provenance_written_outside_repo",
          (work / "teacher_preflight_provenance.json").is_file()
          and REPO not in (work / "teacher_preflight_provenance.json").resolve().parents)
    check("cuda_gate_still_enforced_by_default", cuda_enforced,
          "no --skip-cuda-probe -> refused on this CPU box")
    check("no_mmseg_import_anywhere", "mmseg" not in sys.modules and "mmcv" not in sys.modules)


# ---------------------------------------------------------------- 8. locked config (teacher stack only)
def test_locked_config() -> None:
    if importlib.util.find_spec("mmengine") is None or importlib.util.find_spec("mmseg") is None:
        print("  (teacher stack absent: the merged-config lock check runs in the teacher image)")
        return
    from mmengine.config import Config
    os.chdir(REPO)
    cfg = Config.fromfile(str(L.DEFAULT_CONFIG))
    problems = L.check_locked_config(cfg)
    check("locked_config_has_every_m_lock", problems == [], "; ".join(problems))
    bad = Config.fromfile(str(L.DEFAULT_CONFIG))
    bad.default_hooks.checkpoint.save_best = "mIoU"
    bad.train_cfg.val_interval = 10000
    bad.model.decode_head.loss_decode.avg_non_ignore = False
    bad.test_cfg = dict(type="TestLoop")
    check("locked_config_negative_control_detected", len(L.check_locked_config(bad)) >= 4,
          f"{len(L.check_locked_config(bad))} problems")


def main() -> int:
    print("=" * 78)
    print("TEACHER LAUNCH GATE SMOKE — synthetic temp fixtures; no GPU, no real dataset")
    print(f"temp root: {TMP} | real checkpoint: {REAL_CKPT or 'not provided'}")
    print("=" * 78)
    for fn in (test_authorization, test_data_root, test_splits,
               test_test_surfaces_fail_closed_without_enumeration, test_init_checkpoint,
               test_work_dir, test_governed_gate, test_ordering_and_provenance, test_locked_config):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:48}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
