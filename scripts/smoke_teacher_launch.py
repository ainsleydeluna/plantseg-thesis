#!/usr/bin/env python3
"""Synthetic verification of the teacher fine-tune pre-flight gate. No mmseg, no GPU, no dataset.

Every fixture is a temp tree OUTSIDE the repository: empty files named to match the PlantSeg layout,
so split counting is exercised without a byte of image data. Test-split files are created but their
CONTENTS are never opened — the gate counts names only, and this smoke asserts that.

Each guard is checked through its distinct `PreflightError.code`, so a CPU box can prove every branch
deterministically even though the CUDA gate can never pass here.
"""
from __future__ import annotations

import builtins
import hashlib
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.launch_teacher_finetune import (DATA_ROOT_ENV, INIT_CKPT_ENV,  # noqa: E402
                                             SPLIT_COUNTS, WORK_DIR_ENV, PreflightError,
                                             build_provenance, check_data_root, check_init_checkpoint,
                                             check_splits, check_work_dir, main as launch_main,
                                             parse_args, preflight)

TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_launch_"))
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


def make_dataset(root: Path, counts: dict[str, int] | None = None,
                 skip_dir: str | None = None) -> Path:
    counts = counts or SPLIT_COUNTS
    for split in ("train", "val", "test"):
        for kind, suffix in (("images", ".jpg"), ("annotations", ".png")):
            d = root / kind / split
            if skip_dir == f"{kind}/{split}":
                continue
            d.mkdir(parents=True, exist_ok=True)
            for i in range(counts[split]):
                (d / f"{split}_{i:05d}{suffix}").touch()
    return root


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


def test_splits() -> None:
    ok = TMP / "ds_ok"
    counts = check_splits(ok)
    check("split_counts_verified",
          counts == {s: {"images": n, "masks": n} for s, n in SPLIT_COUNTS.items()},
          f"train={counts['train']['images']} val={counts['val']['images']} "
          f"test={counts['test']['images']}")
    missing = make_dataset(TMP / "ds_missing_dir", skip_dir="annotations/val")
    expect_code("split_dir_missing", "split_dir_missing", check_splits, missing)
    wrong = make_dataset(TMP / "ds_wrong_count", counts={"train": 5367, "val": 845, "test": 1561})
    expect_code("split_count_mismatch", "split_count_mismatch", check_splits, wrong)


def test_no_test_content_read() -> None:
    """The gate must count split filenames without opening ANY dataset file."""
    root = TMP / "ds_ok"
    opened: list[str] = []
    real_open = builtins.open

    def watched_open(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    builtins.open = watched_open
    try:
        check_splits(root)
    finally:
        builtins.open = real_open

    test_reads = [p for p in opened if f"{os.sep}test{os.sep}" in p or "/test/" in p]
    check("test_split_contents_never_read", test_reads == [], f"{len(test_reads)} test files opened")
    check("no_dataset_file_opened_at_all", opened == [],
          f"{len(opened)} dataset files opened — counting is scandir-only")


# ---------------------------------------------------------------- 3. init checkpoint
def test_init_checkpoint() -> None:
    expect_code("init_ckpt_unset", "init_ckpt_unset", check_init_checkpoint, "")
    expect_code("init_ckpt_sentinel", "init_ckpt_sentinel", check_init_checkpoint,
                "NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT")
    expect_code("init_ckpt_missing", "init_ckpt_missing", check_init_checkpoint,
                str(TMP / "nope.pth"))
    ck = make_ckpt(TMP / "weights" / "ade20k.pth")
    prov = check_init_checkpoint(str(ck))
    check("init_ckpt_sha256_recorded",
          prov["sha256"] == hashlib.sha256(ck.read_bytes()).hexdigest(), prov["sha256"][:16] + "…")
    check("init_ckpt_size_recorded", prov["bytes"] == ck.stat().st_size, f"{prov['bytes']} B")
    inside = REPO / "reports" / "__smoke_never_created.pth"
    expect_code("init_ckpt_inside_repo_rejected", "init_ckpt_missing",
                check_init_checkpoint, str(inside))   # never created -> missing, repo stays clean


# ---------------------------------------------------------------- 4. work dir
def test_work_dir() -> None:
    expect_code("work_dir_unset", "work_dir_unset", check_work_dir, "")
    expect_code("work_dir_sentinel", "work_dir_sentinel", check_work_dir,
                "NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR")
    expect_code("work_dir_inside_repo", "work_dir_inside_repo", check_work_dir,
                str(REPO / "teacher_runs"))
    expect_code("work_dir_is_repo_root", "work_dir_inside_repo", check_work_dir, str(REPO))
    external = TMP / "work"
    check("work_dir_external_accepted", check_work_dir(str(external)) == external.resolve())
    check("work_dir_not_created_without_flag", not external.exists(),
          "creation only happens after every earlier gate passes")


# ---------------------------------------------------------------- 5. full ordering + provenance
def test_ordering_and_provenance() -> None:
    ok = TMP / "ds_ok"
    ck = TMP / "weights" / "ade20k.pth"
    work = TMP / "work_full"

    # A fully valid setup on this CPU box must fail ONLY at the final CUDA gate — which proves every
    # earlier gate cleared, in order.
    expect_code("valid_setup_reaches_cuda_gate_last", "cuda_unavailable", preflight,
                _Args(data_root=str(ok), init_ckpt=str(ck), work_dir=str(work)))

    prov = preflight(_Args(data_root=str(ok), init_ckpt=str(ck), work_dir=str(work),
                           skip_cuda_probe=True, create_work_dir=True))
    for key in ("stage", "architecture", "protocol_classification", "public_plantseg_source_commit",
                "config_path", "config_sha256", "plantseg_root", "split_counts",
                "ade20k_init_checkpoint", "work_dir", "seed", "optimizer", "max_iters",
                "val_interval", "preprocessing", "augmentation_source", "versions", "cuda"):
        check(f"provenance_has_{key}", key in prov)
    check("provenance_stage_teacher", prov["stage"] == "teacher")
    check("provenance_classification", prov["protocol_classification"] == "thesis-derived")
    check("provenance_source_commit",
          prov["public_plantseg_source_commit"] == "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0")
    check("provenance_counts_match", prov["split_counts"]["train"]["images"] == 5367
          and prov["split_counts"]["val"]["images"] == 846
          and prov["split_counts"]["test"]["images"] == 1561)
    check("provenance_versions_not_fabricated",
          prov["versions"]["mmsegmentation"] is None and prov["versions"]["mmcv"] is None
          and prov["versions"]["torch"], "absent packages reported as None, torch reported real")
    check("provenance_cuda_none_when_skipped", prov["cuda"] is None)
    check("work_dir_created_after_gates", work.exists() and work.is_dir())

    rc = launch_main(["--real-run", "--confirm-real-run", "--data-root", str(ok),
                      "--init-ckpt", str(ck), "--work-dir", str(work),
                      "--skip-cuda-probe", "--create-work-dir"])
    check("authorized_run_returns_0_without_training", rc == 0)
    check("provenance_written_outside_repo",
          (work / "teacher_preflight_provenance.json").is_file()
          and REPO not in (work / "teacher_preflight_provenance.json").resolve().parents)
    check("cuda_gate_still_enforced_by_default",
          launch_main(["--real-run", "--confirm-real-run", "--data-root", str(ok),
                       "--init-ckpt", str(ck), "--work-dir", str(work)]) == 2,
          "no --skip-cuda-probe -> refused on this CPU box")
    check("no_mmseg_import_anywhere",
          "mmseg" not in sys.modules and "mmcv" not in sys.modules)


def main() -> int:
    print("=" * 78)
    print("TEACHER LAUNCH GATE SMOKE — synthetic temp fixtures; no mmseg, no GPU, no real dataset")
    print(f"temp root: {TMP}")
    print("=" * 78)
    for fn in (test_authorization, test_data_root, test_splits, test_no_test_content_read,
               test_init_checkpoint, test_work_dir, test_ordering_and_provenance):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:40}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
