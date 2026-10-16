#!/usr/bin/env python3
"""Build the frozen corruption cache from split image files. Generation only — never evaluation.

Torch-free: src.corruption_cache, src.corruption_source, src.corruption_roles and the vendored
imagecorruptions. The cache is generated and verified in the student image b80b645d on the session
host (TL-1); nothing here chooses a device or reads a model (LC2).

  --print-pin --split {val,test}      print the generator pin; reads no data; clean governed paths
  --real-run --confirm-real-run --split val --cache-status {smoke,provisional}
      --data-root R --cache-dir C [--expected-pin-sha256 P] [--max-images N] [--workers N]
      [--resume]
  --real-run --confirm-real-run --split test --confirm-test-split --cache-status official
      --data-root R --cache-dir .../corr_cache_tday --expected-pin-sha256 P [--workers N]
      [--resume]
  --real-run --confirm-real-run --repair IMAGE_ID CORRUPTION SEVERITY --fault-report F
      --split S [--confirm-test-split] --data-root R --cache-dir C
      (an official or provisional cache: the generator's stack, image, governed-path and pin
      gates, against the cache's own pin; the source must be the recorded one)

Split and status go together: test is official and official is test; val is provisional or smoke;
--max-images (a deterministic subset) is smoke only. An official or provisional cache needs
--expected-pin-sha256, the pinned corruption stack (requirements.lock), PLANTSEG_IMAGE_DIGEST
naming the student image, clean governed paths, and the pin recomputed at this commit equal to the
expected one.

Every gate refuses by code before anything is written, in this order: the run flags; --cache-dir;
names (no "test" in the absolute cache root, journal or repair paths; LC6); the TEST root's name
corr_cache_tday; --confirm-test-split; split and status; --max-images; the expected pin's presence
and format; the vendored runtime; the pinned stack; the image digest; the severity-5 evidence gate;
git and the governed paths; the pin; the cache location; the cache root's state (LC5); the data
root (the M11 isolation check for val, images/test alone for test); the listing (its count, then
its stems: no "test" in any case, no TL section 2 character). Then the cache is generated, or
resumed from its journal, and its manifest written last.

Exit 0: done. Exit 2: refused. Exit 3: a STOP code (nondeterministic_item, repair_digest_mismatch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import corruption_cache as cc  # noqa: E402
from src import corruption_source as cs  # noqa: E402
from src.corruption_cache import CorruptionCacheError  # noqa: E402
from src.corruption_source import CorruptionSourceError  # noqa: E402
from src.vendor.imagecorruptions.provenance import (VendorIntegrityError,  # noqa: E402
                                                    verify_vendor_integrity)

STOP_CODES = ("nondeterministic_item", "repair_digest_mismatch")
PINNED_STATUSES = ("official", "provisional")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate the frozen PlantSeg corruption cache.")
    p.add_argument("--real-run", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--print-pin", action="store_true",
                   help="print the generator pin for --split and exit; reads no data")
    p.add_argument("--split", choices=list(cs.SPLITS), default=None)
    p.add_argument("--confirm-test-split", action="store_true")
    p.add_argument("--cache-status", choices=list(cc.CACHE_STATUSES), default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--expected-pin-sha256", default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--repair", nargs=3, metavar=("IMAGE_ID", "CORRUPTION", "SEVERITY"),
                   default=None)
    p.add_argument("--fault-report", default=None)
    return p


def _refuse(code: str, message: str):
    raise CorruptionCacheError(code, message)


# ---------------------------------------------------------------- gates, in order
def gate_cache_dir(args) -> None:
    if not args.cache_dir:
        _refuse("cache_dir_unset", "--cache-dir is required")


def gate_names(args) -> None:
    """LC6: a string test on every path this run would create, before any filesystem call."""
    root = Path(args.cache_dir)
    paths = {"cache_root": args.cache_dir, "journal_path": str(cc.journal_path(root))}
    if args.repair is not None:
        paths["repair_record_dir"] = str(root / "repair")
        paths["fault_report"] = args.fault_report
    cc.refuse_test_named(**paths)


def gate_tday_root(args) -> None:
    if args.split == "test" and Path(args.cache_dir).name != cc.TDAY_CACHE_ROOT_NAME:
        _refuse("tday_cache_root_required",
                f"the TEST cache root's last component must be {cc.TDAY_CACHE_ROOT_NAME!r}, got "
                f"{Path(args.cache_dir).name!r}")


def gate_test_confirm(args) -> None:
    if args.split == "test" and not args.confirm_test_split:
        _refuse("test_split_unconfirmed",
                "--split test requires --confirm-test-split (the TEST split is not touched "
                "casually)")


def gate_test_official(args) -> None:
    if args.split == "test" and args.cache_status != "official":
        _refuse("test_split_requires_official",
                f"a TEST cache is official only, got --cache-status {args.cache_status}")


def gate_official_test(args) -> None:
    if args.cache_status == "official" and args.split != "test":
        _refuse("official_requires_test_split",
                f"an official cache is the TEST cache, got --split {args.split}")


def gate_max_images(args) -> None:
    if args.max_images is not None and args.cache_status != "smoke":
        _refuse("max_images_requires_smoke",
                f"--max-images makes a capped smoke cache; refused for {args.cache_status}")


def gate_expected_pin_present(args) -> None:
    if args.cache_status in PINNED_STATUSES and not args.expected_pin_sha256:
        _refuse("expected_pin_required",
                f"a {args.cache_status} cache needs --expected-pin-sha256 (print it with "
                "--print-pin at the frozen commit)")


def gate_expected_pin_format(args) -> None:
    if args.expected_pin_sha256 is not None and \
            not cc._HEX64.fullmatch(args.expected_pin_sha256):
        _refuse("expected_pin_format",
                "--expected-pin-sha256 must be 64 lowercase hex characters")


def gate_vendor(args) -> None:
    try:
        verify_vendor_integrity()
    except VendorIntegrityError as e:
        _refuse("vendor_checksum_mismatch", str(e))


def gate_environment(args) -> None:
    if args.cache_status in PINNED_STATUSES:
        cc.require_official_dependency_environment()


def gate_image_digest(args) -> None:
    if args.cache_status in PINNED_STATUSES:
        got = os.environ.get("PLANTSEG_IMAGE_DIGEST")
        if got != cc.STUDENT_IMAGE_DIGEST:
            _refuse("image_digest_mismatch",
                    f"PLANTSEG_IMAGE_DIGEST is {got!r}; a {args.cache_status} cache is generated "
                    f"in the student image {cc.STUDENT_IMAGE_DIGEST}")


def gate_severity5(args, repo_root) -> None:
    if args.cache_status == "official":
        cc.require_severity5_validated("official", cc.full_grid(cc.load_roles(repo_root)))


def gate_governed(args, repo_root) -> dict:
    if args.cache_status in PINNED_STATUSES:
        return cc.require_governed_clean(repo_root)
    raw = cc.governed_porcelain(repo_root)
    return {"governed_paths_clean": not raw, "worktree_state_sha256": cc._sha256_bytes(raw)}


def pin_matches(pin: dict, expected: str) -> bool:
    return pin["pin_sha256"] == expected


def gate_pin(args, repo_root) -> dict:
    pin = cc.build_pin(args.split, repo_root=repo_root)
    if args.expected_pin_sha256 is not None and not pin_matches(pin, args.expected_pin_sha256):
        _refuse("pin_mismatch",
                f"the pin at this commit is {pin['pin_sha256']}, not the expected "
                f"{args.expected_pin_sha256}")
    return pin


def gate_cache_location(args, repo_root) -> None:
    out = Path(args.cache_dir).resolve()
    for r in {REPO.resolve(), Path(repo_root).resolve()}:
        if out == r or r in out.parents:
            _refuse("cache_inside_repo",
                    "the corruption cache is never written inside the repository")


def gate_cache_state(args) -> None:
    cc.check_cache_root_state(Path(args.cache_dir), resume=args.resume)


def gate_data_root(args) -> dict:
    return cs.check_split_surface(args.data_root, args.split)


def gate_listing(args) -> list:
    sources = cs.list_split(args.data_root, args.split)
    if args.max_images is not None:
        sources = [sources[i] for i in cs.deterministic_subset(len(sources), args.max_images)]
    return sources


# ---------------------------------------------------------------- commands
def generate_cmd(args, repo_root) -> dict:
    gate_cache_dir(args)
    gate_names(args)
    gate_tday_root(args)
    gate_test_confirm(args)
    gate_test_official(args)
    gate_official_test(args)
    gate_max_images(args)
    gate_expected_pin_present(args)
    gate_expected_pin_format(args)
    gate_vendor(args)
    gate_environment(args)
    gate_image_digest(args)
    gate_severity5(args, repo_root)
    worktree = gate_governed(args, repo_root)
    pin = gate_pin(args, repo_root)
    gate_cache_location(args, repo_root)
    gate_cache_state(args)
    gate_data_root(args)
    sources = gate_listing(args)
    expected = None if args.max_images is not None else cs.split_sizes()[args.split]
    return cc.build_cache(args.cache_dir, sources, split=args.split, status=args.cache_status,
                          pin=pin, expected_images=expected, workers=args.workers,
                          resume=args.resume, repo_root=repo_root,
                          provenance={"repo_commit": pin["commit"], **worktree})


def gate_repair_split(args, manifest: dict) -> None:
    if manifest.get("split") != args.split:
        _refuse("repair_split_mismatch",
                f"the cache holds split {manifest.get('split')!r}, --split is {args.split!r}")


def gate_repair_pinned(args, manifest: dict, repo_root) -> None:
    """LC8, TL-1: an official or provisional item is regenerated where its cache was made: the
    pinned stack, the student image, clean governed paths and the cache's own pin. A different
    environment is refused here, before any regeneration, so repair_digest_mismatch stays the STOP
    for true nondeterminism."""
    status = manifest.get("status")
    if status not in PINNED_STATUSES:
        return
    probe = argparse.Namespace(cache_status=status)
    gate_environment(probe)
    gate_image_digest(probe)
    gate_governed(probe, repo_root)
    pin = cc.build_pin(manifest["split"], repo_root=repo_root)
    recorded = (manifest.get("pin") or {}).get("pin_sha256")
    if not pin_matches(pin, recorded):
        _refuse("pin_mismatch",
                f"the pin at this commit is {pin['pin_sha256']}; the cache was generated under "
                f"{recorded}, and an item is repaired only by the generator that made it")


def repair_cmd(args, repo_root) -> dict:
    gate_cache_dir(args)
    gate_names(args)
    gate_test_confirm(args)
    gate_vendor(args)
    cc.require_fault_report(Path(args.fault_report) if args.fault_report else Path(""))
    manifest = cc.load_manifest(args.cache_dir)
    gate_repair_split(args, manifest)
    gate_repair_pinned(args, manifest, repo_root)
    gate_data_root(args)
    image_id, corruption, severity = args.repair
    by_id = {s.image_id: s for s in cs.list_split(args.data_root, args.split)}
    if image_id not in by_id:
        _refuse("cache_entry_missing", f"{image_id} is not a source of split {args.split}")
    return cc.repair_item(args.cache_dir, by_id[image_id], corruption, int(severity),
                          fault_report=args.fault_report)


def gate_print_pin_clean(repo_root) -> None:
    """The pin names HEAD and hashes the working-tree script and protocol, so it is printed only
    from clean governed paths: a pin printed from edited files would never match at its commit."""
    cc.require_governed_clean(repo_root)


def print_pin(args, repo_root) -> int:
    try:
        gate_vendor(args)
        gate_print_pin_clean(repo_root)
        pin = cc.build_pin(args.split, repo_root=repo_root)
    except (CorruptionCacheError, CorruptionSourceError) as e:
        print(f"PIN REFUSED [{e.code}]: {e}", file=sys.stderr)
        return 2
    print(json.dumps(pin, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def main(argv=None, *, repo_root=REPO) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.split is None:
        parser.error("--split is required")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.repair is not None and not args.repair[2].isdigit():
        parser.error("--repair SEVERITY must be an integer")
    if args.print_pin:
        return print_pin(args, repo_root)
    if not (args.real_run and args.confirm_real_run):
        print("REFUSING: cache generation requires --real-run --confirm-real-run. Nothing was "
              "read or written.", file=sys.stderr)
        return 2
    if args.data_root is None:
        parser.error("--data-root is required")
    if args.repair is None and args.cache_status is None:
        parser.error("--cache-status is required")
    try:
        if args.repair is not None:
            record = repair_cmd(args, repo_root)
            print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        manifest = generate_cmd(args, repo_root)
    except (CorruptionCacheError, CorruptionSourceError) as e:
        print(f"CACHE GENERATION FAILED [{e.code}]: {e}", file=sys.stderr)
        return 3 if e.code in STOP_CODES else 2
    print(json.dumps({"cache_dir": str(Path(args.cache_dir).resolve()),
                      "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
                      "inferential_cells_sha256": manifest["inferential_cells_sha256"],
                      "status": manifest["status"], "split": manifest["split"],
                      "source_image_count": manifest["source_image_count"],
                      "items": len(manifest["entries"]),
                      "floor_lifted": manifest["floor_lifted"],
                      "generation_items": manifest["generation"]["items"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
