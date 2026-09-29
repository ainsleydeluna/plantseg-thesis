#!/usr/bin/env python3
"""Lane 8 d1 on synthetic data: scripts/build_calibration_lists.py. No PlantSeg data, no GPU.

A synthetic dataset (150 TRAIN, 20 VAL and 20 TEST tiny image/mask pairs) is built in a temp dir
outside the repository. The builder must draw the four lists by the AM-10 procedure exactly
(random.Random(seed).sample(sorted(train_ids), 128), = src/quant/calibration.py), record the TRAIN
split-list hash under the L-AM17-STRATA definition, write deterministic bytes, pass its own --check,
keep the seed-42 list acceptable to the committed E4/E7 validator while that validator refuses the
sensitivity lists, and refuse every VAL/TEST path, a missing or extra mask, a wrong TRAIN count and an
existing output. A Python audit hook records every file opened and directory listed while the builder
runs: none may lie under a VAL or TEST split directory.

    python -B scripts/smoke_calibration_lists.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import make_tree, safe_tmpdir  # noqa: E402

BASE = safe_tmpdir("smoke_calib_lists_")
ROOT = BASE / "data"
SPLITS = make_tree(ROOT, n_train=150, n_val=20, n_test=20, seed=1)
os.environ["PLANTSEG_DATA_ROOT"] = str(ROOT)     # before configs/data.py is imported
N_TRAIN = 150
UTC = "2026-10-01T00:00:00Z"
SEEDS = (42, 43, 44, 45)

from scripts.build_calibration_lists import (FILENAME, check_lists, main as build_main,  # noqa: E402
                                             split_list_sha256, train_split_list)
from src.quant.calibration import CalibrationIndexError, build_calibration_index  # noqa: E402
from src.quant.ptq import PTQRefused, load_calibration_list  # noqa: E402
from src.quant.stages import require_shared_calibration_index  # noqa: E402

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


def build(argv) -> tuple[int, str]:
    buf = io.StringIO()
    AUDIT["on"] = True
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = build_main(argv)
    finally:
        AUDIT["on"] = False
    return rc, buf.getvalue()


def eval_split_touched(paths) -> list[str]:
    bad = []
    for p in paths:
        parts = Path(os.path.realpath(p)).parts
        for i in range(len(parts) - 1):
            if parts[i] in ("images", "annotations") and parts[i + 1] in ("val", "test"):
                bad.append(p)
                break
    return bad


def list_doc(d: Path, seed: int) -> dict:
    return json.loads((d / FILENAME.format(seed=seed)).read_text(encoding="utf-8"))


def fresh_tree(name: str, n_train: int = 130) -> Path:
    root = BASE / name
    make_tree(root, n_train=n_train, n_val=2, n_test=2, seed=3)
    return root


# ---------------------------------------------------------------- build + content
def test_build() -> Path:
    out = BASE / "lists_a"
    rc, log = build(["--data-root", str(ROOT), "--out-dir", str(out), "--expected-images", str(N_TRAIN),
                     "--generated-utc", UTC])
    check("build_exit_0", rc == 0, log.strip().splitlines()[-1] if log.strip() else "")
    names = sorted(p.name for p in out.glob("*.json"))
    check("four_lists_written", names == [FILENAME.format(seed=s) for s in SEEDS], str(names))
    train, val, test = set(SPLITS["train"]), set(SPLITS["val"]), set(SPLITS["test"])
    docs = {s: list_doc(out, s) for s in SEEDS}
    for s in SEEDS:
        ids = docs[s]["selected_ids"]
        check(f"seed{s}_128_unique_train_ids", len(ids) == 128 == len(set(ids)) and set(ids) <= train)
        check(f"seed{s}_disjoint_from_val_and_test", not (set(ids) & val) and not (set(ids) & test))
        procedure = random.Random(s).sample(sorted(SPLITS["train"]), 128)
        committed_fn = build_calibration_index(SPLITS["train"], seed=s)["selected_ids"]
        check(f"seed{s}_is_the_am10_procedure_draw", ids == procedure == committed_fn,
              "random.Random(seed).sample(sorted(train_ids), 128), in draw order")
        check(f"seed{s}_checksum_recomputes",
              docs[s]["checksum_sha256"] == hashlib.sha256("\n".join(ids).encode()).hexdigest())
    check("roles_by_seed",
          docs[42]["role"] == "am10_list_of_record"
          and all(docs[s]["role"] == "am16_calibration_sensitivity" for s in (43, 44, 45)))
    check("lists_differ_pairwise", len({tuple(d["selected_ids"]) for d in docs.values()}) == 4)
    check("synthetic_count_marks_smoke", {d["artifact_status"] for d in docs.values()} == {"smoke"})
    # split-list hash: an independent re-implementation of the L-AM17-STRATA definition
    names_sorted = sorted((n for n in os.listdir(ROOT / "images" / "train")
                           if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg")))
    independent = hashlib.sha256("".join(os.path.splitext(n)[0] + "\n" for n in names_sorted)
                                 .encode("utf-8")).hexdigest()
    check("split_list_sha256_is_the_strata_definition",
          {d["split_list_sha256"] for d in docs.values()} == {independent}
          and split_list_sha256(train_split_list(ROOT, N_TRAIN)) == independent, independent[:16])
    check("builder_never_touched_val_or_test", not eval_split_touched(AUDIT["paths"]),
          f"{len(AUDIT['paths'])} paths audited")
    return out


def test_determinism(out_a: Path) -> None:
    out_b = BASE / "lists_b"
    rc, _ = build(["--data-root", str(ROOT), "--out-dir", str(out_b), "--expected-images", str(N_TRAIN),
                   "--generated-utc", UTC])
    same = [(out_a / FILENAME.format(seed=s)).read_bytes() == (out_b / FILENAME.format(seed=s)).read_bytes()
            for s in SEEDS]
    check("two_builds_byte_identical", rc == 0 and all(same), f"{sum(same)}/4 files identical")


def test_validators(out: Path) -> None:
    ok42 = require_shared_calibration_index(out / FILENAME.format(seed=42))
    check("seed42_passes_committed_e4_e7_validator", ok42["seed"] == 42)
    refused = []
    for s in (43, 44, 45):
        try:
            require_shared_calibration_index(out / FILENAME.format(seed=s))
        except CalibrationIndexError:
            refused.append(s)
    check("sensitivity_lists_refused_by_the_official_validator", refused == [43, 44, 45], str(refused))
    loaded = [load_calibration_list(out / FILENAME.format(seed=s))[0]["seed"] for s in SEEDS]
    check("ptq_loader_accepts_all_four", loaded == list(SEEDS))
    try:
        load_calibration_list(out / FILENAME.format(seed=42), expected_checksum="0" * 64)
        check("ptq_loader_refuses_wrong_pin", False, "accepted")
    except PTQRefused as e:
        check("ptq_loader_refuses_wrong_pin", e.code == "calibration_list_checksum_mismatch", e.code)


def test_check_mode(out: Path) -> None:
    rc, log = build(["--check", str(out), "--data-root", str(ROOT), "--expected-images", str(N_TRAIN)])
    check("check_mode_passes", rc == 0 and "d1 PASS (7/7)" in log, log.strip().splitlines()[-1])
    results_, record = check_lists(out, data_root=ROOT, expected_images=N_TRAIN)
    check("check_mode_reports_decision_log_values",
          record.get("seed42_checksum_sha256") == list_doc(out, 42)["checksum_sha256"]
          and len(record.get("files_sha256", {})) == 4)
    # strata cross-check: equal and different split-list hashes
    strata_ok = BASE / "strata_same.json"
    strata_ok.write_text(json.dumps({"split_list_sha256": list_doc(out, 42)["split_list_sha256"]}))
    strata_bad = BASE / "strata_other.json"
    strata_bad.write_text(json.dumps({"split_list_sha256": "0" * 64}))
    rc1, _ = build(["--check", str(out), "--strata", str(strata_ok)])
    rc2, _ = build(["--check", str(out), "--strata", str(strata_bad)])
    check("strata_hash_cross_check", rc1 == 0 and rc2 == 1, f"same={rc1} other={rc2}")
    # tampering: an id swapped for a VAL stem, with the checksum left stale and then recomputed
    tam = BASE / "lists_tampered"
    shutil.copytree(out, tam)
    doc = list_doc(tam, 43)
    doc["selected_ids"][0] = SPLITS["val"][0]
    (tam / FILENAME.format(seed=43)).write_text(json.dumps(doc), encoding="utf-8")
    rc3, _ = build(["--check", str(tam), "--data-root", str(ROOT), "--expected-images", str(N_TRAIN)])
    doc["checksum_sha256"] = hashlib.sha256("\n".join(doc["selected_ids"]).encode()).hexdigest()
    (tam / FILENAME.format(seed=43)).write_text(json.dumps(doc), encoding="utf-8")
    rc4, log4 = build(["--check", str(tam), "--data-root", str(ROOT), "--expected-images", str(N_TRAIN)])
    check("check_catches_a_foreign_id", rc3 == 1 and rc4 == 1 and "[FAIL] L6" in log4 and "[FAIL] L7" in log4,
          f"stale checksum exit {rc3}; recomputed checksum exit {rc4}")
    check("check_never_touched_val_or_test", not eval_split_touched(AUDIT["paths"]))


def test_refusals() -> None:
    def refused(label: str, argv, out: Path | None = None) -> None:
        rc, log = build(argv)
        wrote = out is not None and out.exists() and any(out.iterdir())
        check(label, rc == 2 and not wrote, f"exit {rc}; {log.strip().splitlines()[-1][:90] if log.strip() else ''}")

    refused("refuses_wrong_train_count", ["--data-root", str(ROOT), "--out-dir", str(BASE / "o1")],
            BASE / "o1")
    latest = fresh_tree("latest_copy")            # contains "test": refused as scripts/build_train_strata.py
    refused("refuses_a_test_path", ["--data-root", str(latest), "--out-dir", str(BASE / "o2"),
                                    "--expected-images", "130"], BASE / "o2")
    under_val = BASE / "val" / "data"
    make_tree(under_val, n_train=130, seed=4)
    refused("refuses_a_val_component", ["--data-root", str(under_val), "--out-dir", str(BASE / "o3"),
                                        "--expected-images", "130"], BASE / "o3")
    linked = BASE / "linked"
    make_tree(linked, n_train=0, n_val=130, seed=5)
    for sub in ("images", "annotations"):
        os.symlink(linked / sub / "val", linked / sub / "train")
    refused("refuses_train_symlinked_to_val", ["--data-root", str(linked), "--out-dir", str(BASE / "o4"),
                                               "--expected-images", "130"], BASE / "o4")
    nomask = fresh_tree("nomask")
    (nomask / "annotations" / "train" / "plant_leaf_0005.png").unlink()
    refused("refuses_missing_mask", ["--data-root", str(nomask), "--out-dir", str(BASE / "o5"),
                                     "--expected-images", "130"], BASE / "o5")
    extra = fresh_tree("extramask")
    shutil.copy(extra / "annotations" / "train" / "plant_leaf_0001.png",
                extra / "annotations" / "train" / "plant_leaf_9999.png")
    refused("refuses_extra_mask", ["--data-root", str(extra), "--out-dir", str(BASE / "o6"),
                                   "--expected-images", "130"], BASE / "o6")
    before = {p.name: p.read_bytes() for p in (BASE / "lists_a").iterdir()}
    rc, _ = build(["--data-root", str(ROOT), "--out-dir", str(BASE / "lists_a"),
                   "--expected-images", str(N_TRAIN)])
    after = {p.name: p.read_bytes() for p in (BASE / "lists_a").iterdir()}
    check("refuses_to_overwrite", rc == 2 and before == after)
    check("refusals_never_touched_val_or_test", not eval_split_touched(AUDIT["paths"]))
    check("nothing_written_in_the_repository", not (REPO / "configs" / "calibration").exists())


def main() -> int:
    print("=" * 78)
    print("CALIBRATION LISTS SMOKE (lane 8 d1) -- synthetic; no PlantSeg data, no GPU")
    print(f"temp: {BASE}")
    print("=" * 78)
    out = test_build()
    test_determinism(out)
    test_validators(out)
    test_check_mode(out)
    test_refusals()
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
