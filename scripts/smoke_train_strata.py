#!/usr/bin/env python3
"""L-AM17-STRATA smoke (docs/lane_specs/part1.md lane 5 (d) d1; NONOFFICIAL, synthetic masks only).

  S1  d1 unit: 5 synthetic masks x 6 disease classes (plus background and two ignore pixels) ->
      hand-counted pixels and images per class; shares are the exact fractions; ranks descend by share
      with the class-id tie-break (classes 2 and 3 tie at 10 pixels); terciles 2/2/2; rare with
      threshold 2 (the boundary: 2 images is not rare); a class with no pixel is ranked last and flagged.
  S2  the CLI end to end (subprocess): two runs with a pinned --generated-utc write bitwise-identical
      JSON and CSV; unpinned runs differ only in generated_utc; the JSON carries the lane 5 (c) fields,
      the split-list hash of the five stems and the CSV hash; an existing output is never overwritten.
  S3  refusals, exit 2 and nothing written: a data root containing "test", a TRAIN stem containing
      "test", an images/train symlink resolving into a "test" directory (and no mask is read before the
      refusal); a missing mask, an extra mask, a count other than expected (the registered 5,367 by
      default), a label outside the class space and an RGB mask.
  S4  --check (d2) on synthetic documents: the 5-mask build is refused as not the registered build; a
      registered-shape 115-class document built from reports/e1_class_weights.json's pixel counts
      passes all ten checks (Spearman rho 1.0, pixel counts equal the class-weight file); perturbed
      shares, terciles, a missing class and an unclean script commit each fail.

Run:  python -B scripts/smoke_train_strata.py
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402

SCRIPT = REPO / "scripts" / "build_train_strata.py"
UTC = "2026-09-28T00:00:00Z"
SYN = ["--num-classes", "7", "--rare-threshold", "2", "--expected-images", "5"]
CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))


def load_module():
    spec = importlib.util.spec_from_file_location("build_train_strata", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cli(argv):
    p = subprocess.run([sys.executable, "-B", str(SCRIPT), *argv], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def run_in_process(mod, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = mod.main(argv)
    return rc, buf.getvalue()


# --------------------------------------------------------------------------------------------------
# d1 fixture: 4x5 masks, background 0, diseases 1-6, ignore 255
# --------------------------------------------------------------------------------------------------
MASKS = {                                   # stem -> [(value, pixel count)], background fills the rest
    "syn_a": [(1, 6), (2, 3)],
    "syn_b": [(1, 4), (3, 5)],
    "syn_c": [(2, 7), (4, 2)],
    "syn_d": [(3, 5), (5, 1), (255, 2)],
    "syn_e": [(1, 2), (4, 3)],
}
IMAGE_NAMES = {"syn_a": "syn_a.jpg", "syn_b": "syn_b.jpg", "syn_c": "syn_c.jpeg", "syn_d": "syn_d.jpg",
               "syn_e": "syn_e.JPEG"}      # the suffix match is case-insensitive, as PlantSegDataset
#: hand counts, classes 0..6: pixels and images (class 6 never appears)
HAND_PIXELS = [11 + 11 + 11 + 12 + 15, 12, 10, 10, 5, 1, 0]
HAND_IMAGES = [5, 3, 2, 2, 2, 1, 0]
HAND_TOTAL = 38
#: class -> (rank, tercile, rare, zero)
HAND_STRATA = {1: (1, "T1", False, False), 2: (2, "T1", False, False), 3: (3, "T2", False, False),
               4: (4, "T2", False, False), 5: (5, "T3", True, False), 6: (6, "T3", True, True)}


def mask_array(spec, shape=(4, 5)):
    flat = []
    for value, count in spec:
        flat += [value] * count
    flat += [0] * (shape[0] * shape[1] - len(flat))
    return np.array(flat, dtype=np.uint8).reshape(shape)


def make_root(root: Path, masks=MASKS, names=IMAGE_NAMES) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations" / "train").mkdir(parents=True)
    for stem, spec in masks.items():
        (root / "images" / "train" / names.get(stem, f"{stem}.jpg")).write_bytes(b"")   # never decoded
        Image.fromarray(mask_array(spec), mode="L").save(root / "annotations" / "train" / f"{stem}.png")
    (root / "images" / "train" / "README.txt").write_text("not an image\n", encoding="utf-8")
    return root


def registered_document(mod) -> dict:
    """A 115-class document in the registered shape, from the class-weight file's TRAIN pixel counts."""
    ref = json.loads((REPO / "reports" / "e1_class_weights.json").read_text(encoding="utf-8"))
    pixels = ref["pixel_counts"]
    images = [5367] + [20 + (c * 7) % 300 for c in range(1, 116)]
    images[42] = 7                                        # a rare class
    rows = mod.build_rows(pixels, images)
    csv_b = mod.csv_bytes(rows)
    return {"schema": mod.SCHEMA, "lane": mod.LANE, "artifact_status": "provisional", "split": "train",
            "split_list_sha256": "0" * 64, "data_root_note": "SYNTHETIC", "n_images": 5367,
            "num_classes": 116, "background_index": 0, "ignore_index": 255, "rare_threshold": 20,
            "per_class": rows, "tercile_sizes": [38, 38, 39],
            "total_disease_pixels": sum(r["pixels"] for r in rows),
            "background": {"class_id": 0, "pixels": pixels[0], "images": 5367}, "ignore_pixels": 0,
            "csv_sha256": hashlib.sha256(csv_b).hexdigest(), "script_commit": "0" * 40,
            "script_clean_at_commit": True, "generated_utc": UTC}


def dir_link(target: Path, link: Path) -> str:
    """A directory symlink, or on Windows without the symlink privilege a junction (realpath resolves
    both)."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return "junction"


def write_doc(mod, d: Path, doc: dict, csv_rows=None) -> Path:
    d.mkdir(parents=True)
    (d / "train_strata_v1.csv").write_bytes(mod.csv_bytes(csv_rows if csv_rows is not None
                                                           else doc["per_class"]))
    p = d / "train_strata_v1.json"
    p.write_bytes(mod.json_bytes(doc))
    return p


# --------------------------------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    print("=" * 100)
    print("L-AM17-STRATA -- TRAIN strata smoke  [NONOFFICIAL: synthetic masks only]")
    print("=" * 100)
    work = Path(tempfile.mkdtemp(prefix="strata_smoke_"))
    try:
        if "test" in str(work).lower():                  # the fixture itself must not trip the guard
            raise RuntimeError(f"temporary directory {work} contains 'test'; rerun")
        mod = load_module()
        root = make_root(work / "root")

        # ============================ S1 d1 by hand ============================
        pairs = mod.train_split_list(root, 5)
        check("S1-1 TRAIN split list: 5 name-sorted pairs (case-insensitive .jpg/.jpeg; others ignored)",
              [s for s, _ in pairs] == sorted(MASKS) and all(m.name == f"{s}.png" for s, m in pairs))
        pixels, images, ignored = mod.count_masks([m for _, m in pairs], 7, 255)
        check("S1-2 hand-counted pixels and images per class (background 60 pixels in 5 images)",
              pixels.tolist() == HAND_PIXELS and images.tolist() == HAND_IMAGES,
              f"pixels={pixels.tolist()} images={images.tolist()}")
        check("S1-3 the two ignore pixels are excluded and counted", ignored == 2, ignored)
        rows = mod.build_rows(pixels, images, background_index=0, rare_threshold=2)
        check("S1-4 one row per disease class 1-6, ascending; background takes no row",
              [r["class_id"] for r in rows] == [1, 2, 3, 4, 5, 6])
        check("S1-5 share = pixels / 38 exactly (the correctly rounded fraction)",
              all(r["share"] == float(Fraction(HAND_PIXELS[r["class_id"]], HAND_TOTAL)) for r in rows)
              and sum(Fraction(r["pixels"], HAND_TOTAL) for r in rows) == 1)
        got = {r["class_id"]: (r["rank"], r["tercile"], r["rare"], r["zero_train_pixels"]) for r in rows}
        check("S1-6 ranks (2 before 3 at the 10-pixel tie), terciles 2/2/2, rare < 2 images, zero flag",
              got == HAND_STRATA, got)
        check("S1-7 tercile sizes: 115 -> 38/38/39, 6 -> 2/2/2, 7 -> 2/2/3",
              mod.tercile_sizes(115) == [38, 38, 39] and mod.tercile_sizes(6) == [2, 2, 2]
              and mod.tercile_sizes(7) == [2, 2, 3])
        check("S1-8 the builder is a pure function of the counts (two calls equal)",
              mod.build_rows(pixels, images, background_index=0, rare_threshold=2) == rows)

        # ============================ S2 the CLI ============================
        out1, out2 = work / "out1", work / "out2"
        rc1, t1 = run_cli(["--data-root", str(root), "--out-dir", str(out1), "--generated-utc", UTC, *SYN])
        rc2, t2 = run_cli(["--data-root", str(root), "--out-dir", str(out2), "--generated-utc", UTC, *SYN])
        j1, j2 = out1 / "train_strata_v1.json", out2 / "train_strata_v1.json"
        c1, c2 = out1 / "train_strata_v1.csv", out2 / "train_strata_v1.csv"
        check("S2-1 two runs exit 0 and write the JSON and CSV", rc1 == rc2 == 0 and j1.is_file()
              and c1.is_file() and j2.is_file() and c2.is_file(), (t1 + t2)[-300:])
        check("S2-2 two runs are bitwise identical (JSON and CSV)",
              j1.read_bytes() == j2.read_bytes() and c1.read_bytes() == c2.read_bytes())
        doc = json.loads(j1.read_text(encoding="utf-8"))
        need = {"split_list_sha256", "data_root_note", "n_images", "per_class", "tercile_sizes",
                "script_commit", "generated_utc"}
        check("S2-3 lane 5 (c) fields present; per_class rows carry the (c) keys",
              need <= set(doc) and all({"class_id", "pixels", "share", "images", "rank", "tercile",
                                        "rare"} <= set(r) for r in doc["per_class"]), sorted(doc))
        check("S2-4 values: 5 images, tercile sizes [2, 2, 2], rows equal the unit build, status smoke",
              doc["n_images"] == 5 and doc["tercile_sizes"] == [2, 2, 2] and doc["per_class"] == rows
              and doc["artifact_status"] == "smoke" and doc["generated_utc"] == UTC)
        check("S2-5 split_list_sha256 = sha256 of the five stems, one per line",
              doc["split_list_sha256"] == hashlib.sha256("".join(f"{s}\n" for s in sorted(MASKS))
                                                         .encode()).hexdigest())
        check("S2-6 the JSON binds the CSV (csv_sha256); background and ignore pixels recorded",
              doc["csv_sha256"] == hashlib.sha256(c1.read_bytes()).hexdigest()
              and doc["background"] == {"class_id": 0, "pixels": 60, "images": 5}
              and doc["ignore_pixels"] == 2 and doc["zero_train_pixel_classes"] == [6])
        check("S2-7 script provenance: HEAD recorded (40-hex) and the script sha256",
              isinstance(doc["script_commit"], str) and len(doc["script_commit"]) == 40
              and doc["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              doc["script_commit"])
        lines = c1.read_text(encoding="utf-8").splitlines()
        check("S2-8 CSV: header + 6 rows sorted by class id, full-precision shares",
              lines[0] == ",".join(mod.CSV_COLUMNS) and [ln.split(",")[0] for ln in lines[1:]]
              == ["1", "2", "3", "4", "5", "6"] and lines[1].split(",")[2] == repr(12 / 38))
        rc3, _ = run_cli(["--data-root", str(root), "--out-dir", str(work / "u1"), *SYN])
        rc4, _ = run_cli(["--data-root", str(root), "--out-dir", str(work / "u2"), *SYN])
        u1 = json.loads((work / "u1" / "train_strata_v1.json").read_text(encoding="utf-8"))
        u2 = json.loads((work / "u2" / "train_strata_v1.json").read_text(encoding="utf-8"))
        u1.pop("generated_utc"), u2.pop("generated_utc")
        check("S2-9 unpinned runs differ at most in generated_utc; CSVs identical",
              rc3 == rc4 == 0 and u1 == u2 and (work / "u1" / "train_strata_v1.csv").read_bytes()
              == (work / "u2" / "train_strata_v1.csv").read_bytes())
        before = j1.read_bytes()
        rc, text = run_cli(["--data-root", str(root), "--out-dir", str(out1), *SYN])
        check("S2-10 an existing output is never overwritten (exit 2)",
              rc == 2 and "refusing to overwrite" in text and j1.read_bytes() == before, text[-160:])
        rc, text = run_cli(["--data-root", str(root), "--out-dir", str(work / "badutc"), *SYN,
                            "--generated-utc", "yesterday"])
        check("S2-11 a malformed --generated-utc is refused (exit 2)", rc == 2
              and not (work / "badutc").exists(), text[-120:])
        rc, text = run_cli(["--help"])
        check("S2-12 --help renders (exit 0)", rc == 0 and "usage:" in text, text[-200:])
        nan_out = work / "nan_out"
        try:
            mod.write_outputs(nan_out, {**doc, "bad": float("nan")}, c1.read_bytes())
            raised = False
        except ValueError:
            raised = True
        check("S2-13 a document that cannot be serialized creates no file (serialized first)",
              raised and not nan_out.exists())
        real_json_bytes = mod.json_bytes
        mod.json_bytes = lambda d: real_json_bytes(d).decode("utf-8")      # str: the binary write fails
        half = work / "half_out"
        try:
            mod.write_outputs(half, doc, c1.read_bytes())
            raised = False
        except TypeError:
            raised = True
        finally:
            mod.json_bytes = real_json_bytes
        check("S2-14 a write that fails after the CSV exists leaves neither the CSV nor a partial JSON",
              raised and half.is_dir() and not any(half.iterdir()))

        # ============================ S3 refusals ============================
        reads = []
        real_count = mod.count_masks
        mod.count_masks = lambda *a, **k: reads.append(a) or real_count(*a, **k)

        def refused(label, data_root, *, expected=5, marker="contains 'test'"):
            out = work / f"out_{len(CHECKS)}"
            reads.clear()
            rc, text = run_in_process(mod, ["--data-root", str(data_root), "--out-dir", str(out),
                                            "--num-classes", "7", "--rare-threshold", "2",
                                            "--expected-images", str(expected)])
            check(label, rc == 2 and marker in text and not out.exists() and not reads, text[-200:])

        troot = make_root(work / "strata_TEST_root")
        refused("S3-1 a data root containing 'test' (any case) -> exit 2, no mask read", troot)
        croot = make_root(work / "root_contest", masks={**MASKS, "syn_contest": [(1, 1)]},
                          names={**IMAGE_NAMES, "syn_contest": "syn_contest.jpg"})
        refused("S3-2 a TRAIN stem containing 'test' -> exit 2 before any mask is read", croot, expected=6)
        (work / "test_elsewhere").mkdir()
        shutil.move(str(make_root(work / "tmp_src") / "images" / "train"),
                    str(work / "test_elsewhere" / "train"))
        lroot = work / "root_link"
        (lroot / "images").mkdir(parents=True)
        shutil.copytree(work / "tmp_src" / "annotations", lroot / "annotations")
        kind = dir_link(work / "test_elsewhere" / "train", lroot / "images" / "train")
        refused(f"S3-3 images/train resolving into a 'test' directory ({kind}) -> exit 2, no mask read",
                lroot)
        mroot = make_root(work / "root_missing")
        (mroot / "annotations" / "train" / "syn_c.png").unlink()
        refused("S3-4 a TRAIN image without its mask -> exit 2", mroot, marker="have no mask")
        xroot = make_root(work / "root_extra")
        Image.fromarray(mask_array([(1, 1)]), mode="L").save(xroot / "annotations" / "train" / "orphan.png")
        refused("S3-5 an extra mask without a TRAIN image -> exit 2", xroot, marker="without a TRAIN image")
        refused("S3-6 a count other than expected -> exit 2", root, expected=6, marker="expected 6")
        rc, text = run_cli(["--data-root", str(root), "--out-dir", str(work / "reg")])
        check("S3-7 the registered defaults refuse the 5-image root (5,367 expected) -> exit 2",
              rc == 2 and "expected 5367" in text and not (work / "reg").exists(), text[-160:])
        broot = make_root(work / "root_badlabel", masks={**MASKS, "syn_b": [(1, 4), (9, 1)]})
        rc, text = run_cli(["--data-root", str(broot), "--out-dir", str(work / "bl"), *SYN])
        check("S3-8 a label outside 0..6 (not 255) is refused by name -> exit 2",
              rc == 2 and "syn_b.png" in text and "[9]" in text and not (work / "bl").exists(), text[-200:])
        rroot = make_root(work / "root_rgb")
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8), mode="RGB").save(
            rroot / "annotations" / "train" / "syn_a.png")
        rc, text = run_cli(["--data-root", str(rroot), "--out-dir", str(work / "rgb"), *SYN])
        check("S3-9 an RGB (3-channel) mask is refused -> exit 2", rc == 2 and "single-channel" in text,
              text[-160:])
        mod.count_masks = real_count

        # ============================ S4 --check (d2) ============================
        rc, text = run_cli(["--check", str(j1)])
        check("S4-1 --check refuses the synthetic 5-mask build (not the registered build) -> exit 1",
              rc == 1 and "[FAIL] C1" in text and "[FAIL] C2" in text and "STRATA d2 FAIL" in text,
              text[-200:])
        reg = registered_document(mod)
        good = write_doc(mod, work / "reg_ok", reg)
        rc, text = run_cli(["--check", str(good)])
        check("S4-2 a registered-shape 115-class document passes all ten checks -> exit 0",
              rc == 0 and "SUMMARY  10/10" in text and "STRATA d2 PASS" in text, text[-300:])
        results, recorded = mod.check_document(good)
        check("S4-3 Spearman rho = 1.0 and the pixel counts equal the class-weight file (recorded)",
              recorded["spearman_rho"] == 1.0 and recorded["pixel_counts_equal_class_weight_file"]
              and recorded["background_pixels_equal"] and recorded["train_mask_count_equal"],
              {k: recorded[k] for k in ("spearman_rho", "pixel_counts_equal_class_weight_file")})
        check("S4-4 tercile sizes of the 115-class build are 38/38/39 with 1 rare class",
              reg["tercile_sizes"] == [38, 38, 39] and [r["rare"] for r in reg["per_class"]].count(True) == 1)

        def failing(label, fn, which):
            d = json.loads(json.dumps(reg))
            fn(d)
            p = write_doc(mod, work / f"bad_{len(CHECKS)}", d)
            res, _ = mod.check_document(p)
            failed = sorted(n.split()[0] for n, ok, _ in res if not ok)
            check(label, failed == which, failed)

        failing("S4-5 a share off by 1e-9 fails C5, C7 and C9 (the CSV hash)",
                lambda d: d["per_class"][0].__setitem__("share", d["per_class"][0]["share"] + 1e-9),
                ["C5", "C7", "C9"])
        failing("S4-6 two terciles moved to T3 fail C6, C7 and C9",
                lambda d: (d["per_class"][0].__setitem__("tercile", "T3"),
                           d["per_class"][1].__setitem__("tercile", "T3")), ["C6", "C7", "C9"])
        failing("S4-7 an unclean script commit fails C10 only",
                lambda d: d.__setitem__("script_clean_at_commit", False), ["C10"])

        p = write_doc(mod, work / "bad_missing", {**reg, "per_class": reg["per_class"][1:]})
        res, _ = mod.check_document(p)
        check("S4-8 a missing class fails C3 and stops the check", [n.split()[0] for n, ok, _ in res
                                                                    if not ok] == ["C3"] and len(res) == 3)
        zero = json.loads(json.dumps(reg))
        rows0 = mod.build_rows([reg["background"]["pixels"], 0] + [r["pixels"] for r in reg["per_class"][1:]],
                               [5367, 0] + [r["images"] for r in reg["per_class"][1:]])
        zero["per_class"] = rows0
        zero["total_disease_pixels"] = sum(r["pixels"] for r in rows0)
        zero["csv_sha256"] = hashlib.sha256(mod.csv_bytes(rows0)).hexdigest()
        res, _ = mod.check_document(write_doc(mod, work / "bad_zero", zero))
        failed = sorted(n.split()[0] for n, ok, _ in res if not ok)
        check("S4-9 a class with zero TRAIN pixels is ranked last and flagged, and fails C4 (all present)",
              "C4" in failed and rows0[0]["rank"] == 115 and rows0[0]["zero_train_pixels"]
              and rows0[0]["tercile"] == "T3", failed)
    except Exception:                                        # noqa: BLE001
        traceback.print_exc()
        check("FATAL", False, traceback.format_exc(limit=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    check("Z1 temporary synthetic data removed", not work.exists())

    print()
    for n, ok, det in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        if det and not ok:
            print(f"         {det[:200]}")
    good = sum(1 for _, ok, _ in CHECKS if ok)
    allok = good == len(CHECKS)
    print("\n" + "=" * 100)
    print(f"SUMMARY  {good}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'L-AM17-STRATA OK' if allok else 'L-AM17-STRATA BLOCKED'} ({good}/{len(CHECKS)})")
    print("NONOFFICIAL: synthetic masks only; the real build and check d2 run locally "
          "(scripts/build_train_strata.py, TRAIN mounted only).")
    print("=" * 100)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
