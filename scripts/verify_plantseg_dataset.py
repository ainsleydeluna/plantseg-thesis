#!/usr/bin/env python3
"""Full read-only verification of the manually downloaded PlantSeg dataset.

Reads the dataset path from reports/dataset_location_log.md (falls back to the
documented location), discovers the real structure WITHOUT assuming a layout or
that any particular split file exists, then verifies the full dataset and writes:

  - reports/dataset_report.json
  - reports/dataset_report.md

This script ONLY reads dataset files. It never moves, renames, deletes, writes,
or modifies any dataset file or mask. It writes solely into reports/.

Dimension check is EXIF-aware: images are EXIF-transposed before their dimensions
are compared to masks (masks are PNG, have no EXIF, and are never transposed).
Raw-dimension transposes are reported as WARNINGS (they resolve after EXIF), not
as dataset failures. The raw EXIF orientation distribution is also reported.

Label diagnostics (for the num_classes/background question) are included:
pixel/mask counts for index 115, per-split presence, and a metadata/COCO scan to
tell whether index 0 is a disease and what index 115 represents.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "reports" / "dataset_location_log.md"
FALLBACK_ROOT = Path(r"C:\Users\admin\plantseg_data\plantseg")
OUT_JSON = REPO / "reports" / "dataset_report.json"
OUT_MD = REPO / "reports" / "dataset_report.md"

IMG_EXTS = {".jpg", ".jpeg"}
MASK_EXTS = {".png"}
ORIENTATION_TAG = 0x0112
SWAP_ORIENTATIONS = {5, 6, 7, 8}
SPLIT_VOCAB = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}

EXPECTED = {
    "total_images": 7774,
    "splits_approx": {"train": 5442, "val": 778, "test": 1554},
    "ignore_index": 255,
    "documented_mask_indices": "Wei: 0-114 (115 classes); ch3: provisional 116 -> OPEN",
    "image_format": "JPEG",
    "mask_format": "PNG grayscale (single-channel indexed)",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_root() -> Path:
    # Prefer PLANTSEG_DATA_ROOT (the portable root the dataloader uses via configs/data.py) so this
    # verifier is runnable on RunPod too; then fall back to the local log / documented path.
    env = os.environ.get("PLANTSEG_DATA_ROOT")
    if env and Path(env).exists():
        log(f"[root] from PLANTSEG_DATA_ROOT: {env}")
        return Path(env)
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"\*\*Path:\*\*\s*`([^`]+)`", text)
        if m:
            p = Path(m.group(1).strip())
            if p.exists():
                log(f"[root] from log: {p}")
                return p
    if FALLBACK_ROOT.exists():
        log(f"[root] fallback: {FALLBACK_ROOT}")
        return FALLBACK_ROOT
    log("FATAL: could not resolve a dataset root that exists.")
    sys.exit(2)


def canon_split(name: str):
    return SPLIT_VOCAB.get(name.strip().lower())


def split_from_path(path: Path, root: Path):
    for part in path.relative_to(root).parts:
        c = canon_split(part)
        if c:
            return c
    return None


def collect_filename_stems(obj, out: set) -> None:
    if isinstance(obj, str):
        if obj.lower().endswith((".jpg", ".jpeg", ".png")):
            out.add(Path(obj).stem)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_filename_stems(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_filename_stems(v, out)


def discover_top_level(root: Path) -> list:
    out = []
    for e in sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        out.append({"name": e.name, "type": "DIR" if e.is_dir() else "FILE"})
    return out


def scan_metadata_and_json(root: Path) -> dict:
    """Read Metadata.csv class names and COCO category_ids to characterise the label space."""
    info: dict = {"metadata": None, "coco": None}

    csv_path = root / "Metadata.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        idx2name, idx_vals = {}, set()
        for r in rows:
            raw = str(r.get("Index", "")).strip()
            if raw.lstrip("-").isdigit():
                i = int(raw)
                idx_vals.add(i)
                idx2name.setdefault(i, (r.get("Plant"), r.get("Disease")))
        sv = sorted(idx_vals)
        info["metadata"] = {
            "rows": len(rows),
            "index_min": sv[0] if sv else None,
            "index_max": sv[-1] if sv else None,
            "index_distinct": len(sv),
            "index_115_present": 115 in idx_vals,
            "name_index_0": idx2name.get(0),
            "name_index_114": idx2name.get(114),
            "name_index_115": idx2name.get(115),
        }

    cat_ids: set = set()
    cats_defined = 0
    for p in root.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "annotations" in data:
            for a in data.get("annotations", []):
                if isinstance(a, dict) and "category_id" in a:
                    cat_ids.add(a["category_id"])
            cats_defined = max(cats_defined, len(data.get("categories", []) or []))
    if cat_ids:
        info["coco"] = {
            "category_id_min": min(cat_ids),
            "category_id_max": max(cat_ids),
            "category_id_distinct": len(cat_ids),
            "categories_list_len": cats_defined,
            "category_115_used": 115 in cat_ids,
        }
    return info


def determine_background(report: dict, meta: dict) -> dict:
    """Determine the background index empirically (by ubiquity + pixel dominance) and verify the
    category_id+1 mask encoding. Does NOT assume which index is background."""
    m = report["masks"]
    total_masks = report["counts"]["total_masks"]
    md = meta.get("metadata") or {}
    coco = meta.get("coco") or {}
    bs = m["background_scan"]
    enc = m["mask_value_encoding"]

    bg = bs["background_index"]
    bg_masks = bs["masks_containing_background"]
    bg_pix_frac = bs["background_pixel_fraction"]
    ubiquitous = bool(total_masks and bg_masks / total_masks >= 0.99 and bg_pix_frac >= 0.30)

    diseases_named_0_114 = (md.get("index_min") == 0 and md.get("index_max") == 114
                            and md.get("index_distinct") == 115)
    shift_ok = (enc.get("model") == "mask_value = category_id + 1"
                and (enc.get("consistent_fraction") or 0) >= 0.99)

    if bg == 0 and ubiquitous:
        index0_role = (f"BACKGROUND — mask value 0 is in {bg_masks}/{total_masks} masks "
                       f"({bg_pix_frac:.1%} of pixels). NOTE: Metadata Index 0 names a disease "
                       f"({md.get('name_index_0')}), but mask pixel value 0 is background because the "
                       f"mask encoding is category_id+1.")
    else:
        index0_role = f"NEED_TO_CONFIRM (empirical background index = {bg})"

    if shift_ok:
        index115_role = (f"DISEASE class = category_id 114 + 1 (Metadata 114 = "
                         f"{md.get('name_index_114')}); appears in {m['label_115']['masks_containing']} "
                         f"masks only (class-specific, NOT background)")
    else:
        index115_role = "NEED_TO_CONFIRM"

    # "Explicit" would require a literal background category in metadata/annotations; COCO categories
    # list is empty, so the background label is DERIVED, not explicitly named.
    background_explicit = bool(coco.get("categories_list_len", 0))
    return {
        "background_index_empirical": bg,
        "background_ubiquitous_and_dominant": ubiquitous,
        "mask_encoding_model": enc.get("model"),
        "mask_encoding_consistent_fraction": enc.get("consistent_fraction"),
        "disease_classes_named_0_114_in_metadata": bool(diseases_named_0_114),
        "index_0_role": index0_role,
        "index_115_role": index115_role,
        "background_explicitly_named_in_files": background_explicit,
        "disease_only_mapping_empirical": (
            "background = mask value 0 (exclude); diseases = mask values 1-115 (= category_id + 1)"
            if (shift_ok and ubiquitous and bg == 0) else "NEED_TO_CONFIRM"),
        "disease_only_fully_explicit": False,
        "note": ("Determined empirically with high confidence, but no literal 'background' category is "
                 "named in the released metadata/annotations (COCO categories list is empty), so a "
                 "residual NEED_TO_CONFIRM is retained per protocol."),
    }


def main() -> int:
    root = resolve_root()
    report = {"status": "PASS", "dataset_root": str(root)}
    failures: list[str] = []
    warns: list[str] = []
    inconsistencies: list[str] = []

    top_level = discover_top_level(root)
    report["top_level"] = top_level
    log(f"[structure] top level: {[e['name'] for e in top_level]}")

    img_dir = root / "images" if (root / "images").is_dir() else root
    mask_dir = root / "annotations" if (root / "annotations").is_dir() else root

    img_files: dict[str, Path] = {}
    mask_files: dict[str, Path] = {}
    dup_img: list[str] = []
    dup_mask: list[str] = []
    img_ext: Counter = Counter()
    mask_ext: Counter = Counter()
    folder_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}

    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_ext[p.suffix.lower()] += 1
            if p.stem in img_files:
                dup_img.append(p.stem)
            img_files[p.stem] = p
            s = split_from_path(p, root)
            if s:
                folder_split[s].add(p.stem)
    for p in mask_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            mask_ext[p.suffix.lower()] += 1
            if p.stem in mask_files:
                dup_mask.append(p.stem)
            mask_files[p.stem] = p

    total_images, total_masks = len(img_files), len(mask_files)
    log(f"[counts] images={total_images} masks={total_masks}")
    img_stems, mask_stems = set(img_files), set(mask_files)
    shared = img_stems & mask_stems
    imgs_wo_mask = sorted(img_stems - mask_stems)
    masks_wo_img = sorted(mask_stems - img_stems)

    report["extensions"] = {"images": dict(img_ext), "masks": dict(mask_ext)}
    report["counts"] = {
        "total_images": total_images, "total_masks": total_masks,
        "shared_identifiers": len(shared),
        "images_without_masks": len(imgs_wo_mask),
        "masks_without_images": len(masks_wo_img),
        "duplicate_image_stems": len(dup_img), "duplicate_mask_stems": len(dup_mask),
    }
    report["examples"] = {"images_without_masks": imgs_wo_mask[:20],
                          "masks_without_images": masks_wo_img[:20]}

    # ---- split detection (folder / json / csv) ----
    mechanisms = {}
    if any(folder_split.values()):
        mechanisms["folder"] = True
    json_split = {"train": set(), "val": set(), "test": set()}
    json_used = []
    for p in root.glob("*.json"):
        c = next((canon_split(k) for k in ("train", "val", "test") if k in p.stem.lower()), None)
        if not c:
            continue
        try:
            stems: set = set()
            collect_filename_stems(json.loads(p.read_text(encoding="utf-8", errors="replace")), stems)
            json_split[c] |= stems
            json_used.append({"file": p.name, "split": c, "ids": len(stems)})
        except Exception as e:  # noqa: BLE001
            inconsistencies.append(f"could not parse JSON split file {p.name}: {e}")
    if any(json_split.values()):
        mechanisms["json"] = True
    csv_split = {"train": set(), "val": set(), "test": set()}
    csv_meta = None
    for p in root.glob("*.csv"):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                headers = reader.fieldnames or []
                rows = list(reader)
            split_col = next((h for h in headers if "split" in h.lower()), None)
            name_col = next((h for h in headers if "image" in h.lower() and "name" in h.lower()), None)
            if name_col is None:
                name_col = next((h for h in headers if "name" in h.lower()), headers[0] if headers else None)
            csv_meta = {"file": p.name, "headers": headers, "name_col": name_col,
                        "split_col": split_col, "rows": len(rows)}
            if split_col and name_col:
                for r in rows:
                    c = canon_split(str(r.get(split_col, "")))
                    nm = str(r.get(name_col, "")).strip()
                    if c and nm:
                        csv_split[c].add(Path(nm).stem)
        except Exception as e:  # noqa: BLE001
            inconsistencies.append(f"could not parse CSV {p.name}: {e}")
    if any(csv_split.values()):
        mechanisms["csv"] = True

    if "folder" in mechanisms:
        authoritative, auth = "folder", folder_split
    elif "json" in mechanisms:
        authoritative, auth = "json", json_split
    elif "csv" in mechanisms:
        authoritative, auth = "csv", csv_split
    else:
        authoritative, auth = None, {"train": set(), "val": set(), "test": set()}

    stem2split = {s: sp for sp, ids in auth.items() for s in ids}

    # Metadata Name->Index (per-image disease class id) for the mask-value encoding check.
    stem2metaidx: dict[str, int] = {}
    _mp = root / "Metadata.csv"
    if _mp.exists():
        with _mp.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                nm = str(r.get("Name", "")).strip()
                iv = str(r.get("Index", "")).strip()
                if nm and iv.lstrip("-").isdigit():
                    stem2metaidx[Path(nm).stem] = int(iv)
    overlaps = {
        "train_val": len(auth["train"] & auth["val"]),
        "train_test": len(auth["train"] & auth["test"]),
        "val_test": len(auth["val"] & auth["test"]),
    }
    report["split"] = {
        "mechanisms_found": sorted(mechanisms),
        "authoritative": authoritative,
        "json_files": json_used, "csv_meta": csv_meta,
        "counts": {k: len(auth[k]) for k in ("train", "val", "test")},
        "counts_per_mechanism": {
            m: {k: len(v[k]) for k in ("train", "val", "test")}
            for m, v in {"folder": folder_split, "json": json_split, "csv": csv_split}.items()
            if m in mechanisms},
        "overlaps_counts": overlaps,
    }

    # ---- per-file verification (FULL dataset), EXIF-aware ----
    unreadable_images: list[str] = []
    unreadable_masks: list[str] = []
    img_size_corrected: dict[str, tuple] = {}
    img_size_raw: dict[str, tuple] = {}
    mask_size: dict[str, tuple] = {}
    orient_counter: Counter = Counter()
    mode_counter: Counter = Counter()
    ndim_counter: Counter = Counter()
    unique_vals: set = set()
    hist = np.zeros(256, dtype=np.int64)
    masks_with_115 = 0
    masks_with_0 = 0
    split_has_115 = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    masks_containing: Counter = Counter()
    shift_checked = 0
    shift_consistent = 0
    shift_exceptions: list[str] = []

    log(f"[verify] EXIF-aware decode of {total_images} images ...")
    for i, (stem, p) in enumerate(img_files.items(), 1):
        try:
            with Image.open(p) as im:
                orient_counter[im.getexif().get(ORIENTATION_TAG)] += 1
                img_size_raw[stem] = im.size
                corrected = ImageOps.exif_transpose(im)  # applies EXIF; decodes (readability)
                img_size_corrected[stem] = corrected.size
        except Exception as e:  # noqa: BLE001
            unreadable_images.append(f"{p.name}: {e}")
        if i % 1000 == 0:
            log(f"  images {i}/{total_images}")

    log(f"[verify] decode + label-scan of {total_masks} masks ...")
    for i, (stem, p) in enumerate(mask_files.items(), 1):
        try:
            with Image.open(p) as im:
                mode_counter[im.mode] += 1
                mask_size[stem] = im.size
                arr = np.asarray(im)  # preserves palette indices for mode 'P'/'L'
            ndim_counter[arr.ndim] += 1
            if arr.ndim == 2:
                flat = arr.ravel()
                hist += np.bincount(flat, minlength=256)[:256]
                u = np.unique(arr)
                uset = {int(x) for x in u.tolist()}
                unique_vals |= uset
                if 115 in uset:
                    masks_with_115 += 1
                    split_has_115[stem2split.get(stem, "unknown")] += 1
                if 0 in uset:
                    masks_with_0 += 1
                for v in uset:
                    masks_containing[v] += 1
                mi = stem2metaidx.get(stem)
                if mi is not None:
                    shift_checked += 1
                    nonzero = uset - {0}
                    if (not nonzero) or nonzero <= {mi + 1}:
                        shift_consistent += 1
                    elif len(shift_exceptions) < 20:
                        shift_exceptions.append(
                            f"{stem}: metaIndex={mi} expected nonzero {{{mi + 1}}} got {sorted(nonzero)}")
        except Exception as e:  # noqa: BLE001
            unreadable_masks.append(f"{p.name}: {e}")
        if i % 1000 == 0:
            log(f"  masks {i}/{total_masks}")

    # EXIF-aware dimension comparison (real) + raw transpose warnings
    dim_mismatch: list[str] = []
    raw_transpose_warn: list[str] = []
    for stem in shared:
        a = img_size_corrected.get(stem)
        b = mask_size.get(stem)
        raw = img_size_raw.get(stem)
        if a is not None and b is not None and a != b:
            dim_mismatch.append(f"{stem}: exif-corrected image{a} != mask{b}")
        if raw is not None and b is not None and raw != b:
            raw_transpose_warn.append(f"{stem}: raw image{raw} vs mask{b} (resolved by EXIF)")

    color_masks = any(k > 2 for k in ndim_counter) or any(
        mm not in ("L", "P", "I", "I;16", "1") for mm in mode_counter)
    single_channel_indexed = (not color_masks) and ndim_counter.get(2, 0) == sum(ndim_counter.values())

    sorted_unique = sorted(unique_vals)
    non_ignore = [v for v in sorted_unique if v != 255]
    contains_255 = 255 in unique_vals
    max_ni = max(non_ignore) if non_ignore else None
    min_ni = min(non_ignore) if non_ignore else None
    contiguous = bool(non_ignore) and non_ignore == list(range(min_ni, max_ni + 1))
    total_pixels = int(hist.sum())

    report["exif"] = {
        "orientation_distribution": {str(k): v for k, v in orient_counter.items()},
        "raw_transpose_warning_count": len(raw_transpose_warn),
        "raw_transpose_examples": raw_transpose_warn[:20],
        "note": ("raw .size transposes are EXIF orientation; masks match after "
                 "ImageOps.exif_transpose on images. Treated as WARNING, not failure."),
    }
    report["readability"] = {
        "unreadable_images": len(unreadable_images),
        "unreadable_masks": len(unreadable_masks),
        "unreadable_images_examples": unreadable_images[:20],
        "unreadable_masks_examples": unreadable_masks[:20],
    }
    report["dimensions"] = {
        "exif_aware_mismatch_count": len(dim_mismatch),
        "exif_aware_mismatch_examples": dim_mismatch[:20],
        "raw_transpose_warnings": len(raw_transpose_warn),
        "distinct_image_sizes": len(set(img_size_corrected.values())),
    }
    report["masks"] = {
        "channel_modes": dict(mode_counter),
        "ndim_distribution": dict(ndim_counter),
        "is_single_channel_indexed": bool(single_channel_indexed),
        "color_masks_present": bool(color_masks),
        "num_unique_values": len(sorted_unique),
        "unique_values": sorted_unique,
        "contains_255": bool(contains_255),
        "max_non_ignore_label": max_ni,
        "min_non_ignore_label": min_ni,
        "contiguous_from_min_to_max": contiguous,
        "total_pixels": total_pixels,
        "label_0": {
            "pixels": int(hist[0]),
            "pixel_fraction": round(int(hist[0]) / total_pixels, 6) if total_pixels else 0.0,
            "masks_containing": masks_with_0,
        },
        "label_115": {
            "pixels": int(hist[115]),
            "pixel_fraction": round(int(hist[115]) / total_pixels, 6) if total_pixels else 0.0,
            "masks_containing": masks_with_115,
            "splits_present": {k: v for k, v in split_has_115.items() if v},
        },
    }
    # Background = the value present in the most masks (tie-broken by pixel count). Does not assume 0.
    bg_index = (max(masks_containing, key=lambda v: (masks_containing[v], int(hist[v])))
                if masks_containing else None)
    report["masks"]["background_scan"] = {
        "background_index": bg_index,
        "masks_containing_background": int(masks_containing.get(bg_index, 0)),
        "background_pixel_fraction": (round(int(hist[bg_index]) / total_pixels, 6)
                                      if (bg_index is not None and total_pixels) else 0.0),
        "top5_by_mask_presence": [[int(v), int(n)] for v, n in masks_containing.most_common(5)],
    }
    nonzero_vals = [v for v in sorted_unique if v not in (0, 255)]
    shift_frac = round(shift_consistent / shift_checked, 6) if shift_checked else None
    report["masks"]["mask_value_encoding"] = {
        "nonzero_label_min": (nonzero_vals[0] if nonzero_vals else None),
        "nonzero_label_max": (nonzero_vals[-1] if nonzero_vals else None),
        "model": ("mask_value = category_id + 1"
                  if (nonzero_vals and nonzero_vals[0] == 1 and bg_index == 0
                      and shift_frac is not None and shift_frac >= 0.99)
                  else "NEED_TO_CONFIRM"),
        "per_image_shift_checked": shift_checked,
        "per_image_shift_consistent": shift_consistent,
        "consistent_fraction": shift_frac,
        "shift_exceptions_examples": shift_exceptions[:20],
    }
    report["metadata_files"] = [
        e["name"] for e in top_level
        if e["type"] == "FILE" and Path(e["name"]).suffix.lower() not in (IMG_EXTS | MASK_EXTS)]

    meta = scan_metadata_and_json(root)
    report["label_space"] = meta
    report["background_determination"] = determine_background(report, meta)

    # ---- inconsistencies vs contract (soft) ----
    if total_images != EXPECTED["total_images"]:
        inconsistencies.append(f"total images {total_images} != contract {EXPECTED['total_images']}")
    if total_masks != total_images:
        inconsistencies.append(f"mask count {total_masks} != image count {total_images}")
    if [e for e in img_ext if e not in IMG_EXTS]:
        inconsistencies.append(f"unexpected image extensions: {[e for e in img_ext if e not in IMG_EXTS]}")
    if [e for e in mask_ext if e not in MASK_EXTS]:
        inconsistencies.append(f"unexpected mask extensions: {[e for e in mask_ext if e not in MASK_EXTS]}")
    if color_masks:
        inconsistencies.append("masks are RGB/color or multi-channel; contract expects indexed PNG")
    if authoritative is not None:
        for k, approx in EXPECTED["splits_approx"].items():
            actual = len(auth[k])
            if actual != approx:
                inconsistencies.append(
                    f"{k} split {actual} != contract approx {approx} (empirical split is authoritative)")
    note_255 = ("255 absent in raw masks — expected; the ignore label is introduced during "
                "preprocessing padding, not present in the released annotations") if not contains_255 \
        else "255 present in raw masks"

    if max_ni == 115:
        num_classes_finding = ("empirical max label 115; num_classes=116 appears consistent if "
                               "labels include 0-115; do not change masks.")
        num_classes_decision = "num_classes=116 supported (labels 0-115)"
        num_classes_status = "RESOLVED_TOWARD_116 (empirical max=115)"
    elif max_ni == 114:
        num_classes_finding = ("empirical max label 114; dataset appears to contain 115 classes "
                               "indexed 0-114; num_classes/background handling remains an "
                               "OPEN_CONFLICT unless implementation explicitly adds background separately.")
        num_classes_decision = "OPEN_CONFLICT (115 classes 0-114; background unresolved)"
        num_classes_status = "OPEN_CONFLICT (empirical max=114)"
    else:
        num_classes_finding = (f"empirical max non-ignore label = {max_ni}; matches neither 114 nor "
                               f"115; recorded as inconsistency, no winner picked.")
        num_classes_decision = f"UNEXPECTED max={max_ni}; no winner picked"
        num_classes_status = f"UNEXPECTED (empirical max={max_ni})"
        inconsistencies.append(num_classes_finding)

    report["num_classes_finding"] = num_classes_finding
    report["num_classes_status"] = num_classes_status
    report["note_255"] = note_255
    report["inconsistencies"] = inconsistencies

    # ---- warnings (non-fatal) ----
    if raw_transpose_warn:
        warns.append(f"{len(raw_transpose_warn)} raw image/mask dimension transpose(s) "
                     f"due to EXIF orientation (resolved by exif_transpose; not a failure)")
    report["warnings"] = warns

    # ---- hard failures -> FAIL (EXIF-aware; raw transposes are NOT failures) ----
    if unreadable_images:
        failures.append(f"{len(unreadable_images)} unreadable image(s)")
    if unreadable_masks:
        failures.append(f"{len(unreadable_masks)} unreadable mask(s)")
    if imgs_wo_mask:
        failures.append(f"{len(imgs_wo_mask)} image(s) without a matching mask")
    if masks_wo_img:
        failures.append(f"{len(masks_wo_img)} mask(s) without a matching image")
    if dim_mismatch:
        failures.append(f"{len(dim_mismatch)} EXIF-corrected image/mask dimension mismatch(es)")
    if any(overlaps.values()):
        failures.append(f"split identifier overlap: {overlaps}")
    if dup_img or dup_mask:
        failures.append(f"duplicate stems: img={len(dup_img)} mask={len(dup_mask)}")
    if color_masks:
        failures.append("masks not single-channel indexed (label extraction unreliable)")
    if total_images == 0 or total_masks == 0:
        failures.append("no images and/or no masks found")

    report["status"] = "FAIL" if failures else "PASS"
    report["failures"] = failures

    test_size = report["split"]["counts"]["test"]
    report["summary_line"] = (
        f"TOTAL_IMAGES = {total_images} | TEST_SIZE = {test_size} | "
        f"MAX_NON_IGNORE_LABEL = {max_ni} | NUM_CLASSES_DECISION = {num_classes_decision}")

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")

    log("")
    log(f"STATUS: {report['status']}")
    log(report["summary_line"])
    return 0 if report["status"] == "PASS" else 1


def render_md(r: dict) -> str:
    L: list[str] = []
    L.append("# PlantSeg Dataset Verification Report")
    L.append("")
    L.append(f"**STATUS: {r['status']}**  ")
    L.append("_Generated: 2026-06-27 by scripts/verify_plantseg_dataset.py (read-only; full dataset; EXIF-aware)._")
    L.append("")
    L.append(f"- Dataset root: `{r['dataset_root']}`")
    L.append("")
    if r["failures"]:
        L.append("## ❌ FAILURES")
        for f in r["failures"]:
            L.append(f"- {f}")
        L.append("")
    if r.get("warnings"):
        L.append("## ⚠️ Warnings (non-fatal)")
        for w in r["warnings"]:
            L.append(f"- {w}")
        L.append("")
    L.append("## Extracted structure (top level)")
    for e in r["top_level"]:
        L.append(f"- `{e['name']}` ({e['type']})")
    L.append("")
    L.append("## File extensions")
    L.append(f"- images: `{r['extensions']['images']}` | masks: `{r['extensions']['masks']}`")
    L.append(f"- metadata files: `{r['metadata_files']}`")
    L.append("")
    c = r["counts"]
    L.append("## Counts & identifiers")
    L.append(f"- total images: **{c['total_images']}** | total masks: **{c['total_masks']}** | shared: **{c['shared_identifiers']}**")
    L.append(f"- images without mask: {c['images_without_masks']} | masks without image: {c['masks_without_images']}")
    L.append(f"- duplicate stems (img/mask): {c['duplicate_image_stems']}/{c['duplicate_mask_stems']}")
    L.append("")
    sp = r["split"]
    L.append("## Split mechanism")
    L.append(f"- mechanisms found: **{sp['mechanisms_found']}** | authoritative (folder→json→csv): **{sp['authoritative']}**")
    L.append(f"- counts (authoritative): `{sp['counts']}`")
    L.append(f"- counts per mechanism: `{sp['counts_per_mechanism']}`")
    if sp["csv_meta"]:
        L.append(f"- CSV metadata: `{sp['csv_meta']}`")
    L.append(f"- pairwise zero-overlap (must be 0): `{sp['overlaps_counts']}`")
    L.append("")
    ex = r["exif"]
    L.append("## EXIF orientation (raw)")
    L.append(f"- orientation distribution: `{ex['orientation_distribution']}`")
    L.append(f"- raw-dimension transpose warnings: **{ex['raw_transpose_warning_count']}** (resolved by `exif_transpose`)")
    if ex["raw_transpose_examples"]:
        L.append(f"- examples: `{ex['raw_transpose_examples']}`")
    L.append(f"- {ex['note']}")
    L.append("")
    rd, dm = r["readability"], r["dimensions"]
    L.append("## Readability & dimensions (EXIF-aware)")
    L.append(f"- unreadable images: {rd['unreadable_images']} | unreadable masks: {rd['unreadable_masks']}")
    L.append(f"- EXIF-corrected dimension mismatches: **{dm['exif_aware_mismatch_count']}**")
    L.append(f"- raw transpose warnings: {dm['raw_transpose_warnings']} | distinct image sizes: {dm['distinct_image_sizes']}")
    L.append("")
    m = r["masks"]
    L.append("## Mask label analysis (full dataset)")
    L.append(f"- channel modes: `{m['channel_modes']}` | ndim: `{m['ndim_distribution']}`")
    L.append(f"- single-channel indexed: **{m['is_single_channel_indexed']}** | color masks: **{m['color_masks_present']}**")
    L.append(f"- unique values: **{m['num_unique_values']}** | contiguous 0..max: {m['contiguous_from_min_to_max']}")
    L.append(f"- contains 255: **{m['contains_255']}** — {r['note_255']}")
    L.append(f"- max non-ignore: **{m['max_non_ignore_label']}** | min non-ignore: **{m['min_non_ignore_label']}**")
    uv = m["unique_values"]
    L.append(f"- values: `{uv if len(uv) <= 130 else str(uv[:65]) + ' ... ' + str(uv[-3:])}`")
    L.append("")
    L.append("## Label-index diagnostics (index 0 and 115)")
    l0, l1 = m["label_0"], m["label_115"]
    L.append(f"- **index 115**: pixels={l1['pixels']:,} ({l1['pixel_fraction']:.4f} of all pixels), "
             f"masks containing={l1['masks_containing']}/{c['total_masks']}, splits={l1['splits_present']}")
    L.append(f"- **index 0**: pixels={l0['pixels']:,} ({l0['pixel_fraction']:.4f}), "
             f"masks containing={l0['masks_containing']}/{c['total_masks']}")
    ls = r["label_space"]
    if ls.get("metadata"):
        L.append(f"- Metadata.csv: `{ls['metadata']}`")
    if ls.get("coco"):
        L.append(f"- COCO annotations: `{ls['coco']}`")
    bsc = m["background_scan"]
    enc = m["mask_value_encoding"]
    L.append(f"- background scan (most-present value): `{bsc}`")
    L.append(f"- mask-value encoding: `{enc}`")
    bg = r["background_determination"]
    L.append("")
    L.append("## Background / disease determination")
    L.append(f"- empirical background index: **{bg['background_index_empirical']}** "
             f"(ubiquitous & dominant: {bg['background_ubiquitous_and_dominant']})")
    L.append(f"- mask-value encoding: **{bg['mask_encoding_model']}** "
             f"(consistent fraction {bg['mask_encoding_consistent_fraction']})")
    L.append(f"- disease classes named 0–114 in Metadata: **{bg['disease_classes_named_0_114_in_metadata']}**")
    L.append(f"- index 0 → **{bg['index_0_role']}**")
    L.append(f"- index 115 → **{bg['index_115_role']}**")
    L.append(f"- background explicitly named in files: **{bg['background_explicitly_named_in_files']}**")
    L.append(f"- disease-only mapping (empirical): **{bg['disease_only_mapping_empirical']}**")
    L.append(f"- disease-only fully explicit: **{bg['disease_only_fully_explicit']}** "
             f"(False → retain NEED_TO_CONFIRM)")
    L.append(f"- note: {bg['note']}")
    L.append("")
    L.append("## num_classes critical conflict (no silent winner)")
    L.append(f"- status: **{r['num_classes_status']}**")
    L.append(f"- finding: {r['num_classes_finding']}")
    L.append("")
    L.append("## Inconsistencies vs docs/IMPLEMENTATION_CONTRACT.md")
    for it in (r["inconsistencies"] or ["none"]):
        L.append(f"- {it}")
    L.append("")
    L.append("## Final summary")
    L.append("")
    L.append(f"`{r['summary_line']}`")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
