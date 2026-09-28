#!/usr/bin/env python3
"""L-AM13 real-data checks (docs/lane_specs/part1.md lane 2 (d)). LOCAL ONLY; nothing here runs a model.

    python -B scripts/am13_real_checks.py preflight
    python -B scripts/am13_real_checks.py d5 RUN_A RUN_B [--canvas-value V]

preflight (before the upstream runs): reads the HEADERS of the VAL images and masks under
PLANTSEG_DATA_ROOT (size and EXIF orientation; no pixel is decoded, and TEST is never listed). PASS iff
every image's EXIF-transposed shape equals its mask's shape -- the upstream protocol refuses any other
pair, because it never resizes the mask. It also reports the largest image and the memory its logits
take once resized to it (116 float32 channels at the original resolution, as MMSeg's
`postprocess_result` holds them).

d5 (after the runs): two real VAL artifacts of one model scored under the upstream protocol.
PASS iff both artifacts
  * verify (manifest hashes, strict JSON), are VAL, carry 846 rows and are scored under
    `upstream/1.0.0` with the contract `summary.protocol` block (EVALUATION_CONTRACT section 11);
  * are the same model and manifest (stage, role, checkpoint sha256, split manifest sha256);
  * are bitwise identical: per_image.jsonl byte for byte, every sufficient_stats.npz array (dtype,
    shape, bytes), summary.json as canonical JSON once run.run_id and run.timestamp_utc are removed;
  * score every image at its original resolution: each row's shape fields are the upstream geometry
    of its ori_shape, and per image sum_c pred == sum_c gt == H*W of ori_shape (the prediction was
    argmaxed at ori_shape; released masks carry no ignore pixel, EVALUATION_CONTRACT section 0).
Reported, never gated: all-class mIoU with its union-present eligible-class count, disease-only mIoU,
the AM-5 counts, and, with --canvas-value, the protocol effect upstream minus canvas (AM-17 item
1(e)); the spec sets no threshold on these values.

Exit codes: 0 PASS, 1 FAIL (STOP and report), 2 usage error or an artifact that cannot be read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_PASS, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
VAL_ROWS = 846
IDENTITY_KEYS = ("stage", "model_role", "precision", "checkpoint_sha256")
VOLATILE_RUN_KEYS = ("run_id", "timestamp_utc")          # EVALUATION_CONTRACT section 10(d)


class Unreadable(RuntimeError):
    """An artifact that cannot be verified or read: not a verdict."""


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verified_summary(d: Path) -> dict:
    from src.eval.artifacts import verify_artifact
    try:
        return verify_artifact(d)
    except Exception as e:                                # noqa: BLE001 -- reported verbatim
        raise Unreadable(f"verify_artifact({d}) failed: {type(e).__name__}: {e}") from e


def _rows(d: Path) -> list[dict]:
    return [json.loads(ln) for ln in (d / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]


def d5(a_dir: Path, b_dir: Path, canvas_value: float | None) -> int:
    import numpy as np

    from src.eval import protocols as P

    sa, sb = _verified_summary(a_dir), _verified_summary(b_dir)
    problems: list[str] = []
    for label, s in (("A", sa), ("B", sb)):
        ds = s["dataset"]
        if ds["preprocess_protocol"] != P.UPSTREAM_PROTOCOL_ID or s.get("protocol") != P.upstream_block():
            problems.append(f"{label}: not an upstream/1.0.0 artifact with the contract protocol block")
        if ds["split"] != "val" or not ds["actual_rows"] == ds["expected_rows"] == VAL_ROWS:
            problems.append(f"{label}: split {ds['split']!r}, rows {ds['actual_rows']} "
                            f"(expected val and {VAL_ROWS})")
    for k in IDENTITY_KEYS:
        if sa["run"].get(k) != sb["run"].get(k):
            problems.append(f"run.{k} differs: {sa['run'].get(k)!r} vs {sb['run'].get(k)!r}")
    if sa["dataset"]["split_manifest_sha256"] != sb["dataset"]["split_manifest_sha256"]:
        problems.append("split_manifest_sha256 differs")

    # ---- bitwise identity of the two runs ----
    same_rows = (a_dir / "per_image.jsonl").read_bytes() == (b_dir / "per_image.jsonl").read_bytes()
    print(f"per_image.jsonl byte-identical: {same_rows}")
    if not same_rows:
        problems.append("per_image.jsonl differs between the two runs")
    with np.load(a_dir / "sufficient_stats.npz", allow_pickle=False) as za, \
            np.load(b_dir / "sufficient_stats.npz", allow_pickle=False) as zb:
        if sorted(za.files) != sorted(zb.files):
            problems.append(f"NPZ key sets differ: {sorted(za.files)} vs {sorted(zb.files)}")
        else:
            for k in sorted(za.files):
                if za[k].dtype != zb[k].dtype or za[k].shape != zb[k].shape \
                        or za[k].tobytes() != zb[k].tobytes():
                    problems.append(f"NPZ {k} differs (dtype/shape/bytes)")
        image_index, pred, gt = za["image_index"], za["pred"], za["gt"]
    print(f"sufficient_stats.npz arrays bitwise equal: "
          f"{not any(p.startswith('NPZ') for p in problems)}")

    def strip(s: dict) -> dict:
        s = json.loads(json.dumps(s))
        for k in VOLATILE_RUN_KEYS:
            s["run"].pop(k, None)
        return s
    if _canonical(strip(sa)) != _canonical(strip(sb)):
        diff = sorted(k for k in set(sa) | set(sb) if _canonical(strip(sa).get(k))
                      != _canonical(strip(sb).get(k)))
        problems.append(f"summary.json differs outside run_id/timestamp_utc: {diff}")

    # ---- every image scored at its original resolution ----
    rows = _rows(a_dir)
    bad_shape, bad_res = [], []
    per_pred = np.bincount(image_index, weights=pred, minlength=len(rows))
    per_gt = np.bincount(image_index, weights=gt, minlength=len(rows))
    for pos, r in enumerate(rows):
        try:
            shapes = P.validate_shapes({k: r.get(k) for k in P.SHAPE_FIELDS})
        except (P.ProtocolError, TypeError) as e:
            bad_shape.append(f"{r.get('image_id')}: {e}")
            continue
        h, w = shapes["ori_shape"]
        if not int(per_pred[pos]) == int(per_gt[pos]) == h * w:
            bad_res.append(f"{r['image_id']}: sum pred {int(per_pred[pos])}, sum gt "
                           f"{int(per_gt[pos])}, H*W {h * w}")
    print(f"rows: {len(rows)}; shape fields valid: {not bad_shape}; every prediction at ori_shape "
          f"(sum pred == sum gt == H*W): {not bad_res}")
    if bad_shape:
        problems.append(f"{len(bad_shape)} row(s) with invalid shape fields, e.g. {bad_shape[:3]}")
    if bad_res:
        problems.append(f"{len(bad_res)} row(s) not scored at the original resolution, "
                        f"e.g. {bad_res[:3]}")

    # ---- reported values (no threshold) ----
    lvl, am5 = sa["dataset_level"], sa.get("am5") or {}
    print(f"model: stage={sa['run']['stage']} role={sa['run']['model_role']} "
          f"checkpoint_sha256={sa['run'].get('checkpoint_sha256')}")
    print(f"upstream all_class_miou={lvl['all_class_miou']!r} (union-present, "
          f"n_eligible={lvl['all_class_miou_n_eligible']}); disease_only_miou="
          f"{lvl['disease_only_miou']!r} (n_eligible={lvl['disease_only_miou_n_eligible']})")
    print(f"am5: excluded_count={am5.get('excluded_count')} included_count={am5.get('included_count')}")
    if canvas_value is not None and lvl["all_class_miou"] is not None:
        delta = lvl["all_class_miou"] - canvas_value
        print(f"protocol effect (upstream - canvas): {lvl['all_class_miou']!r} - {canvas_value!r} = "
              f"{delta!r} ({delta * 100:+.4f} pp)")

    for p in problems:
        print(f"D5: {p}")
    if problems:
        print("RESULT: D5 FAIL (STOP and report)")
        return EXIT_FAIL
    print("RESULT: D5 PASS -- 846 rows, every prediction at ori_shape, the two runs bitwise identical")
    return EXIT_PASS


def preflight() -> int:
    from PIL import Image

    from src.data.original_resolution import PlantSegOriginalResolutionDataset
    from src.eval import protocols as P

    try:
        pairs = PlantSegOriginalResolutionDataset("val").pairs     # listing + the locked 846 count
    except Exception as e:                                          # noqa: BLE001
        raise Unreadable(f"VAL listing failed: {type(e).__name__}: {e}") from e
    mismatched, transposed = [], 0
    largest = (0, "", (0, 0))
    for img_path, mask_path in pairs:
        with Image.open(img_path) as im, Image.open(mask_path) as mk:
            w, h = im.size
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):        # EXIF orientations that transpose
                w, h = h, w
                transposed += 1
            mw, mh = mk.size
        if (h, w) != (mh, mw):
            mismatched.append(f"{img_path.stem}: image {h}x{w} after EXIF, mask {mh}x{mw}")
        if h * w > largest[0]:
            largest = (h * w, img_path.stem, (h, w))
    _, stem, (h, w) = largest
    shapes = P.upstream_shapes(h, w)
    ph, pw = shapes["padded_shape"]
    mib = 1024 ** 2
    print(f"VAL pairs: {len(pairs)}; EXIF-transposed images: {transposed}; image/mask shape "
          f"mismatches: {len(mismatched)}")
    print(f"largest image: {stem} {h}x{w} -> rescaled {shapes['rescaled_shape']}, padded "
          f"{shapes['padded_shape']}; logits at original resolution {116 * h * w * 4 / mib:.0f} MiB, "
          f"at the padded input {116 * ph * pw * 4 / mib:.0f} MiB")
    for m in mismatched[:10]:
        print(f"PREFLIGHT: {m}")
    if mismatched:
        print("RESULT: PREFLIGHT FAIL -- the upstream protocol refuses these pairs (STOP and report)")
        return EXIT_FAIL
    print("RESULT: PREFLIGHT PASS -- every VAL image matches its mask after EXIF orientation")
    return EXIT_PASS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="L-AM13 real-data checks (local only).")
    sub = p.add_subparsers(dest="check", required=True)
    sub.add_parser("preflight")
    s = sub.add_parser("d5")
    s.add_argument("run_a", type=Path)
    s.add_argument("run_b", type=Path)
    s.add_argument("--canvas-value", type=float, default=None,
                   help="the model's canvas VAL all-class mIoU (e.g. E1 CPU 0.36307525634765625, "
                        "teacher 0.38576993346214294), to report the protocol effect")
    args = p.parse_args(argv)
    try:
        if args.check == "preflight":
            return preflight()
        return d5(args.run_a, args.run_b, args.canvas_value)
    except Unreadable as e:
        print(f"RESULT: ERROR -- {e}")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
