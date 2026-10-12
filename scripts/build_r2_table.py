#!/usr/bin/env python3
"""R2's class table (lane L-TEACHER-ARMS, commit C4; AM-18 item 2(b); ARMS plan decision (ii); DL-24 audit F13,
F16, F18; Phase B-2 ruling R6).

    python -B scripts/build_r2_table.py --data-root ROOT --strata reports/strata/train_strata_v1.json \\
        --out-dir OUT [--expected-images N] [--generated-utc UTC]
    python -B scripts/build_r2_table.py --check TABLE --data-root ROOT --strata STRATA [--expected-images N]

It lists the TRAIN pairs with scripts/build_train_strata.train_split_list (names only; every path through the
"test" guard), reads each TRAIN mask's labels, and takes each image's disease class: the single label in 1..115,
or null when the mask holds no disease pixel. Two or more disease labels in one mask are a STOP (EVALUATION_
CONTRACT §0: every PlantSeg image holds one disease class). The per-class image counts n_c must equal the strata
file's `images` for all 115 classes, and the split list its split_list_sha256; the strata file itself must pass
build_train_strata.check_document (11 checks). Any disagreement writes r2_table_STOP.json (format
plantseg-r2-table-stop/1: every differing class, every two-label image) and no table: R2 does not launch while it
stands, and R1 waits for R2's table (AM-18 items 2(b), 7(a)).

Otherwise it writes r2_table_v1.json (plantseg-r2-table/1): t = 20/5,367; for each class n_c, rare (n_c < 20) and
m_c, the least m with m * m * n_c >= 20, computed in integers and cross-checked against mmengine's float formula
max(1, sqrt(t / (n_c / N))) under ceil; per image (in stem order) its class and m; the sorted source stems of
each rare class; the totals; the strata file's path, sha256 and check; the split list's sha256; the builder's
sha256 and commit. The bytes are canonical (sorted keys, indent 2, LF) and carry no time, so two builds of the
same inputs are identical; src.training.teacher_arm_components.validate_table must accept them. --check re-derives
the table from the same inputs and compares bytes.

Exit: 0 written (or --check identical); 1 STOP (STOP file written) or --check different; 2 refused, nothing
written. --expected-images and --generated-utc are for synthetic fixtures: the registered count is 5,367, and
--generated-utc (the STOP file's time) is refused at that count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXIT_OK, EXIT_STOP, EXIT_REFUSED = 0, 1, 2
TABLE_FILE, STOP_FILE = "r2_table_v1.json", "r2_table_STOP.json"
STRATA_REL = "reports/strata/train_strata_v1.json"
SCRIPT_REL = "scripts/build_r2_table.py"
REGISTERED_IMAGES = 5367
STRATA_CHECKS = 11                    # build_train_strata.check_document's checks of a valid strata file


class Refused(RuntimeError):
    pass


def strata_check(path) -> tuple[bool, str]:
    """The strata file passes build_train_strata.check_document (all 11 checks); a smoke replaces this hook."""
    import scripts.build_train_strata as bst
    results, _ = bst.check_document(path)
    bad = [name for name, ok, _ in results if not ok]
    return len(results) == STRATA_CHECKS and not bad, f"{len(results) - len(bad)}/{len(results)} checks; failing {bad}"


def canonical_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def mask_classes(pairs) -> tuple[dict, list]:
    """{stem: class or None} from each mask's labels, and the stems whose mask holds two or more disease labels."""
    import numpy as np
    from PIL import Image
    by_stem, two = {}, []
    for stem, path in pairs:
        try:
            with Image.open(path) as im:
                a = np.asarray(im)
        except (OSError, ValueError) as e:
            raise Refused(f"{Path(path).name}: unreadable mask ({type(e).__name__}: {e})") from e
        if a.ndim != 2 or a.size == 0 or a.dtype.kind not in "ui":
            raise Refused(f"{Path(path).name}: a mask must be a non-empty single-channel integer array")
        counts = np.bincount(a.reshape(-1).astype(np.int64), minlength=256)
        bad = [int(v) for v in np.flatnonzero(counts) if v > 115 and v != 255]
        if bad or counts.size > 256:
            raise Refused(f"{Path(path).name}: label value(s) {bad[:5]} outside 0..115 and 255")
        labels = [int(v) for v in np.flatnonzero(counts[1:116]) + 1]
        if len(labels) > 1:
            two.append(stem)
        by_stem[stem] = labels[0] if len(labels) == 1 else None
    return by_stem, two


def strata_differences(n_c: dict, strata: dict) -> list:
    """Every disease class whose image count in the masks differs from the strata file's `images` (AM-18 item 2(b):
    a count that disagrees with the strata file is a STOP)."""
    n_strata = {}
    for r in strata.get("per_class", []):
        n_strata[int(r["class_id"])] = int(r["images"])
    return [{"class_id": c, "n_c_masks": n, "images_strata": n_strata.get(c)} for c, n in sorted(n_c.items())
            if n_strata.get(c) != n]


def float_m(n_c: int, n_images: int) -> int:
    """mmengine ClassBalancedDataset's factor for a class of n_c images out of N, under ceil (the cross-check)."""
    t = 20 / 5367
    return math.ceil(max(1.0, math.sqrt(t / (n_c / n_images))))


def build(data_root, strata_path, *, expected_images: int) -> tuple[dict | None, dict | None]:
    """(table, None) or (None, stop) from the TRAIN masks and the strata file."""
    from src.training import teacher_arm_components as tac
    import scripts.build_train_strata as bst
    strata_b = Path(strata_path).read_bytes()
    try:
        ok, detail = strata_check(strata_path)
    except Exception as e:                    # noqa: BLE001 -- an unreadable strata file is a refusal, never a verdict
        raise Refused(f"--strata {strata_path} cannot be checked ({type(e).__name__}: {e})") from e
    if not ok:
        raise Refused(f"--strata {strata_path} does not pass build_train_strata.check_document ({detail})")
    strata = json.loads(strata_b.decode("utf-8"))
    try:
        pairs = bst.train_split_list(data_root, expected_images)
    except bst.StrataError as e:
        raise Refused(f"the TRAIN split list: {e}") from e
    stems = [s for s, _ in pairs]
    by_stem, two = mask_classes(pairs)
    n_c = {c: 0 for c in tac.DISEASE_IDS}
    for c in by_stem.values():
        if c is not None:
            n_c[c] += 1
    differ = strata_differences(n_c, strata)
    split_sha = bst.split_list_sha256(stems)
    strata_ref = {"path": STRATA_REL, "sha256": hashlib.sha256(strata_b).hexdigest(), "check": "PASS"}
    if two or differ or split_sha != strata.get("split_list_sha256") or len(stems) != strata.get("n_images"):
        return None, {"format": tac.STOP_FORMAT, "two_label_images": sorted(two), "differing_classes": differ,
                      "split_list_sha256": {"masks": split_sha, "strata": strata.get("split_list_sha256")},
                      "images": {"masks": len(stems), "strata": strata.get("n_images")}, "strata": strata_ref,
                      "rule": "AM-18 item 2(b): a count that disagrees with the strata file is a STOP; R2 does not "
                              "launch while it stands (two disease labels in one mask: EVALUATION_CONTRACT §0)"}
    per_class = []
    for c in tac.DISEASE_IDS:
        m = tac.m_of(n_c[c])
        if m is not None and m != float_m(n_c[c], len(stems)) and len(stems) == REGISTERED_IMAGES:
            raise Refused(f"class {c}: the exact m_c {m} differs from mmengine's float factor "
                          f"{float_m(n_c[c], len(stems))} (n_c {n_c[c]})")
        per_class.append({"class_id": c, "n_c": n_c[c], "rare": n_c[c] < tac.T_NUM, "m_c": m})
    rare = {r["class_id"]: r["m_c"] for r in per_class if r["rare"] and r["n_c"] >= 1}
    per_image = [{"stem": s, "class_id": by_stem[s], "m": rare.get(by_stem[s], 1) if by_stem[s] is not None else 1}
                 for s in sorted(stems)]
    table = {"format": tac.TABLE_FORMAT, "t": {"num": tac.T_NUM, "den": tac.T_DEN}, "strata": strata_ref,
             "split_list_sha256": split_sha, "per_class": per_class, "per_image": per_image,
             "sources": {str(c): sorted(s for s in stems if by_stem[s] == c) for c in rare},
             "totals": {"images": len(stems), "repeated": sum(r["m"] for r in per_image)},
             "builder": builder_provenance()}
    tac.validate_table(table)
    return table, None


def builder_provenance() -> dict:
    from src.training.sweep_select import HeadSource
    try:
        head = HeadSource(REPO).head() or None
    except OSError:
        head = None
    return {"script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), "commit": head}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R2's class table (L-TEACHER-ARMS C4).", allow_abbrev=False)
    p.add_argument("--data-root", required=True)
    p.add_argument("--strata", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--check", default=None)
    p.add_argument("--expected-images", type=int, default=REGISTERED_IMAGES)
    p.add_argument("--generated-utc", default=None)
    return p.parse_args(argv)


def run(args) -> int:
    from src.eval.teacher_diag import Refused as DiagRefused
    from src.eval.teacher_diag import refuse_test_path, require_outside_repo, utc_now, write_json_exclusive
    try:
        for value, flag in ((args.data_root, "--data-root"), (args.strata, "--strata"), (args.out_dir, "--out-dir"),
                            (args.check, "--check")):
            if value is not None:
                refuse_test_path(value, flag)
        if (args.out_dir is None) == (args.check is None):
            raise Refused("give exactly one of --out-dir (build) and --check (verify)")
        if args.generated_utc is not None and args.expected_images == REGISTERED_IMAGES:
            raise Refused("--generated-utc is for synthetic fixtures; a build at the registered 5,367 images uses "
                          "the clock")
        if not Path(args.strata).is_file():
            raise Refused(f"--strata {args.strata} is not a file")
        out = None
        if args.out_dir is not None:
            out = require_outside_repo(args.out_dir, "--out-dir")
            if out.exists():
                raise Refused(f"--out-dir {out} exists: every build writes a fresh folder")
        table, stop = build(args.data_root, args.strata, expected_images=args.expected_images)
        data = Path(args.check).read_bytes() if args.check is not None else None
    except (Refused, DiagRefused, OSError, ValueError, KeyError, TypeError) as e:
        print(f"RESULT: REFUSED -- {type(e).__name__}: {e}")
        return EXIT_REFUSED
    if args.check is not None:
        same = table is not None and canonical_bytes(table) == data
        print(f"RESULT: {'IDENTICAL' if same else 'DIFFERENT'} -- {args.check} sha256 "
              f"{hashlib.sha256(data).hexdigest()} vs the re-derived "
              f"{'STOP' if table is None else hashlib.sha256(canonical_bytes(table)).hexdigest()}")
        return EXIT_OK if same else EXIT_STOP
    out.mkdir(parents=True)
    if stop is not None:
        stop["generated_utc"] = args.generated_utc or utc_now()
        stop["builder"] = builder_provenance()
        path = write_json_exclusive(out / STOP_FILE, stop)
        print(f"  two-label images: {len(stop['two_label_images'])}; differing classes: "
              f"{[d['class_id'] for d in stop['differing_classes']]}")
        print(f"RESULT: STOP -- {path} (no table; R2 does not launch while it stands)")
        return EXIT_STOP
    data = canonical_bytes(table)
    path = out / TABLE_FILE
    with open(path, "xb") as fh:
        fh.write(data)
    rare = {r["class_id"]: (r["n_c"], r["m_c"]) for r in table["per_class"] if r["rare"]}
    print(f"  rare classes (n_c, m_c): {rare}")
    print(f"  images {table['totals']['images']}, repeated {table['totals']['repeated']}")
    print(f"RESULT: WRITTEN -- {path} sha256 {hashlib.sha256(data).hexdigest()}")
    return EXIT_OK


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
