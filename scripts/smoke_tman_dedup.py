#!/usr/bin/env python3
"""Smoke for the duplicate-free list, src/tman/dedup.py (lane TMAN, B-3a part 1, B3A-1; AM-18 item 8(c); plan
decision (ix); rulings H3-H7).

- The rule (H3) on an 18-image fixture: TRAIN first, then VAL, then the duplicate groups of the split, each keeping its
  first sorted id; an image equal to both a TRAIN and a VAL image removed under equal_train and also counted; a group
  equal to a TRAIN image removed whole. Code-point edges of "first sorted": an upper-case letter before a lower-case
  one, "10" before "9", "-" before "_", an upper-case letter before "_", "-" before a digit.
- The partition invariants (H5), each with a case: the sum of removed_by_reason is n_images minus the kept ids; the
  id lists are disjoint and their union is the image set; and the doc's other rules.
- The TRAIN/VAL list parser: its sha256 is compared before it is parsed; header, row count, split counts and rows are
  checked, the counts passed by parameter as the fixture's; the header is scripts/hash_split_files.py :232's.
- Deterministic canonical bytes; the hash list; the strict reader's refusals (each form rule of doc_errors with every
  earlier rule passing, and a lone surrogate escape); the listed ids' refusals.
- LREP's reader rules restated (src/stats/heldout_inputs.py :126-127 and :1369-1377 at lane/lrep 6c222c9; no LREP
  code is imported: it is not on this branch), the export's DEDUP_SPEC (src/tman/validate.py :1612) and B-4's
  partition equality.
- Issue #11's counts and the plan's labels as values only; HASHER_VALUES; a no-score static scan in B-1's noscore form.

Synthetic inputs only: nothing is written and no data is read. Run with TMPDIR set to a new folder:
  TMPDIR=<new folder> python -B scripts/smoke_tman_dedup.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.smoke_tman_schema as FX  # noqa: E402  (the TMAN fixture library)

check = FX.check
WORD = "te" + "st"                       # project rule 4's word, never written whole here
HSF_HEADER = "split\tstem\tfile\tbytes\tsha256\tsha1"      # scripts/hash_split_files.py :232, restated
SCRIPT = {"path": "scripts/tday_hash_images.py", "sha256": "5" * 64}

# The fixture: TRAIN and VAL images (stem -> bytes) and the split's 18 images
TRAIN = {"tr_a": b"T1", "tr_b": b"T2", "tr_c": b"TB"}
VAL = {"va_a": b"V1", "va_b": b"TB", "va_c": b"V2"}
IMAGES = {"kx_t1": b"T1", "kx_t2a": b"T2", "kx_t2b": b"T2", "kx_tb": b"TB",      # TRAIN (kx_tb is also a VAL image's)
          "kx_v1": b"V1", "kx_v1b": b"V1",                                      # VAL
          "kx_B1": b"G1", "kx_a1": b"G1", "kx_c1": b"G1",                       # B (0x42) before a (0x61)
          "kx_x10": b"G2", "kx_x9": b"G2",                                      # "10" before "9"
          "kx_y_1": b"G3", "kx_y-1": b"G3",                                     # "-" (0x2D) before "_" (0x5F)
          "kx_q_": b"G4", "kx_qA": b"G4",                                       # "A" (0x41) before "_" (0x5F)
          "kx_z-": b"G5", "kx_z0": b"G5",                                       # "-" (0x2D) before "0" (0x30)
          "kx_u1": b"U1"}                                                       # alone: kept
WANT_REMOVED = {"equal_train": ["kx_t1", "kx_t2a", "kx_t2b", "kx_tb"], "equal_val": ["kx_v1", "kx_v1b"],
                "duplicate_in_split": ["kx_a1", "kx_c1", "kx_q_", "kx_x9", "kx_y_1", "kx_z0"]}
WANT_KEPT = ["kx_B1", "kx_qA", "kx_u1", "kx_x10", "kx_y-1", "kx_z-"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tv_bytes(train: dict, val: dict, header: str = HSF_HEADER, tail: str = "\n") -> bytes:
    """A TRAIN/VAL list in scripts/hash_split_files.py's format (:232-233)."""
    rows = [("train", s, c) for s, c in sorted(train.items())] + [("val", s, c) for s, c in sorted(val.items())]
    body = "".join(f"{sp}\t{s}\t{s}.jpg\t{len(c)}\t{sha(c)}\t{hashlib.sha1(c).hexdigest()}\n" for sp, s, c in rows)
    return (header + "\n" + body)[:-1].encode("utf-8") + tail.encode("utf-8")


def outcome(fn, *a, **kw):
    """("ok", value), ("refused", code, detail) or ("raised", type name): never raises (a mutation that crashes a
    call fails its check instead of the case)."""
    from src.tman.schema import TmanRefused
    try:
        return ("ok", fn(*a, **kw))
    except TmanRefused as e:
        return ("refused", e.code, e.detail)
    except Exception as e:  # noqa: BLE001
        return ("raised", type(e).__name__)


def fixture_doc():
    from src.tman import dedup as D
    tvb = tv_bytes(TRAIN, VAL)
    sets = D.parse_train_val_list(tvb, sha256=sha(tvb), rows=6, train=3, val=3)
    return D.build_doc({k: sha(v) for k, v in IMAGES.items()}, sets, hash_list_sha256="4" * 64,
                       train_val_list_sha256=sha(tvb), script=SCRIPT)


def built(images: dict, train=TRAIN, val=VAL):
    from src.tman import dedup as D
    tvb = tv_bytes(train, val)
    sets = D.parse_train_val_list(tvb, sha256=sha(tvb), rows=len(train) + len(val), train=len(train), val=len(val))
    return outcome(D.build_doc, {k: sha(v) for k, v in images.items()}, sets, hash_list_sha256="4" * 64,
                   train_val_list_sha256=sha(tvb), script=SCRIPT)


# ================================================================================================ the rule (H3)
def case_rule():
    got = built(IMAGES)
    doc = got[1] if got[0] == "ok" else {}
    rem, kept = doc.get("removed_ids_by_reason") or {}, doc.get("kept_ids") or []
    check("fixture_builds", got[0] == "ok", str(got))
    check("rule_equal_train_first_H3", rem.get("equal_train") == WANT_REMOVED["equal_train"], str(rem))
    check("rule_equal_val_second_H3", rem.get("equal_val") == WANT_REMOVED["equal_val"], str(rem))
    check("rule_duplicates_of_the_split_third_H3", rem.get("duplicate_in_split") == WANT_REMOVED["duplicate_in_split"],
          str(rem))
    check("both_match_removed_under_equal_train_and_also_counted_plan577", "kx_tb" in (rem.get("equal_train") or [])
          and "kx_tb" not in (rem.get("equal_val") or []) and doc.get("also_counted") == {"equal_train_and_val": 1},
          str(doc.get("also_counted")))
    check("group_equal_to_a_train_image_removed_whole", {"kx_t2a", "kx_t2b"} <= set(rem.get("equal_train") or [])
          and not {"kx_t2a", "kx_t2b"} & set(kept))
    check("kept_ids_are_each_group_s_first_sorted_id", kept == WANT_KEPT, str(kept))
    check("counts_by_reason_4_2_6", doc.get("removed_by_reason") == {"equal_train": 4, "equal_val": 2,
                                                                     "duplicate_in_split": 6})
    check("reasons_partition_true_in_the_built_doc", doc.get("reasons_partition") is True)
    for name, keep, drop in (("upper_case_before_lower_case", "kx_B1", ("kx_a1", "kx_c1")),
                             ("10_before_9", "kx_x10", ("kx_x9",)),
                             ("hyphen_before_underscore", "kx_y-1", ("kx_y_1",)),
                             ("upper_case_before_underscore", "kx_qA", ("kx_q_",)),
                             ("hyphen_before_a_digit", "kx_z-", ("kx_z0",))):
        check(f"code_point_{name}_keeps_{keep}", keep in kept and not set(drop) & set(kept)
              and set(drop) <= set(rem.get("duplicate_in_split") or []), str(kept))
    check("want_lists_sorted_by_code_point", WANT_KEPT == sorted(WANT_KEPT)
          and all(v == sorted(v) for v in WANT_REMOVED.values()))
    from src.tman import dedup as D
    check("reasons_precedence_and_labels_plan574", D.REASONS == ("equal_train", "equal_val", "duplicate_in_split")
          and D.REASON_LABELS == {"equal_train": "= TRAIN", "equal_val": "= VAL",
                                  "duplicate_in_split": "duplicate " + WORD.upper() + " image"})


# ============================================================================================ the invariants (H5)
def case_invariants():
    from src.tman import dedup as D
    doc = fixture_doc()
    n, kept, counts = doc["n_images"], doc["kept_ids"], doc["removed_by_reason"]
    lists = [kept] + [doc["removed_ids_by_reason"][r] for r in D.REASONS]
    every = [i for ids in lists for i in ids]
    check("invariant_sum_removed_is_images_minus_kept", sum(counts.values()) == n - len(kept) == 12 and n == 18)
    check("invariant_lists_disjoint_union_is_the_image_set", len(every) == len(set(every)) == n
          and set(every) == set(IMAGES))
    check("reasons_partition_true", doc["reasons_partition"] is True)
    check("built_doc_has_no_error", D.doc_errors(doc) == [], str(D.doc_errors(doc)))

    def errs_of(edit) -> str:
        d = copy.deepcopy(doc)
        edit(d)
        return " | ".join(D.doc_errors(d))

    def more(d):
        d["n_images"] += 1
    check("invariant_sum_broken_reported", "the sum of removed_by_reason" in errs_of(more))

    def overlap(d):
        d["removed_ids_by_reason"]["duplicate_in_split"] = sorted(d["removed_ids_by_reason"]["duplicate_in_split"]
                                                                  + [d["kept_ids"][0]])
        d["removed_by_reason"]["duplicate_in_split"] += 1
        d["n_images"] += 1
    check("invariant_disjoint_broken_reported", "not disjoint" in errs_of(overlap))

    def foreign(d):
        ids = d["removed_ids_by_reason"]["duplicate_in_split"]
        d["removed_ids_by_reason"]["duplicate_in_split"] = sorted(ids[1:] + ["kx_zz_foreign"])
    check("invariant_union_broken_reported", "not the image set" in errs_of(foreign))

    def off(d):
        d["removed_by_reason"]["equal_val"] += 1
    check("invariant_counts_are_the_lists_lengths", "length of each removed list" in errs_of(off))

    def flag(d):
        d["reasons_partition"] = False
    check("invariant_partition_flag_false_reported", "reasons_partition is not true" in errs_of(flag))

    def kept_fp(d):
        d["kept_ids_sha256"] = "6" * 64
    check("invariant_kept_fingerprint_reported", "kept_ids_sha256 is not" in errs_of(kept_fp))

    def also(d):
        d["also_counted"]["equal_train_and_val"] = d["removed_by_reason"]["equal_train"] + 1
    check("invariant_also_counted_at_most_equal_train", "exceeds equal_train" in errs_of(also))

    def unsorted(d):
        d["kept_ids"] = list(reversed(d["kept_ids"]))
    check("invariant_kept_ids_sorted_and_unique", "kept_ids: not sorted and unique" in errs_of(unsorted))

    def padded(d):
        d["removed_ids_by_reason"]["equal_val"][0] = " " + d["removed_ids_by_reason"]["equal_val"][0]
    check("invariant_ids_without_surrounding_whitespace", "not a list of ids" in errs_of(padded))
    got = built({"kx_t1": b"T1", "kx_t2a": b"T2"})
    check("invariant_kept_not_empty_build_refused", got[0] == "refused" and got[1] == "dedup_invalid"
          and "kept_ids is empty" in got[2], str(got))


# =================================================================================== the TRAIN/VAL list (H9 step 4)
def case_list_parser():
    from src.tman import dedup as D
    tvb = tv_bytes(TRAIN, VAL)
    got = outcome(D.parse_train_val_list, tvb, sha256=sha(tvb), rows=6, train=3, val=3)
    check("list_parses_with_the_fixture_counts", got[0] == "ok" and got[1] == {
        "train": frozenset(sha(c) for c in TRAIN.values()), "val": frozenset(sha(c) for c in VAL.values())}, str(got))
    garbage = b"\xff\xfe not UTF-8, no header\n"
    got = outcome(D.parse_train_val_list, garbage, sha256=sha(tvb), rows=6, train=3, val=3)
    check("list_wrong_sha256_refused_before_parsing", got[0] == "refused" and got[1] == "train_val_list_mismatch"
          and "not parsed" in got[2], str(got))
    for name, data, kw, want in (
            ("list_not_utf8_refused", garbage, {}, "not UTF-8"),
            ("list_without_final_lf_refused", tv_bytes(TRAIN, VAL, tail=""), {}, "does not end with LF"),
            ("list_wrong_header_refused", tv_bytes(TRAIN, VAL, header=HSF_HEADER.replace("sha1", "sha_1")), {},
             "header"),
            ("list_row_count_refused", tvb, {"rows": 7}, "data rows after the header"),
            ("list_split_counts_refused", tvb, {"train": 4, "val": 2}, "split counts"),
            ("list_row_of_five_fields_refused", tvb.replace(b"\tva_c.jpg", b""), {}, "line 7 is not a"),
            ("list_row_with_an_upper_case_sha256_refused", tvb.replace(sha(b"V2").encode(),
                                                                        sha(b"V2").upper().encode()), {},
             "line 7 is not a"),
            ("list_row_of_another_split_refused", tvb.replace(b"val\tva_c", b"eval\tva_c"), {}, "line 7 is not a"),
            ("list_row_with_a_non_digit_byte_count_refused", tvb.replace(b"va_c.jpg\t2\t", b"va_c.jpg\t2x\t"), {},
             "line 7 is not a"),
            ("list_row_with_a_non_hex_sha1_refused", tvb.replace(hashlib.sha1(b"V2").hexdigest().encode(), b"z" * 40),
             {}, "line 7 is not a")):
        args = {"rows": 6, "train": 3, "val": 3}
        args.update(kw)
        got = outcome(D.parse_train_val_list, data, sha256=sha(data), **args)
        check(name, got[0] == "refused" and got[1] == "train_val_list_invalid" and want in got[2], str(got))
    hsf = REPO / "scripts/hash_split_files.py"
    tree = ast.parse(hsf.read_text(encoding="utf-8"))
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == HSF_HEADER + "\n"]
    check("list_header_is_hash_split_files_232", lines == [232] and "\t".join(D.TRAINVAL_HEADER) == HSF_HEADER,
          str(lines))


# ===================================================================================== canonical bytes and the TSV
def case_canonical_and_hash_list():
    from src.eval.artifacts import canonical_json_bytes
    from src.tman import dedup as D
    doc = fixture_doc()
    tvb = tv_bytes(TRAIN, VAL)
    sets = D.parse_train_val_list(tvb, sha256=sha(tvb), rows=6, train=3, val=3)
    again = D.build_doc({k: sha(IMAGES[k]) for k in reversed(sorted(IMAGES))}, sets, hash_list_sha256="4" * 64,
                        train_val_list_sha256=sha(tvb), script=dict(reversed(list(SCRIPT.items()))))
    check("canonical_bytes_independent_of_input_order", D.doc_bytes(doc) == D.doc_bytes(again))
    check("canonical_bytes_are_artifacts_canonical_json", D.doc_bytes(doc) == canonical_json_bytes(doc)
          == json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    rows = [{"stem": s, "file": s + ".jpg", "bytes": len(c), "sha256": sha(c)} for s, c in IMAGES.items()]
    tsv = D.hash_list_bytes(rows)
    want = "stem\tfile\tbytes\tsha256\n" + "".join(f"{s}\t{s}.jpg\t{len(IMAGES[s])}\t{sha(IMAGES[s])}\n"
                                                  for s in sorted(IMAGES))
    check("hash_list_header_sorted_rows_lf_utf8_H6", tsv == want.encode("utf-8"))
    check("hash_list_independent_of_row_order", D.hash_list_bytes(list(reversed(rows))) == tsv)


# ============================================================================================ the strict reader
def case_reader():
    from src.tman import dedup as D
    doc = fixture_doc()
    data = D.doc_bytes(doc)
    got = outcome(D.read_dedup, data, expect_sha256=sha(data))
    check("reader_round_trip", got == ("ok", doc), str(got)[:200])
    got = outcome(D.read_dedup, b"{ not JSON", expect_sha256=sha(data))
    check("reader_sha256_compared_before_parsing", got[0] == "refused" and got[1] == "dedup_invalid"
          and "not parsed" in got[2], str(got))
    dup = data.replace(b'"format":', b'"format":"tday_dedup/1","format":', 1)
    pretty = json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    nan = data.replace(b'"n_images":18', b'"n_images":NaN')
    extra = D.doc_bytes(dict(doc, note="x"))
    missing = D.doc_bytes({k: v for k, v in doc.items() if k != "script"})
    fmt = D.doc_bytes(dict(doc, format="tday_dedup/2"))
    broken = D.doc_bytes(dict(doc, kept_ids=list(reversed(doc["kept_ids"]))))
    lone = data.replace(b'"kx_u1"', b'"kx_u1\\ud800"', 1)              # strict JSON; no UTF-8 for its string

    def edit(**kw):
        return D.doc_bytes(dict(doc, **kw))
    for name, bad, want in (("reader_duplicate_key_refused", dup, "not strict"),
                            ("reader_non_canonical_bytes_refused", pretty, "not the canonical JSON"),
                            ("reader_nan_refused", nan, "not strict"),
                            ("reader_extra_key_refused", extra, "exactly the keys"),
                            ("reader_missing_key_refused", missing, "exactly the keys"),
                            ("reader_other_format_refused", fmt, "is not 'tday_dedup/1'"),
                            ("reader_broken_invariant_refused", broken, "not sorted and unique"),
                            ("reader_lone_surrogate_escape_refused", lone, "lone surrogate"),
                            # each form rule of doc_errors, every earlier rule passing (DL-24 r3-F5)
                            ("reader_n_images_not_an_integer_refused", edit(n_images=float(doc["n_images"])),
                             "n_images is not a positive integer"),
                            ("reader_sha256_field_not_hex64_refused", edit(train_val_list_sha256="abc"),
                             "train_val_list_sha256: 'abc' is not sha256"),
                            ("reader_script_extra_key_refused", edit(script=dict(doc["script"], note="x")),
                             "script: unexpected ['note']"),
                            ("reader_kept_ids_not_a_list_refused", edit(kept_ids=doc["kept_ids"][0]),
                             "kept_ids: not a list of ids"),
                            ("reader_fourth_reason_in_the_id_lists_refused",
                             edit(removed_ids_by_reason=dict(doc["removed_ids_by_reason"], other=[])),
                             "removed_ids_by_reason does not hold exactly"),
                            ("reader_fourth_reason_in_the_counts_refused",
                             edit(removed_by_reason=dict(doc["removed_by_reason"], other=0)),
                             "removed_by_reason is not a count >= 0"),
                            ("reader_also_counted_second_key_refused",
                             edit(also_counted=dict(doc["also_counted"], other=0)),
                             "also_counted is not a count >= 0")):
        got = outcome(D.read_dedup, bad)
        check(name, got[0] == "refused" and got[1] == "dedup_invalid" and want in got[2], str(got)[:240])
    got = outcome(D.hasher_values, D.hash_list_bytes([]), lone)
    check("hasher_values_lone_surrogate_refused", got[0] == "refused" and got[1] == "dedup_invalid"
          and "lone surrogate" in got[2], str(got)[:240])


# ================================================================================================ the listed ids
def case_image_ids():
    from src.tman import dedup as D
    ok = [("kx_a", "kx_a.jpg"), ("kx_b", "kx_b.JPEG")]
    check("ids_clean_accepted", D.image_id_errors(ok) == [])
    for name, pairs, want in (
            ("ids_surrounding_whitespace_refused", ok + [(" kx_c", " kx_c.jpg")], "surrounding whitespace"),
            ("ids_tab_in_a_name_refused", ok + [("kx\tc", "kx\tc.jpg")], "a tab, CR or LF"),
            ("ids_cr_in_a_name_refused", ok + [("kx\rc", "kx\rc.jpg")], "a tab, CR or LF"),
            ("ids_lf_in_a_name_refused", ok + [("kx\nc", "kx\nc.jpg")], "a tab, CR or LF"),
            ("ids_not_utf8_refused", ok + [("kx_\udcff", "kx_\udcff.jpg")], "not UTF-8"),
            ("ids_shared_stem_refused", ok + [("kx_a", "kx_a.jpeg")], "share a stem"),
            ("ids_case_folded_collision_refused", ok + [("KX_A", "KX_A.jpg")], "case-folded")):
        errs = D.image_id_errors(sorted(pairs))
        check(name, any(want in e for e in errs), str(errs))
    errs = D.image_id_errors(sorted(ok + [("kx_secret_name", "kx_secret_name.jpg"),
                                          ("KX_SECRET_NAME", "KX_SECRET_NAME.jpg")]))
    check("ids_refusals_name_no_file_P21", errs and not any("secret" in e.lower() for e in errs), str(errs))


# ======================================================================================== LREP's reader (B5-S2)
def _is_text(v) -> bool:
    """LREP's _is_text (src/stats/heldout_inputs.py :261-262 at 6c222c9), restated."""
    return isinstance(v, str) and bool(v) and v == v.strip()


def lrep_read(export: dict, data: bytes) -> dict:
    """LREP's read_dedup rules (heldout_inputs.py :1334-1379 at 6c222c9), restated without its import: the export's
    {path, sha256, removed_by_reason} (:126); the sha256 compared before parsing; strict JSON; the file holds kept_ids
    and removed_by_reason (:127); kept_ids a non-empty, sorted, unique list of ids (:1371-1374); removed_by_reason
    {id: int >= 0} equal to the export's (:1375-1377); any other key kept as other_fields (:1379)."""
    def counts(v):
        ok = isinstance(v, dict) and all(_is_text(k) and isinstance(n, int) and not isinstance(n, bool) and n >= 0
                                         for k, n in v.items())
        return tuple(sorted(v.items())) if ok else None

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)
    if not (isinstance(export, dict) and all(k in export for k in ("path", "sha256", "removed_by_reason"))):
        raise ValueError("export dedup fields")
    c = counts(export["removed_by_reason"])
    if c is None or hashlib.sha256(data).hexdigest() != export["sha256"]:
        raise ValueError("export counts or sha256")
    doc = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not (isinstance(doc, dict) and "kept_ids" in doc and "removed_by_reason" in doc):
        raise ValueError("file fields")
    ids = doc["kept_ids"]
    if not (isinstance(ids, list) and ids and all(_is_text(i) for i in ids) and ids == sorted(set(ids))):
        raise ValueError("kept_ids")
    if counts(doc["removed_by_reason"]) != c:
        raise ValueError("removed_by_reason")
    return {"kept_ids": ids, "removed_by_reason": c,
            "other_fields": sorted(set(doc) - {"kept_ids", "removed_by_reason"})}


def case_lrep_rules():
    from src.tman import dedup as D
    from src.tman import schema as S
    from src.tman import validate as V
    doc = fixture_doc()
    data = D.doc_bytes(doc)
    export = {"path": D.DEDUP_NAME, "sha256": sha(data), "removed_by_reason": dict(doc["removed_by_reason"])}
    try:
        rec = lrep_read(export, data)
        err = ""
    except (ValueError, KeyError, TypeError) as e:
        rec, err = None, str(e)
    check("lrep_reader_rules_accept_the_file_H5", rec is not None and rec["kept_ids"] == WANT_KEPT, err)
    check("lrep_other_fields_are_the_rest_of_h5", rec is not None and rec["other_fields"] == sorted(
        set(D.DOC_KEYS) - {"kept_ids", "removed_by_reason"}))
    try:
        lrep_read(dict(export, removed_by_reason=dict(export["removed_by_reason"], equal_val=3)), data)
        refused = False
    except ValueError:
        refused = True
    check("lrep_restatement_refuses_another_count", refused)
    check("export_passes_dedup_spec_validate_1612", S.value_errors(export, V.DEDUP_SPEC, "dedup") == [],
          str(S.value_errors(export, V.DEDUP_SPEC, "dedup")))
    every = list(IMAGES)
    check("b4_partition_equality_all_minus_kept_is_the_sum", len(set(every)) - len(doc["kept_ids"])
          == sum(doc["removed_by_reason"].values()) and set(doc["kept_ids"]) <= set(every))
    ids = ["b", "a", "c"]
    check("ids_sha256_is_lrep_s_definition_no_trailing_lf", D.ids_sha256(ids) == sha(b"a\nb\nc")
          and D.ids_sha256(ids) != sha(b"a\nb\nc\n"))
    check("image_and_kept_fingerprints", doc["image_ids_sha256"] == sha("\n".join(sorted(IMAGES)).encode())
          and doc["kept_ids_sha256"] == sha("\n".join(WANT_KEPT).encode()))


# ========================================================================= issue #11's counts, labels, keys (H4)
def keys_of(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from keys_of(v)
    elif isinstance(x, list):
        for v in x:
            yield from keys_of(v)


def case_issue_rows_and_keys():
    from src.tman import dedup as D
    check("issue_counts_pinned_70_11_13", {r: D.ISSUE_ROWS[r][1] for r in D.REASONS}
          == {"equal_train": 70, "equal_val": 11, "duplicate_in_split": 13})
    arrow = "↔"
    check("issue_labels_are_values_of_reason_keys", list(D.ISSUE_ROWS) == list(D.REASONS)
          and [D.ISSUE_ROWS[r][0] for r in D.REASONS] == [f"{WORD} {arrow} train", f"{WORD} {arrow} val",
                                                          f"{WORD} {arrow} {WORD}"])
    pa = (REPO / "docs/PREREGISTRATION_AMENDMENTS.md").read_text(encoding="utf-8").splitlines()
    found = {r: [i + 1 for i, ln in enumerate(pa) if ln.strip() == f"- {D.ISSUE_ROWS[r][0]} {D.ISSUE_ROWS[r][1]};"]
             for r in D.REASONS}
    check("issue_rows_are_the_amendment_s_lines_once_each", all(len(v) == 1 for v in found.values()), str(found))
    doc = fixture_doc()
    keys = list(keys_of(doc))
    check("no_json_key_holds_the_word_or_a_colon_H4", keys and not any(WORD in k.lower() or ":" in k for k in keys),
          str(keys))
    ids = doc["kept_ids"] + [i for r in D.REASONS for i in doc["removed_ids_by_reason"][r]]
    check("no_fixture_id_holds_the_word_or_a_colon", not any(WORD in i.lower() or ":" in i for i in ids))
    check("doc_keys_are_h5_s", sorted(doc) == sorted(D.DOC_KEYS) == sorted(
        ["format", "n_images", "image_ids_sha256", "kept_ids_sha256", "kept_ids", "removed_ids_by_reason",
         "removed_by_reason", "reasons_partition", "also_counted", "hash_list_sha256", "train_val_list_sha256",
         "script"]) and doc["format"] == "tday_dedup/1")
    got = built({"kx_u1": b"U1", "kx_u2": b"U2"})
    check("removed_by_reason_holds_all_three_reasons_zeros_included", got[0] == "ok" and got[1]["removed_by_reason"]
          == {"equal_train": 0, "equal_val": 0, "duplicate_in_split": 0}, str(got)[:200])


# ================================================================================================= HASHER_VALUES
def case_hasher_values():
    from src.tman import dedup as D
    from src.tman import manifest as M
    doc = fixture_doc()
    tsv = D.hash_list_bytes([{"stem": s, "file": s + ".jpg", "bytes": 2, "sha256": sha(c)} for s, c in IMAGES.items()])
    doc["hash_list_sha256"] = sha(tsv)
    data = D.doc_bytes(doc)
    got = outcome(D.hasher_values, tsv, data)
    check("hasher_values_keys_are_manifest_s_544", got[0] == "ok" and tuple(got[1]) == M.HASHER_VALUES
          and D.VALUE_KEYS == M.HASHER_VALUES, str(got)[:200])
    check("hasher_values_are_the_sha256_and_the_counts", got[0] == "ok" and got[1] == {
        "hash_list_sha256": sha(tsv), "dedup_list_sha256": sha(data), "counts_by_reason": doc["removed_by_reason"]})
    got = outcome(D.hasher_values, tsv + b"x", data)
    check("hasher_values_refuse_a_dedup_of_another_hash_list", got[0] == "refused" and got[1] == "dedup_invalid",
          str(got))


# ======================================================================================== static (B-1's noscore form)
def code_literals(path: Path) -> list[str]:
    """The string literals of a module, docstrings excepted (smoke_tman_noscore.py's code_literals, restated)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def case_static():
    from src.tman import dedup as D
    path = Path(D.__file__)
    hits = [s for s in code_literals(path) if FX.SCORE_KEY_RE.search(s)]
    check("noscore_static_scan_of_dedup", not hits, str(hits))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
                  | {("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})
    check("dedup_imports_the_standard_library_and_schema_only", mods == [".", "__future__", "re"]
          and [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 for a in n.names]
          == ["schema"], str(mods))
    check("dedup_imports_no_src_stats", not any("src.stats" in m for m in mods) and "src.stats" not in sys.modules)


if __name__ == "__main__":
    raise SystemExit(FX.run_cases("TMAN DEDUP SMOKE (lane TMAN, B-3a part 1, B3A-1)",
                                  (case_rule, case_invariants, case_list_parser, case_canonical_and_hash_list,
                                   case_reader, case_image_ids, case_lrep_rules, case_issue_rows_and_keys,
                                   case_hasher_values, case_static)))
