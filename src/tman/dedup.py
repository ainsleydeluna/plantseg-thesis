"""The duplicate-free list of the split of record (lane TMAN, B-3a part 1; AM-18 item 8(c); plan decision (ix)).

The in-session hasher (scripts/tday_hash_images.py) hashes the split's image files -- SHA-256 of the file bytes; no
annotation and no model output is read -- and this module turns the hashes into its two outputs:

  tday_image_hashes.tsv  header stem, file, bytes, sha256 (tab-separated), one row per image sorted by stem, LF,
                         UTF-8 (ruling H6); its sha256 is hash_list_sha256
  tday_dedup.json        tday_dedup/1 (ruling H5): canonical JSON (schema.canonical_json_bytes, src/eval/artifacts.py
                         :93), the kept (duplicate-free) ids and the removed ids by reason; its sha256 is
                         dedup_list_sha256

The rule (AM-18 item 8(c), docs/PREREGISTRATION_AMENDMENTS.md :1217-1219 at ee5593e; plan :571-578; ruling H3). An
image's id is its stem; "sorted" is Python's sorted(), in code-point order. A removed image has exactly one reason,
taken in this precedence:
  1. equal_train         its sha256 equals a TRAIN image's;
  2. equal_val           otherwise it equals a VAL image's;
  3. duplicate_in_split  otherwise it belongs to a group of identical images of the split and is not the group's
                         first sorted id.
The reasons partition the removed set (reasons_partition true). An image equal to both a TRAIN and a VAL image is
removed under equal_train and also counted, as equal_train_and_val (plan :577). The plan's labels and issue #11's
counts are pinned string values and numbers (REASON_LABELS, ISSUE_ROWS); no JSON key holds a label (ruling H4).

The TRAIN/VAL list is DL-91's file of record, in scripts/hash_split_files.py's format (:232-233); its sha256 is the
pin's before it is parsed (ruling H9 step 4). The id fingerprints are LREP's (ids_sha256).

Pure functions: the standard library and B-1's schema helpers (src/tman/schema.py), by import. Refusals are
schema.TmanRefused codes; a refusal names offending files by count and sha256, never by name.
"""
from __future__ import annotations

import re

from . import schema as S

# ------------------------------------------------------------------------------------------- formats and names
DEDUP_FORMAT = "tday_dedup/1"                       # here, not in schema.py (ruling H2(a)); part 2 wires it there
HASH_LIST_NAME = "tday_image_hashes.tsv"            # plan :539
DEDUP_NAME = "tday_dedup.json"
HASH_LIST_HEADER = ("stem", "file", "bytes", "sha256")                     # ruling H6
TRAINVAL_HEADER = ("split", "stem", "file", "bytes", "sha256", "sha1")     # scripts/hash_split_files.py :232
TRAINVAL_SPLITS = ("train", "val")
DIGITS_RE = re.compile(r"[0-9]+")
SHA1_RE = re.compile(r"[0-9a-f]{40}")
TSV_UNSAFE = ("\t", "\n", "\r")                     # a name holding one would break its hash-list row
# manifest.HASHER_VALUES (src/tman/manifest.py :544), restated in its order; a smoke case compares the two
VALUE_KEYS = ("hash_list_sha256", "dedup_list_sha256", "counts_by_reason")

# ---------------------------------------------------------------------------------------------- the rule (H3)
REASONS = ("equal_train", "equal_val", "duplicate_in_split")                # in precedence order
ALSO_COUNTED = "equal_train_and_val"                                         # plan :577
# The plan's labels (:574-576), string values only: a JSON key never holds one (ruling H4)
REASON_LABELS = {"equal_train": "= TRAIN", "equal_val": "= VAL", "duplicate_in_split": "duplicate TEST image"}
# Issue #11's counts beside the three reasons (AM-18 item 8(a); plan :578), with the issue's own labels as string
# values: docs/PREREGISTRATION_AMENDMENTS.md :1196, :1199 and :1198 at ee5593e (:1200, :1203 and :1202 at 5838c80)
ISSUE_ROWS = {"equal_train": ("test ↔ train", 70), "equal_val": ("test ↔ val", 11),
              "duplicate_in_split": ("test ↔ test", 13)}
DOC_KEYS = ("also_counted", "format", "hash_list_sha256", "image_ids_sha256", "kept_ids", "kept_ids_sha256",
            "n_images", "reasons_partition", "removed_by_reason", "removed_ids_by_reason", "script",
            "train_val_list_sha256")                                         # ruling H5, sorted


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_id(x) -> bool:
    """LREP's id: a non-empty string without surrounding whitespace (src/stats/heldout_inputs.py :261-262 at
    lane/lrep 6c222c9)."""
    return isinstance(x, str) and x != "" and x == x.strip()


def _utf8(name: str) -> bool:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def ids_sha256(ids) -> str:
    """An id set's fingerprint: the sha256 of its sorted ids joined by LF, UTF-8, with no trailing LF. This is LREP's
    am6_tables.ids_sha256 (src/stats/am6_tables.py :98-100 at lane/lrep 6c222c9), restated: no module of src.stats is
    imported here."""
    return S.sha256_bytes("\n".join(sorted(ids)).encode("utf-8"))


def name_digests(names) -> str:
    """Offending names by count and the sha256 of each one's bytes, never the name (P21's form,
    scripts/hash_split_files.py :73-88)."""
    ds = sorted(S.sha256_bytes(n.encode("utf-8", "surrogateescape"))[:16] for n in names)
    return f"{len(ds)} name(s), sha256 {', '.join(ds[:4])}{' ...' if len(ds) > 4 else ''}"


# ------------------------------------------------------------------------------------------- the listed ids
def image_id_errors(pairs) -> list[str]:
    """The ids of the listed images, [(stem, file name)]: what the outputs and LREP's reader cannot carry, and what the
    statistics ingest refuses (ruling H9 step 3; the STATS contract's §10, "no case-folded ID collision"). Refused: a
    stem with surrounding whitespace (LREP's id); a tab, CR or LF in a name (a hash-list row is one tab-separated
    line); a name that is not UTF-8 (the outputs are); two images with one stem; two stems equal once case-folded
    (str.casefold, as src/stats/ingest.py :327 and src/eval/evaluate.py :301)."""
    errs = []
    spaced = [n for s, n in pairs if s != s.strip()]
    if spaced:
        errs.append(f"stems with surrounding whitespace: {name_digests(spaced)}")
    unsafe = [n for _, n in pairs if any(c in n for c in TSV_UNSAFE)]
    if unsafe:
        errs.append(f"names holding a tab, CR or LF: {name_digests(unsafe)}")
    raw = [n for _, n in pairs if not _utf8(n)]
    if raw:
        errs.append(f"names that are not UTF-8: {name_digests(raw)}")
    seen: dict[str, int] = {}
    for s, _ in pairs:
        seen[s] = seen.get(s, 0) + 1
    shared = [s for s, k in seen.items() if k > 1]
    if shared:
        errs.append(f"two images share a stem: {name_digests(shared)}")
    folded: dict[str, list] = {}
    for s in seen:
        folded.setdefault(s.casefold(), []).append(s)
    clash = [s for group in folded.values() if len(group) > 1 for s in group]
    if clash:
        errs.append(f"stems equal once case-folded: {name_digests(clash)}")
    return errs


# ---------------------------------------------------------------------------------------- the TRAIN/VAL list
def parse_train_val_list(data: bytes, *, sha256: str, rows: int, train: int, val: int) -> dict:
    """The TRAIN/VAL hash list (DL-91; scripts/hash_split_files.py :232-233), strictly. Its sha256 is `sha256` before
    anything is parsed (train_val_list_mismatch). Then (train_val_list_invalid): UTF-8 ending in LF; the header that
    script writes; `rows` data rows after it, each six tab-separated fields (split train or val, stem, file, a byte
    count, a 64-hex sha256, a 40-hex sha1); `train` rows of split train and `val` of split val. The pin gives the four
    values; a fixture passes its own. Returns {"train": frozenset of sha256, "val": frozenset of sha256}."""
    got = S.sha256_bytes(data)
    if got != sha256:
        raise S.TmanRefused("train_val_list_mismatch", f"the TRAIN/VAL list's sha256 is {got}, not the pin's "
                                                       f"{sha256}: it is not parsed")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise S.TmanRefused("train_val_list_invalid", "the TRAIN/VAL list is not UTF-8") from None
    if not text.endswith("\n"):
        raise S.TmanRefused("train_val_list_invalid", "the TRAIN/VAL list does not end with LF")
    lines = text[:-1].split("\n")
    if lines[0] != "\t".join(TRAINVAL_HEADER):
        raise S.TmanRefused("train_val_list_invalid", f"its header is not scripts/hash_split_files.py's "
                                                      f"{list(TRAINVAL_HEADER)}")
    body = lines[1:]
    if len(body) != rows:
        raise S.TmanRefused("train_val_list_invalid", f"{len(body)} data rows after the header; the pin's rows are "
                                                      f"{rows}")
    sets = {k: set() for k in TRAINVAL_SPLITS}
    counts = dict.fromkeys(TRAINVAL_SPLITS, 0)
    for n, line in enumerate(body, start=2):
        f = line.split("\t")
        if len(f) != len(TRAINVAL_HEADER) or f[0] not in sets or not DIGITS_RE.fullmatch(f[3]) \
                or not S.SHA256_RE.fullmatch(f[4]) or not SHA1_RE.fullmatch(f[5]):
            raise S.TmanRefused("train_val_list_invalid", f"line {n} is not a {list(TRAINVAL_HEADER)} row")
        sets[f[0]].add(f[4])
        counts[f[0]] += 1
    if counts != {"train": train, "val": val}:
        raise S.TmanRefused("train_val_list_invalid", f"split counts {counts}; the pin's are train {train} and val "
                                                      f"{val}")
    return {k: frozenset(v) for k, v in sets.items()}


# ------------------------------------------------------------------------------------------ the rule and doc
def classify(images: dict, train_val: dict) -> tuple[dict, list, int]:
    """The rule (ruling H3) over {id: sha256}: ({reason: sorted removed ids}, the sorted kept ids, the number of images
    equal to both a TRAIN and a VAL image). A group of identical images that equals a TRAIN or VAL image is removed
    whole under that reason; any other group keeps its first sorted id."""
    removed = {r: [] for r in REASONS}
    both = 0
    groups: dict[str, list] = {}
    for stem in sorted(images):
        h = images[stem]
        if h in train_val["train"]:
            removed["equal_train"].append(stem)
            both += h in train_val["val"]
        elif h in train_val["val"]:
            removed["equal_val"].append(stem)
        else:
            groups.setdefault(h, []).append(stem)
    kept = []
    for members in groups.values():
        first, *rest = sorted(members)
        kept.append(first)
        removed["duplicate_in_split"] += rest
    return {r: sorted(ids) for r, ids in removed.items()}, sorted(kept), both


def build_doc(images: dict, train_val: dict, *, hash_list_sha256: str, train_val_list_sha256: str,
              script: dict) -> dict:
    """tday_dedup/1 (ruling H5) from {id: sha256} of the split's images and the TRAIN/VAL list's sha256 sets
    (parse_train_val_list), checked by doc_errors before it is returned (dedup_invalid: for example, no image
    kept)."""
    removed, kept, both = classify(images, train_val)
    doc = {"format": DEDUP_FORMAT, "n_images": len(images), "image_ids_sha256": ids_sha256(images),
           "kept_ids_sha256": ids_sha256(kept), "kept_ids": kept, "removed_ids_by_reason": removed,
           "removed_by_reason": {r: len(removed[r]) for r in REASONS}, "reasons_partition": True,
           "also_counted": {ALSO_COUNTED: both}, "hash_list_sha256": hash_list_sha256,
           "train_val_list_sha256": train_val_list_sha256, "script": dict(script)}
    errs = doc_errors(doc)
    if errs:
        raise S.TmanRefused("dedup_invalid", "; ".join(errs[:4]))
    return doc


def _list_errors(ids, where: str) -> list[str]:
    if not isinstance(ids, list) or not all(_is_id(i) for i in ids):
        return [f"{where}: not a list of ids (non-empty strings without surrounding whitespace)"]
    if ids != sorted(set(ids)):
        return [f"{where}: not sorted and unique"]
    return []


def doc_errors(doc) -> list[str]:
    """Every way `doc` breaks tday_dedup/1 (ruling H5): its exact keys, its format and types, then the invariants --
    the counts are the removed lists' lengths; their sum is n_images minus the kept ids; the kept and removed id lists
    are disjoint and their union is the image set (its count and image_ids_sha256); kept_ids is not empty and its
    fingerprint is kept_ids_sha256; reasons_partition is true; equal_train_and_val is at most equal_train."""
    if not isinstance(doc, dict) or sorted(doc) != list(DOC_KEYS):
        return [f"not an object with exactly the keys {list(DOC_KEYS)}"]
    errs = [] if doc["format"] == DEDUP_FORMAT else [f"format {doc['format']!r} is not {DEDUP_FORMAT!r}"]
    if not (_is_int(doc["n_images"]) and doc["n_images"] >= 1):
        errs.append("n_images is not a positive integer")
    for k in ("image_ids_sha256", "kept_ids_sha256", "hash_list_sha256", "train_val_list_sha256"):
        errs += S.value_errors(doc[k], "sha256", k)
    errs += S.field_errors(doc["script"], {"path": "str", "sha256": "sha256"}, "script")
    errs += _list_errors(doc["kept_ids"], "kept_ids")
    rid, counts, ac = doc["removed_ids_by_reason"], doc["removed_by_reason"], doc["also_counted"]
    if not (isinstance(rid, dict) and sorted(rid) == sorted(REASONS)):
        errs.append(f"removed_ids_by_reason does not hold exactly {list(REASONS)}")
    else:
        for r in REASONS:
            errs += _list_errors(rid[r], f"removed_ids_by_reason.{r}")
    if not (isinstance(counts, dict) and sorted(counts) == sorted(REASONS)
            and all(_is_int(counts[r]) and counts[r] >= 0 for r in REASONS)):
        errs.append(f"removed_by_reason is not a count >= 0 for each of exactly {list(REASONS)}")
    if not (isinstance(ac, dict) and list(ac) == [ALSO_COUNTED] and _is_int(ac[ALSO_COUNTED])
            and ac[ALSO_COUNTED] >= 0):
        errs.append(f"also_counted is not a count >= 0 of {ALSO_COUNTED} alone")
    if doc["reasons_partition"] is not True:
        errs.append("reasons_partition is not true: each removed image has exactly one reason (ruling H3)")
    if errs:
        return errs
    kept = doc["kept_ids"]
    if not kept:
        errs.append("kept_ids is empty")
    if any(counts[r] != len(rid[r]) for r in REASONS):
        errs.append("removed_by_reason is not the length of each removed list")
    if sum(counts.values()) != doc["n_images"] - len(kept):
        errs.append("the sum of removed_by_reason is not n_images minus the kept ids")
    every = kept + [i for r in REASONS for i in rid[r]]
    if len(set(every)) != len(every):
        errs.append("the kept and removed id lists are not disjoint")
    if len(every) != doc["n_images"] or ids_sha256(every) != doc["image_ids_sha256"]:
        errs.append("the kept and removed ids are not the image set (its count and image_ids_sha256)")
    if ids_sha256(kept) != doc["kept_ids_sha256"]:
        errs.append("kept_ids_sha256 is not the kept ids' fingerprint")
    if ac[ALSO_COUNTED] > counts["equal_train"]:
        errs.append(f"{ALSO_COUNTED} exceeds equal_train, under which those images are removed")
    return errs


def doc_bytes(doc: dict) -> bytes:
    """tday_dedup.json's bytes: schema's canonical JSON (src/eval/artifacts.py :93), one definition (TB14)."""
    return S.canonical_json_bytes(doc)


def read_dedup(data: bytes, *, expect_sha256: str | None = None) -> dict:
    """tday_dedup.json, strictly (dedup_invalid): its sha256 is `expect_sha256`, when given, before it is parsed;
    strict UTF-8 JSON (schema.strict_json: no duplicate key, no NaN or infinity); its bytes are the canonical JSON of
    what they hold; every rule of doc_errors."""
    if expect_sha256 is not None and S.sha256_bytes(data) != expect_sha256:
        raise S.TmanRefused("dedup_invalid", f"{DEDUP_NAME}: sha256 {S.sha256_bytes(data)}, not {expect_sha256}: it "
                                             "is not parsed")
    doc = S.strict_json(data, DEDUP_NAME, code="dedup_invalid")
    if S.canonical_json_bytes(doc) != data:
        raise S.TmanRefused("dedup_invalid", f"{DEDUP_NAME}: its bytes are not the canonical JSON of what they hold")
    errs = doc_errors(doc)
    if errs:
        raise S.TmanRefused("dedup_invalid", f"{DEDUP_NAME}: " + "; ".join(errs[:4]))
    return doc


def hash_list_bytes(rows) -> bytes:
    """tday_image_hashes.tsv (ruling H6): the header, then one row per image ({stem, file, bytes, sha256}) sorted by
    stem; tab-separated, LF, UTF-8."""
    lines = ["\t".join(HASH_LIST_HEADER)]
    lines += [f"{r['stem']}\t{r['file']}\t{r['bytes']}\t{r['sha256']}" for r in sorted(rows, key=lambda r: r["stem"])]
    return ("\n".join(lines) + "\n").encode("utf-8")


def hasher_values(hash_list: bytes, dedup: bytes) -> dict:
    """The session's three values of manifest.HASHER_VALUES (ruling H7): the sha256 of the hash list's bytes and of
    tday_dedup.json's, and its counts by reason (removed_by_reason). The dedup file is read strictly and must name
    this hash list (dedup_invalid otherwise)."""
    doc = read_dedup(dedup)
    h = S.sha256_bytes(hash_list)
    if doc["hash_list_sha256"] != h:
        raise S.TmanRefused("dedup_invalid", f"{DEDUP_NAME} names the hash list {doc['hash_list_sha256']}, not {h}")
    return dict(zip(VALUE_KEYS, (h, S.sha256_bytes(dedup), dict(doc["removed_by_reason"]))))


def issue_counts_line() -> str:
    """Issue #11's counts beside the reasons (AM-18 item 8(a) and the last point of (c)): the issue's own labels and
    counts, not recomputed (the issue counts SHA-1 groups over the whole release)."""
    parts = [f"{r} {ISSUE_ROWS[r][1]} ({ISSUE_ROWS[r][0]})" for r in REASONS]
    return "issue #11's counts beside them (AM-18 item 8(a), not recomputed): " + "; ".join(parts)
