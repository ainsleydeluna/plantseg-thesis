#!/usr/bin/env python3
"""The in-session image hasher of the split of record (lane TMAN, B-3a part 1; AM-18 item 8(c); plan §4(iv) and
§1.7; rulings H8 and H9). It is not scripts/hash_split_files.py (plan :501, :532) and not a command of
scripts/tday_manifest.py.

  --print-pin --commit <40 hex>   prints the image_hasher pin, dedup.hasher_pin's (its one definition, P2R2), and
                                  reads no data: script {path, commit, sha256 of its bytes}, python, parameters (with
                                  the sha256 of src/tman/dedup.py). The manifest's builder embeds the same pin before
                                  the freeze (manifest.py :879).
  hash                            the session's run, after the unlock record, in Ice's block outside Claude Code
                                  (TB19 Q5). Each step passes before the next starts:
    1. paths: --repo, --root, --train-val-list and --out go through B-1's path guard (schema.py :248) as given and
       then resolved, and --unlock-entry as given and then resolved under --repo: a refused string gets no call.
    2. records, at HEAD of --repo through B-1's readers, in a clone with its full history (a shallow clone, where
       B-1's written-once check cannot see a rewrite, src/training/sweep_select.py :2400-2420, and a repository with
       no commit are refused): the unlock entry at --unlock-entry (a repository path; nothing is listed) is
       committed, written once, of kind unlock, its sha256 in a RECORDED or IMPLEMENTED row
       (entries.read_entry_file, :753); the manifest it names is the committed one, by its file, body and appends
       sha256; that manifest's image_hasher pin is this script's and this dedup module's (the sha256 of their bytes),
       with this interpreter's python major.minor.
    3. listing: only now is the split resolved (src.eval.stage_artifacts.OFFICIAL_SPLIT, imported in
       split_of_record). One os.scandir of <root>/images/<split>; its files whose suffix, lower-cased, is .jpg or
       .jpeg (scripts/hash_split_files.py :96 and :104); a stem is the name without it. The ids are refused when the
       outputs cannot carry them, when two images share a stem and when two stems are equal once case-folded; their
       number is the pin's expected_images.
    4. the TRAIN/VAL list (DL-91): its sha256 is the pin's before it is parsed; its header, rows and TRAIN and VAL
       counts are the pin's (dedup.parse_train_val_list).
    5. hashing: one binary read per image, SHA-256 only. No image library is imported or called, and no annotation
       is touched.
    6. outputs: tday_image_hashes.tsv and tday_dedup.json are built, then created exclusively in --out, an existing
       folder outside the repository at --repo and outside this checkout, once resolved; both or neither (a failure
       removes what the run created). Printed: the three values of manifest.HASHER_VALUES (:544) and, beside them,
       issue #11's counts; never an id. A character stdout cannot encode is printed escaped.

Exit codes (scripts/tday_manifest.py's): 0 printed or written, 2 refused (REFUSED [<code>] -- <detail>), 4 error.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.tman import dedup as D  # noqa: E402
from src.tman import entries as E  # noqa: E402
from src.tman import manifest as M  # noqa: E402
from src.tman import records as R  # noqa: E402
from src.tman import schema as S  # noqa: E402
from src.training.sweep_select import SelectionRefused  # noqa: E402

READ_BLOCK = 1 << 20


# ------------------------------------------------------------------------------------------------- the pin
def running_python() -> str:
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


def pin_errors(pin) -> list[str]:
    """Step 2: the manifest's image_hasher pin is this code's -- dedup.hasher_pin's form (its one definition, P2R2);
    script.path this script's and script.sha256 its bytes'; modules exactly the dedup module's sha256; python this
    interpreter's major.minor; the parameters this code fixes, its own."""
    errs = S.field_errors(pin, D.PIN_SPEC, "image_hasher.pin")
    if errs:
        return errs
    script, module = D.code_sha256()
    if pin["script"]["path"] != D.SCRIPT_REL or pin["script"]["sha256"] != script:
        errs.append(f"the pinned script is {pin['script']['path']} with sha256 {pin['script']['sha256']}; this one is "
                    f"{D.SCRIPT_REL} with {script}")
    if pin["parameters"]["modules"] != {D.DEDUP_REL: module}:
        errs.append(f"the pinned modules are not {D.DEDUP_REL} with sha256 {module}")
    if pin["python"] != running_python():
        errs.append(f"the pinned python is {pin['python']}; this one is {running_python()}")
    own = D.descriptive_parameters()
    if {k: pin["parameters"][k] for k in D.DESCRIPTIVE} != {k: own[k] for k in D.DESCRIPTIVE}:
        errs.append(f"the pinned parameters {list(D.DESCRIPTIVE)} are not this code's")
    return errs


# ------------------------------------------------------------------------------------------ steps 1 and 2
def check_paths(args) -> None:
    """Step 1: --repo, --root, --train-val-list and --out through B-1's guard (schema.py :248), as given and then
    resolved; --unlock-entry, a repository path, resolved under --repo, where it is read. (B-1's reader resolves it
    against the working folder first, entries.py :757, and under the records' root only after, records.py :212.)"""
    for p in (args.repo, args.root, args.train_val_list, args.out):
        S.refuse_test_path(p)
    S.refuse_test_path(args.unlock_entry, args.repo)


def _at(doc, *keys):
    for k in keys:
        doc = doc.get(k) if isinstance(doc, dict) else None
    return doc


def check_records(args) -> dict:
    """Step 2, at HEAD of --repo, a clone with its full history (in a shallow one, B-1's written-once check cannot see
    a rewrite): the unlock entry (B-1's reader), the manifest it names, and the pin; returns the pin."""
    records = R.Records.at_head(args.repo)
    try:
        records.src.require_not_shallow()
    except SelectionRefused as e:
        if e.code == "repository_shallow":
            raise S.TmanRefused("repository_shallow", f"--repo {records.src.root} is a shallow clone: whether the "
                                                      "unlock entry was written once needs its full history (git "
                                                      "fetch --unshallow)") from None
        raise S.TmanRefused("record_uncommitted", f"--repo {records.src.root} is no git repository with a HEAD "
                                                  "commit: the records are read at HEAD") from None
    doc, _ = E.read_entry_file(records, args.unlock_entry, "unlock")
    m = doc["manifest"]
    if m["path"] != S.MANIFEST_REL:
        raise S.TmanRefused("unlock_manifest_mismatch", f"the unlock entry names {m['path']}, not the manifest "
                                                        f"{S.MANIFEST_REL}")
    data = records.read(m["path"])
    if S.sha256_bytes(data) != m["sha256"]:
        raise S.TmanRefused("unlock_manifest_mismatch", f"{m['path']} has sha256 {S.sha256_bytes(data)}; the unlock "
                                                        f"entry records {m['sha256']}")
    man = S.strict_json(data, m["path"], code="manifest_unreadable")
    if not (isinstance(man, dict) and man.get("format") == S.MANIFEST_FORMAT and isinstance(man.get("body"), dict)
            and "appends" in man):
        raise S.TmanRefused("manifest_format", f"{m['path']} is not a {S.MANIFEST_FORMAT} with body and appends")
    if S.canonical_sha256(man["body"]) != m["body_sha256"]:
        raise S.TmanRefused("unlock_manifest_mismatch", f"{m['path']}'s body is not the body_sha256 the unlock entry "
                                                        "records")
    if S.canonical_sha256(man["appends"]) != m["appends_sha256"]:
        raise S.TmanRefused("unlock_manifest_mismatch", f"{m['path']}'s appends are not the appends_sha256 the unlock "
                                                        "entry records")
    pin = _at(man, "body", "generator_pins", "image_hasher", "pin")
    errs = pin_errors(pin)
    if errs:
        raise S.TmanRefused("hasher_pin_mismatch", "; ".join(errs[:4]))
    return pin


# ------------------------------------------------------------------------------------------- steps 3 to 5
def split_of_record() -> str:
    """The split of record, resolved in step 3 by import (ruling H0: never written as a literal)."""
    from src.eval.stage_artifacts import OFFICIAL_SPLIT
    return OFFICIAL_SPLIT


def image_folder(root, split: str) -> str:
    return os.path.join(root, *D.FOLDER.format(split=split).split("/"))


def list_images(folder: str, expected: int) -> list:
    """Step 3: [(stem, name)] sorted, from one os.scandir of the image folder."""
    try:
        with os.scandir(folder) as it:
            names = [e.name for e in it if e.is_file()]
    except OSError as e:
        raise S.TmanRefused("images_unreadable", f"the image folder cannot be listed ({type(e).__name__})") from None
    pairs = sorted((os.path.splitext(n)[0], n) for n in names if os.path.splitext(n)[1].lower() in D.SUFFIXES)
    errs = D.image_id_errors(pairs)
    if errs:
        raise S.TmanRefused("image_ids_invalid", "; ".join(errs))
    if len(pairs) != expected:
        raise S.TmanRefused("image_count_mismatch", f"{len(pairs)} images; the pin's expected_images is {expected}")
    return pairs


def read_train_val(path, tv: dict) -> tuple[dict, str]:
    """Step 4: the TRAIN/VAL list, one binary read; ({"train", "val": sha256 sets}, its sha256)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        raise S.TmanRefused("train_val_list_mismatch", f"the TRAIN/VAL list cannot be read ({type(e).__name__})") \
            from None
    sets = D.parse_train_val_list(data, sha256=tv["sha256"], rows=tv["rows"], train=tv["train"], val=tv["val"])
    return sets, S.sha256_bytes(data)


def hash_images(folder: str, pairs) -> list:
    """Step 5: one binary read per image, SHA-256 of its bytes and their count."""
    rows = []
    for stem, name in pairs:
        h, n = hashlib.sha256(), 0
        try:
            with open(os.path.join(folder, name), "rb") as fh:
                for block in iter(lambda: fh.read(READ_BLOCK), b""):
                    h.update(block)
                    n += len(block)
        except OSError as e:
            raise S.TmanRefused("images_unreadable", f"an image cannot be read ({type(e).__name__}; "
                                                     f"{D.name_digests([name])})") from None
        rows.append({"stem": stem, "file": name, "bytes": n, "sha256": h.hexdigest()})
    return rows


# ------------------------------------------------------------------------------------------------ step 6
def out_folder(path, repo) -> Path:
    """--out: an existing folder outside the repository at --repo and outside this checkout, once resolved."""
    p = Path(path).resolve()
    for r in (REPO.resolve(), Path(repo).resolve()):
        if p == r or r in p.parents:
            raise S.TmanRefused("output_location", f"--out {p} is inside the repository {r}: outputs are written "
                                                   "outside it")
    if not p.is_dir():
        raise S.TmanRefused("output_location", f"--out {p} is not an existing folder")
    return p


def write_outputs(folder: Path, files: list) -> None:
    """Both outputs created exclusively: none may exist yet, and each is opened with "xb". A failure removes every
    file this call created, so both outputs exist or neither (src/eval/teacher_diag.py :892-909's form)."""
    targets = [folder / name for name, _ in files]
    if any(os.path.lexists(t) for t in targets):
        raise S.TmanRefused("output_exists", f"--out already holds {D.HASH_LIST_NAME} or {D.DEDUP_NAME}: outputs are "
                                             "created once")
    made = []
    try:
        for t, (_, data) in zip(targets, files):
            try:
                fh = open(t, "xb")
            except FileExistsError:
                raise S.TmanRefused("output_exists", f"{t.name} exists: outputs are created once") from None
            with fh:
                made.append(t)
                fh.write(data)
    except BaseException:
        for t in made:
            t.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------------------------------------ commands
def cmd_print_pin(args) -> int:
    pin = D.hasher_pin(args.commit)
    errs = M.hasher_pin_errors(pin)
    if errs:
        raise S.TmanRefused("generator_pin_incomplete", "; ".join(errs))
    sys.stdout.write(S.json_file_bytes(pin).decode("utf-8"))
    return S.EXIT_PASS


def cmd_hash(args) -> int:
    check_paths(args)
    params = check_records(args)["parameters"]
    folder = image_folder(args.root, split_of_record())
    pairs = list_images(folder, params["expected_images"])
    train_val, tv_sha = read_train_val(args.train_val_list, params["train_val_list"])
    rows = hash_images(folder, pairs)
    tsv = D.hash_list_bytes(rows)
    doc = D.build_doc({r["stem"]: r["sha256"] for r in rows}, train_val, hash_list_sha256=S.sha256_bytes(tsv),
                      train_val_list_sha256=tv_sha, script={"path": D.SCRIPT_REL, "sha256": D.code_sha256()[0]})
    data = D.doc_bytes(doc)
    write_outputs(out_folder(args.out, args.repo), [(D.HASH_LIST_NAME, tsv), (D.DEDUP_NAME, data)])
    values = D.hasher_values(tsv, data)
    for k in D.VALUE_KEYS:
        v = values[k]
        print(f"{k} {v if isinstance(v, str) else S.canonical_json_bytes(v).decode('utf-8')}")
    print(D.issue_counts_line())
    return S.EXIT_PASS


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="The in-session image hasher of the split of record (B-3a): --print-pin, "
                                            "or the hash command after the unlock record.")
    p.add_argument("--print-pin", action="store_true", help="print the image_hasher pin; reads no data")
    p.add_argument("--commit", help="the pin's commit (40 lower-case hex), with --print-pin")
    sub = p.add_subparsers(dest="cmd")
    h = sub.add_parser("hash", help="hash the split's images and write the two outputs (the session only)")
    for flag, what in (("--repo", "the session checkout whose records are read at HEAD"),
                       ("--unlock-entry", "the committed unlock entry file, a repository path"),
                       ("--root", "the data root holding images/<split>"),
                       ("--train-val-list", "DL-91's TRAIN/VAL hash list"),
                       ("--out", "an existing folder outside the repository")):
        h.add_argument(flag, required=True, help=what)
    return p


def main(argv=None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)      # a text stream's (a StringIO has none)
    if reconfigure is not None:
        # a stdout whose encoding lacks the issue labels' U+2194 (a Windows code page, redirected) gets it escaped:
        # printing never turns a run that wrote its outputs into exit 4
        reconfigure(errors="backslashreplace")
    p = parser()
    args = p.parse_args(argv)
    if args.print_pin == (args.cmd == "hash"):
        p.error("give --print-pin --commit <40 hex>, or the hash command")
    if args.print_pin != (args.commit is not None):
        p.error("--commit goes with --print-pin, and only with it")
    try:
        return cmd_print_pin(args) if args.print_pin else cmd_hash(args)
    except S.TmanRefused as e:
        print(e.line())
        return S.EXIT_REFUSED
    except Exception as e:  # noqa: BLE001 -- an unexpected error is exit 4, never a pass
        print(f"ERROR -- {type(e).__name__}: {e}")
        return S.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
