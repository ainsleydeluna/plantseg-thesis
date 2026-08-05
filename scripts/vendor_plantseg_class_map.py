#!/usr/bin/env python3
"""Vendor the AUTHORITATIVE PlantSeg class map into configs/plantseg_class_map.json.

MAINTENANCE UTILITY ONLY. This script is never imported by evaluator runtime code, never runs
automatically during evaluation, performs NO network access on import, and modifies the class
map only when invoked explicitly. Ordinary offline evaluation does not need it.

Authoritative source (nothing else is acceptable):
    repo   https://github.com/tqwei05/PlantSeg
    path   mmseg/datasets/plantseg115.py
    object PlantSeg115Dataset.METAINFO["classes"]

Explicitly rejected as sources: Metadata.csv["Index"], majority vote over COCO annotations,
annotation/mask frequency, alphabetical ordering, a manually retyped list, the thesis text, and
any unpinned branch URL.

Two modes:

  Network creation (requires the explicit flag -- no flag, no network):
      python scripts/vendor_plantseg_class_map.py --resolve-main

  Offline reproduction (no network at all):
      python scripts/vendor_plantseg_class_map.py \
          --from-file <upstream plantseg115.py> --commit-sha <40-hex> \
          [--retrieved-utc <ISO-8601 Z>]

Add --dry-run to either mode to print the canonical payload to stdout and write nothing.

WRITE SAFETY (there is no arbitrary output option, by design):
  * the only repository path this tool can ever write is configs/plantseg_class_map.json;
  * the destination is re-validated immediately before every write -- the parent must be the
    repository's own `configs` directory, neither the destination nor its parent may be a
    symlink, and a resolved path that escapes the repository is refused;
  * content is written to a temporary sibling INSIDE that same directory, re-read and
    re-validated, and only then moved into place with an atomic same-directory replacement;
  * a failure before that replacement leaves any existing class map byte-identical and removes
    the temporary sibling. A failure DURING the replacement is the only uncovered case, and
    `Path.replace` is atomic on both POSIX and Windows.

Extraction is static via the standard-library `ast` module. The upstream file is never
imported or executed, and never stored inside the repository.

Standard library only. No installs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OWNER = "tqwei05"
NAME = "PlantSeg"
REPO_URL = f"https://github.com/{OWNER}/{NAME}"
SOURCE_PATH = "mmseg/datasets/plantseg115.py"
SOURCE_OBJECT = 'PlantSeg115Dataset.METAINFO["classes"]'
DATASET_CLASS = "PlantSeg115Dataset"

SCHEMA = "plantseg-class-map/1.0.0"
EXTRACTION_METHOD = "python-ast-static/1.0.0"

NUM_CLASSES = 116
BACKGROUND_INDEX = 0
IGNORE_INDEX = 255
ROLE_BG = "background_or_non_disease"
ROLE_DISEASE = "disease"

USER_AGENT = "plantseg-thesis-vendor-class-map/1.0 (A2b-0 one-time provenance pin)"
TIMEOUT = 30

# The ONE repository path this tool may write. There is deliberately no CLI override: an
# arbitrary destination could target an unrelated file, follow a symlink out of the tree, or
# overwrite a protected document.
CONFIG_DIRNAME = "configs"
CLASS_MAP_FILENAME = "plantseg_class_map.json"


class VendorError(RuntimeError):
    """Any detected failure aborts before the atomic replacement, so an existing class-map
    artifact is left byte-identical and no partial artifact is published."""


# --------------------------------------------------------------------------------------------------
# hashing helpers (three DISTINCT concepts -- never conflate them)
# --------------------------------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_entries_bytes(entries) -> bytes:
    """Canonical bytes of the ORDERED entries list only.

    UTF-8, sorted object keys, compact separators, non-finite forbidden, list order preserved.
    Must match `src.eval.artifacts.canonical_json_bytes(list(class_map))`.
    """
    return json.dumps(list(entries), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def semantic_class_map_sha256(entries) -> str:
    return sha256_bytes(canonical_entries_bytes(entries))


def immutable_url(sha: str) -> str:
    return f"https://raw.githubusercontent.com/{OWNER}/{NAME}/{sha}/{SOURCE_PATH}"


# --------------------------------------------------------------------------------------------------
# network (only reachable via --resolve-main)
# --------------------------------------------------------------------------------------------------
def _get(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:   # noqa: S310 -- pinned https
            status = getattr(resp, "status", resp.getcode())
            if status != 200:
                raise VendorError(f"HTTP {status} for {url}")
            return resp.read()
    except VendorError:
        raise
    except Exception as e:                                           # noqa: BLE001
        raise VendorError(f"request failed for {url}: {type(e).__name__}: {e}") from e


def resolve_main_sha() -> tuple[str, str]:
    """Resolve the official `main` branch to a full 40-hex commit SHA."""
    url = f"https://api.github.com/repos/{OWNER}/{NAME}/commits/main"
    data = json.loads(_get(url, accept="application/vnd.github+json").decode("utf-8"))
    sha = str(data.get("sha", ""))
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise VendorError(f"resolved SHA is not full 40-char lowercase hex: {sha!r}")
    date = str(data.get("commit", {}).get("committer", {}).get("date", ""))
    return sha, date


def fetch_pinned_source(sha: str) -> bytes:
    return _get(immutable_url(sha))


# --------------------------------------------------------------------------------------------------
# safe static extraction (no eval / exec / import / regex)
# --------------------------------------------------------------------------------------------------
def extract_classes(source_text: str) -> list[str]:
    """Statically extract PlantSeg115Dataset.METAINFO['classes'] as literal strings."""
    tree = ast.parse(source_text)

    classdefs = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == DATASET_CLASS]
    if len(classdefs) != 1:
        raise VendorError(f"expected exactly one `class {DATASET_CLASS}`, found {len(classdefs)}")

    metainfo = [n for n in classdefs[0].body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "METAINFO" for t in n.targets)]
    if len(metainfo) != 1:
        raise VendorError(f"expected exactly one METAINFO assignment, found {len(metainfo)}")

    value = metainfo[0].value
    candidates = []
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict":
        candidates = [kw.value for kw in value.keywords if kw.arg == "classes"]
    elif isinstance(value, ast.Dict):
        candidates = [v for k, v in zip(value.keys, value.values)
                      if isinstance(k, ast.Constant) and k.value == "classes"]
    else:
        raise VendorError(
            f"unsupported METAINFO structure {type(value).__name__}; expected dict(...) or a "
            "dict literal -- refusing to guess")
    if len(candidates) != 1:
        raise VendorError(f"expected exactly one `classes` entry, found {len(candidates)}")

    node = candidates[0]
    if not isinstance(node, ast.Tuple):
        raise VendorError(f"`classes` must be a literal tuple, got {type(node).__name__}")

    names: list[str] = []
    for i, el in enumerate(node.elts):
        if not isinstance(el, ast.Constant) or not isinstance(el.value, str):
            raise VendorError(
                f"`classes[{i}]` is not a literal string ({type(el).__name__}); dynamic "
                "expressions are rejected")
        names.append(el.value)

    if len(names) != NUM_CLASSES:
        raise VendorError(f"expected exactly {NUM_CLASSES} class entries, got {len(names)}")
    return names


def build_entries(names) -> list[dict]:
    """Ordered descriptor. NOTE: class 0's official name is the EMPTY STRING -- every check
    below uses explicit identity/length tests, never truthiness."""
    if names[BACKGROUND_INDEX] != "":
        raise VendorError(
            f"class {BACKGROUND_INDEX} official name must be the empty string, got "
            f"{names[BACKGROUND_INDEX]!r}")
    entries = []
    for cid, name in enumerate(names):
        role = ROLE_BG if cid == BACKGROUND_INDEX else ROLE_DISEASE
        if cid != BACKGROUND_INDEX and len(name) == 0:
            raise VendorError(f"disease class {cid} has an empty official name")
        entries.append({"class_id": cid, "name": name, "role": role})
    return entries


def build_payload(entries, *, sha, raw_sha, retrieved_utc, upstream_commit_date=None) -> dict:
    return {
        "schema": SCHEMA,
        "source": {
            "repository_owner": OWNER,
            "repository_name": NAME,
            "repository_url": REPO_URL,
            "source_path": SOURCE_PATH,
            "source_commit_sha": sha,
            "immutable_source_url": immutable_url(sha),
            "source_object": SOURCE_OBJECT,
            "raw_source_sha256": raw_sha,
            "retrieved_utc": retrieved_utc,
            "extraction_method": EXTRACTION_METHOD,
            "upstream_commit_date": upstream_commit_date,
            "note": ("Ordered names come verbatim from "
                     f"{SOURCE_OBJECT} at the pinned commit. Metadata.csv['Index'], majority "
                     "vote and annotation frequency are NOT sources."),
        },
        "class_space": {
            "num_classes": NUM_CLASSES,
            "background_index": BACKGROUND_INDEX,
            "ignore_index": IGNORE_INDEX,
        },
        "entries": entries,
    }


# --------------------------------------------------------------------------------------------------
# destination containment + atomic replacement
# --------------------------------------------------------------------------------------------------
def validated_out_path() -> Path:
    """The single writable destination, re-validated at call time. Raises on anything unsafe.

    Derived from REPO on every call (never a cached module constant), so the containment rules
    are enforced against the tree the tool is actually running in.
    """
    repo_root = Path(REPO).resolve()
    cfg_dir = repo_root / CONFIG_DIRNAME
    if cfg_dir.is_symlink():
        raise VendorError(f"refusing to write: {CONFIG_DIRNAME}/ is a symlink ({cfg_dir})")
    if not cfg_dir.is_dir():
        raise VendorError(f"refusing to write: {CONFIG_DIRNAME}/ is not a directory ({cfg_dir})")
    cfg_real = cfg_dir.resolve()
    if cfg_real.parent != repo_root or cfg_real.name != CONFIG_DIRNAME:
        raise VendorError(
            f"refusing to write: {CONFIG_DIRNAME}/ resolves outside the repository -> {cfg_real}")

    dest = cfg_dir / CLASS_MAP_FILENAME
    if dest.is_symlink():
        raise VendorError(f"refusing to write: destination is a symlink ({dest})")
    dest_real = cfg_real / CLASS_MAP_FILENAME
    if dest_real.parent != cfg_real or dest_real.name != CLASS_MAP_FILENAME:
        raise VendorError(f"refusing to write: destination escapes {CONFIG_DIRNAME}/ -> {dest_real}")
    if not dest_real.is_relative_to(repo_root):
        raise VendorError(f"refusing to write: destination escapes the repository -> {dest_real}")
    return dest_real


def atomic_write_json(dest: Path, payload: dict, expected_semantic: str) -> str:
    """Write `payload` to `dest` via a validated temporary sibling + atomic replacement.

    Mirrors the finalisation discipline in src/eval/artifacts.py: write to a sibling in the
    SAME directory, re-read and re-validate it, then replace. Any failure before the replace
    removes the sibling and leaves `dest` untouched.
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp_out = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
        reread = tmp_out.read_bytes()
        if reread.decode("utf-8") != text:
            raise VendorError("temporary artifact did not round-trip byte-for-byte")
        doc = json.loads(reread.decode("utf-8"))
        if doc.get("schema") != SCHEMA:
            raise VendorError(f"temporary artifact schema mismatch: {doc.get('schema')!r}")
        if len(doc.get("entries", [])) != NUM_CLASSES:
            raise VendorError(f"temporary artifact has {len(doc.get('entries', []))} entries")
        if semantic_class_map_sha256(doc["entries"]) != expected_semantic:
            raise VendorError("temporary artifact semantic hash does not match the built payload")
        tmp_out.replace(dest)                    # atomic on POSIX and Windows
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
    return sha256_bytes(dest.read_bytes())


# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolve-main", action="store_true",
                   help="NETWORK MODE: resolve official main to a SHA and fetch the pinned file")
    p.add_argument("--from-file", type=Path,
                   help="OFFLINE MODE: path to an already-obtained upstream plantseg115.py")
    p.add_argument("--commit-sha", help="OFFLINE MODE: the resolved 40-hex commit SHA")
    p.add_argument("--retrieved-utc", help="OFFLINE MODE: original retrieval timestamp (ISO-8601 Z)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the canonical payload to stdout and write nothing")
    args = p.parse_args(argv)

    if args.resolve_main and args.from_file:
        print("ERROR: choose either --resolve-main or --from-file, not both.", file=sys.stderr)
        return 2
    if not args.resolve_main and not args.from_file:
        print("ERROR: no mode selected. Use --resolve-main (network) or --from-file (offline).\n"
              "       Network access NEVER happens without --resolve-main.", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="plantseg_src_"))
    try:
        if args.resolve_main:
            print("[net] resolving official main -> commit SHA ...")
            sha, commit_date = resolve_main_sha()
            print(f"[net] commit SHA        : {sha}")
            print(f"[net] upstream date     : {commit_date}")
            url = immutable_url(sha)
            print(f"[net] fetching pinned   : {url}")
            raw = fetch_pinned_source(sha)
            retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            if not args.commit_sha or len(args.commit_sha) != 40:
                print("ERROR: --from-file requires --commit-sha <40-hex>.", file=sys.stderr)
                return 2
            sha, commit_date = args.commit_sha, None
            raw = args.from_file.read_bytes()
            retrieved = args.retrieved_utc or (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + " (regenerated offline)")

        # hash received bytes -> write to temp -> reopen -> independently re-hash -> must match
        raw_sha = sha256_bytes(raw)
        tmp_src = tmp / "plantseg115.py"
        tmp_src.write_bytes(raw)
        reread_sha = sha256_bytes(tmp_src.read_bytes())
        if raw_sha != reread_sha:
            raise VendorError(f"raw byte hash mismatch after round-trip: {raw_sha} != {reread_sha}")
        print(f"[hash] raw source sha256: {raw_sha}  (round-trip verified)")
        print(f"[src ] bytes            : {len(raw):,}")

        names = extract_classes(tmp_src.read_text(encoding="utf-8"))
        entries = build_entries(names)
        payload = build_payload(entries, sha=sha, raw_sha=raw_sha, retrieved_utc=retrieved,
                                upstream_commit_date=commit_date)

        sem = semantic_class_map_sha256(entries)
        print(f"[hash] semantic class-map sha256: {sem}")

        if args.dry_run:
            print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
            print("[dry ] --dry-run: nothing was written")
            return 0

        dest = validated_out_path()
        local_sha = atomic_write_json(dest, payload, sem)
        print(f"[hash] local vendored file sha256: {local_sha}  (NOT embedded -- recursive)")
        print(f"[out ] wrote {dest.relative_to(Path(REPO).resolve())} "
              f"(temp sibling + atomic replace)")
        print(f"[ok  ] {len(entries)} entries; class 0 name = {entries[0]['name']!r} "
              f"(official empty string preserved)")
        return 0
    except VendorError as e:
        print(f"VENDOR FAILED: {e}", file=sys.stderr)
        print("No partial class-map artifact was written.", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
