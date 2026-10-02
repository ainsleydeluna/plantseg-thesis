#!/usr/bin/env python3
"""AM-7a item 5: the gradient-clipping value derived from E1's own telemetry (lane L-KD-HARDEN).

    python scripts/am7_clip_value.py --telemetry <seed-43 e1_telemetry.jsonl> <seed-44 e1_telemetry.jsonl>
                                     [--out reports/derived/am7_clip_value.json]

Inputs: the e1_telemetry.jsonl files of the real E1 runs of seeds 43 and 44, one file each. A file must
hold exactly one run_meta row (mode real, resumed_from null, grad_clip_norm null: a second row means an
appended run) and one train row for every iteration 1..max_iters, in order, each with a grad_norm:
train_e1 logs the global L2 norm of the student's gradients before any clipping. A run resumed into a
fresh directory is refused by resumed_from and by its missing first iterations. Both files must share
max_iters. train_e1 writes no run_end row, so a run is complete when its train rows are. Numbers are read
as the decimal text written in the file (json parse_float=Decimal), and the arithmetic below is exact.

Value: per seed, the largest grad_norm over all train rows and the first iteration where it occurs;
M = the larger of the two maxima; target = 1.5 x M; the value is the smallest number of the 1-2-5 series
(1, 2 or 5 times a power of ten) that is >= target.

Output: reports/derived/am7_clip_value.json, written once and never overwritten: the inputs with the
sha256 of the exact bytes read and their seed, the per-seed maxima and iterations, M, the factor, the
target and the series value (exact values as decimal text). Exit 0: written, with a RESULT line. Exit 2:
refused with a named reason (input_count, input_missing, duplicate_input, unreadable_input, nonfinite,
run_meta_rows, not_real, resumed, clipped, seeds, iterations, run_unfinished, horizon_mismatch,
no_grad_norm, bad_grad_norm, output_exists), or an argparse usage error; no output file is written.
Exit 4: an unexpected error; no output file is written.
"""
from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "reports" / "derived" / "am7_clip_value.json"
SEEDS = (43, 44)
FACTOR = Decimal("1.5")
SERIES = (1, 2, 5)
UNEXPECTED_ERROR_EXIT = 4


class Refused(RuntimeError):
    """A named refusal; `code` makes every guard testable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if REPO in p.parents else str(p)


class _NonFinite(ValueError):
    pass


def _no_constant(name: str):
    raise _NonFinite(name)


def series_ceiling(x: Decimal) -> Decimal:
    """The smallest 1-2-5 series value (1, 2 or 5 times a power of ten) that is >= x, for x > 0. Exact:
    x = d.ddd x 10^e with 1 <= d < 10, so the answer is among 1, 2, 5 and 10 times 10^e."""
    if not x.is_finite() or x <= 0:
        raise ValueError(f"series_ceiling needs a finite positive number, got {x!r}")
    e = x.adjusted()
    for m in SERIES + (10,):
        v = Decimal(m).scaleb(e)
        if v >= x:
            return v
    raise AssertionError("unreachable: 10 x 10^e > x")


def read_telemetry(path: Path) -> dict:
    """{seed, max_grad_norm (Decimal), iter, n_train_rows, max_iters, sha256} of one E1 telemetry file;
    sha256 is of the exact bytes parsed."""
    path = Path(path)
    if not path.is_file():
        raise Refused("input_missing", f"telemetry file not found: {path}")
    data = path.read_bytes()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise Refused("unreadable_input", f"{path}: not UTF-8 text ({e})") from e
    metas, train = [], []
    for n, ln in enumerate(lines, 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln, parse_float=Decimal, parse_constant=_no_constant)
        except _NonFinite as e:
            raise Refused("nonfinite", f"{path}: line {n} holds the non-finite value {e}") from e
        except json.JSONDecodeError as e:
            if n == len(lines):
                raise Refused("run_unfinished", f"{path}: the last line is torn (the run was cut "
                                                "mid-write)") from e
            raise Refused("unreadable_input", f"{path}: line {n} is not JSON ({e})") from e
        if not isinstance(row, dict):
            raise Refused("unreadable_input", f"{path}: line {n} is not a JSON object")
        if row.get("event") == "run_meta":
            metas.append(row)
        elif row.get("event") == "train":
            train.append(row)
    if len(metas) != 1:
        raise Refused("run_meta_rows", f"{path}: {len(metas)} run_meta rows; exactly one is required "
                                       "(more means a resumed or appended run)")
    meta = metas[0]
    if meta.get("mode") != "real":
        raise Refused("not_real", f"{path}: mode {meta.get('mode')!r}; only a real E1 run counts")
    if meta.get("resumed_from") is not None:
        raise Refused("resumed", f"{path}: run_meta resumed_from {meta.get('resumed_from')!r}; a resumed "
                                 "run lacks its first segment's gradient norms (official runs never resume)")
    if meta.get("grad_clip_norm") is not None:
        raise Refused("clipped", f"{path}: run_meta grad_clip_norm {meta.get('grad_clip_norm')!r}; the E1 "
                                 "runs of record are unclipped (AM-7)")
    seed, max_iters = meta.get("seed"), meta.get("max_iters")
    if isinstance(seed, bool) or not isinstance(seed, int) \
            or isinstance(max_iters, bool) or not isinstance(max_iters, int):
        raise Refused("run_meta_rows", f"{path}: run_meta seed {seed!r} / max_iters {max_iters!r}")
    if not any("grad_norm" in r for r in train):
        raise Refused("no_grad_norm", f"{path}: no train row carries grad_norm ({len(train)} train rows)")
    best, best_it, prev = None, None, None
    for r in train:
        it, g = r.get("iter"), r.get("grad_norm")
        if isinstance(it, bool) or not isinstance(it, int) or (prev is not None and it <= prev):
            raise Refused("iterations", f"{path}: train iteration {it!r} after {prev!r}; iterations "
                                        "must increase strictly")
        prev = it
        if isinstance(g, bool) or not isinstance(g, (int, Decimal)) or Decimal(g) < 0 \
                or not math.isfinite(float(g)):
            raise Refused("bad_grad_norm", f"{path}: the train row at iter {it} has grad_norm {g!r}; every "
                                           "train row needs a finite grad_norm >= 0")
        g = Decimal(g)
        if best is None or g > best:
            best, best_it = g, it
    if prev != max_iters:
        raise Refused("run_unfinished", f"{path}: the last train iteration is {prev}, run_meta max_iters "
                                        f"is {max_iters}")
    if train[0]["iter"] != 1 or len(train) != max_iters:
        raise Refused("iterations", f"{path}: {len(train)} train rows from iter {train[0]['iter']} to "
                                    f"{prev}; a complete run has one row for every iteration 1..{max_iters}")
    return {"seed": seed, "max_grad_norm": best, "iter": best_it, "n_train_rows": len(train),
            "max_iters": max_iters, "sha256": hashlib.sha256(data).hexdigest()}


def clip_value(paths, out: Path) -> dict:
    paths = [Path(p) for p in paths]
    if len(paths) != len(SEEDS):
        raise Refused("input_count", f"{len(paths)} telemetry files given; exactly {len(SEEDS)} are "
                                     f"required (E1 seeds {list(SEEDS)})")
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise Refused("duplicate_input", f"the same file was given twice: {[str(p) for p in paths]}")
    out = Path(out)
    if out.exists():
        raise Refused("output_exists", f"{out} already exists; the clipping value is derived once")
    runs = [read_telemetry(p) for p in paths]
    seeds = sorted(r["seed"] for r in runs)
    if seeds != sorted(SEEDS):
        raise Refused("seeds", f"the files hold E1 seeds {seeds}; exactly {list(SEEDS)} are required")
    horizons = sorted({r["max_iters"] for r in runs})
    if len(horizons) != 1:
        raise Refused("horizon_mismatch", f"the files' run_meta max_iters differ: {horizons}; the E1 runs "
                                          "of seeds 43 and 44 share one schedule")
    with decimal.localcontext() as ctx:
        ctx.prec = 100
        ctx.traps[decimal.Inexact] = True        # the arithmetic below must never round
        top = max(runs, key=lambda r: (r["max_grad_norm"], -r["seed"]))
        m = top["max_grad_norm"]
        if m <= 0:
            raise Refused("bad_grad_norm", f"the largest grad_norm is {m}; a clipping value needs a "
                                           "positive maximum")
        target = FACTOR * m
        value = series_ceiling(target)
    by_seed = sorted(zip(runs, paths), key=lambda rp: rp[0]["seed"])
    doc = {
        "format": "am7_clip_value/1",
        "rule": "AM-7a item 5: the smallest 1-2-5 series value >= 1.5 x the largest pre-clip grad_norm "
                "over all train rows of the E1 runs of seeds 43 and 44",
        "inputs": [{"path": _rel(p), "sha256": r["sha256"], "seed": r["seed"]} for r, p in by_seed],
        "per_seed": {str(r["seed"]): {"max_grad_norm": float(r["max_grad_norm"]),
                                      "max_grad_norm_text": str(r["max_grad_norm"]), "iter": r["iter"],
                                      "n_train_rows": r["n_train_rows"], "max_iters": r["max_iters"]}
                     for r, _ in by_seed},
        "max_grad_norm": float(m), "max_grad_norm_text": str(m), "max_seed": top["seed"],
        "factor": float(FACTOR), "target_text": str(target),     # exact; a float could round across
        "series": "1-2-5", "series_value": float(value),          # the series point (5.0 beside 10)
        "series_value_text": format(value.normalize(), "f"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": {"path": "scripts/am7_clip_value.py", "sha256": sha256_file(Path(__file__))},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, indent=2) + "\n")
        if out.exists():                          # re-checked just before the rename
            raise Refused("output_exists", f"{out} appeared meanwhile; the clipping value is derived once")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AM-7a item 5: the clipping value from E1 seeds 43 and 44.")
    ap.add_argument("--telemetry", nargs="+", required=True,
                    help="the e1_telemetry.jsonl files of the real E1 runs of seeds 43 and 44")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    try:
        doc = clip_value(a.telemetry, Path(a.out))
    except Refused as e:
        print(f"REFUSED [{e.code}]: {e}", file=sys.stderr)
        print(f"RESULT: REFUSED ({e.code})")
        return 2
    except Exception as e:  # noqa: BLE001 — never a value, never a bare traceback
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT: ERROR (nothing was written)")
        return UNEXPECTED_ERROR_EXIT
    for seed, r in doc["per_seed"].items():
        print(f"  seed {seed}: max grad_norm {r['max_grad_norm_text']} at iter {r['iter']} "
              f"({r['n_train_rows']} train rows)")
    print(f"RESULT: AM7_CLIP_VALUE = {doc['series_value_text']} (1.5 x {doc['max_grad_norm_text']} "
          f"[seed {doc['max_seed']}] = {doc['target_text']}; 1-2-5 series) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
