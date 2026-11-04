"""Recorded run-time substitutions for one exploratory process (lane L-AM24).

Some items need one named change inside a pinned module's loop that the module offers no hook for (N2's sites on the
prepared student run_qat builds; A1'''s off-grid alpha and A2's DIST term in train_distill's run). The pinned file is
never edited: for the duration of one exploratory call, one module attribute is bound to a wrapper, then restored.
`substituted` yields the record that the run's x_run_meta.json and run_meta carry: the module, the attribute, the
pinned file and its sha256, the wrapper, and why. A substitution never outlives the call, never reaches another
process, and the official runs (separate processes) never see it.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def substitution_record(module, name: str, replacement, *, item: str, why: str) -> dict:
    src = Path(inspect.getsourcefile(module)).resolve()
    rel = src.relative_to(REPO.resolve()).as_posix() if REPO.resolve() in src.parents else str(src)
    return {"module": module.__name__, "attribute": name, "pinned_file": rel, "pinned_file_sha256": _sha256(src),
            "original": f"{module.__name__}.{name}",
            "replacement": f"{getattr(replacement, '__module__', '?')}.{getattr(replacement, '__qualname__', '?')}",
            "item": item, "why": why, "scope": "this process, this call only; the file is unchanged"}


@contextlib.contextmanager
def substituted(module, name: str, replacement, *, item: str, why: str):
    if not hasattr(module, name):
        raise AttributeError(f"{module.__name__} has no attribute {name!r} to substitute")
    rec = substitution_record(module, name, replacement, item=item, why=why)
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield rec
    finally:
        setattr(module, name, original)
