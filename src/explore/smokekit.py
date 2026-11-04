"""A small case recorder for the lane's smokes (scripts/smoke_explore_*.py): named cases, refusal expectations,
mutation runs and one RESULT line. Synthetic inputs only; nothing here reads a dataset."""
from __future__ import annotations

import contextlib
import tempfile
import traceback
from pathlib import Path

SCRATCH_ENV = "L_AM24_SMOKE_DIR"


class Cases:
    def __init__(self, name: str, verbose: bool = True):
        self.name, self.verbose = name, verbose
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, case: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((case, ok, detail))
        if self.verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {case}{(' -- ' + detail) if detail else ''}")
        return ok

    def refuses(self, case: str, code: str, fn, *args, **kwargs) -> bool:
        """`fn(*args, **kwargs)` must raise an exception whose `code` attribute is `code`."""
        try:
            fn(*args, **kwargs)
        except Exception as e:                                # noqa: BLE001 -- the refusal is the case
            got = getattr(e, "code", None)
            return self.check(case, got == code, f"raised [{got}] {str(e)[:160]}")
        return self.check(case, False, f"no refusal (expected [{code}])")

    def passes(self, case: str, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
        except Exception as e:                                # noqa: BLE001
            self.check(case, False, f"raised {type(e).__name__}: {str(e)[:200]}")
            return None
        self.check(case, True)
        return out

    @property
    def failed(self) -> list[str]:
        return [c for c, ok, _ in self.rows if not ok]

    def result(self) -> int:
        n, bad = len(self.rows), self.failed
        print(f"RESULT: {self.name} {'PASS' if not bad else 'FAIL'} ({n - len(bad)}/{n})"
              + (f" failed: {bad}" if bad else ""))
        return 0 if not bad else 1


def mutation_killed(run_cases, mutate) -> tuple[bool, list[str]]:
    """Apply `mutate` (a context manager factory), run `run_cases()` -> Cases with verbose off, and report whether
    some case failed (the mutation is KILLED) and which."""
    with mutate():
        try:
            c = run_cases()
            failed = c.failed
        except Exception as e:                                # noqa: BLE001 -- a crash also kills the mutation
            failed = [f"crash: {type(e).__name__}: {e}"]
    return bool(failed), failed


@contextlib.contextmanager
def patched(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def scratch_dir(prefix: str) -> Path:
    import os
    base = os.environ.get(SCRATCH_ENV)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base if base else None))


def print_exc_short() -> None:
    traceback.print_exc(limit=3)
