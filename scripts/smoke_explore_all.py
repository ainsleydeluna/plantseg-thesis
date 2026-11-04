#!/usr/bin/env python3
"""Run every lane L-AM24 smoke, each in a fresh process (python -B -I), and print one table of RESULT lines.

    python -B -I scripts/smoke_explore_all.py        (exit 0 when every smoke passes, 1 otherwise)

Synthetic inputs only; no dataset, no GPU, no TEST. L_AM24_SMOKE_DIR, when set, is where the smokes make their
scratch folders (outside the repository); otherwise the system temporary folder is used.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKES = ("smoke_explore_guard", "smoke_explore_decompose", "smoke_explore_sites", "smoke_explore_qat",
          "smoke_explore_kd", "smoke_explore_kl3")


def main() -> int:
    rows, ok = [], True
    for s in SMOKES:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-B", "-I", str(REPO / "scripts" / f"{s}.py")], cwd=str(REPO),
                           capture_output=True, text=True)
        res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT: SMOKE")]
        muts = sum(1 for ln in p.stdout.splitlines() if "mutation KILLED" in ln and "[PASS]" in ln)
        good = p.returncode == 0 and bool(res) and " PASS " in res[-1]
        ok &= good
        rows.append((s, res[-1] if res else f"(no RESULT; exit {p.returncode})", muts, time.time() - t0))
        if not good:
            sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
    for s, r, m, t in rows:
        print(f"{s:<26} {r:<48} mutations killed {m:>2}  {t:6.1f}s")
    print(f"RESULT: L-AM24 SMOKES {'PASS' if ok else 'FAIL'} ({sum(1 for r in rows if ' PASS ' in r[1])}/{len(rows)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
