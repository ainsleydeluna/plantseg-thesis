#!/bin/sh
# L-PROT launcher for protect_reference_pdf.py (PreToolUse). POSIX sh: Git Bash on Windows, sh on Linux.
# Fail-closed: exit 0 only when the policy itself exits 0. Every other outcome exits 2, including no
# usable Python. `python` is tried first: on Windows `python3` may be the Microsoft Store stub.
payload=$(cat)
here=$(dirname "$0")
for py in python python3; do
  command -v "$py" >/dev/null 2>&1 || continue
  printf '%s' "$payload" | "$py" "$here/protect_reference_pdf.py"
  rc=$?
  [ "$rc" -eq 0 ] && exit 0
  [ "$rc" -eq 2 ] && exit 2
done
echo "[L-PROT P] no working Python interpreter for the guard; denied (fail-closed)." >&2
exit 2
