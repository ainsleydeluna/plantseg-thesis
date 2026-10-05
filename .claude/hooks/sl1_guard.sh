#!/bin/sh
# SL-1 launcher for sl1_guard.py (PreToolUse, lane L-SL1-GUARD). POSIX sh: Git Bash on Windows, sh on Linux.
# The policy exits 0 (no objection, or a would-be denial it logged in log mode) or 3 (deny, deny mode only);
# 3 becomes 2 here, the exit code that blocks. Anything else is a hook error: it is logged (rule SL1-E2)
# and, in log mode, never blocks. In deny mode a hook error follows fail mode (a): shell and search tools
# are denied; a file tool only when its path is test-named or NAMED_DENY, or cannot be read. This launcher
# reads only tool_name and file_path / notebook_path from the payload, never content, old_string or
# new_string. MODE and NAMED_DENY come from the policy's own lines, so the one-line MODE flip there flips
# both layers; a missing policy file or an unreadable MODE line counts as log. stdout stays empty.
# `python` is tried first: on Windows `python3` may be the Microsoft Store stub.
here=$(dirname "$0")
policy="$here/sl1_guard.py"
payload=$(cat 2>/dev/null)

mode_of_policy() {
  m=$(sed -n 's/^MODE = "\([a-z]*\)"$/\1/p' "$policy" 2>/dev/null | head -n 1)
  if [ "$m" = deny ]; then echo deny; else echo log; fi
}

field() {  # the string value of the JSON key $1 (no decoding; JSON escapes cannot fake a key)
  printf '%s' "$payload" | tr -d '\r\n' | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -n 1
}

log_line() {  # tool, rule, subject: appended outside the repository
  f=${SL1_GUARD_LOG:-}
  if [ -z "$f" ]; then
    [ -n "${HOME:-}" ] || return 0
    f="$HOME/.claude/sl1_guard.log"
  fi
  if [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
    case "$f" in "$CLAUDE_PROJECT_DIR"/*|"$CLAUDE_PROJECT_DIR"\\*) f="${HOME:-/nonexistent}/.claude/sl1_guard.log" ;; esac
  fi
  {
    mkdir -p "$(dirname "$f")" &&
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" \
      "$(printf '%s' "$3" | tr '\t\r\n' '   ' | cut -c1-200)" >> "$f"
  } 2>/dev/null
  return 0
}

sensitive_path() {  # test-named, or a component matching NAMED_DENY; an unreadable list counts as a match
  lower=$(printf '%s' "$1" | tr 'A-Z' 'a-z')
  case "$lower" in *test*) return 0 ;; esac
  pats=$(sed -n 's/^NAMED_DENY = (\(.*\))$/\1/p' "$policy" 2>/dev/null | head -n 1 | tr -d '" ' | tr ',' ' ')
  [ -n "$pats" ] || return 0
  set -f
  old_ifs=$IFS
  IFS='/\'
  for comp in $lower; do
    IFS=$old_ifs
    for pat in $pats; do
      case "$comp" in $pat) set +f; return 0 ;; esac
    done
    IFS='/\'
  done
  IFS=$old_ifs
  set +f
  return 1
}

fallback() {  # $1: what failed
  tool=$(field tool_name)
  path=""
  case "$tool" in
    Read|Edit|Write|MultiEdit) kind=file; path=$(field file_path) ;;
    NotebookEdit) kind=file; path=$(field notebook_path) ;;
    *) kind=other ;;
  esac
  log_line "${tool:-?}" SL1-E2 "$1${path:+ | $path}"
  [ "$(mode_of_policy)" = deny ] || exit 0
  if [ "$kind" = file ] && [ -n "$path" ] && ! sensitive_path "$path"; then
    exit 0
  fi
  echo "[SL1 E2] guard error ($1); denied by fail mode (a). Allowed: repair the guard." >&2
  exit 2
}

[ -r "$policy" ] || fallback "policy file missing"
last=""
for py in ${SL1_GUARD_PYTHONS:-python python3}; do
  command -v "$py" >/dev/null 2>&1 || continue
  err=$(printf '%s' "$payload" | "$py" "$policy" 2>&1 >/dev/null)
  rc=$?
  [ "$rc" -eq 0 ] && exit 0
  if [ "$rc" -eq 3 ]; then
    [ "$(mode_of_policy)" = deny ] || exit 0
    printf '%s\n' "$err" | head -n 1 >&2
    exit 2
  fi
  last="$py exited $rc: $(printf '%s' "$err" | tail -n 1)"
done
fallback "${last:-no working Python interpreter}"
