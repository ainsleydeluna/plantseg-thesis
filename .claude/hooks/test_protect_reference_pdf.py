#!/usr/bin/env python3
"""L-PROT R4: data-free test of the PreToolUse guard (protect_reference_pdf.py and its launcher).

Nothing here runs git or opens a repository file. Each case sends a synthetic hook payload, with a
made-up project directory, to the guard on stdin and checks the exit code (0 allow / 2 deny). For a
deny, it also checks the rule tag on stderr.

  python .claude/hooks/test_protect_reference_pdf.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE / "protect_reference_pdf.py"
LAUNCHER = HERE / "protect_reference_pdf.sh"
PROJ = "C:\\lab\\proj" if os.name == "nt" else "/lab/proj"
SEP = "\\" if os.name == "nt" else "/"
X = "':(exclude)docs/reference/reference.pdf'"
INCIDENT = "git grep -n -I -w 'G2' 3c43f89 -- docs AGENTS.md configs scripts src"

# (tool, input, cwd relative to the project or an absolute path, expected rc, expected rule or "")
CASES = [
    # --- R4 required: the incident, the allowed probe, and each excluded form
    ("Bash", INCIDENT, "", 2, "G"),
    ("Bash", "git status --porcelain=v1 -- docs/reference/reference.pdf", "", 0, ""),
    ("Bash", f"{INCIDENT} {X}", "", 0, ""),
    ("Bash", f"git status -sb -- . {X}", "", 0, ""),
    ("Bash", f"git status --porcelain=v1 --untracked-files=normal -- . {X}", "", 0, ""),
    ("Bash", f"git diff -- . {X}", "", 0, ""),
    ("Bash", f"git diff --stat -- . {X}", "", 0, ""),
    ("Bash", f"git diff --cached --name-only -- . {X}", "", 0, ""),
    ("Bash", f"git log -p -3 -- . {X}", "", 0, ""),
    ("Bash", f"git log --stat -5 -- . {X}", "", 0, ""),
    ("Bash", f"git show 7c5b990 -- . {X}", "", 0, ""),
    ("Bash", f"git show --stat HEAD -- . {X}", "", 0, ""),
    ("Bash", f"git ls-files -s -- . {X}", "", 0, ""),
    ("Bash", f"git ls-files -- docs {X}", "", 0, ""),
    ("Bash", f"git archive --format=tar -o out.tar HEAD -- . {X}", "", 0, ""),
    ("Bash", f"git stash push -- . {X}", "", 0, ""),
    ("Bash", "git commit --only -m 'B65: x' -- AGENTS.md docs/DECISION_LOG.md", "", 0, ""),
    # --- R4 required: the same forms without the exclude are denied
    ("Bash", "git status -sb", "", 2, "G"),
    ("Bash", "git -c core.quotepath=false status --porcelain=v1 -z --untracked-files=all", "", 2, "G"),
    ("Bash", "git diff", "", 2, "G"),
    ("Bash", "git diff --stat", "", 2, "G"),
    ("Bash", "git diff --stat -- docs", "", 2, "G"),
    ("Bash", "git diff -- 'docs/*.pdf'", "", 2, "N"),
    ("Bash", "git log -p -5", "", 2, "G"),
    ("Bash", "git log --stat -- .", "", 2, "G"),
    ("Bash", "git show 7c5b990", "", 2, "G"),
    ("Bash", "git show --stat HEAD", "", 2, "G"),
    ("Bash", "git ls-files -s", "", 2, "G"),
    ("Bash", "git ls-files docs", "", 2, "G"),
    ("Bash", "git ls-tree -r HEAD", "", 2, "G"),
    ("Bash", "git ls-tree HEAD docs/reference/", "", 2, "G"),
    ("Bash", "git ls-tree HEAD:docs/reference", "", 2, "G"),
    ("Bash", "git cat-file -p HEAD:docs/reference", "", 2, "G"),
    ("Bash", "git cat-file -p 1a2b3c4d", "", 2, "G"),
    ("Bash", "git archive HEAD", "", 2, "G"),
    ("Bash", "git archive -o out.tar HEAD -- docs", "", 2, "G"),
    ("Bash", "git grep -n foo", "", 2, "G"),
    ("Bash", "git grep -n foo -- .", "", 2, "G"),
    # --- rule C: commit
    ("Bash", "git commit -m x", "", 2, "C"),
    ("Bash", "git commit -am x", "", 2, "C"),
    ("Bash", "git commit --only -m x", "", 2, "C"),
    ("Bash", "git commit -m x -- AGENTS.md", "", 2, "C"),
    ("Bash", "git commit --only -a -m x -- AGENTS.md", "", 2, "C"),
    ("Bash", "git commit --only -m x -- docs", "", 2, "G"),
    # --- rule N: naming
    ("Bash", "cat docs/reference/reference.pdf", "", 2, "N"),
    ("Bash", "sha256sum docs/reference/*.pdf", "", 2, "N"),
    ("Bash", "ls -la docs/reference/reference.PDF", "", 2, "N"),
    ("Bash", "git status --porcelain=v1 -- docs/reference/reference.pdf; echo done", "", 2, "N"),
    ("Bash", "git -C . status --porcelain=v1 -- docs/reference/reference.pdf", "", 2, "N"),
    ("Bash", "ls *.pdf", "docs/reference", 2, "N"),
    ("PowerShell", "Get-Content docs\\reference\\reference.pdf", "", 2, "N"),
    # --- parsing, nesting, wrappers, environment
    ("Bash", "bash -c 'git status'", "", 2, "G"),
    ("Bash", "echo \"$(git diff --stat)\"", "", 2, "G"),
    ("Bash", "echo `git status`", "", 2, "G"),
    ("Bash", "git log --oneline -3\ngit status -s", "", 2, "G"),
    ("Bash", "cd src && git status -sb", "", 2, "G"),    # status reports the whole repo from any cwd
    ("Bash", "cd src && git diff --stat -- .", "", 0, ""),
    ("Bash", f"cd src && git diff --stat -- ':/' {X}", "", 2, "G"),
    ("Bash", "cd .. && git diff --stat", "src", 2, "G"),
    ("Bash", "git status 2>/dev/null", "", 2, "G"),
    ("Bash", "if git diff --quiet; then echo same; fi", "", 2, "G"),
    ("Bash", f"GIT_LITERAL_PATHSPECS=1 git diff -- . {X}", "", 2, "G"),
    ("Bash", f"git --literal-pathspecs diff -- . {X}", "", 2, "G"),
    ("Bash", "git -c alias.st=status st", "", 2, "G"),
    ("Bash", "git frobnicate", "", 2, "P"),
    ("Bash", "git --git-dir=.git log", "", 2, "P"),
    ("Bash", "git status \"", "", 2, "P"),
    ("Bash", "cat <<'EOF'\nno terminator", "", 2, "P"),
    ("Bash", "git add -A", "", 2, "G"),
    ("Bash", "git add .", "", 2, "G"),
    ("Bash", "git stash", "", 2, "G"),
    ("Bash", "git reset --hard", "", 2, "G"),
    ("Bash", "git checkout -- .", "", 2, "G"),
    ("Bash", "git restore docs", "", 2, "G"),
    ("Bash", "timeout 30 git diff --stat", "", 2, "G"),
    ("Bash", "echo docs | xargs git ls-files", "", 2, "G"),
    ("PowerShell", "git status -sb", "", 2, "G"),
    ("PowerShell", f"git status -sb -- . {X}", "", 0, ""),
    ("PowerShell", "Get-ChildItem -Recurse docs", "", 2, "S"),
    ("Monitor", "git log -p", "", 2, "G"),
    # --- allowed everyday commands
    ("Bash", "git log --oneline -5", "", 0, ""),
    ("Bash", "git rev-parse HEAD", "", 0, ""),
    ("Bash", "git ls-remote --heads origin master", "", 0, ""),
    ("Bash", "git push origin claude/keen-curie-u4a8ig", "", 0, ""),
    ("Bash", "git show -s --format=%H HEAD", "", 0, ""),
    ("Bash", "git show HEAD:docs/DECISION_LOG.md", "", 0, ""),
    ("Bash", "git diff --stat -- src/eval/artifacts.py", "", 0, ""),
    ("Bash", "git blame -- src/seeds.py", "", 0, ""),
    ("Bash", "git add -- AGENTS.md", "", 0, ""),
    ("Bash", "git ls-tree HEAD docs/", "", 0, ""),
    ("Bash", "git cat-file -p HEAD:docs", "", 0, ""),
    ("Bash", "git checkout -b lprot-test", "", 0, ""),
    ("Bash", "git -C src diff --stat -- .", "", 0, ""),
    ("Bash", "git commit --only -F msg.txt -- .claude/settings.json .claude/hooks/protect_reference_pdf.py",
     "", 0, ""),
    ("Bash", "git commit --only -m \"$(cat <<'EOF'\nB65: don't stop\n\nbody\nEOF\n)\" -- AGENTS.md", "", 0, ""),
    ("Bash", "ls src", "", 0, ""),
    ("Bash", "echo hi && python scripts/smoke_environment.py", "", 0, ""),
    ("Bash", "cat docs/reference/ch3.pdf | head -c 0", "", 0, ""),
    # --- R2b: non-git readers and the Grep/Glob tools
    ("Bash", "grep -rn G2 docs", "", 2, "S"),
    ("Bash", "grep -rn G2 src", "", 0, ""),
    ("Bash", "rg G2", "", 2, "S"),
    ("Bash", "rg -g '*.md' G2", "", 0, ""),
    ("Bash", "find . -name x", "", 2, "S"),
    ("Bash", "find src -name x", "", 0, ""),
    ("Bash", "ls docs/reference", "", 2, "S"),
    ("Bash", "ls -R docs", "", 2, "S"),
    ("Bash", "du -sh .", "", 2, "S"),
    ("Bash", "tar czf /tmp/a.tgz docs", "", 2, "S"),
    ("Bash", "python -c \"import subprocess; subprocess.run(['git','status'])\"", "", 2, "S"),
    ("Grep", {"pattern": "G2"}, "", 2, "S"),
    ("Grep", {"pattern": "G2", "path": "docs"}, "", 2, "S"),
    ("Grep", {"pattern": "G2", "glob": "*.md"}, "", 0, ""),
    ("Grep", {"pattern": "G2", "type": "py"}, "", 0, ""),
    ("Grep", {"pattern": "G2", "path": "src"}, "", 0, ""),
    ("Grep", {"pattern": "G2", "glob": "*.pdf"}, "", 2, "S"),
    ("Glob", {"pattern": "**/*.pdf"}, "", 2, "S"),
    ("Glob", {"pattern": "**/*"}, "", 2, "S"),
    ("Glob", {"pattern": "*", "path": "docs/reference"}, "", 2, "S"),
    ("Glob", {"pattern": "**/*.md"}, "", 0, ""),
    ("Glob", {"pattern": "docs/reference/ch*.pdf"}, "", 0, ""),
    # --- tools outside the matcher pass through (the deny rules cover Read/Edit/Write)
    ("Read", {"file_path": "x"}, "", 0, ""),
]


def _abs(rel: str) -> str:
    if rel.startswith(("/", "C:")):
        return rel
    return PROJ + (SEP + rel.replace("/", SEP) if rel else "")


def run(argv: list[str], payload, env_extra=None) -> tuple[int, str]:
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = PROJ
    env.update(env_extra or {})
    data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    p = subprocess.run(argv, input=data.encode("utf-8") if isinstance(data, str) else data,
                       capture_output=True, env=env, timeout=60)
    return p.returncode, p.stderr.decode("utf-8", "replace").strip()


def payload(tool, inp, cwd_rel):
    ti = {"command": inp} if isinstance(inp, str) else inp
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": ti, "cwd": _abs(cwd_rel)}


def main() -> int:
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    policy = [sys.executable, str(POLICY)]
    for k, (tool, inp, cwd, want_rc, want_rule) in enumerate(CASES):
        rc, err = run(policy, payload(tool, inp, cwd))
        label = inp if isinstance(inp, str) else json.dumps(inp)
        ok = rc == want_rc and (not want_rule or f"[L-PROT {want_rule}]" in err)
        check(f"{k:03d} {tool} {'deny' if want_rc else 'allow'} {want_rule or ''}".rstrip(),
              ok, f"{label[:70]!r} -> rc={rc} {err[:110]}")

    # fail-closed inputs
    for name, raw in (("empty_stdin", b""), ("not_json", b"{nope"), ("json_list", b"[]"),
                      ("no_tool_input", b'{"tool_name": "Bash"}')):
        rc, err = run(policy, raw)
        check(f"failclosed_{name}", rc == 2 and "[L-PROT P]" in err, f"rc={rc} {err[:90]}")
    rc, err = run(policy, payload("Bash", "ls", ""), env_extra={"CLAUDE_PROJECT_DIR": ""})
    check("failclosed_no_project_dir", rc == 2, f"rc={rc} {err[:90]}")

    # launcher (needs a POSIX sh; reported SKIP, never PASS, when none is available)
    sh = shutil.which("sh")
    if sh is None:
        print("[SKIP] launcher_* (no sh on PATH)")
    else:
        rc, _ = run([sh, str(LAUNCHER)], payload("Bash", INCIDENT, ""))
        check("launcher_denies_incident", rc == 2, f"rc={rc}")
        rc, _ = run([sh, str(LAUNCHER)], payload("Bash", "git log --oneline -1", ""))
        check("launcher_allows_log", rc == 0, f"rc={rc}")
        with tempfile.TemporaryDirectory() as empty:
            rc, err = run([sh, str(LAUNCHER)], payload("Bash", "git log --oneline -1", ""),
                          env_extra={"PATH": empty})
            check("launcher_failclosed_without_python", rc == 2, f"rc={rc} {err[:80]}")
        missing = str(HERE / "does_not_exist.sh")
        rc, _ = run([sh, "-c", f'sh "{missing}" || exit 2'], payload("Bash", "ls", ""))
        check("settings_command_failclosed_missing_script", rc == 2, f"rc={rc}")

    n_fail = results.count(False)
    print(f"\nRESULT: {'PASS' if n_fail == 0 else 'FAIL'} ({len(results) - n_fail}/{len(results)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
