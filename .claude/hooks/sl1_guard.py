#!/usr/bin/env python3
"""SL-1 guard: a PreToolUse hook (lane L-SL1-GUARD) for Claude Code's file, search and shell tools.

SL-1: never list, grep, read or member-list a path containing "test" (case-insensitive), a dataset archive
or the issue #11 CSV; a whole-file sha256 is the only operation allowed on them. NAMED_DENY names the
archives and the CSV; every rule treats a NAMED_DENY path exactly as a test-named path. The guard also
refuses destructive commands on protected locations (rule 4 of the lane brief).

The guard reads the hook payload (JSON) on stdin and inspects only the text of the tool call. It never
opens, lists or hashes a file. It stats paths that are neither test-named nor NAMED_DENY (is this a file,
a directory, or inside a clone of this repository?), never a listing.

In log mode every call goes through and each would-be denial is appended to ~/.claude/sl1_guard.log
(outside the repository; SL1_GUARD_LOG overrides the path, but never to a path inside the project). In
deny mode the launcher blocks. The flip is the one MODE line below; sl1_guard.sh reads the same line.

  exit 0  no objection, or a would-be denial logged in log mode.
  exit 3  deny (deny mode only). sl1_guard.sh maps 3 to 2, the code that blocks. 2 itself is not used:
          python and dash both exit 2 when a file is missing, so 2 cannot mean "deny" in the launcher.
  stdout  always empty: Claude Code reads JSON-shaped stdout on exit 0 as a decision.
  stderr  deny mode only: one line naming the rule and the allowed form.

Rules (logged as SL1-<id>)
  R1  Read, Edit, Write or NotebookEdit on a test-named or NAMED_DENY path.
  G1  Grep or Glob over a directory, or the cwd, in a protected location.
  G2  Grep or Glob naming a test-named or NAMED_DENY path, glob filter or pattern.
  B1  Recursive, globbed or directory search or listing, or a bare listing, in a protected location.
  B2  git ls-files, grep or ls-tree, a tree object, or git diff/show/log name output, whose scope can
      include test paths (no ':(exclude,icase)*test*' and not only explicit files).
  B3  pytest, or python -m pytest / unittest.
  B4  A test-named or NAMED_DENY path given to a reader, search, listing, git path or `<` redirect.
  B5  The command could not be analysed.
  B6  Patch or archive content from git whose scope can include test paths.
  B7  Running a test-named script.
  B8  Listing or extracting the members of a test-named or NAMED_DENY archive.
  D1  docker rmi, image rm/remove/prune, system prune, volume rm/remove/prune.
  D2  rm, rmdir, unlink, shred, del, erase, rd, Remove-Item, git clean without -n, or find -delete, on a
      target inside or above a protected location, or one that cannot be resolved.
  E1  The policy failed. Logged; in deny mode fail mode (a) applies (see fail_denies). The launcher's own
      failures are E2.

Exemptions, B1 and B2 only:
  filter  the listing prints one full path per line, and its stdout reaches a later stage
          `grep -vi test` (any spelling of -v and -i, pattern exactly "test", no -w/-x, no files) or
          PowerShell `Select-String -NotMatch test` (not fed FileInfo objects), with no tee or redirect
          before it.
  count   the pipeline ends in `wc -l`, `grep -c` (no files), Measure-Object, or the listing sits in
          `( ... ).Count`; nothing before the count tees or redirects.
Protected location: the project; any directory inside a clone of this repository (an ancestor holds
.claude/hooks/protect_reference_pdf.sh); any path through plantseg_data or plantseg_runs;
$PLANTSEG_DATA_ROOT. A recursive target also counts when it is an ancestor of one of these, /workspace or a
filesystem root. A target that cannot be resolved ($VAR, ~user, cd -) is protected. C:\\x, C:/x and Git
Bash /c/x are one path.
Known gaps: docs/lane_reports/sl1-guard.md.
"""
from __future__ import annotations

import fnmatch
import json
import os
import posixpath
import re
import sys
import time

MODE = "log"
NAMED_DENY = ("plantseg_exact_duplicates.csv", "plantseg*.zip", "zenodo_*")
EXCEPTIONS: tuple = ()  # exact paths exempt from R1/G2/B4, granted by name by the user (none now)

EXCLUDE_TEST = ":(exclude,icase)*test*"
DENY_EXIT = 3
LOG_ENV = "SL1_GUARD_LOG"
REPO_MARKER = "/.claude/hooks/protect_reference_pdf.sh"
DATA_NAMES = ("plantseg_data", "plantseg_runs")
KNOWN_DATA_PARENTS = ("/workspace",)
FILE_TOOLS = ("Read", "Edit", "Write", "NotebookEdit", "MultiEdit")
SHELL_TOOLS = {"Bash": "bash", "Monitor": "bash", "PowerShell": "ps"}
MAX_DEPTH = 8
GLOB_CHARS = frozenset("*?[")
PLACEHOLDER = "$(SUB)"      # stands for a command substitution: never resolvable

WHY = {
    "R1": "file tool on a test-named or NAMED_DENY path",
    "G1": "Grep/Glob over a directory in a protected location",
    "G2": "Grep/Glob naming a test-named or NAMED_DENY path",
    "B1": "recursive, globbed or directory listing or search in a protected location",
    "B2": "git listing whose scope can include test paths",
    "B3": "pytest is not run",
    "B4": "test-named or NAMED_DENY path read, searched or listed",
    "B5": "command could not be analysed",
    "B6": "git patch or archive content whose scope can include test paths",
    "B7": "test-named script run",
    "B8": "member listing or extraction of a test-named or NAMED_DENY archive",
    "D1": "destructive docker image or volume command",
    "D2": "destructive command on a protected location",
    "E1": "guard error",
}
HINT = {
    "R1": "SL-1: sha256 only",
    "G1": "name the file, or use git grep with ':(exclude,icase)*test*'",
    "G2": "SL-1: sha256 only",
    "B1": "pipe through grep -vi test, or name the file",
    "B2": "add ':(exclude,icase)*test*', or pipe through grep -vi test",
    "B3": "python -B scripts/smoke_*.py (no pytest)",
    "B4": "SL-1: sha256 only",
    "B5": "split it into plain commands",
    "B6": "add ':(exclude,icase)*test*', or name the files",
    "B7": "SL-1: sha256 only",
    "B8": "SL-1: sha256 only",
    "D1": "none in Claude Code; a human runs it",
    "D2": "none in Claude Code; a human runs it",
    "E1": "repair the guard; fail mode (a) applies meanwhile",
}


class Finding(Exception):
    def __init__(self, rule: str, subject: str = "", why: str = ""):
        super().__init__(rule)
        self.rule = rule
        self.subject = subject
        self.why = why or WHY[rule]

    def line(self) -> str:
        return f"[SL1 {self.rule}] {self.why}. Allowed: {HINT[self.rule]}."


class Unparsable(Exception):
    pass


class _Unresolved(Exception):
    pass


class _Unknown:
    __slots__ = ()

    def __repr__(self):
        return "UNKNOWN"


UNKNOWN = _Unknown()


# ----------------------------------------------------------------------------------------- names
def _named(comp: str) -> bool:
    """A path component (lower case) that matches NAMED_DENY, or a glob component that can match it."""
    for pat in NAMED_DENY:
        if fnmatch.fnmatchcase(comp, pat):
            return True
    if any(c in GLOB_CHARS for c in comp) and len(re.sub(r"[*?\[\]]", "", comp)) >= 3:
        samples = [p.replace("*", "x") for p in NAMED_DENY]
        return any(fnmatch.fnmatchcase(s, comp) for s in samples)
    return False


def sensitive(text: str) -> bool:
    """True for a test-named path (literal, case-insensitive: "latest" counts) or a NAMED_DENY path."""
    t = text.lower()
    if "test" in t:
        return True
    return any(c and _named(c) for c in re.split(r"[\\/]", t))


# ----------------------------------------------------------------------------------------- paths
_DRIVE = re.compile(r"^([A-Za-z]):(/.*)?$")
_MSYS = re.compile(r"^/([A-Za-z])(/.*)?$")


def _norm(k: str) -> str:
    unc = k.startswith("//") and not k.startswith("///")
    n = posixpath.normpath(k)
    if unc and not n.startswith("//"):
        n = "/" + n
    if re.fullmatch(r"[a-z]:", n):
        n += "/"
    return n


def abs_key(p: str):
    """Comparison key of an absolute path (lower case, '/'), or None for a relative one.
    C:\\x, C:/x and Git Bash /c/x all give c:/x."""
    p = p.replace("\\", "/")
    m = _DRIVE.match(p) or _MSYS.match(p)
    if m:
        k = m.group(1) + ":" + (m.group(2) or "/")
    elif p.startswith("/"):
        k = p
    else:
        return None
    return _norm(k.lower())


def _real_abs(p: str):
    """A path the OS can stat for an absolute path, or None when it cannot exist on this OS."""
    p = p.replace("\\", "/")
    m = _DRIVE.match(p) or _MSYS.match(p)
    if m and os.name == "nt":
        r = posixpath.normpath(m.group(1).upper() + ":" + (m.group(2) or "/"))
        return r + "/" if re.fullmatch(r"[A-Za-z]:", r) else r
    if _DRIVE.match(p):
        return None
    return posixpath.normpath(p)


def _inside(a: str, b: str) -> bool:
    """Key a is b or below b."""
    return a == b or a.startswith(b.rstrip("/") + "/")


class P:
    __slots__ = ("key", "real")

    def __init__(self, key, real):
        self.key = key
        self.real = real

    def __repr__(self):
        return f"P({self.key!r})"


def glob_base(text: str) -> str:
    """The fixed leading part of a path glob ('.' when the first component already globs)."""
    t = text.replace("\\", "/")
    out = []
    for c in t.split("/"):
        if any(ch in GLOB_CHARS for ch in c) or ("{" in c and "," in c):
            break
        out.append(c)
    base = "/".join(out)
    if t.startswith("/") and not base:
        return "/"
    return base or "."


def has_glob(text: str) -> bool:
    return any(c in GLOB_CHARS for c in text)


def has_extension(text: str) -> bool:
    """The last path component looks like a file name (x.py, a.tar.gz), not a folder (e1_s42, .claude)."""
    base = re.split(r"[\\/]", text.rstrip("/\\"))[-1]
    return bool(re.search(r".\.[A-Za-z0-9_]{1,10}$", base))


_VAR = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$env:([A-Za-z_]\w*)|\$([A-Za-z_]\w*)|%([A-Za-z_]\w*)%", re.I)


class State:
    __slots__ = ("cwd", "vars")

    def __init__(self, cwd, vars_):
        self.cwd = cwd
        self.vars = vars_


class Ctx:
    """Everything a decision needs that does not depend on the command text."""

    def __init__(self, env, cwd_text):
        self.env = env
        self.uenv = {k.upper(): v for k, v in env.items()}
        self.home = os.path.expanduser("~")
        self.roots = []
        self.project = None
        proj = env.get("CLAUDE_PROJECT_DIR") or ""
        pk = abs_key(proj) if proj else None
        if pk:
            self.project = P(pk, _real_abs(proj))
            self.roots.append(pk)
        self.cwd = UNKNOWN
        if isinstance(cwd_text, str) and abs_key(cwd_text):
            self.cwd = P(abs_key(cwd_text), _real_abs(cwd_text))
        elif self.project is not None:
            self.cwd = self.project
        data_root = env.get("PLANTSEG_DATA_ROOT")
        if data_root:
            p = self.resolve(data_root, State(self.cwd, {}), "bash")
            if p is not UNKNOWN:
                self.roots.append(p.key)
        self._clone = {}
        self.excepted = set()
        for e in EXCEPTIONS:
            p = self.resolve(e, State(self.project or self.cwd, {}), "bash")
            if p is not UNKNOWN:
                self.excepted.add(p.key)

    # -- environment and expansion
    def getenv(self, name):
        v = self.env.get(name)
        if v is None and os.name == "nt":
            v = self.uenv.get(name.upper())
        return v

    def expand(self, text, st, dialect):
        t = text
        if t == "~" or t.startswith(("~/", "~\\")):
            t = self.home + t[1:]
        elif t.startswith("~"):
            return None
        if "$" not in t and "%" not in t:
            return t

        def rep(m):
            if m.group(4) is not None:
                if dialect != "cmd":
                    return m.group(0)
                v = self.getenv(m.group(4))
            elif m.group(2) is not None:
                v = self.getenv(m.group(2))
            else:
                name = m.group(1) or m.group(3)
                if name in st.vars:
                    v = st.vars[name]
                elif name.upper() == "HOME":
                    v = self.home
                elif name.upper() == "PWD":
                    v = None if st.cwd is UNKNOWN else st.cwd.real
                elif dialect == "ps":
                    v = None                    # PowerShell variables are not environment variables
                else:
                    v = self.getenv(name)
            if v is None or v is UNKNOWN:
                raise _Unresolved()
            return v

        try:
            out = _VAR.sub(rep, t)
        except _Unresolved:
            return None
        if "$" in out:
            return None
        return out

    def resolve(self, text, st, dialect="bash"):
        """P for a path token, or UNKNOWN."""
        t = self.expand(text, st, dialect)
        if t is None or t == "":
            return UNKNOWN
        t = t.replace("\\", "/")
        k = abs_key(t)
        if k is not None:
            return P(k, _real_abs(t))
        cwd = st.cwd
        if cwd is UNKNOWN:
            return UNKNOWN
        key = _norm(cwd.key.rstrip("/") + "/" + t.lower())
        real = posixpath.normpath(cwd.real.rstrip("/") + "/" + t) if cwd.real else None
        return P(key, real)

    # -- predicates
    def sens(self, text, st, dialect="bash"):
        """A test-named or NAMED_DENY path, judged on the token and on its resolved form."""
        p = self.resolve(text, st, dialect)
        if p is not UNKNOWN and p.key in self.excepted:
            return False
        if sensitive(text):
            return True
        return p is not UNKNOWN and sensitive(p.key)

    def _stat(self, fn, p):
        if p is UNKNOWN or p.real is None or sensitive(p.key):
            return False
        try:
            return fn(p.real)
        except (OSError, ValueError):
            return False

    def is_dir(self, p):
        return self._stat(os.path.isdir, p)

    def is_file(self, p):
        return self._stat(os.path.isfile, p)

    def in_clone(self, p):
        """p is inside a checkout of this repository: one of p or its ancestors holds the marker file."""
        if p is UNKNOWN or p.real is None or sensitive(p.key):
            return False
        cur = p.real.rstrip("/") or "/"
        for _ in range(64):
            if cur in self._clone:
                return self._clone[cur]
            try:
                hit = os.path.isfile(cur.rstrip("/") + REPO_MARKER)
            except (OSError, ValueError):
                hit = False
            if hit:
                self._clone[cur] = True
                return True
            parent = posixpath.dirname(cur)
            if parent == cur or not parent:
                break
            cur = parent
        return False

    def protected(self, p, recursive):
        if p is UNKNOWN:
            return True
        k = p.key
        for r in self.roots:
            if _inside(k, r) or (recursive and _inside(r, k)):
                return True
        for c in k.split("/"):
            if c in DATA_NAMES or c.startswith("plantseg-thesis"):
                return True
        if recursive:
            if k == "/" or re.fullmatch(r"[a-z]:/", k):
                return True
            for d in KNOWN_DATA_PARENTS:
                if _inside(d, k):
                    return True
        return self.in_clone(p)

    # -- the four non-shell tools
    def file_tool(self, tool, ti):
        path = ti.get("notebook_path") if tool == "NotebookEdit" else ti.get("file_path")
        if not isinstance(path, str) or not path:
            return None                                       # the tool rejects it itself
        if self.sens(path, State(self.cwd, {})):
            return Finding("R1", path)
        return None

    def home_p(self):
        k = abs_key(self.home)
        return P(k, _real_abs(self.home)) if k else UNKNOWN

    def grep_tool(self, ti):
        path = ti.get("path")
        glob = ti.get("glob")
        subject = f"{path or '<cwd>'}" + (f" glob={glob}" if glob else "")
        st = State(self.cwd, {})
        if isinstance(glob, str) and glob and not glob.startswith("!") and sensitive(glob):
            return Finding("G2", subject)
        if isinstance(path, str) and path:
            if self.sens(path, st):
                return Finding("G2", subject)
            if has_glob(path):
                p = self.resolve(glob_base(path), st)
            else:
                p = self.resolve(path, st)
                if p is not UNKNOWN and (self.is_file(p) or (not self.is_dir(p) and has_extension(path))):
                    return None
        else:
            p = self.cwd
        if self.protected(p, True):
            return Finding("G1", subject)
        return None

    def glob_tool(self, ti):
        pattern = ti.get("pattern") if isinstance(ti.get("pattern"), str) else ""
        path = ti.get("path") if isinstance(ti.get("path"), str) else ""
        subject = f"{path or '<cwd>'} pattern={pattern}"
        st = State(self.cwd, {})
        if sensitive(pattern) or (path and self.sens(path, st)):
            return Finding("G2", subject)
        base = self.resolve(path, st) if path else self.cwd
        fixed = glob_base(pattern)
        if fixed not in (".", ""):
            if abs_key(fixed) is not None:
                base = self.resolve(fixed, st)
            elif base is not UNKNOWN:
                base = self.resolve(fixed, State(base, {}))
        if self.protected(base, True):
            return Finding("G1", subject)
        return None


# ----------------------------------------------------------------------------------------- lexer
class Word:
    __slots__ = ("text", "glob", "quoted", "comma")

    def __init__(self, text, glob=False, quoted=False, comma=False):
        self.text = text
        self.glob = glob
        self.quoted = quoted
        self.comma = comma

    def __repr__(self):
        return f"W({self.text!r})"


class Redir:
    __slots__ = ("op", "fd", "target")

    def __init__(self, op, fd):
        self.op = op
        self.fd = fd
        self.target = None


class Cmd:
    __slots__ = ("words", "redirs")

    def __init__(self, words, redirs):
        self.words = words
        self.redirs = redirs


class Lexer:
    """Splits a command line into pipelines of simple commands. Dialects: bash, ps (PowerShell), cmd.
    Command substitutions, script blocks and subexpressions become separate nested scripts."""

    def __init__(self, src, dialect):
        self.s = src
        self.n = len(src)
        self.d = dialect
        self.esc = {"bash": "\\", "ps": "`", "cmd": "^"}[dialect]
        self.pipelines = []
        self.nested = []
        self.pipe = []
        self.words = []
        self.redirs = []
        self.buf = None
        self.glob = self.quoted = self.brace = self.comma_next = False
        self.pending = None
        self.heredocs = []

    def run(self):
        i = 0
        step = {"bash": self.step_bash, "ps": self.step_ps, "cmd": self.step_cmd}[self.d]
        while i < self.n:
            i = step(i)
        self.end_pipeline()
        return self.pipelines, self.nested

    # -- buffers
    def add(self, ch, quoted=False):
        if self.buf is None:
            self.buf = []
        self.buf.append(ch)
        if quoted:
            self.quoted = True

    def add_text(self, text, quoted=False):
        if self.buf is None:
            self.buf = []
        self.buf.append(text)
        if quoted:
            self.quoted = True

    def mark(self, ch):
        if ch in GLOB_CHARS and not (self.d == "cmd" and ch == "["):
            self.glob = True
        elif ch == "{":
            self.brace = True
        elif ch == "," and self.brace:
            self.glob = True

    def end_word(self):
        if self.buf is None:
            return
        w = Word("".join(self.buf), self.glob, self.quoted, self.comma_next)
        self.buf = None
        self.glob = self.quoted = self.brace = self.comma_next = False
        if self.pending is not None:
            self.pending.target = w
            self.redirs.append(self.pending)
            if self.pending.op in ("<<", "<<-"):
                self.heredocs.append((w.text, self.pending.op == "<<-"))
            self.pending = None
        else:
            self.words.append(w)

    def end_cmd(self):
        self.end_word()
        if self.pending is not None:
            self.redirs.append(self.pending)
            self.pending = None
        if self.words or self.redirs:
            self.pipe.append(Cmd(self.words, self.redirs))
        self.words = []
        self.redirs = []

    def end_pipeline(self):
        self.end_cmd()
        if self.pipe:
            self.pipelines.append(self.pipe)
        self.pipe = []

    # -- scanning helpers
    def capture(self, i, op="(", cl=")"):
        """Text up to the bracket matching an already-consumed opener; returns (text, index after)."""
        s = self.s
        depth = 1
        j = i
        while j < self.n:
            c = s[j]
            if c == self.esc:
                j += 2
                continue
            if c == "'" and self.d != "cmd":
                k = s.find("'", j + 1)
                if k < 0:
                    raise Unparsable("unterminated single quote")
                j = k + 1
                continue
            if c == '"':
                k = j + 1
                while k < self.n and s[k] != '"':
                    if s[k] == self.esc:
                        k += 1
                    k += 1
                if k >= self.n:
                    raise Unparsable("unterminated double quote")
                j = k + 1
                continue
            if c == op:
                depth += 1
            elif c == cl:
                depth -= 1
                if depth == 0:
                    return s[i:j], j + 1
            j += 1
        raise Unparsable(f"unbalanced {op}")

    def find_backtick(self, i):
        s = self.s
        j = i
        while j < self.n:
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == "`":
                return j
            j += 1
        raise Unparsable("unterminated backtick")

    def skip_heredocs(self, i):
        if not self.heredocs:
            return i
        s = self.s
        for delim, strip in self.heredocs:
            while i < self.n:
                j = s.find("\n", i)
                line = s[i:] if j < 0 else s[i:j]
                i = self.n if j < 0 else j + 1
                if (line.lstrip("\t") if strip else line) == delim:
                    break
        self.heredocs = []
        return i

    # -- bash
    def step_bash(self, i):
        s = self.s
        c = s[i]
        if c in " \t\r":
            self.end_word()
            return i + 1
        if c == "\n":
            self.end_pipeline()
            return self.skip_heredocs(i + 1)
        if c == "#" and self.buf is None:
            j = s.find("\n", i)
            return self.n if j < 0 else j
        if c == "\\":
            if i + 1 < self.n:
                if s[i + 1] == "\n":
                    return i + 2
                self.add(s[i + 1], quoted=True)
                return i + 2
            return i + 1
        if c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                raise Unparsable("unterminated single quote")
            self.add_text(s[i + 1:j], quoted=True)
            return j + 1
        if c == '"':
            return self.dquote_bash(i + 1)
        if c == "$":
            return self.dollar_bash(i)
        if c == "`":
            j = self.find_backtick(i + 1)
            self.nested.append(("bash", s[i + 1:j], False))
            self.add_text(PLACEHOLDER)
            return j + 1
        if c in "<>" and i + 1 < self.n and s[i + 1] == "(":
            text, j = self.capture(i + 2)
            self.nested.append(("bash", text, False))
            self.add_text(PLACEHOLDER)
            return j
        if c == "|":
            if s.startswith("||", i):
                self.end_pipeline()
                return i + 2
            self.end_cmd()
            return i + 2 if s.startswith("|&", i) else i + 1
        if c == "&":
            if s.startswith("&&", i):
                self.end_pipeline()
                return i + 2
            if s.startswith("&>", i):
                self.end_word()
                op = "&>>" if s.startswith("&>>", i) else "&>"
                self.pending = Redir(op, None)
                return i + len(op)
            self.end_pipeline()
            return i + 1
        if c == ";":
            self.end_pipeline()
            return i + 2 if s.startswith(";;", i) else i + 1
        if c in "()":
            self.end_pipeline()
            return i + 1
        if c in "<>":
            fd = None
            if self.buf is not None and not self.quoted and "".join(self.buf).isdigit():
                fd = "".join(self.buf)
                self.buf = None
                self.glob = self.quoted = self.brace = False
            else:
                self.end_word()
            for op in ("<<<", "<<-", "<<", "<>", "<&", ">>", ">|", ">&", "<", ">"):
                if s.startswith(op, i):
                    self.pending = Redir(op, fd)
                    return i + len(op)
        self.add(c)
        self.mark(c)
        return i + 1

    def dquote_bash(self, i):
        s = self.s
        self.add_text("", quoted=True)
        while i < self.n:
            c = s[i]
            if c == '"':
                return i + 1
            if c == "\\" and i + 1 < self.n and s[i + 1] in '$`"\\\n':
                if s[i + 1] != "\n":
                    self.buf.append(s[i + 1])
                i += 2
                continue
            if c == "$":
                i = self.dollar_bash(i, in_dq=True)
                continue
            if c == "`":
                j = self.find_backtick(i + 1)
                self.nested.append(("bash", s[i + 1:j], False))
                self.buf.append(PLACEHOLDER)
                i = j + 1
                continue
            self.buf.append(c)
            i += 1
        raise Unparsable("unterminated double quote")

    def dollar_bash(self, i, in_dq=False):
        s = self.s
        if s.startswith("$(", i):
            text, j = self.capture(i + 2)
            if not (text.startswith("(") and text.endswith(")")):      # $(( arithmetic )) holds no command
                self.nested.append(("bash", text, False))
            self.add_text(PLACEHOLDER)
            return j
        if s.startswith("${", i):
            _, j = self.capture(i + 2, "{", "}")
            self.add_text(s[i:j])
            return j
        if s.startswith("$'", i) and not in_dq:
            j = i + 2
            out = []
            while j < self.n and s[j] != "'":
                if s[j] == "\\" and j + 1 < self.n:
                    out.append(s[j + 1])
                    j += 2
                    continue
                out.append(s[j])
                j += 1
            if j >= self.n:
                raise Unparsable("unterminated $'")
            self.add_text("".join(out), quoted=True)
            return j + 1
        if i + 1 < self.n and s[i + 1] in "?*@#$!-0123456789":
            self.add_text(s[i:i + 2])                   # a special parameter ($?, $*, $1 ...): never a glob
            return i + 2
        self.add("$")
        return i + 1

    # -- PowerShell
    def count_suffix(self, j):
        tail = self.s[j:j + 7].lower()
        return tail.startswith(".count") and (len(tail) == 6 or not (tail[6].isalnum() or tail[6] == "_"))

    def step_ps(self, i):
        s = self.s
        c = s[i]
        if c in " \t\r":
            self.end_word()
            return i + 1
        if c == "\n":
            self.end_pipeline()
            return i + 1
        if c == "#" and self.buf is None:
            j = s.find("\n", i)
            return self.n if j < 0 else j
        if s.startswith("<#", i) and self.buf is None:
            j = s.find("#>", i + 2)
            return self.n if j < 0 else j + 2
        if c == "`":
            if i + 1 < self.n:
                if s[i + 1] == "\n":
                    return i + 2
                self.add(s[i + 1], quoted=True)
                return i + 2
            return i + 1
        if c == "'":
            j = i + 1
            out = []
            while True:
                k = s.find("'", j)
                if k < 0:
                    raise Unparsable("unterminated single quote")
                out.append(s[j:k])
                if s.startswith("''", k):
                    out.append("'")
                    j = k + 2
                    continue
                break
            self.add_text("".join(out), quoted=True)
            return k + 1
        if c == '"':
            return self.dquote_ps(i + 1)
        if c in "$@" and i + 1 < self.n and s[i + 1] in "({":
            op, cl = ("(", ")") if s[i + 1] == "(" else ("{", "}")
            text, j = self.capture(i + 2, op, cl)
            self.nested.append(("ps", text, op == "(" and self.count_suffix(j)))
            self.add_text(PLACEHOLDER)
            return j
        if c in "({":
            op, cl = ("(", ")") if c == "(" else ("{", "}")
            text, j = self.capture(i + 1, op, cl)
            self.nested.append(("ps", text, c == "(" and self.count_suffix(j)))
            self.add_text(PLACEHOLDER)
            return j
        if c in ")}":
            self.end_word()
            return i + 1
        if c == "|":
            if s.startswith("||", i):
                self.end_pipeline()
                return i + 2
            self.end_cmd()
            return i + 1
        if c == "&":
            if s.startswith("&&", i):
                self.end_pipeline()
                return i + 2
            if self.buf is None and not self.words:
                return i + 1                                            # the call operator
            self.end_pipeline()
            return i + 1
        if c == ";":
            self.end_pipeline()
            return i + 1
        if c == ",":
            self.end_word()
            self.comma_next = True
            return i + 1
        if c in "<>" or (c in "*123456" and self.buf is None and i + 1 < self.n and s[i + 1] == ">"):
            fd = None
            if c in "*123456":
                fd = c
                i += 1
            self.end_word()
            for op in (">>", ">&", ">", "<"):
                if s.startswith(op, i):
                    self.pending = Redir(op, fd)
                    return i + len(op)
        self.add(c)
        self.mark(c)
        return i + 1

    def dquote_ps(self, i):
        s = self.s
        self.add_text("", quoted=True)
        while i < self.n:
            c = s[i]
            if c == '"':
                if s.startswith('""', i):
                    self.buf.append('"')
                    i += 2
                    continue
                return i + 1
            if c == "`" and i + 1 < self.n:
                self.buf.append(s[i + 1])
                i += 2
                continue
            if c == "$" and i + 1 < self.n and s[i + 1] == "(":
                text, j = self.capture(i + 2)
                self.nested.append(("ps", text, False))
                self.buf.append(PLACEHOLDER)
                i = j
                continue
            self.buf.append(c)
            i += 1
        raise Unparsable("unterminated double quote")

    # -- cmd.exe
    def step_cmd(self, i):
        s = self.s
        c = s[i]
        if c in " \t\r":
            self.end_word()
            return i + 1
        if c == "\n":
            self.end_pipeline()
            return i + 1
        if c == "^":
            if i + 1 < self.n:
                self.add(s[i + 1], quoted=True)
            return i + 2
        if c == '"':
            j = s.find('"', i + 1)
            if j < 0:
                raise Unparsable("unterminated double quote")
            self.add_text(s[i + 1:j], quoted=True)
            return j + 1
        if c == "|":
            if s.startswith("||", i):
                self.end_pipeline()
                return i + 2
            self.end_cmd()
            return i + 1
        if c == "&":
            self.end_pipeline()
            return i + 2 if s.startswith("&&", i) else i + 1
        if c in "()":
            self.end_pipeline()
            return i + 1
        if c in "<>":
            fd = None
            if self.buf is not None and "".join(self.buf).isdigit():
                fd = "".join(self.buf)
                self.buf = None
            else:
                self.end_word()
            for op in (">>", ">&", "<&", ">", "<"):
                if s.startswith(op, i):
                    self.pending = Redir(op, fd)
                    return i + len(op)
        self.add(c)
        self.mark(c)
        return i + 1


# ----------------------------------------------------------------------------------------- options
def parse_opts(args, short_val=(), long_val=(), on_flag=None, on_val=None):
    """POSIX-style options. Returns the positional Words. on_flag(name), on_val(name, text)."""
    pos = []
    i = 0
    end = False
    n = len(args)
    while i < n:
        w = args[i]
        t = w.text
        if end or t == "-" or not t.startswith("-"):
            pos.append(w)
            i += 1
            continue
        if t == "--":
            end = True
            i += 1
            continue
        if t.startswith("--"):
            name, eq, val = t.partition("=")
            if name in long_val:
                if not eq:
                    val = args[i + 1].text if i + 1 < n else ""
                    i += 1
                if on_val:
                    on_val(name, val)
            elif on_flag:
                on_flag(name)
            i += 1
            continue
        j = 1
        while j < len(t):
            name = "-" + t[j]
            if name in short_val:
                rest = t[j + 1:]
                if rest:
                    val = rest
                else:
                    val = args[i + 1].text if i + 1 < n else ""
                    i += 1
                if on_val:
                    on_val(name, val)
                break
            if on_flag:
                on_flag(name)
            j += 1
        i += 1
    return pos


def ps_params(args, vals, switches, positional_names=(), aliases=None):
    """PowerShell parameters: (dict full-name -> list of Words or [None] for a switch, extra positionals)."""
    aliases = aliases or {}
    names = list(vals) + list(switches)
    params = {}
    pos = []
    i = 0
    n = len(args)

    def full(nm):
        nm = aliases.get(nm, nm)
        if nm in names:
            return nm
        hits = [x for x in names if x.startswith(nm)]
        return hits[0] if hits else nm

    def gather(first, k):
        vs = [first]
        while k < n and args[k].comma:
            vs.append(args[k])
            k += 1
        return vs, k

    while i < n:
        w = args[i]
        t = w.text
        if not w.quoted and re.match(r"^-[A-Za-z]", t):
            nm, colon, val = t[1:].partition(":")
            f = full(nm.lower())
            if f in vals:
                if colon:
                    vs, i = gather(Word(val), i + 1)
                elif i + 1 < n:
                    vs, i = gather(args[i + 1], i + 2)
                else:
                    vs, i = [], i + 1
                params.setdefault(f, []).extend(vs)
                continue
            if not (colon and val.lower() in ("$false", "0")):
                params.setdefault(f, []).append(None)
            i += 1
            continue
        vs, i = gather(w, i + 1)
        pos.append(vs)
    rest = []
    free = [x for x in positional_names if x not in params]
    for vs in pos:
        if free:
            params[free.pop(0)] = vs
        else:
            rest.append(vs)
    return params, rest


def cmd_name(text):
    base = re.split(r"[\\/]", text)[-1].lower()
    for ext in (".exe", ".cmd", ".bat", ".com"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


_MAGIC = re.compile(r"^:\(([^)]*)\)(.*)$")


def is_test_exclude(t):
    m = _MAGIC.match(t)
    if not m:
        return False
    words = {w.strip().lower() for w in m.group(1).split(",")}
    return {"exclude", "icase"} <= words and not words & {"glob", "literal"} and m.group(2) == "*test*"


def blob_path(t):
    """The path of a `<rev>:<path>`, `:<path>` or `:<n>:<path>` object name; None for anything else
    (a plain revision, `:/<text>` commit search, pathspec magic, a Windows drive path)."""
    if t.startswith(":"):
        m = re.match(r"^:(?:[0-3]:)?(?![(/!^])(.+)$", t)
        return m.group(1) if m else None
    if ":" in t and not re.match(r"^[A-Za-z]:[\\/]", t):
        return t.split(":", 1)[1]
    return None


def is_exclude_spec(t):
    if t.startswith((":!", ":^")):
        return True
    m = _MAGIC.match(t)
    return bool(m) and "exclude" in {w.strip().lower() for w in m.group(1).split(",")}


ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\+?=(.*)$", re.S)
PS_ASSIGN = re.compile(r"^\$([A-Za-z_]\w*)$")
KEYWORDS = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "!", "{", "}", "function"}
READERS = {"cat", "head", "tail", "less", "more", "most", "tac", "nl", "od", "xxd", "hexdump", "strings",
           "bat", "batcat", "wc", "cut", "sort", "uniq", "diff", "cmp", "comm", "paste", "join", "column",
           "fold", "fmt", "iconv", "base64", "cp", "rsync", "scp", "vi", "vim", "nvim", "view", "nano",
           "emacs", "code", "notepad", "type", "zcat", "gzcat", "bzcat", "xzcat", "zless", "zmore", "file",
           "md5sum", "sha1sum", "sha224sum", "sha384sum", "sha512sum", "b2sum", "cksum", "sum"}
READER_VAL = {"head": {"-n", "-c"}, "tail": {"-n", "-c", "-s"}, "cut": {"-d", "-f", "-c", "-b"},
              "sort": {"-k", "-t", "-o", "-S", "-T"}, "od": {"-A", "-t", "-N", "-j", "-w"},
              "xxd": {"-l", "-s", "-c", "-g"}, "hexdump": {"-n", "-s", "-e", "-f"}, "nl": {"-b", "-s", "-w"},
              "fold": {"-w"}, "fmt": {"-w"}, "iconv": {"-f", "-t"}, "base64": {"-w"}, "uniq": {"-f", "-s", "-w"},
              "bat": {"-l", "-r", "-H"}, "strings": {"-n", "-t", "-e"}, "paste": {"-d"}, "join": {"-t"},
              "less": {"-p", "-x", "-y"}, "diff": {"-x", "-X", "-I"}}
HASHES = {"sha256sum", "sha256"}
GREP_SHORT_VAL = {"-e", "-f", "-m", "-A", "-B", "-C", "-d", "-D", "-X"}
GREP_LONG_VAL = {"--regexp", "--file", "--max-count", "--after-context", "--before-context", "--context",
                 "--directories", "--devices", "--include", "--exclude", "--exclude-dir", "--exclude-from",
                 "--label", "--color", "--colour", "--binary-files", "--group-separator"}
RG_SHORT_VAL = {"-e", "-f", "-g", "-t", "-T", "-m", "-A", "-B", "-C", "-j", "-M", "-d", "-E", "-r"}
RG_LONG_VAL = {"--regexp", "--file", "--glob", "--iglob", "--type", "--type-not", "--max-count",
               "--after-context", "--before-context", "--context", "--threads", "--max-columns",
               "--max-depth", "--max-filesize", "--color", "--colors", "--encoding", "--pre", "--pre-glob",
               "--sort", "--sortr", "--type-add", "--type-clear", "--path-separator", "--context-separator",
               "--field-match-separator", "--field-context-separator", "--replace", "--ignore-file",
               "--engine", "--dfa-size-limit", "--regex-size-limit"}
AG_SHORT_VAL = {"-G", "-A", "-B", "-C", "-m", "-g", "-p", "-W"}
AG_LONG_VAL = {"--file-search-regex", "--ignore", "--ignore-dir", "--max-count", "--depth", "--path-to-ignore",
               "--pager", "--width", "--after", "--before", "--context"}
FIND_NAMEVAL = {"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename", "-regex", "-iregex",
                "-lname", "-ilname"}
FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir"}
FIND_FILE_OUT = {"-fprint", "-fprint0", "-fprintf", "-fls"}
GIT_GLOBAL_VAL = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix"}
BAD_PATHSPEC = {"--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs"}
BAD_PATHSPEC_ENV = ("GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS")
GIT_NAME_FLAGS = {"--stat", "--numstat", "--name-only", "--name-status", "--dirstat", "--raw", "--summary",
                  "--compact-summary", "--check"}
GIT_PATCH_FLAGS = {"-p", "-u", "--patch", "-W", "--function-context", "--cc", "-c", "--patch-with-stat",
                   "--patch-with-raw"}
GIT_DIFF_VAL = {"-S", "-G", "-n", "--max-count", "--skip", "--since", "--until", "--after", "--before",
                "--author", "--committer", "--grep", "--format", "--pretty", "--date", "-O", "--diff-filter",
                "--encoding", "--word-diff-regex", "--src-prefix", "--dst-prefix", "--line-prefix",
                "--output", "--decorate-refs", "--decorate-refs-exclude", "--abbrev", "-L"}
GIT_GREP_SHORT_VAL = {"-e", "-f", "-A", "-B", "-C", "-m", "-O"}
GIT_GREP_LONG_VAL = {"--max-count", "--max-depth", "--threads", "--after-context", "--before-context",
                     "--context", "--color", "--open-files-in-pager"}
PS_ALIASES = {
    "get-childitem": "gci", "gci": "gci", "ls": "gci", "dir": "gci",
    "get-item": "gi", "gi": "gi",
    "get-content": "gc", "gc": "gc", "cat": "gc", "type": "gc",
    "select-string": "sls", "sls": "sls",
    "remove-item": "ri", "ri": "ri", "rm": "ri", "rmdir": "ri", "del": "ri", "erase": "ri", "rd": "ri",
    "set-location": "cd", "sl": "cd", "cd": "cd", "chdir": "cd", "push-location": "cd", "pushd": "cd",
    "expand-archive": "expand", "get-filehash": "filehash",
    "measure-object": "measure", "measure": "measure",
    "tee-object": "tee", "tee": "tee",
    "out-file": "outfile", "set-content": "outfile", "add-content": "outfile",
    "select-object": "select", "select": "select",
    "foreach-object": "foreach", "%": "foreach", "foreach": "foreach",
    "where-object": "where", "?": "where", "where": "where", "sort-object": "where", "sort": "where",
    "invoke-expression": "iex", "iex": "iex",
    "out-string": "tostring",
}
PS_PATH_ALIASES = {"lp": "literalpath", "pspath": "literalpath"}


class Stage:
    __slots__ = ("cands", "is_filter", "filter_ps", "is_count", "tees", "redirects_out", "safe_source",
                 "fileinfo_out")

    def __init__(self):
        self.cands = []
        self.is_filter = False
        self.filter_ps = False
        self.is_count = False
        self.tees = False
        self.redirects_out = False
        self.safe_source = False
        self.fileinfo_out = False


# ----------------------------------------------------------------------------------------- analyser
class Analyzer:
    def __init__(self, ctx, subject):
        self.c = ctx
        self.subject = subject

    # -- candidates
    def deny(self, sg, rule, why=""):
        sg.cands.append((Finding(rule, self.subject, why), False, False, False))

    def listing(self, sg, rule, filterable, fileinfo=False):
        sg.cands.append((Finding(rule, self.subject), True, filterable, fileinfo))

    # -- helpers
    def res(self, text, st, d="bash"):
        return self.c.resolve(text, st, d)

    def sens(self, w, st, d="bash"):
        return self.c.sens(w.text if isinstance(w, Word) else w, st, d)

    def loc(self, w, st, d="bash"):
        """The location a path word names: its glob base for a glob."""
        t = w.text
        if w.glob or (d != "bash" and has_glob(t)):
            return self.res(glob_base(t), st, d)
        return self.res(t, st, d)

    def kind(self, w, st, d="bash"):
        t = w.text
        if w.glob or (d != "bash" and has_glob(t)):
            return "glob", self.res(glob_base(t), st, d)
        p = self.res(t, st, d)
        if p is UNKNOWN:
            return "unknown", p
        if self.c.is_dir(p) or t.endswith(("/", "\\")):
            return "dir", p
        if self.c.is_file(p):
            return "file", p
        # not on this machine (a pod's data folder, another OS): a name with an extension is a file
        return ("missing" if has_extension(t) else "dir"), p

    # -- entry points
    def script(self, text, dialect, st, depth, count_only=False, outer=None):
        """Analyse a command line. `outer` is the stage of an enclosing pipeline that runs this text
        (bash -c, cmd /c, pwsh -Command, eval, iex): a listing in the last pipeline that is not exempt here
        is handed to that stage, so the enclosing pipeline's filter or count can still exempt it."""
        if depth > MAX_DEPTH:
            raise Finding("B5", self.subject, "nesting deeper than 8")
        try:
            pipelines, nested = Lexer(text, dialect).run()
        except Unparsable as e:
            raise Finding("B5", self.subject, f"command could not be analysed ({e})")
        for k, pipe in enumerate(pipelines):
            self.pipeline(pipe, st, dialect, depth, count_only, outer if k == len(pipelines) - 1 else None)
        for d, t, cnt in nested:
            self.script(t, d, State(st.cwd, dict(st.vars)), depth + 1, count_only or cnt)

    def pipeline(self, cmds, st, dialect, depth, count_only, outer=None):
        stages = []
        for cmd in cmds:
            sg = Stage()
            upstream_safe = any(x.safe_source or x.is_filter for x in stages)
            self.stage(cmd, st, sg, dialect, depth, upstream_safe)
            stages.append(sg)
        last_count = bool(stages) and stages[-1].is_count
        for idx, sg in enumerate(stages):
            for cand in sg.cands:
                f, exemptable, filterable, fileinfo = cand
                if exemptable and self.exempt(stages, idx, filterable, fileinfo, count_only, last_count):
                    continue
                if exemptable and outer is not None and not any(s.tees or s.redirects_out for s in stages[idx:]):
                    outer.cands.append(cand)
                    continue
                raise f

    @staticmethod
    def exempt(stages, idx, filterable, fileinfo, count_only, last_count):
        if count_only:
            return True
        n = len(stages)
        if last_count and idx < n - 1 and not any(s.tees or s.redirects_out for s in stages[: n - 1]):
            return True
        if not filterable or stages[idx].redirects_out:
            return False
        fi = fileinfo
        for j in range(idx + 1, n):
            s = stages[j]
            if s.is_filter:
                return not (s.filter_ps and fi)
            if s.tees or s.redirects_out:
                return False
            if s.fileinfo_out is not None:
                fi = s.fileinfo_out
        return False

    # -- one simple command
    def stage(self, cmd, st, sg, d, depth, upstream_safe):
        for r in cmd.redirs:
            if r.op in ("<", "<>") and r.target is not None and self.sens(r.target, st, d):
                self.deny(sg, "B4")
            if r.op in (">", ">>", ">|", "&>", "&>>") and r.fd in (None, "1", "*"):
                sg.redirects_out = True
            if r.op == ">&" and r.fd in (None, "1") and r.target is not None and not r.target.text.isdigit():
                sg.redirects_out = True
        words = list(cmd.words)
        assigns = {}
        fed = False
        if d == "ps" and words:
            m = PS_ASSIGN.match(words[0].text)
            if len(words) >= 2 and m and words[1].text in ("=", "+="):
                val = words[2:]
                st.vars[m.group(1)] = (self.c.expand(val[0].text, st, d) or UNKNOWN) if len(val) == 1 else UNKNOWN
                words = val
                if not words:
                    return
        while True:
            while words and words[0].text in KEYWORDS and not words[0].quoted:
                words = words[1:]
            k = 0
            while k < len(words) and ASSIGN.match(words[k].text) and d != "ps":
                k += 1
            if k:
                pairs = [ASSIGN.match(w.text).groups() for w in words[:k]]
                if k == len(words):
                    for name, val in pairs:
                        v = self.c.expand(val, st, d)
                        st.vars[name] = UNKNOWN if v is None else v
                    return
                for name, val in pairs:
                    assigns[name] = val
                words = words[k:]
            if not words:
                return
            nw = self.unwrap(words, st, sg, d)
            if nw is None:
                return
            if nw is words:
                break
            if nw and nw[0] == "xargs":
                fed = True
                nw = nw[1:]
            words = nw
            if not words:
                return
        name = cmd_name(words[0].text)
        if d == "ps":
            sg.fileinfo_out = False
            alias = PS_ALIASES.get(name)
            if alias is not None:
                return self.ps_stage(alias, words, st, sg, depth, upstream_safe)
        return self.native(name, words, st, sg, d, depth, fed, upstream_safe, assigns)

    def unwrap(self, words, st, sg, d):
        """Strip one wrapper (sudo, env, xargs, ...). Returns words unchanged when there is none, None to stop."""
        name = cmd_name(words[0].text)
        args = words[1:]
        if name in ("sudo", "doas"):
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U") else 1
            return args[i:]
        if name == "env":
            i = 0
            while i < len(args) and (args[i].text.startswith("-") or ASSIGN.match(args[i].text)):
                if args[i].text in ("-u", "-C", "-S", "--unset", "--chdir", "--split-string"):
                    i += 1
                i += 1
            return args[i:]
        if name in ("nice", "ionice"):
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-n", "-c", "-p") else 1
            return args[i:]
        if name in ("nohup", "builtin", "unbuffer", "winpty"):
            return args
        if name == "exec":
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text == "-a" else 1
            return args[i:]
        if name == "command":
            if any(a.text in ("-v", "-V") for a in args[:2]):
                return None
            return [a for a in args if a.text != "-p"] if args and args[0].text == "-p" else args
        if name == "time":
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-f", "-o") else 1
            return args[i:]
        if name == "timeout":
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-s", "-k", "--signal", "--kill-after") else 1
            return args[i + 1:]
        if name == "stdbuf":
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-i", "-o", "-e") else 1
            return args[i:]
        if name in ("uv", "poetry", "pipenv", "pdm", "hatch") and args and args[0].text == "run":
            i = 1
            while i < len(args) and args[i].text.startswith("-"):
                i += 1
            return args[i:]
        if name == "conda" and args and args[0].text == "run":
            i = 1
            while i < len(args) and args[i].text.startswith("-"):
                i += 2 if args[i].text in ("-n", "-p", "--name", "--prefix") else 1
            return args[i:]
        if name == "xargs":
            i = 0
            while i < len(args) and args[i].text.startswith("-"):
                t = args[i].text
                if t in ("-a", "--arg-file") and i + 1 < len(args) and self.sens(args[i + 1], st, d):
                    self.deny(sg, "B4")
                if t.startswith("--arg-file=") and self.sens(t.split("=", 1)[1], st, d):
                    self.deny(sg, "B4")
                i += 2 if t in ("-n", "-L", "-P", "-s", "-d", "-E", "-I", "-a", "--arg-file") else 1
            rest = args[i:] or [Word("echo")]
            return ["xargs"] + rest
        return words

    # -- PowerShell cmdlets
    def ps_stage(self, alias, words, st, sg, depth, upstream_safe):
        args = words[1:]
        d = "ps"
        if alias == "gci":
            params, rest = ps_params(args, ["path", "literalpath", "filter", "include", "exclude", "depth",
                                            "attributes"],
                                     ["recurse", "name", "force", "directory", "file", "hidden", "readonly",
                                      "system", "followsymlink"], ["path", "filter"], PS_PATH_ALIASES)
            rec = "recurse" in params or "depth" in params
            nm = "name" in params
            sg.fileinfo_out = not nm
            for w in (params.get("filter") or []) + (params.get("include") or []):
                if w is not None and sensitive(w.text):
                    self.deny(sg, "B4")
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            return self.list_targets(sg, st, paths, rec, filterable=nm or not rec, d=d, fileinfo=not nm)
        if alias == "gi":
            params, _ = ps_params(args, ["path", "literalpath", "filter", "include", "exclude"], ["force"],
                                  ["path"], PS_PATH_ALIASES)
            sg.fileinfo_out = True
            for w in [x for x in (params.get("path") or []) + (params.get("literalpath") or []) if x is not None]:
                if self.sens(w, st, d):
                    self.deny(sg, "B4")
                elif has_glob(w.text) and self.c.protected(self.loc(w, st, d), False):
                    self.listing(sg, "B1", True, True)
            return
        if alias == "gc":
            params, _ = ps_params(args, ["path", "literalpath", "totalcount", "head", "first", "tail", "last",
                                         "readcount", "encoding", "delimiter", "filter", "include", "exclude",
                                         "stream"], ["raw", "wait", "force", "asbytestream"], ["path"],
                                  PS_PATH_ALIASES)
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            return self.read_paths(sg, st, paths, d)
        if alias == "sls":
            params, _ = ps_params(args, ["pattern", "path", "literalpath", "include", "exclude", "context",
                                         "encoding", "inputobject", "culture"],
                                  ["notmatch", "casesensitive", "simplematch", "list", "quiet", "allmatches",
                                   "raw", "noemphasis"], ["pattern", "path"], PS_PATH_ALIASES)
            pattern = [w.text.lower() for w in (params.get("pattern") or []) if w is not None]
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            sg.filter_ps = True
            sg.is_filter = ("notmatch" in params and "casesensitive" not in params and not paths
                            and "inputobject" not in params and pattern == ["test"])
            for w in (params.get("include") or []):
                if w is not None and sensitive(w.text):
                    self.deny(sg, "B4")
            for w in paths:
                if self.sens(w, st, d):
                    self.deny(sg, "B4")
                elif has_glob(w.text) and self.c.protected(self.loc(w, st, d), False):
                    self.listing(sg, "B1", True)
            return
        if alias == "ri":
            params, rest = ps_params(args, ["path", "literalpath", "filter", "include", "exclude", "stream"],
                                     ["recurse", "force", "confirm", "whatif"], ["path"], PS_PATH_ALIASES)
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            for vs in rest:
                paths.extend(vs)
            return self.remove_targets(sg, st, paths, d)
        if alias == "cd":
            params, _ = ps_params(args, ["path", "literalpath", "stackname"], ["passthru"], ["path"],
                                  PS_PATH_ALIASES)
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            st.cwd = self.res(paths[0].text, st, d) if paths else self.c.home_p()
            return
        if alias == "expand":
            params, _ = ps_params(args, ["path", "literalpath", "destinationpath"], ["force", "passthru"],
                                  ["path", "destinationpath"], PS_PATH_ALIASES)
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            for w in paths:
                self.archive(sg, st, w, "extract", d)
            return
        if alias == "filehash":
            params, _ = ps_params(args, ["path", "literalpath", "algorithm", "inputstream"], [], ["path"],
                                  PS_PATH_ALIASES)
            paths = [w for w in (params.get("path") or []) + (params.get("literalpath") or []) if w is not None]
            alg = [w.text.upper() for w in (params.get("algorithm") or []) if w is not None]
            if alg and alg != ["SHA256"]:
                return self.read_paths(sg, st, paths, d)
            return self.hash_paths(sg, st, paths, d)
        if alias == "measure":
            sg.is_count = True
            return
        if alias == "tee":
            sg.tees = True
            return
        if alias == "outfile":
            sg.redirects_out = True
            return
        if alias == "select":
            params, _ = ps_params(args, ["property", "expandproperty", "first", "last", "skip", "index",
                                         "exclude", "excludeproperty"], ["unique", "wait"], ["property"])
            sg.fileinfo_out = False if "expandproperty" in params else None
            return
        if alias == "foreach":
            sg.fileinfo_out = None if not args or args[0].text == PLACEHOLDER else False
            return
        if alias == "where":
            sg.fileinfo_out = None
            return
        if alias == "iex":
            self.script(" ".join(w.text for w in args), "ps", st, depth + 1, outer=sg)
            return
        return

    # -- native commands (bash, cmd, and executables called from PowerShell)
    def native(self, name, words, st, sg, d, depth, fed, upstream_safe, assigns):
        args = words[1:]
        argv0 = words[0].text
        if name in ("test", "[", "[["):
            return
        if name in ("cd", "pushd", "chdir"):
            target = [a for a in args if not a.text.startswith("-")]
            if not target:
                st.cwd = self.c.home_p()
            else:
                st.cwd = self.res(target[0].text, st, d)
            return
        if name == "popd":
            st.cwd = UNKNOWN
            return
        if name in ("export", "declare", "typeset", "local", "readonly"):
            for a in args:
                m = ASSIGN.match(a.text)
                if m:
                    v = self.c.expand(m.group(2), st, d)
                    st.vars[m.group(1)] = UNKNOWN if v is None else v
            return
        if name == "for":
            if len(args) >= 2 and args[1].text == "in":
                for w in args[2:]:
                    if self.sens(w, st, d):
                        self.deny(sg, "B4")
                    elif w.glob and self.c.protected(self.loc(w, st, d), False):
                        self.listing(sg, "B1", False)
            return
        if name in ("pytest", "py.test"):
            return self.deny(sg, "B3")
        if name in ("python", "python3", "py", "pythonw") or re.fullmatch(r"python3\.\d+", name):
            return self.python(args, st, sg, d)
        if name in ("bash", "sh", "zsh", "dash", "ksh", "ash"):
            return self.shell(args, st, sg, d, depth)
        if name == "eval":
            return self.script(" ".join(a.text for a in args), "bash", st, depth + 1, outer=sg)
        if name in ("source", "."):
            if args and self.sens(args[0], st, d):
                self.deny(sg, "B7")
            return
        if name in ("node", "perl", "ruby", "rscript", "deno", "bun", "php", "lua"):
            for a in args:
                if a.text in ("-e", "-E", "--eval", "-p", "-c"):
                    return
                if not a.text.startswith("-"):
                    if self.sens(a, st, d):
                        self.deny(sg, "B7")
                    return
            return
        if name == "cmd":
            for i, a in enumerate(args):
                if a.text.lower() in ("/c", "/k", "/r"):
                    return self.script(" ".join(x.text for x in args[i + 1:]), "cmd", st, depth + 1, outer=sg)
            return
        if name in ("powershell", "pwsh"):
            return self.powershell(args, st, sg, depth)
        if name in ("grep", "egrep", "fgrep", "zgrep", "zegrep", "zfgrep", "rgrep"):
            return self.grep(name, args, st, sg, d, fed, upstream_safe)
        if name in ("rg", "ag", "ack", "ack-grep", "ugrep"):
            return self.rg(name, args, st, sg, d, fed, upstream_safe)
        if name == "findstr":
            return self.findstr(args, st, sg, d)
        if name in ("find", "gfind", "bfs"):
            return self.find(args, st, sg, d)
        if name in ("fd", "fdfind"):
            return self.fd(args, st, sg, d)
        if name in ("locate", "plocate", "mlocate", "slocate"):
            for a in args:
                if not a.text.startswith("-") and sensitive(a.text):
                    self.deny(sg, "B4")
            return self.listing(sg, "B1", True)
        if name == "dir" and (d == "cmd" or any(re.fullmatch(r"/[A-Za-z?](:\S*)?", a.text) for a in args)):
            return self.cmd_dir(args, st, sg, d)
        if name in ("ls", "dir", "vdir", "exa", "eza", "lsd", "gls"):
            return self.ls(args, st, sg, d, fed, upstream_safe)
        if name == "tree":
            return self.tree(args, st, sg, d)
        if name == "du":
            return self.du(args, st, sg, d)
        if name in ("sed", "awk", "gawk", "mawk", "nawk", "jq", "yq"):
            return self.script_reader(name, args, st, sg, d, fed, upstream_safe)
        if name in HASHES:
            return self.hash_paths(sg, st, [a for a in args if not a.text.startswith("-")], d)
        if name == "shasum":
            alg = None
            for i, a in enumerate(args):
                if a.text in ("-a", "--algorithm") and i + 1 < len(args):
                    alg = args[i + 1].text
                elif a.text.startswith(("-a", "--algorithm=")) and a.text not in ("-a", "--algorithm"):
                    alg = a.text.split("=", 1)[-1] if "=" in a.text else a.text[2:]
            files = parse_opts(args, {"-a"}, {"--algorithm"})
            if alg == "256":
                return self.hash_paths(sg, st, files, d)
            return self.read_paths(sg, st, files, d, fed=fed, upstream_safe=upstream_safe)
        if name == "certutil":
            low = [a.text.lower() for a in args]
            if "-hashfile" in low:
                i = low.index("-hashfile")
                files = args[i + 1:i + 2]
                if i + 2 < len(args) and args[i + 2].text.upper() == "SHA256":
                    return self.hash_paths(sg, st, files, d)
                return self.read_paths(sg, st, files, d)
            return
        if name == "openssl":
            low = [a.text.lower() for a in args]
            if low and (low[0] == "sha256" or (low[0] == "dgst" and "-sha256" in low)):
                return self.hash_paths(sg, st, [a for a in args[1:] if not a.text.startswith("-")], d)
            return
        if name in READERS:
            return self.reader(name, args, st, sg, d, fed, upstream_safe)
        if name in ("echo", "printf", "stat"):
            for w in args:
                if name == "stat" and not w.text.startswith("-") and self.sens(w, st, d):
                    self.deny(sg, "B4")
                elif w.glob and self.c.protected(self.loc(w, st, d), False):
                    self.listing(sg, "B1", True)                  # shell expansion prints the names
            return
        if name == "tee":
            sg.tees = True
            return
        if name in ("tar", "bsdtar"):
            return self.tar(args, st, sg, d)
        if name in ("unzip", "zipinfo", "7z", "7za", "7zr", "unrar", "lsar", "unar"):
            return self.unpack(name, args, st, sg, d)
        if name == "git":
            return self.git(args, st, sg, d, fed, upstream_safe, assigns)
        if name == "docker":
            return self.docker(args, sg)
        if name in ("rm", "rmdir", "unlink", "shred", "del", "erase", "rd"):
            cmdstyle = d == "cmd" or name in ("del", "erase", "rd")
            targets = []
            end = False
            for a in args:
                t = a.text
                if not end and t == "--":
                    end = True
                    continue
                if not end and t.startswith("-") and len(t) > 1:
                    continue
                if cmdstyle and re.fullmatch(r"/[A-Za-z](:\S*)?", t):
                    continue
                targets.append(a)
            if fed and not targets:
                return self.deny(sg, "D2")
            return self.remove_targets(sg, st, targets, d)
        if ("/" in argv0 or "\\" in argv0) and sensitive(argv0):
            return self.deny(sg, "B7")
        return

    # -- interpreters
    def python(self, args, st, sg, d):
        i = 0
        while i < len(args):
            t = args[i].text
            if t == "-m" or (t.startswith("-m") and len(t) > 2):
                mod = t[2:] if len(t) > 2 else (args[i + 1].text if i + 1 < len(args) else "")
                rest = args[i + 1:] if len(t) > 2 else args[i + 2:]
                if mod in ("pytest", "py.test", "unittest"):
                    return self.deny(sg, "B3")
                if mod in ("zipfile", "tarfile"):
                    mode = None
                    for k, a in enumerate(rest):
                        if a.text in ("-l", "--list", "-t", "--test", "-e", "--extract"):
                            mode = "list" if a.text in ("-l", "--list", "-t", "--test") else "extract"
                            if k + 1 < len(rest):
                                self.archive(sg, st, rest[k + 1], mode, d)
                            return
                return
            if t == "-c" or (t.startswith("-c") and len(t) > 2):
                return                                                  # inline code: not analysed
            if t in ("-W", "-X", "-Q"):
                i += 2
                continue
            if t.startswith("-"):
                i += 1
                continue
            if self.sens(args[i], st, d):
                self.deny(sg, "B7")
            return

    def shell(self, args, st, sg, d, depth):
        i = 0
        while i < len(args):
            t = args[i].text
            if t.startswith("-") and not t.startswith("--") and "c" in t[1:]:
                text = args[i + 1].text if i + 1 < len(args) else ""
                return self.script(text, "bash", st, depth + 1, outer=sg)
            if t in ("-o", "-O", "+o", "+O"):
                i += 2
                continue
            if t.startswith(("-", "+")):
                i += 1
                continue
            if self.sens(args[i], st, d):
                self.deny(sg, "B7")
            return

    def powershell(self, args, st, sg, depth):
        i = 0
        while i < len(args):
            t = args[i].text
            low = t.lower()
            if low.startswith("-") and len(low) > 1:
                nm = low[1:]
                if nm in ("enc", "e", "ec") or "encodedcommand".startswith(nm) and len(nm) >= 2 and nm.startswith("en"):
                    return self.deny(sg, "B5", "encoded command")
                if nm in ("c", "command", "cmd") or ("command".startswith(nm) and len(nm) >= 3):
                    return self.script(" ".join(x.text for x in args[i + 1:]), "ps", st, depth + 1, outer=sg)
                if nm in ("f", "file") or ("file".startswith(nm) and len(nm) >= 2):
                    if i + 1 < len(args) and self.sens(args[i + 1], st, "ps"):
                        self.deny(sg, "B7")
                    return
                if nm in ("executionpolicy", "ep", "windowstyle", "w", "workingdirectory", "wd",
                          "inputformat", "outputformat", "configurationname", "version", "v", "settingsfile"):
                    i += 2
                    continue
                i += 1
                continue
            if low.endswith(".ps1"):
                if self.sens(args[i], st, "ps"):
                    self.deny(sg, "B7")
                return
            return self.script(" ".join(x.text for x in args[i:]), "ps", st, depth + 1, outer=sg)

    # -- searches and listings
    def list_targets(self, sg, st, paths, rec, filterable, d, fileinfo=False, dir_self=False):
        """A listing of `paths` (or the cwd when there are none)."""
        if not paths:
            if st.cwd is not UNKNOWN and sensitive(st.cwd.key):
                return self.deny(sg, "B4")
            if self.c.protected(st.cwd, rec):
                self.listing(sg, "B1", filterable, fileinfo)
            return
        for w in paths:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
                continue
            kind, p = self.kind(w, st, d)
            if kind in ("file", "missing"):
                continue
            if dir_self and not rec and kind in ("dir", "unknown"):
                continue
            if self.c.protected(p, rec):
                # a glob that matches a directory lists its members under a header: not filterable
                self.listing(sg, "B1", filterable and (kind != "glob" or dir_self), fileinfo)

    def grep(self, name, args, st, sg, d, fed, upstream_safe):
        o = {"rec": name == "rgrep", "nofn": False, "count": False, "inv": False, "icase": False,
             "word": False, "line": False, "given": False}
        pats = []
        files = []

        def flag(f):
            o["rec"] |= f in ("-r", "-R", "--recursive", "--dereference-recursive")
            o["nofn"] |= f in ("-h", "--no-filename")
            o["count"] |= f in ("-c", "--count")
            o["inv"] |= f in ("-v", "--invert-match")
            o["icase"] |= f in ("-i", "-y", "--ignore-case")
            o["word"] |= f in ("-w", "--word-regexp")
            o["line"] |= f in ("-x", "--line-regexp")

        def val(nm, v):
            if nm in ("-e", "--regexp"):
                o["given"] = True
                pats.append(v)
            elif nm in ("-f", "--file"):
                o["given"] = True
                files.append(v)
            elif nm in ("-d", "--directories") and v == "recurse":
                o["rec"] = True
            elif nm == "--include":
                files.append(v)

        pos = parse_opts(args, GREP_SHORT_VAL, GREP_LONG_VAL, flag, val)
        if not o["given"] and pos:
            pats.append(pos[0].text)
            pos = pos[1:]
        sg.is_filter = (o["inv"] and o["icase"] and not o["word"] and not o["line"] and not o["count"]
                        and not o["rec"] and not pos and not files and [p.lower() for p in pats] == ["test"])
        sg.is_count = o["count"] and not pos and not o["rec"] and not files
        for f in files:
            if self.c.sens(f, st, d):
                self.deny(sg, "B4")
        filterable = not o["nofn"]
        if not pos:
            if o["rec"]:
                if self.c.protected(st.cwd, True):
                    self.listing(sg, "B1", filterable)
            elif fed and not upstream_safe:
                self.listing(sg, "B1", filterable)
            return
        for w in pos:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
                continue
            kind, p = self.kind(w, st, d)
            if kind in ("file", "missing"):
                continue
            if self.c.protected(p, o["rec"]):
                self.listing(sg, "B1", filterable)

    def rg(self, name, args, st, sg, d, fed, upstream_safe):
        o = {"nofn": False, "pretty": False, "files": False, "given": False}
        globs = []
        is_rg = name == "rg"

        def flag(f):
            o["nofn"] |= f in ("-I", "--no-filename", "--nofilename") or (name.startswith("ack") and f == "-h")
            o["pretty"] |= f in ("--heading", "--group") or (is_rg and f in ("-p", "--pretty"))
            o["files"] |= f == "--files" or (name.startswith("ack") and f == "-f")

        def val(nm, v):
            if nm in ("-e", "--regexp", "-f", "--file"):
                o["given"] = True
                if nm in ("-f", "--file"):
                    globs.append(v)
            elif nm in ("-g", "--glob", "--iglob", "-G", "--file-search-regex"):
                if not is_rg and nm == "-g":
                    o["files"] = True
                globs.append(v)

        sv, lv = (RG_SHORT_VAL, RG_LONG_VAL) if is_rg else (AG_SHORT_VAL, AG_LONG_VAL)
        pos = parse_opts(args, sv, lv, flag, val)
        for g in globs:
            if not g.startswith("!") and sensitive(g):
                self.deny(sg, "B4")
        if not o["files"] and not o["given"] and pos:
            pos = pos[1:]
        filterable = not (o["nofn"] or o["pretty"])
        if not pos:
            if self.c.protected(st.cwd, True):
                self.listing(sg, "B1", filterable)
            return
        for w in pos:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
                continue
            kind, p = self.kind(w, st, d)
            if kind in ("file", "missing"):
                continue
            if self.c.protected(p, True):
                self.listing(sg, "B1", filterable)

    def findstr(self, args, st, sg, d):
        rec = given = False
        pos = []
        for a in args:
            t = a.text
            if re.match(r"^/[A-Za-z](:|$)", t):
                letter = t[1].lower()
                if letter == "s":
                    rec = True
                elif letter in ("c", "g"):
                    given = True
                if letter in ("g", "f") and len(t) > 3 and self.c.sens(t[3:], st, d):
                    self.deny(sg, "B4")
                if letter == "d" and len(t) > 3:
                    for dd in t[3:].split(";"):
                        if self.c.protected(self.res(dd, st, d), True):
                            self.listing(sg, "B1", True)
                continue
            pos.append(a)
        if not given and pos:
            pos = pos[1:]
        for w in pos:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
                continue
            kind, p = self.kind(w, st, d)
            if kind == "glob" or (rec and kind in ("file", "missing")):
                base = p if kind == "glob" else self.res(glob_base(posixpath.dirname(w.text.replace("\\", "/")) or "."), st, d)
                if self.c.protected(base, rec):
                    self.listing(sg, "B1", True)
            elif kind in ("dir", "unknown") and self.c.protected(p, rec):
                self.listing(sg, "B1", True)

    def find(self, args, st, sg, d):
        i = 0
        while i < len(args) and (args[i].text in ("-H", "-L", "-P") or args[i].text in ("-D", "-O")
                                 or re.fullmatch(r"-O\d", args[i].text)):
            i += 2 if args[i].text in ("-D", "-O") else 1
        starts = []
        while i < len(args) and not (args[i].text.startswith("-") or args[i].text in ("(", "!", ")", ",")):
            starts.append(args[i])
            i += 1
        exe = delete = fout = False
        rest = args[i:]
        prev = ""
        k = 0
        while k < len(rest):
            t = rest[k].text
            if t in FIND_NAMEVAL and k + 1 < len(rest):
                if prev not in ("!", "-not") and sensitive(rest[k + 1].text):
                    self.deny(sg, "B4")
                prev = t
                k += 2
                continue
            if t in FIND_EXEC:
                exe = True
                k += 1
                while k < len(rest) and rest[k].text not in (";", "+"):
                    k += 1
            elif t == "-delete":
                delete = True
            elif t in FIND_FILE_OUT:
                fout = True
                k += 1
            prev = t
            k += 1
        for w in starts:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
        targets = [self.loc(w, st, d) for w in starts if not self.sens(w, st, d)] if starts else [st.cwd]
        if delete and any(self.c.protected(p, True) for p in targets):
            return self.deny(sg, "D2")
        for p in targets:
            if self.c.protected(p, True):
                self.listing(sg, "B1", not (exe or fout))

    def fd(self, args, st, sg, d):
        exe = False
        cut = len(args)
        for i, a in enumerate(args):
            if a.text in ("-x", "--exec", "-X", "--exec-batch"):
                exe = True
                cut = i
                break
        pos = parse_opts(args[:cut], {"-e", "-t", "-E", "-d", "-c", "-j", "-S", "-o"},
                         {"--extension", "--type", "--exclude", "--max-depth", "--min-depth", "--color",
                          "--threads", "--size", "--owner", "--changed-within", "--changed-before",
                          "--base-directory", "--search-path"})
        if pos and sensitive(pos[0].text):
            self.deny(sg, "B4")
        paths = pos[1:]
        for w in paths:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
        targets = [self.loc(w, st, d) for w in paths] or [st.cwd]
        for p in targets:
            if self.c.protected(p, True):
                self.listing(sg, "B1", not exe)

    def ls(self, args, st, sg, d, fed, upstream_safe):
        o = {"R": False, "d": False}

        def flag(f):
            o["R"] |= f in ("-R", "--recursive")
            o["d"] |= f in ("-d", "--directory")

        pos = parse_opts(args, {"-I", "-w", "-T"}, {"--ignore", "--width", "--tabsize", "--hide", "--format",
                                                    "--sort", "--time", "--time-style", "--color",
                                                    "--indicator-style", "--quoting-style", "--block-size"},
                         flag)
        if not pos and fed:
            if not upstream_safe:
                self.listing(sg, "B1", not o["R"])
            return
        return self.list_targets(sg, st, pos, o["R"], filterable=not o["R"], d=d, dir_self=o["d"])

    def cmd_dir(self, args, st, sg, d):
        rec = bare = False
        pos = []
        for a in args:
            t = a.text
            if re.fullmatch(r"/-?[A-Za-z?](:\S*)?", t):
                letter = t.lstrip("/-")[:1].lower()
                rec |= letter == "s"
                bare |= letter == "b"
                continue
            pos.append(a)
        return self.list_targets(sg, st, pos, rec, filterable=bare or not rec, d=d)

    def tree(self, args, st, sg, d):
        cmdstyle = full = False
        rest = []
        for a in args:
            if re.fullmatch(r"/[A-Za-z]", a.text):
                cmdstyle = True
            else:
                rest.append(a)
        o = {"f": False}

        def flag(f):
            o["f"] |= f == "-f"

        def val(nm, v):
            if nm == "-P" and sensitive(v):
                self.deny(sg, "B4")

        pos = parse_opts(rest, {"-L", "-P", "-I", "-o", "-H", "-T"},
                         {"--filelimit", "--charset", "--sort", "--timefmt"}, flag, val)
        full = o["f"] and not cmdstyle
        paths = pos
        if not paths:
            if self.c.protected(st.cwd, True):
                self.listing(sg, "B1", full)
            return
        for w in paths:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
                continue
            kind, p = self.kind(w, st, d)
            if kind in ("dir", "unknown", "glob") and self.c.protected(p, True):
                self.listing(sg, "B1", full)

    def du(self, args, st, sg, d):
        o = {"s": False}

        def flag(f):
            o["s"] |= f in ("-s", "--summarize")

        def val(nm, v):
            if nm in ("-d", "--max-depth") and v.strip() == "0":
                o["s"] = True

        pos = parse_opts(args, {"-d", "-B", "-t", "-X"},
                         {"--max-depth", "--block-size", "--threshold", "--exclude", "--exclude-from",
                          "--time-style", "--files0-from"}, flag, val)
        for w in pos:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
            elif w.glob and self.c.protected(self.loc(w, st, d), False):
                self.listing(sg, "B1", True)                      # the glob itself lists the folder
        if o["s"]:
            return
        targets = [self.loc(w, st, d) for w in pos if not self.sens(w, st, d)] if pos else [st.cwd]
        for p in targets:
            if self.c.protected(p, True):
                self.listing(sg, "B1", True)

    # -- readers
    def read_paths(self, sg, st, paths, d, fed=False, upstream_safe=False):
        for w in paths:
            if self.sens(w, st, d):
                self.deny(sg, "B4")
            elif (w.glob or (d != "bash" and has_glob(w.text))) and self.c.protected(self.loc(w, st, d), False):
                self.listing(sg, "B1", False)
        if not paths and fed and not upstream_safe:
            self.listing(sg, "B1", False)

    def hash_paths(self, sg, st, paths, d):
        """Whole-file sha256: allowed on any path. A glob over a protected location is still a listing."""
        for w in paths:
            if (w.glob or (d != "bash" and has_glob(w.text))) and not self.sens(w, st, d) \
                    and self.c.protected(self.loc(w, st, d), False):
                self.listing(sg, "B1", True)

    def reader(self, name, args, st, sg, d, fed, upstream_safe):
        o = {"r": False, "l_only": True, "any": False}

        def flag(f):
            o["any"] = True
            if name == "diff" and f in ("-r", "--recursive"):
                o["r"] = True
            if name == "wc" and f not in ("-l", "--lines"):
                o["l_only"] = False

        def val(nm, v):
            o["any"] = True
            o["l_only"] = False

        pos = parse_opts(args, READER_VAL.get(name, set()), set(), flag, val)
        if name == "wc":
            sg.is_count = o["any"] and o["l_only"] and not pos
        if name in ("cp", "rsync", "scp") and len(pos) >= 2:
            pos = pos[:-1]
        if name == "diff" and o["r"]:
            for w in pos:
                if self.sens(w, st, d):
                    self.deny(sg, "B4")
                elif self.c.protected(self.loc(w, st, d), True):
                    self.listing(sg, "B1", False)
            return
        return self.read_paths(sg, st, pos, d, fed, upstream_safe)

    def script_reader(self, name, args, st, sg, d, fed, upstream_safe):
        o = {"given": False}
        files = []
        if name == "sed":
            sv, lv, script_opts, file_opts = {"-e", "-f", "-l"}, {"--expression", "--file", "--line-length"}, \
                {"-e", "--expression"}, {"-f", "--file"}
        elif name == "jq" or name == "yq":
            sv, lv, script_opts, file_opts = {"-f", "-L"}, {"--from-file", "--indent", "--arg", "--argjson",
                                                            "--slurpfile", "--rawfile"}, set(), {"-f", "--from-file"}
        else:
            sv, lv, script_opts, file_opts = {"-f", "-v", "-F"}, {"--file", "--assign", "--field-separator"}, \
                set(), {"-f", "--file"}

        def val(nm, v):
            if nm in script_opts:
                o["given"] = True
            if nm in file_opts:
                o["given"] = True
                files.append(Word(v))

        pos = parse_opts(args, sv, lv, None, val)
        if not o["given"] and pos:
            pos = pos[1:]
        if name in ("awk", "gawk", "mawk", "nawk"):
            pos = [w for w in pos if not ASSIGN.match(w.text)]
        return self.read_paths(sg, st, pos + files, d, fed, upstream_safe)

    # -- archives
    def archive(self, sg, st, w, mode, d):
        if w is None or mode not in ("list", "extract", "test"):
            return
        if self.sens(w, st, d):
            return self.deny(sg, "B8")
        if mode in ("list", "test") and self.c.protected(self.loc(w, st, d), False):
            self.listing(sg, "B1", True)

    def tar(self, args, st, sg, d):
        mode = None
        archive = None
        i = 0

        def setmode(ch):
            nonlocal mode
            if ch == "t":
                mode = "list"
            elif ch == "x":
                mode = "extract"
            elif ch in "cruA" and mode is None:
                mode = "create"

        if args and not args[0].text.startswith("-"):
            pend = False
            for ch in args[0].text:
                setmode(ch)
                pend |= ch == "f"
            i = 1
            if pend and i < len(args):
                archive = args[i]
                i += 1
        while i < len(args):
            t = args[i].text
            if t.startswith("--"):
                nm, eq, v = t.partition("=")
                if nm == "--list":
                    mode = "list"
                elif nm in ("--extract", "--get"):
                    mode = "extract"
                elif nm in ("--create", "--append", "--update", "--concatenate", "--catenate"):
                    mode = mode or "create"
                elif nm == "--file":
                    if eq:
                        archive = Word(v)
                    elif i + 1 < len(args):
                        archive = args[i + 1]
                        i += 1
                elif nm in ("--directory", "--files-from", "--exclude-from", "--transform", "--use-compress-program") and not eq:
                    i += 1
            elif t.startswith("-") and len(t) > 1:
                j = 1
                while j < len(t):
                    ch = t[j]
                    if ch == "f":
                        if t[j + 1:]:
                            archive = Word(t[j + 1:])
                        elif i + 1 < len(args):
                            archive = args[i + 1]
                            i += 1
                        break
                    if ch in "CTXbHKNgV":
                        if not t[j + 1:]:
                            i += 1
                        break
                    setmode(ch)
                    j += 1
            i += 1
        self.archive(sg, st, archive, mode, d)

    def unpack(self, name, args, st, sg, d):
        if name in ("zipinfo", "lsar"):
            pos = [a for a in args if not a.text.startswith("-")]
            return self.archive(sg, st, pos[0] if pos else None, "list", d)
        if name == "unar":
            pos = [a for a in args if not a.text.startswith("-")]
            return self.archive(sg, st, pos[0] if pos else None, "extract", d)
        if name == "unzip":
            o = {"list": False}

            def flag(f):
                o["list"] |= f in ("-l", "-v", "-Z", "-t")

            cut = next((i for i, a in enumerate(args) if a.text == "-x"), len(args))
            pos = parse_opts(args[:cut], {"-d", "-P"}, set(), flag)
            return self.archive(sg, st, pos[0] if pos else None, "list" if o["list"] else "extract", d)
        pos = [a for a in args if not a.text.startswith("-")]
        if len(pos) < 2:
            return
        cmd = pos[0].text.lower()
        if name in ("7z", "7za", "7zr"):
            mode = {"l": "list", "t": "test", "x": "extract", "e": "extract"}.get(cmd)
        else:
            mode = "list" if cmd in ("l", "lt", "lb", "v", "vt", "vb") else \
                "test" if cmd == "t" else "extract" if cmd in ("x", "e", "p") else None
        self.archive(sg, st, pos[1], mode, d)

    # -- removal
    def remove_targets(self, sg, st, targets, d):
        for w in targets:
            if self.c.protected(self.loc(w, st, d), True):
                return self.deny(sg, "D2")

    # -- docker
    def docker(self, args, sg):
        i = 0
        while i < len(args) and args[i].text.startswith("-"):
            t = args[i].text
            i += 2 if t in ("--config", "-c", "--context", "-H", "--host", "-l", "--log-level", "--tlscacert",
                            "--tlscert", "--tlskey") else 1
        sub = [a.text.lower() for a in args[i:i + 2]]
        if not sub:
            return
        if sub[0] == "rmi":
            return self.deny(sg, "D1")
        if len(sub) == 2 and ((sub[0] == "image" and sub[1] in ("rm", "remove", "prune"))
                              or (sub[0] == "system" and sub[1] == "prune")
                              or (sub[0] == "volume" and sub[1] in ("prune", "rm", "remove"))):
            return self.deny(sg, "D1")

    # -- git
    def git(self, args, st, sg, d, fed, upstream_safe, assigns):
        cwd = st.cwd
        bad = False
        i = 0
        while i < len(args) and args[i].text.startswith("-"):
            t = args[i].text
            nm, eq, _ = t.partition("=")
            if nm in BAD_PATHSPEC:
                bad = True
            if nm == "-C" and i + 1 < len(args):
                cwd = self.res(args[i + 1].text, State(cwd, st.vars), d)
                i += 2
                continue
            if nm in GIT_GLOBAL_VAL and not eq:
                i += 2
                continue
            i += 1
        if i >= len(args):
            return
        for e in BAD_PATHSPEC_ENV:
            v = assigns.get(e)
            if v is None:
                v = st.vars.get(e)
            if v is None:
                v = self.c.getenv(e)
            if v not in (None, "", "0", "false") and v is not UNKNOWN:
                bad = True
        sub = args[i].text.lower()
        rest = args[i + 1:]
        gst = State(cwd, st.vars)
        if sub == "grep":
            return self.git_grep(rest, gst, sg, d, bad)
        if sub == "ls-files":
            specs = parse_opts(rest, {"-x", "-X"}, {"--exclude", "--exclude-from", "--exclude-per-directory",
                                                    "--with-tree", "--format", "--abbrev"})
            sens, covered = self.git_scope(specs, gst, d, bad)
            if sens:
                return self.deny(sg, "B4")
            if covered:
                return self.listing(sg, "B2", True)
            sg.safe_source = True
            return
        if sub == "ls-tree":
            o = {"r": False}

            def flag(f):
                o["r"] |= f == "-r"

            pos = parse_opts(rest, set(), {"--format", "--abbrev"}, flag)
            if not pos:
                return
            treeish, paths = pos[0], pos[1:]
            listing = o["r"] or not paths
            if ":" in treeish.text and not treeish.text.startswith(":"):
                path = treeish.text.split(":", 1)[1]
                if sensitive(path):
                    return self.deny(sg, "B4")
                listing = True
            for w in paths:
                if self.sens(w, gst, d):
                    return self.deny(sg, "B4")
                kind, _ = self.kind(w, gst, d)
                if kind in ("dir", "glob", "unknown") or has_glob(w.text):
                    listing = True
            if listing:
                self.listing(sg, "B2", True)
            return
        if sub in ("diff", "show", "log", "whatchanged", "diff-tree", "diff-index", "diff-files", "stash"):
            return self.git_diffish(sub, rest, gst, sg, d, bad)
        if sub == "cat-file":
            return self.git_cat_file(rest, gst, sg, d)
        if sub == "archive":
            pos = parse_opts(rest, {"-o"}, {"--format", "--output", "--prefix", "--remote", "--exec"})
            if not pos:
                return
            sens, covered = self.git_scope(pos[1:], gst, d, bad)
            if sens:
                return self.deny(sg, "B4")
            if covered:
                return self.deny(sg, "B6")
            return
        if sub in ("blame", "annotate"):
            for w in parse_opts(rest, {"-L", "-S", "-C", "-M"}, {"--contents", "--since", "--encoding"}):
                if self.sens(w, gst, d):
                    return self.deny(sg, "B4")
            return
        if sub == "clean":
            dry = any(a.text in ("--dry-run",) or (a.text.startswith("-") and not a.text.startswith("--")
                                                   and "n" in a.text[1:]) for a in rest)
            if not dry and self.c.protected(cwd, False):
                return self.deny(sg, "D2")
            return
        return

    def git_scope(self, specs, st, d, bad):
        """(a positive pathspec is test-named or NAMED_DENY, the scope can include test paths)."""
        has_excl = not bad and any(is_test_exclude(w.text) for w in specs)
        sens = False
        covered = False
        positive = 0
        for w in specs:
            t = w.text
            if is_exclude_spec(t):
                continue
            positive += 1
            if t.startswith(":"):
                m = _MAGIC.match(t)
                body = m.group(2) if m else t.lstrip(":/!^")
                if sensitive(body):
                    sens = True
                covered = True
                continue
            if self.sens(w, st, d):
                sens = True
                continue
            if has_glob(t) or t in (".", "./") or t.endswith(("/", "\\")):
                covered = True
                continue
            p = self.res(t, st, d)
            if p is UNKNOWN or self.c.is_dir(p) or (not self.c.is_file(p) and not has_extension(t)):
                covered = True
        if positive == 0:
            covered = True
        if has_excl:
            covered = False
        return sens, covered

    def git_grep(self, rest, st, sg, d, bad):
        o = {"nofn": False, "heading": False, "given": False}
        files = []

        def flag(f):
            o["nofn"] |= f in ("-h", "--no-filename")
            o["heading"] |= f == "--heading"

        def val(nm, v):
            if nm in ("-e", "-f"):
                o["given"] = True
            if nm == "-f":
                files.append(v)

        split = next((k for k, a in enumerate(rest) if a.text == "--"), None)
        before = rest if split is None else rest[:split]
        after = [] if split is None else rest[split + 1:]
        pos = parse_opts(before, GIT_GREP_SHORT_VAL, GIT_GREP_LONG_VAL, flag, val)
        if not o["given"] and pos:
            pos = pos[1:]
        specs = list(after)
        for w in pos:                       # revisions or paths before `--`
            if self.sens(w, st, d) or self.c.is_dir(self.res(w.text, st, d)) or \
                    self.c.is_file(self.res(w.text, st, d)) or is_exclude_spec(w.text) or w.text.startswith(":("):
                specs.append(w)
        for f in files:
            if self.c.sens(f, st, d):
                self.deny(sg, "B4")
        sens, covered = self.git_scope(specs, st, d, bad)
        if sens:
            return self.deny(sg, "B4")
        if covered:
            return self.listing(sg, "B2", not (o["nofn"] or o["heading"]))
        sg.safe_source = True

    def git_diffish(self, sub, rest, st, sg, d, bad):
        default = {"diff": "patch", "show": "patch", "log": "none", "whatchanged": "names", "diff-tree": "names",
                   "diff-index": "names", "diff-files": "names", "stash": "names"}[sub]
        if sub == "stash":
            if not rest or rest[0].text != "show":
                return
            rest = rest[1:]
        names = patch = quiet = False
        specs = []
        before = []
        end = False
        i = 0
        while i < len(rest):
            w = rest[i]
            t = w.text
            if end:
                specs.append(w)
                i += 1
                continue
            if t == "--":
                end = True
                i += 1
                continue
            if t.startswith("-") and t != "-":
                nm, eq, _ = t.partition("=")
                if nm in GIT_NAME_FLAGS:
                    names = True
                if nm in GIT_PATCH_FLAGS or nm.startswith(("-U", "--unified", "--word-diff", "--color-words")) \
                        or nm.startswith("-L"):
                    patch = True
                if nm in ("--quiet", "-s", "--no-patch"):
                    quiet = True
                if nm in GIT_DIFF_VAL and not eq:
                    i += 1
                i += 1
                continue
            before.append(w)
            i += 1
        revpaths = []
        plain_revs = 0
        for w in before:
            t = w.text
            bp = blob_path(t)
            if bp is not None:                          # one blob (or tree) object: <rev>:<path>, :<path>
                revpaths.append(bp)
            elif self.sens(w, st, d):
                specs.append(w)
            else:
                p = self.res(t, st, d)
                if p is not UNKNOWN and (self.c.is_dir(p) or self.c.is_file(p)):
                    specs.append(w)
                else:
                    plain_revs += 1
        root = State(self.c.project or st.cwd, st.vars)
        for path in revpaths:
            if sensitive(path):
                return self.deny(sg, "B4")
            if path in ("", ".", "./") or path.endswith("/") or self.c.is_dir(self.res(path, root, d)):
                self.listing(sg, "B2", True)
        if quiet and not names and not patch:
            return
        out = "patch" if patch else "names" if names else default
        if out == "none":
            return
        if sub == "show" and revpaths and not plain_revs and not specs:
            return
        sens, covered = self.git_scope(specs, st, d, bad)
        if sens:
            return self.deny(sg, "B4")
        if covered:
            if out == "patch":
                return self.deny(sg, "B6")
            return self.listing(sg, "B2", True)

    def git_cat_file(self, rest, st, sg, d):
        content = False
        objs = []
        for a in rest:
            t = a.text
            if t in ("-p", "--textconv", "--filters") or t in ("blob", "tree", "commit", "tag"):
                content = True
                if t == "tree":
                    self.listing(sg, "B2", True)
                continue
            if t.startswith("-"):
                continue
            objs.append(a)
        if not content:
            return
        root = State(self.c.project or st.cwd, st.vars)
        for w in objs:
            t = w.text
            if ":" in t and not t.startswith(":"):
                path = t.split(":", 1)[1]
                if sensitive(path):
                    return self.deny(sg, "B4")
                if path in ("", ".", "./") or path.endswith("/") or self.c.is_dir(self.res(path, root, d)):
                    self.listing(sg, "B2", True)
            elif t.endswith("^{tree}"):
                self.listing(sg, "B2", True)


# ----------------------------------------------------------------------------------------- decision
def decide(payload, env):
    """None (no objection) or the first Finding for this tool call."""
    tool = payload.get("tool_name")
    ti = payload.get("tool_input")
    ti = ti if isinstance(ti, dict) else {}
    ctx = Ctx(env, payload.get("cwd"))
    if tool in FILE_TOOLS:
        return ctx.file_tool(tool, ti)
    if tool == "Grep":
        return ctx.grep_tool(ti)
    if tool == "Glob":
        return ctx.glob_tool(ti)
    if tool in SHELL_TOOLS:
        cmd = ti.get("command")
        if tool == "Monitor" and cmd is None:
            return None                                             # a WebSocket monitor runs no command
        if not isinstance(cmd, str):
            cmd = next((v for k, v in ti.items() if k in ("script", "code") and isinstance(v, str)), None)
            if cmd is None:
                return Finding("B5", "", "no command field")
        an = Analyzer(ctx, cmd)
        try:
            an.script(cmd, SHELL_TOOLS[tool], State(ctx.cwd, {}), 0)
        except Finding as f:
            return f
    return None


# ----------------------------------------------------------------------------------------- log, main
def log_path(env):
    """The log file: SL1_GUARD_LOG, else ~/.claude/sl1_guard.log; never a path inside the project."""
    default = os.path.join(os.path.expanduser("~"), ".claude", "sl1_guard.log")
    proj = abs_key(env.get("CLAUDE_PROJECT_DIR") or "") if env.get("CLAUDE_PROJECT_DIR") else None

    def inside(p):
        k = abs_key(os.path.abspath(p))
        return bool(proj and k and _inside(k, proj))

    cand = env.get(LOG_ENV) or default
    if inside(cand):
        cand = default
    if inside(cand):
        return None
    return cand


def write_log(env, tool, rule, subject):
    try:
        path = log_path(env)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subj = re.sub(r"[\t\r\n]+", " ", str(subject))[:200]
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{tool}\t{rule}\t{subj}\n"
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line)
    except Exception:
        pass


def fail_denies(tool, payload, raw):
    """Fail mode (a), deny mode only: shell and search tools are denied; a file tool only when its path is
    test-named or NAMED_DENY, or cannot be read. Content, old_string and new_string are never looked at."""
    if tool not in FILE_TOOLS:
        return True
    path = None
    if isinstance(payload, dict) and isinstance(payload.get("tool_input"), dict):
        ti = payload["tool_input"]
        path = ti.get("notebook_path") if tool == "NotebookEdit" else ti.get("file_path")
    if not isinstance(path, str):
        m = re.search(r'"(?:file_path|notebook_path)"\s*:\s*"([^"]*)"', raw or "")
        path = m.group(1) if m else None
    return not path or sensitive(path)


def run(raw, env, mode=None):
    """(exit code, stderr line) for one payload."""
    mode = mode or MODE
    if mode not in ("log", "deny"):
        mode = "log"
    tool = "?"
    payload = None
    text = ""
    try:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        payload = json.loads(text) if text.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("the payload is not a JSON object")
        tool = str(payload.get("tool_name") or "?")
        f = decide(payload, env)
        if f is None:
            return 0, ""
        write_log(env, tool, "SL1-" + f.rule, f.subject)
        if mode == "deny":
            return DENY_EXIT, f.line()
        return 0, ""
    except Exception as e:
        write_log(env, tool, "SL1-E1", f"{type(e).__name__}: {e}")
        if mode == "deny" and fail_denies(tool, payload, text):
            return DENY_EXIT, Finding("E1").line()
        return 0, ""


def main():
    try:
        raw = sys.stdin.buffer.read()
    except Exception:
        raw = b""
    rc, err = run(raw, dict(os.environ))
    if err:
        try:
            sys.stderr.write(err.encode("ascii", "backslashreplace").decode("ascii") + "\n")
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        code = 0
    sys.exit(code)
