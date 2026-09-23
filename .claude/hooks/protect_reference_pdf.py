#!/usr/bin/env python3
"""L-PROT guard: a PreToolUse hook that protects docs/reference/reference.pdf (AGENTS.md rule 2).

The guard reads the hook payload (JSON) on stdin and inspects only the text of the tool call. It never
opens, stats, hashes or lists a file, so the guard itself cannot touch the protected path.

  exit 0  no objection: the normal permission flow continues, and deny rules still apply.
  exit 2  block. stderr names the rule. Every internal error, unparsable command, unknown git
          subcommand or unknown git global option also exits 2 (fail-closed).

Rules
  N  The command names the PDF, in text or with a glob that matches it. The one exception is the
     exact whole command `git status --porcelain=v1 -- docs/reference/reference.pdf`.
  G  A git command whose scope would include the PDF, and which lacks
     `:(exclude)docs/reference/reference.pdf` given from the repository root.
  C  `git commit` without `--only -- <paths>`.
  S  (R2b) A non-git recursive reader, the Grep tool or the Glob tool whose scope includes the PDF.
  P  The command could not be analysed.

Known gaps are listed in the L-PROT proposal. A script run by an interpreter (for example
`python x.py`, or `bash x.sh`) is not read. Remote commands sent over ssh are not analysed. A hook that
cannot start, or that times out, does not block.
"""
from __future__ import annotations

import fnmatch
import json
import os
import posixpath
import re
import shlex
import sys

PDF = "docs/reference/reference.pdf"
PDF_DIR = "docs/reference"
EXCLUDE = ":(exclude)docs/reference/reference.pdf"
ALLOWED_STATUS = "git status --porcelain=v1 -- docs/reference/reference.pdf"
ENABLE_R2B = True

HINT = (" Allowed probe: `git status --porcelain=v1 -- docs/reference/reference.pdf`. For repo-wide git,"
        " run from the repo root and end the command with `-- . ':(exclude)docs/reference/reference.pdf'`"
        " (ls-tree, cat-file and archive have no excluded form; name non-covering paths instead).")

UNKNOWN = object()          # a path or directory the guard cannot resolve; treated as covering the PDF
GLOB_CHARS = set("*?[")


class Deny(Exception):
    def __init__(self, rule: str, why: str):
        super().__init__(f"[L-PROT {rule}] {why}")
        self.rule = rule


# ----------------------------------------------------------------------------------------- paths
def _abs_form(p: str):
    """'C:\\x', 'C:/x' or '/c/x' becomes 'c:/x'. A POSIX absolute path stays as it is. Relative -> None."""
    p = p.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):(/.*)?$", p)
    if m:
        return (m.group(1) + ":" + (m.group(2) or "/")).lower()
    m = re.match(r"^/([A-Za-z])(/.*)?$", p)
    if m and os.name == "nt":
        return (m.group(1) + ":" + (m.group(2) or "/")).lower()
    if p.startswith("/"):
        return p.lower()
    return None


class Ctx:
    def __init__(self, project_dir: str, cwd: str | None):
        root = _abs_form(project_dir or "")
        if root is None:
            raise Deny("P", "CLAUDE_PROJECT_DIR is not set to an absolute path")
        self.root = posixpath.normpath(root)
        # The cwd relative to the project root. A cwd outside the project is treated as the root:
        # it may be another checkout of this repository, where the same pathspec rules apply.
        self.prefix = self.rel_of_abs(_abs_form(cwd)) if cwd and _abs_form(cwd) else ""
        if self.prefix is None:
            self.prefix = ""

    def rel_of_abs(self, a):
        if a is None:
            return None
        a = posixpath.normpath(a)
        if a == self.root:
            return ""
        if a.startswith(self.root.rstrip("/") + "/"):
            return a[len(self.root.rstrip("/")) + 1:]
        return None                                         # outside the project

    def resolve(self, tok: str, prefix):
        """Repo-relative lower-case path. None = outside the project. UNKNOWN = unresolvable."""
        t = tok.replace("\\", "/")
        if not t:
            return UNKNOWN
        if "$" in t or t.startswith("~") or prefix is UNKNOWN:
            return UNKNOWN
        a = _abs_form(t)
        if a is None:
            a = posixpath.normpath(posixpath.join(self.root, prefix or "", t)).lower()
        return self.rel_of_abs(a)


def _has_glob(s: str) -> bool:
    return any(c in GLOB_CHARS for c in s)


def covers(rel, glob: bool = True) -> bool:
    """True if the repo-relative path or pathspec `rel` includes the PDF (as itself or an ancestor)."""
    if rel is UNKNOWN:
        return True
    if rel is None:
        return False
    r = rel.lower().rstrip("/")
    if r in ("", "."):
        return True
    if r == PDF or PDF.startswith(r + "/"):
        return True
    if glob and _has_glob(r):
        return any(fnmatch.fnmatchcase(s, r) for s in (PDF, PDF_DIR, "docs"))
    return False


def glob_names_pdf(tok: str, ctx: Ctx, prefix) -> bool:
    """A glob token that could expand to, or match, the PDF (git pathspec `*` also crosses '/')."""
    t = tok.replace("\\", "/").lower()
    if not _has_glob(t):
        return False
    if any(fnmatch.fnmatchcase(s, t) for s in (PDF, "reference/reference.pdf", "reference.pdf")):
        return True
    r = ctx.resolve(tok, prefix)
    return r is not UNKNOWN and r is not None and fnmatch.fnmatchcase(PDF, r.lower())


# ------------------------------------------------------------------------------- bash scanning
def _match_paren(text: str, i: int) -> int:
    """Index of the ')' closing the '(' at text[i], skipping quoted spans. Raises on no match."""
    depth, q = 0, None
    k = i
    while k < len(text):
        c = text[k]
        if q:
            if c == "\\" and q == '"':
                k += 2
                continue
            if c == q:
                q = None
        elif text.startswith("<<", k) and not text.startswith("<<<", k):
            k = _skip_heredoc(text, k)                      # $(cat <<'EOF' ... EOF) commit messages
            continue
        elif c in "'\"":
            q = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    raise Deny("P", "unbalanced $( ... ) in the command")


def _skip_heredoc(text: str, k: int) -> int:
    """Index just past the delimiter line of the heredoc whose `<<` is at text[k]."""
    j = k + 2
    strip_tabs = j < len(text) and text[j] == "-"
    j += 1 if strip_tabs else 0
    while j < len(text) and text[j] in " \t":
        j += 1
    m = re.match(r"""(['"]?)([^\s;&|<>()'"]+)\1""", text[j:])
    if not m:
        raise Deny("P", "heredoc delimiter could not be parsed")
    delim = m.group(2)
    nl = text.find("\n", j + m.end())
    if nl < 0:
        raise Deny("P", f"heredoc '{delim}' is not terminated")
    pos = nl + 1
    while pos <= len(text):
        e = text.find("\n", pos)
        line = text[pos:] if e < 0 else text[pos:e]
        if (line.lstrip("\t") if strip_tabs else line).rstrip("\r") == delim:
            return len(text) if e < 0 else e
        if e < 0:
            break
        pos = e + 1
    raise Deny("P", f"heredoc '{delim}' is not terminated")


def _subs_in(text: str) -> list[str]:
    """Command substitutions $( ... ) and ` ... ` in text that bash expands (double quotes, heredocs)."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("$((", i):
            i += 3
            continue
        if text.startswith("$(", i):
            k = _match_paren(text, i + 1)
            out.append(text[i + 2:k])
            i = k + 1
            continue
        if text[i] == "`":
            k = text.find("`", i + 1)
            if k < 0:
                raise Deny("P", "unbalanced backquote in the command")
            out.append(text[i + 1:k])
            i = k + 1
            continue
        if text[i] == "\\":
            i += 2
            continue
        i += 1
    return out


def scan_bash(text: str):
    """Quote-aware pass. Returns (flat, subs, heredocs).

    flat      the text with heredoc bodies and comments removed, and with unquoted newlines and
              backquotes turned into ' ; ' separators.
    subs      command substitutions inside double quotes or expanding heredocs (analysed recursively).
    heredocs  (introducing line, body) pairs.
    """
    out, subs, heredocs, pending = [], [], [], []
    line_start = 0
    i, n, q = 0, len(text), None
    while i < n:
        c = text[i]
        if q == "'":
            out.append(c)
            if c == "'":
                q = None
            i += 1
            continue
        if q == '"':
            if c == "\\":
                out.append(text[i:i + 2])
                i += 2
                continue
            if c == '"':
                q = None
                out.append(c)
                i += 1
                continue
            if text.startswith("$(", i) and not text.startswith("$((", i):
                k = _match_paren(text, i + 1)
                subs.append(text[i + 2:k])
                out.append(text[i:k + 1])
                i = k + 1
                continue
            if c == "`":
                k = text.find("`", i + 1)
                if k < 0:
                    raise Deny("P", "unbalanced backquote in the command")
                subs.append(text[i + 1:k])
                out.append(text[i:k + 1])
                i = k + 1
                continue
            out.append(c)
            i += 1
            continue
        # ---- unquoted
        if c == "\\":
            if text.startswith("\\\n", i):
                i += 2                                      # line continuation
                continue
            out.append(text[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            q = c
            out.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;&|()"):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("<<", i) and not text.startswith("<<<", i):
            j = i + 2
            strip_tabs = j < n and text[j] == "-"
            if strip_tabs:
                j += 1
            while j < n and text[j] in " \t":
                j += 1
            m = re.match(r"""(['"]?)([^\s;&|<>()'"]+)\1""", text[j:])
            if not m:
                raise Deny("P", "heredoc delimiter could not be parsed")
            pending.append((m.group(2), strip_tabs, bool(m.group(1))))
            out.append(" << HEREDOC ")
            i = j + m.end()
            continue
        if c == "`":
            out.append(" ; ")
            i += 1
            continue
        if c == "\n":
            intro = "".join(out)[line_start:]
            out.append(" ; ")
            i += 1
            for delim, strip_tabs, quoted in pending:
                body = []
                while True:
                    if i >= n:
                        raise Deny("P", f"heredoc '{delim}' is not terminated")
                    k = text.find("\n", i)
                    line = text[i:] if k < 0 else text[i:k]
                    i = n if k < 0 else k + 1
                    if (line.lstrip("\t") if strip_tabs else line).rstrip("\r") == delim:
                        break
                    body.append(line)
                body_text = "\n".join(body)
                heredocs.append((intro, body_text))
                if not quoted:
                    subs.extend(_subs_in(body_text))
            pending = []
            line_start = len("".join(out))
            continue
        out.append(c)
        i += 1
    if q:
        raise Deny("P", "unterminated quote in the command")
    if pending:
        raise Deny("P", "heredoc is not terminated")
    return "".join(out), subs, heredocs


def tokenize(flat: str, ps: bool = False) -> list[str]:
    lx = shlex.shlex(flat, posix=True, punctuation_chars=";&|()<>" + ("{}" if ps else ""))
    lx.whitespace_split = True
    lx.commenters = ""
    if ps:
        lx.escape = "`"
        lx.escapedquotes = '"'
    try:
        return list(lx)
    except ValueError as e:
        raise Deny("P", f"command could not be tokenised ({e})")


def segments(tokens: list[str]) -> list[list[str]]:
    segs, cur, skip_next = [], [], False
    for t in tokens:
        if skip_next:
            skip_next = False
            continue
        if t and all(ch in ";&|()<>{}" for ch in t):
            if ("<" in t or ">" in t) and "(" not in t and ")" not in t:
                if cur and cur[-1].isdigit():
                    cur.pop()                               # fd number of 2>&1, 2>/dev/null
                skip_next = True                            # the redirect target is not an argument
                continue
            if cur:
                segs.append(cur)
            cur = []
            continue
        cur.append(t)
    if cur:
        segs.append(cur)
    return segs


KEYWORDS = {"!", "{", "}", "if", "then", "else", "elif", "fi", "do", "done", "while", "until", "time",
            "coproc", "exec", "command", "builtin", "nohup", "noglob"}
SKIP_SEGMENT = {"for", "case", "select", "function", "esac", "in"}
SAFE_GIT_ENV = re.compile(r"^GIT_(PAGER|EDITOR|SEQUENCE_EDITOR|TERMINAL_PROMPT|SSH|SSH_COMMAND|ASKPASS|"
                          r"TRACE\w*|AUTHOR_\w+|COMMITTER_\w+|OPTIONAL_LOCKS|PROGRESS_DELAY|MERGE_AUTOEDIT)=")
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
PS_SHELLS = {"powershell", "pwsh"}
INTERPRETERS = {"python", "python3", "py", "node", "perl", "ruby"}


def _base(word: str) -> str:
    b = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return b[:-4] if b.endswith(".exe") else b


def unwrap(seg: list[str]):
    """Strip keywords, env assignments and wrappers. Returns (words, env_names, from_xargs)."""
    env, xargs = [], False
    words = list(seg)
    changed = True
    while words and changed:
        changed = False
        w = words[0]
        b = _base(w)
        if w in KEYWORDS:
            words = words[1:]
            changed = True
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            env.append(w)
            words = words[1:]
            changed = True
        elif b == "env":
            words = words[1:]
            while words and (words[0].startswith("-") or re.match(r"^[A-Za-z_]\w*=", words[0])):
                if words[0] in ("-u", "--unset", "-C", "--chdir", "-S", "--split-string"):
                    if words[0] in ("-C", "--chdir", "-S", "--split-string"):
                        raise Deny("P", f"`env {words[0]}` is not analysed")
                    words = words[2:]
                    continue
                if re.match(r"^[A-Za-z_]\w*=", words[0]):
                    env.append(words[0])
                words = words[1:]
            changed = True
        elif b in ("time", "nice", "timeout", "stdbuf", "sudo", "watch", "xargs"):
            xargs = xargs or b == "xargs"
            words = words[1:]
            value_opts = {"nice": {"-n"}, "timeout": {"-s", "--signal", "-k", "--kill-after"},
                          "sudo": {"-u", "-g", "-h", "-p", "-C", "-D"}, "watch": {"-n", "--interval"},
                          "xargs": {"-I", "-n", "-d", "-L", "-P", "-s", "-a", "-E"},
                          "stdbuf": {"-i", "-o", "-e"}}.get(b, set())
            while words and words[0].startswith("-"):
                words = words[2:] if words[0] in value_opts else words[1:]
            if b == "timeout" and words:
                words = words[1:]                           # the duration
            changed = True
    return words, env, xargs


# ------------------------------------------------------------------------------------ git rules
DIFF_FLAG = re.compile(r"^(-p|-u|--patch.*|--stat.*|--numstat|--shortstat|--dirstat.*|--cumulative|"
                       r"--summary|--name-only|--name-status|--raw|--cc|-c|--combined-all-paths|"
                       r"--full-diff|-S.*|-G.*|--pickaxe.*|--find-object.*|-L.*|--check|"
                       r"--compact-summary|-W|--function-context|--word-diff.*|--color-words.*|"
                       r"--binary|--ext-diff|-U.*|--unified.*|--diff-merges.*|-m|--remerge-diff|"
                       r"--dd|--patch-with-.*)$")
SAFE_SUB = {"rev-parse", "branch", "tag", "remote", "fetch", "push", "ls-remote", "config", "update-ref",
            "symbolic-ref", "show-ref", "for-each-ref", "merge-base", "describe", "name-rev", "cherry",
            "count-objects", "var", "version", "help", "verify-commit", "verify-tag", "check-ref-format",
            "shortlog", "notes", "init", "clean", "check-ignore", "check-attr", "mailinfo",
            "interpret-trailers", "column", "stripspace"}
DENY_SUB = {"clone", "switch", "merge", "rebase", "pull", "cherry-pick", "revert", "am", "apply", "gc",
            "fsck", "repack", "prune", "bundle", "fast-export", "fast-import", "filter-branch",
            "filter-repo", "submodule", "sparse-checkout", "read-tree", "checkout-index", "update-index",
            "hash-object", "mktree", "replace", "bisect", "lfs", "format-patch", "send-email",
            "request-pull", "range-diff", "maintenance", "pack-objects", "unpack-objects",
            "verify-pack", "index-pack", "write-tree", "commit-tree", "annotate-stdin", "p4", "svn"}
HEX = re.compile(r"^[0-9a-fA-F]{4,64}$")


def _split_dd(args):
    if "--" in args:
        k = args.index("--")
        return args[:k], args[k + 1:], True
    return args, [], False


def _positionals(opts, value_opts=()):
    out, skip = [], False
    for t in opts:
        if skip:
            skip = False
            continue
        if t in value_opts:
            skip = True
            continue
        if t.startswith("-"):
            continue
        out.append(t)
    return out


def _parse_magic(spec: str):
    """(magic_words, body) for a pathspec, or (None, spec) when the magic is not understood."""
    if not spec.startswith(":"):
        return set(), spec
    if spec.startswith(":("):
        k = spec.find(")")
        if k < 0:
            return None, spec
        words = {w.strip().lower() for w in spec[2:k].split(",") if w.strip()}
        if not words <= {"top", "exclude", "glob", "icase", "literal"}:
            return None, spec
        return words, spec[k + 1:]
    words, i = set(), 1
    while i < len(spec) and spec[i] in "/!^":
        words.add("top" if spec[i] == "/" else "exclude")
        i += 1
    if i < len(spec) and spec[i] == ":":
        i += 1
    return words, spec[i:]


def scope_includes_pdf(specs, ctx: Ctx, prefix, what: str) -> None:
    """Raise Deny('G') unless the pathspec list provably excludes the PDF."""
    positives, excluded, had_neg = [], False, False
    for s in specs:
        if s == EXCLUDE:
            if prefix != "":
                raise Deny("G", f"{what}: the exclude pathspec is relative to the cwd; run from the"
                                " repository root." + HINT)
            excluded = True
            continue
        if s.lower().startswith("--pathspec-from-file"):
            raise Deny("G", f"{what}: --pathspec-from-file hides the pathspecs." + HINT)
        magic, body = _parse_magic(s)
        if magic is None:
            positives.append(UNKNOWN)
            continue
        if "exclude" in magic:
            had_neg = True                                  # other exclude forms are not trusted
            continue
        r = ctx.resolve(body or ".", "" if "top" in magic else prefix)
        positives.append((r, "literal" not in magic))
    if not positives:
        positives = [("", True)] if (had_neg or excluded or not specs) else []
    hit = any(p is UNKNOWN or covers(p[0], p[1]) for p in positives)
    if hit and not excluded:
        raise Deny("G", f"{what}: the pathspec scope includes {PDF}." + HINT)


def _object_path_hits(obj: str, ctx: Ctx, prefix) -> bool:
    """For '<rev>:<path>' objects: True if the object is the PDF or the tree that lists it."""
    rev, path = obj.split(":", 1)
    if path.startswith(("./", "../", ".\\", "..\\")):
        r = ctx.resolve(path, prefix)
    else:
        r = ctx.resolve(path or ".", "")
    if r is UNKNOWN:
        return True
    if r is None:
        return False
    r = r.lower().rstrip("/")
    return r in (PDF, PDF_DIR)


def check_git(words: list[str], ctx: Ctx, prefix, env: list[str], xargs: bool) -> None:
    for e in env:
        if e.upper().startswith("GIT_") and not SAFE_GIT_ENV.match(e):
            raise Deny("G", f"git with `{e.split('=')[0]}` in the environment is not analysed." + HINT)
    i = 1
    while i < len(words) and words[i].startswith("-"):
        t = words[i]
        if t == "-C":
            if i + 1 >= len(words):
                raise Deny("P", "git -C without a directory")
            r = ctx.resolve(words[i + 1], prefix)
            prefix = UNKNOWN if r is UNKNOWN else (r if r is not None else "")
            i += 2
            continue
        if t == "-c" or (t.startswith("-c") and len(t) > 2):
            kv = words[i + 1] if t == "-c" else t[2:]
            if kv.lower().startswith(("alias.", "core.worktree", "core.sparsecheckout", "include")):
                raise Deny("G", f"git -c {kv.split('=')[0]} is not analysed." + HINT)
            i += 2 if t == "-c" else 1
            continue
        if t in ("--literal-pathspecs", "--noglob-pathspecs"):
            raise Deny("G", f"{t} disables `:(exclude)` magic." + HINT)
        if t in ("--no-pager", "-p", "--paginate", "-P", "--no-replace-objects", "--no-optional-locks",
                 "--no-lazy-fetch", "--no-advice", "--glob-pathspecs", "--icase-pathspecs"):
            i += 1
            continue
        if t in ("--version", "--help", "-h", "--html-path", "--man-path", "--info-path", "--exec-path"):
            return
        raise Deny("P", f"git global option {t} is not analysed.")
    if i >= len(words):
        return
    sub, args = words[i].lower(), words[i + 1:]
    if xargs:
        args = args + ["--", "$XARGS"] if "--" not in args else args + ["$XARGS"]
    opts, after, dd = _split_dd(args)
    what = f"git {sub}"

    if sub in SAFE_SUB:
        return
    if sub in DENY_SUB:
        raise Deny("G", f"{what} works on the whole tree or object store and is not allowed by L-PROT.")
    if sub == "rev-list":
        if any(t.startswith(("--objects", "--filter")) for t in opts):
            raise Deny("G", f"{what} --objects lists blob paths." + HINT)
        return
    if sub == "worktree":
        if opts[:1] and opts[0] in ("list", "prune", "lock", "unlock"):
            return
        raise Deny("G", f"{what} {' '.join(opts[:1])} creates or moves a checkout (a copy of the PDF).")
    if sub == "commit":
        return check_commit(opts, after, dd, ctx, prefix)
    if sub == "grep":
        if not dd:
            raise Deny("G", f"{what} without `--` searches the whole tree." + HINT)
        return scope_includes_pdf(after, ctx, prefix, what)
    if sub in ("diff", "diff-files", "diff-index", "diff-tree", "difftool", "whatchanged", "archive"):
        if sub == "diff" and "--no-index" in opts:
            paths = after if dd else _positionals(opts)
            if any(covers(ctx.resolve(p, prefix)) for p in paths):
                raise Deny("G", f"{what} --no-index names a path that includes {PDF}.")
            return
        if not dd:
            raise Deny("G", f"{what} without `--` covers the whole tree." + HINT)
        return scope_includes_pdf(after, ctx, prefix, what)
    if sub in ("status", "ls-files", "add", "rm", "restore", "blame", "annotate", "stage"):
        value_opts = {"ls-files": ("-x", "-X", "--exclude", "--exclude-from"),
                      "restore": ("-s", "--source"), "blame": ("-L", "-S", "-C", "-M", "--contents")
                      }.get(sub, ())
        specs = after if dd else _positionals(opts, value_opts)
        if sub in ("blame", "annotate") and not dd:
            specs = specs[-1:] if specs else specs
        if not specs and sub in ("blame", "annotate"):
            return
        return scope_includes_pdf(specs, ctx, prefix, what)
    if sub == "mv":
        paths = after if dd else _positionals(opts)
        if any(covers(ctx.resolve(p, prefix)) for p in paths):
            raise Deny("G", f"{what} moves a path that includes {PDF}.")
        return
    if sub == "checkout":
        if dd:
            return scope_includes_pdf(after, ctx, prefix, what)
        pos = _positionals(opts, ("-b", "-B", "--orphan"))
        if any(t in ("-b", "-B", "--orphan") for t in opts) and not pos:
            return                                          # new branch at HEAD; the tree is untouched
        raise Deny("G", f"{what} without `--` switches or restores the whole tree.")
    if sub == "reset":
        if "--soft" in opts:
            return
        if dd and not any(t in ("--hard", "--merge", "--keep") for t in opts):
            return scope_includes_pdf(after, ctx, prefix, what)
        raise Deny("G", f"{what} without `-- <paths>` resets the whole index or tree.")
    if sub == "stash":
        act = opts[0] if opts and not opts[0].startswith("-") else "push"
        if act in ("list", "drop", "clear", "store"):
            return
        if act == "push" and dd and not any(t in ("-p", "--patch") for t in opts):
            return scope_includes_pdf(after, ctx, prefix, what)
        raise Deny("G", f"{what} {act} reads or rewrites the whole worktree." + HINT)
    if sub in ("log", "reflog"):
        if sub == "reflog" and opts[:1] and opts[0] in ("expire", "delete", "exists"):
            return
        if not any(DIFF_FLAG.match(t) for t in opts):
            return
        if not dd:
            raise Deny("G", f"{what} with a diff option and no `--` covers the whole tree." + HINT)
        return scope_includes_pdf(after, ctx, prefix, what)
    if sub == "show":
        objs = _positionals(opts, ("--format", "--pretty"))
        for o in objs:
            if ":" in o and _object_path_hits(o, ctx, prefix):
                raise Deny("G", f"{what} {o} shows {PDF} or the tree that lists it.")
        if objs and all(":" in o for o in objs):
            return
        if any(t in ("-s", "--no-patch", "--quiet") for t in opts) and \
                not any(DIFF_FLAG.match(t) for t in opts if t != "-s"):
            return
        if not dd:
            raise Deny("G", f"{what} without `--` shows a whole-tree diff." + HINT)
        return scope_includes_pdf(after, ctx, prefix, what)
    if sub == "cat-file":
        if any(t.startswith(("--batch", "--path", "--textconv", "--filters")) for t in opts):
            raise Deny("G", f"{what} batch/path/filter modes are not analysed.")
        type_only = any(t in ("-t", "-e") for t in opts)
        for o in _positionals(opts):
            if ":" in o:
                if _object_path_hits(o, ctx, prefix):
                    raise Deny("G", f"{what} {o} reads {PDF} or the tree that lists it.")
            elif HEX.match(o) and not type_only:
                raise Deny("G", f"{what} {o}: a bare object id may be the PDF blob; use <rev>:<path>.")
        return
    if sub == "ls-tree":
        pos = _positionals(opts) + after
        rec = any(t == "-r" or (re.match(r"^-[a-zA-Z]+$", t) and "r" in t) for t in opts)
        full = "--full-tree" in opts
        if not pos:
            raise Deny("P", f"{what} without a tree-ish")
        tree, paths = pos[0], pos[1:]
        base = ""
        if ":" in tree:
            if _object_path_hits(tree, ctx, prefix):
                raise Deny("G", f"{what} {tree} lists {PDF}.")
            base = tree.split(":", 1)[1]
        pfx = "" if full else prefix
        targets = [ctx.resolve(posixpath.join(base, p) if base else p, pfx) for p in paths] or \
                  [ctx.resolve(base or ".", "" if base else pfx)]
        for r in targets:
            if rec and covers(r):
                raise Deny("G", f"{what} -r would list {PDF} (ls-tree has no exclude magic).")
            if r is UNKNOWN or (r is not None and (r.lower().rstrip("/") in (PDF, PDF_DIR)
                                                   or (_has_glob(r) and covers(r)))):
                raise Deny("G", f"{what} would list {PDF}.")
        return
    raise Deny("P", f"{what} is not a subcommand L-PROT knows; denied (fail-closed).")


def check_commit(opts, after, dd, ctx, prefix) -> None:
    if not dd or not after:
        raise Deny("C", "git commit must name its paths: `git commit --only ... -- <paths>`.")
    if "--only" not in opts:
        raise Deny("C", "git commit must use `--only -- <paths>`.")
    value_long = {"-m", "--message", "-F", "--file", "-C", "--reuse-message", "-c", "--reedit-message",
                  "--author", "--date", "-t", "--template", "--cleanup", "--fixup", "--squash",
                  "--trailer"}
    skip = False
    for t in opts:
        if skip:
            skip = False
            continue
        if t in value_long:
            skip = True
            continue
        if t in ("--all", "--include", "--patch", "--interactive") or t.startswith("--pathspec-from-file"):
            raise Deny("C", f"git commit {t} is not allowed; use `--only -- <paths>`.")
        if t.startswith("-") and not t.startswith("--") and len(t) > 1:
            for k, ch in enumerate(t[1:]):
                if ch in "aip":
                    raise Deny("C", f"git commit -{ch} is not allowed; use `--only -- <paths>`.")
                if ch in "mFCctu":
                    if k == len(t) - 2 and ch != "u":
                        skip = True
                    break
    scope_includes_pdf(after, ctx, prefix, "git commit")


# ------------------------------------------------------------------------ R2b non-git readers
RECURSIVE_READERS = {"find", "rg", "ag", "ack", "tree", "du", "tar", "zip", "7z", "rsync"}
FLAG_RECURSIVE = {"grep": {"-r", "-R", "--recursive", "--dereference-recursive"},
                  "egrep": {"-r", "-R", "--recursive"}, "fgrep": {"-r", "-R", "--recursive"},
                  "cp": {"-r", "-R", "-a", "--recursive", "--archive"},
                  "scp": {"-r"}, "rm": {"-r", "-R", "-rf", "-fr", "--recursive"},
                  "chmod": {"-R", "--recursive"}, "chown": {"-R", "--recursive"},
                  "ls": {"-R", "--recursive"}}
INCLUDE_OPTS = {"rg": ("-g", "--glob", "--iglob", "-t", "--type"), "grep": ("--include",),
                "egrep": ("--include",), "fgrep": ("--include",)}


def check_reader(b: str, args: list[str], ctx: Ctx, prefix) -> None:
    if b in ("mv",):
        if any(covers(ctx.resolve(a, prefix), glob=False) for a in args if not a.startswith("-")):
            raise Deny("S", f"`{b}` moves a directory that holds {PDF}.")
        return
    recursive = b in RECURSIVE_READERS or any(
        a in FLAG_RECURSIVE.get(b, ()) or (b in FLAG_RECURSIVE and re.match(r"^-[a-zA-Z]+$", a)
                                           and ("r" in a or "R" in a) and b != "chmod")
        or a.startswith("--directories=recurse") for a in args)
    if b in ("ls", "dir") and not recursive:
        for a in args:
            if not a.startswith("-"):
                r = ctx.resolve(a, prefix)
                if r is UNKNOWN or (r is not None and r.lower().rstrip("/") == PDF_DIR):
                    raise Deny("S", f"`{b} {a}` lists {PDF}.")
        if not [a for a in args if not a.startswith("-")] and prefix == PDF_DIR:
            raise Deny("S", f"`{b}` in {PDF_DIR} lists {PDF}.")
        return
    if not recursive:
        return
    # an include filter that cannot match the PDF keeps it unopened (rg -g/-t, grep --include)
    incl, k = [], 0
    while k < len(args):
        a = args[k]
        for o in INCLUDE_OPTS.get(b, ()):
            if a == o and k + 1 < len(args):
                incl.append((o, args[k + 1]))
            elif a.startswith(o + "="):
                incl.append((o, a.split("=", 1)[1]))
        k += 1
    if incl and all(not v.startswith("!") and not (o in ("-t", "--type") and v.lower() in ("all", "pdf"))
                    and not fnmatch.fnmatchcase("reference.pdf", v.lower()) and
                    not fnmatch.fnmatchcase(PDF, v.lower()) for o, v in incl):
        return
    value_opts = {"find": (), "grep": ("-e", "-f", "-m", "-A", "-B", "-C", "--include", "--exclude"),
                  "rg": ("-e", "-f", "-g", "--glob", "-t", "--type", "-m", "-A", "-B", "-C", "-T",
                         "--type-not", "--iglob", "-j", "-M"),
                  "tar": ("-f", "-C", "--file"), "du": ("-d", "--max-depth")}.get(b, ())
    pos = _positionals(args, value_opts)
    if b in ("grep", "egrep", "fgrep", "rg", "ag", "ack") and not any(
            a in ("-e", "-f", "--regexp", "--file") for a in args):
        pos = pos[1:]                                       # the first positional is the pattern
    if b == "find":
        pos = [a for a in args if not a.startswith(("-", "(", ")", "!"))][:1] or []
        pos = [p for p in pos if not p.startswith("-")]
    targets = [ctx.resolve(p, prefix) for p in pos] or [ctx.resolve(".", prefix)]
    if any(covers(r) for r in targets):
        raise Deny("S", f"`{b}` would read or list {PDF} (recursive scope includes it). Use the Grep/Glob"
                        " tools with a glob, or a narrower path.")


def check_interpreter_inline(code: str) -> None:
    if re.search(r"\bgit\b", code) and re.search(
            r"\b(status|diff|grep|log|show|ls-files|ls-tree|cat-file|archive|add|commit|stash|checkout"
            r"|restore|reset|rm|mv)\b", code):
        raise Deny("S", "inline interpreter code that runs git is not analysed; run git from the shell.")


# ------------------------------------------------------------------------------- analysers
def analyse_bash(cmd: str, ctx: Ctx, prefix, depth: int = 0) -> None:
    if depth > 4:
        raise Deny("P", "shell nesting too deep to analyse")
    flat, subs, heredocs = scan_bash(cmd)
    for s in subs:
        analyse_bash(s, ctx, prefix, depth + 1)
    for intro, body in heredocs:
        words = [w for w in re.split(r"[\s;&|()]+", intro) if w]
        bases = {_base(w) for w in words}
        if bases & SHELLS and "-c" not in words:
            analyse_bash(body, ctx, prefix, depth + 1)
        elif ENABLE_R2B and bases & INTERPRETERS:
            check_interpreter_inline(body)
    for seg in segments(tokenize(flat)):
        prefix = analyse_segment(seg, ctx, prefix, depth, ps=False)


def analyse_segment(seg, ctx: Ctx, prefix, depth: int, ps: bool):
    for t in seg:
        if glob_names_pdf(t, ctx, prefix):
            raise Deny("N", f"the glob `{t}` matches {PDF}." + HINT)
    if seg and seg[0] in SKIP_SEGMENT:
        return prefix
    words, env, xargs = unwrap(seg)
    if not words:
        return prefix
    b = _base(words[0])
    args = words[1:]
    if b in ("cd", "pushd", "set-location", "sl", "chdir"):
        tgt = [a for a in args if not a.startswith("-")]
        if not tgt or tgt[0] in ("~",):
            return ""
        r = ctx.resolve(tgt[0], prefix)
        return UNKNOWN if r is UNKNOWN else (r if r is not None else "")
    if b in ("popd", "pop-location"):
        return UNKNOWN
    if b in ("export", "set", "declare", "typeset") or b.startswith("$env:"):
        for a in args:
            if a.upper().startswith("GIT_") and "=" in a and not SAFE_GIT_ENV.match(a):
                raise Deny("G", f"`{b} {a.split('=')[0]}` changes how git reads pathspecs.")
        return prefix
    if b == "git":
        check_git(words, ctx, prefix, env, xargs)
        return prefix
    if b in SHELLS:
        for k, a in enumerate(args):
            if re.match(r"^-[a-zA-Z]*c[a-zA-Z]*$", a):
                if k + 1 >= len(args):
                    raise Deny("P", f"`{b} -c` without a command string")
                analyse_bash(args[k + 1], ctx, prefix, depth + 1)
                break
        return prefix
    if b in PS_SHELLS:
        for k, a in enumerate(args):
            if a.lower() in ("-command", "-c", "/c"):
                analyse_ps(" ".join(args[k + 1:]), ctx, prefix, depth + 1)
                break
            if a.lower() in ("-encodedcommand", "-ec", "-e"):
                raise Deny("P", f"`{b} {a}` hides the command text")
        return prefix
    if b == "cmd":
        for k, a in enumerate(args):
            if a.lower() in ("/c", "/k"):
                analyse_bash(" ".join(args[k + 1:]), ctx, prefix, depth + 1)
                break
        return prefix
    if b in ("eval", "invoke-expression", "iex"):
        (analyse_ps if ps else analyse_bash)(" ".join(args), ctx, prefix, depth + 1)
        return prefix
    if b == "start-process" and any(_base(a) == "git" for a in args):
        raise Deny("P", "Start-Process git is not analysed; call git directly.")
    if ENABLE_R2B:
        if b in INTERPRETERS:
            for k, a in enumerate(args):
                if a in ("-c", "-e", "--eval") and k + 1 < len(args):
                    check_interpreter_inline(args[k + 1])
        elif ps:
            check_ps_reader(b, args, ctx, prefix)
        elif b in RECURSIVE_READERS or b in FLAG_RECURSIVE or b in ("ls", "dir", "mv"):
            check_reader(b, args, ctx, prefix)
    return prefix


def analyse_ps(cmd: str, ctx: Ctx, prefix, depth: int = 0) -> None:
    if depth > 4:
        raise Deny("P", "shell nesting too deep to analyse")
    # here-strings: drop the bodies (they are data); an unterminated one is fail-closed
    text, out, i = cmd, [], 0
    while True:
        m = re.search(r"@(['\"])[ \t]*\r?\n", text[i:])
        if not m:
            out.append(text[i:])
            break
        start = i + m.start()
        close = re.search(r"\r?\n" + m.group(1) + "@", text[i + m.end():])
        if not close:
            raise Deny("P", "unterminated PowerShell here-string")
        out.append(text[i:start] + " HERESTRING ")
        i = i + m.end() + close.end()
    flat = "".join(out)
    # PowerShell $( ... ) inside double quotes runs commands
    for m in re.finditer(r"\$\(", flat):
        k = _match_paren(flat, m.start() + 1)
        analyse_ps(flat[m.start() + 2:k], ctx, prefix, depth + 1)
    flat = re.sub(r"\r?\n", " ; ", flat)
    for seg in segments(tokenize(flat, ps=True)):
        prefix = analyse_segment(seg, ctx, prefix, depth, ps=True)


PS_LISTERS = {"get-childitem", "gci", "dir", "ls", "select-string", "sls", "copy-item", "cp", "copy",
              "compress-archive", "remove-item", "rm", "del", "ri", "move-item", "mv", "move",
              "robocopy", "xcopy", "findstr", "tree", "get-filehash"}


def check_ps_reader(b: str, args, ctx: Ctx, prefix) -> None:
    if b not in PS_LISTERS:
        return
    low = [a.lower() for a in args]
    recursive = any(a in ("-recurse", "-r", "-s", "/s", "/e", "/mir", "-depth") for a in low) or \
        b in ("robocopy", "tree") or (b == "select-string" and not any(
            a in ("-path", "-literalpath") for a in low))
    paths, k = [], 0
    while k < len(args):
        a = low[k]
        if a in ("-path", "-literalpath", "-destination", "-destinationpath"):
            if k + 1 < len(args):
                paths.append(args[k + 1])
            k += 2
            continue
        if a in ("-filter", "-include", "-exclude", "-pattern", "-depth", "-algorithm"):
            k += 2
            continue
        if not a.startswith(("-", "/")):
            paths.append(args[k])
        k += 1
    if b in ("select-string", "sls") and not any(a in ("-path", "-literalpath") for a in low):
        return                                              # reads the pipeline, not files
    targets = [ctx.resolve(p, prefix) for p in paths] or [ctx.resolve(".", prefix)]
    for r in targets:
        if r is UNKNOWN or (r is not None and (covers(r) if recursive or b in (
                "remove-item", "rm", "del", "ri", "move-item", "mv", "move", "copy-item", "cp", "copy",
                "compress-archive") else r.lower().rstrip("/") == PDF_DIR)):
            raise Deny("S", f"`{b}` scope includes {PDF}.")


# ---------------------------------------------------------------------- Grep / Glob tools (R2b)
def _glob_re(pattern: str) -> re.Pattern:
    """Glob-tool pattern -> regex: ** crosses directories, * and ? do not, {a,b} alternates."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "{":
            k = pattern.find("}", i)
            if k < 0:
                out.append(re.escape(c))
                i += 1
            else:
                out.append("(?:" + "|".join(re.escape(x).replace(r"\*", "[^/]*")
                                            for x in pattern[i + 1:k].split(",")) + ")")
                i = k + 1
        elif c == "[":
            k = pattern.find("]", i)
            if k < 0:
                out.append(re.escape(c))
                i += 1
            else:
                out.append("[" + pattern[i + 1:k].replace("!", "^", 1) + "]")
                i = k + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$", re.I)


def check_search_tool(tool: str, inp: dict, ctx: Ctx) -> None:
    raw = inp.get("path") or ""
    base = ctx.resolve(raw, ctx.prefix) if raw else ctx.prefix
    if base is None:
        return
    if base is UNKNOWN:
        raise Deny("S", f"{tool}: search path could not be resolved.")
    b = base.lower().rstrip("/")
    if b == PDF:
        raise Deny("S", f"{tool} targets {PDF} directly.")
    if not covers(b, glob=False):
        return
    sub = PDF[len(b) + 1:] if b else PDF                    # the PDF relative to the search root
    if tool == "Glob":
        pat = (inp.get("pattern") or "").replace("\\", "/")
        if _glob_re(pat).match(sub) or _glob_re(pat).match(PDF):
            raise Deny("S", f"Glob pattern `{pat}` under `{raw or '.'}` matches {PDF}.")
        return
    glob, typ = inp.get("glob"), inp.get("type")
    if glob:
        g = glob.replace("\\", "/")
        pats = [p for p in re.split(r"[,\s]+", g.strip("{}")) if p] if "{" in g else [g]
        if any(("/" not in p and fnmatch.fnmatchcase("reference.pdf", p.lower())) or
               _glob_re(p).match(sub) or p.startswith("!") for p in pats):
            raise Deny("S", f"Grep glob `{glob}` does not keep {PDF} out of the search.")
        return
    if typ and str(typ).lower() not in ("pdf", "all"):
        return
    raise Deny("S", f"Grep over `{raw or '.'}` would open {PDF}; add a `glob` or `type` filter, or a"
                    " narrower path.")


# ------------------------------------------------------------------------------------- main
def decide(payload: dict, project_dir: str) -> None:
    tool = payload.get("tool_name")
    inp = payload.get("tool_input")
    if not isinstance(tool, str) or not isinstance(inp, dict):
        raise Deny("P", "hook payload has no tool_name/tool_input")
    ctx = Ctx(project_dir, payload.get("cwd"))
    if tool in ("Grep", "Glob"):
        if ENABLE_R2B:
            check_search_tool(tool, inp, ctx)
        return
    if tool not in ("Bash", "PowerShell", "Monitor"):
        return
    cmd = inp.get("command")
    if cmd is None and tool == "Monitor" and "ws" in inp:
        return
    if not isinstance(cmd, str):
        raise Deny("P", f"{tool} call without a command string")
    if cmd.strip() == ALLOWED_STATUS:
        return
    if re.search(r"reference\.pdf", cmd.replace(EXCLUDE, " "), re.I):
        raise Deny("N", f"the command names {PDF}; only the exact status probe may name it." + HINT)
    if tool == "PowerShell":
        analyse_ps(cmd, ctx, ctx.prefix)
    else:
        analyse_bash(cmd, ctx, ctx.prefix)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise Deny("P", "hook payload is not a JSON object")
        decide(payload, os.environ.get("CLAUDE_PROJECT_DIR", ""))
        return 0
    except Deny as e:
        sys.stderr.write(str(e) + "\n")
        return 2
    except BaseException as e:                              # fail-closed on anything unexpected
        sys.stderr.write(f"[L-PROT P] guard error ({type(e).__name__}: {e}); denied (fail-closed).\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
