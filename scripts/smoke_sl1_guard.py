#!/usr/bin/env python3
"""SL1 guard smoke (lane L-SL1-GUARD; DL-41: scripts/smoke_*.py printing a RESULT line, no pytest).

Feeds named PreToolUse payloads to .claude/hooks/sl1_guard.py: in process, through the committed
sl1_guard.sh (the committed MODE), and through temporary copies with the other MODE, so both modes run on
every case. It also checks the launcher's failure paths, the settings.json entry, and that the existing
hook files and settings entry are unchanged since c689634.

SL-1: test-named paths, NAMED_DENY names and archives appear here only as text inside the payloads fed to
the policy. This script never opens, lists, stats or creates them, and the policy never stats them.
Every launcher call logs to a temporary directory outside the repository; the real
~/.claude/sl1_guard.log is checked unchanged. Nothing in the repository is written.

Run:  python -B scripts/smoke_sl1_guard.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HOOKS = REPO / ".claude" / "hooks"
LAUNCHER = HOOKS / "sl1_guard.sh"
POLICY = HOOKS / "sl1_guard.py"
SETTINGS = REPO / ".claude" / "settings.json"
BASE = "c689634617ef9b59473b1a6c5ea64bc99cb08a29"

sys.path.insert(0, str(HOOKS))
import sl1_guard as g  # noqa: E402

PDF_EXCL = "':(exclude)docs/reference/reference.pdf'"
TEST_EXCL = "':(exclude,icase)*test*'"
OLD_ENTRY = {"matcher": "Bash|PowerShell|Monitor|Grep|Glob",
             "hooks": [{"type": "command",
                        "command": "sh \"${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_reference_pdf.sh\" || exit 2",
                        "timeout": 30}]}
OLD_PERMISSIONS = {"deny": ["Read(/docs/reference/reference.pdf)", "Edit(/docs/reference/reference.pdf)"]}
NEW_COMMAND = 'f="${CLAUDE_PROJECT_DIR}/.claude/hooks/sl1_guard.sh"; [ -r "$f" ] || exit 0; sh "$f"'
NEW_ENTRY = {"matcher": "Read|Edit|Write|NotebookEdit|Grep|Glob|Bash|PowerShell|Monitor",
             "hooks": [{"type": "command", "command": NEW_COMMAND, "timeout": 10}]}
NEW_FILES = (".claude/hooks/sl1_guard.sh", ".claude/hooks/sl1_guard.py", "scripts/smoke_sl1_guard.py",
             "docs/lane_reports/sl1-guard.md")
GROUPS = {"A": "file tools (rule 1)", "B": "Grep/Glob tools (rule 2)", "C": "recorded slips",
          "D": "everyday commands", "E": "bare listings", "F": "recursive and glob listings",
          "G": "filter and count exemptions", "H": "git", "I": "pytest", "J": "test-named and NAMED paths",
          "K": "Monitor", "L": "destructive (rule 4)", "M": "nesting and parsing", "N": "archives",
          "W": "Windows path forms", "O": "launcher, modes, settings, invariants"}


def _tmpdir():
    while True:
        d = Path(tempfile.mkdtemp(prefix="sl1_guard_smoke_"))
        if not g.sensitive(str(d)):
            return d
        shutil.rmtree(d, ignore_errors=True)


TMP = _tmpdir()
S = TMP / "scratch"
S.mkdir()
LOGS = TMP / "logs"
LOGS.mkdir()
CLONE = TMP / "thesis_copy"                      # a clone of this repository under another name (Q2)
(CLONE / ".claude" / "hooks").mkdir(parents=True)
(CLONE / ".claude" / "hooks" / "protect_reference_pdf.sh").write_bytes(b"")
R = REPO
P = REPO.parent
REAL_LOG = Path(os.path.expanduser("~")) / ".claude" / "sl1_guard.log"
REAL_LOG_BEFORE = (REAL_LOG.exists(), REAL_LOG.stat().st_size if REAL_LOG.exists() else -1)
WIN_ENV = {"CLAUDE_PROJECT_DIR": "C:\\Users\\admin\\plantseg-thesis"}
WIN_CWD = "C:\\Users\\admin\\plantseg-thesis"
WIN2_ENV = {"CLAUDE_PROJECT_DIR": "C:\\Users\\admin\\thesis2"}
WIN2_CWD = "C:\\Users\\admin\\thesis2"


def px(p):
    return Path(p).as_posix()


def q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def qps(s):
    return "'" + str(s).replace("'", "''") + "'"


# ----------------------------------------------------------------------------------------- cases
CASES = []


def add(cid, name, tool, tinput, expect, cwd=None, env=None, exceptions=None):
    CASES.append({"id": cid, "group": cid[0], "name": name, "tool": tool, "input": tinput, "expect": expect,
                  "cwd": str(cwd or R), "env": env or {}, "exceptions": exceptions})


def bash(cid, name, cmd, expect, **kw):
    add(cid, name, "Bash", {"command": cmd, "description": name}, expect, **kw)


def pwsh(cid, name, cmd, expect, **kw):
    add(cid, name, "PowerShell", {"command": cmd}, expect, **kw)


def build_cases():
    m = str(R / "src" / "eval" / "metrics.py")
    # A: file tools
    add("A01", "read_metrics", "Read", {"file_path": m}, None)
    add("A02", "read_policy", "Read", {"file_path": str(POLICY)}, None)
    add("A03", "read_test_script", "Read", {"file_path": str(R / "scripts" / "test_teacher_init.py")}, "R1")
    add("A04", "read_latest_literal", "Read", {"file_path": str(R / "reports" / "latest_summary.md")}, "R1")
    add("A05", "read_upper_TEST", "Read", {"file_path": str(R / "data" / "TEST_manifest.csv")}, "R1")
    add("A06", "read_windows_path", "Read", {"file_path": "C:\\Users\\admin\\Tests\\notes.txt"}, "R1")
    add("A07", "edit_ok_content_ignored", "Edit", {"file_path": m, "old_string": "a", "new_string": "test"}, None)
    add("A08", "edit_test_dir", "Edit", {"file_path": str(R / "src" / "Tests" / "x.py"), "old_string": "a",
                                          "new_string": "b"}, "R1")
    add("A09", "write_scratch", "Write", {"file_path": str(S / "notes.md"), "content": "a test line"}, None)
    add("A10", "write_contest_outside", "Write", {"file_path": str(S / "contest_notes.md"), "content": "x"}, "R1")
    add("A11", "notebook_ok", "NotebookEdit", {"notebook_path": str(R / "notebooks" / "analysis.ipynb"),
                                               "new_source": "x"}, None)
    add("A12", "notebook_test", "NotebookEdit", {"notebook_path": str(R / "notebooks" / "test_eda.ipynb"),
                                                 "new_source": "x"}, "R1")
    add("A13", "exception_by_name", "Read", {"file_path": str(R / "scripts" / "test_teacher_init.py")}, None,
        exceptions=("scripts/test_teacher_init.py",))
    add("A14", "read_named_csv", "Read", {"file_path": str(S / "plantseg_exact_duplicates.csv")}, "R1")
    add("A15", "read_named_zip_case", "Read", {"file_path": "C:\\dl\\PlantSeg_v2.ZIP"}, "R1")
    add("A16", "write_zenodo_component", "Write",
        {"file_path": str(S / "b66_harness" / "inputs" / "zenodo_123" / "a.txt"), "content": "x"}, "R1")
    # B: Grep and Glob tools
    add("B01", "grep_tool_no_path", "Grep", {"pattern": "seed"}, "G1")
    add("B02", "grep_tool_src", "Grep", {"pattern": "seed", "path": str(R / "src")}, "G1")
    add("B03", "grep_tool_docs_lprot2", "Grep", {"pattern": "seed", "path": str(R / "docs")}, "G1")
    add("B04", "grep_tool_repo_root", "Grep", {"pattern": "seed", "path": str(R)}, "G1")
    add("B05", "grep_tool_parent", "Grep", {"pattern": "seed", "path": str(P)}, "G1")
    add("B06", "grep_tool_data", "Grep", {"pattern": "seed", "path": "/workspace/plantseg_data/plantseg"}, "G1")
    add("B07", "grep_tool_runs", "Grep", {"pattern": "seed", "path": "/workspace/plantseg_runs"}, "G1")
    add("B08", "grep_tool_file", "Grep", {"pattern": "seed", "path": m}, None)
    add("B09", "grep_tool_test_file", "Grep", {"pattern": "seed", "path": str(R / "scripts" / "test_x.py")}, "G2")
    add("B10", "grep_tool_scratch", "Grep", {"pattern": "seed", "path": str(S)}, None)
    add("B11", "grep_tool_glob_test", "Grep", {"pattern": "seed", "path": str(S), "glob": "*test*"}, "G2")
    add("B12", "grep_tool_cwd_scratch", "Grep", {"pattern": "seed"}, None, cwd=S)
    add("B13", "glob_no_path", "Glob", {"pattern": "**/*.py"}, "G1")
    add("B14", "glob_claude_dir", "Glob", {"pattern": "*", "path": str(R / ".claude")}, "G1")
    add("B15", "glob_absolute_pattern", "Glob", {"pattern": px(R) + "/src/**/*.py"}, "G1", cwd=S)
    add("B16", "glob_scratch", "Glob", {"pattern": "*.md", "path": str(S)}, None)
    add("B17", "glob_test_pattern", "Glob", {"pattern": "**/*test*", "path": str(S)}, "G2")
    add("B18", "glob_named_pattern", "Glob", {"pattern": "**/plantseg*.zip", "path": str(S)}, "G2")
    add("B19", "grep_tool_other_clone", "Grep", {"pattern": "seed", "path": str(CLONE / "src")}, "G1", cwd=S)
    add("B20", "grep_tool_new_file", "Grep", {"pattern": "seed", "path": str(R / "src" / "eval" / "new_mod.py")},
        None)
    # C: the recorded slips
    bash("C01", "slip_grep_scripts_glob", 'grep -n "def main" scripts/*.py', "B1")
    bash("C02", "slip_grep_rn_src", "grep -rn seed src/", "B1")
    bash("C03", "slip_grep_R_include", "grep -R --include='*.py' MODE src/eval", "B1")
    bash("C04", "slip_git_ls_files_claude", "git ls-files -- .claude/", "B2")
    bash("C05", "slip_ls_hooks", "ls .claude/hooks/", "B1")
    bash("C06", "slip_find_root_xdev", "find / -xdev -name '*.json'", "B1", cwd=S)
    # D: everyday commands, all allowed
    bash("D01", "agents_rule1_status", f"git status -sb -- . {PDF_EXCL}", None)
    bash("D02", "agents_rule1_log", "git log --oneline -5", None)
    bash("D03", "git_diff_both_excludes", f"git diff -- . {PDF_EXCL} {TEST_EXCL}", None)
    bash("D04", "git_diff_stat_both_excludes", f"git diff --stat -- . {PDF_EXCL} {TEST_EXCL}", None)
    bash("D05", "git_diff_explicit_paths", "git diff -- .claude/settings.json scripts/smoke_sl1_guard.py", None)
    bash("D06", "git_diff_quiet_hooks", f"git diff --quiet {BASE[:7]} HEAD -- .claude/hooks "
                                        "':(exclude).claude/hooks/sl1_guard.*'", None)
    bash("D07", "git_grep_exclude", f"git grep -n MODE -- src {TEST_EXCL}", None)
    bash("D08", "git_ls_files_exclude", f"git ls-files -- scripts {TEST_EXCL}", None)
    bash("D09", "git_ls_files_icase_first", "git ls-files -- ':(icase,exclude)*test*'", None)
    bash("D10", "smoke_run", "python -B scripts/smoke_sl1_guard.py", None)
    bash("D11", "smoke_glob_run", "python -B scripts/smoke_*.py", None)
    mounts = " ".join(f"--mount type=bind,src=/workspace/plantseg_data/plantseg/{s},dst=/data/{s},readonly"
                      for s in ("images/train", "images/val", "annotations/train", "annotations/val"))
    bash("D12", "docker_run_trainval_mounts", f"docker run --rm {mounts} student-image python -B "
                                              "scripts/preflight_e1_trainval.py", None)
    bash("D13", "builtin_test_f", "test -f src/eval/metrics.py && echo ok", None)
    bash("D14", "builtin_bracket", "[ -f src/eval/metrics.py ] || echo missing", None)
    bash("D15", "cat_metrics", "cat src/eval/metrics.py", None)
    bash("D16", "head_agents", "head -n 40 AGENTS.md", None)
    bash("D17", "grep_explicit_file", "grep -n MODE .claude/hooks/sl1_guard.py", None)
    bash("D18", "grep_stdin", "git log --oneline | grep -i fix", None)
    bash("D19", "ls_explicit_file", "ls -la src/eval/metrics.py", None)
    bash("D20", "ls_d_dir", "ls -ld .claude", None)
    bash("D21", "ls_scratch", f"ls -la {q(px(S))}", None)
    bash("D22", "ls_parent_nonrecursive", f"ls {q(px(P))}", None)
    bash("D23", "sha256_test_archive", "sha256sum /workspace/plantseg_data/plantseg_test.zip", None)
    bash("D24", "sed_script_word", "sed -n 's/test/x/p' src/eval/metrics.py", None)
    bash("D25", "git_log_grep_flag", "git log --grep=test --oneline", None)
    bash("D26", "git_show_blob", f"git show {BASE[:7]}:.claude/settings.json", None)
    bash("D27", "git_show_no_patch", "git show -s --format=%H HEAD", None)
    bash("D28", "cat_guard_log", "cat ~/.claude/sl1_guard.log", None)
    pwsh("D29", "ps_get_content", "Get-Content src\\eval\\metrics.py", None)
    pwsh("D30", "ps_filehash_sha256", "Get-FileHash -Algorithm SHA256 C:\\data\\plantseg_data\\plantseg_test.zip",
         None)
    bash("D31", "sha256_named_csv", "sha256sum /workspace/plantseg_runs/evidence_am18/plantseg_exact_duplicates.csv",
         None)
    bash("D32", "tar_extract_export", "tar -xf /workspace/plantseg_runs/smoke_afd2d33.tar -C /tmp/x", None)
    bash("D33", "hash_run_folder_allowed_form",
         "find /workspace/plantseg_runs/e1_s42 -type f | grep -vi test | xargs -d '\\n' sha256sum", None)
    bash("D34", "git_clean_dry_run", "git clean -n", None)
    bash("D35", "count_find_wc", "find . -name '*.py' | wc -l", None)
    bash("D36", "count_ls_R_wc", "ls -R docs | wc -l", None)
    bash("D37", "count_grep_c", "git ls-files | grep -c py", None)
    pwsh("D38", "count_ps_measure", "Get-ChildItem -Recurse src | Measure-Object", None)
    pwsh("D39", "count_ps_dot_count", "Write-Output (Get-ChildItem -Recurse src).Count", None)
    bash("D40", "assigned_var_outside", f"SP={q(px(S))}; ls -R \"$SP\"", None)
    bash("D41", "tar_list_export_filtered", "tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar | grep -vi test",
         None)
    bash("D42", "tar_list_export_count", "tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar | wc -l", None)
    bash("D43", "git_diff_name_only_explicit", "git diff --name-only HEAD -- .claude/settings.json", None)
    bash("D44", "du_summary_explicit", "du -sh /workspace/plantseg_runs/e1_s42", None)
    bash("D45", "git_add_explicit", "git add -- .claude/hooks/sl1_guard.py scripts/smoke_sl1_guard.py", None)
    bash("D46", "git_commit_only", "git commit --only -F /tmp/msg.txt -- .claude/settings.json", None)
    bash("D47", "git_push_lane", "git push -u origin lane/sl1-guard", None)
    bash("D48", "git_ls_remote_lane", "git ls-remote --heads origin lane/sl1-guard", None)
    bash("D49", "acceptance_name_status", f"git diff --name-status 6067e66 HEAD -- . {PDF_EXCL} {TEST_EXCL}", None)
    bash("D50", "acceptance_quiet", f"git diff --quiet 6067e66 HEAD -- . {PDF_EXCL} "
                                    "':(exclude).claude/settings.json' ':(exclude)scripts/smoke_sl1_guard.py'", None)
    # E: bare listings in the repository
    bash("E01", "bare_ls", "ls", "B1")
    bash("E02", "bare_ls_la", "ls -la", "B1")
    bash("E03", "bash_dir", "dir", "B1")
    pwsh("E04", "ps_get_childitem", "Get-ChildItem", "B1")
    pwsh("E05", "ps_dir", "dir", "B1")
    pwsh("E06", "ps_ls", "ls", "B1")
    bash("E07", "bare_tree", "tree", "B1")
    bash("E08", "cd_repo_then_ls", f"cd {q(px(R))} && ls", "B1", cwd=S)
    bash("E09", "cd_scratch_then_ls", f"cd {q(px(S))} && ls", None)
    # F: recursive and glob listings
    bash("F01", "ls_R_docs", "ls -R docs", "B1")
    bash("F02", "ls_glob", "ls src/eval/*.py", "B1")
    bash("F03", "rg_bare", "rg seed", "B1")
    bash("F04", "rg_src", "rg -n seed src", "B1")
    bash("F05", "find_dot", "find . -name '*.py'", "B1")
    bash("F06", "tree_src", "tree src", "B1")
    bash("F07", "du_a", "du -a src", "B1")
    bash("F08", "grep_dir_arg", "grep MODE src", "B1")
    bash("F09", "egrep_r", "egrep -r seed .", "B1")
    pwsh("F10", "ps_gci_recurse", "Get-ChildItem -Recurse src", "B1")
    pwsh("F11", "ps_gci_r_filter", "gci -r -Filter *.py", "B1")
    pwsh("F12", "ps_dir_recurse", "dir -Recurse", "B1")
    bash("F13", "cmd_dir_s_b", 'cmd /c "dir /s /b"', "B1")
    bash("F14", "findstr_s", "findstr /s /i seed *.py", "B1")
    pwsh("F15", "ps_sls_glob", "Select-String -Pattern seed -Path src\\*.py", "B1")
    bash("F16", "cat_glob", "cat scripts/*.py", "B1")
    bash("F17", "for_glob", 'for f in scripts/*.py; do wc -l "$f"; done', "B1")
    bash("F18", "locate", "locate smoke_", "B1")
    bash("F19", "tar_list_export", "tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar", "B1")
    bash("F20", "assigned_var_data", 'SP=/workspace/plantseg_data; ls -R "$SP"', "B1")
    bash("F21", "unresolved_var", 'ls -R "$SL1_UNSET_DIR"', "B1")
    bash("F22", "du_summary_glob", "du -sh /workspace/plantseg_runs/*", "B1")
    bash("F23", "echo_glob_expansion", "echo scripts/*", "B1")
    # G: filter and count exemptions
    bash("G01", "grep_r_filtered", "grep -rn seed src | grep -vi test", None)
    bash("G02", "find_filtered_head", "find . -name '*.py' | grep -iv test | head -n 20", None)
    bash("G03", "git_ls_files_filtered", "git ls-files | grep -v -i test", None)
    bash("G04", "git_grep_filtered_long", "git grep -n seed | grep --invert-match --ignore-case test", None)
    pwsh("G05", "ps_gci_name_filtered", "Get-ChildItem -Recurse -Name src | Select-String -NotMatch test", None)
    pwsh("G06", "ps_git_filtered", "git ls-files | Select-String -NotMatch -Pattern test", None)
    bash("G07", "tree_f_filtered", "tree -f src | grep -vi test", None)
    bash("G08", "cmd_dir_s_b_filtered", 'cmd /c "dir /s /b" | grep -vi test', None)
    pwsh("G09", "ps_gci_fileinfo_into_sls", "Get-ChildItem -Recurse src | Select-String -NotMatch test", "B1")
    bash("G10", "ls_R_filtered_leaks_members", "ls -R docs | grep -vi test", "B1")
    bash("G11", "tree_filtered_leaks_members", "tree src | grep -vi test", "B1")
    bash("G12", "cmd_dir_s_not_bare", 'cmd /c "dir /s" | grep -vi test', "B1")
    bash("G13", "grep_h_filtered", "grep -rhn seed src | grep -vi test", "B1")
    bash("G14", "find_exec_filtered", "find . -name '*.py' -exec cat {} + | grep -vi test", "B1")
    bash("G15", "filter_word_regexp", "grep -rn seed src | grep -viw test", "B1")
    bash("G16", "filter_case_sensitive", "grep -rn seed src | grep -v test", "B1")
    bash("G17", "tee_before_filter", f"find . | tee {q(px(S / 'l.txt'))} | grep -vi test", "B1")
    bash("G18", "redirect_listing", f"find . -name '*.py' > {q(px(S / 'list.txt'))}", "B1")
    bash("G19", "xargs_from_safe_list", f"git ls-files -- src {TEST_EXCL} | xargs grep -n seed", None)
    bash("G20", "xargs_from_unsafe_list", "git ls-files | xargs grep -n seed", "B2")
    bash("G21", "rg_heading_filtered", "rg --heading seed src | grep -vi test", "B1")
    bash("G22", "rg_pretty_filtered", "rg -p seed src | grep -vi test", "B1")
    bash("G23", "git_grep_h_filtered", "git grep -h seed | grep -vi test", "B2")
    bash("G24", "git_grep_heading_filtered", "git grep --heading seed | grep -vi test", "B2")
    bash("G25", "tee_before_count", f"ls -R docs | tee {q(px(S / 'x.txt'))} | wc -l", "B1")
    bash("G26", "find_exec_sha256_run_folder", "find /workspace/plantseg_runs/e1_s42 -type f -exec sha256sum {} +",
         "B1")
    # H: git
    bash("H01", "git_ls_files_bare", "git ls-files", "B2")
    bash("H02", "git_grep_no_exclude", "git grep -n seed", "B2")
    bash("H03", "git_grep_exclude_no_icase", "git grep seed -- ':(exclude)*test*'", "B2")
    bash("H04", "git_literal_pathspecs", f"git --literal-pathspecs ls-files -- {TEST_EXCL}", "B2")
    bash("H05", "git_diff_stat_bare", "git diff --stat", "B2")
    bash("H06", "git_diff_name_only_pdf_only", f"git diff --name-only -- . {PDF_EXCL}", "B2")
    bash("H07", "git_show_stat", "git show --stat HEAD", "B2")
    bash("H08", "git_log_name_status", "git log --name-status -3", "B2")
    bash("H09", "git_log_numstat_excludes", f"git log --numstat -3 -- . {PDF_EXCL} {TEST_EXCL}", None)
    bash("H10", "git_show_patch", "git show HEAD", "B6")
    bash("H11", "git_diff_pdf_exclude_only", f"git diff -- . {PDF_EXCL}", "B6")
    bash("H12", "git_show_test_blob", "git show HEAD:scripts/test_teacher_init.py", "B4")
    bash("H13", "git_show_tree", "git show HEAD:.claude/hooks", "B2")
    bash("H14", "git_ls_tree_r", "git ls-tree -r HEAD", "B2")
    bash("H15", "git_ls_tree_r_filtered", "git ls-tree -r HEAD | grep -vi test", None)
    bash("H16", "git_cat_file_tree", "git cat-file -p HEAD:.claude", "B2")
    bash("H17", "git_C_scratch_ls_files", f"git -C {q(px(S))} ls-files", "B2")
    bash("H18", "git_diff_stat_explicit_file", "git diff --stat -- src/eval/metrics.py", None)
    bash("H19", "git_literal_env_prefix", f"GIT_LITERAL_PATHSPECS=1 git ls-files -- {TEST_EXCL}", "B2")
    bash("H20", "git_archive", "git archive HEAD", "B6")
    # I: pytest
    bash("I01", "pytest_bare", "pytest", "B3")
    bash("I02", "python_m_pytest", "python -m pytest -q scripts", "B3")
    bash("I03", "py_launcher_pytest", "py -m pytest -q", "B3")
    bash("I04", "bash_c_pytest", 'bash -c "cd scripts && pytest -x"', "B3")
    bash("I05", "python_m_unittest", "python -m unittest discover", "B3")
    # J: test-named and NAMED_DENY paths
    bash("J01", "cat_test", "cat scripts/test_teacher_init.py", "B4")
    bash("J02", "head_test", "head -n 5 src/tests/x.py", "B4")
    bash("J03", "tail_test", "tail TEST_split.txt", "B4")
    bash("J04", "less_test", "less docs/testing.md", "B4")
    bash("J05", "more_test", "more Tests/a.txt", "B4")
    bash("J06", "sed_test", "sed -n '1,10p' scripts/test_x.py", "B4")
    bash("J07", "awk_test", "awk '{print}' data/test.csv", "B4")
    pwsh("J08", "ps_get_content_test", "Get-Content scripts\\test_x.py", "B4")
    pwsh("J09", "ps_type_test", "type .\\tests\\a.txt", "B4")
    bash("J10", "cat_latest_literal", "cat reports/latest.md", "B4")
    bash("J11", "grep_test_file", "grep seed scripts/test_x.py", "B4")
    bash("J12", "redirect_in_test", "wc -l < scripts/test_x.py", "B4")
    bash("J13", "tail_contest_outside", f"tail -n 50 {q(px(S / 'contest.log'))}", "B4")
    bash("J14", "python_test_script", "python -B scripts/test_teacher_init.py", "B7")
    bash("J15", "sh_test_script", "sh tools/run_tests.sh", "B7")
    bash("J16", "cat_named_csv", f"cat {q(px(S / 'plantseg_exact_duplicates.csv'))}", "B4")
    bash("J17", "head_named_zip_bytes", "head -c 64 /workspace/plantseg_data/plantseg_v3.zip", "B4")
    bash("J18", "ls_named_glob", "ls -la ~/Downloads/plantseg*.zip", "B4")
    bash("J19", "md5_named_csv", f"md5sum {q(px(S / 'plantseg_exact_duplicates.csv'))}", "B4")
    bash("J20", "stat_test_path", "stat scripts/test_x.py", "B4")
    # K: Monitor
    add("K01", "monitor_tail_log", "Monitor", {"command": f"tail -f {q(px(S / 'run.log'))} | grep --line-buffered "
                                                         "ERROR", "description": "x", "timeout_ms": 60000}, None)
    add("K02", "monitor_grep_r", "Monitor", {"command": "grep -r seed src", "description": "x",
                                             "timeout_ms": 60000}, "B1")
    add("K03", "monitor_websocket", "Monitor", {"ws": {"url": "wss://example.invalid/x"}, "description": "x",
                                                "timeout_ms": 60000}, None)
    # L: destructive
    bash("L01", "docker_rmi", "docker rmi img:tag", "D1")
    bash("L02", "docker_image_rm", "docker image rm img", "D1")
    bash("L03", "docker_image_prune", "docker image prune -a", "D1")
    bash("L04", "docker_system_prune", "docker system prune -af", "D1")
    bash("L05", "docker_volume_prune", "docker volume prune -f", "D1")
    bash("L06", "docker_global_flag_rmi", "docker --context x rmi img", "D1")
    bash("L07", "docker_image_ls", "docker image ls", None)
    bash("L08", "rm_runs", "rm -rf /workspace/plantseg_runs/e1_s42", "D2")
    bash("L09", "rm_data", "rm -f /workspace/plantseg_data/x.zip", "D2")
    bash("L10", "rm_repo_file", "rm scripts/old.py", "D2")
    bash("L11", "rm_repo_root", f"rm -rf {q(px(R))}", "D2")
    bash("L12", "rm_repo_parent", f"rm -rf {q(px(P))}", "D2")
    bash("L13", "rm_root", "rm -rf /", "D2")
    bash("L14", "rm_scratch", f"rm -f {q(px(S / 'x.txt'))}", None)
    bash("L15", "rmdir_repo", "rmdir docs/tmpdir", "D2")
    bash("L16", "cmd_del_runs", 'cmd /c "del /q C:\\work\\plantseg_runs\\e1\\x.pt"', "D2")
    bash("L17", "cmd_rd_repo", 'cmd /c "rd /s /q C:\\work\\plantseg-thesis\\tmp"', "D2")
    pwsh("L18", "ps_remove_item_data", "Remove-Item -Recurse -Force C:\\data\\plantseg_data\\cache", "D2")
    pwsh("L19", "ps_ri_alias_repo", "ri .\\tmp.txt", "D2")
    bash("L20", "rm_unresolved_var", 'rm -rf "$SL1_UNSET_VAR/x"', "D2")
    pwsh("L21", "ps_remove_item_scratch", f"Remove-Item {qps(S / 'x.txt')}", None)
    bash("L22", "rm_glob_runs", "rm -f /workspace/plantseg_runs/*.log", "D2")
    bash("L23", "rm_data_root_env", 'rm -rf "$PLANTSEG_DATA_ROOT"', "D2",
         env={"PLANTSEG_DATA_ROOT": px(S / "pd")})
    bash("L24", "docker_volume_rm", "docker volume rm vol1", "D1")
    bash("L25", "git_clean_force", "git clean -fdx", "D2")
    bash("L26", "find_delete_runs", "find /workspace/plantseg_runs -name '*.tmp' -delete", "D2")
    bash("L27", "rm_other_clone", f"rm -rf {q(px(CLONE / 'src'))}", "D2", cwd=S)
    # M: nesting and parsing
    bash("M01", "bash_c_grep_r", 'bash -c "grep -r seed src"', "B1")
    bash("M02", "subst_ls_R", 'echo "$(ls -R docs)"', "B1")
    bash("M03", "backtick_find", "echo `find . -name x`", "B1")
    bash("M04", "sudo_find", "sudo find / -name x", "B1")
    bash("M05", "eval_find", 'eval "find . -type f"', "B1")
    bash("M06", "heredoc_body_is_data", f"cat > {q(px(S / 'x.md'))} <<'EOF'\ngrep -r seed src\nEOF", None)
    bash("M07", "unbalanced_quote", 'grep "seed src', "B5")
    bash("M08", "env_prefix", "LC_ALL=C grep -r seed src", "B1")
    pwsh("M09", "ps_nested_fullname", "(Get-ChildItem -Recurse src).FullName", "B1")
    bash("M10", "pwsh_command", 'pwsh -NoProfile -Command "Get-ChildItem -Recurse"', "B1")
    pwsh("M11", "ps_encoded_command", "powershell -EncodedCommand ZQBjAGgAbwA=", "B5")
    # N: archives
    bash("N01", "tar_list_data_not_named", "tar -tzf /workspace/plantseg_data/plantseg.tar.gz", "B1")
    bash("N02", "unzip_l_named", "unzip -l /workspace/plantseg_data/plantseg.zip", "B8")
    bash("N03", "7z_l_named", "7z l ~/Downloads/plantseg_v3.zip", "B8")
    bash("N04", "tar_create_scratch", f"tar -czf {q(px(S / 'out.tgz'))} -C {q(px(S))} notes", None)
    bash("N05", "python_zipfile_named", "python -m zipfile -l ~/Downloads/plantseg.zip", "B8")
    pwsh("N06", "ps_expand_archive_named", "Expand-Archive C:\\dl\\plantseg_v2.zip -DestinationPath C:\\x", "B8")
    bash("N07", "unzip_zenodo", "unzip b66_harness/inputs/zenodo_17/data.zip -d /tmp/x", "B8")
    bash("N08", "python_tarfile_export", "python -m tarfile -l /workspace/plantseg_runs/smoke_afd2d33.tar", "B1")
    bash("N09", "tar_test_named_archive", f"tar -tf {q(px(S / 'test_kit.tar'))}", "B8")
    # W: Windows path forms (C:\x, C:/x and /c/x are one path)
    bash("W01", "find_msys_home", "find /c/Users/admin -name '*.json'", "B1", cwd=WIN_CWD, env=WIN_ENV)
    bash("W02", "rm_msys_runs", "rm -rf /c/Users/admin/plantseg_runs/e1", "D2", cwd=WIN_CWD, env=WIN_ENV)
    bash("W03", "grep_msys_repo", "grep -rn seed /c/Users/admin/plantseg-thesis/src", "B1", cwd=WIN_CWD,
         env=WIN_ENV)
    bash("W04", "grep_msys_case_only", "grep -rn seed /c/users/ADMIN/thesis2/src", "B1", cwd=WIN2_CWD,
         env=WIN2_ENV)
    add("W05", "grep_tool_forward_slashes", "Grep", {"pattern": "seed", "path": "C:/Users/Admin/Thesis2/docs"},
        "G1", cwd=WIN2_CWD, env=WIN2_ENV)
    bash("W06", "ls_R_drive_ancestor", "ls -R C:/Users/admin", "B1", cwd=WIN2_CWD, env=WIN2_ENV)
    pwsh("W07", "ps_remove_backslash", "Remove-Item -Recurse C:\\USERS\\admin\\thesis2\\tmp", "D2", cwd=WIN2_CWD,
         env=WIN2_ENV)


# ----------------------------------------------------------------------------------------- running
def find_sh():
    cands = [shutil.which("sh")]
    git = shutil.which("git")
    if git:
        base = Path(git).resolve().parents[1]
        cands += [str(base / "bin" / "sh.exe"), str(base / "usr" / "bin" / "sh.exe")]
    cands += [r"C:\Program Files\Git\bin\sh.exe", r"C:\Program Files\Git\usr\bin\sh.exe"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


SH = find_sh()


def base_env(log, extra=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PLANTSEG_DATA_ROOT", "SL1_GUARD_PYTHONS") and k not in g.BAD_PATHSPEC_ENV}
    env["CLAUDE_PROJECT_DIR"] = str(R)
    env["SL1_GUARD_LOG"] = str(log)
    env.update(extra or {})
    return env


def payload_of(c):
    return {"session_id": "sl1-smoke", "transcript_path": "", "cwd": c["cwd"], "hook_event_name": "PreToolUse",
            "tool_name": c["tool"], "tool_input": c["input"]}


_COPIES = {}


def copy_of(mode, exceptions=None, crash=False, policy=True):
    """A temporary checkout-like directory holding the launcher and a policy with MODE (and EXCEPTIONS)
    rewritten. Never inside the repository."""
    key = (mode, exceptions, crash, policy)
    if key in _COPIES:
        return _COPIES[key]
    root = TMP / f"copy{len(_COPIES)}"
    hooks = root / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, hooks / "sl1_guard.sh")
    if policy:
        text = POLICY.read_text(encoding="utf-8")
        text = re.sub(r'(?m)^MODE = "[a-z]*"$', f'MODE = "{mode}"', text, count=1)
        if exceptions is not None:
            text = re.sub(r"(?m)^EXCEPTIONS: tuple = \(\)", f"EXCEPTIONS: tuple = {exceptions!r}", text, count=1)
        if crash:
            text = text.replace('\nMODE = "', '\nraise RuntimeError("injected crash")\nMODE = "', 1)
        with open(hooks / "sl1_guard.py", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    _COPIES[key] = root
    return root


def launch(root, payload, env):
    """Run <root>/.claude/hooks/sl1_guard.sh (root None = the repository) like Claude Code does."""
    launcher = (root / ".claude" / "hooks" / "sl1_guard.sh") if root else LAUNCHER
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    t0 = time.perf_counter()
    p = subprocess.run([SH, launcher.as_posix()], input=data, capture_output=True, env=env, timeout=60)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace"), (time.perf_counter() - t0) * 1000


def log_lines(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    except FileNotFoundError:
        return []


RESULTS = []          # (group, id, name, ok, detail)
TIMES = []
STDOUT_CLEAN = []
DENY_LINES_OK = []
COUNTS = {}           # group -> [decide ok, n, log ok, n, deny ok, n]


def record(group, cid, name, ok, detail=""):
    RESULTS.append((group, cid, name, bool(ok), detail))


def run_decision_case(c):
    payload = payload_of(c)
    want = None if c["expect"] is None else "SL1-" + c["expect"]
    env = base_env(LOGS / "inproc.log", c["env"])
    saved = g.EXCEPTIONS
    try:
        if c["exceptions"] is not None:
            g.EXCEPTIONS = c["exceptions"]
        f = g.decide(payload, env)
        got = None if f is None else "SL1-" + f.rule
    except Exception as e:  # a crash is a failure of this case
        got = f"crash {type(e).__name__}: {e}"
    finally:
        g.EXCEPTIONS = saved
    ok_d = got == want
    details = [f"decide={got or 'allow'}"]
    cnt = COUNTS.setdefault(c["group"], [0, 0, 0, 0, 0, 0])
    cnt[0] += ok_d
    cnt[1] += 1
    ok_modes = []
    if SH is None:
        details.append("no sh found")
        ok_modes = [False, False]
    else:
        for idx, mode in enumerate(("log", "deny")):
            if mode == g.MODE and c["exceptions"] is None:
                root = None
            else:
                root = copy_of(mode, c["exceptions"])
            log = LOGS / f"{mode}_{c['id']}.log"
            rc, out, err, ms = launch(root, payload, base_env(log, c["env"]))
            if mode == "log":
                TIMES.append(ms)
            STDOUT_CLEAN.append(out == b"")
            lines = log_lines(log)
            if want:
                log_ok = len(lines) == 1 and len(lines[0]) == 4 and lines[0][2] == want
            else:
                log_ok = not lines
            if mode == "log":
                ok = rc == 0 and out == b"" and log_ok
            elif want:
                errl = err.strip().splitlines()
                line_ok = len(errl) == 1 and errl[0].startswith(f"[SL1 {c['expect']}]") and "Allowed:" in errl[0]
                DENY_LINES_OK.append(line_ok)
                ok = rc == 2 and out == b"" and log_ok and line_ok
            else:
                ok = rc == 0 and out == b"" and log_ok and not err.strip()
            ok_modes.append(ok)
            cnt[2 + 2 * idx] += ok
            cnt[3 + 2 * idx] += 1
            details.append(f"{mode}={'ok' if ok else f'FAIL(rc={rc}, log={lines}, err={err.strip()[:120]!r})'}")
    record(c["group"], c["id"], c["name"], ok_d and all(ok_modes),
           f"expect={want or 'allow'} " + " ".join(details))


def run_launcher_checks():
    slip = {"session_id": "s", "cwd": str(R), "hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": {"command": "grep -rn seed src/"}}
    read_ok = {"session_id": "s", "cwd": str(R), "tool_name": "Read", "tool_input": {"file_path": str(POLICY)}}
    read_test = {"session_id": "s", "cwd": str(R), "tool_name": "Read",
                 "tool_input": {"file_path": str(R / "scripts" / "test_x.py")}}
    read_named = {"session_id": "s", "cwd": str(R), "tool_name": "Read",
                  "tool_input": {"file_path": str(S / "plantseg_exact_duplicates.csv")}}
    write_content = {"session_id": "s", "cwd": str(R), "tool_name": "Write",
                     "tool_input": {"file_path": str(S / "notes.md"),
                                    "content": "a test line; \"file_path\": \"tests/x\""}}
    no_path = {"session_id": "s", "cwd": str(R), "tool_name": "Read", "tool_input": {}}
    if SH is None:
        record("O", "O00", "sh_available", False, "no sh found: launcher checks cannot run")
        return
    nopy = {"SL1_GUARD_PYTHONS": "sl1-no-such-python"}
    deny = copy_of("deny")
    logc = copy_of("log")

    def chk(cid, name, root, payload, extra, want_rc, want_rule=None, stderr_has=None):
        log = LOGS / f"{cid}.log"
        rc, out, err, _ = launch(root, payload, base_env(log, extra))
        lines = log_lines(log)
        ok = rc == want_rc and out == b""
        if want_rule:
            ok = ok and any(len(ln) == 4 and ln[2] == want_rule for ln in lines)
        elif want_rule is None and want_rc == 0 and stderr_has is None:
            pass
        if stderr_has:
            ok = ok and stderr_has in err
        record("O", cid, name, ok, f"rc={rc} log={lines} err={err.strip()[:160]!r}")

    chk("O04", "no_python_log_mode_logged", logc, slip, nopy, 0, "SL1-E2")
    chk("O05", "no_python_deny_bash", deny, slip, nopy, 2, "SL1-E2", "[SL1 E2]")
    chk("O06", "no_python_deny_read_ok", deny, read_ok, nopy, 0, "SL1-E2")
    chk("O07", "no_python_deny_read_test", deny, read_test, nopy, 2, "SL1-E2")
    chk("O08", "no_python_deny_read_named", deny, read_named, nopy, 2, "SL1-E2")
    chk("O09", "no_python_deny_write_content_test", deny, write_content, nopy, 0, "SL1-E2")
    chk("O10", "no_python_deny_unreadable_path", deny, no_path, nopy, 2, "SL1-E2")
    crash_log = copy_of("log", crash=True)
    crash_deny = copy_of("deny", crash=True)
    chk("O11", "policy_import_crash_log", crash_log, slip, {}, 0, "SL1-E2")
    chk("O12", "policy_import_crash_deny_bash", crash_deny, slip, {}, 2, "SL1-E2")
    chk("O13", "policy_import_crash_deny_read_ok", crash_deny, read_ok, {}, 0, "SL1-E2")
    chk("O14", "policy_file_missing", copy_of("deny", policy=False), slip, {}, 0, "SL1-E2")
    chk("O17", "non_json_stdin_log", None if g.MODE == "log" else logc, b"not json {", {}, 0, "SL1-E1")
    log = LOGS / "O18.log"
    rc, out, err, _ = launch(None if g.MODE == "log" else logc, b"", base_env(log))
    record("O", "O18", "empty_stdin", rc == 0 and out == b"" and not log_lines(log), f"rc={rc}")

    # in-process: decide() raises
    saved = g.decide

    def boom(payload, env):
        raise RuntimeError("injected")

    g.decide = boom
    try:
        log = LOGS / "O15.log"
        r1 = g.run(json.dumps(slip).encode(), base_env(log), "log")
        record("O", "O15", "policy_exception_log_mode", r1 == (0, "") and any(ln[2] == "SL1-E1" for ln in
                                                                             log_lines(log)), f"{r1}")
        r2 = g.run(json.dumps(slip).encode(), base_env(LOGS / "O16.log"), "deny")
        r3 = g.run(json.dumps(read_ok).encode(), base_env(LOGS / "O16.log"), "deny")
        record("O", "O16", "policy_exception_deny_fail_mode_a",
               r2[0] == 3 and r2[1].startswith("[SL1 E1]") and r3 == (0, ""), f"bash={r2} read={r3}")
    finally:
        g.decide = saved

    # the settings.json command itself
    empty = TMP / "no_launcher"
    empty.mkdir()
    for cid, name, proj, want in (("O19", "settings_cmd_launcher_missing", empty, 0),
                                  ("O20", "settings_cmd_real_files", R, 2 if g.MODE == "deny" else 0),
                                  ("O21", "settings_cmd_deny_copy", deny, 2)):
        log = LOGS / f"{cid}.log"
        env = base_env(log, {"CLAUDE_PROJECT_DIR": str(proj)})
        p = subprocess.run([SH, "-c", NEW_COMMAND], input=json.dumps(slip).encode(), capture_output=True,
                           env=env, timeout=60)
        lines = log_lines(log)
        ok = p.returncode == want and p.stdout == b""
        if proj is not empty:
            ok = ok and any(ln[2] == "SL1-B1" for ln in lines)
        record("O", cid, name, ok, f"rc={p.returncode} log={lines}")

    # log line format and truncation
    log = LOGS / "O22.log"
    long = dict(slip, tool_input={"command": "ls -R docs " + "x" * 400})
    launch(None if g.MODE == "log" else logc, long, base_env(log))
    lines = log_lines(log)
    fmt = (len(lines) == 1 and len(lines[0]) == 4 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lines[0][0])
           and lines[0][1] == "Bash" and lines[0][2] == "SL1-B1" and len(lines[0][3]) == 200)
    record("O", "O22", "log_line_format_and_200_cut", fmt, f"{[(ln[0], ln[1], ln[2], len(ln[3])) for ln in lines]}")

    # a log path inside the repository is refused (HOME points into the temp dir for this call)
    home = TMP / "home"
    home.mkdir()
    inside = R / "sl1_guard_probe.log"
    lp = g.log_path({"CLAUDE_PROJECT_DIR": str(R), "SL1_GUARD_LOG": str(inside)})
    env = base_env(inside, {"HOME": str(home), "USERPROFILE": str(home)})
    rc, out, err, _ = launch(None if g.MODE == "log" else logc, slip, env)
    sh_ok = not inside.exists() and (home / ".claude" / "sl1_guard.log").exists()
    record("O", "O23", "log_path_inside_repo_refused",
           lp is not None and not str(Path(lp).resolve()).startswith(str(R)) and sh_ok and rc == 0,
           f"policy->{lp} launcher_default_used={sh_ok}")


def run_static_checks():
    text = POLICY.read_text(encoding="utf-8")
    modes = re.findall(r'(?m)^MODE = "([a-z]*)"$', text)
    sh_text = LAUNCHER.read_text(encoding="utf-8")
    record("O", "O24", "single_mode_line_parsed_by_launcher",
           modes == [g.MODE] and g.MODE in ("log", "deny") and 's/^MODE = "\\([a-z]*\\)"$/\\1/p' in sh_text,
           f"MODE lines={modes}")
    record("O", "O35", "named_deny_and_exceptions_at_tip",
           g.NAMED_DENY == ("plantseg_exact_duplicates.csv", "plantseg*.zip", "zenodo_*") and g.EXCEPTIONS == (),
           f"NAMED_DENY={g.NAMED_DENY} EXCEPTIONS={g.EXCEPTIONS}")
    try:
        cur = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception as e:
        cur = {}
        record("O", "O26", "settings_parse", False, str(e))
    base = None
    try:
        out = subprocess.run(["git", "-C", str(R), "show", f"{BASE}:.claude/settings.json"], capture_output=True,
                             timeout=60)
        if out.returncode == 0:
            base = json.loads(out.stdout.decode("utf-8"))
    except Exception:
        base = None
    pre = (cur.get("hooks") or {}).get("PreToolUse") or []
    record("O", "O26", "existing_entry_equals_c689634",
           bool(pre) and pre[0] == OLD_ENTRY and base is not None and base["hooks"]["PreToolUse"][0] == pre[0],
           "c689634 unavailable (run git fetch --unshallow)" if base is None else "")
    record("O", "O27", "permissions_equal_c689634",
           cur.get("permissions") == OLD_PERMISSIONS and base is not None and base.get("permissions") ==
           cur.get("permissions"), "")
    same_rest = base is not None and {k: v for k, v in cur.items() if k != "hooks"} == \
        {k: v for k, v in base.items() if k != "hooks"} and set(cur.get("hooks", {})) == {"PreToolUse"}
    record("O", "O28", "new_entry_exact_and_nothing_else",
           len(pre) == 2 and pre[1] == NEW_ENTRY and same_rest, f"entries={len(pre)}")
    for cid, name, extra in (("O29", "hooks_unchanged_c689634_to_HEAD", ["HEAD"]),
                             ("O30", "hooks_unchanged_c689634_to_worktree", [])):
        p = subprocess.run(["git", "-C", str(R), "diff", "--quiet", BASE] + extra +
                           ["--", ".claude/hooks", ":(exclude).claude/hooks/sl1_guard.*"],
                           capture_output=True, timeout=60)
        record("O", cid, name, p.returncode == 0, f"exit={p.returncode}")
    record("O", "O31", "new_names_clean", all(not g.sensitive(n) for n in NEW_FILES), ", ".join(NEW_FILES))


def main():
    build_cases()
    names = [c["id"] for c in CASES]
    assert len(names) == len(set(names)), "duplicate case ids"
    assert not str(TMP.resolve()).startswith(str(R.resolve())), "temp dir inside the repository"
    print(f"SL1 guard smoke  repo={R}  MODE={g.MODE}  sh={SH}  python={sys.executable}")
    try:
        for c in CASES:
            run_decision_case(c)
        run_launcher_checks()
        run_static_checks()
        p95 = statistics.quantiles(TIMES, n=20)[-1] if len(TIMES) >= 20 else (max(TIMES) if TIMES else 0.0)
        med = statistics.median(TIMES) if TIMES else 0.0
        mx = max(TIMES) if TIMES else 0.0
        record("O", "O25", "latency_p95_under_200ms", bool(TIMES) and p95 < 200.0,
               f"median={med:.1f} p95={p95:.1f} max={mx:.1f} ms over {len(TIMES)} log-mode launcher calls")
        record("O", "O32", "stdout_empty_every_call", bool(STDOUT_CLEAN) and all(STDOUT_CLEAN),
               f"{sum(STDOUT_CLEAN)}/{len(STDOUT_CLEAN)}")
        record("O", "O33", "deny_stderr_one_line_with_allowed_form", bool(DENY_LINES_OK) and all(DENY_LINES_OK),
               f"{sum(DENY_LINES_OK)}/{len(DENY_LINES_OK)}")
        after = (REAL_LOG.exists(), REAL_LOG.stat().st_size if REAL_LOG.exists() else -1)
        record("O", "O34", "real_log_untouched", after == REAL_LOG_BEFORE, f"before={REAL_LOG_BEFORE} after={after}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    for group, cid, name, ok, detail in sorted(RESULTS, key=lambda r: r[1]):
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {name}")
        if not ok or os.environ.get("SL1_SMOKE_VERBOSE"):
            print(f"         {detail}")
    print("\nPer group (decision cases: in-process / log mode / deny mode):")
    for grp, (d_ok, d_n, l_ok, l_n, x_ok, x_n) in sorted(COUNTS.items()):
        print(f"  {grp} {GROUPS[grp]:<34} {d_ok}/{d_n}  {l_ok}/{l_n}  {x_ok}/{x_n}")
    o = [r for r in RESULTS if r[0] == "O"]
    print(f"  O {GROUPS['O']:<34} {sum(r[3] for r in o)}/{len(o)}")
    timing = next((r[4] for r in RESULTS if r[1] == "O25"), "")
    print(f"Timing: {timing}")
    good = sum(1 for r in RESULTS if r[3])
    total = len(RESULTS)
    verdict = "PASS" if good == total else "FAIL"
    print(f"SUMMARY  {good}/{total} named cases passed ({len(CASES)} decision cases x in-process, log, deny; "
          f"{len(o)} launcher/settings/invariant checks)")
    print(f"RESULT {verdict}: smoke_sl1_guard {good}/{total} (MODE={g.MODE})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
