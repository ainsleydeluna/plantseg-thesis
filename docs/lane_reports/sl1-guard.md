# Lane report: L-SL1-GUARD (SL1 · hook guard, cloud)

The lane adds the SL-1 PreToolUse guard in **log mode**.

| | |
|---|---|
| Date | 2026-10-04 |
| Session | cloud lane session (DL-37) |
| Branch | `lane/sl1-guard` |
| Base | `6067e662b1d5f4036b38cef7179f8cd94b56e50e` |
| GO | orchestrator, 2026-10-04 23:23 |

Labels: MEASURED (run in this session), DOCUMENTED, REPOSITORY-PROVEN, INFERRED.

## 1. Outcome

- **Last code commit:** `648d38813644ba0cef1ef21e4d52f11a86e1ce8a`, "Add the SL-1 PreToolUse guard in log mode (lane L-SL1-GUARD)".
- **Final tip:** this report is the lane's last commit, so the final tip SHA is given in the session's chat reply.
- **Mode:** `MODE = "log"` at the tip. No commit carries a trailer line.

| Commit | Files |
|---|---|
| `648d388` | M `.claude/settings.json` (blob `022114bb`)<br>A `.claude/hooks/sl1_guard.sh` (`dac9b4c2`)<br>A `.claude/hooks/sl1_guard.py` (`ae38007e`)<br>A `scripts/smoke_sl1_guard.py` (`2a67160c`) |
| this commit | A `docs/lane_reports/sl1-guard.md` |

## 2. Base and branch: PB-1a outputs (MEASURED)

**Step 1**
- `git fetch origin master` exited 0. FETCH_HEAD and `git ls-remote origin master` both returned `6067e662b1d5f4036b38cef7179f8cd94b56e50e`.
- `git merge-base --is-ancestor c689634… 6067e66…` exited 0.
- `git diff --quiet c689634… 6067e66… -- .claude` exited 0.
- The clone was already unshallowed in Phase A, as DL-37 requires. Safety floor `885523a` is an ancestor.

**Step 2: recorded before the chain**
- `git symbolic-ref HEAD` returned `refs/heads/claude/practical-mayer-3owe6m`.
- `git rev-parse HEAD` returned `c689634617ef9b59473b1a6c5ea64bc99cb08a29`.

**Pre-checks** (both printed nothing):
- `git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf'`
- `git ls-remote --heads origin lane/sl1-guard`

**Chain:** `git branch lane/sl1-guard 6067e66… && git symbolic-ref HEAD refs/heads/lane/sl1-guard && git checkout --no-overlay HEAD -- . ':(exclude)docs/reference/reference.pdf'` exited 0.

**Post-checks**
- `git rev-parse HEAD`: `6067e662b1d5f4036b38cef7179f8cd94b56e50e`
- `git symbolic-ref HEAD`: `refs/heads/lane/sl1-guard`
- Status with the protected-file exclude: empty
- `git diff --quiet 6067e66… -- . <exclude>`: exit 0
- `git diff --cached --quiet 6067e66… -- . <exclude>`: exit 0
- The protected-file status probe, run once as a whole command: empty

**Step 3 reads:** `sed -n 1p docs/lane_specs/part1.md` (the smoke convention), and `scripts/smoke_frozen_blobs.py`. No other docs file was read and no folder was listed.

## 3. Acceptance (MEASURED)

These were measured on the staged tree immediately before this report's commit. The index is exactly what HEAD holds after the commit. The chat reply repeats them against the final tip.

**Changed paths.** `git diff --name-status 6067e66… HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'` (run as `--cached` against the index) printed exactly these five paths, in git's order:

```
A	.claude/hooks/sl1_guard.py
A	.claude/hooks/sl1_guard.sh
M	.claude/settings.json
A	docs/lane_reports/sl1-guard.md
A	scripts/smoke_sl1_guard.py
```

**Nothing else changed.** `git diff --quiet 6067e66… HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude).claude/settings.json' ':(exclude).claude/hooks/sl1_guard.sh' ':(exclude).claude/hooks/sl1_guard.py' ':(exclude)scripts/smoke_sl1_guard.py' ':(exclude)docs/lane_reports/sl1-guard.md'; echo $?` printed `0`. That covers AGENTS.md, CLAUDE.md and every docs path except this report.

**Smokes**
- `python -B scripts/smoke_sl1_guard.py` printed `RESULT PASS: smoke_sl1_guard 284/284 (MODE=log)`. Every case is named and both modes ran; O25 p95 was 81.2 ms.
- `python -B scripts/smoke_frozen_blobs.py` printed `RESULT: FROZEN BLOBS OK (10/10)`.

**Mode and trailers.** The tip has exactly one `MODE = "log"` line, and `6067e66..HEAD` has 0 trailer lines.

**Unchanged** (section 6):
- the existing PreToolUse entry (O26);
- the permissions block (O27);
- `protect_reference_pdf.*` and the self-test (O29 and O30, exit code only).

## 4. Design

| File | Role |
|---|---|
| `.claude/settings.json` | A second PreToolUse entry after the existing one.<br>Matcher `Read\|Edit\|Write\|NotebookEdit\|Grep\|Glob\|Bash\|PowerShell\|Monitor`.<br>Command `f="${CLAUDE_PROJECT_DIR}/.claude/hooks/sl1_guard.sh"; [ -r "$f" ] \|\| exit 0; sh "$f"`, timeout 10 s.<br>There is no `\|\| exit 2`: dash and python both exit 2 when a file is missing (MEASURED), and that would block even in log mode. |
| `.claude/hooks/sl1_guard.sh` | A POSIX sh launcher with the same sh-then-python form as L-PROT (`python`, then `python3`).<br>Policy exit 3 becomes 2 (blocks), and only when the policy's MODE line says `deny`. Any other exit code is a hook error (SL1-E2).<br>On a hook error it reads only `tool_name` and `file_path` / `notebook_path`, never content, `old_string` or `new_string`.<br>stdout stays empty. |
| `.claude/hooks/sl1_guard.py` | The policy: bash, cmd and PowerShell lexers, the rule table below, and the log writer.<br>It never opens, lists or hashes a file. It stats only paths that are neither test-named nor NAMED_DENY. |
| `scripts/smoke_sl1_guard.py` | 284 named cases (section 6). |

**Log**
- Location: `~/.claude/sl1_guard.log`. `SL1_GUARD_LOG` overrides it, but a path inside the project is refused.
- Line format: `UTC-time<TAB>tool<TAB>SL1-<rule><TAB>path-or-command[:200]`. Deny mode keeps logging.

**MODE flip**
- The flip is one line in `sl1_guard.py`: `MODE = "log"` becomes `MODE = "deny"`. The launcher reads that same line, and settings.json doesn't change.
- The smoke follows the committed MODE and always runs the other mode from temporary copies, so the flip commit needs no smoke edit.

**Fail mode (a) in deny mode (ruling Q4)**
- A hook error denies Bash, PowerShell, Monitor, Grep and Glob.
- It denies a file tool only when the path is test-named or NAMED_DENY, or the path field can't be read.
- A missing policy file, an unreadable MODE line, a missing launcher or a timeout never blocks, in either mode.

**Protected location**
- The project directory.
- Any directory inside a clone: one of its ancestors holds `.claude/hooks/protect_reference_pdf.sh`, found by a stat of that one file.
- Any path with a component `plantseg_data` or `plantseg_runs`, or starting with `plantseg-thesis`.
- `$PLANTSEG_DATA_ROOT`.

For recursive targets, ancestors of these also count, as do `/workspace` and the filesystem or drive roots. A target that can't be resolved (`$UNSET`, `~user`, `cd -`) counts as protected. Plain assignments made earlier in the same command line are resolved. `C:\x`, `C:/x` and `/c/x` are one path, compared case-insensitively.

**NAMED_DENY** = `plantseg_exact_duplicates.csv`, `plantseg*.zip`, `zenodo_*`: basename globs, case-insensitive, matched on every path component, anywhere on disk. Every rule treats them exactly like a test-named path, so a whole-file sha256 stays allowed and nothing else does. `EXCEPTIONS` is empty.

## 5. Decision table (final, with the GO rulings)

| Rule | Tool or pattern | Deny mode | Exempt by filter or count? |
|---|---|---|---|
| R1 | Read, Edit, Write, NotebookEdit on a test-named path (literal: "latest" and "contest" count) or a NAMED_DENY path, anywhere | deny | no |
| G1 | Grep or Glob over a directory, or with no path (cwd), in a protected location. Glob also uses an absolute pattern's fixed prefix. | deny | no |
| G2 | Grep or Glob with a test-named or NAMED_DENY path, Grep `glob`, or Glob `pattern` | deny | no |
| B1 | Recursive, globbed or directory listing or search in a protected location:<br>`grep -r/-R/-d recurse`, rg, ag, ack, find, fd, locate, `ls -R`, tree, du, `gci -Recurse/-Depth`, `dir /s`, `findstr /s`;<br>grep, ls or readers with a glob argument; grep or ls with a directory argument;<br>bare ls, dir, gci or tree; `for x in <glob>`; echo, printf, stat or du with a glob;<br>member listing of a non-named archive (tar -t, unzip -l, 7z l, `python -m tarfile -l`) | deny | **yes** |
| B2 | git ls-files / grep / ls-tree, show or cat-file of a tree, and diff/show/log/whatchanged/diff-tree names (`--stat`, `--numstat`, `--name-only`, `--name-status`, `--dirstat`, `--raw`, `--summary`, `--check`), all without `:(exclude,icase)*test*`.<br>The exclude doesn't count under `--literal-`, `--glob-` or `--noglob-pathspecs`, or the matching `GIT_*_PATHSPECS` variables.<br>Explicit existing non-test files are not a listing (Q3). | deny | **yes**, except `git grep -h/--heading` |
| B3 | pytest, py.test, `python/py -m pytest`, `-m unittest` | deny | no |
| B4 | A test-named or NAMED_DENY path given to a reader (cat, head, tail, less, more, type, Get-Content, sed, awk, plus extras), a search, a listing, stat, a git path, `rev:path` or a `<` redirect | deny | no |
| B5 | A command that can't be analysed (unbalanced quotes, nesting deeper than 8, `-EncodedCommand`, no command field) | deny | no |
| B6 | git patch or archive output (default `git diff` or `git show`, `-p`, `-L`, `git archive`) whose scope can include test paths | deny | no |
| B7 | Running a test-named script (python, sh/bash, node, pwsh -File, source) | deny | no |
| B8 | Listing or extracting members of a test-named or NAMED_DENY archive (tar, bsdtar, unzip, zipinfo, 7z, unrar, lsar/unar, `python -m zipfile/tarfile`, Expand-Archive) | deny | no |
| D1 | `docker rmi`, `docker image rm\|remove\|prune`, `docker system prune`, `docker volume prune\|rm\|remove` | deny | no |
| D2 | rm, rmdir, unlink, shred, del, erase, rd, Remove-Item and aliases; `git clean` without `-n`/`--dry-run`; `find … -delete`. Applies when the target is in or above a protected location, or can't be resolved. | deny | no |
| E1/E2 | A policy exception, or a launcher failure | logged; fail mode (a) | no |
| (allow) | sha256sum, `shasum -a 256`, `Get-FileHash` (SHA256), `certutil -hashfile … SHA256`; the builtins `test`, `[`, `[[`; git status; `git diff --quiet`; `git show -s`; plain git log; docker run; non-named archive extraction | allow | n/a |

**Filter exemption.** All of these must hold:
- A later stage is `grep` with `-v` and `-i` (any spelling) and the pattern exactly `test`, with no `-w`, `-x` or file arguments. In PowerShell, the equivalent is `Select-String -NotMatch test` with no `-CaseSensitive`, not fed FileInfo objects.
- Nothing before it tees or redirects stdout.
- The listing prints one full path per line.

Not filterable:
- `ls -R`; an `ls` glob without `-d`; `tree` without `-f`; `dir /s` without `/b`; `gci -Recurse` without `-Name`.
- `find -exec/-ok`; `grep -h`; `rg -I/--heading/-p/--pretty`; `git grep -h/--heading`.

**Count exemption (A4).** The pipeline ends in `wc -l`, `grep -c` with no files, or Measure-Object, or the listing sits inside `( … ).Count`, and nothing before the count tees or redirects. Nested `bash -c`, `cmd /c`, `pwsh -Command`, `eval` and `iex` hand their listings to the enclosing pipeline's filter or count.

## 6. Smoke results (MEASURED, cloud VM, Python 3.11.15, dash)

`python -B scripts/smoke_sl1_guard.py` (after the code commit) printed `RESULT PASS: smoke_sl1_guard 284/284 (MODE=log)`.

| Group | Decision cases | in-process | log mode | deny mode |
|---|---|---|---|---|
| A | file tools (rule 1) | 16/16 | 16/16 | 16/16 |
| B | Grep/Glob tools (rule 2) | 20/20 | 20/20 | 20/20 |
| C | recorded slips | 6/6 | 6/6 | 6/6 |
| D | everyday commands | 50/50 | 50/50 | 50/50 |
| E | bare listings | 9/9 | 9/9 | 9/9 |
| F | recursive and glob listings | 23/23 | 23/23 | 23/23 |
| G | filter and count exemptions | 26/26 | 26/26 | 26/26 |
| H | git | 20/20 | 20/20 | 20/20 |
| I | pytest | 5/5 | 5/5 | 5/5 |
| J | test-named and NAMED paths | 20/20 | 20/20 | 20/20 |
| K | Monitor | 3/3 | 3/3 | 3/3 |
| L | destructive (rule 4) | 27/27 | 27/27 | 27/27 |
| M | nesting and parsing | 11/11 | 11/11 | 11/11 |
| N | archives | 9/9 | 9/9 | 9/9 |
| W | Windows path forms | 7/7 | 7/7 | 7/7 |
| O | launcher, modes, settings, invariants | 32/32 | | |

**Timing** (O25: 252 log-mode launcher calls, wall time per call including sh, python and the decision):
- Median 61.4 ms, p95 81.2 ms, max 133.2 ms.
- The four runs in this session had p95 between 80.0 and 85.3 ms.

**`scripts/smoke_frozen_blobs.py`:** `RESULT: FROZEN BLOBS OK (10/10)`.

**Fuzz** (scratch, not committed): 20,012 random and malformed inputs, 0 policy crashes, worst random-case decision 4.1 ms. A 200 KB command takes 88 ms in process.

**Launcher, settings and invariant checks (group O):**

| ID | Check | Result |
|---|---|---|
| O04 | no Python, log mode: exit 0, SL1-E2 logged | PASS |
| O05 | no Python, deny mode, Bash: exit 2, `[SL1 E2]` on stderr | PASS |
| O06 | no Python, deny mode, Read of a non-test file: exit 0 | PASS |
| O07 | no Python, deny mode, Read of a test path: exit 2 | PASS |
| O08 | no Python, deny mode, Read of `plantseg_exact_duplicates.csv`: exit 2 | PASS |
| O09 | no Python, deny mode, Write whose content (not path) contains "test": exit 0 | PASS |
| O10 | no Python, deny mode, file-tool payload with no path: exit 2 | PASS |
| O11 | policy crashes at import, log mode: exit 0, SL1-E2 | PASS |
| O12 | policy crashes at import, deny mode, Bash: exit 2 | PASS |
| O13 | policy crashes at import, deny mode, Read of a non-test file: exit 0 | PASS |
| O14 | launcher present, policy file missing: exit 0 | PASS |
| O15 | `decide()` raises, log mode: exit 0, SL1-E1 logged | PASS |
| O16 | `decide()` raises, deny mode: Bash denied (`[SL1 E1]`), non-test Read allowed | PASS |
| O17 | non-JSON stdin: exit 0, SL1-E1 | PASS |
| O18 | empty stdin: exit 0, nothing logged | PASS |
| O19 | settings command, launcher missing: exit 0 | PASS |
| O20 | settings command, committed files, a slip: exit 0 and logged | PASS |
| O21 | settings command, deny-mode copy, a slip: exit 2 | PASS |
| O22 | log line: 4 tab fields, UTC time, subject cut to 200 characters | PASS |
| O23 | a log path inside the repository is refused (policy and launcher) | PASS |
| O24 | exactly one MODE line, parsed by the launcher's sed | PASS |
| O25 | p95 under 200 ms | PASS |
| O26 | existing PreToolUse entry equals c689634's (parsed JSON) | PASS |
| O27 | permissions block equals c689634's | PASS |
| O28 | new entry second, exact; nothing else in settings.json changed | PASS |
| O29 | `git diff --quiet c689634 HEAD -- .claude/hooks ':(exclude).claude/hooks/sl1_guard.*'` exits 0 (the self-test, by exit code only) | PASS |
| O30 | same, worktree against c689634 | PASS |
| O31 | no new file name is test-named or NAMED_DENY | PASS |
| O32 | stdout empty on every launcher call (504) | PASS |
| O33 | each deny is one stderr line with the rule id and `Allowed:` (171) | PASS |
| O34 | the real `~/.claude/sl1_guard.log` existence and size unchanged by the run (A1) | PASS |
| O35 | NAMED_DENY is exactly the three names and EXCEPTIONS is empty | PASS |

### Case table (all decision cases)

"vs Phase A": 62 cases are new, 3 changed, and the rest are as planned in Phase A. Phase A's O checks were renumbered to O04–O35 above.

| ID | Case | Tool | Input (R = repo root, $TMP = smoke temp dir) | Deny mode | vs Phase A |
|---|---|---|---|---|---|
| A01 | read_metrics | Read | `R/src/eval/metrics.py` | allow | Phase A |
| A02 | read_policy | Read | `R/.claude/hooks/sl1_guard.py` | allow | Phase A |
| A03 | read_test_script | Read | `R/scripts/test_teacher_init.py` | R1 | Phase A |
| A04 | read_latest_literal | Read | `R/reports/latest_summary.md` | R1 | Phase A |
| A05 | read_upper_TEST | Read | `R/data/TEST_manifest.csv` | R1 | Phase A |
| A06 | read_windows_path | Read | `C:\Users\admin\Tests\notes.txt` | R1 | Phase A |
| A07 | edit_ok_content_ignored | Edit | `R/src/eval/metrics.py` | allow | Phase A |
| A08 | edit_test_dir | Edit | `R/src/Tests/x.py` | R1 | Phase A |
| A09 | write_scratch | Write | `$TMP/scratch/notes.md` | allow | Phase A |
| A10 | write_contest_outside | Write | `$TMP/scratch/contest_notes.md` | R1 | Phase A |
| A11 | notebook_ok | NotebookEdit | `R/notebooks/analysis.ipynb` | allow | Phase A |
| A12 | notebook_test | NotebookEdit | `R/notebooks/test_eda.ipynb` | R1 | Phase A |
| A13 | exception_by_name | Read | `R/scripts/test_teacher_init.py` | allow | Phase A |
| A14 | read_named_csv | Read | `$TMP/scratch/plantseg_exact_duplicates.csv` | R1 | new |
| A15 | read_named_zip_case | Read | `C:\dl\PlantSeg_v2.ZIP` | R1 | new |
| A16 | write_zenodo_component | Write | `$TMP/scratch/b66_harness/inputs/zenodo_123/a.txt` | R1 | new |
| B01 | grep_tool_no_path | Grep | `path=<cwd>` | G1 | Phase A |
| B02 | grep_tool_src | Grep | `path=R/src` | G1 | Phase A |
| B03 | grep_tool_docs_lprot2 | Grep | `path=R/docs` | G1 | Phase A |
| B04 | grep_tool_repo_root | Grep | `path=R` | G1 | Phase A |
| B05 | grep_tool_parent | Grep | `path=/home/user` | G1 | Phase A |
| B06 | grep_tool_data | Grep | `path=/workspace/plantseg_data/plantseg` | G1 | Phase A |
| B07 | grep_tool_runs | Grep | `path=/workspace/plantseg_runs` | G1 | Phase A |
| B08 | grep_tool_file | Grep | `path=R/src/eval/metrics.py` | allow | Phase A |
| B09 | grep_tool_test_file | Grep | `path=R/scripts/test_x.py` | G2 | Phase A |
| B10 | grep_tool_scratch | Grep | `path=$TMP/scratch` | allow | Phase A |
| B11 | grep_tool_glob_test | Grep | `path=$TMP/scratch glob=*test*` | G2 | Phase A |
| B12 | grep_tool_cwd_scratch | Grep | `path=<cwd>` | allow | Phase A |
| B13 | glob_no_path | Glob | `pattern=**/*.py path=<cwd>` | G1 | Phase A |
| B14 | glob_claude_dir | Glob | `pattern=* path=R/.claude` | G1 | Phase A |
| B15 | glob_absolute_pattern | Glob | `pattern=R/src/**/*.py path=<cwd>` | G1 | Phase A |
| B16 | glob_scratch | Glob | `pattern=*.md path=$TMP/scratch` | allow | Phase A |
| B17 | glob_test_pattern | Glob | `pattern=**/*test* path=$TMP/scratch` | G2 | Phase A |
| B18 | glob_named_pattern | Glob | `pattern=**/plantseg*.zip path=$TMP/scratch` | G2 | new |
| B19 | grep_tool_other_clone | Grep | `path=$TMP/thesis_copy/src` | G1 | new |
| B20 | grep_tool_new_file | Grep | `path=R/src/eval/new_mod.py` | allow | new |
| C01 | slip_grep_scripts_glob | Bash | `grep -n "def main" scripts/*.py` | B1 | Phase A |
| C02 | slip_grep_rn_src | Bash | `grep -rn seed src/` | B1 | Phase A |
| C03 | slip_grep_R_include | Bash | `grep -R --include='*.py' MODE src/eval` | B1 | Phase A |
| C04 | slip_git_ls_files_claude | Bash | `git ls-files -- .claude/` | B2 | Phase A |
| C05 | slip_ls_hooks | Bash | `ls .claude/hooks/` | B1 | Phase A |
| C06 | slip_find_root_xdev | Bash | `find / -xdev -name '*.json'` | B1 | Phase A |
| D01 | agents_rule1_status | Bash | `git status -sb -- . ':(exclude)docs/reference/reference.pdf'` | allow | Phase A |
| D02 | agents_rule1_log | Bash | `git log --oneline -5` | allow | Phase A |
| D03 | git_diff_both_excludes | Bash | `git diff -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'` | allow | Phase A |
| D04 | git_diff_stat_both_excludes | Bash | `git diff --stat -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'` | allow | Phase A |
| D05 | git_diff_explicit_paths | Bash | `git diff -- .claude/settings.json scripts/smoke_sl1_guard.py` | allow | Phase A |
| D06 | git_diff_quiet_hooks | Bash | `git diff --quiet c689634 HEAD -- .claude/hooks ':(exclude).claude/hooks/sl1_guard.*'` | allow | Phase A |
| D07 | git_grep_exclude | Bash | `git grep -n MODE -- src ':(exclude,icase)*test*'` | allow | Phase A |
| D08 | git_ls_files_exclude | Bash | `git ls-files -- scripts ':(exclude,icase)*test*'` | allow | Phase A |
| D09 | git_ls_files_icase_first | Bash | `git ls-files -- ':(icase,exclude)*test*'` | allow | Phase A |
| D10 | smoke_run | Bash | `python -B scripts/smoke_sl1_guard.py` | allow | Phase A |
| D11 | smoke_glob_run | Bash | `python -B scripts/smoke_*.py` | allow | Phase A |
| D12 | docker_run_trainval_mounts | Bash | `docker run --rm --mount type=bind,src=/workspace/plantseg_data/plantseg/images/train,dst=/dat...` | allow | changed: A6: explicit TRAIN/VAL mounts |
| D13 | builtin_test_f | Bash | `test -f src/eval/metrics.py && echo ok` | allow | Phase A |
| D14 | builtin_bracket | Bash | `[ -f src/eval/metrics.py ] \|\| echo missing` | allow | Phase A |
| D15 | cat_metrics | Bash | `cat src/eval/metrics.py` | allow | Phase A |
| D16 | head_agents | Bash | `head -n 40 AGENTS.md` | allow | Phase A |
| D17 | grep_explicit_file | Bash | `grep -n MODE .claude/hooks/sl1_guard.py` | allow | Phase A |
| D18 | grep_stdin | Bash | `git log --oneline \| grep -i fix` | allow | Phase A |
| D19 | ls_explicit_file | Bash | `ls -la src/eval/metrics.py` | allow | Phase A |
| D20 | ls_d_dir | Bash | `ls -ld .claude` | allow | Phase A |
| D21 | ls_scratch | Bash | `ls -la '$TMP/scratch'` | allow | Phase A |
| D22 | ls_parent_nonrecursive | Bash | `ls '/home/user'` | allow | Phase A |
| D23 | sha256_test_archive | Bash | `sha256sum /workspace/plantseg_data/plantseg_test.zip` | allow | Phase A |
| D24 | sed_script_word | Bash | `sed -n 's/test/x/p' src/eval/metrics.py` | allow | Phase A |
| D25 | git_log_grep_flag | Bash | `git log --grep=test --oneline` | allow | Phase A |
| D26 | git_show_blob | Bash | `git show c689634:.claude/settings.json` | allow | Phase A |
| D27 | git_show_no_patch | Bash | `git show -s --format=%H HEAD` | allow | Phase A |
| D28 | cat_guard_log | Bash | `cat ~/.claude/sl1_guard.log` | allow | Phase A |
| D29 | ps_get_content | PowerShell | `Get-Content src\eval\metrics.py` | allow | Phase A |
| D30 | ps_filehash_sha256 | PowerShell | `Get-FileHash -Algorithm SHA256 C:\data\plantseg_data\plantseg_test.zip` | allow | Phase A |
| D31 | sha256_named_csv | Bash | `sha256sum /workspace/plantseg_runs/evidence_am18/plantseg_exact_duplicates.csv` | allow | new |
| D32 | tar_extract_export | Bash | `tar -xf /workspace/plantseg_runs/smoke_afd2d33.tar -C /tmp/x` | allow | new |
| D33 | hash_run_folder_allowed_form | Bash | `find /workspace/plantseg_runs/e1_s42 -type f \| grep -vi test \| xargs -d '\n' sha256sum` | allow | new |
| D34 | git_clean_dry_run | Bash | `git clean -n` | allow | new |
| D35 | count_find_wc | Bash | `find . -name '*.py' \| wc -l` | allow | new |
| D36 | count_ls_R_wc | Bash | `ls -R docs \| wc -l` | allow | new |
| D37 | count_grep_c | Bash | `git ls-files \| grep -c py` | allow | new |
| D38 | count_ps_measure | PowerShell | `Get-ChildItem -Recurse src \| Measure-Object` | allow | new |
| D39 | count_ps_dot_count | PowerShell | `Write-Output (Get-ChildItem -Recurse src).Count` | allow | new |
| D40 | assigned_var_outside | Bash | `SP='$TMP/scratch'; ls -R "$SP"` | allow | new |
| D41 | tar_list_export_filtered | Bash | `tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar \| grep -vi test` | allow | new |
| D42 | tar_list_export_count | Bash | `tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar \| wc -l` | allow | new |
| D43 | git_diff_name_only_explicit | Bash | `git diff --name-only HEAD -- .claude/settings.json` | allow | new |
| D44 | du_summary_explicit | Bash | `du -sh /workspace/plantseg_runs/e1_s42` | allow | new |
| D45 | git_add_explicit | Bash | `git add -- .claude/hooks/sl1_guard.py scripts/smoke_sl1_guard.py` | allow | new |
| D46 | git_commit_only | Bash | `git commit --only -F /tmp/msg.txt -- .claude/settings.json` | allow | new |
| D47 | git_push_lane | Bash | `git push -u origin lane/sl1-guard` | allow | new |
| D48 | git_ls_remote_lane | Bash | `git ls-remote --heads origin lane/sl1-guard` | allow | new |
| D49 | acceptance_name_status | Bash | `git diff --name-status 6067e66 HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,...` | allow | new |
| D50 | acceptance_quiet | Bash | `git diff --quiet 6067e66 HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude).claud...` | allow | new |
| E01 | bare_ls | Bash | `ls` | B1 | Phase A |
| E02 | bare_ls_la | Bash | `ls -la` | B1 | Phase A |
| E03 | bash_dir | Bash | `dir` | B1 | Phase A |
| E04 | ps_get_childitem | PowerShell | `Get-ChildItem` | B1 | Phase A |
| E05 | ps_dir | PowerShell | `dir` | B1 | Phase A |
| E06 | ps_ls | PowerShell | `ls` | B1 | Phase A |
| E07 | bare_tree | Bash | `tree` | B1 | Phase A |
| E08 | cd_repo_then_ls | Bash | `cd 'R' && ls` | B1 | Phase A |
| E09 | cd_scratch_then_ls | Bash | `cd '$TMP/scratch' && ls` | allow | Phase A |
| F01 | ls_R_docs | Bash | `ls -R docs` | B1 | Phase A |
| F02 | ls_glob | Bash | `ls src/eval/*.py` | B1 | Phase A |
| F03 | rg_bare | Bash | `rg seed` | B1 | Phase A |
| F04 | rg_src | Bash | `rg -n seed src` | B1 | Phase A |
| F05 | find_dot | Bash | `find . -name '*.py'` | B1 | Phase A |
| F06 | tree_src | Bash | `tree src` | B1 | Phase A |
| F07 | du_a | Bash | `du -a src` | B1 | Phase A |
| F08 | grep_dir_arg | Bash | `grep MODE src` | B1 | Phase A |
| F09 | egrep_r | Bash | `egrep -r seed .` | B1 | Phase A |
| F10 | ps_gci_recurse | PowerShell | `Get-ChildItem -Recurse src` | B1 | Phase A |
| F11 | ps_gci_r_filter | PowerShell | `gci -r -Filter *.py` | B1 | Phase A |
| F12 | ps_dir_recurse | PowerShell | `dir -Recurse` | B1 | Phase A |
| F13 | cmd_dir_s_b | Bash | `cmd /c "dir /s /b"` | B1 | Phase A |
| F14 | findstr_s | Bash | `findstr /s /i seed *.py` | B1 | Phase A |
| F15 | ps_sls_glob | PowerShell | `Select-String -Pattern seed -Path src\*.py` | B1 | Phase A |
| F16 | cat_glob | Bash | `cat scripts/*.py` | B1 | Phase A |
| F17 | for_glob | Bash | `for f in scripts/*.py; do wc -l "$f"; done` | B1 | Phase A |
| F18 | locate | Bash | `locate smoke_` | B1 | Phase A |
| F19 | tar_list_export | Bash | `tar -tf /workspace/plantseg_runs/smoke_afd2d33.tar` | B1 | new |
| F20 | assigned_var_data | Bash | `SP=/workspace/plantseg_data; ls -R "$SP"` | B1 | new |
| F21 | unresolved_var | Bash | `ls -R "$SL1_UNSET_DIR"` | B1 | new |
| F22 | du_summary_glob | Bash | `du -sh /workspace/plantseg_runs/*` | B1 | new |
| F23 | echo_glob_expansion | Bash | `echo scripts/*` | B1 | new |
| G01 | grep_r_filtered | Bash | `grep -rn seed src \| grep -vi test` | allow | Phase A |
| G02 | find_filtered_head | Bash | `find . -name '*.py' \| grep -iv test \| head -n 20` | allow | Phase A |
| G03 | git_ls_files_filtered | Bash | `git ls-files \| grep -v -i test` | allow | Phase A |
| G04 | git_grep_filtered_long | Bash | `git grep -n seed \| grep --invert-match --ignore-case test` | allow | Phase A |
| G05 | ps_gci_name_filtered | PowerShell | `Get-ChildItem -Recurse -Name src \| Select-String -NotMatch test` | allow | Phase A |
| G06 | ps_git_filtered | PowerShell | `git ls-files \| Select-String -NotMatch -Pattern test` | allow | Phase A |
| G07 | tree_f_filtered | Bash | `tree -f src \| grep -vi test` | allow | Phase A |
| G08 | cmd_dir_s_b_filtered | Bash | `cmd /c "dir /s /b" \| grep -vi test` | allow | Phase A |
| G09 | ps_gci_fileinfo_into_sls | PowerShell | `Get-ChildItem -Recurse src \| Select-String -NotMatch test` | B1 | Phase A |
| G10 | ls_R_filtered_leaks_members | Bash | `ls -R docs \| grep -vi test` | B1 | Phase A |
| G11 | tree_filtered_leaks_members | Bash | `tree src \| grep -vi test` | B1 | Phase A |
| G12 | cmd_dir_s_not_bare | Bash | `cmd /c "dir /s" \| grep -vi test` | B1 | Phase A |
| G13 | grep_h_filtered | Bash | `grep -rhn seed src \| grep -vi test` | B1 | Phase A |
| G14 | find_exec_filtered | Bash | `find . -name '*.py' -exec cat {} + \| grep -vi test` | B1 | Phase A |
| G15 | filter_word_regexp | Bash | `grep -rn seed src \| grep -viw test` | B1 | Phase A |
| G16 | filter_case_sensitive | Bash | `grep -rn seed src \| grep -v test` | B1 | Phase A |
| G17 | tee_before_filter | Bash | `find . \| tee '$TMP/scratch/l.txt' \| grep -vi test` | B1 | Phase A |
| G18 | redirect_listing | Bash | `find . -name '*.py' > '$TMP/scratch/list.txt'` | B1 | Phase A |
| G19 | xargs_from_safe_list | Bash | `git ls-files -- src ':(exclude,icase)*test*' \| xargs grep -n seed` | allow | Phase A |
| G20 | xargs_from_unsafe_list | Bash | `git ls-files \| xargs grep -n seed` | B2 | Phase A |
| G21 | rg_heading_filtered | Bash | `rg --heading seed src \| grep -vi test` | B1 | new |
| G22 | rg_pretty_filtered | Bash | `rg -p seed src \| grep -vi test` | B1 | new |
| G23 | git_grep_h_filtered | Bash | `git grep -h seed \| grep -vi test` | B2 | new |
| G24 | git_grep_heading_filtered | Bash | `git grep --heading seed \| grep -vi test` | B2 | new |
| G25 | tee_before_count | Bash | `ls -R docs \| tee '$TMP/scratch/x.txt' \| wc -l` | B1 | new |
| G26 | find_exec_sha256_run_folder | Bash | `find /workspace/plantseg_runs/e1_s42 -type f -exec sha256sum {} +` | B1 | new |
| H01 | git_ls_files_bare | Bash | `git ls-files` | B2 | Phase A |
| H02 | git_grep_no_exclude | Bash | `git grep -n seed` | B2 | Phase A |
| H03 | git_grep_exclude_no_icase | Bash | `git grep seed -- ':(exclude)*test*'` | B2 | Phase A |
| H04 | git_literal_pathspecs | Bash | `git --literal-pathspecs ls-files -- ':(exclude,icase)*test*'` | B2 | Phase A |
| H05 | git_diff_stat_bare | Bash | `git diff --stat` | B2 | Phase A |
| H06 | git_diff_name_only_pdf_only | Bash | `git diff --name-only -- . ':(exclude)docs/reference/reference.pdf'` | B2 | Phase A |
| H07 | git_show_stat | Bash | `git show --stat HEAD` | B2 | Phase A |
| H08 | git_log_name_status | Bash | `git log --name-status -3` | B2 | Phase A |
| H09 | git_log_numstat_excludes | Bash | `git log --numstat -3 -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'` | allow | Phase A |
| H10 | git_show_patch | Bash | `git show HEAD` | B6 | Phase A |
| H11 | git_diff_pdf_exclude_only | Bash | `git diff -- . ':(exclude)docs/reference/reference.pdf'` | B6 | Phase A |
| H12 | git_show_test_blob | Bash | `git show HEAD:scripts/test_teacher_init.py` | B4 | Phase A |
| H13 | git_show_tree | Bash | `git show HEAD:.claude/hooks` | B2 | Phase A |
| H14 | git_ls_tree_r | Bash | `git ls-tree -r HEAD` | B2 | Phase A |
| H15 | git_ls_tree_r_filtered | Bash | `git ls-tree -r HEAD \| grep -vi test` | allow | Phase A |
| H16 | git_cat_file_tree | Bash | `git cat-file -p HEAD:.claude` | B2 | Phase A |
| H17 | git_C_scratch_ls_files | Bash | `git -C '$TMP/scratch' ls-files` | B2 | Phase A |
| H18 | git_diff_stat_explicit_file | Bash | `git diff --stat -- src/eval/metrics.py` | allow | Phase A |
| H19 | git_literal_env_prefix | Bash | `GIT_LITERAL_PATHSPECS=1 git ls-files -- ':(exclude,icase)*test*'` | B2 | new |
| H20 | git_archive | Bash | `git archive HEAD` | B6 | new |
| I01 | pytest_bare | Bash | `pytest` | B3 | Phase A |
| I02 | python_m_pytest | Bash | `python -m pytest -q scripts` | B3 | Phase A |
| I03 | py_launcher_pytest | Bash | `py -m pytest -q` | B3 | Phase A |
| I04 | bash_c_pytest | Bash | `bash -c "cd scripts && pytest -x"` | B3 | Phase A |
| I05 | python_m_unittest | Bash | `python -m unittest discover` | B3 | new |
| J01 | cat_test | Bash | `cat scripts/test_teacher_init.py` | B4 | Phase A |
| J02 | head_test | Bash | `head -n 5 src/tests/x.py` | B4 | Phase A |
| J03 | tail_test | Bash | `tail TEST_split.txt` | B4 | Phase A |
| J04 | less_test | Bash | `less docs/testing.md` | B4 | Phase A |
| J05 | more_test | Bash | `more Tests/a.txt` | B4 | Phase A |
| J06 | sed_test | Bash | `sed -n '1,10p' scripts/test_x.py` | B4 | Phase A |
| J07 | awk_test | Bash | `awk '{print}' data/test.csv` | B4 | Phase A |
| J08 | ps_get_content_test | PowerShell | `Get-Content scripts\test_x.py` | B4 | Phase A |
| J09 | ps_type_test | PowerShell | `type .\tests\a.txt` | B4 | Phase A |
| J10 | cat_latest_literal | Bash | `cat reports/latest.md` | B4 | Phase A |
| J11 | grep_test_file | Bash | `grep seed scripts/test_x.py` | B4 | Phase A |
| J12 | redirect_in_test | Bash | `wc -l < scripts/test_x.py` | B4 | Phase A |
| J13 | tail_contest_outside | Bash | `tail -n 50 '$TMP/scratch/contest.log'` | B4 | Phase A |
| J14 | python_test_script | Bash | `python -B scripts/test_teacher_init.py` | B7 | Phase A |
| J15 | sh_test_script | Bash | `sh tools/run_tests.sh` | B7 | Phase A |
| J16 | cat_named_csv | Bash | `cat '$TMP/scratch/plantseg_exact_duplicates.csv'` | B4 | new |
| J17 | head_named_zip_bytes | Bash | `head -c 64 /workspace/plantseg_data/plantseg_v3.zip` | B4 | new |
| J18 | ls_named_glob | Bash | `ls -la ~/Downloads/plantseg*.zip` | B4 | new |
| J19 | md5_named_csv | Bash | `md5sum '$TMP/scratch/plantseg_exact_duplicates.csv'` | B4 | new |
| J20 | stat_test_path | Bash | `stat scripts/test_x.py` | B4 | new |
| K01 | monitor_tail_log | Monitor | `tail -f '$TMP/scratch/run.log' \| grep --line-buffered ERROR` | allow | Phase A |
| K02 | monitor_grep_r | Monitor | `grep -r seed src` | B1 | Phase A |
| K03 | monitor_websocket | Monitor | `(WebSocket, no command)` | allow | Phase A |
| L01 | docker_rmi | Bash | `docker rmi img:tag` | D1 | Phase A |
| L02 | docker_image_rm | Bash | `docker image rm img` | D1 | Phase A |
| L03 | docker_image_prune | Bash | `docker image prune -a` | D1 | Phase A |
| L04 | docker_system_prune | Bash | `docker system prune -af` | D1 | Phase A |
| L05 | docker_volume_prune | Bash | `docker volume prune -f` | D1 | Phase A |
| L06 | docker_global_flag_rmi | Bash | `docker --context x rmi img` | D1 | Phase A |
| L07 | docker_image_ls | Bash | `docker image ls` | allow | Phase A |
| L08 | rm_runs | Bash | `rm -rf /workspace/plantseg_runs/e1_s42` | D2 | Phase A |
| L09 | rm_data | Bash | `rm -f /workspace/plantseg_data/x.zip` | D2 | Phase A |
| L10 | rm_repo_file | Bash | `rm scripts/old.py` | D2 | Phase A |
| L11 | rm_repo_root | Bash | `rm -rf 'R'` | D2 | Phase A |
| L12 | rm_repo_parent | Bash | `rm -rf '/home/user'` | D2 | Phase A |
| L13 | rm_root | Bash | `rm -rf /` | D2 | Phase A |
| L14 | rm_scratch | Bash | `rm -f '$TMP/scratch/x.txt'` | allow | Phase A |
| L15 | rmdir_repo | Bash | `rmdir docs/tmpdir` | D2 | Phase A |
| L16 | cmd_del_runs | Bash | `cmd /c "del /q C:\work\plantseg_runs\e1\x.pt"` | D2 | Phase A |
| L17 | cmd_rd_repo | Bash | `cmd /c "rd /s /q C:\work\plantseg-thesis\tmp"` | D2 | Phase A |
| L18 | ps_remove_item_data | PowerShell | `Remove-Item -Recurse -Force C:\data\plantseg_data\cache` | D2 | Phase A |
| L19 | ps_ri_alias_repo | PowerShell | `ri .\tmp.txt` | D2 | Phase A |
| L20 | rm_unresolved_var | Bash | `rm -rf "$SL1_UNSET_VAR/x"` | D2 | Phase A |
| L21 | ps_remove_item_scratch | PowerShell | `Remove-Item '$TMP/scratch/x.txt'` | allow | Phase A |
| L22 | rm_glob_runs | Bash | `rm -f /workspace/plantseg_runs/*.log` | D2 | Phase A |
| L23 | rm_data_root_env | Bash | `rm -rf "$PLANTSEG_DATA_ROOT"` | D2 | Phase A |
| L24 | docker_volume_rm | Bash | `docker volume rm vol1` | D1 | new |
| L25 | git_clean_force | Bash | `git clean -fdx` | D2 | new |
| L26 | find_delete_runs | Bash | `find /workspace/plantseg_runs -name '*.tmp' -delete` | D2 | new |
| L27 | rm_other_clone | Bash | `rm -rf '$TMP/thesis_copy/src'` | D2 | new |
| M01 | bash_c_grep_r | Bash | `bash -c "grep -r seed src"` | B1 | Phase A |
| M02 | subst_ls_R | Bash | `echo "$(ls -R docs)"` | B1 | Phase A |
| M03 | backtick_find | Bash | `echo `find . -name x`` | B1 | Phase A |
| M04 | sudo_find | Bash | `sudo find / -name x` | B1 | Phase A |
| M05 | eval_find | Bash | `eval "find . -type f"` | B1 | Phase A |
| M06 | heredoc_body_is_data | Bash | `cat > '$TMP/scratch/x.md' <<'EOF' ⏎ grep -r seed src ⏎ EOF` | allow | Phase A |
| M07 | unbalanced_quote | Bash | `grep "seed src` | B5 | Phase A |
| M08 | env_prefix | Bash | `LC_ALL=C grep -r seed src` | B1 | Phase A |
| M09 | ps_nested_fullname | PowerShell | `(Get-ChildItem -Recurse src).FullName` | B1 | changed: A4: old M09 (.Count) is now allowed as D39; M09 is a nested .FullName deny |
| M10 | pwsh_command | Bash | `pwsh -NoProfile -Command "Get-ChildItem -Recurse"` | B1 | Phase A |
| M11 | ps_encoded_command | PowerShell | `powershell -EncodedCommand ZQBjAGgAbwA=` | B5 | new |
| N01 | tar_list_data_not_named | Bash | `tar -tzf /workspace/plantseg_data/plantseg.tar.gz` | B1 | changed: Q1a: B8 is by name; a non-named archive in a protected location is a B1 listing |
| N02 | unzip_l_named | Bash | `unzip -l /workspace/plantseg_data/plantseg.zip` | B8 | Phase A |
| N03 | 7z_l_named | Bash | `7z l ~/Downloads/plantseg_v3.zip` | B8 | Phase A |
| N04 | tar_create_scratch | Bash | `tar -czf '$TMP/scratch/out.tgz' -C '$TMP/scratch' notes` | allow | Phase A |
| N05 | python_zipfile_named | Bash | `python -m zipfile -l ~/Downloads/plantseg.zip` | B8 | new |
| N06 | ps_expand_archive_named | PowerShell | `Expand-Archive C:\dl\plantseg_v2.zip -DestinationPath C:\x` | B8 | new |
| N07 | unzip_zenodo | Bash | `unzip b66_harness/inputs/zenodo_17/data.zip -d /tmp/x` | B8 | new |
| N08 | python_tarfile_export | Bash | `python -m tarfile -l /workspace/plantseg_runs/smoke_afd2d33.tar` | B1 | new |
| N09 | tar_test_named_archive | Bash | `tar -tf '$TMP/scratch/test_kit.tar'` | B8 | new |
| W01 | find_msys_home | Bash | `find /c/Users/admin -name '*.json'` | B1 | new |
| W02 | rm_msys_runs | Bash | `rm -rf /c/Users/admin/plantseg_runs/e1` | D2 | new |
| W03 | grep_msys_repo | Bash | `grep -rn seed /c/Users/admin/plantseg-thesis/src` | B1 | new |
| W04 | grep_msys_case_only | Bash | `grep -rn seed /c/users/ADMIN/thesis2/src` | B1 | new |
| W05 | grep_tool_forward_slashes | Grep | `path=C:/Users/Admin/Thesis2/docs` | G1 | new |
| W06 | ls_R_drive_ancestor | Bash | `ls -R C:/Users/admin` | B1 | new |
| W07 | ps_remove_backslash | PowerShell | `Remove-Item -Recurse C:\USERS\admin\thesis2\tmp` | D2 | new |

## 7. Did the entry go live in this session? Yes (MEASURED)

- settings.json was written at `2026-10-04T15:57:48Z`. Before that, `~/.claude/sl1_guard.log` was absent.
- A deliberate harmless probe, `git diff --stat -- src/eval` (no changes, so no output), was logged 8 seconds later without being blocked. The file watcher picked the entry up, as the GO's DOCUMENTED note predicted.
- From then on every tool call in this session passed through the guard in log mode.

This session's `~/.claude/sl1_guard.log`, read just before this commit:

```
2026-10-04T15:57:56Z	Bash	SL1-B2	git diff --stat -- src/eval; echo "exit=$?"
```

## 8. For the flip review: everyday forms that deny mode will refuse

| Form used today | Rule | Allowed form |
|---|---|---|
| AGENTS.md rule 6: `git diff -- . ':(exclude)docs/reference/reference.pdf'` (also `--cached`) | B6 | add `':(exclude,icase)*test*'`, or name the files: `git diff -- <paths>` |
| `git diff --stat <a> <b>`, `git log --stat`, `--name-only`, `--name-status` | B2 | add `-- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'`, or name the files |
| `git show <commit>`, `git log -p` | B6 | `git show <commit> -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'`, or `git show <commit> -- <paths>` |
| `git ls-files`, `git ls-files <dir>`, `git grep <pat>` | B2 | add `':(exclude,icase)*test*'`, or pipe through `grep -vi test` |
| `ls` / `ls <repo folder>` / `tree` / `dir` / `Get-ChildItem` in the repo | B1 | `git ls-files -- <folder> ':(exclude,icase)*test*'`; `ls <folder> \| grep -vi test` (non-recursive); `ls -d <folder>` |
| `ls -la /workspace/plantseg_runs/` | B1 | `ls -la /workspace/plantseg_runs \| grep -vi test` |
| Hashing a run folder: `find <dir> -exec sha256sum {} +`, `sha256sum <dir>/*` | B1 | `find <dir> -type f \| grep -vi test \| xargs -d '\n' sha256sum`, or `sha256sum <named files>` |
| `du -sh /workspace/plantseg_runs/*` | B1 | `du -sh /workspace/plantseg_runs/* \| grep -vi test` |
| `grep -rn <pat> src`, `rg <pat>` | B1 | `git grep -n <pat> -- src ':(exclude,icase)*test*'`, or `… \| grep -vi test` |
| The Grep or Glob tool on a repo folder | G1 | the Grep tool on an explicit file, or `git grep` / `git ls-files` with the exclude through Bash |
| `python -m pytest` | B3 | `python -B scripts/smoke_*.py` |
| `rm -rf <repo>/…/__pycache__`, `rm <repo file>` | D2 | a human runs it outside Claude Code; keep scratch files in the session scratchpad |
| `docker system prune`, `docker image prune` | D1 | a human runs it |

This session's own commands, replayed through the policy, are all allowed. That covers base checks, PB-1a, the reads, the smokes, `git add -- <paths>`, `git commit --only -F … -- <paths>` and the acceptance commands (cases D45–D50).

## 9. Deferred local block for Ice (after the merge)

Run in the merged local checkout, outside Claude Code unless a step says otherwise.

1. **LB-1 branch check** (Git Bash, repo root):
   ```
   git status -sb -- . ':(exclude)docs/reference/reference.pdf'
   git rev-parse --abbrev-ref HEAD                                      # claude/keen-curie-u4a8ig
   git merge-base --is-ancestor 648d38813644ba0cef1ef21e4d52f11a86e1ce8a HEAD; echo $?   # 0
   git rev-parse HEAD:.claude/hooks/sl1_guard.py                        # ae38007e7a7ff61b1415b521efe424a4116acb04
   git diff --quiet c689634 HEAD -- .claude/hooks ':(exclude).claude/hooks/sl1_guard.*'; echo $?   # 0
   ```
   The blob line holds only if no later commit touched the policy.
2. **Git Bash smoke:**
   ```
   python -B scripts/smoke_sl1_guard.py | tail -n 22
   python -B scripts/smoke_frozen_blobs.py | tail -n 1
   ```
   Expected:
   - The first command's last line is `RESULT PASS: smoke_sl1_guard 284/284 (MODE=log)`, and the header line shows `sh=` with Git's sh.
   - The `Timing:` line gives the Windows median, p95 and max. O25 requires p95 under 200 ms; if O25 is the only failure, record the timing and report it.
   - The second command prints `RESULT: FROZEN BLOBS OK (10/10)`.
3. **PowerShell smoke:**
   ```
   python -B scripts\smoke_sl1_guard.py | Select-Object -Last 22
   python -B scripts\smoke_frozen_blobs.py | Select-Object -Last 1
   ```
   Expected: the same two RESULT lines. The smoke finds `sh.exe` next to `git.exe` or under `C:\Program Files\Git`. If O00 reports `no sh found`, put Git's `bin` on PATH and re-run.
4. **Live check from a local Claude Code session** (restart the session if the entry doesn't load):
   - Ask Claude to **Read `src/eval/metrics.py`**. Expected: the file is shown, and no new line appears in the log.
   - Ask Claude to run a **bare `ls`** in the repo. Expected: the command runs (log mode never blocks), and one new line appears: `<UTC>	Bash	SL1-B1	ls`. If the session uses the PowerShell tool, the line reads `<UTC>	PowerShell	SL1-B1	ls`.
     - The PowerShell payload key `command` is INFERRED. If the line instead reads `SL1-B5` with an empty subject, that key is wrong; report it before any flip.
     - `ls >/dev/null` logs the same rule without printing the root's names.
   - Read the log outside Claude Code. Git Bash: `tail -n 3 ~/.claude/sl1_guard.log`. PowerShell: `Get-Content $HOME\.claude\sl1_guard.log -Tail 3`.
5. **After two days:** review the log. The flip is a one-line commit, `MODE = "deny"`, followed by the smoke, whose expected outcomes follow MODE.

## 10. Known gaps

- **Unanalysed code:** code inside interpreter scripts and `python -c`, `node -e` or `perl -e` is not read. Neither are shell functions or aliases defined earlier, the contents of `source`d scripts, commands run by scripts, make, git hooks or CI, or commands sent over ssh.
- **Docker (A6):** `--mount` and `-v` binds, and commands run inside containers, are not policed.
- **stderr (A4):** the filter and count exemptions cover stdout only. A listing's stderr (permission errors, for example) can still print a test-named path.
- **`--stat` truncation:** `git diff --stat` shortens long paths (`…/x`), which can hide a test-named directory component in a filtered listing.
- **Q3 and missing paths:** a directory that exists only in another revision but has a file-like name passes the explicit-file relaxation. A path that doesn't exist on this machine counts as a file if its name has an extension, and as a directory otherwise.
- **Unpoliced glob expansion:** glob expansion in commands that aren't listings is allowed, for example `python -B scripts/smoke_*.py` or `git add scripts/*.py`. The names reach the program, not the output, unless the program prints them.
- **Unpoliced writes and renames:** `mv`, `git mv`, `git rm` and `git add` of test-named or NAMED paths aren't policed; they print no content or listing.
- **Never blocks:** a hook timeout (10 s) and a missing launcher file never block (DOCUMENTED behaviour, plus the `[ -r ] || exit 0` guard).
- **PowerShell:** the payload key `command` is INFERRED (see the local block). Only Select-String's FileInfo binding is modelled; other cmdlets fed FileInfo objects are judged through the upstream `Get-ChildItem` listing. `--%` and here-strings are approximated.
- **Malformed payloads:** a non-JSON or non-UTF-8 payload is E1, which means fail mode (a) in deny mode.
- **False positive found while writing this report:** `git show :<path>` (the index blob) is read as a revision. It is treated as patch output and logged as B6 in log mode, and deny mode would refuse it. `git cat-file -p :<path>` is allowed. A fix can ride a later commit.

**Observations**
- The existing L-PROT hook refused `git hash-object` and `ls ~/.claude/sl1_guard.log` in this session (rule S treats `~` as unresolvable). Read the SL1 log with Python or outside Claude Code.
- DL-26 lists `switch` among the commands denied to Claude Code. The PB-1a chain (`git branch`, `git symbolic-ref`, `git checkout --no-overlay`) passed L-PROT.

## 11. Fix after review (GO-2, orchestrator, 2026-10-05 00:21)

**Commit.** This one fix commit has parent `3d99edd2c16ed57821129d0b5f0aacb49ce10ce3`. A commit can't name its own SHA, so it is given in the session's chat reply. Files, all modified:
- `.claude/hooks/sl1_guard.py`
- `scripts/smoke_sl1_guard.py`
- `docs/lane_reports/sl1-guard.md`

The MODE line still reads `log`.

**Fixes**
1. **Special parameters.** `Lexer.dollar_bash` now takes a `$` followed by one of `? * @ # $ ! -` or a digit as literal text, with no glob mark.
   - Before the fix, an unquoted `echo $?` or `echo $*` was a glob word, so `…; echo $?` was B1. The live log found this at 16:10:33Z (MEASURED).
   - A real glob is still a glob: `ls *.md` and `ls $1*` remain B1 (F24, F25).
2. **Blob reads in `git show`.** The new `blob_path()` parses `<rev>:<path>`, `:<path>` and `:<n>:<path>`.
   - A test-named or NAMED_DENY blob path is B4. Any other blob path is allowed, never B6 (H21–H25).
   - A tree path, such as `<rev>:<dir>` or `<rev>:`, is still a B2 listing (H13 unchanged).
   - `:/<text>` is a commit search, not a path, so `git show :/fix` shows a commit patch and is B6 (H26).
   - This closes the false positive recorded in §10.

**Orchestrator erratum.** GO-2's ACCEPTANCE line `git diff --name-status 3d99edd… HEAD` had no pathspec.
- Under GO-1 rule 3 it is a B2 listing.
- The existing L-PROT hook already refuses it, under rule G: "git diff without `--` covers the whole tree" (MEASURED by running the L-PROT policy on the text).
- On the orchestrator's ruling it stays a deny case (P19).
- The corrected acceptance line is P23, `git diff --name-status 3d99edd… HEAD -- . ':(exclude)docs/reference/reference.pdf' ':(exclude,icase)*test*'`, together with the five-path `--quiet` check.
- This is an erratum in the GO, not an exception to any rule.

**Observation.** P18, the protected-file status probe copied verbatim with its trailing `# nothing; …` comment, is allowed by SL1 but refused by L-PROT rule N: the comment means it is no longer the exact whole command. Run the probe without the comment.

**Gap carried, outside GO-2's scope.** `git cat-file -p :<path>`, the index form, is not parsed by `blob_path()`. `git cat-file -p :scripts/test_x.py` is allowed, while `git cat-file -p HEAD:scripts/test_x.py` is B4 (MEASURED). The fix is to route cat-file object names through `blob_path()` in a later commit.

**New counts (MEASURED, cloud).** `python -B scripts/smoke_sl1_guard.py` printed `RESULT PASS: smoke_sl1_guard 319/319 (MODE=log)`: 287 decision cases, each run in process, in log mode and in deny mode, plus 32 launcher, settings and invariant checks.

| Group | in-process | log mode | deny mode |
|---|---|---|---|
| A file tools | 16/16 | 16/16 | 16/16 |
| B Grep/Glob tools | 20/20 | 20/20 | 20/20 |
| C recorded slips | 6/6 | 6/6 | 6/6 |
| D everyday commands | 54/54 | 54/54 | 54/54 |
| E bare listings | 9/9 | 9/9 | 9/9 |
| F recursive and glob listings | 25/25 | 25/25 | 25/25 |
| G filter and count exemptions | 26/26 | 26/26 | 26/26 |
| H git | 26/26 | 26/26 | 26/26 |
| I pytest | 5/5 | 5/5 | 5/5 |
| J test-named and NAMED paths | 20/20 | 20/20 | 20/20 |
| K Monitor | 3/3 | 3/3 | 3/3 |
| L destructive | 27/27 | 27/27 | 27/27 |
| M nesting and parsing | 11/11 | 11/11 | 11/11 |
| N archives | 9/9 | 9/9 | 9/9 |
| P GO forms (verbatim, new) | 23/23 | 23/23 | 23/23 |
| W Windows path forms | 7/7 | 7/7 | 7/7 |
| O launcher, settings, invariants | 32/32 | | |

- **Timing:** median 62.4 ms, p95 86.6 ms, max 106.4 ms over 287 log-mode launcher calls.
- **`scripts/smoke_frozen_blobs.py`:** `RESULT: FROZEN BLOBS OK (10/10)`.
- **Cases:** 35 new and none changed; every expectation from 3d99edd is kept. P01–P07 are GO-1 step 1, P08–P18 are PB-1a, and P19–P23 are GO-2's acceptance lines, all verbatim.

**Acceptance (MEASURED on the staged tree before this commit; the chat reply repeats it against the tip)**

- The new tip's parent will be `3d99edd2c16ed57821129d0b5f0aacb49ce10ce3`: the commit is made on that HEAD.
- **Corrected name-status line (P23 form)** against `3d99edd…` printed exactly the three paths:

```
M	.claude/hooks/sl1_guard.py
M	docs/lane_reports/sl1-guard.md
M	scripts/smoke_sl1_guard.py
```

- **Nothing else changed since `3d99edd…`.** The `--quiet` check excluding the protected file and the three paths exits 0, so no other file changed, test-named files included.
- **GO-1's five-path `--quiet` check against `6067e66…`** still exits 0.
- **Mode and trailers.** There is one `MODE = "log"` line, and the commit carries 0 trailer lines.

### New cases (GO-2)

| ID | Case | Tool | Input (R = repo root, $TMP = smoke temp dir) | Deny mode | Status |
|---|---|---|---|---|---|
| D51 | special_param_status | Bash | `echo $?` | allow | new (GO-2) |
| D52 | quiet_diff_then_status | Bash | `git diff --quiet c689634 HEAD -- .claude/hooks ':(exclude).claude/hooks/sl1_guard.*'; echo $?` | allow | new (GO-2) |
| D53 | special_param_count | Bash | `echo $#` | allow | new (GO-2) |
| D54 | positional_param | Bash | `echo $1` | allow | new (GO-2) |
| F24 | ls_glob_md | Bash | `ls *.md` | B1 | new (GO-2) |
| F25 | ls_param_then_glob | Bash | `ls $1*` | B1 | new (GO-2) |
| H21 | git_show_index_blob | Bash | `git show :src/eval/metrics.py` | allow | new (GO-2) |
| H22 | git_show_index_test_blob | Bash | `git show :scripts/test_teacher_init.py` | B4 | new (GO-2) |
| H23 | git_show_stage_named_blob | Bash | `git show :0:data/plantseg_exact_duplicates.csv` | B4 | new (GO-2) |
| H24 | git_show_rev_blob | Bash | `git show 6067e66:src/eval/metrics.py` | allow | new (GO-2) |
| H25 | git_show_rev_named_blob | Bash | `git show HEAD:b66_harness/inputs/zenodo_17/meta.json` | B4 | new (GO-2) |
| H26 | git_show_commit_search_not_path | Bash | `git show :/fix` | B6 | new (GO-2) |
| P01 | go1_step1_shallow_probe | Bash | `git rev-parse --is-shallow-repository` | allow | new (GO-2) |
| P02 | go1_step1_unshallow | Bash | `git fetch --unshallow` | allow | new (GO-2) |
| P03 | go1_step1_fetch_master | Bash | `git fetch origin master` | allow | new (GO-2) |
| P04 | go1_step1_fetch_head | Bash | `git rev-parse FETCH_HEAD` | allow | new (GO-2) |
| P05 | go1_step1_ls_remote_master | Bash | `git ls-remote origin master` | allow | new (GO-2) |
| P06 | go1_step1_is_ancestor_echo_status | Bash | `git merge-base --is-ancestor c689634617ef9b59473b1a6c5ea64bc99cb08a29 6067e662b1d5f4036b38cef...` | allow | new (GO-2) |
| P07 | go1_step1_quiet_claude_echo_status | Bash | `git diff --quiet c689634617ef9b59473b1a6c5ea64bc99cb08a29 6067e662b1d5f4036b38cef7179f8cd94b5...` | allow | new (GO-2) |
| P08 | pb1a_record_symbolic_ref | Bash | `git symbolic-ref HEAD` | allow | new (GO-2) |
| P09 | pb1a_record_rev_parse | Bash | `git rev-parse HEAD` | allow | new (GO-2) |
| P10 | pb1a_precheck_status | Bash | `git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf'` | allow | new (GO-2) |
| P11 | pb1a_precheck_ls_remote | Bash | `git ls-remote --heads origin lane/sl1-guard` | allow | new (GO-2) |
| P12 | pb1a_chain | Bash | `git branch lane/sl1-guard 6067e662b1d5f4036b38cef7179f8cd94b56e50e && git symbolic-ref HEAD r...` | allow | new (GO-2) |
| P13 | pb1a_post_rev_parse | Bash | `git rev-parse HEAD                     # 6067e662b1d5f4036b38cef7179f8cd94b56e50e` | allow | new (GO-2) |
| P14 | pb1a_post_symbolic_ref | Bash | `git symbolic-ref HEAD                  # refs/heads/lane/sl1-guard` | allow | new (GO-2) |
| P15 | pb1a_post_status | Bash | `git status --porcelain=v1 -- . ':(exclude)docs/reference/reference.pdf'                      ...` | allow | new (GO-2) |
| P16 | pb1a_post_quiet_diff | Bash | `git diff --quiet 6067e662b1d5f4036b38cef7179f8cd94b56e50e -- . ':(exclude)docs/reference/refe...` | allow | new (GO-2) |
| P17 | pb1a_post_cached_quiet_diff | Bash | `git diff --cached --quiet 6067e662b1d5f4036b38cef7179f8cd94b56e50e -- . ':(exclude)docs/refer...` | allow | new (GO-2) |
| P18 | pb1a_post_protected_probe | Bash | `git status --porcelain=v1 -- docs/reference/reference.pdf                                    ...` | allow | new (GO-2) |
| P19 | go2_erratum_bare_name_status | Bash | `git diff --name-status 3d99edd2c16ed57821129d0b5f0aacb49ce10ce3 HEAD` | B2 | new (GO-2) |
| P20 | go2_go1_quiet_check | Bash | `git diff --quiet 6067e662b1d5f4036b38cef7179f8cd94b56e50e HEAD -- . ':(exclude)docs/reference...` | allow | new (GO-2) |
| P21 | go2_push | Bash | `git push origin lane/sl1-guard` | allow | new (GO-2) |
| P22 | go2_ls_remote | Bash | `git ls-remote --heads origin lane/sl1-guard` | allow | new (GO-2) |
| P23 | go2_corrected_name_status | Bash | `git diff --name-status 3d99edd2c16ed57821129d0b5f0aacb49ce10ce3 HEAD -- . ':(exclude)docs/ref...` | allow | new (GO-2) |
