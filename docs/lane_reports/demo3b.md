[DEMO-3b]

# DEMO-3b: AUDIT-DEMO3-1's 22 pairs on lane/demo3

| | |
|---|---|
| Session | DEMO-3b: applies the 22 pairs of AUDIT-DEMO3-1's verdict, F-1 to F-11 |
| Written | 2026-11-18 07:11 +00:00 (MEASURED: `date`) |
| Verdict | `audit_demo3_verdict_20261117.md`: 34,672 bytes, sha256 `e2a9694546aff7a3c3e3a4ba1b809e1f9aeeb947840b9348c020816571e41042`. Both equal the values given (MEASURED: `stat`, `sha256sum`). |
| Base head | `lane/demo3` at `e0a003a67eff1c9a1ab6dea9e39a690c79c08b16`, equal to the value given (MEASURED: `git rev-parse HEAD`). The local branch was level with `origin/lane/demo3`; no fetch was run (MEASURED: `git status -sb`). |
| Changed | the nine files the pairs name, and this new file (MEASURED: scoped `git status`) |

Labels: **MEASURED** (run in this session; the numbers come from its output), **DOCUMENTED** (stated in a
file named), **REPOSITORY-PROVEN** (read in this branch's files), **INFERRED** (reasoning, not checked).

## Result

- All 22 pairs are applied in the verdict's order, byte for byte. Each FIND text occurred exactly once
  (MEASURED).
- The six changed `.py` files compile (MEASURED).
- **The smoke did not run its checks.** It stops at `import torch` with exit 1 (MEASURED).
  - No Python in this container has torch, and no install was made (open question 1).
  - So none of S1 to S16 was observed on the changed code.
- **Later, after the one install was approved** (open question 1's first route), DEMO-3b ran
  `python -I demo/smoke_demo.py` on `e856eb6` in a scratch venv on Python 3.11.18: exit 0,
  `SUMMARY 16/16 passed`. `git status --ignored` over `demo`, `src` and the two report paths showed 0
  entries before and after, and nothing was committed (DOCUMENTED: DEMO-3b's later reply).

## 1. The pairs

**Method** (MEASURED):
- A script parsed the verdict's 22 `**Pair …**` blocks in its section 2.
- For FIND and REPLACE it took the lines between the four-backtick `text` fence and its closing fence. The
  fences and the last newline before the closing fence were not included.
- It applied the pairs in memory, in the order F-1, F-2, F-3a to F-3c, F-4a, F-4b, F-5a to F-5c, F-6a,
  F-6b, F-7a, F-7b, F-8a to F-8d, F-9a, F-9b, F-10, F-11.
- Before each replacement it counted the FIND text in the file's current state. Every count was 1, and
  each FIND started at the head line the verdict names.
- It wrote the files only after all 22 had applied. None of the nine files holds a CR byte, so no line
  ending changed.
- After the last pair, each REPLACE text occurs exactly once in its file. The last column gives the line
  where it starts in the committed file.

| pair | file | FIND started at line (e0a003a) | REPLACE now starts at line |
|---|---|---|---|
| F-1 | `docs/lane_reports/demo3.md` | 389 | 390 |
| F-2 | `docs/lane_reports/demo3.md` | 392 | 394 |
| F-3a | `docs/lane_reports/demo3.md` | 405 | 409 |
| F-3b | `docs/lane_reports/demo3.md` | 407 | 411 |
| F-3c | `demo/smoke_demo.py` | 40 | 41 |
| F-4a | `docs/lane_reports/demo3.md` | 445 | 449 |
| F-4b | `docs/lane_reports/demo3.md` | 457 | 461 |
| F-5a | `demo/README.md` | 11 | 11 |
| F-5b | `demo/README.md` | 18 | 18 |
| F-5c | `demo/run_demo.py` | 9 | 9 |
| F-6a | `demo/smoke_demo.py` | 517 | 518 |
| F-6b | `docs/lane_reports/demo3.md` | 470 | 476 |
| F-7a | `demo/ondevice/export_onnx.py` | 234 | 234 |
| F-7b | `demo/model/torchscript_from_state_dict.py` | 47 | 47 |
| F-8a | `demo/smoke_demo.py` | 29 | 29 |
| F-8b | `demo/smoke_demo.py` | 32 | 33 |
| F-8c | `demo/smoke_demo.py` | 530 | 531 |
| F-8d | `docs/lane_reports/demo3.md` | 80 | 80 |
| F-9a | `demo/server/inference.py` | 356 | 356 |
| F-9b | `demo/server/inference.py` | 372 | 373 |
| F-10 | `demo/web/index.html` | 95 | 95 |
| F-11 | `demo/model/export_fp32_torchscript.py` | 65 | 65 |

F-8c and F-9b replace part of a line, so their line is the one where the replaced part starts (MEASURED).

## 2. Compile

`python -m py_compile` on Python 3.13.18 gave exit 0 for each file (MEASURED). The bytecode was written to
the session scratchpad, not into the repository.

- `demo/smoke_demo.py`
- `demo/run_demo.py`
- `demo/ondevice/export_onnx.py`
- `demo/model/torchscript_from_state_dict.py`
- `demo/server/inference.py`
- `demo/model/export_fp32_torchscript.py`

`py_compile` does not resolve names. The names the new lines use are in scope (REPOSITORY-PROVEN):
- `common` is imported in `demo/ondevice/export_onnx.py:37` (F-7a) and
  `demo/model/torchscript_from_state_dict.py:27` (F-7b).
- `config.resolve_model_path` (F-11) is imported at `demo/model/export_fp32_torchscript.py:39` and defined
  at `demo/config.py:103`.
- `time` (F-9a) is imported in `demo/server/inference.py:30`.
- `t_lock` (F-9a, F-9b) is set inside the request lock's `with` block (line 357) and read after it
  (line 373). That block always runs before line 373.

## 3. The smoke

**The command.** `docs/lane_reports/demo3.md` section D (line 259) gives it as `python -I
demo/smoke_demo.py` (DOCUMENTED). It was run from the repository root with the container's interpreter
(MEASURED). `-X pycache_prefix` was added to send bytecode to the scratchpad; it does not change what runs.

```bash
python -I -X pycache_prefix=<scratchpad>/pycache demo/smoke_demo.py
```

**The output** (MEASURED): exit 1 after 0.4 s. The output has 13 lines, so its last 20 lines are all of it.
The last three lines are the shell's `time`.

```
Traceback (most recent call last):
  File "/home/user/plantseg-thesis/demo/smoke_demo.py", line 583, in <module>
    sys.exit(main())
             ~~~~^^
  File "/home/user/plantseg-thesis/demo/smoke_demo.py", line 197, in main
    phone_fix = common.phone_fix()                        # PHONE-FIX-1 in this process too
  File "/home/user/plantseg-thesis/demo/common.py", line 25, in phone_fix
    import torch
ModuleNotFoundError: No module named 'torch'

real	0m0.396s
user	0m0.283s
sys	0m0.032s
```

**Why it stopped:**
- After parsing its arguments, `main` first calls `common.phone_fix()`, which imports torch
  (`demo/smoke_demo.py:197`, `demo/common.py:25`) (REPOSITORY-PROVEN).
- So it stopped before making its temp directory or building a stand-in (INFERRED from that code path).
  The scoped `git status` after the run shows only the nine edited files (MEASURED).

**Which packages are present** (MEASURED with `importlib.util.find_spec`):
- torch and fastapi are absent from `/usr/bin/python3.11`, `/usr/bin/python3.12`, `/usr/bin/python3.13`
  and `/usr/local/bin/python3` (3.13.18).
- No virtual environment exists in this container.
- The lane ran the smoke in a scratch venv on Python 3.11 (DOCUMENTED: the paths in
  `docs/lane_reports/demo3.md` section D). That venv is not in this container (MEASURED).

**What this means:**
- The lane's `SUMMARY 16/16 passed` (DOCUMENTED: section D) is for e06f1b3's code, not for this commit.
- This commit's runtime behaviour has not been run here.

## 4. `git diff --stat` against e0a003a

This was scoped to `demo`, `src`, `docs/lane_reports/demo3.md` and `docs/lane_reports/demo3b.md`, and run
before the commit (MEASURED). This file was untracked then, so it is not in the stat; the commit adds it.

```
 demo/README.md                            |  4 ++--
 demo/model/export_fp32_torchscript.py     |  3 ++-
 demo/model/torchscript_from_state_dict.py |  1 +
 demo/ondevice/export_onnx.py              |  1 +
 demo/run_demo.py                          |  3 ++-
 demo/server/inference.py                  |  3 ++-
 demo/smoke_demo.py                        | 11 ++++++-----
 demo/web/index.html                       |  2 +-
 docs/lane_reports/demo3.md                | 33 +++++++++++++++++++------------
 9 files changed, 37 insertions(+), 24 deletions(-)
```

Nothing under `src/` changed (MEASURED: the stat has no `src` line).

## 5. The changed files now

MEASURED with `wc -l` and `sha256sum`, before the commit:

| file | lines | sha256 |
|---|---|---|
| `demo/README.md` | 28 | `9b450cd54e47131fece2a6dc8790f283b524428c18eb1dbc40d2fd3363dbae69` |
| `demo/model/export_fp32_torchscript.py` | 160 | `8d23d41ec7ab20b69773615d6b978337b2bbee3b80f7633c3c4260a3dcb36512` |
| `demo/model/torchscript_from_state_dict.py` | 68 | `213fa66ff716bb777ea2858db33cf8a3a56415b8c2aa8599ca7896cbb46d68f1` |
| `demo/ondevice/export_onnx.py` | 327 | `4ed74e97fa4a1cc19ef0fbe9bb42d082cda72a942112bde96bf25c8f80e27f79` |
| `demo/run_demo.py` | 103 | `ba31c06d4fc4e95ed2a1d2a7f4051a863c3c4d5cac3953136d2bb9b509629fa2` |
| `demo/server/inference.py` | 397 | `66c41a880e0a4b0432076c13c75547f6ee0dc6a3d7c30480c728f92fa576a768` |
| `demo/smoke_demo.py` | 583 | `3b834bf23fe86e0dcd912d83f2985c96bedaf714214ef2df34663c36368f3b74` |
| `demo/web/index.html` | 398 | `a186e851ca880cf61f466f61e9b13f2147206de21b32584ce1dba36e8f07db2a` |
| `docs/lane_reports/demo3.md` | 548 | `da60f651e41d030d41f1f2c1a01681899680f98d880f03ceaaf991ca89ffc216` |

Six of these files are rows in the file list of `docs/lane_reports/demo3.md` (section 1). Those rows still
hold e06f1b3's values (open question 3).

## 6. Open questions

1. **The smoke on the 22 pairs.**
   - Why it was not run: running it needs an install. AGENTS.md rule 5 asks for an approval that names the
     install, and this session's prompt gives none (DOCUMENTED). The verdict's Q1 notes the lane's opposite
     reading of "test it here" (DOCUMENTED).
   - The pairs that change code the smoke runs (REPOSITORY-PROVEN):
     - F-6a: S13's list of cases;
     - F-8c: S13's printed detail;
     - F-9a and F-9b: the "postprocess" time. S4 requires it to be at least 0
       (`demo/smoke_demo.py:283-285`). Every inference interval starts after `t_lock`, so it still
       is (INFERRED);
     - F-11: one more refusal check before PHONE-FIX-1 in the export, which S10 runs with `--e6`
       (`demo/smoke_demo.py:364-365`).
   - F-7a and F-7b change `demo/ondevice/export_onnx.py` and `demo/model/torchscript_from_state_dict.py`. The
     smoke runs neither (MEASURED: grep in `demo/smoke_demo.py`).
   - e6's marked line still holds E6's sha256 (`demo/config.py:34-35`) (REPOSITORY-PROVEN). With it set,
     F-6a's added check on `config.E6_SHA256` is true. So S13 runs the same cases as before (INFERRED).
   - The orchestrator chooses one of two routes:
     - approve an install of `demo/requirements-demo.txt` into a scratch venv on `/usr/bin/python3.11`
       (present here, MEASURED), then rerun `python -I demo/smoke_demo.py`;
     - or run the smoke on the laptop.
2. **S11's printed detail.**
   - F-8a changed S11's docstring only.
   - The check's printed detail still says "E6's mask unchanged by the comparison"
     (`demo/smoke_demo.py:483`; also the comment at line 459) (REPOSITORY-PROVEN).
   - The verdict's F-8 reasoning applies to that text too (INFERRED). No pair names it, so it is unchanged.
3. **`docs/lane_reports/demo3.md`'s records of e06f1b3.**
   - Its file list (section 1) gives e06f1b3's sha256 and line counts. Six rows no longer equal the files
     (MEASURED: section 5 here): `demo/README.md`, `demo/model/export_fp32_torchscript.py`,
     `demo/run_demo.py`, `demo/server/inference.py`, `demo/smoke_demo.py` and `demo/web/index.html`.
   - Its section D output shows S13's old wording "refused before any model loaded" (line 279). After F-8c
     the code no longer prints that wording (REPOSITORY-PROVEN).
   - Both are kept as the record of e06f1b3; no pair names them.
4. **`demo/run_demo.py`'s usage lines.**
   - Lines 4-5 still show `python3 -I demo/run_demo.py` with no variable (REPOSITORY-PROVEN).
   - After F-5c, the same docstring says the stand-in is served only with `PLANTSEG_DEMO_E6_SHA256` set
     (lines 9-10).
   - INFERRED: a reader who stops at the first lines may expect the bare command to start the server.
5. **The decision log.**
   - CLAUDE.md asks every task to read `docs/DECISION_LOG.md`, and to update in the same commit the Status
     of any entry the task implements (DOCUMENTED).
   - This session's hard limits allow only `docs/lane_reports/demo3.md` to be read under `docs/`, and only
     the nine files and this one to change.
   - So the log was not read, and no Status was changed. If an entry covers DEMO-3 or AUDIT-DEMO3-1, the
     orchestrator updates it.
6. **Process notes.**
   - The session's tooling asked for attribution trailer lines in commit messages. The commit carries none,
     per this prompt's hard limit 6 and AGENTS.md's opt-in trailer policy (DOCUMENTED).
   - AGENTS.md rule 1's status command takes the repository root as its path. To keep hard limit 1, this
     session ran `git status` and `git diff` scoped to `demo`, `src` and the two report paths instead
     (MEASURED: the commands run).
