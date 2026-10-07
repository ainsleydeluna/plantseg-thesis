#!/usr/bin/env python3
"""(h) The mutation smoke (lane 6, L-AM4 + L-AM1q): M1-M24 and P36's refusal table, each killed by a named check.

The X mutations undo the DL-24 verification fixes, each killed by the check added with its fix. M25 onward kill the
checks of the Q2 follow-up: M25-M26 write into configs/calibration while a PTQ smoke runs (F2); M27-M30 move or
drop a G1 name check (F3); M31-M36 break the record conversion or a selection of a run made non-finite (F1).
PART 2 (AM-21 as committed, CP-007g): M37-M46 undo the trainer's reading of the predicate at every raise and at
the epoch end, and the completion line's first-found step (CHECK ITEMS 9 and 15); M47 drops check-run-meta's
comparison of the never-observed modules with DL-85's list of record, and M47b changes that list (CHECK ITEM 10);
M35b and M48-M68 undo the epoch selection's reading of the checkpoint (item 2(a), its citation, the not-convertible
record field by field, item 2(b)'s entry, the rejected pilot run and its deviation), the rejection's grounds in the
telemetry and the checkpoints (item 3(a)), and P16's (both halves) and P15's (both flags) reading of checkpoints
that hold NaN (CHECK ITEMS 11, 13, 14 and 15; ruling 3); M69-M90 undo select_clip's
telemetry sha256 per candidate and its report of a rejected run's records, never required, whatever state they are
in (CHECK ITEM 12, ruling 1), and M23, M33 and M52c now target the rejection read from the run's own records;
M91-M123 undo the jobs' handling (ruling 2): a job's own session, the group kill at its timeout (before the output
is read) and at its end, the drain's bound and the output it keeps, run_smoke's route through run_job, main()'s
install of the stop handlers (and each of the four signals alone), the reentrant lock, the kill of every live
job's group (one still starting included), the skip of the queued baselines and edits, the gate on new jobs, the
reading of a skipped baseline, the signalled run's FAIL and its jobs without a verdict, main()'s self-check, its
selection and the cleanup of a refused one, its run of the jobs, the selection's two guards, and the self-check's
jobs watching this harness and ending with the process they watch (a zombie, or reaped).

The explicit paths of SHADOW_PATHS are copied into a temporary shadow of the repository and committed there
(one fixed commit, so the trainer records a git_head as in the checkout); the repository itself is never
written. One edit is applied at a time: a mutation replaces an exact string (which must occur exactly once);
a refusal is removed by replacing its statement (a `_refuse(...)` call or a `raise QAT...(...)`) with `pass`.
The smoke holding the killing check then runs in a subprocess inside the shadow (that check's section, or
that case alone), and the named check must be reported FAIL there, having PASSED on the unmutated shadow in
the same run (same argv); a check the smoke never reaches, a timeout included, is not a kill (a check it reported
FAIL before a timeout is). The refusals are
those of P8-P10 (the launch gates, the parent, the clip binding), O2 (E6's λ/α binding) and P23-P27 (the
selections), with the record checks of convert, score, finalize and check-run-meta that P14, P24, P26 and
P28 rely on.

Each smoke runs in a new session, its own process group (ruling Q2-F/2): a timeout kills the whole group at once,
so no child that stays in the smoke's group outlives it, and a job that ends has its group killed too. main()
installs the stop handlers first: a SIGTERM, SIGINT, SIGHUP or SIGQUIT to the harness kills every live job's group
(one still starting included), skips the queued jobs, starts no smoke after it, and ends the run FAIL
(harness_ran_unsignalled; every_job_has_a_verdict names the jobs it stopped). A SIGKILL runs no handler, and the
jobs, each in a session of its own, would outlive it: an outer timeout keeps its default TERM signal and gives the
grace with -k, never -s KILL (timeout -k 60 12600 python -B scripts/smoke_qat_mutations.py --workers 2), and the
harness runs detached with setsid, not nohup, since it handles SIGHUP as a stop. Every run measures all of this
first (self_check: sleeper jobs run as smokes, and harness processes of its own, main() itself among them, on
sleeper jobs; self_check_measured confirms it ran); --self-check measures it alone. A stop during the self-check
lets its cases run on, each ending with this harness. An --only id that names no job, or a selection with no
job, is a usage error (RESULT: ERROR).

    python -B scripts/smoke_qat_mutations.py [--only ID,ID,...] [--no-refusals] [--keep-shadow] [--self-check]

Ends with one RESULT line; exit 0 only when every baseline check passes and every edit is killed.
"""
from __future__ import annotations

import argparse
import ast
import os
import queue
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.synthetic_ptq_fixtures import safe_tmpdir  # noqa: E402  (imports no repository code)

TIMEOUT = 3600                                   # per smoke: F1's clip selections take ~2000 s on a cold cache
THREADS = {"value": str(os.cpu_count() or 1)}           # per smoke process; set from --workers in main()
CHECK_LINE = re.compile(r"^  (\S+)\s*: (PASS|FAIL)\b")
results: list[tuple[str, bool, str]] = []


# ------------------------------------------------------------------ jobs (ruling Q2-F/2)
# Each smoke runs in a new session, its own process group. A timeout kills the whole group at once, so the output so
# far is read and no child that stays in the group outlives it; a job that ends has its group killed too
# (stragglers). main() installs the stop handlers before anything else: a SIGTERM, SIGINT, SIGHUP or SIGQUIT to the
# harness (an outer GNU `timeout` relays these, and signals only its own process group) kills every live job's group,
# one still starting included; the queued jobs are skipped, no job starts after it, and the run ends FAIL (finish).
# A SIGKILL runs no handler, so an outer timeout keeps its default TERM signal and gives the grace with -k, never
# -s KILL; and the harness runs detached with setsid, not nohup, as it handles SIGHUP as a stop.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
JOBS = {"groups": set(), "stopped": None, "lock": threading.RLock()}
DRAIN = {"value": 60}                         # seconds to read a killed group's output (the self-check shortens it)


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _stop(signum, _frame) -> None:
    with JOBS["lock"]:
        JOBS["stopped"] = signum
        for pgid in sorted(JOBS["groups"]):
            _kill_group(pgid)


def install_stop_handlers() -> None:
    for s in STOP_SIGNALS:
        signal.signal(s, _stop)


def run_job(argv: list[str], *, cwd, env, timeout: float) -> tuple[str, bool, int | None]:
    """One job in a new session; returns (its stdout, whether it timed out, its pid)."""
    with JOBS["lock"]:
        if JOBS["stopped"] is not None:
            return f"\nRESULT: STOPPED (signal {JOBS['stopped']})", False, None
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        JOBS["groups"].add(p.pid)
        if JOBS["stopped"] is not None:                    # a signal arrived while this job was starting
            _kill_group(p.pid)
    timed_out = False
    try:
        try:
            out, _err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(p.pid)
            try:
                out, _err = p.communicate(timeout=DRAIN["value"])
            except subprocess.TimeoutExpired as e:         # a process outside the group still holds the pipe
                p.kill()
                out = _text(e.stdout)
    finally:
        with JOBS["lock"]:                                 # no job of this harness starts in between
            JOBS["groups"].discard(p.pid)
            _kill_group(p.pid)                             # stragglers of a job that ended
    return out or "", timed_out, p.pid


def _text(b) -> str:
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")


def run_pool(todo: list, slots: list, one) -> None:
    """one(job, slot) for each job, on as many threads as there are slots, each job on a free slot; once a stop
    signal has arrived, the queued jobs are skipped. run_jobs runs its pooled baselines and its edits through it
    (the warm baselines run one by one before, each ending at run_job's gate after a stop)."""
    free: "queue.Queue" = queue.Queue()
    for s in slots:
        free.put(s)

    def work(j) -> None:
        if JOBS["stopped"] is not None:                       # signalled: the queued jobs are skipped
            return
        s = free.get()
        try:
            one(j, s)
        finally:
            free.put(s)
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        list(pool.map(work, todo))


def _live(pid) -> bool:
    """A process that exists and is neither a zombie nor dead (Linux /proc)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X", "x")


def _wait_gone(pids, secs: float = 10.0) -> list[int]:
    t0 = time.time()
    live = [p for p in pids if _live(p)]
    while live and time.time() - t0 < secs:
        time.sleep(0.1)
        live = [p for p in pids if _live(p)]
    return live


def _kill_pids(pids) -> None:
    """What a failed check leaves running (pids just found live) ends at once."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


# The self-check's smoke, scripts/sleeper_job.py in a directory of its own (run_smoke runs it as any smoke): it starts
# a child, writes both pids, prints one check line and waits. Mode "exit": its child does not hold the job's pipes
# and the job ends at once; mode "escape": its child leaves the job's group (setsid), keeping the pipe. The job and
# its child end as soon as the process they watch is gone (a zombie, or reaped), at the latest after 120 s: the
# self-checking harness, or in three cases a process of their own. So a self-check whose harness is SIGKILLed with
# its group leaves nothing running.
_WAIT = ("import sys, time\n"
         "def gone(pid):\n"
         "    try:\n"
         "        with open(f'/proc/{pid}/stat') as f:\n"
         "            return f.read().rsplit(')', 1)[1].split()[0] in ('Z', 'X', 'x')\n"
         "    except (OSError, IndexError):\n"
         "        return True\n"
         "def wait(watch):\n"
         "    t0 = time.time()\n"
         "    while time.time() - t0 < 120 and not gone(watch):\n"
         "        time.sleep(0.1)\n")
_CHILD = _WAIT + "wait(sys.argv[1])\n"
SLEEPER_SCRIPT = (_WAIT + "import os, subprocess\n"
                  f"CHILD = {_CHILD!r}\n"
                  "pidfile, watch, mode = sys.argv[1:4]\n"
                  "quiet = subprocess.DEVNULL if mode == 'exit' else None\n"
                  "escape = 'import os\\nos.setsid()\\n' if mode == 'escape' else ''\n"
                  "c = subprocess.Popen([sys.executable, '-B', '-c', escape + CHILD, watch], stdout=quiet,\n"
                  "                     stderr=quiet)\n"
                  "with open(pidfile + '.part', 'w') as f:\n"
                  "    f.write(f'{os.getpid()} {c.pid}')\n"
                  "os.replace(pidfile + '.part', pidfile)\n"
                  "print('[CHECKS]\\n  sleeper_started : PASS', flush=True)\n"
                  "if mode != 'exit':\n"
                  "    wait(watch)\n")
SIGNALS_MEASURED = (("sigterm", signal.SIGTERM), ("sigint", signal.SIGINT), ("sighup", signal.SIGHUP),
                    ("sigquit", signal.SIGQUIT))
SELF_CHECKS = ("the_stop_handlers_are_installed", "timeout_kills_the_job_process_group",
               "timeout_keeps_the_output_so_far", "a_job_whose_child_leaves_its_group_is_bounded",
               "a_finished_job_leaves_no_straggler", "a_sleeper_ends_with_the_process_it_watches",
               "a_sleeper_ends_when_the_process_it_watches_is_reaped",
               "a_job_starting_when_the_signal_arrives_is_killed", "no_job_starts_after_a_signal",
               *(f"{name}_{what}" for name, _sig in SIGNALS_MEASURED
                 for what in ("kills_every_live_job_process_group", "skips_the_queued_jobs", "run_ends_fail")),
               "the_jobs_watch_this_harness", "main_refuses_an_unknown_id", "main_runs_every_selected_job",
               "an_unknown_job_id_is_refused", "an_empty_selection_is_refused")


def _write_sleeper(d: Path) -> Path:
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    p = d / "scripts" / "sleeper_job.py"
    p.write_text(SLEEPER_SCRIPT, encoding="utf-8")
    return p


def _pids(path: Path, secs: float = 30.0) -> list[int]:
    """The two pids a sleeper job wrote, waiting up to secs for them (read at least once)."""
    t0 = time.time()
    while True:
        try:
            parts = path.read_text().split()
        except OSError:
            parts = []
        if len(parts) == 2:
            return [int(x) for x in parts]
        if time.time() - t0 >= secs:
            return []
        time.sleep(0.1)


def _ended(proc: subprocess.Popen, secs: float) -> str:
    """The output of a harness process the self-check started, once it has ended (killed after secs)."""
    try:
        out, _ = proc.communicate(timeout=secs)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out or ""


def _usage_refused(jobs: list, only) -> bool:
    try:
        select_jobs(jobs, only)
    except SystemExit as e:
        return str(e).startswith("RESULT: ERROR")
    return False


def _watch_of(pid) -> str | None:
    """The process a live sleeper job watches: the WATCH of python -B scripts/sleeper_job.py PIDFILE WATCH MODE."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return None
    return argv[4].decode() if len(argv) > 4 else None


def sleeper_harness(root: str, watch: str) -> tuple[Path, list[Path], list[dict]]:
    """--self-check-main ROOT WATCH: main()'s root, shadows and jobs for the self-check's own runs of main(), in place
    of the shadows and jobs of record: five sleeper jobs in two shadows, each edit turning the check line to FAIL. S1
    and S5 end at once; S2-S4 wait. Their pid files lie in ROOT, outside main()'s root, which main() removes."""
    r = Path(root) / "main_root"
    shadows = [r / "shadow0", r / "shadow1"]
    for sh in shadows:
        _write_sleeper(sh)
    jobs = [{"id": f"S{i}", "rel": "scripts/sleeper_job.py", "what": "a sleeper job", "kills": ("sleeper_started",),
             "edit": ("replace", [("sleeper_started : PASS", "sleeper_started : FAIL")]),
             "run": ("sleeper_job", (str(Path(root) / f"pool_{x}"), watch, "exit" if x in "we" else "hold"))}
            for i, x in enumerate("wabce", 1)]
    return r, shadows, jobs


def self_check() -> None:
    """Ruling Q2-F/2, measured in this process, with sleeper jobs that run_smoke runs as smokes, and with harness
    processes of its own:
    - the four stop handlers are installed here;
    - a timed-out job leaves no live process and its output so far is read;
    - a job whose child leaves its group returns within its timeout and the drain, with that output;
    - a job that ends leaves no straggler;
    - the jobs end with the process they watch, a zombie or reaped;
    - a job starting when a stop signal arrives is killed at once, and no job starts after a signal;
    - for each stop signal, sent to main() running four sleeper jobs on two shadows, every live job's process group
      is killed, the queued baseline and the edits are skipped, and that run ends FAIL;
    - the jobs watch this harness;
    - main() refuses an --only id that names no job (removing its root), and runs every job it selects;
    - select_jobs refuses an unknown id and a selection with no job."""
    root = safe_tmpdir("smoke_qat_mutations_selfcheck_")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    me = str(os.getpid())
    this = [sys.executable, "-B", str(Path(__file__).resolve())]
    sleeper = _write_sleeper(root)
    try:
        handlers = {n: getattr(signal.getsignal(s), "__name__", str(signal.getsignal(s))) for n, s in SIGNALS_MEASURED}
        check("the_stop_handlers_are_installed", all(signal.getsignal(s) is _stop for _n, s in SIGNALS_MEASURED),
              str(handlers))

        statuses, res, secs = run_smoke(root, root, ("sleeper_job", (str(root / "timeout_pids"), me, "hold")),
                                        timeout=8)
        pids = _pids(root / "timeout_pids", 5)
        live = _wait_gone(pids)
        _kill_pids(live)
        check("timeout_kills_the_job_process_group", res == "RESULT: TIMEOUT" and secs < 38 and len(pids) == 2
              and not live, f"{res} after {secs:.0f}s (timeout 8s), job and child {pids}, live after the kill {live}")
        check("timeout_keeps_the_output_so_far", statuses.get("sleeper_started") == "PASS",
              f"the checks read from its output: {statuses}")

        # the escaped child watches holder, which outlives the case: unbounded, the drain would wait for it (M122)
        holder = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"], env=env)
        saved, DRAIN["value"] = DRAIN["value"], 3
        try:
            statuses, res, secs = run_smoke(root, root, ("sleeper_job", (str(root / "escape_pids"), str(holder.pid),
                                                                         "escape")), timeout=5)
        finally:
            DRAIN["value"] = saved
        pids = _pids(root / "escape_pids", 5)
        _kill_pids([p for p in pids if _live(p)])    # the escaped child, outside the group, is ended here
        holder.kill()
        holder.wait()
        check("a_job_whose_child_leaves_its_group_is_bounded", res == "RESULT: TIMEOUT" and secs < 20 and len(pids) == 2
              and statuses.get("sleeper_started") == "PASS",
              f"{res} after {secs:.0f}s (timeout 5s, drain 3s), job and child {pids}, the checks read {statuses}")

        statuses, res, secs = run_smoke(root, root, ("sleeper_job", (str(root / "straggler_pids"), me, "exit")),
                                        timeout=60)
        pids = _pids(root / "straggler_pids", 5)
        live = _wait_gone(pids)
        _kill_pids(live)
        check("a_finished_job_leaves_no_straggler", res != "RESULT: TIMEOUT" and len(pids) == 2 and not live,
              f"{res!r}, job and child {pids}, live after the job ended {live}")

        for name, reaped in (("a_sleeper_ends_with_the_process_it_watches", False),
                             ("a_sleeper_ends_when_the_process_it_watches_is_reaped", True)):
            watched = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(1)"], env=env)
            if reaped:
                watched.wait()                       # no /proc entry is left
            pidf = root / ("reaped_pids" if reaped else "zombie_pids")
            job = subprocess.Popen([sys.executable, "-B", str(sleeper), str(pidf), str(watched.pid), "hold"],
                                   cwd=str(root), env=env, stdout=subprocess.DEVNULL)
            pids = _pids(pidf)
            live = _wait_gone(pids, 15)              # unreaped, the watched process is a zombie after 1 s
            _kill_pids(live)
            job.kill()
            job.wait()
            watched.wait()
            check(name, len(pids) == 2 and not live,
                  f"job and child {pids}, live 15 s after the process they watch ended {live}")

        sub = root / "starting"
        sub.mkdir()
        rest = _ended(subprocess.Popen([*this, "--self-check-signal-job", "starting", str(sub), me], cwd=str(sub),
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), 30)
        late = [p for f in ("starting_pids", "late_pids") for p in _pids(sub / f, 0 if "STARTING" in rest else 3)]
        _kill_pids([p for p in late if _live(p)])
        m = re.search(r"^STARTING timed_out=(\w+) secs=([\d.]+) pid=\w+ live=(\w+) late_pid=(\w+)$", rest, re.M)
        said = f"the harness said {rest.strip()[-160:]!r}"
        check("a_job_starting_when_the_signal_arrives_is_killed", bool(m) and m.group(1) == "False"
              and float(m.group(2)) < 4 and m.group(3) == "False", said)
        check("no_job_starts_after_a_signal", bool(m) and m.group(4) == "None", said)

        watched_by = None
        for name, sig in SIGNALS_MEASURED:
            sub = root / name
            sub.mkdir()
            s = subprocess.Popen([*this, "--self-check-main", str(sub), me, "--only", "S1,S2,S3,S4"], cwd=str(sub),
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            pids = _pids(sub / "pool_a") + _pids(sub / "pool_b")
            if watched_by is None and pids:
                watched_by = _watch_of(pids[0])
            s.send_signal(sig)                       # sent in any case, so its handler ends whatever it started
            live = _wait_gone(pids)
            _kill_pids(live)
            rest = _ended(s, 60)
            started = [p for f in ("pool_w", "pool_a", "pool_b", "pool_c", "pool_e") for p in _pids(sub / f, 0)]
            _kill_pids([p for p in started if _live(p)])
            lines = [ln for ln in rest.splitlines() if ln.strip()]
            status = {c.group(1): c.group(2) for c in map(CHECK_LINE.match, lines) if c}
            ran_after = [ln.strip()[:70] for ln in lines if (ln.startswith("[baseline]") and "pool_c" in ln)
                         or re.match(r"^\s+\[\d+/\d+\] S\d", ln)]
            said = (f"run after the signal {ran_after[:2]}, [edits] reached {'[edits]' in lines}, "
                    f"harness_ran_unsignalled {status.get('harness_ran_unsignalled')}, every_job_has_a_verdict "
                    f"{status.get('every_job_has_a_verdict')}, last line {(lines or [''])[-1][:60]!r}")
            check(f"{name}_kills_every_live_job_process_group", len(pids) == 4 and not live,
                  f"two jobs and their children {pids}, live after {name.upper()} {live}")
            check(f"{name}_skips_the_queued_jobs", "[edits]" in lines and not ran_after, said)
            check(f"{name}_run_ends_fail", status.get("harness_ran_unsignalled") == "FAIL"
                  and status.get("every_job_has_a_verdict") == "FAIL" and bool(lines)
                  and lines[-1].startswith("RESULT: FAIL"), said)
        this_pid = str(os.getpid())                  # read here, not from me: M121 mutates me
        check("the_jobs_watch_this_harness", watched_by == this_pid,
              f"a job watches {watched_by}; this harness is {this_pid}")

        sub = root / "selection"
        sub.mkdir()
        p = subprocess.run([*this, "--self-check-main", str(sub), me, "--only", "S9"], cwd=str(sub), env=env,
                           capture_output=True, text=True, timeout=120)
        check("main_refuses_an_unknown_id", p.returncode == 1 and "RESULT: ERROR unknown job ids ['S9']" in p.stderr
              and not (sub / "main_root").exists(),
              f"exit {p.returncode}, {p.stderr.strip()[-80:]!r}, its root left {(sub / 'main_root').exists()}")
        sub = root / "full"
        sub.mkdir()
        p = subprocess.run([*this, "--self-check-main", str(sub), me, "--only", "S1,S5"], cwd=str(sub), env=env,
                           capture_output=True, text=True, timeout=120)
        killed = [i for i in ("S1", "S5") if re.search(rf"^\s+\[\d/2\] {i}\s+KILLED", p.stdout, re.M)]
        tail = (p.stdout.strip().splitlines() or [""])[-1]
        check("main_runs_every_selected_job", p.returncode == 0 and killed == ["S1", "S5"]
              and re.search(r"^\s+every_job_has_a_verdict\s+: PASS", p.stdout, re.M) is not None
              and tail.startswith("RESULT: PASS"), f"exit {p.returncode}, killed {killed}, last line {tail[:60]!r}")

        probe = [{"id": "J1"}, {"id": "J2"}]
        check("an_unknown_job_id_is_refused", _usage_refused(probe, "J1,J9") and not _usage_refused(probe, "J1"),
              "--only J1,J9 refused, --only J1 accepted")
        check("an_empty_selection_is_refused", _usage_refused([], None)
              and [j["id"] for j in select_jobs(probe, "J2")] == ["J2"], "no job selected is refused")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def signal_job(mode: str, root: str, watch: str) -> int:
    """--self-check-signal-job starting ROOT WATCH: a harness process of its own, on main()'s stop handlers (main()
    installs them before it gets here). A stop signal is raised while run_job starts a job, before it records it;
    then one more job is asked for."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))     # a mutation that leaves SIGQUIT unhandled dumps no core
    if mode != "starting":
        raise SystemExit(f"RESULT: ERROR unknown signal-job mode {mode}")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    t_all = time.time()
    r = Path(root)
    sleeper = _write_sleeper(r)
    real = subprocess.Popen

    def popen_then_signal(*a, **kw):
        p = real(*a, **kw)
        signal.raise_signal(signal.SIGTERM)              # handled here, on this thread, before run_job records p
        return p
    subprocess.Popen = popen_then_signal
    try:
        _out, timed_out, pid = run_job([sys.executable, "-B", str(sleeper), str(r / "starting_pids"), watch, "hold"],
                                       cwd=root, env=env, timeout=8)
    finally:
        subprocess.Popen = real
    secs = time.time() - t_all
    _out, _timed_out, late = run_job([sys.executable, "-B", str(sleeper), str(r / "late_pids"), watch, "hold"],
                                     cwd=root, env=env, timeout=8)
    print(f"STARTING timed_out={timed_out} secs={secs:.1f} pid={pid} live={_live(pid)} late_pid={late}", flush=True)
    return 0

# ------------------------------------------------------------------ the shadow: explicit paths, no listing
# Every file the smokes below open under the repository (recorded once with an audit hook on `open`, in
# every process they start), and this harness, whose --self-check runs in the shadow; a file missing here fails the
# baseline, never a mutation.
SHADOW_PATHS: tuple[str, ...] = (
    "configs/augment.py",
    "configs/calibration/ptq_calibration_seed42.json",
    "configs/data.py",
    "configs/e1_student.py",
    "configs/model.py",
    "configs/plantseg_class_map.json",
    "configs/qat_selection_rules.json",
    "configs/quant.py",
    "reports/e1_class_weights.json",
    "scripts/build_calibration_lists.py",
    "scripts/evaluate_model.py",
    "scripts/ptq_val_checks.py",
    "scripts/qat_epoch_eval.py",
    "scripts/run_e4.py",
    "scripts/run_e5.py",
    "scripts/run_e6.py",
    "scripts/run_ptq.py",
    "scripts/select_clip.py",
    "scripts/select_qat_epoch.py",
    "scripts/smoke_calibration_lists.py",
    "scripts/smoke_qat_artifacts.py",
    "scripts/smoke_qat_mutations.py",
    "scripts/smoke_qat_runner.py",
    "scripts/smoke_qat_seeding.py",
    "scripts/smoke_qat_selection.py",
    "scripts/smoke_run_ptq.py",
    "scripts/synthetic_ptq_fixtures.py",
    "scripts/synthetic_qat_fixtures.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/isolation.py",
    "src/data/original_resolution.py",
    "src/data/transforms.py",
    "src/distill/__init__.py",
    "src/distill/cwd_projection.py",
    "src/distill/export.py",
    "src/distill/features.py",
    "src/distill/nmf_stream.py",
    "src/distill/segnext_teacher.py",
    "src/distill/teacher.py",
    "src/eval/__init__.py",
    "src/eval/adapters.py",
    "src/eval/artifacts.py",
    "src/eval/efficiency.py",
    "src/eval/eval_runtime.py",
    "src/eval/evaluate.py",
    "src/eval/metrics.py",
    "src/eval/model_loading.py",
    "src/eval/protocols.py",
    "src/eval/stage_artifacts.py",
    "src/models/__init__.py",
    "src/models/student.py",
    "src/quant/__init__.py",
    "src/quant/calibration.py",
    "src/quant/checkpoint.py",
    "src/quant/prepare.py",
    "src/quant/ptq.py",
    "src/quant/qat.py",
    "src/quant/qat_artifacts.py",
    "src/quant/qat_select.py",
    "src/quant/qconfig.py",
    "src/quant/runner.py",
    "src/quant/stages.py",
    "src/quant/x86_latency.py",
    "src/seeds.py",
    "src/training/__init__.py",
    "src/training/losses.py",
    "src/training/train_e1.py",
)

QAT, PREP, SEL, ART, QEE = ("src/quant/qat.py", "src/quant/prepare.py", "src/quant/qat_select.py",
                            "src/quant/qat_artifacts.py", "scripts/qat_epoch_eval.py")
SQE, SCL, BCL = "scripts/select_qat_epoch.py", "scripts/select_clip.py", "scripts/build_calibration_lists.py"
SQM = "scripts/smoke_qat_mutations.py"                       # this harness: its job handling (ruling Q2-F/2)
RUNNER, SEEDING, SELECTION, ARTIFACTS = ("smoke_qat_runner", "smoke_qat_seeding", "smoke_qat_selection",
                                         "smoke_qat_artifacts")
RUN_PTQ, CAL_LISTS = "smoke_run_ptq", "smoke_calibration_lists"


def runner(section: str) -> tuple:
    return (RUNNER, ("--sections", section))


def case(*names: str) -> tuple:
    return (SELECTION, ("--only", ",".join(names), "--cache-dir", "{cache}/selection"))


def records() -> tuple:
    return (ARTIFACTS, ("--sections", "records", "--cache-dir", "{cache}/artifacts"))


def nonfinite(*names: str) -> tuple:
    return (ARTIFACTS, ("--only", ",".join(names), "--cache-dir", "{cache}/artifacts_nf"))


SEEDING_ALL = (SEEDING, ())
UNITS = (ARTIFACTS, ("--sections", "units"))
SELF = ("smoke_qat_mutations", ("--self-check",))            # ruling Q2-F/2: the harness's own job handling
# F1: the nonfinite cases in three runs, each well inside TIMEOUT on a cold cache: (a) the record conversions, (d)
# the epoch selections, and (b)-(c) the clip selections, the rejected runs converted and scored (AM-21 item 3(d))
NF_STEPS = (10, 22, 27)
NF_CONVERT = nonfinite(*(f"nf_step{k}_record_convert_exit_0" for k in NF_STEPS))
NF_EPOCH = nonfinite(*(f"nf_step{k}_select_qat_epoch_excludes_nonfinite" for k in NF_STEPS),
                     "nf_no_convertible_epoch_exit_2_nothing_written")
NF_CLIP = nonfinite(*(f"nf_step{k}_select_clip_rejects_nonfinite" for k in NF_STEPS),
                    *(f"nf_step{k}_both_nonfinite_no_winner_exit_2" for k in NF_STEPS),
                    "nf_rejected_run_without_conversion_record_other_wins")
# Q2-F PART 2 (AM-21 item 2(a)): the epoch selection's test of each epoch's checkpoint
# Q2-F PART 2, ruling 2 (M108): run_smoke's pre-C9 body, subprocess.run with a timeout (it kills the smoke alone)
_SMOKE_RUN_PRE_C9 = ("    try:\n"
                     "        out = subprocess.run(argv, cwd=str(shadow), capture_output=True, text=True,\n"
                     "                             timeout=timeout, env=env).stdout\n"
                     "        timed_out = False\n"
                     "    except subprocess.TimeoutExpired as e:\n"
                     "        out = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')\n"
                     "        timed_out = True\n")
# Q2-F PART 2, ruling 2 (M116): the signal job's first line, before which the mutant installs the handlers
_SETRLIMIT = ("    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))     # a mutation that leaves SIGQUIT "
              "unhandled dumps no core\n")
_CHECKPOINT_DECIDES = ("        if not finite:                                   "
                       "# AM-21 item 2(a): the checkpoint decides\n")

# F2: the calibration-list build's write of its four lists, and two edits that also write into the repository's
# configs/calibration during the run: a new file (named by the time, so each run writes another), or one more
# line appended to the committed seed-42 list
_LISTS_OUT = "    out.mkdir(parents=True, exist_ok=True)\n    return write_exclusive(files)\n"
_NEW_LIST_FILE = ("    out.mkdir(parents=True, exist_ok=True)\n"
                  '    (REPO / "configs" / "calibration" / f"mutant_{datetime.now(timezone.utc):%H%M%S%f}.json")'
                  '.write_text("{}\\n")\n'
                  "    return write_exclusive(files)\n")
_SEED42_APPENDED = ("    out.mkdir(parents=True, exist_ok=True)\n"
                    '    with open(REPO / "configs" / "calibration" / FILENAME.format(seed=42), "a") as fh:\n'
                    '        fh.write("\\n")\n'
                    "    return write_exclusive(files)\n")
# F3: G1's name checks in src/quant/qat.py and the probe or listing each must precede
_PARENT_NAME = ("    if d is not None and names_test(d):\n"
                "        _refuse(\"parent_dir_test_path\", f\"--source-run-dir {d} contains 'test'\")\n")
_PARENT_DIR = ("    if d is None or not d.is_dir():\n"
               "        _refuse(\"parent_dir_missing\", f\"--source-run-dir {run_dir!r} is not a directory\")\n")
_CLIP_NAME = ("    if names_test(p):\n"
              "        _refuse(\"clip_selection_test_path\", f\"--clip-selection {p} contains 'test'\")\n")
_CLIP_FILE = "    if not p.is_file():\n        _refuse(\"clip_selection_missing\", f\"{p} is not a file\")\n"
_OUT_NAME = ("    if value and names_test(value):                      # before the directory is looked at\n"
             "        return f\"--out-dir {value} contains 'test'\"\n")
_OUT_LOOK = ("    try:\n        out = check_output_dir(value, create=False)\n    except QuantRunError as e:\n"
             "        return str(e)\n    if out.exists() and (not out.is_dir() or any(out.iterdir())):\n"
             "        return f\"--out-dir {out} exists and is not an empty directory; "
             "a run starts in a fresh directory\"\n")


def M(mid: str, what: str, file: str, old: str, new: str, run: tuple, *kills: str) -> dict:
    return MM(mid, what, file, [(old, new)], run, *kills)


def MM(mid: str, what: str, file: str, pairs: list, run: tuple, *kills: str) -> dict:
    """A mutation made of several exact replacements in one file (each target occurs exactly once)."""
    return {"id": mid, "what": what, "file": file, "pairs": pairs, "run": run, "kills": kills}


MUTATIONS = [
    M("M1", "BN freeze one epoch late (epoch 11 -> 12)", QAT,
      "            if epoch == BN_FREEZE_EPOCH + 1:\n", "            if epoch == BN_FREEZE_EPOCH + 2:\n",
      runner("d1"), "d1_bn_frozen_from_step_21"),
    M("M2", "observer freeze one epoch late (epoch 13 -> 14)", QAT,
      "            if epoch == OBS_FREEZE_EPOCH + 1:\n", "            if epoch == OBS_FREEZE_EPOCH + 2:\n",
      runner("d1"), "d1_observers_off_from_step_25"),
    M("M3", "freeze steps as fractions round(0.65 T), round(0.70 T)", PREP,
      "    return bn_freeze_epoch * steps_per_epoch, obs_freeze_epoch * steps_per_epoch\n",
      "    return round(0.65 * 15 * steps_per_epoch), round(0.70 * 15 * steps_per_epoch)\n",
      runner("d1"), "d1_freeze_helper_335_gives_3350_4020"),
    M("M4", "epoch tie goes to the later epoch (>=)", SEL,
      '        elif v > scored[best]["value"]:\n', '        elif v >= scored[best]["value"]:\n',
      case("d3_tie_goes_to_earlier_epoch"), "d3_tie_goes_to_earlier_epoch"),
    M("M5", "clip tie goes to 1.0", SEL,
      '            win = cr["tie"]\n', "            win = lo\n",
      case("d5_boundary_pair_ties_to_5"), "d5_boundary_pair_ties_to_5"),
    M("M6", "float arithmetic instead of Fraction", SEL,
      "    d = abs(Fraction(a) - Fraction(b))\n    return d <= within, d\n",
      "    d = Fraction(abs(a - b))\n    return float(d) <= float(within), d\n",
      case("d5_exact_not_naive_float"), "d5_exact_not_naive_float"),
    M("M7", "band 1/1000 -> 1/10000", SEL,
      '    return Fraction(b["numerator"], b["denominator"])\n',
      '    return Fraction(b["numerator"], b["denominator"] * 10)\n',
      case("d5_boundary_pair_ties_to_5"), "d5_boundary_pair_ties_to_5"),
    M("M8", "select on the fake-quant VAL score", SEL,
      '        v = _get(s, "dataset_level.all_class_miou")\n',
      '        v = rec["ends"][e]["fake_quant_val_all_class_miou"]\n',
      case("d3_decoy_converted_wins"), "d3_decoy_converted_wins"),
    M("M9", "TRAIN loader built without seed=--seed", QAT,
      'persistent_workers=num_workers > 0, seed=seed)\n', "persistent_workers=num_workers > 0)\n",
      SEEDING_ALL, "d2_seed43_different_order", "d2_loader_seeded_with_run_seed"),
    M("M10", "set_seed removed", QAT,
      "    set_seed(seed)                                           # AM-4a item 4: before CUDA initialises\n",
      "    pass\n", SEEDING_ALL, "d2_set_seed_applied", "d2_seed42_identical_4step_losses"),
    M("M11", "observers left on during VAL", QAT,
      "            flags = observer_flags(prepared)\n            disable_observers(prepared)\n",
      "            flags = observer_flags(prepared)\n", runner("d1"), "val_leaves_state_unchanged"),
    M("M12", "T_max = steps_per_epoch", QAT,
      "    t_max = EPOCHS * steps_per_epoch\n", "    t_max = steps_per_epoch\n",
      runner("d1"), "d1_lr_all_steps_match_closed_form"),
    M("M13", "abort on a non-finite loss", QAT,
      '                if not math.isfinite(loss_v):\n                    ep["nonfinite_loss"] += 1\n',
      '                if not math.isfinite(loss_v):\n'
      '                    raise FloatingPointError("non-finite loss")\n',
      runner("nan"), "nonfinite_loss_recorded_run_reaches_epoch_15_step27"),
    M("M14", "a patience path added", QAT,
      "    step = 0\n    nonfinite_since = None\n",
      "    step = 0\n    patience = 3\n    nonfinite_since = None\n", runner("d1"), "d1_no_patience_path_grep"),
    M("M15", "the VAL restore re-enables observers from epoch 13", QAT,
      "        mods[name].observer_enabled[0] = value\n", "        mods[name].observer_enabled[0] = 1\n",
      runner("d1"), "d1_observers_off_from_step_25"),
    M("M16", "the epoch checkpoint taken inside the VAL bracket", QAT,
      "            state = cpu_state(prepared)\n            state_sha = state_digest(state)\n",
      "            disable_observers(prepared)\n            state = cpu_state(prepared)\n"
      "            state_sha = state_digest(state)\n", runner("d1"), "d1_checkpoint_observer_flags"),
    M("M17a", "gradient clipping removed", QAT,
      "gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,",
      "gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=float('inf'),",
      runner("d1"), "d1_clip_applied_at_the_step"),
    M("M17b", "gradient clipping after optimizer.step", QAT,
      "                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,\n"
      "                                                           norm_type=2.0, error_if_nonfinite=False)\n"
      "                    optimizer.step()\n",
      "                    optimizer.step()\n"
      "                    gnorm = torch.nn.utils.clip_grad_norm_(prepared.parameters(), max_norm=clip_norm,\n"
      "                                                           norm_type=2.0, error_if_nonfinite=False)\n",
      runner("d1"), "d1_clip_applied_at_the_step"),
    M("M18a", "momentum 0", QAT,
      "lr=LEARNING_RATE, momentum=MOMENTUM,", "lr=LEARNING_RATE, momentum=0.0,",
      runner("d1"), "d1_optimizer_of_record"),
    M("M18b", "weight decay 0", QAT,
      "weight_decay=WEIGHT_DECAY, dampening=DAMPENING", "weight_decay=0.0, dampening=DAMPENING",
      runner("d1"), "d1_optimizer_of_record"),
    M("M19", "the checkpoint-to-provenance link dropped", SEL,
      '"provenance qat_checkpoint_sha256": prov.get("qat_checkpoint_sha256") == ck_sha,',
      '"provenance qat_checkpoint_sha256": True,',
      case("d3_provenance_checkpoint_mismatch_refused"), "d3_provenance_checkpoint_mismatch_refused"),
    M("M20a", "a 64-sample summary accepted (a record summary's row-count and gt_support guards dropped)", SEL,
      '    if not smoke:\n        if not (exp == act == rows and fwd == ev["eval_runtime"]["forward_batches"]):\n',
      '    if False:\n        if not (exp == act == rows and fwd == ev["eval_runtime"]["forward_batches"]):\n',
      case("d3_capped_summary_refused"), "d3_capped_summary_refused"),
    M("M20b", "an upstream-protocol summary accepted", SEL,
      '"dataset.preprocess_protocol": ev["preprocess_protocol"],', "",
      case("d3_upstream_protocol_refused"), "d3_upstream_protocol_refused"),
    M("M20c", "a record summary's row counts not compared with the rules (one guard of M20a)", SEL,
      '        if not (exp == act == rows and fwd == ev["eval_runtime"]["forward_batches"]):\n',
      "        if not (exp == act == fwd):\n",
      case("d3_capped_rows_alone_refused"), "d3_capped_rows_alone_refused"),
    M("M20d", "the gt_support sum not compared with the rules (the other guard of M20a)", SEL,
      '        if not isinstance(gt, list) or sum(gt) != ev["gt_support_sum"]:\n',
      "        if not isinstance(gt, list):\n",
      case("d3_gt_support_sum_refused"), "d3_gt_support_sum_refused"),
    M("M21", "the key-set comparison dropped", ART,
      "    if missing or extra:\n", "    if False:\n", UNITS, "d4_key_set_missing_key_refused"),
    M("M22a", "the predicate reduced to 'all tensors finite'", QAT,
      "    never = set(never_observed)\n    fails: list[str] = []\n",
      "    return all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), []\n",
      UNITS, "d4_predicate_accepts_an_observed_state"),
    M("M22b", "the predicate reduced to parameters only", QAT,
      '        if kind in ("parameter", "bn_buffer", "fq_scale") and not bool(torch.isfinite(t).all()):\n',
      '        if kind == "parameter" and not bool(torch.isfinite(t).all()):\n',
      UNITS, "d4_predicate_flags_nonfinite_bn_with_finite_params"),
    M("M23", "the AM-21 rejection removed (the rejected pilot run, holding no selection under item 3(d), is then read "
      "as a live run without one)", SEL,
      '        rejected = rej["rejected"]', "        rejected = False",
      case("d5_nonfinite_candidate_rejected_a"), "d5_nonfinite_candidate_rejected_a"),
    M("M24a", "E6's λ/α binding removed at launch", QAT,
      "        binding = e6_parent_binding(parent, args.seed, *sel)\n", "        binding = None\n",
      runner("gates"), "gate_refuses_e6_parent_not_the_selected_run"),
    M("M24b", "E6's λ/α binding removed from check-run-meta", QEE,
      '            Q.e6_parent_binding(meta.get("parent") or {}, args.seed, lam, alpha)\n', "",
      runner("profile"), "profile_stops_on_e6_parent_lambda"),
    # DL-24 (C6): each fix of the verification workflow, with the check that kills its undoing
    MM("X1", "scheduler.step() moved before optimizer.step() (P1)", QAT,
       [("                    optimizer.step()\n                except Exception as e:",
         "                    scheduler.step()\n                    optimizer.step()\n"
         "                except Exception as e:"),
        ("                scheduler.step()\n                loss_v, ce_v, dice_v, gn = ",
         "                loss_v, ce_v, dice_v, gn = ")],
       runner("d1"), "d1_lr_applied_at_each_optimizer_step"),
    M("X2", "check-run-meta's repeat of the clip binding dropped (P10)", QEE,
      "    # the clip and its source (P10)\n    if stage == Q.U4_STAGE and args.seed == Q.U4_SEED:\n",
      "    # the clip and its source (P10)\n    if True:\n        pass\n"
      "    elif stage == Q.U4_STAGE and args.seed == Q.U4_SEED:\n",
      runner("profile"), "profile_pilot_checked_without_u4_pilot_stops", "profile_stops_on_other_clip_selection"),
    M("X3", "train mode not restored after a VAL pass that raised in a non-finite state (P12)", QAT,
      "                prepared.train()                           # validate() returns the model to train mode only "
      "on success\n", "", runner("nan"), "d1_val_error_recorded_and_train_mode_restored"),
    M("X4", "an exception before the run_meta row reported as an aborted run (P37)", QAT,
      "        if not (tel_p.is_file() and tel_p.stat().st_size > 0):", "        if False:",
      runner("gates"), "main_error_before_telemetry_exit_4"),
    M("X6", "epoch_end rows not marked state_nonfinite (P12)", QAT,
      '"nonfinite_since_step": nonfinite_since, "state_nonfinite": nonfinite_since is not None,',
      '"nonfinite_since_step": nonfinite_since,', runner("nan"), "d1_nonfinite_rows_marked_step10"),
    M("X8", "a non-finite run rejected on its epoch_end rows only (AM-21 item 3)", SEL,
      '    flagged = (any(r.get("nonfinite_since_step") is not None for r in rows)\n',
      '    flagged = (any(r.get("nonfinite_since_step") is not None for r in ends.values())\n',
      case("d5_nonfinite_row_only_rejected"), "d5_nonfinite_row_only_rejected"),
    M("X9", "a failed convert worker leaves its eval directory unspent (P26)", QEE,
      "            if not (E / CONVERT_STOP).exists():\n", "            if False:\n",
      records(), "d4_convert_worker_refusal_spends_the_eval_dir"),
    M("X10", "a failed evaluator leaves its eval directory unspent (P26)", QEE,
      "            write_exclusive({E / SCORE_STOP: ", "            (lambda d: None)({E / SCORE_STOP: ",
      records(), "d4_score_failure_spends_the_eval_dir"),
    M("X11", "P16's observer half dropped from the freeze cross-check", ART,
      '            "ok": not any(bn_diff.values()) and not any(obs_diff.values())}',
      '            "ok": not any(bn_diff.values())}',
      records(), "d4_convert_stops_when_an_observer_moved_after_its_freeze"),
    M("X12", "the stored observer and fake-quant flags not checked (P15)", ART,
      '    if flags["observer_enabled"] != want or flags["fake_quant_enabled"] != [1]:\n', "    if False:\n",
      records(), "d4_convert_epoch_stored_flags_stop"),
    M("X13", "the copy's dtype comparison dropped (P15; torch.equal ignores dtype)", ART,
      "live[k].shape == v.shape and live[k].dtype == v.dtype and torch.equal",
      "live[k].shape == v.shape and torch.equal", records(), "d4_convert_epoch_copy_dtype_stop"),
    M("X14", "the checkpoint's identity not checked against the run (P15)", ART,
      "    wrong = {k: payload.get(k) for k, v in ident.items() if payload.get(k) != v}\n", "    wrong = {}\n",
      records(), "d4_convert_epoch_identity_refused"),
    M("X15", "the by-name copy's alias-key check dropped (P15)", ART,
      "    if sorted(unexpected) != aliases:\n", "    if False:\n", UNITS, "d4_alias_mismatch_stops"),
    M("X16", "the run's qconfig fingerprint not compared at conversion (P15)", ART,
      '    if qsum["fingerprint"] != (meta.get("qconfig") or {}).get("fingerprint"):\n', "    if False:\n",
      records(), "d4_convert_epoch_qconfig_changed_stop"),
    M("X17", "select_clip compares the stored selection on the old eight keys only (P27)", SEL,
      "    keys = (set(stored) | set(recomputed)) - set(SELECTION_PATH_KEYS)\n",
      '    keys = {"winner", "values", "tied_epochs", "excluded_epochs", "telemetry_sha256", "run_id",\n'
      '            "summary_fields", "eval_identity"}\n',
      case("d5_edited_selection_trace_refused"), "d5_edited_selection_trace_refused"),
    # Q2-F, F2: a PTQ smoke judges "untouched" by the repository's git status before and after its run, so files
    # tracked at HEAD do not count as a write; each write below is caught by both smokes
    M("M25a", "a calibration-list build writes a new file into configs/calibration", BCL, _LISTS_OUT, _NEW_LIST_FILE,
      (RUN_PTQ, ()), "repository_untouched"),
    M("M25b", "a calibration-list build writes a new file into configs/calibration", BCL, _LISTS_OUT, _NEW_LIST_FILE,
      (CAL_LISTS, ()), "nothing_written_in_the_repository"),
    M("M26a", "a calibration-list build appends to the committed seed-42 list in configs/calibration", BCL,
      _LISTS_OUT, _SEED42_APPENDED, (RUN_PTQ, ()), "repository_untouched"),
    M("M26b", "a calibration-list build appends to the committed seed-42 list in configs/calibration", BCL,
      _LISTS_OUT, _SEED42_APPENDED, (CAL_LISTS, ()), "nothing_written_in_the_repository"),
    # Q2-F, F3: G1, each half of names_test and each name check's place (the launch gates; nothing named "test"
    # is created)
    M("M27", "resolve_parent probes --source-run-dir (is_dir) before its name check", QAT,
      _PARENT_NAME + _PARENT_DIR, _PARENT_DIR + _PARENT_NAME, runner("gates"),
      "g1_given_name_refused_before_any_probe"),
    M("M28", "names_test reads the given path only (G1's resolved half dropped)", QAT,
      '    return "test" in str(p).lower() or "test" in str(p.resolve()).lower()\n',
      '    return "test" in str(p).lower()\n', runner("gates"), "g1_resolved_name_refused_before_is_dir"),
    M("M29", "read_clip_selection probes --clip-selection (is_file) before its name check", QAT,
      _CLIP_NAME + _CLIP_FILE, _CLIP_FILE + _CLIP_NAME, runner("gates"), "g1_clip_selection_name_refused_before_probe"),
    M("M30", "--out-dir checked and listed before its name check (G1's --out-dir order)", QAT,
      _OUT_NAME + _OUT_LOOK, _OUT_LOOK + _OUT_NAME, runner("gates"), "g1_out_dir_name_refused_before_listing"),
    # Q2-F, F1: real 15-epoch runs made non-finite at steps 10, 22 and 27 (before, between and after the two
    # freezes): their record conversion, their epoch selection and the U4 clip selection (AM-21)
    M("M31", "a run that recorded a non-finite state reads as incomplete, so its records are never converted", ART,
      '    if torn:\n        reasons.append("the telemetry\'s last line is torn")\n',
      '    if torn:\n        reasons.append("the telemetry\'s last line is torn")\n'
      '    if any(r.get("state_nonfinite") for r in rows):\n'
      '        reasons.append("the run recorded a non-finite state")\n',
      NF_CONVERT, *(f"nf_step{k}_record_convert_exit_0" for k in NF_STEPS)),
    MM("M32", "P16's freeze cross-check compares floats (torch.equal; NaN != NaN), not the raw bytes", ART,
       [("[k for k in bn_keys if not bytes_equal(states[e][k], states[bn_from][k])]",
         "[k for k in bn_keys if not torch.equal(states[e][k], states[bn_from][k])]"),
        ("[k for k in obs_keys if not bytes_equal(states[e][k], states[obs_from][k])]",
         "[k for k in obs_keys if not torch.equal(states[e][k], states[obs_from][k])]")],
       NF_CONVERT, "nf_step10_record_convert_exit_0", "nf_step22_record_convert_exit_0"),
    M("M33", "select_clip does not reject a non-finite pilot run (AM-21 item 3; it is then read as a live run "
      "without a selection)", SEL,
      '        rejected = rej["rejected"]', "        rejected = False",
      NF_CLIP, *(f"nf_step{k}_select_clip_rejects_nonfinite" for k in NF_STEPS)),
    M("M34", "select_clip goes on when both pilot runs are rejected", SEL,
      '        raise QATRefused("no_winner", NO_WINNER)\n', "        pass\n",
      NF_CLIP, *(f"nf_step{k}_both_nonfinite_no_winner_exit_2" for k in NF_STEPS)),
    M("M35", "select_qat_epoch takes an excluded (non-finite) epoch for a scored one", SEL,
      _CHECKPOINT_DECIDES, "        if False:\n",
      NF_EPOCH, *(f"nf_step{k}_select_qat_epoch_excludes_nonfinite" for k in NF_STEPS)),
    M("M35b", "select_qat_epoch reads the record, not the checkpoint: a non-finite epoch scored highest wins (R3)",
      SEL, _CHECKPOINT_DECIDES, "        if False:\n",
      case("d3_nonfinite_epoch_with_highest_score_still_excluded"),
      "d3_nonfinite_epoch_with_highest_score_still_excluded"),
    M("M36", "select_qat_epoch goes on with no convertible epoch", SEL,
      "    if not scored:\n", "    if False:\n", NF_EPOCH, "nf_no_convertible_epoch_exit_2_nothing_written"),
    # Q2-F PART 2, CHECK ITEM 9 (AM-21 items 1(b), 1(e)): the predicate at every raise, abort only when finite
    M("M37", "a step that raises after an earlier non-finite flag never aborts (the predicate is not read)", QAT,
      "                    if step_ok:\n", "                    if step_ok and nonfinite_since is None:\n",
      runner("nan"), "d1_step_raise_after_repair_aborts"),
    M("M38", "a VAL pass that raises after an earlier non-finite flag never aborts (the predicate is not read)", QAT,
      "                if val_ok:\n", "                if val_ok and nonfinite_since is None:\n",
      runner("nan"), "d1_val_raise_after_repair_aborts"),
    M("M39", "a step that raises while the state is finite does not abort", QAT,
      "                    if step_ok:\n", "                    if False:\n",
      runner("nan"), "d1_step_raise_while_finite_aborts"),
    M("M40", "a VAL pass that raises while the state is finite does not abort", QAT,
      "                if val_ok:\n", "                if False:\n", runner("nan"), "d1_val_raise_while_finite_aborts"),
    # CHECK ITEM 15 (AM-21 item 5): the completion line names the step the state was first found non-finite
    M("M41", "the completion line reads 'non-finite since step N'", QAT,
      "STATE NON-FINITE first found at step ", "STATE NON-FINITE since step ",
      runner("nan"), "d1_result_line_nonfinite_step10", "d1_result_line_nonfinite_step27"),
    M("M42", "a step that raises before the first non-finite flag aborts, whatever the state", QAT,
      "                    if step_ok:\n", "                    if step_ok or nonfinite_since is None:\n",
      runner("nan"), "d1_step_raise_in_unflagged_nonfinite_state_recorded"),
    M("M43", "a raise that first meets a non-finite state does not record its step", QAT,
      "                        _mark_nonfinite(step, step_fails)\n", "                        pass\n",
      runner("nan"), "d1_step_raise_in_unflagged_nonfinite_state_recorded"),
    M("M44", "a step that raises in a non-finite state ends the run (AM-21 item 1(c))", QAT,
      "                    if step_ok:\n", "                    if True:\n",
      runner("nan"), "d1_step_errors_recorded_in_nonfinite_state",
      "nonfinite_loss_recorded_run_reaches_epoch_15_step10"),
    M("M45", "a VAL pass that raises in a non-finite state ends the run (AM-21 item 1(c))", QAT,
      "                if val_ok:\n", "                if True:\n",
      runner("nan"), "d1_val_error_recorded_and_train_mode_restored"),
    M("M46", "the epoch end reads the sticky flag, not the checkpoint's state (AM-21 item 1(b))", QAT,
      "            finite, fails = state_predicate(state, kinds, never)\n",
      "            finite, fails = nonfinite_since is None, []\n",
      runner("nan"), "d1_epoch_end_alone_finds_a_nonfinite_state"),
    # CHECK ITEM 10 (AM-21 item 1(a), DL-85): check-run-meta compares the never-observed list with the list of record
    M("M47", "check-run-meta does not compare never_observed_modules with DL-85's list of record", QEE,
      '            "never_observed_modules": NEVER_OBSERVED_OF_RECORD}\n', "            }\n",
      runner("profile"), "profile_stops_on_never_observed_modules_four",
      "profile_stops_on_never_observed_modules_six", "profile_stops_on_never_observed_modules_one_swapped"),
    M("M47b", "the list of record names the skip_add of features 14 for that of features 13 (DL-85)", QEE,
      "for i in (2, 4, 7, 11, 13))", "for i in (2, 4, 7, 11, 14))",
      runner("profile"), "profile_e5_s42_pilot_passes"),
    # CHECK ITEMS 11, 13, 14 and 15, ruling 3 (AM-21 items 2(a), 2(b), 3(a), 3(d)): the epoch selection reads the
    # checkpoint, the rejection reads the telemetry and the checkpoints, P16 and P15 read checkpoints that hold NaN
    M("M48", "the not-convertible rule cites AM-19 item 3(a) again, not AM-21 item 2(a) (CHECK ITEM 15)", ART,
      'NOT_CONVERTIBLE_RULE = "excluded: non-finite state (AM-21 item 2(a))"\n',
      'NOT_CONVERTIBLE_RULE = "excluded: non-finite state (AM-19 item 3(a))"\n',
      case("d3_nonfinite_epochs_excluded"), "d3_nonfinite_epochs_excluded"),
    M("M49", "a rejected U4 pilot run gets an epoch selection (AM-21 item 3(d))", SEL,
      '    if meta.get("u4_pilot") is True:\n', "    if False:\n",
      case("d3_rejected_pilot_run_needs_no_selection", "d3_rejected_pilot_without_convertible_epoch_is_not_no_model"),
      "d3_rejected_pilot_run_needs_no_selection", "d3_rejected_pilot_without_convertible_epoch_is_not_no_model"),
    MM("M50", "the no-convertible-epoch refusal drops item 2(b)'s entry, non-finite: no model (CHECK ITEM 13)", SEL,
       [('f"{NO_MODEL_ENTRY}: every epoch of {rec[\'run_id\']} is excluded "',
         'f"every epoch of {rec[\'run_id\']} is excluded "'),
        ('f"today as \\"{NO_MODEL_ENTRY}\\"")', '"today")')],
       case("d3_no_convertible_epoch_refused_exit_2"), "d3_no_convertible_epoch_refused_exit_2"),
    M("M51", "a train row that logs a non-finite loss or pre-clip gradient norm does not reject the run (item 3(a))",
      SEL, '                    and {"loss", "grad_norm"} & set(r.get("nonfinite") or {}))\n',
      '                    and {"loss", "grad_norm"} & set(r.get("nonfinite") or {}) and False)\n',
      case("d5_rejection_reads_a_nonfinite_loss_row", "d5_rejection_reads_a_nonfinite_grad_norm_row",
           "d5_logged_nonfinite_loss_candidate_loses", "d3_rejected_pilot_run_by_a_logged_nonfinite_loss"),
      "d5_rejection_reads_a_nonfinite_loss_row", "d5_rejection_reads_a_nonfinite_grad_norm_row",
      "d5_logged_nonfinite_loss_candidate_loses", "d3_rejected_pilot_run_by_a_logged_nonfinite_loss"),
    M("M52", "a checkpoint that fails item 1(a) does not reject the run (item 3(a))", SEL,
      "    failing = sorted(e for e, (finite, *_rest) in checks.items() if not finite)\n", "    failing = []\n",
      case("d5_rejection_reads_a_failing_checkpoint_and_reports_the_missing_flag",
           "d5_unflagged_failing_checkpoint_candidate_loses"),
      "d5_rejection_reads_a_failing_checkpoint_and_reports_the_missing_flag",
      "d5_unflagged_failing_checkpoint_candidate_loses"),
    M("M52b", "epoch_selection passes run_rejected no checkpoint states: a failing checkpoint does not reject a pilot",
      SEL, "        rej = run_rejected(rec, checks=checks)\n", "        rej = run_rejected(rec, checks={})\n",
      case("d3_rejected_pilot_run_reports_the_missing_flag"), "d3_rejected_pilot_run_reports_the_missing_flag"),
    M("M52c", "clip_selection passes run_rejected no checkpoint states: a failing checkpoint does not reject a pilot",
      SEL, "rej = run_rejected(rec) ", "rej = run_rejected(rec, checks={}) ",
      case("d5_unflagged_failing_checkpoint_candidate_loses"), "d5_unflagged_failing_checkpoint_candidate_loses"),
    M("M53", "a failing checkpoint with no state flag is not reported as a deviation (item 3(a))", SEL,
      '    unflagged = [e for e in failing if (ends.get(e) or {}).get("state_finite") is not False]\n',
      "    unflagged = []\n",
      case("d5_rejection_reads_a_failing_checkpoint_and_reports_the_missing_flag",
           "d3_rejected_pilot_run_reports_the_missing_flag"),
      "d5_rejection_reads_a_failing_checkpoint_and_reports_the_missing_flag",
      "d3_rejected_pilot_run_reports_the_missing_flag"),
    M("M54", "P16's observer half compares floats (torch.equal; NaN != NaN): NaN observer buffers fail it (CHECK "
      "ITEM 14)", ART,
      "[k for k in obs_keys if not bytes_equal(states[e][k], states[obs_from][k])]",
      "[k for k in obs_keys if not torch.equal(states[e][k], states[obs_from][k])]",
      case("d3_freeze_cross_check_passes_on_nan_checkpoints"), "d3_freeze_cross_check_passes_on_nan_checkpoints"),
    M("M55", "P16 compares floats with NaN replaced: a frozen buffer whose NaN changed passes (CHECK ITEM 14)", ART,
      "    return torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8),\n"
      "                       b.detach().reshape(-1).contiguous().view(torch.uint8))\n",
      "    return torch.equal(torch.nan_to_num(a.detach()), torch.nan_to_num(b.detach()))\n",
      case("d3_freeze_cross_check_catches_a_changed_nan_buffer"), "d3_freeze_cross_check_catches_a_changed_nan_buffer"),
    M("M56", "P15 wants the observers off from e12, not from e13 (CHECK ITEM 14)", ART,
      "    want = [1] if epoch <= Q.OBS_FREEZE_EPOCH else [0]\n",
      "    want = [1] if epoch < Q.OBS_FREEZE_EPOCH else [0]\n",
      case("d3_stored_flags_read_on_nan_checkpoints"), "d3_stored_flags_read_on_nan_checkpoints"),
    M("M57", "P25 expects score directories only for the epochs selected from: an epoch excluded despite its record "
      "refuses the selection (ruling 3)", SEL,
      '    expected = [f"e{e:02d}" for e in sorted(rows) if rows[e].get("status") == "scored"]\n',
      '    expected = [f"e{e:02d}" for e in sorted(scored)]\n',
      case("d3_nonfinite_epoch_with_highest_score_still_excluded"),
      "d3_nonfinite_epoch_with_highest_score_still_excluded"),
    # DL-24 wf_b44aeaa6-a03 (C7-1, C7-2, F14-1, F14-2): the deviation reported, the not-convertible record field by
    # field, P16's BN half on NaN, P15's fake-quant flag
    M("M58", "the rejected pilot run's refusal drops the trainer's deviation (item 3(a))", SEL,
      "f\"{'; '.join(rej['grounds'] + rej['deviations'])}): it needs no \"",
      "f\"{'; '.join(rej['grounds'])}): it needs no \"",
      case("d3_rejected_pilot_run_reports_the_missing_flag"), "d3_rejected_pilot_run_reports_the_missing_flag"),
    M("M59", "the not-convertible record's own rule is not read", SEL,
      '                    ("rule of the record", ncd.get("rule") == A.NOT_CONVERTIBLE_RULE),\n',
      '                    ("rule of the record", True),\n',
      case("d3_excluded_epoch_record_of_am19_refused"), "d3_excluded_epoch_record_of_am19_refused"),
    M("M60", "the excluded row's rule is not read", SEL,
      '                    ("rule of the row", row.get("rule") == A.NOT_CONVERTIBLE_RULE),\n',
      '                    ("rule of the row", True),\n',
      case("d3_excluded_epoch_row_of_am19_refused"), "d3_excluded_epoch_row_of_am19_refused"),
    M("M61", "the not-convertible record's status is not read", SEL,
      '                    ("status", ncd.get("status") == "not convertible"),\n',
      '                    ("status", True),\n',
      case("d3_excluded_epoch_record_status_refused"), "d3_excluded_epoch_record_status_refused"),
    M("M62", "the not-convertible record's checkpoint is not compared", SEL,
      '                    ("checkpoint", ncd.get("qat_checkpoint_sha256") == ck_sha),\n',
      '                    ("checkpoint", True),\n',
      case("d3_excluded_epoch_record_differs_refused"), "d3_excluded_epoch_record_differs_refused"),
    M("M63", "the not-convertible record's epoch is not compared", SEL,
      '                    ("epoch", ncd.get("epoch") == e),\n', '                    ("epoch", True),\n',
      case("d3_excluded_epoch_record_of_another_epoch_refused"), "d3_excluded_epoch_record_of_another_epoch_refused"),
    M("M64", "the not-convertible record's run is not compared", SEL,
      '                    ("run", (ncd.get("run_id"), ncd.get("telemetry_sha256")) == run_ident),\n',
      '                    ("run", True),\n',
      case("d3_excluded_epoch_record_of_another_run_refused"), "d3_excluded_epoch_record_of_another_run_refused"),
    M("M65", "the not-convertible record's commit, host, CPU and purpose are not compared", SEL,
      "                    *((k, ncd.get(k) == eval_identity[k]) for k in EVAL_IDENTITY_KEYS)) if not same]\n",
      "                    ) if not same]\n",
      case("d3_excluded_epoch_record_of_another_commit_refused"),
      "d3_excluded_epoch_record_of_another_commit_refused"),
    M("M66", "P16's BN half compares floats (torch.equal; NaN != NaN): a frozen NaN statistic fails it", ART,
      "[k for k in bn_keys if not bytes_equal(states[e][k], states[bn_from][k])]",
      "[k for k in bn_keys if not torch.equal(states[e][k], states[bn_from][k])]",
      case("d3_freeze_cross_check_passes_on_bn_nan"), "d3_freeze_cross_check_passes_on_bn_nan"),
    M("M67", "P16's BN half compares floats with NaN replaced: a changed NaN statistic passes", ART,
      "[k for k in bn_keys if not bytes_equal(states[e][k], states[bn_from][k])]",
      "[k for k in bn_keys if not torch.equal(torch.nan_to_num(states[e][k]), "
      "torch.nan_to_num(states[bn_from][k]))]",
      case("d3_freeze_cross_check_catches_a_changed_bn_nan"), "d3_freeze_cross_check_catches_a_changed_bn_nan"),
    M("M68", "P15 does not read the stored fake-quant flags", ART,
      '    if flags["observer_enabled"] != want or flags["fake_quant_enabled"] != [1]:\n',
      '    if flags["observer_enabled"] != want:\n',
      case("d3_stored_flags_read_on_nan_checkpoints"), "d3_stored_flags_read_on_nan_checkpoints"),
    # CHECK ITEM 12 and ruling 1 (AM-21 item 3(d)): the telemetry sha256 per candidate; a rejected run's conversion
    # and scoring reported (present, missing, stopped or failed), never required
    M("M69", "a rejected candidate's missing record is required again (SEL-2's anchor)", SEL,
      '    if not path.is_file():\n        return "missing", f"{path.name} is missing", None\n',
      '    if not path.is_file():\n        raise QATIncomplete("eval_record_missing", f"{path} is missing")\n',
      case("d5_rejected_candidate_without_records_other_wins"), "d5_rejected_candidate_without_records_other_wins"),
    M("M69b", "a rejected candidate's missing record is required again (SEL-2's anchor), end to end", SEL,
      '    if not path.is_file():\n        return "missing", f"{path.name} is missing", None\n',
      '    if not path.is_file():\n        raise QATIncomplete("eval_record_missing", f"{path} is missing")\n',
      NF_CLIP, "nf_rejected_run_without_conversion_record_other_wins"),
    M("M70", "a STOP file of a rejected run's conversion or scoring is not reported as stopped", SEL,
      '    if stops:\n        return "stopped"', '    if False:\n        return "stopped"',
      case("d5_rejected_candidate_stopped_conversion_reported",
           "d5_rejected_candidate_stopped_epoch_conversion_reported", "d5_rejected_candidate_stopped_scoring_reported"),
      "d5_rejected_candidate_stopped_conversion_reported", "d5_rejected_candidate_stopped_epoch_conversion_reported",
      "d5_rejected_candidate_stopped_scoring_reported"),
    M("M71", "another run's record reads as the rejected run's own (present)", SEL,
      '    if doc.get("telemetry_sha256") != rec["telemetry_sha256"] or doc.get("run_id") != rec["run_id"]:\n'
      '        return "failed", (f"{path.name} names run',
      '    if False:\n        return "failed", (f"{path.name} names run',
      case("d5_rejected_candidate_record_of_another_run_reported_failed"),
      "d5_rejected_candidate_record_of_another_run_reported_failed"),
    M("M72", "a timing record reads as a record of purpose record", SEL,
      '    if doc.get("purpose") != "record":\n        return "failed"', '    if False:\n        return "failed"',
      case("d5_rejected_candidate_timing_record_reported_failed"),
      "d5_rejected_candidate_timing_record_reported_failed"),
    M("M73", "a rejected run's scores are not reported", SEL,
      '                scores = {f"e{e:02d}": rows[e]["all_class_miou"] for e in scored}\n',
      '                scores = {}\n',
      case("d5_rejected_candidate_scores_reported"), "d5_rejected_candidate_scores_reported"),
    M("M74", "select_clip prints no REPORT line", SCL,
      '        print(f"REPORT: {line}")\n', "        pass\n",
      case("d5_rejected_candidate_without_records_other_wins"), "d5_rejected_candidate_without_records_other_wins"),
    M("M75", "the telemetry sha256s are paired with the candidates in reverse order", SEL,
      "    for (run_dir, eval_dir), want_sha in zip(candidates, expect):\n",
      "    for (run_dir, eval_dir), want_sha in zip(candidates, expect[::-1]):\n",
      case("d5_telemetry_sha256s_swapped_refused"), "d5_telemetry_sha256s_swapped_refused"),
    M("M76", "a rejected candidate's REPORT line drops the trainer's deviation (item 3(a))", SEL,
      '    parts += [f"deviation: {d}" for d in rej["deviations"]]\n', "    parts += []\n",
      case("d5_rejected_candidate_deviation_reported"), "d5_rejected_candidate_deviation_reported"),
    M("M77", "no REPORT line reaches select_clip (and none is printed when both runs are rejected)", SEL,
      "    if report is not None:\n        report.extend(lines)\n", "    if False:\n        report.extend(lines)\n",
      case("d5_both_rejected_records_reported"), "d5_both_rejected_records_reported"),
    # DL-24 wf_def5109c-338 (C8-1, C8-2, C8-3, C8-SC-1, C8-SC-3): a malformed record of the rejected run reported
    # failed, never a crash; every failed branch; the first-found step; REPORT before RESULT; a surplus sha256
    M("M78", "the rejected run's records are parsed loosely: a NaN score crashes select_clip (exit 4)", SEL,
      '        doc = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)\n',
      '        doc = json.loads(path.read_text(encoding="utf-8"))\n',
      case("d5_rejected_candidate_nan_score_reported_failed"), "d5_rejected_candidate_nan_score_reported_failed"),
    M("M79", "a record whose outcomes are not a list crashes select_clip", SEL,
      "(items if isinstance(items, list) else [])", "(items or [])",
      case("d5_rejected_candidate_malformed_outcomes_reported_failed"),
      "d5_rejected_candidate_malformed_outcomes_reported_failed"),
    M("M80", "an unreadable record of the rejected run stops select_clip", SEL,
      '        return "failed", f"{path.name} is unreadable ({e})", None\n', "        raise\n",
      case("d5_rejected_candidate_unreadable_record_reported_failed"),
      "d5_rejected_candidate_unreadable_record_reported_failed"),
    M("M81", "a record that is not a JSON object crashes select_clip", SEL,
      "    if not isinstance(doc, dict):\n", "    if False:\n",
      case("d5_rejected_candidate_non_object_record_reported_failed"),
      "d5_rejected_candidate_non_object_record_reported_failed"),
    M("M82", "a smoke-input record of the rejected run reads as present", SEL,
      '    if bool(doc.get("smoke_inputs")) and not smoke:\n        return "failed"',
      '    if False:\n        return "failed"',
      case("d5_rejected_candidate_smoke_record_reported_failed"), "d5_rejected_candidate_smoke_record_reported_failed"),
    M("M83", "a conversion record that does not cover e01-e15 reads as present", SEL,
      '        if sorted(outcomes) != list(range(1, Q.EPOCHS + 1)):\n            c_state, c_detail = "failed"',
      '        if False:\n            c_state, c_detail = "failed"',
      case("d5_rejected_candidate_partial_conversion_reported_failed"),
      "d5_rejected_candidate_partial_conversion_reported_failed"),
    M("M84", "a score record that does not cover e01-e15 reads as present", SEL,
      '        if sorted(rows) != list(range(1, Q.EPOCHS + 1)):\n            s_state, s_detail = "failed"',
      '        if False:\n            s_state, s_detail = "failed"',
      case("d5_rejected_candidate_partial_scoring_reported_failed"),
      "d5_rejected_candidate_partial_scoring_reported_failed"),
    M("M85", "a scored epoch without a finite score is reported as a score", SEL,
      '            if unfit:\n                s_state, s_detail = "failed"',
      '            if False:\n                s_state, s_detail = "failed"',
      case("d5_rejected_candidate_text_score_reported_failed"), "d5_rejected_candidate_text_score_reported_failed"),
    M("M86", "the rejected run's conversion detail drops its not-convertible epochs", SEL,
      '            bad = [e for e in sorted(outcomes) if outcomes[e].get("status") != "converted"]\n',
      "            bad = []\n",
      case("d5_both_rejected_records_reported"), "d5_both_rejected_records_reported"),
    M("M87", "the rejection's ground drops the step at which the state was first found non-finite", SEL,
      '                       + (f", first found at step {first}" if first is not None else ""))\n',
      '                       + "")\n',
      case("d5_rejected_candidate_without_records_other_wins"), "d5_rejected_candidate_without_records_other_wins"),
    M("M88", "select_clip prints the REPORT lines after the RESULT line of a selection", SCL,
      '    _print_report(report)\n    print(f"RESULT: CLIP SELECTED {w[\'clip_norm\']} (tie '
      '{str(sel[\'tie\']).lower()}{tail})")\n',
      '    print(f"RESULT: CLIP SELECTED {w[\'clip_norm\']} (tie {str(sel[\'tie\']).lower()}{tail})")\n'
      '    _print_report(report)\n',
      case("d5_rejected_candidate_without_records_other_wins"), "d5_rejected_candidate_without_records_other_wins"),
    M("M89", "select_clip prints the REPORT lines after the RESULT line of a refusal", SCL,
      '        _print_report(report)\n        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")\n',
      '        print(f"RESULT: REFUSED [{e.code}] -- {e}. Nothing was written.")\n        _print_report(report)\n',
      case("d5_both_rejected_records_reported"), "d5_both_rejected_records_reported"),
    M("M90", "a surplus --expect-telemetry-sha256 is accepted", SEL,
      "    if len(expect) != len(candidates):\n", "    if len(expect) < len(candidates):\n",
      case("d5_expect_telemetry_sha256_required_three"), "d5_expect_telemetry_sha256_required_three"),
    # ruling 2: a job's whole process group ends at its timeout and at its end, and a stop signal to the harness
    # ends every live job's group, skips the queued jobs, starts no smoke after it and ends the run FAIL
    M("M91", "a timed-out job is killed alone, not with its process group", SQM,
      "        os.killpg(pgid, signal.SIGKILL)\n", "        os.kill(pgid, signal.SIGKILL)\n",
      SELF, "timeout_kills_the_job_process_group"),
    M("M92", "a job runs in the harness's own session, so the group kill cannot reach its children",
      SQM, "                             start_new_session=True)\n",
      "                             start_new_session=False)\n", SELF, "timeout_kills_the_job_process_group"),
    M("M93", "a signal to the harness leaves the live jobs running", SQM,
      '        for pgid in sorted(JOBS["groups"]):\n            _kill_group(pgid)\n',
      '        for pgid in sorted(JOBS["groups"]):\n            pass\n',
      SELF, "sigterm_kills_every_live_job_process_group"),
    M("M94", "a job starts after a stop signal", SQM,
      '        if JOBS["stopped"] is not None:\n            return f', '        if False:\n            return f',
      SELF, "no_job_starts_after_a_signal"),
    M("M95", "the queued jobs run after a stop signal", SQM,
      '        if JOBS["stopped"] is not None:                       # signalled: the queued jobs are skipped\n',
      '        if False:                       # signalled: the queued jobs are skipped\n',
      SELF, "sigterm_skips_the_queued_jobs"),
    M("M96", "a SIGHUP to the harness is not handled", SQM,
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)\n",
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)\n", SELF,
      "sighup_kills_every_live_job_process_group"),
    M("M97", "a job that ends leaves its stragglers running", SQM,
      "            _kill_group(p.pid)                             # stragglers of a job that ended\n",
      "            pass                                           # stragglers of a job that ended\n",
      SELF, "a_finished_job_leaves_no_straggler"),
    M("M98", "a timed-out job's group is killed only after its output has been waited for", SQM,
      "            timed_out = True\n            _kill_group(p.pid)\n", "            timed_out = True\n",
      SELF, "timeout_kills_the_job_process_group"),
    M("M99", "a stop signal kills only the first live job's group", SQM,
      '        for pgid in sorted(JOBS["groups"]):\n', '        for pgid in sorted(JOBS["groups"])[:1]:\n',
      SELF, "sigterm_kills_every_live_job_process_group"),
    M("M100", "a signalled run does not end FAIL", SQM,
      '    check("harness_ran_unsignalled", stopped is None,\n', '    check("harness_ran_unsignalled", True,\n',
      SELF, "sigterm_run_ends_fail"),
    M("M101", "a job starting when a stop signal arrives runs on", SQM,
      "            _kill_group(p.pid)\n    timed_out = False\n", "            pass\n    timed_out = False\n",
      SELF, "a_job_starting_when_the_signal_arrives_is_killed"),
    M("M102", "a SIGQUIT to the harness is not handled", SQM,
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)\n",
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)\n", SELF,
      "sigquit_kills_every_live_job_process_group"),
    M("M103", "a SIGINT to the harness is not handled", SQM,
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)\n",
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)\n", SELF,
      "sigint_kills_every_live_job_process_group"),
    M("M104", "a SIGTERM to the harness is not handled", SQM,
      "STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)\n",
      "STOP_SIGNALS = (signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)\n", SELF,
      "sigterm_kills_every_live_job_process_group"),
    M("M105", "the self-check's jobs ignore the harness they watch, so a SIGKILLed self-check leaves them running",
      SQM, "< 120 and not gone(watch):\\n", "< 120:\\n", SELF, "a_sleeper_ends_with_the_process_it_watches"),
    M("M106", "the self-check's jobs outlive a watched process that is already reaped (no /proc entry)", SQM,
      '"        return True\\n"', '"        return False\\n"', SELF,
      "a_sleeper_ends_when_the_process_it_watches_is_reaped"),
    M("M107", "main() installs no stop handler, so a signal ends the harness and orphans its live jobs", SQM,
      "    install_stop_handlers()                          # first: the jobs run in sessions of their own\n",
      "    pass\n", SELF, "the_stop_handlers_are_installed", "sigterm_kills_every_live_job_process_group",
      "a_job_starting_when_the_signal_arrives_is_killed"),
    M("M108", "run_smoke runs the smoke without run_job (the pre-C9 subprocess.run, which kills the smoke alone)",
      SQM, "    out, timed_out, _pid = run_job(argv, cwd=str(shadow), env=env, timeout=timeout)\n",
      _SMOKE_RUN_PRE_C9, SELF, "timeout_kills_the_job_process_group"),
    M("M109", "main() runs no self-check, so --self-check would pass with nothing measured", SQM,
      "        self_check()\n", "        pass\n", SELF, "self_check_measured"),
    M("M110", "an --only id that names no job is ignored", SQM,
      '        if unknown:\n            raise SystemExit(f"RESULT: ERROR unknown job ids {unknown}")\n',
      '        if False:\n            raise SystemExit(f"RESULT: ERROR unknown job ids {unknown}")\n',
      SELF, "an_unknown_job_id_is_refused"),
    M("M111", "a selection with no job runs nothing and passes", SQM,
      '    if not jobs:\n        raise SystemExit("RESULT: ERROR no edit selected")\n',
      '    if False:\n        raise SystemExit("RESULT: ERROR no edit selected")\n', SELF,
      "an_empty_selection_is_refused"),
    M("M112", "a baseline that a signal skipped crashes the verdicts (KeyError): the run ends with no RESULT line",
      SQM, '        seen = base.get(j["run"], {})                                  # a skipped baseline passes none\n',
      '        seen = base[j["run"]]\n', SELF,
      "sigterm_run_ends_fail"),
    M("M113", "the pooled baselines run on a plain thread pool: a signal does not skip the queued ones", SQM,
      "    run_pool([s for s in specs if s not in warm], shadows, lambda spec, sh: baseline(sh, spec))\n",
      "    with ThreadPoolExecutor(max_workers=len(shadows)) as pool:\n"
      "        list(pool.map(lambda spec: baseline(shadows[0], spec), [s for s in specs if s not in warm]))\n",
      SELF, "sigterm_skips_the_queued_jobs"),
    M("M114", "the edits run on a plain thread pool: a signal does not skip the queued ones", SQM,
      "    run_pool(todo, shadows, work)\n",
      "    with ThreadPoolExecutor(max_workers=len(shadows)) as pool:\n"
      "        list(pool.map(work, todo, [shadows[i % len(shadows)] for i in range(len(todo))]))\n",
      SELF, "sigterm_skips_the_queued_jobs"),
    M("M115", "the jobs lock is not reentrant: a signal while this thread starts a job deadlocks the harness", SQM,
      '"lock": threading.RLock()}\n', '"lock": threading.Lock()}\n', SELF,
      "a_job_starting_when_the_signal_arrives_is_killed"),
    MM("M116", "the stop handlers are installed only in the signal job, not by main(), so the harness itself has none",
       SQM, [("    install_stop_handlers()                          # first: the jobs run in sessions of their own\n",
              "    pass\n"), (_SETRLIMIT, "    install_stop_handlers()\n" + _SETRLIMIT)],
       SELF, "the_stop_handlers_are_installed"),
    M("M117", "main() filters --only itself (the pre-C9 filter): an id that names no job runs nothing and passes", SQM,
      "        jobs = select_jobs(all_jobs, args.only)\n",
      "        jobs = [j for j in all_jobs if not args.only or j[\"id\"] in args.only.split(\",\")]\n", SELF,
      "main_refuses_an_unknown_id"),
    M("M118", "main() leaves its root behind when it refuses a selection", SQM,
      "    except SystemExit:\n        if not args.keep_shadow:\n            shutil.rmtree(root, ignore_errors=True)\n"
      "        raise\n", "    except SystemExit:\n        raise\n", SELF, "main_refuses_an_unknown_id"),
    M("M119", "main() runs none of the jobs it selected", SQM,
      "    run_jobs(jobs, shadows, root / \"cache\")\n", "    pass\n", SELF, "main_runs_every_selected_job"),
    M("M120", "a job without a verdict (one a signal stopped) passes unnoticed", SQM,
      '    check("every_job_has_a_verdict", not missing,\n', '    check("every_job_has_a_verdict", True,\n', SELF,
      "sigterm_run_ends_fail"),
    M("M121", "the self-check's jobs watch the harness that started this one, not this one", SQM,
      "    me = str(os.getpid())\n", "    me = str(os.getppid())\n", SELF, "the_jobs_watch_this_harness"),
    M("M122", "a timed-out job's output is waited for without a bound once its group is killed", SQM,
      '                out, _err = p.communicate(timeout=DRAIN["value"])\n',
      '                out, _err = p.communicate()\n', SELF, "a_job_whose_child_leaves_its_group_is_bounded"),
    M("M123", "the output so far is dropped when a process outside the group holds the pipe", SQM,
      "                out = _text(e.stdout)\n", "                out = \"\"\n", SELF,
      "a_job_whose_child_leaves_its_group_is_bounded"),
]

# ------------------------------------------------------------------ the refusal table (P36)
# file -> the functions whose refusals P8-P10, O2, P14 and P23-P28 cover
SCOPE = {
    QAT: {"resolve_parent", "read_clip_selection", "clip_binding", "read_tracked_selection", "e6_parent_binding",
          "real_run_gates", "load_source", "run_qat"},
    SEL: None,                                                   # every function
    QEE: {"_epochs", "cmd_convert", "cmd_convert_epoch", "cmd_score", "_read_json", "_pilot_clip_winner",
          "cmd_finalize", "cmd_check_run_meta"},
    # the run's record and P24's first link, read by both selectors, convert and finalize
    ART: {"refuse_test_path", "read_run_record", "require_complete", "verified_checkpoint", "read_ts_identity"},
    SQE: {"main"},
    SCL: {"main"},
}
GATES, PROFILE = runner("gates"), runner("profile")
RECORDS = records()


def K(rel: str, func: str, code: str, run: tuple, *checks: str, k: int = 0) -> tuple:
    return (rel, func, code, k), (run, checks)


KILLS = dict([
    # P9: the parent run (AM-19 item 4(a))
    K(QAT, "resolve_parent", "parent_dir_test_path", GATES, "refuses_parent_dir_test_path"),
    K(QAT, "resolve_parent", "parent_dir_missing", GATES, "refuses_parent_dir_missing"),
    K(QAT, "resolve_parent", "expect_source_sha256_format", GATES, "refuses_expect_source_sha256_format"),
    K(QAT, "resolve_parent", "parent_best_json_missing", GATES, "refuses_parent_best_json_missing"),
    K(QAT, "resolve_parent", "parent_best_json_format", GATES, "refuses_parent_best_json_not_json"),
    K(QAT, "resolve_parent", "parent_best_json_format", GATES, "refuses_parent_best_json_format", k=1),
    K(QAT, "resolve_parent", "parent_checkpoint_test_path", GATES, "refuses_parent_checkpoint_test_name"),
    K(QAT, "resolve_parent", "parent_checkpoint_missing", GATES, "refuses_parent_checkpoint_missing"),
    K(QAT, "resolve_parent", "parent_sha256_mismatch", GATES, "refuses_parent_sha256_mismatch"),
    K(QAT, "resolve_parent", "parent_records_missing", GATES, "refuses_parent_records_missing",
      "refuses_e1_layout_for_e6"),
    K(QAT, "resolve_parent", "parent_run_meta_rows", GATES, "refuses_parent_run_meta_rows"),
    K(QAT, "resolve_parent", "parent_run_meta_key", GATES, "refuses_parent_run_meta_without_mode"),
    K(QAT, "resolve_parent", "parent_mode_not_real", GATES, "refuses_parent_mode_not_real"),
    K(QAT, "resolve_parent", "parent_seed_mismatch", GATES, "refuses_parent_seed_mismatch"),
    K(QAT, "resolve_parent", "parent_stage_mismatch", GATES, "refuses_parent_stage_mismatch"),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e3_abort"),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e3_no_run_end", k=1),
    K(QAT, "resolve_parent", "parent_stage_mismatch", GATES, "refuses_e1_parent_of_another_stage", k=1),
    K(QAT, "resolve_parent", "parent_run_meta_key", GATES, "refuses_parent_run_meta_key", k=1),
    K(QAT, "resolve_parent", "parent_incomplete", GATES, "refuses_parent_incomplete_e1", k=2),
    # P10: the clip and its source
    K(QAT, "read_clip_selection", "clip_selection_sha256_format", GATES, "refuses_clip_selection_sha_format"),
    K(QAT, "read_clip_selection", "clip_selection_test_path", GATES, "refuses_clip_selection_test_path"),
    K(QAT, "read_clip_selection", "clip_selection_missing", GATES, "refuses_clip_selection_path_absent"),
    K(QAT, "read_clip_selection", "clip_selection_missing", GATES, "refuses_clip_selection_file_missing", k=1),
    K(QAT, "read_clip_selection", "clip_selection_sha256_mismatch", GATES, "refuses_clip_selection_sha_mismatch"),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_not_json"),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_format", k=1),
    K(QAT, "read_clip_selection", "clip_selection_format", GATES, "refuses_clip_selection_winner_not_a_candidate",
      k=2),
    K(QAT, "clip_binding", "grad_clip_norm_invalid", GATES, "refuses_clip_absent"),
    K(QAT, "clip_binding", "grad_clip_norm_not_candidate", GATES, "refuses_clip_not_candidate"),
    K(QAT, "clip_binding", "u4_pilot_not_e5_s42", GATES, "refuses_u4_pilot_outside_e5_s42"),
    K(QAT, "clip_binding", "u4_pilot_with_clip_selection", GATES, "refuses_u4_pilot_with_selection"),
    K(QAT, "clip_binding", "u4_pilot_required", GATES, "refuses_e5_s42_without_u4_pilot"),
    K(QAT, "clip_binding", "clip_selection_required", GATES, "refuses_missing_clip_selection"),
    K(QAT, "clip_binding", "clip_selection_winner_mismatch", GATES, "refuses_clip_not_the_winner"),
    # O2: E6's λ and α selections and the parent they name
    K(QAT, "read_tracked_selection", "f'{what}_selection_missing'", GATES, "refuses_lambda_selection_missing",
      "refuses_alpha_selection_missing"),
    K(QAT, "read_tracked_selection", "selection_sha256_format", GATES, "refuses_selection_sha_format"),
    K(QAT, "read_tracked_selection", "selection_path_not_repo_relative", GATES, "refuses_selection_absolute_path"),
    K(QAT, "read_tracked_selection", "selection_test_path", GATES, "refuses_selection_test_path"),
    K(QAT, "read_tracked_selection", "selection_missing_file", GATES, "refuses_selection_missing_file"),
    K(QAT, "read_tracked_selection", "selection_not_tracked", GATES, "refuses_selection_untracked"),
    K(QAT, "read_tracked_selection", "selection_changed_since_head", GATES, "refuses_selection_changed_since_head"),
    K(QAT, "read_tracked_selection", "selection_sha256_mismatch", GATES, "refuses_selection_sha_mismatch"),
    K(QAT, "read_tracked_selection", "selection_format", GATES, "refuses_selection_not_json"),
    K(QAT, "read_tracked_selection", "selection_format", GATES, "refuses_selection_format",
      "refuses_alpha_file_as_lambda", k=1),
    K(QAT, "e6_parent_binding", "parent_stage_mismatch", GATES, "refuses_e6_parent_not_e3"),
    K(QAT, "e6_parent_binding", "parent_seed_mismatch", GATES, "refuses_e6_parent_seed"),
    K(QAT, "e6_parent_binding", "parent_lambda_mismatch", GATES, "refuses_e6_parent_lambda",
      "gate_refuses_e6_parent_not_the_selected_run"),
    K(QAT, "e6_parent_binding", "parent_alpha_mismatch", GATES, "refuses_e6_parent_alpha"),
    K(QAT, "e6_parent_binding", "parent_run_id_mismatch", GATES, "refuses_e6_s42_parent_not_alpha_winner"),
    # P8: the launch gates, in order
    K(QAT, "real_run_gates", "real_run_flag", GATES, "gate_refuses_without_real_run"),
    K(QAT, "real_run_gates", "confirm_real_run_flag", GATES, "gate_refuses_without_confirm"),
    K(QAT, "real_run_gates", "config_pins", GATES, "gate_refuses_config_pins"),
    K(QAT, "real_run_gates", "expect_head_format", GATES, "gate_refuses_head_format"),
    K(QAT, "real_run_gates", "seed", GATES, "gate_refuses_seed"),
    K(QAT, "real_run_gates", "num_workers", GATES, "gate_refuses_num_workers_0"),
    K(QAT, "real_run_gates", "selection_not_applicable", GATES, "gate_refuses_selection_flags_for_e5"),
    K(QAT, "real_run_gates", "out_dir", GATES, "gate_refuses_out_dir_in_repo", "gate_refuses_out_dir_not_empty",
      "gate_refuses_out_dir_test_path"),
    K(QAT, "real_run_gates", "cuda_required", GATES, "gate_refuses_cpu"),
    K(QAT, "real_run_gates", "cuda_initialized_before_seed", GATES, "gate_refuses_cuda_initialised"),
    K(QAT, "real_run_gates", "tf32_not_default", GATES, "gate_refuses_tf32_not_default"),
    K(QAT, "real_run_gates", "backend_unavailable", GATES, "gate_refuses_backend_unavailable"),
    K(QAT, "real_run_gates", "data_root_not_trainval_only", GATES, "gate_refuses_data_root_not_trainval_only"),
    K(QAT, "real_run_gates", "expect_head_mismatch", GATES, "gate_refuses_head_mismatch"),
    K(QAT, "real_run_gates", "code_not_clean", GATES, "gate_refuses_code_not_clean"),
    K(QAT, "load_source", "e.code", GATES, "load_source_refuses_projection_in_e6_student"),
    K(QAT, "load_source", "source_changed", GATES, "load_source_refuses_changed_source"),
    K(QAT, "load_source", "source_projection_keys", GATES, "load_source_refuses_projection_keys"),
    # P8 inside run_qat, and its two STOPs (P2's optimizer of record, P5's VAL bracket)
    K(QAT, "run_qat", "stage_not_qat", GATES, "run_refuses_ptq_stage"),
    K(QAT, "run_qat", "mode_invalid", GATES, "run_refuses_mode"),
    K(QAT, "run_qat", "real_run_test_hooks", GATES, "run_refuses_test_hooks_in_real"),
    K(QAT, "run_qat", "clip_source", GATES, "run_refuses_clip_source_in_real"),
    K(QAT, "run_qat", "config_pins", GATES, "run_refuses_config_pins"),
    K(QAT, "run_qat", "grad_clip_norm_invalid", GATES, "run_refuses_bad_clip"),
    K(QAT, "run_qat", "cuda_required", GATES, "run_refuses_cpu_in_real"),
    K(QAT, "run_qat", "cuda_initialized_before_seed", GATES, "run_refuses_cuda_initialised_in_real"),
    K(QAT, "run_qat", "out_dir_not_empty", GATES, "run_refuses_nonempty_out_dir"),
    K(QAT, "run_qat", "steps_per_epoch", GATES, "run_refuses_steps_per_epoch_in_real"),
    K(QAT, "run_qat", "loader_of_record", GATES, "run_refuses_loader_not_of_record_in_real"),
    K(QAT, "run_qat", "empty_loader", GATES, "run_refuses_empty_loader"),
    K(QAT, "run_qat", "backend_unavailable", GATES, "run_refuses_unavailable_backend"),
    K(QAT, "run_qat", "unfused_batchnorm", GATES, "run_refuses_unfused_bn_in_real"),
    K(QAT, "run_qat", "optimizer_of_record", GATES, "run_stops_on_an_optimizer_not_of_record"),
    K(QAT, "run_qat", "val_bracket_changed_state", GATES, "run_stops_when_val_changes_the_state"),
    # P23-P27: the epoch selection
    K(SEL, "load_rules", "rules_mismatch", case("d3_rules_disagreeing_with_config_refused"),
      "d3_rules_disagreeing_with_config_refused"),
    K(SEL, "_json", "record_unreadable", case("d3_unreadable_record_refused"), "d3_unreadable_record_refused"),
    K(SEL, "epoch_selection", "telemetry_sha256_mismatch", case("d3_telemetry_sha_mismatch_refused"),
      "d3_telemetry_sha_mismatch_refused"),
    K(SEL, "epoch_selection", "run_not_real", case("d3_smoke_run_refused"), "d3_smoke_run_refused"),
    K(SEL, "epoch_selection", "stage", case("d3_non_qat_stage_refused"), "d3_non_qat_stage_refused"),
    K(SEL, "epoch_selection", "stop_present", case("d3_stop_file_refused"), "d3_stop_file_refused"),
    K(SEL, "epoch_selection", "eval_record_missing", case("d3_missing_eval_record_exit_3"),
      "d3_missing_eval_record_exit_3"),
    K(SEL, "epoch_selection", "eval_record_mismatch", case("d3_swapped_eval_dir_refused"),
      "d3_swapped_eval_dir_refused"),
    K(SEL, "epoch_selection", "purpose_not_record", case("d3_timing_purpose_refused"), "d3_timing_purpose_refused"),
    K(SEL, "epoch_selection", "smoke_inputs", case("d3_smoke_input_records_refused"), "d3_smoke_input_records_refused"),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_two_commits_refused"), "d3_two_commits_refused"),
    K(SEL, "epoch_selection", "eval_incomplete", case("d3_partial_eval_incomplete"), "d3_partial_eval_incomplete"),
    K(SEL, "epoch_selection", "rejected_pilot_run", case("d3_rejected_pilot_run_needs_no_selection"),
      "d3_rejected_pilot_run_needs_no_selection"),
    K(SEL, "epoch_selection", "chain_checkpoint", case("d3_row_checkpoint_mismatch_refused"),
      "d3_row_checkpoint_mismatch_refused"),
    K(SEL, "epoch_selection", "exclusion_inconsistent", case("d3_excluding_a_finite_epoch_refused"),
      "d3_excluding_a_finite_epoch_refused"),
    K(SEL, "epoch_selection", "exclusion_inconsistent", case("d3_excluded_epoch_record_differs_refused"),
      "d3_excluded_epoch_record_differs_refused", k=1),
    K(SEL, "epoch_selection", "score_missing", case("d3_unscored_epoch_exit_3"), "d3_unscored_epoch_exit_3"),
    K(SEL, "epoch_selection", "conversion_missing", case("d3_missing_provenance_exit_3"),
      "d3_missing_provenance_exit_3"),
    K(SEL, "epoch_selection", "chain_provenance", case("d3_provenance_not_scored_one_refused"),
      "d3_provenance_not_scored_one_refused"),
    K(SEL, "epoch_selection", "conversion_missing", case("d3_missing_torchscript_exit_3"),
      "d3_missing_torchscript_exit_3", k=1),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_provenance_other_commit_refused"),
      "d3_provenance_other_commit_refused", k=1),
    K(SEL, "epoch_selection", "chain_broken", case("d3_identity_sha_mismatch_refused",
                                                   "d3_provenance_checkpoint_mismatch_refused"),
      "d3_identity_sha_mismatch_refused", "d3_provenance_checkpoint_mismatch_refused"),
    K(SEL, "epoch_selection", "score_missing", case("d3_missing_score_exit_3"), "d3_missing_score_exit_3", k=1),
    K(SEL, "epoch_selection", "score_artifact_invalid", case("d3_sha_mismatch_refused"), "d3_sha_mismatch_refused"),
    K(SEL, "epoch_selection", "chain_summary", case("d3_summary_not_scored_one_refused"),
      "d3_summary_not_scored_one_refused"),
    K(SEL, "epoch_selection", "eval_dir_mixed", case("d3_summary_commit_not_eval_commit_refused"),
      "d3_summary_commit_not_eval_commit_refused", k=2),
    K(SEL, "epoch_selection", "chain_summary", case("d3_summary_identity_refused"), "d3_summary_identity_refused", k=1),
    K(SEL, "epoch_selection", "evaluator_not_literal", case("d3_evaluator_not_literal_refused"),
      "d3_evaluator_not_literal_refused"),
    K(SEL, "epoch_selection", "summary_values", case("d3_capped_summary_refused", "d3_non_val_split_refused",
                                                     "d3_upstream_protocol_refused", "d3_smoke_status_refused"),
      "d3_capped_summary_refused", "d3_non_val_split_refused", "d3_upstream_protocol_refused",
      "d3_smoke_status_refused"),
    K(SEL, "epoch_selection", "score_invalid", case("d3_score_outside_unit_interval_refused",
                                                    "d3_score_not_float_refused"),
      "d3_score_outside_unit_interval_refused", "d3_score_not_float_refused"),
    K(SEL, "epoch_selection", "scores_dir_contents", case("d3_extra_scores_entry_refused"),
      "d3_extra_scores_entry_refused"),
    K(SEL, "epoch_selection", "no_convertible_epoch", case("d3_no_convertible_epoch_refused_exit_2"),
      "d3_no_convertible_epoch_refused_exit_2"),
    K(SEL, "epoch_selection", "summaries_differ", case("d3_metric_impl_differs_across_summaries",
                                                       "d3_null_image_digest_everywhere_refused",
                                                       "d3_null_commit_everywhere_refused"),
      "d3_metric_impl_differs_across_summaries", "d3_null_image_digest_everywhere_refused",
      "d3_null_commit_everywhere_refused"),
    K(SEL, "epoch_selection", "winner_changed", case("d3_winner_changed_during_selection_refused"),
      "d3_winner_changed_during_selection_refused"),
    # P27 and O1: the U4 clip
    K(SEL, "clip_selection", "candidates", case("d5_one_candidate_refused"), "d5_one_candidate_refused"),
    K(SEL, "clip_selection", "not_a_pilot_run", case("d5_non_pilot_run_refused"), "d5_non_pilot_run_refused"),
    # C8 (AM-21 item 3(d), ruling Q2-F/1): clip_selection's eval_record_missing, eval_record_mismatch and
    # purpose_not_record rows left with their raise sites (a rejected run's records are reported, never required)
    K(SEL, "clip_selection", "expect_telemetry_sha256_required",
      case("d5_expect_telemetry_sha256_required_one", "d5_expect_telemetry_sha256_required_none"),
      "d5_expect_telemetry_sha256_required_one", "d5_expect_telemetry_sha256_required_none"),
    K(SEL, "clip_selection", "telemetry_sha256_mismatch",
      case("d5_telemetry_sha256_mismatch_refused_live", "d5_telemetry_sha256_mismatch_refused_rejected"),
      "d5_telemetry_sha256_mismatch_refused_live", "d5_telemetry_sha256_mismatch_refused_rejected"),
    K(SEL, "clip_selection", "epoch_selection_missing", case("d5_missing_run_refused"), "d5_missing_run_refused"),
    K(SEL, "clip_selection", "epoch_selection_format", case("d5_selection_of_other_rules_refused"),
      "d5_selection_of_other_rules_refused"),
    K(SEL, "clip_selection", "selection_differs", case("d5_edited_epoch_selection_refused"),
      "d5_edited_epoch_selection_refused"),
    K(SEL, "clip_selection", "clip_values", case("d5_two_clip_1_runs_refused"), "d5_two_clip_1_runs_refused"),
    K(SEL, "clip_selection", "recipe_mismatch", case("d5_recipe_mismatch_refused", "d5_cpu_model_difference_refused"),
      "d5_recipe_mismatch_refused", "d5_cpu_model_difference_refused"),
    K(SEL, "clip_selection", "recipe_identity_null", case("d5_null_git_head_refused"), "d5_null_git_head_refused"),
    K(SEL, "clip_selection", "batch_order_differs", case("d5_batch_order_mismatch_refused"),
      "d5_batch_order_mismatch_refused"),
    K(SEL, "clip_selection", "pilot_evals_differ", case("d5_pilot_summary_commits_differ_refused",
                                                        "d5_pilot_host_labels_differ_refused"),
      "d5_pilot_summary_commits_differ_refused", "d5_pilot_host_labels_differ_refused"),
    K(SEL, "clip_selection", "no_winner", case("d5_both_rejected_no_winner_exit_2_a"),
      "d5_both_rejected_no_winner_exit_2_a"),
    # P24, P26: the run's record and its checkpoints, read by both selectors (and convert, score, finalize)
    K(ART, "refuse_test_path", "f'{what}_test_path'", case("d3_test_path_refused"), "d3_test_path_refused"),
    K(ART, "read_run_record", "telemetry_missing", case("d3_run_without_telemetry_refused"),
      "d3_run_without_telemetry_refused"),
    K(ART, "read_run_record", "telemetry_not_strict_json", case("d3_telemetry_not_strict_json_refused"),
      "d3_telemetry_not_strict_json_refused"),
    K(ART, "read_run_record", "run_meta_rows", case("d3_first_row_not_run_meta_refused"),
      "d3_first_row_not_run_meta_refused"),
    K(ART, "read_run_record", "epoch_end_duplicate", case("d3_duplicate_epoch_end_refused"),
      "d3_duplicate_epoch_end_refused"),
    K(ART, "require_complete", "run_incomplete", case("d3_incomplete_run_refused_exit_3"),
      "d3_incomplete_run_refused_exit_3"),
    K(ART, "verified_checkpoint", "epoch_unknown", RECORDS, "d4_convert_epoch_unknown_epoch_refused"),
    K(ART, "verified_checkpoint", "checkpoint_name", case("d3_checkpoint_name_refused"), "d3_checkpoint_name_refused"),
    K(ART, "verified_checkpoint", "checkpoint_missing", case("d3_missing_checkpoint_exit_3"),
      "d3_missing_checkpoint_exit_3"),
    K(ART, "verified_checkpoint", "checkpoint_sha256_mismatch", case("d3_checkpoint_bytes_changed_refused"),
      "d3_checkpoint_bytes_changed_refused"),
    K(ART, "read_ts_identity", "identity_missing", case("d3_torchscript_without_identity_refused"),
      "d3_torchscript_without_identity_refused"),
    K(SQE, "main", "output_exists", case("d3_refuses_existing_output"), "d3_refuses_existing_output"),
    K(SCL, "main", "output_exists", case("d5_refuses_existing_output"), "d5_refuses_existing_output"),
    K(QEE, "cmd_convert_epoch", "telemetry_sha256_mismatch", RECORDS, "d4_convert_epoch_telemetry_sha_mismatch"),
    # P24, P26: convert's and score's own record checks; P16's freeze cross-check
    K(QEE, "_epochs", "epochs_with_record", RECORDS, "d4_convert_record_takes_no_epochs",
      "d4_score_record_takes_no_epochs"),
    K(QEE, "_epochs", "epochs_range", RECORDS, "d4_convert_epochs_out_of_range"),
    K(QEE, "cmd_convert", "expect_telemetry_sha256_required", RECORDS, "d4_convert_record_requires_telemetry_sha"),
    K(QEE, "cmd_convert", "eval_dir_not_fresh", RECORDS, "d4_convert_eval_dir_not_fresh"),
    K(QEE, "cmd_convert", "telemetry_sha256_mismatch", RECORDS, "d4_convert_telemetry_sha_mismatch"),
    K(QEE, "cmd_convert", "run_not_real", RECORDS, "d4_convert_smoke_run_refused"),
    K(QEE, "cmd_convert", "freeze_cross_check", RECORDS, "d4_convert_stops_when_a_freeze_did_not_take"),
    K(QEE, "cmd_score", "stop_present", RECORDS, "d4_score_refuses_a_stop_file"),
    K(QEE, "cmd_score", "convert_record_missing", RECORDS, "d4_score_without_convert_record_exit_3"),
    K(QEE, "cmd_score", "convert_record_mismatch", RECORDS, "d4_score_convert_record_of_another_run"),
    K(QEE, "cmd_score", "purpose_mismatch", RECORDS, "d4_score_purpose_mismatch"),
    K(QEE, "cmd_score", "convert_record_epochs", RECORDS, "d4_score_record_needs_all_15_epochs"),
    K(QEE, "cmd_score", "epoch_not_converted", RECORDS, "d4_score_unconverted_epoch_refused"),
    K(QEE, "cmd_score", "scores_not_fresh", RECORDS, "d4_score_scores_dir_not_fresh"),
    K(QEE, "cmd_score", "output_exists", RECORDS, "d4_score_runs_once"),
    # P28: finalize
    K(QEE, "_read_json", "code", case("finalize_selection_not_json"), "finalize_selection_not_json"),
    K(QEE, "_read_json", "code", case("finalize_selection_not_an_object"), "finalize_selection_not_an_object", k=1),
    K(QEE, "_pilot_clip_winner", "clip_selection_missing", case("finalize_clip_selection_file_missing"),
      "finalize_clip_selection_file_missing"),
    K(QEE, "_pilot_clip_winner", "clip_selection_sha256_mismatch", case("finalize_checks_a_pinned_clip_selection_sha"),
      "finalize_checks_a_pinned_clip_selection_sha"),
    K(QEE, "_pilot_clip_winner", "clip_selection_format", case("finalize_clip_selection_of_another_format"),
      "finalize_clip_selection_of_another_format"),
    K(QEE, "_pilot_clip_winner", "clip_selection_winner_clip", case("finalize_refuses_an_edited_winner_clip"),
      "finalize_refuses_an_edited_winner_clip"),
    K(QEE, "_pilot_clip_winner", "not_the_clip_winner", case("finalize_refuses_the_retained_loser"),
      "finalize_refuses_the_retained_loser"),
    K(QEE, "_pilot_clip_winner", "clip_selection_stale", case("finalize_refuses_a_stale_clip_selection"),
      "finalize_refuses_a_stale_clip_selection"),
    K(QEE, "cmd_finalize", "fresh_process_required", case("finalize_fresh_process_guard"),
      "finalize_fresh_process_guard"),
    K(QEE, "cmd_finalize", "run_not_real", case("finalize_smoke_run_refused"), "finalize_smoke_run_refused"),
    K(QEE, "cmd_finalize", "output_exists", case("finalize_serves_the_winner"), "finalize_runs_once"),
    K(QEE, "cmd_finalize", "epoch_selection_missing", case("finalize_without_epoch_selection_exit_3"),
      "finalize_without_epoch_selection_exit_3"),
    K(QEE, "cmd_finalize", "epoch_selection_format", case("finalize_selection_of_other_rules_refused"),
      "finalize_selection_of_other_rules_refused"),
    K(QEE, "cmd_finalize", "epoch_selection_mismatch", case("finalize_selection_of_another_run_refused"),
      "finalize_selection_of_another_run_refused"),
    K(QEE, "cmd_finalize", "smoke_inputs", case("finalize_smoke_input_selection_refused"),
      "finalize_smoke_input_selection_refused"),
    K(QEE, "cmd_finalize", "clip_selection_required", case("finalize_pilot_requires_clip_selection"),
      "finalize_pilot_requires_clip_selection"),
    K(QEE, "cmd_finalize", "clip_selection_not_applicable", case("finalize_non_pilot_takes_no_clip_selection"),
      "finalize_non_pilot_takes_no_clip_selection"),
    K(QEE, "cmd_finalize", "winner_checkpoint_changed", case("finalize_winner_checkpoint_changed_refused"),
      "finalize_winner_checkpoint_changed_refused"),
    K(QEE, "cmd_finalize", "winner_files_changed", case("finalize_winner_artifact_changed_refused"),
      "finalize_winner_artifact_changed_refused"),
    K(QEE, "cmd_finalize", "winner_files_changed", case("finalize_state_dict_companion_changed_refused"),
      "finalize_state_dict_companion_changed_refused", k=1),
    K(QEE, "cmd_finalize", "x86_backend_unavailable", case("finalize_refuses_a_non_x86_engine"),
      "finalize_refuses_a_non_x86_engine"),
    K(QEE, "cmd_finalize", "x86_copy_checks", case("finalize_stops_on_a_failed_x86_check"),
      "finalize_stops_on_a_failed_x86_check"),
    # P14 / P10 / O2: check-run-meta
    K(QEE, "cmd_check_run_meta", "telemetry_missing", PROFILE, "profile_run_without_telemetry_exit_3"),
    K(QEE, "cmd_check_run_meta", "run_meta_rows", PROFILE, "profile_first_row_not_json"),
    K(QEE, "cmd_check_run_meta", "run_meta_rows", PROFILE, "profile_first_row_not_run_meta", k=1),
    K(QEE, "cmd_check_run_meta", "run_meta_profile", PROFILE, "profile_stops_on_num_workers", "profile_stops_on_lr"),
])


# ------------------------------------------------------------------ machinery
def build_shadow(dst: Path) -> str:
    """Copy SHADOW_PATHS and commit exactly them, so the trainer records a git_head as it does in the checkout.

    The commit's author, committer, date and message are fixed, so every shadow has the same HEAD and the runs
    kept in the cache (recorded in one shadow) match the runs of every other shadow. Returns that HEAD.
    """
    for rel in SHADOW_PATHS:
        src = REPO / rel
        if not src.is_file():
            raise SystemExit(f"RESULT: ERROR shadow path {rel} is missing from the checkout")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / rel)
    env = {**os.environ, "GIT_AUTHOR_NAME": "shadow", "GIT_AUTHOR_EMAIL": "shadow@example.invalid",
           "GIT_COMMITTER_NAME": "shadow", "GIT_COMMITTER_EMAIL": "shadow@example.invalid",
           "GIT_AUTHOR_DATE": "2026-10-05T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-05T00:00:00+00:00"}
    for argv in (["init", "-q"], ["add", "--", *SHADOW_PATHS],
                 ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "shadow"]):
        subprocess.run(["git", "-C", str(dst), *argv], check=True, capture_output=True, env=env)
    return subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def run_smoke(shadow: Path, cache: Path, spec: tuple, *, timeout: float = TIMEOUT) -> tuple[dict, str, float]:
    smoke, args = spec
    argv = [sys.executable, "-B", f"scripts/{smoke}.py", *[a.replace("{cache}", str(cache)) for a in args]]
    t0 = time.time()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": THREADS["value"],
           "MKL_NUM_THREADS": THREADS["value"]}
    out, timed_out, _pid = run_job(argv, cwd=str(shadow), env=env, timeout=timeout)
    if timed_out:
        out += "\nRESULT: TIMEOUT"
    statuses = {}
    in_checks = False
    for line in out.splitlines():
        if line.startswith("[CHECKS]"):
            in_checks = True
            continue
        if in_checks:
            m = CHECK_LINE.match(line)
            if m:
                statuses[m.group(1)] = m.group(2)
    tail = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
    return statuses, (tail[-1] if tail else out.strip()[-200:]), time.time() - t0


def _char_offset(line: str, col: int) -> int:
    return len(line.encode("utf-8")[:col].decode("utf-8"))


def refusal_sites(text: str, rel: str) -> list[dict]:
    tree = ast.parse(text)
    owner = {}
    for fn in sorted((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                     key=lambda n: n.lineno):
        for n in ast.walk(fn):
            if n is not fn:
                owner[id(n)] = fn.name
    out, seen = [], {}
    for node in sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
        call = None
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            call = node.exc
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        if call is None:
            continue
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if isinstance(node, ast.Expr) and name != "_refuse":
            continue
        if name not in ("_refuse", "QATRefused", "QATStop", "QATIncomplete") or not call.args:
            continue
        func = owner.get(id(node))
        scope = SCOPE.get(rel)
        if func is None or func == "_refuse" or (scope is not None and func not in scope):
            continue
        a0 = call.args[0]
        if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)):
            code = ast.unparse(a0)
            k = seen.get((func, code), 0)
            seen[(func, code)] = k + 1
            out.append({"rel": rel, "func": func, "code": code, "k": k, "node": node, "literal": False})
            continue
        k = seen.get((func, a0.value), 0)
        seen[(func, a0.value)] = k + 1
        out.append({"rel": rel, "func": func, "code": a0.value, "k": k, "node": node, "literal": True})
    return out


def removed(text: str, node) -> str:
    lines = text.splitlines(keepends=True)
    start = sum(len(x) for x in lines[:node.lineno - 1]) + _char_offset(lines[node.lineno - 1], node.col_offset)
    end = sum(len(x) for x in lines[:node.end_lineno - 1]) + _char_offset(lines[node.end_lineno - 1],
                                                                            node.end_col_offset)
    return text[:start] + "pass" + text[end:]


def check(name: str, ok, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def jobs_of(shadow: Path, refusals: bool) -> list[dict]:
    jobs = [{"id": m["id"], "rel": m["file"], "what": m["what"], "run": m["run"], "kills": m["kills"],
             "edit": ("replace", m["pairs"])} for m in MUTATIONS]
    if not refusals:
        return jobs
    found = set()
    for rel in SCOPE:
        text = (shadow / rel).read_text(encoding="utf-8")
        for site in refusal_sites(text, rel):
            key = (rel, site["func"], site["code"], site["k"])
            found.add(key)
            sid = f"{Path(rel).name}:{site['func']}:{site['code']}#{site['k']}"
            if key not in KILLS:
                check(f"{sid}_has_a_killing_check", False, "no entry in the refusal table")
                continue
            run, kills = KILLS[key]
            jobs.append({"id": sid, "rel": rel, "what": f"refusal removed (line {site['node'].lineno})", "run": run,
                         "kills": kills, "edit": ("site", key)})
    stale = sorted(set(KILLS) - found)
    check("refusal_table_names_only_real_sites", not stale, str(stale[:5]))
    return jobs


def edited(text: str, rel: str, edit: tuple) -> str:
    if edit[0] == "replace":
        out = text
        for old, new in edit[1]:
            if out.count(old) != 1:
                raise ValueError(f"the target occurs {out.count(old)} times in {rel}")
            out = out.replace(old, new)
    else:
        site = next(x for x in refusal_sites(text, rel) if (rel, x["func"], x["code"], x["k"]) == edit[1])
        out = removed(text, site["node"])
    ast.parse(out)                                             # every edit still compiles
    return out


def select_jobs(jobs: list[dict], only: str | None) -> list[dict]:
    """The jobs --only names, all of them when it is not given. An id that names no job, or no job at all, is a usage
    error (RESULT: ERROR, exit 1): a run with no edit would otherwise pass on the self-check's checks alone."""
    if only:
        wanted = set(only.split(","))
        unknown = sorted(wanted - {j["id"] for j in jobs})
        if unknown:
            raise SystemExit(f"RESULT: ERROR unknown job ids {unknown}")
        jobs = [j for j in jobs if j["id"] in wanted]
    if not jobs:
        raise SystemExit("RESULT: ERROR no edit selected")
    return jobs


def run_jobs(jobs: list[dict], shadows: list[Path], cache: Path) -> None:
    """main()'s baselines and edits; the self-check's signalled harness runs it on sleeper jobs (ruling Q2-F/2).

    The baselines: every killing check passes on the unmutated shadow, in the run each job makes (same argv); one
    run per smoke first, on one shadow (it fills that smoke's cache), then the rest on every shadow. Then one edit
    per job, in a free shadow."""
    specs = []
    for j in jobs:
        if j["run"] not in specs:
            specs.append(j["run"])
    base = {}

    def baseline(sh: Path, spec: tuple) -> None:
        statuses, res, secs = run_smoke(sh, cache, spec)
        base[spec] = statuses
        print(f"[baseline] {spec[0]} {' '.join(spec[1]).replace('{cache}', 'CACHE')}: {res} ({secs:.0f}s)",
              flush=True)
    # one run per smoke first, on one shadow (it fills that smoke's cache); then the rest on every shadow
    warm = list({spec[0]: spec for spec in reversed(specs)}.values())
    for spec in warm:
        baseline(shadows[0], spec)
    run_pool([s for s in specs if s not in warm], shadows, lambda spec, sh: baseline(sh, spec))
    for j in jobs:
        seen = base.get(j["run"], {})                                  # a skipped baseline passes none
        bad = [c for c in j["kills"] if seen.get(c) != "PASS"]
        if j["run"][0] == SELECTION:
            bad += [c for c in j["run"][1][1].split(",") if seen.get(c) != "PASS"]
        j["baseline_ok"] = not bad
        if bad:
            check(f"{j['id']}_baseline", False, f"killing checks not passing unmutated: {sorted(set(bad))}")

    # one edit per job, in its own shadow
    lock = threading.Lock()
    todo = [j for j in jobs if j["baseline_ok"]]
    done = {"n": 0}

    def work(j: dict, sh: Path) -> None:
        target = sh / j["rel"]
        original = target.read_text(encoding="utf-8")
        try:
            target.write_text(edited(original, j["rel"], j["edit"]), encoding="utf-8")
        except Exception as e:                           # noqa: BLE001 -- the edit is reported, never run
            with lock:
                check(f"{j['id']}_killed", False, f"the edit could not be applied: {type(e).__name__}: {e}")
            return
        try:
            statuses, res, secs = run_smoke(sh, cache, j["run"])
        finally:
            target.write_text(original, encoding="utf-8")
        failed = [c for c in j["kills"] if statuses.get(c) == "FAIL"]
        unreached = [c for c in j["kills"] if c not in statuses]
        usage_error = res.startswith("RESULT: ERROR unknown")
        killed = j["baseline_ok"] and not usage_error and bool(failed)       # a named check reported FAIL
        how = (f"by {', '.join(failed)}" if failed else
               f"NOT KILLED: the smoke did not reach {', '.join(unreached)} ({res})" if unreached
               else "SURVIVED")
        with lock:
            done["n"] += 1
            print(f"  [{done['n']}/{len(todo)}] {j['id']:58} {'KILLED' if killed else 'SURVIVED'}  {how} "
                  f"({secs:.0f}s)", flush=True)
            check(f"{j['id']}_killed", killed, f"{j['what']}: {how}")

    print("\n[edits]")
    run_pool(todo, shadows, work)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated job ids (M1, ..., or file:function:code#k)")
    ap.add_argument("--no-refusals", action="store_true", help="the M1-M24 mutations only")
    ap.add_argument("--workers", type=int, default=2, help="shadows edited and run in parallel")
    ap.add_argument("--keep-shadow", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="ruling Q2-F/2's job handling only: the self-check, with no shadow and no smoke")
    ap.add_argument("--self-check-signal-job", nargs=3, metavar=("MODE", "ROOT", "WATCH"), help=argparse.SUPPRESS)
    ap.add_argument("--self-check-main", nargs=2, metavar=("ROOT", "WATCH"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    install_stop_handlers()                          # first: the jobs run in sessions of their own
    if args.self_check_signal_job:
        return signal_job(*args.self_check_signal_job)
    t_all = time.time()
    if args.self_check_main:                         # the self-check's own run of main(), on sleeper jobs
        root, shadows, all_jobs = sleeper_harness(*args.self_check_main)
        head = "sleeper jobs"
    else:
        self_check()
        measured = {name for name, _ok, _detail in results}
        check("self_check_measured", set(SELF_CHECKS) <= measured,
              f"{len(SELF_CHECKS)} checks" if set(SELF_CHECKS) <= measured
              else f"not measured: {sorted(set(SELF_CHECKS) - measured)}")
        if args.self_check:
            return finish(t_all)
        root = safe_tmpdir("smoke_qat_mutations_")
        shadows = [root / f"shadow{i}" for i in range(max(1, args.workers))]
        # two torch processes spinning on the same cores slow each other many times over: share the CPUs out
        THREADS["value"] = str(max(1, (os.cpu_count() or 1) // len(shadows)))
        heads = {build_shadow(sh) for sh in shadows}
        if len(heads) != 1:
            raise SystemExit(f"RESULT: ERROR the shadows' commits differ: {sorted(heads)}")
        head = f"commit {heads.pop()[:12]}"
        all_jobs = jobs_of(shadows[0], refusals=not args.no_refusals)
    try:
        jobs = select_jobs(all_jobs, args.only)
    except SystemExit:
        if not args.keep_shadow:
            shutil.rmtree(root, ignore_errors=True)
        raise
    print("=" * 78)
    print("QAT MUTATION SMOKE (h) -- M1-M24 and P36's refusal table, each killed by a named check")
    print(f"{len(jobs)} edits | {len(SHADOW_PATHS)} shadow files x {len(shadows)} at {head} | temp {root}")
    print("=" * 78)
    run_jobs(jobs, shadows, root / "cache")
    verdicts = {name for name, _ok, _detail in results}
    missing = [j["id"] for j in jobs if f"{j['id']}_killed" not in verdicts and f"{j['id']}_baseline" not in verdicts]
    check("every_job_has_a_verdict", not missing,
          f"no verdict for {missing[:5]}" if missing else f"{len(jobs)} jobs, each killed or with its baseline failing")
    if not args.keep_shadow:
        shutil.rmtree(root, ignore_errors=True)
    return finish(t_all)


def finish(t_all: float) -> int:
    stopped = JOBS["stopped"]
    check("harness_ran_unsignalled", stopped is None,
          "" if stopped is None else f"signal {stopped}: every live job's process group was killed and the queued "
          "jobs were skipped; the verdicts of the jobs it stopped are void")
    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:64}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n[time] {time.time() - t_all:.0f}s")
    print(f"RESULT: {'PASS' if passed == len(results) and results else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
