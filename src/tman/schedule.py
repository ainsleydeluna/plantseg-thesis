"""The freeze instant, the TEST deadline and the dates of the cut rules (lane TMAN, B-1, C2; plan §2.3).

`freeze_instant` walks an ordered list of providers and returns one answer:
  1. the AM-22 freeze entry (B-2 puts its provider in front; absent in B-1);
  2. an AM-19 item 1(d) new-freeze entry after a delay entry dated the original freeze date;
  3. the schedule in force (configs/kd_schedule.json through sweep_select.load_schedule,
     src/training/sweep_select.py:799-819) and FREEZE_DATES[in_force].
Cut, completeness and existence checks use the original instant; "no run is added to TEST after the freeze" uses
the moved one (AM-19 item 1(c)). Instants are sweep_select.cutoff_instant (16:00:00 UTC, the end of the Manila
day) and `meets` is strict (sweep_select.py:420-428).

FREEZE_DATES and TEST_DATES are AM-19 item 1(a)'s table (docs/PREREGISTRATION_AMENDMENTS.md:1401-1402 and
:1408-1409), kept as constants (TB14); the smoke cross-checks FREEZE_DATES with scripts/preflight_arms.py:131 and
both with the committed amendment text. Under item 1(d) the freeze moves to the first later date at whose end every
never-cut run is complete, and the TEST date moves by the same number of days (AM-22 item 1(e): 2026-11-08, moved
as the resolver moves it; TB14).

The dates provider (TB12) reads AM-22's item 2(a) cutoff date and ceiling from the committed amendment text once
it is committed, and from a fixture until then (`am22_dates`). Dating is rule-aware (T-15): a cut is dated its
rule's day; a cutoff cut (B-2) is entered the day after its cutoff date (AM-22 item 2(a)), every other cut on its
day (AM-19 item 3(c)).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from src.training import sweep_select as ss

from . import schema as S

FREEZE_DATES = {"T": "2026-11-03", "R": "2026-12-18"}        # AM-19 item 1(a): TEST manifest freeze
TEST_DATES = {"T": "2026-11-08", "R": "2026-12-21"}          # AM-19 item 1(a): single TEST evaluation, no later than
AM22_HEADING = "## AM-22 — "
AM22_PUSH_BASE = "2026-10-17"                                # AM-22 header: dates move by the days after this one
AM22_CUTOFF_RE = re.compile(r"Its cutoff date is (\d{4}-\d{2}-\d{2})")
AM22_CEILING_RE = re.compile(r"in no case later than (\d{4}-\d{2}-\d{2})")
AM22_TEST_DEADLINE = "2026-11-08"                            # AM-22 item 1(e)
ENTERED_DAY_OFFSET = {"cutoff": 1}                           # AM-22 item 2(a): made on the day after its cutoff date


def iso(d: date) -> str:
    return d.isoformat()


def day(s: str) -> date:
    return date.fromisoformat(s)


def cutoff_instant(d: str) -> int:
    return ss.cutoff_instant(d)


def meets(ts, cutoff) -> bool:
    return ss.meets(ts, cutoff)


@dataclass(frozen=True)
class FreezeInstant:
    schedule: dict                      # {in_force, file, file_sha256, decision_log_entry}
    original_date: str
    moved_date: str | None
    source: dict = field(default_factory=dict)          # {provider, path, sha256}
    test_deadline: str = ""

    @property
    def original_utc(self) -> int:
        return cutoff_instant(self.original_date)

    @property
    def moved_utc(self) -> int:
        return cutoff_instant(self.moved_date or self.original_date)

    def as_block(self) -> dict:
        """The body's schedule block (plan §1.2)."""
        return {"in_force": self.schedule["in_force"], "file": self.schedule["file"],
                "file_sha256": self.schedule["file_sha256"],
                "decision_log_entry": self.schedule["decision_log_entry"],
                "original_freeze_date": self.original_date, "original_freeze_utc": ss.iso_utc(self.original_utc),
                "moved_freeze_date": self.moved_date, "freeze_source": dict(self.source),
                "test_deadline": self.test_deadline}


def schedule_block(src: ss.HeadSource) -> dict:
    """The schedule in force, read and checked by sweep_select.load_schedule (schedule_invalid otherwise)."""
    try:
        sch = ss.load_schedule(src)
    except ss.SelectionRefused as e:
        raise S.TmanRefused("schedule_invalid", f"[{e.code}] {e}") from None
    return {"in_force": sch["in_force"], "file": sch["file"], "file_sha256": sch["file_sha256"],
            "decision_log_entry": sch["decision_log_entry"]}


# ------------------------------------------------------------------------------------------------- providers
def schedule_provider(schedule: dict) -> FreezeInstant:
    in_force = schedule.get("in_force")
    if in_force not in FREEZE_DATES:
        raise S.TmanRefused("schedule_invalid", f"schedule in force {in_force!r} is neither T nor R")
    return FreezeInstant(dict(schedule), FREEZE_DATES[in_force], None,
                         {"provider": "schedule", "path": schedule.get("file"), "sha256": schedule.get("file_sha256")},
                         TEST_DATES[in_force])


def _conflict(detail: str):
    raise S.TmanRefused("freeze_source_conflict", detail + " (AM-19 item 1(d))")


def new_freeze_provider(base: FreezeInstant, delays: list, news: list) -> FreezeInstant | None:
    """AM-19 item 1(d): one delay entry, dated the original freeze date, and at most one new-freeze entry, dated the
    later date it names and citing the delay entry; the TEST deadline moves by the same number of days. `delays`
    and `news` are entry files read as {doc, path, sha256}. None when there is neither."""
    if not delays and not news:
        return None
    if len(delays) != 1:
        _conflict(f"{len(delays)} freeze delay entries; a new freeze needs exactly one")
    delay = delays[0]
    if delay["doc"].get("date") != base.original_date or delay["doc"].get("original_date") != base.original_date:
        _conflict(f"the delay entry {delay['path']} is dated {delay['doc'].get('date')!r} for the freeze "
                  f"{delay['doc'].get('original_date')!r}, not the original freeze date {base.original_date}")
    if not news:
        return FreezeInstant(base.schedule, base.original_date, None,
                             {"provider": "freeze_delay", "path": delay["path"], "sha256": delay["sha256"]},
                             base.test_deadline)
    if len(news) != 1:
        _conflict(f"{len(news)} new-freeze entries; the freeze moves once")
    new = news[0]
    nd = new["doc"].get("new_date")
    if not (isinstance(nd, str) and S.ISO_DAY_RE.fullmatch(nd)) or day(nd) <= day(base.original_date):
        _conflict(f"new freeze date {nd!r} is not later than the original {base.original_date}")
    if new["doc"].get("date") != nd:
        _conflict(f"the new freeze {nd} is entered on {new['doc'].get('date')!r}, not on the day it occurs")
    if new["doc"].get("delay_entry") != {"path": delay["path"], "sha256": delay["sha256"]}:
        _conflict("the new-freeze entry does not cite the delay entry")
    moved = (day(nd) - day(base.original_date)).days
    return FreezeInstant(base.schedule, base.original_date, nd,
                         {"provider": "new_freeze", "path": new["path"], "sha256": new["sha256"]},
                         iso(day(base.test_deadline) + timedelta(days=moved)))


def freeze_instant(schedule: dict, *, delays=(), news=(), providers=()) -> FreezeInstant:
    """The first provider that answers. `providers` are callables placed in front (B-2's AM-22 freeze entry); each
    takes the schedule's FreezeInstant and returns a FreezeInstant or None."""
    base = schedule_provider(schedule)
    for p in providers:
        got = p(base)
        if got is not None:
            return got
    return new_freeze_provider(base, list(delays), list(news)) or base


def first_complete_day(instants: list, original_date: str) -> str | None:
    """AM-19 item 1(d): the first date later than the original freeze date at whose end every never-cut run is
    complete, from their completion instants; None when one of them is not complete (None). A run complete at an
    instant t is complete at the end of t's Manila day and of every later day."""
    if any(t is None for t in instants):
        return None
    return iso(max([day(original_date) + timedelta(days=1)] + [ss.manila_day(t) for t in instants]))


# ------------------------------------------------------------------------------------- AM-22's dates (TB12)
@dataclass(frozen=True)
class Am22Dates:
    cutoff: str
    ceiling: str
    base_cutoff: str
    base_ceiling: str
    shift_days: int
    test_deadline: str = AM22_TEST_DEADLINE


def am22_text(amendments_text: str) -> str | None:
    """AM-22's section of the committed amendments file, or None when it is not committed."""
    i = amendments_text.find(AM22_HEADING)
    if i < 0:
        return None
    j = amendments_text.find("\n## ", i + 1)
    return amendments_text[i:] if j < 0 else amendments_text[i:j]


def am22_dates(text: str, pushed_day: str | None) -> Am22Dates:
    """Item 2(a)'s cutoff date and ceiling, read from AM-22's text, moved later by the days from 2026-10-17 to the
    day its commit reached the remote when that day is after 2026-10-17 (AM-22 header)."""
    c, g = AM22_CUTOFF_RE.search(text), AM22_CEILING_RE.search(text)
    if not (c and g):
        raise S.TmanRefused("schedule_invalid", "AM-22's text gives no item 2(a) cutoff date and ceiling")
    shift = 0
    if pushed_day is not None:
        shift = max(0, (day(pushed_day) - day(AM22_PUSH_BASE)).days)
    move = timedelta(days=shift)
    return Am22Dates(iso(day(c.group(1)) + move), iso(day(g.group(1)) + move), c.group(1), g.group(1), shift)


# ---------------------------------------------------------------------------------- rule-aware dating (T-15)
def expected_entered_day(rule: str, rule_day: str) -> str:
    """The day a cut entry is due in the decision log: its rule's day, or the day after for a cutoff cut."""
    return iso(day(rule_day) + timedelta(days=ENTERED_DAY_OFFSET.get(rule, 0)))


def manila_day(ts) -> str:
    return iso(ss.manila_day(ts))


def parse_utc_any(s: str) -> float:
    """An instant written by sweep_select.iso_utc ('...Z', with or without fractional seconds) as Unix time."""
    if not isinstance(s, str) or not s.endswith("Z"):
        raise ValueError(f"not a UTC instant: {s!r}")
    return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc).timestamp()
