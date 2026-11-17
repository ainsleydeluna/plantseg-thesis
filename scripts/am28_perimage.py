#!/usr/bin/env python3
"""AM-28 items 7 and 9(f) (draft v1): the per-image readings, descriptive, with no test. This is a driver outside the
repository. It imports the LREP code of record unchanged (at E = dc40dd2: src/stats is the same at master aa8cac2) and
calls the path that src/stats/arm_tables.py's arm_row uses:
  - load the official statistics artifact with its 37 inputs (heldout_inputs.load_stats_artifact) and the TEST
    inventory (open_inventory, its sha256 checked), then the family slots E1, E3 and E6 at seed 42 (family_inputs) and
    the arm slots A, F and G (load_slot);
  - for each pair, the per-image disease-only mIoU vectors are aligned (align_runs under the artifact's policy, or
    dt_pair for a descriptive_test read, as arm_row);
  - the mean delta and its two-sided BCa 95% interval are computed at B = bootstrap.PRODUCTION_B
    (lrep_descriptive.mean_delta_task), with seeds from LREP's descriptive namespace.
Contrast ids, fixed in AM-28 before any reading: am17b__f__e1, am17b__g__e1, am17b__a__e1 and am17b__e3__a (DL-212 (c));
arm__r2__e6 and arm__r1__e6 (lrep_descriptive.arm_contrast_id, registered in the code before TEST) for E6-X − E6.
  python -I -B am28_perimage_20261115.py dl212c <S> <OUT.json>
  python -I -B am28_perimage_20261115.py e6x <S> <LOOKS> <OUT.json>
<S> is the TEST session root (its reports/derived/tday_inventory.json and reports/tday_results/stats/...); <LOOKS> is
the look kit's output folder. It prints one ROW line per contrast (n, mean delta and the interval, in pp) and nothing
per image, and writes <OUT.json> once. Run it with the code clone at E as the working folder. A run that stops
before printing its rows is made again after a fix; a run that printed its rows is never repeated (AM-28 item 7).
"""
import json
import math
import os
import sys
from pathlib import Path

if not os.path.isfile(os.path.join(os.getcwd(), "src", "stats", "arm_tables.py")):
    print("STOP: run this from the code clone at E (src/stats/arm_tables.py is not in the working folder)")
    raise SystemExit(1)
sys.path.insert(0, os.getcwd())                       # python -I adds neither the script's folder nor PYTHONPATH

from src.stats import bootstrap as BS  # noqa: E402
from src.stats import driver as D  # noqa: E402
from src.stats import heldout_inputs as H  # noqa: E402
from src.stats import lrep_descriptive as LD  # noqa: E402
from src.stats.align import METRIC_DISEASE_ONLY, align_runs  # noqa: E402

INV_REL = "reports/derived/tday_inventory.json"
INV_SHA256 = "e4e3b90f13121fe863436f671b73d87ae2ea344bebf8b513a9851a9cf3a64650"       # DL-248
STATS_REL = "reports/tday_results/stats/tday-official-20261105"                         # DL-249
ARM_SLOTS = (("A", "A:s42:alpha25:80k"), ("F", "F:s42:alpha25:80k"), ("G", "G:s42:80k"))
DL212C = (("am17b__f__e1", "F", "E1"), ("am17b__g__e1", "G", "E1"), ("am17b__a__e1", "A", "E1"),
          ("am17b__e3__a", "E3", "A"))
E6X = (("r2", "E6-R2", ("am28_clean_e6r2_a1", "am28_clean_e6r2_a2")),
       ("r1", "E6-R1", ("am28_clean_e6r1_a1", "am28_clean_e6r1_a2")))


def die(msg: str) -> None:
    print(f"STOP: {msg}")
    print("RESULT: PER-IMAGE STOP (nothing was written)")
    raise SystemExit(1)


def pp(v) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{100.0 * v:+.2f}"


def row_line(r: dict) -> str:
    if r.get("status") != "produced":
        return f"ROW {r['contrast']}: {r['status']}"
    return (f"ROW {r['contrast']} ({r['contrast_id']}): n_paired {r['n_paired']}, per-image mean delta "
            f"{pp(r['observed'])} pp, BCa 95% CI [{pp(r['lower_bound'])}, {pp(r['upper_bound'])}] pp "
            f"(B {r['B']}, {r['interval_method']}, status {r['bca_status']}, warnings {len(r['warnings'])})")


def pair_row(cid: str, label: str, base, cand, policy, *, s_root: Path, dt: bool) -> dict:
    if dt:
        pv = H.dt_pair(base, cand, input_root=s_root, metric=METRIC_DISEASE_ONLY)
    else:
        pv = align_runs(base.run, cand.run, policy=policy, metric=METRIC_DISEASE_ONLY)
    r = LD.task_json(LD.mean_delta_task(LD.check_contrast_id(cid), pv.delta, B=BS.PRODUCTION_B))
    r["bca_status"] = r.pop("status")
    return {"contrast": label, "status": "produced", "metric": METRIC_DISEASE_ONLY, "n_paired": pv.n,
            "alignment_status": pv.alignment_status, "pair_policy": pv.policy.value, **r}


def open_session(s_root: Path):
    inv_p, stats_p = s_root / INV_REL, s_root / STATS_REL
    if H.sha256_file(inv_p) != INV_SHA256:
        die(f"{INV_REL} is not sha256 {INV_SHA256} (DL-248)")
    stats = H.load_stats_artifact(stats_p, s_root)
    inv = H.open_inventory(inv_p, policy=stats.policy, expect_sha256=INV_SHA256)
    cells = H.planned_cells(inv)
    return stats, inv, cells


def main(argv) -> int:
    if len(argv) not in (3, 4) or argv[0] not in ("dl212c", "e6x") or (argv[0] == "e6x") != (len(argv) == 4):
        die("usage: dl212c <S> <OUT.json> | e6x <S> <LOOKS> <OUT.json>")
    mode, s_root, out = argv[0], Path(argv[1]), Path(argv[-1])
    if out.exists():
        die(f"{out} exists")
    if any("test" in str(p).lower() for p in (s_root, out) + ((Path(argv[2]),) if mode == "e6x" else ())):
        die("a path argument names 'test'")
    try:
        stats, inv, cells = open_session(s_root)
        rows = []
        if mode == "dl212c":
            fam = H.family_inputs(stats, inv, cells, stages=("E1", "E3"))
            arms = {st: H.load_slot(stats, inv, slot=sl, stage=st, precision="fp32", input_root=s_root)
                    for st, sl in ARM_SLOTS}
            side = {"E1": fam["E1"], "E3": fam["E3"], **arms}
            for cid, cand, base in DL212C:
                c = side[cand]
                dt = cand in arms and H.dt_read(stats.policy, c.read)
                rows.append(pair_row(cid, f"{cand} - {base}", side[base], c, stats.policy, s_root=s_root, dt=dt))
        else:
            looks = Path(argv[2])
            fam = H.family_inputs(stats, inv, cells, stages=("E6",))
            for x, stage, names in E6X:
                found = [n for n in names if (looks / n).is_dir()]
                if len(found) != 1:
                    rows.append({"contrast": f"{stage} - E6", "status": f"not produced: {len(found)} clean looks found"})
                    continue
                ev_stage = H.stage_spec(stage)[0]
                cand = D.load_input(ev_stage, D.CLEAN, found[0], looks, stats.policy)
                rows.append(pair_row(LD.arm_contrast_id(x, "e6"), f"{stage} - E6", fam["E6"], cand, stats.policy,
                                     s_root=s_root, dt=False))
    except (H.LrepRefusal, D.DriverRefusal) as e:
        die(f"refused: {type(e).__name__}: {e}")
    doc = {"format": "am28_perimage/1", "amendment": "AM-28 items 7 and 9(f)", "mode": mode, "descriptive": True,
           "test": None, "inventory_sha256": INV_SHA256, "stats_artifact": STATS_REL, "B": BS.PRODUCTION_B,
           "rows": rows}
    text = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(out, "x", encoding="utf-8") as f:
        f.write(text)
    for r in rows:
        print(row_line(r))
    print(f"RESULT: PER-IMAGE {mode.upper()} DONE ({len(rows)} rows; {out.name} written; paste the ROW lines to ORCH)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
