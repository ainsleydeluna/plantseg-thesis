#!/usr/bin/env python3
"""Smoke for the exploratory guard, src/explore/guard.py (lane L-AM24). Synthetic fixtures only; no dataset.

Cases: the split refusals (TEST in any case first, then any non-TRAIN/VAL split, VAL-only tools), the run ID rules,
SL-1 on input paths, every output-path refusal (repository, name, official folder, official records in an ancestor,
inside an input, not fresh), the data-root isolation call (a missing root, a count mismatch, a passing fixture), the
x_run_meta.json fields, the exclusive write, and one mutation per guard: each guard replaced by a permissive version
must make a case fail.

    python -I scripts/smoke_explore_guard.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.explore import guard as G                                   # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402

TEST_WORD = "te" + "st"          # built, so no path literal in this file names it


def _fixture_root(base: Path, counts: dict) -> Path:
    root = base / "plantseg_fixture"
    for split, n in counts.items():
        for sub, ext in (("images", ".jpg"), ("annotations", ".png")):
            d = root / sub / split
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"img{i:03d}{ext}").write_bytes(b"")
    return root


def run_cases(verbose: bool = False) -> Cases:
    c = Cases("SMOKE_EXPLORE_GUARD", verbose=verbose)
    base = scratch_dir("x_guard_")

    # ---- step 2: splits
    for s in (TEST_WORD, TEST_WORD.upper(), "Te" + "st", f"val_{TEST_WORD}"):
        c.refuses(f"split {s!r} refused as split_test", "split_test", G.refuse_split, s)
    c.refuses("split None refused", "split_test", G.refuse_split, None)
    c.refuses("split 'trainval' refused", "split_not_trainval", G.refuse_split, "trainval")
    c.refuses("split 'TRAIN' (case) refused", "split_not_trainval", G.refuse_split, "TRAIN")
    c.check("split 'val' accepted", G.refuse_split("val") == "val")
    c.check("split 'train' accepted by a train/val tool", G.refuse_split("train") == "train")
    c.refuses("split 'train' refused by a VAL-only tool", "split_not_trainval", G.refuse_split, "train",
              allowed=G.VAL_ONLY)

    # ---- step 3: run IDs
    c.refuses("run ID without x- refused", "run_id_prefix", G.check_run_id, "e6-e2_s43_clip5.0_a1")
    c.refuses("run ID 'X-' (case) refused", "run_id_prefix", G.check_run_id, "X-c1-s43")
    c.refuses("run ID naming the forbidden word refused", "run_id_test", G.check_run_id, f"x-c1-{TEST_WORD}")
    c.refuses("run ID with a slash refused", "run_id_format", G.check_run_id, "x-c1/s43")
    c.refuses("bare 'x-' refused", "run_id_format", G.check_run_id, "x-")
    c.check("run ID x-c1-s43 accepted", G.check_run_id("x-c1-s43") == "x-c1-s43")

    # ---- step 4: SL-1 input paths (string first; a link whose target names the word, never followed further)
    c.refuses("input path naming the forbidden word refused", "input_path_test", G.check_input_paths,
              {"--run-dir": f"/data/{TEST_WORD}/run"})
    link = base / "innocent_link"
    os.symlink(str(base / f"dangling_{TEST_WORD.upper()}_target"), link)
    c.refuses("input link resolving to a forbidden name refused", "input_path_test", G.check_input_paths,
              {"--run-dir": str(link)})
    ok_inp = base / "inputs" / "parent_copy"
    ok_inp.mkdir(parents=True)
    c.check("plain input path accepted", G.check_input_paths({"--run-dir": str(ok_inp)})["--run-dir"]
            == os.path.realpath(ok_inp))

    # ---- step 5: output paths
    rid = "x-c1-s43"
    c.refuses("out dir inside the repository refused", "out_dir_in_repo", G.check_out_dir, REPO / "runs" / rid, rid)
    c.refuses("out dir not named after the run ID refused", "out_dir_not_run_id", G.check_out_dir,
              base / "x-other", rid)
    c.refuses("out dir under an official run folder refused", "out_dir_official", G.check_out_dir,
              base / "e2_s42_lambda0.5_80k_a1" / rid, rid)
    c.refuses("out dir under qat_* records folder refused", "out_dir_official", G.check_out_dir,
              base / "qat_e6e2_20261020" / "evals" / rid, rid)
    rec_dir = base / "somewhere"
    rec_dir.mkdir()
    (rec_dir / "best.json").write_text("{}")
    c.refuses("out dir under a folder holding best.json refused", "out_dir_official", G.check_out_dir,
              rec_dir / rid, rid)
    c.refuses("out dir inside an input refused", "out_dir_inside_input", G.check_out_dir, ok_inp / rid, rid,
              inputs=[str(ok_inp)])
    busy = base / "busy" / rid
    busy.mkdir(parents=True)
    (busy / "x.txt").write_text("x")
    c.refuses("non-empty out dir refused", "out_dir_not_fresh", G.check_out_dir, busy, rid)
    x_parent = base / "x-parent-item" / rid               # an x- ancestor with records is allowed
    (base / "x-parent-item").mkdir()
    (base / "x-parent-item" / "qat_telemetry.jsonl").write_text("")
    c.check("out dir under an x- folder with records accepted", G.out_dir_error(x_parent, rid) is None)
    good = base / "explore" / rid
    c.check("fresh out dir accepted", G.out_dir_error(good, rid) is None)

    # ---- step 6: the data root, through src.data.isolation
    from configs.data import DATA
    with patched(G, "EXPECTED_COUNTS", {"train": 3, "val": 2}):
        root = _fixture_root(base, {"train": 3, "val": 2})
        with patched_dict(DATA, "root", str(base / "no_such_root")):
            c.refuses("missing data root refused", "data_root_not_trainval_only", G.check_data_root)
        short = _fixture_root(base / "short", {"train": 2, "val": 2})
        with patched_dict(DATA, "root", str(short)):
            c.refuses("TRAIN/VAL count mismatch refused", "data_root_not_trainval_only", G.check_data_root)
        with patched_dict(DATA, "root", str(root)):
            iso = c.passes("fixture TRAIN/VAL-only root accepted", G.check_data_root)
            c.check("isolation record names the counts", bool(iso) and iso.get("counts") == {
                "train": {"images": 3, "masks": 3}, "val": {"images": 2, "masks": 2}})
            c.check("records-only tool skips the data root and says so",
                    G.check_data_root(records_only=True)["checked"] is False)

            # ---- step 7: x_run_meta.json
            out = base / "explore2" / "x-c1-s43"
            st = c.passes("guarded_start writes x_run_meta.json", G.guarded_start, item="C1", run_id="x-c1-s43",
                          out_dir=out, split="val", recipe={"what": "smoke"}, seeds=[43],
                          pinned_paths=["src/quant/qat.py", "src/models/student.py"],
                          input_paths={"--parent": str(ok_inp)})
            if st:
                doc = json.loads((out / G.RUN_META_NAME).read_text(encoding="utf-8"))
                want = {"exploratory": True, "amendment": "AM-24", "split": "val", "item": "C1",
                        "artifact_status": "exploratory", "run_id": "x-c1-s43", "registered": False,
                        "test_scored": False}
                c.check("run_meta carries the exploratory fields", all(doc.get(k) == v for k, v in want.items()),
                        str({k: doc.get(k) for k in want}))
                c.check("run_meta carries the full recipe", doc.get("recipe") == {"what": "smoke"})
                c.check("run_meta carries the git commit", isinstance(doc.get("git_head"), str)
                        and len(doc["git_head"]) == 40)
                c.check("run_meta carries the pinned modules' sha256",
                        set(doc.get("pinned_files_sha256") or {}) == {"src/quant/qat.py", "src/models/student.py"}
                        and all(len(v) == 64 for v in doc["pinned_files_sha256"].values()))
                c.check("run_meta carries the lane files' sha256",
                        "src/explore/guard.py" in (doc.get("lane_files_sha256") or {}))
                c.check("run_meta claim line", doc.get("claim_line", "").startswith("exploratory; VAL (846 images)"))
                c.check("run_meta records the isolation result", doc.get("isolation", {}).get("checked") is True)
                c.refuses("second guarded_start into the same dir refused", "out_dir_not_fresh", G.guarded_start,
                          item="C1", run_id="x-c1-s43", out_dir=out, split="val", recipe={}, seeds=[43])
                try:
                    G.write_exclusive(out / G.RUN_META_NAME, b"{}")
                    c.check("x_run_meta.json is never overwritten", False)
                except FileExistsError:
                    c.check("x_run_meta.json is never overwritten", True)
            c.refuses("guarded_start refuses split TEST before writing", "split_test", G.guarded_start, item="C1",
                      run_id="x-c1-s44", out_dir=base / "explore3" / "x-c1-s44", split=TEST_WORD, recipe={},
                      seeds=[44])
            c.check("nothing written after the TEST refusal", not (base / "explore3").exists())
            c.refuses("guarded_start refuses an unknown item", "item_unknown", G.guarded_start, item="Z9",
                      run_id="x-z9", out_dir=base / "explore4" / "x-z9", split="val", recipe={}, seeds=[42])
    return c


class patched_dict:
    def __init__(self, d: dict, key, value):
        self.d, self.key, self.value = d, key, value

    def __enter__(self):
        self.old = self.d[self.key]
        self.d[self.key] = self.value

    def __exit__(self, *exc):
        self.d[self.key] = self.old


def main() -> int:
    print("[smoke_explore_guard] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_guard] mutations (each guard made permissive must fail a case)")
    muts = {
        "refuse_split -> identity": lambda: patched(G, "refuse_split", lambda split, allowed=G.TRAINVAL: split),
        "check_run_id -> identity": lambda: patched(G, "check_run_id", lambda run_id: run_id),
        "check_input_paths -> no SL-1": lambda: patched(G, "check_input_paths",
                                                       lambda paths: {k: str(v) for k, v in paths.items() if v}),
        "out_dir_error -> None": lambda: patched(G, "out_dir_error", lambda out_dir, run_id, inputs=(): None),
        "check_data_root -> unchecked": lambda: patched(G, "check_data_root",
                                                       lambda records_only=False: {"checked": False}),
    }
    for name, mut in muts.items():
        killed, failed = mutation_killed(run_cases, mut)
        c.check(f"mutation KILLED: {name}", killed, f"{len(failed)} case(s) failed, e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
