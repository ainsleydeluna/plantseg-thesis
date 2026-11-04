"""Lane L-AM24: the exploratory, VAL-only code of ORCH-5's ruling EXPLORE-2 (audit EXPLORE-AUDIT-1; AM-24 draft).

Every module here is new. Nothing here is a run of record, selects anything of record or reaches TEST: each entry
point (scripts/explore_*.py) runs src/explore/guard.py first, which refuses a TEST split, a data root that fails
src/data/isolation.py's TRAIN/VAL-only check, a run ID without the "x-" prefix and an output path inside the
repository or an official run directory. The pinned trainers, quantizers, loaders and evaluator are imported
unchanged; where an item needs one named change inside a pinned module's loop, the change is applied at run time in
the exploratory process only (src/explore/patching.py), recorded in the run's x_run_meta.json, and the pinned
file's sha256 is recorded beside it.

This package imports nothing at import time.
"""
