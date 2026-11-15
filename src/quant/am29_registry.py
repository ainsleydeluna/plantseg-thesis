"""AM-29's stages, registered into three pinned registries of the calling process only (lane L-AM29).

The pinned code resolves a stage by its name in three tables:

  src/quant/stages.py          QUANT_STAGES      resolve_quant_stage (convert_epoch, load_source_for_stage)
  src/quant/qat_select.py      RULES_FILES       load_rules' check of a rules file (rules_spec, rules_rel_of)
  src/eval/stage_artifacts.py  STAGE_ARTIFACTS   validate_int8_artifact (resolve_stage_artifact)

None of the three files is edited (lane L-AM29, hard limit 3). An AM-29 entry point calls register() before it
calls any pinned function that reads one of them: register() adds AM-29's keys to the three tables of its own
process, after asserting that no key it adds exists already, so every key that existed resolves exactly as
before (register_report() records the before/after comparison). A process that never imports this module, as
every AM-28 and earlier entry point never does, holds the tables exactly as the pinned files define them.

The AM-29 stages: one per arm of the allow-list (src/quant/am29.py ARMS), method "qat" (T1-T4, LR-s42/43/44) or
"qat_kd" (A3-s43/44, E6-KD's recipe), source stage E3 (E3 and E3-R2 checkpoints record stage E3), named
"AM29-<ARM>" in capitals because src/eval/model_loading.py compares a state_dict artifact's stage in capitals.
Their converted artifacts are int8_qat artifacts whose conversion records method "qat", as E6-KD's does.
"""
from __future__ import annotations

import json

# arm -> stage name; the allow-list itself (factor, rate, seed, parent) is src/quant/am29.py ARMS
ARM_STAGES = {
    "T1": "AM29-T1", "T2": "AM29-T2", "T3": "AM29-T3", "T4": "AM29-T4",
    "A3-s43": "AM29-A3-S43", "A3-s44": "AM29-A3-S44",
    "LR-s42": "AM29-LR-S42", "LR-s43": "AM29-LR-S43", "LR-s44": "AM29-LR-S44",
}
STAGE_ARMS = {v: k for k, v in ARM_STAGES.items()}
AM29_STAGES = tuple(ARM_STAGES.values())
RULES_REL = "configs/qat_selection_rules_am29.json"


def _objective(arm: str) -> str:
    if arm.startswith("T"):
        return ("INT8 QAT (E6's recipe of record) from E3-R2 seed 42 with a frozen FP32 copy of the parent as the "
                "teacher of one logit-KD term (AM-29)")
    if arm.startswith("A3"):
        return "E6-KD's INT8 QAT recipe (AM-3; DL-39, DL-51) from E3 at seed 43 or 44 (AM-29)"
    return "INT8 QAT (E6's recipe of record) from E3 at learning rate 6.25e-5 (AM-29)"


QUANT_ENTRIES = {stage.lower(): {"name": stage, "method": "qat_kd" if arm.startswith("A3") else "qat",
                                 "source_stage": "E3", "objective": _objective(arm)}
                 for arm, stage in ARM_STAGES.items()}
RULES_ENTRY = {"stages": list(AM29_STAGES), "qat_clip": False}
STAGE_ARTIFACT_ENTRIES = {stage: {"kind": "int8_artifact", "precision": "int8_qat", "source_stage": "E3",
                                  "method": "qat"} for stage in AM29_STAGES}


class RegistryClash(RuntimeError):
    """A key AM-29 would add already exists with another value: nothing is added."""


def _tables():
    from src.eval import stage_artifacts as SA

    from . import qat_select as S
    from . import stages as ST
    return ((ST, "QUANT_STAGES", QUANT_ENTRIES), (S, "RULES_FILES", {RULES_REL: RULES_ENTRY}),
            (SA, "STAGE_ARTIFACTS", STAGE_ARTIFACT_ENTRIES))


def _dump(d: dict) -> str:
    return json.dumps(d, sort_keys=True, default=str)


_REPORT: dict = {}


def register() -> dict:
    """Add AM-29's keys to the three tables of this process. Idempotent; refuses a clash before adding any key."""
    tables = _tables()
    for mod, name, entries in tables:
        table = getattr(mod, name)
        clash = [k for k, v in entries.items() if k in table and _dump(table[k]) != _dump(v)]
        if clash:
            raise RegistryClash(f"{mod.__name__}.{name} already holds {clash} with other values; nothing was added")
    if _REPORT:
        return dict(_REPORT)
    before = {f"{mod.__name__}.{name}": _dump(getattr(mod, name)) for mod, name, _ in tables}
    for mod, name, entries in tables:
        table = getattr(mod, name)
        for k, v in entries.items():
            table.setdefault(k, json.loads(json.dumps(v)))
    after_existing = {}
    for mod, name, entries in tables:
        table = getattr(mod, name)
        after_existing[f"{mod.__name__}.{name}"] = _dump({k: v for k, v in table.items() if k not in entries})
    _REPORT.update({"tables": sorted(before), "added": {f"{m.__name__}.{n}": sorted(e) for m, n, e in tables},
                    "existing_keys_unchanged": all(before[k] == after_existing[k] for k in before)})
    if not _REPORT["existing_keys_unchanged"]:
        raise RegistryClash("registering AM-29 changed an existing entry")
    return dict(_REPORT)


def registered() -> bool:
    return bool(_REPORT)


def stage_of_arm(arm: str) -> str:
    if arm not in ARM_STAGES:
        raise KeyError(f"arm {arm!r} is not on AM-29's allow-list {sorted(ARM_STAGES)}")
    return ARM_STAGES[arm]


__all__ = ["AM29_STAGES", "ARM_STAGES", "QUANT_ENTRIES", "RULES_ENTRY", "RULES_REL", "RegistryClash",
           "STAGE_ARMS", "STAGE_ARTIFACT_ENTRIES", "register", "registered", "stage_of_arm"]
