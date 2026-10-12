"""AM-18 item 3's KD arm registry (lane L-TEACHER-ARMS, commit C1; ARMS plan section 3.0, audit rulings A1-A6).

An arm is one JSON file in configs/kd_arms/; the registry is those files, read in sorted order. Each file
is held to the strict schema kd_arm/1: exactly these keys, with these types and values.

  format       "kd_arm/1"
  name         the --arm value: the file's stem with "_" read as "-" (e3_r1.json -> "e3-r1")
  run_stage    the run's name, which run_meta's arm field and the payload record: "E3-" + the arm's suffix;
               name is run_stage in lower case; with teacher "arm" the suffix is the arm teacher (AM-18
               item 3(a): E3-X distils from X)
  e7_stage     the name of its PTQ run, "E7-" + the same suffix (AM-18 item 3(a): E7-X = PTQ of E3-X)
  stage        "e3": every arm of AM-18 item 3 is E3's recipe of record (item 3(a): no E2-X)
  seed         42 (item 3)
  horizon      80000 (item 3(a): 80,000 iterations)
  teacher      "arm" (an arm teacher in place of the teacher of record, item 3(a)) or "record"
  arm_teacher  "R1" or "R2" (AM-18 item 2) with teacher "arm"; null with teacher "record"
  go_rule      true exactly with teacher "arm": item 5's go rule gates the run (an arm launch gate's check;
               the trainer does not check it)
  K            1: the NMF draws per view and step. DL-84 makes item 6(a)'s correction unavailable and DL-96
               (F < 0.03) adds no K8 arm, so the single-draw targets of record stand for every KD run
  views        1 or 2: the teacher views per step
  targets      null when views is 1 (the record's construction, no builder); otherwise
               {"module": "src.distill.<name>", "builder": "<ClassName>"}, a TargetBuilder (below) whose
               module resolves to a file in src/distill
  cut_place    the arm's place in AM-18 item 7(b)'s cut order: 1, 2 or 4 (places 3 and 5 name arms that
               are not built)

Names, run stages, E7 stages, arm teachers and cut places are unique across the registry. A file that
cannot be read or parsed or breaks the schema, a path or targets module containing "test" (SL-1; refused
as a string, before any filesystem call, and a path again as resolved), an unknown name and a builder that
does not resolve (its module fails to import or is not a file in src/distill, or the class, K, VIEWS or
build is missing or wrong) are all refused as ArmSpecError, the trainer's [arm_spec] refusal. Nothing here
runs unless a run names an arm: train_distill.py imports this module only in main(), for --arm (a direct
run(arm=...) caller imports it to build the ArmSpec), so a run without an arm never reads the registry.

TargetBuilder interface (frozen: audit ruling A4; any change after the C1 push is a STOP for a ruling)

  A builder is the class an arm's `targets` names, with
    K        class attribute, an int equal to the arm's K
    VIEWS    class attribute, an int equal to the arm's views
    build(teacher, model_input, teacher_out) -> Targets
  The trainer instantiates the class once per run, with no arguments, and calls build() once per training
  step, after the step-1 input probe's try/finally: with the run's FrozenTeacher, the step's input tensor
  (the one augmented batch the student received) and the TeacherOutput of the trainer's own teacher call
  on that tensor. The Targets that build() returns exposes
    logit_kd_loss(head_logits, valid, T)    the Logit-KD term, used in place of
                                            logit_kd_kl(head_logits, <teacher logits>, valid, T)
    cwd_logit_loss(head_logits, valid, T)   the CWD logit-map term, used in place of
                                            cwd_channelwise_kl(head_logits, <teacher logits>, valid, T)
    finite                                  true when every teacher output the targets were built from is
                                            finite; the AM-7 abort record's teacher_finite requires it
  head_logits is the student's native OS8 head map [B, 116, 64, 64], valid the shared OS8 validity mask
  [B, 64, 64] (bool) and T the term's temperature (T_logit or T_cwd). Each loss returns a 0-dim tensor on
  the student's graph. The CWD feature-map term is not part of the interface: it always uses the
  trainer's own call's Stage 3 (teacher_out.feat_s16).
  TargetBuilder and Targets below state the same interface as typing.Protocol classes. The trainer checks
  a builder structurally (ArmSpec.builder_class), so a builder need not subclass them.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

REPO = Path(__file__).resolve().parents[2]
ARM_DIR = REPO / "configs" / "kd_arms"
DISTILL_DIR = REPO / "src" / "distill"

FORMAT = "kd_arm/1"
KEYS = ("format", "name", "run_stage", "e7_stage", "stage", "seed", "horizon", "teacher", "arm_teacher",
        "go_rule", "K", "views", "targets", "cut_place")
STAGE = "e3"                                   # AM-18 item 3(a): E3-X; no E2-X, no E4/E5/E6-X
RUN_PREFIX, E7_PREFIX = "E3-", "E7-"
SEED = 42                                      # AM-18 item 3: seed 42
HORIZON = 80000                                # AM-18 item 3(a): 80,000 iterations
TEACHERS = ("arm", "record")
ARM_TEACHERS = ("R1", "R2")                    # AM-18 item 2 (B-avg is not executable); teacher_diag.ARM_IDS
K_VALUES = (1,)                                # DL-84, DL-96: no K-draw average in any KD run
VIEW_COUNTS = (1, 2)
CUT_PLACES = (1, 2, 4)                         # AM-18 item 7(b); places 3 and 5 name arms that are not built
TARGET_KEYS = ("builder", "module")
SUFFIX_RE = re.compile(r"[A-Za-z0-9]+")
MODULE_RE = re.compile(r"src\.distill\.[A-Za-z_][A-Za-z0-9_]*")
CLASS_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ArmSpecError(ValueError):
    """A registry file, arm name or target builder that breaks kd_arm/1: the trainer's [arm_spec] refusal."""

    code = "arm_spec"


class Targets(Protocol):
    """One step's arm targets, as build() returns them (the module docstring's interface)."""

    finite: bool

    def logit_kd_loss(self, head_logits, valid, T): ...

    def cwd_logit_loss(self, head_logits, valid, T): ...


class TargetBuilder(Protocol):
    """The class an arm's `targets` names (the module docstring's interface)."""

    K: ClassVar[int]
    VIEWS: ClassVar[int]

    def build(self, teacher, model_input, teacher_out) -> Targets: ...


def _int(value) -> bool:
    """A JSON integer (a bool is never one)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_object(pairs):
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ArmSpecError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _refuse_test_path(path) -> None:
    """SL-1: a path containing "test" (case-insensitive) is refused as a string, before any filesystem call,
    then as resolved (a link into a "test" location)."""
    if "test" in str(path).lower():
        raise ArmSpecError(f"{path}: a path containing 'test' is refused (SL-1)")
    if "test" in os.path.realpath(path).lower():
        raise ArmSpecError(f"{path}: a path resolving to a 'test' location is refused (SL-1)")


@dataclass(frozen=True)
class ArmSpec:
    """One registered arm (kd_arm/1). `targets` is None or (module, builder class name)."""

    name: str
    run_stage: str
    e7_stage: str
    stage: str
    seed: int
    horizon: int
    teacher: str
    arm_teacher: str | None
    go_rule: bool
    K: int
    views: int
    targets: tuple[str, str] | None
    cut_place: int
    path: str
    sha256: str

    @property
    def target_construction(self) -> dict:
        """The run's target construction (AM-18 item 3's header): K and the number of views."""
        return {"K": self.K, "views": self.views}

    def run_error(self, *, stage_key: str, seed: int, horizon: int, mode: str) -> tuple[str, str] | None:
        """(code, text) when the run cannot be this arm, else None: the trainer's three arm refusals."""
        if stage_key != self.stage:
            return ("arm_stage", f"--arm {self.name} is an arm of stage {self.stage}, not {stage_key} "
                                 "(AM-18 item 3(a): E3-X only)")
        if mode == "real" and seed != self.seed:
            return ("arm_seed", f"--arm {self.name} runs at seed {self.seed} only, not {seed} (AM-18 item 3)")
        if horizon != self.horizon:
            return ("arm_iterations", f"--arm {self.name} runs {self.horizon} iterations only, not {horizon} "
                                      "(AM-18 item 3(a))")
        return None

    def builder_class(self):
        """The TargetBuilder class the arm names, checked against the interface; None without targets. A module
        that raises while it imports, or a class whose attributes raise when read, is refused as ArmSpecError
        like any other builder that does not resolve."""
        if self.targets is None:
            return None
        module_name, class_name = self.targets
        if "test" in module_name.lower():
            raise ArmSpecError(f"{self.path}: targets module {module_name}: a name containing 'test' is refused "
                               "(SL-1)")
        try:
            return self._checked_class(module_name, class_name)
        except ArmSpecError:
            raise
        except Exception as e:           # e.g. ModuleNotFoundError, or a SyntaxError or NameError in the module
            raise ArmSpecError(f"{self.path}: targets {module_name}.{class_name} does not resolve "
                               f"({type(e).__name__}: {e})") from None

    def _checked_class(self, module_name: str, class_name: str):
        module = importlib.import_module(module_name)
        file = getattr(module, "__file__", None)
        if file is None or Path(file).resolve().parent != DISTILL_DIR.resolve():
            raise ArmSpecError(f"{self.path}: targets module {module_name} resolves to {file}, not a file in "
                               f"{DISTILL_DIR}")
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type):
            raise ArmSpecError(f"{self.path}: {module_name} defines no class {class_name}")
        for attr, want in (("K", self.K), ("VIEWS", self.views)):
            got = getattr(cls, attr, None)
            if not (_int(got) and got == want):
                raise ArmSpecError(f"{self.path}: {class_name}.{attr} is {got!r}, the arm's is {want}")
        if not callable(getattr(cls, "build", None)):
            raise ArmSpecError(f"{self.path}: {class_name} has no build(teacher, model_input, teacher_out)")
        return cls

    def make_builder(self):
        """A new builder instance for one run (no arguments); None for an arm without targets."""
        cls = self.builder_class()
        return None if cls is None else cls()


def parse_arm(path) -> ArmSpec:
    """One registry file, held to kd_arm/1."""
    path = Path(path)
    _refuse_test_path(path)
    try:
        data = path.read_bytes()
        doc = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
    except ArmSpecError as e:
        raise ArmSpecError(f"{path}: {e}") from None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:   # RecursionError: nested too deep
        raise ArmSpecError(f"{path}: unreadable or not JSON ({type(e).__name__}: {e})") from None
    bad = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)
    if not isinstance(doc, dict):
        raise ArmSpecError(f"{path}: not a JSON object")
    if sorted(doc) != sorted(KEYS):
        raise ArmSpecError(f"{path}: keys {sorted(doc)}; kd_arm/1 has exactly {list(KEYS)}")
    need(doc["format"] == FORMAT, f"format {doc['format']!r} != {FORMAT!r}")
    name, run_stage, e7_stage = doc["name"], doc["run_stage"], doc["e7_stage"]
    need(isinstance(name, str) and name == path.stem.replace("_", "-"),
         f"name {name!r} is not the file's stem with '_' read as '-' ({path.stem.replace('_', '-')!r})")
    suffix = run_stage[len(RUN_PREFIX):] if isinstance(run_stage, str) and run_stage.startswith(RUN_PREFIX) else ""
    need(bool(SUFFIX_RE.fullmatch(suffix)), f"run_stage {run_stage!r} is not {RUN_PREFIX!r} + letters or digits")
    need(isinstance(name, str) and isinstance(run_stage, str) and name == run_stage.lower(),
         f"name {name!r} is not run_stage {run_stage!r} in lower case")
    need(e7_stage == E7_PREFIX + suffix, f"e7_stage {e7_stage!r} != {E7_PREFIX + suffix!r}")
    need(doc["stage"] == STAGE, f"stage {doc['stage']!r} != {STAGE!r}")
    need(_int(doc["seed"]) and doc["seed"] == SEED, f"seed {doc['seed']!r} != {SEED}")
    need(_int(doc["horizon"]) and doc["horizon"] == HORIZON, f"horizon {doc['horizon']!r} != {HORIZON}")
    teacher, arm_teacher = doc["teacher"], doc["arm_teacher"]
    need(teacher in TEACHERS, f"teacher {teacher!r} not in {TEACHERS}")
    need((arm_teacher in ARM_TEACHERS) if teacher == "arm" else arm_teacher is None,
         f"arm_teacher {arm_teacher!r}: one of {ARM_TEACHERS} with teacher 'arm', null with teacher 'record'")
    need(teacher != "arm" or arm_teacher == suffix,
         f"arm_teacher {arm_teacher!r} is not run_stage's suffix {suffix!r} (AM-18 item 3(a): E3-X distils from X)")
    need(doc["go_rule"] is (teacher == "arm"), f"go_rule {doc['go_rule']!r}: true exactly with teacher 'arm'")
    need(_int(doc["K"]) and doc["K"] in K_VALUES, f"K {doc['K']!r} not in {K_VALUES} (DL-84, DL-96)")
    need(_int(doc["views"]) and doc["views"] in VIEW_COUNTS, f"views {doc['views']!r} not in {VIEW_COUNTS}")
    targets = doc["targets"]
    if doc["views"] == 1:
        need(targets is None, f"targets {targets!r}: null with one view (the record's construction)")
        target_pair = None
    else:
        ok = (isinstance(targets, dict) and sorted(targets) == list(TARGET_KEYS)
              and isinstance(targets.get("module"), str) and bool(MODULE_RE.fullmatch(targets["module"]))
              and isinstance(targets.get("builder"), str) and bool(CLASS_RE.fullmatch(targets["builder"])))
        need(ok, f"targets {targets!r}: with views 2, {{'module': 'src.distill.<name>', 'builder': '<Class>'}}")
        target_pair = (targets["module"], targets["builder"]) if ok else None
    need(_int(doc["cut_place"]) and doc["cut_place"] in CUT_PLACES,
         f"cut_place {doc['cut_place']!r} not in {CUT_PLACES} (AM-18 item 7(b))")
    if bad:
        raise ArmSpecError(f"{path}: {'; '.join(bad)}")
    return ArmSpec(name=name, run_stage=run_stage, e7_stage=e7_stage, stage=doc["stage"], seed=doc["seed"],
                   horizon=doc["horizon"], teacher=teacher, arm_teacher=arm_teacher, go_rule=doc["go_rule"],
                   K=doc["K"], views=doc["views"], targets=target_pair, cut_place=doc["cut_place"],
                   path=str(path), sha256=hashlib.sha256(data).hexdigest())


def load_arms(arm_dir=None) -> dict[str, ArmSpec]:
    """Every registered arm, from sorted(<arm_dir>/*.json) (default configs/kd_arms/), keyed by name."""
    d = ARM_DIR if arm_dir is None else Path(arm_dir)
    _refuse_test_path(d)
    if not d.is_dir():
        raise ArmSpecError(f"{d}: the arm registry folder does not exist")
    arms: dict[str, ArmSpec] = {}
    for p in sorted(d.glob("*.json")):
        spec = parse_arm(p)
        if spec.name in arms:              # e3_r1.json and e3-r1.json both name e3-r1
            raise ArmSpecError(f"{d}: name {spec.name!r} registered by {Path(arms[spec.name].path).name} and "
                               f"{p.name}")
        arms[spec.name] = spec
    for field in ("run_stage", "e7_stage", "arm_teacher", "cut_place"):
        values = [getattr(s, field) for s in arms.values() if getattr(s, field) is not None]
        repeated = sorted({v for v in values if values.count(v) > 1}, key=str)
        if repeated:
            raise ArmSpecError(f"{d}: {field} {repeated} registered by more than one arm")
    return arms


def load_arm(name: str, arm_dir=None) -> ArmSpec:
    """The arm `name` (an --arm value), with its target builder, if it names one, resolved and checked."""
    arms = load_arms(arm_dir)
    if name not in arms:
        raise ArmSpecError(f"--arm {name!r} is not a registered arm; registered: {sorted(arms) or 'none'}")
    spec = arms[name]
    spec.builder_class()
    return spec
