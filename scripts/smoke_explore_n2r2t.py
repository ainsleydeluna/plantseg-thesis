#!/usr/bin/env python3
"""Smoke for N2R2T (addendum L-AM24-ADD-3; AM-29a's E6-R2-MT, run ID x-n2r2t-s42): src/explore/n2r2t.py and the N2R2T
branches of src/explore/qat_explore.py, src/explore/guard.py, scripts/explore_qat.py and scripts/explore_qat_blockb.py.
Synthetic students (random weights), loaders, run folders and records checkout (scripts/smoke_explore_n2r2.py's
fixtures, imported unchanged); no dataset, no checkpoint of record, no GPU, no TEST.

    python -B -I scripts/smoke_explore_n2r2t.py --am29-root <dir> [--sections copies,A,B,C,E,F] [--no-mutations]

--am29-root is a separate checkout of lane/l-am29 at 2a94019f (at least its src/, configs/ and
reports/e1_class_weights.json); its src/quant/am29.py must have the sha256 and git blob id that src/explore/n2r2t.py
records. L-AM29's code runs there, in a second process (this file with --ref) whose sys.path holds that checkout and
not this one. Sections:

  copies  each unchanged copy in src/explore/n2r2t.py is src/quant/am29.py's lines at 2a94019f, byte for byte (sha256
          recomputed from the file and from this module's source); run_am29_qat's source hash; the term's constants;
          N2R2T on the allow-lists; the existing items' entries as at e11073a;
  A       at factor 0, N2R2T's train rows (loss, CE, Dice, gradient norm, clipped flag, learning rate, batch) and its
          15 epoch states, final model state and optimizer state after 30 steps (15 epochs x 2 steps) are bitwise those
          of N2R2's path (src/explore/qat_explore.py run_n2r2), both with the ten sites and TF32 off, from the same
          parent through E6-R2's parent loader, on the same synthetic loaders and seed; at factor 0.5 the loss differs
          and is CE + Dice + 0.25 x the term from step 1 (no ramp);
  B       with the sites and TF32 both off, N2R2T's rows and states after 30 steps are bitwise those of run_am29_qat, arm
          T1, run from --am29-root, both from one in-memory model state (one state file) on the same loaders and seed;
  C       the term on one fixed batch equals src/quant/am29.py t_logit_kd's to the last bit (in the --am29-root
          process, on the same tensors; and from the same state through both checkouts' forwards), and equals
          0.25 x losses.logit_kd_kl(head logits, teacher logits, downsample_validity, T=4); the student tap is on
          head.ff_add's fake-quant grid; the teacher is a frozen FP32 copy built without a draw from the global stream;
  E       a real run refuses a test hook, injected loaders, a step cap, NaN injection, a VAL cap, a factor other than
          0.5, another E3 lambda, another seed, the change removed and a teacher that is not frozen, writing nothing;
          the CLI's N2R2T gates (in process, up to the run call) and its refusals (subprocesses);
  F       block B accepts N2R2T and refuses an unknown item; N2R2T's run of another stage; card fix 1 ends in
          [fqoff_stage_not_in_scope] for N2R2T as for N2R2 (the evaluator in process and the script itself); block B's
          whole flow with its four registered steps stood in for writes x_blockb.json as for N2R2; the CLI.
Mutations: the term's weight dropped, run_n2r2t without the sites, N2R2T on E6-E2's stage, any factor accepted in a
real run, and block B without N2R2T's card fix 1 each fail a case.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
SECTIONS = ("copies", "A", "B", "C", "E", "F")
SHARED = ("src/quant/qat.py", "src/quant/prepare.py", "src/quant/qconfig.py", "src/quant/stages.py",
          "src/quant/ptq.py", "src/training/train_distill.py", "src/training/losses.py", "src/training/train_e1.py",
          "src/distill/features.py", "src/distill/teacher.py", "src/models/student.py", "src/seeds.py",
          "configs/distill.py", "configs/quant.py", "configs/model.py", "reports/e1_class_weights.json")
TRAIN_KEYS = ("epoch", "step", "epoch_step", "lr", "loss", "ce", "dice", "grad_norm", "clipped", "bn_frozen",
              "observers_enabled", "batch_sha256", "state_nonfinite")
TERM_KEYS = ("sup", "logit_kd", "kd_weight")
END_KEYS = ("epoch", "step", "steps_in_epoch", "state_sha256", "fake_quant_val_all_class_miou", "clipped_steps",
            "max_pre_clip_norm", "state_finite", "nonfinite_since_step", "bn_frozen", "observers_enabled",
            "run_complete")
VAL_KEYS = ("epoch", "step", "all_class_miou", "disease_only_miou_PROVISIONAL", "val_batches", "val_total_px")


# ------------------------------------------------------------------ helpers both processes use (torch and stdlib only)
def quiet(*_a, **_k):
    return None


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob_id(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def canon(v):
    """Values as compared bitwise: a float by its hex form (so 0.0 and -0.0 differ), containers element by element."""
    if isinstance(v, float):
        return v.hex()
    if isinstance(v, dict):
        return {k: canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    return v


def read_rows(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out / "qat_telemetry.jsonl").read_text(encoding="utf-8").splitlines()]


def pick(rows: list[dict], event: str, keys) -> list[dict]:
    return [{k: canon(r.get(k)) for k in keys} for r in rows if r.get("event") == event]


def make_loaders(init: dict):
    """The synthetic loaders: 4 TRAIN images (batch 2, shuffled by a seeded generator, drop_last: 2 steps an epoch) and
    2 VAL images, 64 x 64; the same tensors and generators in every run of both processes."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    train = DataLoader(TensorDataset(init["train_x"], init["train_y"]), batch_size=2, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(1234))
    val = DataLoader(TensorDataset(init["val_x"], init["val_y"]), batch_size=2, shuffle=False, drop_last=False,
                     generator=torch.Generator().manual_seed(4321))
    return train, val


def tensor_digest(t) -> str:
    t = t.detach().cpu().contiguous()
    return sha_bytes(f"{t.dtype}{tuple(t.shape)}".encode() + t.numpy().tobytes())


def optimizer_digest(opt) -> str:
    """sha256 over the optimizer's state_dict: its groups (lr, initial_lr, momentum, ...) and every momentum buffer."""
    import torch
    sd = opt.state_dict()
    h = hashlib.sha256(json.dumps(canon(sd["param_groups"]), sort_keys=True).encode())
    for idx in sorted(sd["state"]):
        for k in sorted(sd["state"][idx]):
            v = sd["state"][idx][k]
            h.update(f"{idx}:{k}:".encode())
            h.update(tensor_digest(v).encode() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def grab_hooks(QATHooks):
    class Grab(QATHooks):
        """Smoke-only: keeps the run's prepared student and optimizer (run_qat's own QATHooks interface)."""
        prepared = optimizer = scheduler = run_meta = None

        def on_start(self, prepared, optimizer, scheduler, run_meta):
            self.prepared, self.optimizer, self.scheduler, self.run_meta = prepared, optimizer, scheduler, run_meta
    return Grab()


def run_record(out: Path, grab, Q) -> dict:
    rows = read_rows(out)
    return {"train": pick(rows, "train", TRAIN_KEYS + TERM_KEYS), "ends": pick(rows, "epoch_end", END_KEYS),
            "vals": pick(rows, "val", VAL_KEYS),
            "teacher_states": [r.get("teacher_state_sha256") for r in rows if r.get("event") == "epoch_end"],
            "step1": [r.get("checks") for r in rows if r.get("event") == "step1_checks"],
            "model_digest": Q.state_digest(grab.prepared), "optimizer_digest": optimizer_digest(grab.optimizer)}


def term_bits(total, parts) -> dict:
    return {"total": float(total).hex(), "total_tensor": tensor_digest(total), "dtype": str(total.dtype),
            "logit_kd": parts["logit_kd"].hex(), "keys": sorted(parts)}


# ------------------------------------------------------------------ the --am29-root process
def ref_main(argv) -> int:
    """L-AM29's side: imports src from --am29-root only, runs run_am29_qat (arm T1) and t_logit_kd, writes JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args(argv)
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    root = Path(task["am29_root"]).resolve()
    sys.path.insert(0, str(root))
    os.environ["PLANTSEG_DATA_ROOT"] = str(Path(task["work"]) / "absent_data_root")  # as smoke_am29_runner sets it
    import torch
    torch.set_num_threads(int(task["threads"]))
    from src.distill import teacher as TT
    from src.distill.features import StudentTaps
    from src.models.student import build_student
    from src.quant import am29 as M
    from src.quant import am29_registry as R
    from src.quant import qat as Q
    R.register()
    init = torch.load(task["init"])
    out = {"am29_file": M.__file__, "qat_file": Q.__file__,
           "am29_sha256": sha_bytes(Path(M.__file__).read_bytes()), "am29_blob": blob_id(Path(M.__file__).read_bytes()),
           "torch": torch.__version__, "threads": torch.get_num_threads()}

    def fresh():
        m = build_student(num_classes=116, pretrained=False)
        m.load_state_dict(init["state"])
        return m
    # B: run_am29_qat, arm T1, on the shared state and loaders
    m = fresh()
    out["init_digest"] = Q.state_digest(m)
    teacher = M.SelfTeacher(m, checkpoint_sha256=task["teacher_sha256"])
    grab = grab_hooks(Q.QATHooks)
    run_dir = Path(task["work"]) / "ref_t1" / "x-ref-t1"
    summ = M.run_am29_qat(arm="T1", mode="smoke", model=m, source_meta={"sha256": task["source_sha256"], "path": None},
                          out_dir=run_dir, seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu",
                          num_workers=0, lr=3e-4, factor=0.5, e3_lambda=0.5, teacher=teacher,
                          loaders=make_loaders(init), max_val_batches=1, hooks=grab, log=quiet)
    out["B"] = {**run_record(run_dir, grab, Q), "summary": summ, "teacher_end": teacher.state_sha256(),
                "stage": read_rows(run_dir)[0].get("stage")}
    # C1: t_logit_kd on the tensors the main process computed (N2R2T's path, sites on)
    ti = torch.load(task["term_inputs"])
    tout = TT.TeacherOutput(logits=ti["teacher_logits"], feat_s16=None)
    total, parts = M.t_logit_kd(logits=ti["logits"], head_logits=ti["head_logits"], c5=ti["c5"], mask=ti["mask"],
                                teacher_out=tout, weight=0.25)
    out["C1"] = term_bits(total, parts)
    # C2: the same state through this checkout's forward (no sites): SelfTeacher, prepare_qat_model, StudentTaps
    m2 = fresh()
    t2 = M.SelfTeacher(m2, checkpoint_sha256=task["teacher_sha256"])
    p2 = M.prepare_qat_model(m2)
    p2.train()
    with StudentTaps(p2) as taps:
        logits = p2(init["c_img"])
        c5, head = taps.require()
    tout2 = t2(init["c_img"])
    total2, parts2 = M.t_logit_kd(logits=logits, head_logits=head, c5=c5, mask=init["c_mask"], teacher_out=tout2,
                                  weight=0.25)
    out["C2"] = {**term_bits(total2, parts2), "head": tensor_digest(head), "teacher": tensor_digest(tout2.logits)}
    mods = sorted({str(getattr(v, "__file__", "") or "") for k, v in list(sys.modules.items())
                   if k == "src" or k.startswith(("src.", "configs"))})
    out["module_files"] = mods
    Path(task["result"]).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ this checkout's process
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--am29-root", default=None)
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--no-mutations", action="store_true")
    a = ap.parse_args(argv)
    if not a.am29_root:
        print("RESULT: SMOKE_EXPLORE_N2R2T FAIL (0/1) failed: ['--am29-root is required (checks B and C run L-AM29's "
              "code at 2a94019f from it)']")
        return 1
    sys.path.insert(0, str(REPO))
    sections = tuple(s for s in a.sections.split(",") if s)
    env = Env(Path(a.am29_root).resolve())
    print("[smoke_explore_n2r2t] cases")
    c = run_cases(env, sections, verbose=True, full=True)
    if not a.no_mutations:
        print("[smoke_explore_n2r2t] mutations")
        mutations(env, c)
    return c.result()


class Env:
    """The modules and fixtures of this checkout, imported once."""

    def __init__(self, am29_root: Path):
        import importlib.util
        import torch
        from src.explore import guard as G
        from src.explore import n2r2t as T
        from src.explore import qat_explore as X
        from src.explore import smokekit as SK
        from src.quant import qat as Q
        self.torch, self.G, self.T, self.X, self.SK, self.Q = torch, G, T, X, SK, Q
        self.am29_root = am29_root

        def load(path: Path, name: str):
            spec = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = m
            spec.loader.exec_module(m)
            return m
        self.N = load(REPO / "scripts" / "smoke_explore_n2r2.py", "x_smoke_n2r2_fixtures")  # N2R2's fixtures, unchanged
        self.BB = self.N.blockb()                     # scripts/explore_qat_blockb.py, the module N2R2's smoke drives
        self.EQ = load(REPO / "scripts" / "explore_qat.py", "x_smoke_explore_qat_cli")
        self.work = SK.scratch_dir("x_n2r2t_")
        self.threads = torch.get_num_threads()
        self.cache: dict = {}

    def init(self) -> dict:
        """One state file: a random student (scripts/smoke_explore_n2r2.py's fixture state, seed 7) and the loaders'
        tensors; both processes start from it."""
        if "init" not in self.cache:
            torch = self.torch
            g = torch.Generator().manual_seed(2026)
            train_x = torch.randn(4, 3, 64, 64, generator=g)
            train_y = torch.randint(0, 116, (4, 64, 64), generator=g)
            train_y[:, :4, :] = 255
            val_x = torch.randn(2, 3, 64, 64, generator=g)
            val_y = torch.randint(0, 116, (2, 64, 64), generator=g)
            init = {"state": self.N.student_state(), "train_x": train_x, "train_y": train_y, "val_x": val_x,
                    "val_y": val_y, "c_img": train_x[:2].clone(), "c_mask": train_y[:2].clone()}
            path = self.work / "init.pt"
            torch.save(init, path)
            self.cache["init"] = (init, path)
        return self.cache["init"]

    def fresh_student(self):
        from src.models.student import build_student
        init, _ = self.init()
        m = build_student(num_classes=116, pretrained=False)
        m.load_state_dict(init["state"])
        return m

    def parent(self) -> dict:
        """N2R2's parent check on the fixture E3-R2 run (DL-136's table with the fixture's hashes)."""
        if "parent" not in self.cache:
            N, X = self.N, self.X
            fx = N.fixtures()
            with self.SK.patched(X, "E3R2_PARENT_OF_RECORD", N.table_of(fx["d"])):
                self.cache["parent"] = (fx, X.n2r2_parent(fx["d"], **N.n2r2_kw(fx)))
        return self.cache["parent"]


def run_pair(env: Env, *, steps, factor: float, tag: str) -> dict:
    """N2R2's path (run_n2r2, the pinned run_qat with N2's change) and N2R2T (run_n2r2t at `factor`), each from the
    parent loaded through E6-R2's loader, on fresh copies of the same loaders, both with the ten sites and TF32 off."""
    Q, X, T, N, SK = env.Q, env.X, env.T, env.N, env.SK
    fx, got = env.parent()
    init, _ = env.init()
    mA, metaA = Q.load_source(got["stage"], got["parent"])
    mB, metaB = Q.load_source(got["stage"], got["parent"])
    states = (Q.state_digest(mA), Q.state_digest(mB))
    teacher = T.SelfTeacher(mB, checkpoint_sha256=metaB["sha256"])
    gA, gB = grab_hooks(Q.QATHooks), grab_hooks(Q.QATHooks)
    outA = SK.scratch_dir(f"x_n2r2t_{tag}_n2r2_") / "x-n2r2-s42"
    outB = SK.scratch_dir(f"x_n2r2t_{tag}_n2r2t_") / "x-n2r2t-s42"
    pinned = Q.run_qat
    before = N.tf32_flags()
    try:
        with SK.patched(Q, "run_qat", lambda **kw: pinned(**kw, hooks=gA)):
            resA = X.run_n2r2(mode="smoke", model=mA, source_meta=metaA, out_dir=outA, seed=42, clip_norm=5.0,
                              clip_source="smoke", device="cpu", num_workers=0, parent=got["parent"],
                              extra_meta={"am23_parent": got["am23_binding"], "x_item": "N2R2"}, sites=True,
                              loaders=make_loaders(init), max_steps=steps, max_val_batches=1, log=quiet)
        resB = T.run_n2r2t(mode="smoke", model=mB, teacher=teacher, source_meta=metaB, out_dir=outB, seed=42,
                           clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, parent=got["parent"],
                           factor=factor, e3_lambda=0.5, extra_meta={"am23_parent": got["am23_binding"],
                                                                     "x_item": "N2R2T"},
                           sites=True, loaders=make_loaders(init), hooks=gB, max_steps=steps, max_val_batches=1,
                           log=quiet)
    finally:
        N.tf32_restore(before)
    return {"A": run_record(outA, gA, Q), "B": run_record(outB, gB, Q), "resA": resA, "resB": resB,
            "metaA": read_rows(outA)[0], "metaB": read_rows(outB)[0], "outA": outA, "outB": outB,
            "load_digests": (metaA["sha256"], metaB["sha256"]), "states": states, "teacher": teacher,
            "rowsB": read_rows(outB), "rowsA": read_rows(outA)}


def f32_sum(sup: float, weight: float, term: float) -> float:
    """CE + Dice + weight x term in float32, as the loop adds them: sup + (0 + 1.0 * weight * term)."""
    import torch
    t = torch.tensor(term, dtype=torch.float32)
    distill = torch.zeros((), dtype=torch.float32) + (1.0 * weight) * t
    return float(torch.tensor(sup, dtype=torch.float32) + distill)


# ------------------------------------------------------------------ the cases
def run_cases(env: Env, sections, *, verbose: bool, full: bool):
    c = env.SK.Cases("SMOKE_EXPLORE_N2R2T", verbose=verbose)
    for s in SECTIONS:
        if s in sections:
            {"copies": sec_copies, "A": sec_a, "B": sec_b, "C": sec_c, "E": sec_e, "F": sec_f}[s](env, c, full)
    return c


def sec_copies(env: Env, c, full: bool) -> None:
    T, X, G, Q = env.T, env.X, env.G, env.Q
    from src.training import train_distill as TD
    am29_file = env.am29_root / "src" / "quant" / "am29.py"
    data = am29_file.read_bytes() if am29_file.is_file() else b""
    c.check("copies: --am29-root's src/quant/am29.py is 2a94019f's (sha256 and git blob id as recorded)",
            sha_bytes(data) == T.AM29_FILE_SHA256 and blob_id(data) == T.AM29_FILE_BLOB,
            f"sha256 {sha_bytes(data)[:16]}..., blob {blob_id(data)[:12]}...")
    lines = data.decode("utf-8").splitlines(keepends=True)
    mine = Path(T.__file__).read_text(encoding="utf-8")
    objs = {"SelfTeacher": T.SelfTeacher, "t_logit_kd": T.t_logit_kd, "_refuse": T._refuse, "_is_float": T._is_float}
    for name, rec in T.COPIED.items():
        a, b = rec["lines"]
        text = "".join(lines[a - 1:b])
        ok = sha_bytes(text.encode("utf-8")) == rec["sha256"] and rec["commit"] == T.AM29_COMMIT
        if rec["body"] == "unchanged":
            ok = ok and text in mine and (name not in objs or inspect.getsource(objs[name]) == text)
        c.check(f"copies: {name} = src/quant/am29.py:{a}-{b} at 2a94019f"
                + (", byte for byte in src/explore/n2r2t.py" if rec["body"] == "unchanged" else
                   " (the source of run_n2r2t_qat, by its sha256)"), ok, f"sha256 {rec['sha256'][:16]}...")
    src_am = "".join(lines[T.COPIED["run_am29_qat"]["lines"][0] - 1:T.COPIED["run_am29_qat"]["lines"][1]])
    src_t = inspect.getsource(T.run_n2r2t_qat)
    diff = list(difflib.unified_diff(src_am.splitlines(), src_t.splitlines(), "am29.py run_am29_qat (2a94019f)",
                                     "n2r2t.py run_n2r2t_qat", lineterm="", n=0))
    plus = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    minus = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    (env.work / "run_am29_qat_vs_run_n2r2t_qat.diff").write_text("\n".join(diff) + "\n", encoding="utf-8")
    c.check("copies: run_n2r2t_qat is run_am29_qat with its changes only (unified diff written to the scratch folder)",
            0 < plus and 0 < minus and len(src_t.splitlines()) > 300,
            f"{len(src_am.splitlines())} lines -> {len(src_t.splitlines())}; -{minus} +{plus}")
    c.check("copies: the term's constants are E3's: T 4 (train_distill.T_LOGIT), stage e2's switches, lambda 0.5; "
            "factor 0.5, weight 0.25, no ramp; the rate of record 3e-4",
            T.T_LOGIT == TD.T_LOGIT == 4 and T.E2_SWITCHES == TD.resolve_stage("e2")
            and T.E2_SWITCHES["logit_kd"] is True and not T.E2_SWITCHES["cwd_feat"] and not T.E2_SWITCHES["cwd_logit"]
            and T.E3_LAMBDA_OF_RECORD == 0.5 and T.FACTOR == 0.5 and T.FACTOR * T.E3_LAMBDA_OF_RECORD == 0.25
            and T.T_RAMP is None and Q.LEARNING_RATE == 3e-4,
            f"T {T.T_LOGIT}, switches {[k for k in TD.TERMS if T.E2_SWITCHES[k]]}, weight "
            f"{T.FACTOR * T.E3_LAMBDA_OF_RECORD}, lr {Q.LEARNING_RATE}")
    n2 = {"sites_on_input_grid", "tf32_off"}
    c.check("copies: N2R2T on the allow-lists: guard ITEMS, ITEMS, ITEM_SEEDS (42), ITEM_FLAGS (N2R2's, the same "
            "object), CHANGES, run ID x-n2r2t-s42",
            "N2R2T" in G.ITEMS and "N2R2T" in X.ITEMS and X.ITEM_SEEDS["N2R2T"] == (42,)
            and X.ITEM_FLAGS["N2R2T"] is X.ITEM_FLAGS["N2R2"] is X.ITEM_FLAGS["N2"] and "N2R2T" in X.CHANGES
            and X.N2R2T_RUN_ID == T.RUN_ID == "x-n2r2t-s42"
            and X.RECORDS_ITEMS == {"N2R2": "x-n2r2-s42", "N2R2T": "x-n2r2t-s42"})
    c.check("copies: N2R2T takes exactly N2R2's flags at seed 42; one flag, an added flag or seed 43 refused",
            X.item_flag_error("N2R2T", n2, 42) is None
            and (X.item_flag_error("N2R2T", {"tf32_off"}, 42) or ("",))[0] == "item_flags"
            and (X.item_flag_error("N2R2T", n2 | {"kd_logit"}, 42) or ("",))[0] == "item_flags"
            and (X.item_flag_error("N2R2T", n2, 43) or ("",))[0] == "item_seed")
    base_items = ("C1", "N1", "N2", "N2R2", "N5", "D")
    base_seeds = {"C1": (43, 44), "N1": (42,), "N2": (42,), "N2R2": (42,), "N5": (42,)}
    base_flags = {"C1": set(), "N1": {"kd_logit"}, "N2": n2, "N2R2": n2, "N5": {"fp32_head"}, "D": set()}
    c.check("copies: the existing items' entries are e11073a's (ITEMS order, seeds, flags, guard ITEMS keys)",
            tuple(i for i in X.ITEMS if i != "N2R2T") == base_items
            and {k: v for k, v in X.ITEM_SEEDS.items() if k != "N2R2T"} == base_seeds
            and all(set(X.ITEM_FLAGS[k]) == v for k, v in base_flags.items())
            and sorted(k for k in G.ITEMS if k != "N2R2T") == sorted(["N4", "B1", "C1", "N1", "N2", "N2R2", "N5", "D",
                                                                     "N3", "A1pp", "A2", "KL3"]))


def sec_a(env: Env, c, full: bool) -> None:
    Q, X = env.Q, env.X
    steps = None if full else 2
    p = run_pair(env, steps=steps, factor=0.0, tag="a")
    A, B, rA, rB = p["A"], p["B"], p["resA"], p["resB"]
    n = 30 if full else 2
    sA, sB = rA[0], rB[0]
    c.check(f"A: both runs took {n} steps" + (" (15 epochs x 2), complete" if full else ""),
            sA["steps"] == sB["steps"] == n and (not full or (sA["complete"] and sB["complete"]
                                                              and sA["epochs_completed"] == sB["epochs_completed"] == 15)),
            f"N2R2 {sA['steps']} steps, {sA['epochs_completed']} epochs; N2R2T {sB['steps']}, {sB['epochs_completed']}")
    c.check("A: the same parent through E6-R2's loader: one checkpoint, equal states",
            p["load_digests"][0] == p["load_digests"][1] and p["states"][0] == p["states"][1],
            f"checkpoint {p['load_digests'][0][:12]}, state {p['states'][0][:12]}")
    rowsA = [{k: r[k] for k in TRAIN_KEYS} for r in A["train"]]
    rowsB = [{k: r[k] for k in TRAIN_KEYS} for r in B["train"]]
    diffs = [r["step"] for r, s in zip(rowsA, rowsB) if r != s]
    c.check(f"A: at factor 0, every train row is bitwise N2R2's: loss, CE, Dice, gradient norm, clipped flag, learning "
            f"rate, batch fingerprint, epoch flags ({n} steps)",
            len(rowsA) == len(rowsB) == n and not diffs,
            f"differing steps {diffs}; loss step 1 {A['train'][0]['loss'] if A['train'] else None}, "
            f"step {n} {A['train'][-1]['loss'] if A['train'] else None}")
    c.check("A: N2R2T computed the term at every step, at weight 0 (kd_weight 0.0, logit_kd finite and above 0)",
            all(r["kd_weight"] == canon(0.0) and isinstance(r["logit_kd"], str)
                and float.fromhex(r["logit_kd"]) > 0 for r in B["train"]),
            f"logit_kd step 1 {float.fromhex(B['train'][0]['logit_kd']) if B['train'] else None}")
    c.check("A: the epoch states (state_sha256 of every epoch checkpoint) and the VAL rows are bitwise N2R2's",
            A["ends"] == B["ends"] and A["vals"] == B["vals"] and len(A["ends"]) == (15 if full else 1),
            f"{len(A['ends'])} epochs; last state {A['ends'][-1]['state_sha256'][:12] if A['ends'] else None}")
    c.check("A: the final model state and the optimizer state (groups and momentum buffers) are bitwise N2R2's",
            A["model_digest"] == B["model_digest"] and A["optimizer_digest"] == B["optimizer_digest"],
            f"model {B['model_digest'][:12]}, optimizer {B['optimizer_digest'][:12]}")
    mA, mB = rA[2], rB[2]
    c.check("A: the ten sites were installed and fired in both runs, the same number of times",
            mA is not None and mB is not None and mA.calls == mB.calls and all(v > 0 for v in mB.calls.values())
            and len(mB.calls) == 10, str(None if mB is None else mB.calls))
    metaA, metaB = p["metaA"], p["metaB"]
    differ = sorted(k for k in set(metaA) | set(metaB) if json.dumps(metaA.get(k), sort_keys=True)
                    != json.dumps(metaB.get(k), sort_keys=True))
    want = {"wall_clock", "out_dir", "run_id", "objective", "distillation", "x_term", "x_substitution", "x_item"}
    c.check("A: N2R2T's run_meta is N2R2's but for the run's own labels and the term's fields",
            set(differ) <= want and {"objective", "distillation", "x_term"} <= set(differ)
            and metaB.get("stage") == "E6-R2" and metaB.get("source_stage") == "E3"
            and metaB["x_tf32_set"]["cudnn_allow_tf32"] is False and metaB["tf32"]["cuda_matmul_allow_tf32"] is False
            and metaB["x_substitution"]["original"] == "src.quant.qat.prepare_qat_model"
            and metaB["x_substitution"]["item"] == "N2R2T" and metaB["schema"] == metaA["schema"],
            f"differ {differ}")
    xt = metaB.get("x_term") or {}
    c.check("A: run_meta's x_term names the item, the run ID and the term: weight, T, ramp none, the teacher, the "
            "copies with their sha256",
            xt.get("item") == "N2R2T" and xt.get("run_id") == "x-n2r2t-s42" and xt.get("factor") == 0.0
            and xt.get("e3_lambda_logit") == 0.5 and xt.get("T_logit") == 4 and xt.get("ramp") is None
            and xt.get("lr") == 3e-4 and xt["teacher"]["checkpoint_sha256"] == p["load_digests"][1]
            and set(xt.get("copied", {})) == set(env.T.COPIED)
            and xt["code_files_sha256"].get("src/explore/n2r2t.py") == env.G.sha256_file(env.T.__file__),
            str({k: xt.get(k) for k in ("item", "run_id", "factor", "kd_weight", "T_logit", "ramp", "lr")}))
    s1 = B["step1"][0] if B["step1"] else {}
    c.check("A: N2R2T's step-1 checks pass: the term alone, the head grid, the teacher frozen, outside the optimizer and "
            "fed the student's input batch",
            bool(s1) and all(s1.values()) and s1.get("teacher_same_input") is True, str(s1))
    c.check("A: the teacher's state is the parent's at every epoch end (frozen)",
            len(set(B["teacher_states"])) == 1 and B["teacher_states"][0] == p["teacher"].state_sha256()
            and B["teacher_states"][0] == Q.state_digest(env.Q.load_source(env.parent()[1]["stage"],
                                                                            env.parent()[1]["parent"])[0]))
    if full:
        rows = [r for r in p["rowsB"] if r.get("event") == "train"]
        ends = [r for r in p["rowsB"] if r.get("event") == "epoch_end"]
        import math
        t_max = 30
        cos = [3e-4 * (1 + math.cos(math.pi * k / t_max)) / 2 for k in range(t_max)]
        c.check("A: the recipe's schedule ran: lr 3e-4 at step 1 on the cosine of record (T_max 30), BN frozen from "
                "epoch 11, observers off from epoch 13, 15 epoch checkpoints",
                rows[0]["lr"] == 3e-4 and all(math.isclose(r["lr"], cos[i], rel_tol=1e-9, abs_tol=1e-15)
                                               for i, r in enumerate(rows))
                and all(r["bn_frozen"] == (r["epoch"] >= 11) for r in rows)
                and all(r["observers_enabled"] == (r["epoch"] <= 12) for r in rows)
                and [e["epoch"] for e in ends] == list(range(1, 16)) and ends[-1]["run_complete"] is True,
                f"lr {rows[0]['lr']} .. {rows[-1]['lr']:.3e}; run_meta T_max {p['metaB']['scheduler']['T_max']}")
    # the comparison discriminates: at factor 0.5 the loss is CE + Dice + 0.25 x the term from step 1
    q = run_pair(env, steps=2, factor=0.5, tag="a5")
    tA, tB = q["A"]["train"], q["B"]["train"]
    r1 = tB[0] if tB else {}
    sup, term = float.fromhex(r1.get("sup", "0x0p+0")), float.fromhex(r1.get("logit_kd", "0x0p+0"))
    c.check("A: at factor 0.5 the first loss differs from N2R2's: the comparison discriminates",
            bool(tA) and bool(tB) and tA[0]["loss"] != tB[0]["loss"] and tA[0]["batch_sha256"] == tB[0]["batch_sha256"],
            f"{float.fromhex(tA[0]['loss']) if tA else None!r} vs {float.fromhex(tB[0]['loss']) if tB else None!r}")
    c.check("A: at factor 0.5, step 1's loss is CE + Dice + 0.25 x logit_kd, in float32, bitwise (no ramp); its CE + "
            "Dice is N2R2's loss",
            bool(r1) and r1.get("kd_weight") == canon(0.25) and term > 0
            and float.fromhex(r1["loss"]) == f32_sum(sup, 0.25, term) and r1.get("sup") == tA[0]["loss"],
            f"loss {float.fromhex(r1['loss']) if r1 else None!r}, sup {sup!r}, logit_kd {term!r}")
    c.check("A: N2R2T's run is stage E6-R2 (source E3), its epoch checkpoint too",
            q["metaB"].get("stage") == "E6-R2" and q["metaB"].get("source_stage") == "E3"
            and env.torch.load(q["outB"] / "epoch_ckpts" / "e01.pt")["stage"] == "E6-R2")
    for d in (p["outA"], p["outB"], q["outA"], q["outB"]):               # the epoch checkpoints are large; rows stay
        for f in sorted((d / "epoch_ckpts").glob("e*.pt")):
            f.unlink()


def ref_run(env: Env, term_inputs: Path) -> dict:
    """The --am29-root process, once: B's run_am29_qat (arm T1) and C's t_logit_kd."""
    if "ref" not in env.cache:
        _, init_path = env.init()
        task = {"am29_root": str(env.am29_root), "work": str(env.work), "init": str(init_path), "threads": env.threads,
                "teacher_sha256": "ab" * 32, "source_sha256": "cd" * 32, "term_inputs": str(term_inputs),
                "result": str(env.work / "ref_result.json")}
        tp = env.work / "ref_task.json"
        tp.write_text(json.dumps(task, indent=1), encoding="utf-8")
        envv = {k: v for k, v in os.environ.items() if k not in ("PLANTSEG_DATA_ROOT", "PYTHONPATH")}
        p = subprocess.run([sys.executable, "-B", "-I", str(HERE), "--ref", "--task", str(tp)], capture_output=True,
                           text=True, env=envv)
        res = json.loads(Path(task["result"]).read_text(encoding="utf-8")) if p.returncode == 0 else {}
        env.cache["ref"] = {"exit": p.returncode, "err": (p.stderr or "")[-600:], **res}
    return env.cache["ref"]


def c_inputs(env: Env) -> tuple[dict, Path]:
    """C's fixed batch through N2R2T's path: the student from the state file, prepared by the pinned prepare with the
    ten sites installed (N2's change), a train-mode forward under StudentTaps; the teacher a SelfTeacher of it."""
    if "c_inputs" not in env.cache:
        torch, T, Q = env.torch, env.T, env.Q
        from src.distill.features import StudentTaps
        from src.explore.sites import SITE_NAMES, SiteMatcher
        init, _ = env.init()
        m = env.fresh_student()
        rng0 = torch.get_rng_state()
        teacher = T.SelfTeacher(m, checkpoint_sha256="ab" * 32)
        rng1 = torch.get_rng_state()
        prepared = Q.prepare_qat_model(m)
        matcher = SiteMatcher(prepared, SITE_NAMES)
        prepared.train()
        with StudentTaps(prepared) as taps:
            logits = prepared(init["c_img"])
            c5, head = taps.require()
        tout = teacher(init["c_img"])
        total, parts = T.t_logit_kd(logits=logits, head_logits=head, c5=c5, mask=init["c_mask"], teacher_out=tout,
                                    weight=T.FACTOR * T.E3_LAMBDA_OF_RECORD)
        ti = {"logits": logits.detach().clone(), "head_logits": head.detach().clone(), "c5": c5.detach().clone(),
              "mask": init["c_mask"].clone(), "teacher_logits": tout.logits.detach().clone()}
        path = env.work / "term_inputs.pt"
        torch.save(ti, path)
        env.cache["c_inputs"] = ({"total": total, "parts": parts, "ti": ti, "prepared": prepared, "teacher": teacher,
                                  "model": m, "rng_same": bool(torch.equal(rng0, rng1)), "matcher": matcher}, path)
    return env.cache["c_inputs"]


def sec_b(env: Env, c, full: bool) -> None:
    if not full:
        return
    Q, T = env.Q, env.T
    _, tpath = c_inputs(env)
    ref = ref_run(env, tpath)
    c.check("B: L-AM29's process ran from --am29-root at 2a94019f and loaded no module of this checkout",
            ref.get("exit") == 0 and ref.get("am29_sha256") == T.AM29_FILE_SHA256
            and ref.get("am29_blob") == T.AM29_FILE_BLOB and str(ref.get("am29_file", "")).startswith(str(env.am29_root))
            and all(not f.startswith(str(REPO) + os.sep) for f in ref.get("module_files", ["?"])),
            f"exit {ref.get('exit')}; {ref.get('am29_file')}; {len(ref.get('module_files', []))} src/configs modules "
            f"{ref.get('err', '')[-200:]}")
    same = [p for p in SHARED if (REPO / p).is_file() and (env.am29_root / p).is_file()
            and env.G.sha256_file(REPO / p) == env.G.sha256_file(env.am29_root / p)]
    c.check("B: the modules both loops import are byte-identical in the two checkouts (only the loops differ)",
            len(same) == len(SHARED), f"{len(same)}/{len(SHARED)} identical; differ {sorted(set(SHARED) - set(same))}")
    init, _ = env.init()
    m = env.fresh_student()
    teacher = T.SelfTeacher(m, checkpoint_sha256="ab" * 32)
    grab = grab_hooks(Q.QATHooks)
    out = env.SK.scratch_dir("x_n2r2t_b_") / "x-n2r2t-nosites"
    summ, sub, matcher = T.run_n2r2t(mode="smoke", model=m, teacher=teacher, source_meta={"sha256": "cd" * 32,
                                                                                          "path": None},
                                     out_dir=out, seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu",
                                     num_workers=0, parent=None, factor=0.5, e3_lambda=0.5, sites=False,
                                     loaders=make_loaders(init), hooks=grab, max_val_batches=1, log=quiet)
    mine = run_record(out, grab, Q)
    rb = ref.get("B") or {}
    meta = read_rows(out)[0]
    c.check("B: N2R2T with the substitution and TF32 change off: no substitution, no site, TF32 at the torch defaults",
            sub is None and matcher is None and "x_substitution" not in meta and "x_tf32_set" not in meta
            and Q.tf32_error(meta["tf32"]) is None, str(meta["tf32"]))
    c.check("B: both runs start from one in-memory state (the state file) and run 30 steps, 15 epochs",
            ref.get("init_digest") == Q.state_digest(env.fresh_student()) and summ["steps"] == 30
            and (rb.get("summary") or {}).get("steps") == 30 and summ["complete"]
            and (rb.get("summary") or {}).get("complete") is True,
            f"init {str(ref.get('init_digest'))[:12]}; steps {summ['steps']} vs {(rb.get('summary') or {}).get('steps')}")
    d_rows = [r["step"] for r, s in zip(mine["train"], rb.get("train", [])) if r != s]
    c.check("B: every train row is bitwise run_am29_qat T1's: loss, CE, Dice, CE + Dice, the term, its weight, gradient "
            "norm, clipped flag, learning rate, batch fingerprint (30 steps)",
            len(mine["train"]) == len(rb.get("train", [])) == 30 and not d_rows,
            f"differing steps {d_rows}; loss step 30 {mine['train'][-1]['loss'] if mine['train'] else None}")
    c.check("B: the 15 epoch states, the VAL rows and the teacher's state at every epoch end are bitwise T1's",
            mine["ends"] == rb.get("ends") and mine["vals"] == rb.get("vals")
            and mine["teacher_states"] == rb.get("teacher_states") and len(mine["ends"]) == 15)
    c.check("B: the final model state and optimizer state are bitwise T1's",
            mine["model_digest"] == rb.get("model_digest") and mine["optimizer_digest"] == rb.get("optimizer_digest"),
            f"model {mine['model_digest'][:12]} vs {str(rb.get('model_digest'))[:12]}; optimizer "
            f"{mine['optimizer_digest'][:12]} vs {str(rb.get('optimizer_digest'))[:12]}")
    c.check("B: the step-1 checks are T1's and pass; T1's stage is AM29-T1, N2R2T's E6-R2",
            mine["step1"] == rb.get("step1") and all((mine["step1"] or [{"x": False}])[0].values())
            and rb.get("stage") == "AM29-T1" and meta.get("stage") == "E6-R2")
    for f in sorted((out / "epoch_ckpts").glob("e*.pt")):
        f.unlink()


def sec_c(env: Env, c, full: bool) -> None:
    torch, T, Q = env.torch, env.T, env.Q
    from src.distill.features import StudentTaps
    from src.training.losses import downsample_validity, logit_kd_kl
    from src.training import train_distill as TD
    ci, tpath = c_inputs(env)
    total, parts, ti = ci["total"], ci["parts"], ci["ti"]
    valid = downsample_validity(ti["mask"], ti["head_logits"].shape[-2:], ignore_index=255)
    direct = 0.25 * logit_kd_kl(ti["head_logits"], ti["teacher_logits"], valid, T=TD.T_LOGIT)
    c.check("C: the term (N2R2T's path, sites on) is 0.25 x losses.logit_kd_kl(head logits, teacher logits, "
            "downsample_validity(target, that grid), T=4), to the last bit",
            tensor_digest(total) == tensor_digest(direct) and float(total).hex() == float(direct).hex()
            and parts["logit_kd"] == float(direct / 0.25),
            f"term {float(total).hex()} ({float(total)!r}); unweighted {parts['logit_kd']!r}")
    p = ci["prepared"]
    ff = p.head.ff_add.activation_post_process
    from src.explore.sites import _off_grid_fraction
    frac = _off_grid_fraction(ti["head_logits"], float(ff.scale), int(ff.zero_point))
    c.check("C: the student's logits are the head_logits tap of the prepared student, on head.ff_add's fake-quant "
            "grid (a fake-quant tap); the teacher's on the same OS8 grid",
            bool(ff.fake_quant_enabled[0]) and frac == 0.0
            and tuple(ti["head_logits"].shape) == tuple(ti["teacher_logits"].shape) == (2, 116, 8, 8),
            f"off-grid fraction {frac}; shapes {tuple(ti['head_logits'].shape)}")
    tch, m = ci["teacher"], ci["model"]
    before = tch.student.training
    tch.train(True)
    c.check("C: the teacher is a frozen FP32 copy of the same state: eval mode (also after train()), no gradient, no "
            "fake-quant, its state the parent's; building it drew nothing from the global random stream",
            ci["rng_same"] and not before and not tch.student.training
            and all(not q.requires_grad for q in tch.parameters()) and not tch.trainable_parameters()
            and not Q.fake_quant_modules(tch.student) and Q.state_digest(tch.student) == Q.state_digest(m)
            and tch.state_sha256() == Q.state_digest(env.fresh_student()))
    if not full:
        return
    ref = ref_run(env, tpath)
    c.check("C: on the same tensors, src/quant/am29.py t_logit_kd (2a94019f, its own process) gives the same term to "
            "the last bit",
            ref.get("exit") == 0 and ref.get("C1") == term_bits(total, parts),
            f"{(ref.get('C1') or {}).get('total')} vs {term_bits(total, parts)['total']}")
    m2 = env.fresh_student()
    t2 = T.SelfTeacher(m2, checkpoint_sha256="ab" * 32)
    p2 = Q.prepare_qat_model(m2)
    p2.train()
    init, _ = env.init()
    with StudentTaps(p2) as taps:
        logits = p2(init["c_img"])
        c5, head = taps.require()
    tout2 = t2(init["c_img"])
    total2, parts2 = T.t_logit_kd(logits=logits, head_logits=head, c5=c5, mask=init["c_mask"], teacher_out=tout2,
                                  weight=0.25)
    mine2 = {**term_bits(total2, parts2), "head": tensor_digest(head), "teacher": tensor_digest(tout2.logits)}
    c.check("C: from one state through each checkout's own SelfTeacher, prepare and forward (no sites), the term is the "
            "same to the last bit, and so are the head and teacher logits",
            ref.get("C2") == mine2, f"{(ref.get('C2') or {}).get('total')} vs {mine2['total']}")


def sec_e(env: Env, c, full: bool) -> None:
    T, Q, X, N, SK, G = env.T, env.Q, env.X, env.N, env.SK, env.G
    init, _ = env.init()
    work = SK.scratch_dir("x_n2r2t_e_")
    clipdir = work / "clipdir"                       # the clip record's folder (an input), apart from the outputs
    clipdir.mkdir()
    count = iter(range(10 ** 6))

    def real(**over):
        m = env.fresh_student()
        kw = dict(mode="real", model=m, teacher=T.SelfTeacher(m, checkpoint_sha256="ab" * 32),
                  source_meta={"sha256": "cd" * 32, "path": None}, out_dir=work / f"r{next(count)}" / "x-r",
                  seed=42, clip_norm=5.0, clip_source="clip_selection", device="cpu", num_workers=1, parent=None,
                  factor=0.5, e3_lambda=0.5)
        kw.update(over)
        return kw
    trains = T.SelfTeacher(env.fresh_student())
    for q in trains.student.parameters():
        q.requires_grad_(True)
    cases = (("a test hook (QATHooks)", "real_run_test_hooks", {"hooks": Q.QATHooks()}),
             ("injected loaders", "real_run_test_hooks", {"loaders": make_loaders(init)}),
             ("a step cap", "real_run_test_hooks", {"max_steps": 1}),
             ("NaN injection", "real_run_test_hooks", {"inject_nan_at_steps": (1,)}),
             ("a VAL cap", "real_run_test_hooks", {"max_val_batches": 1}),
             ("factor 0.0", "n2r2t_factor", {"factor": 0.0}),
             ("factor 0.25", "n2r2t_factor", {"factor": 0.25}),
             ("factor 1.0 (T3's)", "n2r2t_factor", {"factor": 1.0}),
             ("factor 0.5000001", "n2r2t_factor", {"factor": 0.5000001}),
             ("no factor", "weight_invalid", {"factor": None}),
             ("E3 lambda 1.0", "e3_lambda", {"e3_lambda": 1.0}),
             ("seed 43", "n2r2t_seed", {"seed": 43}),
             ("the change removed (sites False)", "n2r2t_change_required", {"sites": False}),
             ("a teacher that is not a SelfTeacher", "teacher_not_frozen", {"teacher": None}),
             ("a teacher with trainable parameters", "teacher_not_frozen", {"teacher": trains}),
             ("another clip source", "clip_source", {"clip_source": "smoke"}))
    before = N.tf32_flags()
    try:
        for name, code, over in cases:
            kw = real(**over)
            c.refuses(f"E: a real N2R2T run refuses {name} [{code}]", code, T.run_n2r2t, **kw)
            c.check(f"E: ... nothing written, the pinned prepare_qat_model restored ({name})",
                    not Path(kw["out_dir"]).exists() and Q.prepare_qat_model is N.PINNED_PREPARE)
        c.refuses("E: the loop itself (run_n2r2t_qat) refuses a step cap in a real run", "real_run_test_hooks",
                  T.run_n2r2t_qat, **{k: v for k, v in real(max_steps=1).items() if k != "sites"})
        c.refuses("E: the loop itself refuses factor 1.0 in a real run", "n2r2t_factor", T.run_n2r2t_qat,
                  **real(factor=1.0))
        c.refuses("E: with every value of record, a real run on the CPU stops at the CUDA gate (the values pass)",
                  "cuda_required", T.run_n2r2t, **real())
    finally:
        N.tf32_restore(before)
    if not full:
        return
    # the CLI's N2R2T gates in process, up to the run call (no data root and no CUDA here: those two gates and the clip
    # file's hash, DL-139's, are stood in for; N2R2's parent check, census and teacher are the code's own)
    fx, got = env.parent()
    head = N._git(REPO, "rev-parse", "HEAD").strip()
    calls, seen = {}, {}
    iso = {"checked": True, "data_root": "<synthetic>", "counts": {"train": 5367, "val": 846}}

    def clip_stub(stage_name, seed, clip, **kw):
        calls["clip_stage"] = stage_name
        return {"clip_source": "clip_selection", "u4_pilot": False, "clip_selection": {"winner": clip, "stub": True}}

    def run_stub(**kw):
        seen.update(kw)
        raise X.ExploreQATError("smoke_recorder", "the smoke's recorder stands in for the run")
    table = N.table_of(fx["d"])
    rec = [*fx_records(N, fx), "--run-id", "x-n2r2t-s42", "--expect-head", head]
    out_dir = work / "cli" / "x-n2r2t-s42"
    argv = ["--real-run", "--confirm-real-run", "--item", "N2R2T", "--seed", "42", "--source-run-dir", str(fx["d"]),
            "--expect-source-sha256", G.sha256_file(fx["d"] / N.REC["checkpoint"]), "--clip-selection",
            str(clipdir / "clip.json"), "--clip-selection-sha256", X.CLIP_SELECTION_SHA256, "--num-workers", "12",
            "--sites-on-input-grid", "--tf32-off", "--out-dir", str(out_dir), *rec]
    buf = io.StringIO()
    with SK.patched(G, "check_data_root", lambda records_only=False: dict(iso)), \
            SK.patched(env.torch.cuda, "is_available", lambda: True), SK.patched(Q, "clip_binding", clip_stub), \
            SK.patched(X, "E3R2_PARENT_OF_RECORD", table), SK.patched(T, "run_n2r2t", run_stub), \
            contextlib.redirect_stdout(buf):
        code = env.EQ.main(argv)
    text = buf.getvalue()
    xm = json.loads((out_dir / G.RUN_META_NAME).read_text(encoding="utf-8")) if (out_dir / G.RUN_META_NAME).is_file() \
        else {}
    tch = seen.get("teacher")
    loaded = Q.load_source(got["stage"], got["parent"])[0]
    c.check("E: the CLI's N2R2T gates pass in process up to the run call: E6-R2's clip binding, N2R2's parent check, "
            "the site census, then run_n2r2t called in real mode",
            code == 2 and "[smoke_recorder]" in text and calls.get("clip_stage") == "E6-R2"
            and seen.get("mode") == "real" and "sites" not in seen, text.strip()[-200:])
    c.check("E: ... with factor 0.5, E3's lambda 0.5 from the λ selection the binding read, and a SelfTeacher that is "
            "a frozen copy of the loaded parent",
            seen.get("factor") == 0.5 and seen.get("e3_lambda") == 0.5 and isinstance(tch, T.SelfTeacher)
            and tch.state_sha256() == Q.state_digest(loaded) and not tch.trainable_parameters()
            and tch.checkpoint_sha256 == got["parent"]["checkpoint_sha256"])
    ex = seen.get("extra_meta") or {}

    def same(a, b) -> bool:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c.check("E: ... run_meta's extra fields: x_item N2R2T, its change, the AM-23 binding, the parent of record and "
            "the records checkout, the x_run_meta.json sha256",
            ex.get("x_item") == "N2R2T" and ex.get("x_change") == X.CHANGES["N2R2T"]
            and same(ex.get("am23_parent"), got["am23_binding"]) and same(ex.get("x_parent_of_record"), table)
            and (ex.get("x_records_root") or {}).get("commit") == fx["head"]
            and ex.get("x_run_meta_sha256") == G.sha256_file(out_dir / G.RUN_META_NAME))
    c.check("E: ... x_run_meta.json: item N2R2T, run ID x-n2r2t-s42, E6-R2's recipe with the term, the teacher record, "
            "the census, the term's code pinned and clean at HEAD",
            xm.get("item") == "N2R2T" and xm.get("run_id") == "x-n2r2t-s42" and xm.get("quant_stage") == "E6-R2"
            and (xm.get("recipe") or {}).get("recipe_of_record") == X.RECIPE_OF_RECORD_N2R2
            and ((xm.get("recipe") or {}).get("term") or {}).get("kd_weight") == 0.25
            and (xm.get("teacher_record") or {}).get("kind", "").startswith("frozen FP32 copy")
            and (xm.get("site_census") or {}).get("matches") is True
            and "src/training/train_distill.py" in (xm.get("pinned_files_sha256") or {})
            and xm.get("pinned_clean_at_head") is True and xm.get("lane_clean_at_head") is True
            and xm.get("precision_label") == "int8 QNNPACK (as E6-R2)" and same(xm.get("am23_parent"), got["am23_binding"]))
    # the CLI's refusals (subprocesses: no data root, no CUDA in this session)
    py, cli = sys.executable, str(REPO / "scripts" / "explore_qat.py")
    envv = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    common = ["--real-run", "--confirm-real-run", "--source-run-dir", str(fx["d"]), "--expect-source-sha256",
              N.REC["sha256"], "--clip-selection", str(clipdir / "clip.json"), "--clip-selection-sha256",
              X.CLIP_SELECTION_SHA256, "--num-workers", "12", "--expect-head", "0" * 40]
    recs = fx_records(N, fx)

    def cli_run(*extra):
        return subprocess.run([py, "-I", cli, *common, *extra], capture_output=True, text=True, env=envv)
    n2f = ["--sites-on-input-grid", "--tf32-off"]
    for label, code, args in (
            ("N2R2T at seed 43", "item_seed",
             ["--item", "N2R2T", "--seed", "43", *n2f, *recs, "--run-id", "x-n2r2t-s42"]),
            ("N2R2T under N2R2's run ID", "n2r2_run_id",
             ["--item", "N2R2T", "--seed", "42", *n2f, *recs, "--run-id", "x-n2r2-s42"]),
            ("N2R2T without its records flags", "n2r2_records_required",
             ["--item", "N2R2T", "--seed", "42", *n2f, "--run-id", "x-n2r2t-s42"]),
            ("N2R2T with one of N2R2's two flags", "item_flags",
             ["--item", "N2R2T", "--seed", "42", "--sites-on-input-grid", *recs, "--run-id", "x-n2r2t-s42"]),
            ("N2R2T with a teacher flag", "teacher_not_applicable",
             ["--item", "N2R2T", "--seed", "42", *n2f, *recs, "--run-id", "x-n2r2t-s42", "--teacher-ckpt-sha256",
              "8" * 64]),
            ("N2R2T with --split TEST (first, by name)", "split_test",
             ["--item", "N2R2T", "--seed", "42", *n2f, *recs, "--split", "TE" + "ST", "--run-id", "x-n2r2t-s42"]),
            ("N2R2T with a data root that fails the TRAIN/VAL-only check", "data_root_not_trainval_only",
             ["--item", "N2R2T", "--seed", "42", *n2f, *recs, "--run-id", "x-n2r2t-s42"]),
            ("N2R2 under N2R2T's run ID (N2R2's own run ID unchanged)", "n2r2_run_id",
             ["--item", "N2R2", "--seed", "42", *n2f, *recs, "--run-id", "x-n2r2t-s42"])):
        o = work / f"o{next(count)}" / args[args.index("--run-id") + 1]
        r = cli_run(*args, "--out-dir", str(o))
        c.check(f"E: CLI: {label} refused (exit 2, [{code}]), nothing written",
                r.returncode == 2 and f"[{code}]" in r.stdout and not o.exists(), r.stdout.strip()[-160:])


def fx_records(N, fx: dict) -> list[str]:
    """N2R2's nine records flags for the fixture records checkout (scripts/smoke_explore_n2r2.py's values)."""
    return ["--records-root", str(fx["root"]), "--records-commit", fx["head"], "--lambda-selection", N.LAM_REL,
            "--lambda-selection-sha256", N.LAM_SHA, "--alpha-selection", N.ALPHA_REL, "--alpha-selection-sha256",
            N.ALPHA_SHA, "--condition-record", N.COND_REL, "--condition-sha256", fx["rec_sha"], "--condition-dl",
            N.DL_ID]


def short_run(env: Env, out: Path, *, steps: int = 2) -> None:
    """A two-epoch N2R2T smoke run (one step an epoch) for block B's cases."""
    T, N = env.T, env.N
    m = env.fresh_student()
    before = N.tf32_flags()
    try:
        T.run_n2r2t(mode="smoke", model=m, teacher=T.SelfTeacher(m, checkpoint_sha256="ab" * 32),
                    source_meta={"sha256": "cd" * 32, "path": None}, out_dir=out, seed=42, clip_norm=5.0,
                    clip_source="smoke", device="cpu", num_workers=0, parent=None, factor=0.5, e3_lambda=0.5,
                    loaders=N.one_batch(), max_steps=steps, max_val_batches=1, log=quiet)
    finally:
        N.tf32_restore(before)


def item_dir(out_run: Path, item: str, run_id: str) -> None:
    import json as _j
    from src.explore import guard as G
    (out_run.parent / G.RUN_META_NAME).write_text(_j.dumps({
        "exploratory": True, "item": item, "run_id": run_id, "seeds": [42], "amendment": G.AMENDMENT,
        "precision_label": "int8 QNNPACK (as E6-R2)", "claim_line": G.CLAIM_LINE.format(seeds=42)}), encoding="utf-8")


def sec_f(env: Env, c, full: bool) -> None:
    BB, N, G, Q, SK = env.BB, env.N, env.G, env.Q, env.SK
    work = SK.scratch_dir("x_n2r2t_f_")
    item = work / "x-n2r2t-s42"
    run = item / "x-n2r2t-s42"
    item.mkdir()
    short_run(env, run)
    item_dir(run, "N2R2T", "x-n2r2t-s42")
    doc = c.passes("F: block B accepts an N2R2T run (run_item)", BB.run_item, run)
    c.check("F: ... a registered item of stage E6-R2 with card fix 1, its run of stage E6-R2",
            bool(doc) and doc["item"] == "N2R2T" and "N2R2T" in BB.REGISTERED_ITEMS
            and "N2R2T" in BB.FQOFF_ITEMS + BB.FQOFF_ITEMS_ADD3 and BB.ITEM_STAGE.get("N2R2T") == "E6-R2"
            and BB.run_stage(run) == "E6-R2")
    c.check("F: the existing items' lists are e11073a's: REGISTERED_ITEMS but N2R2T, FQOFF_ITEMS, ITEM_STAGE but N2R2T",
            tuple(i for i in BB.REGISTERED_ITEMS if i != "N2R2T") == ("C1", "N1", "N2", "N2R2", "D")
            and tuple(BB.FQOFF_ITEMS) == ("N2", "N2R2")
            and {k: v for k, v in BB.ITEM_STAGE.items() if k != "N2R2T"} == {"N2": "E6-E2", "N2R2": "E6-R2"})
    unknown = work / "x-z9"
    (unknown / "r").mkdir(parents=True)
    (unknown / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "Z9"}), encoding="utf-8")
    c.refuses("F: block B refuses an unknown item (run_item)", "run_item", BB.run_item, unknown / "r")
    bad = work / "bbs" / "x-n2r2t-s42"
    (bad / "r").mkdir(parents=True)
    (bad / G.RUN_META_NAME).write_text(json.dumps({"exploratory": True, "item": "N2R2T", "seeds": [42]}))
    (bad / "r" / Q.TELEMETRY_NAME).write_text(json.dumps({"event": "run_meta", "stage": "E6-E2"}) + "\n")
    bso = work / "bbs_o" / "x-n2r2t-bbs"
    code2, text2 = N.blockb_main(["--run-dir", str(bad / "r"), "--expect-telemetry-sha256",
                                  G.sha256_file(bad / "r" / Q.TELEMETRY_NAME), "--host-label", "smoke", "--run-id",
                                  "x-n2r2t-bbs", "--out-dir", str(bso)])
    c.check("F: block B refuses an N2R2T item whose run is stage E6-E2 (run_stage), before any step",
            code2 == 2 and "[run_stage]" in text2 and not (bso / "eval").exists(), text2.strip()[-160:])
    tel = G.sha256_file(run / Q.TELEMETRY_NAME)
    bbo = work / "bbo" / "x-n2r2t-bb"
    code, text = N.blockb_main(["--run-dir", str(run), "--expect-telemetry-sha256", tel, "--host-label", "smoke",
                                "--run-id", "x-n2r2t-bb", "--out-dir", str(bbo)])
    xb = json.loads((bbo / "x_blockb.json").read_text(encoding="utf-8")) if (bbo / "x_blockb.json").is_file() else {}
    steps = xb.get("steps") or [{}]
    c.check("F: block B on an N2R2T run runs the pinned scripts: convert, from the checkout, refuses the 2-epoch smoke "
            "run (exit 3); card fix 1 recorded as not run; INCOMPLETE (as N2R2's smoke shows for N2R2)",
            code == 3 and xb.get("item") == "N2R2T" and steps[0].get("step") == "convert"
            and steps[0].get("argv", ["", "", ""])[1:3] == ["scripts/qat_epoch_eval.py", "convert"]
            and steps[0].get("exit") == 3
            and (xb.get("selected_epoch_fake_quant_off_val") or {}).get("status") == "not run"
            and "RESULT: X BLOCK B INCOMPLETE" in text, text.strip()[-200:])
    ev = work / "fq_eval"
    ev.mkdir()
    (ev / "qat_selection.json").write_text(json.dumps(N.selection_doc(run, 2)), encoding="utf-8")
    r = BB.selected_epoch_vals(run, ev, work / "fq_o", SimpleNamespace(expect_telemetry_sha256=tel), stage="E6-R2",
                               runner=N.evaluator_cli_runner)
    fo = r.get("fake_quant_off") or {}
    c.check("F: card fix 1 for N2R2T: the pinned evaluator's CLI refuses stage E6-R2 [fqoff_stage_not_in_scope]; "
            "recorded, not ok; the selected epoch's fake-quant VAL taken from its epoch_end row",
            r.get("ok") is False and fo.get("status") == "failed" and fo.get("exit") == 1
            and "fqoff_stage_not_in_scope" in str(fo.get("message"))
            and (r.get("fake_quant") or {}).get("epoch") == 2, str(fo.get("message"))[:160])
    if full:
        rs = BB.selected_epoch_vals(run, ev, work / "fq_os", SimpleNamespace(expect_telemetry_sha256=tel),
                                    stage="E6-R2", runner=N.script_runner)
        fs = rs["fake_quant_off"]
        c.check("F: card fix 1 for N2R2T through scripts/evaluate_model.py itself: exit 1, [fqoff_stage_not_in_scope], "
                "nothing written",
                fs.get("exit") == 1 and "fqoff_stage_not_in_scope" in str(fs.get("message"))
                and not Path(fs["out_dir"]).exists(), str(fs.get("message"))[:200])
    # block B's whole flow, its four registered steps stood in for (they need a real 15-epoch run): N2R2 and N2R2T
    rules = REPO / BB.AM23_RULES_REL
    ends = {}

    def stand_in(run_dir, eval_dir, args):
        eval_dir.mkdir(parents=True)
        doc = N.selection_doc(run_dir, 2)
        doc["rules_sha256"] = G.sha256_file(rules)
        (eval_dir / "qat_selection.json").write_text(json.dumps(doc), encoding="utf-8")
        return [{"step": s, "argv": ["(stood in)"], "exit": 0, "result": f"RESULT: {s} (stood in)", "seconds": 0.0}
                for s in ("convert", "score", "select", "finalize")]
    for it in ("N2R2", "N2R2T"):
        rid = {"N2R2": "x-n2r2-s42", "N2R2T": "x-n2r2t-s42"}[it]
        idir = work / f"whole_{it}" / rid
        idir.mkdir(parents=True)
        rr = idir / rid
        if it == "N2R2":
            before = N.tf32_flags()
            try:
                env.X.run_n2r2(mode="smoke", model=env.fresh_student(), source_meta={"sha256": "cd" * 32}, out_dir=rr,
                               seed=42, clip_norm=5.0, clip_source="smoke", device="cpu", num_workers=0, parent=None,
                               loaders=N.one_batch(), max_steps=2, max_val_batches=1, log=quiet)
            finally:
                N.tf32_restore(before)
        else:
            short_run(env, rr)
        item_dir(rr, it, rid)
        o = work / f"whole_{it}_o" / f"{rid}-bb"
        with SK.patched(BB, "registered_block_b", stand_in):
            code_w, text_w = N.blockb_main(["--run-dir", str(rr), "--expect-telemetry-sha256",
                                            G.sha256_file(rr / Q.TELEMETRY_NAME), "--host-label", "smoke", "--run-id",
                                            f"{rid}-bb", "--out-dir", str(o)])
        ends[it] = (code_w, text_w, json.loads((o / "x_blockb.json").read_text(encoding="utf-8"))
                    if (o / "x_blockb.json").is_file() else {})
    code_t, text_t, xt = ends["N2R2T"]
    code_r, text_r, xr = ends["N2R2"]
    fo_t = xt.get("selected_epoch_fake_quant_off_val") or {}
    c.check("F: block B's whole flow for N2R2T (registered steps stood in): AM-23's rules file (sha256 e2999415...), "
            "card fix 1 run, the evaluator's [fqoff_stage_not_in_scope] recorded, INCOMPLETE (exit 3)",
            code_t == 3 and xt.get("item") == "N2R2T" and (xt.get("selection_rules") or {}).get("ok") is True
            and (xt.get("selection_rules") or {}).get("rules_sha256")
            == "e299941524968f143359d154566930819cd064152691395fb3647f4de0d98f3d"
            and fo_t.get("status") == "failed" and "fqoff_stage_not_in_scope" in str(fo_t.get("message"))
            and (xt.get("selected_epoch_fake_quant_val") or {}).get("epoch") == 2
            and "RESULT: X BLOCK B INCOMPLETE" in text_t and "fqoff_stage_not_in_scope" in text_t, text_t.strip()[-220:])

    def shape(x: dict) -> dict:
        fo = x.get("selected_epoch_fake_quant_off_val") or {}
        return {"keys": sorted(x), "steps": [s.get("step") for s in x.get("steps") or []],
                "rules": (x.get("selection_rules") or {}).get("rules_file"), "fqoff_status": fo.get("status"),
                "fqoff_exit": fo.get("exit"), "fqoff_code": "fqoff_stage_not_in_scope" in str(fo.get("message")),
                "format": x.get("format"), "artifact_status": x.get("artifact_status")}
    c.check("F: ... x_blockb.json as for N2R2: the same keys, steps, rules file and card fix 1 end; the same exit",
            code_t == code_r == 3 and shape(xt) == shape(xr) and xr.get("item") == "N2R2",
            f"{shape(xt)} vs {shape(xr)}")
    if full:
        py, bbs = sys.executable, str(REPO / "scripts" / "explore_qat_blockb.py")
        envv = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
        r1 = subprocess.run([py, "-I", bbs, "--run-dir", str(run), "--expect-telemetry-sha256", tel, "--host-label",
                             "smoke", "--run-id", "x-n2r2t-bbc", "--out-dir", str(work / "bbc" / "x-n2r2t-bbc")],
                            capture_output=True, text=True, env=envv)
        c.check("F: block B CLI: an N2R2T run passes run_item and reaches the guard's data-root check (exit 2 there)",
                r1.returncode == 2 and "data_root_not_trainval_only" in r1.stdout and "run_item" not in r1.stdout,
                r1.stdout.strip()[-160:])
        r2 = subprocess.run([py, "-I", bbs, "--run-dir", str(unknown / "r"), "--expect-telemetry-sha256", "0" * 64,
                             "--host-label", "smoke", "--run-id", "x-z9-bb", "--out-dir",
                             str(work / "bbz" / "x-z9-bb")], capture_output=True, text=True, env=envv)
        c.check("F: block B CLI: an unknown item refused (exit 2, run_item)",
                r2.returncode == 2 and "run_item" in r2.stdout, r2.stdout.strip()[-160:])


# ------------------------------------------------------------------ mutations
def mutations(env: Env, c) -> None:
    T, X, BB, SK = env.T, env.X, env.BB, env.SK
    orig_term, orig_run = T.t_logit_kd, T.run_n2r2t
    quick = ("copies", "A", "C", "E", "F")
    for name, mut in {
        "the term's weight dropped": lambda: SK.patched(T, "t_logit_kd", lambda **kw: orig_term(**{**kw, "weight": 0.0})),
        "run_n2r2t without the sites": lambda: SK.patched(T, "run_n2r2t", lambda **kw: orig_run(**{**kw,
                                                                                                     "sites": False})),
        "N2R2T on E6-E2's stage": lambda: SK.patched(X, "N2R2_STAGE", "e6-e2"),
        "any factor accepted in a real run": lambda: SK.patched(T, "values_error", lambda **kw: None),
        "block B without N2R2T's card fix 1": lambda: SK.patched(BB, "FQOFF_ITEMS_ADD3", ()),
    }.items():
        killed, failed = SK.mutation_killed(lambda: run_cases(env, quick, verbose=False, full=False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ref":
        raise SystemExit(ref_main(sys.argv[2:]))
    raise SystemExit(main())
