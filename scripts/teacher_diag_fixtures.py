"""Synthetic fixtures for the L-TEACHER-DIAG smokes (a helper module, not a smoke; no RESULT line).

No PlantSeg data, no checkpoint of record, no network, no GPU. Everything is written under a fresh
temporary directory whose path names neither TEST nor VAL (scripts/synthetic_ptq_fixtures.safe_tmpdir).

  stub teacher      P31: the stream-consuming pattern of scripts/smoke_teacher.py (StubIsolatedNMF,
                    StubHamHead): one torch.rand basis draw per head call, routed through the attached
                    NMF stream, so different draws give different logits. Weights are seeded inside a
                    forked RNG, so building a stub never moves the caller's CPU RNG. Variants: no
                    stream, an in-place-mutating head, a NaN head, a global-RNG draw, an oversized
                    head, an mmseg import.
  checkpoints       small stub checkpoints holding exactly a stub factory's state_dict, so
                    validate_teacher_artifact and K-part's strict load (R6) accept them.
  data root         images/{train,val} + annotations/{train,val} at the locked counts 5,367 / 846
                    (scripts/synthetic_ptq_fixtures.make_tree), plus a matching strata file.
  patchers          fake_git(): HEAD and `git status` answers for the commit-binding gates;
                    patched(): module attributes for one case.
  load counter      count_loads(): calls of validate_teacher_artifact and of both load lines.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

NC = 116


# ---------------------------------------------------------------- stubs (P31; scripts/smoke_teacher.py)
class StubMSCAN(nn.Module):
    """Stand-in for MSCAN-B: four stage features at strides 4/8/16/32."""

    def __init__(self, dims=(64, 128, 320, 512), strides=(4, 8, 16, 32)):
        super().__init__()
        self.dims, self.strides = dims, strides
        self.projs = nn.ModuleList([nn.Conv2d(3, d, 1, bias=False) for d in dims])

    def forward(self, x):
        out = []
        for proj, s in zip(self.projs, self.strides):
            h, w = max(x.shape[-2] // s, 1), max(x.shape[-1] // s, 1)
            out.append(proj(torch.nn.functional.adaptive_avg_pool2d(x, (h, w))))
        return tuple(out)


class StubLightHamHead(nn.Module):
    """Stand-in for LightHamHead without NMF: stride-8 logits from the stride-8 stage feature."""

    def __init__(self, in_ch=128, channels=512, num_classes=NC):
        super().__init__()
        self.num_classes = num_classes
        self.squeeze = nn.Conv2d(in_ch, channels, 1, bias=False)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        return self.conv_seg(self.squeeze(feats[1]))


class StubIsolatedNMF(nn.Module):
    """Stand-in for IsolatedNMF2D: one torch.rand basis draw per forward, routed through a stream."""
    SUPPORTS_NMF_STREAM = True

    def __init__(self):
        super().__init__()
        self.nmf_stream = None
        self.draw_log = []                                   # the bases drawn, for sequence checks

    def forward(self, x):
        draw = lambda: torch.rand(x.shape[0], x.shape[1], 1, 1)  # noqa: E731 -- the upstream-style draw
        bases = draw() if self.nmf_stream is None else self.nmf_stream.draw(draw)
        self.draw_log.append(bases.clone())
        return x * bases


class StubHamHead(StubLightHamHead):
    def __init__(self, nmf: nn.Module | None = None, **kw):
        super().__init__(**kw)
        self.ham = nmf if nmf is not None else StubIsolatedNMF()

    def forward(self, feats):
        return self.conv_seg(self.ham(self.squeeze(feats[1])))


class MutatingHamHead(StubHamHead):
    """Changes the backbone feature it was given, in place: the feature-hash check must catch it."""

    def forward(self, feats):
        feats[1].add_(1e-3)
        return super().forward(feats)


class NaNHamHead(StubHamHead):
    def forward(self, feats):
        z = super().forward(feats)
        z[:, 3, 0, 0] = float("nan")
        return z


class GlobalDrawHamHead(StubHamHead):
    """Draws once from the caller's global CPU stream outside the NMF window: the RNG check must catch it."""

    def forward(self, feats):
        torch.rand(1)
        return super().forward(feats)


class DoubleDrawHamHead(StubHamHead):
    """Runs the isolated NMF module twice per call: the one-draw-per-head-call check must catch it."""

    def forward(self, feats):
        return self.conv_seg(self.ham(self.ham(self.squeeze(feats[1]))))


class StubSegNeXt(nn.Module):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone or StubMSCAN()
        self.decode_head = head or StubHamHead()

    def extract_feat(self, x):
        return self.backbone(x)


def _seeded(build, seed: int = 1234):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build()


def stub_factory(_config_path=None, _ckpt_path=None):
    """The default stub: deterministic weights, one isolated NMF module in the head."""
    return _seeded(lambda: StubSegNeXt(head=StubHamHead()))


def streamless_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=StubLightHamHead()))


def mutating_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=MutatingHamHead()))


def nan_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=NaNHamHead()))


def global_draw_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=GlobalDrawHamHead()))


def double_draw_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=DoubleDrawHamHead()))


def big_factory(_config_path=None, _ckpt_path=None):
    return _seeded(lambda: StubSegNeXt(head=StubHamHead(channels=8192)))     # > 1e6 parameters


def mmseg_factory(_config_path=None, _ckpt_path=None):
    """A build that leaves a module named mmseg in sys.modules (a stand-in, removed by the caller)."""
    sys.modules.setdefault("mmseg", types.ModuleType("mmseg"))
    return stub_factory()


def head_of(loaded_or_frozen):
    """The stub head's isolated NMF module, through the loaded teacher."""
    frozen = getattr(loaded_or_frozen, "frozen", loaded_or_frozen)
    return frozen.teacher.model.decode_head.ham


# ---------------------------------------------------------------- checkpoints
def write_stub_ckpt(path, tag: str = "stub", pad_bytes: int = 0, factory=None) -> tuple[Path, str]:
    """A stub checkpoint holding exactly `factory()`'s state_dict (default stub_factory), so K-part's strict
    load (segnext_teacher.strict_load_teacher_state) accepts it and the loaded weights are the factory's own
    seeded weights. pad_bytes adds a zero tensor to exceed size limits; that checkpoint is refused by stat
    before any load."""
    import hashlib
    p = Path(path)
    state = {k: v.detach().clone() for k, v in (factory or stub_factory)().state_dict().items()}
    if pad_bytes:
        state["backbone.pad"] = torch.zeros(pad_bytes // 4 + 1)
    torch.save({"meta": {"stub": tag}, "state_dict": state}, p)
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- data root and strata
def make_data_root(prefix: str = "diag_fix_", n_train: int = 5367, n_val: int = 846) -> tuple[Path, dict]:
    from scripts.synthetic_ptq_fixtures import make_tree, safe_tmpdir
    base = safe_tmpdir(prefix)
    root = base / "plantseg"
    stems = make_tree(root, n_train=n_train, n_val=n_val, seed=7)
    return root, stems


def write_strata(path, stems, *, status: str = "provisional") -> Path:
    """A synthetic strata file in the plantseg-train-strata/1.0.0 layout over these TRAIN stems."""
    from scripts.build_train_strata import split_list_sha256
    rows = []
    for c in range(1, NC):
        rank = c
        tercile = "T1" if rank <= 38 else ("T2" if rank <= 76 else "T3")
        rows.append({"class_id": c, "pixels": 1000 - c, "share": 1.0 / 115, "images": 30 - (c % 13),
                     "rank": rank, "tercile": tercile, "rare": (c % 10 == 0), "zero_train_pixels": False})
    doc = {"schema": "plantseg-train-strata/1.0.0", "lane": "L-AM17-STRATA", "artifact_status": status,
           "split": "train", "split_list_sha256": split_list_sha256(list(stems)), "n_images": len(stems),
           "num_classes": NC, "per_class": rows, "tercile_sizes": [38, 38, 39]}
    p = Path(path)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return p


def set_data_root(root) -> None:
    """Point configs/data.py at a synthetic root (it reads PLANTSEG_DATA_ROOT once, at import)."""
    os.environ["PLANTSEG_DATA_ROOT"] = str(root)
    if "configs.data" in sys.modules:
        sys.modules["configs.data"].DATA["root"] = str(root)


# ---------------------------------------------------------------- patchers
@contextlib.contextmanager
def fake_git(head: str = "a" * 40, status: str = "", tree_status: str | None = None):
    """Answer the commit-binding gates as a checkout at `head` would, without touching git."""
    import src.eval.teacher_diag as td
    old = (td.git_head, td.git_status)
    td.git_head = lambda: head

    def _status(paths):
        if tree_status is not None and list(paths) == ["src", "configs", "scripts"]:
            return tree_status
        return status
    td.git_status = _status
    try:
        yield head
    finally:
        td.git_head, td.git_status = old


@contextlib.contextmanager
def count_loads():
    """Count calls of validate_teacher_artifact and of both teacher load lines."""
    import src.distill.teacher as dt
    import src.eval.model_loading as ml
    import src.eval.stage_artifacts as sa
    calls = {"validate_teacher_artifact": 0, "load_frozen_teacher": 0, "load_teacher_model": 0}
    orig = (sa.validate_teacher_artifact, dt.load_frozen_teacher, ml.load_teacher_model)

    def wrap(name, fn):
        def inner(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return inner
    sa.validate_teacher_artifact = wrap("validate_teacher_artifact", orig[0])
    dt.load_frozen_teacher = wrap("load_frozen_teacher", orig[1])
    ml.load_teacher_model = wrap("load_teacher_model", orig[2])
    try:
        yield calls
    finally:
        sa.validate_teacher_artifact, dt.load_frozen_teacher, ml.load_teacher_model = orig


@contextlib.contextmanager
def patched(obj, **attrs):
    """Set attributes on a module or object for the duration of a case."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def argv(**kw) -> list[str]:
    out = []
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is None or v is False:
            continue
        else:
            out += [flag, str(v)]
    return out


def reference_artifact(out_dir, ckpt, sha256: str, *, n: int = 3, factory=None, run_id: str = "ref_run") -> Path:
    """scripts/evaluate_model.py run() on the stub teacher: a smoke's R3-equivalent --val-reference. `sha256` is
    the one write_stub_ckpt returned; the evaluator's teacher stage requires it (R6)."""
    from scripts import evaluate_model as em
    from src.distill.segnext_teacher import segnext_builder
    from src.eval.teacher_diag import REPO, TEACHER_CONFIG_REL
    args = em.build_parser().parse_args([
        "--stage", "teacher", "--model-role", "teacher", "--split", "val", "--checkpoint", str(ckpt),
        "--teacher-ckpt-sha256", sha256,
        "--teacher-config", str(REPO / TEACHER_CONFIG_REL), "--artifact-status", "smoke",
        "--max-samples", str(n), "--batch-size", "1", "--out-dir", str(out_dir), "--run-id", run_id])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return em.run(args, teacher_builder=segnext_builder(model_factory=factory or stub_factory))


def call_run(module, *, factory=stub_factory, **flags) -> tuple[int, str]:
    """Parse `flags` with the script's own parser and call its run(); (exit code, stderr)."""
    args = module.build_parser().parse_args(argv(**flags))
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = module.run(args) if factory is None else module.run(args, model_factory=factory)
    return code, err.getvalue()


# ---------------------------------------------------------------- synthetic VAL re-scores (S3's pattern)
SYN_SHA_T, SYN_SHA_E = "a" * 64, "b" * 64
SYN_DIG_T, SYN_DIG_E = "sha256:" + "1" * 64, "sha256:" + "2" * 64


def clean_git_pin(root) -> Path:
    """A throwaway repository with one commit and nothing dirty: the pin of synthetic re-scores
    (scripts/smoke_gap_bootstrap_val.py's clean_git_fixture). Never the project repository."""
    import subprocess
    root = Path(root)
    root.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=diag-fixture", "-c", "user.email=diag@invalid",
                        "-c", "commit.gpgsign=false", *args], cwd=str(root), check=True, capture_output=True)
    git("init", "-q")
    (root / "seed.txt").write_text("diag fixture\n", encoding="utf-8")
    git("add", "--", "seed.txt")
    git("commit", "-q", "-m", "fixture baseline")
    return root


def core_result(ids, gts, preds):
    """The real evaluation core on one-hot logits of `preds` ([n, h, w]) against `gts`."""
    from src.eval.evaluate import Condition, EvalBatch, ManifestEntry, evaluate_model
    manifest = [ManifestEntry(i, iid, iid) for i, iid in enumerate(ids)]
    logits = torch.nn.functional.one_hot(preds, NC).permute(0, 3, 1, 2).float()
    batches = [EvalBatch(images=logits[i:i + 64], targets=gts[i:i + 64], image_ids=ids[i:i + 64],
                         clean_image_ids=ids[i:i + 64], manifest_indices=list(range(i, min(i + 64, len(ids)))))
               for i in range(0, len(ids), 64)]
    res = evaluate_model(None, batches, expected_manifest=manifest, condition=Condition(), num_classes=NC,
                         background_index=0, ignore_index=255, forward=lambda _m, x: x)
    return res, manifest


def cpu_runtime(digest, n_rows: int = 846) -> dict:
    from src.eval.artifacts import EVAL_RUNTIME_VERSION
    from src.eval.eval_runtime import RECORD_KEYS
    values = {"eval_runtime_version": EVAL_RUNTIME_VERSION, "model_device": "cpu", "model_tensor_devices": ["cpu"],
              "input_devices": ["cpu"], "forward_batches": n_rows, "batch_size": 1, "num_workers": 0,
              "determinism_policy_applied": False, "cuda_initialized_before_policy": None,
              "inherited_cublas_workspace_config": None, "determinism": {}, "fill_uninitialized_memory": None,
              "cudnn_enabled": True, "tf32": {}, "gpu_name": None, "gpu_capability": None, "cudnn_version": None,
              "torch_cuda": None, "torch_num_threads": 4, "pillow": None, "checkpoint_iteration": None,
              "checkpoint_best_val_miou_all_class": None, "image_digest": digest, "eval_warnings": [],
              "eval_warnings_truncated": False, "nondeterministic_alert_count": 0}
    return {k: values[k] for k in RECORD_KEYS}


def write_rescore(out_dir, res, manifest, *, stage, role, sha, digest, pin, status="provisional") -> Path:
    """A provisional CPU batch-1 VAL re-score written by the real writer at the throwaway pin."""
    from src.eval.artifacts import (DatasetMeta, RunMeta, prepare_artifact_request, validate_artifact_request,
                                    write_artifact)
    from src.eval.evaluate import Condition
    class_map = [{"class_id": 0, "name": "", "role": "background_or_non_disease"}] + \
                [{"class_id": c, "name": f"syn_{c:03d}", "role": "disease"} for c in range(1, NC)]
    out_dir = Path(out_dir)
    req = prepare_artifact_request(
        out_dir=out_dir, artifact_status=status, run_id=f"syn_{out_dir.name}",
        run=RunMeta(stage=stage, model_role=role, precision="fp32", checkpoint_path=f"synthetic_{stage}.pt",
                    checkpoint_sha256=sha, device="cpu"),
        dataset=DatasetMeta(name="SYNTHETIC diag fixture (NOT PlantSeg data)", doi="10.5281/zenodo.17719108",
                            split="val", condition=Condition(), preprocess_protocol="core_preprocess/1.0.0",
                            expected_rows=len(manifest)),
        expected_manifest=manifest, class_map=class_map, repo_root=pin)
    return write_artifact(res, req, validate_artifact_request(req), timestamp_utc="2026-10-02T00:00:00Z",
                          eval_runtime=cpu_runtime(digest, len(manifest)))


def rehash_artifact(d) -> None:
    from src.eval.artifacts import ARTIFACT_FILES, MANIFEST_NAME
    import hashlib
    d = Path(d)
    (d / MANIFEST_NAME).write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n"
                                           for n in ARTIFACT_FILES), encoding="utf-8", newline="\n")


@contextlib.contextmanager
def synthetic_pairing(v_t: float, v_e: float):
    """S3's smoke pattern: point scripts/gap_bootstrap_val.PAIRING at the synthetic shas, digests and
    references for the duration of a case."""
    import scripts.gap_bootstrap_val as gap
    old = {k: dict(v) for k, v in gap.PAIRING.items()}
    gap.PAIRING["teacher"].update(checkpoint_sha256=SYN_SHA_T, image_digest=SYN_DIG_T, reference=v_t)
    gap.PAIRING["e1"].update(checkpoint_sha256=SYN_SHA_E, image_digest=SYN_DIG_E, reference=v_e)
    try:
        yield gap.PAIRING
    finally:
        for k, v in old.items():
            gap.PAIRING[k].clear()
            gap.PAIRING[k].update(v)


def gap_output(path, art_t, art_e, *, e1_tolerance: float = 1e-7) -> Path:
    """A synthetic S3 gap output: the fields the D4 reducer binds to, the point from S3's own RuleGap."""
    from src.stats.eligibility import UNION_PRESENT
    from src.stats.gap import RuleGap
    from src.stats.noninferiority import PooledStages
    stages = PooledStages.from_runs(art_e.run, art_t.run)
    point = RuleGap(stages, UNION_PRESENT, range(NC)).point()["point"]
    doc = {"lane": "L-AM17B-GAP", "artifact_status": "provisional",
           "inputs": {"teacher": art_t.describe(), "e1": art_e.describe()},
           "rules": {"union_present": {"point": point}}, "e1_tolerance": {"value": e1_tolerance}}
    p = Path(path)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return p


def output_files(out_dir) -> list[Path]:
    d = Path(out_dir)
    return sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []
