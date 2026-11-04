"""E5 (addendum L-AM24-ADD-1): KL-3 parity, the x86 fbgemm latency copies against their QNNPACK artifacts on VAL.

DL-181 records that E5/E6's x86 latency copy (finalize's translation of the selected epoch's pre-convert state into an
x86-prepared student, reduce_range true, quantization parameters recomputed, the x86 engine) is certified only by
finalize's translation check on synthetic inputs, and that its accuracy parity with the QNNPACK artifact is not
measured (KL-3; errata E-68, E-69). This tool measures it on VAL, descriptively: for one pair, the QNNPACK artifact of
record and its x86 copy, each scored on the CPU through the evaluator library (src/explore/valscore.py: canvas, the
846 VAL images, union-present all-class mIoU), each under its own quantized engine (EngineBound sets it before every
forward; TorchScript packs weights for the engine active at load), plus the per-pixel argmax agreement of the two on
every VAL image (every output pixel of the canvas, padding included) and their largest logit difference.

READ-ONLY on both artifacts: each is hashed before it is loaded and again after the last pass, and each must equal the
sha256 the operator names (and, with a slot, the TEST-day manifest's artifact_sha256 / x86_copy_sha256 for that slot);
their embedded identities are read from the TorchScript zip without loading (src/quant/qat_artifacts.py
read_ts_identity). The output is exploratory only.

AM-24 clause 3 (ordering): the tool runs only after the TEST results are committed. `clause3_gate` requires a
decision-log row, at HEAD, with status RECORDED, holding the word TEST and the commit the operator names as the TEST
results' commit, and that commit in HEAD's history. Before the TEST session no such row exists and the tool refuses.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from . import guard as G

FORMAT = "x_kl3_parity/1"
QNNPACK = "qnnpack"
X86_ROLE = "x86_cpu_proxy_latency"
ACCURACY_ROLE = "accuracy"
_DL_ID = re.compile(r"DL-\d+[a-z]?")
_HEX = re.compile(r"[0-9a-f]{7,40}")


class KL3Error(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


# ------------------------------------------------------------------ AM-24 clause 3
def git_head_text(rel: str, repo_root: Path = G.REPO) -> str | None:
    p = subprocess.run(["git", "-C", str(repo_root), "show", f"HEAD:{rel}"], capture_output=True)
    return p.stdout.decode("utf-8") if p.returncode == 0 else None


def is_ancestor_of_head(commit: str, repo_root: Path = G.REPO) -> bool:
    p = subprocess.run(["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, "HEAD"],
                       capture_output=True)
    return p.returncode == 0


def clause3_gate(dl_id, test_commit, *, dl_text: str | None = None, is_ancestor=None) -> dict:
    """The TEST results are committed: `dl_id`'s row at HEAD is RECORDED, names TEST and `test_commit`, and that commit
    is in HEAD's history. Returns the evidence; raises KL3Error("clause3_not_met")."""
    if not isinstance(dl_id, str) or not _DL_ID.fullmatch(dl_id):
        raise KL3Error("clause3_not_met", f"--test-results-dl {dl_id!r} is not a decision-log id DL-<n>")
    if not isinstance(test_commit, str) or not _HEX.fullmatch(test_commit):
        raise KL3Error("clause3_not_met", f"--test-results-commit {test_commit!r} is not 7 to 40 lowercase hex")
    text = git_head_text("docs/DECISION_LOG.md") if dl_text is None else dl_text
    if text is None:
        raise KL3Error("clause3_not_met", "docs/DECISION_LOG.md is not readable at HEAD")
    row = next((ln for ln in text.splitlines() if ln.lstrip().startswith("|")
                and ln.strip().strip("|").split("|")[0].strip() == dl_id), None)
    if row is None:
        raise KL3Error("clause3_not_met", f"{dl_id} is not a row of docs/DECISION_LOG.md at HEAD: the TEST results "
                                          "are not on record, so KL-3 parity does not run yet (AM-24 clause 3)")
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    status = (cells[1].split() or [""])[0] if len(cells) > 1 else ""
    if status != "RECORDED":
        raise KL3Error("clause3_not_met", f"{dl_id} has status {status!r}, not RECORDED")
    if "TEST" not in row or test_commit not in row:
        raise KL3Error("clause3_not_met", f"{dl_id} does not name TEST and the commit {test_commit}")
    ok = (is_ancestor_of_head if is_ancestor is None else is_ancestor)(test_commit)
    if not ok:
        raise KL3Error("clause3_not_met", f"{test_commit} is not in HEAD's history")
    return {"dl": dl_id, "test_results_commit": test_commit, "row_status": status, "rule": "AM-24 clause 3"}


# ------------------------------------------------------------------ the pair, read-only
def manifest_pair(slot: str, repo_root: Path = G.REPO) -> dict:
    doc = json.loads((Path(repo_root) / "reports" / "derived" / "tday_manifest.json").read_text(encoding="utf-8"))
    hits = [e for e in doc["body"]["entries"] if isinstance(e, dict) and e.get("slot") == slot]
    if len(hits) != 1 or not isinstance(hits[0].get("hashed"), dict):
        raise KL3Error("manifest_slot", f"the TEST-day manifest holds no single hashed entry for slot {slot!r}")
    h = hits[0]["hashed"]
    if not h.get("x86_copy_sha256"):
        raise KL3Error("manifest_no_x86_copy", f"slot {slot!r} records no x86 latency copy")
    return {"slot": slot, "stage": hits[0].get("stage"), "artifact_sha256": h.get("artifact_sha256"),
            "x86_copy_sha256": h.get("x86_copy_sha256"), "record": h.get("record"),
            "selected_checkpoint": h.get("selected_checkpoint")}


class EngineBound(nn.Module):
    """A loaded INT8 TorchScript module run under its own quantized engine (set before every forward)."""

    def __init__(self, module, engine: str):
        super().__init__()
        self.module, self.engine = module, engine

    def forward(self, x):
        torch.backends.quantized.engine = self.engine
        return self.module(x)


def read_identity(path) -> dict:
    from src.quant.qat_artifacts import read_ts_identity
    return read_ts_identity(path)


def load_pair(q_path, x_path, *, stage: str, expect_q_sha256: str, expect_x_sha256: str) -> dict:
    """Both artifacts, sha-checked before the load, identities checked (roles, engines, stage), each loaded under its
    engine through the evaluator's loader (src/eval/model_loading.py load_int8_torchscript)."""
    from src.eval.model_loading import load_int8_torchscript
    shas = {"qnnpack": G.sha256_file(q_path), "x86": G.sha256_file(x_path)}
    for k, want in (("qnnpack", expect_q_sha256), ("x86", expect_x_sha256)):
        if shas[k] != want:
            raise KL3Error("artifact_sha256_mismatch", f"the {k} artifact has sha256 {shas[k]}, expected {want}")
    iq, ix = read_identity(q_path), read_identity(x_path)
    if iq.get("artifact_role") != ACCURACY_ROLE or iq.get("engine") != QNNPACK:
        raise KL3Error("qnnpack_identity", f"{q_path} identity {iq.get('artifact_role')!r}/{iq.get('engine')!r} is "
                                           "not the QNNPACK accuracy artifact")
    if ix.get("artifact_role") != X86_ROLE or ix.get("engine") not in ("x86", "fbgemm"):
        raise KL3Error("x86_identity", f"{x_path} identity {ix.get('artifact_role')!r}/{ix.get('engine')!r} is not "
                                       "an x86 latency copy")
    for k in ("stage", "epoch", "run_id", "qat_checkpoint_sha256"):
        if iq.get(k) != ix.get(k):
            raise KL3Error("pair_mismatch", f"the two artifacts differ in {k}: {iq.get(k)!r} vs {ix.get(k)!r}")
    if iq.get("stage") != stage:
        raise KL3Error("stage_mismatch", f"the artifacts are stage {iq.get('stage')!r}, --stage {stage!r}")
    supported = list(torch.backends.quantized.supported_engines)
    if QNNPACK not in supported or ix["engine"] not in supported:
        raise KL3Error("engine_unavailable", f"engines {supported} lack {QNNPACK} or {ix['engine']}")
    torch.backends.quantized.engine = QNNPACK
    q_mod, _ = load_int8_torchscript(q_path, stage=stage, method="qat", engine=QNNPACK, artifact_role=ACCURACY_ROLE)
    torch.backends.quantized.engine = ix["engine"]
    x_mod, _ = load_int8_torchscript(x_path, stage=stage, method="qat", engine=ix["engine"], artifact_role=X86_ROLE)
    torch.backends.quantized.engine = QNNPACK
    return {"qnnpack": EngineBound(q_mod, QNNPACK), "x86": EngineBound(x_mod, ix["engine"]),
            "identities": {"qnnpack": iq, "x86": ix}, "sha256_before": shas}


class AgreementForward:
    """evaluate_model's forward hook for the x86 pass: runs the x86 copy (returned, so it is what is scored) and the
    QNNPACK artifact on the same input, accumulating argmax agreement and the largest logit difference."""

    def __init__(self, other: nn.Module, device: str = "cpu"):
        self.other, self.device = other, torch.device(device)
        self.pixels = self.agree = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.elems = 0

    @torch.no_grad()
    def __call__(self, model, images):
        x = images.to(self.device)
        y = model(x)
        z = self.other(x)
        a, b = y.argmax(1), z.argmax(1)
        self.pixels += int(a.numel())
        self.agree += int((a == b).sum())
        d = (y - z).abs()
        self.max_abs = max(self.max_abs, float(d.max()))
        self.sum_abs += float(d.sum())
        self.elems += int(d.numel())
        return y

    def summary(self) -> dict:
        return {"pixels": self.pixels, "argmax_agreement": (self.agree / self.pixels) if self.pixels else None,
                "max_abs_logit_diff": self.max_abs, "mean_abs_logit_diff": (self.sum_abs / self.elems)
                if self.elems else None, "pixels_note": "every output pixel of the canvas, padding included"}


def parity(pair: dict, *, score_q, score_x_with_forward) -> dict:
    """Pass A: the QNNPACK artifact; pass B: the x86 copy, with the agreement hook. `score_q(model)` and
    `score_x_with_forward(model, forward)` return evaluator summaries (all_class_miou)."""
    a = score_q(pair["qnnpack"])
    fwd = AgreementForward(pair["qnnpack"])
    b = score_x_with_forward(pair["x86"], fwd)
    torch.backends.quantized.engine = QNNPACK
    vq, vx = a.get("all_class_miou"), b.get("all_class_miou")
    return {"qnnpack": a, "x86": b, "x86_minus_qnnpack_pp": (None if vq is None or vx is None
                                                             else 100.0 * (vx - vq)),
            "agreement": fwd.summary()}
