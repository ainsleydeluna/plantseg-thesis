#!/usr/bin/env python3
"""Smoke for E5, KL-3 parity (src/explore/kl3_parity.py, scripts/explore_kl3_parity.py). Synthetic models only.

A synthetic pair is built the way block B builds the real one: a random student, prepared and calibrated; its
pre-convert state saved as an epoch checkpoint (src/quant/qat.py save_epoch_checkpoint); the QNNPACK artifact
converted and serialized with its identity (src/quant/ptq.py torchscript_bytes, as convert_epoch does); the x86 copy
translated from the checkpoint (src/quant/x86_latency.py translate_qat_state_to_x86), converted under the x86 engine
and serialized with finalize's identity. Cases: the pair loads, sha-checked, identities read without loading; each
module runs under its own engine; parity on synthetic batches through the evaluator core (both mIoUs, the
difference, the argmax agreement, the largest logit difference); both artifacts byte-unchanged afterwards (read
only); refusals of a wrong sha256, a swapped pair, a pair of different epochs and a wrong stage; the TEST-day
manifest's E6-E2 pair is DL-146's; AM-24 clause 3 refuses at the current HEAD (no TEST results on record) and on rows
that are not RECORDED, lack TEST or the commit, or name a commit outside HEAD's history, and accepts a row that meets
it; the CLI refuses TEST and an unmet clause 3, writing nothing. Mutations: a clause-3 gate that always passes, an
identity reader that hides roles, and an engine wrapper that does not set the engine each fail a case.

    python -I scripts/smoke_explore_kl3.py        (exit 0 PASS, 1 FAIL)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402

from src.explore import guard as G                                   # noqa: E402
from src.explore import kl3_parity as P                              # noqa: E402
from src.explore.smokekit import Cases, mutation_killed, patched, scratch_dir  # noqa: E402

_PAIR: dict = {}


def build_pair(base: Path, *, epoch: int = 6, tag: str = "a") -> dict:
    from src.explore.synthetic import calibrated_prepared, random_student
    from src.models.student import build_student
    from src.quant import qat as Q
    from src.quant.prepare import convert_model
    from src.quant.ptq import qconfig_summary, torchscript_bytes
    from src.quant.qconfig import qat_qconfig, select_qnnpack_backend, select_x86_backend, x86_qat_qconfig
    from src.quant.x86_latency import X86_LATENCY_ARTIFACT_ROLE, load_qat_sidecar, translate_qat_state_to_x86
    p = calibrated_prepared(random_student(seed=7), size=128)
    state = Q.cpu_state(p)
    run = base / f"run_{tag}"
    ck, ck_sha, _ = Q.save_epoch_checkpoint(run, stage_name="E6", source_stage="E3", epoch=epoch, step=10,
                                           state=state, state_sha256=Q.state_digest(state), seed=42, clip_norm=5.0,
                                           run_id="x-kl3-synthetic", source_sha256="0" * 64, state_finite=True,
                                           nonfinite_since_step=None)
    select_qnnpack_backend()
    conv = convert_model(p)
    common = {"schema": "plantseg-int8-torchscript/1.0.0", "stage": "E6", "quantization": "qat", "num_classes": 116,
              "source_checkpoint_sha256": "0" * 64, "qat_checkpoint_sha256": ck_sha, "epoch": epoch,
              "run_id": "x-kl3-synthetic"}
    q_b, _ = torchscript_bytes(conv, {**common, "engine": "qnnpack", "artifact_role": "accuracy",
                                      "qconfig_fingerprint": qconfig_summary(qat_qconfig())["fingerprint"],
                                      "state_dict_companion": "none"})
    payload = load_qat_sidecar(ck)
    px, _report = translate_qat_state_to_x86("E6", build_student(num_classes=116, pretrained=False), payload)
    px.eval()
    engine = select_x86_backend()
    xconv = convert_model(px)
    x_b, _ = torchscript_bytes(xconv, {**common, "engine": engine, "artifact_role": X86_LATENCY_ARTIFACT_ROLE,
                                       "qconfig_fingerprint": qconfig_summary(x86_qat_qconfig())["fingerprint"],
                                       "usable_for_accuracy": False, "reduce_range": True})
    torch.backends.quantized.engine = "qnnpack"
    ev = base / f"eval_{tag}"
    (ev / "converted").mkdir(parents=True)
    (ev / "final").mkdir()
    qp, xp = ev / "converted" / f"e{epoch:02d}_int8_qnnpack.torchscript.pt", ev / "final" / f"e{epoch:02d}_int8_x86_latency.torchscript.pt"
    qp.write_bytes(q_b)
    xp.write_bytes(x_b)
    return {"q": qp, "x": xp, "q_sha": G.sha256_file(qp), "x_sha": G.sha256_file(xp), "engine": engine}


def pair_cached(base: Path):
    if "a" not in _PAIR:
        _PAIR["a"] = build_pair(base, epoch=6, tag="a")
        _PAIR["b"] = build_pair(base, epoch=7, tag="b")
    return _PAIR["a"], _PAIR["b"]


def run_cases(verbose: bool = False) -> Cases:
    c = Cases("SMOKE_EXPLORE_KL3", verbose=verbose)
    base = scratch_dir("x_kl3_")
    a, b = pair_cached(base)
    pair = c.passes("the synthetic pair loads (sha-checked, identities checked)", P.load_pair, a["q"], a["x"],
                    stage="E6", expect_q_sha256=a["q_sha"], expect_x_sha256=a["x_sha"])
    if pair:
        c.check("identities: QNNPACK accuracy artifact and x86 latency copy",
                pair["identities"]["qnnpack"]["engine"] == "qnnpack"
                and pair["identities"]["x86"]["artifact_role"] == "x86_cpu_proxy_latency"
                and pair["identities"]["x86"]["engine"] in ("x86", "fbgemm"))
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            pair["x86"](x)
            e_x = torch.backends.quantized.engine
            pair["qnnpack"](x)
            e_q = torch.backends.quantized.engine
        c.check("each module runs under its own engine", e_x == a["engine"] and e_q == "qnnpack", f"{e_x}, {e_q}")
        from src.explore.valscore import score_batches, synthetic_batches
        batches, manifest = synthetic_batches(3, size=(64, 64))
        res = P.parity(pair, score_q=lambda m: score_batches(m, batches, manifest)[0],
                       score_x_with_forward=lambda m, f: score_batches(m, batches, manifest, forward=f)[0])
        c.check("parity: both mIoUs and their difference", isinstance(res["qnnpack"]["all_class_miou"], float)
                and isinstance(res["x86"]["all_class_miou"], float) and isinstance(res["x86_minus_qnnpack_pp"], float))
        ag = res["agreement"]
        c.check("parity: argmax agreement over every pixel, and the logit differences", ag["pixels"] == 3 * 64 * 64
                and 0.0 <= ag["argmax_agreement"] <= 1.0 and ag["max_abs_logit_diff"] >= 0.0,
                f"agreement {ag['argmax_agreement']:.4f}, max |diff| {ag['max_abs_logit_diff']:.4f}")
        c.check("both artifacts are byte-unchanged (read only)", G.sha256_file(a["q"]) == a["q_sha"]
                and G.sha256_file(a["x"]) == a["x_sha"])
    c.refuses("a wrong sha256 refused", "artifact_sha256_mismatch", P.load_pair, a["q"], a["x"], stage="E6",
              expect_q_sha256="0" * 64, expect_x_sha256=a["x_sha"])
    c.refuses("a swapped pair refused (the x86 copy as the QNNPACK artifact)", "qnnpack_identity", P.load_pair,
              a["x"], a["q"], stage="E6", expect_q_sha256=a["x_sha"], expect_x_sha256=a["q_sha"])
    c.refuses("a pair of different epochs refused", "pair_mismatch", P.load_pair, a["q"], b["x"], stage="E6",
              expect_q_sha256=a["q_sha"], expect_x_sha256=b["x_sha"])
    c.refuses("a wrong stage refused", "stage_mismatch", P.load_pair, a["q"], a["x"], stage="E5",
              expect_q_sha256=a["q_sha"], expect_x_sha256=a["x_sha"])
    m = P.manifest_pair("E6-E2:s42:sel")
    c.check("the manifest's E6-E2 pair is DL-146's", m["artifact_sha256"].startswith("12e60c895ff0")
            and m["x86_copy_sha256"].startswith("eed07f984c6b"))
    # ---- AM-24 clause 3
    c.refuses("clause 3 at the current HEAD: no TEST results on record", "clause3_not_met", P.clause3_gate,
              "DL-999", "abcdef0")
    commit = "a" * 40
    rows = {"draft": f"| DL-900 | DRAFT | TEST results committed at {commit} | x | no |",
            "no_test": f"| DL-900 | RECORDED | results committed at {commit} | x | no |",
            "no_commit": "| DL-900 | RECORDED | TEST results committed | x | no |",
            "ok": f"| DL-900 | RECORDED (records commit) | TEST results committed at {commit} | x | yes |"}
    c.refuses("clause 3: a row that is not RECORDED", "clause3_not_met", P.clause3_gate, "DL-900", commit,
              dl_text=rows["draft"], is_ancestor=lambda h: True)
    c.refuses("clause 3: a row without TEST", "clause3_not_met", P.clause3_gate, "DL-900", commit,
              dl_text=rows["no_test"], is_ancestor=lambda h: True)
    c.refuses("clause 3: a row without the commit", "clause3_not_met", P.clause3_gate, "DL-900", commit,
              dl_text=rows["no_commit"], is_ancestor=lambda h: True)
    c.refuses("clause 3: a commit outside HEAD's history", "clause3_not_met", P.clause3_gate, "DL-900", commit,
              dl_text=rows["ok"], is_ancestor=lambda h: False)
    c.check("clause 3: a RECORDED row naming TEST and an ancestor commit passes",
            P.clause3_gate("DL-900", commit, dl_text=rows["ok"], is_ancestor=lambda h: True)["dl"] == "DL-900")
    # ---- the CLI
    py, cli = sys.executable, str(REPO / "scripts" / "explore_kl3_parity.py")
    env = {k: v for k, v in os.environ.items() if k != "PLANTSEG_DATA_ROOT"}
    common = ["--stage", "E6", "--qnnpack-artifact", str(a["q"]), "--qnnpack-sha256", a["q_sha"], "--x86-copy",
              str(a["x"]), "--x86-sha256", a["x_sha"], "--test-results-dl", "DL-999", "--test-results-commit", "abcdef0"]
    r = subprocess.run([py, "-I", cli, *common, "--split", "TE" + "ST", "--run-id", "x-kl3",
                        "--out-dir", str(base / "o" / "x-kl3")], capture_output=True, text=True, env=env)
    c.check("CLI: --split TEST refused first (exit 2)", r.returncode == 2 and "split_test" in r.stdout)
    r = subprocess.run([py, "-I", cli, *common, "--run-id", "x-kl3", "--out-dir", str(base / "o" / "x-kl3")],
                       capture_output=True, text=True, env=env)
    c.check("CLI: clause 3 unmet at HEAD refused, nothing written", r.returncode == 2 and "clause3_not_met" in r.stdout
            and not (base / "o").exists())
    return c


def main() -> int:
    print("[smoke_explore_kl3] cases")
    c = run_cases(verbose=True)
    print("[smoke_explore_kl3] mutations")

    class NoEngine(torch.nn.Module):
        def __init__(self, module, engine):
            super().__init__()
            self.module, self.engine = module, engine

        def forward(self, x):
            return self.module(x)
    for name, mut in {
        "clause 3 always met": lambda: patched(P, "clause3_gate", lambda *a, **k: {"dl": "DL-900"}),
        "identity reader that hides roles": lambda: patched(P, "read_identity", lambda p: {
            "artifact_role": "accuracy", "engine": "qnnpack", "stage": "E6"}),
        "engine wrapper that does not set the engine": lambda: patched(P, "EngineBound", NoEngine),
    }.items():
        killed, failed = mutation_killed(lambda: run_cases(False), mut)
        c.check(f"mutation KILLED: {name}", killed, f"e.g. {failed[:2]}")
    return c.result()


if __name__ == "__main__":
    raise SystemExit(main())
