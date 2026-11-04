"""B1, the D-1 conversion-gap split (audit EXPLORE-AUDIT-1 §2 B1, items (i)-(v); lane L-AM24, build item E2).

For one QAT model's selected epoch, VAL is scored through the fake-quant graph (qat_artifacts.fake_quant_model: the
prepared student holding the epoch's state exactly, observers off, eval) in these passes:

  baseline        the registered graph, TF32 as the session has it (torch defaults on a GPU; none on a CPU)
  site:<name>     one of the ten sites on its input's grid (src/explore/sites.py), SE-pool sites first (item (v))
  all10           all ten sites together
  tf32off         the registered graph with TF32 off (both flags; CUDA only: a CPU has no TF32)
  all10_tf32off   all ten and TF32 off: the full corrected simulation, whose residual against the converted VAL
                  validates N2 (item (i))
  converted       the converted QNNPACK artifact of record, re-scored on the CPU by the same scorer

and the shares are reconciled (item (ii)): the total gap is baseline - converted; a site's share is baseline -
site:<name>; the interaction is (baseline - all10) - sum of the ten shares; TF32's share is baseline - tf32off; the
residual is all10_tf32off - converted (all10 - converted on a CPU, said so). Item (iii) is N4's table
(src/explore/decompose.py). Item (iv): the fake-quant passes run on the device the operator names; only a CUDA run on
the hardware class of the recorded fake-quant VAL (the QAT pod's GPU, TF32 at defaults) meets it, and the record says
which it was. Every value is the evaluator library's union-present all-class mIoU (src/explore/valscore.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import guard as G
from .sites import SE_SITES, SITE_NAMES, SiteMatcher, set_tf32, site_table, tf32_flags

FORMAT = "x_b1_split/1"
PASS_ORDER = ("baseline", *[f"site:{n}" for n in SITE_NAMES], "all10", "tf32off", "all10_tf32off", "converted")
CUDA_ONLY = ("tf32off", "all10_tf32off")


class B1Error(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


def pass_plan(device: str, passes=None) -> list[str]:
    want = list(PASS_ORDER) if passes is None else list(passes)
    unknown = [p for p in want if p not in PASS_ORDER]
    if unknown:
        raise B1Error("pass_unknown", f"unknown passes {unknown}; known: {list(PASS_ORDER)}")
    on_cuda = str(device).startswith("cuda")
    return [p for p in PASS_ORDER if p in want and (on_cuda or p not in CUDA_ONLY)]


def _sites_of(pass_name: str) -> tuple[str, ...]:
    if pass_name.startswith("site:"):
        return (pass_name.split(":", 1)[1],)
    if pass_name.startswith("all10"):
        return SITE_NAMES
    return ()


def run_passes(fq_model, converted, score, *, device: str, passes=None, log=print, on_pass=None) -> dict:
    """Score every planned pass. `score(model, device) -> summary dict` (all_class_miou). Returns {pass: summary}."""
    import torch
    plan = pass_plan(device, passes)
    tf32_before = tf32_flags()
    out = {}
    try:
        for p in plan:
            t0 = time.time()
            if p == "converted":
                summ = score(converted, "cpu")
                summ["device"] = "cpu"
            else:
                tf32_off = p.endswith("tf32off")
                if str(device).startswith("cuda"):
                    if tf32_off:
                        set_tf32(False)                          # both flags off (N2's TF32 switch)
                    else:                                        # the session's own state, e.g. torch's defaults
                        torch.backends.cudnn.allow_tf32 = tf32_before["cudnn_allow_tf32"]
                        torch.backends.cuda.matmul.allow_tf32 = tf32_before["cuda_matmul_allow_tf32"]
                matcher = SiteMatcher(fq_model, _sites_of(p)) if _sites_of(p) else None
                try:
                    with torch.no_grad():
                        summ = score(fq_model, device)
                finally:
                    if matcher is not None:
                        matcher.remove()
                summ.update({"device": str(device), "sites": list(_sites_of(p)), "tf32": tf32_flags(),
                             "site_calls": None if matcher is None else dict(matcher.calls)})
            summ["pass"] = p
            summ["pass_seconds"] = time.time() - t0
            out[p] = summ
            if on_pass is not None:
                on_pass(p, summ)
            log(f"[b1] {p:<20} all-class mIoU {summ.get('all_class_miou')!r} ({summ['pass_seconds']:.1f}s)")
    finally:
        torch.backends.cudnn.allow_tf32 = tf32_before["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = tf32_before["cuda_matmul_allow_tf32"]
    return out


def _v(scores: dict, p: str):
    s = scores.get(p)
    v = None if s is None else s.get("all_class_miou")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def reconcile(scores: dict, *, converted_of_record: float | None = None, device: str = "cpu") -> dict:
    """Item (ii): the gap, the shares (SE-pool sites first), their sum, the interaction, TF32 and the residual, in pp."""
    pp = lambda a, b: None if a is None or b is None else 100.0 * (a - b)   # noqa: E731
    base, conv, all10 = _v(scores, "baseline"), _v(scores, "converted"), _v(scores, "all10")
    shares = {n: pp(base, _v(scores, f"site:{n}")) for n in SITE_NAMES}
    known = [s for s in shares.values() if s is not None]
    total_sites = pp(base, all10)
    sum_shares = sum(known) if len(known) == len(SITE_NAMES) else None
    cuda = str(device).startswith("cuda")
    full = _v(scores, "all10_tf32off") if cuda else all10
    return {"gap_baseline_minus_converted_pp": pp(base, conv),
            "converted_rescored": conv, "converted_of_record": converted_of_record,
            "converted_rescored_minus_record_pp": pp(conv, converted_of_record),
            "site_shares_pp": shares, "se_pool_shares_pp": {n: shares[n] for n in SE_SITES},
            "se_pool_sum_pp": (sum(shares[n] for n in SE_SITES) if all(shares[n] is not None for n in SE_SITES)
                               else None),
            "sum_of_ten_shares_pp": sum_shares, "all10_share_pp": total_sites,
            "interaction_pp": None if sum_shares is None or total_sites is None else total_sites - sum_shares,
            "tf32_share_pp": pp(base, _v(scores, "tf32off")) if cuda else None,
            "tf32_note": None if cuda else "CPU run: no TF32 exists, so TF32's share is not measured here",
            "full_corrected_pass": "all10_tf32off" if cuda else "all10 (CPU: no TF32)",
            "residual_full_corrected_minus_converted_pp": pp(full, conv),
            "explained_fraction": (None if pp(base, conv) in (None, 0.0) or pp(base, full) is None
                                   else pp(base, full) / pp(base, conv))}


def table_text(scores: dict, rec: dict) -> str:
    f = lambda v: "—" if v is None else f"{v:.5f}"      # noqa: E731
    g = lambda v: "—" if v is None else f"{v:+.3f}"     # noqa: E731
    lines = [f"{'pass':<22} {'VAL mIoU':>9} {'share pp':>9}", "-" * 42]
    for p in PASS_ORDER:
        if p not in scores:
            continue
        share = rec["site_shares_pp"].get(p.split(":", 1)[1]) if p.startswith("site:") else None
        lines.append(f"{p:<22} {f(_v(scores, p)):>9} {g(share):>9}")
    lines += ["-" * 42,
              f"gap baseline - converted      {g(rec['gap_baseline_minus_converted_pp'])} pp",
              f"SE-pool shares (8), sum       {g(rec['se_pool_sum_pp'])} pp",
              f"sum of ten shares             {g(rec['sum_of_ten_shares_pp'])} pp",
              f"all ten together              {g(rec['all10_share_pp'])} pp",
              f"interaction (all10 - sum)     {g(rec['interaction_pp'])} pp",
              f"TF32 share                    {g(rec['tf32_share_pp'])} pp"
              + ("" if rec["tf32_note"] is None else f"  ({rec['tf32_note']})"),
              f"residual {rec['full_corrected_pass']:<14} {g(rec['residual_full_corrected_minus_converted_pp'])} pp",
              f"converted re-scored - of record {g(rec['converted_rescored_minus_record_pp'])} pp"]
    return "\n".join(lines)


# ------------------------------------------------------------------ the inputs of a real B1 run (read-only)
def read_selection(path, expect_sha256: str) -> dict:
    p = Path(path)
    sha = G.sha256_file(p)
    if sha != expect_sha256:
        raise B1Error("selection_sha256_mismatch", f"{p} has sha256 {sha}, expected {expect_sha256}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("format") != "qat_selection/1" or not isinstance((doc.get("winner") or {}).get("epoch"), int):
        raise B1Error("selection_format", f"{p} is not a qat_selection/1 record with a winner epoch")
    return {"path": str(p.resolve()), "sha256": sha, "doc": doc}


def load_selected_fake_quant(run_dir, epoch: int):
    """The selected epoch's fake-quant model, through the pinned record readers (sha-checked against its row)."""
    from src.quant import qat_artifacts as A
    from src.quant.x86_latency import load_qat_sidecar
    rec = A.read_run_record(run_dir)
    A.require_complete(rec)
    ck, ck_sha = A.verified_checkpoint(rec, epoch)
    payload = load_qat_sidecar(ck)
    model = A.fake_quant_model(payload["model_state_dict"])
    return model, {"run_id": rec["run_id"], "stage": rec["run_meta"].get("stage"), "seed": rec["run_meta"].get("seed"),
                   "epoch": epoch, "checkpoint": str(ck), "checkpoint_sha256": ck_sha,
                   "telemetry_sha256": rec["telemetry_sha256"],
                   "fake_quant_val_recorded": rec["ends"][epoch].get("fake_quant_val_all_class_miou"),
                   "recorded_device": rec["run_meta"].get("device"), "recorded_gpu": rec["run_meta"].get("gpu_name"),
                   "recorded_tf32": rec["run_meta"].get("tf32")}


def load_converted(path, *, stage: str, expect_sha256: str):
    """The converted QNNPACK TorchScript artifact, read-only: hashed before the load (and again by the caller after)."""
    from src.eval.model_loading import load_int8_torchscript
    from src.quant.qconfig import select_qnnpack_backend
    sha = G.sha256_file(path)
    if sha != expect_sha256:
        raise B1Error("artifact_sha256_mismatch", f"{path} has sha256 {sha}, expected {expect_sha256}")
    select_qnnpack_backend()
    model, meta = load_int8_torchscript(path, stage=stage, method="qat", engine="qnnpack")
    return model, {"path": str(Path(path).resolve()), "sha256": sha, "identity": meta}


def describe_sites(fq_model) -> list[dict]:
    return site_table(fq_model)
