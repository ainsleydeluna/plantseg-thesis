"""Lane 6 (L-AM4 + L-AM1q): one QAT epoch converted to the INT8 artifacts of record (AM-4a item 2).

    epoch_ckpts/eNN.pt   (the pre-convert QAT state src/quant/qat.py wrote, in the x86 sidecar schema)
      -> its sha256 against the epoch_end row; loaded weights_only; the state predicate (a non-finite
         state is "not convertible": recorded, never converted)
      -> the observer and fake-quant flags its epoch must carry
      -> prepare_qat_model(build_student()) rebuilt and every tensor copied by name: equal key sets both
         ways, every entry equal with the same shape and dtype
      -> convert under QNNPACK
      -> the QNNPACK TorchScript artifact (traced at 1x3x512x512, its identity in plantseg_int8.json)
         and the converted state_dict companion (quantization "qat"), both written exclusively
      -> both re-read from disk and checked as src/quant/ptq.py checks its artifacts: per-channel INT8
         weights, no undeclared float region, bitwise parity on ptq.synthetic_parity_inputs()
      -> eNN_run_meta.json, the evaluator's --provenance (src/quant/runner.py build_run_provenance)

Conversion is data-free. Each epoch converts in a fresh process: TorchScript renames a class traced
twice in one process, so a second trace would write different bytes. A failed check writes eNN_STOP.json
and no provenance.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from . import qat as Q                                                      # noqa: E402
from .qat import QATRefused, QATStop                                        # noqa: E402

LANE = Q.LANE
RUN_META_SCHEMA = "plantseg-qat-epoch-run-meta/1.0.0"
CONVERTED_DIR = "converted"
SCORES_DIR = "scores"
PURPOSES = ("record", "timing")
NOT_CONVERTIBLE_RULE = "excluded: non-finite state (AM-19 item 3(a))"
CONVERT_CODE_PATHS = (
    "src/quant/qat_artifacts.py", "src/quant/qat.py", "src/quant/ptq.py", "src/quant/prepare.py",
    "src/quant/qconfig.py", "src/quant/x86_latency.py", "src/quant/runner.py", "src/quant/stages.py",
    "src/models/student.py", "src/eval/model_loading.py", "configs/quant.py", "configs/model.py",
    "configs/data.py", "scripts/qat_epoch_eval.py",
)


class QATIncomplete(RuntimeError):
    """The run or the eval directory is incomplete (exit 3)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def epoch_names(epoch: int) -> dict:
    e = f"e{epoch:02d}"
    return {"torchscript": f"{e}_int8_qnnpack.torchscript.pt", "state_dict": f"{e}_int8_student.pt",
            "run_meta": f"{e}_run_meta.json", "stop": f"{e}_STOP.json",
            "not_convertible": f"{e}_not_convertible.json"}


def json_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def refuse_test_path(path, what: str) -> None:
    for form in (str(path), os.path.realpath(path)):
        if "test" in form.lower():
            raise QATRefused(f"{what}_test_path", f"{what} {form!r} contains 'test'")


# ------------------------------------------------------------------ the run's record (AM-19)
def _strict(line: str):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(line, parse_constant=bad)


def read_run_record(run_dir) -> dict:
    """The run's telemetry, read strictly. Refuses an unusable record; reports completeness (AM-19).

    Complete: exactly one run_meta row, one epoch_end row per epoch 1..15 in order, the last row the
    epoch-15 epoch_end with run_complete true, no run_abort or run_stop row and no torn line.
    """
    d = Path(run_dir)
    refuse_test_path(d, "run_dir")
    tel = d / Q.TELEMETRY_NAME
    if not tel.is_file():
        raise QATRefused("telemetry_missing", f"{tel} is missing")
    raw = tel.read_bytes()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    torn = bool(lines) and lines[-1] != ""
    if lines and lines[-1] == "":
        lines = lines[:-1]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(_strict(line))
        except ValueError as e:
            if i == len(lines) - 1:
                torn = True
                break
            raise QATRefused("telemetry_not_strict_json", f"{tel} line {i + 1}: {e}") from e
    metas = [r for r in rows if r.get("event") == "run_meta"]
    if len(metas) != 1 or rows[0].get("event") != "run_meta":
        raise QATRefused("run_meta_rows", f"{tel} holds {len(metas)} run_meta rows; exactly one, first, is required")
    meta = metas[0]
    ends: dict[int, dict] = {}
    order_ok = True
    for r in rows:
        if r.get("event") == "epoch_end":
            e = r.get("epoch")
            if e in ends:
                raise QATRefused("epoch_end_duplicate", f"{tel} holds two epoch_end rows for epoch {e}")
            if ends and e != max(ends) + 1:
                order_ok = False
            ends[e] = r
    reasons = []
    if torn:
        reasons.append("the telemetry's last line is torn")
    if any(r.get("event") in ("run_abort", "run_stop") for r in rows):
        reasons.append("the telemetry holds a run_abort or run_stop row")
    if sorted(ends) != list(range(1, Q.EPOCHS + 1)) or not order_ok:
        reasons.append(f"epoch_end rows for epochs {sorted(ends)}, not 1..{Q.EPOCHS} in order")
    last = rows[-1] if rows else {}
    if not (last.get("event") == "epoch_end" and last.get("epoch") == Q.EPOCHS and last.get("run_complete") is True):
        reasons.append("the last row is not the epoch-15 completion record")
    return {"run_dir": str(d.resolve()), "run_id": meta.get("run_id"), "run_meta": meta, "ends": ends,
            "rows": rows, "telemetry": str(tel), "telemetry_sha256": Q.sha256_bytes(raw),
            "complete": not reasons, "incomplete_reasons": reasons}


def require_complete(rec: dict) -> None:
    if not rec["complete"]:
        raise QATIncomplete("run_incomplete", f"{rec['run_dir']}: {'; '.join(rec['incomplete_reasons'])}")


def checkpoint_path(rec: dict, epoch: int) -> Path:
    end = rec["ends"][epoch]
    return Path(rec["run_dir"]) / end["checkpoint"]


def verified_checkpoint(rec: dict, epoch: int) -> tuple[Path, str]:
    end = rec["ends"].get(epoch)
    if end is None:
        raise QATRefused("epoch_unknown", f"the run records no epoch_end row for epoch {epoch}")
    p = checkpoint_path(rec, epoch)
    if p.name != f"e{epoch:02d}.pt" or p.parent.name != Q.EPOCH_DIR:
        raise QATRefused("checkpoint_name", f"epoch {epoch}'s row names {end['checkpoint']!r}")
    if not p.is_file():
        raise QATIncomplete("checkpoint_missing", f"{p} is missing")
    sha = Q.sha256_file(p)
    if sha != end.get("checkpoint_sha256"):
        raise QATRefused("checkpoint_sha256_mismatch", f"{p} has sha256 {sha}; its epoch_end row records "
                                                       f"{end.get('checkpoint_sha256')}")
    return p, sha


# ------------------------------------------------------------------ the prepared student and its state
def prepared_skeleton() -> nn.Module:
    from src.models.student import build_student
    from .prepare import prepare_qat_model
    return prepare_qat_model(build_student(num_classes=Q.NUM_CLASSES, pretrained=False), select_backend=False)


def alias_keys(prepared: nn.Module) -> list[str]:
    """state_dict entries that are not a named parameter or buffer: the FixedQParams fake-quants'
    scale/zero_point, which alias their observer's (18 in the student)."""
    named = {n for n, _ in prepared.named_parameters()} | {n for n, _ in prepared.named_buffers()}
    return sorted(k for k in prepared.state_dict() if k not in named)


def load_state_by_name(prepared: nn.Module, state: dict) -> dict:
    """Copy a QAT state into a freshly prepared student by name and prove the copy exact."""
    from .x86_latency import copy_qat_state_by_name
    target = set(prepared.state_dict())
    missing, extra = sorted(target - set(state)), sorted(set(state) - target)
    if missing or extra:
        raise QATStop("state_key_mismatch", f"the checkpoint and the prepared student differ in keys: missing "
                                            f"{missing[:5]} ({len(missing)}), extra {extra[:5]} ({len(extra)})")
    aliases = alias_keys(prepared)
    copied, unexpected = copy_qat_state_by_name(prepared, state)
    if sorted(unexpected) != aliases:
        raise QATStop("state_alias_mismatch", f"the copy left {sorted(unexpected)[:5]} uncopied; the alias keys "
                                              f"are {aliases[:5]} ({len(aliases)})")
    live = prepared.state_dict()
    bad = [k for k, v in state.items()
           if not (live[k].shape == v.shape and live[k].dtype == v.dtype and torch.equal(live[k].cpu(), v.cpu()))]
    if bad:
        raise QATStop("state_copy_mismatch", f"{len(bad)} entries differ after the copy, e.g. {bad[:5]}")
    return {"entries": len(state), "copied": copied, "alias_keys": len(aliases), "equal_entries": len(state) - len(bad)}


def fake_quant_model(state: dict) -> nn.Module:
    """The epoch's fake-quant model: the prepared student with the state, eval mode, observers off."""
    from .prepare import disable_observers
    m = prepared_skeleton()
    load_state_by_name(m, state)
    disable_observers(m)
    return m.eval()


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8),
                       b.detach().reshape(-1).contiguous().view(torch.uint8))


def freeze_key_sets(prepared: nn.Module) -> tuple[list[str], list[str]]:
    """The BN statistics and the observer / scale / zero_point buffers, from the module structure."""
    bn, obs = [], []
    for mname, mod in prepared.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            bn += [f"{mname}.{b}" for b in ("running_mean", "running_var", "num_batches_tracked")]
    for mname, fq in Q.fake_quant_modules(prepared):
        obs += [f"{mname}.scale", f"{mname}.zero_point"]
        obs += [f"{mname}.activation_post_process.{b}" for b, _ in fq.activation_post_process.named_buffers()]
    return bn, obs


def freeze_cross_check(rec: dict) -> dict:
    """The freezes took: BN statistics of e11-e15 equal e10's, observer buffers of e13-e15 equal e12's."""
    bn_keys, obs_keys = freeze_key_sets(prepared_skeleton())
    states = {}
    for e in range(1, Q.EPOCHS + 1):
        p, _sha = verified_checkpoint(rec, e)
        states[e] = torch.load(p, map_location="cpu", weights_only=True)["model_state_dict"]
    bn_from, obs_from = Q.BN_FREEZE_EPOCH, Q.OBS_FREEZE_EPOCH
    bn_diff = {e: [k for k in bn_keys if not bytes_equal(states[e][k], states[bn_from][k])]
               for e in range(bn_from + 1, Q.EPOCHS + 1)}
    obs_diff = {e: [k for k in obs_keys if not bytes_equal(states[e][k], states[obs_from][k])]
                for e in range(obs_from + 1, Q.EPOCHS + 1)}
    return {"bn_keys": len(bn_keys), "observer_keys": len(obs_keys),
            "bn_equal_to_e10": {f"e{e:02d}": not v for e, v in bn_diff.items()},
            "observers_equal_to_e12": {f"e{e:02d}": not v for e, v in obs_diff.items()},
            "first_differences": {**{f"bn e{e:02d}": v[:3] for e, v in bn_diff.items() if v},
                                  **{f"obs e{e:02d}": v[:3] for e, v in obs_diff.items() if v}},
            "ok": not any(bn_diff.values()) and not any(obs_diff.values())}


# ------------------------------------------------------------------ the TorchScript identity, read without loading
def read_ts_identity(path) -> dict:
    """plantseg_int8.json from a TorchScript zip, read with zipfile (no model is loaded)."""
    from src.eval.model_loading import INT8_TORCHSCRIPT_META
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("/extra/" + INT8_TORCHSCRIPT_META)]
        if len(names) != 1:
            raise QATRefused("identity_missing", f"{path} holds {len(names)} extra/{INT8_TORCHSCRIPT_META} entries")
        return json.loads(z.read(names[0]).decode("utf-8"))


IDENTITY_KEYS = ("schema", "stage", "quantization", "num_classes", "engine", "artifact_role",
                 "source_checkpoint_sha256", "qat_checkpoint_sha256", "epoch", "run_id", "qconfig_fingerprint",
                 "state_dict_companion")


# ------------------------------------------------------------------ descriptive agreement
@torch.no_grad()
def argmax_agreement(fq: nn.Module, int8: nn.Module, inputs, lsb: float, valid_masks=None) -> dict:
    """Fake-quant vs converted argmax agreement, by top-1 / top-2 margin of the fake-quant logits in LSB.

    LSB is the scale of head.ff_add, the fake-quant step of the logits before the final upsample.
    """
    bands = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf")))
    tot = agree = 0
    by_band = {f"[{lo:g},{hi:g})": [0, 0] for lo, hi in bands}
    diffs = []
    for i, x in enumerate(inputs):
        a, b = fq(x), int8(x)
        top2 = a.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) / lsb
        same = a.argmax(1) == b.argmax(1)
        valid = torch.ones_like(same) if valid_masks is None else valid_masks[i]
        tot += int(valid.sum())
        agree += int((same & valid).sum())
        for lo, hi in bands:
            sel = valid & (margin >= lo) & (margin < hi)
            k = f"[{lo:g},{hi:g})"
            by_band[k][0] += int(sel.sum())
            by_band[k][1] += int((same & sel).sum())
        diffs.append(((a - b).abs() / lsb)[valid.unsqueeze(1).expand_as(a)].flatten())
    d = torch.cat(diffs) if diffs else torch.zeros(1)
    confident = by_band["[8,inf)"]
    return {"lsb": lsb, "valid_pixels": tot, "agreement": (agree / tot) if tot else None,
            "confident_pixels": confident[0], "confident_fraction": (confident[0] / tot) if tot else None,
            "confident_agreement": (confident[1] / confident[0]) if confident[0] else None,
            "by_margin_band": {k: {"pixels": n, "agreement": (s / n) if n else None} for k, (n, s) in by_band.items()},
            "max_logit_diff_lsb": float(d.max()), "p999_logit_diff_lsb": float(torch.quantile(d.float(), 0.999))
            if d.numel() <= 16_000_000 else None}


def head_lsb(prepared: nn.Module) -> float:
    return float(prepared.head.ff_add.activation_post_process.scale.reshape(-1)[0])


# ------------------------------------------------------------------ one epoch, in a fresh process
def convert_epoch(run_dir, eval_dir, epoch: int, *, purpose: str, host_label: str,
                  allow_smoke_inputs: bool = False, log=print) -> dict:
    """Convert one epoch. Returns the outcome; raises QATRefused, QATIncomplete or QATStop (after writing
    eNN_STOP.json)."""
    from .ptq import student_traced_in_process, write_exclusive
    from .stages import resolve_quant_stage
    from .x86_latency import load_qat_sidecar

    # 1. the fresh-process guard
    if student_traced_in_process():
        raise QATRefused("fresh_process_required", "the student was already traced in this process; convert "
                                                   "each epoch in a fresh process (qat_epoch_eval.py convert)")
    if purpose not in PURPOSES:
        raise QATRefused("purpose", f"--purpose must be one of {PURPOSES}")
    if not host_label or not str(host_label).strip():
        raise QATRefused("host_label", "--host-label is required")
    refuse_test_path(eval_dir, "eval_dir")
    rec = read_run_record(run_dir)
    require_complete(rec)
    meta = rec["run_meta"]
    if meta.get("mode") != "real" and not allow_smoke_inputs:
        raise QATRefused("run_not_real", f"the run's mode is {meta.get('mode')!r}; only a real run converts")
    st = resolve_quant_stage(meta["stage"])
    # 2. the checkpoint against its epoch_end row
    ck, ck_sha = verified_checkpoint(rec, epoch)
    E = Path(eval_dir)
    conv = E / CONVERTED_DIR
    names = epoch_names(epoch)
    paths = {k: conv / v for k, v in names.items()}
    present = [p.name for p in paths.values() if p.exists()]
    if present:
        raise QATRefused("output_exists", f"{conv} already holds {present}; an eval directory is never reused")
    # 3. weights_only, with the sidecar's kind and role checks
    payload = load_qat_sidecar(ck)
    ident = {"stage": st["name"], "epoch": epoch, "run_id": rec["run_id"], "seed": meta["seed"],
             "source_checkpoint_sha256": meta["source_checkpoint_sha256"]}
    wrong = {k: payload.get(k) for k, v in ident.items() if payload.get(k) != v}
    if wrong:
        raise QATRefused("checkpoint_identity", f"{ck} records {wrong}, the run {ident}")
    state = payload["model_state_dict"]
    skeleton = prepared_skeleton()
    kinds = Q.state_kinds(skeleton)
    never = meta.get("never_observed_modules") or []
    common = {"schema": RUN_META_SCHEMA, "lane": LANE, "stage": st["name"], "epoch": epoch, "run_id": rec["run_id"],
              "qat_checkpoint": f"{Q.EPOCH_DIR}/{ck.name}", "qat_checkpoint_sha256": ck_sha,
              "telemetry_sha256": rec["telemetry_sha256"], "purpose": purpose, "host_label": host_label,
              "cpu_model": Q.cpu_model(), "smoke_inputs": bool(allow_smoke_inputs)}
    # 4. the state predicate: a non-finite state is recorded and never converted
    ok, fails = Q.state_predicate(state, kinds, never)
    if not ok:
        conv.mkdir(parents=True, exist_ok=True)
        doc = {**common, "status": "not convertible", "rule": NOT_CONVERTIBLE_RULE, "failing": fails,
               "git_head": Q.code_identity()["git_head"], "created_wall_clock": time.time()}
        write_exclusive({paths["not_convertible"]: json_bytes(doc)})
        log(f"[convert] e{epoch:02d}: not convertible ({len(fails)} failing entries)")
        return {"epoch": epoch, "status": "not convertible", "file": str(paths["not_convertible"])}
    try:
        return _convert_checked(rec, st, epoch, ck_sha, state, skeleton, paths, names, common, purpose, host_label, log)
    except QATStop as e:
        if not paths["stop"].exists():
            conv.mkdir(parents=True, exist_ok=True)
            write_exclusive({paths["stop"]: json_bytes({**common, "status": "STOP", "code": e.code,
                                                        "message": str(e)[:2000], "created_wall_clock": time.time()})})
        raise


def _convert_checked(rec, st, epoch, ck_sha, state, skeleton, paths, names, common, purpose, host_label, log) -> dict:
    from src.eval.model_loading import (ACCURACY_ARTIFACT_ROLE, INT8_TORCHSCRIPT_SCHEMA, load_int8_torchscript,
                                        rebuild_int8_from_state_dict)
    from .prepare import convert_model, disable_observers
    from .ptq import (EXAMPLE_SHAPE, code_provenance, environment, graph_census, output_parity, qconfig_summary,
                      runtime_census, state_dict_bytes, synthetic_parity_inputs, torchscript_bytes,
                      weight_scheme_report, write_exclusive)
    from .qconfig import QUANT_BACKEND, QuantBackendUnavailable, qat_qconfig, select_qnnpack_backend
    from .runner import build_run_provenance
    meta = rec["run_meta"]
    conv = paths["torchscript"].parent
    # 5. the flags its epoch must carry
    flags = Q.flag_summary(state)
    want = [1] if epoch <= Q.OBS_FREEZE_EPOCH else [0]
    if flags["observer_enabled"] != want or flags["fake_quant_enabled"] != [1]:
        raise QATStop("stored_flags", f"e{epoch:02d} stores observer_enabled {flags['observer_enabled']} and "
                                      f"fake_quant_enabled {flags['fake_quant_enabled']}; expected {want} and [1]")
    # 6. the prepared student, rebuilt and copied by name
    copy_report = load_state_by_name(skeleton, state)
    lsb = head_lsb(skeleton)
    # 7. convert under QNNPACK, serialize, write, verify from disk
    try:
        select_qnnpack_backend()
    except QuantBackendUnavailable as e:
        raise QATRefused("backend_unavailable", str(e)) from e
    converted = convert_model(skeleton)
    qsum = qconfig_summary(qat_qconfig())
    if qsum["fingerprint"] != (meta.get("qconfig") or {}).get("fingerprint"):
        raise QATStop("qconfig_changed", f"the QAT qconfig fingerprint is {qsum['fingerprint']}; the run "
                                         f"recorded {(meta.get('qconfig') or {}).get('fingerprint')}")
    identity = {"schema": INT8_TORCHSCRIPT_SCHEMA, "stage": st["name"], "quantization": "qat",
                "num_classes": Q.NUM_CLASSES, "engine": QUANT_BACKEND, "artifact_role": ACCURACY_ARTIFACT_ROLE,
                "source_checkpoint_sha256": meta["source_checkpoint_sha256"], "qat_checkpoint_sha256": ck_sha,
                "epoch": epoch, "run_id": rec["run_id"], "qconfig_fingerprint": qsum["fingerprint"],
                "state_dict_companion": names["state_dict"]}
    if tuple(identity) != IDENTITY_KEYS:
        raise QATStop("identity_keys", f"the identity keys {tuple(identity)} differ from {IDENTITY_KEYS}")
    ts_b, ts_notes = torchscript_bytes(converted, identity)
    sd_b = state_dict_bytes(converted, st["name"], "qat")
    conv.mkdir(parents=True, exist_ok=True)
    write_exclusive({paths["torchscript"]: ts_b, paths["state_dict"]: sd_b})
    shas = {"torchscript": Q.sha256_bytes(ts_b), "state_dict": Q.sha256_bytes(sd_b)}
    for k in shas:
        if Q.sha256_file(paths[k]) != shas[k]:
            raise QATStop("artifact_write_mismatch", f"{paths[k]} on disk differs from what was serialized")
    ts_model, _ = load_int8_torchscript(paths["torchscript"], stage=st["name"], method="qat", engine=QUANT_BACKEND)
    sd_model = rebuild_int8_from_state_dict(paths["state_dict"], stage=st["name"], method="qat")
    parity = {
        "torchscript_vs_state_dict": output_parity({"torchscript": ts_model, "state_dict": sd_model},
                                                   synthetic_parity_inputs(), [("torchscript", "state_dict")]),
        "eager_vs_torchscript": output_parity({"eager": converted, "torchscript": ts_model},
                                              synthetic_parity_inputs(), [("eager", "torchscript")]),
    }
    parity["all_equal"] = all(v["all_equal"] for v in parity.values())
    weights = {"eager": weight_scheme_report(converted), "torchscript": weight_scheme_report(ts_model)}
    census = runtime_census(converted, torch.zeros(EXAMPLE_SHAPE))
    graph = graph_census(ts_model)
    fq = skeleton
    disable_observers(fq)
    fq.eval()
    agreement = argmax_agreement(fq, converted, synthetic_parity_inputs()[:2], lsb)
    checks = {"weights_per_channel_int8": all(w["ok"] for w in weights.values()),
              "no_undeclared_float_module": census["ok"], "torchscript_float_boundary": graph["ok"],
              "parity_torchscript_state_dict_eager": parity["all_equal"]}
    verification = {"checks": checks, "weights": weights, "census": census, "graph": graph, "parity": parity,
                    "state_copy": copy_report, "flags": flags,
                    "fake_quant_vs_converted_agreement": {**agreement, "inputs": "ptq.synthetic_parity_inputs()[:2]",
                                                          "descriptive": True},
                    "trace_warnings": ts_notes}
    code = code_provenance(CONVERT_CODE_PATHS)
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        per_tensor = any(w["violations"] for w in weights.values())
        float_region = any(w["fp32_convs"] for w in weights.values()) or not (
            checks["no_undeclared_float_module"] and checks["torchscript_float_boundary"])
        code_ = ("per_tensor_weight" if per_tensor else "undeclared_float_fallback" if float_region
                 else "weights_unreadable" if not checks["weights_per_channel_int8"] else "parity_mismatch")
        stop = {**common, "status": "STOP", "code": code_, "failed_checks": failed,
                "artifacts": {k: {"path": names[k], "sha256": shas[k]} for k in shas}, "verification": verification,
                "git_head": code["commit"], "created_wall_clock": time.time()}
        write_exclusive({paths["stop"]: json_bytes(stop)})
        raise QATStop(code_, f"e{epoch:02d} conversion STOP ({code_}): {failed}; see {paths['stop']}")
    source_meta = {"path": meta.get("source_checkpoint"), "sha256": meta["source_checkpoint_sha256"],
                   "bytes": meta.get("source_checkpoint_bytes")}
    prov = build_run_provenance(
        stage=st, source_meta=source_meta, method="qat",
        qconfig_summary={"activation": qsum["activation"], "weight": qsum["weight"],
                         "fingerprint": qsum["fingerprint"]},
        calibration=None,
        training={"run_id": rec["run_id"], "epoch": epoch, "step": rec["ends"][epoch]["step"], "seed": meta["seed"],
                  "clip_norm": meta["clip_norm"], "clip_source": meta.get("clip_source"), "epochs": Q.EPOCHS,
                  "steps_per_epoch": meta["steps_per_epoch"], "mode": meta["mode"],
                  "telemetry_sha256": rec["telemetry_sha256"],
                  "fake_quant_val_all_class_miou": rec["ends"][epoch].get("fake_quant_val_all_class_miou"),
                  "fake_quant_val_note": "recorded during training; never selects (AM-4a item 2)"},
        converted_path=paths["torchscript"], coverage=None,
        forward={"ok": True, "output_shape": census["output"]["shape"], "dtype": census["output"]["dtype"],
                 "error": None},
        backend=QUANT_BACKEND)
    prov.update({**common,
                 "artifact_format": "torchscript",
                 "converted_artifact": names["torchscript"],
                 "converted_artifact_sha256": shas["torchscript"], "converted_artifact_bytes": len(ts_b),
                 "converted_artifact_role": ACCURACY_ARTIFACT_ROLE,
                 "state_dict_artifact": {"path": names["state_dict"], "sha256": shas["state_dict"],
                                         "bytes": len(sd_b), "schema": "src/quant/runner.py converted state_dict",
                                         "role": "companion of the TorchScript artifact of record"},
                 "identity": identity, "engine": QUANT_BACKEND,
                 "observers": {"activation": qsum["activation"]["observer_class"],
                               "weight": qsum["weight"]["observer_class"]},
                 "verification": verification, "conversion": "data-free (ptq.synthetic_parity_inputs for parity)",
                 "git_head": code["commit"], "code": code, "environment": environment(),
                 "artifact_bytes_note": ("the TorchScript file's debug records keep absolute source paths and the "
                                         "call stack of the trace, so its sha256 reproduces for the same invocation "
                                         "(scripts/qat_epoch_eval.py convert-epoch, same checkout path and "
                                         "interpreter, one epoch per process)"),
                 "status": "converted", "created_wall_clock": time.time()})
    write_exclusive({paths["run_meta"]: json_bytes(prov)})
    log(f"[convert] e{epoch:02d}: {names['torchscript']} {shas['torchscript'][:12]}, all checks passed")
    return {"epoch": epoch, "status": "converted", "run_meta": str(paths["run_meta"]),
            "torchscript_sha256": shas["torchscript"], "state_dict_sha256": shas["state_dict"]}


__all__ = ["CONVERTED_DIR", "IDENTITY_KEYS", "NOT_CONVERTIBLE_RULE", "PURPOSES", "QATIncomplete", "SCORES_DIR",
           "alias_keys", "argmax_agreement", "bytes_equal", "convert_epoch", "epoch_names", "fake_quant_model",
           "freeze_cross_check", "freeze_key_sets", "head_lsb", "json_bytes", "load_state_by_name",
           "prepared_skeleton", "read_run_record", "read_ts_identity", "refuse_test_path", "require_complete",
           "verified_checkpoint"]
