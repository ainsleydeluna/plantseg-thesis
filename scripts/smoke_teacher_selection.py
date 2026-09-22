#!/usr/bin/env python3
"""B62 — M12 checkpoint selection, M13 CE normalisation, the init-compatibility rule and R3 wiring.

Teacher image, CPU, synthetic data in a temp dir (no PlantSeg data, no checkpoint download).

M12 is exercised through REAL MMEngine `Runner.train()` mini-runs (tiny iteration counts, the real
CheckpointHook, the real ValLoop and the thesis hooks). A scripted subclass of the thesis metric
injects chosen full-precision values, so the selection rules can be tested exactly:
  * run A — a rounded FALSE TIE (0.312341 vs 0.312344 both display as 31.23): full precision selects
    the later, higher checkpoint; a selection on the rounded value would pick the earlier one;
  * run B — an EXACT tie: the earliest iteration wins; deleting/altering the selected checkpoint is
    caught by the verifier;
  * run C — retention negative control: max_keep_ckpts=1 deletes the selected earlier checkpoint and
    the end-of-run assertion fails closed;
  * run D — an init checkpoint violating the classifier-only rule fails closed at load;
  * run E — a run without the M4-V hook has no ordered VAL manifest, so no selection record is written.
The ordered VAL manifest (one identity: `src.eval.artifacts.hash_split_manifest`) is checked end to end:
every record carries the same non-empty hash, teacher_selection.json carries it, the thesis evaluator's
own listing reproduces it, and R3 refuses before evaluating when order or membership differs, or when
the selection or the evaluator artifact (schema: `dataset.split_manifest_sha256`) lacks it.
Exit codes: 0 pass · 1 a check fails · 2 the teacher stack is not importable.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
STOCK = "/usr/local/lib/python3.11/site-packages/mmseg/.mim/configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py"
TMP = Path(tempfile.mkdtemp(prefix="smoke_teacher_selection_"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def staged_root() -> Path:
    import numpy as np
    from PIL import Image
    rs = np.random.RandomState(11)
    root = TMP / "staged"
    for split, n in (("train", 4), ("val", 3)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(n):
            h, w = int(rs.randint(200, 600)), int(rs.randint(200, 600))
            Image.fromarray((rs.rand(h, w, 3) * 255).astype(np.uint8)).save(root / "images" / split / f"{split}_{i}.jpg")
            Image.fromarray(rs.randint(0, 6, (h, w)).astype(np.uint8), mode="L").save(
                root / "annotations" / split / f"{split}_{i}.png")
    return root


def run_cfg(root: Path, work: Path, *, max_iters: int, load_from: str, keep: int = -1):
    from mmengine.config import Config
    cfg = Config.fromfile(str(CONFIG))
    for dl in (cfg.train_dataloader, cfg.val_dataloader):
        dl.dataset.data_root = str(root)
    cfg.train_dataloader.update(batch_size=2, num_workers=0, persistent_workers=False)
    cfg.val_dataloader.update(num_workers=0, persistent_workers=False)
    cfg.train_cfg.update(max_iters=max_iters, val_interval=2)
    cfg.default_hooks.checkpoint.update(interval=2, max_keep_ckpts=keep)
    cfg.val_evaluator = dict(type="ScriptedThesisMetric", num_classes=116, ignore_index=255, background_index=0)
    cfg.work_dir = str(work)
    cfg.load_from = load_from
    cfg.randomness = dict(seed=42)          # the G18 strict policy is verified by smoke_teacher_runner
    cfg.default_hooks.logger.interval = 1
    return cfg


@contextlib.contextmanager
def evaluator_root(root: Path, n_val: int = 3):
    """Point the thesis evaluator's own VAL listing (configs.data.DATA, EXPECTED_SPLIT_ROWS) at `root`."""
    from configs.data import DATA
    import scripts.evaluate_model as em
    saved = (DATA["root"], DATA["splits"]["sizes"]["val"], em.EXPECTED_SPLIT_ROWS["val"])
    DATA["root"], DATA["splits"]["sizes"]["val"], em.EXPECTED_SPLIT_ROWS["val"] = str(root), n_val, n_val
    try:
        yield
    finally:
        DATA["root"], DATA["splits"]["sizes"]["val"], em.EXPECTED_SPLIT_ROWS["val"] = saved


def train(cfg, script: list[float]):
    from mmengine.runner import Runner
    import src.training.teacher_components as tc  # noqa: F401 — registers the thesis components
    SCRIPTED.script[:] = list(script)
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner


SCRIPTED = SimpleNamespace(script=[])


def main() -> int:
    try:
        import torch
        import torch.nn.functional as F
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
    except ImportError as e:
        print(f"ENVIRONMENT: teacher stack not importable ({e}); run in the plantseg-teacher image.")
        return 2
    os.chdir(REPO)
    Config.fromfile(str(CONFIG))                    # imports the thesis components (custom_imports)
    init_default_scope("mmseg")
    import src.training.teacher_components as tc
    from mmseg.registry import METRICS, MODELS

    @METRICS.register_module(force=True)
    class ScriptedThesisMetric(tc.ThesisConfusionMIoUMetric):
        """The real thesis metric, with the selection value replaced by a scripted one (test only)."""

        def compute_metrics(self, results_):
            out = super().compute_metrics(results_)
            value = SCRIPTED.script.pop(0)
            out[tc.SELECTION_KEY] = value
            out[tc.DISPLAY_KEY] = round(value * 100, 2)
            return out

    # ---- M13 ----
    cfg = Config.fromfile(str(CONFIG))
    loss = MODELS.build(cfg.model.decode_head.loss_decode)
    torch.manual_seed(0)
    logits = torch.randn(2, 116, 8, 8, requires_grad=True)
    label = torch.randint(0, 116, (2, 8, 8))
    label[:, :, :3] = 255
    value = loss(logits, label, ignore_index=255)
    value.backward()
    per_px = F.cross_entropy(logits.detach(), label, ignore_index=255, reduction="none")
    check("m13_equals_valid_pixel_mean", abs(float(value) - float(F.cross_entropy(logits.detach(), label,
                                                                                    ignore_index=255))) < 1e-6)
    check("m13_denominator_excludes_ignored", abs(float(value) - float(per_px.sum() / (label != 255).sum())) < 1e-6
          and abs(float(value) - float(per_px.mean())) > 1e-3, "the MMSeg default would give the all-pixel mean")
    check("m13_ignored_pixels_zero_loss", float(per_px[label == 255].abs().max()) == 0.0)
    check("m13_ignored_pixels_zero_gradient", float(logits.grad.permute(0, 2, 3, 1)[label == 255].abs().max()) == 0.0)
    all_ignore = torch.randn(1, 116, 4, 4, requires_grad=True)
    ai = loss(all_ignore, torch.full((1, 4, 4), 255), ignore_index=255)
    check("m13_all_ignore_finite_zero", bool(torch.isfinite(ai)) and float(ai) == 0.0,
          "sum / (0 + eps) under the pinned mmseg 1.2.2")
    check("m13_no_class_weight", loss.class_weight is None and loss.avg_non_ignore is True)

    # ---- metric: full precision vs display ----
    met = tc.ThesisConfusionMIoUMetric()
    pred = torch.randint(0, 116, (512, 512))
    gt = pred.clone()
    gt[:200] = 3
    gt[:, :5] = 255
    met.process(None, [{"pred_sem_seg": {"data": pred[None]}, "gt_sem_seg": {"data": gt[None]}}])
    out = met.compute_metrics(met.results)
    from src.eval.metrics import confusion_matrix, miou_from_confusion
    ref = miou_from_confusion(confusion_matrix(pred, gt, 116, ignore_index=255))
    check("m12_metric_is_thesis_reducer_full_precision", out["mIoU_full"] == ref
          and out["mIoU"] == round(ref * 100, 2), f"{out['mIoU_full']!r} vs display {out['mIoU']}")

    # ---- init checkpoints: a stock-shaped 150-class model; and one violating the rule ----
    stock_cfg = Config.fromfile(STOCK)
    stock_cfg.model.backbone.init_cfg = None
    torch.manual_seed(1)
    stock = MODELS.build(stock_cfg.model)
    good_init = TMP / "ade20k_like_150.pth"
    torch.save({"meta": {}, "state_dict": stock.state_dict()}, good_init)
    bad_state = dict(stock.state_dict())
    bad_state["backbone.patch_embed1.proj.0.weight"] = torch.zeros(16, 3, 3, 3)
    bad_init = TMP / "ade20k_like_bad.pth"
    torch.save({"meta": {}, "state_dict": bad_state}, bad_init)
    root = staged_root()

    # ---- run A: rounded false tie ----
    work_a = TMP / "run_a"
    runner = train(run_cfg(root, work_a, max_iters=4, load_from=str(good_init)), [0.312341, 0.312344])
    records = [json.loads(line) for line in (work_a / tc.RECORDS_FILE).read_text().splitlines()]
    selection = json.loads((work_a / tc.SELECTION_FILE).read_text())
    hub = runner.message_hub
    check("m12_checkpointhook_sees_mIoU_full_key",
          hub.get_info("best_score") == 0.312344 and (work_a / "best_mIoU_full_iter_4.pth").is_file(),
          f"best_score {hub.get_info('best_score')!r}, best file best_mIoU_full_iter_4.pth")
    check("m12_full_precision_selects_later_higher", selection["selected_iteration"] == 4
          and selection["selected_mIoU_full"] == 0.312344 and selection["status"] == "final")
    rounded_pick = tc.select_checkpoint([{**r, "d": r["display_mIoU_percent"]} for r in records], "d")
    check("m12_negative_control_rounded_value_would_pick_earlier", rounded_pick["iteration"] == 2
          and records[0]["display_mIoU_percent"] == records[1]["display_mIoU_percent"] == 31.23,
          "both display 31.23 -> a rounded selection keeps iteration 2")
    check("m12_records_every_validation", [r["iteration"] for r in records] == [2, 4]
          and all(r["nmf_seed"] == 42 and len(r["val_manifest_sha256"]) == 64 and r["val_images"] == 3
                  and len(r["checkpoint_sha256"]) == 64 for r in records))
    manifest_a = records[0]["val_manifest_sha256"]
    check("manifest_every_record_nonempty_and_identical",
          all(isinstance(r["val_manifest_sha256"], str) and r["val_manifest_sha256"] for r in records)
          and len({r["val_manifest_sha256"] for r in records}) == 1, manifest_a)
    check("manifest_persisted_in_teacher_selection_json", selection.get("val_manifest_sha256") == manifest_a)
    from src.eval.artifacts import hash_split_manifest
    from src.eval.evaluate import ManifestEntry
    import scripts.teacher_readiness_r3 as R3
    val_stems = sorted(p.stem for p in (root / "images" / "val").iterdir())
    check("manifest_is_the_evaluator_ordered_hash", manifest_a == hash_split_manifest(
        [ManifestEntry(i, s, s) for i, s in enumerate(val_stems)]), "hash_split_manifest(position, stem, stem)")
    with evaluator_root(root):
        evaluator_rows = R3.evaluator_val_manifest()
    check("manifest_selection_equals_evaluator_listing",
          hash_split_manifest(evaluator_rows) == manifest_a and [r.image_id for r in evaluator_rows] == val_stems,
          "MMSeg VAL pass order == thesis evaluator order (same stems, same positions)")
    check("m12_selected_sha_matches_file", selection["checkpoint_sha256"] == sha256(work_a / "iter_4.pth"))
    check("m12_all_checkpoints_retained", (work_a / "iter_2.pth").is_file() and (work_a / "iter_4.pth").is_file())
    init = hub.get_info("init_compatibility")
    check("init_classifier_only_rule_passes", init["pass"] is True
          and init["classifier_mismatch"] == ["decode_head.conv_seg.bias", "decode_head.conv_seg.weight"]
          and init["missing"] == [] and init["unexpected"] == [] and init["source_classes"] == 150
          and init["model_classes"] == 116, f"{init['matched_same_shape']} matched")

    # ---- run B: exact tie -> earliest; deletion / alteration caught ----
    work_b = TMP / "run_b"
    train(run_cfg(root, work_b, max_iters=6, load_from=str(good_init)), [0.5, 0.5, 0.4])
    sel_b = json.loads((work_b / tc.SELECTION_FILE).read_text())
    check("m12_exact_tie_keeps_earliest", sel_b["selected_iteration"] == 2
          and (work_b / "best_mIoU_full_iter_2.pth").is_file(), str(sel_b["validations"]))
    records_b = [json.loads(line) for line in (work_b / tc.RECORDS_FILE).read_text().splitlines()]
    check("manifest_run_b_three_validations_one_manifest",
          [r["iteration"] for r in records_b] == [2, 4, 6]
          and {r["val_manifest_sha256"] for r in records_b} == {manifest_a}
          and sel_b.get("val_manifest_sha256") == manifest_a)
    tc.verify_selected_checkpoint(sel_b)
    target = Path(sel_b["checkpoint_path"])
    original = target.read_bytes()
    target.write_bytes(original + b"x")
    try:
        tc.verify_selected_checkpoint(sel_b)
        altered = False
    except RuntimeError:
        altered = True
    target.unlink()
    try:
        tc.verify_selected_checkpoint(sel_b)
        deleted = False
    except RuntimeError:
        deleted = True
    check("m12_selected_checkpoint_alteration_caught", altered)
    check("m12_selected_checkpoint_deletion_caught", deleted)

    # ---- run C: retention negative control (max_keep_ckpts=1 deletes the earlier selection) ----
    try:
        train(run_cfg(root, TMP / "run_c", max_iters=4, load_from=str(good_init), keep=1), [0.9, 0.1])
        caught = ""
    except RuntimeError as e:
        caught = str(e)
    check("m12_end_of_run_assertion_catches_lost_selection", "no longer exists" in caught, caught[:90])

    # ---- run D: init checkpoint violating the classifier-only rule ----
    try:
        train(run_cfg(root, TMP / "run_d", max_iters=2, load_from=str(bad_init)), [0.1])
        refused = ""
    except RuntimeError as e:
        refused = str(e)
    check("init_non_classifier_mismatch_fails_closed", "classifier-only rule" in refused, refused[:90])

    # ---- the ten official validations must share one non-empty ordered VAL manifest ----
    def ten(**override):
        recs = [{"iteration": 4000 * k, "val_manifest_sha256": manifest_a} for k in range(1, 11)]
        for it, value in override.items():
            rec = recs[int(it[1:]) // 4000 - 1]
            if value is KeyError:
                rec.pop("val_manifest_sha256")
            else:
                rec["val_manifest_sha256"] = value
        return recs

    def raised_msg(fn):
        try:
            fn()
            return ""
        except RuntimeError as e:
            return str(e)

    check("manifest_ten_identical_records_accepted",
          tc.require_uniform_val_manifest(ten()) == manifest_a and len(ten()) == 10)
    check("manifest_one_different_validation_fails_closed",
          "different VAL manifests" in raised_msg(lambda: tc.require_uniform_val_manifest(ten(i20000="b" * 64))))
    check("manifest_absent_fails_closed",
          "lacks its ordered VAL manifest" in raised_msg(lambda: tc.require_uniform_val_manifest(ten(i8000=KeyError))))
    check("manifest_null_fails_closed",
          "lacks its ordered VAL manifest" in raised_msg(lambda: tc.require_uniform_val_manifest(ten(i40000=None))))
    check("manifest_empty_fails_closed",
          "lacks its ordered VAL manifest" in raised_msg(lambda: tc.require_uniform_val_manifest(ten(i4000=""))))

    # The record hook's own after_val_epoch path (not only the helper): a changed manifest, a stale M4-V
    # record and an empty manifest are refused before any record is written.
    from mmengine.logging import MessageHub
    from mmengine.runner import BaseLoop

    class _Loop(BaseLoop):
        def run(self):
            pass

    stub_dir = TMP / "stub_hook"
    stub_dir.mkdir()
    (stub_dir / "iter_8000.pth").write_bytes(b"x")
    hub_stub = MessageHub.get_instance("smoke_selection_stub")
    stub = SimpleNamespace(iter=8000, work_dir=str(stub_dir), message_hub=hub_stub,
                           _train_loop=_Loop(runner=None, dataloader=[]))
    metrics = {tc.SELECTION_KEY: 0.5, tc.DISEASE_KEY: 0.4, tc.DISPLAY_KEY: 50.0, "val_images": 3}

    def hook_with_first_record():
        h = tc.TeacherSelectionRecordHook()
        h._active = True
        h.records = [{"iteration": 4000, "val_manifest_sha256": manifest_a}]
        return h

    hub_stub.update_info("m4v/last_pass", {"iteration": 8000, "seed": 42, "manifest_sha256": "c" * 64})
    changed = raised_msg(lambda: hook_with_first_record().after_val_epoch(stub, metrics))
    hub_stub.update_info("m4v/last_pass", {"iteration": 4000, "seed": 42, "manifest_sha256": manifest_a})
    stale = raised_msg(lambda: hook_with_first_record().after_val_epoch(stub, metrics))
    hub_stub.update_info("m4v/last_pass", {"iteration": 8000, "seed": 42, "manifest_sha256": ""})
    empty = raised_msg(lambda: hook_with_first_record().after_val_epoch(stub, metrics))
    check("manifest_hook_refuses_changed_order_or_membership", "different VAL manifests" in changed, changed[:80])
    check("manifest_hook_refuses_stale_m4v_record", "no M4-V pass record" in stale, stale[:80])
    check("manifest_hook_refuses_empty_manifest", "lacks its ordered VAL manifest" in empty
          and not (stub_dir / tc.RECORDS_FILE).exists(), "no record written")

    # ---- run E: no M4-V hook -> no ordered VAL manifest -> no selection record (real Runner) ----
    cfg_e = run_cfg(root, TMP / "run_e", max_iters=2, load_from=str(good_init))
    cfg_e.custom_hooks = [h for h in cfg_e.custom_hooks if h["type"] != "TeacherNMFEvalStreamHook"]
    no_m4v = raised_msg(lambda: train(cfg_e, [0.2]))
    check("manifest_run_without_m4v_hook_fails_closed", "no M4-V pass record" in no_m4v
          and not (TMP / "run_e" / tc.SELECTION_FILE).exists(), no_m4v[:90])

    # ---- official cadence ----
    official = Config.fromfile(str(CONFIG))
    cadence = list(range(official.train_cfg.val_interval, official.train_cfg.max_iters + 1,
                         official.train_cfg.val_interval))
    check("m12_official_cadence_4k_to_40k", cadence == [4000 * k for k in range(1, 11)]
          and official.default_hooks.checkpoint.interval == 4000
          and official.default_hooks.checkpoint.max_keep_ckpts == -1, f"{len(cadence)} validations")

    # ---- R3: ordered-manifest gate BEFORE evaluation; artifact manifest (real schema) AFTER ----
    import shutil
    from mmseg.registry import DATASETS
    import scripts.teacher_readiness_r3 as R3
    check("r3_strictly_greater_no_margin", R3.decide(R3.E1_VAL_ALL_CLASS_MIOU)["readiness_verdict"] == "FAIL"
          and R3.decide(R3.E1_VAL_ALL_CLASS_MIOU + 1e-12)["readiness_verdict"] == "PASS"
          and R3.E1_VAL_ALL_CLASS_MIOU == 0.36314016580581665)
    ckpt = TMP / "r3" / "iter_8.pth"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"selected-teacher")

    def selection_file(name: str, **fields) -> Path:
        """A final selection whose manifest is run A's REAL M4-V pass manifest (unless overridden)."""
        body = {"status": "final", "selected_iteration": 8, "selected_mIoU_full": 0.5,
                "checkpoint_sha256": sha256(ckpt), "val_manifest_sha256": manifest_a, **fields}
        path = TMP / "r3" / name
        path.write_text(json.dumps({k: v for k, v in body.items() if v is not KeyError}))
        return path

    sel_path = selection_file("teacher_selection.json")

    def args(tag, selection=None, **kw):
        base = dict(checkpoint=str(ckpt), teacher_config=str(CONFIG), selection=str(selection or sel_path),
                    out_dir=str(TMP / "r3" / f"art_{tag}"), readiness_json=str(TMP / "r3" / f"r3_{tag}.json"),
                    device="cpu", run_id=None)
        base.update(kw)
        return SimpleNamespace(**base)

    ev = R3.evaluator_args(args("argcheck"))
    check("r3_evaluator_args_batch1_provisional_val_teacher",
          ev.batch_size == 1 and ev.artifact_status == "provisional" and ev.split == "val"
          and ev.model_role == "teacher" and ev.teacher_config == str(CONFIG))

    evaluated: list[str] = []

    def fake_eval(miou, sha=None, manifest="evaluator"):
        """Writes summary.json in the REAL frozen schema (src/eval/artifacts.build_summary): the split
        manifest lives under "dataset" and is what scripts/evaluate_model.run computes for R3 —
        hash_split_manifest of build_expected_manifest_for(split, range(EXPECTED_SPLIT_ROWS[split]))."""
        def _run(ev_args, teacher_builder=None):
            import scripts.evaluate_model as em
            from src.eval.adapters import build_expected_manifest_for
            evaluated.append(ev_args.out_dir)
            out_dir = Path(ev_args.out_dir)
            out_dir.mkdir(parents=True)
            rows = build_expected_manifest_for(ev_args.split, list(range(em.EXPECTED_SPLIT_ROWS[ev_args.split])))
            dataset = {"split": ev_args.split, "split_manifest_sha256": hash_split_manifest(rows)}
            run = {"artifact_status": ev_args.artifact_status, "model_role": ev_args.model_role,
                   "checkpoint_sha256": sha or sha256(Path(ev_args.checkpoint))}
            if manifest is KeyError:
                dataset.pop("split_manifest_sha256")
            elif manifest == "run_only":                  # the superseded wrong schema assumption
                run["split_manifest_sha256"] = dataset.pop("split_manifest_sha256")
            elif manifest != "evaluator":
                dataset["split_manifest_sha256"] = manifest
            (out_dir / "summary.json").write_text(json.dumps({
                "dataset": dataset, "run": run,
                "dataset_level": {"all_class_miou": miou, "disease_only_miou": miou - 0.05}}))
            return out_dir
        return _run

    def r3_refusal(tag, selection=None, evaluate=None, resolve_manifest=None):
        """(refusal code or None, evaluator calls made, readiness record written?)."""
        before = len(evaluated)
        a = args(tag, selection)
        try:
            R3.run_r3(a, evaluate=evaluate or fake_eval(0.41), resolve_manifest=resolve_manifest)
            code = None
        except R3.R3Refused as e:
            code = e.code
        return code, len(evaluated) - before, Path(a.readiness_json).exists()

    with evaluator_root(root):
        code, rec = R3.run_r3(args("pass"), evaluate=fake_eval(0.41))
        check("r3_pass_path", code == R3.EXIT_PASS and rec["readiness_verdict"] == "PASS"
              and rec["role"] == "same_val_operational_floor" and rec["artifact_status"] == "provisional"
              and Path(args("pass").readiness_json).is_file())
        check("r3_matching_ordered_manifest_passes_and_is_recorded",
              rec["split_manifest_sha256"] == manifest_a and rec["val_manifest"]["pre_evaluation_sha256"] == manifest_a
              and rec["val_manifest"]["selection_val_manifest_sha256"] == manifest_a
              and rec["selection"]["val_manifest_sha256"] == manifest_a and rec["val_manifest"]["rows"] == 3,
              manifest_a)
        code, rec = R3.run_r3(args("fail"), evaluate=fake_eval(0.30))
        check("r3_fail_path_exit_3", code == R3.EXIT_FAIL and rec["readiness_verdict"] == "FAIL")
        other = r3_refusal("other", evaluate=fake_eval(0.5, sha="f" * 64))
        check("r3_refuses_evaluator_scoring_another_checkpoint", other[0] == "r3_refused" and not other[2])

        # ORDER: the same VAL members in a different order must be refused before any evaluation.
        rows = R3.evaluator_val_manifest()
        swapped = [ManifestEntry(r.manifest_index, rows[j].image_id, rows[j].image_id)
                   for r, j in zip(rows, [1, 0, 2])]
        reorder = r3_refusal("reorder", resolve_manifest=lambda: swapped)
        check("r3_reordered_manifest_refused_before_evaluation",
              reorder == ("val_manifest_mismatch", 0, False)
              and sorted(r.image_id for r in swapped) == sorted(r.image_id for r in rows), str(reorder))
        ds_cfg = Config.fromfile(str(CONFIG)).val_dataloader.dataset
        ds_cfg.data_root = str(root)
        ds_cfg.indices = [2, 1, 0]
        rev_stems, rev_manifest = tc.val_manifest(DATASETS.build(ds_cfg))
        rev_sel = selection_file("sel_reversed_pass.json", val_manifest_sha256=rev_manifest)
        rev = r3_refusal("rev_pass", selection=rev_sel)
        check("r3_selection_pass_in_other_order_refused",
              rev == ("val_manifest_mismatch", 0, False) and sorted(rev_stems) == val_stems != rev_stems,
              f"real MMSeg pass order {rev_stems} vs evaluator {val_stems}")

        # MEMBERSHIP: one VAL stem replaced must be refused before any evaluation.
        replaced = rows[:-1] + [ManifestEntry(rows[-1].manifest_index, "val_9", "val_9")]
        member = r3_refusal("member", resolve_manifest=lambda: replaced)
        check("r3_changed_membership_refused_before_evaluation", member == ("val_manifest_mismatch", 0, False),
              str(member))
        alt = TMP / "staged_alt"
        shutil.copytree(root, alt)
        (alt / "images" / "val" / "val_2.jpg").rename(alt / "images" / "val" / "val_9.jpg")
        (alt / "annotations" / "val" / "val_2.png").rename(alt / "annotations" / "val" / "val_9.png")
        with evaluator_root(alt):
            member_real = r3_refusal("member_real")
        check("r3_changed_membership_real_listing_refused", member_real == ("val_manifest_mismatch", 0, False),
              str(member_real))

        # MISSING / NULL / EMPTY selection manifest: refused before any evaluation.
        for label, value in (("absent", KeyError), ("null", None), ("empty", "")):
            got = r3_refusal(f"selmiss_{label}", selection=selection_file(f"sel_{label}.json",
                                                                          val_manifest_sha256=value))
            check(f"r3_selection_manifest_{label}_refused", got == ("val_manifest_missing", 0, False), str(got))

        # MISSING / NULL / wrong-schema / different ARTIFACT manifest: refused, no readiness record.
        for label, value, want in (("absent", KeyError, "artifact_manifest_missing"),
                                   ("null", None, "artifact_manifest_missing"),
                                   ("run_block_only", "run_only", "artifact_manifest_missing"),
                                   ("different", "0" * 64, "artifact_manifest_mismatch")):
            got = r3_refusal(f"artmiss_{label}", evaluate=fake_eval(0.41, manifest=value))
            check(f"r3_artifact_manifest_{label}_refused", got == (want, 1, False), str(got))

    wrong_sel = TMP / "r3" / "wrong_selection.json"
    wrong_sel.write_text(json.dumps({"status": "final", "checkpoint_sha256": "0" * 64}))
    try:
        R3.verify_selection(ckpt, wrong_sel)
        refused_sha = False
    except R3.R3Refused:
        refused_sha = True
    check("r3_refuses_non_selected_checkpoint", refused_sha)
    running = TMP / "r3" / "running_selection.json"
    running.write_text(json.dumps({"status": "running", "checkpoint_sha256": sha256(ckpt)}))
    try:
        R3.verify_selection(ckpt, running)
        refused_running = False
    except R3.R3Refused:
        refused_running = True
    check("r3_refuses_non_final_selection", refused_running)
    saved_run = R3.run_r3
    R3.run_r3 = lambda a: (R3.EXIT_FAIL, {"readiness_verdict": "FAIL"})
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = R3.main(["--checkpoint", "x", "--teacher-config", "y", "--selection", "z",
                      "--out-dir", "o", "--readiness-json", "r"])
    R3.run_r3 = saved_run
    check("r3_failure_prints_stop", rc == 3 and "STOP" in err.getvalue() and "No retraining" in err.getvalue())

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:52}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
