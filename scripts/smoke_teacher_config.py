#!/usr/bin/env python3
"""Verification of the SegNeXt-B teacher fine-tune config through MMEngine's own loader.

The config is loaded with `mmengine.config.Config.fromfile` — the call `mmseg.apis.init_model` makes — from
the repository root, and the checks run on the MERGED config (the thesis deltas plus everything `_base_`
supplies), so a config the production loader cannot parse fails here. CPU only: nothing here builds a model,
reads the dataset, opens a checkpoint, uses the network or trains anything. It asserts that the locked B1
recipe and every B60/B61 lock (M2-M5, M11-M13; implemented by B62) are what the loaded config actually says,
and that no student-stage setting leaked into the teacher.
Needs mmengine 0.10.7 + mmsegmentation 1.2.2 (the teacher stack).

Exit codes: 0 all checks pass · 1 the config does not load or a check fails · 2 mmengine is not importable.
"""
from __future__ import annotations

import ast
import io
import json
import os
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = REPO / "configs" / "teacher" / "segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
CLASS_MAP = REPO / "configs" / "plantseg_class_map.json"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def strip_comments(src: str) -> str:
    """Return the config's CODE only, with `#` comments removed.

    Token scans must not read the file's own prose: the header comment legitimately names the very
    things being forbidden ("no KD, no CWD, no QAT", "No MMSeg 0.x keys … total_iters"), so scanning
    raw source produces false positives. String literals are preserved.
    """
    out, last_row, last_col = [], 1, 0
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        ttype, text, (srow, scol), (erow, ecol), _ = tok
        if srow > last_row:
            out.append("\n" * (srow - last_row))
            last_col = 0
        if scol > last_col:
            out.append(" " * (scol - last_col))
        if ttype != tokenize.COMMENT:
            out.append(text)
        last_row, last_col = erow, ecol
    return "".join(out)


def config_types(obj):
    """Every `type` value anywhere in a merged config (nested dicts and lists)."""
    if isinstance(obj, dict):
        yield obj.get("type")
        for value in obj.values():
            yield from config_types(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from config_types(value)


def load_config(env: dict[str, str] | None = None) -> dict:
    """Load the MERGED config through mmengine's Config.fromfile, optionally with env vars applied."""
    from mmengine.config import Config

    saved = {k: os.environ.get(k) for k in (env or {})}
    try:
        for k, v in (env or {}).items():
            os.environ[k] = v
        return Config.fromfile(str(CONFIG)).to_dict()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    try:
        import mmengine
    except ImportError as e:
        print(f"ENVIRONMENT: mmengine is not importable ({e}). Run inside the teacher stack "
              "(mmengine 0.10.7 + mmsegmentation 1.2.2), e.g. the plantseg-teacher image.")
        return 2
    # Load from the repository root, as launches do: mmengine decides whether a file is a `_base_` config or
    # a lazy-import config relative to the working directory, so a load from elsewhere can pass while the
    # operational load fails.
    os.chdir(REPO)

    print("=" * 78)
    print("TEACHER FINE-TUNE CONFIG — verification through mmengine Config.fromfile (CPU only)")
    print(f"config: {CONFIG.relative_to(REPO)} | mmengine {mmengine.__version__} | cwd: {Path.cwd()}")
    print("=" * 78)

    src = CONFIG.read_text(encoding="utf-8")
    code = strip_comments(src)          # token scans run against CODE ONLY, never the comments
    try:
        cfg = load_config()
    except Exception as e:  # noqa: BLE001 — any loader failure is the finding; report it verbatim
        print(f"\n[LOAD] Config.fromfile failed: {type(e).__name__}: {e}")
        print("\nRESULT: FAIL (the config does not load through mmengine)")
        return 1
    check("config_loads_via_mmengine_config_fromfile", "model" in cfg, f"cwd={Path.cwd()}")

    # ---- provenance classification (A3) ----
    check("classified_thesis_derived", cfg["PROTOCOL_CLASSIFICATION"] == "thesis-derived",
          cfg["PROTOCOL_CLASSIFICATION"])
    check("records_public_source_commit",
          cfg["PUBLIC_PLANTSEG_SOURCE_COMMIT"] == "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0")
    check("published_42_05_is_contextual_only",
          cfg["PUBLISHED_MSCAN_B_REFERENCE"]["miou"] == 42.05
          and cfg["PUBLISHED_MSCAN_B_REFERENCE"]["role"] == "contextual-reference-only",
          str(cfg["PUBLISHED_MSCAN_B_REFERENCE"]))
    check("published_protocol_recorded_as_not_used",
          "NOT used here" in cfg["PUBLISHED_BENCHMARK_OPTIMIZER"],
          "paper trains benchmarks with SGD; this config does not")
    check("no_verbatim_reproduction_claim",
          "verbatim" not in code and "reproduction" not in code,
          "config code makes no published-B reproduction claim")

    # ---- class space ----
    classes = cfg["PLANTSEG_CLASSES"]
    authoritative = [e["name"] for e in sorted(json.loads(CLASS_MAP.read_text(encoding="utf-8"))["entries"],
                                               key=lambda x: x["class_id"])]
    check("class_count_is_116", len(classes) == 116, f"{len(classes)}")
    check("classes_match_authoritative_map_verbatim", list(classes) == authoritative,
          "configs/plantseg_class_map.json")
    check("decode_head_num_classes_116", cfg["model"]["decode_head"]["num_classes"] == 116)
    check("ignore_index_255", cfg["model"]["decode_head"]["ignore_index"] == 255
          and cfg["data_preprocessor"]["seg_pad_val"] == 255)
    check("background_index_0", cfg["BACKGROUND_INDEX"] == 0)

    # ---- dataset (M11: TRAIN and VAL only; no active TEST surface) ----
    for split, loader in (("train", "train_dataloader"), ("val", "val_dataloader")):
        ds = cfg[loader]["dataset"]
        check(f"{split}_reduce_zero_label_false", ds["reduce_zero_label"] is False)
        check(f"{split}_uses_{split}_split",
              ds["data_prefix"] == {"img_path": f"images/{split}",
                                    "seg_map_path": f"annotations/{split}"},
              str(ds["data_prefix"]))
    check("m11_no_active_test_surface",
          cfg["test_dataloader"] is None and cfg["test_evaluator"] is None and cfg["test_cfg"] is None,
          "test_dataloader / test_evaluator / test_cfg are all None (MMEngine all-or-none rule)")
    check("m11_no_test_pipeline_or_tta",
          cfg["test_pipeline"] is None and cfg.get("tta_pipeline") is None,
          "the inherited upstream (2048, 512) pipelines are removed")
    check("m11_no_test_path_anywhere_in_code", "images/test" not in code and "annotations/test" not in code)
    check("plantseg_data_root_env_is_used",
          cfg["PLANTSEG_DATA_ROOT_ENV"] == "PLANTSEG_DATA_ROOT"
          and "os.environ.get(PLANTSEG_DATA_ROOT_ENV" in src)
    piped = load_config({"PLANTSEG_DATA_ROOT": "/pod/plantseg_data/plantseg"})
    check("data_root_resolves_from_env",
          piped["train_dataloader"]["dataset"]["data_root"] == "/pod/plantseg_data/plantseg",
          piped["train_dataloader"]["dataset"]["data_root"])
    for bad in ("C:\\Users", "C:/Users", "/workspace/plantseg", "plantseg_data/plantseg"):
        check(f"no_hardcoded_path_{bad[:12]!r}", bad not in code)

    # ---- M12: full-precision same-pass selection; the test split never reaches selection ----
    ckpt_hook = cfg["default_hooks"]["checkpoint"]
    check("m12_selects_on_full_precision_key",
          ckpt_hook["save_best"] == "mIoU_full" and cfg["SELECTION_KEY"] == "mIoU_full"
          and ckpt_hook["rule"] == "greater", "never the 2-decimal display 'mIoU'")
    check("m12_checkpoint_every_validation",
          cfg["train_cfg"]["val_interval"] == cfg["VAL_INTERVAL"] == 4000
          and ckpt_hook["interval"] == cfg["VAL_INTERVAL"] and ckpt_hook["by_epoch"] is False)
    check("m12_all_ten_checkpoints_retained", ckpt_hook["max_keep_ckpts"] == -1,
          "explicit -1: the full 4k..40k selection trail is kept")
    check("m12_same_pass_thesis_metric",
          cfg["val_evaluator"]["type"] == "ThesisConfusionMIoUMetric"
          and cfg["val_evaluator"]["num_classes"] == 116 and cfg["val_evaluator"]["ignore_index"] == 255
          and cfg["val_evaluator"]["background_index"] == 0)
    check("m12_selection_record_hook_registered",
          [h["type"] for h in cfg["custom_hooks"]] == ["TeacherInitCompatibilityHook",
                                                        "TeacherNMFEvalStreamHook",
                                                        "TeacherSelectionRecordHook"],
          str([h["type"] for h in cfg["custom_hooks"]]))
    check("m4v_val_batch_1_frozen_order",
          cfg["val_dataloader"]["batch_size"] == 1
          and cfg["val_dataloader"]["sampler"]["type"] == "DefaultSampler"
          and cfg["val_dataloader"]["sampler"]["shuffle"] is False)
    check("test_split_not_in_train_or_val_path",
          "test" not in str(cfg["train_dataloader"]["dataset"]["data_prefix"])
          and "test" not in str(cfg["val_dataloader"]["dataset"]["data_prefix"]))

    # ---- normalization ----
    norms = [cfg["norm_cfg"]["type"], cfg["model"]["backbone"]["norm_cfg"]["type"]]
    check("norm_is_BN", norms == ["BN", "BN"], str(norms))
    check("no_syncbn_anywhere", "SyncBN" not in code and "SyncBN" not in set(config_types(cfg)),
          "delta code and every type in the merged config, including what _base_ supplies")

    # ---- optimizer / schedule ----
    opt = cfg["optim_wrapper"]["optimizer"]
    check("optimizer_is_adamw", opt["type"] == "AdamW", opt["type"])
    check("lr_6e-5", opt["lr"] == 6e-5, repr(opt["lr"]))
    check("weight_decay_0.01", opt["weight_decay"] == 0.01, repr(opt["weight_decay"]))
    check("betas_0.9_0.999", tuple(opt["betas"]) == (0.9, 0.999), str(opt["betas"]))
    check("decode_head_lr_mult_10",
          cfg["optim_wrapper"]["paramwise_cfg"]["custom_keys"]["head"]["lr_mult"] == 10.)
    # SGD may appear exactly once, inside the provenance constant recording the PAPER's optimizer.
    check("not_student_sgd",
          opt["type"] == "AdamW" and code.count("SGD") == 1
          and "SGD" in cfg["PUBLISHED_BENCHMARK_OPTIMIZER"],
          "AdamW in use; the single SGD mention is the recorded published-benchmark protocol")
    scheds = [s["type"] for s in cfg["param_scheduler"]]
    check("poly_schedule_present", "PolyLR" in scheds, str(scheds))
    check("max_iters_40000", cfg["train_cfg"]["max_iters"] == 40000
          and cfg["MAX_ITERS"] == 40000, str(cfg["MAX_ITERS"]))
    poly = [s for s in cfg["param_scheduler"] if s["type"] == "PolyLR"][0]
    lin = [s for s in cfg["param_scheduler"] if s["type"] == "LinearLR"][0]
    check("poly_ends_at_max_iters", poly["end"] == 40000 and poly["end"] != 160000,
          "horizon corrected from the public family's stale end=160000")
    check("m5_poly_power_1_0", cfg["POLY_POWER"] == 1.0 and poly["power"] == 1.0
          and poly["begin"] == 1500 and poly["eta_min"] == 0.0,
          "not schedule_40k.py's 0.9")
    check("m5_warmup_1500_from_1e-6",
          cfg["WARMUP_ITERS"] == 1500 and lin["begin"] == 0 and lin["end"] == 1500
          and lin["start_factor"] == 1e-6)
    check("poly_horizon_choice_is_labelled",
          cfg["POLY_END_SOURCE"] == "horizon-corrected-to-max-iters", cfg["POLY_END_SOURCE"])
    check("m5_val_interval_4000", cfg["VAL_INTERVAL"] == 4000, "B60 M5; not the TEST-selected 10000")
    check("batch_size_16", cfg["train_dataloader"]["batch_size"] == 16,
          str(cfg["train_dataloader"]["batch_size"]))
    check("crop_512x512", tuple(cfg["crop_size"]) == (512, 512)
          and tuple(cfg["data_preprocessor"]["size"]) == (512, 512))

    # ---- M3 core preprocessing canvas + M2 augmentation (both reuse src/data/transforms.py) ----
    check("m3_core_preprocessing_labelled",
          cfg["CORE_PREPROCESSING_SOURCE"] == "M3:src/data/transforms.core_preprocess")
    check("m3_val_pipeline_is_thesis_canvas",
          [t["type"] for t in cfg["val_dataloader"]["dataset"]["pipeline"]]
          == ["ThesisTeacherEvalTransform", "PackSegInputs"],
          "core_preprocess canvas; semantics proven in smoke_teacher_pipeline")
    dp = cfg["data_preprocessor"]
    check("imagenet_mean_std_match_student",
          dp["mean"] == [123.675, 116.28, 103.53] and dp["std"] == [58.395, 57.12, 57.375],
          "configs/data.py (0.485,0.456,0.406)/(0.229,0.224,0.225) x 255")
    check("m3_pad_applied_by_thesis_transform",
          cfg["PAD_MODE"] == "thesis-canvas-8bit-imagenet-mean-pad-applied-by-transform"
          and dp["test_cfg"] is None and dp["pad_val"] == 0,
          "the transforms emit the padded 512x512 canvas; no evaluation-time padding")
    check("seg_pad_is_ignore_index", dp["seg_pad_val"] == 255)

    check("m2_augmentation_source_labelled",
          cfg["AUGMENTATION_SOURCE"] == "M2:src/data/transforms.train_preprocess+configs/augment.AUGMENT")
    check("m2_train_pipeline_reuses_e1_recipe",
          [t["type"] for t in cfg["train_pipeline"]] == ["ThesisTeacherTrainTransform", "PackSegInputs"])
    upstream = {"PhotoMetricDistortion", "RandomResize", "RandomCrop", "RandomFlip", "Resize",
                "LoadImageFromFile", "LoadAnnotations", "RandomRotate"}
    found = sorted(upstream & set(t for t in config_types(cfg) if isinstance(t, str)))
    check("m2_no_upstream_augmentation_anywhere", found == [],
          "no PhotoMetricDistortion / short-side RandomResize anywhere in the merged config" if not found
          else str(found))
    from configs.augment import AUGMENT
    rrc, rot, photo = AUGMENT["random_resized_crop"], AUGMENT["rotation"], AUGMENT["photometric"]
    check("m2_m3_recipe_values_match_locks",
          tuple(rrc["scale_range"]) == (0.75, 2.0) and tuple(rrc["size"]) == (512, 512)
          and rrc["cat_max_ratio"] == 0.95 and rrc["aspect_ratio_preserved"] is True
          and rot == {"degrees": 10, "p": 0.5, "apply": "before_crop", "image_fill": "imagenet_mean",
                      "mask_fill": 255}
          and AUGMENT["horizontal_flip_p"] == 0.5 and AUGMENT["vertical_flip_p"] == 0.5
          and photo == {"hue": 0.015, "saturation_factor": (0.8, 1.2), "p": 0.5}
          and set(AUGMENT["excluded"]) == {"brightness", "contrast", "blur", "noise", "jpeg"},
          "configs/augment.AUGMENT == B60 M2/M3")

    # ---- M4: upstream NMF algorithm kept, isolated for evaluation and the frozen teacher ----
    head = cfg["model"]["decode_head"]
    check("m4_isolated_nmf_head", head["type"] == "IsolatedNMFLightHamHead"
          and head["ham_kwargs"]["rand_init"] is True, "rand_init=True retained (upstream algorithm)")
    check("m4_policy_marker", cfg["M4_NMF_POLICY"]["seed"] == 42 == cfg["M4_NMF_SEED"]
          and cfg["M4_NMF_POLICY"]["train"] == "M4-T" and cfg["M4_NMF_POLICY"]["val_and_final_eval"] == "M4-V")
    check("m4v_hook_seed_42", [h for h in cfg["custom_hooks"]
                               if h["type"] == "TeacherNMFEvalStreamHook"][0]["seed"] == 42)
    check("m4_no_fake_ham_seed_key", "seed" not in head["ham_kwargs"],
          "MMSeg has no NMF seed key; the control lives in IsolatedNMF2D, not a silently ignored key")
    check("custom_imports_registers_components",
          cfg["custom_imports"]["imports"] == ["src.training.teacher_components"]
          and cfg["custom_imports"]["allow_failed_imports"] is False)

    # ---- loss: cross-entropy ONLY ----
    loss = cfg["model"]["decode_head"]["loss_decode"]
    check("loss_is_cross_entropy_only",
          isinstance(loss, dict) and loss["type"] == "CrossEntropyLoss"
          and loss["use_sigmoid"] is False, str(loss))
    check("m13_avg_non_ignore_true", loss.get("avg_non_ignore") is True,
          "ignore/padded pixels leave both numerator and denominator (B61 M13)")
    check("m5_m13_unweighted", "class_weight" not in loss, "no class weighting in the teacher CE")
    check("no_dice_in_teacher_loss", "Dice" not in code and "dice" not in code,
          "CE+Dice belongs to the student stages")

    # ---- initialization contract ----
    check("backbone_init_cfg_none", cfg["model"]["backbone"]["init_cfg"] is None)
    check("external_ade20k_init_required",
          cfg["SEGNEXT_ADE20K_CKPT_ENV"] == "SEGNEXT_ADE20K_CKPT"
          and cfg["load_from"] == cfg["ADE20K_CKPT_UNSET_SENTINEL"],
          "unset -> non-path sentinel, never silent random init")
    resolved = load_config({"SEGNEXT_ADE20K_CKPT": "/ext/weights/segnext_ade20k.pth"})
    check("ade20k_ckpt_resolves_from_env",
          resolved["load_from"] == "/ext/weights/segnext_ade20k.pth")
    ident = cfg["ADE20K_INIT_IDENTITY"]
    check("ade20k_identity_is_readiness_checkpoint",
          cfg["EXPECTED_ADE20K_SHA256"] == ident["sha256"]
          == "647a0cda7678a35396689a4f8e9fddc33a088d8b539195d0dc97485ab8640ef1"
          and ident["config"] == "segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512"
          and ident["filename"] == "segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth"
          and tuple(ident["classifier_only_mismatch"]) == ("decode_head.conv_seg.weight",
                                                           "decode_head.conv_seg.bias"))
    from configs.teacher_finetune import TEACHER_FINETUNE
    init = TEACHER_FINETUNE["init_checkpoint"]
    check("g10_metadata_uses_mmseg1x_identity",
          init["config"] == ident["config"] and init["filename"] == ident["filename"]
          and init["sha256"] == ident["sha256"]
          and "segnext_mscan-b_512x512_160k_ade20k" not in str(TEACHER_FINETUNE),
          "configs/teacher_finetune.py: no 0.x alias")
    check("m5_metadata_readiness_rule", "R1-R4" in TEACHER_FINETUNE["success_criterion"]
          and "NEED_TO_CONFIRM" not in str(TEACHER_FINETUNE))
    check("resume_disabled", cfg["resume"] is False, "load_from, not resume (runbook §5)")
    check("work_dir_is_external_and_env_driven",
          cfg["TEACHER_WORK_DIR_ENV"] == "TEACHER_WORK_DIR"
          and cfg["work_dir"] == cfg["WORK_DIR_UNSET_SENTINEL"])
    external = load_config({"TEACHER_WORK_DIR": "/ext/teacher_work"})
    check("work_dir_resolves_from_env", external["work_dir"] == "/ext/teacher_work", external["work_dir"])

    # ---- determinism ----
    check("seed_42_deterministic",
          cfg["randomness"]["seed"] == 42 and cfg["randomness"]["deterministic"] is True,
          str(cfg["randomness"]))
    check("nmf_control_not_a_placeholder", "NMF_SEED_CONTROL" not in code,
          "the NEED_TO_CONFIRM placeholder is replaced by the implemented M4 policy")

    # ---- MMSeg 1.x conventions, no 0.x leakage ----
    for key in ("optim_wrapper", "param_scheduler", "train_dataloader", "val_dataloader",
                "train_cfg", "default_hooks", "randomness"):
        check(f"mmseg1x_key_{key}", key in cfg)
    for legacy in ("optimizer_config", "lr_config", "total_iters", "samples_per_gpu",
                   "workers_per_gpu", "evaluation"):
        check(f"no_mmseg0x_key_{legacy}", legacy not in cfg and legacy not in code)
    check("iter_based_train_loop", cfg["train_cfg"]["type"] == "IterBasedTrainLoop")
    check("packsegimputs_pipeline_end",
          cfg["train_pipeline"][-1]["type"] == "PackSegInputs"
          and cfg["val_dataloader"]["dataset"]["pipeline"][-1]["type"] == "PackSegInputs")
    # `_base_` is consumed by the merge, so its declaration is read from the source; the merged model
    # shows what the installed `mmseg::` base actually resolves to.
    loader_stmts = [n for n in ast.parse(src).body
                    if isinstance(n, (ast.Import, ast.ImportFrom, ast.With))
                    or (isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_base_" for t in n.targets))]
    base_stmts = [n for n in loader_stmts if isinstance(n, ast.Assign)]
    check("base_precedes_every_import", bool(loader_stmts) and loader_stmts[0] in base_stmts,
          "keeps _base_ ahead of every import, so mmengine treats the file as a _base_ config from any cwd")
    base = ast.literal_eval(base_stmts[0].value) if len(base_stmts) == 1 else None
    check("base_is_pinned_stock_segnext_b",
          base == ["mmseg::segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py"], str(base))
    model = cfg["model"]
    check("merged_model_is_segnext_mscan_b",
          model["type"] == "EncoderDecoder" and model["backbone"]["type"] == "MSCAN"
          and list(model["backbone"]["embed_dims"]) == [64, 128, 320, 512]
          and list(model["backbone"]["depths"]) == [3, 3, 12, 3]
          and model["decode_head"]["type"] == "IsolatedNMFLightHamHead",
          "SegNeXt-B as resolved: MSCAN [64,128,320,512] / [3,3,12,3] + LightHamHead (M4-isolated)")
    import scripts.launch_teacher_finetune as launcher
    from mmengine.config import Config
    problems = launcher.check_locked_config(Config.fromfile(str(CONFIG)))
    check("launcher_locked_config_check_passes", problems == [], "; ".join(problems))

    # ---- no student-stage leakage ----
    for token in ("kd", "KD", "cwd", "CWD", "distill", "quant", "QAT", "PTQ", "lambda_logit",
                  "T_logit", "observer", "fuse_modules"):
        check(f"no_student_leak_{token}", token not in code)

    print("\n[CHECKS]")
    for name, ok, detail in results:
        print(f"  {name:42}: {'PASS' if ok else 'FAIL'}{('  ' + detail) if detail else ''}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
