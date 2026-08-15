#!/usr/bin/env python3
"""Static verification of the SegNeXt-B teacher fine-tune config. No mmseg, no GPU, no training.

The config is plain Python (dicts, strings, `_base_` as a list of strings), so it can be `exec`'d and
inspected directly without MMSegmentation installed. Nothing here builds a model, reads the dataset,
downloads a checkpoint or trains anything — it asserts that the locked B1 recipe is what the file
actually says, and that no student-stage setting leaked into the teacher.
"""
from __future__ import annotations

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


def load_config(env: dict[str, str] | None = None) -> dict:
    """exec the config in an isolated namespace, optionally with env vars applied."""
    saved = {k: os.environ.get(k) for k in (env or {})}
    try:
        for k, v in (env or {}).items():
            os.environ[k] = v
        ns: dict = {}
        exec(compile(CONFIG.read_text(encoding="utf-8"), str(CONFIG), "exec"), ns)
        return ns
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    print("=" * 78)
    print("TEACHER FINE-TUNE CONFIG — static verification (no mmseg, no GPU, no training)")
    print(f"config: {CONFIG.relative_to(REPO)}")
    print("=" * 78)

    src = CONFIG.read_text(encoding="utf-8")
    code = strip_comments(src)          # token scans run against CODE ONLY, never the comments
    cfg = load_config()
    check("config_is_valid_python", isinstance(cfg, dict) and "model" in cfg)

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

    # ---- dataset ----
    for split, loader in (("train", "train_dataloader"), ("val", "val_dataloader"),
                          ("test", "test_dataloader")):
        ds = cfg[loader]["dataset"]
        check(f"{split}_reduce_zero_label_false", ds["reduce_zero_label"] is False)
        check(f"{split}_uses_{split}_split",
              ds["data_prefix"] == {"img_path": f"images/{split}",
                                    "seg_map_path": f"annotations/{split}"},
              str(ds["data_prefix"]))
    check("plantseg_data_root_env_is_used",
          cfg["PLANTSEG_DATA_ROOT_ENV"] == "PLANTSEG_DATA_ROOT"
          and "os.environ.get(PLANTSEG_DATA_ROOT_ENV" in src)
    piped = load_config({"PLANTSEG_DATA_ROOT": "/pod/plantseg_data/plantseg"})
    check("data_root_resolves_from_env",
          piped["train_dataloader"]["dataset"]["data_root"] == "/pod/plantseg_data/plantseg",
          piped["train_dataloader"]["dataset"]["data_root"])
    for bad in ("C:\\Users", "C:/Users", "/workspace/plantseg", "plantseg_data/plantseg"):
        check(f"no_hardcoded_path_{bad[:12]!r}", bad not in code)

    # ---- test split must not reach training or model selection ----
    ckpt_hook = cfg["default_hooks"]["checkpoint"]
    check("checkpoint_selection_on_val_miou",
          ckpt_hook["save_best"] == "mIoU" and ckpt_hook["rule"] == "greater")
    check("val_interval_drives_selection",
          cfg["train_cfg"]["val_interval"] == cfg["VAL_INTERVAL"]
          and ckpt_hook["interval"] == cfg["VAL_INTERVAL"])
    check("test_split_not_in_train_or_val_path",
          "test" not in str(cfg["train_dataloader"]["dataset"]["data_prefix"])
          and "test" not in str(cfg["val_dataloader"]["dataset"]["data_prefix"]))

    # ---- normalization ----
    norms = [cfg["norm_cfg"]["type"], cfg["model"]["backbone"]["norm_cfg"]["type"]]
    check("norm_is_BN", norms == ["BN", "BN"], str(norms))
    check("no_syncbn_anywhere", "SyncBN" not in code)

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
    check("poly_power_is_source_backed_1_0", cfg["POLY_POWER"] == 1.0 and poly["power"] == 1.0,
          "public PlantSeg SegNeXt-family override, not schedule_40k.py's 0.9")
    check("warmup_1500_source_backed",
          cfg["WARMUP_ITERS"] == 1500 and lin["begin"] == 0 and lin["end"] == 1500)
    check("poly_horizon_choice_is_labelled",
          cfg["POLY_END_SOURCE"] == "horizon-corrected-to-max-iters", cfg["POLY_END_SOURCE"])
    check("val_interval_is_source_backed_10000", cfg["VAL_INTERVAL"] == 10000,
          "public schedule_40k.py cadence, not an unlabelled 4000")
    check("batch_size_16", cfg["train_dataloader"]["batch_size"] == 16,
          str(cfg["train_dataloader"]["batch_size"]))
    check("crop_512x512", tuple(cfg["crop_size"]) == (512, 512)
          and tuple(cfg["data_preprocessor"]["size"]) == (512, 512))

    # ---- core preprocessing parity (runbook §11) vs teacher-specific augmentation (A5) ----
    check("core_preprocessing_labelled_thesis_parity",
          cfg["CORE_PREPROCESSING_SOURCE"] == "thesis-parity-runbook-s11")
    resize = [t for t in cfg["test_pipeline"] if t["type"] == "Resize"][0]
    check("eval_resize_makes_long_side_512",
          tuple(resize["scale"]) == (512, 512) and resize["keep_ratio"] is True,
          "aspect-preserving long-side->512, NOT the upstream (2048,512) short-side scale")
    dp = cfg["data_preprocessor"]
    check("imagenet_mean_std_match_student",
          dp["mean"] == [123.675, 116.28, 103.53] and dp["std"] == [58.395, 57.12, 57.375],
          "configs/data.py (0.485,0.456,0.406)/(0.229,0.224,0.225) x 255")
    check("image_pad_is_imagenet_mean_equivalent",
          cfg["PAD_MODE"] == "imagenet-mean-equivalent-post-normalisation" and dp["pad_val"] == 0,
          "SegDataPreProcessor normalises then pads, so pad_val=0 == mean padding in raw space")
    check("seg_pad_is_ignore_index", dp["seg_pad_val"] == 255)

    aug = {t["type"]: t for t in cfg["train_pipeline"]}
    check("augmentation_source_labelled",
          cfg["AUGMENTATION_SOURCE"] == "public-plantseg-segnext-family",
          "teacher-specific; §11 governs core preprocessing only")
    check("augmentation_matches_public_family",
          tuple(aug["RandomResize"]["scale"]) == (2048, 512)
          and tuple(aug["RandomResize"]["ratio_range"]) == (0.5, 2.0)
          and tuple(aug["RandomCrop"]["crop_size"]) == (512, 512)
          and aug["RandomCrop"]["cat_max_ratio"] == 0.75
          and aug["RandomFlip"]["prob"] == 0.5
          and "PhotoMetricDistortion" in aug)
    check("student_only_augmentation_not_imposed",
          "rotation" not in code and "vertical" not in code and "saturation" not in code,
          "no E1 vertical flip / +-10deg rotation / hue-saturation added to the teacher")
    check("nmf_seed_control_marked_unresolved",
          cfg["NMF_SEED_CONTROL"] == "NEED_TO_CONFIRM")

    # ---- loss: cross-entropy ONLY ----
    loss = cfg["model"]["decode_head"]["loss_decode"]
    check("loss_is_cross_entropy_only",
          isinstance(loss, dict) and loss["type"] == "CrossEntropyLoss"
          and loss["use_sigmoid"] is False, str(loss))
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
    check("resume_disabled", cfg["resume"] is False, "load_from, not resume (runbook §5)")
    check("work_dir_is_external_and_env_driven",
          cfg["TEACHER_WORK_DIR_ENV"] == "TEACHER_WORK_DIR"
          and cfg["work_dir"] == cfg["WORK_DIR_UNSET_SENTINEL"])

    # ---- determinism ----
    check("seed_42_deterministic",
          cfg["randomness"]["seed"] == 42 and cfg["randomness"]["deterministic"] is True,
          str(cfg["randomness"]))
    check("nmf_seed_documented_not_faked",
          "NEED_TO_CONFIRM" in src and "ham_head.py" in src,
          "no invented ham_kwargs seed key")

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
          and cfg["test_pipeline"][-1]["type"] == "PackSegInputs")
    check("base_is_pinned_stock_segnext_b",
          cfg["_base_"] == ["mmseg::segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py"],
          str(cfg["_base_"]))

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
