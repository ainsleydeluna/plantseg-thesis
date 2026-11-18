# Lane DEMO-3: model registry, FP32 path for E3, comparison switch (lane report)

Session DEMO-3 (Claude Code cloud session), under the prompt `demo3_prompt_20261116.md` of 16 Nov 2026
and the orchestrator's rulings DEMO-MODEL-2, DEMO-3MODEL-1 and DEMO-3MODEL-2.
Branch `lane/demo3`, made from `lane/demo` at `72e3196e` (the demo of record; DL-264, DL-265).
Changes are under `demo/` only, plus this report.

Labels, as in [demo.md](demo.md):
- **MEASURED**: run in this session; the numbers come from the output.
- **DOCUMENTED**: stated in the prompt or in a file named here.
- **INFERRED**: reasoning, not checked here.

## 0. Status

- **Done: A to E.**
  - A: the model registry.
  - B: the FP32 export, loader and parity check.
  - C: the page's comparison switch and the API.
  - D: the smoke, **16/16 passed**. S1 to S8 are kept unchanged; S9 to S16 are new.
  - E: the build context, which now writes `SHA256SUMS` and `demo.env`.
- **Everything ran on CPU, on synthetic images (`demo/synthetic.py`) and random-weights stand-ins.**
  - No thesis checkpoint, no dataset file and no model file of record was opened, loaded, copied or hashed.
  - E3's weights were not loaded in any form.
  - The E3 export and its parity check on VAL images are for the laptop, after AM-30 (section 3).
- **Commits on `lane/demo3`** (author Claude, no trailer, each `git commit --only` with explicit paths):
  - `e06f1b3`: the code (14 files under `demo/`);
  - the next commit: this report.

  Only `lane/demo3` was pushed. Nothing was merged.

## 1. What changed

### File list

These rows are `e06f1b3`'s. Six of these files changed in `e856eb6` (DEMO-3b: [demo3b.md](demo3b.md)
section 5).

| file | status | lines | sha256 |
|---|---|---|---|
| `demo/Dockerfile` | M | 35 | `854f308718b1cfd8b43cad51321dce02282168bb95b6189abcce17cf4ef88580` |
| `demo/README.md` | M | 28 | `51baebbbc8f24b82c218a5b5b69d229d4a84bf10c5d58d2f5547f4f11b05c19e` |
| `demo/common.py` | M | 109 | `006b9a5efd48cdef9512e8925f224e22d912d8791d67b4976dc3039531c61bd0` |
| `demo/config.py` | M | 142 | `789cd8ec2106ba79bbb231c6819252629f23889ccdf75b78930afa2ffbaa2046` |
| `demo/docker/build_context.py` | M | 158 | `a690f16cce57263a41aed94771bcd6b0f759670ecf858452c885c0165f357674` |
| `demo/model/build_stand_in.py` | M | 178 | `8cef768877cfb29ba25885e4364aaab6a8e459ffc8c4ab7ef31597156063bbf0` |
| `demo/model/export_fp32_torchscript.py` | A | 159 | `bf4f51970b8d2ab9145c8003adf19928abeced3141713d70a091b3bb26f51e0a` |
| `demo/model/parity_fp32.py` | A | 140 | `a88cb4b8f7e345fe3d0dde55f7fa8906165f54fc9c7ce30ea2a5b0584afc6b5c` |
| `demo/run_demo.py` | M | 102 | `31081d83bac8834431e186273987cde6a503c0606ef3a935c28fef0672fa7fa7` |
| `demo/server/app.py` | M | 129 | `5cfed23a2a8ba7269cea928e76eb0d4b4bfdadd2c769bba97f77900c549f9fa4` |
| `demo/server/inference.py` | M | 396 | `00efdb80b567f692dff6ab1ee13bd7be826851b481b65e95b0ff8805ffbb679d` |
| `demo/smoke_demo.py` | M | 582 | `66e4aeb89997130bbb7d87afc47745b234319dfdd80c78345d053f3f043dd58d` |
| `demo/tools/bench_server.py` | M | 97 | `1fc8b25f1109bb876c83efed1b43cfe160d9b75a3a3f3d587d45d84e314251e1` |
| `demo/web/index.html` | M | 398 | `199d3a1b32a975d44ca8fa064988229ed3823fe726d66b242cfd548170f6ade5` |
| `docs/lane_reports/demo3.md` | A | (this file) | (this file) |

- Scoped diff `72e3196..e06f1b3` over `demo` and `docs/lane_reports` (test-named paths excluded):
  14 files, 1,468 insertions and 214 deletions (MEASURED).
- No file outside `demo/` changed. Nothing in `src/`, `scripts/`, `configs/` or any other lane changed.
- `demo/screenshots/` is unchanged (open question 12).

### A. The model registry (`demo/config.py`)

**The three slots.** Labels and descriptions are fixed strings, never derived from a score.

| slot | label | description | page option | kind | required |
|---|---|---|---|---|---|
| `e6` | E6 · INT8 (QAT) | the demo model | (always shown) | INT8 QNNPACK TorchScript (`plantseg_int8.json`) | yes |
| `e3` | E3 · FP32 | the model E6 was quantized from | E3 (FP32) | FP32 TorchScript (`plantseg_fp32.json`), stage must be E3 | no |
| `shown` | AM-29's shown student · INT8 | the highest of the candidates on VAL (exploratory) | shown student (INT8, exploratory) | INT8 QNNPACK TorchScript (`plantseg_int8.json`) | no |

**Where each setting comes from.** Each slot's file and expected sha256 come from an environment
variable, or else from its marked line in `demo/config.py`.

| slot | file: variable / marked line | expected sha256: variable / marked line |
|---|---|---|
| e6 | `PLANTSEG_DEMO_MODEL` / `MODEL_PATH` (default: the stand-in, as today) | `PLANTSEG_DEMO_E6_SHA256` / `E6_SHA256 = cf14ae03…dbb0bb` (open question 1) |
| e3 | `PLANTSEG_DEMO_E3_MODEL` / `E3_PATH = ""` | `PLANTSEG_DEMO_E3_SHA256` / `E3_SHA256 = ""` |
| shown | `PLANTSEG_DEMO_SHOWN_MODEL` / `SHOWN_PATH = ""` | `PLANTSEG_DEMO_SHOWN_SHA256` / `SHOWN_SHA256 = ""` |

`run_demo.py --model PATH` still overrides e6's file for one run. Today's route for e6 is kept:
`PLANTSEG_DEMO_MODEL` names its file, and `MODEL_PATH` is its marked line.

**The rules** (`demo/server/inference.py load_registry`; MEASURED in S9 to S16, except the order "hashed before
deserialized", which is read from the code; S13 does not observe it):

- A slot loads only if its expected sha256 is set and the file's sha256 equals it. No other model is loaded.
- Every configured file is hashed and compared **before any file is deserialized**.
- A slot with no expected sha256 is not loaded, and its option is hidden. This holds even when a file is
  named for it; a start-up note says so.
- e6 is required. These stop start-up, with exit 2 and the line `[demo] refused to start: …`:
  - a missing e6;
  - a wrong, missing or malformed sha256 in any configured slot;
  - a file of the wrong kind for its slot;
  - a bad embedded identity;
  - two slots naming the same file.
- **The e3 cross-check.** When E6's identity names its `source_checkpoint_sha256`
  (`src/quant/qat_artifacts.py:453` writes it), e3's identity must name the same checkpoint. Otherwise
  the label "the model E6 was quantized from" would be false, and start-up stops (open question 8).

**The start-up print.** For each loaded file it prints the slot, the file name, the full sha256, the
engine and the stage. MEASURED from the unpacked build context (section 1 E):

```
[demo] PHONE-FIX-1: mkldnn enabled=False, quantized engine qnnpack
[demo] loaded e6: model.pt sha256 8bfd44b0d2b8cb47ab7e638d3aafbbf9c5e7a227f925726cb6c93b5fa24f6c30 engine qnnpack stage DEMO_STAND_IN (STAND-IN, random weights)
[demo] loaded e3: e3.pt sha256 6cd3ab0307f30ae660c08a1044751fd6fab68d5efcafb5e6bcc4bd1e1b9379fd engine fp32 CPU (mkldnn off) stage E3 (STAND-IN, random weights)
[demo] loaded shown: shown.pt sha256 9af67bbab87653ec61c7775ec07d982b2482112e6327dbb9c13665b3a5948128 engine qnnpack stage DEMO_STAND_IN (STAND-IN, random weights)
[demo] e3 source checkpoint not cross-checked: E6's identity names no source_checkpoint_sha256
```

The stand-in's identity has no source checkpoint, so the cross-check is skipped here. With the real E6
it runs.

### B. The FP32 path for E3

**`demo/model/export_fp32_torchscript.py`**

```
CHECKPOINT --source-sha256 HEX --stage E3 --seed 42 --out OUT.pt [--e6 E6.pt]
```

It runs these steps in order:

1. It refuses any path containing "test".
2. It requires the checkpoint's sha256 to equal `--source-sha256` before the file is read.
3. With `--e6`, it reads only E6's embedded `plantseg_int8.json`, from the zip, without loading the
   model. That identity must name this checkpoint as its source.
4. It builds the student with the repository's own code, `src/quant/checkpoint.py load_student_from_e3`.
   This is the E3 loader E6 was quantized through:
   - the checkpoint's stage must be E3;
   - there must be no CWD projection, teacher or quantization state;
   - the state loads strictly into `src/models/student.py build_student(pretrained=False)`.
5. It traces at 1×3×512×512 (`src/quant/ptq.py EXAMPLE_SHAPE`) with PHONE-FIX-1 on, and embeds
   `plantseg_fp32.json`.
6. It writes `OUT.partial`, loads it back through the demo's FP32 loader, and compares it with the eager
   model on `src/quant/ptq.py synthetic_parity_inputs` (a batch of 2 and a 384×512 input among them).
7. Only if that comparison is within the parity limits does it rename the file to `OUT`. It refuses to
   overwrite an existing file.

**The embedded identity** (`plantseg_fp32.json`):

- required keys:
  - `schema` = `plantseg-fp32-torchscript/1.0.0`;
  - `precision` = `fp32`;
  - `stage` (E3);
  - `seed` (42);
  - `num_classes` = 116;
  - `source_checkpoint_sha256`;
  - `preprocess_protocol` = `core_preprocess/1.0.0` (open question 5);
  - `artifact_role` = `demo_fp32_export`;
- also recorded: the source file name and iteration, `stand_in`, the traced shape,
  `mkldnn_enabled_at_trace` (false), the torch version and the writer.

**The loader** (`demo/server/inference.py load_fp32_model`) checks the identity the same way the INT8
loader does:

- It checks before loading (read from the zip) and again after `torch.jit.load` (read through its
  `_extra_files`). The two reads must be equal.
- Every required key must be present.
- `schema`, `precision`, `num_classes`, `preprocess_protocol`, `artifact_role` and the slot's stage must
  have the values above.
- `seed` must be an integer, and the source sha256 must be 64 hex digits.
- A warm-up forward must give [1, 116, 512, 512] float32.

S10 shows tampered identities refused, 5/5 (MEASURED): 115 classes, another preprocessing id, stage E2,
no seed, and an INT8 file.

**`demo/model/parity_fp32.py`**

```
--model OUT.pt --checkpoint CKPT --source-sha256 HEX --images PATH… [--n N]
```

- It refuses test-named paths.
- It requires checkpoint sha256 = `--source-sha256` = the identity's source, before the checkpoint is read.
- It builds the eager model the same way as the export.
- It takes the first N images (`.jpg`, `.jpeg`, `.png`, in name order) through the demo's own
  preprocessing (core_preprocess + finalize).
- For each image it prints the largest absolute logit difference and the argmax agreement, over the
  canvas and over the photo's region.
- It ends with `PASS parity_fp32: …` (exit 0) or `STOP parity_fp32: …` (exit 1).
- The limits: max |Δ| ≤ 1e-4, and 100% canvas argmax agreement on every image (open question 9).
- On the stand-in, the export is **bitwise equal** to the eager model on 4/4 synthetic tensors and on
  6/6 synthetic image files (MEASURED, S10).

**PHONE-FIX-1** (`demo/common.py phone_fix`: `torch.backends.mkldnn.enabled = False`) is applied in
every demo process:

- the server, at `demo/server/inference.py` import and again in `Segmenter`;
- the export, the parity check, the stand-in builder and the smoke.

INT8 keeps the qnnpack engine through the evaluator's `select_int8_backend`. S15 checks the state in
all four smoke servers and in the export, parity and smoke processes. At `72e3196` the fix was not in
the code (open question 7).

**Stand-ins** (`demo/model/build_stand_in.py`):

- `--seed N` builds a second INT8 stand-in, for slot `shown`.
- `--e3-checkpoint PATH` writes a random-weights stand-in in E3's checkpoint schema: stage `E3`,
  `model_state_dict`, `num_classes` 116, `stand_in: true`.
- With the default seed, its weights are exactly the FP32 weights the default INT8 stand-in was
  quantized from. The defaults are otherwise unchanged.

### C. The page and the API

**API** (`demo/server/app.py`). All existing fields are kept.

- `POST /api/segment` takes an optional form field `compare` (`none` | `e3` | `shown`).
  - The second model runs on **the same input tensor** as E6.
  - The response keeps every build-2 field for E6 and adds:
    - `compare`;
    - `panels`: one per model, E6 first. Each panel has its slot, label, description, sha256 and short
      sha256, engine, stage, inference ms, top classes, overlay and mask;
    - `latency_ms.inference_compare`.
  - A comparison with a slot that is not loaded is HTTP 400.
- `GET /api/info` adds:
  - `models`: the loaded slots, in registry order;
  - `compare_options`: loaded optional slots only;
  - `runtime`: PHONE-FIX-1's state;
  - `startup_notes`.
- `/healthz` adds `slots`, the sha256 of each loaded slot.
- Tools keep `load_model(path)` (INT8, no registry). `demo/tools/bench_server.py` now calls
  `Segmenter.for_file`; the server itself goes only through the registry.

**Page** (`demo/web/index.html`):

- **E6's result is always shown.** Its panel carries its label, its description, a short sha256 and
  its engine.
- **The switch "Compare with: none | E3 (FP32) | shown student (INT8, exploratory)".**
  - It is built from `compare_options`.
  - The page has no built-in e3 or shown option (S14). If no optional slot is loaded, the switch is
    hidden altogether.
  - Changing it re-sends the same photo.
- **Compare mode.** The second panel sits beside E6's, in two columns, with the same photo.
  - Each panel shows its label, description, short sha256, engine, overlay and top classes.
  - Each also shows "Time for this photo (a demonstration, not a measurement)" with its own model
    inference time.
  - Server total, round trip and upload are shown once, below both panels.
  - The overlay slider drives both overlays.
- **Build 2's wording rules (DEMO-LABEL-1), kept:**
  - the note under the "Top classes" heading, verbatim, shown once above the class lists;
  - the time heading;
  - "(page to server and back)";
  - no link to the in-browser page in the footer.
- **The Model box** lists E6, then each loaded optional slot. Each shows its label and description,
  file, SHA-256, identity (stage and method; for FP32 also the seed and the source checkpoint) and
  runtime. There is no "role" field.
- **No accuracy figure appears anywhere on the page.** The stand-in banner names which loaded models
  are stand-ins.
- **Layout check** (MEASURED in headless Chromium, iPhone 15 Plus viewport, 430 CSS px):
  - single and compare views render;
  - switching none → E3 → shown → none works;
  - there is no horizontal scroll;
  - the screenshots were not committed.
- **Times seen here** (4-vCPU Xeon, stand-ins; a demonstration, not a measurement):
  - E6 INT8: 110–220 ms;
  - E3 FP32 with mkldnn off: 590–940 ms;
  - shown INT8: about 150 ms.

### D. The smoke: full output

`/tmp/…/venv-demo/bin/python -I demo/smoke_demo.py` (MEASURED; the code is that of `e06f1b3`):

```
[smoke] /home/user/plantseg-thesis/demo/models/stand_in_int8_qnnpack.pt missing: building a stand-in at /tmp/plantseg_demo_smoke_3x68b3jl/stand_in_int8_qnnpack.pt
/tmp/claude-0/-home-user-plantseg-thesis/32b518f2-2eff-5d46-a49f-2f3e0d0ad16c/scratchpad/venv-demo/lib/python3.11/site-packages/torch/ao/quantization/observer.py:1207: UserWarning: must run observer before calling calculate_qparams.                                    Returning default scale and zero point 
  warnings.warn(
RESULT PASS S1 server_starts: healthz={'ok': True, 'model_sha256': '8bfd44b0d2b8cb47ab7e638d3aafbbf9c5e7a227f925726cb6c93b5fa24f6c30'} after 2.5s; file sha256 8bfd44b0d2b8cb47...
RESULT PASS S2 class_list_is_repos: 116 served == 116 raw == 116 loader; configs/data.py num_classes 116; model outputs 116; class 0 name '' role background_or_non_disease
RESULT PASS S3 mask_shape_and_class_range: leaf_1024x768.jpg: mask (384, 512) classes 100..100 inference 173.89 ms; leaf_480x640.png: mask (512, 384) classes 100..100 inference 113.77 ms; leaf_800x800.jpg: mask (512, 512) classes 100..100 inference 139.65 ms; leaf_720x1280.jpg: mask (512, 288) classes 100..100 inference 112.51 ms
RESULT PASS S4 latency_field_present: decode/preprocess/inference/postprocess/total in ms
RESULT PASS S5 preprocessing_byte_equal: in-process wrapper == dataset path: True; server input sha256 == dataset path: True; placeholder independent: True; protocol core_preprocess/1.0.0 == src.eval.adapters.PREPROCESS_PROTOCOL; over 4 images (JPEG and PNG)
RESULT PASS S6 exif_orientation_applied: stored 800x600 with orientation 6 -> photo box 384x512 (w x h)
RESULT PASS S7 bad_upload_refused: HTTP 400 could not decode the image (UnidentifiedImageError): send a JPEG or PNG
RESULT PASS S8 no_src_stats: server [], smoke []
RESULT PASS S9 e6_alone: loaded ['e6']; options []; start-up line 'loaded e6: stand_in_int8_qnnpack.pt sha256 8bfd44b0d2b8... engine qnnpack stage DEMO_STAND_IN'; label 'E6 · INT8 (QAT)' / 'the demo model'; 1 panel
/tmp/claude-0/-home-user-plantseg-thesis/32b518f2-2eff-5d46-a49f-2f3e0d0ad16c/scratchpad/venv-demo/lib/python3.11/site-packages/torch/ao/quantization/observer.py:1207: UserWarning: must run observer before calling calculate_qparams.                                    Returning default scale and zero point 
  warnings.warn(
RESULT PASS S10 fp32_export_round_trip: export e3_fp32_stand_in.pt sha256 c92c7f41e4c6...; identity stage E3 seed 42 source 82f34c61ee4d... (= stand-in checkpoint: True) classes 116 preprocessing core_preprocess/1.0.0; round trip bitwise 4/4; refused 5/5 (classes_115, other_preprocessing, stage_E2, no_seed, int8_file); parity: 'PASS parity_fp32: 6 images, max |logit diff| 0.000e+00 (limit 0.0001), min argmax agreement 1.000000 (min 1), bitwise-equal 6/6'; wrong checkpoint sha256 -> exit 1
RESULT PASS S11 e6_with_fp32_stand_in_in_e3: loaded ['e6', 'e3']; options ['E3 (FP32)']; e3 e3_fp32_stand_in.pt sha256 c92c7f41e4c6... engine fp32 CPU (mkldnn off) stage E3; compare=e3: HTTP 200, 2 panels, inference e6 110.86 ms / e3 593.91 ms; E6's mask unchanged by the comparison: True
RESULT PASS S12 e6_with_int8_stand_in_in_shown: loaded ['e6', 'shown']; options ['shown student (INT8, exploratory)']; shown shown_stand_in_int8_qnnpack.pt sha256 9af67bbab876... engine qnnpack; compare=shown: HTTP 200, 2 panels, inference e6 117.34 ms / shown 157.11 ms; compare=e3 -> HTTP 400
RESULT PASS S13 wrong_sha256_refused: e6_other_files_sha256: exit 2, refused before any model loaded: True; e3_other_files_sha256: exit 2, refused before any model loaded: True; e6_malformed_sha256: exit 2, refused before any model loaded: True; e6_marked_line_sha256: exit 2, refused before any model loaded: True
RESULT PASS S14 unset_slot_hidden: e6 alone: options [] and compare=e3/shown -> HTTP 400/400: True; shown named without a sha256 beside a loaded e3: not loaded, not offered, compare=shown -> HTTP 400: True; the page has no built-in e3/shown option and its switch starts hidden: True
RESULT PASS S15 phone_fix_1: servers ['A_e6_alone', 'B_e6_e3', 'C_e6_shown', 'D_all_three']: mkldnn off and engine qnnpack in all: True; export trace mkldnn off: True; parity process mkldnn off: True; smoke process mkldnn off: True
RESULT PASS S16 all_three_slots: loaded ['e6', 'e3', 'shown']; options ['E3 (FP32)', 'shown student (INT8, exploratory)']; both comparisons answered with two panels: True
SUMMARY 16/16 passed

real	1m17.798s
user	2m23.423s
sys	0m23.609s
```

- The two warnings come from the INT8 stand-in builds. Their cause is the 5 unused `skip_add`
  observers, as in [demo.md](demo.md) section 4.
- S3 shows one class everywhere (class 100), because the weights are random.

How S9 to S16 map to the prompt's checks:

| prompt's check | smoke check(s) |
|---|---|
| E6 alone | S9 |
| E6 with an FP32 stand-in in e3 | S11 |
| E6 with an INT8 stand-in in shown | S12 (S16 loads all three) |
| a wrong sha256 is refused | S13 |
| an unset slot is hidden | S14 |
| the FP32 export round trip, with parity PASS | S10 |
| PHONE-FIX-1 (added) | S15 |

### E. The build context (`demo/docker/build_context.py`, `demo/Dockerfile`)

**What it copies.**

- Every configured slot, after checking its file against its expected sha256. A mismatch stops the build.
- e6 goes to `demo/models/model.pt`, e3 to `demo/models/e3.pt`, and shown to `demo/models/shown.pt`.
  `--model`, `--e3` and `--shown` override a slot's file.
- The tracked tree (`git ls-files`, test-named paths and `src/stats` excluded) now includes the two new
  scripts.

**What it writes.**

- `demo.env`: each copied slot's path, relative to the context root, and its expected sha256.
- `SHA256SUMS` last. It lists every other file, in `sha256sum -c` form.
- The Dockerfile sets the three paths. The documented run passes `--env-file demo.env`.

**Measured here.** Context built from `e06f1b3` with the three stand-ins (e6 `8bfd44b0…`, e3 `6cd3ab03…`,
shown `9af67bba…`):

- **48 files copied**:
  - 45 tracked files: 15 under `src/`, 4 under `configs/`, 26 under `demo/`;
  - 3 model files.
- In build 1's terms, the same command gives 43 tracked files plus 1 model. The 2 new tracked files are
  `export_fp32_torchscript.py` and `parity_fp32.py`.
- **51 files are listed in `SHA256SUMS`**: the 48 above plus `Dockerfile`, `CONTEXT_MANIFEST.json` and
  `demo.env`. The context holds **52 files** with `SHA256SUMS` itself.
- **`SHA256SUMS` sha256: `6371793ea0fafcba6dff022ee5c07bab59653986baa6ae31b21585eb51a76752`.**
  - This value depends on the three model files and on `demo.env`.
  - `CONTEXT_MANIFEST.json` names the model sources' absolute paths, so it depends on them too.
  - The laptop's context will have its own value.
- No listed path contains "test", `src/stats` or `docs/`.
- The phone unpack, simulated:
  - pack (`tar` of the 7 top-level entries), unpack into a new directory;
  - `sha256sum -c SHA256SUMS`: all OK (51 files);
  - `set -a; . ./demo.env; set +a; python -I demo/run_demo.py --no-build`;
  - the server loaded all three slots (the start-up print in section 1 A);
  - `/healthz` named the three sha256 values;
  - a compare=e3 request returned two panels;
  - the page answered HTTP 200.
- A context build with a wrong e3 sha256 is refused: exit 1, `refused: slot e3: … has sha256 6cd3ab03…,
  expected 9af67bba…`.
- The Docker image was not built, because there is no Docker daemon here.

## 2. Hard limits, as kept

1. **docs/ and the repository root.**
   - Nothing under `docs/reference/` was opened, read, listed, grepped or hashed.
   - Searches ran only inside `demo/` and `src/`.
   - In `docs/`, only `docs/lane_reports/demo.md` was read, by its exact path. This file was written
     directly.
   - `git status` was always scoped to `demo` and `docs/lane_reports`, with test-named paths excluded.
2. **TEST data.** No TEST data, dataset file, archive or issue #11 CSV was read. No test-named path was
   read. The new scripts refuse test-named paths.
3. **Models.** No thesis checkpoint and no model file of record was loaded, copied or downloaded. Only
   stand-ins were built.
4. **Changed files.** The changes are under `demo/` plus this report. `src.stats` is not imported (S8),
   and the new scripts assert it.
5. **Images.** Only synthetic images were used.
6. **Commits and pushes.** No commit has a trailer. Only `lane/demo3` was pushed. There was no merge.
7. **Words.** None of the four words appears in `demo/` (`grep -w`, 0 hits) or in this report.

## 3. Commands for the later steps

### On the laptop (WSL2 or Linux, Python 3.11, `demo/requirements-demo.txt`), after AM-30 is on master

**1. Export E3 to the FP32 TorchScript file.** Write it outside the repository.

```bash
python -I demo/model/export_fp32_torchscript.py /path/to/E3_seed42_checkpoint.pt --source-sha256 4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d --stage E3 --seed 42 --e6 /path/to/E6_seed42_int8_qnnpack.pt --out /path/outside/repo/demo_e3_fp32.pt
```

The output must show:
- `"e6_cross_check": "… names this checkpoint as its source"`;
- a round trip within limits;
- the line `[export] wrote demo_e3_fp32.pt sha256 <HEX>`.

It refuses any other checkpoint before reading it.

**2. Its parity check on VAL images.**

```bash
python -I demo/model/parity_fp32.py --model /path/outside/repo/demo_e3_fp32.pt --checkpoint /path/to/E3_seed42_checkpoint.pt --source-sha256 4baa5bad682995091b66c67fb2d2ce467b3f139e02cc5a5cc62c243fe00f1a0d --images "$PLANTSEG_DATA_ROOT/images/val" --n 32
```

The VAL path is INFERRED (open question 10). This step is not run until the orchestrator has checked, in
the repository's data configuration, that `$PLANTSEG_DATA_ROOT/images/val` holds the VAL split and nothing
else. The last line must start with `PASS parity_fp32`. A `STOP` line means: do not use the file.

**3. The orchestrator sets e3's expected sha256.** Set it to the `<HEX>` from step 1 as
`PLANTSEG_DEMO_E3_SHA256`, as steps 4 and 6 pass it. Do the same for `shown` once AM-29's file exists. The
e3 and shown marked lines in `demo/config.py` stay empty. The smoke needs them empty: a sha256 line without
its path line stops S1 (and every start-up that does not set the path variable), and both lines set make
some of S9, S11, S12 and S14 fail. e6's line already holds `cf14ae03…dbb0bb`.

**4. Run on the laptop.**

```bash
PLANTSEG_DEMO_MODEL=/path/to/E6_seed42_int8_qnnpack.pt PLANTSEG_DEMO_E3_MODEL=/path/outside/repo/demo_e3_fp32.pt PLANTSEG_DEMO_E3_SHA256=<HEX> python -I demo/run_demo.py --no-build
```

The start-up lines must show:
- `loaded e6: … sha256 cf14ae0362d0…dbb0bb engine qnnpack stage E6`;
- `loaded e3: demo_e3_fp32.pt sha256 <HEX> engine fp32 CPU (mkldnn off) stage E3`;
- `e3 source checkpoint 4baa5bad6829... equals the one E6's identity names`.

**5. The smoke against the real E6.** S1 to S9 run on the real E6. S10 to S16 use an INT8 stand-in as e6,
because the real E6's identity would correctly refuse an E3 stand-in. Expect `SUMMARY 16/16 passed`.

```bash
python -I demo/smoke_demo.py --model /path/to/E6_seed42_int8_qnnpack.pt
```

**6. The build.** The context goes outside the repository.

```bash
PLANTSEG_DEMO_E3_SHA256=<HEX> python -I demo/docker/build_context.py ../plantseg-demo-context --model /path/to/E6_seed42_int8_qnnpack.pt --e3 /path/outside/repo/demo_e3_fp32.pt
```

```bash
tar -C ../plantseg-demo-context -czf ../plantseg-demo-context.tgz demo src configs Dockerfile CONTEXT_MANIFEST.json demo.env SHA256SUMS
```

```bash
sha256sum ../plantseg-demo-context/SHA256SUMS ../plantseg-demo-context.tgz
```

- Record the printed file count and both sha256 values.
- Add `--shown /path/to/shown.pt` (with its sha256 set) and/or `--with-ondevice` as needed.
- With Docker:

```bash
docker build -t plantseg-demo ../plantseg-demo-context
```

```bash
docker run --rm -p 8000:8000 --env-file ../plantseg-demo-context/demo.env plantseg-demo
```

### On the phone: the unpack

This assumes a POSIX shell with Python 3.11 and the demo's pins installed, as for build 2 (INFERRED;
open question 11). PHONE-FIX-1 is now in the code, so no patch is applied on the phone.

```bash
sha256sum plantseg-demo-context.tgz && mkdir ~/plantseg-demo3 && tar -C ~/plantseg-demo3 -xzf plantseg-demo-context.tgz
```

```bash
cd ~/plantseg-demo3 && sha256sum -c SHA256SUMS && sha256sum SHA256SUMS
```

```bash
set -a; . ./demo.env; set +a; python -I demo/run_demo.py --no-build --host 0.0.0.0 --port 8000
```

- The tgz's sha256 must equal the laptop's from step 6, and `mkdir` must succeed (a new, empty directory).
- `sha256sum -c` must print OK for every listed file.
- The sha256 of `SHA256SUMS` must equal the laptop's from step 6.
- The start-up lines must show step 4's sha256 values, engines and stages; here the files are named
  `model.pt` and `e3.pt`.

## 4. Open questions

Where the prompt left a choice open, I took the option closest to build 2 and listed it here.

1. **e6's expected sha256 is pre-filled** with `cf14ae0362d0…dbb0bb`, the value DEMO-MODEL-2 documents.
   - Why: so today's route keeps working with no new variable. `PLANTSEG_DEMO_MODEL=<E6 file>
     python -I demo/run_demo.py --no-build` still serves E6.
   - The consequence: the out-of-the-box stand-in no longer starts by itself. `run_demo.py` builds it,
     refuses it (wrong sha256), and prints the `PLANTSEG_DEMO_E6_SHA256=…` line that would serve it.
   - If the orchestrator wants every value empty until set, blank the `E6_SHA256` line. With it blank,
     the server refuses to start until it is set, and steps 4 and 6 of section 3 also pass
     `PLANTSEG_DEMO_E6_SHA256`.
2. **Expected sha256 values can also come from environment variables** (`PLANTSEG_DEMO_*_SHA256`), which
   override the marked lines.
   - They are used by the smoke's stand-ins, by Docker's `--env-file` and by the unpacked context
     (`demo.env`).
   - The source of each value is shown in `/api/info` (`sha256_source`).
   - If the marked lines must be the only route, the override can be removed. The smoke and `demo.env`
     would then need another route.
3. **A bad configured slot stops start-up rather than being hidden.** This covers a wrong sha256, a
   missing file and a bad identity. Build 2's loader also stops start-up on any mismatch.
4. **Label and description split.** Each prompt string was split at its last " · ": the label is the
   first two parts and the description is the last. For example, "E6 · INT8 (QAT)" / "the demo model".
5. **"The preprocessing id that the INT8 identity names."** Neither E6's QAT identity
   (`src/quant/qat_artifacts.py:453`) nor the PTQ identity carries a preprocessing key.
   - The id used is `core_preprocess/1.0.0`. It is the one the INT8 path applies and the page shows:
     `demo/server/preprocess.py PROTOCOL_ID` = `src.eval.adapters.PREPROCESS_PROTOCOL`.
   - The loader requires it, because both panels get the same input tensor.
6. **The seed.** E3's checkpoint payload (`src/training/train_distill.py`) has no seed key, so `--seed
   42` is declared on the command line. The export checks it against the checkpoint only if the
   checkpoint carries a `seed`.
7. **PHONE-FIX-1 was not in the code at `72e3196`.** A grep for "mkldnn" in `demo/` and `src/` gave 0
   hits (MEASURED), so it was added here.
   - With mkldnn off, the FP32 stand-in's inference was 590–940 ms here. On a phone, E3 will be
     slower than E6 (INFERRED).
   - This session's e6 stand-in hashes to `8bfd44b0…`, not build 1's `a5c699b5…`. The checkout path
     and the mkldnn setting during calibration both differ, and which one causes the change was not
     separated.
8. **Two added refusals:**
   - e3's source checkpoint must equal the one E6's identity names, when E6 names one;
   - two slots may not hold the same file.
9. **The parity limits** are max |Δ logit| ≤ 1e-4 and 100% canvas argmax agreement on every image.
   - On the stand-in, the export is bitwise equal.
   - If real E3 on VAL is not bitwise equal, a single tied pixel would STOP. The numbers are printed for
     the orchestrator to judge.
10. **The VAL directory** is assumed to be `$PLANTSEG_DATA_ROOT/images/val` (INFERRED from `src/quant/ptq.py`'s
    `images/train` layout). `--images` takes any files or directories.
11. **Build 2's phone setup is not visible to me.** I don't know how the context was unpacked and served
    on the phone. The unpack commands in section 3 are INFERRED.
12. **`demo/screenshots/` still shows build 1's page.** The prompt did not ask for new screenshots. The
    comparison view was checked in headless Chromium here, but not committed.
13. **Install.** torch was not installed in this container. The prompt requires running here, so I took
    "test it here" as approval and installed `demo/requirements-demo.txt`'s pins into a venv in the
    session scratchpad, outside the repository:
    - uv, in two steps: PyPI, then the PyTorch CPU index;
    - plus `setuptools==65.5.0`, because a uv venv lacks it and torch 2.1 imports `pkg_resources`.
      `python3.11 -m venv` includes it;
    - onnx and onnxruntime were not installed, because they were not needed.

    AGENTS.md rule 5 asks for an explicit approval of installs, and the prompt says to ask nothing.
14. **The decision log was not read.** CLAUDE.md asks for `docs/DECISION_LOG.md` to be read and its
    Status updated. Hard limits 1 and 4 allow only `docs/lane_reports/demo.md` to be read and only this
    report to be written in `docs/`, so I did neither.
15. **Tooling conflicts, reported once.**
    - The harness asked for `Co-Authored-By`/`Claude-Session` trailers and named `lane/demo` as the
      development branch.
    - The prompt overrides both: there are no trailers, and the work is on `lane/demo3`.
16. **The repository's L-PROT hook refused six of my shell commands before they ran.** The hook cannot
    see paths held in shell variables, or `.` and `*` used outside the repository, so it treated them as
    possibly covering the protected file:
    - `rm -rf $W`;
    - `curl --noproxy '*'`;
    - `ls $SP`;
    - `git add -- $P`;
    - `find $SP/ctx3`;
    - `tar … .`.

    Each was re-run with literal paths, all in the scratchpad or naming explicit `demo/` files. None
    touched `docs/reference/`.
17. **Memory.** A comparison holds two models' logits ([1, 116, 512, 512] float32, 121 MB each) only until
    each argmax. Peak memory with E3 loaded was not measured; `bench_server.py` was not re-run.
18. **The in-browser page** (`/ondevice/`) and its ONNX exports remain E6-only.
