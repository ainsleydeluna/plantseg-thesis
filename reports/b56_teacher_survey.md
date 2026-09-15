# B56 — teacher survey on CPU: A, B and E pass; C and D are blocked by a config mmengine cannot load

**Type:** survey record, independently re-measured. **Status:** A, B and E **measured**; C and D **not
run** — blocked at `Config.fromfile`, root cause established at source, fix queued as **G16** (governed,
not applied). **G14 closed.** **Date:** 2026-09-14.
**Where:** the operator's Docker machine, CPU only, `docker run --rm --network none`. No pod, no GPU, no
download, no training.
**Image:** `plantseg-teacher:local`, image id
`sha256:8c1f31f6e7fc09c6de9d404cf76a647614748f6bc1765775391b4411bae9ba45`, labels
`org.opencontainers.image.revision=1dbd1a1d…` and `org.opencontainers.image.base.digest=sha256:b80b645d…866aaf`.
**Repository:** `1dbd1a1`. **Config under test:** the HEAD blob `9752882`, bind-mounted read-only at its
repository path.

> ## The findings in two lines
>
> **The teacher config has never been through mmengine's loader, and cannot load:** `Config.fromfile`
> classifies it as a lazy-import config because `import os` (`:42`) precedes `_base_` (`:44`) and
> Python 3.11 freezes `os`.
> **The teacher path runs strict determinism with no cuBLAS configuration:** MMEngine calls
> `use_deterministic_algorithms(True)` without `warn_only`, and nothing sets `CUBLAS_WORKSPACE_CONFIG`.

---

## 1. How this record was produced

The survey was first run by the previous session, and **its raw output was not preserved** — nothing in
the repository, nothing in this project's session scratch directories on this machine, an empty
terminal. Under N17 no
finding is taken from that session's summary. **Every item below was re-measured in this session**, and
every claim quotes the output it rests on. Where the re-measurement refines the summary, the section
says so.

| Item | Summary carried into this session | Re-measured here |
|---|---|---|
| A | registry holds 92 entries at runtime | **92**, and 92 in the build log (§2) |
| B | three of ch3 §D's four measures; `CUBLAS_WORKSPACE_CONFIG` `None → None` | `None → None` confirmed, and a framework-wide scan finds **no** mmengine, mmseg or mmcv source that sets it. **Refined:** two measures set exactly, the third only in *strict* form, the fourth absent. **New:** `set_random_seed` resets a caller's `warn_only=True` to `False` (§3) |
| `warn_only` consequence | `[INFERRED]` the fine-tune aborts at its first CUDA CE loss | Still `[INFERRED]`, **site refined**: mmseg's CE is not the call E1's warning attests; a cuBLAS raise in the decode head precedes it; a `histc` raise at the first validation follows it (§4) |
| C, D | blocked in `Config.fromfile` by a lazy-import misclassification; order-dependent | Confirmed through the whole source chain, **in both images**. **New:** the classification is **CWD-dependent** (§5) |
| D's instrument | double build in one process, `torch.equal` on NMF/Hamburger tensors | **Vacuous for NMF as specified** — the bases are drawn inside `forward`, not at construction (§5.6) |
| E | `mim` present, `mim.commands.download` importable | Confirmed, after correcting this session's own first instrument (§6) |

**Instruments.** Eight throwaway scripts in session scratch, plus two one-line `python -c` reads, each
run as

```bash
docker run --rm --network none \
  --mount type=bind,source=<scratch>/probe,target=/probe,readonly \
  --mount type=bind,source=<repo>/configs/teacher,target=/workspace/plantseg-thesis/configs/teacher,readonly \
  plantseg-teacher:local python -B /probe/<script>.py
```

Both mounts are read-only, so nothing could be written into a governed path; the container has no
network, so nothing could be downloaded. The two scripts that adjudicate an absence or carry the
simulation are embedded in full in the appendix. The rest print raw values and source excerpts, and
every line quoted from them below is verbatim. Like B55's probes, they are diagnostics rather than
repository tooling, which is why they live in `reports/`.

The config under test is the file the image carries: `git diff f77d05d7 1dbd1a1 -- configs/teacher/`
is empty, and the worktree blob equals the HEAD blob (`git hash-object` = `git rev-parse HEAD:<path>` =
`9752882…`).

---

## 2. A — the registry is populated (layer 3): PASS

**At build time** — `docker buildx history logs` for build `4wobye95magc9kohwtnsmxn50`, step `[5/5]`:

```text
#9 6.478 mmseg 1.2.2 imports; MODELS registry entries = 92
```

**At runtime**, in a fresh container:

```text
python      3.11.16 /usr/local/bin/python
torch       2.1.0+cu121
mmengine    0.10.7
mmcv        2.1.0
mmsegmentation 1.2.2
ftfy        6.3.0
openmim     0.3.9
CUBLAS_WORKSPACE_CONFIG at process start: None

A  len(MODELS.module_dict) = 92
A  'EncoderDecoder'         in MODELS.module_dict -> True
A  'MSCAN'                  in MODELS.module_dict -> True
A  'LightHamHead'           in MODELS.module_dict -> True
A  'CrossEntropyLoss'       in MODELS.module_dict -> True
A  'SegDataPreProcessor'    in MODELS.module_dict -> True
```

The five names are the model-level types the merged config names, and each resolves through the
registry (merged from the reordered copy of §5.3):

```text
model.type                           = 'EncoderDecoder'         MODELS.get -> <class 'mmseg.models.segmentors.encoder_decoder.EncoderDecoder'>
model.data_preprocessor.type         = 'SegDataPreProcessor'    MODELS.get -> <class 'mmseg.models.data_preprocessor.SegDataPreProcessor'>
model.backbone.type                  = 'MSCAN'                  MODELS.get -> <class 'mmseg.models.backbones.mscan.MSCAN'>
model.decode_head.type               = 'LightHamHead'           MODELS.get -> <class 'mmseg.models.decode_heads.ham_head.LightHamHead'>
model.decode_head.loss_decode.type   = 'CrossEntropyLoss'       MODELS.get -> <class 'mmseg.models.losses.cross_entropy_loss.CrossEntropyLoss'>
```

So this is layer 3 for **this model**, not merely a non-empty registry. Build and runtime agree.
Normalization-layer types are deliberately not walked here, because that is survey item C. `[MEASURED]`

**Image state.** The image `Env` sets `PLANTSEG_GIT_COMMIT=1dbd1a1…` and carries no
`CUBLAS_WORKSPACE_CONFIG`. The unfiltered build log shows export to the local store only (`naming to
docker.io/library/plantseg-teacher:local`), with no push step, and no registry-qualified tag for the image
exists on this machine. A pod cannot pull the image until it is pushed — G12's open remainder.

---

## 3. B — what MMEngine's determinism setup does, and G14

### 3.1 The measurement

`mmengine.runner.set_random_seed(42, deterministic=True)`, from process defaults:

```text
B1 CUBLAS_WORKSPACE_CONFIG                        before=None    after=None
B1 cudnn.deterministic                            before=False   after=True
B1 cudnn.benchmark                                before=False   after=False
B1 are_deterministic_algorithms_enabled           before=False   after=True
B1 is_deterministic_algorithms_warn_only_enabled  before=False   after=False
```

The code, `mmengine/runner/utils.py:78-90`:

```text
     78|     if deterministic:
     79|         if torch.backends.cudnn.benchmark:
     80|             print_log(
     81|                 'torch.backends.cudnn.benchmark is going to be set as '
     82|                 '`False` to cause cuDNN to deterministically select an '
     83|                 'algorithm',
     84|                 logger='current',
     85|                 level=logging.WARNING)
     86|         torch.backends.cudnn.deterministic = True
     87|         torch.backends.cudnn.benchmark = False
     88|
     89|         if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
     90|             torch.use_deterministic_algorithms(True)
```

Against ch3 §D's four measures, as tabulated in [B52 §6.6](b52_e1_seed42_completion.md):

| ch3 §D measure | E1 — `src/seeds.py` | Teacher — MMEngine |
|---|---|---|
| `cudnn.deterministic = True` | `:35` | **set** — `utils.py:86` |
| `cudnn.benchmark = False` | `:36` | **set** — `utils.py:87` |
| `use_deterministic_algorithms(True, warn_only=True)` | `:37` | **set in strict form** — `utils.py:90`, no `warn_only`; measured `False` |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `:26` | **absent** — `None → None` |

"Three of four" counts the third row as set. It is set, but in a **different mode** from the one ch3 §D
names, and that difference is §4's subject.

### 3.2 Coverage beyond one function — the whole framework, instrument stated per N17

That `set_random_seed` leaves the variable unset does not show that nothing on the teacher path sets it.
The adjudicated claim is **"no mmengine, mmseg or mmcv code sets it"**, so the instrument covers every
source file of all three (script: Appendix A.1):

```text
=== mmengine: 153 .py files and 0 .so files scanned under /usr/local/lib/python3.11/site-packages/mmengine
  [cublas (case-insensitive)] 0 hit(s)
  [use_deterministic_algorithms] 1 hit(s)
      mmengine/runner/utils.py:90: torch.use_deterministic_algorithms(True)
  [warn_only] 0 hit(s)
  [.so literal CUBLAS_WORKSPACE_CONFIG] 0 hit(s) []
=== mmseg: 997 .py files and 0 .so files scanned under /usr/local/lib/python3.11/site-packages/mmseg
  [cublas (case-insensitive)] 0 hit(s)
  [use_deterministic_algorithms] 0 hit(s)
  [warn_only] 0 hit(s)
  [.so literal CUBLAS_WORKSPACE_CONFIG] 0 hit(s) []
=== mmcv: 128 .py files and 1 .so files scanned under /usr/local/lib/python3.11/site-packages/mmcv
  [cublas (case-insensitive)] 2 hit(s)
      mmcv/ops/conv2d_gradfix.py:179: # Simple 1x1 convolution => cuBLAS (only on Volta, not on Ampere).
      mmcv/ops/conv2d_gradfix.py:249: # Simple 1x1 convolution => cuBLAS (on both Volta and Ampere).
  [use_deterministic_algorithms] 0 hit(s)
  [warn_only] 0 hit(s)
  [.so literal CUBLAS_WORKSPACE_CONFIG] 0 hit(s) []
=== positive control: identical line scanner on torch/__init__.py
  [cublas (case-insensitive)] 4 hit(s)
  [use_deterministic_algorithms] 6 hit(s)
  [warn_only] 8 hit(s)
```

- **Coverage.** Every `.py` under each package (`os.walk`, hidden `.mim` included). The file counts
  agree with an independent `find -name '*.py' | wc -l`: `153`, `997`, `128`. Each pattern contains its
  adjudicated string, and `cublas` is matched case-insensitively. Compiled extensions are byte-searched
  for the literal.
- **Sensitivity.** The identical scanner finds all three strings in `torch/__init__.py`, so it detects
  them where they are present.
- mmcv's two hits are comments, and set nothing.
- *Limit:* a variable name assembled at runtime from fragments would evade a literal scan. That is not
  claimed impossible; nothing indicates it.

**Repository code**, re-scanned at `1dbd1a1` — case-insensitive `cublas` over every text file under
`src/`, `scripts/` and `configs/` (115 `.py`, 2 `.json`, 1 `.sh`, 1 licence file), both Dockerfiles and the
`requirements*` files: hits only at `src/seeds.py:21,24,26`, on a path the teacher does not call
([B55 §6.3](b55_teacher_acquisition.md)). The image `Env` does not set it either (§2). `[MEASURED]`

### 3.3 The Runner reaches the same call

`Runner.__init__` runs `setup_env` before `set_randomness`, and `set_randomness` makes exactly the call
measured above:

```text
    372|         self.setup_env(env_cfg)
    375|         self._randomness_cfg = randomness
    376|         self.set_randomness(**randomness)
    ...
    666|         if env_cfg.get('cudnn_benchmark'):
    667|             torch.backends.cudnn.benchmark = True
    ...
    716|         self._seed = set_random_seed(
    718|             deterministic=deterministic,
    719|             diff_rank_seed=diff_rank_seed)
```

The merged config sets `env_cfg = {'cudnn_benchmark': True, …}` and `randomness = {'seed': 42,
'deterministic': True}`. So `setup_env` turns benchmark on (`:666-667`), and `set_randomness` turns it
off again (§3.4 shows the flip). `[MEASURED — source order, merged config, and the function's effect; no
Runner was constructed]`

### 3.4 Also measured: `set_random_seed` overrides a caller's `warn_only`

Pre-set to E1's mode, then the same call:

```text
09/14 07:44:36 - mmengine - WARNING - torch.backends.cudnn.benchmark is going to be set as `False` to cause cuDNN to deterministically select an algorithm
B2 CUBLAS_WORKSPACE_CONFIG                        before=None    after=None
B2 cudnn.deterministic                            before=True    after=True
B2 cudnn.benchmark                                before=True    after=False
B2 are_deterministic_algorithms_enabled           before=True    after=True
B2 is_deterministic_algorithms_warn_only_enabled  before=True    after=False
```

**A `warn_only=True` set before the Runner is constructed is silently reset to strict** — including one
set by calling `src.seeds.set_seed`. Recorded as a fact bearing on G15 and G18 (the teacher launcher,
raised as N24). No remedy is designed here.

### 3.5 G14 — CLOSED

**MMEngine 0.10.7 does not set `CUBLAS_WORKSPACE_CONFIG`, and no mmengine, mmseg, mmcv or teacher-path
repository code does.** B55 §6.3's first branch holds:

- **ch3 §D's reproducibility configuration does not hold for the teacher stage as the repository
  stands** — one measure absent, one in a different mode.
- **G8 is PRIMARY and NECESSARY for the teacher path** (`Dockerfile.teacher`) — not defence in depth.
  *[UPDATED 2026-09-15 — B58 reconciliation]*
  - **Status:** at HEAD it is **TO SHIP**, not an existing safeguard. No `ENV` for the variable exists in
    `Dockerfile.teacher` yet ([B57 E-2](b57_strict_mode_evidence.md)).
  - **Division of labour:** the selected G18 `TeacherRunner` *asserts* the Python-visible value before CUDA
    initialization and never creates it ([B58 §9–10](b58_teacher_runner_adjudication.md)).
  - **Evidentiary roles:**
    - the image `ENV` declaration and digest attest what the image supplies;
    - the launcher-entry echo and G18's first-call assertion attest what the run's own process saw.

    Neither they nor the subclass inspects cuBLAS's internal workspace state.
  - The default-DON'T of [B52 §11.3](b52_e1_seed42_completion.md) stands for the **E1 image**, which seeds
    2 and 3 must run against unchanged.
- The `warn_only` half of G14's "fixed or amended" is carried by **G15**.

---

## 4. The strict-mode consequence — `[INFERRED]`, with what confirms it

### 4.1 What is measured

- The teacher path enables strict determinism, with `warn_only=False` (§3.1).
- `CUBLAS_WORKSPACE_CONFIG` is unset on that path (§3.2).
- torch 2.1.0's own installed docstring for `use_deterministic_algorithms`:

```text
doc:  48|     The following normally-nondeterministic operations will throw a
doc:  49|     :class:`RuntimeError` when ``mode=True``:
...
doc:  75|         * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
...
doc:  81|         * :func:`torch.histc` when called on a CUDA tensor
...
doc:  91|     A handful of CUDA operations are nondeterministic if the CUDA version is
doc:  92|     10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
doc:  93|     or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set. See the CUDA documentation for more
doc:  94|     details: `<https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
doc:  95|     If one of these environment variable configurations is not set, a :class:`RuntimeError`
doc:  96|     will be raised from these operations when called with CUDA tensors:
doc:  97|
doc:  98|         * :func:`torch.mm`
doc:  99|         * :func:`torch.mv`
doc: 100|         * :func:`torch.bmm`
...
doc: 119|         warn_only (:class:`bool`, optional): If True, operations that do not
doc: 120|             have a deterministic implementation will throw a warning instead of
doc: 121|             an error. Default: ``False``
```

- The matching alert strings are compiled into this build — `uses CuBLAS` in `libtorch_cpu.so`;
  `nll_loss2d_forward_out_cuda_template` and `_histc_cuda` in `libtorch_cuda.so`. *Presence only:* a
  string in a binary shows that the check exists, not which branch reaches it.

### 4.2 Candidate raise sites, in execution order

These were found by reading the source of the teacher's own modules. **The list is not exhaustive** —
only a CUDA run enumerates what the real graph reaches.

| # | Site | Reached at | Evidence | What weakens it |
|---|---|---|---|---|
| 1 | `torch.bmm` in `NMF2D` — `ham_head.py:64`, `:100`, `:131-140`, `:149-151` | the first training forward, inside the decode head | docstring `doc:95-100`; the calls, by source | nothing found — strict mode with the variable unset is exactly the documented condition. **G8 removes this site** |
| 2 | the CE loss → `nll_loss2d` on CUDA | the first training forward, at the loss | docstring `doc:75`, which does not qualify by reduction | **the teacher's call is not E1's.** mmseg calls `F.cross_entropy(…, reduction='none', …)` (`cross_entropy_loss.py:45-50`) and reduces the result itself (`:75-76`); E1 calls it with the default `mean` (`src/training/losses.py:53`), which is the call B52 §6.2's production warning attests. Whether the alert fires on the unreduced branch of `nll_loss2d_forward_out_cuda_template` is **not determinable from the wheel** — the kernel is compiled and no C++ source ships |
| 3 | `torch.histc` in `IoUMetric.intersect_and_union` — `iou_metric.py:190-198`, on `pred_label`'s device, with `.cpu()` applied after | the first validation, at iteration 10,000 (config `:325`, `:338`) | docstring `doc:81`; the calls, by source | that `pred_label` is on CUDA when `process` runs is read from the flow (`:79`, `:82-83`), not executed |

**The summary's "abort at its first CUDA CE loss" is refined, not confirmed.** Site 1 comes earlier in
the same iteration and does not depend on the CE question. Site 2 rests on a call E1 never made. Site 3
survives any fix to sites 1 and 2, and would surface only after 10,000 paid iterations — the most
expensive place to discover it.

### 4.3 What confirms it

One CUDA session in `plantseg-teacher:local`, in the determinism state the teacher actually gets:
`mmengine.runner.set_random_seed(42, deterministic=True)`, nothing else, `CUBLAS_WORKSPACE_CONFIG` unset.
**Each call is isolated in its own `try`/`except`** and its exception text printed verbatim. The
isolation makes the result independent of the warning-deduplication regime that
[B52 §6.4](b52_e1_seed42_completion.md) leaves unverified.

1. `torch.bmm` on two CUDA tensors — site 1.
2. mmseg `CrossEntropyLoss(use_sigmoid=False)` on CUDA logits `(N, 116, H, W)` with `ignore_index=255` —
   site 2.
3. `IoUMetric.intersect_and_union` on CUDA tensors — site 3.
4. One `model.loss` → `backward` → AdamW step on the built teacher, from random initialisation — whatever
   else the real graph reaches first.

Then run 1–4 again with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before CUDA initialisation, which
separates what G8 alone removes. This takes seconds of GPU time and rides on the G2 pod (N25).

### 4.4 The G2 probe as written cannot see any of this — N25

[B55 Appendix A](b55_teacher_acquisition.md) seeds through `src.seeds.set_seed(42)`, which sets
`warn_only=True` **and** `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Both differ from the teacher's state (§3.1),
so the probe reproduces **neither site 1 nor site 2**. It runs no validation, so it cannot reach
**site 3** in any mode. A clean VRAM number from it would say nothing about whether the teacher can run.

**No G2 memory reading exists yet, and a future reading is usable for provisioning only conditionally.**
*[UPDATED 2026-09-15 — AMEND 1]* The current defect is a coverage/design defect in an unexecuted probe,
not a validated memory measurement with incomplete provenance. As written, the probe records only three
of the five required determinism states. Any future figure is NOT provisioning-grade unless the
launch/pre-CUDA CUBLAS state and all five states at first training forward and before validation are
recorded.

`[INFERRED]` The one determinism-driven memory term on record — D30's decomposition reroute of bilinear
`interpolate` — is keyed on whether determinism is enabled, not on `warn_only`. From
`torch/nn/functional.py` (torch 2.1.0):

```text
 4014|             if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
```

Determinism is enabled under both modes (§3.1, §3.4). The remedy is deferred: the G2 probe is not touched
until this session's items 1–3 land.

---

## 5. C and D — BLOCKED at `Config.fromfile`

### 5.1 The failure, verbatim

```text
   Config._is_lazy_import -> True
   Config.fromfile -> RAISED mmengine.config.utils.ConfigParsingError
   message: The configuration file type in the inheritance chain must match the current configuration file type, either "lazy_import" or non-"lazy_import". You got this error since you use the syntax like `_base_ = "_base_"` in your config. You should use `with read_base(): ... to` mark the inherited config file. See more information in https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
     at /usr/local/lib/python3.11/site-packages/mmengine/config/config.py:494 in fromfile: raise e
     at /usr/local/lib/python3.11/site-packages/mmengine/config/config.py:492 in fromfile: cfg_dict, imported_names = Config._parse_lazy_import(filename)
     at /usr/local/lib/python3.11/site-packages/mmengine/config/config.py:1044 in _parse_lazy_import: base_modules = Config._get_base_modules(parsed_codes.body)
     at /usr/local/lib/python3.11/site-packages/mmengine/config/config.py:579 in _get_base_modules: raise ConfigParsingError(
```

C (the SyncBN walk over the merged model) and D (the NMF double build) both begin with this call, so
neither ran.

### 5.2 The chain, link by link (script: Appendix A.2)

**1. Python 3.11 freezes `os`.**

```text
find_spec('os').origin           'frozen'
_imp.is_frozen('os')             True
'os' in sys.builtin_module_names False
```

**2. mmengine's builtin test therefore returns `False` for `os`** — `mmengine/config/utils.py:159-184`.
`os` is not in `sys.builtin_module_names` (`:169`), so the test falls through to `find_spec` (`:171`),
takes `osp.abspath` of the origin (`:178`), and requires the result to start with `PYTHON_ROOT_DIR` or
`SYSTEM_PYTHON_PREFIX` (`:179-182`). The origin `'frozen'` is not a path, so `abspath` resolves it
against the working directory:

```text
PYTHON_ROOT_DIR                  '/usr/local'
SYSTEM_PYTHON_PREFIX             '/usr/lib/python'
osp.abspath('frozen')            '/workspace/plantseg-thesis/frozen'
_is_builtin_module('os')         False
```

**3. The lazy-import predicate returns `True` at the first node it inspects** — `Config._is_lazy_import`,
`config.py:1658-1696`. `ast.walk` is breadth-first, so top-level statements are visited in file order.
The first relevant node is `import os` (`:42`), and the `ast.Import` branch returns `True` for any
name the builtin test rejects:

```text
   1692|             if isinstance(node, ast.Import):
   1693|                 for alias_node in node.names:
   1694|                     if not _is_builtin_module(alias_node.name):
   1695|                         return True
```

The `_base_` branch, which would have returned `False` (`:1666-1669`), is never reached.

**4. `fromfile` takes the lazy branch, and the lazy parser rejects `_base_`.** `config.py:459-460`
routes to lazy parsing when the predicate returns `True`. `_parse_lazy_import` (`:492`) calls
`_get_base_modules` (`:1044`), which raises at `:579` on `_base_ = […]` syntax (§5.1).

### 5.3 Order dependence — both orderings through mmengine's own predicate and loader

The reordered file is a scratch copy of the HEAD blob with lines 42 and 44 swapped. The repository file
is untouched.

| | HEAD | Reordered copy |
|---|---|---|
| sha256 | `6004209f14302737d70a2272772ada286de1e10b0a00e21bc9eb59ef5bed9cf4` | `510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5` |
| lines | 370 | 370 |
| first two top-level statements | `Import` `:42`, `Assign` `:44` | `Assign` `:42`, `Import` `:44` |
| predicate-relevant nodes, in `ast.walk` order | `Import :42`, `_base_ :44` | `_base_ :42`, `Import :44` |
| `Name 'os'` sites (AST) | `186, 364, 370` | `186, 364, 370` |
| `Config._is_lazy_import` | **`True`** | **`False`** |
| `Config.fromfile` | **raises** `ConfigParsingError` | **loads** — `decode_head.num_classes: 116`, no `os` key, `load_from` = the unset sentinel |

Every use of `os` sits at `:186` or later in both files, so the swap cannot put a use before its import.
The file has no other `import` or `from`, no `with`, and no `{{…}}` substitution, and it names `_base_`
only where it assigns it. `[MEASURED]`

**Independent of G12.** The identical script, run in the **E1 image**
(`sha256:b80b645d…866aaf`, no `ftfy`), produces **byte-identical output** (`cmp` exit 0) — `True` and a
raise for HEAD, `False` and a load for the copy:

```text
   Config._is_lazy_import -> True
   Config.fromfile -> RAISED mmengine.config.utils.ConfigParsingError
   ...
   Config._is_lazy_import -> False
   Config.fromfile -> LOADED | type(cfg): Config | decode_head.num_classes: 116 | 'os' key present: False | load_from: NEED_TO_CONFIRM__SET_SEGNEXT_ADE20K_CKPT
```

The E1 image cannot import `ftfy` (`ModuleNotFoundError: No module named 'ftfy'`), so it cannot import
`mmseg.models` either — yet the copy loads there. Loading a config does not need `mmseg.models`, so this
defect is older than the `ftfy` blocker and unrelated to it. `[MEASURED]`

### 5.4 The classification is CWD-dependent

Same process, with the working directory changed:

```text
cwd=/usr/local -> osp.abspath('frozen') = '/usr/local/frozen' -> _is_builtin_module('os') = True
```

Because the test applies `abspath` to a literal, **the failure reproduces only when the loader runs from
outside the Python prefix.** That includes the image's `WORKDIR`, `/workspace/plantseg-thesis`, where
B55 §7.1's steps run. A load check run from somewhere else can pass while the operational load fails, so
any regression check must run from the operational working directory. `[MEASURED]`

### 5.5 Why it was never caught

**The config has never been through mmengine's loader.**

- `scripts/smoke_teacher_config.py` loads the file with `exec(compile(…))` (`:59`). The file's own
  `import os` runs as plain Python, and mmengine's classifier never runs. The smoke's PASS is correct for
  what it checks and silent on loadability. Its `no_syncbn_anywhere` check (`:141`) scans only the delta
  file's code, so it cannot see anything `_base_` supplies — which is exactly what C exists to check.
- **No repository code has loaded it, and one repository path will try.** Nothing under `src/`,
  `scripts/` or `configs/` calls `Config.fromfile` directly; a repo-wide grep finds the call only in
  `reports/`. But the E2/E3 frozen-teacher loader reaches it indirectly. Its default factory
  (`src/distill/segnext_teacher.py:206-208`) calls `mmseg.apis.init_model(str(config_path), …)` (`:228`),
  and `init_model` calls `Config.fromfile` on any string path:

  ```text
     40|     if isinstance(config, (str, Path)):
     41|         config = Config.fromfile(config)
  ```

  `mmseg/apis/inference.py:40-41`. The factory's own error text names this derived config as the one to
  pass (`:226`). **G16 therefore blocks the E2/E3 teacher load, not only the fine-tune.** `[MEASURED —
  source chain; the loader was not run]`
- The instruments written to call `Config.fromfile` on this file — B55 Appendices A and B — have never
  executed (B55 §5, and the appendix preamble). On a pod with a working image, Appendix A would have
  exited **4** (".mim/configs/segnext PRESENT and `Config.fromfile` failed anyway → a different fault")
  before printing any VRAM line. `[INFERRED — from Appendix A's source and §5.2–5.4, for a run from the
  WORKDIR]` The 3/4 split in its exit contract was right, and the step would still have cost a pod cycle.

Queued as **G17**.

### 5.6 D — the instrument, settled at source before it is run

The merged config inherits `rand_init=True`, and the SegNeXt-B base neither overrides `ham_kwargs` nor
uses `_delete_` (zero occurrences of each):

```text
--- merged (reordered copy): decode_head.type = LightHamHead
--- merged: decode_head.ham_kwargs = {'MD_S': 1, 'MD_R': 16, 'train_steps': 6, 'eval_steps': 7, 'inv_t': 100, 'rand_init': True}
```

With `rand_init=True`, `mmseg/models/decode_heads/ham_head.py` behaves as follows:

- `Matrix_Decomposition_2D_Base.__init__` (`:37-54`) stores scalars only, so **NMF holds no tensor at
  construction.** A `bases` buffer is registered only when `rand_init` is `False`, and even then only on
  the first forward (`:84-86`).
- `forward` draws fresh bases **on every call** (`:89-90`).
- `NMF2D._build_bases` draws them from the **CPU default generator** and then moves them:
  `torch.rand((B * S, D, R)).to(device)` (`:123`).

**Consequences for D as [B55 §7.1](b55_teacher_acquisition.md) step 5b specifies it** — build twice in one
process, then `torch.equal` on LightHamHead's NMF/Hamburger tensors:

- A comparison after construction covers the Hamburger's `ConvModule` parameters and **no NMF
  randomness**, because none exists yet. An equal result would be vacuous for the question D exists to
  answer.
- Step 5b's premise, "NMF init happens during model construction", is **false for this config**.
- To cover the draw that is actually being adjudicated, each build must be followed by a **forward pass on
  a fixed input**, comparing the drawn bases or the decode-head output. It also needs a **negative
  control**: a forward without reseeding must differ, which shows that `torch.equal` was able to return
  `False`. The same-process requirement stands. CPU suffices, because the draw uses the CPU generator.

`[MEASURED — source and merged config; D itself not run]`

---

## 6. E — `mim` present, download command usable: PASS

```text
E  shutil.which('mim') = /usr/local/bin/mim
E  openmim version     = 0.3.9
E  module object       = module mim.commands.download /usr/local/lib/python3.11/site-packages/mim/commands/download.py
E  module.download     = function | callable: True
E  signature           = (package: str, configs: Optional[List[str]] = None, dest_root: Optional[str] = None, check_certificate: bool = True, dataset: Optional[str] = None) -> Optional[List[str]]
E  mim.commands.download is the function (shadowing): True
E  `mim download --help` returncode = 0
E  | Usage: mim download [OPTIONS] PACKAGE
```

The CLI command is registered, and its function imports and is callable. Nothing was downloaded:
`--help` exits before the command body runs, and the container had no network. `[MEASURED]`

**Instrument note — this session's own slip, recorded.** The first E probe used
`import mim.commands.download as mcd` and printed `download() callable: False`. That import form binds
by attribute lookup, and `mim.commands` re-exports the *function* `download`, which shadows the submodule
of the same name — so the check tested an attribute of a function. The raw output exposed this
(`__name__` printed `download`, not `mim.commands.download`) before anything was recorded. The corrected
probe above fetches the module through `importlib`.

---

## 7. Corrections to committed records, and the pattern ledger

**B55 §3.1, `[CORRECTION 2026-09-14 — layer 4 was never at risk]`.** Its conclusion that "layer 4 was
fine throughout" is **withdrawn for the operative config**. What the wheel inspection established still
stands: `.mim/configs/segnext/` ships and `mmseg::` resolution works — reconfirmed here, because the
reordered copy resolves its `_base_` and loads. What the inspection did not establish is that **the
thesis config loads**, and it does not (§5). Layer 4 failed for the config that matters, by a mechanism
the correction never examined.

**B55 §7.1 step 5b.** The premise that NMF initialisation happens during model construction is corrected
by §5.6, and the step's instrument with it.

**B55 §7.1 step 5** (the MMEngine determinism echo plus `getsource(set_random_seed)`) is **done**, without
a pod (§3). It drops out of the next pod's sequence.

The ledger continues from [B53 §5](b53_uvm_host_fault.md) (instances 1–5) and B55 §3.2 (instance 6):

| # | Instance | Caught |
|---|---|---|
| 7 | B55 §3.1's "layer 4 was fine throughout": evidence about the wheel's stock config tree, read as evidence that the operative config resolves. Committed in `f2e8186`, which the local remote-tracking ref shows on `origin` | by a CPU survey on the operator's machine, before a pod was rented — zero cost |
| 8 | B55 §7.1 step 5b's D instrument: a construction-time `torch.equal`, adjudicating randomness that is drawn per forward | at source, before it was run |
| 9 | This session's first E probe: an import form that bound a re-exported function instead of the module it named | from its own raw output, before recording |

All three share N17's shape: an instrument whose coverage of the adjudicated object was never
established.

---

## 8. Register changes — applied to B52 §11 in the same change

| # | Change | Class |
|---|---|---|
| G8 | **Updated:** PRIMARY and NECESSARY for the teacher path, still to ship (§3.5, updated 2026-09-15); default-DON'T unchanged for the E1 image | governed — by consequence |
| G12 | **Built** 2026-09-14 and verified (§2); push to a registry outstanding | governed — `requirements*` |
| G13 | Carried in from B55 §9, unchanged | governed — `scripts/**` |
| G14 | **Closed** (§3.5); its consequences move to G8 and G15 | governed |
| G15 | **New:** the teacher's strict-mode mismatch (§4) | governed — resolution needs launch code, or a ch3 §D / contract B6 amendment |
| G16 | **New:** the two-line config reorder (§5); the defect blocks the fine-tune and the E2/E3 teacher load | governed — `configs/**` |
| G17 | **New:** `smoke_teacher_config.py` never exercises mmengine's loader or the merged config (§5.5) | governed — `scripts/**` |
| G18 *(raised as N24)* | **Reclassified:** the teacher launcher. Its deliverable is code on the path list, so B52 §11.1's floor rule governs it, as it did G9 and G11 | governed — by path |
| N22, N23 | Carried in from B55 §9. N23 was exercised locally (§2) but is not yet in the runbook | non-governed |
| N25 | **New:** the G2 probe's coverage defect (§4.4) | non-governed — the probe lives in `reports/` |

---

## 9. What this survey establishes, and what it does not

**Establishes:**

- The teacher image imports `mmseg.models`, and its registry resolves the five model-level types this
  config names (§2).
- MMEngine 0.10.7's determinism setup: two of ch3 §D's measures set exactly, one in strict form, the cuBLAS
  configuration absent — across the whole framework, not only in one function (§3).
- Why the teacher config cannot load, to the source line, in both images; that a two-line reorder
  makes it load; and that the same defect sits on the E2/E3 teacher-load path (§5).
- That D, as specified, would not have tested NMF seeding (§5.6).
- That `mim` is usable in the image (§6).

**Does not establish:**

- Whether the teacher raises on CUDA, or where it raises first — `[INFERRED]`, G15, confirmation in §4.3.
- Teacher VRAM — G2, unchanged.
- Anything C or D would show — neither was run.
- Anything about the checkpoint — the four `docs/teacher_init_source.md` fields stay `NEED_TO_CONFIRM`.

**Next, in this session's order:** the G16 plan (item 2); then, after an explicit go, C and D, with D's
instrument as corrected in §5.6 (item 3). G18, the checkpoint download and the G2 probe stay untouched
until those land.

---

## Appendix — the two instruments that adjudicate an absence or carry the simulation

Both ran unmodified in the containers described in §1. The remaining scripts are not embedded: they
print raw values and source excerpts, and every line quoted from them above is verbatim.

### A.1 `b56_scan.py`

sha256 `af83717a8c32ba54977f3218d43b23763060cdc753a4dc7230d314b361e8bc08` · 75 lines · 3042 bytes

```python
#!/usr/bin/env python3
"""B56 -- coverage-stated scans of the installed teacher-path frameworks.

Adjudicates: does any mmengine / mmseg / mmcv source set CUBLAS_WORKSPACE_CONFIG, call
use_deterministic_algorithms, or pass warn_only?

Instrument: every *.py under each package directory (os.walk, hidden dirs such as .mim included),
three per-line regexes whose patterns contain the adjudicated strings; plus a literal byte search of
every compiled extension (*.so) for b'CUBLAS_WORKSPACE_CONFIG'.
Positive control: the same line scanner on torch/__init__.py, which is known to document the variable.
Supporting presence checks: alert strings in this exact torch build's shared libraries.
"""
import sys

sys.dont_write_bytecode = True

import mmap
import os
import re

SP = "/usr/local/lib/python3.11/site-packages"
PATS = [
    ("cublas (case-insensitive)", re.compile(r"cublas", re.I)),
    ("use_deterministic_algorithms", re.compile(r"use_deterministic_algorithms")),
    ("warn_only", re.compile(r"warn_only")),
]


def scan_file(path, hits):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            for key, rx in PATS:
                if rx.search(line):
                    hits[key].append(f"{os.path.relpath(path, SP)}:{i}: {line.strip()[:140]}")


def bytes_in(path, needle):
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size == 0:
            return False
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(needle) != -1


for pkg in ("mmengine", "mmseg", "mmcv"):
    root = os.path.join(SP, pkg)
    hits = {k: [] for k, _ in PATS}
    n_py, so_files = 0, []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if f.endswith(".py"):
                n_py += 1
                scan_file(p, hits)
            elif f.endswith(".so"):
                so_files.append(p)
    so_hits = [os.path.relpath(p, SP) for p in so_files if bytes_in(p, b"CUBLAS_WORKSPACE_CONFIG")]
    print(f"=== {pkg}: {n_py} .py files and {len(so_files)} .so files scanned under {root}")
    for key, _ in PATS:
        print(f"  [{key}] {len(hits[key])} hit(s)")
        for h in hits[key]:
            print("      " + h)
    print(f"  [.so literal CUBLAS_WORKSPACE_CONFIG] {len(so_hits)} hit(s) {so_hits}")

print("=== positive control: identical line scanner on torch/__init__.py")
ctl = {k: [] for k, _ in PATS}
scan_file(os.path.join(SP, "torch", "__init__.py"), ctl)
for key, _ in PATS:
    print(f"  [{key}] {len(ctl[key])} hit(s)")

print("=== presence of alert strings in this torch build's shared libraries (presence only)")
lib = os.path.join(SP, "torch", "lib")
for needle in (b"CUBLAS_WORKSPACE_CONFIG", b"uses CuBLAS", b"nll_loss2d_forward_out_cuda_template", b"_histc_cuda"):
    found = [f for f in sorted(os.listdir(lib)) if f.endswith(".so") and bytes_in(os.path.join(lib, f), needle)]
    print(f"  {needle.decode():40s} -> {found}")
```

### A.2 `b56_lazy.py`

sha256 `a89395e72d303b97efa27003a4669d5a6644da7efeecbe308fffcdd815bb28fa` · 94 lines · 4427 bytes

```python
#!/usr/bin/env python3
"""B56 -- why Config.fromfile cannot load the teacher config: the source chain, and both orderings
run through mmengine's own predicate and loader. Read-only; CPU; --network none.

CUR = the HEAD config, bind-mounted read-only at its repository path.
REO = a scratch copy with lines 42 and 44 swapped (the proposed reorder). Not the repository file.
"""
import sys

sys.dont_write_bytecode = True

import _imp
import ast
import hashlib
import inspect
import os
import traceback
from importlib.util import find_spec

CUR = "/workspace/plantseg-thesis/configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"
REO = "/probe/reordered/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py"


def src(obj):
    lines, start = inspect.getsourcelines(obj)
    print(f"  source {inspect.getsourcefile(obj)}:{start}-{start + len(lines) - 1}")
    for i, line in enumerate(lines):
        print(f"  {start + i:5d}| {line.rstrip()}")


print("== 1. os is frozen on Python 3.11")
print("python                          ", sys.version.split()[0], sys.executable)
print("cwd                             ", os.getcwd())
print("find_spec('os').origin          ", repr(find_spec("os").origin))
print("_imp.is_frozen('os')            ", _imp.is_frozen("os"))
print("'os' in sys.builtin_module_names", "os" in sys.builtin_module_names)

print("\n== 2. mmengine's builtin-module test")
import mmengine  # noqa: E402
import mmengine.config.utils as cu  # noqa: E402

print("mmengine                        ", mmengine.__version__)
print("PYTHON_ROOT_DIR                 ", repr(cu.PYTHON_ROOT_DIR))
print("SYSTEM_PYTHON_PREFIX            ", repr(cu.SYSTEM_PYTHON_PREFIX))
print("osp.abspath('frozen')           ", repr(os.path.abspath("frozen")))
print("_is_builtin_module('os')        ", cu._is_builtin_module("os"))
src(cu._is_builtin_module)

print("\n== 3. the lazy-import predicate and its call site in Config.fromfile")
from mmengine.config import Config  # noqa: E402

src(Config._is_lazy_import)
lines, start = inspect.getsourcelines(Config.fromfile)
print(f"  Config.fromfile lines mentioning lazy import ({inspect.getsourcefile(Config.fromfile)}):")
for i, line in enumerate(lines):
    if "lazy" in line.lower():
        print(f"  {start + i:5d}| {line.rstrip()}")

print("\n== 4. both orderings")
for label, path in (("CURRENT (HEAD)", CUR), ("REORDERED (scratch copy)", REO)):
    raw = open(path, "rb").read()
    text = raw.decode("utf-8")
    tree = ast.parse(text)
    print(f"\n-- {label}: {path}")
    print("   sha256", hashlib.sha256(raw).hexdigest(), "| lines", text.count("\n"))
    print("   first 2 top-level statements:", [(type(s).__name__, s.lineno) for s in tree.body[:2]])
    relevant = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With)):
            relevant.append((type(node).__name__, node.lineno))
        elif (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == "_base_"):
            relevant.append(("Assign _base_", node.lineno))
    print("   predicate-relevant nodes, in ast.walk order:", relevant)
    print("   Name 'os' sites (AST):", sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "os"}))
    print("   Name '_base_' sites (AST):", sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "_base_"}))
    print("   '{{' substitution markers in file:", text.count("{{"))
    print("   Config._is_lazy_import ->", Config._is_lazy_import(path))
    try:
        cfg = Config.fromfile(path)
        print("   Config.fromfile -> LOADED | type(cfg):", type(cfg).__name__,
              "| decode_head.num_classes:", cfg.model.decode_head.num_classes,
              "| 'os' key present:", "os" in cfg,
              "| load_from:", cfg.load_from)
    except Exception as e:  # noqa: BLE001
        print("   Config.fromfile -> RAISED", f"{type(e).__module__}.{type(e).__name__}")
        print("   message:", str(e))
        for fr in traceback.extract_tb(e.__traceback__)[-4:]:
            print(f"     at {fr.filename}:{fr.lineno} in {fr.name}: {fr.line}")

print("\n== 5. CWD dependence of the builtin test (same process)")
os.chdir("/usr/local")
print("cwd=/usr/local -> osp.abspath('frozen') =", repr(os.path.abspath("frozen")),
      "-> _is_builtin_module('os') =", cu._is_builtin_module("os"))
```
