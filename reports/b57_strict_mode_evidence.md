# B57 — strict-mode evidence pass, E-1 to E-6

**Type:** read-only evidence pass. **Scope:** no existing file edited, no governed path touched, nothing
committed; this report is the only file written. **Date:** 2026-09-14. **Repository:** `1dbd1a1`.
**Images:** E1 `sha256:b80b645d…866aaf`; teacher `plantseg-teacher:local` (`sha256:8c1f31f6…9ba45`).
**Execution:** CPU only, `--network none`, read-only mounts. **No GPU reproduction was attempted.**

> **Every finding below is a hypothesis.** Each item shows the bare command, its raw output and its exit
> code, taken verbatim from a transcript runner that prints `$ <command>`, then the command's stdout and
> stderr, then `[exit N]`. Where a command as given cannot reach what it was meant to adjudicate, a
> coverage note says so, and a covering command follows in the same transcript. The probe scripts that
> the transcripts call are embedded verbatim in the appendix.

## Hypotheses at a glance

| # | Hypothesis | Rests on |
|---|---|---|
| H-E1 | **MEASURED:** the verified E1 log contains one line of the not-deterministic alert class, naming `nll_loss2d_forward_out_cuda_template` — one *printed* occurrence, not one execution. **[INFERRED — COUNTERFACTUAL]:** under `warn_only=False`, E1 would have raised at that op within its first 50 iterations. *[UPDATED 2026-09-15 — B58 reconciliation]* | measured log (E-1); the counterfactual rests on E-6's CPU mechanism |
| H-E2 | All four ch3 §D measures are set by `src/seeds.py`, in-process; neither image's ENV supplies any. `PYTHONHASHSEED` at `:23` does not seed the interpreter that sets it | measured |
| H-E3 | G14 is pinned: mmengine `0.10.7`, `mmengine/runner/utils.py` (sha256 `2649223c…a5a3`, equal to the wheel's RECORD digest), calls `torch.use_deterministic_algorithms(True)` with no `warn_only` — a call its public docstring never mentions | measured |
| H-E4 | The operative teacher config sets `deterministic=True` (`:352`), and nothing in its `_base_` chain sets `randomness`, so MMEngine's strict path fires. The E-4 command as given read a different file | measured, text only |
| H-E5 | At pinned mmseg 1.2.2, CE calls `F.cross_entropy(…, reduction='none')`, and a bilinear `resize` sits inside `loss_by_feat` before the loss. That resize is probably **not** a strict-mode raise site | measured source; the last sentence is inferred |
| H-E6 | `set_random_seed` turns a warning into a raise; a later `use_deterministic_algorithms(True, warn_only=True)` turns it back; the cuBLAS variable survives both. *[UPDATED 2026-09-15 — B58 reconciliation]* A one-shot override after construction is **not** the selected fix: it lands after CUDA discovery and is not resume-safe. G18 selects a `TeacherRunner` that establishes the policy inside `set_randomness`, before CUDA initialization ([B58 §9](b58_teacher_runner_adjudication.md)) | measured on CPU; source reads |

---

## E-1 — the E1 production log, by alert class

**Question.** Which operations raised the "does not have a deterministic implementation" alert in the E1
seed-42 production run? That set is what becomes fatal under `warn_only=False`.

**The log.** `~/plantseg_runs/e1_seed42/e1_stdout.log` on the operator's machine. Its sha256, first in the
transcript, equals [B52 §8.1](b52_e1_seed42_completion.md) and the directory's `SHA256SUMS.txt`.

````text
$ sha256sum ~/plantseg_runs/e1_seed42/e1_stdout.log; wc -c -l ~/plantseg_runs/e1_seed42/e1_stdout.log
fa183481140147102342606237ff3d702b7d971abb2ded71e7336d977439044b */c/Users/admin/plantseg_runs/e1_seed42/e1_stdout.log
  1706 124276 /c/Users/admin/plantseg_runs/e1_seed42/e1_stdout.log
[exit 0]

$ grep -n "does not have a deterministic implementation" ~/plantseg_runs/e1_seed42/e1_stdout.log
14:/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
[exit 0]

$ grep -o -E "[^[:space:]]+ does not have a deterministic implementation" ~/plantseg_runs/e1_seed42/e1_stdout.log | sed 's/ does not have a deterministic implementation$//' | sort | uniq -c
      1 nll_loss2d_forward_out_cuda_template
[exit 0]

$ grep -n -i -E "determinis|cublas" ~/plantseg_runs/e1_seed42/e1_stdout.log
14:/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
[exit 0]

$ grep -n -E "Warning|warn" ~/plantseg_runs/e1_seed42/e1_stdout.log
14:/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
[exit 0]

$ grep -rn -E "filterwarnings|simplefilter|PYTHONWARNINGS" src/ scripts/ --include=*.py
src/stats/tests.py:94:        _warnings.simplefilter("always")
src/stats/tests.py:140:        _warnings.simplefilter("always")
scripts/smoke_qnnpack_full_student.py:25:warnings.filterwarnings("ignore", category=DeprecationWarning)
scripts/smoke_qnnpack_full_student.py:26:warnings.filterwarnings("ignore", message=".*reduce_range.*")
scripts/smoke_qnnpack_head.py:29:warnings.filterwarnings("ignore", category=DeprecationWarning)
scripts/smoke_qnnpack_head.py:30:warnings.filterwarnings("ignore", message=".*reduce_range.*")
scripts/verify_plantseg_dataset.py:39:warnings.simplefilter("ignore", Image.DecompressionBombWarning)
[exit 0]

$ grep -n -E "^\s*(import|from)\s" src/training/train_e1.py
18:from __future__ import annotations
20:import argparse
21:import json
22:import os
23:import random
24:import subprocess
25:import sys
26:import tempfile
27:import time
28:from collections import deque
29:from pathlib import Path
31:import numpy as np
37:import torch  # noqa: E402
39:from configs.data import DATA                                   # noqa: E402
40:from configs.e1_student import E1_STUDENT                       # noqa: E402
41:from src.data import NUM_CLASSES, build_dataloader             # noqa: E402
42:from src.eval.metrics import confusion_matrix, miou_from_confusion  # noqa: E402
43:from src.models.student import build_student                    # noqa: E402
44:from src.seeds import set_seed                                  # noqa: E402
45:from src.training.losses import CombinedCEDiceLoss             # noqa: E402
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf python -c "import os, warnings; print(warnings.filters); print('defaultaction =', repr(warnings.defaultaction)); print('PYTHONWARNINGS =', repr(os.environ.get('PYTHONWARNINGS')))"
[('ignore', re.compile('.+ distutils\\b.+ deprecated', re.IGNORECASE), <class 'DeprecationWarning'>, None, 0), ('default', None, <class 'DeprecationWarning'>, '__main__', 0), ('ignore', None, <class 'DeprecationWarning'>, None, 0), ('ignore', None, <class 'PendingDeprecationWarning'>, None, 0), ('ignore', None, <class 'ImportWarning'>, None, 0), ('ignore', None, <class 'ResourceWarning'>, None, 0)]
defaultaction = 'default'
PYTHONWARNINGS = None
[exit 0]

$ sed -n '12,16p' ~/plantseg_runs/e1_seed42/e1_stdout.log
[provenance] git_head=f77d05d7b35187bf0da7e7b94a629549fe2e1c05 (source=git_checkout) image_digest=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
[jsonl] telemetry -> /workspace/e1_ckpts/e1_telemetry.jsonl
/usr/local/lib/python3.11/site-packages/torch/nn/functional.py:3053: UserWarning: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
  return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
[iter   50/80000] loss=5.3376 ce=4.3811 dice=0.9565 lr=9.99437482e-03
[exit 0]
````

**Coverage.**

- The given pattern matches the not-deterministic alert class: its text is fixed at `Context.cpp:71`,
  and the op name comes first. It does **not** match the cuBLAS-configuration alert, which is worded
  differently. The case-insensitive `determinis|cublas` grep and the case-sensitive `Warning|warn` grep
  return the same single line, so **within those instruments the 1,706-line log holds one matching line.**
  The case-sensitive pattern cannot see an uppercase `WARNING:` line, so "one warning of any kind" is not
  claimed. *[UPDATED 2026-09-15 — B58 reconciliation]*
- **Counts are printed occurrences, not executions.** `UserWarning` has no entry in the interpreter's filter
  list, `defaultaction` is `'default'`, and `PYTHONWARNINGS` is unset. Python's documented `default` action
  prints the first occurrence per issuing location. Distinct ops carry distinct message text, so each would
  print at least once.
- No warning suppression reaches the E1 launch. The `filterwarnings`/`simplefilter` hits are in
  `src/stats/tests.py`, whose `"always"` prints more rather than less, and in three `scripts/` entry
  points. `train_e1.py` imports no `scripts/` module — its import list is in the transcript.
- The warning, at log line 14, comes before the first progress line (`[iter 50/80000]`, line 16).

**H-E1 — MEASURED.** In the verified log, the only line of this alert class names
`nll_loss2d_forward_out_cuda_template`, printed once, before the first progress line. One printed warning is
not one execution: the loss is computed every iteration, and Python's `default` action prints a given
warning once per issuing location.

**H-E1 — [INFERRED — COUNTERFACTUAL].** The alert fires whenever determinism is enabled; `warn_only` only
chooses between a warning and a raise, and E-6 shows that flip on CPU for an op of the same class. So under
`warn_only=False`, E1 would have raised at its CE forward before iteration 50 — and, within what this log
can show, at no other op of this class. No strict-mode E1 run exists; only such a run would confirm it.
*[UPDATED 2026-09-15 — B58 reconciliation]*

**Limits.**

1. E1's `CUBLAS_WORKSPACE_CONFIG` is attested at code level only ([B52 §6.6](b52_e1_seed42_completion.md);
   runtime-attested for the same image and commit in [B55 §8.2](b55_teacher_acquisition.md), not for this
   run), so its log cannot reveal cuBLAS-class raise sites.
2. E1's CE call used the default `mean` reduction, so the log cannot show whether mmseg's
   `reduction='none'` call (E-5) alerts.
3. The teacher reaches ops E1 never ran ([B56 §4.2](b56_teacher_survey.md)), and E1's one alerting call is
   not the call the teacher's CE makes (limit 2). **E1's set neither bounds nor answers the teacher's.**
   *[UPDATED 2026-09-15 — B58 reconciliation]*

---

## E-2 — what `src/seeds.py` sets, who calls it, and what the images supply

````text
$ grep -n -E "use_deterministic_algorithms|warn_only|CUBLAS_WORKSPACE_CONFIG|PYTHONHASHSEED" src/seeds.py
20:    Sets cuDNN deterministic mode, torch deterministic algorithms (warn_only=True), and
21:    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
23:    os.environ["PYTHONHASHSEED"] = str(seed)
26:    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
37:    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops
[exit 0]

$ grep -rn "set_seed(" src/ scripts/
src/seeds.py:17:def set_seed(seed: int = 42) -> int:
src/training/train_distill.py:260:    set_seed(seed)
src/training/train_e1.py:5:  set_seed(42) -> build_student -> build_dataloader(train/val) -> CombinedCEDiceLoss(class-weighted
src/training/train_e1.py:301:    set_seed(seed)
Binary file src/training/__pycache__/train_e1.cpython-313.pyc matches
scripts/smoke_aug_stochasticity.py:83:    set_seed(42)
scripts/smoke_aug_stochasticity.py:95:        set_seed(42)
scripts/smoke_aug_stochasticity.py:105:    set_seed(42)
scripts/smoke_aug_stochasticity.py:109:    set_seed(42)
scripts/smoke_cwd_projection.py:24:    set_seed()
scripts/smoke_dataloader.py:43:    set_seed()
scripts/smoke_distill.py:56:    set_seed(42)
scripts/smoke_distill.py:93:    set_seed(42)
scripts/smoke_distill.py:160:    set_seed(42)
scripts/smoke_distill.py:194:    set_seed(42)
scripts/smoke_distill.py:246:    set_seed(42)
scripts/smoke_distill.py:397:    set_seed(7)
scripts/smoke_eval_int8.py:57:    set_seed(42)
scripts/smoke_kd_resolution.py:62:    set_seed(42)
scripts/smoke_loss.py:49:    set_seed()
scripts/smoke_losses_metrics.py:37:    set_seed()
scripts/smoke_metrics.py:39:    set_seed()
scripts/smoke_qnnpack_full_student.py:40:    set_seed()
scripts/smoke_qnnpack_head.py:112:    set_seed()
scripts/smoke_quant_e4_e5.py:126:    set_seed(42)
scripts/smoke_quant_e4_e5.py:200:    set_seed(42)
scripts/smoke_quant_e4_e5.py:263:    set_seed(0)
scripts/smoke_quant_e6_e7.py:96:    set_seed(0)
scripts/smoke_quant_e6_e7.py:192:    set_seed(42)
scripts/smoke_student_forward.py:24:    set_seed()
scripts/smoke_teacher.py:165:    set_seed(42)
scripts/smoke_teacher.py:192:    set_seed(42)
scripts/smoke_teacher.py:216:    set_seed(42)
scripts/smoke_teacher.py:250:    set_seed(42)
scripts/smoke_train_step.py:33:    set_seed(42)
[exit 0]

$ grep -n -E "cudnn\.(deterministic|benchmark)" src/seeds.py
35:    torch.backends.cudnn.deterministic = True
36:    torch.backends.cudnn.benchmark = False
[exit 0]

$ docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf
PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LANG=C.UTF-8
GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D
PYTHON_VERSION=3.11.16
PYTHON_SHA256=91bcdebfdde239a003ae93738a7fce0f9230fee5c4bc2b86f6e6e8c6f98aabe8
DEBIAN_FRONTEND=noninteractive
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_ROOT_USER_ACTION=ignore
PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
PYTHONPATH=/workspace/plantseg-thesis
PLANTSEG_GIT_COMMIT=f77d05d7b35187bf0da7e7b94a629549fe2e1c05

[exit 0]

$ docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' plantseg-teacher:local
PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LANG=C.UTF-8
GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D
PYTHON_VERSION=3.11.16
PYTHON_SHA256=91bcdebfdde239a003ae93738a7fce0f9230fee5c4bc2b86f6e6e8c6f98aabe8
DEBIAN_FRONTEND=noninteractive
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_ROOT_USER_ACTION=ignore
PLANTSEG_DATA_ROOT=/workspace/plantseg_data/plantseg
PYTHONPATH=/workspace/plantseg-thesis
PLANTSEG_GIT_COMMIT=1dbd1a1dcc03a70f4ef5b329be9e1b59df4ab088

[exit 0]

$ grep -n -i -E "cublas|PYTHONHASHSEED|determinis|cudnn" Dockerfile Dockerfile.teacher
Dockerfile:19:# Deterministic, non-interactive, no bytecode, unbuffered logs (RunPod streams stdout).
Dockerfile:40:# that residual non-determinism is recorded in docs/runpod_environment.md.
[exit 0]

$ for run in 1 2; do MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/plantseg-thesis/src,target=/workspace/plantseg-thesis/src,readonly ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf python -B -c "import os; from src.seeds import set_seed; set_seed(42); print('after set_seed: PYTHONHASHSEED =', os.environ['PYTHONHASHSEED'], '| hash(\"plantseg\") =', hash('plantseg'))"; done
after set_seed: PYTHONHASHSEED = 42 | hash("plantseg") = 5517237912415358457
after set_seed: PYTHONHASHSEED = 42 | hash("plantseg") = 257990339446526992
[exit 0]

$ for run in 1 2; do MSYS_NO_PATHCONV=1 docker run --rm --network none -e PYTHONHASHSEED=42 ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf python -B -c "print('PYTHONHASHSEED=42 at interpreter start | hash(\"plantseg\") =', hash('plantseg'))"; done
PYTHONHASHSEED=42 at interpreter start | hash("plantseg") = -7394851924393689482
PYTHONHASHSEED=42 at interpreter start | hash("plantseg") = -7394851924393689482
[exit 0]
````

**Coverage.** The given pattern contains neither `cudnn.deterministic` nor `cudnn.benchmark`, so on its own
it answers for two of ch3 §D's four measures; the third command covers the other two. The caller grep also
matched the definition, a docstring line and a `.pyc`. Its scope (`src/`, `scripts/`) excludes `reports/`,
where B55 Appendix A calls `set_seed`.

**H-E2.**

| ch3 §D measure | `src/seeds.py` | E1 image ENV | teacher image ENV |
|---|---|---|---|
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `:26`, in-process | not set | not set |
| `cudnn.deterministic = True` | `:35` | — (API call) | — (API call) |
| `cudnn.benchmark = False` | `:36` | — (API call) | — (API call) |
| `use_deterministic_algorithms(True, warn_only=True)` | `:37` | — (API call) | — (API call) |

All four are set by `src/seeds.py` in-process. Neither image's ENV carries `CUBLAS_WORKSPACE_CONFIG`,
`PYTHONHASHSEED` or any other determinism variable; both Dockerfiles match only in comments
(`Dockerfile:19`, `:40`). The training-path callers are `train_e1.py:301` and `train_distill.py:260`.
`scripts/launch_teacher_finetune.py` is not among the callers.

**Also measured — `:23` does not seed the interpreter that runs it.** Two fresh interpreters, each after
`set_seed(42)`, give different `hash("plantseg")` values. The control — `PYTHONHASHSEED=42` exported
before the interpreter starts — gives the same value twice, so the instrument can show equality. Whether
the variable reaches DataLoader workers is **not measured**: fork-started workers inherit the parent's
hash secret, while spawn-started ones would read it. `PYTHONHASHSEED` is not one of ch3 §D's four
measures.

---

## E-3 — G14, pinned to a version and a file

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local python -c "import mmengine, inspect, mmengine.runner.utils as u; \
  print(mmengine.__version__); print(inspect.getsourcefile(u)); \
  print(inspect.getsource(u.set_random_seed))"
0.10.7
/usr/local/lib/python3.11/site-packages/mmengine/runner/utils.py
def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    elif is_musa_available():
        torch.musa.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print_log(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm',
                logger='current',
                level=logging.WARNING)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
            torch.use_deterministic_algorithms(True)
    return seed

[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/974001ee-7c4c-479a-97fa-064c9a759cdd/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b57_record.py
file               /usr/local/lib/python3.11/site-packages/mmengine/runner/utils.py
bytes              3918
sha256 hex         2649223cf04b2d4b48ee179bdc3ec93d67f6f35e2a787569339ece9b2fe3a5a3
recomputed RECORD  sha256=JkkiPPBLLUtI7heb3D7JPWf2814qeHVpM57Omy_jpaM
RECORD files       ['/usr/local/lib/python3.11/site-packages/mmengine-0.10.7.dist-info/RECORD']
RECORD line        mmengine/runner/utils.py,sha256=JkkiPPBLLUtI7heb3D7JPWf2814qeHVpM57Omy_jpaM,3918
digest equal       True
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local python -c "from mmengine.runner import set_random_seed as s; import mmengine.runner.utils as u; print(s is u.set_random_seed); print(repr(s.__doc__))"
True
'Set random seed.\n\n    Args:\n        seed (int, optional): Seed to be used.\n        deterministic (bool): Whether to set the deterministic option for\n            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`\n            to True and `torch.backends.cudnn.benchmark` to False.\n            Defaults to False.\n        diff_rank_seed (bool): Whether to add rank number to the random seed to\n            have different random seed in different threads. Defaults to False.\n    '
[exit 0]
````

**H-E3.** mmengine `0.10.7`, `/usr/local/lib/python3.11/site-packages/mmengine/runner/utils.py`: 3,918
bytes, sha256 `2649223cf04b2d4b48ee179bdc3ec93d67f6f35e2a787569339ece9b2fe3a5a3`, and a RECORD digest
equal to the 0.10.7 wheel's (`digest equal True`) — the file the wheel shipped, unmodified.

With `deterministic=True`, `set_random_seed` sets `cudnn.deterministic=True` and `cudnn.benchmark=False`
and calls `torch.use_deterministic_algorithms(True)` with no `warn_only`. It never mentions
`CUBLAS_WORKSPACE_CONFIG`, and its `PYTHONHASHSEED` line is commented out. The public
`mmengine.runner.set_random_seed` is the same object (`True`), and its docstring names only the two cuDNN
flags. *[UPDATED 2026-09-15]* The examined public docstring does not mention the
`use_deterministic_algorithms` call. No broader search of MMEngine documentation was performed, so no
framework-wide documentation-absence claim is made. The framework-wide absence of the cuBLAS variable is
B56 §3.2.

---

## E-4 — the teacher's determinism keys, read as text

````text
$ ls -la configs/teacher_finetune*.py configs/teacher/
-rw-r--r-- 1 admin 197121 1095 Jun 27 13:59 configs/teacher_finetune.py

configs/teacher/:
total 24
drwxr-xr-x 1 admin 197121     0 Aug 15 14:48 .
drwxr-xr-x 1 admin 197121     0 Sep  2 20:50 ..
-rw-r--r-- 1 admin 197121 20170 Aug 15 14:48 segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
[exit 0]

$ grep -n -A3 -E "randomness|deterministic|cudnn_benchmark" configs/teacher_finetune*.py
[exit 1]

$ cat configs/teacher_finetune.py
# Teacher fine-tune config — Blocker B1
# Source of truth: docs/IMPLEMENTATION_CONTRACT.md  section (d) "B1 - Teacher fine-tune"
# Every value traced to ch3.pdf (method authority); versions corroborated by context.md.
# Analysis/config artifact only — contains NO training logic.

TEACHER_FINETUNE = {
    # Base / initialization
    "model": "SegNeXt-B / MSCAN-B",
    "init_checkpoint": "segnext_mscan-b_512x512_160k_ade20k",  # ADE20K-pretrained, MMSeg zoo (or equivalent)
    "framework": "MMSegmentation 1.2.2 + mmcv 2.1.0",

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 6e-5,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "decode_head_lr_mult": 10,

    # Schedule / budget
    "lr_schedule": "poly",
    "iterations": 40000,
    "batch_size": 16,
    "crop": (512, 512),

    # Loss
    "loss": "cross_entropy",

    # Success criterion (protocol match, not max accuracy)
    "success_criterion": "recover 42.05% mIoU within +/-1.5-2.0 pp",

    # Role
    "role": "descriptive upper-bound reference only (not deployed, not an inferential comparator)",
}
[exit 0]

$ grep -n -A3 -E "randomness|deterministic|cudnn_benchmark" configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
352:randomness = dict(seed=42, deterministic=True)
353-
354:# NMF / Hamburger randomness: MMSeg's LightHamHead exposes NO seed key of its own. It is governed by
355-# the global torch RNG seeded above, inside mmseg/models/decode_heads/ham_head.py :: NMF2D._build_bases.
356-# A dedicated control remains NEED_TO_CONFIRM, to be settled against the pinned stack; it is
357-# deliberately NOT faked as a config key MMSeg would silently ignore.
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'cd /usr/local/lib/python3.11/site-packages/mmseg/.mim/configs && sed -n "1,4p" segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py && grep -n -E "_base_|randomness|deterministic|cudnn_benchmark|env_cfg" segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py _base_/default_runtime.py _base_/schedules/schedule_160k.py _base_/datasets/ade20k.py'
_base_ = './segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py'

# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth'  # noqa
segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py:1:_base_ = './segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py'
segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py:1:_base_ = [
segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py:2:    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py:3:    '../_base_/datasets/ade20k.py'
_base_/default_runtime.py:2:env_cfg = dict(
_base_/default_runtime.py:3:    cudnn_benchmark=True,
[exit 0]
````

**Coverage — the command as given read the wrong file.** `configs/teacher_finetune*.py` matches exactly one
file, `configs/teacher_finetune.py`, whose header reads "Analysis/config artifact only — contains NO
training logic". Its `exit 1` is an absence from a file that never held these keys. The operative,
unloadable config is `configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py`, which the
fourth command covers.

*Instrument note.* `sed -n "1,4p"` over two files prints the first four lines of their concatenation, so
only the SegNeXt-B head appears. The SegNeXt-T `_base_` list is shown by the grep lines that follow.

**H-E4.** The operative config sets `randomness = dict(seed=42, deterministic=True)` at `:352`. Neither
`randomness` nor `deterministic` appears anywhere in its `_base_` chain (SegNeXt-B → SegNeXt-T →
`default_runtime.py`, `schedule_160k.py`, `ade20k.py`), so the delta's value stands after merging.
`env_cfg.cudnn_benchmark=True` enters from `_base_/default_runtime.py:3`.

*[UPDATED 2026-09-15 — B58 reconciliation]* That inherited value is a **requested** setting, which
`Runner.setup_env` applies before randomness handling. It is not itself a methodology violation. The final
runtime value is set afterwards: `False` under MMEngine's strict path ([B56 §3.3–3.4](b56_teacher_survey.md)),
`False` in the G18 scratch prototype on CPU ([B58 §6](b58_teacher_runner_adjudication.md)), and `False` by
design under the selected G18 architecture (B58 §9.1), which is not yet implemented. The config text and
MMEngine's environment log keep the requested `True`.

**`randomness.deterministic` is True, so MMEngine's strict path fires on the teacher.** ch3 §D fails
there through strict mode and the missing cuBLAS variable, not because determinism is off. B56 §3.3 read
the merged values from a scratch reordered copy; this item stays text-only, as scoped.

---

## E-5 — the loss path at pinned mmseg 1.2.2

The second command as given abbreviates the mmseg directory as `$(...)`; it is expanded here to the same
`python -c` expression the first command uses.

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/974001ee-7c4c-479a-97fa-064c9a759cdd/scratchpad/probe,target=/probe,readonly plantseg-teacher:local bash -c 'python -c "import mmseg; print(mmseg.__version__, mmseg.__file__)" && python -B /probe/b57_pin.py "mmsegmentation-*.dist-info" mmseg/models/losses/cross_entropy_loss.py mmseg/models/decode_heads/decode_head.py mmseg/models/decode_heads/ham_head.py mmseg/models/utils/wrappers.py'
1.2.2 /usr/local/lib/python3.11/site-packages/mmseg/__init__.py
RECORD files ['/usr/local/lib/python3.11/site-packages/mmsegmentation-1.2.2.dist-info/RECORD']
mmseg/models/losses/cross_entropy_loss.py
    sha256 hex         2d2d024b5ef3a433f784a97cc282ec4b485549fbe4ade93701ac96a2a365ac7a
    RECORD line        mmseg/models/losses/cross_entropy_loss.py,sha256=LS0CS17zpDP3hKl8woLsS0hVSfvkrek3AayWoqNlrHo,12510
    recomputed digest  sha256=LS0CS17zpDP3hKl8woLsS0hVSfvkrek3AayWoqNlrHo
    digest equal       True
mmseg/models/decode_heads/decode_head.py
    sha256 hex         cd7a2fa6d9963f2dcb51fa5c5f21433d9d7aafa56c39769f4b83ebaa88a5890e
    RECORD line        mmseg/models/decode_heads/decode_head.py,sha256=zXovptmWPy3LUfpcXyFDPZ16r6VsOXafS4PrqoiliQ4,14571
    recomputed digest  sha256=zXovptmWPy3LUfpcXyFDPZ16r6VsOXafS4PrqoiliQ4
    digest equal       True
mmseg/models/decode_heads/ham_head.py
    sha256 hex         eb2f0963769926e303a832874d01c7d854596fe9839f400bcaf3c596110790b3
    RECORD line        mmseg/models/decode_heads/ham_head.py,sha256=6y8JY3aZJuMDqDKHTQHH2FRZb-mDn0ALyvPFlhEHkLM,8298
    recomputed digest  sha256=6y8JY3aZJuMDqDKHTQHH2FRZb-mDn0ALyvPFlhEHkLM
    digest equal       True
mmseg/models/utils/wrappers.py
    sha256 hex         5b1f2676f506dde9b483b1aba43cb5e3b441887642ed955dfea9b75123c3fd0c
    RECORD line        mmseg/models/utils/wrappers.py,sha256=Wx8mdvUG3em0g7GrpDy147RBiHZC7ZVd_qm3USPD_Qw,1861
    recomputed digest  sha256=Wx8mdvUG3em0g7GrpDy147RBiHZC7ZVd_qm3USPD_Qw
    digest equal       True
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'grep -n -B2 -A8 "F.cross_entropy" $(python -c "import mmseg,os;print(os.path.dirname(mmseg.__file__))")/models/losses/cross_entropy_loss.py'
18-                  ignore_index=-100,
19-                  avg_non_ignore=False):
20:    """cross_entropy. The wrapper function for :func:`F.cross_entropy`
21-
22-    Args:
23-        pred (torch.Tensor): The prediction with shape (N, 1).
24-        label (torch.Tensor): The learning label of the prediction.
25-        weight (torch.Tensor, optional): Sample-wise loss weight.
26-            Default: None.
27-        class_weight (list[float], optional): The weight for each class.
28-            Default: None.
--
43-    # class_weight is a manual rescaling weight given to each class.
44-    # If given, has to be a Tensor of size C element-wise losses
45:    loss = F.cross_entropy(
46-        pred,
47-        label,
48-        weight=class_weight,
49-        reduction='none',
50-        ignore_index=ignore_index)
51-
52-    # apply weights and do the reduction
53-    # average loss over non-ignored elements
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'grep -n -A6 "def loss_by_feat" $(python -c "import mmseg,os;print(os.path.dirname(mmseg.__file__))")/models/decode_heads/decode_head.py'
291:    def loss_by_feat(self, seg_logits: Tensor,
292-                     batch_data_samples: SampleList) -> dict:
293-        """Compute segmentation loss.
294-
295-        Args:
296-            seg_logits (Tensor): The output from decode head forward function.
297-            batch_data_samples (List[:obj:`SegDataSample`]): The seg
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'F=$(python -c "import mmseg,os;print(os.path.dirname(mmseg.__file__))")/models/decode_heads/decode_head.py; grep -n "" "$F" | sed -n "/^[0-9]*:    def loss(/,/^[0-9]*:    def predict(/p"; grep -n "" "$F" | sed -n "/^[0-9]*:    def loss_by_feat(/,/^[0-9]*:    def predict_by_feat(/p"'
247:    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
248:             train_cfg: ConfigType) -> dict:
249:        """Forward function for training.
250:
251:        Args:
252:            inputs (Tuple[Tensor]): List of multi-level img features.
253:            batch_data_samples (list[:obj:`SegDataSample`]): The seg
254:                data samples. It usually includes information such
255:                as `img_metas` or `gt_semantic_seg`.
256:            train_cfg (dict): The training config.
257:
258:        Returns:
259:            dict[str, Tensor]: a dictionary of loss components
260:        """
261:        seg_logits = self.forward(inputs)
262:        losses = self.loss_by_feat(seg_logits, batch_data_samples)
263:        return losses
264:
265:    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
291:    def loss_by_feat(self, seg_logits: Tensor,
292:                     batch_data_samples: SampleList) -> dict:
293:        """Compute segmentation loss.
294:
295:        Args:
296:            seg_logits (Tensor): The output from decode head forward function.
297:            batch_data_samples (List[:obj:`SegDataSample`]): The seg
298:                data samples. It usually includes information such
299:                as `metainfo` and `gt_sem_seg`.
300:
301:        Returns:
302:            dict[str, Tensor]: a dictionary of loss components
303:        """
304:
305:        seg_label = self._stack_batch_gt(batch_data_samples)
306:        loss = dict()
307:        seg_logits = resize(
308:            input=seg_logits,
309:            size=seg_label.shape[2:],
310:            mode='bilinear',
311:            align_corners=self.align_corners)
312:        if self.sampler is not None:
313:            seg_weight = self.sampler.sample(seg_logits, seg_label)
314:        else:
315:            seg_weight = None
316:        seg_label = seg_label.squeeze(1)
317:
318:        if not isinstance(self.loss_decode, nn.ModuleList):
319:            losses_decode = [self.loss_decode]
320:        else:
321:            losses_decode = self.loss_decode
322:        for loss_decode in losses_decode:
323:            if loss_decode.loss_name not in loss:
324:                loss[loss_decode.loss_name] = loss_decode(
325:                    seg_logits,
326:                    seg_label,
327:                    weight=seg_weight,
328:                    ignore_index=self.ignore_index)
329:            else:
330:                loss[loss_decode.loss_name] += loss_decode(
331:                    seg_logits,
332:                    seg_label,
333:                    weight=seg_weight,
334:                    ignore_index=self.ignore_index)
335:
336:        loss['acc_seg'] = accuracy(
337:            seg_logits, seg_label, ignore_index=self.ignore_index)
338:        return loss
339:
340:    def predict_by_feat(self, seg_logits: Tensor,
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'grep -n -E "^class |^    def " $(python -c "import mmseg,os;print(os.path.dirname(mmseg.__file__))")/models/decode_heads/ham_head.py'
15:class Matrix_Decomposition_2D_Base(nn.Module):
37:    def __init__(self,
56:    def _build_bases(self, B, S, D, R, device=None):
59:    def local_step(self, x, bases, coef):
62:    def local_inference(self, x, bases):
73:    def compute_coef(self, x, bases, coef):
76:    def forward(self, x, return_bases=False):
108:class NMF2D(Matrix_Decomposition_2D_Base):
114:    def __init__(self, args=dict()):
119:    def _build_bases(self, B, S, D, R, device=None):
128:    def local_step(self, x, bases, coef):
146:    def compute_coef(self, x, bases, coef):
158:class Hamburger(nn.Module):
168:    def __init__(self,
183:    def forward(self, x):
194:class LightHamHead(BaseDecodeHead):
212:    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
234:    def forward(self, inputs):
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'F=$(python -c "import mmseg,os;print(os.path.dirname(mmseg.__file__))")/models/utils/wrappers.py; grep -n "" "$F" | sed -n "/^[0-9]*:def resize(/,/^[0-9]*:class /p"'
8:def resize(input,
9:           size=None,
10:           scale_factor=None,
11:           mode='nearest',
12:           align_corners=None,
13:           warning=True):
14:    if warning:
15:        if size is not None and align_corners:
16:            input_h, input_w = tuple(int(x) for x in input.shape[2:])
17:            output_h, output_w = tuple(int(x) for x in size)
18:            if output_h > input_h or output_w > output_h:
19:                if ((output_h > 1 and output_w > 1 and input_h > 1
20:                     and input_w > 1) and (output_h - 1) % (input_h - 1)
21:                        and (output_w - 1) % (input_w - 1)):
22:                    warnings.warn(
23:                        f'When align_corners={align_corners}, '
24:                        'the output would more aligned if '
25:                        f'input size {(input_h, input_w)} is `x+1` and '
26:                        f'out size {(output_h, output_w)} is `nx+1`')
27:    return F.interpolate(input, size, scale_factor, mode, align_corners)
28:
29:
30:class Upsample(nn.Module):
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/974001ee-7c4c-479a-97fa-064c9a759cdd/scratchpad/probe,target=/probe,readonly plantseg-teacher:local bash -c 'python -B /probe/b57_doc.py; grep -n "" /usr/local/lib/python3.11/site-packages/torch/nn/functional.py | sed -n "4007,4018p"'
torch 2.1.0+cu121
doc:  47| 
doc:  48|     The following normally-nondeterministic operations will throw a
doc:  49|     :class:`RuntimeError` when ``mode=True``:
doc:  50| 
doc:  51|         * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
doc:  52|         * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
doc:  53|         * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
doc:  54|         * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
doc:  55|         * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
doc:  56|         * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
doc:  57|         * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
doc:  58|         * :class:`torch.nn.MaxUnpool1d`
doc:  59|         * :class:`torch.nn.MaxUnpool2d`
doc:  60|         * :class:`torch.nn.MaxUnpool3d`
doc:  61|         * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
doc:  62|           and one of the following modes is used:
doc:  63| 
doc:  64|           - ``linear``
doc:  65|           - ``bilinear``
doc:  66|           - ``bicubic``
doc:  67|           - ``trilinear``
doc:  68| 
doc:  69|         * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
doc:  70|         * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
doc:  71|         * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
doc:  72|         * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
doc:  73|         * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
doc:  74|         * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
doc:  75|         * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
doc:  76|         * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
doc:  77|         * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
doc:  78|           ``mode='max'``
doc:  79|         * :func:`torch.Tensor.put_` when ``accumulate=False``
doc:  80|         * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
doc:  81|         * :func:`torch.histc` when called on a CUDA tensor
doc:  82|         * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
4007:    if input.dim() == 4 and mode == "bilinear":
4008:        assert align_corners is not None
4009:        if antialias:
4010:            return torch._C._nn._upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors)
4011:        # Two levels are necessary to prevent TorchScript from touching
4012:        # are_deterministic_algorithms_enabled.
4013:        if not torch.jit.is_scripting():
4014:            if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
4015:                # Use slow decomp whose backward will be in terms of index_put
4016:                # importlib is required because the import cannot be top level
4017:                # (cycle) and cannot be nested (TS doesn't support)
4018:                return importlib.import_module('torch._decomp.decompositions').upsample_bilinear2d_vec(
[exit 0]
````

**Coverage — `-A6` cannot show the resize.** The given `grep -n -A6 "def loss_by_feat"` ends at `:297`,
inside the docstring. The covering command prints `loss` and `loss_by_feat` in full, each through the next
method's `def`, which shows the window reached the end of the function.

**H-E5.**

- **(a) Confirmed at the pinned version.** `cross_entropy()` calls `F.cross_entropy(pred, label,
  weight=class_weight, reduction='none', ignore_index=ignore_index)` (`cross_entropy_loss.py:45-50`). All
  four files read are byte-identical to the 1.2.2 wheel's RECORD.
- **(b) Confirmed.** `BaseDecodeHead.loss` runs `forward`, then `loss_by_feat` (`decode_head.py:261-262`).
  `loss_by_feat` resizes the logits to label size with `mode='bilinear'` (`:307-311`) before it calls
  `loss_decode` (`:324-328`). `LightHamHead` defines only `__init__` and `forward`
  (`ham_head.py:212`, `:234`), so it inherits this path, and `resize` returns `F.interpolate(…)`
  (`wrappers.py:27`).
- **Implication — inferred, confirmable only on CUDA.** torch 2.1.0 documents bilinear `interpolate` as
  raising when differentiated on CUDA under `mode=True` (`doc:61-65`). However, `F.interpolate` reroutes
  4-D bilinear CUDA input to `upsample_bilinear2d_vec` whenever determinism is **enabled**
  (`functional.py:4014-4018`). That condition holds in both modes, so both modes run the same kernels.
  E1 ran the reroute for 80,000 iterations without an interpolate alert (E-1); the reroute's memory
  transient is D30's measurement. **The loss-path resize is probably not a strict-mode raise site.**

---

## E-6 — flag-state ordering, and whether a one-shot override holds

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/plantseg-thesis/src,target=/workspace/plantseg-thesis/src,readonly --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/974001ee-7c4c-479a-97fa-064c9a759cdd/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b57_flags.py
[0] process start
    are_deterministic_algorithms_enabled()          = False
    is_deterministic_algorithms_warn_only_enabled() = False
    cudnn.deterministic = False | cudnn.benchmark = False | CUBLAS_WORKSPACE_CONFIG = None
    CPU put_(accumulate=False) -> returned
[1] after src.seeds.set_seed(42)
    are_deterministic_algorithms_enabled()          = True
    is_deterministic_algorithms_warn_only_enabled() = True
    cudnn.deterministic = True | cudnn.benchmark = False | CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    CPU put_(accumulate=False) -> returned
    warning captured: UserWarning - put_ does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
[2] after mmengine.runner.set_random_seed(42, deterministic=True)
    are_deterministic_algorithms_enabled()          = True
    is_deterministic_algorithms_warn_only_enabled() = False
    cudnn.deterministic = True | cudnn.benchmark = False | CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    CPU put_(accumulate=False) -> RuntimeError: put_ does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
[3] after torch.use_deterministic_algorithms(True, warn_only=True)
    are_deterministic_algorithms_enabled()          = True
    is_deterministic_algorithms_warn_only_enabled() = True
    cudnn.deterministic = True | cudnn.benchmark = False | CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    CPU put_(accumulate=False) -> returned
    warning captured: UserWarning - put_ does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:71.)
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'cd /usr/local/lib/python3.11/site-packages && grep -rn --include=*.py -E "set_random_seed\(|set_randomness\(|use_deterministic_algorithms\(" mmengine mmseg mmcv'
mmengine/_strategy/deepspeed.py:498:            self._set_randomness(**self._randomness)
mmengine/_strategy/base.py:246:        self._set_randomness(**randomness)
mmengine/_strategy/base.py:252:    def _set_randomness(
mmengine/_strategy/base.py:273:        self._seed = set_random_seed(
mmengine/_strategy/single_device.py:213:            self._set_randomness(**self._randomness)
mmengine/_strategy/colossalai.py:392:            self._set_randomness(**self._randomness)
mmengine/runner/utils.py:48:def set_random_seed(seed: Optional[int] = None,
mmengine/runner/utils.py:90:            torch.use_deterministic_algorithms(True)
mmengine/runner/runner.py:376:        self.set_randomness(**randomness)
mmengine/runner/runner.py:698:    def set_randomness(self,
mmengine/runner/runner.py:716:        self._seed = set_random_seed(
mmengine/runner/runner.py:2060:            self.set_randomness(**self._randomness_cfg)
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/974001ee-7c4c-479a-97fa-064c9a759cdd/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b57_enclosing.py mmengine/runner/runner.py:376 mmengine/runner/runner.py:2060 mmengine/_strategy/base.py:246 mmengine/_strategy/single_device.py:213 mmengine/_strategy/deepspeed.py:498 mmengine/_strategy/colossalai.py:392
=== mmengine/runner/runner.py:376  enclosed by: class Runner (:77) > def __init__ (:262)
  262|     def __init__(
  ...
  365|             self._distributed = False
  366|         else:
  367|             self._distributed = True
  368| 
  369|         # self._timestamp will be set in the `setup_env` method. Besides,
  370|         # it also will initialize multi-process and (or) distributed
  371|         # environment.
  372|         self.setup_env(env_cfg)
  373|         # self._deterministic and self._seed will be set in the
  374|         # `set_randomness`` method
  375|         self._randomness_cfg = randomness
  376|         self.set_randomness(**randomness)
=== mmengine/runner/runner.py:2060  enclosed by: class Runner (:77) > def resume (:1997)
 1997|     def resume(self,
  ...
 2049| 
 2050|         # resume random seed
 2051|         resumed_seed = checkpoint['meta'].get('seed', None)
 2052|         current_seed = self._randomness_cfg.get('seed')
 2053|         if resumed_seed is not None and resumed_seed != current_seed:
 2054|             if current_seed is not None:
 2055|                 self.logger.warning(f'The value of random seed in the '
 2056|                                     f'checkpoint "{resumed_seed}" is '
 2057|                                     f'different from the value in '
 2058|                                     f'`randomness` config "{current_seed}"')
 2059|             self._randomness_cfg.update(seed=resumed_seed)
 2060|             self.set_randomness(**self._randomness_cfg)
=== mmengine/_strategy/base.py:246  enclosed by: class BaseStrategy (:31) > def _setup_env (:168)
  168|     def _setup_env(
  ...
  235|         # set resource limit
  236|         if platform.system() != 'Windows':
  237|             import resource
  238|             rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  239|             base_soft_limit = rlimit[0]
  240|             hard_limit = rlimit[1]
  241|             soft_limit = min(max(resource_limit, base_soft_limit), hard_limit)
  242|             resource.setrlimit(resource.RLIMIT_NOFILE,
  243|                                (soft_limit, hard_limit))
  244| 
  245|         self._randomness = randomness
  246|         self._set_randomness(**randomness)
=== mmengine/_strategy/single_device.py:213  enclosed by: class SingleDeviceStrategy (:17) > def resume (:158)
  158|     def resume(
  ...
  202| 
  203|         # resume random seed
  204|         resumed_seed = checkpoint['meta'].get('seed', None)
  205|         current_seed = self._randomness.get('seed')
  206|         if resumed_seed is not None and resumed_seed != current_seed:
  207|             if current_seed is not None:
  208|                 self.logger.warning(f'The value of random seed in the '
  209|                                     f'checkpoint "{resumed_seed}" is '
  210|                                     f'different from the value in '
  211|                                     f'`randomness` config "{current_seed}"')
  212|             self._randomness.update(seed=resumed_seed)
  213|             self._set_randomness(**self._randomness)
=== mmengine/_strategy/deepspeed.py:498  enclosed by: class DeepSpeedStrategy (:219) > def resume (:445)
  445|     def resume(
  ...
  487| 
  488|         # resume random seed
  489|         resumed_seed = extra_ckpt['meta'].get('seed', None)
  490|         current_seed = self._randomness.get('seed')
  491|         if resumed_seed is not None and resumed_seed != current_seed:
  492|             if current_seed is not None:
  493|                 self.logger.warning(f'The value of random seed in the '
  494|                                     f'checkpoint "{resumed_seed}" is '
  495|                                     f'different from the value in '
  496|                                     f'`randomness` config "{current_seed}"')
  497|             self._randomness.update(seed=resumed_seed)
  498|             self._set_randomness(**self._randomness)
=== mmengine/_strategy/colossalai.py:392  enclosed by: class ColossalAIStrategy (:191) > def resume (:355)
  355|     def resume(
  ...
  381| 
  382|         # resume random seed
  383|         resumed_seed = extra_ckpt['meta'].get('seed', None)
  384|         current_seed = self._randomness.get('seed')
  385|         if resumed_seed is not None and resumed_seed != current_seed:
  386|             if current_seed is not None:
  387|                 self.logger.warning(f'The value of random seed in the '
  388|                                     f'checkpoint "{resumed_seed}" is '
  389|                                     f'different from the value in '
  390|                                     f'`randomness` config "{current_seed}"')
  391|             self._randomness.update(seed=resumed_seed)
  392|             self._set_randomness(**self._randomness)
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'grep -n "" /usr/local/lib/python3.11/site-packages/mmengine/runner/runner.py | sed -n "1679,1701p"'
1679:    def load_or_resume(self) -> None:
1680:        """Load or resume checkpoint."""
1681:        if self._has_loaded:
1682:            return None
1683:
1684:        # decide to load from checkpoint or resume from checkpoint
1685:        resume_from = None
1686:        if self._resume and self._load_from is None:
1687:            # auto resume from the latest checkpoint
1688:            resume_from = find_latest_checkpoint(self.work_dir)
1689:            self.logger.info(
1690:                f'Auto resumed from the latest checkpoint {resume_from}.')
1691:        elif self._resume and self._load_from is not None:
1692:            # resume from the specified checkpoint
1693:            resume_from = self._load_from
1694:
1695:        if resume_from is not None:
1696:            self.resume(resume_from)
1697:            self._has_loaded = True
1698:        elif self._load_from is not None:
1699:            self.load_checkpoint(self._load_from)
1700:            self._has_loaded = True
1701:
[exit 0]

$ grep -n -E "^(resume|load_from) =" configs/teacher/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py
364:load_from = os.environ.get(SEGNEXT_ADE20K_CKPT_ENV, ADE20K_CKPT_UNSET_SENTINEL)
365:resume = False
[exit 0]
````

**H-E6 — the overwrite and its reversal, shown as behaviour on CPU, not only as flag values.**

| point | `enabled` | `warn_only` | CPU `put_(accumulate=False)` | `CUBLAS_WORKSPACE_CONFIG` |
|---|---|---|---|---|
| [0] process start | False | False | returns | `None` |
| [1] after `src.seeds.set_seed(42)` | True | True | **warns** | `':4096:8'` |
| [2] after mmengine `set_random_seed(42, deterministic=True)` | True | **False** | **raises `RuntimeError`** | `':4096:8'` |
| [3] after `use_deterministic_algorithms(True, warn_only=True)` | True | True | **warns** | `':4096:8'` |

`Tensor.put_` with `accumulate=False` is on torch 2.1.0's "will throw a RuntimeError when mode=True" list
with no device qualifier (`doc:79`, E-5 transcript), and its alert text belongs to E-1's class. The same op
in the same process moves from warning to raise at [2], and back at [3]. `cudnn.deterministic=True` and
`cudnn.benchmark=False` hold from [1] to [3], and `set_random_seed` does not unset the cuBLAS variable.

**Would a one-shot override hold — inferred from source.** In mmengine, mmseg and mmcv, `use_deterministic_algorithms(`
appears only in `set_random_seed` (`utils.py:90`), and every route back to that call goes through one of
these callers:

- `Runner.__init__`, once (`runner.py:376`, after `setup_env` at `:372`);
- `Runner.resume`, and only when a checkpoint's seed differs from the config's (`:2053-2060`);
- the FlexibleRunner strategies — `BaseStrategy._setup_env` (`_strategy/base.py:246`), and the
  seed-mismatch branches of the `resume` methods in `single_device.py`, `deepspeed.py` and
  `colossalai.py`.

`Runner.load_or_resume` calls `resume` only when `self._resume` is true (`:1686-1696`); otherwise it calls
`load_checkpoint` (`:1699`). The teacher config sets `resume = False` (`:365`) and supplies its weights
through `load_from` (`:364`).

**Consequence.** On a classic Runner with `resume=False`, nothing in mmengine, mmseg or mmcv would undo an
override placed after `Runner` construction and before `runner.train()`. It is **not resume-safe**: a
`resume` with a mismatched seed re-enters MMEngine's strict path and undoes it.

> *[UPDATED 2026-09-15 — B58 reconciliation]* This one-shot override is **not** the selected fix. It lands
> after the Runner's environment logging can initialize CUDA (`runner.py:403`), which contract B6's
> unqualified "before CUDA init" does not allow, and it is not resume-safe. G18 instead selects a minimal
> `TeacherRunner` that overrides only `set_randomness()`, establishes the registered policy there before
> CUDA initialization, and is re-entered by a differing-seed resume
> ([B58 §9](b58_teacher_runner_adjudication.md)).

**Not shown here — CUDA only:**

- that `CUBLAS_WORKSPACE_CONFIG` set in-process reaches cuBLAS, which reads it when the first handle is
  created;
- that the override changes what the teacher's real CUDA graph does ([B56 §4.3](b56_teacher_survey.md)).

---

## Protected-file note — a breach of AGENTS.md rule 2, recorded

While this report was being verified, `git diff --stat` was run **without a pathspec**. To compute the
diff stat for the working tree, git read `docs/reference/reference.pdf` and printed only
`docs/reference/reference.pdf | Bin 129010 -> 163179 bytes`. The file was not opened directly, and it was
not modified, staged or copied; no content was printed, and its status is unchanged (` M`).

Rule 2 permits recording the file's status, size and mtime, but it forbids reading the file — and a tool
invocation read it. Recorded here rather than left out. From that point on, every git command in this pass
names its paths.

Two commands in the pass were not path-scoped: the rule-1 `git status`, and
`git diff --cached --name-only`, which compares the index with HEAD and reads no working-tree files. Every
other git and grep command named paths that do not contain the file.

---

## What this pass bears on — nothing was edited

- **B56 §4.2.** H-E5 supports keeping the loss-path resize off the candidate raise-site list. H-E1 narrows
  what E1's log shows about the CE site: it is evidence for the *reduced* call only.
- **G15 and G18.** H-E6 fixes the ordering any launch-level fix must respect, and names the one Runner path
  that would undo it: `Runner.resume` with a mismatched seed.
- **`src/seeds.py:23`.** H-E2's `PYTHONHASHSEED` finding is recorded, not queued. Acting on it would be a
  governed `src/**` edit, and ch3 §D makes no hash-seed claim.

The previous item's B56 and its B52 register edit are still uncommitted, and this report is added
alongside them. **Stopped after E-6:** no GPU reproduction, no config edit, no reorder plan.

---

## Appendix — probe scripts called by the transcripts

Each ran unmodified from a read-only mount at `/probe`.

### `b57_record.py` — used by E-3

sha256 `b92051b32a4653ef69e117a5f8424670bb7474b3bbd9e5ecf2ac6cb43ea9b78f` · 29 lines · 1169 bytes

````python
#!/usr/bin/env python3
"""B57 E-3 pin -- is the installed mmengine/runner/utils.py the file the 0.10.7 wheel shipped?

Prints the file's sha256 (hex), the wheel RECORD line for it, and the RECORD-format digest
(urlsafe base64 of the sha256, unpadded) recomputed from the installed bytes. Raw values only.
"""
import base64
import glob
import hashlib
import os
import sys

sys.dont_write_bytecode = True

SP = "/usr/local/lib/python3.11/site-packages"
REL = "mmengine/runner/utils.py"
data = open(os.path.join(SP, REL), "rb").read()
print("file              ", os.path.join(SP, REL))
print("bytes             ", len(data))
print("sha256 hex        ", hashlib.sha256(data).hexdigest())
recomputed = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
print("recomputed RECORD ", recomputed)
records = glob.glob(os.path.join(SP, "mmengine-*.dist-info", "RECORD"))
print("RECORD files      ", records)
for rec in records:
    for line in open(rec, encoding="utf-8"):
        if line.startswith(REL + ","):
            print("RECORD line       ", line.rstrip())
            print("digest equal      ", line.split(",")[1] == recomputed)
````

### `b57_pin.py` — used by E-5

sha256 `98562047bae604ed661033d1b89a3b8e9da4a4eb276b149c592d26092f8bc6ab` · 31 lines · 1188 bytes

````python
#!/usr/bin/env python3
"""B57 pin -- are the installed files the ones their wheel shipped?

usage: b57_pin.py <dist-info glob> <relative path>...
For each path: the installed package version, the file's sha256 (hex), its RECORD line, the RECORD-format
digest recomputed from the installed bytes, and whether the two digests are equal. Raw values only.
"""
import base64
import glob
import hashlib
import os
import sys

sys.dont_write_bytecode = True

SP = "/usr/local/lib/python3.11/site-packages"
records = glob.glob(os.path.join(SP, sys.argv[1], "RECORD"))
print("RECORD files", records)
lines = {}
for rec in records:
    for line in open(rec, encoding="utf-8"):
        lines[line.split(",")[0]] = line.rstrip()
for rel in sys.argv[2:]:
    data = open(os.path.join(SP, rel), "rb").read()
    recomputed = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rec = lines.get(rel)
    print(rel)
    print("    sha256 hex        ", hashlib.sha256(data).hexdigest())
    print("    RECORD line       ", rec)
    print("    recomputed digest ", recomputed)
    print("    digest equal      ", rec is not None and rec.split(",")[1] == recomputed)
````

### `b57_doc.py` — used by E-5

sha256 `0c9aabc54144dc0cc8e850e2f8718fd6697f865c530268ba96d2638001bb5356` · 14 lines · 458 bytes

````python
#!/usr/bin/env python3
"""B57 -- torch 2.1.0's installed docstring for use_deterministic_algorithms, lines 47-82 verbatim:
the start of the "will throw a RuntimeError when mode=True" list, through the interpolate, put_ and
histc entries."""
import sys

sys.dont_write_bytecode = True

import torch

doc = torch.use_deterministic_algorithms.__doc__.splitlines()
print("torch", torch.__version__)
for i in range(47, 83):
    print(f"doc:{i:4d}| {doc[i - 1]}")
````

### `b57_flags.py` — used by E-6

sha256 `8d9c1321ac79aa9b93d478d0bcb9f7026ea6b9e9148d3d5e4a3c1f1cb0a96c13` · 47 lines · 2002 bytes

````python
#!/usr/bin/env python3
"""B57 E-6 -- CPU flag-state ordering probe. No CUDA. Raw values only.

At each point it prints the two determinism flags, the other two ch3 D measures, and what one CPU
operation does. torch 2.1.0's installed docstring lists `Tensor.put_` with `accumulate=False` among the
operations that "will throw a RuntimeError when mode=True", with no device qualifier, so the call shows
the flags' consequence (raise vs warn), not only their values. The full exception or warning text is
printed, untruncated.
"""
import sys

sys.dont_write_bytecode = True

import os
import warnings

import torch
from mmengine.runner import set_random_seed
from src.seeds import set_seed


def point(tag):
    print(tag)
    print("    are_deterministic_algorithms_enabled()          =", torch.are_deterministic_algorithms_enabled())
    print("    is_deterministic_algorithms_warn_only_enabled() =", torch.is_deterministic_algorithms_warn_only_enabled())
    print("    cudnn.deterministic =", torch.backends.cudnn.deterministic,
          "| cudnn.benchmark =", torch.backends.cudnn.benchmark,
          "| CUBLAS_WORKSPACE_CONFIG =", repr(os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    t = torch.zeros(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            t.put_(torch.tensor([0, 0]), torch.tensor([1.0, 2.0]), accumulate=False)
            print("    CPU put_(accumulate=False) -> returned")
        except RuntimeError as e:
            print("    CPU put_(accumulate=False) -> RuntimeError:", e)
    for w in caught:
        print("    warning captured:", w.category.__name__, "-", w.message)


point("[0] process start")
set_seed(42)
point("[1] after src.seeds.set_seed(42)")
set_random_seed(42, deterministic=True)
point("[2] after mmengine.runner.set_random_seed(42, deterministic=True)")
torch.use_deterministic_algorithms(True, warn_only=True)
point("[3] after torch.use_deterministic_algorithms(True, warn_only=True)")
````

### `b57_enclosing.py` — used by E-6

sha256 `82458affc6ceb443a6b4ee18938c42fb7eefa48560a28e20beb87b95b0c53f11` · 41 lines · 1607 bytes

````python
#!/usr/bin/env python3
"""B57 E-6 support -- for each file:line: the enclosing class/def chain (AST), the def line itself, and
the 12 source lines ending at the target line. Raw source only; the window size is fixed by design."""
import ast
import sys

sys.dont_write_bytecode = True

SP = "/usr/local/lib/python3.11/site-packages/"
SCOPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def chain_for(tree, line):
    chain, node = [], tree
    while True:
        inner = [n for n in ast.walk(node) if isinstance(n, SCOPES) and n is not node
                 and n.lineno <= line <= n.end_lineno]
        inner = [n for n in inner if not any(m is not n and isinstance(m, SCOPES)
                                             and m.lineno <= n.lineno and n.end_lineno <= m.end_lineno
                                             and m in inner for m in inner)]
        if not inner:
            return chain
        node = inner[0]
        chain.append(node)


for spec in sys.argv[1:]:
    rel, line = spec.rsplit(":", 1)
    line = int(line)
    src = open(SP + rel, encoding="utf-8").read()
    lines = src.splitlines()
    chain = chain_for(ast.parse(src), line)
    print(f"=== {rel}:{line}  enclosed by: " + " > ".join(
        f"{'class' if isinstance(n, ast.ClassDef) else 'def'} {n.name} (:{n.lineno})" for n in chain))
    if chain:
        d = chain[-1].lineno
        print(f"{d:5d}| {lines[d - 1]}")
        if line - 12 > d + 1:
            print("  ...")
    for i in range(max(line - 11, (chain[-1].lineno + 1) if chain else 1), line + 1):
        print(f"{i:5d}| {lines[i - 1]}")
````
