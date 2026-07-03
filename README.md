# plantseg-thesis (THESIS2)

Resource-constrained plant-lesion segmentation — knowledge distillation + INT8 quantization. Current
focus: the **E1 FP32 MobileNetV3-Large + LR-ASPP** student baseline.

## Start here
- **Claude Code / AI workflow:** read [CLAUDE.md](CLAUDE.md) first, then [docs/ai_guardrails.md](docs/ai_guardrails.md) and [docs/task_templates/](docs/task_templates/).
- **E1 RunPod launch:** [reports/e1_runpod_launch_runbook.md](reports/e1_runpod_launch_runbook.md) (pre-flight checklist: [docs/task_templates/runpod_preflight_template.md](docs/task_templates/runpod_preflight_template.md)).
- **Teacher prep (separate A6000 workflow, not needed for E1):** [docs/teacher_prep_runbook.md](docs/teacher_prep_runbook.md).
- **Methodology / locked configs:** [docs/IMPLEMENTATION_CONTRACT.md](docs/IMPLEMENTATION_CONTRACT.md).

## Environments
- **E1 (student only):** `requirements-e1.txt` — torch / torchvision / numpy / pillow; no mmcv/mmseg/teacher stack.
- **Full pinned stack (teacher / MMSeg):** `requirements.lock`.
