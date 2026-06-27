# Dataset Location Log — PlantSeg (manual download)

_Generated: 2026-06-27. Read-only registration; no dataset file was moved, renamed, deleted, or modified._
_Updated: 2026-06-27 — the 9 missing `.gitignore` safety patterns were added and re-verified (see below)._

## Manual download status
- **Status:** Manually downloaded by the user. **Nothing was downloaded by this task.**

## Dataset path
- **Path:** `C:\Users\admin\plantseg_data\plantseg`
- **Exists:** ✅ **Yes** (`Test-Path` → `True`)

## Top-level structure (depth 1 only — no recursive scan)
| Name | Type |
|---|---|
| `annotations/` | DIR |
| `images/` | DIR |
| `annotation_train.json` | FILE |
| `annotation_val.json` | FILE |
| `annotation_test.json` | FILE |
| `Metadata.csv` | FILE |

This matches the official PlantSeg layout described in Wei et al. (2026) (images / annotations folders,
three split JSONs, `Metadata.csv`).

## Raw data — outside or inside the repo?
- **Repo work-tree root:** `C:\Users\admin\plantseg-thesis` (`git rev-parse --show-toplevel`).
- **Dataset root:** `C:\Users\admin\plantseg_data\plantseg`.
- **Verdict:** ✅ **OUTSIDE the repository.** The dataset is in a sibling top-level folder
  (`plantseg_data\`), not under the repo. Git for this repo **cannot track it** regardless of `.gitignore`.

## Git safety status
- Raw dataset is **not staged and not committed** (it is outside the work-tree; see `git status`).
- `.gitignore` patterns are a secondary defense against *accidentally copying raw data into the repo*.
- Verified with `git check-ignore -v --non-matching` against representative in-repo paths
  (**re-run after the patterns were added — all 12 now match**):

| Required pattern | In `.gitignore`? | Git actually ignores it? | Matching rule |
|---|---|---|---|
| `data/` | ✅ | ✅ | `.gitignore:5:data/` |
| `datasets/` | ✅ (added) | ✅ | `.gitignore:15:datasets/` |
| `plantseg_data/` | ✅ (added) | ✅ | `.gitignore:16:plantseg_data/` |
| `*.zip` | ✅ (added) | ✅ | `.gitignore:19:*.zip` |
| `*.rar` | ✅ (added) | ✅ | `.gitignore:20:*.rar` |
| `*.7z` | ✅ (added) | ✅ | `.gitignore:21:*.7z` |
| `*.tar` | ✅ (added) | ✅ | `.gitignore:22:*.tar` |
| `*.tar.gz` | ✅ (added) | ✅ | `.gitignore:23:*.tar.gz` |
| `*.pth` | ✅ | ✅ | `.gitignore:8:*.pth` |
| `*.pt` | ✅ | ✅ | `.gitignore:9:*.pt` |
| `*.ckpt` | ✅ (added) | ✅ | `.gitignore:26:*.ckpt` |
| `*.onnx` | ✅ (added) | ✅ | `.gitignore:27:*.onnx` |

## Is `.gitignore` sufficient?
- ✅ **Yes (now).** All 12 requested safety patterns are present and confirmed effective by
  `git check-ignore`. The actual raw dataset is additionally safe because it lives outside the repo.

## Missing safety rules — `NEED_TO_CONFIRM`
- **None.** The 9 patterns previously missing (`datasets/`, `plantseg_data/`, `*.zip`, `*.rar`, `*.7z`,
  `*.tar`, `*.tar.gz`, `*.ckpt`, `*.onnx`) were added to `.gitignore` and re-verified.

## PASS summary
| PASS condition | Result |
|---|---|
| Dataset path exists | ✅ Yes |
| Raw data is not being committed | ✅ Yes (outside the repo; not staged/tracked) |
| `.gitignore` protects raw/large files | ✅ Yes (all 12 patterns present and verified) |

**Overall:** ✅ **PASS.** Raw dataset exists, is external to the repo, and is not committed; `.gitignore`
now protects all requested raw/large/model-artifact file types.
