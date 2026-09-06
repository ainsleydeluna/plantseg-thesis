# B41 — runtime provenance inside the official image

**Type:** code change + image rebuild. **Status:** COMPLETE — committed, pushed, image published.
**Date:** 2026-09-06. **Closes:** B40 finding #19.

**Image commit:** `f77d05d7b35187bf0da7e7b94a629549fe2e1c05`
**New authoritative digest:** `sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf`

---

## 1. What was wrong

B40 #19 established, by running the published image **by digest**, that:

- `.git` is excluded from the image (`.dockerignore:57`), so `_git_head()` returned `"UNKNOWN"`;
- `ARG GIT_COMMIT` is build-time only — it becomes a `LABEL`, and **a process inside a container
  cannot read its own labels**;
- no image-digest field existed anywhere in `src/`.

So every `run_meta` row E1 wrote on the pod would have carried `"git_head": "UNKNOWN"` and no image
identifier — unattributable to a commit or an image. That nullifies the B37 pre-registration anchor
and the B38 digest **for exactly the artifacts those exist to support**. Training would have run
correctly and produced 80,000 iterations of telemetry nobody could tie to anything.

---

## 2. The four changes

### 2.1 `Dockerfile` — bake the commit in, and correct a false rationale

```dockerfile
ARG GIT_COMMIT
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
ENV PLANTSEG_GIT_COMMIT=${GIT_COMMIT}
```

Declared last, so the `ARG` still does not invalidate the ~8.6 GB pip layer.

The comment at `Dockerfile:30-32` previously read:

> `git -> LOAD-BEARING, not convenience: src/eval/artifacts.py records the commit and the
> governed-path porcelain in every official artifact, so official evaluation cannot be finalised
> without it.`

That is **false as written** — the binary is installed, the `.git` directory is not, so installing git
does not make official evaluation possible in-image. It is corrected **in place**, saying explicitly
that the earlier claim was wrong, rather than deleted. A known-false rationale left standing is how
the original defect survived a build gate that passed **36/36**: the gate proved software identity,
and the comment asserted a capability nobody re-tested.

### 2.2 `_git_provenance()` — resolve the commit *and* say where it came from

The two sources are not equally trustworthy, so the artifact records which one it got:

| source | proves |
|---|---|
| `git_checkout` | what was built **and** the tree the code is running from |
| `image_env` | what was **BUILT** — cannot detect a tree modified after the build (in-image there is none to modify) |
| `unavailable` | neither; commit is `"UNKNOWN"` |

The env var is preferred because in the image the subprocess silently yields `"UNKNOWN"`. Where a
real checkout exists the env var is unset, so `git_checkout` still wins — verified below. Blank or
whitespace-only env values fall through rather than being emitted.

### 2.3 `_image_digest()` — from `PLANTSEG_IMAGE_DIGEST`, absent means `null`

A digest **cannot** be baked into the image it identifies: it is the hash of the config that would
have to contain it. It therefore arrives at `docker run -e PLANTSEG_IMAGE_DIGEST=sha256:…`. When
unset the field is JSON `null` — **absent, never `""` and never guessed**. A fabricated digest is
worse than a missing one: a missing one is visibly missing, a wrong one silently misattributes a run.

`run_meta` now carries `git_head`, `git_head_source`, `image_digest`, and a `[provenance]` line prints
at startup so the operator sees the resolution without reading the JSONL.

### 2.4 `src/eval/artifacts.py` — functionally UNCHANGED, deliberately

> ### The recorded rationale: training records provenance best-effort and degrades; official evaluation demands proof and refuses. The asymmetry is intended.

`git_commit` *could* have read `PLANTSEG_GIT_COMMIT`. Its companion `git_porcelain_bytes` **cannot**.
That call is the only evidence the governed paths were clean **at evaluation time**, it feeds
`worktree_state_sha256` / `governed_violations` / `governed_paths_clean`
(`artifacts.py:288-301`), and it gates official artifacts:

```python
if req.artifact_status == "official" and not summary["run"]["governed_paths_clean"]:
    raise ArtifactWriteError("official artifact requires governed_paths_clean=True")
```

No constant baked at build time can attest to a working tree at evaluation time. Making `git_commit`
env-aware while the porcelain still raised would produce a **half-verified artifact claiming more
than it can show**, failing later and less clearly than it does now.

**eval-on-pod:** official evaluation genuinely cannot run in the current image, and raising is
correct. The remedy is to give the evaluation stage a real checkout (mount or clone), not to weaken
the check. Training is unaffected — it never touches this path.

**eval-on-laptop:** works today, unchanged. `.git` is present; both functions succeed.

Both error messages were rewritten (text only, no logic) to state that the raise is by design, name
the remedy, and say plainly that weakening the check is not the fix — because a future reader hitting
`git rev-parse HEAD failed` in a container will otherwise reach for the gate.

---

## 3. An incident worth recording: the CRLF that a clean diff would have hidden

The first edit to `src/training/train_e1.py` silently converted the **whole file** from LF to CRLF —
648 CRLF line endings against HEAD's 609 LF.

**`.gitattributes` (`* text=auto eol=lf`) would have normalized the committed blob**, so
`git diff` showed only the 47 lines actually changed and the commit would have looked clean. Review
would have passed. The working copy, however, would have been left wrong — contradicting the
repository's own declared `eol=lf` — and the next tool to read bytes rather than go through git's
normalization would have seen a file nothing in the diff explained.

Detected by comparing raw byte counts against `git show HEAD:<path>` rather than trusting the diff:

```
src/training/train_e1.py     working CRLF=648    LF-only=0      | HEAD CRLF=0      LF-only=609
```

Normalized back to LF, recompiled, and every verification in §5 re-run afterwards.

**The finding, not the apology:** a normalizing `.gitattributes` makes an entire class of working-tree
defect invisible in review, because the artifact under review is the normalized blob and not the file
on disk. `git diff --stat` is not sufficient evidence that a file is unchanged in the ways that
matter. This is the same shape as B40 #19 — a check that passes while the thing it is supposed to
protect is broken — and the same shape as the unconditional-echo defect from B34c: **verification
that inspects a derived view rather than the actual state.**

---

## 4. Diff

```diff
--- a/Dockerfile
+++ b/Dockerfile
-#   git                    -> LOAD-BEARING, not convenience: src/eval/artifacts.py records the commit
-#                             and the governed-path porcelain in every official artifact, so official
-#                             evaluation cannot be finalised without it.
+#   git                    -> present for tooling that shells out to it. NOTE (B40/B41): this image
+#                             does NOT carry a .git directory (.dockerignore excludes it), so the git
+#                             BINARY alone does not make official evaluation possible here:
+#                             src/eval/artifacts.py needs the porcelain of a real working tree to
+#                             prove governed_paths_clean, and it correctly RAISES without one. An
+#                             earlier version of this comment claimed git made official evaluation
+#                             finalisable in-image; that was false and is corrected here. Official
+#                             evaluation requires a real checkout (mount or clone); TRAINING is
+#                             unaffected and records its commit from PLANTSEG_GIT_COMMIT below.
@@
 ARG GIT_COMMIT
 LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
+# The LABEL is readable only from OUTSIDE (docker inspect). A process INSIDE the container cannot
+# read its own labels, and .git is excluded, so without this ENV the training loop has no way to name
+# the commit it is running and every run_meta row records "UNKNOWN" (B40 #19). The image digest
+# cannot be baked in at all -- it is the hash of the config that would have to contain it -- so it is
+# supplied at `docker run` via PLANTSEG_IMAGE_DIGEST and recorded as absent when unset.
+ENV PLANTSEG_GIT_COMMIT=${GIT_COMMIT}
```

```diff
--- a/src/training/train_e1.py
+++ b/src/training/train_e1.py
-def _git_head() -> str:
+def _git_provenance() -> tuple[str, str]:
+    """Resolve the source commit AND say where it came from. Returns (commit, source).
+      "git_checkout" -- live `git rev-parse HEAD`. Proves what was built AND the tree it runs from.
+      "image_env"    -- PLANTSEG_GIT_COMMIT, baked at build time. Proves what was BUILT only.
+      "unavailable"  -- neither; commit is "UNKNOWN".
+    """
+    env = os.environ.get("PLANTSEG_GIT_COMMIT", "").strip()
+    if env:
+        return env, "image_env"
     try:
         r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True, timeout=15)
-        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
+        if r.returncode == 0 and r.stdout.strip():
+            return r.stdout.strip(), "git_checkout"
     except Exception:  # noqa: BLE001
-        return "UNKNOWN"
+        pass
+    return "UNKNOWN", "unavailable"
+
+
+def _image_digest() -> str | None:
+    """The running image's registry digest, or None when it was not supplied. Cannot be baked in."""
+    v = os.environ.get("PLANTSEG_IMAGE_DIGEST", "").strip()
+    return v or None
@@
+    git_head, git_head_source = _git_provenance()
+    image_digest = _image_digest()
+    print(f"[provenance] git_head={git_head} (source={git_head_source}) "
+          f"image_digest={image_digest if image_digest else 'ABSENT (PLANTSEG_IMAGE_DIGEST unset)'}")
     _jsonl(jsonl_path, {
         "event": "run_meta", "wall_clock": time.time(), "mode": mode, "seed": seed,
-        "git_head": _git_head(), "torch": torch.__version__, "numpy": np.__version__,
+        "git_head": git_head, "git_head_source": git_head_source, "image_digest": image_digest,
+        "torch": torch.__version__, "numpy": np.__version__,
```

`src/eval/artifacts.py`: two raise messages extended with the by-design explanation, the remedy, and
an explicit "do not weaken this check". **No control flow changed.**

```
 Dockerfile               | 18 +++++++++++++++---
 src/eval/artifacts.py    | 27 +++++++++++++++++++++++++--
 src/training/train_e1.py | 47 +++++++++++++++++++++++++++++++++++++++++++----
 3 files changed, 83 insertions(+), 9 deletions(-)
```

---

## 5. Verification

### 5.1 Against the PUSHED image, addressed by digest

`ghcr.io/ainsleydeluna/plantseg-thesis@sha256:b80b645d…866aaf`

**A — `PLANTSEG_IMAGE_DIGEST` unset:**

```
{"git_head": "f77d05d7b35187bf0da7e7b94a629549fe2e1c05", "git_head_source": "image_env", "image_digest": null}
```

**B — digest supplied at `docker run`:**

```
{"git_head": "f77d05d7b35187bf0da7e7b94a629549fe2e1c05", "git_head_source": "image_env", "image_digest": "sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf"}
```

`"UNKNOWN"` is gone. Absence is `null`, not `""`.

**Label round-trip, read from the registry** (`buildx imagetools`, not locally):

```
  revision label: f77d05d7b35187bf0da7e7b94a629549fe2e1c05
```

equal to `git rev-parse HEAD` → `f77d05d7b35187bf0da7e7b94a629549fe2e1c05`.

### 5.2 On a real checkout — `git_checkout` still wins

```
--- local checkout, no env vars ---
   ('c059e1b0cb6945d5f985d2f19132e33b422fed28', 'git_checkout') None
--- env vars set (env must win) ---
   ('deadbeefdeadbeefdeadbeefdeadbeefdeadbeef', 'image_env') sha256:test
--- env vars blank/whitespace (must fall through, never emit empty) ---
   ('c059e1b0cb6945d5f985d2f19132e33b422fed28', 'git_checkout') None
```

End-to-end dry run:

```
[provenance] git_head=c059e1b0cb6945d5f985d2f19132e33b422fed28 (source=git_checkout) image_digest=ABSENT (PLANTSEG_IMAGE_DIGEST unset)
RESULT: PASS (6/6 checks exercised, 0 skipped)
```

### 5.3 `artifacts.py` still refuses, now with guidance

```
git_commit
git rev-parse HEAD failed.
BY DESIGN: inside the official container image there is no .git directory. This function deliberately
does NOT fall back to the PLANTSEG_GIT_COMMIT environment variable that the training loop uses:
training records provenance best-effort and degrades, whereas an OFFICIAL artifact must prove it...
REMEDY: give the evaluation stage a real checkout (mount the repository into the container, or clone
it there)...
DO NOT weaken or bypass this check to make the error go away.
```

### 5.4 The build gates ran

The push went through `scripts/build_and_push_image.sh`, not a direct `docker push`:

```
  commit   : f77d05d7b35187bf0da7e7b94a629549fe2e1c05
  branch   : master (present on origin)
  label in image : f77d05d7b35187bf0da7e7b94a629549fe2e1c05
  expected       : f77d05d7b35187bf0da7e7b94a629549fe2e1c05
    class_weights             PASS          6.3  116 weights verified; ratio 259.551929
```

---

## 6. The new authoritative image, and the superseded list

| field | value |
|---|---|
| **registry digest (cite this)** | `sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf` |
| tags | `:f77d05d7b35187bf0da7e7b94a629549fe2e1c05` · `:official` |
| revision label | `f77d05d7b35187bf0da7e7b94a629549fe2e1c05` |

`docs/runpod_environment.md` now carries this digest, the `docker run -e PLANTSEG_IMAGE_DIGEST=…`
invocation, and a **growable superseded table** — restructured from the previous one-off note because
there are now two entries and there will be more. Each row carries digest, tags, source commit, and
reason:

| digest | tags | built from | superseded because |
|---|---|---|---|
| `sha256:5f5dba46b866…` | `:148af2c`, `:runpod-preflight-2026-08-17` | `148af2c9b30f…` | predates `scripts/preflight_e1.py` entirely |
| `sha256:0572c116…cedbe6` | `:28d1038a4b36…`, formerly `:official` | `28d1038a4b36…` | no runtime provenance; `run_meta` unattributable |

Both remain **on the registry and pullable, deliberately** — a digest cited in an old note must
resolve to something marked superseded rather than to nothing. The doc says the list grows by
appending and never by deleting.

---

## 7. What this does NOT do

- **It does not make official evaluation possible in the image.** `artifacts.py` still raises, by
  design (§2.4). That is a separate decision requiring a real checkout in the evaluation environment.
- **It does not make the digest automatic.** `PLANTSEG_IMAGE_DIGEST` must be passed at `docker run`;
  if the operator forgets, `image_digest` is `null` and the run is image-unattributable. The runbook
  now shows the flag, but nothing enforces it — a gate on that would belong in the launch checklist,
  not in the training loop.
- **It changes no training behaviour.** No metric, loss, seed, schedule or data path was touched; the
  only new work per run is two `os.environ.get` calls and one extra print at startup.
- **It was not verified on a GPU.** Everything above ran on CPU, in-container and by digest.
