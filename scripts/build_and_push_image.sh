#!/usr/bin/env bash
# Build and push the official PlantSeg experiment image (B38).
#
# WHY THIS EXISTS. The image already on GHCR was built and pushed by hand, and its
# org.opencontainers.image.revision label was applied on the docker CLI rather than from the
# Dockerfile -- the Dockerfile has never contained a LABEL. Nothing recorded the procedure, so the
# published image silently fell 19 commits behind HEAD while still looking authoritative. Every
# guard below exists to make that specific failure impossible to repeat.
#
# WHAT IT REFUSES TO DO
#   * build from a dirty governed path      -> the image would not correspond to any commit
#   * build from a commit not on origin     -> the digest would cite a hash nobody else can resolve
#   * take the commit as an argument        -> a typo'd label is a provenance lie; it is DERIVED
#   * handle credentials                    -> it fails with an instruction to run docker login
#
# The commit is ALWAYS `git rev-parse HEAD` of the tree being built. It is never an argument.
#
# Usage:
#   scripts/build_and_push_image.sh              # build, verify, push
#   scripts/build_and_push_image.sh --build-only # build + verify, no push
#   scripts/build_and_push_image.sh --dry-run    # run every gate, then stop before building
set -euo pipefail

REGISTRY="ghcr.io/ainsleydeluna/plantseg-thesis"
LOCAL_TAG="plantseg-thesis:official"

# EVALUATION_CONTRACT.md section 7.1. A blanket "repo must be clean" check is deliberately NOT used:
# this repository is intentionally never globally clean (docs/reference/reference.pdf stays dirty).
GOVERNED=(src configs scripts requirements.lock requirements-e1.txt requirements-runpod.lock
          requirements-runpod.in docs/EVALUATION_CONTRACT.md docs/IMPLEMENTATION_CONTRACT.md)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_ONLY=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --build-only) BUILD_ONLY=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "" >&2; echo "REFUSING: $*" >&2; exit 1; }

echo "=============================================================================="
echo "OFFICIAL IMAGE BUILD"
echo "=============================================================================="
echo "  repo     : $REPO_ROOT"
echo "  registry : $REGISTRY"

# ---- gate 1: the Docker engine is actually reachable -------------------------------------------
docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable. Start Docker Desktop (or the
          engine) and re-run. 'docker --version' succeeding only proves the CLI exists."

# ---- gate 2: governed paths are clean ----------------------------------------------------------
# Run bare and read the verdict off the actual output. A label printed unconditionally would report
# "clean" underneath a dirty result.
DIRTY="$(git status --porcelain -- "${GOVERNED[@]}")"
if [ -n "$DIRTY" ]; then
  echo "" >&2
  echo "$DIRTY" >&2
  die "governed paths are dirty (listed above). The image would not correspond to any commit.
          Commit or stash them first. EVALUATION_CONTRACT.md section 7.1 defines this set;
          docs/reference/reference.pdf is NOT governed and is expected to stay dirty."
fi

# ---- gate 3: the commit is DERIVED, and must exist on origin -----------------------------------
GIT_COMMIT="$(git rev-parse HEAD)"                 # full 40 chars, never an argument
SHORT="$(git rev-parse --short=7 HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git fetch --quiet origin || echo "  WARNING: could not fetch origin; the push check uses stale refs."
if ! git merge-base --is-ancestor "$GIT_COMMIT" origin/"$BRANCH" 2>/dev/null; then
  die "HEAD ($SHORT) is not on origin/$BRANCH. An image digest citing an unpushed commit
          names a tree nobody else can resolve. Push the branch first:
              git push origin $BRANCH"
fi

echo "  commit   : $GIT_COMMIT"
echo "  branch   : $BRANCH (present on origin)"
echo "  tags     : $REGISTRY:$GIT_COMMIT"
echo "             $REGISTRY:official"
echo "             $LOCAL_TAG"

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "RESULT: DRY-RUN — every gate passed. Nothing was built or pushed."
  exit 0
fi

# ---- build -------------------------------------------------------------------------------------
# The final Dockerfile layer runs preflight_environment.py --mode image, which is fail-closed, so a
# drifted stack fails the BUILD rather than producing a wrong image.
echo ""
echo "---- docker build ----"
docker build \
  --build-arg GIT_COMMIT="$GIT_COMMIT" \
  -t "$LOCAL_TAG" \
  -t "$REGISTRY:$GIT_COMMIT" \
  -t "$REGISTRY:official" \
  .

# ---- verify: the image self-identifies ---------------------------------------------------------
echo ""
echo "---- verifying the revision label ----"
LABEL="$(docker image inspect "$LOCAL_TAG" \
         --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
echo "  label in image : $LABEL"
echo "  expected       : $GIT_COMMIT"
[ "$LABEL" = "$GIT_COMMIT" ] || die "the image's revision label does not match the commit it was
          built from. Do not push it."

# ---- verify: the E1 gate reaches its class-weight stage ----------------------------------------
# Stage 1/5 asserts the CE class-weight artifact, which must be INSIDE the image: train_e1.py loads
# it through a fail-closed loader, so an image missing it cannot run E1 at all. The later stages
# need a GPU and a mounted dataset and are expected to NO-GO on a builder.
echo ""
echo "---- verifying the E1 gate's class-weight stage inside the image ----"
GATE="$(docker run --rm "$LOCAL_TAG" python -B scripts/preflight_e1.py 2>&1 || true)"
if ! grep -q "class_weights             PASS" <<<"$GATE"; then
  echo "$GATE" | head -30 >&2
  die "preflight_e1.py stage 1/5 (class_weights) did not PASS inside the image. E1 would abort at
          loss construction on the pod. Check that reports/e1_class_weights.json is COPYed in and
          that the .dockerignore re-include sits BELOW 'reports/**' (last-match-wins)."
fi
grep -E "class_weights|verify_env" <<<"$GATE" | sed 's/^/  /'

if [ "$BUILD_ONLY" = "1" ]; then
  echo ""
  echo "RESULT: BUILT AND VERIFIED — --build-only, nothing pushed."
  exit 0
fi

# ---- push --------------------------------------------------------------------------------------
# No credential is read, written, printed or stored by this script. If the daemon is not already
# authenticated, the push fails and the operator authenticates themselves.
echo ""
echo "---- docker push ----"
if ! docker push "$REGISTRY:$GIT_COMMIT"; then
  die "push failed. If this is an authentication error, authenticate yourself and re-run:
              docker login ghcr.io
          This script never handles credentials."
fi
docker push "$REGISTRY:official"

# ---- the digest is what Chapter 4 cites; tags move ---------------------------------------------
echo ""
echo "=============================================================================="
echo "PUSHED"
echo "=============================================================================="
DIGEST="$(docker image inspect "$REGISTRY:$GIT_COMMIT" --format '{{json .RepoDigests}}')"
echo "  commit       : $GIT_COMMIT"
echo "  tags         : $REGISTRY:$GIT_COMMIT"
echo "                 $REGISTRY:official"
echo "  RepoDigests  : $DIGEST"
echo ""
echo "  Record the DIGEST (not the tag) in the run provenance of every official artifact."
echo "  Tags move; a digest does not. Any earlier tag of this repository is superseded."
