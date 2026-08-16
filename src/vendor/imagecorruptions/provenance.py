"""Machine-readable provenance for the vendored imagecorruptions closure.

CHECKSUM DESIGN — non-self-referential by construction. `VENDOR_RUNTIME_SHA256` is the SHA-256 of
`corruptions.py` ONLY: the runtime numerics live in a different file from the recorded hash, so the
value never covers bytes that contain itself. LICENSE, this provenance module, generated cache
manifests and `__pycache__` are all excluded from that digest.

Two identities are recorded separately and must not be conflated:
  * `UPSTREAM_CORRUPTIONS_SHA256` — the PRISTINE upstream file we extracted from;
  * `VENDOR_RUNTIME_SHA256`       — our LOCAL, patched runtime bytes.
The vendored bytes are NOT pristine upstream bytes; `COMPATIBILITY_PATCHES` enumerates every
difference.

THE LICENSE CARRIES THE SAME TWO-IDENTITY SPLIT, for a line-ending reason rather than a code one:
  * `UPSTREAM_LICENSE_SHA256`  — the raw bytes as RETRIEVED from the pinned upstream commit, which
    arrived CRLF-terminated (11,558 bytes). Provenance evidence only. It is deliberately NOT the
    digest of the committed file, and must never be relabelled as such.
  * `VENDORED_LICENSE_SHA256`  — the LF-normalised text actually committed (11,357 bytes).
`.gitattributes` declares `* text=auto eol=lf`, so Git stores this file LF regardless of the
working copy. Verifying a fresh checkout against the CRLF retrieval digest would therefore fail on
every clone, so `verify_vendored_license()` — the integrity check that matters after checkout —
verifies the VENDORED digest. `LICENSE_NORMALIZATION` records the transformation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

UPSTREAM_PROJECT = "imagecorruptions"
UPSTREAM_VERSION = "1.1.2"
UPSTREAM_TAG = "v1.1.2"
UPSTREAM_COMMIT = "d03ee68843a9be8fda4c94a9ad8aad767a3f437e"
UPSTREAM_SDIST_SHA256 = "044e173f24d5934899bdbf3596bfbec917e8083e507eed583ab217abebbe084d"
UPSTREAM_CORRUPTIONS_SHA256 = "adb5944eccfafe0118e777e3300c94420eab486556ca805ea101d3374d130cbb"
UPSTREAM_LICENSE = "Apache-2.0"
# Raw retrieval bytes from the pinned commit (CRLF, 11,558 bytes). Evidence of WHAT WAS FETCHED --
# never the digest of the committed file. See the module docstring.
UPSTREAM_LICENSE_SHA256 = "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6"

# The LF-normalised license text actually committed and checked out (11,357 bytes, 201 lines).
VENDORED_LICENSE_FILE = "LICENSE"
VENDORED_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
LICENSE_NORMALIZATION = "CRLF upstream retrieval -> LF vendored text"

VENDOR_SCOPE = "selected function closure only (not the full upstream package)"
SELECTED_CORRUPTIONS = ("motion_blur", "gaussian_noise", "jpeg_compression", "brightness", "fog")
VENDORED_FUNCTIONS = (
    "gauss_function", "getOptimalKernelWidth1D", "getMotionBlurKernel", "shift", "_motion_blur",
    "plasma_fractal", "next_power_of_2",
    "gaussian_noise", "jpeg_compression", "brightness", "fog", "motion_blur",
)

COMPATIBILITY_PATCHES = (
    {"function": "plasma_fractal", "change": "np.float_ -> np.float64",
     "reason": "NumPy 2.x removed the np.float_ alias; it names the same float64 dtype that the "
               "pinned NumPy 1.26.4 resolves it to, so the patch is output-preserving.",
     "verified_output_preserving_by": "scripts/smoke_corruption_vendor.py reference equivalence"},
)

# Expected digest of the runtime file. Recomputed and checked by `verify_vendor_integrity()`.
VENDOR_RUNTIME_FILE = "corruptions.py"
VENDOR_RUNTIME_SHA256 = "caad653555d6d4098c5b6c11d8e821f3c3a86eaf56cbeaca1603fc7d9237e90e"

# The per-item seed derivation below is a THESIS implementation choice, not an upstream feature.
SEED_POLICY_ID = "plantseg-corruption-v1"


class VendorIntegrityError(RuntimeError):
    """The vendored runtime bytes do not match the recorded provenance."""


def runtime_path() -> Path:
    return Path(__file__).resolve().parent / VENDOR_RUNTIME_FILE


def compute_runtime_sha256() -> str:
    """SHA-256 over the runtime implementation bytes ONLY (never over this file)."""
    return hashlib.sha256(runtime_path().read_bytes()).hexdigest()


def license_path() -> Path:
    return Path(__file__).resolve().parent / VENDORED_LICENSE_FILE


def compute_license_sha256() -> str:
    """SHA-256 over the committed license bytes as they exist on disk."""
    return hashlib.sha256(license_path().read_bytes()).hexdigest()


def verify_vendored_license() -> str:
    """Checkout-time license integrity. Verifies the VENDORED (LF) digest, not the CRLF retrieval.

    Kept out of `verify_vendor_integrity()` on purpose: that runs on every corruption call, and
    re-hashing 11 KB of licence text per cached item would be pure waste.
    """
    actual = compute_license_sha256()
    if actual != VENDORED_LICENSE_SHA256:
        raise VendorIntegrityError(
            f"vendored {VENDORED_LICENSE_FILE} sha256 {actual[:16]}… does not match the recorded "
            f"{VENDORED_LICENSE_SHA256[:16]}… — the Apache-2.0 licence text changed")
    return actual


def verify_vendor_integrity() -> str:
    actual = compute_runtime_sha256()
    if actual != VENDOR_RUNTIME_SHA256:
        raise VendorIntegrityError(
            f"vendored {VENDOR_RUNTIME_FILE} sha256 {actual[:16]}… does not match the recorded "
            f"{VENDOR_RUNTIME_SHA256[:16]}… — the vendored numerics changed")
    return actual


def as_dict() -> dict:
    return {
        "upstream_project": UPSTREAM_PROJECT,
        "upstream_version": UPSTREAM_VERSION,
        "upstream_tag": UPSTREAM_TAG,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_sdist_sha256": UPSTREAM_SDIST_SHA256,
        "pristine_upstream_corruptions_sha256": UPSTREAM_CORRUPTIONS_SHA256,
        "upstream_license": UPSTREAM_LICENSE,
        "upstream_license_sha256_as_retrieved": UPSTREAM_LICENSE_SHA256,
        "vendored_license_sha256": compute_license_sha256(),
        "license_normalization": LICENSE_NORMALIZATION,
        "vendor_scope": VENDOR_SCOPE,
        "selected_corruptions": list(SELECTED_CORRUPTIONS),
        "vendored_functions": list(VENDORED_FUNCTIONS),
        "compatibility_patches": [dict(p) for p in COMPATIBILITY_PATCHES],
        "patched_vendor_runtime_sha256": compute_runtime_sha256(),
        "seed_policy_id": SEED_POLICY_ID,
        "bytes_are_pristine_upstream": False,
    }
