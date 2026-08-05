#!/usr/bin/env python3
"""Vendor-safety smoke for scripts/vendor_plantseg_class_map.py (A2b-0 hardening).

STANDARD LIBRARY ONLY. FULLY OFFLINE. Network access is actively blocked for every subprocess
via a generated `sitecustomize.py`, so a request would raise rather than silently succeed.

This is an AUXILIARY suite. It does not replace scripts/smoke_plantseg_class_map.py, which
remains the frozen 21-check semantic/provenance gate for the artifact itself.

Every write performed here happens inside one temporary simulation directory under the system
temp location. The real configs/plantseg_class_map.json is READ ONLY -- it is used to build the
upstream fixture and as the comparison target, and is never written, moved, or replaced.

Run:  set PYTHONIOENCODING=utf-8 && python -B scripts/smoke_vendor_plantseg_class_map.py
Exit: 0 only if all six checks pass.

No dataset, checkpoint, training, CUDA/GPU, or project-repository mutation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_SRC = REPO / "scripts" / "vendor_plantseg_class_map.py"
REAL_MAP = REPO / "configs" / "plantseg_class_map.json"

NUM_CLASSES = 116
SEMANTIC_SHA = "d14182423b6f176f940cada979adb364701fe186be091655ce38a80ce791a729"
FIXTURE_SHA_PLACEHOLDER = "0" * 40

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_entries_sha(entries) -> str:
    return sha256_bytes(json.dumps(list(entries), sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8"))


def snapshot(root: Path) -> dict:
    """path -> sha256 for every regular file under `root` (used to prove no side effects)."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root)).replace("\\", "/")] = sha256_bytes(p.read_bytes())
    return out


def build_sim(tmp: Path) -> Path:
    """A repository-shaped simulation holding only a copy of the vendor tool."""
    sim = tmp / "sim_repo"
    (sim / "scripts").mkdir(parents=True)
    (sim / "configs").mkdir(parents=True)
    shutil.copyfile(VENDOR_SRC, sim / "scripts" / "vendor_plantseg_class_map.py")
    return sim


def write_netblock(tmp: Path) -> Path:
    """sitecustomize.py that denies every socket operation in child processes."""
    d = tmp / "netblock"
    d.mkdir()
    # Patch the resolution/connection entry points only. Replacing `socket.socket` itself would
    # break interpreter startup, since it is a class other stdlib modules subclass.
    (d / "sitecustomize.py").write_text(
        "import socket\n"
        "class NetworkBlocked(RuntimeError):\n"
        "    pass\n"
        "def _deny(*a, **k):\n"
        "    raise NetworkBlocked('network access blocked by smoke_vendor_plantseg_class_map')\n"
        "socket.getaddrinfo = _deny\n"
        "socket.create_connection = _deny\n"
        "socket.gethostbyname = _deny\n",
        encoding="utf-8", newline="\n")
    return d


def child_env(netblock: Path, extra: str | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PLANTSEG_DATA_ROOT", None)
    env["PYTHONPATH"] = str(netblock) + (os.pathsep + extra if extra else "")
    return env


def run(args, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", *args], cwd=str(cwd),
                          capture_output=True, env=env, text=True, encoding="utf-8",
                          errors="replace")


def make_fixture(tmp: Path, names) -> Path:
    """A synthetic upstream plantseg115.py carrying exactly the committed ordered names.

    Includes a `palette` subscript so the extractor's "palette is ignored, never evaluated"
    property is exercised.
    """
    body = ",\n            ".join(repr(n) for n in names)
    src = ("# synthetic upstream fixture -- structure mirrors the official module\n"
           "class PlantSeg115Dataset:\n"
           "    METAINFO = dict(\n"
           "        classes=(\n"
           f"            {body},\n"
           "        ),\n"
           "        palette=[[0, 0, 0]] * 200,\n"
           "    )\n")
    f = tmp / "plantseg115_fixture.py"
    f.write_text(src, encoding="utf-8", newline="\n")
    return f


def main() -> int:
    print("=" * 96)
    print("A2b-0 hardening -- vendor-tool safety smoke")
    print(f"python {sys.version.split()[0]} | stdlib only | OFFLINE, network actively blocked")
    print("=" * 96)

    real_doc = json.loads(REAL_MAP.read_text(encoding="utf-8"))
    names = [e["name"] for e in real_doc["entries"]]
    real_map_bytes = REAL_MAP.read_bytes()

    tmp = Path(tempfile.mkdtemp(prefix="vendor_safety_"))
    try:
        sim = build_sim(tmp)
        netblock = write_netblock(tmp)
        env = child_env(netblock)
        vend = sim / "scripts" / "vendor_plantseg_class_map.py"
        dest = sim / "configs" / "plantseg_class_map.json"
        fixture = make_fixture(tmp, names)
        outside = tmp / "ESCAPED_TARGET.json"

        # ---------- 1. import is side-effect-free ----------
        before = snapshot(sim)
        drv = tmp / "drv_import.py"
        drv.write_text(
            "import sys, pathlib\n"
            f"sys.path.insert(0, r'{sim / 'scripts'}')\n"
            "import vendor_plantseg_class_map as V\n"
            "print('IMPORTED', bool(V.SCHEMA))\n"
            "print('NETMOD', [m for m in ('urllib.request',) if m in sys.modules])\n",
            encoding="utf-8", newline="\n")
        r = run([str(drv)], cwd=sim, env=env)
        after = snapshot(sim)
        check("1 import is side-effect-free",
              r.returncode == 0 and "IMPORTED True" in r.stdout and before == after
              and not dest.exists(),
              f"rc={r.returncode} tree_unchanged={before == after} dest_created={dest.exists()}")

        # ---------- 2. no-mode invocation refuses safely ----------
        r = run([str(vend)], cwd=sim, env=env)
        check("2 no-mode invocation refuses safely",
              r.returncode != 0 and not dest.exists() and "NEVER happens without" in r.stderr,
              f"rc={r.returncode} dest_created={dest.exists()}")

        # ---------- 3. arbitrary output destination is refused ----------
        r = run([str(vend), "--from-file", str(fixture), "--commit-sha", FIXTURE_SHA_PLACEHOLDER,
                 "--out", str(outside)], cwd=sim, env=env)
        check("3 arbitrary output destination is refused",
              r.returncode != 0 and not outside.exists()
              and ("unrecognized arguments" in r.stderr or "--out" in r.stderr),
              f"rc={r.returncode} outside_created={outside.exists()} "
              f"stderr={r.stderr.strip().splitlines()[-1][:70] if r.stderr.strip() else ''}")

        # ---------- 4. symlink or path-escape target is refused ----------
        esc = tmp / "escape_target_dir"
        esc.mkdir()
        cfg = sim / "configs"
        mechanism = None
        shutil.rmtree(cfg)
        try:
            os.symlink(esc, cfg, target_is_directory=True)
            mechanism = "directory symlink"
        except OSError:
            rc = subprocess.run(["cmd", "/c", "mklink", "/J", str(cfg), str(esc)],
                                capture_output=True)
            if rc.returncode == 0 and cfg.exists():
                mechanism = "NTFS directory junction"
            else:
                cfg.write_text("not a directory\n", encoding="utf-8", newline="\n")
                mechanism = "configs-is-not-a-directory probe (symlink/junction unavailable)"
        r = run([str(vend), "--from-file", str(fixture), "--commit-sha", FIXTURE_SHA_PLACEHOLDER],
                cwd=sim, env=env)
        escaped = (esc / "plantseg_class_map.json").exists()
        check("4 symlink or path-escape target is refused",
              r.returncode != 0 and "refusing to write" in (r.stdout + r.stderr) and not escaped,
              f"mechanism={mechanism} rc={r.returncode} escaped_write={escaped}")
        # restore a clean configs/ for the remaining checks
        try:
            if cfg.is_symlink():
                os.rmdir(cfg)                      # dir symlinks and junctions
            elif cfg.is_dir():
                shutil.rmtree(cfg, ignore_errors=True)
            elif cfg.exists():
                cfg.unlink()
        except OSError:
            shutil.rmtree(cfg, ignore_errors=True)
        cfg.mkdir(parents=True, exist_ok=True)

        # ---------- 5. offline canonical generation is deterministic ----------
        r = run([str(vend), "--from-file", str(fixture), "--commit-sha", FIXTURE_SHA_PLACEHOLDER,
                 "--retrieved-utc", "2026-01-01T00:00:00Z"], cwd=sim, env=env)
        gen_ok = False
        detail = f"rc={r.returncode}"
        if r.returncode == 0 and dest.exists():
            gen = json.loads(dest.read_text(encoding="utf-8"))
            gen_ids = [e["class_id"] for e in gen["entries"]]
            gen_sem = canonical_entries_sha(gen["entries"])
            gen_ok = (gen["schema"] == real_doc["schema"]
                      and gen["class_space"] == real_doc["class_space"]
                      and len(gen["entries"]) == NUM_CLASSES
                      and gen_ids == list(range(NUM_CLASSES))
                      and gen["entries"] == real_doc["entries"]
                      and gen["entries"][0]["name"] == ""
                      and gen["entries"][0]["role"] == "background_or_non_disease"
                      and gen_sem == SEMANTIC_SHA)
            differing = sorted(k for k in gen["source"]
                               if gen["source"][k] != real_doc["source"].get(k))
            detail = (f"semantic={gen_sem[:16]}... entries_match={gen['entries'] == real_doc['entries']} "
                      f"provenance_fields_differing={differing}")
        check("5 offline canonical generation is deterministic", gen_ok, detail)

        # ---------- 6. pre-replace failure preserves existing artifact ----------
        sentinel = b'{"sentinel": "pre-existing artifact -- must not change"}\n'
        dest.write_bytes(sentinel)
        drv6 = tmp / "drv_fault.py"
        drv6.write_text(
            "import sys, pathlib\n"
            f"sys.path.insert(0, r'{sim / 'scripts'}')\n"
            "import vendor_plantseg_class_map as V\n"
            "orig = pathlib.Path.replace\n"
            "def boom(self, target):\n"
            "    raise OSError('injected pre-replace failure')\n"
            "pathlib.Path.replace = boom\n"
            "try:\n"
            f"    V.main(['--from-file', r'{fixture}', '--commit-sha', '{FIXTURE_SHA_PLACEHOLDER}'])\n"
            "    print('NO_EXCEPTION')\n"
            "except Exception as e:\n"
            "    print('RAISED', type(e).__name__)\n",
            encoding="utf-8", newline="\n")
        r = run([str(drv6)], cwd=sim, env=env)
        leftovers = sorted(p.name for p in (sim / "configs").glob(".plantseg_class_map.json.tmp-*"))
        check("6 pre-replace failure preserves existing artifact",
              "RAISED" in r.stdout and dest.read_bytes() == sentinel and not leftovers,
              f"artifact_unchanged={dest.read_bytes() == sentinel} leftover_temp={leftovers} "
              f"{r.stdout.strip().splitlines()[-1][:40] if r.stdout.strip() else ''}")

        # every write stayed inside the temporary simulation
        real_unchanged = REAL_MAP.read_bytes() == real_map_bytes
        print(f"\nreal configs/plantseg_class_map.json unchanged: {real_unchanged}")
        print(f"all simulation writes confined to: {tmp}")
        if not real_unchanged:
            check("real artifact untouched", False, "REAL ARTIFACT CHANGED")
        return finish()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def finish() -> int:
    print()
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail[:150]}")
    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    print("\n" + "=" * 96)
    print(f"SUMMARY  {n_ok}/{len(CHECKS)} checks passed")
    print(f"RESULT: {'VENDOR TOOL SAFE' if n_ok == len(CHECKS) else 'VENDOR SAFETY FAILURE'}")
    print("=" * 96)
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
