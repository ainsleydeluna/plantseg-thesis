#!/usr/bin/env python3
"""Smokes for the mobile web demo (synthetic images only; no dataset, no network; CPU).

    python3 -I demo/smoke_demo.py [--model PATH]

S1 the server starts (demo/run_demo.py in a subprocess) and /healthz names the model file's sha256
S2 the class list served is the repository's: ids, names and roles equal configs/plantseg_class_map.json
   (read raw here) and src.eval.adapters.load_class_map; 116 = configs/data.py num_classes = model output
S3 a synthetic image returns a mask of the right shape (the photo's box on the canvas) and class range
   (0..115), and logits [1, 116, 512, 512]
S4 the latency field is present (decode, preprocess, inference, postprocess, total; inference > 0)
S5 the preprocessing equals the evaluator's function byte for byte on the same input: the server's
   input tensor (sha256 in the response, and the in-process wrapper) equals core_preprocess + finalize
   run exactly as src/data/dataset.py runs them on image and mask files on disk; JPEG, PNG and several
   aspect ratios; the placeholder mask's value and size do not change the image tensor
S6 EXIF orientation is applied (by core_preprocess): a landscape-stored JPEG tagged orientation 6 is
   segmented as a portrait photo
S7 a non-image upload is refused with HTTP 400
S8 src.stats is imported neither by this process nor by the server
The model registry (demo/config.py slots e6, e3, shown; DEMO-3):
S9  E6 alone: the S1 server loads e6 only, prints its file name, sha256, engine and stage at start-up,
    offers no comparison and answers with one panel carrying e6's fixed label and description
S10 the FP32 export round trip on a random-weights stand-in in E3's checkpoint schema:
    demo/model/export_fp32_torchscript.py writes the file with plantseg_fp32.json (stage E3, seed 42,
    the stand-in checkpoint's sha256, 116 classes, core_preprocess/1.0.0), bitwise equal to the eager
    model on its synthetic tensors; load_fp32_model refuses tampered identities and an INT8 file;
    demo/model/parity_fp32.py prints PASS on synthetic image files and STOP-refuses a wrong checkpoint sha256
S11 E6 with the FP32 stand-in in slot e3: both load, the option "E3 (FP32)" is offered, and one upload
    returns E6's panel beside E3's on the same photo (this e3 stand-in holds the E6 stand-in's FP32 weights,
    so equal masks do not tell the two panels apart)
S12 E6 with a second INT8 stand-in in slot shown: the same, with the option "shown student (INT8, exploratory)"
S13 a wrong sha256 is refused: e6 or e3 naming another file's sha256, e6 under the marked line's sha256
    (when the file is not that one) and a malformed sha256 each stop start-up (exit 2) with the sha256 refusal
S14 an unset slot is hidden: a slot without an expected sha256 (with or without a file named) is not
    loaded, is absent from /api/info's options, is refused as a comparison (HTTP 400), and the page has no
    built-in option for it
S15 PHONE-FIX-1 holds: mkldnn is off and the quantized engine is qnnpack in every server, and mkldnn was
    off in the export and parity processes
S16 all three slots load together and the switch offers both options, in registry order
Exit 0 only if every check passes. If MODEL_PATH does not exist, a stand-in is built in a temp dir. The
registry checks S10-S16 use an INT8 stand-in as e6 (the --model file if it is one); S9 uses the S1 server.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo import common, config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))    # localhost: never a proxy
SLOT_ENV = tuple(v for s in config.SLOTS for v in (s.path_env, s.sha256_env))
SHOWN_SEED = 20261116            # the second INT8 stand-in (slot shown)
EXPORT_SEED = 42                 # the seed the e3 identity records (E3 seed 42)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return bool(ok)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 30.0) -> dict:
    with OPENER.open(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_file(url: str, filename: str, data: bytes, ctype: str, fields: dict | None = None) -> tuple[int, dict]:
    boundary = "----demo" + secrets.token_hex(8)
    body = b""
    for k, v in (fields or {}).items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with OPENER.open(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def decode_data_url(url: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def evaluator_reference(img_path: Path, mask_path: Path):
    """core_preprocess + finalize exactly as src/data/dataset.py:86-97 runs them for VAL/TEST."""
    from PIL import Image

    from src.data.transforms import core_preprocess, finalize

    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        img_np, mask_np = core_preprocess(im, mk)
    return finalize(img_np, mask_np)[0]


def slot_env(**slots) -> dict:
    """This process's environment without any registry variable, plus the given ones."""
    env = {k: v for k, v in os.environ.items() if k not in SLOT_ENV}
    env.update({k: str(v) for k, v in slots.items()})
    return env


def run_py(args: list, env: dict | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", *map(str, args)], capture_output=True, text=True,
                          env=env, timeout=timeout)


class Server:
    """demo/run_demo.py in a subprocess on a free local port."""

    def __init__(self, tmp: Path, name: str, env: dict, args: tuple = ()):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = tmp / f"server_{name}.log"
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen([sys.executable, "-I", str(config.DEMO_DIR / "run_demo.py"), "--host",
                                      "127.0.0.1", "--port", str(self.port), "--no-build", *map(str, args)],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.waited = 0.0

    def wait(self, timeout: float = 180) -> dict | None:
        t0, health = time.time(), None
        while time.time() - t0 < timeout and self.proc.poll() is None:
            try:
                health = get_json(self.base + "/healthz", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.5)
        self.waited = time.time() - t0
        return health

    def text(self) -> str:
        self.log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def startup_line_ok(log: str, slot: str, path: Path, sha: str, engine: str, stage: str) -> bool:
    pat = (rf"\[demo\] loaded {re.escape(slot)}: {re.escape(path.name)} sha256 {sha} "
           rf"engine {re.escape(engine)} stage {re.escape(stage)}")
    return re.search(pat, log) is not None


def is_stand_in(path: Path) -> bool:
    from demo.server.inference import read_embedded_meta

    try:
        return bool(read_embedded_meta(path).get("stand_in", False))
    except Exception:  # noqa: BLE001 -- not an INT8 file of the demo's form
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="demo smokes")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)

    phone_fix = common.phone_fix()                        # PHONE-FIX-1 in this process too
    import numpy as np
    from PIL import Image, ImageOps

    from demo.server.inference import META_NAME
    from demo.server.preprocess import PROTOCOL_ID, evaluator_preprocess
    from demo.synthetic import make_leaf_image
    from src.eval.adapters import PREPROCESS_PROTOCOL, load_class_map
    from src.eval.model_loading import INT8_TORCHSCRIPT_META

    tmp = Path(tempfile.mkdtemp(prefix="plantseg_demo_smoke_"))
    builder = config.DEMO_DIR / "model" / "build_stand_in.py"
    model = config.resolve_model_path(a.model)
    if not model.is_file():
        model = tmp / "stand_in_int8_qnnpack.pt"
        print(f"[smoke] {config.MODEL_PATH} missing: building a stand-in at {model}", flush=True)
        subprocess.run([sys.executable, "-I", str(builder), "--out", str(model)], check=True,
                       stdout=subprocess.DEVNULL)
    model_sha = common.sha256_file(model)
    servers: dict[str, dict] = {}                         # name -> {"info", "log"} for S15

    # ------------------------------------------------------------------ S1 server starts
    srv = Server(tmp, "A_e6_alone", slot_env(PLANTSEG_DEMO_E6_SHA256=model_sha), ("--model", model))
    base = srv.base
    try:
        health = srv.wait()
        check("S1 server_starts", bool(health and health.get("ok") and health.get("model_sha256") == model_sha),
              f"healthz={ {k: health[k] for k in ('ok', 'model_sha256')} if health else None} after "
              f"{srv.waited:.1f}s; file sha256 {model_sha[:16]}...")
        if not health:
            print(srv.text()[-3000:])
            return 1

        # -------------------------------------------------------------- S2 class list is the repo's
        info = get_json(base + "/api/info")
        raw = json.loads((REPO_ROOT / "configs" / "plantseg_class_map.json").read_text(encoding="utf-8"))
        raw_list = [(e["class_id"], e["name"], e["role"]) for e in raw["entries"]]
        loader_list = [(e["class_id"], e["name"], e["role"]) for e in load_class_map().entries]
        served = [(c["class_id"], c["name"], c["role"]) for c in info["classes"]]
        nc = common.data_facts()["num_classes"]
        check("S2 class_list_is_repos", served == raw_list == loader_list
              and len(served) == nc == info["model"]["num_classes"],
              f"{len(served)} served == {len(raw_list)} raw == {len(loader_list)} loader; configs/data.py "
              f"num_classes {nc}; model outputs {info['model']['num_classes']}; class 0 name "
              f"{served[0][1]!r} role {served[0][2]}")

        # -------------------------------------------------------------- S3-S5 synthetic images
        cases = [("leaf_1024x768.jpg", (1024, 768), "JPEG"), ("leaf_480x640.png", (480, 640), "PNG"),
                 ("leaf_800x800.jpg", (800, 800), "JPEG"), ("leaf_720x1280.jpg", (720, 1280), "JPEG")]
        shape_ok = range_ok = lat_ok = bytes_ok = http_ok = placeholder_ok = True
        details = []
        for i, (fname, size, fmt) in enumerate(cases):
            img = make_leaf_image(100 + i, size)
            path = tmp / fname
            img.save(path, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
            data = path.read_bytes()
            # evaluator reference on files, exactly as the dataset reads them
            with Image.open(path) as im:
                upright = ImageOps.exif_transpose(im).size
            mask_path = tmp / (path.stem + "_placeholder.png")
            Image.new("L", upright, 0).save(mask_path)
            ref = evaluator_reference(path, mask_path)
            ref_bytes = ref.numpy().tobytes()
            # the server's wrapper, in process, on the uploaded bytes
            pre = evaluator_preprocess(Image.open(io.BytesIO(data)))
            alt = evaluator_preprocess(Image.open(io.BytesIO(data)), placeholder_value=7)
            bytes_ok &= pre.tensor[0].numpy().tobytes() == ref_bytes and pre.tensor.dtype == ref.dtype
            placeholder_ok &= alt.tensor.numpy().tobytes() == pre.tensor.numpy().tobytes()
            # the running server
            status, res = post_file(base + "/api/segment", fname, data, f"image/{fmt.lower()}")
            if status != 200:
                check(f"S3 segment_{fname}", False, f"HTTP {status} {res}")
                shape_ok = False
                continue
            http_ok &= res["input"]["input_tensor_sha256"] == hashlib.sha256(ref_bytes).hexdigest()
            box = res["input"]["photo_box_on_canvas"]
            mask = decode_data_url(res["images"]["mask_png"])
            m = np.asarray(mask)
            overlay = decode_data_url(res["images"]["overlay_png"])
            exp_hw = (box["height"], box["width"])
            shape_ok &= (mask.mode == "L" and m.shape == exp_hw and overlay.mode == "RGBA"
                         and overlay.size == (box["width"], box["height"])
                         and res["output"]["logits_shape"] == [1, nc, 512, 512]
                         and max(exp_hw) == 512 and tuple(res["input"]["model_input_shape"]) == (1, 3, 512, 512))
            range_ok &= bool(m.min() >= 0 and m.max() <= nc - 1)
            lat = res.get("latency_ms", {})
            lat_ok &= (all(isinstance(lat.get(k), (int, float)) and lat[k] >= 0
                           for k in ("decode", "preprocess", "inference", "postprocess", "total"))
                       and lat["inference"] > 0)
            details.append(f"{fname}: mask {m.shape} classes {int(m.min())}..{int(m.max())} "
                           f"inference {lat.get('inference')} ms")
        check("S3 mask_shape_and_class_range", shape_ok and range_ok, "; ".join(details))
        check("S4 latency_field_present", lat_ok, "decode/preprocess/inference/postprocess/total in ms")
        check("S5 preprocessing_byte_equal", bytes_ok and http_ok and placeholder_ok
              and PROTOCOL_ID == PREPROCESS_PROTOCOL and META_NAME == INT8_TORCHSCRIPT_META,
              f"in-process wrapper == dataset path: {bytes_ok}; server input sha256 == dataset path: "
              f"{http_ok}; placeholder independent: {placeholder_ok}; protocol {PROTOCOL_ID} == "
              f"src.eval.adapters.PREPROCESS_PROTOCOL; over {len(cases)} images (JPEG and PNG)")

        # -------------------------------------------------------------- S6 EXIF orientation
        img = make_leaf_image(201, (800, 600))
        exif = Image.Exif()
        exif[0x0112] = 6                                   # rotate 90 degrees to display
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, exif=exif.tobytes())
        status, res = post_file(base + "/api/segment", "rotated.jpg", buf.getvalue(), "image/jpeg")
        box = res.get("input", {}).get("photo_box_on_canvas", {})
        check("S6 exif_orientation_applied", status == 200 and box.get("height") == 512 and box.get("width") == 384
              and res["input"]["original_size_wh"] == [800, 600],
              f"stored 800x600 with orientation 6 -> photo box {box.get('width')}x{box.get('height')} (w x h)")

        # -------------------------------------------------------------- S7 bad upload
        status, res = post_file(base + "/api/segment", "notes.txt", b"not an image", "text/plain")
        check("S7 bad_upload_refused", status == 400, f"HTTP {status} {res.get('detail', '')}")

        # -------------------------------------------------------------- S8 no src.stats
        info = get_json(base + "/api/info")
        check("S8 no_src_stats", info.get("src_stats_imported") == [] and common.stats_modules_loaded() == [],
              f"server {info.get('src_stats_imported')}, smoke {common.stats_modules_loaded()}")

        # -------------------------------------------------------------- S9 E6 alone (this server)
        photo = tmp / "compare_1024x768.jpg"
        make_leaf_image(150, (1024, 768)).save(photo, format="JPEG", quality=92)
        photo_bytes = photo.read_bytes()
        st_one, one = post_file(base + "/api/segment", photo.name, photo_bytes, "image/jpeg")
        st_e3, r_e3 = post_file(base + "/api/segment", photo.name, photo_bytes, "image/jpeg", {"compare": "e3"})
        st_sh, r_sh = post_file(base + "/api/segment", photo.name, photo_bytes, "image/jpeg", {"compare": "shown"})
        log = srv.text()
        e6_info = info["models"][0]
        spec6 = config.SLOTS[0]
        servers["A_e6_alone"] = {"info": info}
        a_hidden = (info["compare_options"] == [] and st_e3 == 400 and st_sh == 400
                    and [m["slot"] for m in info["models"]] == ["e6"])
        check("S9 e6_alone", [m["slot"] for m in info["models"]] == ["e6"] and info["compare_options"] == []
              and e6_info["label"] == spec6.label and e6_info["description"] == spec6.description
              and startup_line_ok(log, "e6", model, model_sha, "qnnpack", e6_info["stage"])
              and st_one == 200 and one["compare"] is None and len(one["panels"]) == 1
              and one["panels"][0]["slot"] == "e6" and one["panels"][0]["sha256"] == model_sha
              and one["panels"][0]["images"]["mask_png"] == one["images"]["mask_png"],
              f"loaded {[m['slot'] for m in info['models']]}; options {info['compare_options']}; start-up line "
              f"'loaded e6: {model.name} sha256 {model_sha[:12]}... engine qnnpack stage {e6_info['stage']}'; "
              f"label {e6_info['label']!r} / {e6_info['description']!r}; 1 panel")
    finally:
        srv.stop()

    # ------------------------------------------------------------------ the registry's stand-ins
    reg_e6 = model if is_stand_in(model) else tmp / "registry_e6_stand_in.pt"
    if reg_e6 != model:
        subprocess.run([sys.executable, "-I", str(builder), "--out", str(reg_e6)], check=True,
                       stdout=subprocess.DEVNULL)
    shown = tmp / "shown_stand_in_int8_qnnpack.pt"
    subprocess.run([sys.executable, "-I", str(builder), "--out", str(shown), "--seed", str(SHOWN_SEED)],
                   check=True, stdout=subprocess.DEVNULL)
    e6_sha, shown_sha = common.sha256_file(reg_e6), common.sha256_file(shown)

    # ------------------------------------------------------------------ S10 FP32 export round trip
    import torch

    from demo.server.inference import (FP32_META_NAME, NUM_CLASSES, SlotRefused, load_fp32_model,
                                       read_embedded_meta)
    from src.eval.model_loading import FROZEN_NUM_CLASSES

    ckpt = tmp / "e3_stand_in_checkpoint.pt"
    subprocess.run([sys.executable, "-I", str(builder), "--e3-checkpoint", str(ckpt)], check=True,
                   stdout=subprocess.DEVNULL)
    ckpt_sha = common.sha256_file(ckpt)
    fp32 = tmp / "e3_fp32_stand_in.pt"
    ex = run_py([config.DEMO_DIR / "model" / "export_fp32_torchscript.py", ckpt, "--source-sha256", ckpt_sha,
                 "--stage", "E3", "--seed", EXPORT_SEED, "--out", fp32, "--e6", reg_e6])
    export_ok = ex.returncode == 0 and fp32.is_file()
    facts = json.loads(ex.stdout[:ex.stdout.rindex("\n[export]")]) if export_ok else {}
    if not export_ok:
        print(ex.stdout[-2000:], ex.stderr[-3000:])
    fp32_sha = common.sha256_file(fp32) if export_ok else ""
    meta = read_embedded_meta(fp32, FP32_META_NAME) if export_ok else {}
    ident_ok = (meta.get("stage") == "E3" and meta.get("seed") == EXPORT_SEED
                and meta.get("source_checkpoint_sha256") == ckpt_sha and meta.get("num_classes") == 116
                and meta.get("preprocess_protocol") == PREPROCESS_PROTOCOL == PROTOCOL_ID
                and meta.get("stand_in") is True and NUM_CLASSES == FROZEN_NUM_CLASSES == 116)
    rt = facts.get("round_trip", {})
    rt_ok = rt.get("bitwise_equal") == len(rt.get("output_shapes", [])) == 4 and rt.get("max_abs_diff") == 0.0
    refused = []
    if export_ok:
        loaded = load_fp32_model(fp32, "E3")
        extra = {FP32_META_NAME: ""}
        ts = torch.jit.load(str(fp32), map_location="cpu", _extra_files=extra)
        good = json.loads(extra[FP32_META_NAME])
        bad_ids = {"classes_115": {**good, "num_classes": 115},
                   "other_preprocessing": {**good, "preprocess_protocol": "upstream/1.0.0"},
                   "stage_E2": {**good, "stage": "E2"},
                   "no_seed": {k: v for k, v in good.items() if k != "seed"}}
        for name, bad in bad_ids.items():
            p = tmp / f"tampered_{name}.pt"
            torch.jit.save(ts, str(p), _extra_files={FP32_META_NAME: json.dumps(bad)})
            try:
                load_fp32_model(p, "E3")
                refused.append((name, False))
            except SlotRefused:
                refused.append((name, True))
        try:
            load_fp32_model(reg_e6, "E3")
            refused.append(("int8_file", False))
        except RuntimeError:
            refused.append(("int8_file", True))
        del ts, loaded
    pdir = tmp / "parity_images"
    pdir.mkdir()
    for i, (size, fmt) in enumerate((((1024, 768), "JPEG"), ((480, 640), "PNG"), ((800, 800), "JPEG"),
                                     ((720, 1280), "JPEG"), ((1200, 900), "PNG"), ((600, 900), "JPEG"))):
        make_leaf_image(400 + i, size).save(pdir / f"syn_{i}.{fmt.lower()[:3] if fmt == 'PNG' else 'jpg'}",
                                            format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    par = run_py([config.DEMO_DIR / "model" / "parity_fp32.py", "--model", fp32, "--checkpoint", ckpt,
                  "--source-sha256", ckpt_sha, "--images", pdir, "--n", 6])
    par_line = (par.stdout.strip().splitlines() or [""])[-1]
    par_facts = json.loads(par.stdout[par.stdout.index("\n{") + 1:par.stdout.rindex("}") + 1]) \
        if par.returncode == 0 else {}
    wrong = run_py([config.DEMO_DIR / "model" / "parity_fp32.py", "--model", fp32, "--checkpoint", ckpt,
                    "--source-sha256", shown_sha, "--images", pdir, "--n", 1])
    wrong_ok = wrong.returncode != 0 and "STOP checkpoint sha256" in (wrong.stdout + wrong.stderr)
    check("S10 fp32_export_round_trip", export_ok and ident_ok and rt_ok and len(refused) == 5
          and all(ok for _, ok in refused) and par.returncode == 0 and par_line.startswith("PASS parity_fp32")
          and wrong_ok,
          f"export {fp32.name} sha256 {fp32_sha[:12]}...; identity stage {meta.get('stage')} seed {meta.get('seed')} "
          f"source {str(meta.get('source_checkpoint_sha256'))[:12]}... (= stand-in checkpoint: "
          f"{meta.get('source_checkpoint_sha256') == ckpt_sha}) classes {meta.get('num_classes')} preprocessing "
          f"{meta.get('preprocess_protocol')}; round trip bitwise {rt.get('bitwise_equal')}/"
          f"{len(rt.get('output_shapes', []))}; refused {sum(ok for _, ok in refused)}/{len(refused)} "
          f"({', '.join(n for n, ok in refused if ok)}); parity: '{par_line}'; wrong checkpoint sha256 -> "
          f"exit {wrong.returncode}")

    # ------------------------------------------------------------------ S11, S12, S16: registry servers
    def registry_server(name: str, env: dict, compare: list[str]) -> dict:
        s = Server(tmp, name, env)
        out = {"ok": False, "detail": "", "info": None}
        try:
            h = s.wait()
            if not h:
                out["detail"] = f"did not start: {s.text()[-1500:]}"
                return out
            inf = get_json(s.base + "/api/info")
            out["info"], out["log"], out["health"] = inf, s.text(), h
            out["single"] = post_file(s.base + "/api/segment", photo.name, photo_bytes, "image/jpeg")
            out["compare"] = {c: post_file(s.base + "/api/segment", photo.name, photo_bytes, "image/jpeg",
                                           {"compare": c}) for c in compare}
            out["ok"] = True
        finally:
            s.stop()
        return out

    def panels_ok(res: dict, slot: str, kind: str, engine: str, sha: str) -> bool:
        p = res.get("panels", [])
        if len(p) != 2 or res.get("compare") != slot:
            return False
        m0, m1 = (np.asarray(decode_data_url(x["images"]["mask_png"])) for x in p)
        box = res["input"]["photo_box_on_canvas"]
        ok = (p[0]["slot"] == "e6" and p[0]["sha256"] == e6_sha and p[1]["slot"] == slot
              and p[1]["kind"] == kind and p[1]["engine"] == engine and p[1]["sha256"] == sha
              and p[1]["sha256_short"] == common.short_sha(sha)
              and m0.shape == m1.shape == (box["height"], box["width"])
              and int(m1.max()) <= nc - 1 and p[1]["latency_ms"]["inference"] > 0
              and p[1]["output"]["logits_shape"] == [1, nc, 512, 512]
              and res["latency_ms"]["inference_compare"] == p[1]["latency_ms"]["inference"])
        if model == reg_e6:          # the same e6 file as server A: its panel is unchanged by the comparison
            ok = ok and p[0]["images"]["mask_png"] == one["panels"][0]["images"]["mask_png"]
        return ok

    spec = {s.key: s for s in config.SLOTS}
    env_b = slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256=e6_sha, PLANTSEG_DEMO_E3_MODEL=fp32,
                     PLANTSEG_DEMO_E3_SHA256=fp32_sha, PLANTSEG_DEMO_SHOWN_MODEL=shown)   # shown: no sha256
    b = registry_server("B_e6_e3", env_b, ["e3", "shown"]) if export_ok else {"ok": False, "detail": "no export"}
    if b["ok"]:
        inf, (st, res) = b["info"], b["compare"]["e3"]
        e3m = next((m for m in inf["models"] if m["slot"] == "e3"), {})
        servers["B_e6_e3"] = {"info": inf}
        b_ok = ([m["slot"] for m in inf["models"]] == ["e6", "e3"]
                and inf["compare_options"] == [{"slot": "e3", "option": "E3 (FP32)", "label": spec["e3"].label,
                                                "description": spec["e3"].description}]
                and startup_line_ok(b["log"], "e6", reg_e6, e6_sha, "qnnpack", "DEMO_STAND_IN")
                and startup_line_ok(b["log"], "e3", fp32, fp32_sha, "fp32 CPU (mkldnn off)", "E3")
                and b["health"]["slots"] == {"e6": e6_sha, "e3": fp32_sha}
                and st == 200 and panels_ok(res, "e3", "fp32", "fp32 CPU (mkldnn off)", fp32_sha)
                and inf["src_stats_imported"] == [])
        detail = (f"loaded {[m['slot'] for m in inf['models']]}; options {[o['option'] for o in inf['compare_options']]}; "
                  f"e3 {e3m.get('file')} sha256 {fp32_sha[:12]}... engine {e3m.get('engine')} stage {e3m.get('stage')}; "
                  f"compare=e3: HTTP {st}, {len(res.get('panels', []))} panels, inference e6 "
                  f"{res.get('latency_ms', {}).get('inference')} ms / e3 {res.get('latency_ms', {}).get('inference_compare')} "
                  f"ms; E6's panel mask compared with server A's: {model == reg_e6}")
        check("S11 e6_with_fp32_stand_in_in_e3", b_ok, detail)
    else:
        check("S11 e6_with_fp32_stand_in_in_e3", False, b.get("detail", ""))

    env_c = slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256=e6_sha,
                     PLANTSEG_DEMO_SHOWN_MODEL=shown, PLANTSEG_DEMO_SHOWN_SHA256=shown_sha)
    c = registry_server("C_e6_shown", env_c, ["shown", "e3"])
    if c["ok"]:
        inf, (st, res) = c["info"], c["compare"]["shown"]
        servers["C_e6_shown"] = {"info": inf}
        c_ok = ([m["slot"] for m in inf["models"]] == ["e6", "shown"]
                and inf["compare_options"] == [{"slot": "shown", "option": "shown student (INT8, exploratory)",
                                                "label": spec["shown"].label,
                                                "description": spec["shown"].description}]
                and startup_line_ok(c["log"], "shown", shown, shown_sha, "qnnpack", "DEMO_STAND_IN")
                and st == 200 and panels_ok(res, "shown", "int8", "qnnpack", shown_sha)
                and c["compare"]["e3"][0] == 400 and inf["src_stats_imported"] == [])
        check("S12 e6_with_int8_stand_in_in_shown", c_ok,
              f"loaded {[m['slot'] for m in inf['models']]}; options {[o['option'] for o in inf['compare_options']]}; "
              f"shown {shown.name} sha256 {shown_sha[:12]}... engine qnnpack; compare=shown: HTTP {st}, "
              f"{len(res.get('panels', []))} panels, inference e6 {res.get('latency_ms', {}).get('inference')} ms / "
              f"shown {res.get('latency_ms', {}).get('inference_compare')} ms; compare=e3 -> HTTP "
              f"{c['compare']['e3'][0]}")
    else:
        check("S12 e6_with_int8_stand_in_in_shown", False, c.get("detail", ""))

    # ------------------------------------------------------------------ S13 a wrong sha256 is refused
    run_demo = config.DEMO_DIR / "run_demo.py"
    cases13 = {
        "e6_other_files_sha256": slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256=shown_sha),
        "e3_other_files_sha256": slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256=e6_sha,
                                          PLANTSEG_DEMO_E3_MODEL=fp32, PLANTSEG_DEMO_E3_SHA256=shown_sha),
        "e6_malformed_sha256": slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256="abc123"),
    }
    if config.E6_SHA256 and config.E6_SHA256 != e6_sha and not os.environ.get("PLANTSEG_DEMO_E6_SHA256"):
        cases13["e6_marked_line_sha256"] = slot_env(PLANTSEG_DEMO_MODEL=reg_e6)
    outcomes = []
    for name, env in cases13.items():
        r = run_py([run_demo, "--host", "127.0.0.1", "--port", free_port(), "--no-build"], env=env, timeout=180)
        text = r.stdout + r.stderr
        ok = (r.returncode == 2 and "[demo] refused to start" in text and "[demo] loaded" not in text
              and "Uvicorn running" not in text)
        slot = "e3" if name.startswith("e3") else "e6"
        ok &= (f"slot {slot}" in text) and ("not 64 hex digits" in text if "malformed" in name
                                             else "was not loaded" in text)
        outcomes.append((name, ok, r.returncode))
    check("S13 wrong_sha256_refused", all(ok for _, ok, _ in outcomes) and len(outcomes) >= 3,
          "; ".join(f"{n}: exit {rc}, sha256 refusal, server not started: {ok}" for n, ok, rc in outcomes))

    # ------------------------------------------------------------------ S14 an unset slot is hidden
    page = (config.DEMO_DIR / "web" / "index.html").read_text(encoding="utf-8")
    page_ok = ('value="e3"' not in page and 'value="shown"' not in page
               and re.search(r'<fieldset class="compare" id="compareBox" hidden>', page) is not None)
    b_hidden = bool(b.get("ok")) and (
        [o["slot"] for o in b["info"]["compare_options"]] == ["e3"]
        and "shown" not in [m["slot"] for m in b["info"]["models"]]
        and b["compare"]["shown"][0] == 400
        and any(n.startswith("shown: not loaded: no expected sha256 is set (a file is named")
                for n in b["info"]["startup_notes"]))
    check("S14 unset_slot_hidden", a_hidden and b_hidden and page_ok,
          f"e6 alone: options [] and compare=e3/shown -> HTTP {st_e3}/{st_sh}: {a_hidden}; shown named without "
          f"a sha256 beside a loaded e3: not loaded, not offered, compare=shown -> HTTP 400: {b_hidden}; the page "
          f"has no built-in e3/shown option and its switch starts hidden: {page_ok}")

    # ------------------------------------------------------------------ S16 all three slots
    env_d = slot_env(PLANTSEG_DEMO_MODEL=reg_e6, PLANTSEG_DEMO_E6_SHA256=e6_sha, PLANTSEG_DEMO_E3_MODEL=fp32,
                     PLANTSEG_DEMO_E3_SHA256=fp32_sha, PLANTSEG_DEMO_SHOWN_MODEL=shown,
                     PLANTSEG_DEMO_SHOWN_SHA256=shown_sha)
    d = registry_server("D_all_three", env_d, ["e3", "shown"]) if export_ok else {"ok": False, "detail": "no export"}
    d_ok = False
    if d["ok"]:
        servers["D_all_three"] = {"info": d["info"]}
        d_ok = ([m["slot"] for m in d["info"]["models"]] == ["e6", "e3", "shown"]
                and [o["slot"] for o in d["info"]["compare_options"]] == ["e3", "shown"]
                and d["compare"]["e3"][0] == d["compare"]["shown"][0] == 200
                and panels_ok(d["compare"]["e3"][1], "e3", "fp32", "fp32 CPU (mkldnn off)", fp32_sha)
                and panels_ok(d["compare"]["shown"][1], "shown", "int8", "qnnpack", shown_sha))

    # ------------------------------------------------------------------ S15 PHONE-FIX-1
    rts = {n: s["info"]["runtime"] for n, s in servers.items()}
    servers_ok = len(rts) == 4 and all(r["mkldnn_enabled"] is False and r["quantized_engine"] == "qnnpack"
                                       for r in rts.values())
    export_fix = facts.get("identity", {}).get("mkldnn_enabled_at_trace") is False
    parity_fix = par_facts.get("phone_fix", {}).get("mkldnn_enabled") is False
    check("S15 phone_fix_1", servers_ok and export_fix and parity_fix and phone_fix["mkldnn_enabled"] is False,
          f"servers {sorted(rts)}: mkldnn off and engine qnnpack in all: {servers_ok}; export trace mkldnn off: "
          f"{export_fix}; parity process mkldnn off: {parity_fix}; smoke process mkldnn off: "
          f"{phone_fix['mkldnn_enabled'] is False}")
    check("S16 all_three_slots", d_ok,
          f"loaded {[m['slot'] for m in d['info']['models']] if d.get('ok') else None}; options "
          f"{[o['option'] for o in d['info']['compare_options']] if d.get('ok') else None}; both comparisons "
          f"answered with two panels: {d_ok}" if d.get("ok") else d.get("detail", ""))

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"SUMMARY {len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
