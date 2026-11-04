#!/usr/bin/env python3
"""Smokes for the mobile web demo (synthetic images only; no dataset, no network).

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
Exit 0 only if every check passes. If MODEL_PATH does not exist, a stand-in is built in a temp dir.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
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


def post_file(url: str, filename: str, data: bytes, ctype: str) -> tuple[int, dict]:
    boundary = "----demo" + secrets.token_hex(8)
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="demo smokes")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)

    import numpy as np
    from PIL import Image, ImageOps

    from demo.server.inference import META_NAME
    from demo.server.preprocess import PROTOCOL_ID, evaluator_preprocess
    from demo.synthetic import make_leaf_image
    from src.eval.adapters import PREPROCESS_PROTOCOL, load_class_map
    from src.eval.model_loading import INT8_TORCHSCRIPT_META

    tmp = Path(tempfile.mkdtemp(prefix="plantseg_demo_smoke_"))
    model = config.resolve_model_path(a.model)
    if not model.is_file():
        model = tmp / "stand_in_int8_qnnpack.pt"
        print(f"[smoke] {config.MODEL_PATH} missing: building a stand-in at {model}", flush=True)
        subprocess.run([sys.executable, "-I", str(config.DEMO_DIR / "model" / "build_stand_in.py"),
                        "--out", str(model)], check=True, stdout=subprocess.DEVNULL)
    model_sha = common.sha256_file(model)

    # ------------------------------------------------------------------ S1 server starts
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(tmp / "server.log", "wb")
    proc = subprocess.Popen([sys.executable, "-I", str(config.DEMO_DIR / "run_demo.py"), "--host",
                             "127.0.0.1", "--port", str(port), "--model", str(model), "--no-build"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        t0, health = time.time(), None
        while time.time() - t0 < 180 and proc.poll() is None:
            try:
                health = get_json(base + "/healthz", timeout=2)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.5)
        check("S1 server_starts", bool(health and health.get("ok") and health.get("model_sha256") == model_sha),
              f"healthz={health} after {time.time() - t0:.1f}s; file sha256 {model_sha[:16]}...")
        if not health:
            print((tmp / "server.log").read_text(errors="replace")[-3000:])
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
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"SUMMARY {len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; FAILED {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
