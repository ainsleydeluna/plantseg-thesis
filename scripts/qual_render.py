#!/usr/bin/env python3
"""QUAL-1, step 3 of 3 (AM-27 items 3, 4 and 6): the five qualitative figures, drawn from the arrays that the
predict steps saved. No model is loaded and nothing is predicted or scored here.

Gate (AM-27 item 4): on TEST the step draws nothing unless every predict record exists, names the same
select/S.json, checked every tile it must (teacher 9; E1, E3 and E6 9 clean + 3 x 3 corrupted) and passed each
one. Every array is re-hashed against the record that wrote it before it is drawn.

Figures (AM-27 item 3; the brief's item 3; ruling QUAL-A-1; numbering as the brief gives it):
  fig4_2.png         the 50th-percentile image (position 2): the image and its mask as read, the 512 x 512
                     canvas, E6's error map, the four predictions, and under each its per-image disease-only
                     mIoU of record as a percentage (Q8: the only numbers any figure prints)
  fig4_3{a,b,c}.png  k = 9, S[0..2], S[3..5], S[6..8]: input, ground truth, Teacher, E1, E3, E6
  fig4_4{a,b,c}.png  k = 9, the same parts: input and the class-level error maps of the four models (Q4)
  fig4_5{a,b,c}.png  k = 3, S[0], S[1], S[2]: clean, motion blur, Gaussian noise and fog (severity 2) rows;
                     input, ground truth, E1, E3, E6 columns
  fig4_6.png         k = 3: input and the class-confusion overlays of the four models, each with the class
                     pairs it confuses (Q5; "and further pairs" without a number: QUAL-AUD-1 F-05; names wrapped at
                     a space, never cut: F-07)
Style (the brief's item 6): PNG, 300 dpi, 6.5 in wide (1,950 px), Arial or Liberation Sans in black, no title
inside, column headers inside; the canvas is drawn whole, padding included (in error maps and overlays the padding
is grey whatever was predicted there); one colour per class index across every figure (Q3, PALETTE_RULE);
categorical keys from a colour-blind-safe set (CATEGORY); every part's legend names every class it shows, black
included ("Background": F-08). Scaling (QUAL-AUD-1 N-10): class maps and the categorical layer of error maps and
overlays with nearest, so that no colour outside a legend appears; photographs (the inputs and the grey underlay)
with the box filter.
A layout fault is corrected by rendering again from the saved arrays into a new folder (AM-27 item 5; Q10:
--rerender, from a commit that differs from the predict commit in scripts/qual_render.py alone).

Writes <run>/<out>/ (the figures, PNG files only) and <run>/manifest_<out>.json, outside the figures folder
(QUAL-AUD-1 N-06; AM-27 item 6: the identities of S, the sha256 of every input and output, per position the sha256
of the image, mask and cached copy read, all thirteen records' files_sha256, each tile's check, the largest gap,
the versions). Prints counts, hashes and PASS or STOP lines only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.qual_select import (CELLS, CLEAN, CONDITIONS, K_ALL, K_CELLS, MODEL_ORDER, MODELS,  # noqa: E402
                                 P50_POSITION, RECORD_FORMAT, TIE_READING, QualStop, canonical_bytes,
                                 expected_commit, k_of, no_test, read_s, repo_state, run_dir, run_step, say,
                                 sha256_bytes, sha256_file, strict_json, write_new)

DPI = 300
WIDTH = 1950                       # 6.5 in at 300 dpi
MARGIN = 6
GAP = 10
HEADER_PT, TEXT_PT = 8, 7
FONT_FAMILIES = ("Arial", "Liberation Sans")
ROW_LABELS = {CLEAN: "Clean", "motion_blur_s2": "Motion blur", "gaussian_noise_s2": "Gaussian noise",
              "fog_s2": "Fog"}
BACKGROUND = (0, 0, 0)             # class 0
IGNORE = (128, 128, 128)           # mask 255: the canvas padding of the ground truth
FIG_BG = (255, 255, 255)
BORDER = (160, 160, 160)
# Categorical keys (Okabe-Ito blue, orange, sky blue; Tol red). Worst case over normal, protan, deutan and tritan
# vision (Machado 2009 at full severity), CIEDE2000: every pair >= 19.6, every colour >= 18.5 from the greys of
# the error-map underlay (QUAL-1 measurement, Phase A report).
CATEGORY = {"tp": (0x00, 0x72, 0xB2), "fp": (0xE6, 0x9F, 0x00), "fn": (0xCC, 0x33, 0x11), "wrong": (0x56, 0xB4, 0xE9)}
CATEGORY_TEXT = {"tp": "True positive (its own class)", "fn": "False negative (background or another class)",
                 "fp": "False positive (background predicted as disease)"}
CONFUSION_TEXT = {"tp": "Correct class", "wrong": "Wrong disease class", "fn": "Missed (predicted background)",
                  "fp": "False alarm (background predicted as disease)"}
# Error maps (ruling QUAL-A-1 Q4): class-level, so that a map agrees with the metric of record. On a ground-truth
# disease pixel: TP when predicted as its own class, FN otherwise (background or another disease class); FP is a
# background pixel predicted as a disease class; padding (255) is drawn in the padding grey whatever was predicted;
# a background pixel predicted as background has no colour. A tile's TP area is the tp count of its per-image record.
# Figure 4.6 splits FN into "wrong disease class" and "missed".
ERROR_MAP_RULE = "class"
# The underlay of error maps and overlays: the canvas in grey, mapped into [UNDERLAY_MIN, 255], so that it stays
# lighter than the padding grey (IGNORE) on any image, a black background included.
UNDERLAY_MIN = 170
GREY_KEY = ((212, 212, 212), "Image in grey: background")
PAD_KEY = (IGNORE, "Padding (ignored)")
# Class colours (ruling QUAL-A-1 Q3): one colour per class index for the whole run. The classes present in any saved
# array of the run (ground truth or prediction, any model, any condition) take CLASS_COLOURS' first colours in
# ascending class index; background black and padding grey are fixed; a re-render from the same arrays gives the
# same colours, and every figure part's legend names every class it shows. CLASS_COLOURS is a greedy
# max-min list over a 16-level sRGB grid (only colours of CIELAB L* >= 45 and chroma >= 15: visible on the black
# background, never grey like the padding), distance = the worst CIEDE2000 over normal, protan, deutan and tritan
# vision (Machado 2009), seeded with black and the padding grey; smoke_qual --palette regenerates it. Measured
# worst-case minimum distance of the first k colours: k 4: 25.2, 8: 14.6, 12: 11.2, 16: 8.9, 32: 6.2, 115: 3.1.
PALETTE_RULE = "present_first"
CLASS_COLOURS = (
    "#22FF33", "#FFDDFF", "#EE00FF", "#DD1111", "#EEFFDD", "#88BB88", "#BB99BB", "#557799", "#996655", "#FF6644",
    "#FF88FF", "#AAEE99", "#EEBBBB", "#779977", "#BB00CC", "#CC77CC", "#44AA55", "#557722", "#CC0088", "#CCBBDD",
    "#DD7799", "#AACC66", "#FFFF88", "#DDAA00", "#338866", "#BB55BB", "#44DDBB", "#AA8800", "#99EEDD", "#EEBBFF",
    "#0088FF", "#8888AA", "#DD3399", "#CCEEBB", "#337766", "#CC3344", "#DD99DD", "#9922FF", "#EE0066", "#CC9999",
    "#BBCC99", "#CC6688", "#CCEE77", "#66BB66", "#EEFFAA", "#22BBFF", "#447744", "#99CCCC", "#77AA77", "#886677",
    "#33AAAA", "#33FFFF", "#BBCC44", "#EE00DD", "#DD2277", "#7777CC", "#776688", "#FF55AA", "#CCAABB", "#DDAAAA",
    "#559944", "#CCFFEE", "#EE9988", "#AA5511", "#339966", "#33CCAA", "#CC9900", "#5599BB", "#AADDDD", "#EE77EE",
    "#CCFF66", "#CCCC88", "#FFCCCC", "#0077AA", "#BBDD77", "#77EECC", "#CCFFBB", "#BBAACC", "#338899", "#AA6688",
    "#668877", "#BB8888", "#4477EE", "#BBBB99", "#997733", "#CCAA44", "#889933", "#CC11AA", "#EEDD00", "#CC8855",
    "#8899DD", "#88BBBB", "#CCDDBB", "#FFFFCC", "#AA7722", "#EECCEE", "#BBDDAA", "#DD66BB", "#DDBBCC", "#FF8866",
    "#BB88AA", "#9955AA", "#88CCEE", "#AA5577", "#BBEEEE", "#BB9988", "#DDAA77", "#886644", "#FFEECC", "#BB5599",
    "#667755", "#AA7788", "#DD0055", "#AA44BB", "#88FFDD")
CLASS_COLOURS_SHA256 = "447de3521ed2f22a7cc1ea82e1054739d9ef440b5972ecae4eaee0ae87816285"


def pt(points: float) -> int:
    return round(points * DPI / 72)


# --------------------------------------------------------------------------------------------- the inputs
class Inputs:
    """S, the predict records, the checks and the arrays of one run, re-verified."""

    def __init__(self, run: Path, split: str, select_sha256: str | None):
        self.run, self.split = run, split
        self.s, self.s_sha = read_s(run, select_sha256)
        if self.s["split"] != split:
            raise QualStop("s_split", "S.json is of another split")
        self.select_record_bytes = (run / "select" / "record.json").read_bytes()
        self.select_record = strict_json(self.select_record_bytes, "select record")
        if (self.select_record.get("outputs") or {}).get("select/S.json") != self.s_sha:
            raise QualStop("record_invalid", "the select record does not name this S.json")
        self.repos = {"select": self.select_record.get("repo")}
        self.records, self.checks, self.values, self.files = {}, {}, {}, {}
        for model in MODEL_ORDER:
            p = run / "records" / f"predict_{model}.json"
            if not p.is_file():
                raise QualStop("record_missing", f"no predict record of {model}")
            data = p.read_bytes()
            rec = strict_json(data, f"predict record {model}")
            self.files[f"records/predict_{model}.json"] = sha256_bytes(data)
            if rec.get("format") != RECORD_FORMAT or rec.get("model") != model or rec.get("split") != split \
                    or rec.get("select_sha256") != self.s_sha or rec.get("streamcheck") is not None:
                raise QualStop("record_invalid", f"the predict record of {model} is not this run's")
            for name, rel in (("checks", f"checks/{model}.json"), ("values", f"values/{model}.json")):
                data = (run / rel).read_bytes()
                if sha256_bytes(data) != rec["outputs"][name]:
                    raise QualStop("record_changed", f"{rel} is not the file its record names")
                self.files[rel] = sha256_bytes(data)
                getattr(self, name)[model] = strict_json(data, rel)
            for rel, digest in rec["outputs"]["arrays"].items():
                if sha256_file(run / rel) != digest:
                    raise QualStop("array_changed", f"{rel} is not the array its record names")
                self.files[rel] = digest
            self.records[model] = rec
            self.repos[model] = rec.get("repo")
        self.gate()

    def gate(self) -> None:
        """AM-27 item 4: every tile of every figure checked and passed (TEST); VAL draws regardless."""
        self.n_tiles, self.largest = 0, None
        for model in MODEL_ORDER:
            tiles = self.checks[model]
            for t in tiles:
                if t["gap"] is not None:
                    self.largest = t["gap"] if self.largest is None else max(self.largest, t["gap"])
            self.n_tiles += len(tiles)
            if self.split != "test":
                continue
            need = sum(k_of(c) for c in CONDITIONS[model])
            got = {(t["condition"], t.get("position")) for t in tiles if t["pass"]}
            want = {(c, p) for c in CONDITIONS[model] for p in range(1, k_of(c) + 1)}
            if len(tiles) != need or got != want or not self.records[model]["summary"]["all_pass"]:
                raise QualStop("check_incomplete", f"{model}: not every tile was checked and passed; nothing is drawn")

    def array(self, rel: str):
        import numpy as np
        from PIL import Image
        if rel not in self.files:
            raise QualStop("array_unrecorded", f"{rel} is not an array a predict record wrote")
        with Image.open(io.BytesIO((self.run / rel).read_bytes())) as im:
            return np.asarray(im).copy()

    def pred(self, model: str, condition: str, pos: int):
        return self.array(f"arrays/pred/{model}/{condition}/{pos}.png")

    def canvas(self, condition: str, pos: int):
        return self.array(f"arrays/input/{condition}/{pos}.png")

    def mask(self, pos: int):
        return self.array(f"arrays/mask/{pos}.png")

    def value(self, model: str, condition: str, pos: int):
        return (self.values.get(model) or {}).get(condition, {}).get(str(pos))


# ------------------------------------------------------------------------------------------------ colours
def hex_rgb(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def class_lut(present: list):
    """A 256 x 3 lookup table: background black, padding grey, the present classes in CLASS_COLOURS order."""
    import numpy as np
    if len(present) > len(CLASS_COLOURS):
        raise QualStop("palette", "more classes than colours")
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:] = (255, 0, 255)                                # never shown: a class outside `present` is an error
    lut[0], lut[255] = BACKGROUND, IGNORE
    for i, c in enumerate(present):
        lut[c] = hex_rgb(CLASS_COLOURS[i])
    return lut


def colourise(arr, lut, present_set):
    import numpy as np
    vals = set(int(v) for v in np.unique(arr)) - {0, 255}
    if not vals <= present_set:
        raise QualStop("palette", "a class map holds a class without an assigned colour")
    return lut[arr]


def underlay(canvas):
    """The background of error maps and overlays: the canvas in grey, mapped into [UNDERLAY_MIN, 255], so that every
    key colour stands out on it and the padding grey stays darker than any of it."""
    import numpy as np
    y = 0.299 * canvas[..., 0] + 0.587 * canvas[..., 1] + 0.114 * canvas[..., 2]
    g = np.clip(np.rint(UNDERLAY_MIN + y * (255 - UNDERLAY_MIN) / 255.0), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def error_layers(gt, pred) -> dict:
    """Ruling QUAL-A-1 Q4, class-level, as the metric of record. On a ground-truth disease pixel: `tp` when the
    prediction is its own class, `fn` otherwise, split into `wrong` (another disease class) and `missed`
    (background) for Figure 4.6; `fp` is a background pixel predicted as a disease class; `pad` is the padding
    (255, the evaluator's ignore_index), which is none of them."""
    pad = gt == 255
    g_d = ~pad & (gt >= 1)
    tp = g_d & (pred == gt)
    wrong = g_d & (pred >= 1) & (pred != gt)
    missed = g_d & (pred == 0)
    return {"tp": tp, "fn": wrong | missed, "fp": ~pad & (gt == 0) & (pred >= 1), "wrong": wrong, "missed": missed,
            "pad": pad}


def _layer(shape, parts) -> tuple:
    import numpy as np
    rgb = np.zeros(tuple(shape) + (3,), dtype=np.uint8)
    where = np.zeros(tuple(shape), dtype=bool)
    for sel, colour in parts:
        rgb[sel] = colour
        where |= sel
    return rgb, where


def error_layer(gt, pred) -> tuple:
    """The categorical layer of an error map (Q4): padding grey, then TP, FN and FP; (rgb, where)."""
    lay = error_layers(gt, pred)
    return _layer(gt.shape, [(lay["pad"], IGNORE)] + [(lay[k], CATEGORY[k]) for k in ("tp", "fn", "fp")])


def confusion_layer(gt, pred) -> tuple:
    """The categorical layer of a Figure 4.6 overlay: padding grey, correct class, wrong disease class, missed and
    false alarm; (rgb, where)."""
    lay = error_layers(gt, pred)
    return _layer(gt.shape, [(lay["pad"], IGNORE), (lay["tp"], CATEGORY["tp"]), (lay["wrong"], CATEGORY["wrong"]),
                             (lay["missed"], CATEGORY["fn"]), (lay["fp"], CATEGORY["fp"])])


def composite(canvas, layer):
    """The full-size picture of a photograph's grey underlay with a categorical layer over it."""
    out = underlay(canvas)
    rgb, where = layer
    out[where] = rgb[where]
    return out


def error_map(canvas, gt, pred):
    return composite(canvas, error_layer(gt, pred))


def confusion_map(canvas, gt, pred):
    return composite(canvas, confusion_layer(gt, pred))


def confused_pairs(gt, pred) -> list:
    """(ground-truth class, predicted class, pixels) of the wrong-disease-class pixels, most pixels first, ties to
    the lower (ground-truth class, predicted class) (ruling QUAL-A-1 Q5)."""
    import numpy as np
    lay = error_layers(gt, pred)["wrong"]
    if not lay.any():
        return []
    pairs, counts = np.unique(np.stack([gt[lay], pred[lay]], axis=1), axis=0, return_counts=True)
    out = [(int(a), int(b), int(n)) for (a, b), n in zip(pairs, counts)]
    return sorted(out, key=lambda t: (-t[2], t[0], t[1]))


# ------------------------------------------------------------------------------------------------ drawing
class Canvas:
    """A figure being drawn: a white page 1,950 px wide that grows downward. `placed` lists every panel drawn, as
    (left, top, width, height) on the page, for the smoke's wiring check (QUAL-AUD-1 T7 (3))."""

    def __init__(self, fonts):
        from PIL import Image, ImageDraw
        self.fonts = fonts
        self.img = Image.new("RGB", (WIDTH, 8000), FIG_BG)
        self.draw = ImageDraw.Draw(self.img)
        self.y = MARGIN
        self.placed = []

    def text_size(self, text, size):
        box = self.draw.textbbox((0, 0), text, font=self.fonts[size])
        return box[2] - box[0], box[3] - box[1]

    def text(self, x, y, text, size, *, anchor="la"):
        self.draw.text((x, y), text, fill=(0, 0, 0), font=self.fonts[size], anchor=anchor)

    def fits(self, text, size, width) -> bool:
        return self.text_size(text, size)[0] <= width

    def tile(self, arr, x, y, side, *, nearest=False, layer=None):
        """One panel: the array scaled into a side x side box, aspect kept, centred, with a thin border. A class map
        is scaled with nearest (`nearest`); a photograph with the box filter. `layer` = (rgb, where), at the array's
        own size, is a categorical layer over the photograph `arr`: it is scaled with nearest and painted where
        `where` holds, so that no blended colour appears (QUAL-AUD-1 N-10)."""
        import numpy as np
        from PIL import Image
        im = Image.fromarray(arr)
        w, h = im.size
        s = side / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.NEAREST if nearest else Image.BOX)
        if layer is not None:
            rgb, where = layer
            if rgb.shape[:2] != (h, w) or where.shape != (h, w):
                raise QualStop("layout", "a categorical layer is not the size of its photograph")
            top = Image.fromarray(np.ascontiguousarray(rgb)).resize((nw, nh), Image.NEAREST)
            sel = Image.fromarray(np.where(where, 255, 0).astype(np.uint8)).resize((nw, nh), Image.NEAREST)
            im.paste(top, (0, 0), sel)
        ox, oy = x + (side - nw) // 2, y + (side - nh) // 2
        self.img.paste(im, (ox, oy))
        self.draw.rectangle([ox - 1, oy - 1, ox + nw, oy + nh], outline=BORDER, width=1)
        self.placed.append((ox, oy, nw, nh))

    def headers(self, xs, side, labels, size=HEADER_PT):
        h = self.text_size("Ag", size)[1]
        for x, lab in zip(xs, labels):
            if not self.fits(lab, size, side):
                raise QualStop("layout", f"the header {lab!r} does not fit its column")
            self.text(x + side // 2, self.y + h, lab, size, anchor="ms")
        self.y += h + pt(3)

    def legend(self, entries, size=TEXT_PT):
        """A flowing key: (rgb, label) entries left to right, wrapped at the page width."""
        sw = pt(TEXT_PT)
        lh = max(sw, self.text_size("Ag", size)[1]) + pt(2)
        x = MARGIN
        self.y += pt(3)
        for rgb, label in entries:
            w = sw + pt(2) + self.text_size(label, size)[0] + pt(8)
            if x + w > WIDTH - MARGIN and x > MARGIN:
                x, self.y = MARGIN, self.y + lh
            self.draw.rectangle([x, self.y, x + sw, self.y + sw], fill=tuple(rgb), outline=(0, 0, 0), width=1)
            self.text(x + sw + pt(2), self.y + sw // 2, label, size, anchor="lm")
            x += w
        self.y += lh

    def png(self) -> bytes:
        im = self.img.crop((0, 0, WIDTH, self.y + MARGIN))
        buf = io.BytesIO()
        im.save(buf, format="PNG", dpi=(DPI, DPI))
        return buf.getvalue()


def columns(n: int, left: int = 0) -> tuple:
    side = (WIDTH - 2 * MARGIN - left - (n - 1) * GAP) // n
    xs = [MARGIN + left + i * (side + GAP) for i in range(n)]
    return xs, side


def name_of(names, c: int) -> str:
    return names[c].replace("_", " ")


# ------------------------------------------------------------------------------------------------ figures
BACKGROUND_KEY = (BACKGROUND, "Background")


def class_legend(cv, classes, lut, names, padding: bool):
    """Every class the part shows, by the dataset's own names, black first ("Background": QUAL-AUD-1 F-08); the
    padding key only when a panel shows padding."""
    cv.legend([BACKGROUND_KEY] + [(lut[c], name_of(names, c)) for c in sorted(classes)]
              + ([PAD_KEY] if padding else []))


def map_legend(cv, keys, texts, padding: bool):
    """The key of error maps or overlays: the categories, the grey image, the padding only when a panel shows it."""
    cv.legend([(CATEGORY[k], texts[k]) for k in keys] + [GREY_KEY] + ([PAD_KEY] if padding else []))


def has_pad(*masks) -> bool:
    return any(bool((m == 255).any()) for m in masks)


def value_lines(cv, v, split: str, width: int) -> list:
    """Ruling QUAL-A-1 Q8: the per-image value of record from the official artifact (never the recomputed one), as a
    percentage with two decimals, on one line or on two (split after the colon)."""
    if v is None:
        if split == "test":
            raise QualStop("value_missing", "a Figure 4.2 prediction has no per-image value of record")
        return ["Per-image disease-only mIoU:", "no VAL record"]
    one = f"Per-image disease-only mIoU: {100.0 * v:.2f}%"
    lines = [one] if cv.fits(one, TEXT_PT, width) else ["Per-image disease-only mIoU:", f"{100.0 * v:.2f}%"]
    if not all(cv.fits(t, TEXT_PT, width) for t in lines):
        raise QualStop("layout", "a value label does not fit its column")
    return lines


def classes_in(*arrays) -> set:
    import numpy as np
    out = set()
    for a in arrays:
        out |= set(int(v) for v in np.unique(a))
    return out


def fig4_2(inp, cv, lut, present, names):
    p = P50_POSITION
    xs, side = columns(4)
    rgb = inp.array(f"arrays/original/{p}_image.png")
    omask = inp.array(f"arrays/original/{p}_mask.png")
    canvas, gt = inp.canvas(CLEAN, p), inp.mask(p)
    cv.headers(xs, side, ["Image", "Mask", "Canvas input", "E6 error map"])
    cv.tile(rgb, xs[0], cv.y, side)
    cv.tile(colourise(omask, lut, present), xs[1], cv.y, side, nearest=True)
    cv.tile(canvas, xs[2], cv.y, side)
    cv.tile(underlay(canvas), xs[3], cv.y, side, layer=error_layer(gt, inp.pred("E6", CLEAN, p)))
    cv.y += side + GAP
    cv.headers(xs, side, [MODELS[m]["header"] for m in MODEL_ORDER])
    classes = classes_in(omask)                        # the classes shown in class colours: the mask as read and
    for x, m in zip(xs, MODEL_ORDER):                  # the four predictions
        pr = inp.pred(m, CLEAN, p)
        classes |= classes_in(pr)
        cv.tile(colourise(pr, lut, present), x, cv.y, side, nearest=True)
    cv.y += side + pt(2)
    h = cv.text_size("Ag", TEXT_PT)[1] + pt(1)
    labels = {x: value_lines(cv, inp.value(m, CLEAN, p), inp.split, side) for x, m in zip(xs, MODEL_ORDER)}
    for x, lines in labels.items():
        for j, t in enumerate(lines):
            cv.text(x + side // 2, cv.y + (j + 1) * h, t, TEXT_PT, anchor="ms")
    cv.y += max(len(v) for v in labels.values()) * h + pt(4)
    class_legend(cv, classes - {0, 255}, lut, names, padding=has_pad(omask))
    map_legend(cv, ("tp", "fn", "fp"), {k: "E6 error map: " + CATEGORY_TEXT[k].lower() for k in CATEGORY_TEXT},
               padding=has_pad(gt))
    return cv.png()


def fig4_3(inp, cv, lut, present, names, positions):
    xs, side = columns(6)
    cv.headers(xs, side, ["Input", "Ground truth"] + [MODELS[m]["header"] for m in MODEL_ORDER])
    classes, gts = set(), []
    for i, p in enumerate(positions):
        gt = inp.mask(p)
        gts.append(gt)
        maps = [inp.canvas(CLEAN, p), colourise(gt, lut, present)]
        classes |= classes_in(gt)
        for m in MODEL_ORDER:
            pr = inp.pred(m, CLEAN, p)
            classes |= classes_in(pr)
            maps.append(colourise(pr, lut, present))
        for j, (x, a) in enumerate(zip(xs, maps)):
            cv.tile(a, x, cv.y, side, nearest=j > 0)
        cv.y += side + (GAP if i < len(positions) - 1 else pt(2))
    class_legend(cv, classes - {0, 255}, lut, names, padding=has_pad(*gts))
    return cv.png()


def fig4_4(inp, cv, positions):
    xs, side = columns(5)
    cv.headers(xs, side, ["Input"] + [MODELS[m]["header"] for m in MODEL_ORDER])
    gts = []
    for i, p in enumerate(positions):
        canvas, gt = inp.canvas(CLEAN, p), inp.mask(p)
        gts.append(gt)
        cv.tile(canvas, xs[0], cv.y, side)
        for x, m in zip(xs[1:], MODEL_ORDER):
            cv.tile(underlay(canvas), x, cv.y, side, layer=error_layer(gt, inp.pred(m, CLEAN, p)))
        cv.y += side + (GAP if i < len(positions) - 1 else pt(2))
    map_legend(cv, ("tp", "fn", "fp"), CATEGORY_TEXT, padding=has_pad(*gts))
    return cv.png()


def fig4_5(inp, cv, lut, present, names, pos):
    from PIL import Image, ImageDraw
    label_w = cv.text_size("Ag", TEXT_PT)[1] + pt(6)
    xs, side = columns(5, left=label_w)
    cv.headers(xs, side, ["Input", "Ground truth"] + [MODELS[m]["header"] for m in ("E1", "E3", "E6")])
    gt = inp.mask(pos)
    classes = classes_in(gt)
    for i, cond in enumerate((CLEAN, *CELLS)):
        lab = ROW_LABELS[cond]
        box = cv.draw.textbbox((0, 0), lab, font=cv.fonts[TEXT_PT])
        if box[2] - box[0] > side:
            raise QualStop("layout", f"the row label {lab!r} does not fit")
        strip = Image.new("RGB", (box[2] + 4, box[3] + 4), FIG_BG)
        ImageDraw.Draw(strip).text((2, 2), lab, fill=(0, 0, 0), font=cv.fonts[TEXT_PT])
        strip = strip.rotate(90, expand=True)
        cv.img.paste(strip, (MARGIN, cv.y + (side - strip.size[1]) // 2))
        maps = [inp.canvas(cond, pos), colourise(gt, lut, present)]
        for m in ("E1", "E3", "E6"):
            pr = inp.pred(m, cond, pos)
            classes |= classes_in(pr)
            maps.append(colourise(pr, lut, present))
        for j, (x, a) in enumerate(zip(xs, maps)):
            cv.tile(a, x, cv.y, side, nearest=j > 0)
        cv.y += side + (GAP if i < len(CELLS) else pt(2))
    class_legend(cv, classes - {0, 255}, lut, names, padding=has_pad(gt))
    return cv.png()


MAX_PAIRS = 3     # ruling QUAL-A-1 Q5: up to three pairs a panel by pixel count, then FURTHER_PAIRS
# Ruling QUAL-AUD-1 F-05 (AM-27 item 4: "The only numbers a figure prints are per-image values of record"): the
# legend says that further pairs exist, without their number.
FURTHER_PAIRS = "and further pairs"


def wrap(cv, text, width) -> list:
    """`text` on as few lines as fit `width`, broken at spaces only (ruling QUAL-AUD-1 F-07: never cut, no
    ellipsis); a single word wider than the column stops the step (layout)."""
    lines, cur = [], ""
    for word in text.split(" "):
        cand = word if not cur else f"{cur} {word}"
        if cv.fits(cand, TEXT_PT, width):
            cur = cand
            continue
        if not cv.fits(word, TEXT_PT, width):
            raise QualStop("layout", "a class name holds a word wider than its column")
        lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    return lines


def pair_lines(cv, names, pairs, width) -> list:
    """The panel's legend lines: each pair "ground truth → predicted" on one line, or, when it does not fit, the
    ground truth and the arrow, then the prediction, each wrapped at a space (F-07); then FURTHER_PAIRS (F-05)."""
    out = []
    for a, b, _ in pairs[:MAX_PAIRS]:
        one = f"{name_of(names, a)} → {name_of(names, b)}"
        if cv.fits(one, TEXT_PT, width):
            out.append(one)
        else:
            out += wrap(cv, f"{name_of(names, a)} →", width) + wrap(cv, name_of(names, b), width)
    if len(pairs) > MAX_PAIRS:
        out.append(FURTHER_PAIRS)
    return out or ["No wrong-class pixels"]


def fig4_6(inp, cv, names, positions):
    xs, side = columns(5)
    cv.headers(xs, side, ["Input"] + [MODELS[m]["header"] for m in MODEL_ORDER])
    h = cv.text_size("Ag", TEXT_PT)[1] + pt(1)
    gts = []
    for i, p in enumerate(positions):
        canvas, gt = inp.canvas(CLEAN, p), inp.mask(p)
        gts.append(gt)
        cv.tile(canvas, xs[0], cv.y, side)
        lines = {}
        for x, m in zip(xs[1:], MODEL_ORDER):
            pr = inp.pred(m, CLEAN, p)
            cv.tile(underlay(canvas), x, cv.y, side, layer=confusion_layer(gt, pr))
            lines[x] = pair_lines(cv, names, confused_pairs(gt, pr), side)
        cv.y += side + pt(2)
        n = max(len(v) for v in lines.values())
        for x, text in lines.items():
            for j, t in enumerate(text):
                cv.text(x + side // 2, cv.y + (j + 1) * h, t, TEXT_PT, anchor="ms")
        cv.y += n * h + (GAP if i < len(positions) - 1 else pt(2))
    map_legend(cv, ("tp", "wrong", "fn", "fp"), CONFUSION_TEXT, padding=has_pad(*gts))
    return cv.png()


# ---------------------------------------------------------------------------------------------------- run
def load_fonts(path, synthetic: bool):
    from PIL import ImageFont
    p = Path(path)
    sizes = {s: ImageFont.truetype(str(p), pt(s)) for s in (HEADER_PT, TEXT_PT, TEXT_PT - 1)}
    family, style = sizes[TEXT_PT].getname()
    if family not in FONT_FAMILIES and not synthetic:
        raise QualStop("font", f"the font is {family}, not one of {FONT_FAMILIES}")
    return sizes, {"family": family, "style": style, "sha256": sha256_file(p), "name": p.name}


def present_classes(inp) -> list:
    import numpy as np
    seen = set()
    for p in range(1, K_ALL + 1):
        seen |= set(int(v) for v in np.unique(inp.mask(p)))
        for m in MODEL_ORDER:
            seen |= set(int(v) for v in np.unique(inp.pred(m, CLEAN, p)))
    for p in range(1, K_CELLS + 1):
        for m in ("E1", "E3", "E6"):
            for c in CELLS:
                seen |= set(int(v) for v in np.unique(inp.pred(m, c, p)))
    seen |= set(int(v) for v in np.unique(inp.array(f"arrays/original/{P50_POSITION}_mask.png")))
    bad = seen - set(range(0, 116)) - {255}
    if bad:
        raise QualStop("class_range", "a map holds a value outside 0..115 and 255")
    return sorted(seen - {0, 255})


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=("test", "val"))
    ap.add_argument("--select-sha256", required=True)
    ap.add_argument("--font", required=True, help="Arial or Liberation Sans, regular (a .ttf file)")
    ap.add_argument("--out", default="figures", help="the folder under the run for this rendering (never reused)")
    ap.add_argument("--synthetic", action="store_true", help="smoke_qual only: any font, synthetic class names")
    ap.add_argument("--expect-commit", help="the commit this step runs from (TEST: required; the audited commit, "
                                            "or for a re-render the render commit)")
    ap.add_argument("--rerender", help="a re-render from the saved arrays (ruling QUAL-A-1 Q10): its reason; on TEST "
                                       "the decision-log row (DL-n) that states the layout fault")
    ap.add_argument("--predict-commit", help="TEST re-render: the audited commit the select and predict steps ran from")
    return ap


def render_only_change(repo: Path, base: str, head: str) -> list:
    """Ruling QUAL-A-1 Q10: the paths that differ between the predict commit `base` and the render commit `head`, by
    git in the checkout; `base` must be an ancestor of `head`."""
    import subprocess
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", base, head], env=env,
                         capture_output=True, text=True)
    if anc.returncode != 0:
        raise QualStop("rerender_refused", "the predict commit is not an ancestor of the render commit",
                       private=anc.stderr)
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", base, head], env=env, capture_output=True,
                          text=True)
    if diff.returncode != 0:
        raise QualStop("rerender_refused", "git could not compare the two commits", private=diff.stderr)
    return sorted(x for x in diff.stdout.splitlines() if x)


def rerender_check(args, repo: dict, repos: dict):
    """One checkout for every step (a first rendering), or a re-render (Q10): the select and predict steps from one
    commit, the render step from a later one that differs from it in scripts/qual_render.py alone; on TEST also a
    decision-log row and the audited commit named. Returns the re-render's record, or None."""
    commits = {(r or {}).get("commit") for r in repos.values()}
    if len(commits) != 1 or None in commits:
        raise QualStop("repo_mismatch", "the select and predict steps did not all run from one checkout")
    base = next(iter(commits))
    if base == repo["commit"]:
        return {"reason": args.rerender, "predict_commit": base, "changed": []} if args.rerender else None
    if not args.rerender:
        raise QualStop("repo_mismatch", "the select, predict and render steps did not all run from this checkout")
    if args.split == "test":
        if not re.fullmatch(r"DL-[0-9]+", args.rerender):
            raise QualStop("rerender_refused", "on TEST a re-render names the decision-log row of its fault (DL-n)")
        if args.predict_commit != base:
            raise QualStop("rerender_refused", "the select and predict steps did not run from the audited commit named")
    changed = render_only_change(REPO, base, repo["commit"])
    if changed != ["scripts/qual_render.py"]:
        raise QualStop("rerender_refused", f"the render commit changes {len(changed)} paths; only "
                                           "scripts/qual_render.py may differ from the predict commit")
    say("PASS", f"re-render ({args.rerender}): the render commit {repo['commit'][:12]}... differs from the predict "
                f"commit {base[:12]}... in scripts/qual_render.py alone")
    return {"reason": args.rerender, "predict_commit": base, "changed": changed}


def main(argv=None, *, class_names=None) -> int:
    args = build_parser().parse_args(argv)
    return run_step("render", args, lambda a: _main(a, class_names))


def _main(args, class_names) -> int:
    import platform
    import numpy as np
    import PIL
    run = run_dir(args.run)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", args.out) or args.out in (".", ".."):
        raise QualStop("cli", f"--out {args.out!r} is not a plain folder name")
    out = no_test(run / args.out)
    manifest_rel = f"manifest_{args.out}.json"                  # QUAL-AUD-1 N-06: outside the figures folder
    if out.exists() or (run / manifest_rel).exists():           # N-03: before any read
        raise QualStop("output_exists", f"the folder {args.out} or {manifest_rel} exists already; a rendering is "
                                        "never overwritten")
    if args.synthetic and args.split == "test":
        raise QualStop("cli", "--synthetic is for smoke_qual only")
    repo = repo_state(expected_commit(args))
    inp = Inputs(run, args.split, args.select_sha256)
    rerender = rerender_check(args, repo, inp.repos)
    say("PASS", f"inputs: S.json {inp.s_sha[:12]}..., 4 predict records, {len(inp.files)} files re-hashed; "
                f"{inp.n_tiles} tiles checked" + (", every one passed" if args.split == "test" else "")
                + (f"; every step ran from {repo['commit'][:12]}..." if rerender is None
                   else f"; rendered again from {repo['commit'][:12]}..."))
    if class_names is None:
        from src.eval.adapters import load_class_map
        class_names = [e["name"] for e in load_class_map().entries]
    if not args.synthetic and len(class_names) != 116:
        raise QualStop("class_map", "the class map does not name 116 classes")
    if sha256_bytes(json.dumps(list(CLASS_COLOURS)).encode()) != CLASS_COLOURS_SHA256:
        raise QualStop("palette", "CLASS_COLOURS is not the measured list")
    fonts, font_info = load_fonts(args.font, args.synthetic)
    present = present_classes(inp)
    lut = class_lut(present)
    pset = set(present)
    figs = {
        "fig4_2.png": fig4_2(inp, Canvas(fonts), lut, pset, class_names),
        "fig4_3a.png": fig4_3(inp, Canvas(fonts), lut, pset, class_names, [1, 2, 3]),
        "fig4_3b.png": fig4_3(inp, Canvas(fonts), lut, pset, class_names, [4, 5, 6]),
        "fig4_3c.png": fig4_3(inp, Canvas(fonts), lut, pset, class_names, [7, 8, 9]),
        "fig4_4a.png": fig4_4(inp, Canvas(fonts), [1, 2, 3]),
        "fig4_4b.png": fig4_4(inp, Canvas(fonts), [4, 5, 6]),
        "fig4_4c.png": fig4_4(inp, Canvas(fonts), [7, 8, 9]),
        "fig4_5a.png": fig4_5(inp, Canvas(fonts), lut, pset, class_names, 1),
        "fig4_5b.png": fig4_5(inp, Canvas(fonts), lut, pset, class_names, 2),
        "fig4_5c.png": fig4_5(inp, Canvas(fonts), lut, pset, class_names, 3),
        "fig4_6.png": fig4_6(inp, Canvas(fonts), class_names, [1, 2, 3]),
    }
    digests = {name: write_new(out / name, data) for name, data in figs.items()}
    manifest = {
        "format": "qual1-manifest/1", "split": args.split, "out": args.out, "repo": repo, "step_repos": inp.repos,
        "rerender": rerender, "s": inp.s, "select_sha256": inp.s_sha,
        "select_record": {"sha256": sha256_bytes(inp.select_record_bytes), "inputs": inp.select_record["inputs"],
                          "tie_readings": inp.select_record.get("tie_readings")},
        "rules": {"ruling": "QUAL-A-1, QUAL-AUD-1", "tie_reading": TIE_READING, "error_map": ERROR_MAP_RULE,
                  "palette": PALETTE_RULE, "max_pairs": MAX_PAIRS, "pair_ties": "lower (ground truth, predicted)",
                  "further_pairs": FURTHER_PAIRS, "names": "wrapped at a space, never cut (F-07)",
                  "legend_background": BACKGROUND_KEY[1], "scaling": "class maps and categorical layers nearest; "
                  "photographs box (N-10)",
                  "value_label": "Per-image disease-only mIoU: xx.xx% (the official value of record)",
                  "underlay_min": UNDERLAY_MIN},
        "versions": {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": np.__version__},
        "font": font_info, "dpi": DPI, "width_px": WIDTH,
        "palette": {str(c): CLASS_COLOURS[i] for i, c in enumerate(present)},
        "predict_inputs": {m: inp.records[m]["inputs"] for m in MODEL_ORDER},
        # QUAL-AUD-1 F-06 (AM-27 item 6): per model and condition, the record's files_sha256 and, per position, the
        # sha256 of the image file, the mask file and the cached copy read; and the select step's thirteen records
        "predict_conditions": {m: inp.records[m].get("conditions") for m in MODEL_ORDER},
        "select_records": inp.select_record.get("records"),
        "select_preflight": inp.select_record.get("preflight"),
        "predict_versions": {m: inp.records[m]["versions"] for m in MODEL_ORDER},
        "nmf": inp.records["teacher"].get("nmf"),
        "files": inp.files, "figures": digests,
        "checks": {m: inp.checks[m] for m in MODEL_ORDER}, "tiles": inp.n_tiles, "largest_gap": inp.largest,
    }
    man_sha = write_new(run / manifest_rel, canonical_bytes(manifest))
    for name, d in digests.items():
        say("FIGURE", f"{name} sha256 {d}")
    say("RESULT:", f"QUAL RENDER OK: {len(digests)} figure files; {inp.n_tiles} tiles; largest gap {inp.largest!r}; "
                   f"{manifest_rel} sha256 {man_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
