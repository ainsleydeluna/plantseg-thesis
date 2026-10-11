"""R2's TRAIN-data components (lane L-TEACHER-ARMS, commit C3; AM-18 item 2(b); ARMS plan decision (ii); DL-24
audit F13 to F18; Phase B-2 ruling R6).

Loaded only by configs/teacher/arms/r2_bema_rare.py through `custom_imports`, so it runs only where
MMSegmentation is installed. teacher_components.py and src/data/transforms.py are frozen and stay unchanged:
the record transform's pieces are imported, never copied. It registers:

* `RareClassTrainDataset` (mmseg DATASETS): the record's TRAIN dataset (`BaseSegDataset`, the same listing,
  order and keys) plus `table_path`, the R2 table (scripts/build_r2_table.py, `plantseg-r2-table/1`).
  `load_data_list` reads and validates the table once and attaches two fields to each data_info:
  `rare_class` (the image's class when it is rare, else None) and `paste_sources` (the other images of that
  class, sorted stems; [] when not rare). `get_cat_ids` reads `rare_class` only: [c] or []. Background is
  never a category, so an image with no disease pixel, or of a class that is not rare, has repeat factor 1
  under ClassBalancedDataset; a rare class c gets ceil(sqrt(t / (n_c / N))) = m_c (the launcher asserts
  the per-index counts and the length against the table before train()).
* `RareClassCopyPasteTrainTransform` (mmseg TRANSFORMS): the record transform's per-sample RandomState
  (teacher_components.py:118), then same-class copy-paste for a rare T on the EXIF-corrected raw image and
  mask, then the record's `train_preprocess(..., rng, AUGMENT)` and canvas packing. It reads only the two
  data_info fields: no table I/O and no environment lookup. A T that is not rare takes no extra draw, so its
  output is bit-identical to `ThesisTeacherTrainTransform`'s.

The paste (`copy_paste`), per ruling R6:
  frame      once a source is drawn, T's paste frame is the EXIF-corrected image; a mask of another size is
             first resized NEAREST to the image's size, as `_resize_long_side` does to it anyway (F17); the same
             for S; counted. When no pixel is pasted the opened files go to train_preprocess unchanged.
  draws      all on the sample's RandomState, before M2/M3, in this order: the decision rng.rand() < 0.5
             (drawn for every rare T, even when its class has no other image); the source
             pool[rng.randint(len(pool))] (pool = paste_sources; empty: nothing more is drawn); u =
             rng.uniform(0.75, 1.25); up to 10 placements, each y = rng.randint(0, H_T - h + 1) then
             x = rng.randint(0, W_T - w + 1), the first accepted wins.
  patch      the bounding box of S's pixels labelled c, scaled by f = max(H_T, W_T) / max(H_S, W_S) * u;
             sizes exact: Fraction of the integer sizes and Fraction(u), Python round() (half to even, the
             repository's convention, transforms.py:35, :108), then max(1, .); when a size exceeds T,
             g = min(H_T / h', W_T / w') in Fraction, nearest rounding, max(1, .) and min(H_T, .) / min(W_T, .)
             clamps. One resize of the crop to the final size: BILINEAR image, NEAREST mask; the patch is
             the resized mask's pixels equal to c.
  accept     20 * overlap <= n_patch, in integers (overlap: patch pixels on T pixels labelled 1..115). An
             empty patch draws one placement, which is accepted (20 * 0 <= 0), and pastes nothing: no
             special case.
  overwrite  patch pixels overwrite T's image and set T's label to c; no blending, one class per image.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

# Imported-code attestation (the teacher_components pattern): the bytes this import resolved to.
_MODULE_PATH = Path(__file__).resolve()
ARM_COMPONENTS_PROVENANCE = {"path": str(_MODULE_PATH),
                             "sha256": hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()}

import numpy as np  # noqa: E402
from mmcv.transforms import BaseTransform  # noqa: E402
from mmseg.datasets import BaseSegDataset  # noqa: E402
from mmseg.registry import DATASETS, TRANSFORMS  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from configs.augment import AUGMENT  # noqa: E402
from src.data.transforms import train_preprocess  # noqa: E402
from src.eval.teacher_diag import refuse_test_path  # noqa: E402
from src.training.teacher_components import _pack_canvas  # noqa: E402

# ------------------------------------------------------------------------------------------ values of record
TABLE_FORMAT = "plantseg-r2-table/1"
STOP_FORMAT = "plantseg-r2-table-stop/1"      # scripts/build_r2_table.py's r2_table_STOP.json: never a table
TABLE_KEYS = ("builder", "format", "per_class", "per_image", "sources", "split_list_sha256", "strata", "t",
              "totals")
T_NUM, T_DEN = 20, 5367              # AM-18 item 2(b): t = 20 / 5,367 (the TRAIN image count)
DISEASE_IDS = range(1, 116)          # class 0 is background, never a category; 255 is ignore
PASTE_P = 0.5                        # the paste decision
SCALE_RANGE = (0.75, 1.25)           # u
MAX_TRIES = 10                       # placements
OVERLAP_DEN = 20                     # accept iff overlap <= 5% of the patch: 20 * overlap <= n_patch
IMG_SUFFIX, MASK_SUFFIX = ".jpg", ".png"
COUNT_KEYS = ("rare", "decision", "empty_pool", "accepted", "pasted", "empty_patch", "no_placement",
              "frame_resized_t", "frame_resized_s", "pasted_with_resized_frame")


class R2TableError(ValueError):
    """The R2 table is not a valid plantseg-r2-table/1 document, or does not match the TRAIN listing."""


def m_of(n_c: int) -> int | None:
    """m_c = ceil(sqrt(20 / n_c)) in exact integers: the least m with m * m * n_c >= 20 (1 when n_c >= 20);
    None for a class with no TRAIN image (it is never a category)."""
    if n_c < 1:
        return None
    m = 1
    while m * m * n_c < T_NUM:
        m += 1
    return m


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_table(doc) -> dict:
    """The R2 table's internal rules; returns the lookups the dataset and the launcher read.

    Every class 1..115 once, in order, with n_c = its count in per_image, rare = n_c < 20 and m_c = m_of(n_c);
    per_image in stem order, unique stems, class_id null or 1..115, m = the class's m_c when rare, else 1;
    sources = exactly the rare classes with images, each the sorted stems of that class; totals = the image
    count and the sum of m. Raises R2TableError on the first violation."""
    def need(ok, msg):
        if not ok:
            raise R2TableError(msg)

    need(isinstance(doc, dict) and tuple(sorted(doc)) == TABLE_KEYS,
         f"the table's keys are {list(TABLE_KEYS)}, got {sorted(doc) if isinstance(doc, dict) else type(doc)}")
    need(doc["format"] == TABLE_FORMAT, f"format {doc['format']!r} is not {TABLE_FORMAT!r} (a STOP file has no "
                                        "table)")
    need(doc["t"] == {"num": T_NUM, "den": T_DEN} and all(_is_int(v) for v in doc["t"].values()),
         f"t {doc['t']!r} is not {{num {T_NUM}, den {T_DEN}}}")
    pc = doc["per_class"]
    need(isinstance(pc, list) and len(pc) == len(DISEASE_IDS), "per_class must list the 115 disease classes")
    images = doc["per_image"]
    need(isinstance(images, list) and images, "per_image must be a non-empty list")
    by_stem, m_by_stem, counts = {}, {}, dict.fromkeys(DISEASE_IDS, 0)
    for row in images:
        need(isinstance(row, dict) and sorted(row) == ["class_id", "m", "stem"], f"per_image row {row!r}")
        stem, c, m = row["stem"], row["class_id"], row["m"]
        need(isinstance(stem, str) and stem and "/" not in stem and stem not in by_stem,
             f"per_image stem {stem!r}: empty, a path or repeated")
        need(c is None or (_is_int(c) and c in DISEASE_IDS), f"{stem}: class_id {c!r} is not null or 1..115")
        need(_is_int(m) and m >= 1, f"{stem}: m {m!r}")
        by_stem[stem], m_by_stem[stem] = c, m
        if c is not None:
            counts[c] += 1
    stems = list(by_stem)
    need(stems == sorted(stems), "per_image is not in stem order")
    rare = {}
    for want_id, row in zip(DISEASE_IDS, pc):
        need(isinstance(row, dict) and sorted(row) == ["class_id", "m_c", "n_c", "rare"], f"per_class row {row!r}")
        need(row["class_id"] == want_id and _is_int(row["class_id"]), f"per_class row {row!r} out of order")
        n_c = row["n_c"]
        need(_is_int(n_c) and n_c == counts[want_id], f"class {want_id}: n_c {n_c!r} != its per_image count "
                                                       f"{counts[want_id]}")
        need(row["rare"] is (n_c < T_NUM), f"class {want_id}: rare {row['rare']!r} with n_c {n_c}")
        need(row["m_c"] == m_of(n_c) and (row["m_c"] is None or _is_int(row["m_c"])),
             f"class {want_id}: m_c {row['m_c']!r} != {m_of(n_c)!r}")
        if row["rare"] and n_c >= 1:
            rare[want_id] = row["m_c"]
    for stem, c in by_stem.items():
        want = rare.get(c, 1) if c is not None else 1
        need(m_by_stem[stem] == want, f"{stem}: m {m_by_stem[stem]} != {want} (class {c})")
    src = doc["sources"]
    need(isinstance(src, dict) and sorted(src) == sorted(str(c) for c in rare),
         f"sources must name exactly the rare classes with images: {sorted(rare)}")
    sources = {}
    for c in rare:
        want = sorted(s for s, k in by_stem.items() if k == c)
        need(src[str(c)] == want, f"sources[{c}] is not the sorted stems of class {c}")
        sources[c] = want
    tot = doc["totals"]
    need(tot == {"images": len(stems), "repeated": sum(m_by_stem.values())},
         f"totals {tot!r} != {{images {len(stems)}, repeated {sum(m_by_stem.values())}}}")
    st = doc["strata"]
    need(isinstance(st, dict) and sorted(st) == ["check", "path", "sha256"] and st["check"] == "PASS",
         f"strata {st!r}: needs {{path, sha256, check PASS}}")
    need(isinstance(doc["split_list_sha256"], str) and len(doc["split_list_sha256"]) == 64,
         "split_list_sha256 must be a sha256")
    need(isinstance(doc["builder"], dict) and sorted(doc["builder"]) == ["commit", "script_sha256"],
         "builder needs {script_sha256, commit}")
    return {"by_stem": by_stem, "m_by_stem": m_by_stem, "rare": rare, "sources": sources,
            "n_c": {row["class_id"]: row["n_c"] for row in pc}, "images": len(stems),
            "repeated": tot["repeated"]}


def read_table(path) -> tuple[dict, dict, str]:
    """(document, lookups, sha256) of the table file at `path`, refused by name when it is not valid."""
    refuse_test_path(path, "R2 table")
    data = Path(path).read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise R2TableError(f"{Path(path).name}: not JSON ({e})") from e
    return doc, validate_table(doc), hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------ dataset
@DATASETS.register_module()
class RareClassTrainDataset(BaseSegDataset):
    """The record's TRAIN dataset plus the R2 table: see the module docstring."""

    def __init__(self, table_path: str, **kwargs):
        self.table_path = str(table_path)
        self.table_sha256 = None
        super().__init__(**kwargs)

    def load_data_list(self) -> list:
        if (self.img_suffix, self.seg_map_suffix) != (IMG_SUFFIX, MASK_SUFFIX):
            raise R2TableError(f"R2 pairs <stem>{IMG_SUFFIX} with <stem>{MASK_SUFFIX} (the record dataset's "
                               f"suffixes), got {self.img_suffix!r} and {self.seg_map_suffix!r}")
        for key in ("img_path", "seg_map_path"):
            refuse_test_path(self.data_prefix.get(key), f"R2 TRAIN {key}")
        _, table, self.table_sha256 = read_table(self.table_path)
        data_list = super().load_data_list()
        listed = [Path(d["img_path"]).name[:-len(IMG_SUFFIX)] for d in data_list]
        if sorted(listed) != sorted(table["by_stem"]):
            missing, extra = sorted(set(table["by_stem"]) - set(listed)), sorted(set(listed) - set(table["by_stem"]))
            raise R2TableError(f"the TRAIN listing ({len(listed)} images) and the R2 table ({table['images']}) "
                               f"differ: {len(missing)} in the table only, {len(extra)} listed only")
        for d, stem in zip(data_list, listed):
            c = table["by_stem"][stem]
            rare = c is not None and c in table["rare"]
            d["rare_class"] = c if rare else None
            d["paste_sources"] = [s for s in table["sources"][c] if s != stem] if rare else []
        return data_list

    def get_cat_ids(self, idx: int) -> list:
        c = self.get_data_info(idx)["rare_class"]
        return [] if c is None else [int(c)]


# ---------------------------------------------------------------------------------------------- the paste
def paste_frame(im: Image.Image, mk: Image.Image) -> tuple[Image.Image, Image.Image, bool]:
    """F17: the EXIF-corrected RGB image (as train_preprocess makes it) and the raw mask, resized NEAREST to
    the image's size when the two differ; the flag says whether it was."""
    img = ImageOps.exif_transpose(im).convert("RGB")
    resized = mk.size != img.size
    return img, (mk.resize(img.size, Image.NEAREST) if resized else mk), resized


def patch_size(h: int, w: int, t_hw: tuple, s_hw: tuple, u: float) -> tuple[int, int]:
    """The resized patch's (height, width), exactly (ruling R6): f = max(H_T, W_T) / max(H_S, W_S) * u, round()
    then max(1, .); a size over T's fits down by g = min(H_T / h', W_T / w') with nearest rounding and clamps."""
    (H_T, W_T), (H_S, W_S) = t_hw, s_hw
    f = Fraction(max(H_T, W_T), max(H_S, W_S)) * Fraction(u)
    hp, wp = max(1, round(h * f)), max(1, round(w * f))
    if hp > H_T or wp > W_T:
        g = min(Fraction(H_T, hp), Fraction(W_T, wp))
        hp, wp = min(H_T, max(1, round(hp * g))), min(W_T, max(1, round(wp * g)))
    return hp, wp


def _mask_array(mk: Image.Image) -> np.ndarray:
    a = np.array(mk)
    if a.ndim != 2 or a.dtype.kind not in "ui" or (a.size and (int(a.min()) < 0 or int(a.max()) > 255)):
        raise ValueError(f"R2 paste: a mask must be single-channel 0..255, got {a.shape} {a.dtype}")
    return a.astype(np.uint8)


def copy_paste(im: Image.Image, mk: Image.Image, c: int, pool: list, img_dir: str, ann_dir: str,
               rng: np.random.RandomState) -> tuple[Image.Image, Image.Image, dict]:
    """One rare T's paste (see the module docstring): (image, mask) for train_preprocess, and what happened.
    When no pixel is pasted (no decision, no other image, no accepted placement, an empty patch) the opened
    files go to train_preprocess unchanged, as in the record transform; only the draws differ."""
    info = {"decision": False, "source": None, "u": None, "size": None, "tries": 0, "accepted": False,
            "pasted_pixels": 0, "frame_resized_t": False, "frame_resized_s": False}
    if not rng.rand() < PASTE_P:                                       # 1. the decision
        return im, mk, info
    info["decision"] = True
    if not pool:                                                      # no other image of the class
        return im, mk, info
    stem = pool[rng.randint(len(pool))]                               # 2. the source
    u = float(rng.uniform(*SCALE_RANGE))                              # 3. u
    info.update(source=stem, u=u)
    s_paths = (str(Path(img_dir) / f"{stem}{IMG_SUFFIX}"), str(Path(ann_dir) / f"{stem}{MASK_SUFFIX}"))
    for p, what in zip(s_paths, ("R2 source image", "R2 source mask")):
        refuse_test_path(p, what)
    with Image.open(s_paths[0]) as s_im, Image.open(s_paths[1]) as s_mk:
        s_img, s_mask, info["frame_resized_s"] = paste_frame(s_im, s_mk)
        s_img.load()
        s_lab = _mask_array(s_mask)
    t_img, t_mask, info["frame_resized_t"] = paste_frame(im, mk)
    t_rgb, t_lab = np.array(t_img, dtype=np.uint8), _mask_array(t_mask)
    p_rgb, patch = make_patch(s_img, s_lab, c, t_lab.shape, u)
    info["size"] = list(patch.shape)
    info["tries"], at = place(t_lab, patch, rng)                      # 4. the placements
    if at is None:
        return im, mk, info
    n_patch = int(patch.sum())
    info.update(accepted=True, at=list(at), pasted_pixels=n_patch)
    if n_patch == 0:                                                  # an empty patch pastes nothing
        return im, mk, info
    overwrite(t_rgb, t_lab, p_rgb, patch, at, c)
    return Image.fromarray(t_rgb), Image.fromarray(t_lab), info


def make_patch(s_img: Image.Image, s_lab: np.ndarray, c: int, t_hw: tuple, u: float) -> tuple:
    """S's patch: the bounding box of its pixels labelled c, resized once to patch_size (BILINEAR image, NEAREST
    mask); (image, patch pixels), the patch pixels being the resized mask's pixels equal to c (possibly none)."""
    ys, xs = np.nonzero(s_lab == c)
    if ys.size:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:                                                             # c lost in S's frame: an empty patch
        y0 = y1 = x0 = x1 = 0
    hp, wp = patch_size(y1 - y0, x1 - x0, t_hw, s_lab.shape, u)
    if not ys.size:
        return np.zeros((hp, wp, 3), dtype=np.uint8), np.zeros((hp, wp), dtype=bool)
    box = (x0, y0, x1, y1)
    p_rgb = np.asarray(s_img.crop(box).resize((wp, hp), Image.BILINEAR), dtype=np.uint8)
    patch = np.asarray(Image.fromarray(s_lab).crop(box).resize((wp, hp), Image.NEAREST)) == c
    return p_rgb, patch


def place(t_lab: np.ndarray, patch: np.ndarray, rng) -> tuple:
    """Up to MAX_TRIES placements, each y = rng.randint(0, H_T - h + 1) then x = rng.randint(0, W_T - w + 1); the
    first with 20 * overlap <= n_patch (overlap: patch pixels on T pixels labelled 1..115) wins. An empty patch is
    accepted at its first placement. (tries, (y, x) or None)."""
    (H_T, W_T), (hp, wp) = t_lab.shape, patch.shape
    n_patch = int(patch.sum())
    for k in range(1, MAX_TRIES + 1):
        y = int(rng.randint(0, H_T - hp + 1))
        x = int(rng.randint(0, W_T - wp + 1))
        region = t_lab[y:y + hp, x:x + wp]
        overlap = int((patch & (region >= DISEASE_IDS.start) & (region < DISEASE_IDS.stop)).sum())
        if OVERLAP_DEN * overlap <= n_patch:
            return k, (y, x)
    return MAX_TRIES, None


def overwrite(t_rgb: np.ndarray, t_lab: np.ndarray, p_rgb: np.ndarray, patch: np.ndarray, at: tuple, c: int) -> None:
    """The patch pixels overwrite T's image and set T's label to c, in place; no blending."""
    y, x = at
    hp, wp = patch.shape
    t_rgb[y:y + hp, x:x + wp][patch] = p_rgb[patch]
    t_lab[y:y + hp, x:x + wp][patch] = c


@TRANSFORMS.register_module()
class RareClassCopyPasteTrainTransform(BaseTransform):
    """The record transform with R2's copy-paste before M2/M3 for a rare T (see the module docstring)."""

    def __init__(self):
        self.counts = dict.fromkeys(COUNT_KEYS, 0)
        self._noted = set()

    def transform(self, results: dict) -> dict:
        rng = np.random.RandomState(int(np.random.randint(0, 2 ** 31 - 1)))   # teacher_components.py:118
        c = results["rare_class"]
        with Image.open(results["img_path"]) as im, Image.open(results["seg_map_path"]) as mk:
            if c is None:
                img, mask = train_preprocess(im, mk, rng, AUGMENT)
            else:
                pool = list(results["paste_sources"])
                img_p, mask_p, info = copy_paste(im, mk, int(c), pool, str(Path(results["img_path"]).parent),
                                                 str(Path(results["seg_map_path"]).parent), rng)
                self._count(results, info)
                img, mask = train_preprocess(img_p, mask_p, rng, AUGMENT)
        return _pack_canvas(results, img, mask)

    def _count(self, results: dict, info: dict) -> None:
        """Per-process counts; the first pasted sample of a stem whose frame was resized is printed once (the
        EXIF-frame rule's evidence for the run log; F17)."""
        k = self.counts
        k["rare"] += 1
        k["decision"] += info["decision"]
        k["empty_pool"] += info["decision"] and info["source"] is None
        k["accepted"] += info["accepted"]
        k["pasted"] += info["accepted"] and info["pasted_pixels"] > 0
        k["empty_patch"] += info["accepted"] and info["pasted_pixels"] == 0
        k["no_placement"] += info["source"] is not None and not info["accepted"]
        k["frame_resized_t"] += info["frame_resized_t"]
        k["frame_resized_s"] += info["frame_resized_s"]
        resized = info["accepted"] and info["pasted_pixels"] > 0 and (info["frame_resized_t"]
                                                                       or info["frame_resized_s"])
        k["pasted_with_resized_frame"] += resized
        stem = Path(results["img_path"]).stem
        if resized and stem not in self._noted:
            self._noted.add(stem)
            print(f"[R2 paste] EXIF frame: {stem} <- {info['source']} pasted with a mask resized to its image's "
                  f"size (T {info['frame_resized_t']}, S {info['frame_resized_s']}; ruling R6, audit F17)",
                  flush=True)
