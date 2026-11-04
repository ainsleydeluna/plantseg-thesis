"""Self-made synthetic images (no dataset): a drawn leaf with lesion-like spots on a textured ground.

Used for the stand-in's calibration forwards, the smokes, the parity checks and the screenshots.
Deterministic for a given (seed, size).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# (width, height): landscape, portrait, square and phone-like aspect ratios.
SIZES = ((640, 480), (480, 640), (800, 800), (1024, 768), (720, 1280), (1200, 900), (900, 600),
         (600, 900))


def make_leaf_image(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    rng = np.random.RandomState(seed)
    w, h = size
    # ground: soil-to-grass gradient plus noise
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array([96, 82, 58], np.float32)
    bottom = np.array([70, 104, 52], np.float32)
    ground = top * (1 - yy) + bottom * yy
    ground = np.broadcast_to(ground, (h, w, 3)).copy()
    ground += rng.normal(0, 14, size=(h, w, 3)).astype(np.float32)
    img = Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(1.2))
    draw = ImageDraw.Draw(img)
    # leaf: rotated ellipse polygon with a midrib and veins
    cx, cy = w * (0.45 + 0.1 * rng.rand()), h * (0.45 + 0.1 * rng.rand())
    a, b = min(w, h) * (0.38 + 0.06 * rng.rand()), min(w, h) * (0.20 + 0.05 * rng.rand())
    ang = rng.uniform(-0.8, 0.8)
    pts = []
    for t in np.linspace(0, 2 * math.pi, 120, endpoint=False):
        x, y = a * math.cos(t), b * math.sin(t) * (1 - 0.25 * math.cos(t))
        pts.append((cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang)))
    leaf_rgb = tuple(int(v) for v in (rng.randint(40, 80), rng.randint(120, 170), rng.randint(40, 70)))
    draw.polygon(pts, fill=leaf_rgb)
    tip = (cx + a * math.cos(ang), cy + a * math.sin(ang))
    base = (cx - a * math.cos(ang), cy - a * math.sin(ang))
    draw.line([base, tip], fill=(170, 200, 120), width=max(2, int(min(w, h) / 160)))
    for k in range(1, 7):
        f = k / 7.0
        px, py = base[0] + (tip[0] - base[0]) * f, base[1] + (tip[1] - base[1]) * f
        for sgn in (-1, 1):
            vx = px + 0.5 * b * math.cos(ang + sgn * 1.0) * 1.4
            vy = py + 0.5 * b * math.sin(ang + sgn * 1.0) * 1.4
            draw.line([(px, py), (vx, vy)], fill=(150, 190, 110), width=1)
    # lesions: brown/yellow spots with darker rims, inside the leaf's ellipse
    for _ in range(rng.randint(5, 12)):
        r = min(w, h) * rng.uniform(0.015, 0.05)
        t = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.0, 0.75)
        x, y = a * rr * math.cos(t), b * rr * math.sin(t)
        lx, ly = cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang)
        rim = (int(rng.randint(60, 90)), int(rng.randint(40, 60)), int(rng.randint(20, 35)))
        core = (int(rng.randint(150, 200)), int(rng.randint(110, 150)), int(rng.randint(40, 70)))
        draw.ellipse([lx - r * 1.25, ly - r * 1.25, lx + r * 1.25, ly + r * 1.25], fill=rim)
        draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=core)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def calibration_images(n: int, seed: int) -> list[Image.Image]:
    return [make_leaf_image(seed + i, SIZES[i % len(SIZES)]) for i in range(n)]
