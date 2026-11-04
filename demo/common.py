"""Helpers shared by the demo server, the stand-in builder, the exports and the smokes.

Everything that has a repository definition comes from the repository: the class list from
configs/plantseg_class_map.json through src.eval.adapters.load_class_map (which verifies its pinned
semantic hash), the class count and the data licence from configs/data.py, file hashing from
src.eval.model_loading.sha256_file. The only demo-owned definition is the display palette.
"""
from __future__ import annotations

import colorsys
import sys
from dataclasses import dataclass
from pathlib import Path

from demo.config import REPO_ROOT

if str(REPO_ROOT) not in sys.path:          # python3 -I does not put the checkout on sys.path
    sys.path.insert(0, str(REPO_ROOT))


def stats_modules_loaded() -> list[str]:
    """Names of any src.stats modules in this process. The demo must never import src.stats."""
    return sorted(m for m in sys.modules if m == "src.stats" or m.startswith("src.stats."))


def assert_no_stats() -> None:
    loaded = stats_modules_loaded()
    if loaded:
        raise RuntimeError(f"src.stats was imported ({loaded}); the demo must never import it")


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    name: str           # verbatim from the class map ("" for class 0, the official empty name)
    role: str           # "background_or_non_disease" (class 0) or "disease"
    color: tuple[int, int, int]

    @property
    def label(self) -> str:
        """Display text. Class 0 has no name in the class map, so the page shows its role instead."""
        return self.name if self.name else "background / non-disease (class 0, unnamed)"


def palette(n: int) -> list[tuple[int, int, int]]:
    """Deterministic, well-spread display colours (golden-angle hues). Display only, never data."""
    cols = [(0, 0, 0)]                                   # class 0 is never painted on the overlay
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        s = 0.70 + 0.25 * ((i * 7) % 3) / 2.0
        v = 0.95 - 0.20 * ((i * 5) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    return cols


def load_classes() -> tuple[list[ClassInfo], dict]:
    """The repository's class list (116 entries, ids 0..115) plus its provenance."""
    from src.eval.adapters import load_class_map

    cm = load_class_map()
    cols = palette(len(cm.entries))
    classes = [ClassInfo(int(e["class_id"]), str(e["name"]), str(e["role"]), cols[i])
               for i, e in enumerate(cm.entries)]
    provenance = {"path": "configs/plantseg_class_map.json",
                  "semantic_sha256": cm.semantic_sha256,
                  "source_commit_sha": cm.source_commit_sha,
                  "immutable_source_url": cm.immutable_source_url,
                  "loader": "src.eval.adapters.load_class_map"}
    return classes, provenance


def data_facts() -> dict:
    """Class count, licence and DOI exactly as configs/data.py records them."""
    from configs.data import DATA

    return {"num_classes": int(DATA["num_classes"]), "license": str(DATA["license"]),
            "doi": str(DATA["doi"]), "background_index": int(DATA["background_index"]),
            "ignore_index": int(DATA["ignore_index"])}


def sha256_file(path: Path) -> str:
    from src.eval.model_loading import sha256_file as _sha

    return _sha(Path(path))
