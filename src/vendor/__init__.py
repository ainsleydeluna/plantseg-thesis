"""Third-party code vendored verbatim, kept deliberately separate from thesis-owned modules.

BOUNDARY RULE (load-bearing, not cosmetic): nothing under `src.vendor` may import `src.data`,
`torch` or `torchvision`, and this package init stays side-effect free. The corruption closure is
pure NumPy/Pillow/scikit-image, so the cache generator that drives it must be runnable without the
deep-learning stack at all.

That rule is also what keeps generation usable on a machine whose BLAS/OpenMP runtimes clash: the
Anaconda development box aborts with `OMP: Error #15` when torch's OpenMP runtime initialises
before scikit-image's, which `src.data.__init__` (-> `dataset` -> `torch`) would otherwise force on
every corruption call. Vendored code lives here, thesis-owned cache infrastructure lives in
`src.corruption_cache`, and neither reaches for the model stack.
"""
