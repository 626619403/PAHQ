"""Visualization utilities for PAHQ experiments.

Includes:
- Colormap helpers (adapted from cmapy for OpenCV compatibility)
- ROC curve plotting via ``pahq.roc.RocEvaluator.plot``
"""
from __future__ import annotations

import functools
from typing import List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Colormap utilities (adapted from cmapy)
# ---------------------------------------------------------------------------

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import cv2 as cv
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@functools.lru_cache(maxsize=None)
def _get_cmap_lut(cmap_name: str, n: int = 256) -> np.ndarray:
    """Return an (n, 1, 3) uint8 BGR lookup table for the given colormap."""
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for colormap utilities")
    cmap = mpl.colormaps[cmap_name]
    lut = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
    # Convert RGB -> BGR for OpenCV
    return lut[:, ::-1].reshape(n, 1, 3)


def cmap(cmap_name: str, n: int = 256) -> np.ndarray:
    """Return an OpenCV-compatible (n, 1, 3) uint8 BGR colormap LUT."""
    return _get_cmap_lut(cmap_name, n)


def color(cmap_name: str, value: float, n: int = 256):
    """Return the BGR color for a scalar value in [0, 1]."""
    lut = _get_cmap_lut(cmap_name, n)
    idx = max(0, min(n - 1, int(value * (n - 1))))
    return tuple(int(c) for c in lut[idx, 0])


def colorize(image: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply a colormap to a grayscale image using OpenCV."""
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required for colorize()")
    lut = _get_cmap_lut(cmap_name)
    return cv.LUT(cv.cvtColor(image, cv.COLOR_GRAY2BGR), lut)


# ---------------------------------------------------------------------------
# ROC plotting
# ---------------------------------------------------------------------------

def plot_roc(
    points_by_method: dict,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot multiple ROC curves on the same axes.

    Parameters
    ----------
    points_by_method : dict
        Maps method name -> list of RocPoints (dicts with ``edge_fpr``,
        ``edge_tpr`` keys, as returned by ``RocEvaluator.sweep``).
    title : str
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        Call ``plt.show()`` after plotting.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_roc()")

    fig, ax = plt.subplots(figsize=(6, 6))
    for method, points in points_by_method.items():
        from pahq.roc import RocEvaluator
        RocEvaluator.plot(points, label=method, ax=ax, show=False)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC plot to {save_path}")

    if show:
        plt.show()

    return fig, ax
