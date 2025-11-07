
"""Minimal fuzzy/non-binary graphing helpers.

A fuzzy graph visualizes an implicit equation F(x,y)=0 by mapping |F(x,y)| to intensity/color.
Lower residual => brighter; higher residual => darker "shadow".

Core idea: compute residual on a grid; map residual via a compressive transfer function.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

Array = np.ndarray

@dataclass
class View:
    xlim: Tuple[float,float] = (-4.0, 4.0)
    ylim: Tuple[float,float] = (-4.0, 4.0)
    res: Tuple[int,int] = (1200, 1200)  # (H, W)

def mesh(view: View) -> Tuple[Array, Array]:
    H, W = view.res
    xs = np.linspace(view.xlim[0], view.xlim[1], W, dtype=np.float64)
    ys = np.linspace(view.ylim[0], view.ylim[1], H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def residual(F: Callable[[Array,Array], Array], X: Array, Y: Array) -> Array:
    R = F(X, Y)
    if not np.issubdtype(R.dtype, np.floating):
        R = R.astype(np.float64)
    return np.abs(R)

def transfer(residual_abs: Array, *, alpha: float=1.0, gamma: float=1.0, clip: Optional[float]=None) -> Array:
    """Map residual -> intensity in [0,1]. Smaller residual => brighter.

    I = 1 / (1 + alpha * residual_abs)
    Then apply gamma correction: I **= (1/gamma).

    If clip is provided, first clip residuals to [0, clip] to keep extremely large values from going pure black.
    """
    R = residual_abs.copy()
    if clip is not None:
        R = np.minimum(R, clip)
    I = 1.0 / (1.0 + alpha * R)
    if gamma != 1.0:
        I = I ** (1.0/gamma)
    return I

def normalize01(a: Array) -> Array:
    mn, mx = np.nanmin(a), np.nanmax(a)
    if mx == mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def fuzzy_image(F: Callable[[Array,Array], Array], view: View, *, alpha=1.0, gamma=1.0, clip=None, eps_stabilizer=1e-9) -> Array:
    X, Y = mesh(view)
    # Stabilize denominators by adding a tiny epsilon where needed; users can build that into F as well.
    R = np.abs(F(X, Y))
    # Map to intensity
    I = transfer(R, alpha=alpha, gamma=gamma, clip=clip)
    return I
def colorize(intensity, cmap_name="magma", invert=False):
    import matplotlib.cm as cm
    import numpy as np
    I = np.clip(intensity, 0.0, 1.0)
    if invert:
        I = 1.0 - I
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(I, bytes=True)
    return rgba
def levelset_mask(F: Callable[[Array,Array], Array], view: View, *, tol: float=1e-3) -> Array:
    """Binary mask of near-solution zone for comparison."""
    X, Y = mesh(view)
    return (np.abs(F(X, Y)) <= tol).astype(float)

# Common transfer presets approximating the "shadowy" aesthetic
def preset_soft(R: Array) -> Array:
    return transfer(R, alpha=2.0, gamma=1.6, clip=np.percentile(R, 99))

def preset_deep(R: Array) -> Array:
    return transfer(R, alpha=5.0, gamma=2.2, clip=np.percentile(R, 98))

# Utility to save as PNG with matplotlib (kept optional to keep core pure numpy)
def save_png(intensity: Array, path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(intensity, cmap='gray', origin='lower', interpolation='bicubic')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
