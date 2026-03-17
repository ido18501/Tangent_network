from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def overlay_curves(image: np.ndarray, curves: list[np.ndarray], *, linewidth: float = 1.6, annotate: bool = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap=None if image.ndim == 3 else "gray")
    for i, curve in enumerate(curves):
        ax.plot(curve[:, 0], curve[:, 1], linewidth=linewidth)
        if annotate and len(curve):
            ax.text(curve[0, 0], curve[0, 1], str(i), fontsize=8)
    ax.set_axis_off()
    ax.set_title("Extracted curves overlay")
    fig.tight_layout()
    return fig, ax


def plot_curves(curves: list[np.ndarray], *, equal_axis: bool = True):
    fig, ax = plt.subplots(figsize=(6, 6))
    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1])
    if equal_axis:
        ax.set_aspect("equal", adjustable="box")
    ax.set_title("Curves")
    fig.tight_layout()
    return fig, ax


def visualize_sampling_order(curve: np.ndarray, every: int = 16):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curve[:, 0], curve[:, 1])
    idx = np.arange(0, len(curve), every)
    ax.scatter(curve[idx, 0], curve[idx, 1], s=18)
    for i in idx:
        ax.text(curve[i, 0], curve[i, 1], str(i), fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Curve sampling order")
    fig.tight_layout()
    return fig, ax


def visualize_arclength_parameterization(curve: np.ndarray):
    diffs = np.diff(curve, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(s)), s)
    ax.set_xlabel("Point index")
    ax.set_ylabel("Cumulative arc length")
    ax.set_title("Arc-length parameterization")
    fig.tight_layout()
    return fig, ax
