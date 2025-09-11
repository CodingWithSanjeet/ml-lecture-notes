from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def contours(ax, a: float, b: float, title: str) -> None:
    w1 = np.linspace(-1.2, 1.2, 300)
    w2 = np.linspace(-1.2, 1.2, 300)
    W1, W2 = np.meshgrid(w1, w2)
    Z = a * W1 ** 2 + b * W2 ** 2
    cs = ax.contour(W1, W2, Z, levels=14, colors="#1f77b4", linewidths=1)
    ax.clabel(cs, inline=True, fontsize=7, fmt="")
    ax.set_title(title)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    # Fix axes so plotting the path won't rescale due to overshoot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def gd_path(a: float, b: float, alpha: float, steps: int = 25, start=(-1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    w1, w2 = start
    path_w1 = [w1]
    path_w2 = [w2]
    for _ in range(steps):
        # grad of J = [2 a w1, 2 b w2]
        g1, g2 = 2 * a * w1, 2 * b * w2
        w1 = w1 - alpha * g1
        w2 = w2 - alpha * g2
        path_w1.append(w1)
        path_w2.append(w2)
    return np.array(path_w1), np.array(path_w2)


def build_figure() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Without scaling: elongated contours cause zig-zag
    contours(ax1, a=25.0, b=1.0, title="Without scaling: zig‑zag path")
    # Choose alpha to cause stable oscillation along steep axis: 0.5/a < alpha < 1/a
    p1x, p1y = gd_path(a=25.0, b=1.0, alpha=0.03, steps=22, start=(-1.0, 1.0))
    ax1.plot(p1x, p1y, marker="o", markersize=3, color="#d62728")

    # With scaling: near circular contours → direct path
    contours(ax2, a=5.0, b=5.0, title="With scaling: direct path")
    p2x, p2y = gd_path(a=5.0, b=5.0, alpha=0.06, steps=22, start=(-1.0, 1.0))
    ax2.plot(p2x, p2y, marker="o", markersize=3, color="#2ca02c")

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture5_contours_scaled_unscaled.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build_figure()
    print(f"Saved: {path}")


