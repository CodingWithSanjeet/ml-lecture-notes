from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_feature_data(n: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(300, 2000, size=n)
    x2 = rng.integers(0, 6, size=n) + rng.normal(0, 0.15, size=n)
    return x1, x2


def rescale_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def draw_contours(ax: plt.Axes, scale_x: float, scale_y: float, title: str) -> None:
    w1 = np.linspace(-1.0, 1.0, 200)
    w2 = np.linspace(-1.0, 1.0, 200)
    W1, W2 = np.meshgrid(w1, w2)
    # Quadratic form to mimic cost landscape J(w1, w2)
    Z = (scale_x * W1) ** 2 + (scale_y * W2) ** 2
    cs = ax.contour(W1, W2, Z, levels=12, colors="#1f77b4")
    ax.clabel(cs, inline=True, fontsize=7, fmt="")
    ax.set_title(title)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def build_figure(output_path: Path | None = None) -> Path:
    x1, x2 = make_feature_data()
    x1_s, x2_s = rescale_minmax(x1), rescale_minmax(x2)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x1, x2, c="#ff7f0e", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax1.set_title("Features (raw)")
    ax1.set_xlabel("x1: size in ft^2 (300–2000)")
    ax1.set_ylabel("x2: # bedrooms (0–5)")
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(x1_s, x2_s, c="#ff7f0e", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_title("Features (scaled 0–1)")
    ax2.set_xlabel("x1 scaled")
    ax2.set_ylabel("x2 scaled")
    ax2.grid(alpha=0.2)

    # Cost contours: tall-and-skinny vs near-circular
    ax3 = fig.add_subplot(gs[1, 0])
    draw_contours(ax3, scale_x=25.0, scale_y=1.0, title="Cost contours (before scaling)")

    ax4 = fig.add_subplot(gs[1, 1])
    draw_contours(ax4, scale_x=5.0, scale_y=5.0, title="Cost contours (after scaling)")

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / "lecture5-feature-scaling.png"
    if output_path is not None:
        out = output_path
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build_figure()
    print(f"Saved visualization to: {path}")


