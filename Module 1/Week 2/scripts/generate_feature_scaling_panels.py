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


def save_features_scatter(x1: np.ndarray, x2: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(x1, x2, c="#ff7f0e", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_contours(scale_x: float, scale_y: float, title: str, out_path: Path) -> None:
    w1 = np.linspace(-1.0, 1.0, 200)
    w2 = np.linspace(-1.0, 1.0, 200)
    W1, W2 = np.meshgrid(w1, w2)
    Z = (scale_x * W1) ** 2 + (scale_y * W2) ** 2
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    cs = ax.contour(W1, W2, Z, levels=12, colors="#1f77b4")
    ax.clabel(cs, inline=True, fontsize=7, fmt="")
    ax.set_title(title)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    x1, x2 = make_feature_data()
    x1s, x2s = rescale_minmax(x1), rescale_minmax(x2)

    save_features_scatter(
        x1,
        x2,
        title="Features (raw)",
        xlabel="x1: size in ft^2 (300–2000)",
        ylabel="x2: # bedrooms (0–5)",
        out_path=assets / "lecture5-features-raw.png",
    )

    save_features_scatter(
        x1s,
        x2s,
        title="Features (scaled 0–1)",
        xlabel="x1 scaled",
        ylabel="x2 scaled",
        out_path=assets / "lecture5-features-scaled.png",
    )

    save_contours(
        scale_x=25.0,
        scale_y=1.0,
        title="Cost contours (before scaling)",
        out_path=assets / "lecture5-contours-before.png",
    )

    save_contours(
        scale_x=5.0,
        scale_y=5.0,
        title="Cost contours (after scaling)",
        out_path=assets / "lecture5-contours-after.png",
    )

    print("Saved feature scaling panel images to:")
    for name in [
        "lecture5-features-raw.png",
        "lecture5-features-scaled.png",
        "lecture5-contours-before.png",
        "lecture5-contours-after.png",
    ]:
        print(" -", assets / name)


if __name__ == "__main__":
    main()


