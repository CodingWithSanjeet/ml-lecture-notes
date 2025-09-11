from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_data(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(300, 2000, size=n)
    x2 = rng.integers(0, 6, size=n) + rng.normal(0, 0.2, size=n)
    x2 = np.clip(x2, 0, 5)
    return x1, x2


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)


def main() -> None:
    x1, x2 = make_data()
    x1z, x2z = zscore(x1), zscore(x2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax1.scatter(x1, x2, c="#ff7f0e", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_title("Before scaling (stretched)")
    ax1.set_xlabel("x1 (size in ft^2)")
    ax1.set_ylabel("x2 (# bedrooms)")
    ax1.grid(alpha=0.2)

    ax2.scatter(x1z, x2z, c="#2ca02c", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_title("After scaling (compact, centered)")
    ax2.set_xlabel("x1 (scaled)")
    ax2.set_ylabel("x2 (scaled)")
    ax2.grid(alpha=0.2)

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture6_visual_intuition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


