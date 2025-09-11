from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_feature_data(n: int = 80, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(300, 2000, size=n)
    x2 = rng.integers(0, 6, size=n) + rng.normal(0, 0.2, size=n)
    x2 = np.clip(x2, 0, 5)
    return x1, x2


def plot_raw_scaled(x1_raw, x2_raw, x1_new, x2_new, title_new: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].scatter(x1_raw, x2_raw, c="#ff7f0e", edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[0].set_title("Raw features")
    axes[0].set_xlabel("x1 (size in ft^2)")
    axes[0].set_ylabel("x2 (# bedrooms)")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(x1_new, x2_new, c="#ff7f0e", edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[1].set_title(title_new)
    axes[1].set_xlabel("x1 (scaled)")
    axes[1].set_ylabel("x2 (scaled)")
    axes[1].grid(alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def method_divide_by_max(x1, x2):
    x1s = x1 / 2000.0
    x2s = x2 / 5.0
    return x1s, x2s


def method_mean_normalization(x1, x2):
    mu1 = x1.mean()
    mu2 = x2.mean()
    x1s = (x1 - mu1) / (x1.max() - x1.min())
    x2s = (x2 - mu2) / (x2.max() - x2.min())
    return x1s, x2s


def method_zscore(x1, x2):
    mu1, mu2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=0), x2.std(ddof=0)
    x1s = (x1 - mu1) / (s1 + 1e-12)
    x2s = (x2 - mu2) / (s2 + 1e-12)
    return x1s, x2s


def main() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    x1, x2 = make_feature_data()

    # 1) Divide by max
    x1_div, x2_div = method_divide_by_max(x1, x2)
    plot_raw_scaled(x1, x2, x1_div, x2_div, "Divide by max (0–1)", assets / "lecture6_divide_by_max.png")

    # 2) Mean normalization
    x1_mn, x2_mn = method_mean_normalization(x1, x2)
    plot_raw_scaled(x1, x2, x1_mn, x2_mn, "Mean normalization", assets / "lecture6_mean_normalization.png")

    # 3) Z-score normalization
    x1_z, x2_z = method_zscore(x1, x2)
    plot_raw_scaled(x1, x2, x1_z, x2_z, "Z-score normalization", assets / "lecture6_zscore_normalization.png")

    print("Saved Lecture 6 feature scaling images in:", assets)


if __name__ == "__main__":
    main()


