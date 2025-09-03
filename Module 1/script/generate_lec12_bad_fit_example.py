from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def make_mock_housing() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    # house sizes in sq ft
    x = np.linspace(800, 3200, 20)
    # underlying relation ~ 0.12*x + 20 with small noise (prices in $1000s)
    y = 0.12 * x / 1.0 + 20 + rng.normal(0, 8, size=x.size)
    return x, y


def main():
    assets = ensure_assets_dir()
    x, y = make_mock_housing()

    # Bad fit from the lecture narrative
    w_bad = 0.06  # per sq ft (1000s of $ per sq ft)
    b_bad = 50.0

    xs = np.linspace(x.min() - 100, x.max() + 100, 200)
    y_bad = w_bad * xs + b_bad

    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(x, y, color="#000000", marker="x", s=40, label="training data")
    plt.plot(xs, y_bad, color="#d62728", linewidth=2.2, label=f"f(x) = {w_bad}·x + {b_bad}")
    # Visual hint of underestimation: sample vertical residual from a mid point
    mid_idx = x.size // 2
    xi, yi = x[mid_idx], y[mid_idx]
    plt.plot([xi, xi], [w_bad * xi + b_bad, yi], linestyle=":", color="#d62728")
    plt.annotate("underestimates", xy=(xi, (w_bad * xi + b_bad + yi) / 2), xytext=(10, 10),
                 textcoords="offset points", color="#d62728", fontsize=9)
    plt.title("Lecture 12: Example of a bad fit (w=0.06, b=50)")
    plt.xlabel("size (sq ft)")
    plt.ylabel("price ($1000s)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    out = assets / "lec12_bad_fit_example.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


