from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def make_data(seed: int = 3):
    rng = np.random.default_rng(seed)
    x = np.linspace(600, 2600, 40)
    true_w, true_b = 0.14, 60.0
    y = true_w * x + true_b + rng.normal(0, 28, size=x.shape)
    return x, y


def plot_cost_intuition(output_path: Path) -> None:
    x, y = make_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Poor fit parameters
    w_bad, b_bad = 0.05, 150
    y_bad = w_bad * x + b_bad
    axes[0].scatter(x, y, color="#d62728", marker="x", alpha=0.8)
    axes[0].plot(x, y_bad, color="#1f77b4", linewidth=2)
    axes[0].set_title("High J(w,b) — poor fit")
    axes[0].set_xlabel("size (sq ft)")
    axes[0].set_ylabel("price ($1000s)")
    axes[0].grid(True, alpha=0.2)

    # Good fit parameters (close to true)
    w_good, b_good = 0.14, 60
    y_good = w_good * x + b_good
    axes[1].scatter(x, y, color="#d62728", marker="x", alpha=0.8)
    axes[1].plot(x, y_good, color="#2ca02c", linewidth=2)
    axes[1].set_title("Low J(w,b) — good fit")
    axes[1].set_xlabel("size (sq ft)")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    assets_dir = ensure_assets_dir()
    out = assets_dir / "cost_intuition.png"
    plot_cost_intuition(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


