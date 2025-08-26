from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def make_dataset():
    # Clarity dataset: points lie on y = x
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    return x, y


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float = 0.0) -> float:
    m = x.size
    y_hat = w * x + b
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def plot_pair_for_w(x: np.ndarray, y: np.ndarray, w_value: float, b: float, assets: Path) -> Path:
    # Prepare J(w) curve
    w_grid = np.linspace(-2.0, 5.5, 500)
    j_grid = np.array([compute_cost(x, y, w, b) for w in w_grid])
    j_val = compute_cost(x, y, w_value, b)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # Left: f(x) fit for this w
    ax0 = axes[0]
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    ax0.scatter(x, y, color="#d62728", marker="x", s=60)
    ax0.plot(xs, w_value * xs + b, color="#1f77b4", linewidth=2)
    ax0.set_title(f"f(x)=w x + b  (w={w_value:.1f}, b={b:.1f})\nJ={j_val:.3f}")
    ax0.set_xlabel("x (input)")
    ax0.set_ylabel("y")
    ax0.grid(True, alpha=0.25)

    # Right: J(w) curve with point
    ax1 = axes[1]
    ax1.plot(w_grid, j_grid, color="#1f77b4", linewidth=2)
    ax1.scatter([w_value], [j_val], color="#d62728")
    ax1.axvline(w_value, color="#d62728", linestyle=":", alpha=0.7)
    ax1.set_title("J(w) vs w (point shows current w)")
    ax1.set_xlabel("w (parameter)")
    ax1.set_ylabel("J(w)")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    out = assets / f"pair_w_{str(w_value).replace('.', '_')}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    assets = ensure_assets_dir()
    x, y = make_dataset()
    b = 0.0
    for w in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        out = plot_pair_for_w(x, y, w, b, assets)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()


