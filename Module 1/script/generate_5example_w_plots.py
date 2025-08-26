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


def plot_single(ax, x, y, w: float, b: float = 0.0):
    j = compute_cost(x, y, w, b)
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    ax.scatter(x, y, color="#d62728", marker="x", s=60)
    ax.plot(xs, w * xs + b, color="#1f77b4", linewidth=2)
    ax.set_title(f"w={w:.1f}, b={b:.1f}  J={j:.3f}")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xs.min(), xs.max())


def main():
    assets = ensure_assets_dir()
    x, y = make_dataset()
    ws = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # Individual images
    for w in ws:
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_single(ax, x, y, w=w, b=0.0)
        fig.tight_layout()
        out = assets / f"w_plot_{str(w).replace('.', '_')}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)

    # Grid image
    cols = min(5, len(ws))
    rows = int(np.ceil(len(ws) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3*rows), sharey=True)
    axes = np.atleast_2d(axes)
    for ax, w in zip(axes.ravel(), ws):
        plot_single(ax, x, y, w=w, b=0.0)
    fig.suptitle("5-example dataset: fits for different w (b=0)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    grid_out = assets / "w_plots_grid.png"
    fig.savefig(grid_out, dpi=160)
    plt.close(fig)
    print(f"Saved: {grid_out}")


if __name__ == "__main__":
    main()


