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


def main():
    assets = ensure_assets_dir()
    x, y = make_dataset()
    b = 0.0
    ws = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # Color palette (will cycle automatically if ws is longer)
    colors = [
        "#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
        "#8c564b", "#17becf", "#7f7f7f", "#bcbd22", "#e377c2",
        "#7f0000", "#6600cc", "#009e73", "#f781bf"
    ]

    # Prepare global J(w) curve
    w_grid = np.linspace(-2.0, 5.5, 500)
    j_grid = np.array([compute_cost(x, y, w, b) for w in w_grid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # Left: overlay fits (all w)
    ax0 = axes[0]
    ax0.scatter(x, y, color="#000000", marker="x", s=70, label="data")
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    for i, w in enumerate(ws):
        c = colors[i % len(colors)]
        ax0.plot(xs, w * xs + b, color=c, linewidth=2, label=f"w={w:.1f}")
    ax0.set_title("All fit lines on the same dataset (b=0)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    # Right: J(w) with the chosen w points
    ax1 = axes[1]
    ax1.plot(w_grid, j_grid, color="#1f77b4", linewidth=2)
    # Highlight min J(w)
    min_idx = int(np.argmin(j_grid))
    w_min, j_min = float(w_grid[min_idx]), float(j_grid[min_idx])
    ax1.scatter([w_min], [j_min], marker="*", s=160, color="#2ca02c", edgecolor="#145a32", zorder=4, label=f"min J at w={w_min:.2f}")
    ax1.annotate(
        f"min w={w_min:.2f}\nJ={j_min:.3f}",
        xy=(w_min, j_min),
        xytext=(10, -16),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.85, lw=0.8),
        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8),
        ha="left", va="top",
    )
    # Predefined annotation offsets to avoid label overlap
    base_offsets = [(8, 8), (8, -12), (8, 20), (8, -20), (-40, 8), (-40, -12), (12, 14), (12, -14), (14, 22), (14, -22), (8, 26)]
    for idx, w in enumerate(ws):
        c = colors[idx % len(colors)]
        j = compute_cost(x, y, w, b)
        ax1.scatter([w], [j], color=c, zorder=3)
        dx, dy = base_offsets[idx % len(base_offsets)]
        ax1.annotate(
            f"w={w}\nJ={j:.3f}",
            xy=(w, j),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.75, lw=0.8),
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=0),
            zorder=4,
        )
    ax1.set_title("J(w) curve with selected w values (b=0) — minimum highlighted")
    ax1.set_xlabel("w")
    ax1.set_ylabel("J(w)")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    out = assets / "common_w_fits_and_cost.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


