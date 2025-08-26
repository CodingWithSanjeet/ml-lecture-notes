from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def make_dataset():
    # Clarity dataset: exact line y = x
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    return x, y


def compute_cost(x: np.ndarray, y: np.ndarray, w: float) -> float:
    m = x.size
    y_hat = w * x
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def main():
    assets = ensure_assets_dir()
    x, y = make_dataset()

    # Choose a w to visualize residuals clearly
    w_focus = 1.5
    w_grid = np.linspace(-1.5, 2.5, 400)
    j_grid = np.array([compute_cost(x, y, w) for w in w_grid])

    # Additional points to show how J(w) is built point by point
    w_points = [-0.5, 0.5, 1.0, 1.5, 2.0]
    j_points = [compute_cost(x, y, w) for w in w_points]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: residuals for chosen w
    ax0 = axes[0]
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
    ax0.plot(xs, w_focus * xs, color="#1f77b4", label=f"f(x)=w x (w={w_focus})")
    ax0.scatter(x, y, color="#000000", marker="x", s=70, label="data")

    # Draw residual lines from each data point to the model prediction at same x
    for xi, yi in zip(x, y):
        y_hat_i = w_focus * xi
        ax0.plot([xi, xi], [yi, y_hat_i], linestyle=":", color="#d62728")
        ax0.annotate(
            f"e={y_hat_i - yi:+.1f}",
            xy=(xi, (yi + y_hat_i) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )

    ax0.set_title("Residuals (errors) for one w")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    # Right: J(w) curve and selected points
    ax1 = axes[1]
    ax1.plot(w_grid, j_grid, color="#1f77b4")
    ax1.scatter(w_points, j_points, color="#d62728", zorder=3)
    # Highlight the minimum of J(w)
    min_idx = int(np.argmin(j_grid))
    w_min, j_min = float(w_grid[min_idx]), float(j_grid[min_idx])
    ax1.scatter([w_min], [j_min], marker="*", s=180, color="#2ca02c", edgecolor="#145a32", zorder=4, label=f"min J at w={w_min:.2f}")
    ax1.annotate(
        f"min\nw={w_min:.2f}\nJ={j_min:.3f}",
        xy=(w_min, j_min),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.85, lw=0.8),
        arrowprops=dict(arrowstyle="-", color="#2ca02c", lw=0.8),
        ha="left", va="top",
    )
    for w, j in zip(w_points, j_points):
        ax1.annotate(
            f"w={w}\nJ={j:.3f}",
            xy=(w, j),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.8, lw=0.8),
            arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8),
        )
    ax1.set_title("Building J(w) point by point (minimum highlighted)")
    ax1.set_xlabel("w")
    ax1.set_ylabel("J(w)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out = assets / "residuals_and_cost.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


