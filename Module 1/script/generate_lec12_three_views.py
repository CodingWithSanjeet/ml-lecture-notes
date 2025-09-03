from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def make_mock_housing() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = np.linspace(800, 3200, 26)
    # true relation ~ 0.12*x + 20 (prices in $1000s) with noise
    y = 0.12 * x + 20 + rng.normal(0, 10, size=x.size)
    return x, y


def compute_cost_wb(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    m = x.size
    y_hat = w * x + b
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def plot_lines_view(x: np.ndarray, y: np.ndarray, params: list[tuple[float, float]], assets: Path) -> Path:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    xs = np.linspace(x.min() - 100, x.max() + 100, 200)

    plt.figure(figsize=(6.6, 4.6))
    plt.scatter(x, y, color="#d62728", marker="x", s=40, label="data")
    for (w, b), c in zip(params, colors):
        plt.plot(xs, w * xs + b, color=c, linewidth=2.2, label=f"w={w:.2f}, b={b:.0f}")
    plt.title("Different (w,b) lines over the dataset")
    plt.xlabel("size (sq ft)")
    plt.ylabel("price ($1000s)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    out = assets / "lec12_multi_lines.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_contour_view(x: np.ndarray, y: np.ndarray, params: list[tuple[float, float]], assets: Path) -> Path:
    # grid for w and b
    w_vals = np.linspace(0.0, 0.24, 140)
    b_vals = np.linspace(0.0, 90.0, 140)
    W, B = np.meshgrid(w_vals, b_vals)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost_wb(x, y, float(W[i, j]), float(B[i, j]))

    # locate minimum on grid
    min_idx = np.unravel_index(np.argmin(J), J.shape)
    w_min, b_min, j_min = float(W[min_idx]), float(B[min_idx]), float(J[min_idx])

    plt.figure(figsize=(6.6, 4.6))
    cs = plt.contour(W, B, J, levels=30, cmap="viridis")
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    # scatter selected (w,b)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for (w, b), c in zip(params, colors):
        j = compute_cost_wb(x, y, w, b)
        plt.scatter([w], [b], color=c, s=50, zorder=3)
        plt.annotate(f"w={w:.2f}\nb={b:.0f}\nJ={j:.0f}",
                     xy=(w, b), xytext=(8, -8), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.85, lw=0.8))

    # mark minimum
    plt.scatter([w_min], [b_min], marker="*", s=140, color="#2ca02c", edgecolor="#145a32", zorder=3)
    plt.annotate("minimum", xy=(w_min, b_min), xytext=(10, 12), textcoords="offset points",
                 color="#2ca02c")

    plt.title("Contour of J(w,b) with selected (w,b) points")
    plt.xlabel("w")
    plt.ylabel("b")
    plt.grid(True, alpha=0.15)
    out = assets / "lec12_multi_contour.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_surface_view(x: np.ndarray, y: np.ndarray, params: list[tuple[float, float]], assets: Path) -> Path:
    # grid for w and b
    w_vals = np.linspace(0.0, 0.24, 120)
    b_vals = np.linspace(0.0, 90.0, 120)
    W, B = np.meshgrid(w_vals, b_vals)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost_wb(x, y, float(W[i, j]), float(B[i, j]))

    # min
    min_idx = np.unravel_index(np.argmin(J), J.shape)
    w_min, b_min, j_min = float(W[min_idx]), float(B[min_idx]), float(J[min_idx])

    fig = plt.figure(figsize=(8.8, 4.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surf = ax.plot_surface(W, B, J, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    # scatter selected points
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for (w, b), c in zip(params, colors):
        j = compute_cost_wb(x, y, w, b)
        ax.scatter([w], [b], [j], color=c, s=35)
    # min
    ax.scatter([w_min], [b_min], [j_min], color="#2ca02c", s=60, marker="*", label="min J")
    ax.set_title("J(w,b) surface with selected points")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("J(w,b)")
    fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)
    out = assets / "lec12_multi_surface.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    assets = ensure_assets_dir()
    x, y = make_mock_housing()
    # three example parameter sets + one near min
    params = [(0.06, 50.0), (0.10, 30.0), (0.14, 15.0), (0.18, 0.0)]
    p1 = plot_lines_view(x, y, params, assets)
    p2 = plot_contour_view(x, y, params, assets)
    p3 = plot_surface_view(x, y, params, assets)
    print(f"Saved: {p1}\nSaved: {p2}\nSaved: {p3}")


if __name__ == "__main__":
    main()


