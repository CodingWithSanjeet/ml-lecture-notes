from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir():
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def make_mock_housing():
    # Use the same mock dataset as the bad-fit example so visuals match
    rng = np.random.default_rng(42)
    x = np.linspace(800, 3200, 20)
    y = 0.12 * x + 20 + rng.normal(0, 8, size=x.size)
    return x, y


def compute_cost_wb(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    m = x.size
    y_hat = w * x + b
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def main():
    assets = ensure_assets_dir()
    x, y = make_mock_housing()

    # Grids for w and b; ranges chosen to show a clear bowl-like shape
    w_vals = np.linspace(0.0, 0.24, 120)
    b_vals = np.linspace(0.0, 80.0, 120)
    W, B = np.meshgrid(w_vals, b_vals)
    J = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost_wb(x, y, float(W[i, j]), float(B[i, j]))

    # Find minimum on the grid
    min_idx = np.unravel_index(np.argmin(J), J.shape)
    w_min, b_min, j_min = float(W[min_idx]), float(B[min_idx]), float(J[min_idx])

    fig = plt.figure(figsize=(12, 4.5))

    # Left: 3D surface
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax0.plot_surface(W, B, J, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax0.scatter([w_min], [b_min], [j_min], color="#2ca02c", s=60, marker="*", label="min J")
    ax0.set_title("J(w,b) surface (mock housing)")
    ax0.set_xlabel("w")
    ax0.set_ylabel("b")
    ax0.set_zlabel("J(w,b)")
    fig.colorbar(surf, ax=ax0, fraction=0.046, pad=0.04)

    # Right: contour plot
    ax1 = fig.add_subplot(1, 2, 2)
    cs = ax1.contour(W, B, J, levels=25, cmap="viridis")
    ax1.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax1.scatter([w_min], [b_min], color="#2ca02c", s=60, marker="*", zorder=3, label=f"min (w={w_min:.3f}, b={b_min:.1f})")
    ax1.set_title("Contours of J(w,b) — minimum highlighted")
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.legend(fontsize=8, loc="upper right")

    out = assets / "lec12_surface_and_contour.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


