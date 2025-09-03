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
    y = 0.12 * x + 20 + rng.normal(0, 10, size=x.size)
    return x, y


def compute_cost_wb(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    m = x.size
    y_hat = w * x + b
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def build_j_grid(x: np.ndarray, y: np.ndarray,
                 w_range: tuple[float, float] = (-0.4, 0.5),
                 b_range: tuple[float, float] = (-1000.0, 1000.0),
                 num_w: int = 180, num_b: int = 180):
    w_vals = np.linspace(w_range[0], w_range[1], num_w)
    b_vals = np.linspace(b_range[0], b_range[1], num_b)
    W, B = np.meshgrid(w_vals, b_vals)
    # Vectorized cost over the grid
    y_hat = (W[..., None] * x[None, None, :]) + B[..., None]
    errs = y_hat - y[None, None, :]
    J = (errs ** 2).sum(axis=-1) / (2 * x.size)
    # grid minimum
    min_idx = np.unravel_index(np.argmin(J), J.shape)
    w_min, b_min, j_min = float(W[min_idx]), float(B[min_idx]), float(J[min_idx])
    return W, B, J, (w_min, b_min, j_min)


def plot_example(x: np.ndarray, y: np.ndarray, W, B, J, j_min, w: float, b: float, out_path: Path) -> None:
    from matplotlib.gridspec import GridSpec

    j_val = compute_cost_wb(x, y, w, b)

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, height_ratios=[1, 1.15])

    # Top-left: data + line
    ax0 = fig.add_subplot(gs[0, 0])
    xs = np.linspace(x.min() - 100, x.max() + 100, 200)
    ax0.scatter(x, y, color="#000000", marker="x", s=40, label="data")
    ax0.plot(xs, w * xs + b, color="#1f77b4", linewidth=2.2, label=f"w={w:.3f}, b={b:.1f}")
    ax0.set_title("Data with model line f(x)=w x + b")
    ax0.set_xlabel("size (sq ft)")
    ax0.set_ylabel("price ($1000s)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=9)

    # Top-right: contour J(w,b)
    ax1 = fig.add_subplot(gs[0, 1])
    cs = ax1.contour(W, B, J, levels=30, cmap="viridis")
    ax1.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    ax1.scatter([w], [b], color="#d62728", s=50, zorder=3)
    ax1.annotate(f"J={j_val:.0f}", xy=(w, b), xytext=(8, -8), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.85, lw=0.8))
    ax1.scatter([j_min[0]], [j_min[1]], marker="*", s=120, color="#2ca02c", zorder=3)
    # Guide lines like in the lecture figure
    ax1.axvline(w, color="#1f77b4", ls="--", lw=1.0, alpha=0.6)
    ax1.axhline(b, color="#1f77b4", ls="--", lw=1.0, alpha=0.6)
    ax1.set_title("Contour of J(w,b)")
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.grid(True, alpha=0.15)

    # Bottom: 3D surface J(w,b)
    ax2 = fig.add_subplot(gs[1, :], projection="3d")
    surf = ax2.plot_surface(W, B, J, cmap="viridis", linewidth=0, antialiased=True, alpha=0.75)
    ax2.scatter([w], [b], [j_val], color="#d62728", s=45)
    ax2.scatter([j_min[0]], [j_min[1]], [j_min[2]], color="#2ca02c", s=60, marker="*")
    ax2.set_title("3D surface J(w,b)")
    ax2.set_xlabel("w")
    ax2.set_ylabel("b")
    ax2.set_zlabel("J(w,b)")
    # Low, oblique viewpoint similar to lecture
    ax2.view_init(elev=18, azim=-135)
    # Keep axes limits consistent with contour
    ax2.set_xlim(W.min(), W.max())
    ax2.set_ylim(B.min(), B.max())
    fig.colorbar(surf, ax=ax2, fraction=0.03, pad=0.06)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    assets = ensure_assets_dir()
    x, y = make_mock_housing()
    W, B, J, j_min = build_j_grid(x, y)

    examples = [(-0.15, 800.0), (0.0, 360.0), (0.14, 15.0), (0.12, 20.0)]
    for idx, (w, b) in enumerate(examples, start=1):
        out = assets / f"lec13_example_{idx}_w_{w:+.2f}_b_{b:+.0f}.png"
        plot_example(x, y, W, B, J, j_min, w, b, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()


