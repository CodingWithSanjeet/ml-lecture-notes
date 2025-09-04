from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def generate_toy_data() -> tuple[np.ndarray, np.ndarray]:
    # Simple 1D linear data with small noise
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 25)
    y = 2.5 * x + 5.0 + rng.normal(0, 2.0, size=x.size)
    return x, y


def compute_cost(w: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    m = x.size
    preds = w * x + b
    return float(np.sum((preds - y) ** 2) / (2 * m))


def compute_gradients(w: float, b: float, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # dJ/dw = (1/m) sum( (w x + b - y) x )
    # dJ/db = (1/m) sum( (w x + b - y) )
    m = x.size
    preds = w * x + b
    err = preds - y
    dj_dw = float(np.sum(err * x) / m)
    dj_db = float(np.sum(err) / m)
    return dj_dw, dj_db


def run_gradient_descent(x: np.ndarray, y: np.ndarray, w0: float, b0: float, alpha: float, steps: int):
    w, b = w0, b0
    path = [(w, b, compute_cost(w, b, x, y))]
    for _ in range(steps):
        dj_dw, dj_db = compute_gradients(w, b, x, y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        path.append((w, b, compute_cost(w, b, x, y)))
    return path


def plot_surface_and_contour(x: np.ndarray, y: np.ndarray, path: list[tuple[float, float, float]], assets: Path) -> None:
    w_vals = np.linspace(-1.0, 5.0, 80)
    b_vals = np.linspace(-5.0, 15.0, 80)
    W, B = np.meshgrid(w_vals, b_vals)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        preds = W[i, :, None] * x[None, :] + B[i, :, None]
        J[i, :] = np.sum((preds - y[None, :]) ** 2, axis=1) / (2 * x.size)

    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(W, B, J, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.85)
    fig.colorbar(surf, ax=ax1, fraction=0.035, pad=0.05)
    pw, pb, pj = zip(*path)
    ax1.plot(pw, pb, pj, color="#d62728", marker="o", markersize=3, linewidth=2)
    ax1.set_title("J(w,b) surface with GD path (squared error)")
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.set_zlabel("J(w,b)")
    ax1.view_init(elev=25, azim=-140)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(W, B, J, levels=30, cmap="viridis")
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    ax2.plot(pw, pb, color="#d62728", marker="o", markersize=3, linewidth=2)
    ax2.set_title("J(w,b) contour with GD path")
    ax2.set_xlabel("w")
    ax2.set_ylabel("b")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    out = assets / "lec18_gd_linear_regression.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    assets = ensure_assets_dir()
    x, y = generate_toy_data()
    path = run_gradient_descent(x, y, w0=-0.5, b0=0.0, alpha=0.05, steps=40)
    plot_surface_and_contour(x, y, path, assets)
    print("Saved:", assets / "lec18_gd_linear_regression.png")


if __name__ == "__main__":
    main()


