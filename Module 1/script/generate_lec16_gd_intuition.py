from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def j_of_w(w: np.ndarray | float) -> np.ndarray | float:
    # A simple convex function to illustrate J(w)
    return (w - 2.0) ** 2 + 0.5


def plot_gd_intuition(assets: Path) -> None:
    ws = np.linspace(-1.0, 5.0, 300)
    J = j_of_w(ws)

    # Two starting points: right side (positive slope) and left side (negative slope)
    w_right = 3.2
    w_left = 0.4

    def derivative_at(w: float) -> float:
        # d/dw [(w-2)^2 + 0.5] = 2*(w-2)
        return 2.0 * (w - 2.0)

    # Compute tangents
    def tangent_line(w0: float, ws: np.ndarray) -> np.ndarray:
        m = derivative_at(w0)
        b = j_of_w(w0) - m * w0
        return m * ws + b

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Right-side start (positive slope)
    ax = axes[0]
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    ax.scatter([w_right], [j_of_w(w_right)], color="#1f77b4")
    ax.plot(ws, tangent_line(w_right, ws), color="#d62728", ls="-", label="tangent")
    ax.annotate("positive slope\n(dJ/dw > 0)", xy=(w_right, j_of_w(w_right)), xytext=(-40, 40),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("step left (decrease w)", xy=(w_right, j_of_w(w_right) - 0.5), xytext=(-50, -40),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_title("Right side start")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)

    # Left-side start (negative slope)
    ax = axes[1]
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    ax.scatter([w_left], [j_of_w(w_left)], color="#1f77b4")
    ax.plot(ws, tangent_line(w_left, ws), color="#d62728", ls="-")
    ax.annotate("negative slope\n(dJ/dw < 0)", xy=(w_left, j_of_w(w_left)), xytext=(-10, 40),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("step right (increase w)", xy=(w_left, j_of_w(w_left) - 0.5), xytext=(-10, -40),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_title("Left side start")
    ax.set_xlabel("w")
    ax.grid(alpha=0.25)

    fig.suptitle("Gradient Descent Intuition on J(w)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = assets / "lec16_gd_intuition.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_jw_only(assets: Path) -> None:
    ws = np.linspace(-1.0, 5.0, 300)
    J = j_of_w(ws)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(ws, J, color="#1f77b4", linewidth=2)
    ax.set_title("Cost curve J(w) (1-parameter view)")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    out = assets / "lec16_jw_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_cases_individual(assets: Path) -> None:
    ws = np.linspace(-1.0, 5.0, 300)
    J = j_of_w(ws)

    def derivative_at(w: float) -> float:
        return 2.0 * (w - 2.0)

    def tangent_line(w0: float, ws: np.ndarray) -> np.ndarray:
        m = derivative_at(w0)
        b = j_of_w(w0) - m * w0
        return m * ws + b

    # Positive slope case (right side)
    w_right = 3.2
    alpha = 0.3
    next_right = w_right - alpha * derivative_at(w_right)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    # Minimum marker
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")
    # Current and next points
    ax.scatter([w_right], [j_of_w(w_right)], color="#1f77b4", zorder=3, label="current")
    ax.scatter([next_right], [j_of_w(next_right)], color="#17becf", zorder=3, label="next")
    # Arrow from current to next (left)
    ax.annotate("", xy=(next_right, j_of_w(next_right)), xytext=(w_right, j_of_w(w_right)),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#2ca02c"))
    ax.plot(ws, tangent_line(w_right, ws), color="#d62728")
    ax.annotate("dJ/dw > 0", xy=(w_right, j_of_w(w_right)), xytext=(-20, 35),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("step left (decrease w)", xy=(next_right, j_of_w(next_right)), xytext=(-60, -30),
                textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.9))
    ax.set_title("Positive slope: move left")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = assets / "lec16_gd_positive.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # Negative slope case (left side)
    w_left = 0.4
    next_left = w_left - alpha * derivative_at(w_left)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    # Minimum marker
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")
    # Current and next points
    ax.scatter([w_left], [j_of_w(w_left)], color="#1f77b4", zorder=3, label="current")
    ax.scatter([next_left], [j_of_w(next_left)], color="#17becf", zorder=3, label="next")
    # Arrow from current to next (right)
    ax.annotate("", xy=(next_left, j_of_w(next_left)), xytext=(w_left, j_of_w(w_left)),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#1f77b4"))
    ax.plot(ws, tangent_line(w_left, ws), color="#d62728")
    ax.annotate("dJ/dw < 0", xy=(w_left, j_of_w(w_left)), xytext=(-5, 35),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("step right (increase w)", xy=(next_left, j_of_w(next_left)), xytext=(10, -30),
                textcoords="offset points", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.9))
    ax.set_title("Negative slope: move right")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = assets / "lec16_gd_negative.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    assets = ensure_assets_dir()
    plot_gd_intuition(assets)
    plot_jw_only(assets)
    plot_cases_individual(assets)
    print("Saved:", assets / "lec16_gd_intuition.png")
    print("Saved:", assets / "lec16_jw_curve.png")
    print("Saved:", assets / "lec16_gd_positive.png")
    print("Saved:", assets / "lec16_gd_negative.png")


if __name__ == "__main__":
    main()


