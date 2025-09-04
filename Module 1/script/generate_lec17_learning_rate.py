from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets = module_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def j_of_w(w: np.ndarray | float) -> np.ndarray | float:
    return (w - 2.0) ** 2 + 1.0


def derivative(w: float) -> float:
    return 2.0 * (w - 2.0)


def run_gd_path(w0: float, alpha: float, steps: int = 10) -> list[float]:
    w = w0
    path = [w]
    for _ in range(steps):
        w = w - alpha * derivative(w)
        path.append(w)
    return path


def plot_small_alpha(assets: Path) -> None:
    ws = np.linspace(-1.0, 6.0, 300)
    J = j_of_w(ws)
    w_path = run_gd_path(w0=4.0, alpha=0.05, steps=12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    jp = [j_of_w(w) for w in w_path]
    ax.scatter(w_path, jp, color="#ff7f0e")
    for i in range(len(w_path) - 1):
        ax.annotate("", xy=(w_path[i + 1], jp[i + 1]), xytext=(w_path[i], jp[i]),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#ff7f0e"))
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")
    ax.set_title("Small learning rate α: slow progress")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = assets / "lec17_lr_small.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_large_alpha(assets: Path) -> None:
    ws = np.linspace(-1.0, 6.0, 300)
    J = j_of_w(ws)
    # Start near min; large alpha causes overshoot
    w_path = run_gd_path(w0=2.5, alpha=1.2, steps=6)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    jp = [j_of_w(w) for w in w_path]
    ax.scatter(w_path, jp, color="#9467bd")
    for i in range(len(w_path) - 1):
        ax.annotate("", xy=(w_path[i + 1], jp[i + 1]), xytext=(w_path[i], jp[i]),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#9467bd"))
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")
    ax.set_title("Large learning rate α: overshoot/diverge")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = assets / "lec17_lr_large.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_at_minimum_zero_grad(assets: Path) -> None:
    ws = np.linspace(-1.0, 6.0, 300)
    J = j_of_w(ws)
    w0 = 5.0
    # create a toy multi-well by adjusting the view: we just annotate slope=0 at min
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, J, color="#1f77b4")
    # show a local minimum at w=2 with slope 0
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3)
    ax.annotate("slope = 0", xy=(2.0, j_of_w(2.0)), xytext=(20, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.0))
    # Formula effect
    ax.text(0.05, 0.92, "w := w - α·(dJ/dw)\nAt min: dJ/dw=0 → w stays same",
            transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec="#999", alpha=0.9))
    ax.set_title("At a minimum: gradient is zero → w unchanged")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = assets / "lec17_zero_grad.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_fixed_alpha_converge(assets: Path) -> None:
    ws = np.linspace(-1.0, 6.0, 300)
    J = j_of_w(ws)
    # Run GD with fixed alpha; steps naturally shrink as derivative shrinks
    alpha = 0.35
    w_path = run_gd_path(w0=4.5, alpha=alpha, steps=8)
    jp = [j_of_w(w) for w in w_path]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#2ca02c", "#1f77b4"]
    for i in range(len(w_path) - 1):
        ax.annotate("", xy=(w_path[i + 1], jp[i + 1]), xytext=(w_path[i], jp[i]),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=colors[i % len(colors)]))
        ax.scatter([w_path[i]], [jp[i]], color=colors[i % len(colors)], zorder=3)
    ax.scatter([w_path[-1]], [jp[-1]], color=colors[(len(w_path)-1) % len(colors)], zorder=3)
    ax.set_title("Fixed α: derivative shrinks → steps shrink near minimum")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = assets / "lec17_fixed_alpha_converge.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

def plot_fixed_alpha_annotated(assets: Path) -> None:
    ws = np.linspace(-1.0, 6.0, 300)
    J = j_of_w(ws)
    alpha = 0.35
    w_path = run_gd_path(w0=4.5, alpha=alpha, steps=7)
    jp = [j_of_w(w) for w in w_path]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ws, J, color="#1f77b4", label="J(w)")
    ax.scatter([2.0], [j_of_w(2.0)], color="#d62728", zorder=3, label="minimum")

    # Arrows showing update steps (decreasing length)
    for i in range(len(w_path) - 1):
        ax.annotate("", xy=(w_path[i + 1], jp[i + 1]), xytext=(w_path[i], jp[i]),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#2ca02c"))

    # Small slope arrows (magnitude proportional to |dJ/dw|)
    for w in w_path[:-1]:
        dj = derivative(w)
        scale = 0.06
        dx = -np.sign(dj) * scale  # arrow points opposite slope to descend
        dy = (-dj) * scale
        ax.annotate("", xy=(w + dx, j_of_w(w) + dy), xytext=(w, j_of_w(w)),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#9467bd", alpha=0.9))

    # Tangent lines at successive points showing slopes flattening near the minimum
    def draw_tangent(w0: float, width: float = 0.6, color: str = "#e377c2"):
        m = derivative(w0)
        x1, x2 = w0 - width / 2.0, w0 + width / 2.0
        y1 = m * (x1 - w0) + j_of_w(w0)
        y2 = m * (x2 - w0) + j_of_w(w0)
        ax.plot([x1, x2], [y1, y2], color=color, lw=2.4, alpha=0.9)

    # Draw tangents for the last few points (closer to the minimum → flatter)
    palette = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    candidates = w_path[-5:]
    for idx, w in enumerate(candidates):
        draw_tangent(w, width=0.9, color=palette[idx % len(palette)])
    ax.annotate("flatter near minimum", xy=(candidates[-1], j_of_w(candidates[-1])),
                xytext=(30, -25), textcoords="offset points",
                bbox=dict(fc="white", ec="#999", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.text(0.02, 0.94, "fixed α; slopes shrink → steps shrink",
            transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec="#bbb", alpha=0.9))
    ax.set_title("Fixed α with shrinking derivative and steps")
    ax.set_xlabel("w")
    ax.set_ylabel("J(w)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = assets / "lec17_fixed_alpha_annotated.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

def main():
    assets = ensure_assets_dir()
    plot_small_alpha(assets)
    plot_large_alpha(assets)
    plot_at_minimum_zero_grad(assets)
    # Extra: fixed alpha path with shrinking steps near minimum
    plot_fixed_alpha_converge(assets)
    plot_fixed_alpha_annotated(assets)
    for name in [
        "lec17_lr_small.png",
        "lec17_lr_large.png",
        "lec17_zero_grad.png",
        "lec17_fixed_alpha_converge.png",
        "lec17_fixed_alpha_annotated.png",
    ]:
        print("Saved:", assets / name)


if __name__ == "__main__":
    main()


