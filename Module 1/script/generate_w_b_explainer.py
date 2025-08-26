from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def plot_w_b_explainer(w: float = 0.5, b: float = 1.0, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = ensure_assets_dir() / "w_b_explainer.png"

    x = np.linspace(0, 10, 200)
    y = w * x + b

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color="#1f77b4", linewidth=2, label=f"y = {w}x + {b}")
    plt.xlim(0, 10)
    plt.ylim(0, max(12, w * 10 + b + 1))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)

    # Intercept annotation (b)
    plt.scatter([0], [b], color="#1f77b4", zorder=5)
    plt.annotate("b (y-intercept)", xy=(0, b), xytext=(0.8, b + 1.2),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"), color="#1f77b4")

    # Slope annotation (rise over run)
    x0, x1 = 2, 4
    y0, y1 = w * x0 + b, w * x1 + b
    plt.plot([x0, x1], [y0, y0], color="#ff9900", linewidth=2)  # run
    plt.plot([x1, x1], [y0, y1], color="#ff00aa", linewidth=2)  # rise
    plt.annotate("run = 2", xy=((x0 + x1) / 2, y0), xytext=(0, -16), textcoords="offset points",
                 ha="center", color="#ff9900")
    plt.annotate(f"rise = {w * (x1 - x0):.1f}", xy=(x1, (y0 + y1) / 2), xytext=(8, 0),
                 textcoords="offset points", color="#ff00aa")
    plt.annotate("slope w = rise/run", xy=(x1 + 0.2, (y0 + y1) / 2 + 0.4), color="#ff00aa")

    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def main() -> None:
    out = plot_w_b_explainer()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


