from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def plot_variations(output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.linspace(0, 3, 100)

    configs = [
        {"w": 0.0, "b": 1.5, "title": r"$f(x)=0\cdot x + 1.5$"},
        {"w": 0.5, "b": 0.0, "title": r"$f(x)=0.5\,x$"},
        {"w": 0.5, "b": 1.0, "title": r"$f(x)=0.5\,x + 1$"},
    ]

    for ax, cfg in zip(axes, configs):
        w, b, title = cfg["w"], cfg["b"], cfg["title"]
        y = w * x + b
        ax.plot(x, y, color="#1f77b4", linewidth=2)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.2)
        # Mark intercept and slope guide where helpful
        if b != 0:
            ax.scatter([0], [b], color="#1f77b4")
            ax.text(0.05, b + 0.1, f"b={b}", color="#1f77b4")
        if w != 0:
            # Show slope as rise/run around x=1
            x0, x1 = 1.0, 2.0
            y0, y1 = w * x0 + b, w * x1 + b
            ax.plot([x0, x1], [y0, y0], color="#ff9900")
            ax.plot([x1, x1], [y0, y1], color="#ff00aa")
            ax.text(1.4, y0 - 0.15, "run=1", color="#ff9900")
            ax.text(x1 + 0.05, (y0 + y1) / 2, f"rise={w}", color="#ff00aa")

    axes[0].set_ylabel("y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    assets_dir = ensure_assets_dir()
    output_path = assets_dir / "linear_variations_w_b.png"
    plot_variations(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


