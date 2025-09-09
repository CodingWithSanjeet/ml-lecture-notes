from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_without_vectorization(ax, num_ops: int = 16) -> None:
    ax.set_title("Without vectorization (sequential)")
    ax.set_xlim(0, 7.5)
    y_step = 0.7
    top = num_ops * y_step
    ax.set_ylim(-0.5, top + 0.5)
    ax.set_axis_off()

    ax.text(0.1, top, "time →", fontsize=10, va="top")

    for k in range(num_ops):
        y = k * y_step
        ax.text(0.0, y, f"t{k}", fontsize=8, ha="left", va="center")
        row_h = 0.38
        r = Rectangle((1.0, y - row_h / 2), 5.6, row_h, edgecolor="#444", facecolor="#E6F2FF")
        ax.add_patch(r)
        ax.text(
            1.15,
            y,
            f"f = f + w[{k}] * x[{k}]",
            fontsize=8,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.5},
        )


def main(num_ops: int = 16, output_path: Path | None = None) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    draw_without_vectorization(ax, num_ops=num_ops)

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / "lecture3-without-vectorization-timeline.png"
    if output_path is not None:
        out = output_path
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = main(num_ops=16)
    print(f"Saved visualization to: {path}")


