import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def draw_without_vectorization(ax, num_ops: int = 16) -> None:
    """Draw timeline showing sequential multiply-add operations.

    Each row represents one time step t_k computing w[k] * x[k] and adding to f.
    """
    ax.set_title("Without vectorization (sequential)")
    ax.set_xlim(0, 7.5)
    # Use larger row spacing so text in boxes doesn't collide
    y_step = 0.7
    top = num_ops * y_step
    ax.set_ylim(-0.5, top + 0.5)
    ax.set_axis_off()

    # Column labels
    ax.text(0.1, top, "time →", fontsize=10, va="top")

    for k in range(num_ops):
        y = k * y_step
        # time label t_k on the left
        ax.text(0.0, y, f"t{k}", fontsize=8, ha="left", va="center")
        # operation box
        row_h = 0.38
        r = Rectangle((1.0, y - row_h / 2), 5.6, row_h, edgecolor="#444", facecolor="#E6F2FF")
        ax.add_patch(r)
        # Slightly smaller font, keep inside the box with padding, and add white bbox
        ax.text(
            1.15,
            y,
            f"f = f + w[{k}] * x[{k}]",
            fontsize=8,
            va="center",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.5},
        )


def draw_with_vectorization(ax, num_ops: int = 16) -> None:
    """Draw parallel multiply followed by fast reduction (sum)."""
    ax.set_title("With vectorization (parallel)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 6)
    ax.set_axis_off()

    # t0: all multiplications in parallel (grid of small boxes)
    ax.text(0.0, 4.8, "t0: multiply all pairs in parallel", fontsize=9, va="top")
    cols = min(8, num_ops)
    rows = math.ceil(num_ops / cols)
    box_w, box_h = 0.9, 0.5
    x0, y0 = 0.5, 3.2
    for k in range(num_ops):
        rr = k // cols
        cc = k % cols
        x = x0 + cc * (box_w + 0.1)
        y = y0 - rr * (box_h + 0.1)
        r = Rectangle((x, y), box_w, box_h, edgecolor="#444", facecolor="#E6F2FF")
        ax.add_patch(r)
        ax.text(x + box_w / 2, y + box_h / 2, f"w[{k}]*x[{k}]", fontsize=7, ha="center", va="center")

    # arrow to reduction box
    arr = FancyArrowPatch((4.8, 2.2), (6.0, 1.2), arrowstyle="->", mutation_scale=12, color="#444")
    ax.add_patch(arr)

    # t1: fast reduction (sum)
    ax.text(6.4, 1.8, "t1: fast sum (reduction)", fontsize=9, va="top")
    r_sum = Rectangle((6.2, 0.7), 3.0, 1.0, edgecolor="#444", facecolor="#FFEFD6")
    ax.add_patch(r_sum)
    ax.text(7.7, 1.2, "sum(w[j]*x[j])", fontsize=9, ha="center", va="center")

    # final add b
    arr2 = FancyArrowPatch((7.7, 0.7), (7.7, 0.2), arrowstyle="->", mutation_scale=12, color="#444")
    ax.add_patch(arr2)
    ax.text(7.7, 0.0, "+ b → f", fontsize=9, ha="center", va="bottom")


def build_figure(num_ops: int = 16, output_path: Path | None = None) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    draw_without_vectorization(ax1, num_ops=num_ops)
    draw_with_vectorization(ax2, num_ops=num_ops)

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / "lecture3-vectorization-timeline.png"
    if output_path is not None:
        out = output_path
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build_figure(num_ops=16)
    print(f"Saved visualization to: {path}")


