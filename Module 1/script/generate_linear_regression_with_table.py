from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def generate_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.uniform(600, 2600, size=35)
    true_intercept = 60.0
    true_slope = 0.14  # price ($1000s) per sq ft
    noise = rng.normal(0.0, 30.0, size=sizes.shape)
    prices = true_intercept + true_slope * sizes + noise
    return sizes, prices


def make_plot_with_table(sizes: np.ndarray, prices: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[3.0, 2.0], figure=fig)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    # Left: scatter + linear fit
    ax_scatter.set_title("House sizes and prices")
    ax_scatter.scatter(sizes, prices, marker="x", color="#d62728")

    coef = np.polyfit(sizes, prices, 1)
    poly = np.poly1d(coef)
    x_line = np.linspace(0, 3000, 200)
    y_line = poly(x_line)
    ax_scatter.plot(x_line, y_line, color="#1f77b4", linewidth=2)

    ax_scatter.set_xlabel("size in feet^2")
    ax_scatter.set_ylabel("price in $1000's")
    ax_scatter.set_xlim(0, 3000)
    ax_scatter.set_ylim(0, 500)
    ax_scatter.grid(True, alpha=0.2)

    # Right: data table (illustrative values similar to lecture)
    ax_table.axis("off")
    col_labels = ["size in feet^2", "price in $1000's"]
    table_rows = [
        ("2104", "400"),
        ("1416", "232"),
        ("1534", "315"),
        ("852", "178"),
        ("…", "…"),
        ("3210", "870"),
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)
    ax_table.set_title("Data table")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    assets_dir = ensure_assets_dir()
    output_path = assets_dir / "linear_regression_plot_with_table.png"
    sizes, prices = generate_data()
    make_plot_with_table(sizes, prices, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


