from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def build_figure(x1: float = 2000.0, x2: float = 5.0, actual_k: float = 500.0) -> Path:
    # Scenario 1: large w1, small w2
    w1_a, w2_a, b_a = 50.0, 0.1, 50.0
    c1_a = w1_a * x1
    c2_a = w2_a * x2
    pred_a = c1_a + c2_a + b_a

    # Scenario 2: small w1, large w2
    w1_b, w2_b, b_b = 0.1, 50.0, 50.0
    c1_b = w1_b * x1
    c2_b = w2_b * x2
    pred_b = c1_b + c2_b + b_b

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    def plot(ax, title, c1, c2, b, pred):
        labels = ["w1*x1", "w2*x2", "b", "pred", "actual"]
        values = [c1, c2, b, pred, actual_k]
        bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"])
        ax.set_title(title)
        ax.set_ylabel("Value (thousands of $)")
        # Use log scale to expose order-of-magnitude differences
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both", axis="y")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{val:,.1f}k",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plot(
        axes[0],
        title="Case 1: w1=50, w2=0.1, b=50",
        c1=c1_a,
        c2=c2_a,
        b=b_a,
        pred=pred_a,
    )

    plot(
        axes[1],
        title="Case 2: w1=0.1, w2=50, b=50",
        c1=c1_b,
        c2=c2_b,
        b=b_b,
        pred=pred_b,
    )

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = assets_dir / "lecture5-parameter-scenarios.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build_figure()
    print(f"Saved parameter scenarios figure to: {path}")


