from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def synthetic_cost(iters: np.ndarray, alpha: float) -> np.ndarray:
    """Generate stylized cost curves for different learning rates.

    - Very small alpha: slow, smooth decrease
    - Just right alpha: faster, smooth decrease
    - Too big alpha: oscillations or divergence
    """
    # base decreasing shape
    base = 0.08 + (1.5 - 0.08) * np.exp(-iters * (alpha * 1.2))
    if alpha < 0.002:
        # very small: extra slow decay
        return 0.08 + (1.5 - 0.08) * np.exp(-iters * (alpha * 0.6))
    if 0.005 <= alpha <= 0.05:
        # good range: smooth and relatively quick
        return base
    if alpha > 0.2:
        # too big: oscillatory and sometimes increasing
        osc = 0.15 * np.sin(0.2 * iters) * (1 + 0.3 * (alpha - 0.2))
        return np.clip(base + osc + 0.02 * (alpha - 0.2) * iters / iters.max(), 0, None)
    # intermediate slightly wobbly
    return np.clip(base + 0.05 * np.sin(0.15 * iters), 0, None)


def main() -> None:
    iters = np.arange(1, 201)

    # Top row: diagnosing too big vs bug vs oscillation isn't converging
    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: too small alpha vs just right (comparison)
    for a, c, lbl in [(0.001, "#1f77b4", "α = 0.001 (too small)"), (0.01, "#2ca02c", "α = 0.01 (just right)")]:
        ax1.plot(iters, synthetic_cost(iters, a), color=c, label=lbl)
    ax1.set_title("Effect of small vs adequate learning rate")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost J(\\vec{w}, b)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    # Panel 2: too big alpha
    for a, c in [(0.3, "#d62728"), (0.6, "#ff7f0e")]:
        ax2.plot(iters, synthetic_cost(iters, a), color=c, label=f"α = {a}")
    ax2.set_title("Too large α → oscillations/divergence")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Cost J(\\vec{w}, b)")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    # Panel 3: picking α by ~3× ladder
    ladder = [0.001, 0.003, 0.01, 0.03, 0.1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(ladder)))
    for a, color in zip(ladder, cmap):
        ax3.plot(iters, synthetic_cost(iters, a), color=color, label=f"α = {a}")
    ax3.set_title("Try α values spaced by ~3×")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Cost J(\\vec{w}, b)")
    ax3.legend(frameon=False, ncol=3)
    ax3.grid(alpha=0.25)

    # Panel 4: guidance text via annotations
    ax4.axis("off")
    ax4.text(
        0.0,
        0.95,
        "Guidelines",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    lines = [
        "• Too small α → very slow decrease in J",
        "• Too large α → oscillations or divergence",
        "• Use tiny α to debug: J should decrease every iteration",
        "• Sweep α ≈ 3× steps; pick largest that still converges smoothly",
        "• Feature scaling helps make α selection easier",
    ]
    ax4.text(0.02, 0.85, "\n".join(lines), fontsize=11, va="top")

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture8_learning_rate.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()



