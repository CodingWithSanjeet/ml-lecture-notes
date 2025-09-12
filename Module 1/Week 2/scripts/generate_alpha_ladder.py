from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def synthetic_cost(iters: np.ndarray, alpha: float) -> np.ndarray:
    base = 0.08 + (1.5 - 0.08) * np.exp(-iters * (alpha * 1.2))
    if alpha < 0.002:
        return 0.08 + (1.5 - 0.08) * np.exp(-iters * (alpha * 0.6))
    if alpha > 0.2:
        osc = 0.15 * np.sin(0.2 * iters)
        return np.clip(base + osc + 0.02 * (alpha - 0.2) * iters / iters.max(), 0, None)
    return base


def main() -> None:
    iters = np.arange(1, 201)
    ladder = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(ladder)))
    for idx, (a, color) in enumerate(zip(ladder, cmap)):
        y = synthetic_cost(iters, a)
        ax.plot(iters, y, color=color)
        # place alpha label slightly to the right of the last point, jitter vertically to avoid overlap
        y_offset = 0.03 * (len(ladder) - idx)
        ax.text(
            iters[-1] + 8,
            y[-1] + y_offset,
            f"α = {a}",
            color=color,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    ax.set_title("Learning‑rate ladder (~3× steps): compare J vs iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost J(\\vec{w}, b)")
    # add space on the right for text labels
    ax.set_xlim(1, iters[-1] + 40)
    ax.grid(alpha=0.3)

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture8_alpha_ladder.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


