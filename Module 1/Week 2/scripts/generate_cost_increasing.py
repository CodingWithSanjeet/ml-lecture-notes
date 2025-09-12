from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def increasing_cost(iters: np.ndarray) -> np.ndarray:
    """Stylized monotonically increasing cost curve.

    Models a scenario where α is too large or the update sign is wrong,
    causing J to grow with iterations.
    """
    # Start near a reasonable value and increase with a slight acceleration
    return 0.2 + 0.002 * iters + 0.00002 * (iters ** 1.5)


def main() -> None:
    iters = np.arange(1, 301)
    J = increasing_cost(iters)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(iters, J, color="#d62728", linewidth=2)
    ax.set_title("Cost J vs iterations: consistently increasing (sign error or α too large)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost J(\\vec{w}, b)")
    ax.grid(alpha=0.3)

    ax.annotate(
        "increasing cost",
        xy=(220, J[219]),
        xytext=(160, J[219] + 0.15),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        color="#d62728",
        fontsize=10,
    )

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture8_cost_increasing.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()



