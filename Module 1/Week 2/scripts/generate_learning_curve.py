from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def learning_curve(iterations: int = 450, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    iters = np.arange(1, iterations + 1)
    # Construct a monotonically decreasing curve that plateaus
    # J0 is initial cost; decay_rate controls speed; plateau is minimal value
    J0 = 1.5
    decay_rate = 0.025
    plateau = 0.08
    J = plateau + (J0 - plateau) * np.exp(-decay_rate * iters)
    if noise > 0:
        rng = np.random.default_rng(7)
        J = J + rng.normal(0, noise, size=iters.size)
        # Ensure nonincreasing (clip small ups to a tiny downward step)
        for i in range(1, len(J)):
            if J[i] > J[i - 1]:
                J[i] = J[i - 1] - 1e-4
    return iters, J


def main() -> None:
    iters, J = learning_curve()

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.set_title("Learning curve: cost J vs iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost J(\\vec{w}, b)")
    ax.grid(alpha=0.25)

    # Annotate checkpoints
    checkpoints = [100, 200, 400]
    for c in checkpoints:
        ax.scatter([c], [J[c - 1]], color="#9467bd")
        ax.annotate(f"{c}", (c, J[c - 1]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

    # Automatic convergence threshold epsilon line
    # Mark the iteration where delta J < epsilon
    eps = 1e-3
    deltas = np.abs(np.diff(J))
    mask = deltas < eps
    if mask.any():
        idx = int(np.argmax(mask))  # first iteration where drop < eps
        x_conv = idx + 1
        y_conv = J[x_conv - 1]

        # plot pre-convergence in blue, post-convergence in green (switch color AFTER convergence)
        ax.plot(iters[:x_conv], J[:x_conv], color="#1f77b4", linewidth=2, label="pre‑convergence")
        ax.plot(iters[x_conv:], J[x_conv:], color="#2ca02c", linewidth=2, label="post‑convergence")

        # vertical dashed line indicating convergence threshold
        ax.axvline(
            x_conv,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"convergence threshold (ε={eps})",
        )
        ax.scatter([x_conv], [y_conv], color="#d62728", zorder=3)
        ax.legend(frameon=False)
    else:
        # Fallback: plot entire curve in blue
        ax.plot(iters, J, color="#1f77b4", linewidth=2)

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture7_learning_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


