from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def oscillating_cost(iters: np.ndarray, amplitude: float = 0.2, frequency: float = 0.25) -> np.ndarray:
    """Stylized non‑monotonic cost curve showing oscillations due to too‑large α or a bug."""
    baseline = 0.08 + (1.6 - 0.08) * np.exp(-0.02 * iters)
    osc = amplitude * np.sin(frequency * iters) * (1 - np.exp(-0.01 * iters))
    return np.clip(baseline + osc, 0, None)


def main() -> None:
    iters = np.arange(1, 301)
    J = oscillating_cost(iters)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(iters, J, color="#1f77b4", linewidth=2)
    ax.set_title("Cost J vs iterations: non‑monotonic (oscillation) due to too‑large α or bug")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost J(\\vec{w}, b)")
    ax.grid(alpha=0.3)

    # Annotate up/down segments
    ax.annotate(
        "oscillation",
        xy=(60, J[59]),
        xytext=(90, J[59] + 0.25),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        color="#d62728",
        fontsize=10,
    )
    ax.annotate(
        "not decreasing every iteration",
        xy=(170, J[169]),
        xytext=(200, J[169] + 0.25),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        color="#d62728",
        fontsize=10,
    )

    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = assets / "lecture8_cost_oscillation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()



