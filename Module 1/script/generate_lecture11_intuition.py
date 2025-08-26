from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def make_mock_data():
    # Simple line y=x with three points (1,1), (2,2), (3,3)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    return x, y


def compute_cost_w(x: np.ndarray, y: np.ndarray, w: float) -> float:
    m = x.size
    y_hat = w * x
    return float(((y_hat - y) ** 2).sum() / (2 * m))


def plot_scenarios_and_cost(output_path: Path) -> None:
    x, y = make_mock_data()
    ws = [1.0, 0.5, 0.0, -0.5]
    costs = [compute_cost_w(x, y, w) for w in ws]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: four scenarios
    ax = axes[0]
    ax.scatter(x, y, color="#d62728", marker="x", s=60, label="data")
    xs = np.linspace(0, 3.2, 100)
    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    labels = [
        f"w=1.0 (J={costs[0]:.2f})",
        f"w=0.5 (J={costs[1]:.2f})",
        f"w=0.0 (J={costs[2]:.2f})",
        f"w=-0.5 (J={costs[3]:.2f})",
    ]
    for w, c, lab in zip(ws, colors, labels):
        ax.plot(xs, w * xs, color=c, linewidth=2, label=lab)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Different w values and their fits")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    # Right: J(w) curve (sampled grid)
    ax2 = axes[1]
    w_grid = np.linspace(-1.0, 1.5, 120)
    j_grid = np.array([compute_cost_w(x, y, w) for w in w_grid])
    ax2.plot(w_grid, j_grid, color="#1f77b4", linewidth=2)
    ax2.scatter(ws, costs, color="#d62728")
    for w, j in zip(ws, costs):
        ax2.annotate(f"w={w}\nJ={j:.2f}", xy=(w, j), xytext=(8, 4), textcoords="offset points")
    ax2.set_xlabel("w")
    ax2.set_ylabel("J(w)")
    ax2.set_title("Cost J(w) for different w")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    assets = ensure_assets_dir()
    out = assets / "lecture11_scenarios_and_cost.png"
    plot_scenarios_and_cost(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


