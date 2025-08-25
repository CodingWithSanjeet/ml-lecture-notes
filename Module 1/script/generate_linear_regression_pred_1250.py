from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def generate_portland_like(seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(600, 2600, size=40)
    intercept = 60.0
    slope = 0.13
    noise = rng.normal(0.0, 25.0, size=x.shape)
    y = intercept + slope * x + noise
    return x, y


def plot_with_prediction(
    x: np.ndarray,
    y: np.ndarray,
    x_target: float,
    output_path: Path,
    y_target_k: float | None = None,
) -> None:
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    y_pred = float(poly(x_target))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x", color="#d62728", label="data")

    x_line = np.linspace(0, 3000, 200)
    y_line = poly(x_line)
    plt.plot(x_line, y_line, color="#1f77b4", linewidth=2, label="best-fit line")

    # Dotted guide lines limited to the intersection point
    y_line_pred = y_pred
    # vertical: from bottom up to the prediction
    plt.plot([x_target, x_target], [0, y_line_pred], color="#ff9900", linestyle="--", linewidth=1.5)
    # horizontal: from left to the intersection (use provided target value if any)
    y_h = y_target_k if y_target_k is not None else y_line_pred
    plt.plot([0, x_target], [y_h, y_h], color="#ff9900", linestyle="--", linewidth=1.5)

    # Mark model prediction at x=1250
    plt.scatter([x_target], [y_pred], color="#ff9900", zorder=5,
                label=f"prediction @ {x_target:g} sq ft ~ {y_pred:.0f}k")
    plt.text(x_target + 25, y_pred + 6, f"~${y_pred:.0f}k", color="#ff9900")

    # Only one price annotation near the intersection point

    plt.title("Prediction at 1250 sq ft using Linear Regression")
    plt.xlabel("House size (sq ft)")
    plt.ylabel("Price (in $1000s)")
    plt.xlim(0, 3000)
    plt.ylim(0, 500)

    # Emphasize magnitudes on axes: add ticks at 1250 and at predicted price (e.g., ~227k)
    ax = plt.gca()
    xticks = [0, 500, 1000, 1250, 1500, 2000, 2500, 3000]
    ax.set_xticks(xticks)
    y_key = int(round(y_pred))
    yticks = sorted(set([0, 100, 200, y_key, 300, 400, 500]))
    ax.set_yticks(yticks)

    # Annotate the axes at the target magnitudes
    ax.annotate(f"{int(x_target)} sq ft", xy=(x_target, 0), xytext=(0, -22),
                textcoords="offset points", ha="center", va="top", color="#ff9900")
    ax.annotate(f"{y_key}k", xy=(0, y_pred), xytext=(-28, 0),
                textcoords="offset points", ha="right", va="center", color="#ff9900")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    assets_dir = ensure_assets_dir()
    output_path = assets_dir / "linear_regression_pred_1250.png"
    x, y = generate_portland_like()
    # Draw dotted guides at x=1250 and y=227k as requested
    plot_with_prediction(x, y, x_target=1250.0, output_path=output_path, y_target_k=227.0)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


