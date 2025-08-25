from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def generate_linear_data(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 2400, 20)
    true_intercept = 50.0
    true_slope = 0.12  # price ($1000s) per sq ft
    noise = rng.normal(0.0, 12.0, size=x.shape)
    y = true_intercept + true_slope * x + noise
    return x, y


def make_plot(x: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    # Scatter data
    plt.scatter(x, y, marker="x", color="#d62728", label="data")

    # Linear fit
    coef = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coef)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = poly(x_line)
    plt.plot(x_line, y_line, color="#1f77b4", linewidth=2, label="linear model")

    # Example prediction at 750 sq ft
    x_target = 750
    y_pred = float(poly(x_target))
    plt.axvline(x_target, color="#ff9900", linestyle=":", linewidth=2)
    plt.scatter([x_target], [y_pred], color="#ff9900", zorder=5, label=f"prediction @ {x_target} sq ft")
    plt.text(x_target + 20, y_pred + 5, f"~{y_pred:.0f}", color="#ff9900")

    plt.title("Linear Regression: Price vs House Size")
    plt.xlabel("House size (sq ft)")
    plt.ylabel("Price (in $1000s)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    assets_dir = ensure_assets_dir()
    output_path = assets_dir / "linear_regression_price_vs_size.png"
    x, y = generate_linear_data()
    make_plot(x, y, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


