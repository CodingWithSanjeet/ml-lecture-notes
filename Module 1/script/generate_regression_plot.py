import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_assets_dir() -> Path:
    """Ensure the assets directory exists next to the Module 1 folder.

    Returns the path to the assets directory inside Module 1.
    """
    # This script is expected at Module 1/script/
    module_dir = Path(__file__).resolve().parents[1]
    assets_dir = module_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def generate_sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate simple, reproducible housing data (size vs price in $1000s)."""
    rng = np.random.default_rng(42)
    sizes = np.array([400, 600, 750, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300])
    # Base linear trend (in $1000s) plus slight nonlinearity and noise
    base = 60 + 0.12 * sizes  # roughly 60k + 120 per sqft (scaled to $1000s)
    curve = base + 0.00002 * (sizes - 1200) ** 2
    noise = rng.normal(0, 10, size=sizes.shape)
    prices = curve + noise
    return sizes, prices


def plot_regression(sizes: np.ndarray, prices: np.ndarray, output_path: Path) -> None:
    """Create the regression illustration and save to output_path."""
    plt.figure(figsize=(10, 6))

    # Scatter points
    plt.scatter(sizes, prices, marker="x", color="red", label="data")

    # Linear fit
    coef_lin = np.polyfit(sizes, prices, deg=1)
    poly_lin = np.poly1d(coef_lin)
    x_line = np.linspace(sizes.min(), sizes.max(), 200)
    y_line = poly_lin(x_line)
    plt.plot(x_line, y_line, color="#1f77b4", linewidth=2, label="linear fit")

    # Curved fit (degree 3)
    coef_curve = np.polyfit(sizes, prices, deg=3)
    poly_curve = np.poly1d(coef_curve)
    y_curve = poly_curve(x_line)
    plt.plot(x_line, y_curve, color="#d62728", alpha=0.6, linewidth=2, label="curved fit")

    # Example prediction at 750 sq ft
    x_target = 750
    y_lin_pred = float(poly_lin(x_target))
    y_curve_pred = float(poly_curve(x_target))
    plt.axvline(x_target, color="#ff9900", linestyle=":", linewidth=2)
    plt.scatter([x_target, x_target], [y_lin_pred, y_curve_pred], color=["#1f77b4", "#d62728"], zorder=5)
    plt.text(x_target + 15, y_lin_pred + 5, f"linear ~ {y_lin_pred:.0f}", color="#1f77b4")
    plt.text(x_target + 15, y_curve_pred - 15, f"curved ~ {y_curve_pred:.0f}", color="#d62728")

    # Labels and title
    plt.title("Regression: Housing price prediction")
    plt.xlabel("House size (sq ft)")
    plt.ylabel("Price (in $1000s)")
    plt.xlim(sizes.min() - 50, sizes.max() + 50)
    plt.ylim(max(0, prices.min() - 30), prices.max() + 40)
    plt.legend()
    plt.grid(True, alpha=0.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    assets_dir = ensure_assets_dir()
    output_path = assets_dir / "regression_price_prediction.png"
    sizes, prices = generate_sample_data()
    plot_regression(sizes, prices, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


