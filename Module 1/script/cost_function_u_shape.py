import numpy as np
import matplotlib.pyplot as plt

# Generate w values (parameter values to test)
w_values = np.linspace(-2, 4, 100)

# Simulate a U-shaped cost function J(w)
# Using a quadratic function to create the classic "soup bowl" shape
# Let's say the optimal w is around 1, so we'll center the parabola there
optimal_w = 1.0
J_values = (w_values - optimal_w)**2 + 0.5  # Parabola with minimum at w=1

plt.figure(figsize=(10, 6))

# Plot the cost function
plt.plot(w_values, J_values, 'b-', linewidth=2, label='Cost Function J(w)')

# Mark the minimum point (best w value)
min_idx = np.argmin(J_values)
best_w = w_values[min_idx]
best_J = J_values[min_idx]

plt.scatter(best_w, best_J, color='red', s=100, zorder=5, label=f'Minimum at w = {best_w:.1f}')
plt.annotate('Best value of w\n(Lowest Cost)', 
             xy=(best_w, best_J), 
             xytext=(best_w + 0.8, best_J + 1),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, color='red', ha='center')

# Add labels and title
plt.xlabel('Parameter w (slope)', fontsize=14)
plt.ylabel('Cost Function J(w)', fontsize=14)
plt.title('U-Shaped Cost Function: Finding the Best Parameter w', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add some visual context
plt.text(-1.5, 4, 'Higher Cost\n(Bad Fit)', fontsize=10, ha='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
plt.text(3.5, 4, 'Higher Cost\n(Bad Fit)', fontsize=10, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

plt.tight_layout()
plt.savefig('cost_function_u_shape.png', dpi=300)
plt.show()
