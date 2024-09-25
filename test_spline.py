import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def fit_spline_loop(x, y, num_points=1000):
    """
    Fit a spline through a set of (x, y) coordinates that form a loop.

    Parameters:
    x (array-like): x-coordinates of the points.
    y (array-like): y-coordinates of the points.
    num_points (int): Number of points to sample along the fitted spline.

    Returns:
    tuple: (xs, ys) The x and y coordinates of the fitted spline.
    """
    # Ensure the input arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Create an array of parametric values (e.g., arc length, or normalized index)
    t = np.linspace(0, 1, len(x))

    # Fit cubic splines with periodic boundary conditions for both x and y
    cs_x = CubicSpline(t, x, bc_type='periodic')
    cs_y = CubicSpline(t, y, bc_type='periodic')

    # Create a finer array of parametric values to sample the spline
    t_fine = np.linspace(0, 1, num_points)

    # Evaluate the spline at these points
    xs = cs_x(t_fine)
    ys = cs_y(t_fine)

    return xs, ys

# Example usage
x = [0, 1, 1, 0]
y = [0, 0, 1, 0]

# Fit the spline loop
xs, ys = fit_spline_loop(x, y)

# Plot the original points and the fitted spline
plt.figure(figsize=(6,6))
plt.plot(x, y, 'ro', label='Original Points')
plt.plot(xs, ys, 'b-', label='Fitted Spline')
plt.legend()
plt.axis('equal')
plt.title('Cubic Spline Loop')
plt.show()
