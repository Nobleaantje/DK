import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def compute_curvature(x, y):
    """Compute the curvature for the given x and y points of the raceline."""
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    return curvature

def raceline_objective(checkpoints, n_points=100):
    """Objective function to minimize the curvature of the raceline."""
    x = checkpoints[::2]
    y = checkpoints[1::2]
    
    # Ensure the raceline forms a loop by repeating the first point
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    # Fit a cubic spline through the checkpoints to get a smooth raceline
    spline = CubicSpline(np.linspace(0, 1, len(x)), np.vstack((x, y)), axis=1, bc_type='periodic')
    
    # Sample points along the spline
    t_values = np.linspace(0, 1, n_points)
    raceline_x, raceline_y = spline(t_values)
    
    # Compute curvature
    curvature = compute_curvature(raceline_x, raceline_y)
    
    # Minimize the sum of curvature
    return np.sum(curvature**2)

def optimize_raceline(checkpoints, r, n_points=100):
    """
    Optimize the raceline by moving the checkpoints within a radius `r`.
    
    Parameters:
    - checkpoints: A list of (x, y) tuples of checkpoints [(x1, y1), (x2, y2), ...]
    - r: Maximum distance each checkpoint can move from its original position.
    - n_points: Number of points used for smoothing the raceline.
    
    Returns:
    - optimized_raceline: Optimized (x, y) positions of the checkpoints.
    """
    initial_checkpoints = np.array(checkpoints).flatten()
    
    def constraint_func(checkpoints):
        """Ensure that each checkpoint stays within radius r of the original."""
        new_checkpoints = checkpoints.reshape(-1, 2)
        original_checkpoints = np.array(initial_checkpoints).reshape(-1, 2)
        return r - np.linalg.norm(new_checkpoints - original_checkpoints, axis=1)

    constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Optimize the raceline by minimizing curvature
    result = minimize(raceline_objective, initial_checkpoints, args=(n_points,), constraints=constraints, method='SLSQP')
    
    # Reshape the result into x, y coordinates
    optimized_checkpoints = result.x.reshape(-1, 2)
    return optimized_checkpoints

def plot_raceline(checkpoints, optimized_checkpoints=None):
    """Plot the raceline and the optimized raceline if provided."""
    checkpoints = np.array(checkpoints)
    
    plt.figure(figsize=(8, 8))
    plt.plot(checkpoints[:, 0], checkpoints[:, 1], 'bo-', label='Original Checkpoints')
    
    if optimized_checkpoints is not None:
        optimized_checkpoints = np.array(optimized_checkpoints)
        # Append the first checkpoint at the end to close the loop
        optimized_checkpoints = np.vstack([optimized_checkpoints, optimized_checkpoints[0]])
        plt.plot(optimized_checkpoints[:, 0], optimized_checkpoints[:, 1], 'ro-', label='Optimized Raceline')
    
    plt.legend()
    plt.title("Raceline Optimization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage
checkpoints = [
    (694.59796, -220.4221),
    (733.31953, -226.15862),
    (759.13391, -248.62669),
    (762.00217, -278.26542),
    (743.8365, -314.11872),
    (687.90535, -346.14767),
    (617.63288, -371.96205),
    (400.12285, -427.41516),
    (202.69067, -459.92215),
    (131.46211, -447.01496),
    (102.77946, -414.98601),
    (105.64773, -391.08381),
    (136.24255, -382.00098),
    (310.25057, -384.86924),
    (401.07893, -371.48401),
    (424.02505, -355.70855),
    (436.45419, -321.76743),
    (432.62984, -287.8263),
    (410.16177, -264.40215),
    (387.6937, -254.84127),
    (79.833352, -222.81232),
    (48.76049, -203.21251),
    (35.853302, -168.3153),
    (43.502006, -138.19852),
    (66.448119, -116.2085),
    (105.16968, -109.51588),
    (241.41223, -121.94503),
    (283.4801, -113.81828),
    (384.34739, -44.501896),
    (416.85439, -37.80928),
    (462.74661, -44.023852),
    (609.9841700000001, -96.608695),
    (686.9492600000001, -110.47197),
    (709.8953700000001, -102.82327),
    (781.1239300000001, -58.843217),
    (821.2796300000001, -48.804293),
    (873.3864300000001, -55.974953),
    (909.2397300000001, -87.047814),
    (914.9762600000001, -120.03285),
    (896.3325400000001, -137.72048),
    (840.8794300000001, -141.54483),
    (380.0450000000001, -140.58874),
    (339.4112600000001, -154.93007),
    (320.2894900000001, -175.964),
    (325.0699300000001, -197.95403),
    (358.05497, -214.68557),
    (414.46417, -222.33427),
]  # Sample checkpoints

r = 15  # Maximum radius the checkpoints can move
optimized_checkpoints = optimize_raceline(checkpoints, r)

# Plot original and optimized raceline
plot_raceline(checkpoints, optimized_checkpoints)
