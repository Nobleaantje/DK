import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def optimal_race_line_loop(checkpoints, radius):
    # Extract x and y coordinates from checkpoints
    x_points = np.array([point[0] for point in checkpoints])
    y_points = np.array([point[1] for point in checkpoints])

    # Add the first checkpoint to the end of the list to form a loop
    x_points = np.append(x_points, x_points[0])
    y_points = np.append(y_points, y_points[0])

    # Generate a periodic cubic spline for the x and y coordinates
    t = np.linspace(0, 1, len(x_points))  # A normalized parameter for the spline
    cs_x = CubicSpline(t, x_points, bc_type='periodic')  # Periodic boundary condition
    cs_y = CubicSpline(t, y_points, bc_type='periodic')  # Periodic boundary condition

    # Generate dense points on the spline
    t_fine = np.linspace(0, 1, 1000)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)

    # Adjust spline to stay within radius at each checkpoint
    adjusted_x = []
    adjusted_y = []
    
    for i in range(len(x_points)):
        # Compute the tangent vector at the checkpoint (derivative of the spline)
        dx_dt = cs_x(t[i], 1)  # First derivative wrt t for x
        dy_dt = cs_y(t[i], 1)  # First derivative wrt t for y
        
        # Compute the normal vector (perpendicular to the tangent)
        normal_x = -dy_dt
        normal_y = dx_dt
        normal_length = np.sqrt(normal_x**2 + normal_y**2)
        
        # Normalize the normal vector
        normal_x /= normal_length
        normal_y /= normal_length
        
        # Move the point along the normal direction by the specified radius
        new_x = x_points[i] + radius * normal_x
        new_y = y_points[i] + radius * normal_y
        
        adjusted_x.append(new_x)
        adjusted_y.append(new_y)

    # Return the new list of optimized points
    optimized_points = list(zip(adjusted_x, adjusted_y))

    # Plotting for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, label='Smooth Spline Path (Loop)', linestyle='--')
    plt.scatter(x_points, y_points, label='Checkpoints', color='red', zorder=5)
    plt.scatter(adjusted_x, adjusted_y, label='Adjusted Optimal Path', color='green', zorder=5)
    plt.legend()
    plt.title("Optimal Race Line with Loop Based on Checkpoints and Deviation Radius")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

    return optimized_points

# Example usage:
checkpoints = [(0, 0), (1, 2), (4, 3), (6, 1), (6, 5), (3,6), (-1,3)]
radius = 0.5  # Maximum allowed deviation
optimal_race_line_loop(checkpoints, radius)
