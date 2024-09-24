import numpy as np
import matplotlib.pyplot as plt

def compute_optimal_raceline(checkpoints, r):
    """
    Computes the optimal raceline through a set of checkpoints.
    
    Parameters:
    - checkpoints: List of tuples (x, y) representing the coordinates of the middle of the track.
    - r: The width of the track (distance from the center to the edge).
    
    Returns:
    - optimal_path: List of (x, y) coordinates representing the optimal raceline.
    """
    n = len(checkpoints)
    
    if n == 0:
        return []
    
    # Positions for edges of the track
    edge_positions = []
    for (x, y) in checkpoints:
        edge_positions.append([(x, y + r), (x, y - r)])  # Upper and Lower edges

    # Initialize the DP table
    dp = np.inf * np.ones((n, 2))  # Two edges per checkpoint
    dp[0, 0] = 0  # Starting from the upper edge of the first checkpoint (outer edge)
    dp[0, 1] = np.inf  # We will not consider starting from the lower edge

    # To reconstruct the path later
    path = np.zeros((n, 2), dtype=int)

    # Dynamic programming to find minimum distances
    for i in range(1, n):
        for j in range(2):  # Previous edge (0: upper, 1: lower)
            for k in range(2):  # Current edge
                distance = np.linalg.norm(np.array(edge_positions[i][k]) - np.array(edge_positions[i - 1][j]))
                if dp[i - 1, j] + distance < dp[i, k]:
                    dp[i, k] = dp[i - 1, j] + distance
                    path[i, k] = j  # Store the previous edge index

    # Always end at the upper edge of the last checkpoint
    last_edge = 0  # Upper edge of the last checkpoint

    # Backtrack to find the optimal path
    optimal_path = []
    for i in range(n - 1, -1, -1):
        optimal_path.append(edge_positions[i][last_edge])
        last_edge = path[i, last_edge]
    
    optimal_path.reverse()  # Reverse to get the correct order

    return optimal_path

def plot_raceline(checkpoints, track_width, optimal_path):
    """
    Plots the track, checkpoints, and optimal raceline.
    
    Parameters:
    - checkpoints: List of tuples (x, y) representing the coordinates of the middle of the track.
    - track_width: The width of the track.
    - optimal_path: List of (x, y) coordinates representing the optimal raceline.
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot checkpoints
    checkpoints = np.array(checkpoints)
    plt.plot(checkpoints[:, 0], checkpoints[:, 1], marker='o', color='blue', label='Checkpoints')
    
    # Plot track edges
    upper_edge = checkpoints + np.array([0, track_width])
    lower_edge = checkpoints - np.array([0, track_width])
    
    plt.plot(upper_edge[:, 0], upper_edge[:, 1], linestyle='--', color='red', label='Upper Edge')
    plt.plot(lower_edge[:, 0], lower_edge[:, 1], linestyle='--', color='green', label='Lower Edge')
    
    # Plot the optimal raceline
    optimal_path = np.array(optimal_path)
    plt.plot(optimal_path[:, 0], optimal_path[:, 1], color='orange', linewidth=2, label='Optimal Raceline')

    # Labels and title
    plt.title('Optimal Raceline through Checkpoints')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axhline(0, color='black', lw=0.5, ls='--')  # Add a center line for reference
    plt.axvline(0, color='black', lw=0.5, ls='--')  # Add a center line for reference
    plt.grid()
    plt.legend()
    plt.axis('equal')  # Equal scaling for X and Y axes
    plt.show()

# Example Usage
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
track_width = 10
optimal_raceline = compute_optimal_raceline(checkpoints, track_width)

print(optimal_raceline)

# Plotting the results
plot_raceline(checkpoints, track_width, optimal_raceline)
