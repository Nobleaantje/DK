import numpy as np
import matplotlib.pyplot as plt

def compute_optimal_raceline(checkpoints, track_width):
    raceline = []
    num_checkpoints = len(checkpoints)

    # Loop through each segment defined by checkpoints
    for i in range(num_checkpoints):
        p1 = np.array(checkpoints[i])
        p2 = np.array(checkpoints[(i + 1) % num_checkpoints])
        
        # Compute the direction vector from p1 to p2
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            continue  # Skip overlapping points
        
        # Normalize the direction vector
        direction /= distance
        
        # Perpendicular direction for the outer edge of the track
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Calculate the outer points for the arc
        outer_p1 = p1 + (track_width / 2) * perpendicular
        outer_p2 = p2 + (track_width / 2) * perpendicular
        
        # Compute the center of the arc
        mid_point = (outer_p1 + outer_p2) / 2
        radius = track_width / 2
        angle = np.arctan2(direction[1], direction[0])

        # Calculate the angle for the arc
        arc_points = 50  # Number of points to represent the arc
        angles = np.linspace(angle - np.pi/2, angle + np.pi/2, arc_points)
        
        # Generate points for the arc
        arc_x = mid_point[0] + radius * np.cos(angles)
        arc_y = mid_point[1] + radius * np.sin(angles)

        raceline.extend(zip(arc_x, arc_y))

    return np.array(raceline)

def plot_raceline(checkpoints, raceline):
    plt.figure(figsize=(10, 6))

    # Plot the track (middle line)
    checkpoints = np.array(checkpoints)
    plt.plot(checkpoints[:, 0], checkpoints[:, 1], 'ko-', label='Track Center')

    # Plot the raceline
    plt.plot(raceline[:, 0], raceline[:, 1], 'r-', linewidth=2, label='Optimal Raceline')

    # Mark the checkpoints
    plt.scatter(checkpoints[:, 0], checkpoints[:, 1], color='blue', label='Checkpoints')

    plt.title('Optimal Raceline through Checkpoints')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()

# Example usage:
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
track_width = 10  # Width of the track

raceline = compute_optimal_raceline(checkpoints, track_width)
plot_raceline(checkpoints, raceline)
