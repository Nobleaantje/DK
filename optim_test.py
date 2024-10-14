import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from copy import deepcopy

def compute_curvature(x, y):
    """Compute the curvature for the given x and y points of the raceline."""
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    # curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    return curvature

def compute_angles(checkpoints):

    checkpoints = deepcopy(checkpoints.reshape(-1, 2))

    nTargets = len(checkpoints)

    angleList = [0] * nTargets

    for section in range(nTargets):

        id1 = section
        id2 = (section - 1) % (nTargets - 1)
        id3 = (section + 2) % (nTargets - 1)

        point1 = checkpoints[id1]
        point2 = checkpoints[id2]
        point3 = checkpoints[id3]

        a = [point1[0] - point2[0], point1[1] - point2[1]]
        b = [point1[0] - point3[0], point1[1] - point3[1]]

        lengthA = math.sqrt(a[0]**2 + a[1]**2)
        lengthB = math.sqrt(b[0]**2 + b[1]**2)

        angleList[section] = math.acos( np.dot(a,b) / ( lengthA * lengthB ) )

    return angleList
        

def raceline_objective(checkpoints, n_points=100):
    """Objective function to minimize the curvature of the raceline."""
    x = checkpoints[::2]
    y = checkpoints[1::2]
    
    # Ensure the raceline forms a loop by repeating the first point
    # x = np.append(x, x[0])
    # y = np.append(y, y[0])
    
    # Fit a cubic spline through the checkpoints to get a smooth raceline
    # spline = CubicSpline(np.linspace(0, 1, len(x)), np.vstack((x, y)), axis=1, bc_type='periodic')
    
    # Sample points along the spline
    # t_values = np.linspace(0, 1, n_points)
    # raceline_x, raceline_y = spline(t_values)
    
    # Compute curvature
    # curvature = compute_curvature(x, y)
    # curvature = [x*1000 for x in curvature]
    # curvature = curvature*100
    # racelineCheckpoints = [raceline_x, raceline_y]

    curvature = 3.1415 - np.array(compute_angles(checkpoints))
    angleList = np.array(compute_angles(checkpoints))

    tot_dist = 0

    tmpcheckpoint = checkpoints.reshape(-1, 2)

    nTargets = len(tmpcheckpoint)

    time = 0
    velPrev = 9999

    for section, angle in zip(range(nTargets),angleList):
        section1ID = (section ) % (nTargets - 1)
        section2ID = (section + 1 ) % (nTargets - 1)
        sectionPrev = (section - 1 ) % (nTargets - 1)

        acc = 100

        target1 = tmpcheckpoint[section1ID]
        target2 = tmpcheckpoint[section2ID]
        targetPrev = tmpcheckpoint[sectionPrev]

        distPrev = np.linalg.norm(target1 - targetPrev)

        t = ((2*acc*distPrev + velPrev**2)**(1/2)-velPrev)/acc
        velAcc = velPrev + acc*t

        tot_dist += np.linalg.norm(target2 - target1)

        vel = ( angle / 3.1415 )**2 * 300



        vel = min(vel,velAcc)
        velPrev = vel
        # Is this the correct calculation?
        time += np.linalg.norm(target2 - target1) / vel

    # Minimize the sum of curvature
    return np.sum(time**2)

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

        xpoints = optimized_checkpoints[:,0]
        ypoints = optimized_checkpoints[:,1]

        xpoints = np.append(xpoints,xpoints[0])
        ypoints = np.append(ypoints,ypoints[0])

        xs, ys = fit_spline_loop(xpoints, ypoints)
        
        # plt.plot(xs, ys, 'ko-', label='Optimized Raceline')
    
    plt.legend()
    plt.title("Raceline Optimization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

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

checkpoints = [(1389.19592, 440.8442), (1466.63906, 452.31724), (1518.26782, 497.25338), (1524.00434, 556.53084), (1487.673, 628.23744), (1375.8107, 692.29534), (1235.26576, 743.9241), (800.2457, 854.83032), (405.38134, 919.8443), (262.92422, 894.02992), (205.55892, 829.97202), (211.29546, 782.16762), (272.4851, 764.00196), (620.50114, 769.73848), (802.15786, 742.96802), (848.0501, 711.4171), (872.90838, 643.53486), (865.25968, 575.6526), (820.32354, 528.8043), (775.3874, 509.68254), (159.666704, 445.62464), (97.52098, 406.42502), (71.706604, 336.6306), (87.004012, 276.39704), (132.896238, 232.417), (210.33936, 219.03176), (482.82446, 243.89006), (566.9602, 227.63656), (768.69478, 89.003792), (833.70878, 75.61856), (925.49322, 88.047704), (1219.9683400000001, 193.21739), (1373.8985200000002, 220.94394), (1419.7907400000001, 205.64654), (1562.2478600000002, 117.686434), (1642.5592600000002, 97.608586), (1746.7728600000003, 111.949906), (1818.4794600000002, 174.095628), (1829.9525200000003, 240.0657), (1792.6650800000002, 275.44096), (1681.7588600000001, 283.08966), (760.0900000000001, 281.17748), (678.8225200000002, 309.86014), (640.5789800000002, 351.928), (650.1398600000002, 395.90806), (716.10994, 429.37114), (828.92834, 444.66854)]

checkpoints = [(x,-y) for x,y in checkpoints]

r = 11.25  # Maximum radius the checkpoints can move
optimized_checkpoints = optimize_raceline(checkpoints, r)
# print([(x,y) for x,y in optimized_checkpoints])

# Plot original and optimized raceline
plot_raceline(checkpoints, optimized_checkpoints)
