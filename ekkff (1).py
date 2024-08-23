# import numpy as np
# from filterpy.kalman import KalmanFilter

# def create_kalman_filter(dim_x, dim_z, dt, process_var):
#     kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
#     kf.x = np.zeros(dim_x)  # initial state (position and velocity)
#     kf.F = np.array([[1, dt],  # state transition matrix
#                      [0, 1]])
#     kf.H = np.array([[1, 0]])  # measurement function
#     kf.P *= 10.  # covariance matrix
#     kf.R = np.eye(dim_z)  # measurement uncertainty
#     kf.Q = np.eye(dim_x) * process_var  # process uncertainty
#     return kf

# # Define parameters
# dt = 1.0  # time step
# process_var = 1e-4  # process variance
# dim_x = 2  # state vector dimension (position, velocity)
# dim_z = 1  # measurement vector dimension (position)

# # Initialize Kalman filters for each GPS module
# kf1 = create_kalman_filter(dim_x=dim_x, dim_z=dim_z, dt=dt, process_var=process_var)
# kf2 = create_kalman_filter(dim_x=dim_x, dim_z=dim_z, dt=dt, process_var=process_var)
# kf3 = create_kalman_filter(dim_x=dim_x, dim_z=dim_z, dt=dt, process_var=process_var)

# # Function to update Kalman filters with new measurements
# def update_kalman_filters(kfs, measurements, errors):
#     estimates = []
#     for kf, measurement, error in zip(kfs, measurements, errors):
#         kf.R = np.array([[error]])  # update measurement uncertainty
#         kf.predict()
#         kf.update(measurement)
#         estimates.append(kf.x[0])  # only the position component
#     return estimates

# # Simulated GPS measurements with different errors at each time step
# gps_measurements = [
#     [10.0, 10.2, 9.8, 10.1, 10.3],  # GPS 1 measurements
#     [9.8, 10.0, 9.9, 10.2, 10.1],   # GPS 2 measurements
#     [10.1, 10.3, 10.0, 10.1, 10.2]  # GPS 3 measurements
# ]

# # Simulated measurement errors for each GPS at each time step
# measurement_errors = [
#     [0.2, 0.3, 0.1, 0.2, 0.25],  # GPS 1 errors
#     [0.15, 0.1, 0.2, 0.1, 0.2],  # GPS 2 errors
#     [0.25, 0.3, 0.2, 0.3, 0.1]   # GPS 3 errors
# ]

# # Initialize Kalman filters list
# kfs = [kf1, kf2, kf3]

# # Process the measurements and fuse them using Kalman filters
# fused_estimates = []
# for i in range(len(gps_measurements[0])):
#     measurements = [gps_measurements[0][i], gps_measurements[1][i], gps_measurements[2][i]]
#     errors = [measurement_errors[0][i], measurement_errors[1][i], measurement_errors[2][i]]
#     estimates = update_kalman_filters(kfs, measurements, errors)
#     fused_estimate = np.mean(estimates)  # simple average of the estimates
#     fused_estimates.append(fused_estimate)

# print("Fused Estimates:", fused_estimates)














# ****************************************************************************************




# import numpy as np

# dt = 1.0 

# F = np.array([
#     [1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0],
#     [0, 1, dt, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0, 0],
#     [0, 0, 0, 0, 1, dt, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0]
#     [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2]
#     [0, 0, 0, 0, 0, 0, 0, 1, dt]
#     [0, 0, 0, 0, 0, 0, 0, 0,1]
# ])

# # Process noise covariance matrix Q
# sigma_a = 1 
# Q = sigma_a * np.array([
#     [dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
#     [dt**3/2, dt**2, dt, 0, 0, 0],
#     [dt**2/2, dt, 1, 0, 0, 0],
#     [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
#     [0, 0, 0, dt**3/2, dt**2, dt],
#     [0, 0, 0, dt**2/2, dt, 1]
# ])

# # Measurement matrix H
# H = np.array([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0]
# ])

# # Measurement noise covariance matrix R
# R1 = np.diag([1**2, 1**2])  
# R2 = np.diag([0.5**2, 0.5**2]) 

# # Initial state estimate and covariance
# x = np.zeros((6, 1))  # [x, x_dot, x_ddot, y, y_dot, y_ddot]
# P = np.eye(6) * 1000  # initial large uncertainty

# def kalman_filter(x, P, z, R):
#     # Predict
#     x_pred = F @ x
#     P_pred = F @ P @ F.T + Q
    
#     # Update
#     y = z - (H @ x_pred)
#     S = H @ P_pred @ H.T + R
#     K = P_pred @ H.T @ np.linalg.inv(S)
#     x_upd = x_pred + K @ y
#     P_upd = (np.eye(len(K)) - K @ H) @ P_pred
    
#     return x_upd, P_upd

# # Example measurements from GPS1 and GPS2
# measurements_gps1 = np.array([
#     [1, 2],
#     [1.1, 2.1],
#     [0.9, 2.0]
# ])

# measurements_gps2 = np.array([
#     [1.05, 2.05],
#     [1.15, 2.15],
#     [1.0, 2.0]
# ])

# for i in range(len(measurements_gps1)):
#     if np.linalg.norm(R1) < np.linalg.norm(R2):
#         z = measurements_gps1[i].reshape(-1, 1)
#         R = R1
#     else:
#         z = measurements_gps2[i].reshape(-1, 1)
#         R = R2
    
#     x, P = kalman_filter(x, P, z, R)
#     print(f"Updated state at step {i}:\n{x}\n")







# ********************************************************************************************



# import numpy as np
# import matplotlib.pyplot as plt

# # Define the state transition matrix F
# def state_transition_matrix(delta_t):
#     dt = delta_t
#     dt2 = dt**2 / 2
#     F = np.array([[1, dt, dt2, 0, 0, 0],
#                   [0, 1,  dt, 0, 0, 0],
#                   [0, 0,  1, 0, 0, 0],
#                   [0, 0, 0, 1, dt, dt2],
#                   [0, 0, 0, 0, 1,  dt],
#                   [0, 0, 0, 0, 0,  1]])
#     return F

# # Define the process noise covariance matrix Q
# def process_noise_covariance(delta_t, sigma_a):
#     dt = delta_t
#     dt2 = dt**2
#     dt3 = dt**3
#     dt4 = dt**4
#     Q = sigma_a**2 * np.array([[dt4/4, dt3/2, dt2/2,     0,     0,     0],
#                                [dt3/2,    dt2,     dt,     0,     0,     0],
#                                [dt2/2,     dt,      1,     0,     0,     0],
#                                [    0,     0,      0, dt4/4, dt3/2, dt2/2],
#                                [    0,     0,      0, dt3/2,    dt2,    dt],
#                                [    0,     0,      0,  dt2/2,     dt,     1]])
#     return Q

# # Define the initial state estimate and its covariance matrix
# initial_state_estimate = np.array([301.5, 0, 0, -401.46, 0, 0])
# initial_covariance_estimate = np.eye(6) * 1000  # Large initial uncertainty

# # Define the observation matrix H
# H = np.array([[1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0]])

# # Define the measurement noise covariance matrix R
# R = np.array([[1, 0],
#               [0, 1]]) * 10  # Assuming some measurement noise

# # Define the measurement data
# measurements = np.array([
#     [301.5, -401.46],
#     [298.23, -375.44],
#     [297.83, -346.15],
#     [300.42, -320.2],
#     [301.94, -300.08],
#     [299.5, -274.12],
#     [305.98, -253.45],
#     [301.25, -226.4],
#     [299.73, -200.65],
#     [299.2, -171.62],
#     [298.62, -152.11],
#     [301.84, -125.19],
#     [299.6, -93.4],
#     [295.3, -74.79],
#     [299.3, -49.12],
#     [301.95, -28.73],
#     [296.3, 2.99],
#     [295.11, 25.65],
#     [295.12, 49.86],
#     [289.9, 72.87],
#     [283.51, 96.34],
#     [276.42, 120.4],
#     [264.22, 144.69],
#     [250.25, 168.06],
#     [236.66, 184.99],
#     [217.47, 205.11],
#     [199.75, 221.82],
#     [179.7, 238.3],
#     [160, 253.02],
#     [140.92, 267.19],
#     [113.53, 270.71],
#     [93.68, 285.86],
#     [69.71, 288.48],
#     [45.93, 292.9],
#     [20.87, 298.77]
# ])

# # Define the time step between measurements
# delta_t = 1  # Assuming 1 second for simplicity

# # Define the process noise variance
# sigma_a = 0.1  # Assumed constant acceleration noise

# # Initialize state and covariance
# x = initial_state_estimate
# P = initial_covariance_estimate

# # Storage for estimates
# estimates = []

# for z in measurements:
#     # Prediction step
#     F = state_transition_matrix(delta_t)
#     Q = process_noise_covariance(delta_t, sigma_a)
#     x = F @ x
#     P = F @ P @ F.T + Q
    
#     # Measurement update step
#     y = z - H @ x
#     S = H @ P @ H.T + R
#     K = P @ H.T @ np.linalg.inv(S)
#     x = x + K @ y
#     P = P - K @ H @ P
    
#     # Store the estimate
#     estimates.append(x)

# # Convert estimates to a NumPy array for easier handling
# estimates = np.array(estimates)

# # Extract position estimates
# x_estimates = estimates[:, 0]
# y_estimates = estimates[:, 3]

# # Print the estimated positions
# print("Estimated positions:")
# print("X:", x_estimates)
# print("Y:", y_estimates)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(measurements[:, 0], measurements[:, 1], 'ro', label='Measurements')
# plt.plot(x_estimates, y_estimates, 'b-', label='Kalman Filter Estimate')
# plt.xlabel('X position (m)')
# plt.ylabel('Y position (m)')
# plt.legend()
# plt.title('Kalman Filter Position Estimation')
# plt.show()



















# **************************************************************************************




# import numpy as np
# import matplotlib.pyplot as plt

# # Define the state transition matrix F
# def state_transition_matrix(delta_t):
#     dt = delta_t
#     dt2 = dt**2 / 2
#     F = np.array([[1, dt, dt2, 0, 0, 0, 0, 0, 0],
#                   [0, 1,  dt, 0, 0, 0, 0, 0, 0],
#                   [0, 0,  1, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 1, dt, dt2, 0, 0, 0],
#                   [0, 0, 0, 0, 1,  dt, 0, 0, 0],
#                   [0, 0, 0, 0, 0,  1, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 1, dt, dt2],
#                   [0, 0, 0, 0, 0, 0, 0, 1,  dt],
#                   [0, 0, 0, 0, 0, 0, 0, 0,  1]])
#     return F

# # Define the process noise covariance matrix Q
# def process_noise_covariance(delta_t, sigma_a):
#     dt = delta_t
#     dt2 = dt**2
#     dt3 = dt**3
#     dt4 = dt**4
#     Q = sigma_a**2 * np.array([[dt4/4, dt3/2, dt2/2,     0,     0,     0,     0,     0,     0],
#                                [dt3/2,    dt2,     dt,     0,     0,     0,     0,     0,     0],
#                                [dt2/2,     dt,      1,     0,     0,     0,     0,     0,     0],
#                                [    0,     0,      0, dt4/4, dt3/2, dt2/2,     0,     0,     0],
#                                [    0,     0,      0, dt3/2,    dt2,    dt,     0,     0,     0],
#                                [    0,     0,      0,  dt2/2,     dt,     1,     0,     0,     0],
#                                [    0,     0,      0,     0,     0,     0, dt4/4, dt3/2, dt2/2],
#                                [    0,     0,      0,     0,     0,     0, dt3/2,    dt2,    dt],
#                                [    0,     0,      0,     0,     0,     0,  dt2/2,     dt,     1]])
#     return Q

# # Define the initial state estimate and its covariance matrix
# initial_state_estimate = np.array([301.5, 0, 0, -401.46, 0, 0, 0, 0, 0])  # Add z initial values
# initial_covariance_estimate = np.eye(9) * 1000  # Large initial uncertainty

# # Define the observation matrix H
# H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0, 0]])  # Include z observation

# # Define the measurement noise covariance matrix R
# R = np.eye(3) * 10  # Assuming some measurement noise

# # Define the measurement data
# measurements = np.array([
#     [301.5, -401.46, 100],
#     [298.23, -375.44, 110],
#     [297.83, -346.15, 120],
#     [300.42, -320.2, 130],
#     [301.94, -300.08, 140],
#     [299.5, -274.12, 150],
#     [305.98, -253.45, 160],
#     [301.25, -226.4, 170],
#     [299.73, -200.65, 180],
#     [299.2, -171.62, 190],
#     [298.62, -152.11, 200],
#     [301.84, -125.19, 210],
#     [299.6, -93.4, 220],
#     [295.3, -74.79, 230],
#     [299.3, -49.12, 240],
#     [301.95, -28.73, 250],
#     [296.3, 2.99, 260],
#     [295.11, 25.65, 270],
#     [295.12, 49.86, 280],
#     [289.9, 72.87, 290],
#     [283.51, 96.34, 300],
#     [276.42, 120.4, 310],
#     [264.22, 144.69, 320],
#     [250.25, 168.06, 330],
#     [236.66, 184.99, 340],
#     [217.47, 205.11, 350],
#     [199.75, 221.82, 360],
#     [179.7, 238.3, 370],
#     [160, 253.02, 380],
#     [140.92, 267.19, 390],
#     [113.53, 270.71, 400],
#     [93.68, 285.86, 410],
#     [69.71, 288.48, 420],
#     [45.93, 292.9, 430],
#     [20.87, 298.77, 440]
# ])

# # Define the time step between measurements
# delta_t = 1  # Assuming 1 second for simplicity

# # Define the process noise variance
# sigma_a = 0.1  # Assumed constant acceleration noise

# # Initialize state and covariance
# x = initial_state_estimate
# P = initial_covariance_estimate

# # Storage for estimates
# estimates = []

# for z in measurements:
#     # Prediction step
#     F = state_transition_matrix(delta_t)
#     Q = process_noise_covariance(delta_t, sigma_a)
#     x = F @ x
#     P = F @ P @ F.T + Q
    
#     # Measurement update step
#     y = z - H @ x
#     S = H @ P @ H.T + R
#     K = P @ H.T @ np.linalg.inv(S)
#     x = x + K @ y
#     P = P - K @ H @ P
    
#     # Store the estimate
#     estimates.append(x)

# # Convert estimates to a NumPy array for easier handling
# estimates = np.array(estimates)

# # Extract position estimates
# x_estimates = estimates[:, 0]
# y_estimates = estimates[:, 3]
# z_estimates = estimates[:, 6]

# # Print the estimated positions
# print("Estimated positions:")
# print("X:", x_estimates)
# print("Y:", y_estimates)
# print("Z:", z_estimates)

# # Plot the results
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(measurements[:, 0], measurements[:, 1], measurements[:, 2], 'ro', label='Measurements')
# ax.plot(x_estimates, y_estimates, z_estimates, 'b-', label='Kalman Filter Estimate')
# ax.set_xlabel('X position (m)')
# ax.set_ylabel('Y position (m)')
# ax.set_zlabel('Z position (m)')
# ax.legend()
# ax.set_title('Kalman Filter 3D Position Estimation')
# plt.show()

















# *******************************************************************************










import numpy as np
import matplotlib.pyplot as plt

# Define the state transition matrix F
def state_transition_matrix(delta_t):
    dt = delta_t
    dt2 = dt**2 / 2
    F = np.array([[1, dt, dt2, 0, 0, 0, 0, 0, 0],
                  [0, 1,  dt, 0, 0, 0, 0, 0, 0],
                  [0, 0,  1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, dt, dt2, 0, 0, 0],
                  [0, 0, 0, 0, 1,  dt, 0, 0, 0],
                  [0, 0, 0, 0, 0,  1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, dt, dt2],
                  [0, 0, 0, 0, 0, 0, 0, 1,  dt],
                  [0, 0, 0, 0, 0, 0, 0, 0,  1]])
    return F

# Define the process noise covariance matrix Q
def process_noise_covariance(delta_t, sigma_a):
    dt = delta_t
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    Q = sigma_a**2 * np.array([[dt4/4, dt3/2, dt2/2,     0,     0,     0,     0,     0,     0],
                               [dt3/2,    dt2,     dt,     0,     0,     0,     0,     0,     0],
                               [dt2/2,     dt,      1,     0,     0,     0,     0,     0,     0],
                               [    0,     0,      0, dt4/4, dt3/2, dt2/2,     0,     0,     0],
                               [    0,     0,      0, dt3/2,    dt2,    dt,     0,     0,     0],
                               [    0,     0,      0,  dt2/2,     dt,     1,     0,     0,     0],
                               [    0,     0,      0,     0,     0,     0, dt4/4, dt3/2, dt2/2],
                               [    0,     0,      0,     0,     0,     0, dt3/2,    dt2,    dt],
                               [    0,     0,      0,     0,     0,     0,  dt2/2,     dt,     1]])
    return Q

initial_state_estimate = np.array([301.5, 0, 0, -401.46, 0, 0, 100, 0, 0])  # initial
initial_covariance_estimate = np.eye(9) * 1000  # initial

H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # gps1 x
              [0, 0, 0, 1, 0, 0, 0, 0, 0],  # gps1 y
              [0, 0, 0, 0, 0, 0, 1, 0, 0],  # gps1 z
              [1, 0, 0, 0, 0, 0, 0, 0, 0],  # gps2 x
              [0, 0, 0, 1, 0, 0, 0, 0, 0],  # gps2 y
              [0, 0, 0, 0, 0, 0, 1, 0, 0]]) # gps2 z


R = np.diag([10, 10, 10, 20, 20, 20])  # measurement error

# gps1 measurements
measurements_gps1 = np.array([
    [301.5, -401.46, 100],
    [298.23, -375.44, 110],
    [297.83, -346.15, 120],
    [300.42, -320.2, 130],
    [301.94, -300.08, 140],
    [299.5, -274.12, 150],
    [305.98, -253.45, 160],
    [301.25, -226.4, 170],
    [299.73, -200.65, 180],
    [299.2, -171.62, 190],
    [298.62, -152.11, 200],
    [301.84, -125.19, 210],
    [299.6, -93.4, 220],
    [295.3, -74.79, 230],
    [299.3, -49.12, 240],
    [301.95, -28.73, 250],
    [296.3, 2.99, 260],
    [295.11, 25.65, 270],
    [295.12, 49.86, 280],
    [289.9, 72.87, 290],
    [283.51, 96.34, 300],
    [276.42, 120.4, 310],
    [264.22, 144.69, 320],
    [250.25, 168.06, 330],
    [236.66, 184.99, 340],
    [217.47, 205.11, 350],
    [199.75, 221.82, 360],
    [179.7, 238.3, 370],
    [160, 253.02, 380],
    [140.92, 267.19, 390],
    [113.53, 270.71, 400],
    [93.68, 285.86, 410],
    [69.71, 288.48, 420],
    [45.93, 292.9, 430],
    [20.87, 298.77, 440]
])

# gps2 measurements
measurements_gps2 = np.array([
    [300.5, -402.46, 102],
    [297.23, -374.44, 112],
    [298.83, -345.15, 122],
    [299.42, -319.2, 132],
    [302.94, -299.08, 142],
    [298.5, -273.12, 152],
    [306.98, -252.45, 162],
    [300.25, -125.4, 172],
    [300.73, -199.65, 182],
    [298.2, -170.62, 192],
    [297.62, -151.11, 202],
    [300.84, -124.19, 212],
    [298.6, -92.4, 222],
    [294.3, -73.79, 232],
    [298.3, -48.12, 242],
    [300.95, -27.73, 252],
    [295.3, 3.99, 262],
    [294.11, 26.65, 272],
    [294.12, 50.86, 282],
    [288.9, 73.87, 292],
    [282.51, 97.34, 302],
    [275.42, 121.4, 312],
    [263.22, 145.69, 322],
    [249.25, 169.06, 332],
    [235.66, 185.99, 342],
    [216.47, 206.11, 352],
    [198.75, 222.82, 362],
    [178.7, 239.3, 372],
    [159, 254.02, 382],
    [139.92, 268.19, 392],
    [112.53, 271.71, 402],
    [92.68, 286.86, 412],
    [68.71, 289.48, 422],
    [44.93, 293.9, 432],
    [19.87, 299.77, 442]
])

delta_t = 1 
# process noise
sigma_a = 0.1 

# Initialize state and covariance
x = initial_state_estimate
P = initial_covariance_estimate

estimates = []
i = 0
for z1, z2 in zip(measurements_gps1, measurements_gps2):
    i +=1
    if i == 4:
        R = np.diag([1000, 1000, 1000, 20, 20, 20])

    else:
        R = np.diag([10, 10, 10, 20, 20, 20])
        
    z = np.hstack((z1, z2))
    
    # Prediction step
    F = state_transition_matrix(delta_t)
    Q = process_noise_covariance(delta_t, sigma_a)
    x = F @ x
    P = F @ P @ F.T + Q
    
    # Measurement update step
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = P - K @ H @ P
    
    estimates.append(x)

estimates = np.array(estimates)

x_estimates = estimates[:, 0]
y_estimates = estimates[:, 3]
z_estimates = estimates[:, 6]

# Print the estimated positions
print("Estimated positions:")
print("X:", x_estimates)
print("Y:", y_estimates)
print("Z:", z_estimates)

# Plot the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(measurements_gps1[:, 0], measurements_gps1[:, 1], measurements_gps1[:, 2], 'ro', label='GPS1 Measurements')
ax.plot(measurements_gps2[:, 0], measurements_gps2[:, 1], measurements_gps2[:, 2], 'go', label='GPS2 Measurements')
ax.plot(x_estimates, y_estimates, z_estimates, 'b-', label='Kalman Filter Estimate')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.set_zlabel('Z position (m)')
ax.legend()
ax.set_title('Kalman Filter 3D Position Estimation with Two GPS Modules')
plt.show()
