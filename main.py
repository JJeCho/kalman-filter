import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the uploaded files
flight_logs = pd.read_csv('./flight_logs.csv')
sensor_data = pd.read_csv('./sensor_data.csv')

# Simulation Parameters
np.random.seed(42)
timesteps = len(flight_logs)  # Use length of flight_logs as timesteps
dt = 60  # Time interval between steps (seconds)

# Use positions, velocities, and accelerations from the flight logs
true_positions = flight_logs[['Latitude', 'Longitude', 'Altitude']].to_numpy()
true_velocities = np.zeros((timesteps, 3))  # Placeholder, calculate velocity if needed
true_accelerations = sensor_data[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()

# Generate GPS data by adding noise to the true positions
def generate_gps_data(true_position, noise_std=0.5):
    noise = np.random.normal(0, noise_std, size=true_position.shape)
    return true_position + noise

gps_data = np.array([generate_gps_data(pos) for pos in true_positions])

# Generate accelerometer data with additional noise
def generate_accelerometer_data(true_acceleration, noise_std=0.05, drift=0.0):
    noise = np.random.normal(0, noise_std, size=true_acceleration.shape)
    drift_component = drift * np.ones_like(true_acceleration)
    return true_acceleration + noise + drift_component

accelerometer_data = np.array([generate_accelerometer_data(accel) for accel in true_accelerations])

# Kalman Filter Implementation with Adjustments
def kalman_filter(timesteps, dt, gps_data, accel_data):
    state = np.zeros(6)
    P = np.diag([0.6, 0.6, 0.6, 0.05, 0.05, 0.05])


    F = np.array([
        [1, 0, 0, dt,  0,  0],
        [0, 1, 0,  0, dt,  0],
        [0, 0, 1,  0,  0, dt],
        [0, 0, 0,  1,  0,  0],
        [0, 0, 0,  0,  1,  0],
        [0, 0, 0,  0,  0,  1],
    ])

    B = np.array([
        [0.5 * dt ** 2,               0,               0],
        [              0, 0.5 * dt ** 2,               0],
        [              0,               0, 0.5 * dt ** 2],
        [            dt,               0,               0],
        [              0,             dt,               0],
        [              0,               0,             dt],
    ])

    H_gps = np.array([
        [1, 0, 0, 0, 0, 0],  # Position x
        [0, 1, 0, 0, 0, 0],  # Position y
        [0, 0, 1, 0, 0, 0],  # Position z
    ])

    R_u = np.diag([0.0005, 0.0005, 0.0005])
    Q = B @ R_u @ B.T
    R_gps = np.eye(3) * 20
    kalman_estimates = []

    for t in range(timesteps):
        u = accel_data[t]
        state = F @ state + B @ u
        P = F @ P @ F.T + Q

        z = gps_data[t]
        y = z - (H_gps @ state)  # Measurement residual
        S = H_gps @ P @ H_gps.T + R_gps  # Residual covariance
        K = P @ H_gps.T @ np.linalg.inv(S)  # Kalman gain

        state = state + K @ y
        P = (np.eye(6) - K @ H_gps) @ P
        kalman_estimates.append(state.copy())

    return np.array(kalman_estimates)

# Run Kalman Filter
kalman_estimates = kalman_filter(timesteps, dt, gps_data, accelerometer_data)

# Visualization
plt.figure(figsize=(12, 8))

# 3D Trajectory Plot
ax = plt.subplot(2, 1, 1, projection='3d')
ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label="True Path", color='blue')
ax.scatter(gps_data[:, 0], gps_data[:, 1], gps_data[:, 2], color='red', alpha=0.6, label="GPS Data (Noisy)")
ax.plot(kalman_estimates[:, 0], kalman_estimates[:, 1], kalman_estimates[:, 2], label="Kalman Estimate", color='green')
ax.set_title("3D Trajectory")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()

# Position Error Plot
plt.subplot(2, 1, 2)
position_errors = np.linalg.norm(kalman_estimates[:, 0:3] - true_positions, axis=1)
plt.plot(position_errors, label="Position Estimation Error")
plt.title("Position Estimation Error Over Time")
plt.xlabel("Time Step")
plt.ylabel("Error (meters)")
plt.legend()

plt.tight_layout()
plt.show()
