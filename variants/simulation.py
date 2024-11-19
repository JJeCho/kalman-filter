from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
np.random.seed(42)
timesteps = 1000  # Number of time steps in the simulation
dt = 0.1  # Time interval between steps (seconds)

# Function to select movement patterns
def select_movement_pattern(pattern):
    if pattern == "linear_constant_velocity":
        initial_position = np.array([0.0, 0.0, 0.0])
        initial_velocity = np.array([1.0, 0.5, 0.0])
        acceleration = np.array([0.0, 0.0, 0.0])
        return simulate_movement_path(timesteps, dt, initial_position, initial_velocity, acceleration)

    elif pattern == "linear_acceleration":
        initial_position = np.array([0.0, 0.0, 0.0])
        initial_velocity = np.array([0.5, 0.0, 0.0])
        acceleration = np.array([0.1, 0.0, 0.0])
        return simulate_movement_path(timesteps, dt, initial_position, initial_velocity, acceleration)

    elif pattern == "circular":
        radius = 5.0
        angular_velocity = 0.2
        return simulate_circular_path(timesteps, dt, radius, angular_velocity), np.zeros((timesteps, 3))

    elif pattern == "spiral":
        radius = 5.0
        angular_velocity = 0.2
        vertical_velocity = 0.1
        return simulate_spiral_path(timesteps, dt, radius, angular_velocity, vertical_velocity), np.zeros((timesteps, 3))

    elif pattern == "oscillatory":
        amplitude = 3.0
        frequency = 0.5
        return simulate_oscillatory_path(timesteps, dt, amplitude, frequency), np.zeros((timesteps, 3))

    elif pattern == "random_walk":
        step_size = 0.1
        return simulate_random_walk(timesteps, step_size), np.zeros((timesteps, 3))

    elif pattern == "figure_eight":
        amplitude = 3.0
        frequency = 0.5
        return simulate_figure_eight_path(timesteps, dt, amplitude, frequency), np.zeros((timesteps, 3))

# Define individual movement functions
def simulate_movement_path(timesteps, dt, initial_position, velocity, acceleration):
    positions = [initial_position]
    velocities = [velocity]
    for t in range(1, timesteps):
        velocity = velocities[-1] + acceleration * dt
        position = positions[-1] + velocity * dt + 0.5 * acceleration * (dt ** 2)
        positions.append(position)
        velocities.append(velocity)
    return np.array(positions), np.array(velocities)

def simulate_circular_path(timesteps, dt, radius, angular_velocity):
    positions = []
    for t in range(timesteps):
        angle = angular_velocity * t * dt
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        positions.append([x, y, z])
    return np.array(positions)


def simulate_spiral_path(timesteps, dt, radius, angular_velocity, vertical_velocity):
    positions = []
    for t in range(timesteps):
        angle = angular_velocity * t * dt
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = vertical_velocity * t * dt
        positions.append([x, y, z])
    return np.array(positions)

def simulate_oscillatory_path(timesteps, dt, amplitude, frequency):
    positions = []
    for t in range(timesteps):
        x = amplitude * np.sin(2 * np.pi * frequency * t * dt)
        y = 0.0
        z = 0.0
        positions.append([x, y, z])
    return np.array(positions)

def simulate_random_walk(timesteps, step_size):
    positions = [np.array([0.0, 0.0, 0.0])]
    for t in range(1, timesteps):
        step = np.random.uniform(-step_size, step_size, 3)
        new_position = positions[-1] + step
        positions.append(new_position)
    return np.array(positions)

def simulate_figure_eight_path(timesteps, dt, amplitude, frequency):
    positions = []
    for t in range(timesteps):
        x = amplitude * np.sin(2 * np.pi * frequency * t * dt)
        y = amplitude * np.sin(4 * np.pi * frequency * t * dt)
        z = 0.0
        positions.append([x, y, z])
    return np.array(positions)


# Sensor simulation functions
def generate_gps_data(true_position, noise_std=0.5):
    noise = np.random.normal(0, noise_std, size=true_position.shape)
    return true_position + noise

def generate_accelerometer_data(true_acceleration, noise_std=0.05, drift=0.01):
    noise = np.random.normal(0, noise_std, size=true_acceleration.shape)
    drift_component = drift * np.ones_like(true_acceleration)
    return true_acceleration + noise + drift_component

def generate_gyroscope_data(true_angular_velocity, noise_std=0.01, bias=0.02):
    noise = np.random.normal(0, noise_std, size=true_angular_velocity.shape)
    return true_angular_velocity + noise + bias


# Choose movement pattern (change pattern name to test others)
pattern = "spiral"  # Options: linear_constant_velocity, linear_acceleration, circular, spiral, oscillatory, random_walk, figure_eight
true_positions, true_velocities = select_movement_pattern(pattern)

# Simulating Sensor Data
gps_data = np.array([generate_gps_data(pos) for pos in true_positions])
acceleration = np.gradient(true_velocities, axis=0) / dt  # Estimated acceleration from velocity gradient
accelerometer_data = np.array([generate_accelerometer_data(accel) for accel in acceleration])

# Generate simulated angular velocity
angular_velocity = np.array([0.1, 0.05, 0.0])  # Constant angular velocity
gyroscope_data = np.array([generate_gyroscope_data(angular_velocity) for _ in range(timesteps)])

# Visualization
fig = plt.figure(figsize=(15, 12))

# 3D GPS Data Plot
ax = fig.add_subplot(3, 1, 1, projection='3d')
ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label="True Path")
ax.scatter(gps_data[:, 0], gps_data[:, 1], gps_data[:, 2], color='r', alpha=0.6, label="GPS Data (Noisy)")
ax.set_title("Simulated GPS Data (3D Trajectory)")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()

# Accelerometer Data Plot (X, Y, Z)
axes = ['X', 'Y', 'Z']
for i in range(3):
    plt.subplot(3, 3, 4 + i)
    plt.plot(acceleration[:, i], label=f"True Acceleration ({axes[i]})")
    plt.plot(accelerometer_data[:, i], label=f"Accelerometer Data (Noisy)", alpha=0.7)
    plt.title(f"Simulated Accelerometer Data ({axes[i]}-axis)")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()

# Gyroscope Data Plot (X, Y, Z)
for i in range(3):
    plt.subplot(3, 3, 7 + i)
    plt.plot([angular_velocity[i]] * timesteps, label=f"True Angular Velocity ({axes[i]})")
    plt.plot(gyroscope_data[:, i], label=f"Gyroscope Data (Noisy)", alpha=0.7)
    plt.title(f"Simulated Gyroscope Data ({axes[i]}-axis)")
    plt.xlabel("Time Step")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()

plt.tight_layout()
plt.show()
