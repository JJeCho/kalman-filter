from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
np.random.seed(42)
timesteps = 1000  # Number of time steps in the simulation
dt = 0.1  # Time interval between steps (seconds)

# Initial GPS coordinates (approximate location over San Francisco)
initial_lat = 37.7749    # Degrees
initial_lon = -122.4194  # Degrees
initial_altitude = 1000  # Altitude in meters (approximate cruising altitude)

# Function to generate realistic GPS data with multiple movement patterns
def generate_realistic_gps_data(pattern, timesteps, initial_lat, initial_lon, initial_altitude, speed=250, climb_rate=1):
    latitudes = [initial_lat]
    longitudes = [initial_lon]
    altitudes = [initial_altitude]
    
    for t in range(1, timesteps):
        if pattern == "linear_constant_velocity":
            delta_lat = (speed * dt / 111320) * np.random.normal(1, 0.05)
            delta_lon = (speed * dt / (111320 * np.cos(np.radians(latitudes[-1])))) * np.random.normal(1, 0.05)
            new_alt = altitudes[-1]
        
        elif pattern == "linear_acceleration":
            speed += 0.1  # Accelerate gradually
            delta_lat = (speed * dt / 111320) * np.random.normal(1, 0.05)
            delta_lon = (speed * dt / (111320 * np.cos(np.radians(latitudes[-1])))) * np.random.normal(1, 0.05)
            new_alt = altitudes[-1] + climb_rate * dt * np.random.normal(1, 0.1)
        
        elif pattern == "circular":
            angle = t * dt * 0.1  # Control angular speed
            radius = 5.0 / 111320  # Approx radius in degrees
            delta_lat = radius * np.cos(angle)
            delta_lon = radius * np.sin(angle)
            new_alt = altitudes[-1]
        
        elif pattern == "spiral":
            angle = t * dt * 0.1
            radius = 5.0 / 111320
            delta_lat = radius * np.cos(angle)
            delta_lon = radius * np.sin(angle)
            new_alt = altitudes[-1] + climb_rate * dt * np.random.normal(1, 0.1)
        
        elif pattern == "oscillatory":
            delta_lat = 0
            delta_lon = 0
            new_alt = initial_altitude + 10 * np.sin(2 * np.pi * 0.5 * t * dt)  # Oscillate altitude

        elif pattern == "random_walk":
            delta_lat = np.random.uniform(-0.0001, 0.0001)
            delta_lon = np.random.uniform(-0.0001, 0.0001)
            new_alt = altitudes[-1] + np.random.uniform(-1, 1) * climb_rate * dt
        
        elif pattern == "figure_eight":
            angle = 2 * np.pi * 0.5 * t * dt
            delta_lat = 0.00005 * np.sin(angle)
            delta_lon = 0.00005 * np.sin(2 * angle)
            new_alt = altitudes[-1]

        # Update position
        new_lat = latitudes[-1] + delta_lat
        new_lon = longitudes[-1] + delta_lon
        latitudes.append(new_lat)
        longitudes.append(new_lon)
        altitudes.append(new_alt)
    
    gps_data = np.column_stack((latitudes, longitudes, altitudes))
    return gps_data

# Choose movement pattern
# Options: linear_constant_velocity, linear_acceleration, circular, spiral, oscillatory, random_walk, figure_eight
pattern = "spiral"  # Change this to test different patterns
gps_data = generate_realistic_gps_data(pattern, timesteps, initial_lat, initial_lon, initial_altitude)

# Define function for realistic accelerometer data
def generate_realistic_accelerometer_data(timesteps, true_acceleration, noise_std=0.05, drift=0.01):
    accelerometer_data = []
    for _ in range(timesteps):
        noise = np.random.normal(0, noise_std, size=3)
        drift_component = drift * np.ones(3)  # Adding a constant drift
        gravity = np.array([0, 0, 9.81])      # Simulate Earth's gravity in Z-axis
        realistic_accel = true_acceleration + noise + drift_component + gravity
        accelerometer_data.append(realistic_accel)
    return np.array(accelerometer_data)

# Define function for realistic gyroscope data
def generate_realistic_gyroscope_data(timesteps, true_angular_velocity, noise_std=0.01, drift=0.005, bias=0.02):
    gyroscope_data = []
    for _ in range(timesteps):
        noise = np.random.normal(0, noise_std, size=3)
        drift_component = drift * np.ones(3)
        realistic_gyro = true_angular_velocity + noise + drift_component + bias
        gyroscope_data.append(realistic_gyro)
    return np.array(gyroscope_data)

# Adjust true acceleration and angular velocity based on movement pattern
if pattern in ["linear_constant_velocity", "linear_acceleration", "random_walk"]:
    true_acceleration = np.array([0.1, 0.0, 0.0])  # Forward acceleration
    true_angular_velocity = np.array([0.0, 0.0, 0.0])
elif pattern == "circular":
    true_acceleration = np.array([0.0, 0.0, 0.0])
    true_angular_velocity = np.array([0.0, 0.0, 0.1])  # Constant yaw rate
elif pattern == "spiral":
    true_acceleration = np.array([0.05, 0.0, 0.0])  # Slow climb
    true_angular_velocity = np.array([0.0, 0.0, 0.1])
elif pattern == "oscillatory":
    true_acceleration = np.array([0.0, 0.0, 0.0])
    true_angular_velocity = np.array([0.0, 0.1, 0.0])  # Pitch oscillation
elif pattern == "figure_eight":
    true_acceleration = np.array([0.0, 0.0, 0.0])
    true_angular_velocity = np.array([0.0, 0.0, 0.1])

# Generate accelerometer and gyroscope data
accelerometer_data = generate_realistic_accelerometer_data(timesteps, true_acceleration)
gyroscope_data = generate_realistic_gyroscope_data(timesteps, true_angular_velocity)

# Visualization
fig = plt.figure(figsize=(15, 12))

# 3D GPS Data Plot
ax = fig.add_subplot(3, 1, 1, projection='3d')
ax.plot(gps_data[:, 1], gps_data[:, 0], gps_data[:, 2], label="True Path")
ax.set_title(f"Simulated GPS Data (3D Trajectory in Lat/Lon/Alt - {pattern})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude (m)")
ax.legend()

# Accelerometer Data Plot (X, Y, Z)
axes = ['X', 'Y', 'Z']
for i in range(3):
    plt.subplot(3, 3, 4 + i)
    plt.plot(accelerometer_data[:, i], label=f"Accelerometer Data ({axes[i]}-axis)", alpha=0.7)
    plt.title(f"Simulated Accelerometer Data ({axes[i]}-axis)")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()

# Gyroscope Data Plot (X, Y, Z)
for i in range(3):
    plt.subplot(3, 3, 7 + i)
    plt.plot(gyroscope_data[:, i], label=f"Gyroscope Data ({axes[i]}-axis)", alpha=0.7)
    plt.title(f"Simulated Gyroscope Data ({axes[i]}-axis)")
    plt.xlabel("Time Step")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()

plt.tight_layout()
plt.show()
