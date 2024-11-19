import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from geopy.distance import geodesic, distance as geopy_distance
from geopy import Point
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd  # Import pandas for data handling

# Simulation Parameters
np.random.seed(42)
dt = 60  # Time interval between steps (seconds)

# Initial GPS coordinates (approximate location over San Francisco)
initial_lat = 37.7749    # Degrees
initial_lon = -122.4194  # Degrees
initial_altitude = 1000  # Altitude in meters

# Initialize drift variables
accelerometer_drift = np.zeros(3)
gyroscope_drift = np.zeros(3)

# Aircraft Class
class Aircraft:
    def __init__(self, initial_position, initial_velocity, initial_orientation):
        self.position = np.array(initial_position)  # [lat, lon, alt]
        self.velocity = np.array(initial_velocity)  # m/s in NED frame
        self.orientation = np.array(initial_orientation)  # [roll, pitch, yaw] in degrees
        self.acceleration = np.array([0.0, 0.0, 0.0])  # m/s^2 in NED frame
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # rad/s
        self.mass = 50000  # kg (Typical commercial aircraft mass)
        self.wing_area = 122.6  # m^2 (Typical commercial aircraft wing area)
        self.C_L0 = 0.5  # Base lift coefficient
        self.C_D0 = 0.02  # Base drag coefficient

# Sensor Error Models
def accelerometer_error_model(true_acceleration, bias_instability, random_walk_std, scale_factor_error, temperature):
    global accelerometer_drift
    bias = np.random.normal(0, bias_instability, size=true_acceleration.shape)
    random_walk = np.random.normal(0, random_walk_std, size=true_acceleration.shape)
    scale_error = scale_factor_error * true_acceleration
    # Update drift with random walk
    drift_step = np.random.normal(0, 0.000001, size=true_acceleration.shape)
    accelerometer_drift += drift_step
    measured_acceleration = true_acceleration + bias + random_walk + scale_error + accelerometer_drift
    # Apply saturation limits
    measured_acceleration = np.clip(measured_acceleration, -16 * 9.81, 16 * 9.81)  # +/-16g
    return measured_acceleration

def gyroscope_error_model(true_angular_velocity, bias_instability, random_walk_std, scale_factor_error, temperature):
    global gyroscope_drift
    bias = np.random.normal(0, bias_instability, size=true_angular_velocity.shape)
    random_walk = np.random.normal(0, random_walk_std, size=true_angular_velocity.shape)
    scale_error = scale_factor_error * true_angular_velocity
    # Update drift with random walk
    drift_step = np.random.normal(0, 0.0000001, size=true_angular_velocity.shape)
    gyroscope_drift += drift_step
    measured_angular_velocity = true_angular_velocity + bias + random_walk + scale_error + gyroscope_drift
    # Apply saturation limits in rad/s
    measured_angular_velocity = np.clip(measured_angular_velocity, np.radians(-500), np.radians(500))
    return measured_angular_velocity

# Environment Class
class Environment:
    def __init__(self):
        self.temperature = 15  # Celsius at sea level
        self.pressure = 101325  # Pa at sea level
        self.density = 1.225  # kg/m^3 at sea level

    def update_conditions(self, altitude):
        # Simple atmospheric model: ISA standard atmosphere
        temp_lapse_rate = -0.0065  # Temperature lapse rate in K/m
        self.temperature = 15 + temp_lapse_rate * altitude  # Celsius
        # Avoid negative temperatures in Kelvin
        temp_kelvin = self.temperature + 273.15
        temp_kelvin = max(temp_kelvin, 1.0)
        # Update pressure
        self.pressure = 101325 * (temp_kelvin / 288.15) ** (-9.80665 / (temp_lapse_rate * 287.05))
        # Update density
        self.density = self.pressure / (287.05 * temp_kelvin)

    def get_wind(self, altitude, location, time):
        # Simplistic wind model with reduced variability
        base_wind_speed = 5  # m/s at sea level
        wind_speed = base_wind_speed  # Constant wind speed for simplicity
        wind_direction = 270  # Westward wind
        wind_vector = wind_speed * np.array([
            np.sin(np.radians(wind_direction)),  # East component
            np.cos(np.radians(wind_direction)),  # North component
            0.0  # Assuming horizontal wind
        ])
        return wind_vector  # m/s

    def get_turbulence(self, altitude, flight_phase):
        # Adjust turbulence intensity based on flight phase
        if flight_phase == 'cruise':
            turbulence_intensity = 0.05  # Very low turbulence during cruise
            angular_turbulence_intensity = 0.005  # Reduced angular turbulence
            # Introduce minor turbulence variations during cruise
            turbulence_intensity += 0.02 * np.sin(np.radians(altitude % 360))
        elif flight_phase == 'climb' or flight_phase == 'descent':
            turbulence_intensity = 0.1  # Moderate turbulence during climb/descent
            angular_turbulence_intensity = 0.01
        else:
            turbulence_intensity = 0.2  # Higher turbulence during takeoff/landing
            angular_turbulence_intensity = 0.02

        # Ensure turbulence intensity doesn't go below a minimum value
        turbulence_intensity = max(turbulence_intensity, 0.05)
        turbulence = np.random.normal(0, turbulence_intensity, size=3)  # m/s^2
        # Add angular turbulence
        angular_turbulence = np.random.normal(0, np.radians(angular_turbulence_intensity), size=3)
        return turbulence, angular_turbulence

# Generate Waypoints for Flight Plan
def generate_flight_plan():
    waypoints = [
        {'lat': 37.7749, 'lon': -122.4194, 'alt': 1000, 'phase': 'takeoff', 'speed': 80},   # San Francisco
        {'lat': 34.0522, 'lon': -118.2437, 'alt': 10000, 'phase': 'climb', 'speed': 250},   # Los Angeles
        {'lat': 36.1699, 'lon': -115.1398, 'alt': 10000, 'phase': 'cruise', 'speed': 250},  # Las Vegas
        {'lat': 40.7128, 'lon': -74.0060, 'alt': 1000, 'phase': 'descent', 'speed': 250},    # New York
    ]
    return waypoints

# Compute Bearing between two points
def compute_bearing(pointA, pointB):
    lat1 = np.radians(pointA.latitude)
    lat2 = np.radians(pointB.latitude)
    diffLong = np.radians(pointB.longitude - pointA.longitude)
    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
            * np.cos(lat2) * np.cos(diffLong))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# Simulate Accelerometer and Gyroscope Data
def simulate_sensors(aircraft, environment, dt, flight_plan):
    global accelerometer_drift, gyroscope_drift
    accelerometer_data = []
    gyroscope_data = []
    positions = []
    orientations = []
    time_stamps = []
    waypoint_index = 0
    total_waypoints = len(flight_plan)
    current_waypoint = flight_plan[waypoint_index]
    next_waypoint = flight_plan[waypoint_index + 1] if waypoint_index + 1 < total_waypoints else flight_plan[waypoint_index]

    plane_stationary = False  # Flag to check if plane has reached the final destination
    current_flight_phase = current_waypoint['phase']  # Initialize flight phase
    t = 0  # Time step counter
    simulation_start_time = datetime.now()
    previous_heading = aircraft.orientation[2]
    desired_heading = aircraft.orientation[2]
    previous_desired_heading = desired_heading

    while not plane_stationary:
        # Update time
        current_time = simulation_start_time + timedelta(seconds=dt * t)
        time_stamps.append(current_time)

        # Update environmental conditions
        altitude = aircraft.position[2]
        environment.update_conditions(altitude)
        # Get wind and turbulence
        wind = environment.get_wind(altitude, aircraft.position[:2], current_time)
        turbulence, angular_turbulence = environment.get_turbulence(altitude, current_flight_phase)

        if not plane_stationary:
            # Flight dynamics calculations
            # Determine desired speed based on flight phase
            desired_speed = next_waypoint['speed']
            if current_flight_phase == 'descent':
                # Gradually reduce speed during descent
                min_descent_speed = 80  # m/s
                max_descent_speed = desired_speed  # Start descent at cruise speed
                altitude_range = current_waypoint['alt'] - next_waypoint['alt']  # Positive value
                if altitude_range > 0:
                    altitude_fraction = (aircraft.position[2] - next_waypoint['alt']) / altitude_range
                    desired_speed = min_descent_speed + altitude_fraction * (max_descent_speed - min_descent_speed)
                else:
                    desired_speed = min_descent_speed

            # Compute airspeed (horizontal components)
            airspeed = aircraft.velocity[:2] - wind[:2]  # Relative to air mass (horizontal components)
            airspeed_mag = np.linalg.norm(airspeed)
            airspeed_mag = max(airspeed_mag, 1e-3)  # Prevent division by zero
            # Compute current heading from airspeed
            current_heading = (np.degrees(np.arctan2(airspeed[0], airspeed[1])) + 360) % 360
            # Compute desired heading towards next waypoint
            current_position = Point(aircraft.position[0], aircraft.position[1])
            waypoint_position = Point(next_waypoint['lat'], next_waypoint['lon'])
            heading_to_waypoint = compute_bearing(current_position, waypoint_position)

            # Compute desired heading smoothly
            desired_heading_error = (heading_to_waypoint - previous_desired_heading + 180) % 360 - 180  # [-180, 180]
            # Adjust max desired heading rate based on flight phase
            if current_flight_phase == 'cruise':
                max_desired_heading_rate = 2.0  # degrees per second
            else:
                max_desired_heading_rate = 1.0  # degrees per second
            desired_heading_change = np.clip(desired_heading_error, -max_desired_heading_rate * dt, max_desired_heading_rate * dt)
            desired_heading = (previous_desired_heading + desired_heading_change) % 360

            # Compute heading error
            heading_error = (desired_heading - current_heading + 180) % 360 - 180  # Range [-180, 180]
            # Adjust max_turn_rate based on flight phase and remaining distance
            remaining_distance = geodesic((aircraft.position[0], aircraft.position[1]),
                                          (next_waypoint['lat'], next_waypoint['lon'])).meters
            if current_flight_phase == 'cruise':
                if remaining_distance < 50000:
                    max_turn_rate = 3.0  # degrees per second
                else:
                    max_turn_rate = 1.5  # degrees per second
            else:
                max_turn_rate = 1.0  # degrees per second
            max_turn_angle = max_turn_rate * dt  # degrees per time step
            heading_change = np.clip(heading_error, -max_turn_angle, max_turn_angle)
            # Compute new heading
            new_heading = (current_heading + heading_change) % 360
            # Update airspeed magnitude towards desired speed
            speed_error = desired_speed - airspeed_mag
            max_acceleration = 2.0  # m/s^2
            acceleration = np.clip(speed_error / dt, -max_acceleration, max_acceleration)
            airspeed_mag += acceleration * dt
            airspeed_mag = max(airspeed_mag, 0)
            # Update airspeed direction
            airspeed_x = airspeed_mag * np.sin(np.radians(new_heading))
            airspeed_y = airspeed_mag * np.cos(np.radians(new_heading))
            airspeed = np.array([airspeed_x, airspeed_y])
            # Update aircraft's velocity (considering wind)
            aircraft.velocity[0] = airspeed[0] + wind[0]
            aircraft.velocity[1] = airspeed[1] + wind[1]
            # Update vertical component of velocity
            # Adjust climb rate based on phase
            alt_diff = next_waypoint['alt'] - aircraft.position[2]
            if current_flight_phase == 'descent':
                max_climb_rate = 3  # m/s (Gentler descent)
            else:
                max_climb_rate = 5  # m/s (Typical for climb)
            climb_rate = np.clip(alt_diff / dt, -max_climb_rate, max_climb_rate)
            aircraft.velocity[2] = climb_rate
            # Update lift and drag coefficients
            C_L = aircraft.C_L0
            C_D = aircraft.C_D0
            # Compute aerodynamic forces
            lift = aircraft.mass * 9.81  # Adjusted to balance weight (steady flight)
            drag = 0.5 * environment.density * airspeed_mag**2 * aircraft.wing_area * C_D
            lift_force = np.array([0, 0, lift])
            drag_force = -drag * (airspeed / airspeed_mag) if airspeed_mag != 0 else np.zeros(2)
            drag_force_3d = np.array([drag_force[0], drag_force[1], 0.0])  # Only horizontal drag
            # Compute thrust
            thrust_magnitude = drag  # Assume thrust balances drag in steady flight
            thrust_force = thrust_magnitude * (airspeed / airspeed_mag) if airspeed_mag != 0 else np.zeros(2)
            thrust_force_3d = np.array([thrust_force[0], thrust_force[1], 0.0])  # Only horizontal thrust
            # Total forces
            total_force = lift_force + drag_force_3d + thrust_force_3d + turbulence * aircraft.mass
            # Update acceleration
            aircraft.acceleration = total_force / aircraft.mass - np.array([0, 0, 9.81])  # Subtract gravity
            # Limit accelerations to realistic values
            max_total_acceleration = 2 * 9.81  # Max 2g acceleration
            aircraft.acceleration = np.clip(aircraft.acceleration, -max_total_acceleration, max_total_acceleration)
            # Update position using geodesic calculations
            current_position = Point(aircraft.position[0], aircraft.position[1])
            # Calculate true ground speed
            ground_speed = np.linalg.norm(aircraft.velocity[:2])
            distance_traveled = ground_speed * dt
            new_heading = (current_heading + heading_change) % 360
            new_position = geopy_distance(meters=distance_traveled).destination(current_position, new_heading)
            aircraft.position[0] = new_position.latitude
            aircraft.position[1] = new_position.longitude
            aircraft.position[2] += aircraft.velocity[2] * dt
            # Ensure altitude is not below ground level
            aircraft.position[2] = max(aircraft.position[2], 0)
            # Update orientation
            # Compute angular velocity around Z-axis (yaw rate)
            heading_difference = (new_heading - previous_heading + 180) % 360 - 180  # Handle wrap-around
            angular_velocity_z = np.radians(heading_difference) / dt  # rad/s
            aircraft.angular_velocity = np.array([0.0, 0.0, angular_velocity_z]) + angular_turbulence
            previous_heading = new_heading
            # Compute pitch angle
            pitch_angle = np.degrees(np.arctan2(aircraft.velocity[2], airspeed_mag))
            # Compute roll angle
            rate_of_heading_change = heading_difference / dt  # degrees per second
            roll_angle = rate_of_heading_change * 5.0  # Adjust factor as needed
            max_roll_angle = 30.0  # degrees
            roll_angle = np.clip(roll_angle, -max_roll_angle, max_roll_angle)
            # Add small random variations to pitch and roll
            roll_angle += np.random.normal(0, 1.0)
            pitch_angle += np.random.normal(0, 0.5)
            aircraft.orientation[0] = roll_angle  # Roll angle
            aircraft.orientation[1] = pitch_angle  # Pitch angle
            aircraft.orientation[2] = new_heading
            # Check if reached next waypoint
            if waypoint_index < total_waypoints - 1:
                if remaining_distance < 5000:  # Reduced threshold distance
                    waypoint_index += 1
                    current_waypoint = next_waypoint
                    current_flight_phase = current_waypoint['phase']  # Update flight phase
                    if waypoint_index + 1 < total_waypoints:
                        next_waypoint = flight_plan[waypoint_index + 1]
            else:
                # At final waypoint
                if remaining_distance < 5000:  # Close enough to final destination
                    # Plane has reached final destination
                    aircraft.velocity = np.zeros(3)
                    aircraft.acceleration = np.zeros(3)
                    aircraft.angular_velocity = np.zeros(3)
                    aircraft.position[0] = next_waypoint['lat']
                    aircraft.position[1] = next_waypoint['lon']
                    aircraft.position[2] = next_waypoint['alt']
                    plane_stationary = True
        else:
            # Plane is stationary at final destination
            aircraft.velocity = np.zeros(3)
            aircraft.acceleration = np.zeros(3)
            aircraft.angular_velocity = np.zeros(3)
            # Position remains constant

        positions.append(aircraft.position.copy())
        orientations.append(aircraft.orientation.copy())

        # True sensor readings
        true_acceleration = aircraft.acceleration
        true_angular_velocity = aircraft.angular_velocity

        # Adjust sensor noise parameters based on flight phase
        if current_flight_phase == 'cruise':
            acc_bias_instability = 0.00001
            acc_random_walk_std = 0.001
            gyro_bias_instability = 0.000001
            gyro_random_walk_std = 0.0001
        elif current_flight_phase == 'climb' or current_flight_phase == 'descent':
            acc_bias_instability = 0.00005
            acc_random_walk_std = 0.005
            gyro_bias_instability = 0.000005
            gyro_random_walk_std = 0.0005
        else:
            acc_bias_instability = 0.0001
            acc_random_walk_std = 0.01
            gyro_bias_instability = 0.00001
            gyro_random_walk_std = 0.001

        # Sensor error models
        acc_measured = accelerometer_error_model(
            true_acceleration, bias_instability=acc_bias_instability, random_walk_std=acc_random_walk_std,
            scale_factor_error=0.0005, temperature=environment.temperature)
        gyro_measured = gyroscope_error_model(
            true_angular_velocity, bias_instability=gyro_bias_instability, random_walk_std=gyro_random_walk_std,
            scale_factor_error=0.00005, temperature=environment.temperature)
        accelerometer_data.append(acc_measured)
        gyroscope_data.append(gyro_measured)

        # Increment time step
        t += 1
        previous_desired_heading = desired_heading

    accelerometer_data = np.array(accelerometer_data)
    gyroscope_data = np.array(gyroscope_data)
    positions = np.array(positions)
    orientations = np.array(orientations)
    total_flight_time = t * dt  # in seconds
    return accelerometer_data, gyroscope_data, positions, orientations, time_stamps, total_flight_time

# Main Simulation
flight_plan = generate_flight_plan()

# Initialize Environment
environment = Environment()

# Initialize Aircraft
initial_position = [flight_plan[0]['lat'], flight_plan[0]['lon'], flight_plan[0]['alt']]
# Compute initial bearing
initial_position_point = Point(initial_position[0], initial_position[1])
next_waypoint_position = Point(flight_plan[1]['lat'], flight_plan[1]['lon'])
initial_bearing = compute_bearing(initial_position_point, next_waypoint_position)
initial_speed = flight_plan[0]['speed']  # 80 m/s
initial_velocity = initial_speed * np.array([
    np.sin(np.radians(initial_bearing)),
    np.cos(np.radians(initial_bearing)),
    0.0
])
initial_orientation = np.array([0.0, 0.0, initial_bearing])  # Degrees
aircraft = Aircraft(initial_position, initial_velocity, initial_orientation)

# Simulate Sensors
accelerometer_data, gyroscope_data, positions, orientations, time_stamps, total_flight_time = simulate_sensors(
    aircraft, environment, dt, flight_plan)

# Display Total Flight Time
flight_duration = timedelta(seconds=total_flight_time)
print(f"Total Predicted Flight Time: {flight_duration}")

# Collect Data into DataFrames
sensor_data = pd.DataFrame({
    'Time': time_stamps,
    'Acc_X': accelerometer_data[:, 0],
    'Acc_Y': accelerometer_data[:, 1],
    'Acc_Z': accelerometer_data[:, 2],
    'Gyro_X': np.degrees(gyroscope_data[:, 0]),
    'Gyro_Y': np.degrees(gyroscope_data[:, 1]),
    'Gyro_Z': np.degrees(gyroscope_data[:, 2]),
})

flight_logs = pd.DataFrame({
    'Time': time_stamps,
    'Latitude': positions[:, 0],
    'Longitude': positions[:, 1],
    'Altitude': positions[:, 2],
    'Roll': orientations[:, 0],
    'Pitch': orientations[:, 1],
    'Yaw': orientations[:, 2],
})

# Export Data to CSV
sensor_data.to_csv('sensor_data.csv', index=False)
flight_logs.to_csv('flight_logs.csv', index=False)

# Export Data to JSON (Optional)
sensor_data.to_json('sensor_data.json', orient='records', date_format='iso')
flight_logs.to_json('flight_logs.json', orient='records', date_format='iso')

# Visualization
fig = plt.figure(figsize=(15, 12))

# Map Plot using Cartopy
ax_map = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
ax_map.set_title("Flight Path on Map")
ax_map.add_feature(cfeature.LAND)
ax_map.add_feature(cfeature.OCEAN)
ax_map.add_feature(cfeature.COASTLINE)
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.add_feature(cfeature.LAKES, alpha=0.5)
ax_map.add_feature(cfeature.RIVERS)
ax_map.set_extent([-130, -65, 25, 50], crs=ccrs.PlateCarree())
ax_map.plot(positions[:, 1], positions[:, 0], 'b-', transform=ccrs.Geodetic(), label="Flight Path")
ax_map.scatter([wp['lon'] for wp in flight_plan], [wp['lat'] for wp in flight_plan],
               color='red', transform=ccrs.Geodetic(), zorder=5, label="Waypoints")
ax_map.legend()

# Accelerometer Data Plot (X, Y, Z)
axes = ['X', 'Y', 'Z']
for i in range(3):
    plt.subplot(3, 3, 4 + i)
    plt.plot(time_stamps, accelerometer_data[:, i], label=f"Accelerometer ({axes[i]})", alpha=0.7)
    plt.title(f"Accelerometer Data ({axes[i]}-axis)")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Gyroscope Data Plot (X, Y, Z)
gyroscope_data_deg = np.degrees(gyroscope_data)  # Convert from rad/s to deg/s
for i in range(3):
    plt.subplot(3, 3, 7 + i)
    plt.plot(time_stamps, gyroscope_data_deg[:, i], label=f"Gyroscope ({axes[i]})", alpha=0.7)
    plt.title(f"Gyroscope Data ({axes[i]}-axis)")
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity (deg/s)")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()
