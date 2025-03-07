import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math
import random
import time

def generate_gps_route_with_noise(start_lat, start_lon, num_points=100, 
                                 route_type="random", 
                                 noise_level="medium",
                                 record_noise=True,
                                 simulate_signal_loss=True):
    """
    Generate a simulated GPS route with realistic noise patterns.
    
    Parameters:
    ----------
    start_lat, start_lon : float
        Starting coordinates
    num_points : int
        Number of GPS points to generate along the route
    route_type : str
        Type of route: "random", "straight", "circle", "zigzag", or "realistic"
    noise_level : str
        Level of GPS noise: "low" (open sky), "medium" (urban), "high" (dense urban/trees)
    record_noise : bool
        Whether to record the true position and noise components
    simulate_signal_loss : bool
        Whether to simulate occasional GPS signal loss
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame containing the route coordinates, metadata, and noise information
    """
    # Define noise levels (in meters, converted to degrees)
    noise_levels = {
        "low": {"bias": 1.0, "random": 2.0, "drift_factor": 0.05, "signal_loss_prob": 0.01},  # Open sky
        "medium": {"bias": 3.0, "random": 5.0, "drift_factor": 0.15, "signal_loss_prob": 0.05},  # Urban
        "high": {"bias": 8.0, "random": 12.0, "drift_factor": 0.3, "signal_loss_prob": 0.1}  # Dense urban/trees
    }
    
    # Use selected noise profile
    noise_profile = noise_levels.get(noise_level, noise_levels["medium"])
    
    # Approximately 111,320 meters per degree at the equator (for lat/lon conversion)
    # This changes with latitude, but this approximation works for small movements
    meters_per_degree = 111320 * math.cos(math.radians(start_lat))
    
    route_data = []
    
    # Initialize variables
    current_lat = start_lat
    current_lon = start_lon
    
    # True positions (without noise)
    true_lat = start_lat
    true_lon = start_lon
    
    # Set average speed between 5-60 km/h (1.4-16.7 m/s)
    if route_type == "realistic":
        # Realistic urban travel - varies more
        avg_speed = random.uniform(2.0, 15.0)  # m/s
    else:
        avg_speed = random.uniform(5.0, 16.7)  # m/s
    
    # Time between points (1-5 seconds)
    time_step = random.uniform(1, 5)
    
    # Calculate typical position change based on speed
    degree_step = avg_speed / meters_per_degree  # degrees per second
    
    # Initialize timestamp
    timestamp = pd.Timestamp.now()
    
    # Initialize noise drift components
    # Bias: systematic error that stays somewhat consistent
    # Random: completely random error each reading
    # Drift: slowly changing error component
    bias_lat = random.uniform(-noise_profile["bias"], noise_profile["bias"]) / meters_per_degree
    bias_lon = random.uniform(-noise_profile["bias"], noise_profile["bias"]) / meters_per_degree
    
    drift_lat = 0
    drift_lon = 0
    drift_direction = random.uniform(0, 2*math.pi)
    drift_speed = noise_profile["drift_factor"] * noise_profile["random"] / meters_per_degree / 10  # slow drift
    
    # For realistic route
    realistic_waypoints = []
    if route_type == "realistic":
        # Create some waypoints to simulate realistic travel
        num_waypoints = random.randint(3, 8)  # 3-8 waypoints
        
        # Create a sequence of waypoints within a reasonable area
        bearing = random.uniform(0, 360)
        distance = random.uniform(0.01, 0.05)  # ~1-5km 
        
        for i in range(num_waypoints):
            bearing_rad = math.radians(bearing)
            wp_lat = start_lat + distance * math.cos(bearing_rad)
            wp_lon = start_lon + distance * math.sin(bearing_rad) / math.cos(math.radians(start_lat))
            realistic_waypoints.append((wp_lat, wp_lon))
            
            # Change bearing significantly for next waypoint
            bearing = (bearing + random.uniform(60, 120)) % 360
            distance = random.uniform(0.01, 0.05)
    
    # Initial bearing (direction of travel)
    if route_type == "realistic" and realistic_waypoints:
        # Set initial bearing toward first waypoint
        target = realistic_waypoints[0]
        y = math.sin(target[1] - true_lon) * math.cos(math.radians(target[0]))
        x = (math.cos(math.radians(true_lat)) * math.sin(math.radians(target[0])) - 
             math.sin(math.radians(true_lat)) * math.cos(math.radians(target[0])) * 
             math.cos(math.radians(target[1] - true_lon)))
        bearing = math.degrees(math.atan2(y, x))
        current_waypoint_idx = 0
    else:
        bearing = random.uniform(0, 360)
    
    # Signal loss simulation
    in_signal_loss = False
    signal_loss_duration = 0
    last_valid_lat = current_lat
    last_valid_lon = current_lon
    
    # Main loop to generate points
    for i in range(num_points):
        # Check if we should simulate a signal loss
        if simulate_signal_loss and random.random() < noise_profile["signal_loss_prob"] and not in_signal_loss:
            in_signal_loss = True
            signal_loss_duration = random.randint(3, 10)  # 3-10 points of signal loss
        
        # Update drift component (slowly changing error)
        drift_direction += random.uniform(-0.1, 0.1)  # Slight change in drift direction
        drift_lat += drift_speed * math.cos(drift_direction)
        drift_lon += drift_speed * math.sin(drift_direction)
        
        # Calculate new true position based on route type
        if route_type == "straight":
            # Move in a fairly straight line with slight variations
            if i > 0:
                bearing += random.uniform(-5, 5)  # Slight variations in direction
                
        elif route_type == "circle":
            # Move in a rough circle
            bearing = (i * 360 / num_points) + random.uniform(-5, 5)
            
        elif route_type == "zigzag":
            # Move in a zigzag pattern
            if i % 10 == 0:
                bearing = bearing + 90 if i % 20 == 0 else bearing - 90
                
        elif route_type == "realistic" and realistic_waypoints:
            # Move toward the current waypoint
            target = realistic_waypoints[current_waypoint_idx]
            
            # Calculate bearing to waypoint
            y = math.sin(math.radians(target[1] - true_lon)) * math.cos(math.radians(target[0]))
            x = (math.cos(math.radians(true_lat)) * math.sin(math.radians(target[0])) - 
                 math.sin(math.radians(true_lat)) * math.cos(math.radians(target[0])) * 
                 math.cos(math.radians(target[1] - true_lon)))
            target_bearing = math.degrees(math.atan2(y, x)) % 360
            
            # Gradually adjust bearing toward target (simulates realistic turning)
            bearing_diff = (target_bearing - bearing + 180) % 360 - 180
            turn_rate = min(abs(bearing_diff), 15)  # Max 15 degrees turn per point
            if bearing_diff > 0:
                bearing += turn_rate
            else:
                bearing -= turn_rate
            
            # Check if we've reached the waypoint
            dist_to_waypoint = math.sqrt((true_lat - target[0])**2 + (true_lon - target[1])**2)
            if dist_to_waypoint < 0.0005:  # ~50 meters
                current_waypoint_idx = (current_waypoint_idx + 1) % len(realistic_waypoints)
                
        else:  # random
            # Randomly change direction every few points
            if i % 5 == 0:
                bearing = random.uniform(0, 360)
        
        # Calculate movement distance
        move_distance = degree_step * time_step
        
        # Add some random variation to the speed
        move_distance *= (0.8 + 0.4 * random.random())
        
        # Convert bearing to radians
        bearing_rad = math.radians(bearing)
        
        # Calculate new true position
        true_lat += move_distance * math.cos(bearing_rad)
        true_lon += move_distance * math.sin(bearing_rad) / math.cos(math.radians(true_lat))
        
        # Apply noise to get the measured position
        if in_signal_loss:
            # During signal loss, either report last known position or generate highly inaccurate readings
            if random.random() < 0.5:
                # Report last known position
                measured_lat = last_valid_lat
                measured_lon = last_valid_lon
                noise_lat = measured_lat - true_lat
                noise_lon = measured_lon - true_lon
                accuracy = 100.0  # Very poor accuracy during signal loss
            else:
                # Generate highly inaccurate reading
                random_noise_lat = random.uniform(-50, 50) / meters_per_degree
                random_noise_lon = random.uniform(-50, 50) / meters_per_degree
                measured_lat = true_lat + random_noise_lat
                measured_lon = true_lon + random_noise_lon
                noise_lat = random_noise_lat
                noise_lon = random_noise_lon
                accuracy = 200.0  # Extremely poor accuracy
                
            signal_loss_duration -= 1
            if signal_loss_duration <= 0:
                in_signal_loss = False
                
            speed = 0.0  # During signal loss, speed is often reported as zero
            
        else:
            # Normal GPS reading with applied noise
            # Random noise component (different every reading)
            random_noise_lat = random.uniform(-noise_profile["random"], noise_profile["random"]) / meters_per_degree
            random_noise_lon = random.uniform(-noise_profile["random"], noise_profile["random"]) / meters_per_degree
            
            # Calculate measured position with all noise components
            measured_lat = true_lat + bias_lat + drift_lat + random_noise_lat
            measured_lon = true_lon + bias_lon + drift_lon + random_noise_lon
            
            # Record the noise components
            noise_lat = bias_lat + drift_lat + random_noise_lat
            noise_lon = bias_lon + drift_lon + random_noise_lon
            
            # Update last valid position
            last_valid_lat = measured_lat
            last_valid_lon = measured_lon
            
            # Realistic accuracy value (in meters)
            if in_signal_loss:
                accuracy = random.uniform(50.0, 100.0)
            else:
                accuracy = random.uniform(noise_profile["random"]/2, noise_profile["random"]*2)
                
            # Calculate speed with some noise
            speed = avg_speed * (0.9 + 0.2 * random.random())
            if i % 10 == 0:  # Occasionally change average speed
                avg_speed = random.uniform(5.0, 16.7)
        
        # Add the point to our dataset
        point_data = {
            'latitude': measured_lat,
            'longitude': measured_lon,
            'timestamp': timestamp,
            'accuracy': accuracy,
            'speed': speed,
            'altitude': random.uniform(0, 500) + (300 if true_lat > 30 else 0),  # Higher altitudes in the north
            'source': 'Route',
            'signal_loss': in_signal_loss
        }
        
        # Add noise information if requested
        if record_noise:
            point_data.update({
                'true_latitude': true_lat,
                'true_longitude': true_lon,
                'noise_latitude': noise_lat * meters_per_degree,  # Convert to meters
                'noise_longitude': noise_lon * meters_per_degree,  # Convert to meters
                'bias_latitude': bias_lat * meters_per_degree,  # Convert to meters
                'bias_longitude': bias_lon * meters_per_degree,  # Convert to meters
                'drift_latitude': drift_lat * meters_per_degree,  # Convert to meters
                'drift_longitude': drift_lon * meters_per_degree,  # Convert to meters
                'random_noise_latitude': (noise_lat - bias_lat - drift_lat) * meters_per_degree,  # Convert to meters
                'random_noise_longitude': (noise_lon - bias_lon - drift_lon) * meters_per_degree,  # Convert to meters
                'total_noise_meters': math.sqrt((noise_lat * meters_per_degree)**2 + 
                                              (noise_lon * meters_per_degree)**2)
            })
        
        route_data.append(point_data)
        
        # Update timestamp
        timestamp += pd.Timedelta(seconds=time_step)
        
    return pd.DataFrame(route_data)

def plot_noisy_route(df, title="GPS Route with Noise"):
    """Plot both the measured and true positions to visualize noise effects"""
    plt.figure(figsize=(15, 10))
    
    # Check if we have true position data
    has_true_pos = 'true_latitude' in df.columns and 'true_longitude' in df.columns
    
    # Plot measured positions
    plt.plot(df['longitude'], df['latitude'], 'b-', alpha=0.6, label='Measured GPS Path')
    plt.scatter(df['longitude'], df['latitude'], c=df['accuracy'], 
                cmap='YlOrRd', s=30, alpha=0.7, label='GPS Points')
    
    # Plot true positions if available
    if has_true_pos:
        plt.plot(df['true_longitude'], df['true_latitude'], 'g-', 
                linewidth=2, alpha=0.7, label='True Path')
    
    # Highlight signal loss areas if available
    if 'signal_loss' in df.columns:
        signal_loss_points = df[df['signal_loss']]
        if not signal_loss_points.empty:
            plt.scatter(signal_loss_points['longitude'], signal_loss_points['latitude'], 
                       c='red', s=50, marker='x', label='Signal Loss')
    
    plt.colorbar(label='GPS Accuracy (m)')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a zoom inset of a particularly noisy section if we have true data
    if has_true_pos and len(df) > 20:
        # Find section with high noise
        df['noise_magnitude'] = df['total_noise_meters']
        high_noise_idx = df['noise_magnitude'].nlargest(10).index[0]
        
        # Get a range around the high noise section
        range_start = max(0, high_noise_idx - 10)
        range_end = min(len(df), high_noise_idx + 10)
        zoom_df = df.iloc[range_start:range_end]
        
        # Create inset axes
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        axins = zoomed_inset_axes(plt.gca(), 6, loc='upper right')
        
        axins.plot(zoom_df['longitude'], zoom_df['latitude'], 'b-', alpha=0.6)
        axins.scatter(zoom_df['longitude'], zoom_df['latitude'], c=zoom_df['accuracy'], 
                     cmap='YlOrRd', s=30, alpha=0.7)
        axins.plot(zoom_df['true_longitude'], zoom_df['true_latitude'], 'g-', 
                  linewidth=2, alpha=0.7)
        
        # Draw connecting lines between true and measured positions
        for i, row in zoom_df.iterrows():
            axins.plot([row['true_longitude'], row['longitude']], 
                      [row['true_latitude'], row['latitude']], 
                      'r-', alpha=0.3, linewidth=0.5)
        
        # Set limits for inset
        mean_lon = zoom_df['longitude'].mean()
        mean_lat = zoom_df['latitude'].mean()
        lon_range = zoom_df['longitude'].max() - zoom_df['longitude'].min()
        lat_range = zoom_df['latitude'].max() - zoom_df['latitude'].min()
        buffer = max(lon_range, lat_range) * 0.2
        
        axins.set_xlim(mean_lon - buffer, mean_lon + buffer)
        axins.set_ylim(mean_lat - buffer, mean_lat + buffer)
        axins.set_title("Noise Detail View")
        axins.grid(True, linestyle='--', alpha=0.7)
        
        # Draw a box around the zoomed region
        mark_inset(plt.gca(), axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    plt.tight_layout()
    plt.savefig('india_gps_route_with_noise.png', dpi=300)
    
    # Create a separate plot showing noise components over time if available
    if has_true_pos:
        plt.figure(figsize=(15, 8))
        
        # Convert timestamp to numeric for plotting
        df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        # Plot different noise components
        plt.plot(df['time_numeric'], df['noise_latitude'], 'r-', label='Latitude Noise')
        plt.plot(df['time_numeric'], df['noise_longitude'], 'b-', label='Longitude Noise')
        plt.plot(df['time_numeric'], df['total_noise_meters'], 'k-', label='Total Noise Magnitude')
        
        if 'bias_latitude' in df.columns:
            plt.plot(df['time_numeric'], df['bias_latitude'], 'r--', alpha=0.5, label='Bias (Lat)')
            plt.plot(df['time_numeric'], df['bias_longitude'], 'b--', alpha=0.5, label='Bias (Lon)')
        
        if 'drift_latitude' in df.columns:
            plt.plot(df['time_numeric'], df['drift_latitude'], 'r:', alpha=0.5, label='Drift (Lat)')
            plt.plot(df['time_numeric'], df['drift_longitude'], 'b:', alpha=0.5, label='Drift (Lon)')
        
        plt.title('GPS Noise Components Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Noise (meters)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('india_gps_noise_components.png', dpi=300)
    
    plt.show()

def analyze_noise(df):
    """Analyze and summarize the noise characteristics"""
    if 'true_latitude' not in df.columns:
        print("No noise data available for analysis")
        return
    
    # Basic statistics on total noise
    print("\nNoise Analysis Summary:")
    print(f"Average total noise: {df['total_noise_meters'].mean():.2f} meters")
    print(f"Maximum total noise: {df['total_noise_meters'].max():.2f} meters")
    print(f"Standard deviation of noise: {df['total_noise_meters'].std():.2f} meters")
    
    # Component breakdown
    print("\nNoise Component Analysis:")
    
    # Random noise component
    random_lat = df['random_noise_latitude']
    random_lon = df['random_noise_longitude']
    random_magnitude = np.sqrt(random_lat**2 + random_lon**2)
    
    print(f"Random noise component: {random_magnitude.mean():.2f} meters (avg)")
    
    # Bias component
    bias_lat = df['bias_latitude']
    bias_lon = df['bias_longitude']
    bias_magnitude = np.sqrt(bias_lat**2 + bias_lon**2)
    
    print(f"Bias component: {bias_magnitude.mean():.2f} meters (avg)")
    
    # Drift component
    drift_lat = df['drift_latitude']
    drift_lon = df['drift_longitude']
    drift_magnitude = np.sqrt(drift_lat**2 + drift_lon**2)
    
    print(f"Drift component: {drift_magnitude.mean():.2f} meters (avg)")
    
    # Signal loss statistics if available
    if 'signal_loss' in df.columns:
        signal_loss_count = df['signal_loss'].sum()
        signal_loss_pct = signal_loss_count / len(df) * 100
        print(f"\nSignal loss events: {signal_loss_count} points ({signal_loss_pct:.1f}% of data)")
    
    # Generate a histogram of noise magnitude
    plt.figure(figsize=(10, 6))
    plt.hist(df['total_noise_meters'], bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of GPS Noise Magnitude')
    plt.xlabel('Noise Magnitude (meters)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('india_gps_noise_histogram.png', dpi=300)
    plt.show()
    
    # Component contribution pie chart
    plt.figure(figsize=(8, 8))
    components = ['Random', 'Bias', 'Drift']
    values = [random_magnitude.mean(), bias_magnitude.mean(), drift_magnitude.mean()]
    plt.pie(values, labels=components, autopct='%1.1f%%', startangle=90)
    plt.title('Contribution of Different Noise Components')
    plt.axis('equal')
    plt.savefig('india_gps_noise_components_pie.png', dpi=300)
    plt.show()

# Generate example routes with different noise levels
def generate_example_routes():
    """Generate example routes with different noise profiles"""
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define starting points for routes (major Indian cities)
    start_points = {
        'Mumbai': (19.076, 72.8777),
        'Delhi': (28.7041, 77.1025),
        'Bangalore': (12.9716, 77.5946)
    }
    
    # Generate routes with different noise levels
    all_routes = {}
    
    for city, coords in start_points.items():
        for noise_level in ['low', 'medium', 'high']:
            route_df = generate_gps_route_with_noise(
                coords[0], coords[1],
                num_points=100,
                route_type="realistic",
                noise_level=noise_level,
                record_noise=True
            )
            
            # Add city and noise level info
            route_df['city'] = city
            route_df['noise_level'] = noise_level
            
            # Save to file
            filename = f"{city.lower()}_route_{noise_level}_noise.csv"
            route_df.to_csv(filename, index=False)
            print(f"Generated route with {noise_level} noise in {city} - Saved to {filename}")
            
            # Plot the route
            plot_noisy_route(route_df, f"GPS Route in {city} with {noise_level.capitalize()} Noise")
            
            # Analyze noise characteristics
            analyze_noise(route_df)
            
            all_routes[f"{city}_{noise_level}"] = route_df
    
    return all_routes

# Example usage to generate a single route with detailed noise analysis
if __name__ == "__main__":
    # Starting in Hyderabad
    hyd_lat , hyd_lon = 17.3850, 78.4867

    # Generate a noisy GPS route
    route = generate_gps_route_with_noise(
        hyd_lat , hyd_lon,
        num_points=200,
        route_type="realistic",
        noise_level="medium",  # Choose from: low, medium, high
        record_noise=True,
        simulate_signal_loss=True
    )
    
    # Save to CSV
    route.to_csv("/content/track_points.csv", index=False)
    print(f"Generated route with {len(route)} points - Saved to mumbai_gps_route_with_noise.csv")
    
    # Visualize the route
    plot_noisy_route(route, "GPS Route in Mumbai with Noise")
    
    # Analyze the noise characteristics
    analyze_noise(route)
    
    print("\nDone!")
