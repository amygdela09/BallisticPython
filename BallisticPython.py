import math

def get_float_input(prompt):
    """
    Prompts the user for a floating-point number and handles invalid input.
    """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def calculate_trajectory(mass, velocity, angle, area, drag_coefficient, time_step=0.01):
    """
    Calculates the trajectory of a projectile considering air resistance.

    Args:
        mass (float): Mass of the projectile in kilograms (kg).
        velocity (float): Initial velocity (muzzle velocity) in meters/second (m/s).
        angle (float): Launch angle in degrees.
        area (float): Cross-sectional area of the projectile in square meters (m^2).
        drag_coefficient (float): Drag coefficient of the projectile (dimensionless).
        time_step (float): The time step for the simulation in seconds.

    Returns:
        A tuple containing:
        - range (float): The horizontal distance traveled in meters.
        - max_height (float): The maximum height reached in meters.
        - time_of_flight (float): The total time of flight in seconds.
        - trajectory (list): A list of (x, y) coordinates representing the path.
    """
    # --- Constants ---
    g = 9.81  # Acceleration due to gravity (m/s^2)
    rho = 1.225  # Density of air at sea level (kg/m^3)

    # --- Initial Conditions ---
    angle_rad = math.radians(angle)
    vx = velocity * math.cos(angle_rad)
    vy = velocity * math.sin(angle_rad)
    x = 0.0
    y = 0.0
    t = 0.0
    
    max_height = 0.0
    trajectory = [(x, y)]

    print("\nSimulating trajectory...")

    # --- Simulation Loop ---
    # The loop continues as long as the projectile is above the ground (y >= 0)
    while y >= 0:
        # Calculate current speed
        current_velocity = math.sqrt(vx**2 + vy**2)
        
        # Calculate drag force (Fd = 0.5 * rho * v^2 * A * Cd)
        # This force opposes the direction of velocity.
        drag_force_magnitude = 0.5 * rho * (current_velocity**2) * area * drag_coefficient
        
        # Resolve drag force into x and y components
        drag_force_x = -drag_force_magnitude * (vx / current_velocity)
        drag_force_y = -drag_force_magnitude * (vy / current_velocity)
        
        # Calculate net forces
        force_x = drag_force_x
        force_y = drag_force_y - (mass * g) # Gravity acts downwards
        
        # Calculate acceleration (a = F/m)
        ax = force_x / mass
        ay = force_y / mass
        
        # Update velocity using the calculated acceleration
        vx += ax * time_step
        vy += ay * time_step
        
        # Update position
        x += vx * time_step
        y += vy * time_step
        
        # Update time
        t += time_step
        
        # Track maximum height
        if y > max_height:
            max_height = y
            
        # Store the current position for plotting/analysis
        if y >= 0:
            trajectory.append((x, y))

    print("Simulation complete.")
    
    # --- Final Calculations ---
    # To get a more accurate range, we can interpolate the last two points
    # to find where the trajectory crosses y=0.
    if len(trajectory) > 1:
        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]
        # Using linear interpolation: x = x1 - y1 * (x2 - x1) / (y2 - y1)
        final_range = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        final_range = x

    return final_range, max_height, t

def main():
    """
    Main function to run the ballistic computer.
    """
    print("--- Simple Ballistic Computer ---")
    print("This script calculates the trajectory of a projectile with air resistance.")
    print("Please provide the following inputs:")

    # --- Get User Inputs ---
    muzzle_velocity = get_float_input("Enter muzzle velocity (m/s): ")
    launch_angle = get_float_input("Enter launch angle (degrees): ")
    projectile_mass = get_float_input("Enter projectile mass (kg): ")
    projectile_area = get_float_input("Enter projectile cross-sectional area (m^2): ")
    drag_coeff = get_float_input("Enter drag coefficient (e.g., 0.47 for a sphere): ")
    
    # --- Run Calculation ---
    flight_range, max_height, time_of_flight = calculate_trajectory(
        projectile_mass, 
        muzzle_velocity, 
        launch_angle, 
        projectile_area, 
        drag_coeff
    )

    # --- Display Results ---
    print("\n--- Ballistic Solution ---")
    print(f"Time of Flight: {time_of_flight:.2f} seconds")
    print(f"Maximum Height: {max_height:.2f} meters")
    print(f"Range (Distance): {flight_range:.2f} meters")
    print("--------------------------")

if __name__ == "__main__":
    main()
