# ==============================================================================
#
#                           Ballistic Solver Script
#
#  This script functions as a sophisticated ballistic computer. It takes
#  a projectile profile (pre-loaded or manual) and environmental data to
#  calculate a firing solution for a given target range.
#
#  The primary function is to solve for the required launch angle (holdover)
#  to hit a target at a known distance, accounting for gravity, atmospheric
#  drag, and wind deflection. It also estimates hit probability based on
#  weapon system accuracy and a standard adult human target.
#
# ==============================================================================

import math
import sys
import scipy.stats as st # Make sure to install this: pip install scipy

# --- GLOBAL CONSTANTS ---

# The G1 standard projectile, a common reference for calculating ballistic
# coefficients, has a standard drag coefficient (Cd) of approximately 0.519.
# This constant is used when converting a bullet's Ballistic Coefficient (BC)
# into a raw drag coefficient for the simulation.
G1_STD_DRAG_COEFF = 0.519

# This dictionary stores pre-loaded ballistic data for common ammunition types.
# Each entry contains the projectile's mass, muzzle velocity, diameter, its
# G1 Ballistic Coefficient, and a new 'moa_accuracy' field.
CALIBER_DATA = {
    "1": {"name": "6.5 Creedmoor (143gr Hornady ELD-X)", "mass_kg": 0.009266, "velocity_mps": 822.96, "diameter_m": 0.0067056, "g1_bc_lb_in2": 0.625, "moa_accuracy": 0.75}, # Example accuracy
    "2": {"name": ".308 Winchester (168gr Fed. MatchKing)", "mass_kg": 0.010886, "velocity_mps": 792.48, "diameter_m": 0.0078232, "g1_bc_lb_in2": 0.462, "moa_accuracy": 1.0},
    "3": {"name": ".338 Lapua Magnum (250gr Lapua Scenar)", "mass_kg": 0.01620, "velocity_mps": 900.0, "diameter_m": 0.0085852, "g1_bc_lb_in2": 0.625, "moa_accuracy": 0.5},
    "4": {"name": ".50 BMG (750gr Hornady A-MAX)", "mass_kg": 0.048598, "velocity_mps": 859.536, "diameter_m": 0.012954, "g1_bc_lb_in2": 1.050, "moa_accuracy": 1.25},
    "5": {"name": "Manual Input"},
}

# Conversion factor for degrees to Milliradians (MILs).
# 1 degree is approximately 17.4533 MILs.
DEG_TO_MIL = 17.4533

# --- TARGET DIMENSIONS (APPROXIMATE adult male standing target) ---
TARGET_HEIGHT_M = 1.75 # meters
TARGET_WIDTH_M = 0.5   # meters


# ==============================================================================
#
#                            HELPER FUNCTIONS
#
#  These functions handle user input and data validation to ensure the main
#  program receives clean, usable data.
#
# ==============================================================================

def get_float_input(prompt, default=None):
    """
    Prompts the user for a floating-point number and handles invalid input.
    If a default value is provided, it's used when the user just presses Enter.
    """
    while True:
        try:
            val_str = input(prompt)
            if not val_str and default is not None:
                return default
            return float(val_str)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_menu_choice(prompt, choices, default=None):
    """
    Prompts the user to choose from a dictionary of options.
    If a default value is provided, it's used when the user just presses Enter.
    """
    while True:
        choice = input(prompt)
        if not choice and default is not None:
            return default
        if choice in choices:
            return choice
        else:
            print("Invalid selection.")


# ==============================================================================
#
#                         PHYSICS & CALCULATION FUNCTIONS
#
#  These functions form the core of the ballistic engine. They calculate the
#  physical properties and simulate the projectile's flight.
#
# ==============================================================================

def calculate_air_density(elevation_m=0, temp_c=15, humidity_percent=50):
    """
    Calculates air density (rho) in kg/m^3 based on atmospheric conditions.
    This uses the International Standard Atmosphere (ISA) model to first find
    the standard pressure at a given elevation, then adjusts for the actual
    temperature and humidity. Denser air creates more drag.
    """
    # --- Step 1: Calculate pressure at altitude using the barometric formula ---
    P0 = 1013.25  # Standard sea-level pressure in hectopascals (hPa)
    T0 = 288.15   # Standard sea-level temperature in Kelvin (15°C)
    L = 0.0065    # Standard temperature lapse rate in K/m
    EXPONENT = 5.25588 # Exponent derived from (g*M)/(R*L)

    # Calculate pressure at the given elevation
    pressure_hpa = P0 * (1 - (L * elevation_m) / T0) ** EXPONENT

    # --- Step 2: Adjust for humidity and actual temperature ---
    temp_k = temp_c + 273.15  # Convert current temperature to Kelvin
    # Saturation vapor pressure and partial pressure of water vapor
    p_sat = 6.1078 * 10**((7.5 * temp_c) / (temp_c + 237.3))
    p_v = (humidity_percent / 100) * p_sat
    # Pressure of dry air
    p_d = pressure_hpa - p_v

    # Gas constants for dry air and water vapor (J/(kg*K))
    R_d, R_v = 287.058, 461.495

    # Final density is the sum of the densities of dry air and water vapor
    rho = (p_d * 100) / (R_d * temp_k) + (p_v * 100) / (R_v * temp_k)
    return rho

def calculate_derived_properties(profile):
    """
    Calculates simulation-specific properties from the base projectile profile.
    This function converts the user-friendly Ballistic Coefficient (BC) into
    the Drag Coefficient (Cd) that the physics simulation requires.
    """
    # Calculate the projectile's cross-sectional area (m^2)
    radius = profile["diameter_m"] / 2
    profile["area_m2"] = math.pi * (radius ** 2)

    # Convert the G1 BC from imperial (lb/in^2) to metric (kg/m^2)
    bc_kg_m2 = profile["g1_bc_lb_in2"] * 703.06958
    # Calculate sectional density (mass per unit of cross-sectional area)
    sectional_density_kg_m2 = profile["mass_kg"] / profile["area_m2"]

    # The form factor 'i' relates the bullet's drag to the G1 standard projectile.
    form_factor = sectional_density_kg_m2 / bc_kg_m2

    # The final drag coefficient (Cd) is the form factor multiplied by the
    # G1 standard projectile's drag coefficient.
    profile["drag_coefficient"] = form_factor * G1_STD_DRAG_COEFF
    return profile

def calculate_trajectory(angle, profile, wind_speed_mps, wind_angle_deg, air_density):
    """
    Simulates the flight of a projectile for a given launch angle.
    This is the core physics engine, using the Euler method for numerical integration.
    It steps through the flight path in small time increments, applying forces
    of gravity and drag at each step to find the final landing point.

    Coordinate System:
    - x: Downrange distance from the shooter.
    - y: Vertical distance (height) from the barrel line.
    - z: Horizontal distance (wind drift) from the barrel line.
    """
    g = 9.81  # Acceleration due to gravity (m/s^2)

    # --- Initial Conditions ---
    launch_angle_rad = math.radians(angle)
    # Resolve initial velocity into x, y, and z components
    vx = profile['velocity_mps'] * math.cos(launch_angle_rad)
    vy = profile['velocity_mps'] * math.sin(launch_angle_rad)
    vz = 0.0
    # Initialize position
    x, y, z = 0.0, 0.0, 0.0

    # --- Wind Components ---
    # Resolve wind speed into headwind (affecting drag) and crosswind (pushing the bullet)
    wind_angle_rad = math.radians(wind_angle_deg)
    headwind_mps = wind_speed_mps * math.cos(wind_angle_rad)
    crosswind_mps = wind_speed_mps * math.sin(wind_angle_rad)

    # Using a smaller time step for the solver improves accuracy
    time_step = 0.005

    # --- Simulation Loop ---
    # Continue the simulation as long as the projectile is above the ground (y >= 0)
    while y >= 0:
        # Calculate the projectile's velocity relative to the air mass
        vx_rel = vx - headwind_mps
        vy_rel = vy
        vz_rel = vz - crosswind_mps
        v_rel_mag = math.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
        if v_rel_mag == 0: break # Avoid division by zero if projectile stops

        # Calculate the magnitude of the drag force using the drag equation
        drag_force_mag = 0.5 * air_density * (v_rel_mag**2) * profile['area_m2'] * profile['drag_coefficient']

        # Resolve total drag force into x, y, and z components
        force_x = -drag_force_mag * (vx_rel / v_rel_mag)
        force_y = -drag_force_mag * (vy_rel / v_rel_mag) - (profile['mass_kg'] * g) # Include gravity
        force_z = -drag_force_mag * (vz_rel / v_rel_mag)

        # Update velocity for the next time step using acceleration (a = F/m)
        vx += (force_x / profile['mass_kg']) * time_step
        vy += (force_y / profile['mass_kg']) * time_step
        vz += (force_z / profile['mass_kg']) * time_step

        # Update position for the next time step
        x += vx * time_step
        y += vy * time_step
        z += vz * time_step

        # Failsafe to prevent an infinite loop if something goes wrong
        if y < -10: break

    # --- Final Point Calculation ---
    # Interpolate the final landing point (where y=0) for better accuracy
    final_range = x - (vx * y / vy) if vy != 0 else x
    final_deflection = z - (vz * y / vy) if vy != 0 else z

    return final_range, final_deflection

def solve_for_angle(target_range_m, profile, wind_speed_mps, wind_angle_deg, air_density):
    """
    Finds the required launch angle to hit a target at a specific range.
    This function is an iterative solver. It uses the secant method, a root-finding
    algorithm, to determine the angle where the output of `calculate_trajectory`
    matches the desired `target_range_m`.
    """
    # Initial low and high guesses for the launch angle in degrees
    angle_low, angle_high = 0.0, 5.0

    # Calculate the trajectory range for the initial guesses
    range_low, _ = calculate_trajectory(angle_low, profile, wind_speed_mps, wind_angle_deg, air_density)
    range_high, _ = calculate_trajectory(angle_high, profile, wind_speed_mps, wind_angle_deg, air_density)

    # Check if the target range is bracketed by the initial guesses. If not, it's likely unreachable.
    if not (range_low < target_range_m < range_high):
        return None, None # Return None if no solution is possible

    # Iterate to refine the angle guess
    for _ in range(20): # Limit to 20 iterations to prevent infinite loops
        if abs(range_high - range_low) < 1e-6: return None, None # Avoid division by zero
        
        # The secant method finds a new, better guess for the angle
        angle_next = angle_high - (range_high - target_range_m) * (angle_high - angle_low) / (range_high - range_low)
        
        # Calculate the trajectory for the new guess
        range_next, deflection = calculate_trajectory(angle_next, profile, wind_speed_mps, wind_angle_deg, air_density)

        # If the new range is within a tolerance (0.1m) of the target, we found the solution
        if abs(range_next - target_range_m) < 0.1:
            return angle_next, deflection

        # Update the high or low bound for the next iteration
        if range_next < target_range_m:
            angle_low, range_low = angle_next, range_next
        else:
            angle_high, range_high = angle_next, range_next
            
    return None, None # Return None if the solver fails to converge on a solution

def calculate_hit_probability(moa_accuracy, target_range_m, target_height_m, target_width_m):
    """
    Calculates the hit probability assuming a circular normal distribution
    of shots centered on the target, and a rectangular target.
    """
    # MOA spread in meters at target range:
    # 1 MOA at 100 yards (91.44m) is 1.047 inches (0.0265938 m)
    # So, moa_spread_at_range_m for 1 MOA at target_range_m:
    # (0.0265938 m / 91.44 m) * target_range_m = 0.000290888 * target_range_m
    
    # Total MOA spread at range in meters
    moa_spread_at_range_m = (moa_accuracy * target_range_m / 91.44) * 0.0254
    
    # Standard deviation (sigma) of the shot group.
    # Assuming 'moa_accuracy' is the *diameter* of the group at range,
    # and this group represents roughly the 95% confidence interval (approx 4 standard deviations diameter).
    # So, sigma_xy = moa_spread_at_range_m / 4.0
    # Or, if MOA is usually the 1-sigma spread, sigma_xy = moa_spread_at_range_m.
    # For a typical advertised MOA group, it's often the maximum spread for a significant portion of shots.
    # A common rule of thumb for converting advertised group size to standard deviation for a Gaussian:
    # sigma = Group Size / 2.355 (if Group Size is CEP, 50%)
    # sigma = Group Size / 4 to 6 (if Group Size is max spread)
    # Let's use a conservative approximation where MOA_accuracy represents the 2-sigma radius (95% of shots).
    sigma_xy = moa_spread_at_range_m / 2.0 # Roughly half the stated MOA spread as std dev

    # Half-width and half-height of the target
    half_width = target_width_m / 2.0
    half_height = target_height_m / 2.0

    # Calculate the probability for the x-dimension (width)
    prob_x = st.norm.cdf(half_width, loc=0, scale=sigma_xy) - st.norm.cdf(-half_width, loc=0, scale=sigma_xy)
    
    # Calculate the probability for the y-dimension (height)
    prob_y = st.norm.cdf(half_height, loc=0, scale=sigma_xy) - st.norm.cdf(-half_height, loc=0, scale=sigma_xy)
    
    # The total probability of hitting the rectangle is the product of the two
    total_probability = prob_x * prob_y
    
    return total_probability


# ==============================================================================
#
#                         MAIN PROGRAM EXECUTION
#
#  This section controls the user interface, setup, and the main calculation
#  loop of the script.
#
# ==============================================================================

def setup_profile():
    """
    Handles the one-time setup for the projectile profile and scope units.
    This function is called once at the start or when the user types 'profile'.
    """
    print("--- Ballistic Profile Setup ---")
    for key, value in CALIBER_DATA.items(): print(f"  {key}: {value['name']}")
    choice_key = get_menu_choice("Select Profile: ", CALIBER_DATA)
    
    profile = {}
    if choice_key == "5": # Manual user input
        print("\n--- Manual Projectile Data ---")
        mass_gr = get_float_input("Mass (grains): ")
        vel_fps = get_float_input("Muzzle Velocity (fps): ")
        diam_in = get_float_input("Diameter (in): ")
        bc_g1 = get_float_input("G1 Ballistic Coefficient: ")
        moa_acc = get_float_input("Weapon System Accuracy (MOA): ")
        profile.update({
            'mass_kg': mass_gr * 0.0000647989,
            'velocity_mps': vel_fps * 0.3048,
            'diameter_m': diam_in * 0.0254,
            'g1_bc_lb_in2': bc_g1,
            'name': "Manual Entry",
            'moa_accuracy': moa_acc
        })
    else: # Use pre-loaded data
        profile = CALIBER_DATA[choice_key].copy() # Use .copy() to avoid modifying the global CALIBER_DATA
    
    # Let the user choose their scope's adjustment unit (MIL is the default)
    unit_choices = {"1": "MIL", "2": "MOA"}
    print("\nSelect Adjustment Unit:")
    for key, value in unit_choices.items(): print(f"  {key}: {value}")
    unit_choice = get_menu_choice("Unit (default: 1 for MIL): ", unit_choices, default="1")
    unit = unit_choices[unit_choice]
    
    # Return the fully processed profile and the chosen unit
    return calculate_derived_properties(profile), unit

def setup_environment():
    """
    Handles the one-time setup for environmental conditions.
    Called at the start or when the user types 'env'.
    """
    print("\n--- Environmental Setup (press Enter for defaults) ---")
    elevation_ft = get_float_input("Elevation ft (default: 0): ", 0)
    temp_f = get_float_input("Temperature °F (default: 59): ", 59)
    humidity = get_float_input("Humidity % (default: 50): ", 50)
    
    # Convert imperial units to metric for the physics calculations
    elevation_m = elevation_ft * 0.3048
    temp_c = (temp_f - 32) * 5/9
    
    # Return the final calculated air density
    return calculate_air_density(elevation_m, temp_c, humidity)

def main():
    """
    The main function that runs the user interface and calculation loop.
    """
    # Perform initial setup for the rifle profile and environment
    derived_profile, unit = setup_profile()
    air_density = setup_environment()
    
    # Display the current settings to the user
    print("\n--- Ready for Firing Solution ---")
    print(f"Profile: {derived_profile['name']} | Unit: {unit} | Air Density: {air_density:.4f} kg/m^3 | Accuracy: {derived_profile['moa_accuracy']} MOA")
    print("Commands: 'quit', 'profile' (to reset rifle & unit), 'env'")

    # --- Main Calculation Loop ---
    # This loop allows for rapid, iterative calculations without restarting the script.
    while True:
        try:
            # Prompt the user for the primary variables
            prompt = "Enter [Range yd] [Wind mph] [Wind deg]: "
            user_input = input(prompt)

            # --- Handle User Commands ---
            if user_input.lower() in ['q', 'quit', 'exit']:
                sys.exit("Exiting Ballistic Solver.")
            if user_input.lower() == 'profile':
                derived_profile, unit = setup_profile()
                print(f"\nProfile changed to: {derived_profile['name']} | Unit: {unit} | Accuracy: {derived_profile['moa_accuracy']} MOA")
                continue # Restart loop
            if user_input.lower() == 'env':
                air_density = setup_environment()
                print(f"\nEnvironment updated. Air Density: {air_density:.4f} kg/m^3")
                continue # Restart loop

            # --- Parse User Input for a Firing Solution ---
            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input. Please provide three values separated by spaces.")
                continue

            # Convert parsed strings to numbers
            target_range_yd, wind_speed_mph, wind_angle = map(float, parts)
            # Convert inputs to metric for the simulation
            target_range_m = target_range_yd * 0.9144
            wind_speed_mps = wind_speed_mph * 0.44704

            # --- Call the Solver ---
            # This is the main call to find the required angle and resulting deflection
            solution_angle, solution_deflection = solve_for_angle(
                target_range_m, derived_profile, wind_speed_mps, wind_angle, air_density
            )
            
            # --- Format and Display the Output ---
            if solution_angle is not None:
                # Convert the elevation angle (holdover) to the user's chosen unit (MOA or MIL)
                hold_adjustment = solution_angle * DEG_TO_MIL if unit == "MIL" else solution_angle * 60
                
                # Convert the linear wind deflection (in meters) to an angular value for scope adjustment
                deflection_angle_deg = math.degrees(math.atan(solution_deflection / target_range_m))
                wind_adjustment = deflection_angle_deg * DEG_TO_MIL if unit == "MIL" else deflection_angle_deg * 60
                
                # Also provide the linear deflection in inches for reference
                deflection_in = solution_deflection * 39.3701

                # Calculate hit probability
                hit_prob = calculate_hit_probability(
                    derived_profile['moa_accuracy'],
                    target_range_m,
                    TARGET_HEIGHT_M,
                    TARGET_WIDTH_M
                )
                
                # Print the final, formatted firing solution
                print(f"  HOLD: {hold_adjustment:.1f} {unit} UP | "
                      f"WIND: {abs(wind_adjustment):.1f} {unit} {'Right' if solution_deflection > 0 else 'Left'} ({abs(deflection_in):.1f} in) | "
                      f"HIT PROB: {hit_prob:.1%}")
            else:
                # If the solver returns None, the target is unreachable
                print("  SOLUTION NOT FOUND. Target may be out of range.")

        except (ValueError, IndexError):
            print("Invalid format. Please use: <range> <wind_speed> <wind_direction>")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An error occurred: {e}")

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()
