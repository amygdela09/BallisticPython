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
import json
import os

# Attempt to import scipy.stats; if it fails, use the custom erf-based CDF.
try:
    import scipy.stats as st
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not found. Hit probability will use a less precise approximation.")

# Attempt to import matplotlib.pyplot; if it fails, plotting feature will be disabled.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not found. Plotting feature will be unavailable.")


# --- GLOBAL CONSTANTS ---

# The G1 standard projectile, a common reference for calculating ballistic
# coefficients, has a standard drag coefficient (Cd) of approximately 0.519.
G1_STD_DRAG_COEFF = 0.519

# This dictionary stores pre-loaded ballistic data for common ammunition types,
# now organized into nested submenus by caliber family.
CALIBER_DATA = {
    "1": {
        "name": "6.5 Creedmoor Profiles",
        "loads": {
            "1": {"name": "143gr Hornady ELD-X", "mass_kg": 0.009266, "velocity_mps": 822.96, "diameter_m": 0.0067056, "g1_bc_lb_in2": 0.625, "moa_accuracy": 0.75},
            "2": {"name": "140gr Hornady ELD Match", "mass_kg": 0.009072, "velocity_mps": 829.0, "diameter_m": 0.0067056, "g1_bc_lb_in2": 0.640, "moa_accuracy": 0.65},
            "3": {"name": "130gr Berger VLD Hunting", "mass_kg": 0.008424, "velocity_mps": 870.0, "diameter_m": 0.0067056, "g1_bc_lb_in2": 0.562, "moa_accuracy": 0.7}
        }
    },
    "2": {
        "name": ".308 Winchester Profiles",
        "loads": {
            "1": {"name": "168gr Fed. MatchKing", "mass_kg": 0.010886, "velocity_mps": 792.48, "diameter_m": 0.0078232, "g1_bc_lb_in2": 0.462, "moa_accuracy": 1.0},
            "2": {"name": "175gr Sierra MatchKing", "mass_kg": 0.011340, "velocity_mps": 780.0, "diameter_m": 0.0078232, "g1_bc_lb_in2": 0.505, "moa_accuracy": 0.9},
            "3": {"name": "150gr FMJ", "mass_kg": 0.009720, "velocity_mps": 850.0, "diameter_m": 0.0078232, "g1_bc_lb_in2": 0.390, "moa_accuracy": 1.2}
        }
    },
    "3": {
        "name": ".338 Lapua Magnum Profiles",
        "loads": {
            "1": {"name": "250gr Lapua Scenar", "mass_kg": 0.01620, "velocity_mps": 900.0, "diameter_m": 0.0085852, "g1_bc_lb_in2": 0.625, "moa_accuracy": 0.5},
            "2": {"name": "300gr Sierra MatchKing", "mass_kg": 0.019440, "velocity_mps": 825.0, "diameter_m": 0.0085852, "g1_bc_lb_in2": 0.768, "moa_accuracy": 0.4},
            "3": {"name": "250gr Hornady A-MAX", "mass_kg": 0.01620, "velocity_mps": 890.0, "diameter_m": 0.0085852, "g1_bc_lb_in2": 0.670, "moa_accuracy": 0.55}
        }
    },
    "4": {
        "name": ".50 BMG Profiles",
        "loads": {
            "1": {"name": "750gr Hornady A-MAX", "mass_kg": 0.048598, "velocity_mps": 859.536, "diameter_m": 0.012954, "g1_bc_lb_in2": 1.050, "moa_accuracy": 1.25},
            "2": {"name": "660gr FMJ (M33)", "mass_kg": 0.042768, "velocity_mps": 900.0, "diameter_m": 0.012954, "g1_bc_lb_in2": 0.700, "moa_accuracy": 1.5}
        }
    },
    "5": {
        "name": "5.56 NATO / .223 Rem Profiles",
        "loads": {
            "1": {"name": "55gr FMJ", "mass_kg": 0.003564, "velocity_mps": 944.88, "diameter_m": 0.00569, "g1_bc_lb_in2": 0.245, "moa_accuracy": 1.5},
            "2": {"name": "62gr FMJ/M855", "mass_kg": 0.0040175, "velocity_mps": 914.4, "diameter_m": 0.00569, "g1_bc_lb_in2": 0.304, "moa_accuracy": 1.25},
            "3": {"name": "69gr Sierra MatchKing", "mass_kg": 0.004471, "velocity_mps": 883.92, "diameter_m": 0.00569, "g1_bc_lb_in2": 0.335, "moa_accuracy": 1.0},
            "4": {"name": "77gr OTM", "mass_kg": 0.0049895, "velocity_mps": 838.2, "diameter_m": 0.00569, "g1_bc_lb_in2": 0.372, "moa_accuracy": 0.75}
        }
    },
    "6": { "name": "Manual Input", "loads": {} } # Manual input at top level
}


# Conversion factor for degrees to Milliradians (MILs).
DEG_TO_MIL = 17.4533

# --- TARGET DIMENSIONS (APPROXIMATE adult male standing target) ---
TARGET_HEIGHT_M = 1.75 # meters
TARGET_WIDTH_M = 0.5   # meters
TARGET_MIDPOINT_M = TARGET_HEIGHT_M / 2 # Midpoint for aiming

# --- Profile Management File ---
PROFILE_FILE = 'rifle_profiles.json'

# --- Unit Conversion Factors ---
YD_TO_M = 0.9144
IN_TO_M = 0.0254
FT_TO_M = 0.3048
MPH_TO_MPS = 0.44704
GR_TO_KG = 0.0000647989


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
#                           PROFILE MANAGEMENT
#
# ==============================================================================

def load_profiles():
    """Loads saved rifle profiles from JSON file."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    """Saves current rifle profiles to JSON file."""
    with open(PROFILE_FILE, 'w') as f:
        json.dump(profiles, f, indent=4)

def manage_profiles(current_profiles):
    """Allows user to view and delete saved profiles."""
    print("\n--- Manage Saved Profiles ---")
    if not current_profiles:
        print("No custom profiles saved yet.")
        return current_profiles

    print("Saved Profiles:")
    profile_names = list(current_profiles.keys())
    for i, name in enumerate(profile_names):
        print(f"  {i+1}: {name}")
    print("  0: Back to main menu")

    while True:
        choice_str = input("Enter number to delete, or 0 to go back: ")
        try:
            choice = int(choice_str)
            if choice == 0:
                return current_profiles
            elif 1 <= choice <= len(profile_names):
                name_to_delete = profile_names[choice-1]
                confirm = input(f"Are you sure you want to delete '{name_to_delete}'? (y/N): ").lower()
                if confirm == 'y':
                    del current_profiles[name_to_delete]
                    save_profiles(current_profiles)
                    print(f"Profile '{name_to_delete}' deleted.")
                    return current_profiles # Return updated profiles and exit management
                else:
                    print("Deletion cancelled.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# ==============================================================================
#
#                         PHYSICS & CALCULATION FUNCTIONS
#
# ==============================================================================

def calculate_air_density(elevation_m=0, temp_c=15, humidity_percent=50):
    """
    Calculates air density (rho) in kg/m^3 based on atmospheric conditions.
    """
    P0 = 1013.25  # Standard sea-level pressure in hectopascals (hPa)
    T0 = 288.15   # Standard sea-level temperature in Kelvin (15°C)
    L = 0.0065    # Standard temperature lapse rate in K/m
    EXPONENT = 5.25588 

    pressure_hpa = P0 * (1 - (L * elevation_m) / T0) ** EXPONENT

    temp_k = temp_c + 273.15
    p_sat = 6.1078 * 10**((7.5 * temp_c) / (temp_c + 237.3))
    p_v = (humidity_percent / 100) * p_sat
    p_d = pressure_hpa - p_v

    R_d, R_v = 287.058, 461.495

    rho = (p_d * 100) / (R_d * temp_k) + (p_v * 100) / (R_v * temp_k)
    return rho

def calculate_derived_properties(profile):
    """
    Calculates simulation-specific properties from the base projectile profile.
    """
    radius = profile["diameter_m"] / 2
    profile["area_m2"] = math.pi * (radius ** 2)

    bc_kg_m2 = profile["g1_bc_lb_in2"] * 703.06958
    sectional_density_kg_m2 = profile["mass_kg"] / profile["area_m2"]

    form_factor = sectional_density_kg_m2 / bc_kg_m2

    profile["drag_coefficient"] = form_factor * G1_STD_DRAG_COEFF
    return profile

def calculate_trajectory_with_points(angle, profile, wind_speed_mps, wind_angle_deg, air_density, scope_height_m, target_range_for_solver_m, scope_zero_m, barrel_angle_for_zero_m):
    """
    Simulates the flight of a projectile for a given launch angle,
    and returns a list of (x, y) points for plotting, the height at target range,
    and a flag if the bullet hit the ground before reaching the target range.
    y coordinate is height relative to shooter's ground level.
    Also tracks line of sight (LOS) intersection points relative to a horizontal LOS at scope_height_m.
    """
    g = 9.81  # Acceleration due to gravity (m/s^2)

    # --- Initial Conditions ---
    launch_angle_rad = math.radians(angle)
    vx = profile['velocity_mps'] * math.cos(launch_angle_rad)
    vy = profile['velocity_mps'] * math.sin(launch_angle_rad)
    vz = 0.0
    x, y, z = 0.0, scope_height_m, 0.0 # Start at scope height above shooter's ground

    # Store trajectory points for plotting
    trajectory_points = [(x, y)]
    los_intersection_points = []
    
    # Store projectile height at target_range_for_solver_m (used by the solver)
    height_at_target_range = None
    hit_ground_before_target_range = False # Flag to track if ground hit occurred before target

    # Line of sight for ZEROING: From (0, scope_height_m) to (scope_zero_m, 0)
    # This function defines the specific zeroing LOS that the bullet trajectory should cross.
    def get_zeroing_los_height_at_x(current_x):
        if scope_zero_m == 0: # If zero is at muzzle, LOS is horizontal at scope height
            return scope_height_m
        # LOS is a line from (0, scope_height_m) to (scope_zero_m, 0)
        # Equation: y = y1 + m(x - x1) where (x1, y1) = (0, scope_height_m) and m = (0 - scope_height_m) / scope_zero_m
        return scope_height_m + (current_x * (-scope_height_m / scope_zero_m))


    # Variables to track previous state for interpolation for LOS intersections
    prev_x, prev_y = x, y
    prev_los_y = get_zeroing_los_height_at_x(prev_x)
    prev_bullet_above_los = y > prev_los_y 


    # --- Wind Components ---
    wind_angle_rad = math.radians(wind_angle_deg)
    headwind_mps = wind_speed_mps * math.cos(wind_angle_rad)
    crosswind_mps = wind_speed_mps * math.sin(wind_angle_rad)

    time_step = 0.005

    # --- Simulation Loop ---
    max_sim_range = max(target_range_for_solver_m * 1.5, scope_zero_m * 1.5, 1000.0)
    if profile['velocity_mps'] > 1500: max_sim_range = max(max_sim_range, 2000.0)

    while True:
        # Check if the projectile has hit the shooter's ground (y <= 0)
        if y <= 0 and prev_y > 0: # Check for crossing the ground from above
            # Interpolate to find exact ground impact point
            ratio = prev_y / (prev_y - y)
            ground_impact_x = prev_x + (x - prev_x) * ratio
            trajectory_points.append((ground_impact_x, 0.0))
            
            if ground_impact_x < target_range_for_solver_m - 0.1: # Small buffer
                hit_ground_before_target_range = True
                height_at_target_range = 0.0 # If hit ground before target, treat height at target as 0
            break # Terminate simulation

        # Capture height at target range for the solver
        # Only capture if we are approaching or just crossed it for the first time
        if height_at_target_range is None and prev_x < target_range_for_solver_m and x >= target_range_for_solver_m:
            if x - prev_x != 0:
                ratio_to_target = (target_range_for_solver_m - prev_x) / (x - prev_x)
                height_at_target_range = prev_y + (y - prev_y) * ratio_to_target
            else: # Vertical trajectory or very small step
                height_at_target_range = y

        # Check for LOS intersection (zeroing point)
        current_los_y = get_zeroing_los_height_at_x(x)
        current_bullet_above_los = y > current_los_y

        # Only check for intersection if we've moved beyond a very small distance from muzzle
        # to avoid false positives at the very start where bullet is at scope height and LOS starts there.
        # Also ensure scope_zero_m is not 0 for this specific LOS calculation (handled in get_zeroing_los_height_at_x).
        if x > 0.01 and scope_zero_m != 0 and current_bullet_above_los != prev_bullet_above_los: # If status changed
            # Calculate difference between bullet y and LOS y at previous and current points
            diff_prev_y_los = prev_y - get_zeroing_los_height_at_x(prev_x)
            diff_curr_y_los = y - current_los_y

            if diff_prev_y_los * diff_curr_y_los < 0: # If signs are different, a crossing occurred
                # Linear interpolation for x
                lerp_factor = abs(diff_prev_y_los) / (abs(diff_prev_y_los) + abs(diff_curr_y_los))
                
                intersect_x = prev_x + (x - prev_x) * lerp_factor
                intersect_y = prev_y + (y - prev_y) * lerp_factor 
                los_intersection_points.append((intersect_x, intersect_y))

        # --- Physics calculations for next step ---
        v_air_x = vx - headwind_mps
        v_air_y = vy
        v_air_z = vz - crosswind_mps
        v_rel_mag = math.sqrt(v_air_x**2 + v_air_y**2 + v_air_z**2)
        
        if v_rel_mag == 0: break # Avoid division by zero if projectile stops

        drag_force_mag = 0.5 * air_density * (v_rel_mag**2) * profile['area_m2'] * profile['drag_coefficient']

        force_x = -drag_force_mag * (v_air_x / v_rel_mag)
        force_y = -drag_force_mag * (v_air_y / v_rel_mag) - (profile['mass_kg'] * g)
        force_z = -drag_force_mag * (v_air_z / v_rel_mag)

        vx += (force_x / profile['mass_kg']) * time_step
        vy += (force_y / profile['mass_kg']) * time_step
        vz += (force_z / profile['mass_kg']) * time_step

        x_new = x + vx * time_step
        y_new = y + vy * time_step
        z_new = z + vz * time_step
        
        # Update for next iteration
        prev_x, prev_y = x, y
        prev_bullet_above_los = current_bullet_above_los
        x, y, z = x_new, y_new, z_new
        trajectory_points.append((x, y))

        # Termination conditions based on range or very low velocity
        if x > max_sim_range and vy < 0: # Past max sim range and falling
            break
        if v_rel_mag < 20: # Velocity too low (e.g., bullet has stopped moving effectively)
            break
        # If it passed the target and is significantly below ground relative to target_range_for_solver_m
        if x > target_range_for_solver_m * 1.5 and y < -20: # Far past target range and significantly below ground
            break


    # Final interpolation for height_at_target_range if not captured precisely or if hit ground
    if height_at_target_range is None: # If it was never set (e.g. simulation ended before reaching target_range_for_solver_m)
        if not trajectory_points: # No points generated at all
            height_at_target_range = scope_height_m # Default to starting height
            hit_ground_before_target_range = True # Assume it couldn't even start properly
        else:
            # If target range is outside the simulated trajectory points, extrapolate.
            last_x, last_y = trajectory_points[-1]
            
            if last_x < target_range_for_solver_m: # If bullet didn't reach target range
                if vx > 0: # Only if moving forward
                    # Use a rough extrapolation (linear + gravity effect over remaining distance)
                    time_to_target = (target_range_for_solver_m - last_x) / vx
                    height_at_target_range = last_y + vy * time_to_target - 0.5 * g * time_to_target**2
                else: # Not moving forward, effectively didn't reach
                    height_at_target_range = -1000.0 # Large finite negative number
                hit_ground_before_target_range = True # Bullet did not reach target while in air
            else: # Target range is within or past simulated trajectory, height_at_target_range should already be set.
                  height_at_target_range = last_y


    return x, z, trajectory_points, height_at_target_range, los_intersection_points, hit_ground_before_target_range


def _find_zero_barrel_angle(profile, air_density, scope_height_m, scope_zero_m):
    """
    Helper function to find the precise barrel angle required for the projectile
    to hit the ground (y=0) at scope_zero_m, assuming a flat range.
    This angle is then used to define the true LOS for zeroing.
    """
    if scope_zero_m <= 0.0: # Cannot zero at or before muzzle
        return 0.0 # Default to horizontal if invalid zero distance

    def zeroing_error(angle):
        g_const = 9.81
        launch_angle_rad = math.radians(angle)
        vx = profile['velocity_mps'] * math.cos(launch_angle_rad)
        vy = profile['velocity_mps'] * math.sin(launch_angle_rad)
        
        # Start at muzzle (0,0) for finding barrel angle to hit ground at zero distance
        # We want the bullet's *trajectory* from the muzzle to hit ground at scope_zero_m.
        x_curr, y_curr = 0.0, 0.0 
        
        time_step = 0.005
        
        # Simulate until past zero distance or hit ground
        while x_curr <= scope_zero_m + 1.0 and y_curr >= -1.0: # Sim slightly past zero or slightly below ground
            v_mag = math.sqrt(vx**2 + vy**2)
            if v_mag == 0: break
            
            drag_force_mag = 0.5 * air_density * (v_mag**2) * profile['area_m2'] * profile['drag_coefficient']
            
            force_x = -drag_force_mag * (vx / v_mag)
            force_y = -drag_force_mag * (vy / v_mag) - (profile['mass_kg'] * g_const)
            
            vx += (force_x / profile['mass_kg']) * time_step
            vy += (force_y / profile['mass_kg']) * time_step
            
            x_prev_step, y_prev_step = x_curr, y_curr # Store previous for interpolation
            x_curr += vx * time_step
            y_curr += vy * time_step
            
            # If crossed zero distance, interpolate y
            if x_curr >= scope_zero_m and x_prev_step < scope_zero_m:
                if vx != 0:
                    ratio = (scope_zero_m - x_prev_step) / (x_curr - x_prev_step)
                    interp_y = y_prev_step + (y_curr - y_prev_step) * ratio
                    return interp_y # Return height at scope_zero_m
                else:
                    return y_curr # Vertical trajectory at zero distance

            # If hit ground before reaching zero distance
            if y_curr < 0 and y_prev_step >= 0:
                return -1000.0 # Indicate it hit ground early

        # If it went past zero, and didn't hit ground (still high)
        if x_curr > scope_zero_m + 1.0 and y_curr > 0:
            return y_curr # Return positive height
        
        # Default if it just didn't get anywhere or stopped
        return -1000.0


    # Use Secant method to find the angle that results in y=0 at scope_zero_m
    angle_low, angle_high = 0.0, 5.0 # Typical starting guesses

    # Ensure initial bracket covers positive and negative errors
    error_low = zeroing_error(angle_low)
    error_high = zeroing_error(angle_high)
    
    # Simple bracket finding if initial fails
    if error_low * error_high >= 0:
        found_bracket = False
        test_angles = [-5.0, -2.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        for a1 in test_angles:
            e1 = zeroing_error(a1)
            for a2 in [angle for angle in test_angles if angle > a1]:
                e2 = zeroing_error(a2)
                if e1 * e2 < 0:
                    angle_low, error_low = a1, e1
                    angle_high, error_high = a2, e2
                    found_bracket = True
                    break
            if found_bracket: break
        
        if not found_bracket:
            # print("Warning: Could not robustly bracket zeroing angle. Using 0.0 as default.")
            return 0.0 # Fallback to 0 degrees if zero angle cannot be found

    for _ in range(50): # Max iterations for zeroing angle solver
        if abs(error_high - error_low) < 1e-9:
            break
        if error_high == error_low: break # Avoid division by zero
        
        angle_next = angle_high - error_high * (angle_high - angle_low) / (error_high - error_low)
        error_next = zeroing_error(angle_next)
        
        if abs(error_next) < 0.001: # 1mm tolerance for zero
            return angle_next
            
        angle_low, error_low = angle_high, error_high
        angle_high, error_high = angle_next, error_next
    
    return angle_high # Return best guess after iterations


def solve_for_angle(target_range_m, target_aim_height_m, profile, wind_speed_mps, wind_angle_deg, air_density, scope_height_m, scope_zero_m):
    """
    Finds the required launch angle to hit the target_aim_height at target_range_m using the secant method.
    """
    # First, calculate the barrel angle required to zero at scope_zero_m on a flat range
    # This is needed to correctly define the true LOS for the zeroing intersections
    barrel_angle_for_zero_m = _find_zero_barrel_angle(profile, air_density, scope_height_m, scope_zero_m)
    # print(f"DEBUG: Barrel angle for zeroing at {scope_zero_m:.0f}m is {barrel_angle_for_zero_m:.3f}°") # Debugging print

    # Define a function to find the error (difference from target aim height)
    def get_error_and_reach(angle):
        _, _, _, height_at_target_range, _, hit_ground_early = calculate_trajectory_with_points(angle, profile, wind_speed_mps, wind_angle_deg, air_density, scope_height_m, target_range_m, scope_zero_m, barrel_angle_for_zero_m)
        
        # If the bullet hit the ground before reaching the target range, return a large negative error
        # This biases the solver to find angles that keep the bullet in the air.
        if hit_ground_early:
            return -1000.0, True # Return a large finite negative value to guide the secant method
        
        return height_at_target_range - target_aim_height_m, False

    # Initialize solution_angle, final_deflection, trajectory_final, and los_intersections_final
    solution_angle = None
    final_deflection = None
    trajectory_final = []
    los_intersections_final = [] # Initialize as an empty list

    # Initial low and high guesses for the launch angle in degrees
    angle_low, angle_high = 0.0, 5.0 

    # Get initial errors and reach status
    error_low, hit_early_low = get_error_and_reach(angle_low)
    error_high, hit_early_high = get_error_and_reach(angle_high)

    # Robust initial bracketing for Secant Method
    search_angles = sorted(list(set([-10.0, -5.0, -2.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0])))
    
    found_bracket = False
    
    # Iterate through test angles to find a low_angle (error < 0 or hits early)
    for angle_val in search_angles:
        err, hit_early = get_error_and_reach(angle_val)
        if err < 0 or hit_early:
            angle_low = angle_val
            error_low = err
            hit_early_low = hit_early
        else: # Found an angle that's too high or just right, implies previous angle_low is now valid
            angle_high = angle_val
            error_high = err
            hit_early_high = hit_early
            found_bracket = True
            break
    
    # If a bracket wasn't found by stepping up, try stepping down if both were too high
    if not found_bracket and error_low > 0 and error_high > 0:
        for angle_val in sorted(test_angles, reverse=True): # Start high, go low
            if angle_val < min(angle_low, angle_high): # Only check lower angles
                err, hit_early = get_error_and_reach(angle_val)
                if err < 0 or hit_early:
                    angle_low = angle_val
                    error_low = err
                    hit_early_low = hit_early
                    found_bracket = True
                    break
    
    # Final check for a valid bracket (one error negative/hit_early, one positive/not hit_early)
    if not found_bracket or (error_low * error_high >= 0 and not (hit_early_low and error_high > 0)):
        print("Target is likely out of effective range (cannot find an angle that keeps the bullet in the air long enough, or it's always too high/low).")
        return None, None, [], [] # Return empty lists if no solution

    print(f"Initial bracket: Angle range [{angle_low:.2f}°, {angle_high:.2f}°] (Errors: {error_low:.2f}, {error_high:.2f})")


    # Iterate to refine the angle guess using Secant Method
    
    for i in range(50): # Increased iterations for Secant precision
        # Prevent division by zero or extremely small denominator
        if abs(error_high - error_low) < 1e-9:
            solution_angle = angle_high # Use the last high angle as the solution
            break

        # Secant method formula
        if error_high == error_low:
            solution_angle = angle_high # Cannot proceed with secant, use current best guess
            break
        
        angle_next = angle_high - error_high * (angle_high - angle_low) / (error_high - error_low)
        
        # Evaluate error at the new angle
        error_next, hit_early_next = get_error_and_reach(angle_next)

        # Check for convergence
        if abs(error_next) < 0.05: # Converged within 5 cm tolerance
            solution_angle = angle_next
            break
        
        # Update angles for next iteration
        angle_low_temp, error_low_temp = angle_high, error_high 
        angle_high, error_high = angle_next, error_next
        angle_low, error_low = angle_low_temp, error_low_temp
        
        # Prevent divergence to extreme angles.
        if not (-10 <= angle_next <= 45): 
            print(f"Solver diverged to an unrealistic angle: {angle_next:.2f}°. Stopping.")
            solution_angle = angle_next 
            break

    # If loop finished without perfect convergence, take the last calculated angle as the best effort
    if solution_angle is None:
        solution_angle = angle_high # Fallback to the last angle calculated
        print(f"Warning: Solver did not converge to exact tolerance within {0.05}m. Best angle found: {solution_angle:.2f}°")


    # Once solution_angle is found (or best effort), run trajectory one last time
    if solution_angle is not None:
         _, final_deflection, trajectory_final, height_at_target_final, los_intersections_final, hit_ground_early_final = \
             calculate_trajectory_with_points(solution_angle, profile, wind_speed_mps, wind_angle_deg, air_density, scope_height_m, target_range_m, scope_zero_m, barrel_angle_for_zero_m)
         
         # Final validation: Ensure the solution actually hits the target within tolerance and not ground early
         if hit_ground_early_final or abs(height_at_target_final - target_aim_height_m) > 0.1: 
             print(f"Validation failed: Solved angle {solution_angle:.2f}° resulted in a height of {height_at_target_final:.2f}m at {target_range_m:.0f}m, aiming for {target_aim_height_m:.2f}m. (Hit ground early: {hit_ground_early_final})")
             print("Target might be beyond bullet's effective range for this configuration.")
             return None, None, [], [] # Return empty lists if validation fails

         return solution_angle, final_deflection, trajectory_final, los_intersections_final

    return None, None, [], [] # Return None and empty lists if the solver fails to converge on a solution


def calculate_hit_probability(moa_accuracy, target_range_m, target_height_m, target_width_m):
    """
    Calculates the hit probability assuming a circular normal distribution
    of shots centered on the target, and a rectangular target.
    Uses scipy if available, otherwise falls back to math.erf.
    """
    # MOA spread in meters at target range:
    # 1 MOA at 100 yards (91.44m) is 1.047 inches (0.0265938 m)
    
    # Total MOA spread at range in meters
    moa_spread_at_range_m = (moa_accuracy * target_range_m / 91.44) * 0.0254
    
    # Standard deviation (sigma) of the shot group.
    # Assuming 'moa_accuracy' is the *diameter* of the group at range,
    # and this group represents roughly the 95% confidence interval (approx 4 standard deviations diameter).
    sigma_xy = moa_spread_at_range_m / 2.0 # Roughly half the stated MOA spread as std dev

    # Handle cases where sigma_xy might be zero (e.g., if moa_accuracy is 0)
    if sigma_xy == 0:
        return 1.0 # If accuracy is perfect, probability is 100%

    # Half-width and half-height of the target
    half_width = target_width_m / 2.0
    half_height = target_height_m / 2.0

    if SCIPY_AVAILABLE:
        # Use scipy's normal CDF for higher precision
        prob_x = st.norm.cdf(half_width, loc=0, scale=sigma_xy) - st.norm.cdf(-half_width, loc=0, scale=sigma_xy)
        prob_y = st.norm.cdf(half_height, loc=0, scale=sigma_xy) - st.norm.cdf(-half_height, loc=0, scale=sigma_xy)
    else:
        # Custom CDF implementation using math.erf
        def erf_cdf(x, mu, sigma):
            # For sigma = 0, erf will raise an error. Handle explicitly.
            if sigma == 0: return 1.0 if x >= mu else 0.0
            return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

        prob_x = erf_cdf(half_width, 0, sigma_xy) - erf_cdf(-half_width, 0, sigma_xy)
        prob_y = erf_cdf(half_height, 0, sigma_xy) - erf_cdf(-half_height, 0, sigma_xy)
    
    # The total probability of hitting the rectangle is the product of the two
    total_probability = prob_x * prob_y
    
    return total_probability


# ==============================================================================
#
#                            PLOTTING FUNCTIONS
#
#  These functions visualize the trajectory, line of sight, and target.
#
# ==============================================================================

def plot_trajectory(trajectory_points, barrel_angle_deg, scope_height_m, target_range_m, target_angle_deg, target_height_m, scope_zero_m, los_intersection_points, system_units, barrel_angle_for_zero_m):
    """
    Plots the bullet trajectory, line of sight, barrel angle, and target.
    Accounts for target elevation angle and marks LOS intersections, including zero.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not installed. Cannot generate plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Convert coordinates if Imperial units are selected for display
    unit_factor = 1.0 # default to meters
    distance_unit_label = "meters"
    if system_units == 'IMPERIAL':
        unit_factor = 1.0 / YD_TO_M # meters to yards
        distance_unit_label = "yards"

    traj_x = [p[0] * unit_factor for p in trajectory_points]
    traj_y = [p[1] * unit_factor for p in trajectory_points]
    scope_height_plot = scope_height_m * unit_factor
    scope_zero_plot = scope_zero_m * unit_factor

    # Calculate target's vertical position based on target_angle_deg relative to shooter's ground
    target_elevation_at_range_plot = math.tan(math.radians(target_angle_deg)) * target_range_m * unit_factor

    # --- Plot the bullet trajectory ---
    ax.plot(traj_x, traj_y, color='blue', linewidth=2, label='Bullet Trajectory')

    # --- Plot the ground terrain ---
    # Shooter's ground is at y=0. Target's ground is at target_elevation_at_range_plot.
    max_plot_x = max(target_range_m * 1.1 * unit_factor, max(traj_x) * 1.1 if traj_x else 0)
    ax.plot([0, max_plot_x], [0, 0], color='brown', linestyle='-', linewidth=2, label='Shooter\'s Ground Level')
    # Draw sloping terrain to target elevation
    ax.plot([0, target_range_m * unit_factor], [0, target_elevation_at_range_plot], color='darkgreen', linestyle='-', linewidth=1.5, label='Terrain (Slope to Target)')
    # Extend target's ground level
    ax.plot([target_range_m * unit_factor, max_plot_x], [target_elevation_at_range_plot, target_elevation_at_range_plot], color='sienna', linestyle='-', linewidth=2, label='Target\'s Ground Level')


    # --- Plot the Line of Sight (LOS) to Target ---
    # LOS from scope (0, scope_height_plot) to actual target midpoint
    los_target_x = target_range_m * unit_factor
    los_target_y = target_elevation_at_range_plot + TARGET_MIDPOINT_M * unit_factor
    
    # Calculate angle for LOS to target
    los_angle_to_target_rad = math.atan2( (los_target_y - scope_height_plot), los_target_x)
    
    # Extend LOS across the plot
    los_end_x = max_plot_x
    los_end_y = scope_height_plot + math.tan(los_angle_to_target_rad) * los_end_x
    ax.plot([0, los_end_x], [scope_height_plot, los_end_y], color='green', linestyle='--', label='Line of Sight (to target midpoint)')

    # --- Plot the Barrel Line ---
    # The barrel line starts at (0, 0) and extends with the barrel_angle_deg.
    barrel_end_x = max_plot_x
    barrel_end_y = math.tan(math.radians(barrel_angle_deg)) * barrel_end_x
    ax.plot([0, barrel_end_x], [0, barrel_end_y], color='red', linestyle=':', label='Barrel Line (from bore)')

    # --- Plot the Target ---
    # Target is a rectangle at target_range_m, from target_elevation_plot to target_elevation_plot + target_height_m.
    target_bottom_y_plot = target_elevation_at_range_plot
    target_top_y_plot = target_elevation_at_range_plot + target_height_m * unit_factor
    ax.plot([target_range_m * unit_factor, target_range_m * unit_factor], [target_bottom_y_plot, target_top_y_plot],
            color='purple', linewidth=3, label=f'Target (Adult Male at {target_range_m * unit_factor:.0f} {distance_unit_label})')
    
    # Add a dot at the precise target midpoint for aiming reference
    ax.plot(target_range_m * unit_factor, target_elevation_at_range_plot + TARGET_MIDPOINT_M * unit_factor, 'o', color='black', markersize=5, label='Target Midpoint')

    # --- Plot the Zeroing Line of Sight and its Intersections ---
    # This LOS goes from (0, scope_height_plot) to (scope_zero_plot, 0)
    # This is the line that the bullet intersects at its zero distance.
    if scope_zero_plot > 0:
        zeroing_los_slope = (0 - scope_height_plot) / scope_zero_plot
    else: # For 0m zero, treat as horizontal from scope height
        zeroing_los_slope = 0.0

    zeroing_los_x_end = max_plot_x
    zeroing_los_y_end = scope_height_plot + zeroing_los_slope * zeroing_los_x_end
    
    ax.plot([0, zeroing_los_x_end], [scope_height_plot, zeroing_los_y_end], color='grey', linestyle='-.', label='Zeroing Line of Sight')


    # Mark intersections from calculate_trajectory_with_points
    los_intersection_points_plot = [(p[0] * unit_factor, p[1] * unit_factor) for p in los_intersection_points]
    los_intersection_points_plot.sort(key=lambda p: p[0])

    marked_primary_zero = False # To track if the primary zero has been marked with the special icon
    for i_x, i_y in los_intersection_points_plot:
        # Check if this point is close to the defined scope_zero_m
        # Using a slightly more robust check for primary zero, and ensure it's not the initial 0m mark if scope_zero_m is not 0
        is_primary_zero = (abs(i_x - scope_zero_plot) < (scope_zero_plot * 0.02 + 0.1) and scope_zero_plot != 0) or (scope_zero_plot == 0 and i_x < 5 * unit_factor and i_x > -0.1) # Within 2% or 0.1m, or near 0 for 0m zero
        
        label_str = ''
        marker_style = 'x' # Default for general LOS crossings
        marker_color = 'darkgreen'
        marker_size = 10
        marker_edge = 2

        if is_primary_zero and not marked_primary_zero:
            label_str = f'Primary Zero ({i_x:.0f}{distance_unit_label})'
            marker_style = 'X' # Capital X for primary zero
            marker_color = 'darkorange'
            marker_size = 12
            marker_edge = 2
            marked_primary_zero = True
        elif i_x > 0.01: # Mark other positive crossings as 'LOS Crossing'
            label_str = f'LOS Crossing ({i_x:.0f}{distance_unit_label})'
        elif i_x <= 0.01 and i_x > -1.0: # Mark initial close-to-muzzle intersection
            label_str = f'Initial LOS Crossing ({i_x:.0f}{distance_unit_label})'

        if label_str: # Only add if we have a label
            ax.plot(i_x, i_y, marker=marker_style, color=marker_color, markersize=marker_size, markeredgewidth=marker_edge)
            # Dynamic offset based on plot height
            offset_y = (max(traj_y) - min(traj_y) if traj_y else 1) * 0.05 
            ax.text(i_x, i_y + offset_y, label_str.split('(')[0].strip(),
                    verticalalignment='bottom', horizontalalignment='center', color=marker_color, fontsize=9)


    # --- Labels and Title ---
    ax.set_xlabel(f"Downrange Distance ({distance_unit_label})")
    ax.set_ylabel(f"Height Above Shooter's Ground ({distance_unit_label})")
    ax.set_title("Bullet Trajectory Simulation")
    
    # Consolidate duplicate labels for the legend
    legend_handles = []
    legend_labels = []

    # Add handles/labels for lines (from ax.plot)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels_lines = []
    unique_handles_lines = []
    for h, l in zip(handles, labels):
        if l not in unique_labels_lines:
            unique_labels_lines.append(l)
            unique_handles_lines.append(h)
    
    # Manually add proxy artists for the 'X' markers to ensure they appear in the legend
    proxy_primary_zero = plt.Line2D([0], [0], linestyle='None', marker='X', markersize=12, markeredgewidth=2, color='darkorange')
    proxy_los_crossing = plt.Line2D([0], [0], linestyle='None', marker='x', markersize=10, markeredgewidth=2, color='darkgreen')
    
    # Add unique marker labels/handles
    legend_handles.extend(unique_handles_lines)
    legend_labels.extend(unique_labels_lines)

    if not marked_primary_zero: # Only add this if no primary zero was actually marked (e.g. if scope_zero_m was 0)
        # This condition helps ensure "Primary Zero" only appears if an orange X was actually used.
        pass # The logic below will handle the LOS Crossing entry.
    else:
        legend_handles.append(proxy_primary_zero)
        legend_labels.append('Primary Zero')
    
    # Always add a general LOS crossing marker legend entry if there are any crossings at all
    if any(p[0] > 0.01 for p in los_intersection_points_plot): # If there was at least one crossing detected
        legend_handles.append(proxy_los_crossing)
        legend_labels.append('LOS Crossing')

    ax.legend(legend_handles, legend_labels)
    ax.grid(True)
    
    # --- Adjust Aspect Ratio ---
    min_y_val = min(0, min(traj_y) if traj_y else 0, target_elevation_at_range_plot * 1.2 if target_elevation_at_range_plot < 0 else 0, scope_height_plot * 0.5) * 1.1
    max_y_val = max(max(traj_y) if traj_y else 0, scope_height_plot * 1.5, (target_elevation_at_range_plot + target_height_m * unit_factor) * 1.1, target_elevation_at_range_plot * 1.2 if target_elevation_at_range_plot > 0 else 0) * 1.1

    if min_y_val >= max_y_val:
        min_y_val = min(0, target_elevation_at_range_plot) - 1 * unit_factor # Give some buffer
        max_y_val = max(scope_height_plot, target_elevation_at_range_plot + target_height_m * unit_factor) + 1 * unit_factor # Give some buffer
        if min_y_val >= max_y_val:
            min_y_val, max_y_val = -1 * unit_factor, 1 * unit_factor


    ax.set_xlim(0, max_plot_x)
    ax.set_ylim(min_y_val, max_y_val)

    # Set y-axis scale at 3:1 ratio to the x-axis
    # This means one unit on the Y-axis is visually three times longer as one unit on the X-axis.
    # In Matplotlib, aspect = Y_unit_length / X_unit_length = 3 / 1 = 3.
    ax.set_aspect(3, adjustable='box') 
    plt.show()


# ==============================================================================
#
#                         MAIN PROGRAM EXECUTION
#
# ==============================================================================

def setup_system_units():
    """Prompts user to select system units (Imperial/Metric)."""
    print("\n--- System Unit Setup ---")
    unit_choices = {"1": "IMPERIAL", "2": "METRIC"}
    print("Select System Units:")
    for key, value in unit_choices.items(): print(f"  {key}: {value}")
    choice = get_menu_choice("Unit (default: 1 for Imperial): ", unit_choices, default="1")
    return unit_choices[choice]

def setup_profile(saved_profiles, system_units):
    """
    Handles the one-time setup for the projectile profile and scope units.
    """
    print("--- Ballistic Profile Setup ---")
    
    available_choices = {}
    choice_idx = 1
    
    # Add saved profiles first
    print("\n--- Saved Profiles ---")
    if saved_profiles:
        for name in saved_profiles:
            available_choices[str(choice_idx)] = {"name": f"(Saved) {name}", "data": saved_profiles[name]}
            print(f"  {choice_idx}: (Saved) {name}")
            choice_idx += 1
    else:
        print("  No custom profiles saved yet.")

    # Add caliber families (top level)
    print("\n--- Pre-loaded Caliber Families ---")
    caliber_family_choices = {}
    for key, data in CALIBER_DATA.items():
        caliber_family_choices[key] = data["name"]
        print(f"  {key}: {data['name']}")
    
    # Get user's caliber family choice
    top_level_choice_key = get_menu_choice("Select Caliber Family or Manual Input: ", caliber_family_choices)
    
    profile_data = {}
    if top_level_choice_key == "6": # Manual Input (using the key defined in CALIBER_DATA)
        print("\n--- Manual Projectile Data ---")
        mass_gr = get_float_input("Mass (grains): ")
        vel_fps = get_float_input("Muzzle Velocity (fps): ")
        diam_in = get_float_input("Diameter (in): ")
        bc_g1 = get_float_input("G1 Ballistic Coefficient: ")
        moa_acc = get_float_input("Weapon System Accuracy (MOA): ", 1.0)
        
        profile_data.update({
            'mass_kg': mass_gr * GR_TO_KG,
            'velocity_mps': vel_fps * 0.3048, # Corrected: fps to mps
            'diameter_m': diam_in * IN_TO_M,
            'g1_bc_lb_in2': bc_g1,
            'moa_accuracy': moa_acc,
            'name': "Manual Entry"
        })
        
        # BC Estimator placeholder
        # print("\nBC Estimator: (Feature not yet implemented - input BC manually)")
        
        save_prompt = input("Do you want to save this custom profile? (y/N): ").lower()
        if save_prompt == 'y':
            profile_name = input("Enter a name for this profile: ")
            if profile_name:
                profile_data['name'] = profile_name # Update name for saving
                saved_profiles[profile_name] = profile_data # Add to dict
                save_profiles(saved_profiles)
                print(f"Profile '{profile_name}' saved.")
            else:
                print("Profile name cannot be empty. Profile not saved.")
    else: # User selected a pre-loaded caliber family or a saved profile
        if top_level_choice_key in CALIBER_DATA: # It's a pre-loaded family
            selected_family = CALIBER_DATA[top_level_choice_key]
            if selected_family['loads']: # If there are specific loads
                print(f"\n--- {selected_family['name']} Loads ---")
                load_choices = {}
                # Create load choices with numeric keys for display
                load_idx = 1
                for key_inner, load_data in selected_family['loads'].items():
                    load_choices[str(load_idx)] = load_data
                    print(f"  {load_idx}: {load_data['name']}")
                    load_idx += 1
                
                load_choice_key_str = get_menu_choice("Select specific load: ", load_choices)
                profile_data = load_choices[load_choice_key_str].copy()
            else: # Should not happen with current structure, but as a fallback
                print("No specific loads found for this caliber. Please select manual input.")
                return setup_profile(saved_profiles, system_units) # Re-run setup
        else: # It's a loaded custom profile
            profile_name_from_choice = available_choices[top_level_choice_key]["name"].replace("(Saved) ", "")
            profile_data = saved_profiles[profile_name_from_choice].copy()


    # Now ask for scope specific data (scope height and zero) for ANY selected profile
    print("\n--- Scope Settings ---")
    scope_h_in_default = profile_data.get('scope_height_m', 0.0381) / IN_TO_M # Default 1.5 inches
    scope_h_in = get_float_input("Scope Height Above Bore (inches): ", scope_h_in_default) * IN_TO_M
    
    scope_z_m_default = profile_data.get('scope_zero_m', 91.44) # Default 100m
    if system_units == 'IMPERIAL':
        scope_z_input_unit = "yards"
        scope_z_val = get_float_input(f"Scope Zero Distance ({scope_z_input_unit}): ", scope_z_m_default / YD_TO_M)
        scope_z_m = scope_z_val * YD_TO_M
    else:
        scope_z_input_unit = "meters"
        scope_z_val = get_float_input(f"Scope Zero Distance ({scope_z_input_unit}): ", scope_z_m_default)
        scope_z_m = scope_z_val

    profile_data['scope_height_m'] = scope_h_in
    profile_data['scope_zero_m'] = scope_z_m

    # Let the user choose their scope's adjustment unit (MIL is the default)
    unit_choices = {"1": "MIL", "2": "MOA"}
    print("\nSelect Adjustment Unit:")
    for key, value in unit_choices.items(): print(f"  {key}: {value}")
    scope_unit = unit_choices[get_menu_choice("Unit (default: 1 for MIL): ", unit_choices, default="1")]
    
    return calculate_derived_properties(profile_data), scope_unit

def setup_environment(system_units):
    """
    Handles the one-time setup for environmental conditions.
    """
    print("\n--- Environmental Setup (press Enter for defaults) ---")
    
    elevation_unit = "feet" if system_units == 'IMPERIAL' else "meters"
    elevation_val = get_float_input(f"Elevation {elevation_unit} (shooter's elevation, default: 0): ", 0)
    elevation_m = elevation_val * FT_TO_M if system_units == 'IMPERIAL' else elevation_val

    temp_f = get_float_input("Temperature °F (default: 59): ", 59)
    humidity = get_float_input("Humidity % (default: 50): ", 50)
    
    temp_c = (temp_f - 32) * 5/9
    
    return calculate_air_density(elevation_m, temp_c, humidity)

def main():
    """
    The main function that runs the user interface and calculation loop.
    """
    # Load saved profiles at start
    saved_profiles = load_profiles()

    # Setup global system units
    system_units = setup_system_units()
    distance_unit_display = "yards" if system_units == 'IMPERIAL' else "meters"

    # Perform initial setup for the rifle profile and environment
    derived_profile, scope_unit = setup_profile(saved_profiles, system_units)
    air_density = setup_environment(system_units)
    
    # Display the current settings to the user
    print("\n--- Ready for Firing Solution ---")
    print(f"Profile: {derived_profile['name']} | Scope Unit: {scope_unit} | Air Density: {air_density:.4f} kg/m^3 | Accuracy: {derived_profile['moa_accuracy']} MOA | Scope Height: {derived_profile['scope_height_m']/IN_TO_M:.1f} in | Scope Zero: {derived_profile['scope_zero_m']/YD_TO_M:.0f} yd")
    print("\nCommands: 'profile', 'env', 'manage', 'quit'")
    print(f"Input Format: [Range {distance_unit_display}] [Wind mph] [Wind deg] [Target Angle deg] [optional g/p]")


    # --- Main Calculation Loop ---
    while True:
        try:
            prompt_str = f"Enter [Range {distance_unit_display}] [Wind mph] [Wind deg] [Target Angle deg] [optional g/p]: "
            user_input = input(prompt_str).strip()

            # --- Handle In-App Commands ---
            if user_input.lower() == 'quit':
                sys.exit("Exiting Ballistic Solver.")
            elif user_input.lower() == 'profile':
                derived_profile, scope_unit = setup_profile(saved_profiles, system_units)
                print(f"\nProfile changed to: {derived_profile['name']} | Scope Unit: {scope_unit} | Accuracy: {derived_profile['moa_accuracy']} MOA | Scope Height: {derived_profile['scope_height_m']/IN_TO_M:.1f} in | Scope Zero: {derived_profile['scope_zero_m']/YD_TO_M:.0f} yd")
                continue # Restart loop
            elif user_input.lower() == 'env':
                air_density = setup_environment(system_units)
                print(f"\nEnvironment updated. Air Density: {air_density:.4f} kg/m^3")
                continue # Restart loop
            elif user_input.lower() == 'manage':
                saved_profiles = manage_profiles(saved_profiles) # Update saved_profiles after management
                continue # Restart loop

            # --- Parse User Input for a Firing Solution ---
            parts = user_input.split()
            
            if len(parts) < 4: # Need at least Range, Wind Speed, Wind Angle, Target Angle
                print("Invalid input. Please provide at least four values (range, wind speed, wind direction, target angle).")
                continue

            # Check for plotting flags ('g' or 'p')
            plot_console_requested = False
            plot_matplotlib_requested = False
            if len(parts) > 4:
                plot_flag = parts[-1].lower()
                if plot_flag == 'g':
                    plot_console_requested = True
                    parts = parts[:-1]
                elif plot_flag == 'p':
                    plot_matplotlib_requested = True
                    parts = parts[:-1]

            if len(parts) != 4: # After removing plot flag, ensure 4 numeric inputs remain
                 print("Invalid input format after parsing plot flag. Please provide four numeric values.")
                 continue

            # Convert parsed strings to numbers
            try:
                raw_range, raw_wind_speed, raw_wind_angle, raw_target_angle = map(float, parts)
            except ValueError:
                print("Invalid numeric input. Please ensure range, wind speed/angle, and target angle are numbers.")
                continue

            # Convert inputs to metric for the simulation based on system units
            if system_units == 'IMPERIAL':
                target_range_m = raw_range * YD_TO_M
            else:
                target_range_m = raw_range
            
            wind_speed_mps = raw_wind_speed * MPH_TO_MPS
            wind_angle_deg = raw_wind_angle
            target_angle_deg = raw_target_angle

            # Calculate the target's vertical position relative to shooter's ground
            # This is the height the bullet needs to hit at the target range
            # Note: The target_elevation_m from setup_environment is the shooter's actual elevation,
            # which affects air density. target_angle_deg defines the target's *relative* elevation.
            target_aim_height_m = math.tan(math.radians(target_angle_deg)) * target_range_m + TARGET_MIDPOINT_M


            # --- Call the Solver ---
            solution_angle, solution_deflection, trajectory_points, los_intersection_points = solve_for_angle(
                target_range_m, target_aim_height_m, derived_profile, wind_speed_mps, wind_angle_deg, air_density, derived_profile['scope_height_m'], derived_profile['scope_zero_m']
            )
            
            # --- Format and Display the Output ---
            if solution_angle is not None:
                # Convert the elevation angle (holdover) to the user's chosen unit (MOA or MIL)
                hold_adjustment = solution_angle * DEG_TO_MIL if scope_unit == "MIL" else solution_angle * 60
                
                # Convert the linear wind deflection (in meters) to an angular value for scope adjustment
                deflection_angle_rad = math.atan(solution_deflection / target_range_m)
                deflection_angle_deg = math.degrees(deflection_angle_rad) # Angle from LOS to bullet impact due to wind
                wind_adjustment = deflection_angle_deg * DEG_TO_MIL if scope_unit == "MIL" else deflection_angle_deg * 60
                
                # Also provide the linear deflection in inches for reference
                deflection_in = solution_deflection / IN_TO_M
                
                # Calculate hit probability
                hit_prob = calculate_hit_probability(
                    derived_profile['moa_accuracy'],
                    target_range_m,
                    TARGET_HEIGHT_M, # Always adult man target height
                    TARGET_WIDTH_M   # Always adult man target width
                )
                
                # Print the final, formatted firing solution
                print(f"  HOLD: {hold_adjustment:.1f} {scope_unit} UP | "
                      f"WIND: {abs(wind_adjustment):.1f} {scope_unit} {'Right' if solution_deflection > 0 else 'Left'} ({abs(deflection_in):.1f} in) | "
                      f"HIT PROB: {hit_prob:.1%}")

                # Plot if requested
                if plot_matplotlib_requested:
                    if MATPLOTLIB_AVAILABLE:
                        # Pass the barrel_angle_for_zero_m to the plot function.
                        # It's already calculated inside solve_for_angle and used to get los_intersection_points.
                        # Recalculating here is fine as it's a fixed value for the profile/env.
                        barrel_angle_for_zero = _find_zero_barrel_angle(derived_profile, air_density, derived_profile['scope_height_m'], derived_profile['scope_zero_m'])
                        plot_trajectory(trajectory_points, solution_angle, derived_profile['scope_height_m'], target_range_m, target_angle_deg, TARGET_HEIGHT_M, derived_profile['scope_zero_m'], los_intersection_points, system_units, barrel_angle_for_zero)
                    else:
                        print("Plotting requested, but Matplotlib is not installed.")
                elif plot_console_requested:
                    plot_console_trajectory(trajectory_points, derived_profile['scope_height_m'], target_range_m, target_angle_deg, TARGET_HEIGHT_M, system_units)

            else:
                # If the solver returns None, the target is unreachable
                print("  SOLUTION NOT FOUND. Target may be out of range or parameters too extreme.")

        except (ValueError, IndexError) as e:
            print(f"Input Error: {e}. Please use: <range> <wind_speed> <wind_angle> <target_angle> [optional g/p]")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()
