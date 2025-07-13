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
    # The slope of this line is (0 - scope_height_m) / (scope_zero_m - 0)
    # So, LOS height at any x is: scope_height_m + (x * (0 - scope_height_m) / scope_zero_m)
    # This assumes a flat range for zeroing.
    def get_zeroing_los_height_at_x(current_x):
        if scope_zero_m == 0: # If zero is at muzzle, LOS is horizontal at scope height
            return scope_height_m
        # LOS is a line from (0, scope_height_m) to (scope_zero_m, 0)
        # Equation: y - y1 = m(x - x1)
        # m = (0 - scope_height_m) / (scope_zero_m - 0) = -scope_height_m / scope_zero_m
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
        # and if the bullet hasn't hit ground already, and if the LOS itself is valid (not dividing by zero if scope_zero_m is 0)
        if x > 0.01 and current_bullet_above_los != prev_bullet_above_los: # If status changed
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
    to hit the ground (y=0) at scope_zero_m,
