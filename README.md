# Ballistic Python

A command-line ballistic calculator in Python. Get precise firing solutions for long-range shooting by simulating trajectory with environmental factors, custom rifle profiles, and trajectory visualization.

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Command-Line Arguments](#command-line-arguments)
  - [In-App Commands](#in-app-commands)
- [Profile Management](#profile-management)
- [Contributing](#contributing)
- [License](#license)

## Features

This program is designed to be both powerful and flexible, incorporating features essential for accurate long-range calculations:

-   **Physics-Based Simulation:** The core of the calculator is a numerical integration engine that accounts for:
    -   **Gravity:** Applied correctly as a constant downward vector.
    -   **Atmospheric Drag:** Calculated using projectile-specific G1 ballistic coefficients.
    -   **Wind Deflection:** Accounts for wind speed and angle relative to the shooter.
-   **Uphill/Downhill Shooting:** Accurately calculates solutions for inclined or declined targets by factoring in the angle to the target. This is not a simple "Rifleman's Rule" approximation but a full physics calculation.
-   **Hit Probability Estimation:** Estimates the probability of hitting a standard adult human target based on the weapon system's inherent accuracy (MOA), disregarding shooter skill.
-   **Extensive Caliber Library:** Comes pre-loaded with a wide variety of common calibers and popular bullet loads, including various 5.56 NATO / .223 Remington bullet weights (55gr, 62gr, 69gr, 77gr).
-   **Full Profile Customization:**
    -   **Manual Entry:** Create a profile for any projectile by entering its physical characteristics.
    -   **Scope Settings:** Define your scope's height above the bore and its zero distance.
    -   **Profile Management:** Save, load, and delete your custom rifle profiles. Your setups are stored locally in a `rifle_profiles.json` file and are available every time you run the script.
-   **Environmental Adjustments:** The simulation accounts for changes in air density based on:
    -   Geographic Elevation (in feet or meters).
    -   Temperature.
    -   Humidity.
-   **Flexible Unit Support:**
    -   **System Units:** Choose between **Imperial (yards, feet, mph, inches)** or **Metric (meters, mps, cm)** for all distance, speed, and size inputs and outputs.
    -   **Scope Adjustments:** Get your final holdover and windage in either **MILs** or **MOA**.
-   **Trajectory Visualization:**
    -   **Console Graph (`g` flag):** Instantly generate a simple text-based graph of the trajectory in your terminal.
    -   **Matplotlib Plot (`p` flag):** For more detail, generate a high-quality plot in a separate window, showing the bullet's path, line of sight, barrel angle, ground terrain (including slope to target), and marked zero intersection points. This feature is optional and requires Matplotlib installation.

## How It Works

The program uses a numerical method to simulate the bullet's flight step-by-step.

1.  **Angle Solving:** When you input a target range and angle, the script uses a **Secant Method root-finding algorithm**. It makes initial guesses for the barrel's launch angle and runs full trajectory simulations. It checks if the bullet's height at the target range is above or below the desired target height (midpoint). Based on this "error," it refines its guess, rapidly converging on the precise angle where the bullet's path will intersect the target's coordinates. It also ensures the bullet remains in the air until the target range is reached.
2.  **Trajectory Simulation:** Each simulation uses **Euler integration** over small time steps (0.005s). In each step, it calculates the net forces on the projectile:
    * **Gravity:** A constant downward acceleration.
    * **Drag:** A force opposing the direction of motion, calculated using the standard drag equation `Fd = 0.5 * ρ * v² * A * Cd`, where the drag coefficient `Cd` is derived from the bullet's G1 Ballistic Coefficient.
    * **Wind:** A force applied perpendicular (crosswind) or parallel (headwind/tailwind) to the initial direction of fire.
3.  **Final Solution:** Once the correct launch angle is found, the script calculates the difference between this angle and the angle required for your rifle's zero. This difference is your final scope adjustment (your "dial").

## Installation

1.  **Python:** Ensure you have Python 3.6 or newer installed on your system.

2.  **Matplotlib (Optional):** This library is required *only* if you want to use the detailed plotting feature (`p` flag). If you don't need it, the script will still run perfectly without it.
    ```bash
    pip install matplotlib
    ```

3.  **SciPy (Optional):** This library is recommended for more precise hit probability calculations. If not installed, a less precise approximation will be used.
    ```bash
    pip install scipy
    ```

4.  **Download:** Download the `ballistic_solver.py` script to a directory on your computer.

## Usage

### Quick Start

1.  Open a terminal or command prompt.
2.  Navigate to the directory where you saved the file.
3.  Run the script with `python ballistic_solver.py`.
4.  Follow the on-screen prompts to set up your unit system, rifle profile, and environment.
5.  At the main prompt, enter your target data to get a firing solution.

### Command-Line Arguments (In-App Input)

The main input prompt expects data in the following format:

`[Range] [Wind Speed] [Wind Angle] [Target Angle] (optional: g/p)`

| Parameter        | Description                                                                                             | Example      |
| :--------------- | :------------------------------------------------------------------------------------------------------ | :----------- |
| `[Range]`        | The distance to the target (in your chosen system units, e.g., yards or meters).                        | `850`        |
| `[Wind Speed]`   | The wind speed in miles per hour (mph).                                                                 | `10`         |
| `[Wind Angle]`   | The wind direction in degrees (0=headwind, 90=right crosswind, 180=tailwind, 270=left crosswind).       | `270`        |
| `[Target Angle]` | The uphill/downhill angle to the target in degrees (positive for uphill, negative for downhill).        | `-5`         |
| `(g/p)`          | **Optional.** A flag to generate a visual representation of the trajectory.                               | `p`          |
|                  | `g`: Generates a simple text-based **g**raph in the console.                                            |              |
|                  | `p`: Generates a detailed Matplotlib **p**lot in a new window.                                           |              |

**Full Example:**
```bash
850 10 270 -5 p
