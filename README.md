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

- **Physics-Based Simulation:** The core of the calculator is a numerical integration engine that accounts for:
  - **Gravity:** Applied correctly as a constant downward vector.
  - **Atmospheric Drag:** Calculated using projectile-specific G1 ballistic coefficients.
  - **Wind Deflection:** Accounts for wind speed and angle relative to the shooter.
- **Uphill/Downhill Shooting:** Accurately calculates solutions for inclined or declined targets by factoring in the angle to the target. This is not a simple "Rifleman's Rule" approximation but a full physics calculation.
- **Extensive Caliber Library:** Comes pre-loaded with a wide variety of common calibers and popular bullet loads.
- **Full Profile Customization:**
  - **Manual Entry:** Create a profile for any projectile by entering its physical characteristics.
  - **BC Estimator:** If the ballistic coefficient is unknown, an optional calculator can provide a solid estimate based on the bullet's weight, diameter, and shape (form factor).
  - **Profile Management:** Save, load, and delete your custom rifle profiles. Your setups are stored locally in a `rifle_profiles.json` file and are available every time you run the script.
- **Environmental Adjustments:** The simulation accounts for changes in air density based on:
  - Geographic Elevation (in feet or meters).
  - Temperature.
  - Humidity.
- **Flexible Unit Support:**
  - **System Units:** Choose between **Imperial (yards)** or **Metric (meters)** for all distance inputs and outputs.
  - **Scope Adjustments:** Get your final holdover and windage in either **MILs** or **MOA**.
- **Trajectory Visualization:**
  - **Console Graph:** Instantly generate a simple text-based graph of the trajectory in your console.
  - **Matplotlib Plot:** For more detail, generate a high-quality plot in a separate window, showing the bullet's path, line of sight, and barrel angle relative to the horizon.

## How It Works

The program uses a numerical method to simulate the bullet's flight step-by-step.

1.  **Angle Solving:** When you input a target range, the script uses a **bisection method root-finding algorithm**. It makes an initial guess for the barrel's launch angle and runs a full trajectory simulation. It checks if the bullet's final position is above or below the target. Based on this "error," it adjusts its guess and re-runs the simulation, rapidly converging on the precise angle where the bullet's path will intersect the target's coordinates.
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

3.  **Download:** Download the `ballistic_solver.py` script to a directory on your computer.

## Usage

### Quick Start

1.  Open a terminal or command prompt.
2.  Navigate to the directory where you saved the file.
3.  Run the script with `python ballistic_solver.py`.
4.  Follow the on-screen prompts to set up your unit system, rifle profile, and environment.
5.  At the main prompt, enter your target data to get a firing solution.

### Command-Line Arguments

The main input prompt expects data in the following format:

`[Range] [Wind Speed] [Wind Angle] [Target Angle] (optional: g/p)`

| Parameter      | Description                                                                                             | Example      |
| :------------- | :------------------------------------------------------------------------------------------------------ | :----------- |
| `[Range]`      | The distance to the target (in yards or meters, based on your system setting).                          | `850`        |
| `[Wind Speed]` | The wind speed in miles per hour (mph).                                                                 | `10`         |
| `[Wind Angle]` | The wind direction in degrees (0=headwind, 90=right crosswind, 180=tailwind, 270=left crosswind).       | `270`        |
| `[Target Angle]`| The uphill/downhill angle to the target in degrees (positive for uphill, negative for downhill).          | `-5`         |
| `(g/p)`        | **Optional.** A flag to generate a visual representation of the trajectory.                               | `p`          |
|                | `g`: Generates a simple text-based **g**raph in the console.                                            |              |
|                | `p`: Generates a detailed Matplotlib **p**lot in a new window.                                           |              |

**Full Example:**
Enter [Range yd] [Wind mph] [Wind deg] [Target Angle deg] (optional: g/p): 850 10 270 -5 pThis command calculates a solution for a target at 850 yards, with a 10 mph wind coming from the left, at a 5-degree downhill angle, and generates a detailed plot.

### In-App Commands

At the main prompt, you can also enter the following commands instead of firing data:

-   `profile`: Restarts the rifle profile setup process from the beginning.
-   `env`: Restarts the environmental setup process.
-   `manage`: Opens the profile manager to view or delete saved profiles.
-   `quit`: Exits the program.

## Profile Management

Your custom rifle profiles are the heart of this tool. A profile consists of your bullet's data, your scope's adjustment unit (MIL/MOA), and your rifle's zero distance.

-   **Saving:** When you create a new profile using the "Manual Input" option, you will be prompted to save it. Give it a descriptive name (e.g., "My Tikka 6.5CM"), and it will be available the next time you run the script.
-   **Loading:** Saved profiles appear at the top of the selection list when you start the script or use the `profile` command.
-   **Deleting:** Use the `manage` command to see a list of your saved profiles and choose one to delete.
-   **File Location:** All profiles are stored in a human-readable `rifle_profiles.json` file in the same directory as the script. You can manually edit or back up this file.

## Contributing

Contributions are welcome! If you have suggestions for new features, improvements to the physics model, or bug fixes, please feel free to open an issue or submit a pull request.

Areas for future improvement include:
-   Support for G7 ballistic coefficients.
-   Calculation of spin drift (Coriolis effect).
-   Generation of full drop charts/dope cards.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
