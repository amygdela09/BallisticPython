# Ballistic Python

A command-line ballistic calculator in Python. Get precise firing solutions for long-range shooting by simulating trajectory with environmental factors, custom rifle profiles, and trajectory visualization.

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

## Contributing

Contributions are welcome! If you have suggestions for new features, improvements to the physics model, or bug fixes, please feel free to open an issue or submit a pull request.

Areas for future improvement include:
-   **G7 Ballistic Coefficient Support:** Implement drag models and calculations for G7 BCs, which are more accurate for modern, long-range projectiles.
-   **Spin Drift Calculation:** Incorporate the effect of bullet spin, which causes a small horizontal deflection over long distances.
-   **Coriolis Effect:** Account for the Earth's rotation impacting projectile trajectory over very long ranges.
-   **Full Drop Charts/Dope Cards:** Generate printable tables of holdovers and windage for various distances.
-   **BC Estimator Refinement:** Develop a more robust and accurate ballistic coefficient estimator based on projectile dimensions and known bullet types.
-   **Advanced Wind Models:** Implement more sophisticated wind inputs (e.g., full value, half value, different wind zones).
-   **Target Customization:** Allow users to define custom target sizes and shapes for hit probability calculations.

## Distant Future
1. Implementing More Advanced Ballistic Models:

    G7 Ballistic Coefficients: This is crucial. We'd need to expand the projectile profiles to include G7 BCs and, more importantly, modify the drag calculation within calculate_trajectory_with_points to use the G7 drag function (typically a different set of drag tables) instead of or in addition to G1. This involves finding or creating the mathematical representation of the G7 drag curve.

    Spin Drift: This is the horizontal deviation caused by the bullet's spin interacting with air resistance. It requires knowing the bullet's twist rate, velocity, and a formula to calculate the force. This would involve adding new terms to the force calculations within the calculate_trajectory_with_points loop.

    Coriolis Effect: This is the deflection due to the Earth's rotation. It depends on latitude, azimuth of fire, and time of flight. This is a very subtle effect only significant at extreme long ranges (typically beyond 1000-1500 yards) but is present in high-end solvers.

2. Enhancing Environmental Inputs and Modeling:

    Atmospheric Pressure/Density Altitude: While our script uses elevation, temperature, and humidity to estimate air density, a Kestrel directly measures or allows input of barometric pressure, which is a more direct and accurate way to determine density altitude. We could add direct pressure input and refine the air density calculation based on standard atmospheric models.

    Wind Models: The current constant wind model is a major simplification. Kestrels allow for:

   Wind Averaging: Averaging wind speed and direction over a period.

   Wind Zones: Inputting different wind speeds and directions at various ranges downrange. This would be a significant overhaul of the wind inputs and how calculate_trajectory_with_points applies wind.

   Headwind/Tailwind Component: While we calculate this internally, a Kestrel often gives you direct readouts.

3. "Truing" and Calibration:

    Muzzle Velocity (MV) Truing: A critical feature of high-end solvers. This involves shooting at a known long distance, measuring the actual bullet drop, and then adjusting the calculated MV in the software until the predicted drop matches the observed drop. This corrects for unquantified variables like barrel wear, atmospheric variations, and slight discrepancies in manufacturer-stated BCs. This would require a new "Truing" mode in the script, where the user inputs observed drop, and the script iteratively adjusts MV.

    Ballistic Coefficient (BC) Truing: Similar to MV truing, but adjusts the BC based on observed drop. Often, a combination of MV and BC truing is performed.

4. Advanced Usability and Output:

    Customizable Targets for Probability: Allow the user to define arbitrary target shapes (circles, rectangles, complex outlines) for hit probability calculations.

    Drop Charts/Dope Cards: Generate a table of holdovers/clicks for various ranges, wind speeds, and wind angles. This is a common and highly desired output.

    Graphical User Interface (GUI): A command-line interface, while functional, is not as user-friendly or quick in the field as a GUI (like Kestrel's buttons and screen). This would involve moving to a library like Tkinter, PyQt, or Kivy, which is a massive undertaking.

    Internal Sensors (Raspberry Pi Integration): wind, pressure, moisture, gps for elevation and coordinates.

Challenges and Scope:

Complexity: Each of these features, particularly G7 modeling, spin drift, and sophisticated truing, involves complex mathematical models and careful implementation.

Data Acquisition: Finding accurate drag tables for G7 BCs can be challenging.

Validation: Rigorous testing against real-world data and other professional solvers would be crucial to ensure accuracy.

Time Commitment: This would be a project spanning many hours, requiring deep dives into ballistic theory and numerical methods.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
