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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
