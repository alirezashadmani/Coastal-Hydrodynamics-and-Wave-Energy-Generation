"""
Post-Processing Visualization for SWAN Wave Model Output
=========================================================
Demonstrates three visualization types using synthetic data:
  1. Temporal  — Time series of significant wave height at a point
  2. Spatial   — 2D contour map of Hs with wave-direction quiver overlay
  3. Spectral  — Polar plot of directional wave energy spectrum E(f, theta)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# --- 1. Temporal Visualization (Time Series) ---
def plot_time_series_example():
    """Plot a 72-hour time series of significant wave height at a single point."""
    time_hours = np.arange(0, 72, 1)
    # Simulate Hs with diurnal variation + random noise
    sample_hs = 1.5 + 0.8 * np.sin(time_hours * 2 * np.pi / 24) \
                + np.random.normal(0, 0.1, len(time_hours))

    plt.figure(figsize=(10, 4))
    plt.plot(time_hours, sample_hs, label='$H_s$ (m)', color='darkblue')
    plt.title('Temporal Evolution of Significant Wave Height')
    plt.xlabel('Time (Hours Since Start)')
    plt.ylabel('Significant Wave Height ($H_s$, m)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- 2. Spatial Visualization (2D Map with Vectors) ---
def plot_spatial_map_example():
    """Plot a 2D contour map of Hs with mean wave direction quiver overlay."""
    grid_res = 50
    X = np.linspace(0, 10000, grid_res)   # Cross-shore (m)
    Y = np.linspace(0, 5000, grid_res)    # Along-shore (m)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    # Simulate shoaling: Hs decreases towards the coast (X = 10000)
    Hs_data = 3.0 * (1 - (X_mesh / 10000) * 0.5) + 0.5

    # Simulate wave direction (nautical, 'coming from') with refraction
    Dir_data_nautical = 270 - 10 * (Y_mesh / 5000)
    Dir_angle_cartesian = 270 - Dir_data_nautical   # Convert to 'going to'

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scalar field: significant wave height
    contour = ax.contourf(X_mesh, Y_mesh, Hs_data, levels=20, cmap='inferno')
    fig.colorbar(contour, label='Significant Wave Height ($H_s$, m)')

    # Vector field: mean wave direction (downsampled for clarity)
    skip = (slice(None, None, 5), slice(None, None, 5))
    U = np.cos(np.deg2rad(Dir_angle_cartesian[skip]))
    V = np.sin(np.deg2rad(Dir_angle_cartesian[skip]))
    ax.quiver(X_mesh[skip], Y_mesh[skip], U, V,
              color='cyan', pivot='mid', scale=40,
              headwidth=3, headlength=4, alpha=0.9)

    ax.set_title('Spatial Distribution of Wave Height and Direction (Refraction)')
    ax.set_xlabel('Cross-shore Distance (m)')
    ax.set_ylabel('Along-shore Distance (m)')
    plt.tight_layout()
    plt.show()


# --- 3. Spectral Visualization (Directional Wave Spectrum) ---
def plot_spectral_example():
    """
    Plot a directional wave energy spectrum E(f, theta) as a polar contour.

    Simulates a JONSWAP-like frequency spectrum combined with a cos^2s
    directional spreading function — the standard representation used in
    spectral wave models like SWAN.
    """
    frequencies = np.linspace(0.03, 0.5, 80)      # Hz
    directions = np.linspace(0, 2 * np.pi, 90)    # radians (full circle)
    F, D = np.meshgrid(frequencies, directions)

    # Frequency spectrum: simplified JONSWAP shape
    fp = 0.10                # Peak frequency (Hz)
    gamma_peak = 3.3         # Peak enhancement factor
    sigma = np.where(F <= fp, 0.07, 0.09)
    S_f = F**(-5) * np.exp(-1.25 * (fp / F)**4) \
        * gamma_peak ** np.exp(-0.5 * ((F - fp) / (sigma * fp))**2)

    # Directional spreading: cos^2s (Mitsuyasu et al., 1975)
    theta_mean = np.deg2rad(270)     # Mean wave direction (from West)
    s = 15                           # Spreading parameter (narrow swell)
    D_theta = np.cos(0.5 * (D - theta_mean)) ** (2 * s)

    # Combined spectrum E(f, theta)
    E = S_f * D_theta
    E = E / np.nanmax(E)   # Normalize for display

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    contour = ax.contourf(D, F, E, levels=20, cmap='magma')
    fig.colorbar(contour, pad=0.1, label='Normalized Spectral Density')
    ax.set_theta_zero_location('N')     # North at top (nautical convention)
    ax.set_theta_direction(-1)          # Clockwise
    ax.set_title('Directional Wave Spectrum $E(f, \\theta)$', pad=20)
    ax.set_ylabel('Frequency (Hz)', labelpad=30)
    plt.tight_layout()
    plt.show()


# --- Run all examples ---
if __name__ == "__main__":
    np.random.seed(42)
    print("Generating temporal plot...")
    plot_time_series_example()
    print("Generating spatial map...")
    plot_spatial_map_example()
    print("Generating spectral polar plot...")
    plot_spectral_example()
    print("Done.")