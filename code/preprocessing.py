"""
Pre-Processing Visualization — Bathymetry, Wave & Wind Climate
===============================================================
Produces publication-quality maps for both case-study regions:
  • GEBCO bathymetry profiles
  • ERA5 mean significant wave height & mean wave direction
  • ERA5 mean wind speed & wind rose

Regions:
  1. California Bay  (california_bay.F/)
  2. Oman Coast      (Oman/)

Dependencies: numpy, scipy, netCDF4, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import netCDF4 as nc
from scipy.interpolate import RectBivariateSpline

__authors__ = "Alireza Shadmani", "Mohammad Reza Nikoo", "Amir H. Gandomi"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.dirname(os.path.abspath(__file__))

REGIONS = {
    'California Bay': {
        'gebco': os.path.join(BASE, 'california_bay.F', 'GEBCO.asc'),
        'wave':  os.path.join(BASE, 'california_bay.F', 'WAVE.nc'),
        'wind':  os.path.join(BASE, 'california_bay.F', 'WIND.nc'),
        'tag':   'california',
    },
    'Oman Coast': {
        'gebco': os.path.join(BASE, 'Oman', 'gebco.asc'),
        'wave':  os.path.join(BASE, 'Oman', 'Wave-2021-April-May.nc'),
        'wind':  os.path.join(BASE, 'Oman', 'Wind-2021-April-May.nc'),
        'tag':   'oman',
    },
}


# ===================================================================
# DATA LOADERS
# ===================================================================

def load_gebco(path):
    """Load GEBCO ASCII-grid bathymetry."""
    with open(path) as f:
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xll   = float(f.readline().split()[1])
        yll   = float(f.readline().split()[1])
        cs    = float(f.readline().split()[1])
        nd    = float(f.readline().split()[1])
    elev = np.loadtxt(path, skiprows=6)
    elev[elev == nd] = np.nan
    lons = xll + (np.arange(ncols) + 0.5) * cs
    lats = yll + (nrows - 1 - np.arange(nrows) + 0.5) * cs
    return lons, lats, elev


def load_wave(path):
    """Load ERA5 wave NetCDF → dict with time-averaged and raw fields."""
    ds = nc.Dataset(path)
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    swh = ds.variables['swh'][:]       # (t, lat, lon) Hs
    pp1d = ds.variables['pp1d'][:]     # Tp
    mwd = ds.variables['mwd'][:]       # mean wave direction (°)

    # Parse timestamps
    tvar = ds.variables['valid_time']
    cal = tvar.calendar if hasattr(tvar, 'calendar') else 'standard'
    times = nc.num2date(tvar[:], tvar.units, cal)
    ds.close()

    # Time-averages
    mean_hs = np.ma.mean(swh, axis=0)
    mean_tp = np.ma.mean(pp1d, axis=0)

    # Circular mean direction
    mwd_rad = np.deg2rad(mwd)
    mean_dir = np.rad2deg(np.arctan2(
        np.ma.mean(np.sin(mwd_rad), axis=0),
        np.ma.mean(np.cos(mwd_rad), axis=0),
    )) % 360

    # Wave power density (kW/m)
    rho_g2 = 1025 * 9.81**2 / (64 * np.pi)
    P = rho_g2 * swh**2 * pp1d / 1000
    mean_P = np.ma.mean(P, axis=0)

    return dict(lat=lat, lon=lon, mean_hs=mean_hs, mean_tp=mean_tp,
                mean_dir=mean_dir, mean_P=mean_P,
                swh=swh, pp1d=pp1d, mwd=mwd, times=times)


def load_wind(path):
    """Load ERA5 wind NetCDF → dict with time-averaged fields."""
    ds = nc.Dataset(path)
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    u10 = ds.variables['u10'][:]
    v10 = ds.variables['v10'][:]
    ds.close()

    speed = np.sqrt(u10**2 + v10**2)
    direction = (np.rad2deg(np.arctan2(-u10, -v10)) + 360) % 360  # "from" direction

    mean_speed = np.ma.mean(speed, axis=0)
    mean_u = np.ma.mean(u10, axis=0)
    mean_v = np.ma.mean(v10, axis=0)

    return dict(lat=lat, lon=lon, mean_speed=mean_speed,
                mean_u=mean_u, mean_v=mean_v,
                speed=speed, direction=direction)


# ===================================================================
# INTERPOLATION HELPER
# ===================================================================

def upscale_and_mask(coarse_lon, coarse_lat, field, gebco,
                     target_res=0.02):
    """
    Smooth a coarse ERA5 field onto a finer regular grid and mask land.

    Parameters
    ----------
    coarse_lon, coarse_lat : 1-D arrays (ERA5 coordinates).
    field        : 2-D array on the coarse grid.
    gebco        : (lons, lats, elev) tuple from load_gebco().
    target_res   : output grid spacing in degrees (~2 km).

    Returns
    -------
    fine_lon, fine_lat, fine_field  (all 2-D meshgrids / masked array)
    """
    # Fill any masked values with nearest-neighbour so the spline works
    arr = np.ma.filled(field, np.nan)

    # Latitude must be ascending for RectBivariateSpline
    lat_asc = coarse_lat if coarse_lat[0] < coarse_lat[-1] else coarse_lat[::-1]
    dat = arr if coarse_lat[0] < coarse_lat[-1] else arr[::-1, :]

    # Replace remaining NaNs with column/row means for spline stability
    for j in range(dat.shape[1]):
        col = dat[:, j]
        nans = np.isnan(col)
        if nans.all():
            dat[:, j] = 0.0
        elif nans.any():
            dat[nans, j] = np.nanmean(col)

    # Build cubic spline on the coarse grid
    spline = RectBivariateSpline(lat_asc, coarse_lon, dat, kx=3, ky=3)

    # Fine output grid covering the ERA5 extent
    fine_lat_1d = np.arange(lat_asc[0], lat_asc[-1] + 1e-9, target_res)
    fine_lon_1d = np.arange(coarse_lon[0], coarse_lon[-1] + 1e-9, target_res)
    fine_vals = spline(fine_lat_1d, fine_lon_1d)

    FLON, FLAT = np.meshgrid(fine_lon_1d, fine_lat_1d)

    # --- Land mask from GEBCO (subsample for speed) ---
    g_lon, g_lat, g_elev = gebco
    # For each fine-grid point, look up nearest GEBCO cell
    lat_idx = np.searchsorted(g_lat[::-1], fine_lat_1d)  # g_lat is N→S
    lat_idx = np.clip(len(g_lat) - 1 - lat_idx, 0, len(g_lat) - 1)
    lon_idx = np.searchsorted(g_lon, fine_lon_1d)
    lon_idx = np.clip(lon_idx, 0, len(g_lon) - 1)

    li, lj = np.meshgrid(lat_idx, lon_idx, indexing='ij')
    land = g_elev[li, lj] >= 0
    fine_vals[land] = np.nan

    return FLON, FLAT, fine_vals


# ===================================================================
# PLOT FUNCTIONS
# ===================================================================

def plot_bathymetry(lons, lats, elev, title, ax):
    """Filled-contour bathymetry with land mask and depth contours."""
    LON, LAT = np.meshgrid(lons, lats)
    bathy = np.where(elev < 0, elev, np.nan)

    d_min = np.nanmin(bathy)
    levels = np.linspace(d_min, 0, 30)
    cf = ax.contourf(LON, LAT, bathy, levels=levels,
                     cmap='ocean_r', extend='min')

    # Land
    ax.contourf(LON, LAT, np.where(elev >= 0, elev, np.nan),
                levels=[0, np.nanmax(elev) or 1], colors=['#c8b882'], zorder=2)
    ax.contour(LON, LAT, elev, levels=[0], colors='k', linewidths=0.8, zorder=3)

    # Labelled isobaths
    iso = np.arange(int(d_min / 1000) * 1000, 0, 500)
    if len(iso) > 1:
        cs = ax.contour(LON, LAT, bathy, levels=iso,
                        colors='white', linewidths=0.4, alpha=0.6)
        ax.clabel(cs, fontsize=6, fmt='%d m', inline=True)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Depth (m)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15, ls='--')


def plot_wave_climate(wave, title, ax, gebco=None):
    """Mean Hs contour map with mean wave direction quiver overlay."""
    hs = np.ma.filled(wave['mean_hs'], np.nan)
    d = np.ma.filled(wave['mean_dir'], np.nan)

    # Smooth interpolation + land mask if GEBCO available
    if gebco is not None:
        FLON, FLAT, hs_fine = upscale_and_mask(
            wave['lon'], wave['lat'], hs, gebco)
    else:
        FLON, FLAT = np.meshgrid(wave['lon'], wave['lat'])
        hs_fine = hs

    vmin = np.nanmin(hs_fine[hs_fine > 0]) if np.any(hs_fine > 0) else 0
    levels = np.linspace(vmin, np.nanmax(hs_fine), 25)
    cf = ax.contourf(FLON, FLAT, hs_fine, levels=levels, cmap='YlOrRd')
    cbar = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Mean $H_s$ (m)', fontsize=10)

    # Direction quivers on the original coarse grid
    LON_c, LAT_c = np.meshgrid(wave['lon'], wave['lat'])
    angle = np.deg2rad(270 - d)
    U, V = np.cos(angle), np.sin(angle)
    skip = max(1, len(wave['lat']) // 12)
    sl = (slice(None, None, skip), slice(None, None, skip))
    ax.quiver(LON_c[sl], LAT_c[sl], U[sl], V[sl],
              color='navy', scale=30, headwidth=3, headlength=4, alpha=0.85,
              zorder=4)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15, ls='--')


def plot_wave_power(wave, title, ax, gebco=None):
    """Mean wave power density contour map."""
    P = np.ma.filled(wave['mean_P'], np.nan)

    if gebco is not None:
        FLON, FLAT, P_fine = upscale_and_mask(
            wave['lon'], wave['lat'], P, gebco)
    else:
        FLON, FLAT = np.meshgrid(wave['lon'], wave['lat'])
        P_fine = P

    vmin = np.nanmin(P_fine[P_fine > 0]) if np.any(P_fine > 0) else 0
    levels = np.linspace(vmin, np.nanmax(P_fine), 25)
    cf = ax.contourf(FLON, FLAT, P_fine, levels=levels, cmap='inferno')
    cbar = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Wave Power (kW/m)', fontsize=10)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15, ls='--')


def plot_wind_climate(wind, title, ax, gebco=None):
    """Mean wind speed contour map with mean wind vectors."""
    spd = np.ma.filled(wind['mean_speed'], np.nan)

    if gebco is not None:
        FLON, FLAT, spd_fine = upscale_and_mask(
            wind['lon'], wind['lat'], spd, gebco)
    else:
        FLON, FLAT = np.meshgrid(wind['lon'], wind['lat'])
        spd_fine = spd

    levels = np.linspace(np.nanmin(spd_fine), np.nanmax(spd_fine), 25)
    cf = ax.contourf(FLON, FLAT, spd_fine, levels=levels, cmap='Blues')
    cbar = plt.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Mean Wind Speed (m/s)', fontsize=10)

    # Wind vectors on original coarse grid
    LON_c, LAT_c = np.meshgrid(wind['lon'], wind['lat'])
    u = np.ma.filled(wind['mean_u'], 0)
    v = np.ma.filled(wind['mean_v'], 0)
    skip = max(1, len(wind['lat']) // 14)
    sl = (slice(None, None, skip), slice(None, None, skip))
    ax.quiver(LON_c[sl], LAT_c[sl], u[sl], v[sl],
              color='darkred', scale=80, headwidth=3, alpha=0.8, zorder=4)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15, ls='--')


def plot_wind_rose(wind, title, ax):
    """Wind-rose histogram on a polar axes."""
    spd_flat = np.ma.filled(wind['speed'], np.nan).ravel()
    dir_flat = np.ma.filled(wind['direction'], np.nan).ravel()
    valid = ~np.isnan(spd_flat) & ~np.isnan(dir_flat)
    spd_flat = spd_flat[valid]
    dir_flat = dir_flat[valid]

    n_dir_bins = 16
    dir_edges = np.linspace(0, 360, n_dir_bins + 1)
    dir_centres = np.deg2rad((dir_edges[:-1] + dir_edges[1:]) / 2)
    bar_width = np.deg2rad(360 / n_dir_bins) * 0.9

    speed_bins = [0, 3, 6, 9, 12, 25]
    colors = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62', '#b2182b']
    labels = ['0–3', '3–6', '6–9', '9–12', '>12 m/s']

    bottom = np.zeros(n_dir_bins)
    for k in range(len(speed_bins) - 1):
        mask = (spd_flat >= speed_bins[k]) & (spd_flat < speed_bins[k + 1])
        counts, _ = np.histogram(dir_flat[mask], bins=dir_edges)
        pct = 100 * counts / len(spd_flat)
        ax.bar(dir_centres, pct, width=bar_width, bottom=bottom,
               color=colors[k], edgecolor='white', linewidth=0.3,
               label=labels[k], zorder=3)
        bottom += pct

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.12),
              fontsize=8, ncol=3, title='Speed')


def plot_wave_timeseries(wave, region_name, tag, out_dir):
    """
    Plot time series of domain-averaged Hs and Tp with min/max envelope
    and 24-h rolling mean.
    """
    import datetime

    times = wave['times']
    swh = np.ma.filled(wave['swh'], np.nan)   # (t, lat, lon)
    tp  = np.ma.filled(wave['pp1d'], np.nan)

    # Convert cftime to python datetime for matplotlib
    py_times = np.array([datetime.datetime(t.year, t.month, t.day,
                                           t.hour, t.minute, t.second)
                         for t in times])

    # Spatial statistics per timestep
    hs_mean = np.nanmean(swh, axis=(1, 2))
    hs_min  = np.nanmin(swh, axis=(1, 2))
    hs_max  = np.nanmax(swh, axis=(1, 2))

    tp_mean = np.nanmean(tp, axis=(1, 2))
    tp_min  = np.nanmin(tp, axis=(1, 2))
    tp_max  = np.nanmax(tp, axis=(1, 2))

    # 24-h rolling mean (window=24 for hourly data)
    def rolling(arr, w=24):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    hs_roll = rolling(hs_mean)
    tp_roll = rolling(tp_mean)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f'Wave Climate Time Series — {region_name}',
                 fontsize=15, fontweight='bold')

    # ---- Hs panel ----
    ax1.fill_between(py_times, hs_min, hs_max,
                     color='#fddbc7', alpha=0.5, label='Spatial range')
    ax1.plot(py_times, hs_mean, color='#d6604d', lw=0.6,
             alpha=0.7, label='Domain mean')
    ax1.plot(py_times, hs_roll, color='#b2182b', lw=1.8,
             label='24-h rolling mean')
    ax1.set_ylabel('Significant Wave Height $H_s$ (m)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, ls='--')
    ax1.set_ylim(bottom=0)

    # ---- Tp panel ----
    ax2.fill_between(py_times, tp_min, tp_max,
                     color='#d1e5f0', alpha=0.5, label='Spatial range')
    ax2.plot(py_times, tp_mean, color='#4393c3', lw=0.6,
             alpha=0.7, label='Domain mean')
    ax2.plot(py_times, tp_roll, color='#2166ac', lw=1.8,
             label='24-h rolling mean')
    ax2.set_ylabel('Peak Wave Period $T_p$ (s)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2, ls='--')
    ax2.set_ylim(bottom=0)

    # Date formatting
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, f'wave_timeseries_{tag}.png')
    fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f"  ➤ Saved: {os.path.basename(p)}")
    return fig


# ===================================================================
# MAIN
# ===================================================================

def main():
    plt.rcParams.update({'figure.max_open_warning': 0})

    print("=" * 60)
    print("Pre-Processing Visualization")
    print("=" * 60)

    for region_name, paths in REGIONS.items():
        tag = paths['tag']
        print(f"\n{'─'*60}")
        print(f"  Region: {region_name}")
        print(f"{'─'*60}")

        # --- Load data ---
        print("  Loading bathymetry …")
        g_lon, g_lat, g_elev = load_gebco(paths['gebco'])

        print("  Loading wave data …")
        wave = load_wave(paths['wave'])

        print("  Loading wind data …")
        wind = load_wind(paths['wind'])

        # --- Multi-panel figure (2×2): bathy, wave Hs, wave power, wind ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle(f'Pre-Processing Overview — {region_name}',
                     fontsize=16, fontweight='bold', y=0.98)

        gebco_tuple = (g_lon, g_lat, g_elev)

        plot_bathymetry(g_lon, g_lat, g_elev,
                        'Bathymetry (GEBCO)', axes[0, 0])

        print("  Interpolating wave data to fine grid …")
        plot_wave_climate(wave,
                          'Mean Wave Height & Direction (ERA5)', axes[0, 1],
                          gebco=gebco_tuple)

        plot_wave_power(wave,
                        'Mean Wave Power Density (ERA5)', axes[1, 0],
                        gebco=gebco_tuple)

        print("  Interpolating wind data to fine grid …")
        plot_wind_climate(wind,
                          'Mean Wind Speed & Vectors (ERA5)', axes[1, 1],
                          gebco=gebco_tuple)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(OUT, f'preprocessing_{tag}.png')
        fig.savefig(p, dpi=180, bbox_inches='tight')
        print(f"  ➤ Saved: {os.path.basename(p)}")

        # --- Wind rose (separate polar figure) ---
        fig_wr, ax_wr = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=(7, 7))
        plot_wind_rose(wind, f'Wind Rose — {region_name}', ax_wr)
        plt.tight_layout()
        p_wr = os.path.join(OUT, f'windrose_{tag}.png')
        fig_wr.savefig(p_wr, dpi=150, bbox_inches='tight')
        print(f"  ➤ Saved: {os.path.basename(p_wr)}")

        # --- Wave time series ---
        print("  Plotting wave time series …")
        plot_wave_timeseries(wave, region_name, tag, OUT)

    # --- Side-by-side bathymetry comparison ---
    print(f"\n{'─'*60}")
    print("  Side-by-side bathymetry comparison")
    fig_cmp, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10))
    ca = REGIONS['California Bay']
    om = REGIONS['Oman Coast']
    l1, l2, e1 = load_gebco(ca['gebco'])
    l3, l4, e2 = load_gebco(om['gebco'])
    plot_bathymetry(l1, l2, e1, 'California Bay', ax_a)
    plot_bathymetry(l3, l4, e2, 'Oman Coast', ax_b)
    fig_cmp.suptitle('GEBCO Bathymetry — Study Regions',
                     fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    p_cmp = os.path.join(OUT, 'bathymetry_comparison.png')
    fig_cmp.savefig(p_cmp, dpi=150, bbox_inches='tight')
    print(f"  ➤ Saved: {os.path.basename(p_cmp)}")

    # plt.show()
    print("\n  Done.")


if __name__ == "__main__":
    main()
