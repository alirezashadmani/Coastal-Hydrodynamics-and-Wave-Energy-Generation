"""
Post-Processing Visualization for SWAN Wave Model Output
=========================================================
Reads **real** SWAN output files from the California Bay case study:

  1. TABLE output  (site_output.tab)  → time-series at candidate sites
  2. BLOCK output  (ga_block.mat)     → 2-D spatial fields (Hsig, Tp, Dir)
  3. Spectral reconstruction          → JONSWAP directional spectrum from
                                         SWAN-resolved parameters

Dependencies: numpy, scipy, matplotlib
"""

import os
import sys
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# =====================================================================
# Paths — works both as a script (__file__ defined) and in a notebook
# =====================================================================
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running inside a Jupyter notebook — use the notebook's working directory
    CODE_DIR = os.getcwd()
BASE_DIR = os.path.join(CODE_DIR, '..')
CALIF_DIR = os.path.join(BASE_DIR, 'california_bay.F')

TABLE_FILE = os.path.join(CALIF_DIR, 'site_output.tab')
MAT_FILE   = os.path.join(CALIF_DIR, 'ga_block.mat')
SITES_FILE = os.path.join(CALIF_DIR, 'candidate_sites.dat')

OUT_DIR = CODE_DIR  # save figures next to the script


# =====================================================================
# 1.  TABLE reader — parse site_output.tab
# =====================================================================

def read_swan_table(filepath, n_sites):
    """
    Parse a SWAN TABLE output with *n_sites* output points.

    SWAN writes all sites sequentially for each timestep:
        t0-site0, t0-site1, ..., t0-siteN, t1-site0, ...

    Returns
    -------
    times  : list[datetime]  — length n_times
    data   : dict            — {var_name: 2-D array (n_times, n_sites)}
    """
    headers, data_lines = [], []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s.startswith('%'):
                headers.append(s)
            elif s:
                data_lines.append(s)

    # ---- Variable names from header ----
    var_names = None
    skip_prefixes = ('Run:', 'Table:', 'SWAN')
    for h in reversed(headers):
        tokens = h.lstrip('%').split()
        if tokens and '[' not in h:
            if not all(t.startswith(skip_prefixes) for t in tokens):
                var_names = tokens
                break
    if var_names is None:
        raise ValueError("Could not find variable names in TABLE header")

    raw = np.loadtxt(data_lines)
    n_total = raw.shape[0]
    n_times = n_total // n_sites
    print(f"  TABLE: {n_total} rows → {n_times} timesteps × {n_sites} sites, "
          f"variables: {var_names}")

    data = {}
    for i, v in enumerate(var_names):
        col = raw[:, i]
        data[v] = col.reshape(n_times, n_sites)

    # ---- Reconstruct timestamps ----
    # The TABLE was produced with OUTPUT 20240101.000000 1 HR
    t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    times = [t0 + datetime.timedelta(hours=h) for h in range(n_times)]

    return times, data


# =====================================================================
# 2.  BLOCK (.MAT) reader
# =====================================================================

def read_swan_block(filepath):
    """
    Read SWAN BLOCK .MAT file and extract 3-D arrays + coordinate grids.

    Returns
    -------
    Xp, Yp : 2-D arrays (ny, nx) — longitude / latitude grids
    fields  : dict  — {'Hsig': (nt, ny, nx), 'RTpeak': ..., 'Dir': ...}
    times   : list[datetime]
    """
    import scipy.io
    mat = scipy.io.loadmat(filepath)

    Xp = mat['Xp']
    Yp = mat['Yp']

    fields = {}
    time_stamps = set()
    pattern = re.compile(r'^(\w+?)_(\d{8})_(\d{6})$')
    for key in mat:
        m = pattern.match(key)
        if m:
            var, date_s, time_s = m.groups()
            time_stamps.add(f"{date_s}_{time_s}")
            fields.setdefault(var, {})[f"{date_s}_{time_s}"] = mat[key]

    sorted_stamps = sorted(time_stamps)
    times = [datetime.datetime.strptime(s, '%Y%m%d_%H%M%S') for s in sorted_stamps]

    result = {}
    for var, stamp_dict in fields.items():
        arrays = [stamp_dict[s] for s in sorted_stamps if s in stamp_dict]
        if len(arrays) == len(sorted_stamps):
            result[var] = np.stack(arrays, axis=0)

    print(f"  BLOCK: grid {Xp.shape}, {len(times)} timesteps, "
          f"variables: {list(result.keys())}")
    return Xp, Yp, result, times


# =====================================================================
# 3.  Plotting functions
# =====================================================================

def plot_temporal(times, table_data, site_locs, site_indices=None,
                  n_show=4):
    """
    Plot Hs and Tp time series at selected sites from TABLE output.
    """
    if site_indices is None:
        # Pick evenly-spaced ocean sites (skip land = -9 exception)
        valid_mask = table_data['Hsig'][10, :] > 0
        valid_ids = np.where(valid_mask)[0]
        step = max(1, len(valid_ids) // n_show)
        site_indices = valid_ids[::step][:n_show]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle('SWAN Post-Processing — Temporal Evolution at Candidate Sites',
                 fontsize=14, fontweight='bold')

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(site_indices) - 1))
              for i in range(len(site_indices))]

    for idx, si in enumerate(site_indices):
        hs = table_data['Hsig'][:, si].copy()
        tp = table_data['RTpeak'][:, si].copy()

        # Mask exception values
        hs[hs < 0] = np.nan
        tp[tp < 0] = np.nan

        lon, lat = site_locs[si]
        label = f'Site {si} ({lon:.2f}°, {lat:.2f}°)'

        ax1.plot(times, hs, lw=0.8, color=colors[idx], label=label, alpha=0.9)
        ax2.plot(times, tp, lw=0.8, color=colors[idx], label=label, alpha=0.9)

    ax1.set_ylabel('Significant Wave Height $H_s$ (m)', fontsize=11)
    ax1.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.2, ls='--')
    ax1.set_ylim(bottom=0)

    ax2.set_ylabel('Peak Wave Period $T_p$ (s)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.2, ls='--')
    ax2.set_ylim(bottom=0)

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, 'postproc_temporal.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print(f"  ➤ Saved: {os.path.basename(path)}")
    return fig


def plot_spatial(Xp, Yp, block_fields, block_times, time_idx=None):
    """
    Plot 2-D spatial map of Hs with direction quivers from BLOCK output.
    """
    if time_idx is None:
        # Pick a mid-simulation timestep
        time_idx = len(block_times) // 2

    Hs = block_fields['Hsig'][time_idx]
    Dir = block_fields['Dir'][time_idx]
    Depth = block_fields['Depth'][time_idx]
    Tp = block_fields['RTpeak'][time_idx]

    # Mask land / exception values
    land = np.isnan(Hs) | (Hs <= 0) | np.isnan(Depth) | (Depth <= 0)
    Hs_plot = np.where(land, np.nan, Hs)
    Dir_plot = np.where(land, np.nan, Dir)

    timestamp = block_times[time_idx].strftime('%Y-%m-%d %H:%M UTC')

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(f'SWAN Post-Processing — Spatial Fields ({timestamp})',
                 fontsize=14, fontweight='bold')

    # ---- Panel 1: Hs with direction vectors ----
    ax = axes[0]
    vmax = np.nanpercentile(Hs_plot, 99)
    levels = np.linspace(0, vmax, 25)
    cf = ax.contourf(Xp, Yp, Hs_plot, levels=levels, cmap='YlOrRd',
                     extend='max')
    plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02,
                 label='$H_s$ (m)')

    # Land shading
    ax.contourf(Xp, Yp, np.where(land, 1, np.nan),
                levels=[0.5, 1.5], colors=['#c8b882'], zorder=2)
    ax.contour(Xp, Yp, np.where(land, 1, 0),
               levels=[0.5], colors='k', linewidths=0.8, zorder=3)

    # Wave direction quivers
    angle = np.deg2rad(270 - Dir_plot)
    U = np.cos(angle)
    V = np.sin(angle)
    skip = 8
    sl = (slice(None, None, skip), slice(None, None, skip))
    ax.quiver(Xp[sl], Yp[sl], U[sl], V[sl],
              color='navy', scale=35, headwidth=3, headlength=4,
              alpha=0.8, zorder=4)

    ax.set_title('Significant Wave Height & Direction', fontsize=12,
                 fontweight='bold')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, ls='--')

    # ---- Panel 2: Wave Period ----
    ax2 = axes[1]
    Tp_plot = np.where(land, np.nan, Tp)
    Tp_plot[Tp_plot <= 0] = np.nan
    vmax_tp = np.nanpercentile(Tp_plot, 99)
    levels_tp = np.linspace(0, vmax_tp, 25)
    cf2 = ax2.contourf(Xp, Yp, Tp_plot, levels=levels_tp, cmap='viridis',
                       extend='max')
    plt.colorbar(cf2, ax=ax2, shrink=0.85, pad=0.02,
                 label='$T_p$ (s)')
    ax2.contourf(Xp, Yp, np.where(land, 1, np.nan),
                 levels=[0.5, 1.5], colors=['#c8b882'], zorder=2)
    ax2.contour(Xp, Yp, np.where(land, 1, 0),
                levels=[0.5], colors='k', linewidths=0.8, zorder=3)
    ax2.set_title('Peak Wave Period', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (°E)')
    ax2.set_ylabel('Latitude (°N)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15, ls='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, 'postproc_spatial.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print(f"  ➤ Saved: {os.path.basename(path)}")
    return fig


def plot_spectral(table_data, site_locs, site_idx=None):
    """
    Reconstruct and plot a JONSWAP directional spectrum using SWAN-resolved
    parameters (Hs, Tp, Dir) at a selected output site.
    """
    if site_idx is None:
        # Find site with largest mean Hs
        hs_valid = np.where(table_data['Hsig'] > 0,
                            table_data['Hsig'], np.nan)
        with np.errstate(all='ignore'):
            mean_hs = np.nanmean(hs_valid, axis=0)
        site_idx = int(np.nanargmax(mean_hs))

    hs_ts = table_data['Hsig'][:, site_idx].copy()
    tp_ts = table_data['RTpeak'][:, site_idx].copy()
    dir_ts = table_data['Dir'][:, site_idx].copy()

    # Mask exceptions
    valid = (hs_ts > 0) & (tp_ts > 0) & (dir_ts > -900)
    hs_mean = np.mean(hs_ts[valid])
    tp_mean = np.mean(tp_ts[valid])

    # Circular mean direction
    dir_rad = np.deg2rad(dir_ts[valid])
    dir_mean = np.rad2deg(np.arctan2(np.mean(np.sin(dir_rad)),
                                      np.mean(np.cos(dir_rad)))) % 360

    fp = 1.0 / tp_mean  # peak frequency from SWAN

    lon, lat = site_locs[site_idx]
    print(f"  Spectral plot for Site {site_idx} ({lon:.2f}°, {lat:.2f}°)")
    print(f"    Mean Hs = {hs_mean:.2f} m, Tp = {tp_mean:.1f} s, "
          f"Dir = {dir_mean:.0f}°")

    # ---- Construct JONSWAP directional spectrum ----
    frequencies = np.linspace(0.03, 0.5, 100)
    directions = np.linspace(0, 2 * np.pi, 120)
    F, D = np.meshgrid(frequencies, directions)

    # Frequency spectrum: JONSWAP
    gamma = 3.3
    sigma = np.where(F <= fp, 0.07, 0.09)
    S_f = (F ** -5) * np.exp(-1.25 * (fp / F) ** 4) * \
          gamma ** np.exp(-0.5 * ((F - fp) / (sigma * fp)) ** 2)

    # Scale to match Hs: integral of S(f) ≈ Hs² / 16
    S_f *= (hs_mean ** 2 / 16.0) / np.trapezoid(S_f[0, :], frequencies)

    # Directional spreading: cos^2s
    theta_mean = np.deg2rad(dir_mean)
    s_spread = 15  # narrow swell
    D_theta = np.cos(0.5 * (D - theta_mean)) ** (2 * s_spread)
    D_theta /= np.trapezoid(D_theta[:, 0], directions)  # normalize

    E = S_f * D_theta

    # ---- Plot ----
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    levels = np.linspace(0, np.nanmax(E), 25)
    cf = ax.contourf(D, F, E, levels=levels, cmap='magma')
    fig.colorbar(cf, ax=ax, pad=0.1, shrink=0.85,
                 label='Spectral Density $E(f, \\theta)$ (m²/Hz/rad)')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(
        f'Directional Wave Spectrum — Site {site_idx}\n'
        f'$H_s$ = {hs_mean:.2f} m, $T_p$ = {tp_mean:.1f} s, '
        f'Dir = {dir_mean:.0f}°',
        fontsize=12, fontweight='bold', pad=20)
    ax.set_ylabel('Frequency (Hz)', labelpad=30)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'postproc_spectral.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print(f"  ➤ Saved: {os.path.basename(path)}")
    return fig


# =====================================================================
# MAIN
# =====================================================================

def main():
    plt.rcParams.update({'figure.max_open_warning': 0})

    print("=" * 65)
    print(" SWAN Post-Processing — California Bay Case Study")
    print("=" * 65)

    # ---- Check files exist ----
    for label, fp in [('TABLE', TABLE_FILE), ('BLOCK', MAT_FILE),
                      ('SITES', SITES_FILE)]:
        if not os.path.isfile(fp):
            print(f"  ERROR: {label} file not found: {fp}")
            sys.exit(1)

    # ---- Load site locations ----
    site_locs = np.loadtxt(SITES_FILE)
    n_sites = site_locs.shape[0]
    print(f"\n  Loaded {n_sites} candidate sites")

    # ============================================================
    # 1.  TEMPORAL  — time series from TABLE
    # ============================================================
    print(f"\n{'─'*65}")
    print("  1. Temporal Visualization (TABLE output)")
    print(f"{'─'*65}")
    times, table_data = read_swan_table(TABLE_FILE, n_sites)
    plot_temporal(times, table_data, site_locs)

    # ============================================================
    # 2.  SPATIAL  — 2-D map from BLOCK .MAT
    # ============================================================
    print(f"\n{'─'*65}")
    print("  2. Spatial Visualization (BLOCK .MAT output)")
    print(f"{'─'*65}")
    Xp, Yp, block_fields, block_times = read_swan_block(MAT_FILE)
    plot_spatial(Xp, Yp, block_fields, block_times)

    # ============================================================
    # 3.  SPECTRAL  — reconstructed JONSWAP from SWAN parameters
    # ============================================================
    print(f"\n{'─'*65}")
    print("  3. Spectral Visualization (JONSWAP from SWAN parameters)")
    print(f"{'─'*65}")
    plot_spectral(table_data, site_locs)

    plt.show()
    print(f"\n{'─'*65}")
    print("  Done. All post-processing figures saved.")


if __name__ == "__main__":
    main()