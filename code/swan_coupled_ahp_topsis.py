"""
Live-Coupled SWAN + AHP-TOPSIS Site Selection — California Bay
===============================================================
This script **runs the SWAN wave model**, reads the resolved nearshore
wave field, computes site-selection criteria from the SWAN output, and
performs AHP-TOPSIS MCDM ranking.

Workflow:
  1. Define candidate sites within the SWAN computational domain
  2. Inject POINTS + TABLE output commands into the SWAN input file
  3. Execute SWAN  →  resolves refraction, shoaling, breaking, etc.
  4. Read SWAN TABLE output at each candidate site
  5. Compute 5 criteria from SWAN-resolved parameters
  6. Run AHP-TOPSIS and rank sites

Criteria (all from SWAN output):
  1. Mean Wave Power Density (kW/m)          — Benefit
  2. Energy Consistency (% hours Hs >= 1 m)  — Benefit
  3. Water Depth (m, from SWAN DEPTH)        — Cost
  4. Distance to Shore (km, nearest 0-depth) — Cost
  5. Directional Variability (°, circ std)   — Cost

Dependencies: numpy, netCDF4, matplotlib
Requires: SWAN executable on PATH (e.g., swanrun)
"""

import json
import os
import sys
import numpy as np

# Add code directory to path for swan_runner import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import swan_runner

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'california_bay.F')
INPUT_TEMPLATE = os.path.join(MODEL_DIR, 'INPUT')
RUN_INPUT = os.path.join(MODEL_DIR, 'ahp_run')  # modified copy (no ext)
OUTPUT_TABLE = 'site_output.tab'
POINTS_FILE = 'candidate_sites.dat'
OUTPUT_JSON = os.path.join(SCRIPT_DIR, 'best_site.json')
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, 'ahp_topsis_map.png')

# SWAN domain (rotated grid approximate geographic extent)
DOMAIN_LON = (-122.55, -121.90)
DOMAIN_LAT = (37.35, 38.25)
SITE_SPACING = 0.04  # degrees (~4.5 km)

# SWAN configuration
SWAN_CMD = 'swanrun'           # change to match your installation
SWAN_NPROC = 4
SWAN_VARS = ['HSIGN', 'RTP', 'DIR', 'DEPTH']
TIME_START = '20240101.000000'
TIME_STEP = '1 HR'


# ===================================================================
# CANDIDATE-SITE GENERATION
# ===================================================================

def generate_candidate_sites():
    """Create a regular grid of candidate sites within the SWAN domain."""
    lons = np.arange(DOMAIN_LON[0], DOMAIN_LON[1] + 1e-9, SITE_SPACING)
    lats = np.arange(DOMAIN_LAT[0], DOMAIN_LAT[1] + 1e-9, SITE_SPACING)
    sites = [(float(lo), float(la)) for la in lats for lo in lons]
    return sites


# ===================================================================
# CRITERIA FROM SWAN OUTPUT
# ===================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km (scalar)."""
    R = 6371.0
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def circular_std(angles_deg):
    """Circular standard deviation for directional data."""
    rad = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-10)))))


def compute_criteria_from_swan(table_data, locations):
    """
    Compute 5 criteria for each candidate site from SWAN TABLE output.

    The TABLE contains one row per (timestep × location), cycling through
    locations at each timestep.  We reshape and compute temporal statistics.
    """
    n_sites = len(locations)
    n_total = len(table_data['Hsig'])
    n_times = n_total // n_sites

    if n_times * n_sites != n_total:
        raise ValueError(
            f"TABLE rows ({n_total}) not divisible by sites ({n_sites}). "
            f"Check SWAN output."
        )

    # Reshape to (n_times, n_sites) for each variable
    Hs = table_data['Hsig'].reshape(n_times, n_sites)
    Tp = table_data['RTpeak'].reshape(n_times, n_sites)
    Dir = table_data['Dir'].reshape(n_times, n_sites)
    Dep = table_data['Depth'].reshape(n_times, n_sites)

    rho_g2 = 1025 * 9.81 ** 2 / (64 * np.pi)

    # Identify land sites (SWAN returns Hs = -9 or Depth <= 0 for land)
    shore_locs = []  # (lon, lat) of sites with depth ~0 (shoreline)

    sites = []
    for j in range(n_sites):
        hs = Hs[:, j]
        tp = Tp[:, j]
        dr = Dir[:, j]
        dep = Dep[:, j]

        mean_dep = np.mean(dep)
        mean_hs = np.mean(hs)

        # Record shore locations
        if mean_dep <= 1.0 or mean_hs < 0:
            shore_locs.append(locations[j])
            continue

        # 1. Mean wave power (kW/m)
        valid = (hs > 0) & (tp > 0)
        if np.sum(valid) < 10:
            continue
        P = rho_g2 * hs[valid] ** 2 * tp[valid] / 1000
        mean_power = float(np.mean(P))

        # 2. Energy consistency (% time Hs >= 1 m)
        consistency = 100.0 * np.sum(hs[valid] >= 1.0) / len(hs[valid])

        # 3. Depth (m, positive below sea level)
        depth = float(np.mean(dep[dep > 0])) if np.any(dep > 0) else 0.0

        # 4. Distance to shore (nearest 0-depth site)
        lon_j, lat_j = locations[j]
        if shore_locs:
            dists = [haversine(lat_j, lon_j, la, lo) for lo, la in shore_locs]
            dist_shore = min(dists)
        else:
            dist_shore = 1.0

        # 5. Directional variability
        dr_valid = dr[valid]
        dir_var = circular_std(dr_valid)

        sites.append(dict(
            lon=lon_j, lat=lat_j,
            mean_power=round(mean_power, 2),
            consistency=round(consistency, 2),
            depth=round(depth, 1),
            dist_shore=round(dist_shore, 2),
            dir_variability=round(dir_var, 2),
            mean_hs=round(float(np.mean(hs[valid])), 3),
            mean_tp=round(float(np.mean(tp[valid])), 3),
        ))

    return sites


# ===================================================================
# AHP-TOPSIS (self-contained)
# ===================================================================

def ahp_weights(A):
    n = A.shape[0]
    evals, evecs = np.linalg.eig(A)
    idx = np.argmax(evals.real)
    w = np.abs(evecs[:, idx].real)
    w /= w.sum()
    ci = (evals[idx].real - n) / (n - 1) if n > 1 else 0.0
    ri = {1: 0, 2: 0, 3: .58, 4: .90, 5: 1.12}.get(n, 1.49)
    return w, float(ci / ri) if ri else 0.0


def topsis(X, w, benefit):
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    nc = V.shape[1]
    pis = np.array([V[:, j].max() if benefit[j] else V[:, j].min() for j in range(nc)])
    nis = np.array([V[:, j].min() if benefit[j] else V[:, j].max() for j in range(nc)])
    sp = np.sqrt(((V - pis) ** 2).sum(axis=1))
    sm = np.sqrt(((V - nis) ** 2).sum(axis=1))
    C = sm / (sp + sm)
    return C, np.argsort(C)[::-1]


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 65)
    print("Live-Coupled SWAN + AHP-TOPSIS — California Bay")
    print("=" * 65)

    # --- 0. Check SWAN ---
    if not swan_runner.check_swan_available(SWAN_CMD):
        print(f"\n  ERROR: '{SWAN_CMD}' not found on PATH.")
        print("  Install SWAN and ensure the executable is accessible.")
        print("  On Linux/macOS: compile from https://swanmodel.sourceforge.io")
        print("  Then re-run this script.\n")
        sys.exit(1)

    # --- 1. Candidate sites ---
    locations = generate_candidate_sites()
    print(f"\n[1/5] {len(locations)} candidate sites in "
          f"({DOMAIN_LAT[0]:.2f}–{DOMAIN_LAT[1]:.2f}°N, "
          f"{DOMAIN_LON[0]:.2f}–{DOMAIN_LON[1]:.2f}°E)")

    # --- 2. Prepare SWAN input ---
    print("[2/5] Preparing SWAN input …")
    pts_path = os.path.join(MODEL_DIR, POINTS_FILE)
    swan_runner.write_points_file(pts_path, locations)

    var_str = ' '.join(SWAN_VARS)
    extra_cmds = [
        f"POINTS 'sites' FILE '{POINTS_FILE}'",
        f"TABLE 'sites' HEADER '{OUTPUT_TABLE}' {var_str} "
        f"OUTPUT {TIME_START} {TIME_STEP}",
    ]
    swan_runner.prepare_input(INPUT_TEMPLATE, RUN_INPUT + '.swn', extra_cmds)

    # --- 3. Run SWAN ---
    print("[3/5] Running SWAN (this may take several minutes) …")
    swan_runner.run_swan(MODEL_DIR, input_name='ahp_run',
                         swan_cmd=SWAN_CMD, nproc=SWAN_NPROC)

    # --- 4. Read output & compute criteria ---
    print("[4/5] Reading SWAN output & computing criteria …")
    table_path = os.path.join(MODEL_DIR, OUTPUT_TABLE)
    table_data = swan_runner.read_table(table_path)
    sites = compute_criteria_from_swan(table_data, locations)
    print(f"       → {len(sites)} valid ocean sites")

    # --- 5. AHP-TOPSIS ---
    print("[5/5] AHP-TOPSIS ranking …")
    A = np.array([
        [1, 3, 5, 7, 9], [1/3, 1, 3, 5, 7], [1/5, 1/3, 1, 3, 5],
        [1/7, 1/5, 1/3, 1, 3], [1/9, 1/7, 1/5, 1/3, 1],
    ])
    crit = ["Power", "Consist", "Depth", "Shore", "DirVar"]
    benefit = [True, True, False, False, False]
    weights, cr = ahp_weights(A)
    print(f"       Weights: {dict(zip(crit, [f'{w:.3f}' for w in weights]))}")
    print(f"       CR = {cr:.4f} {'(OK)' if cr < 0.10 else '(REVISE!)'}")

    X = np.array([[s['mean_power'], s['consistency'], s['depth'],
                    s['dist_shore'], s['dir_variability']] for s in sites])
    C, ranking = topsis(X, weights, benefit)

    # Print top 10
    n_show = min(10, len(ranking))
    print(f"\n{'Rank':<5}{'Lat':>7}{'Lon':>9}{'P':>8}{'C%':>6}{'D':>7}"
          f"{'Shr':>6}{'DV':>6}{'C*':>8}")
    print("-" * 55)
    for r in range(n_show):
        i = ranking[r]
        s = sites[i]
        print(f"{r+1:<5}{s['lat']:7.3f}{s['lon']:9.3f}{s['mean_power']:8.1f}"
              f"{s['consistency']:6.1f}{s['depth']:7.1f}"
              f"{s['dist_shore']:6.1f}{s['dir_variability']:6.1f}{C[i]:8.4f}")

    best = sites[ranking[0]]
    print(f"\n  ➤ Best site: ({best['lat']:.3f}°N, {best['lon']:.3f}°E) "
          f" C*={C[ranking[0]]:.4f}")

    # Save for GA script
    out = {**best, 'c_star': round(float(C[ranking[0]]), 4)}
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  ➤ Saved to {os.path.basename(OUTPUT_JSON)}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter([s['lon'] for s in sites], [s['lat'] for s in sites],
                        c=[C[i] for i in range(len(sites))],
                        cmap='RdYlGn', s=100, edgecolors='k', linewidths=.4)
        fig.colorbar(sc, ax=ax, label='C*', shrink=0.8)
        ax.plot(best['lon'], best['lat'], '*', color='blue', ms=16, mec='k')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('AHP-TOPSIS (SWAN-resolved) — California Bay')
        ax.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150)
        print(f"  ➤ Plot → {os.path.basename(OUTPUT_PLOT)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
