"""
Live-Coupled SWAN + GA Array Optimisation — California Bay
============================================================
This script **runs the SWAN wave model** to resolve the 2-D wave field
over a deployment area, then uses the SWAN output as the baseline for
a Genetic Algorithm that optimises WEC array layout with q-factor
hydrodynamic interactions.

Workflow:
  1. Read best site from AHP-TOPSIS (best_site.json)
  2. Inject BLOCK output into SWAN input for the deployment area
  3. Execute SWAN  →  resolved Hs, Tp, Dir over 2-D grid
  4. Read SWAN .MAT block output  →  time-averaged wave field
  5. GA fitness: interpolate SWAN wave field to WEC positions,
     apply analytical q-factor corrections for array interactions
  6. Return optimal layout

Dependencies: numpy, scipy, matplotlib
Requires: SWAN executable on PATH
"""

import json
import os
import sys
import numpy as np
from scipy.special import j0
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import swan_runner

# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'california_bay.F')
INPUT_TEMPLATE = os.path.join(MODEL_DIR, 'INPUT')
BEST_SITE_JSON = os.path.join(SCRIPT_DIR, 'best_site.json')
BLOCK_FILE = 'ga_block.mat'
LAYOUT_PLOT = os.path.join(SCRIPT_DIR, 'array_layout.png')
CONV_PLOT = os.path.join(SCRIPT_DIR, 'ga_convergence.png')

SWAN_CMD = 'swanrun'
SWAN_NPROC = 4


# ===================================================================
# Q-FACTOR & ARRAY POWER
# ===================================================================

def q_factor(d, k, alpha=0.8, decay=0.05):
    if d < 1e-6:
        return 1.0
    kd = k * d
    return max(1.0 + alpha * j0(kd) * np.exp(-decay * kd), 0.1)


def array_power(pos, k, P_field, k_field=None):
    """
    Array power using spatially-varying isolated power from SWAN field.

    Parameters
    ----------
    pos     : (N, 2) WEC positions in metres within local domain.
    k       : dominant wavenumber (rad/m).
    P_field : callable(x, y) → isolated WEC power (kW) at (x, y).
    """
    n = len(pos)
    powers = np.zeros(n)
    for i in range(n):
        P_iso_i = P_field(pos[i])
        if n == 1:
            powers[i] = P_iso_i
        else:
            qs = sum(q_factor(np.linalg.norm(pos[i] - pos[j]), k)
                     for j in range(n) if j != i)
            powers[i] = P_iso_i * qs / (n - 1)
    tot = powers.sum()
    P_iso_sum = sum(P_field(pos[i]) for i in range(n))
    q_arr = tot / P_iso_sum if P_iso_sum > 0 else 1.0
    return tot, q_arr


def spacing_pen(pos, d_min):
    n = len(pos)
    p = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < d_min:
                p += (d_min - d) ** 2
    return p


# ===================================================================
# GA
# ===================================================================

def run_ga(n_wecs, P_field, k, domain, d_min,
           pop_size=120, n_gen=300, mut_rate=0.15):
    nv = 2 * n_wecs
    xm, ym = domain
    lo = np.tile([0.0, 0.0], n_wecs)
    hi = np.tile([float(xm), float(ym)], n_wecs)

    pop = np.column_stack(
        [np.random.uniform(lo[j], hi[j], pop_size) for j in range(nv)])
    hist = np.zeros(n_gen)

    def fit(ch):
        pos = ch.reshape(n_wecs, 2)
        pw, _ = array_power(pos, k, P_field)
        return pw - 1e4 * spacing_pen(pos, d_min)

    for gen in range(n_gen):
        sc = np.array([fit(ind) for ind in pop])
        hist[gen] = sc.max()
        np_ = max(pop_size // 5, 2)
        parents = pop[np.argsort(sc)[::-1][:np_]].copy()
        no = pop_size - np_
        off = np.empty((no, nv))
        for i in range(no):
            p1, p2 = parents[i % np_], parents[(i + 1) % np_]
            off[i] = np.where(np.random.rand(nv) < .5, p1, p2)
        for i in range(no):
            for j in range(nv):
                if np.random.rand() < mut_rate:
                    off[i, j] = np.random.uniform(lo[j], hi[j])
        pop[:np_] = parents
        pop[np_:] = off

    sc = np.array([fit(ind) for ind in pop])
    bi = np.argmax(sc)
    bp = pop[bi].reshape(n_wecs, 2)
    _, qa = array_power(bp, k, P_field)
    return bp, sc[bi], qa, hist


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 65)
    print("Live-Coupled SWAN + GA Array Optimisation — California Bay")
    print("=" * 65)

    # --- 0. Check SWAN ---
    if not swan_runner.check_swan_available(SWAN_CMD):
        print(f"\n  ERROR: '{SWAN_CMD}' not found.")
        print("  Install SWAN and re-run.\n")
        sys.exit(1)

    # --- 1. Load best site ---
    if os.path.exists(BEST_SITE_JSON):
        with open(BEST_SITE_JSON) as f:
            site = json.load(f)
    else:
        print("  best_site.json not found; using default.")
        site = dict(lat=37.6, lon=-122.4, mean_hs=1.5, mean_tp=10.0,
                    mean_power=20.0)

    print(f"\n  Site: ({site['lat']:.3f}°N, {site['lon']:.3f}°E)")
    print(f"  Hs={site['mean_hs']:.2f} m  Tp={site['mean_tp']:.2f} s  "
          f"P={site['mean_power']:.1f} kW/m")

    # --- 2. Prepare SWAN with BLOCK output ---
    print("\n[1/4] Preparing SWAN input with block output …")
    extra_cmds = [
        f"BLOCK 'COMPGRID' NOHEAD '{BLOCK_FILE}' LAY-OUT 3 "
        f"XP YP HSIGN RTP DIR DEPTH OUTPUT 20240101.030000 6 HR",
    ]
    swan_runner.prepare_input(
        INPUT_TEMPLATE, os.path.join(MODEL_DIR, 'ga_run.swn'), extra_cmds)

    # --- 3. Run SWAN ---
    print("[2/4] Running SWAN (block output, may take several minutes) …")
    swan_runner.run_swan(MODEL_DIR, input_name='ga_run',
                         swan_cmd=SWAN_CMD, nproc=SWAN_NPROC)

    # --- 4. Read SWAN block output → build wave-field interpolator ---
    print("[3/4] Reading SWAN block output …")
    mat = swan_runner.read_mat_block(os.path.join(MODEL_DIR, BLOCK_FILE))

    Xp = mat['Xp']  # (ny, nx) lon grid
    Yp = mat['Yp']  # (ny, nx) lat grid
    Hs_all = swan_runner.extract_block_variable(mat, 'Hsig')  # (nt, ny, nx)
    Tp_all = swan_runner.extract_block_variable(mat, 'RTpeak')

    # Time-averaged fields
    Hs_mean = np.nanmean(Hs_all, axis=0)
    Tp_mean = np.nanmean(Tp_all, axis=0)

    # Wave power field (kW/m)
    rho_g2 = 1025 * 9.81 ** 2 / (64 * np.pi)
    P_map = rho_g2 * Hs_mean ** 2 * Tp_mean / 1000

    # Convert to local metres from southwest corner
    lat0, lon0 = Yp.min(), Xp.min()
    deg2m_x = 111320 * np.cos(np.radians(lat0))
    deg2m_y = 110540
    x_m = (Xp[0, :] - lon0) * deg2m_x   # 1-D x axis in metres
    y_m = (Yp[:, 0] - lat0) * deg2m_y   # 1-D y axis in metres

    capture_width = 5.0  # metres
    P_wec_map = P_map * capture_width  # kW per isolated WEC

    # Build interpolator for isolated WEC power
    interp = RegularGridInterpolator(
        (y_m, x_m), P_wec_map,
        method='linear', bounds_error=False, fill_value=0.0,
    )

    def P_field(pos_xy):
        """pos_xy = [x, y] in metres → isolated WEC power (kW)."""
        return float(interp([[pos_xy[1], pos_xy[0]]]))

    # Wave parameters
    Tp_dom = float(np.nanmean(Tp_mean[Tp_mean > 0]))
    wavelength = 1.56 * Tp_dom ** 2
    k = 2 * np.pi / wavelength

    domain_m = (float(x_m.max()), float(y_m.max()))

    print(f"       Grid: {Xp.shape[1]}×{Xp.shape[0]} → "
          f"{domain_m[0]:.0f}×{domain_m[1]:.0f} m")
    print(f"       Tp_dom={Tp_dom:.1f} s  λ={wavelength:.0f} m  k={k:.4f}")

    # --- 5. GA ---
    print("[4/4] Running GA …")
    n_wecs = 5
    d_min = 100  # metres
    np.random.seed(42)

    pos, best_fit, q_arr, hist = run_ga(
        n_wecs, P_field, k, domain_m, d_min)

    total_p, _ = array_power(pos, k, P_field)
    dists = [np.linalg.norm(pos[i] - pos[j])
             for i in range(n_wecs) for j in range(i + 1, n_wecs)]

    print(f"\n  {'='*50}")
    print(f"  RESULTS (SWAN-resolved wave field)")
    print(f"  {'='*50}")
    print(f"    q-factor   : {q_arr:.4f}")
    print(f"    Total power: {total_p:.1f} kW")
    print(f"    Min spacing: {min(dists):.1f} m")
    for i, (x, y) in enumerate(pos):
        print(f"    WEC {i+1}: ({x:7.1f}, {y:7.1f}) m")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(pos[:, 0], pos[:, 1], s=250, c='dodgerblue',
                   edgecolors='navy', linewidths=1.5, zorder=5, label='WEC')
        for i, (x, y) in enumerate(pos):
            ax.annotate(f' {i+1}', (x, y), fontsize=10, fontweight='bold')
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((0, 0), domain_m[0], domain_m[1],
                               lw=1.5, ec='grey', fc='lightyellow', ls='--'))
        ax.set_xlim(-100, domain_m[0] + 100)
        ax.set_ylim(-100, domain_m[1] + 100)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f"SWAN-Coupled Array Layout — {n_wecs} WECs\n"
                     f"q={q_arr:.3f}  P={total_p:.0f} kW  "
                     f"spacing≥{min(dists):.0f} m")
        ax.legend(); ax.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(LAYOUT_PLOT, dpi=150)
        print(f"\n  ➤ {os.path.basename(LAYOUT_PLOT)}")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(hist, color='darkblue', lw=1.2)
        ax2.set_xlabel('Generation'); ax2.set_ylabel('Best Fitness (kW)')
        ax2.set_title('GA Convergence (SWAN-coupled)')
        ax2.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(CONV_PLOT, dpi=150)
        print(f"  ➤ {os.path.basename(CONV_PLOT)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
