"""
Hybrid AHP-TOPSIS Multi-Criteria Decision Making for Wave Energy Site Selection
================================================================================
Implements the Analytic Hierarchy Process (AHP) for criteria weighting and
the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)
for ranking candidate deployment sites.

Dependencies: numpy
"""

import numpy as np


# =====================================================================
# AHP: Analytic Hierarchy Process for Criteria Weighting
# =====================================================================

def ahp_weights(pairwise_matrix):
    """
    Calculate criteria weights using the Analytic Hierarchy Process.

    The principal eigenvector of the pairwise comparison matrix gives
    the relative priority (weight) of each criterion (Saaty, 1980).

    Parameters:
        pairwise_matrix (np.ndarray): Square matrix of pairwise comparisons
            using Saaty's 1-9 scale. Must be positive reciprocal (a_ij = 1/a_ji).

    Returns:
        weights (np.ndarray): Normalized criteria weights summing to 1.
        cr (float): Consistency Ratio. CR < 0.10 indicates acceptable consistency.
    """
    n = pairwise_matrix.shape[0]

    # Principal eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real
    priority_vector = np.abs(eigenvectors[:, max_idx].real)

    # Normalize to obtain weights
    weights = priority_vector / priority_vector.sum()

    # Consistency Index (CI)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    # Random Index (RI) table for matrices of size 1-10 (Saaty, 1980)
    ri_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(n, 1.49)

    # Consistency Ratio (CR)
    cr = ci / ri if ri > 0 else 0.0

    return weights, cr


# =====================================================================
# TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
# =====================================================================

def topsis_rank(decision_matrix, weights, benefit_criteria):
    """
    Rank alternatives using the TOPSIS method.

    Parameters:
        decision_matrix (np.ndarray): Shape (n_alternatives, n_criteria).
        weights (np.ndarray): Criteria weights from AHP (length n_criteria).
        benefit_criteria (list of bool): True if the criterion is a benefit
            (higher is better), False if it is a cost (lower is better).

    Returns:
        relative_closeness (np.ndarray): C* score for each alternative in [0, 1].
        ranking (np.ndarray): Indices sorted by C* in descending order (best first).
    """
    # Step 1: Vector normalization (column-wise L2 norm)
    norms = np.sqrt((decision_matrix ** 2).sum(axis=0))
    normalized = decision_matrix / norms

    # Step 2: Weighted normalized matrix
    weighted = normalized * weights

    # Step 3: Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
    n_crit = weighted.shape[1]
    pis = np.zeros(n_crit)
    nis = np.zeros(n_crit)
    for j in range(n_crit):
        if benefit_criteria[j]:
            pis[j] = weighted[:, j].max()
            nis[j] = weighted[:, j].min()
        else:
            pis[j] = weighted[:, j].min()
            nis[j] = weighted[:, j].max()

    # Step 4: Euclidean distances to PIS and NIS
    s_plus = np.sqrt(((weighted - pis) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - nis) ** 2).sum(axis=1))

    # Step 5: Relative closeness to ideal solution
    relative_closeness = s_minus / (s_plus + s_minus)

    # Rank in descending order of C*
    ranking = np.argsort(relative_closeness)[::-1]

    return relative_closeness, ranking


# =====================================================================
# Example: Wave Energy Site Selection
# =====================================================================

if __name__ == "__main__":
    # --- Define Criteria ---
    criteria_names = [
        "Wave Power (kW/m)",
        "Energy Consistency (%)",
        "Dist. to Shore (km)",
        "Dist. to Grid (km)",
        "Env. Sensitivity (1-5)"
    ]
    # Benefit = True (higher is better), Cost = False (lower is better)
    benefit_flags = [True, True, False, False, False]

    # --- Step 1: AHP Pairwise Comparison Matrix (Saaty's 1-9 scale) ---
    # Reflects expert judgment on relative importance of criteria
    ahp_matrix = np.array([
        [1,     3,     5,     7,     9],      # Wave Power
        [1/3,   1,     3,     5,     7],      # Energy Consistency
        [1/5,   1/3,   1,     3,     5],      # Distance to Shore
        [1/7,   1/5,   1/3,   1,     3],      # Distance to Grid
        [1/9,   1/7,   1/5,   1/3,   1]       # Env. Sensitivity
    ])

    weights, cr = ahp_weights(ahp_matrix)

    print("=" * 60)
    print("AHP CRITERIA WEIGHTING")
    print("=" * 60)
    for name, w in zip(criteria_names, weights):
        print(f"  {name:30s}: {w:.4f}")
    print(f"\n  Consistency Ratio (CR): {cr:.4f}")
    if cr < 0.10:
        print("  CR < 0.10 — Judgments are consistent.")
    else:
        print("  CR >= 0.10 — Revise pairwise comparisons!")

    # --- Step 2: Decision Matrix (6 candidate sites x 5 criteria) ---
    site_names = ["Site A", "Site B", "Site C", "Site D", "Site E", "Site F"]
    decision_matrix = np.array([
        [35.0,  55.0,  15.0,  20.0,  3],    # Site A
        [50.0,  70.0,  25.0,  30.0,  2],    # Site B
        [28.0,  45.0,   8.0,  12.0,  5],    # Site C
        [45.0,  65.0,  18.0,  22.0,  1],    # Site D
        [40.0,  60.0,  12.0,  15.0,  4],    # Site E
        [55.0,  75.0,  30.0,  35.0,  2],    # Site F
    ])

    # --- Step 3: TOPSIS Ranking ---
    closeness, ranking = topsis_rank(decision_matrix, weights, benefit_flags)

    print("\n" + "=" * 60)
    print("TOPSIS RANKING RESULTS")
    print("=" * 60)
    print(f"  {'Rank':<6} {'Site':<10} {'C*':<10}")
    print("  " + "-" * 26)
    for rank, idx in enumerate(ranking, 1):
        print(f"  {rank:<6} {site_names[idx]:<10} {closeness[idx]:.4f}")
    print(f"\n  Best site: {site_names[ranking[0]]}")
