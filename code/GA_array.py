"""
Genetic Algorithm for WEC Design and Array Layout Optimization
==============================================================
Part 1: Single WEC Design Optimization
    Optimizes geometric parameters [radius, draft] of an individual WEC
    to maximize the power-to-cost ratio.

Part 2: WEC Array Layout Optimization
    Optimizes (x, y) positions of N WECs within a deployment area,
    considering hydrodynamic interactions via q-factors (Parks, 1976)
    and enforcing minimum inter-device spacing constraints.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.special import j0   # Bessel function of the first kind, order 0


# =====================================================================
# PART 1: SINGLE WEC DESIGN OPTIMIZATION
# =====================================================================

def calculate_fitness(individual):
    """Single WEC fitness: simplified power-to-cost ratio."""
    radius, draft = individual
    wave_power_coeff = np.sin(radius) * np.cos(draft) + 1.5
    cost_factor = radius * 0.5 + draft * 0.8
    fitness = wave_power_coeff / cost_factor
    return fitness


def create_initial_population(pop_size, bounds):
    """Create random population within parameter bounds.
    bounds shape: (2, n_vars) — row 0 = lower, row 1 = upper."""
    num_vars = bounds.shape[1]
    population = np.empty((pop_size, num_vars))
    for j in range(num_vars):
        population[:, j] = np.random.uniform(bounds[0, j], bounds[1, j], size=pop_size)
    return population


def select_parents(population, fitness_scores, num_parents):
    """Truncation selection: keep top individuals."""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    return population[sorted_indices[:num_parents]]


def crossover(parents, offspring_size):
    """Single-point crossover."""
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2
    num_parents = parents.shape[0]
    for k in range(offspring_size[0]):
        p1 = k % num_parents
        p2 = (k + 1) % num_parents
        offspring[k, :crossover_point] = parents[p1, :crossover_point]
        offspring[k, crossover_point:] = parents[p2, crossover_point:]
    return offspring


def mutate(offspring, mutation_rate, bounds):
    """Uniform random mutation within bounds."""
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] = np.random.uniform(bounds[0, j], bounds[1, j])
    return offspring


def run_single_wec_ga():
    """Run GA for single WEC design optimization."""
    pop_size = 50
    num_generations = 100
    mutation_rate = 0.1
    bounds = np.array([[1.0, 1.0], [10.0, 10.0]])  # [radius, draft]

    population = create_initial_population(pop_size, bounds)

    for gen in range(num_generations):
        fitness_scores = np.array([calculate_fitness(ind) for ind in population])
        num_parents = pop_size // 5
        parents = select_parents(population, fitness_scores, num_parents)
        offspring_size = (pop_size - num_parents, parents.shape[1])
        offspring = crossover(parents, offspring_size)
        offspring = mutate(offspring, mutation_rate, bounds)
        population[:num_parents] = parents
        population[num_parents:] = offspring

    fitness_scores = np.array([calculate_fitness(ind) for ind in population])
    best_idx = np.argmax(fitness_scores)
    return population[best_idx], fitness_scores[best_idx]


# =====================================================================
# PART 2: WEC ARRAY LAYOUT OPTIMIZATION
# =====================================================================

def compute_q_factor(distance, wavenumber, alpha=0.8, decay=0.05):
    """
    Hydrodynamic interaction factor between two WECs (far-field approximation).

    Uses Bessel function J0(kd) to model the oscillatory constructive /
    destructive interference of cylindrical scattered waves, with exponential
    decay for increasing separation (Parks, 1976).

        q > 1 : constructive interference (power gain)
        q < 1 : destructive interference (power loss)
        q -> 1: no interaction (large separation)

    Parameters:
        distance : Separation between two WECs (m).
        wavenumber : Wave number k = 2*pi/wavelength (rad/m).
        alpha : Interaction strength coefficient.
        decay : Exponential decay rate for distant interactions.

    Returns:
        float: Interaction factor q (clamped >= 0.1).
    """
    if distance < 1e-6:
        return 1.0
    kd = wavenumber * distance
    q = 1.0 + alpha * j0(kd) * np.exp(-decay * kd)
    return max(q, 0.1)


def compute_array_power(positions, wavenumber, isolated_power):
    """
    Total array power considering pairwise q-factor interactions.

    For each WEC i, effective power = P_isolated * mean(q_ij for all j != i).

    Parameters:
        positions : WEC positions, shape (N, 2).
        wavenumber : Dominant wave number (rad/m).
        isolated_power : Single isolated WEC power (kW).

    Returns:
        total_power : Total array power (kW).
        q_array : Array interaction factor = total / (N * P_isolated).
    """
    n = positions.shape[0]
    wec_powers = np.zeros(n)

    for i in range(n):
        if n == 1:
            wec_powers[i] = isolated_power
        else:
            q_sum = sum(
                compute_q_factor(np.linalg.norm(positions[i] - positions[j]), wavenumber)
                for j in range(n) if j != i
            )
            wec_powers[i] = isolated_power * (q_sum / (n - 1))

    total_power = wec_powers.sum()
    q_array = total_power / (n * isolated_power)
    return total_power, q_array


def spacing_penalty(positions, min_spacing):
    """Quadratic penalty for WEC pairs closer than min_spacing."""
    n = positions.shape[0]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_spacing:
                penalty += (min_spacing - dist) ** 2
    return penalty


def array_fitness(chromosome, n_wecs, wavenumber, isolated_power,
                  min_spacing, penalty_weight=1e4):
    """
    Fitness for array layout GA.
    Maximizes array power while penalizing spacing violations.
    """
    positions = chromosome.reshape(n_wecs, 2)
    power, _ = compute_array_power(positions, wavenumber, isolated_power)
    pen = spacing_penalty(positions, min_spacing)
    return power - penalty_weight * pen


def run_array_ga(n_wecs=5, pop_size=100, num_generations=200,
                 mutation_rate=0.15, domain_size=(1000, 500),
                 min_spacing=100, wave_period=8.0, isolated_power=100.0):
    """
    GA for optimizing WEC array layout within a rectangular domain.

    Parameters:
        n_wecs : Number of WECs in the array.
        pop_size : Population size.
        num_generations : Number of GA generations.
        mutation_rate : Per-gene mutation probability.
        domain_size : (width, height) of deployment area (m).
        min_spacing : Minimum inter-device spacing (m).
        wave_period : Dominant wave period (s).
        isolated_power : Power of a single isolated WEC (kW).

    Returns:
        best_positions : Optimal (x, y) positions, shape (N, 2).
        best_fitness : Best fitness value.
        q_array : Array interaction factor.
        history : Fitness history per generation.
    """
    # Wave parameters via deep-water dispersion relation
    wavelength = 1.56 * wave_period ** 2
    wavenumber = 2 * np.pi / wavelength

    n_vars = 2 * n_wecs
    x_max, y_max = domain_size

    # Build per-gene bounds [x1, y1, x2, y2, ...]
    lo = np.tile([0.0, 0.0], n_wecs)
    hi = np.tile([x_max, y_max], n_wecs)

    # Initialize population
    population = np.empty((pop_size, n_vars))
    for j in range(n_vars):
        population[:, j] = np.random.uniform(lo[j], hi[j], size=pop_size)

    history = np.zeros(num_generations)

    for gen in range(num_generations):
        scores = np.array([
            array_fitness(ind, n_wecs, wavenumber, isolated_power, min_spacing)
            for ind in population
        ])
        history[gen] = scores.max()

        # Selection (truncation: top 20%)
        n_parents = max(pop_size // 5, 2)
        parents = population[np.argsort(scores)[::-1][:n_parents]].copy()

        # Uniform crossover
        n_offspring = pop_size - n_parents
        offspring = np.empty((n_offspring, n_vars))
        for k in range(n_offspring):
            p1 = parents[k % n_parents]
            p2 = parents[(k + 1) % n_parents]
            mask = np.random.rand(n_vars) < 0.5
            offspring[k] = np.where(mask, p1, p2)

        # Mutation
        for i in range(n_offspring):
            for j in range(n_vars):
                if np.random.rand() < mutation_rate:
                    offspring[i, j] = np.random.uniform(lo[j], hi[j])

        population[:n_parents] = parents
        population[n_parents:] = offspring

    # Final evaluation
    scores = np.array([
        array_fitness(ind, n_wecs, wavenumber, isolated_power, min_spacing)
        for ind in population
    ])
    best_idx = np.argmax(scores)
    best_pos = population[best_idx].reshape(n_wecs, 2)
    _, q_arr = compute_array_power(best_pos, wavenumber, isolated_power)

    return best_pos, scores[best_idx], q_arr, history


# =====================================================================
# MAIN: Run both optimizations
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # --- Part 1 ---
    print("=" * 60)
    print("PART 1: Single WEC Design Optimization")
    print("=" * 60)
    best_design, best_fit = run_single_wec_ga()
    print(f"  Best design:  radius = {best_design[0]:.4f},  draft = {best_design[1]:.4f}")
    print(f"  Best fitness: {best_fit:.4f}")

    # --- Part 2 ---
    print("\n" + "=" * 60)
    print("PART 2: WEC Array Layout Optimization")
    print("=" * 60)
    positions, fitness, q, hist = run_array_ga(
        n_wecs=5, pop_size=100, num_generations=200,
        domain_size=(1000, 500), min_spacing=100,
        wave_period=8.0, isolated_power=100.0
    )
    print(f"  WECs in array : {positions.shape[0]}")
    print(f"  Best fitness  : {fitness:.2f} kW")
    print(f"  Array q-factor: {q:.4f}")
    print(f"  Optimal positions:")
    for i, (x, y) in enumerate(positions):
        print(f"    WEC {i+1}: ({x:7.1f}, {y:7.1f}) m")

    # Verify spacing
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    print(f"  Min spacing   : {min(dists):.1f} m")