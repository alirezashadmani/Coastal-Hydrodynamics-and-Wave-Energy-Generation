// Pseudocode for WEC Array Layout Optimization using Genetic Algorithm

// --- Hydrodynamic Interaction Model (q-factor) ---
FUNCTION Compute_Q_Factor(distance, wavenumber, alpha, decay):
  // Far-field approximation using Bessel function J0 (Parks, 1976)
  // q > 1: constructive interference; q < 1: destructive interference
  kd = wavenumber * distance;
  q = 1.0 + alpha * J0(kd) * EXP(-decay * kd);
  RETURN MAX(q, 0.1); // Physical lower bound
END FUNCTION

// --- Array Power Calculation ---
FUNCTION Compute_Array_Power(Positions, wavenumber, P_isolated):
  N = NUMBER_OF_WECS(Positions);
  Total_Power = 0;

  FOR each WEC i FROM 1 TO N:
    q_sum = 0;
    FOR each WEC j FROM 1 TO N WHERE j != i:
      d_ij = EUCLIDEAN_DISTANCE(Positions[i], Positions[j]);
      q_sum = q_sum + Compute_Q_Factor(d_ij, wavenumber);
    END FOR
    q_avg_i = q_sum / (N - 1);
    Total_Power = Total_Power + P_isolated * q_avg_i;
  END FOR

  q_array = Total_Power / (N * P_isolated); // Array interaction factor
  RETURN Total_Power, q_array;
END FUNCTION

// --- Spacing Constraint Penalty ---
FUNCTION Spacing_Penalty(Positions, min_spacing):
  penalty = 0;
  FOR each pair (i, j) WHERE i < j:
    d = EUCLIDEAN_DISTANCE(Positions[i], Positions[j]);
    IF d < min_spacing:
      penalty = penalty + (min_spacing - d)^2;
    END IF
  END FOR
  RETURN penalty;
END FUNCTION

// --- Fitness Function ---
FUNCTION Array_Fitness(Chromosome, N_WECs, wavenumber, P_isolated, min_spacing):
  Positions = RESHAPE(Chromosome, [N_WECs, 2]); // Decode (x,y) pairs
  Power, _ = Compute_Array_Power(Positions, wavenumber, P_isolated);
  Penalty = Spacing_Penalty(Positions, min_spacing);
  RETURN Power - 10000 * Penalty; // Penalize constraint violations
END FUNCTION

// --- Main GA Loop for Array Optimization ---
FUNCTION Run_Array_GA(N_WECs, Pop_Size, Num_Generations, Domain_Bounds):
  // 1. Derive wave parameters
  wavelength = 1.56 * T^2; // Deep-water dispersion relation
  wavenumber = 2 * PI / wavelength;

  // 2. Initialize population: each chromosome = [x1, y1, x2, y2, ..., xN, yN]
  Population = CREATE_RANDOM(Pop_Size, 2 * N_WECs, Domain_Bounds);

  // 3. Evolution loop
  FOR generation FROM 1 TO Num_Generations:
    Fitness = EVALUATE(Population, Array_Fitness);

    // Selection (truncation: top 20%)
    Parents = SELECT_TOP(Population, Fitness, fraction=0.2);

    // Crossover (uniform: random gene selection from each parent)
    Offspring = UNIFORM_CROSSOVER(Parents, Pop_Size - |Parents|);

    // Mutation (uniform random within domain bounds)
    MUTATE(Offspring, Mutation_Rate, Domain_Bounds);

    // Replacement
    Population = COMBINE(Parents, Offspring);
  END FOR

  // 4. Return best layout
  Best = FIND_BEST(Population);
  Positions = RESHAPE(Best, [N_WECs, 2]);
  _, q_array = Compute_Array_Power(Positions, wavenumber, P_isolated);
  RETURN Positions, q_array;
END FUNCTION
