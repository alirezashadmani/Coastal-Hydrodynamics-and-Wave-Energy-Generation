// Pseudocode for Hybrid AHP-TOPSIS Site Selection
FUNCTION Perform_MCDM_Site_Selection(Criteria, Alternatives):
  // Step 1: AHP for Criteria Weighting
  Create Pairwise_Comparison_Matrix (A) for all criteria; 
  Calculate Criteria_Weights (W) as the principal eigenvector of A; 

  // Step 2: TOPSIS for Ranking Alternatives
  Create Decision_Matrix (X) of size [number of alternatives, number of criteria];

  // Step 3: Normalize the Decision Matrix
  Create Normalized_Matrix (R) of the same size;
  FOR each column j in X:
    norm_factor = SQRT(SUM of squares of all values in column j);
    FOR each row i in X:
      R[i][j] = X[i][j] / norm_factor; // Vector normalization 
    END FOR
  END FOR

  // Step 4: Create Weighted Normalized Matrix
  Create Weighted_Normalized_Matrix (V) of the same size;
  FOR each column j in R:
    FOR each row i in R:
      V[i][j] = W[j] * R[i][j]; // Apply AHP weights 
    END FOR
  END FOR

  // Step 5: Determine Positive and Negative Ideal Solutions
  PIS = array of max values for benefit criteria and min for cost criteria from V; 
  NIS = array of min values for benefit criteria and max for cost criteria from V; 

  // Step 6: Calculate Distance to PIS and NIS
  Create array of PIS_Distances (S_plus) of size [number of alternatives];
  Create array of NIS_Distances (S_minus) of size [number of alternatives];
  FOR each row i in V:
    S_plus[i] = SQRT(SUM of (V[i][j] - PIS[j])^2 for all columns j); // Euclidean distance 
    S_minus[i] = SQRT(SUM of (V[i][j] - NIS[j])^2 for all columns j); 
  END FOR

  // Step 7: Calculate Relative Closeness and Rank
  Create array of Relative_Closeness (C_star) of size [number of alternatives];
  FOR each alternative i:
    C_star[i] = S_minus[i] / (S_plus[i] + S_minus[i]); 
  END FOR

  Rank alternatives based on C_star in descending order;
  RETURN Ranked_Alternatives;
END FUNCTION