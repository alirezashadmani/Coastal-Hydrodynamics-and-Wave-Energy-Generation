// Pseudocode for a Genetic Algorithm for WEC Optimization
FUNCTION Run_GA(Population_Size, Number_of_Generations, Parameters_Bounds):
  // 1. Initialization
  Population = Create_Random_Population(Population_Size, Parameters_Bounds);

  // 2. Iteration Loop
  FOR generation FROM 1 TO Number_of_Generations:
    // 3. Evaluation
    Fitness_Scores = Evaluate_Fitness(Population); 

    // 4. Selection 
    Parents = Select_Fittest_Individuals(Population, Fitness_Scores);

    // 5. Crossover 
    Offspring = Create_Offspring_from_Parents(Parents);

    // 6. Mutation 
    Mutate(Offspring, Mutation_Rate); 

    // 7. Replacement 
    Population = Replace_Weakest_with_Offspring(Population, Offspring);
  END FOR

  // 8. Result
  Best_Individual = Find_Best_Individual_in_Population(Population);
  RETURN Best_Individual;
END FUNCTION

FUNCTION Evaluate_Fitness(Population):
  FOR each individual in Population:
    power_output = P_WEC(individual.radius, individual.draft, wave_climate);
    cost_value = Cost_Function(power_output, lifetime, capital_cost);
    individual.fitness = power_output / cost_value;
  END FOR
  RETURN Fitness_Scores;