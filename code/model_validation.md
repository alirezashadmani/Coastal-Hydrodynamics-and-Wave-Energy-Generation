// Pseudocode for Model Validation
FUNCTION Validate_Model(Simulated_Data, Observed_Data):
  IF datasets are not independent:
    THROW ERROR "Calibration and Validation datasets must be independent."
  END IF

  N = LENGTH(Simulated_Data)
  Bias = 0
  RMSE = 0
  MAE = 0
  CC_Numerator = 0
  CC_Denominator_S = 0
  CC_Denominator_O = 0

  Avg_Sim = MEAN(Simulated_Data)
  Avg_Obs = MEAN(Observed_Data)
  Std_Sim = STDEV(Simulated_Data)
  Std_Obs = STDEV(Observed_Data)

  FOR i FROM 1 TO N:
    Bias = Bias + (Simulated_Data[i] - Observed_Data[i])
    MAE = MAE + ABS(Simulated_Data[i] - Observed_Data[i])
    RMSE = RMSE + (Simulated_Data[i] - Observed_Data[i])^2
    CC_Numerator = CC_Numerator + (Simulated_Data[i] - Avg_Sim) * (Observed_Data[i] - Avg_Obs)
    CC_Denominator_S = CC_Denominator_S + (Simulated_Data[i] - Avg_Sim)^2
    CC_Denominator_O = CC_Denominator_O + (Observed_Data[i] - Avg_Obs)^2
  END FOR

  Bias = Bias / N
  MAE = MAE / N
  RMSE = SQRT(RMSE / N)
  CC = CC_Numerator / (SQRT(CC_Denominator_S) * SQRT(CC_Denominator_O))

  // NSE (Nash-Sutcliffe Efficiency)
  NSE_Numerator = SUM of (Observed_Data - Simulated_Data)^2
  NSE_Denominator = SUM of (Observed_Data - Avg_Obs)^2
  NSE = 1 - (NSE_Numerator / NSE_Denominator)

  // KGE (Kling-Gupta Efficiency) components
  r = CC // Correlation component
  beta = Avg_Sim / Avg_Obs // Bias ratio component
  gamma = Std_Sim / Std_Obs // Variability ratio component
  KGE = 1 - SQRT((r-1)^2 + (beta-1)^2 + (gamma-1)^2)

  RETURN Bias, MAE, RMSE, CC, NSE, KGE
END FUNCTION