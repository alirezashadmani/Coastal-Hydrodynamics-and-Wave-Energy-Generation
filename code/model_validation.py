import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_validation_metrics(simulated_data, observed_data):
    """
    Calculates bias, MAE, RMSE, CC, NSE, and KGE for model validation.
    
    Parameters:
    simulated_data (np.array): Array of simulated values.
    observed_data (np.array): Array of observed values.
    
    Returns:
    tuple: (bias, mae, rmse, correlation_coefficient, nse, kge)
    """
    # Ensure inputs are NumPy arrays for efficient computation
    simulated_data = np.array(simulated_data)
    observed_data = np.array(observed_data)

    # Calculate Bias (Mean Error)
    bias = np.mean(simulated_data - observed_data)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(observed_data, simulated_data)

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(observed_data, simulated_data))

    # Calculate Correlation Coefficient (CC) - Pearson's r
    # np.corrcoef returns a 2x2 matrix; we extract the scalar value for correlation
    correlation_coefficient = np.corrcoef(simulated_data, observed_data)[0, 1]

    # Calculate NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - (np.sum((observed_data - simulated_data)**2) / np.sum((observed_data - np.mean(observed_data))**2))

    # Calculate KGE (Kling-Gupta Efficiency)
    r = correlation_coefficient
    beta = np.mean(simulated_data) / np.mean(observed_data) # Bias ratio
    gamma = np.std(simulated_data) / np.std(observed_data)   # Variability ratio
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

    return bias, mae, rmse, correlation_coefficient, nse, kge


# --- Runnable example ---
if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic observed vs. simulated wave height data
    t = np.linspace(0, 10, 200)
    observed = 2.0 * np.sin(t) + 0.5 * np.random.randn(len(t))
    simulated = 2.1 * np.sin(t + 0.1) + 0.3 * np.random.randn(len(t))

    bias, mae, rmse, cc, nse, kge = calculate_validation_metrics(simulated, observed)

    print("=" * 40)
    print("Model Validation Metrics")
    print("=" * 40)
    print(f"  Bias:  {bias:+.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  CC:    {cc:.4f}")
    print(f"  NSE:   {nse:.4f}")
    print(f"  KGE:   {kge:.4f}")