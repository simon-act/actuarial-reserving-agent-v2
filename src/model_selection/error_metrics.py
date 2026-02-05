"""
Error Metrics for Reserving Model Validation

Implements various error metrics appropriate for evaluating predictive
performance of actuarial reserving methods on out-of-sample data.
"""

import numpy as np
from typing import Callable, Dict


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    
    MAE = (1/n) Σ|yᵢ - ŷᵢ|
    
    Interpretable in original units. Robust to outliers.
    Less sensitive to large errors than RMSE.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).
    
    RMSE = sqrt((1/n) Σ(yᵢ - ŷᵢ)²)
    
    Penalizes large errors more heavily than MAE.
    Standard metric for comparing predictive accuracy.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) Σ|100 × (yᵢ - ŷᵢ) / yᵢ|
    
    Scale-independent metric. Useful for comparing across different
    loss magnitudes. Undefined when actual = 0.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0)
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs(100 * (actual[mask] - predicted[mask]) / actual[mask]))


def symmetric_mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    
    sMAPE = (100/n) Σ|yᵢ - ŷᵢ| / ((|yᵢ| + |ŷᵢ|) / 2)
    
    Symmetric version of MAPE. Avoids issues when actual = 0.
    Bounded between 0 and 200.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        sMAPE value (as percentage)
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    numerator = np.abs(actual[mask] - predicted[mask])
    denominator = (np.abs(actual[mask]) + np.abs(predicted[mask])) / 2
    
    # Avoid division by zero
    valid = denominator > 0
    if valid.sum() == 0:
        return np.nan
    
    return np.mean(100 * numerator[valid] / denominator[valid])


def mean_squared_logarithmic_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Squared Logarithmic Error (MSLE).
    
    MSLE = (1/n) Σ(log(1 + yᵢ) - log(1 + ŷᵢ))²
    
    Penalizes underestimation more than overestimation.
    Appropriate when errors scale with magnitude.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MSLE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual >= 0) & (predicted >= 0)
    if mask.sum() == 0:
        return np.nan
    
    log_actual = np.log1p(actual[mask])
    log_predicted = np.log1p(predicted[mask])
    
    return np.mean((log_actual - log_predicted) ** 2)


def mean_bias_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Bias Error (MBE).
    
    MBE = (1/n) Σ(ŷᵢ - yᵢ)
    
    Measures systematic bias. Positive = overestimation, negative = underestimation.
    Can be zero even with large errors if they cancel out.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MBE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(predicted[mask] - actual[mask])


def median_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Median Absolute Error (MedAE).
    
    MedAE = median(|yᵢ - ŷᵢ|)
    
    Robust to outliers. Represents typical error.
    Less affected by extreme mispredictions than mean-based metrics.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MedAE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    return np.median(np.abs(actual[mask] - predicted[mask]))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    
    R² = 1 - (SS_res / SS_tot)
    
    Proportion of variance explained by the model.
    R² = 1 is perfect, R² = 0 is baseline (mean), R² < 0 is worse than baseline.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        R² value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - actual_clean.mean()) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def normalized_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error (NRMSE).
    
    NRMSE = RMSE / (max(y) - min(y))
    
    Scale-independent version of RMSE.
    Useful for comparing across different loss magnitudes.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        NRMSE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    rmse = root_mean_squared_error(actual, predicted)
    actual_range = actual[mask].max() - actual[mask].min()
    
    if actual_range == 0:
        return np.nan
    
    return rmse / actual_range


def max_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Maximum Absolute Error (MaxAE).
    
    MaxAE = max(|yᵢ - ŷᵢ|)
    
    Identifies worst-case prediction error.
    Useful for understanding extreme mispredictions.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        MaxAE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    return np.max(np.abs(actual[mask] - predicted[mask]))


def weighted_mean_absolute_error(
    actual: np.ndarray, 
    predicted: np.ndarray, 
    weights: np.ndarray = None
) -> float:
    """
    Weighted Mean Absolute Error.
    
    WMAE = Σwᵢ|yᵢ - ŷᵢ| / Σwᵢ
    
    Allows different errors to have different importance.
    Useful when certain predictions are more critical (e.g., recent years).
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        weights: Weights for each observation (default: equal weights)
        
    Returns:
        WMAE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    if weights is None:
        weights = np.ones_like(actual)
    
    weights_clean = weights[mask]
    errors = np.abs(actual[mask] - predicted[mask])
    
    return np.sum(weights_clean * errors) / np.sum(weights_clean)


def mean_absolute_scaled_error(
    actual: np.ndarray, 
    predicted: np.ndarray,
    actual_full: np.ndarray = None
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    
    MASE = MAE / MAE_naive
    
    Scales error relative to naive forecast (random walk).
    MASE < 1 means better than naive, MASE > 1 means worse.
    
    Args:
        actual: Actual observed values (test set)
        predicted: Predicted values
        actual_full: Full actual series for computing naive benchmark
        
    Returns:
        MASE value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    mae = mean_absolute_error(actual, predicted)
    
    # Use full series if provided, otherwise use test set
    if actual_full is None:
        actual_full = actual
    
    # Naive forecast error: |yₜ - yₜ₋₁|
    naive_errors = np.abs(np.diff(actual_full[~np.isnan(actual_full)]))
    
    if len(naive_errors) == 0:
        return np.nan
    
    mae_naive = np.mean(naive_errors)
    
    if mae_naive == 0:
        return np.nan
    
    return mae / mae_naive


def log_likelihood_normal(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Log-likelihood under normal distribution assumption.
    
    LL = -n/2 × log(2π) - n/2 × log(σ²) - (1/2σ²) Σ(yᵢ - ŷᵢ)²
    
    Probabilistic metric. Higher is better.
    Assumes errors are normally distributed.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        Log-likelihood value
    """
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return np.nan
    
    residuals = actual[mask] - predicted[mask]
    n = len(residuals)
    
    sigma_sq = np.var(residuals, ddof=1)
    
    if sigma_sq <= 0:
        return np.nan
    
    ll = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma_sq) - np.sum(residuals**2) / (2 * sigma_sq)
    
    return ll


def get_default_metrics() -> Dict[str, Callable]:
    """
    Get default set of error metrics for model validation.
    
    Returns:
        Dictionary mapping metric names to metric functions
    """
    return {
        'MAE': mean_absolute_error,
        'RMSE': root_mean_squared_error,
        'MAPE': mean_absolute_percentage_error,
        'sMAPE': symmetric_mean_absolute_percentage_error,
        'MBE': mean_bias_error,
        'MedAE': median_absolute_error,
        'R²': r_squared,
        'MaxAE': max_absolute_error
    }


def get_all_metrics() -> Dict[str, Callable]:
    """
    Get comprehensive set of all available error metrics.
    
    Returns:
        Dictionary mapping metric names to metric functions
    """
    return {
        'MAE': mean_absolute_error,
        'RMSE': root_mean_squared_error,
        'MAPE': mean_absolute_percentage_error,
        'sMAPE': symmetric_mean_absolute_percentage_error,
        'MSLE': mean_squared_logarithmic_error,
        'MBE': mean_bias_error,
        'MedAE': median_absolute_error,
        'R²': r_squared,
        'NRMSE': normalized_rmse,
        'MaxAE': max_absolute_error,
        'LogLik': log_likelihood_normal
    }


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics for a prediction.
    
    Args:
        actual: Actual observed values
        predicted: Predicted values
        
    Returns:
        Dictionary of all metric values
    """
    metrics = get_all_metrics()
    results = {}
    
    for name, func in metrics.items():
        try:
            results[name] = func(actual, predicted)
        except Exception as e:
            results[name] = np.nan
    
    return results