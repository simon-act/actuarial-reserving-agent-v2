"""
Statistical Tests for Model Comparison

Implements statistical hypothesis tests for comparing predictive performance
of different reserving methods, including Diebold-Mariano test and 
model confidence sets.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.stats_utils import norm_cdf, t_cdf


@dataclass
class TestResult:
    """
    Container for statistical test results.
    
    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value of the test
        conclusion: Textual conclusion
        details: Additional details as dictionary
    """
    test_name: str
    statistic: float
    p_value: float
    conclusion: str
    details: Dict = None
    
    def __repr__(self):
        return f"TestResult(test={self.test_name}, p_value={self.p_value:.4f}, conclusion={self.conclusion})"


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Diebold-Mariano test for comparing predictive accuracy.
    
    Tests the null hypothesis that two forecasting methods have equal
    predictive accuracy against the alternative that one is better.
    
    H₀: E[L(e₁) - L(e₂)] = 0
    H₁: E[L(e₁) - L(e₂)] ≠ 0  (two-sided)
        E[L(e₁) - L(e₂)] < 0  (less: method 1 better)
        E[L(e₁) - L(e₂)] > 0  (greater: method 2 better)
    
    Reference: Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
    accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.
    
    Args:
        errors1: Forecast errors from method 1
        errors2: Forecast errors from method 2
        h: Forecast horizon (for HAC adjustment, default=1)
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult object
    """
    # Remove NaN values
    mask = ~np.isnan(errors1) & ~np.isnan(errors2)
    e1 = errors1[mask]
    e2 = errors2[mask]
    
    if len(e1) < 2:
        return TestResult(
            test_name="Diebold-Mariano",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for test"
        )
    
    # Loss differential (using squared errors as loss function)
    d = e1**2 - e2**2
    
    # Mean of loss differential
    d_mean = np.mean(d)
    
    # Variance of loss differential (HAC-robust if h > 1)
    n = len(d)
    
    if h == 1:
        # Simple variance
        d_var = np.var(d, ddof=1) / n
    else:
        # Newey-West HAC variance
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        
        for lag in range(1, h):
            gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1]
            gamma_sum += (1 - lag / h) * gamma_lag
        
        d_var = (gamma_0 + 2 * gamma_sum) / n
    
    # DM statistic
    if d_var <= 0:
        return TestResult(
            test_name="Diebold-Mariano",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Zero or negative variance"
        )
    
    dm_stat = d_mean / np.sqrt(d_var)
    
    # P-value (asymptotic normal distribution)
    if alternative == 'two-sided':
        p_value = 2 * (1 - norm_cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = norm_cdf(dm_stat)
    elif alternative == 'greater':
        p_value = 1 - norm_cdf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    # Conclusion
    if p_value < 0.05:
        if alternative == 'two-sided':
            if dm_stat < 0:
                conclusion = "Method 1 significantly better (α=0.05)"
            else:
                conclusion = "Method 2 significantly better (α=0.05)"
        elif alternative == 'less':
            conclusion = "Method 1 significantly better (α=0.05)"
        else:
            conclusion = "Method 2 significantly better (α=0.05)"
    else:
        conclusion = "No significant difference (α=0.05)"
    
    return TestResult(
        test_name="Diebold-Mariano",
        statistic=dm_stat,
        p_value=p_value,
        conclusion=conclusion,
        details={
            'alternative': alternative,
            'n_observations': n,
            'mean_loss_diff': d_mean,
            'std_loss_diff': np.sqrt(d_var * n)
        }
    )


def paired_t_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Paired t-test for comparing squared forecast errors.
    
    Tests whether the mean squared error differs significantly between
    two methods. Less robust than DM test but simpler.
    
    Args:
        errors1: Forecast errors from method 1
        errors2: Forecast errors from method 2
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult object
    """
    # Remove NaN values
    mask = ~np.isnan(errors1) & ~np.isnan(errors2)
    e1 = errors1[mask]
    e2 = errors2[mask]
    
    if len(e1) < 2:
        return TestResult(
            test_name="Paired t-test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for test"
        )
    
    # Squared errors
    se1 = e1**2
    se2 = e2**2

    # Paired t-test (native implementation)
    d = se1 - se2  # Differences
    n = len(d)
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)

    if d_std == 0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = d_mean / (d_std / np.sqrt(n))
        df = n - 1

        # P-value using t-distribution approximation
        if alternative == 'two-sided':
            p_value = 2 * (1 - t_cdf(abs(t_stat), df))
        elif alternative == 'less':
            p_value = t_cdf(t_stat, df)
        else:  # greater
            p_value = 1 - t_cdf(t_stat, df)
    
    # Conclusion
    if p_value < 0.05:
        if alternative == 'two-sided':
            if t_stat < 0:
                conclusion = "Method 1 significantly better (α=0.05)"
            else:
                conclusion = "Method 2 significantly better (α=0.05)"
        elif alternative == 'less':
            conclusion = "Method 1 significantly better (α=0.05)"
        else:
            conclusion = "Method 2 significantly better (α=0.05)"
    else:
        conclusion = "No significant difference (α=0.05)"
    
    return TestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_value,
        conclusion=conclusion,
        details={
            'alternative': alternative,
            'n_observations': len(e1),
            'mse1': np.mean(se1),
            'mse2': np.mean(se2)
        }
    )


def wilcoxon_signed_rank_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    alternative: str = 'two-sided'
) -> TestResult:
    """
    Wilcoxon signed-rank test for comparing forecast errors.
    
    Non-parametric test. Does not assume normality of error differences.
    More robust to outliers than t-test.
    
    Args:
        errors1: Forecast errors from method 1
        errors2: Forecast errors from method 2
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        TestResult object
    """
    # Remove NaN values
    mask = ~np.isnan(errors1) & ~np.isnan(errors2)
    e1 = errors1[mask]
    e2 = errors2[mask]
    
    if len(e1) < 5:
        return TestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for test (n<5)"
        )
    
    # Squared errors
    se1 = e1**2
    se2 = e2**2

    # Wilcoxon signed-rank test (native implementation)
    d = se1 - se2  # Differences
    d_nonzero = d[d != 0]
    n = len(d_nonzero)

    if n < 5:
        return TestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient non-zero differences"
        )

    # Rank the absolute differences
    abs_d = np.abs(d_nonzero)
    ranks = np.argsort(np.argsort(abs_d)) + 1  # Ranks from 1 to n

    # Sum of ranks for positive and negative differences
    w_plus = np.sum(ranks[d_nonzero > 0])
    w_minus = np.sum(ranks[d_nonzero < 0])

    # Test statistic
    w_stat = min(w_plus, w_minus)

    # Normal approximation for p-value (n >= 10)
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if sigma == 0:
        z_stat = 0
        p_value = 1.0
    else:
        z_stat = (w_stat - mu) / sigma

        if alternative == 'two-sided':
            p_value = 2 * norm_cdf(z_stat)
        elif alternative == 'less':
            # Method 1 better means w_plus is small
            p_value = norm_cdf((w_plus - mu) / sigma)
        else:  # greater
            p_value = 1 - norm_cdf((w_plus - mu) / sigma)

    # Conclusion
    if p_value < 0.05:
        if alternative == 'two-sided':
            if np.median(se1) < np.median(se2):
                conclusion = "Method 1 significantly better (α=0.05)"
            else:
                conclusion = "Method 2 significantly better (α=0.05)"
        elif alternative == 'less':
            conclusion = "Method 1 significantly better (α=0.05)"
        else:
            conclusion = "Method 2 significantly better (α=0.05)"
    else:
        conclusion = "No significant difference (α=0.05)"

    return TestResult(
        test_name="Wilcoxon Signed-Rank",
        statistic=w_stat,
        p_value=p_value,
        conclusion=conclusion,
        details={
            'alternative': alternative,
            'n_observations': len(e1),
            'n_nonzero': n,
            'w_plus': w_plus,
            'w_minus': w_minus
        }
    )


def compute_pairwise_comparisons(
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray,
    test_type: str = 'dm'
) -> pd.DataFrame:
    """
    Compute pairwise statistical comparisons between all methods.
    
    Args:
        predictions_dict: Dictionary mapping method names to predictions
        actuals: Actual observed values
        test_type: Type of test ('dm', 't_test', or 'wilcoxon')
        
    Returns:
        DataFrame with p-values for all pairwise comparisons
    """
    methods = list(predictions_dict.keys())
    n_methods = len(methods)
    
    # Initialize p-value matrix
    p_values = pd.DataFrame(
        index=methods,
        columns=methods,
        dtype=float
    )
    
    # Compute errors for each method
    errors = {}
    for name, preds in predictions_dict.items():
        errors[name] = actuals - preds
    
    # Compute pairwise comparisons
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                p_values.loc[method1, method2] = 1.0
            elif i < j:
                # Compute test
                if test_type == 'dm':
                    result = diebold_mariano_test(
                        errors[method1], 
                        errors[method2],
                        alternative='two-sided'
                    )
                elif test_type == 't_test':
                    result = paired_t_test(
                        errors[method1],
                        errors[method2],
                        alternative='two-sided'
                    )
                elif test_type == 'wilcoxon':
                    result = wilcoxon_signed_rank_test(
                        errors[method1],
                        errors[method2],
                        alternative='two-sided'
                    )
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                
                p_values.loc[method1, method2] = result.p_value
                p_values.loc[method2, method1] = result.p_value
    
    return p_values


def model_confidence_set(
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray,
    alpha: float = 0.10,
    test_type: str = 'dm'
) -> Dict:
    """
    Model Confidence Set (MCS) procedure.
    
    Identifies the set of models that are not significantly worse than
    the best model. Sequential elimination procedure.
    
    Reference: Hansen, P. R., Lunde, A., & Nason, J. M. (2011). 
    The model confidence set. Econometrica, 79(2), 453-497.
    
    Args:
        predictions_dict: Dictionary mapping method names to predictions
        actuals: Actual observed values
        alpha: Significance level (default 0.10)
        test_type: Type of test to use
        
    Returns:
        Dictionary with MCS results
    """
    methods = list(predictions_dict.keys())
    remaining_methods = methods.copy()
    
    # Compute losses (squared errors)
    losses = {}
    for name, preds in predictions_dict.items():
        losses[name] = (actuals - preds) ** 2
    
    eliminated = []
    elimination_order = []
    
    # Sequential elimination
    while len(remaining_methods) > 1:
        # Compute pairwise tests among remaining methods
        remaining_preds = {m: predictions_dict[m] for m in remaining_methods}
        p_values = compute_pairwise_comparisons(remaining_preds, actuals, test_type)
        
        # Find method with highest p-value (least significant difference from others)
        # This is the "safest" to eliminate
        min_p_values = p_values.min(axis=1)
        
        # Eliminate method with highest minimum p-value
        to_eliminate = min_p_values.idxmax()
        
        # Check if we should stop (all remaining are significantly different)
        if min_p_values.max() < alpha:
            break
        
        eliminated.append(to_eliminate)
        elimination_order.append((to_eliminate, min_p_values.max()))
        remaining_methods.remove(to_eliminate)
    
    # Compute average loss for each method
    avg_losses = {name: np.nanmean(loss) for name, loss in losses.items()}
    
    return {
        'mcs': remaining_methods,
        'eliminated': eliminated,
        'elimination_order': elimination_order,
        'alpha': alpha,
        'avg_losses': avg_losses,
        'best_model': min(avg_losses, key=avg_losses.get)
    }


def bootstrap_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> TestResult:
    """
    Bootstrap test for comparing forecast accuracy.
    
    Non-parametric bootstrap test. Makes no distributional assumptions.
    
    Args:
        errors1: Forecast errors from method 1
        errors2: Forecast errors from method 2
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        TestResult object
    """
    np.random.seed(random_state)
    
    # Remove NaN values
    mask = ~np.isnan(errors1) & ~np.isnan(errors2)
    e1 = errors1[mask]
    e2 = errors2[mask]
    
    n = len(e1)
    
    if n < 2:
        return TestResult(
            test_name="Bootstrap Test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Insufficient data for test"
        )
    
    # Observed test statistic (difference in MSE)
    mse1 = np.mean(e1**2)
    mse2 = np.mean(e2**2)
    observed_stat = mse1 - mse2
    
    # Bootstrap samples
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        boot_e1 = e1[indices]
        boot_e2 = e2[indices]
        
        boot_mse1 = np.mean(boot_e1**2)
        boot_mse2 = np.mean(boot_e2**2)
        
        bootstrap_stats.append(boot_mse1 - boot_mse2)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # P-value: proportion of bootstrap stats as extreme as observed
    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
    
    # Conclusion
    if p_value < 0.05:
        if observed_stat < 0:
            conclusion = "Method 1 significantly better (α=0.05)"
        else:
            conclusion = "Method 2 significantly better (α=0.05)"
    else:
        conclusion = "No significant difference (α=0.05)"
    
    return TestResult(
        test_name="Bootstrap Test",
        statistic=observed_stat,
        p_value=p_value,
        conclusion=conclusion,
        details={
            'n_bootstrap': n_bootstrap,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats)
        }
    )