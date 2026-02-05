"""
Native Statistical Utilities

Provides statistical functions without scipy dependency.
These are approximations suitable for actuarial reserving purposes.
"""

import numpy as np
from typing import Tuple, List, Optional


def norm_ppf(p: float) -> float:
    """
    Inverse normal CDF (quantile function).

    Uses rational approximation from Abramowitz & Stegun.

    Args:
        p: Probability (0 < p < 1)

    Returns:
        z-score corresponding to probability p
    """
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p == 0.5:
        return 0.0

    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def norm_cdf(z: float) -> float:
    """
    Normal CDF using error function approximation.

    Args:
        z: z-score

    Returns:
        Cumulative probability
    """
    return 0.5 * (1 + _erf(z / np.sqrt(2)))


def _erf(x: float) -> float:
    """Error function approximation."""
    # Horner form coefficients
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def linregress(x, y) -> Tuple[float, float, float, float, float]:
    """
    Simple linear regression.

    Args:
        x: Independent variable (array-like)
        y: Dependent variable (array-like)

    Returns:
        Tuple of (slope, intercept, r_value, p_value, std_err)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    if ss_xx == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    if ss_yy == 0:
        r_value = 1.0 if ss_xy == 0 else 0.0
    else:
        r_value = ss_xy / np.sqrt(ss_xx * ss_yy)

    # Standard error of slope
    residuals = y - (slope * x + intercept)
    mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0
    std_err = np.sqrt(mse / ss_xx) if ss_xx > 0 else np.nan

    # P-value (approximate using t-distribution)
    if std_err > 0:
        t_stat = slope / std_err
        # Approximate p-value using normal distribution for large n
        p_value = 2 * (1 - norm_cdf(abs(t_stat)))
    else:
        p_value = 0.0

    return slope, intercept, r_value, p_value, std_err


def t_cdf(t: float, df: int) -> float:
    """
    Approximate Student's t CDF.

    For df > 30, uses normal approximation.
    For smaller df, uses a simple approximation.
    """
    if df > 30:
        return norm_cdf(t)

    # Simple approximation using adjusted normal
    adj = np.sqrt(df / (df - 2)) if df > 2 else 1.5
    return norm_cdf(t / adj)


def pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation coefficient.

    Args:
        x, y: Arrays to correlate

    Returns:
        Tuple of (correlation, p_value)
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator == 0:
        return 0.0, 1.0

    r = numerator / denominator

    # T-test for correlation
    if abs(r) < 1:
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p_value = 2 * (1 - t_cdf(abs(t_stat), n - 2))
    else:
        p_value = 0.0

    return r, p_value


def skewness(data: np.ndarray) -> float:
    """Calculate sample skewness."""
    n = len(data)
    if n < 3:
        return np.nan

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        return 0.0

    m3 = np.mean((data - mean) ** 3)
    return m3 / (std ** 3)


def kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis (Fisher's definition)."""
    n = len(data)
    if n < 4:
        return np.nan

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        return 0.0

    m4 = np.mean((data - mean) ** 4)
    return m4 / (std ** 4) - 3


def f_oneway(*groups) -> Tuple[float, float]:
    """
    One-way ANOVA F-test.

    Args:
        *groups: Variable number of group arrays

    Returns:
        Tuple of (F-statistic, p-value)
    """
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    # Total observations and grand mean
    n_total = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()

    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # Within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 or ss_within == 0:
        return np.nan, np.nan

    # F-statistic
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # P-value approximation using chi-square approximation for F
    # For large df_within, F * df_between ~ chi-square(df_between)
    # Use normal approximation
    if df_within > 30:
        z = (f_stat - 1) * np.sqrt(df_within / (2 * f_stat)) if f_stat > 0 else 0
        p_value = 1 - norm_cdf(z)
    else:
        # Very rough approximation
        p_value = 0.05 if f_stat > 3 else 0.5

    return f_stat, p_value


def levene(*groups) -> Tuple[float, float]:
    """
    Levene's test for equality of variances.

    Uses mean-based method (Brown-Forsythe variant uses median).

    Returns:
        Tuple of (statistic, p_value)
    """
    # Transform to absolute deviations from group mean
    transformed = [np.abs(g - np.mean(g)) for g in groups]

    # Apply one-way ANOVA to transformed data
    return f_oneway(*transformed)


def bartlett(*groups) -> Tuple[float, float]:
    """
    Bartlett's test for equality of variances.

    Assumes normality of data.

    Returns:
        Tuple of (statistic, p_value)
    """
    k = len(groups)
    if k < 2:
        return np.nan, np.nan

    n = [len(g) for g in groups]
    n_total = sum(n)

    # Pooled variance
    vars = [np.var(g, ddof=1) for g in groups]
    s_pooled = sum((ni - 1) * vi for ni, vi in zip(n, vars)) / (n_total - k)

    if s_pooled <= 0:
        return np.nan, np.nan

    # Bartlett statistic
    numerator = (n_total - k) * np.log(s_pooled) - sum(
        (ni - 1) * np.log(vi) for ni, vi in zip(n, vars) if vi > 0
    )

    c = 1 + (1 / (3 * (k - 1))) * (sum(1 / (ni - 1) for ni in n) - 1 / (n_total - k))

    stat = numerator / c

    # P-value from chi-square approximation
    # Rough approximation
    p_value = 0.05 if stat > k * 2 else 0.5

    return stat, p_value


def shapiro_wilk(data: np.ndarray) -> Tuple[float, float]:
    """
    Simplified Shapiro-Wilk test for normality.

    Returns:
        Tuple of (statistic, p_value)
    """
    n = len(data)
    if n < 3:
        return np.nan, np.nan

    # Sort data
    x = np.sort(data)
    x_mean = np.mean(x)

    # Simplified W statistic using correlation with expected order statistics
    # This is an approximation
    ss = np.sum((x - x_mean) ** 2)

    # Expected order statistics (approximate)
    expected = np.array([norm_ppf((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)])

    numerator = np.sum((x - x_mean) * (expected - np.mean(expected)))
    denominator = np.sqrt(ss * np.sum((expected - np.mean(expected))**2))

    if denominator == 0:
        return 1.0, 1.0

    r = numerator / denominator
    w = r ** 2

    # Very rough p-value approximation
    # W close to 1 indicates normality
    if w > 0.95:
        p_value = 0.5
    elif w > 0.90:
        p_value = 0.1
    elif w > 0.85:
        p_value = 0.05
    else:
        p_value = 0.01

    return w, p_value


def jarque_bera(data: np.ndarray) -> Tuple[float, float]:
    """
    Jarque-Bera test for normality.

    Returns:
        Tuple of (statistic, p_value)
    """
    n = len(data)
    if n < 4:
        return np.nan, np.nan

    s = skewness(data)
    k = kurtosis(data)

    if np.isnan(s) or np.isnan(k):
        return np.nan, np.nan

    # JB statistic
    jb = (n / 6) * (s**2 + (k**2) / 4)

    # P-value approximation (chi-square with 2 df)
    # Rough approximation
    if jb < 1:
        p_value = 0.5
    elif jb < 3:
        p_value = 0.2
    elif jb < 6:
        p_value = 0.05
    else:
        p_value = 0.01

    return jb, p_value


def kstest(data: np.ndarray, dist: str = 'norm', args: Tuple = None) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test.

    Args:
        data: Sample data
        dist: Distribution to test against ('norm')
        args: Distribution parameters (mean, std)

    Returns:
        Tuple of (statistic, p_value)
    """
    n = len(data)
    if n < 5:
        return np.nan, np.nan

    if args is None:
        mean, std = np.mean(data), np.std(data)
    else:
        mean, std = args

    # Sort data
    x_sorted = np.sort(data)

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF (standard normal)
    if std > 0:
        z = (x_sorted - mean) / std
        tcdf = np.array([norm_cdf(zi) for zi in z])
    else:
        return np.nan, np.nan

    # KS statistic
    d_plus = np.max(ecdf - tcdf)
    d_minus = np.max(tcdf - np.concatenate([[0], ecdf[:-1]]))
    ks_stat = max(d_plus, d_minus)

    # P-value approximation
    # Critical values approximately: 0.05 -> 1.36/sqrt(n), 0.01 -> 1.63/sqrt(n)
    sqrt_n = np.sqrt(n)
    if ks_stat < 1.0 / sqrt_n:
        p_value = 0.5
    elif ks_stat < 1.36 / sqrt_n:
        p_value = 0.1
    elif ks_stat < 1.63 / sqrt_n:
        p_value = 0.05
    else:
        p_value = 0.01

    return ks_stat, p_value
