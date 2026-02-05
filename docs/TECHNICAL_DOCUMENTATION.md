# 📖 Technical Documentation
## Actuarial Reserving Analysis - Methods & Formulas

---

## Table of Contents

1. [Chain-Ladder Method](#1-chain-ladder-method)
2. [Mack's Stochastic Model](#2-macks-stochastic-model)
3. [Bootstrap Methods](#3-bootstrap-methods)
4. [Bornhuetter-Ferguson Method](#4-bornhuetter-ferguson-method)
5. [Cape Cod Method](#5-cape-cod-method)
6. [Tail Factor Estimation](#6-tail-factor-estimation)
7. [Model Selection Framework](#7-model-selection-framework)
8. [Cross-Validation for Triangles](#8-cross-validation-for-triangles)
9. [Diagnostic Tests](#9-diagnostic-tests)
10. [Stress Testing](#10-stress-testing)
11. [References](#11-references)

---

## 1. Chain-Ladder Method

### 1.1 Overview
The Chain-Ladder (CL) method is the most widely used deterministic reserving technique. It assumes that claims develop proportionally across accident years.

### 1.2 Notation
- $C_{i,j}$ = Cumulative claims for accident year $i$ at development period $j$
- $n$ = Number of accident years
- $f_j$ = Age-to-age factor (link ratio) for period $j$

### 1.3 Age-to-Age Factors

Individual factors:
$$\hat{f}_{i,j} = \frac{C_{i,j+1}}{C_{i,j}}$$

Selected factor (volume-weighted average):
$$f_j = \frac{\sum_{i=1}^{n-j} C_{i,j+1}}{\sum_{i=1}^{n-j} C_{i,j}}$$

### 1.4 Cumulative Development Factors
$$F_j = \prod_{k=j}^{n-1} f_k$$

### 1.5 Ultimate Claims
$$\hat{C}_{i,\infty} = C_{i,n-i+1} \cdot F_{n-i+1}$$

### 1.6 Reserve
$$R_i = \hat{C}_{i,\infty} - C_{i,n-i+1}$$

### 1.7 Implementation
```python
# File: src/chain_ladder.py
class ChainLadder:
    def calculate_age_to_age_factors(self)
    def select_factors(self, method='volume_weighted')
    def calculate_cumulative_factors(self)
    def project_ultimate_losses(self)
```

**Reference:** Taylor, G. (2000). *Loss Reserving: An Actuarial Perspective*. Kluwer Academic Publishers.

---

## 2. Mack's Stochastic Model

### 2.1 Overview
Mack (1993) developed a distribution-free stochastic model that provides standard errors for Chain-Ladder estimates without requiring distributional assumptions.

### 2.2 Model Assumptions
1. $E[C_{i,j+1}|C_{i,1},...,C_{i,j}] = C_{i,j} \cdot f_j$ (expected development)
2. $Var[C_{i,j+1}|C_{i,1},...,C_{i,j}] = C_{i,j} \cdot \sigma_j^2$ (variance)
3. Accident years are independent

### 2.3 Variance Parameter Estimation
$$\hat{\sigma}_j^2 = \frac{1}{n-j-1} \sum_{i=1}^{n-j} C_{i,j} \left(\frac{C_{i,j+1}}{C_{i,j}} - f_j\right)^2$$

For the last period ($j = n-1$):
$$\hat{\sigma}_{n-1}^2 = \min\left(\frac{\sigma_{n-2}^4}{\sigma_{n-3}^2}, \min(\sigma_{n-3}^2, \sigma_{n-2}^2)\right)$$

### 2.4 Mean Square Error of Reserve
$$MSE(\hat{R}_i) = \hat{C}_{i,\infty}^2 \sum_{j=n-i+1}^{n-1} \frac{\hat{\sigma}_j^2}{\hat{f}_j^2} \left(\frac{1}{\hat{C}_{i,j}} + \frac{1}{\sum_{k=1}^{n-j} C_{k,j}}\right)$$

### 2.5 Standard Error
$$SE(\hat{R}_i) = \sqrt{MSE(\hat{R}_i)}$$

### 2.6 Confidence Intervals
Assuming log-normal distribution:
$$CI_{1-\alpha} = \left[\hat{R}_i \cdot e^{-z_{\alpha/2} \cdot \sigma_R}, \hat{R}_i \cdot e^{z_{\alpha/2} \cdot \sigma_R}\right]$$

where $\sigma_R = \sqrt{\log(1 + CV^2)}$ and $CV = SE/\hat{R}$

### 2.7 Implementation
```python
# File: src/stochastic_reserving/mack_model.py
class MackChainLadder:
    def fit(self)
    def _estimate_sigma(self)
    def _calculate_mse(self)
    def get_confidence_intervals(self, alpha=0.95)
```

**Reference:** Mack, T. (1993). *Distribution-free calculation of the standard error of chain ladder reserve estimates*. ASTIN Bulletin, 23(2), 213-225.

---

## 3. Bootstrap Methods

### 3.1 Overview
Bootstrap provides a non-parametric way to estimate the full distribution of reserves through simulation.

### 3.2 Residual Bootstrap Algorithm

**Step 1: Calculate Pearson Residuals**
$$r_{i,j} = \frac{C_{i,j+1} - C_{i,j} \cdot \hat{f}_j}{\sqrt{C_{i,j} \cdot \hat{\sigma}_j^2}}$$

**Step 2: Resample Residuals**
Sample with replacement from $\{r_{i,j}\}$ to get $\{r^*_{i,j}\}$

**Step 3: Reconstruct Triangle**
$$C^*_{i,j+1} = C^*_{i,j} \cdot \hat{f}_j + r^*_{i,j} \cdot \sqrt{C^*_{i,j} \cdot \hat{\sigma}_j^2}$$

**Step 4: Re-estimate Factors and Project**
Calculate new factors $\hat{f}^*_j$ and ultimate $\hat{C}^*_{i,\infty}$

**Step 5: Repeat**
Repeat steps 2-4 $B$ times (typically $B = 10,000$)

### 3.3 Over-Dispersed Poisson (ODP) Bootstrap

Assumes:
$$C_{i,j} \sim ODP(\mu_{i,j}, \phi)$$

where $\phi$ is the over-dispersion parameter:
$$\hat{\phi} = \frac{1}{n_{df}} \sum_{i,j} \frac{(C_{i,j} - \hat{\mu}_{i,j})^2}{\hat{\mu}_{i,j}}$$

### 3.4 Output Percentiles
From $B$ simulated reserves:
- P50 (median)
- P75
- P90
- P95 (common regulatory)
- P99

### 3.5 Implementation
```python
# File: src/stochastic_reserving/bootstrap.py
class BootstrapChainLadder:
    def fit(self)
    def _calculate_residuals(self)
    def _simulate_triangle(self)
    def get_reserve_distribution(self)
    def get_percentiles(self, percentiles=[50, 75, 90, 95, 99])
```

**Reference:** England, P. & Verrall, R. (2002). *Stochastic claims reserving in general insurance*. British Actuarial Journal, 8(3), 443-518.

---

## 4. Bornhuetter-Ferguson Method

### 4.1 Overview
The BF method combines Chain-Ladder with a priori expected losses, making it more stable for immature years.

### 4.2 Formula
$$\hat{C}_{i,\infty}^{BF} = C_{i,j} + ELR_i \cdot P_i \cdot (1 - \frac{1}{F_j})$$

where:
- $ELR_i$ = Expected Loss Ratio for accident year $i$
- $P_i$ = Earned Premium for accident year $i$
- $F_j$ = Cumulative development factor to ultimate
- $(1 - 1/F_j)$ = Percent unreported

### 4.3 Reserve
$$R_i^{BF} = ELR_i \cdot P_i \cdot (1 - \frac{1}{F_j})$$

### 4.4 Comparison with Chain-Ladder

| Aspect | Chain-Ladder | Bornhuetter-Ferguson |
|--------|--------------|---------------------|
| Weight on reported | 100% | $1/F_j$ |
| Weight on a priori | 0% | $1 - 1/F_j$ |
| Stability for immature years | Low | High |
| Responsiveness to data | High | Low |

### 4.5 ELR Estimation
When ELR is not provided, estimate from mature years:
$$\widehat{ELR} = \frac{\sum_{i: mature} C_{i,\infty}^{CL}}{\sum_{i: mature} P_i}$$

### 4.6 Implementation
```python
# File: src/alternative_methods/bornhuetter_ferguson.py
class BornhuetterFerguson:
    def fit(self)
    def _estimate_elr_from_mature_years(self)
    def _calculate_percent_reported(self)
    def get_comparison(self)  # vs Chain-Ladder
```

**Reference:** Bornhuetter, R. & Ferguson, R. (1972). *The Actuary and IBNR*. Proceedings of the Casualty Actuarial Society, 59, 181-195.

---

## 5. Cape Cod Method

### 5.1 Overview
The Cape Cod (Stanard-Bühlmann) method estimates an implicit Expected Loss Ratio from the data itself using "used-up" premium.

### 5.2 Used-Up Premium
$$P_i^{used} = P_i \cdot \frac{1}{F_j}$$

This represents the portion of premium "earned" for claims already reported.

### 5.3 Implicit ELR
$$\widehat{ELR}_{CC} = \frac{\sum_i C_{i,j}}{\sum_i P_i^{used}} = \frac{\sum_i C_{i,j}}{\sum_i P_i / F_j}$$

### 5.4 Ultimate Claims
$$\hat{C}_{i,\infty}^{CC} = C_{i,j} + \widehat{ELR}_{CC} \cdot P_i \cdot (1 - \frac{1}{F_j})$$

### 5.5 Key Properties
- Single ELR for all years (homogeneity assumption)
- Self-calibrating from triangle data
- Robust to individual year volatility

### 5.6 Generalized Cape Cod
Allows different weights or year-specific adjustments:
$$\widehat{ELR}_{GCC} = \frac{\sum_i w_i \cdot C_{i,j}}{\sum_i w_i \cdot P_i / F_j}$$

### 5.7 Implementation
```python
# File: src/alternative_methods/cape_cod.py
class CapeCod:
    def fit(self)
    def _calculate_used_up_premium(self)
    def _calculate_cape_cod_elr(self)
    def fit_generalized(self, weights)
```

**Reference:** Stanard, J. (1985). *A Simulation Test of Prediction Errors of Loss Reserve Estimation Techniques*. Proceedings of the CAS, 72, 124-148.

---

## 6. Tail Factor Estimation

### 6.1 Overview
Tail factors extrapolate development beyond the observed triangle to estimate ultimate losses.

### 6.2 Methods Implemented

#### 6.2.1 Exponential Decay
$$f(k) = 1 + a \cdot e^{-bk}$$

Linearization: $\log(f - 1) = \log(a) - bk$

#### 6.2.2 Inverse Power
$$f(k) = 1 + \frac{a}{k^b}$$

Linearization: $\log(f - 1) = \log(a) - b\log(k)$

#### 6.2.3 Weibull
$$f(k) = 1 + a \cdot e^{-(k/b)^c}$$

#### 6.2.4 Sherman Curve
$$f(k) = 1 + \frac{a}{b + k}$$

Linearization: $\frac{1}{f-1} = \frac{b}{a} + \frac{k}{a}$

#### 6.2.5 Bondy Method
$$Tail = f_{last}^{f_{last}/(f_{last}-1)}$$

Quick estimate based on last observed factor.

### 6.3 Model Selection
Use AIC (Akaike Information Criterion):
$$AIC = n \cdot \log(RSS/n) + 2k$$

where $k$ = number of parameters

### 6.4 Tail Factor Calculation
$$Tail = \prod_{j=n}^{\infty} f_j$$

In practice, extrapolate until $f_j < 1.0005$ (convergence).

### 6.5 Implementation
```python
# File: src/tail_fitting/tail_estimator.py
class TailEstimator:
    METHODS = ['exponential', 'inverse_power', 'weibull',
               'sherman', 'bondy', 'linear_decay', 'log_linear']

    def fit(self, methods=None)
    def get_comparison_table(self)
    def get_full_cumulative_factors(self)
```

**Reference:** Sherman, R. (1984). *Extrapolating, Smoothing and Interpolating Development Factors*. Proceedings of the CAS.

---

## 7. Model Selection Framework

### 7.1 Factor Estimation Methods

| Method | Formula | Properties |
|--------|---------|-----------|
| Simple Average | $\bar{f}_j = \frac{1}{n-j}\sum_{i=1}^{n-j} \hat{f}_{i,j}$ | Equal weight |
| Volume Weighted | $f_j = \frac{\sum C_{i,j+1}}{\sum C_{i,j}}$ | Larger years dominate |
| Medial | Exclude min/max, then average | Robust to outliers |
| Geometric | $f_j = \left(\prod \hat{f}_{i,j}\right)^{1/(n-j)}$ | Less sensitive to large values |
| Harmonic | $f_j = \frac{n-j}{\sum 1/\hat{f}_{i,j}}$ | Conservative |
| Regression | $C_{j+1} = \beta_j \cdot C_j$ (OLS) | Statistical framework |
| Exponential | $f_j = \sum w_i \hat{f}_{i,j}$, $w_i \propto e^{-\lambda(n-i)}$ | Recent years weighted |

### 7.2 Windowed Estimators
Apply estimation to a rolling window of $w$ years:
$$f_j^{(window)} = Estimator(C_{n-w+1,j}, ..., C_{n,j})$$

### 7.3 Validation
- **Holdout Validation**: Remove last diagonal, predict, compare
- **K-Fold Cross-Validation**: Multiple holdouts

### 7.4 Error Metrics
- RMSE: $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$
- MAE: $\frac{1}{n}\sum|y_i - \hat{y}_i|$
- MAPE: $\frac{100}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|$

### 7.5 Statistical Tests

#### Diebold-Mariano Test
Tests equal predictive accuracy:
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}$$
where $d_t = L(e_{1,t}) - L(e_{2,t})$ (loss differential)

#### Model Confidence Set (MCS)
Sequential elimination to find set of models not significantly worse than the best.

### 7.6 Implementation
```python
# File: src/model_selection/model_selector.py
class ModelSelector:
    def create_with_windowed_grid(triangle, min_window, max_window)
    def run_windowed_analysis(selection_criterion='RMSE')
    def get_model_confidence_set(alpha=0.10)
```

**Reference:** Diebold, F. & Mariano, R. (1995). *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics.

---

## 8. Cross-Validation for Triangles

### 8.1 Challenges
- Triangles have dependencies (rows, columns, diagonals)
- Standard random CV not appropriate

### 8.2 CV Strategies

#### Leave-One-Year-Out (LOYO)
Hold out one accident year, predict using others.

#### Grouped CV
Group consecutive years, hold out groups.

#### Diagonal CV
Hold out one calendar year (diagonal).

#### Time-Series CV
Expanding window: train on years 1..t, predict t+1.

### 8.3 Implementation
```python
# File: src/model_selection/kfold_validation.py
class KFoldTriangleValidator:
    def __init__(self, triangle, n_folds=5, strategy='loyo')
    def get_folds(self)
    def cross_validate(self, estimator)

class NestedCrossValidation:
    # For hyperparameter tuning
```

---

## 9. Diagnostic Tests

### 9.1 Calendar Year Effect Test
Tests if there's a systematic effect across diagonals:
$$H_0: \text{No calendar year effect}$$

Uses ANOVA on diagonal residuals.

### 9.2 Accident Year Effect Test
Tests for heterogeneity across accident years.

### 9.3 Independence Test
Tests if development factors are independent across periods.

### 9.4 Proportionality Test
Tests the key CL assumption:
$$H_0: E[C_{i,j+1}] = f_j \cdot C_{i,j}$$

### 9.5 Model Adequacy Score
Composite score (0-100%) based on:
- Number of failed tests
- Severity of violations
- Residual patterns

### 9.6 Implementation
```python
# File: src/diagnostics/diagnostic_tests.py
class DiagnosticTests:
    def run_all_tests(self)
    def test_calendar_year_effect(self)
    def test_proportionality(self)
    def get_model_adequacy_score(self)
```

**Reference:** Barnett, G. & Zehnwirth, B. (2000). *Best Estimates for Reserves*. Proceedings of the CAS.

---

## 10. Stress Testing

### 10.1 Scenarios

| Scenario | Description | Shock |
|----------|-------------|-------|
| Uniform +5% | All factors increased | $f_j \times 1.05$ |
| Uniform +10% | All factors increased | $f_j \times 1.10$ |
| Uniform +20% | Severe adverse | $f_j \times 1.20$ |
| Early Period | First half shocked | $f_{j<n/2} \times 1.15$ |
| Late Period | Second half shocked | $f_{j>n/2} \times 1.15$ |
| Tail Shock | Only tail factor | $Tail \times 1.50$ |

### 10.2 Regulatory Scenarios (Solvency II)
- 1-in-200 year event (99.5th percentile)
- Combined ratio stress
- Reserve risk capital

### 10.3 Sensitivity Analysis
$$\frac{\partial R}{\partial f_j} \approx \frac{R(f_j + \Delta) - R(f_j)}{\Delta}$$

### 10.4 Implementation
```python
# File: src/scenario_analysis/stress_testing.py
class StressTestFramework:
    def run_standard_scenarios(self)
    def run_regulatory_scenarios(self)
    def sensitivity_analysis(self)
    def get_summary_table(self)
```

---

## 11. References

### Primary References

1. **Mack, T. (1993)**. *Distribution-free calculation of the standard error of chain ladder reserve estimates*. ASTIN Bulletin, 23(2), 213-225.

2. **England, P. & Verrall, R. (2002)**. *Stochastic claims reserving in general insurance*. British Actuarial Journal, 8(3), 443-518.

3. **Bornhuetter, R. & Ferguson, R. (1972)**. *The Actuary and IBNR*. Proceedings of the CAS, 59, 181-195.

4. **Stanard, J. (1985)**. *A Simulation Test of Prediction Errors of Loss Reserve Estimation Techniques*. Proceedings of the CAS, 72, 124-148.

5. **Taylor, G. (2000)**. *Loss Reserving: An Actuarial Perspective*. Kluwer Academic Publishers.

### Additional References

6. **Sherman, R. (1984)**. *Extrapolating, Smoothing and Interpolating Development Factors*. Proceedings of the CAS.

7. **Bardis, E., Majidi, A., Murphy, D. (2012)**. *A Family of Chain-Ladder Factor Models*. Variance, 6(2).

8. **Diebold, F. & Mariano, R. (1995)**. *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics, 13(3), 253-263.

9. **Hansen, P., Lunde, A., Nason, J. (2011)**. *The Model Confidence Set*. Econometrica, 79(2), 453-497.

10. **Wüthrich, M. & Merz, M. (2008)**. *Stochastic Claims Reserving Methods in Insurance*. Wiley.

---

## Appendix: Statistical Functions

All statistical functions implemented natively in `src/utils/stats_utils.py`:

| Function | Description |
|----------|-------------|
| `norm_ppf(p)` | Normal inverse CDF |
| `norm_cdf(x)` | Normal CDF |
| `t_cdf(x, df)` | Student's t CDF |
| `linregress(x, y)` | Linear regression |
| `pearsonr(x, y)` | Pearson correlation |
| `f_oneway(*groups)` | One-way ANOVA |
| `levene(*groups)` | Levene's test |
| `shapiro_wilk(x)` | Shapiro-Wilk normality test |
| `jarque_bera(x)` | Jarque-Bera test |
| `skewness(x)` | Sample skewness |
| `kurtosis(x)` | Sample kurtosis |

---

*Document generated for the Actuarial Reserving Project*
*Swiss Re Property Reinsurance Analysis*
