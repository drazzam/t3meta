"""
Utility functions for T3-Meta.

This module provides statistical utilities and helper functions
used throughout the T3-Meta package.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats, optimize


# ============================================================================
# Statistical Utilities
# ============================================================================

def z_score(level: float = 0.95) -> float:
    """
    Get z-score for a confidence level.
    
    Args:
        level: Confidence level (0 to 1)
        
    Returns:
        z-score for two-tailed confidence interval
    """
    return stats.norm.ppf((1 + level) / 2)


def se_from_ci(
    ci_lower: float,
    ci_upper: float,
    level: float = 0.95,
    log_scale: bool = False
) -> float:
    """
    Compute standard error from confidence interval.
    
    Args:
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        level: Confidence level
        log_scale: Whether to use log scale
        
    Returns:
        Standard error
    """
    z = z_score(level)
    
    if log_scale:
        return (np.log(ci_upper) - np.log(ci_lower)) / (2 * z)
    else:
        return (ci_upper - ci_lower) / (2 * z)


def ci_from_se(
    estimate: float,
    se: float,
    level: float = 0.95,
    log_scale: bool = False
) -> Tuple[float, float]:
    """
    Compute confidence interval from standard error.
    
    Args:
        estimate: Point estimate
        se: Standard error
        level: Confidence level
        log_scale: Whether estimate is on log scale
        
    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    z = z_score(level)
    
    if log_scale:
        return (
            np.exp(estimate - z * se),
            np.exp(estimate + z * se)
        )
    else:
        return (
            estimate - z * se,
            estimate + z * se
        )


def p_value_from_z(z: float, two_tailed: bool = True) -> float:
    """
    Compute p-value from z-score.
    
    Args:
        z: z-score
        two_tailed: Use two-tailed test
        
    Returns:
        p-value
    """
    if two_tailed:
        return 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        return 1 - stats.norm.cdf(z)


def p_value_from_estimate(
    estimate: float,
    se: float,
    null_value: float = 0.0,
    two_tailed: bool = True
) -> float:
    """
    Compute p-value for testing estimate against null.
    
    Args:
        estimate: Point estimate
        se: Standard error
        null_value: Null hypothesis value
        two_tailed: Use two-tailed test
        
    Returns:
        p-value
    """
    z = (estimate - null_value) / se
    return p_value_from_z(z, two_tailed)


# ============================================================================
# Effect Size Conversions
# ============================================================================

def cohens_d_to_hedges_g(d: float, n: int) -> float:
    """
    Convert Cohen's d to Hedges' g (bias-corrected SMD).
    
    Args:
        d: Cohen's d
        n: Total sample size
        
    Returns:
        Hedges' g
    """
    # Hedges' correction factor
    j = 1 - 3 / (4 * n - 9)
    return d * j


def hedges_g_se(g: float, n1: int, n2: int) -> float:
    """
    Compute standard error of Hedges' g.
    
    Args:
        g: Hedges' g
        n1: Sample size group 1
        n2: Sample size group 2
        
    Returns:
        Standard error
    """
    n = n1 + n2
    j = 1 - 3 / (4 * n - 9)
    
    # Variance of d
    var_d = (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))
    
    # Variance of g
    var_g = j**2 * var_d
    
    return np.sqrt(var_g)


def odds_ratio_to_risk_ratio(
    or_value: float,
    baseline_risk: float
) -> float:
    """
    Convert odds ratio to risk ratio (Zhang-Yu formula).
    
    Args:
        or_value: Odds ratio
        baseline_risk: Baseline risk in control group (p0)
        
    Returns:
        Risk ratio
    """
    p0 = baseline_risk
    return or_value / (1 - p0 + p0 * or_value)


def risk_ratio_to_risk_difference(
    rr: float,
    baseline_risk: float
) -> float:
    """
    Convert risk ratio to risk difference.
    
    Args:
        rr: Risk ratio
        baseline_risk: Baseline risk in control group
        
    Returns:
        Risk difference
    """
    return baseline_risk * (rr - 1)


def hazard_ratio_to_risk(
    hr: float,
    baseline_risk: float,
    time: float = 1.0
) -> float:
    """
    Convert hazard ratio to cumulative incidence ratio at time t.
    
    Assumes exponential survival: S(t) = exp(-λt)
    
    Args:
        hr: Hazard ratio
        baseline_risk: Baseline cumulative incidence at time t
        time: Time point (for reference)
        
    Returns:
        Cumulative incidence in treatment group
    """
    # Baseline survival
    s0 = 1 - baseline_risk
    
    # Treatment survival (assuming proportional hazards)
    s1 = s0 ** hr
    
    return 1 - s1


def nnt_from_rd(rd: float) -> float:
    """
    Compute Number Needed to Treat from risk difference.
    
    Args:
        rd: Risk difference (absolute)
        
    Returns:
        NNT (or NNH if negative)
    """
    if rd == 0:
        return np.inf
    return 1 / rd


# ============================================================================
# Pooling Functions
# ============================================================================

def pooled_estimate_fixed(
    estimates: np.ndarray,
    variances: np.ndarray
) -> Tuple[float, float]:
    """
    Fixed-effect pooled estimate using inverse variance weighting.
    
    Args:
        estimates: Array of effect estimates
        variances: Array of variances
        
    Returns:
        Tuple of (pooled_estimate, pooled_variance)
    """
    weights = 1 / variances
    pooled = np.sum(weights * estimates) / np.sum(weights)
    pooled_var = 1 / np.sum(weights)
    
    return pooled, pooled_var


def pooled_estimate_random(
    estimates: np.ndarray,
    variances: np.ndarray,
    tau_squared: float
) -> Tuple[float, float]:
    """
    Random-effects pooled estimate.
    
    Args:
        estimates: Array of effect estimates
        variances: Array of variances
        tau_squared: Between-study variance
        
    Returns:
        Tuple of (pooled_estimate, pooled_variance)
    """
    total_var = variances + tau_squared
    weights = 1 / total_var
    pooled = np.sum(weights * estimates) / np.sum(weights)
    pooled_var = 1 / np.sum(weights)
    
    return pooled, pooled_var


def combine_estimates(
    estimate1: float,
    var1: float,
    estimate2: float,
    var2: float
) -> Tuple[float, float]:
    """
    Combine two independent estimates.
    
    Args:
        estimate1: First estimate
        var1: Variance of first estimate
        estimate2: Second estimate
        var2: Variance of second estimate
        
    Returns:
        Tuple of (combined_estimate, combined_variance)
    """
    w1 = 1 / var1
    w2 = 1 / var2
    
    combined = (w1 * estimate1 + w2 * estimate2) / (w1 + w2)
    combined_var = 1 / (w1 + w2)
    
    return combined, combined_var


# ============================================================================
# Matrix Operations
# ============================================================================

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add intercept column to design matrix."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean."""
    return np.sum(weights * values) / np.sum(weights)


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted variance."""
    mean = weighted_mean(values, weights)
    return np.sum(weights * (values - mean) ** 2) / np.sum(weights)


# ============================================================================
# Bayesian Utilities
# ============================================================================

def log_sum_exp(log_values: np.ndarray) -> float:
    """
    Compute log(sum(exp(log_values))) in a numerically stable way.
    
    Args:
        log_values: Array of log values
        
    Returns:
        log(sum(exp(log_values)))
    """
    max_val = np.max(log_values)
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def hpd_interval(
    samples: np.ndarray,
    level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Highest Posterior Density (HPD) interval.
    
    Args:
        samples: MCMC samples
        level: Credible level
        
    Returns:
        Tuple of (lower, upper) bounds
    """
    samples = np.sort(samples)
    n = len(samples)
    interval_size = int(np.ceil(level * n))
    
    # Find shortest interval
    min_width = np.inf
    best_interval = (samples[0], samples[-1])
    
    for i in range(n - interval_size):
        width = samples[i + interval_size - 1] - samples[i]
        if width < min_width:
            min_width = width
            best_interval = (samples[i], samples[i + interval_size - 1])
    
    return best_interval


def effective_sample_size(samples: np.ndarray) -> float:
    """
    Compute effective sample size accounting for autocorrelation.
    
    Args:
        samples: MCMC samples
        
    Returns:
        Effective sample size
    """
    n = len(samples)
    
    # Compute autocorrelation
    mean = np.mean(samples)
    var = np.var(samples)
    
    if var == 0:
        return float(n)
    
    # Autocorrelation function
    acf = np.correlate(samples - mean, samples - mean, mode='full')
    acf = acf[n-1:] / (var * n)
    
    # Sum until first negative autocorrelation
    rho_sum = 0
    for i in range(1, n):
        if acf[i] < 0:
            break
        rho_sum += acf[i]
    
    ess = n / (1 + 2 * rho_sum)
    return max(1, ess)


def gelman_rubin(chains: List[np.ndarray]) -> float:
    """
    Compute Gelman-Rubin convergence diagnostic (R-hat).
    
    Args:
        chains: List of MCMC chains
        
    Returns:
        R-hat statistic (< 1.1 indicates convergence)
    """
    m = len(chains)  # Number of chains
    n = len(chains[0])  # Length of each chain
    
    # Chain means
    chain_means = [np.mean(c) for c in chains]
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n / (m - 1) * np.sum([(cm - overall_mean) ** 2 for cm in chain_means])
    
    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains])
    
    # Pooled variance estimate
    var_est = (n - 1) / n * W + 1 / n * B
    
    # R-hat
    rhat = np.sqrt(var_est / W)
    
    return rhat


# ============================================================================
# Validation Utilities
# ============================================================================

def is_valid_effect_measure(measure: str) -> bool:
    """Check if effect measure string is valid."""
    valid = {"HR", "RR", "OR", "IRR", "RD", "MD", "SMD", "RMST", "MSR"}
    return measure.upper() in valid


def is_ratio_measure(measure: str) -> bool:
    """Check if effect measure is a ratio (requires log transformation)."""
    ratio_measures = {"HR", "RR", "OR", "IRR", "MSR"}
    return measure.upper() in ratio_measures


def validate_probability(p: float, name: str = "probability") -> None:
    """Validate that value is a valid probability."""
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {p}")


def validate_positive(x: float, name: str = "value") -> None:
    """Validate that value is positive."""
    if x <= 0:
        raise ValueError(f"{name} must be positive, got {x}")


def validate_non_negative(x: float, name: str = "value") -> None:
    """Validate that value is non-negative."""
    if x < 0:
        raise ValueError(f"{name} must be non-negative, got {x}")


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_estimate(
    estimate: float,
    se: Optional[float] = None,
    ci: Optional[Tuple[float, float]] = None,
    decimals: int = 3,
    exponentiate: bool = False
) -> str:
    """
    Format estimate with optional SE or CI.
    
    Args:
        estimate: Point estimate
        se: Standard error (optional)
        ci: Confidence interval (optional)
        decimals: Number of decimal places
        exponentiate: Whether to exponentiate
        
    Returns:
        Formatted string
    """
    if exponentiate:
        estimate = np.exp(estimate)
        if ci:
            ci = (np.exp(ci[0]), np.exp(ci[1]))
    
    result = f"{estimate:.{decimals}f}"
    
    if se is not None:
        result += f" (SE: {se:.{decimals}f})"
    
    if ci is not None:
        result += f" [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"
    
    return result


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """Format p-value with appropriate precision."""
    if p < threshold:
        return f"p < {threshold}"
    return f"p = {p:.3f}"
