"""
Heterogeneity Analysis for T3-Meta.

This module provides tools for analyzing and decomposing heterogeneity
in meta-analysis, with specific focus on separating design-related bias
from true clinical heterogeneity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from scipy import stats


def compute_tau_squared(
    y: np.ndarray,
    se: np.ndarray,
    method: str = "DL"
) -> float:
    """
    Estimate between-study variance (tau-squared).
    
    Args:
        y: Effect estimates
        se: Standard errors
        method: Estimation method ('DL', 'PM', 'REML', 'ML', 'HS', 'SJ')
        
    Returns:
        Estimated tau-squared
    """
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    variances = se ** 2
    weights = 1 / variances
    
    # Fixed-effect pooled estimate
    theta_fe = np.sum(weights * y) / np.sum(weights)
    
    # Cochran's Q
    q = np.sum(weights * (y - theta_fe) ** 2)
    
    if method == "DL":
        # DerSimonian-Laird
        c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau_sq = max(0, (q - (n - 1)) / c)
        
    elif method == "PM":
        # Paule-Mandel
        from scipy.optimize import brentq
        
        def pm_eq(tau_sq):
            w = 1 / (variances + tau_sq)
            theta = np.sum(w * y) / np.sum(w)
            return np.sum(w * (y - theta) ** 2) - (n - 1)
        
        if pm_eq(0) <= 0:
            tau_sq = 0.0
        else:
            try:
                tau_sq = brentq(pm_eq, 0, 10 * q / n)
            except ValueError:
                tau_sq = max(0, (q - (n - 1)) / np.sum(weights))
                
    elif method == "HS":
        # Hunter-Schmidt
        theta_fe = np.mean(y)
        var_obs = np.var(y, ddof=1)
        var_err = np.mean(variances)
        tau_sq = max(0, var_obs - var_err)
        
    elif method == "SJ":
        # Sidik-Jonkman
        theta_0 = np.mean(y)
        tau_sq_0 = np.sum((y - theta_0) ** 2) / (n - 1)
        
        w = 1 / (variances + tau_sq_0)
        theta_1 = np.sum(w * y) / np.sum(w)
        
        tau_sq = np.sum((y - theta_1) ** 2 / (variances / tau_sq_0 + 1)) / (n - 1)
        tau_sq = max(0, tau_sq)
        
    else:
        # Default to DL
        c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau_sq = max(0, (q - (n - 1)) / c)
    
    return tau_sq


def compute_i_squared(
    y: np.ndarray,
    se: np.ndarray,
    tau_squared: Optional[float] = None
) -> float:
    """
    Compute I-squared statistic (percentage of variance due to heterogeneity).
    
    Args:
        y: Effect estimates
        se: Standard errors
        tau_squared: Pre-computed tau-squared (optional)
        
    Returns:
        I-squared value (0 to 1)
    """
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    variances = se ** 2
    weights = 1 / variances
    
    # Pooled estimate
    theta = np.sum(weights * y) / np.sum(weights)
    
    # Cochran's Q
    q = np.sum(weights * (y - theta) ** 2)
    df = n - 1
    
    if q <= df:
        return 0.0
    
    return (q - df) / q


def compute_h_squared(
    y: np.ndarray,
    se: np.ndarray
) -> float:
    """
    Compute H-squared statistic.
    
    H² = Q / (k-1) where k is number of studies.
    
    Args:
        y: Effect estimates
        se: Standard errors
        
    Returns:
        H-squared value
    """
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    weights = 1 / (se ** 2)
    
    theta = np.sum(weights * y) / np.sum(weights)
    q = np.sum(weights * (y - theta) ** 2)
    
    return max(1, q / (n - 1))


def cochran_q_test(
    y: np.ndarray,
    se: np.ndarray
) -> Dict[str, float]:
    """
    Perform Cochran's Q test for heterogeneity.
    
    Args:
        y: Effect estimates
        se: Standard errors
        
    Returns:
        Dictionary with Q statistic, degrees of freedom, and p-value
    """
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    weights = 1 / (se ** 2)
    
    theta = np.sum(weights * y) / np.sum(weights)
    q = np.sum(weights * (y - theta) ** 2)
    df = n - 1
    pvalue = 1 - stats.chi2.cdf(q, df)
    
    return {
        "Q": float(q),
        "df": int(df),
        "p_value": float(pvalue)
    }


@dataclass
class HeterogeneityAnalysis:
    """
    Comprehensive heterogeneity analysis for T3-Meta.
    
    Provides methods for decomposing heterogeneity into components
    explained by design features vs. residual heterogeneity.
    
    Attributes:
        y: Effect estimates
        se: Standard errors
        X: Design covariate matrix (optional)
        feature_names: Names of design features
    """
    
    y: np.ndarray
    se: np.ndarray
    X: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    
    def __post_init__(self):
        self.y = np.asarray(self.y).flatten()
        self.se = np.asarray(self.se).flatten()
        if self.X is not None:
            self.X = np.asarray(self.X)
    
    @property
    def n_studies(self) -> int:
        return len(self.y)
    
    def total_heterogeneity(self, method: str = "DL") -> Dict[str, float]:
        """
        Compute total heterogeneity statistics.
        
        Returns:
            Dictionary with tau_squared, tau, I_squared, H_squared, Q test
        """
        tau_sq = compute_tau_squared(self.y, self.se, method)
        i_sq = compute_i_squared(self.y, self.se, tau_sq)
        h_sq = compute_h_squared(self.y, self.se)
        q_test = cochran_q_test(self.y, self.se)
        
        return {
            "tau_squared": tau_sq,
            "tau": np.sqrt(tau_sq),
            "I_squared": i_sq,
            "H_squared": h_sq,
            **q_test
        }
    
    def residual_heterogeneity(
        self,
        fitted: np.ndarray,
        n_params: int,
        method: str = "DL"
    ) -> Dict[str, float]:
        """
        Compute residual heterogeneity after adjusting for covariates.
        
        Args:
            fitted: Fitted values from meta-regression
            n_params: Number of parameters in the model
            method: Tau-squared estimation method
            
        Returns:
            Dictionary with residual heterogeneity statistics
        """
        residuals = self.y - fitted
        variances = self.se ** 2
        weights = 1 / variances
        
        # Residual Q
        q_res = np.sum(weights * residuals ** 2)
        df_res = self.n_studies - n_params
        
        if df_res <= 0:
            return {
                "tau_squared_residual": 0.0,
                "tau_residual": 0.0,
                "I_squared_residual": 0.0,
                "Q_residual": float(q_res),
                "df_residual": int(df_res),
                "p_value_residual": np.nan
            }
        
        # Residual tau-squared (DL-style)
        c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau_sq_res = max(0, (q_res - df_res) / c)
        
        # Residual I-squared
        i_sq_res = max(0, (q_res - df_res) / q_res) if q_res > 0 else 0
        
        p_val = 1 - stats.chi2.cdf(q_res, df_res)
        
        return {
            "tau_squared_residual": tau_sq_res,
            "tau_residual": np.sqrt(tau_sq_res),
            "I_squared_residual": i_sq_res,
            "Q_residual": float(q_res),
            "df_residual": int(df_res),
            "p_value_residual": float(p_val)
        }
    
    def r_squared(
        self,
        tau_sq_total: float,
        tau_sq_residual: float
    ) -> float:
        """
        Compute R-squared (proportion of heterogeneity explained).
        
        Args:
            tau_sq_total: Total tau-squared (without covariates)
            tau_sq_residual: Residual tau-squared (with covariates)
            
        Returns:
            R-squared value (0 to 1)
        """
        if tau_sq_total <= 0:
            return 0.0
        
        return max(0, 1 - tau_sq_residual / tau_sq_total)
    
    def prediction_interval(
        self,
        theta: float,
        se_theta: float,
        tau_squared: float,
        level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute prediction interval for a new study.
        
        Args:
            theta: Pooled effect estimate
            se_theta: Standard error of pooled estimate
            tau_squared: Between-study variance
            level: Confidence level
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        # Prediction variance includes both estimation uncertainty and tau
        pred_var = se_theta ** 2 + tau_squared
        pred_se = np.sqrt(pred_var)
        
        # Use t-distribution with k-2 df (Higgins et al.)
        df = max(1, self.n_studies - 2)
        t_crit = stats.t.ppf((1 + level) / 2, df)
        
        lower = theta - t_crit * pred_se
        upper = theta + t_crit * pred_se
        
        return lower, upper
    
    def outlier_detection(
        self,
        theta: Optional[float] = None,
        tau_squared: Optional[float] = None,
        threshold: float = 2.0
    ) -> List[int]:
        """
        Detect potential outlier studies.
        
        Args:
            theta: Pooled estimate (computed if not provided)
            tau_squared: Tau-squared (computed if not provided)
            threshold: Number of SDs for outlier detection
            
        Returns:
            Indices of potential outlier studies
        """
        if theta is None:
            weights = 1 / (self.se ** 2)
            theta = np.sum(weights * self.y) / np.sum(weights)
        
        if tau_squared is None:
            tau_squared = compute_tau_squared(self.y, self.se)
        
        # Studentized residuals
        total_var = self.se ** 2 + tau_squared
        residuals = self.y - theta
        std_residuals = residuals / np.sqrt(total_var)
        
        outliers = np.where(np.abs(std_residuals) > threshold)[0]
        
        return outliers.tolist()


def decompose_heterogeneity(
    y: np.ndarray,
    se: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    tau_squared_total: Optional[float] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Decompose heterogeneity into components.
    
    Separates total heterogeneity into:
    - Explained by design features (systematic bias)
    - Residual heterogeneity (true clinical variation + unexplained)
    
    Args:
        y: Effect estimates
        se: Standard errors
        X: Design covariate matrix
        beta: Fitted coefficients
        tau_squared_total: Total tau-squared (computed if not provided)
        feature_names: Names of design features
        
    Returns:
        Dictionary with decomposition results
    """
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    X = np.asarray(X)
    beta = np.asarray(beta).flatten()
    
    n = len(y)
    p = X.shape[1]
    variances = se ** 2
    weights = 1 / variances
    
    # Total heterogeneity
    if tau_squared_total is None:
        tau_squared_total = compute_tau_squared(y, se)
    
    # Fitted values
    fitted = X @ beta
    
    # Systematic bias component (variance of X @ beta)
    bias_variance = np.var(fitted, ddof=0)
    
    # Residual heterogeneity
    residuals = y - fitted
    q_res = np.sum(weights * residuals ** 2)
    df_res = n - p
    c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau_squared_residual = max(0, (q_res - df_res) / c) if df_res > 0 else 0
    
    # R-squared
    if tau_squared_total > 0:
        r_squared = 1 - tau_squared_residual / tau_squared_total
        r_squared = max(0, min(1, r_squared))
    else:
        r_squared = 0.0
    
    # Proportion of total variance explained by each feature
    feature_contributions = {}
    if feature_names is not None and len(feature_names) == p - 1:
        for i, name in enumerate(feature_names):
            # Contribution = variance explained by this feature
            x_i = X[:, i + 1]  # Skip intercept
            contrib_i = beta[i + 1] ** 2 * np.var(x_i, ddof=0)
            feature_contributions[name] = {
                "coefficient": float(beta[i + 1]),
                "variance_contribution": float(contrib_i),
                "proportion_of_explained": float(contrib_i / bias_variance) if bias_variance > 0 else 0
            }
    
    # I-squared components
    total_q = np.sum(weights * (y - np.mean(y)) ** 2)
    i_squared_total = max(0, (total_q - (n - 1)) / total_q) if total_q > 0 else 0
    i_squared_residual = max(0, (q_res - df_res) / q_res) if q_res > 0 and df_res > 0 else 0
    i_squared_explained = i_squared_total - i_squared_residual
    
    return {
        "tau_squared_total": tau_squared_total,
        "tau_squared_explained": bias_variance,
        "tau_squared_residual": tau_squared_residual,
        "r_squared": r_squared,
        "i_squared_total": i_squared_total,
        "i_squared_explained": i_squared_explained,
        "i_squared_residual": i_squared_residual,
        "feature_contributions": feature_contributions,
        "q_residual": q_res,
        "df_residual": df_res,
        "p_value_residual": 1 - stats.chi2.cdf(q_res, df_res) if df_res > 0 else np.nan
    }
