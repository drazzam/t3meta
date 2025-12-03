"""
Influence Analysis for T3-Meta.

This module provides tools for assessing the influence of individual
studies on the meta-analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from scipy import stats


@dataclass
class InfluenceAnalysis:
    """
    Influence diagnostics for T3-Meta.
    
    Provides methods to assess how individual studies affect
    the overall meta-analysis results.
    
    Attributes:
        y: Effect estimates
        se: Standard errors
        X: Design covariate matrix (optional)
        study_names: Names of studies
    """
    
    y: np.ndarray
    se: np.ndarray
    X: Optional[np.ndarray] = None
    study_names: Optional[List[str]] = None
    
    def __post_init__(self):
        self.y = np.asarray(self.y).flatten()
        self.se = np.asarray(self.se).flatten()
        self.n = len(self.y)
        
        if self.study_names is None:
            self.study_names = [f"Study_{i+1}" for i in range(self.n)]
        
        if self.X is not None:
            self.X = np.asarray(self.X)
    
    def leave_one_out(
        self,
        fit_func: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Leave-one-out analysis.
        
        Args:
            fit_func: Function that takes (y, se, X) and returns (estimate, se)
            
        Returns:
            Dictionary with LOO results for each study
        """
        results = {}
        
        for i in range(self.n):
            # Create LOO datasets
            mask = np.ones(self.n, dtype=bool)
            mask[i] = False
            
            y_loo = self.y[mask]
            se_loo = self.se[mask]
            X_loo = self.X[mask] if self.X is not None else None
            
            # Fit without study i
            try:
                est, est_se = fit_func(y_loo, se_loo, X_loo)
                results[self.study_names[i]] = {
                    "estimate": float(est),
                    "se": float(est_se),
                    "excluded_study": self.study_names[i],
                    "excluded_effect": float(self.y[i]),
                }
            except Exception as e:
                results[self.study_names[i]] = {
                    "estimate": np.nan,
                    "se": np.nan,
                    "excluded_study": self.study_names[i],
                    "error": str(e)
                }
        
        return results
    
    def dfbetas(
        self,
        beta_full: np.ndarray,
        vcov_full: np.ndarray
    ) -> np.ndarray:
        """
        Compute DFBETAS influence measures.
        
        DFBETAS measures how much each coefficient changes when
        a study is removed, scaled by its standard error.
        
        Args:
            beta_full: Full model coefficients
            vcov_full: Full model variance-covariance matrix
            
        Returns:
            Matrix of DFBETAS (n_studies x n_coefficients)
        """
        if self.X is None:
            raise ValueError("Design matrix X required for DFBETAS")
        
        p = len(beta_full)
        dfbetas = np.zeros((self.n, p))
        
        variances = self.se ** 2
        
        for i in range(self.n):
            mask = np.ones(self.n, dtype=bool)
            mask[i] = False
            
            y_loo = self.y[mask]
            se_loo = self.se[mask]
            X_loo = self.X[mask]
            var_loo = variances[mask]
            
            # Fit LOO model
            W = np.diag(1 / var_loo)
            XtWX = X_loo.T @ W @ X_loo
            
            try:
                XtWX_inv = np.linalg.inv(XtWX)
                beta_loo = XtWX_inv @ X_loo.T @ W @ y_loo
            except np.linalg.LinAlgError:
                beta_loo = beta_full
            
            # DFBETAS
            se_beta = np.sqrt(np.diag(vcov_full))
            dfbetas[i] = (beta_full - beta_loo) / se_beta
        
        return dfbetas
    
    def cooks_distance(
        self,
        fitted: np.ndarray,
        tau_squared: float = 0.0
    ) -> np.ndarray:
        """
        Compute Cook's distance for each study.
        
        Args:
            fitted: Fitted values from the model
            tau_squared: Between-study variance
            
        Returns:
            Array of Cook's distances
        """
        variances = self.se ** 2
        total_var = variances + tau_squared
        weights = 1 / total_var
        
        residuals = self.y - fitted
        
        # Leverage (hat values)
        if self.X is not None:
            W = np.diag(weights)
            H = self.X @ np.linalg.pinv(self.X.T @ W @ self.X) @ self.X.T @ W
            leverage = np.diag(H)
        else:
            leverage = weights / np.sum(weights)
        
        # Standardized residuals
        std_res = residuals / np.sqrt(total_var * (1 - leverage + 1e-10))
        
        # Cook's distance
        p = self.X.shape[1] if self.X is not None else 1
        cooks_d = (std_res ** 2 / p) * (leverage / (1 - leverage + 1e-10))
        
        return cooks_d
    
    def dffits(
        self,
        fitted: np.ndarray,
        tau_squared: float = 0.0
    ) -> np.ndarray:
        """
        Compute DFFITS for each study.
        
        Args:
            fitted: Fitted values
            tau_squared: Between-study variance
            
        Returns:
            Array of DFFITS values
        """
        variances = self.se ** 2
        total_var = variances + tau_squared
        weights = 1 / total_var
        
        residuals = self.y - fitted
        
        if self.X is not None:
            W = np.diag(weights)
            H = self.X @ np.linalg.pinv(self.X.T @ W @ self.X) @ self.X.T @ W
            leverage = np.diag(H)
        else:
            leverage = weights / np.sum(weights)
        
        # DFFITS = (y_i - fitted_i) * sqrt(h_i / (1-h_i)) / s_i
        dffits = residuals * np.sqrt(leverage / (1 - leverage + 1e-10)) / np.sqrt(total_var)
        
        return dffits
    
    def hat_values(
        self,
        tau_squared: float = 0.0
    ) -> np.ndarray:
        """
        Compute leverage (hat) values for each study.
        
        Args:
            tau_squared: Between-study variance
            
        Returns:
            Array of hat values
        """
        variances = self.se ** 2
        total_var = variances + tau_squared
        weights = 1 / total_var
        
        if self.X is not None:
            W = np.diag(weights)
            try:
                XtWX_inv = np.linalg.inv(self.X.T @ W @ self.X)
            except np.linalg.LinAlgError:
                XtWX_inv = np.linalg.pinv(self.X.T @ W @ self.X)
            H = self.X @ XtWX_inv @ self.X.T @ W
            leverage = np.diag(H)
        else:
            leverage = weights / np.sum(weights)
        
        return leverage
    
    def studentized_residuals(
        self,
        fitted: np.ndarray,
        tau_squared: float = 0.0
    ) -> np.ndarray:
        """
        Compute studentized residuals.
        
        Args:
            fitted: Fitted values
            tau_squared: Between-study variance
            
        Returns:
            Array of studentized residuals
        """
        variances = self.se ** 2
        total_var = variances + tau_squared
        
        residuals = self.y - fitted
        leverage = self.hat_values(tau_squared)
        
        # Internally studentized residuals
        student_res = residuals / np.sqrt(total_var * (1 - leverage + 1e-10))
        
        return student_res
    
    def externally_studentized_residuals(
        self,
        fitted: np.ndarray,
        tau_squared: float = 0.0
    ) -> np.ndarray:
        """
        Compute externally studentized (jackknife) residuals.
        
        Args:
            fitted: Fitted values
            tau_squared: Between-study variance
            
        Returns:
            Array of externally studentized residuals
        """
        internal = self.studentized_residuals(fitted, tau_squared)
        leverage = self.hat_values(tau_squared)
        n = self.n
        p = self.X.shape[1] if self.X is not None else 1
        
        # External studentization
        df = n - p - 1
        if df <= 0:
            return internal
        
        external = internal * np.sqrt((df) / (df + 1 - internal ** 2 + 1e-10))
        
        return external
    
    def summary(
        self,
        fitted: np.ndarray,
        tau_squared: float = 0.0,
        threshold_cooks: float = 4.0,
        threshold_residual: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive influence summary.
        
        Args:
            fitted: Fitted values
            tau_squared: Between-study variance
            threshold_cooks: Threshold for flagging Cook's distance
            threshold_residual: Threshold for flagging residuals
            
        Returns:
            Dictionary with influence diagnostics
        """
        cooks = self.cooks_distance(fitted, tau_squared)
        dffits = self.dffits(fitted, tau_squared)
        leverage = self.hat_values(tau_squared)
        std_res = self.studentized_residuals(fitted, tau_squared)
        
        n = self.n
        p = self.X.shape[1] if self.X is not None else 1
        
        # Thresholds
        cooks_threshold = threshold_cooks / n
        dffits_threshold = 2 * np.sqrt(p / n)
        leverage_threshold = 2 * p / n
        
        # Identify influential studies
        influential_cooks = np.where(cooks > cooks_threshold)[0]
        influential_dffits = np.where(np.abs(dffits) > dffits_threshold)[0]
        influential_leverage = np.where(leverage > leverage_threshold)[0]
        outliers = np.where(np.abs(std_res) > threshold_residual)[0]
        
        all_influential = set(influential_cooks) | set(influential_dffits) | set(influential_leverage)
        
        return {
            "cooks_distance": cooks,
            "dffits": dffits,
            "leverage": leverage,
            "studentized_residuals": std_res,
            "thresholds": {
                "cooks": cooks_threshold,
                "dffits": dffits_threshold,
                "leverage": leverage_threshold,
                "residual": threshold_residual
            },
            "influential_studies": {
                "by_cooks": [self.study_names[i] for i in influential_cooks],
                "by_dffits": [self.study_names[i] for i in influential_dffits],
                "by_leverage": [self.study_names[i] for i in influential_leverage],
                "outliers": [self.study_names[i] for i in outliers],
                "any": [self.study_names[i] for i in all_influential]
            }
        }


def leave_one_out(
    y: np.ndarray,
    se: np.ndarray,
    method: str = "DL"
) -> Dict[str, Dict[str, float]]:
    """
    Simple leave-one-out analysis for random-effects meta-analysis.
    
    Args:
        y: Effect estimates
        se: Standard errors
        method: Tau-squared estimation method
        
    Returns:
        Dictionary with LOO results
    """
    from t3meta.diagnostics.heterogeneity import compute_tau_squared
    
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    variances = se ** 2
    
    results = {}
    
    # Full analysis
    tau_sq_full = compute_tau_squared(y, se, method)
    w_full = 1 / (variances + tau_sq_full)
    theta_full = np.sum(w_full * y) / np.sum(w_full)
    se_full = 1 / np.sqrt(np.sum(w_full))
    
    results["full"] = {
        "estimate": float(theta_full),
        "se": float(se_full),
        "tau_squared": float(tau_sq_full)
    }
    
    # LOO
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        y_loo = y[mask]
        se_loo = se[mask]
        var_loo = variances[mask]
        
        tau_sq_loo = compute_tau_squared(y_loo, se_loo, method)
        w_loo = 1 / (var_loo + tau_sq_loo)
        theta_loo = np.sum(w_loo * y_loo) / np.sum(w_loo)
        se_loo = 1 / np.sqrt(np.sum(w_loo))
        
        results[f"excluding_{i}"] = {
            "estimate": float(theta_loo),
            "se": float(se_loo),
            "tau_squared": float(tau_sq_loo),
            "change_estimate": float(theta_full - theta_loo),
            "excluded_index": i
        }
    
    return results


def cumulative_meta_analysis(
    y: np.ndarray,
    se: np.ndarray,
    order: Optional[np.ndarray] = None,
    method: str = "DL"
) -> Dict[str, List[Dict[str, float]]]:
    """
    Cumulative meta-analysis (adding studies one at a time).
    
    Args:
        y: Effect estimates
        se: Standard errors
        order: Order in which to add studies (default: as given)
        method: Tau-squared estimation method
        
    Returns:
        Dictionary with cumulative results
    """
    from t3meta.diagnostics.heterogeneity import compute_tau_squared
    
    y = np.asarray(y).flatten()
    se = np.asarray(se).flatten()
    n = len(y)
    
    if order is None:
        order = np.arange(n)
    
    results = []
    
    for k in range(2, n + 1):
        indices = order[:k]
        y_cum = y[indices]
        se_cum = se[indices]
        var_cum = se_cum ** 2
        
        tau_sq = compute_tau_squared(y_cum, se_cum, method)
        w = 1 / (var_cum + tau_sq)
        theta = np.sum(w * y_cum) / np.sum(w)
        se_theta = 1 / np.sqrt(np.sum(w))
        
        results.append({
            "n_studies": k,
            "estimate": float(theta),
            "se": float(se_theta),
            "ci_lower": float(theta - 1.96 * se_theta),
            "ci_upper": float(theta + 1.96 * se_theta),
            "tau_squared": float(tau_sq),
            "last_added_index": int(order[k-1])
        })
    
    return {"cumulative_results": results}
