"""
Sensitivity Analysis for T3-Meta.

This module provides tools for conducting sensitivity analyses
to assess the robustness of meta-analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
import numpy as np
from scipy import stats


@dataclass
class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for T3-Meta.
    
    Provides methods to assess robustness of results under
    different modeling assumptions, prior specifications,
    and data perturbations.
    
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
        self.se = np.asarray(se).flatten() if isinstance(se, np.ndarray) else np.asarray(self.se).flatten()
        self.n = len(self.y)
        
        if self.study_names is None:
            self.study_names = [f"Study_{i+1}" for i in range(self.n)]
    
    def vary_tau(
        self,
        tau_values: List[float],
        fit_func: Optional[Callable] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Sensitivity to different tau (heterogeneity) values.
        
        Args:
            tau_values: List of tau values to evaluate
            fit_func: Optional fitting function
            
        Returns:
            Dictionary with results for each tau value
        """
        results = []
        variances = self.se ** 2
        
        for tau in tau_values:
            tau_sq = tau ** 2
            total_var = variances + tau_sq
            weights = 1 / total_var
            
            # Random-effects estimate
            theta = np.sum(weights * self.y) / np.sum(weights)
            se_theta = 1 / np.sqrt(np.sum(weights))
            
            results.append({
                "tau": float(tau),
                "tau_squared": float(tau_sq),
                "estimate": float(theta),
                "se": float(se_theta),
                "ci_lower": float(theta - 1.96 * se_theta),
                "ci_upper": float(theta + 1.96 * se_theta),
                "weights": weights.tolist()
            })
        
        return {"tau_sensitivity": results}
    
    def vary_prior_mean(
        self,
        prior_means: List[float],
        prior_sd: float = 1.0
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Sensitivity to prior mean specification (Bayesian).
        
        Args:
            prior_means: List of prior means to evaluate
            prior_sd: Prior standard deviation
            
        Returns:
            Dictionary with results for each prior mean
        """
        from t3meta.diagnostics.heterogeneity import compute_tau_squared
        
        results = []
        variances = self.se ** 2
        tau_sq = compute_tau_squared(self.y, self.se)
        
        for prior_mean in prior_means:
            # Simplified Bayesian posterior with normal prior
            prior_precision = 1 / (prior_sd ** 2)
            total_var = variances + tau_sq
            data_precision = np.sum(1 / total_var)
            
            # Posterior mean is weighted average
            data_weighted_mean = np.sum(self.y / total_var) / data_precision
            
            post_precision = prior_precision + data_precision
            post_mean = (prior_precision * prior_mean + data_precision * data_weighted_mean) / post_precision
            post_sd = 1 / np.sqrt(post_precision)
            
            results.append({
                "prior_mean": float(prior_mean),
                "prior_sd": float(prior_sd),
                "posterior_mean": float(post_mean),
                "posterior_sd": float(post_sd),
                "ci_lower": float(post_mean - 1.96 * post_sd),
                "ci_upper": float(post_mean + 1.96 * post_sd),
                "shrinkage": float(1 - data_precision / post_precision)
            })
        
        return {"prior_mean_sensitivity": results}
    
    def trim_and_fill(
        self,
        side: str = "right",
        estimator: str = "L0"
    ) -> Dict[str, Any]:
        """
        Trim-and-fill analysis for publication bias.
        
        Args:
            side: Which side to check for asymmetry ('left', 'right', 'auto')
            estimator: Method for estimating missing studies ('L0', 'R0', 'Q0')
            
        Returns:
            Dictionary with trim-and-fill results
        """
        from t3meta.diagnostics.heterogeneity import compute_tau_squared
        
        # Sort by effect size
        order = np.argsort(self.y)
        y_sorted = self.y[order]
        se_sorted = self.se[order]
        
        # Estimate tau
        tau_sq = compute_tau_squared(y_sorted, se_sorted)
        total_var = se_sorted ** 2 + tau_sq
        weights = 1 / total_var
        
        # Pooled estimate
        theta = np.sum(weights * y_sorted) / np.sum(weights)
        
        # Centered effects
        centered = y_sorted - theta
        
        # Determine side
        if side == "auto":
            # Check which side has more extreme values
            left_extreme = np.sum(centered < -np.std(centered))
            right_extreme = np.sum(centered > np.std(centered))
            side = "right" if right_extreme > left_extreme else "left"
        
        # Estimate number of missing studies (L0 estimator)
        if side == "right":
            ranks = np.arange(1, self.n + 1)
            T_stat = np.sum(ranks * (centered > 0))
        else:
            ranks = np.arange(self.n, 0, -1)
            T_stat = np.sum(ranks * (centered < 0))
        
        # Expected value under no asymmetry
        E_T = self.n * (self.n + 1) / 4
        Var_T = self.n * (self.n + 1) * (2 * self.n + 1) / 24
        
        # Number of missing studies
        k0 = max(0, round((T_stat - E_T) / np.sqrt(Var_T)))
        
        # Create imputed studies
        if k0 > 0:
            # Mirror the k0 most extreme studies
            if side == "right":
                extreme_idx = order[:k0]
            else:
                extreme_idx = order[-k0:]
            
            imputed_y = 2 * theta - self.y[extreme_idx]
            imputed_se = self.se[extreme_idx]
            
            # Combined analysis
            y_combined = np.concatenate([self.y, imputed_y])
            se_combined = np.concatenate([self.se, imputed_se])
            
            tau_sq_adj = compute_tau_squared(y_combined, se_combined)
            var_combined = se_combined ** 2 + tau_sq_adj
            w_combined = 1 / var_combined
            
            theta_adj = np.sum(w_combined * y_combined) / np.sum(w_combined)
            se_adj = 1 / np.sqrt(np.sum(w_combined))
        else:
            theta_adj = theta
            se_adj = 1 / np.sqrt(np.sum(weights))
            imputed_y = np.array([])
            imputed_se = np.array([])
        
        return {
            "original_estimate": float(theta),
            "original_se": float(1 / np.sqrt(np.sum(weights))),
            "adjusted_estimate": float(theta_adj),
            "adjusted_se": float(se_adj),
            "n_missing": int(k0),
            "side": side,
            "imputed_effects": imputed_y.tolist(),
            "imputed_ses": imputed_se.tolist()
        }
    
    def egger_test(self) -> Dict[str, float]:
        """
        Egger's regression test for funnel plot asymmetry.
        
        Returns:
            Dictionary with test statistics
        """
        # Precision = 1/SE
        precision = 1 / self.se
        
        # Standardized effect = y / SE
        std_effect = self.y / self.se
        
        # Regress standardized effect on precision
        # Under no bias: intercept = 0
        n = self.n
        
        mean_prec = np.mean(precision)
        mean_std = np.mean(std_effect)
        
        # Slope and intercept
        ss_prec = np.sum((precision - mean_prec) ** 2)
        sp = np.sum((precision - mean_prec) * (std_effect - mean_std))
        
        slope = sp / ss_prec
        intercept = mean_std - slope * mean_prec
        
        # Standard errors
        residuals = std_effect - (intercept + slope * precision)
        mse = np.sum(residuals ** 2) / (n - 2)
        
        se_intercept = np.sqrt(mse * (1/n + mean_prec**2 / ss_prec))
        se_slope = np.sqrt(mse / ss_prec)
        
        # t-test for intercept
        t_stat = intercept / se_intercept
        df = n - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return {
            "intercept": float(intercept),
            "se_intercept": float(se_intercept),
            "slope": float(slope),
            "se_slope": float(se_slope),
            "t_statistic": float(t_stat),
            "df": int(df),
            "p_value": float(p_value),
            "bias_detected": p_value < 0.1
        }
    
    def begg_test(self) -> Dict[str, float]:
        """
        Begg's rank correlation test for publication bias.
        
        Returns:
            Dictionary with test statistics
        """
        from t3meta.diagnostics.heterogeneity import compute_tau_squared
        
        # Standardized effects
        tau_sq = compute_tau_squared(self.y, self.se)
        total_var = self.se ** 2 + tau_sq
        
        # Pooled estimate
        weights = 1 / total_var
        theta = np.sum(weights * self.y) / np.sum(weights)
        
        # Standardized residuals
        std_res = (self.y - theta) / np.sqrt(total_var)
        
        # Variance (precision)
        variance = self.se ** 2
        
        # Kendall's tau correlation
        n = self.n
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sign_res = np.sign(std_res[i] - std_res[j])
                sign_var = np.sign(variance[i] - variance[j])
                
                if sign_res * sign_var > 0:
                    concordant += 1
                elif sign_res * sign_var < 0:
                    discordant += 1
        
        tau = (concordant - discordant) / (n * (n - 1) / 2)
        
        # Standard error under null
        se_tau = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
        
        z = tau / se_tau
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            "kendall_tau": float(tau),
            "se_tau": float(se_tau),
            "z_statistic": float(z),
            "p_value": float(p_value),
            "concordant_pairs": int(concordant),
            "discordant_pairs": int(discordant),
            "bias_detected": p_value < 0.1
        }
    
    def selection_model(
        self,
        steps: List[float] = None,
        method: str = "step"
    ) -> Dict[str, Any]:
        """
        Selection model for publication bias adjustment.
        
        Args:
            steps: P-value cutpoints for selection (default: [0.025, 0.5])
            method: 'step' for step function, 'beta' for beta function
            
        Returns:
            Dictionary with adjusted estimate and selection parameters
        """
        if steps is None:
            steps = [0.025, 0.5]
        
        from t3meta.diagnostics.heterogeneity import compute_tau_squared
        from scipy.optimize import minimize
        
        tau_sq = compute_tau_squared(self.y, self.se)
        
        # Compute two-sided p-values
        z_scores = self.y / self.se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        def neg_log_lik(params):
            theta = params[0]
            # Selection weights for each step
            weights = np.ones_like(p_values)
            
            if method == "step":
                for i, step in enumerate(steps):
                    if i == 0:
                        # Reference category: p < steps[0]
                        continue
                    prev_step = steps[i-1] if i > 0 else 0
                    # Relative selection probability for this interval
                    delta = params[i]  # log-odds of selection
                    mask = (p_values >= prev_step) & (p_values < step)
                    weights[mask] = np.exp(delta)
                
                # Last interval: p >= last step
                if len(params) > len(steps):
                    delta = params[-1]
                    weights[p_values >= steps[-1]] = np.exp(delta)
            
            # Likelihood
            total_var = self.se ** 2 + tau_sq
            ll = -0.5 * np.sum(np.log(total_var) + (self.y - theta) ** 2 / total_var)
            ll += np.sum(np.log(weights))  # Selection contribution
            
            return -ll
        
        # Initial values
        w = 1 / (self.se ** 2 + tau_sq)
        theta_init = np.sum(w * self.y) / np.sum(w)
        n_params = 1 + len(steps)  # theta + selection parameters
        x0 = np.concatenate([[theta_init], np.zeros(len(steps))])
        
        # Optimize
        result = minimize(neg_log_lik, x0, method='L-BFGS-B')
        
        theta_adj = result.x[0]
        selection_params = result.x[1:]
        
        # Standard error from Hessian (approximate)
        eps = 1e-5
        hess_diag = (
            neg_log_lik(result.x + eps * np.eye(n_params)[0]) - 
            2 * neg_log_lik(result.x) + 
            neg_log_lik(result.x - eps * np.eye(n_params)[0])
        ) / (eps ** 2)
        se_adj = 1 / np.sqrt(max(hess_diag, 1e-10))
        
        return {
            "original_estimate": float(theta_init),
            "adjusted_estimate": float(theta_adj),
            "adjusted_se": float(se_adj),
            "selection_steps": steps,
            "selection_params": selection_params.tolist(),
            "converged": result.success
        }
    
    def robust_variance(
        self,
        cluster: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Robust variance estimation (sandwich estimator).
        
        Args:
            cluster: Cluster indicators for clustered errors
            
        Returns:
            Dictionary with robust variance estimate
        """
        variances = self.se ** 2
        weights = 1 / variances
        
        # Weighted least squares estimate
        theta = np.sum(weights * self.y) / np.sum(weights)
        
        residuals = self.y - theta
        
        if cluster is None:
            # HC0 sandwich variance
            meat = np.sum(weights ** 2 * residuals ** 2)
            bread = np.sum(weights) ** 2
            robust_var = meat / bread
        else:
            # Clustered robust variance
            unique_clusters = np.unique(cluster)
            meat = 0
            for c in unique_clusters:
                mask = cluster == c
                cluster_contrib = np.sum(weights[mask] * residuals[mask])
                meat += cluster_contrib ** 2
            bread = np.sum(weights) ** 2
            robust_var = meat / bread
        
        # Model-based variance for comparison
        model_var = 1 / np.sum(weights)
        
        return {
            "estimate": float(theta),
            "model_se": float(np.sqrt(model_var)),
            "robust_se": float(np.sqrt(robust_var)),
            "robust_ci_lower": float(theta - 1.96 * np.sqrt(robust_var)),
            "robust_ci_upper": float(theta + 1.96 * np.sqrt(robust_var)),
            "variance_ratio": float(robust_var / model_var)
        }
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive sensitivity analysis summary.
        
        Returns:
            Dictionary with all sensitivity analyses
        """
        return {
            "egger_test": self.egger_test(),
            "begg_test": self.begg_test(),
            "trim_and_fill": self.trim_and_fill(),
            "robust_variance": self.robust_variance(),
            "tau_sensitivity": self.vary_tau([0.0, 0.1, 0.2, 0.3, 0.5]),
        }
