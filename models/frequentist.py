"""
Frequentist Models for T3-Meta.

This module implements frequentist meta-regression models for
estimating the target trial effect with bias adjustment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats, optimize

from t3meta.models.base import (
    BaseModel, ModelResults, 
    compute_heterogeneity_stats,
    compute_tau_squared_dl,
    compute_tau_squared_reml
)


@dataclass
class FrequentistModel(BaseModel):
    """
    Frequentist meta-regression model for T3-Meta.
    
    Implements the model:
        θ̂_j | θ_j, s_j² ~ N(θ_j, s_j²)
        θ_j = θ* + X_j'β + u_j
        u_j ~ N(0, τ²)
    
    Attributes:
        method: Estimation method ('REML', 'DL', 'ML', 'FE')
        ci_level: Confidence interval level
        max_iter: Maximum iterations for iterative methods
        tol: Convergence tolerance
        knha: Use Knapp-Hartung adjustment for CIs
    """
    
    method: str = "REML"
    ci_level: float = 0.95
    max_iter: int = 100
    tol: float = 1e-6
    knha: bool = True  # Knapp-Hartung-Sidik-Jonkman adjustment
    
    # Fitted results
    _results: Optional[ModelResults] = field(default=None, repr=False)
    _beta: Optional[np.ndarray] = field(default=None, repr=False)
    _vcov: Optional[np.ndarray] = field(default=None, repr=False)
    _tau_sq: float = field(default=0.0, repr=False)
    
    def __post_init__(self):
        """Validate inputs."""
        valid_methods = {"REML", "DL", "ML", "FE", "PM", "EB"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
    
    @property
    def results(self) -> Optional[ModelResults]:
        return self._results
    
    @property
    def is_fitted(self) -> bool:
        return self._results is not None
    
    def fit(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        Z: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        modifier_names: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResults:
        """
        Fit the frequentist meta-regression model.
        
        Args:
            y: Effect estimates (n_studies,)
            se: Standard errors (n_studies,)
            X: Design matrix (n_studies, n_features) - should include intercept
            Z: Effect modifier matrix (optional)
            feature_names: Names for design features
            modifier_names: Names for effect modifiers
            **kwargs: Additional arguments
            
        Returns:
            ModelResults object
        """
        y = np.asarray(y).flatten()
        se = np.asarray(se).flatten()
        X = np.asarray(X)
        
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        # Validate dimensions
        if len(se) != n:
            raise ValueError("y and se must have same length")
        if X.shape[0] != n:
            raise ValueError("X must have same number of rows as y")
        
        # Feature names
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(p)]
        
        warnings = []
        
        # Step 1: Estimate tau-squared
        if self.method == "FE":
            tau_sq = 0.0
            converged = True
        elif self.method == "DL":
            # Initial fixed-effect fit for residuals
            W_fe = np.diag(1 / variances)
            beta_fe = np.linalg.lstsq(X.T @ W_fe @ X, X.T @ W_fe @ y, rcond=None)[0]
            fitted_fe = X @ beta_fe
            tau_sq = compute_tau_squared_dl(y, se, fitted_fe)
            converged = True
        elif self.method == "REML":
            tau_sq, converged = compute_tau_squared_reml(y, se, X, self.max_iter, self.tol)
            if not converged:
                warnings.append("REML did not converge; using last estimate")
        elif self.method == "ML":
            tau_sq, converged = self._estimate_tau_ml(y, se, X)
            if not converged:
                warnings.append("ML did not converge")
        elif self.method == "PM":
            # Paule-Mandel estimator
            tau_sq = self._estimate_tau_pm(y, se, X)
            converged = True
        elif self.method == "EB":
            # Empirical Bayes
            tau_sq = self._estimate_tau_eb(y, se, X)
            converged = True
        else:
            tau_sq = compute_tau_squared_dl(y, se)
            converged = True
        
        self._tau_sq = tau_sq
        
        # Step 2: Weighted least squares with total variance
        total_var = variances + tau_sq
        W = np.diag(1 / total_var)
        
        # Solve weighted least squares: (X'WX)^-1 X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
            warnings.append("Design matrix near-singular; used pseudo-inverse")
        
        beta = XtWX_inv @ XtWy
        self._beta = beta
        
        # Step 3: Variance-covariance matrix
        # Standard: Var(beta) = (X'WX)^-1
        vcov = XtWX_inv
        
        # Knapp-Hartung adjustment
        if self.knha and self.method != "FE":
            fitted = X @ beta
            residuals = y - fitted
            
            # Scale factor
            df = n - p
            if df > 0:
                qm = np.sum(residuals ** 2 / total_var)
                scale = qm / df
                vcov = scale * vcov
            else:
                warnings.append("Insufficient df for Knapp-Hartung adjustment")
        
        self._vcov = vcov
        
        # Step 4: Standard errors and confidence intervals
        beta_se = np.sqrt(np.diag(vcov))
        
        # t-distribution for CIs with Knapp-Hartung
        alpha = 1 - self.ci_level
        df = n - p
        if self.knha and df > 0:
            t_crit = stats.t.ppf(1 - alpha / 2, df)
        else:
            t_crit = stats.norm.ppf(1 - alpha / 2)
        
        beta_ci = np.column_stack([
            beta - t_crit * beta_se,
            beta + t_crit * beta_se
        ])
        
        # Step 5: Fitted values and residuals
        fitted = X @ beta
        residuals = y - fitted
        
        # Step 6: Study weights
        weights = 1 / total_var
        weights = weights / weights.sum()
        
        # Step 7: Heterogeneity statistics
        het_stats = compute_heterogeneity_stats(y, se, fitted, p)
        
        # I-squared explained by model
        # Compare residual heterogeneity to total
        q_total = np.sum((1 / variances) * (y - np.average(y, weights=1/variances)) ** 2)
        if q_total > 0:
            i_sq_explained = 1 - het_stats["q_statistic"] / q_total
            i_sq_explained = max(0, min(1, i_sq_explained))
        else:
            i_sq_explained = 0.0
        
        # Step 8: Study-level bias estimates
        # Bias = X_j'β (excluding intercept)
        if p > 1:
            study_bias = X[:, 1:] @ beta[1:]
        else:
            study_bias = np.zeros(n)
        
        # Step 9: Log-likelihood and information criteria
        log_lik = -0.5 * (
            n * np.log(2 * np.pi) + 
            np.sum(np.log(total_var)) + 
            np.sum(residuals ** 2 / total_var)
        )
        
        n_params = p + 1  # beta + tau
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n)
        
        # Step 10: Tau-squared confidence interval (Q-profile method)
        tau_sq_ci = self._tau_squared_ci(y, se, X, tau_sq, alpha)
        
        # Build results
        self._results = ModelResults(
            theta_star=float(beta[0]),
            theta_star_se=float(beta_se[0]),
            theta_star_ci=(float(beta_ci[0, 0]), float(beta_ci[0, 1])),
            beta=beta[1:] if p > 1 else np.array([]),
            beta_se=beta_se[1:] if p > 1 else np.array([]),
            beta_ci=beta_ci[1:, :] if p > 1 else np.array([]),
            tau_squared=float(tau_sq),
            tau_squared_ci=tau_sq_ci,
            study_effects=y,
            study_weights=weights,
            study_bias=study_bias,
            fitted_values=fitted,
            residuals=residuals,
            convergence=converged,
            n_iter=self.max_iter if not converged else 0,
            log_likelihood=float(log_lik),
            aic=float(aic),
            bic=float(bic),
            i_squared=het_stats["i_squared"],
            i_squared_explained=i_sq_explained,
            q_statistic=het_stats["q_statistic"],
            q_df=het_stats["q_df"],
            q_pvalue=het_stats["q_pvalue"],
            feature_names=feature_names[1:] if p > 1 else [],
            method=self.method,
            model_type="frequentist",
            warnings=warnings,
        )
        
        return self._results
    
    def predict(
        self,
        X_new: np.ndarray,
        Z_new: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict effects for new design configurations.
        
        Args:
            X_new: Design matrix for prediction
            Z_new: Effect modifiers (not used in basic model)
            
        Returns:
            Tuple of (predictions, standard_errors)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_new = np.atleast_2d(X_new)
        
        predictions = X_new @ self._beta
        
        # Prediction variance
        pred_var = np.array([
            X_new[i] @ self._vcov @ X_new[i].T 
            for i in range(X_new.shape[0])
        ])
        
        # Add tau-squared for prediction interval
        pred_var += self._tau_sq
        
        return predictions, np.sqrt(pred_var)
    
    def _estimate_tau_ml(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray
    ) -> Tuple[float, bool]:
        """Estimate tau-squared using ML."""
        n = len(y)
        variances = se ** 2
        
        def neg_log_lik(tau_sq):
            if tau_sq < 0:
                return np.inf
            total_var = variances + tau_sq
            W = np.diag(1 / total_var)
            
            try:
                XtWX_inv = np.linalg.inv(X.T @ W @ X)
            except np.linalg.LinAlgError:
                return np.inf
            
            beta = XtWX_inv @ X.T @ W @ y
            residuals = y - X @ beta
            
            ll = -0.5 * (np.sum(np.log(total_var)) + np.sum(residuals ** 2 / total_var))
            return -ll
        
        # Initial value from DL
        tau_sq_init = compute_tau_squared_dl(y, se)
        
        result = optimize.minimize_scalar(
            neg_log_lik,
            bounds=(0, 10 * tau_sq_init + 1),
            method='bounded'
        )
        
        return result.x, result.success
    
    def _estimate_tau_pm(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray
    ) -> float:
        """Estimate tau-squared using Paule-Mandel method."""
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        def q_func(tau_sq):
            total_var = variances + tau_sq
            W = np.diag(1 / total_var)
            
            try:
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.inf
            
            residuals = y - X @ beta
            q = np.sum(residuals ** 2 / total_var)
            return q - (n - p)
        
        # Find root
        tau_sq_dl = compute_tau_squared_dl(y, se)
        
        if q_func(0) <= 0:
            return 0.0
        
        try:
            result = optimize.brentq(q_func, 0, 10 * tau_sq_dl + 10)
            return max(0, result)
        except ValueError:
            return tau_sq_dl
    
    def _estimate_tau_eb(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray
    ) -> float:
        """Estimate tau-squared using empirical Bayes."""
        # Morris (1983) empirical Bayes estimator
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        # Initial fixed-effect estimate
        W = np.diag(1 / variances)
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Moment estimator
        q = np.sum(residuals ** 2 / variances)
        tau_sq = max(0, (q - (n - p)) / np.sum(1 / variances))
        
        return tau_sq
    
    def _tau_squared_ci(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        tau_sq: float,
        alpha: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for tau-squared using Q-profile."""
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        df = n - p
        
        if df <= 0:
            return (0.0, np.inf)
        
        def q_func(tau):
            if tau < 0:
                return np.inf
            total_var = variances + tau
            W = np.diag(1 / total_var)
            
            try:
                beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.inf
            
            residuals = y - X @ beta
            return np.sum(residuals ** 2 / total_var)
        
        q_crit_lower = stats.chi2.ppf(1 - alpha / 2, df)
        q_crit_upper = stats.chi2.ppf(alpha / 2, df)
        
        # Find bounds
        try:
            if q_func(0) >= q_crit_lower:
                tau_lower = 0.0
            else:
                tau_lower = optimize.brentq(
                    lambda t: q_func(t) - q_crit_lower,
                    0, tau_sq * 10 + 10
                )
        except (ValueError, RuntimeError):
            tau_lower = 0.0
        
        try:
            tau_upper = optimize.brentq(
                lambda t: q_func(t) - q_crit_upper,
                tau_sq, tau_sq * 100 + 100
            )
        except (ValueError, RuntimeError):
            tau_upper = tau_sq * 3
        
        return (max(0, tau_lower), tau_upper)


@dataclass
class MixedEffectsModel(FrequentistModel):
    """
    Mixed-effects meta-regression model.
    
    Extends FrequentistModel with support for effect modifiers
    and multi-level structure.
    """
    
    include_study_random_effects: bool = True
    
    def fit(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        Z: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        modifier_names: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResults:
        """
        Fit mixed-effects model with effect modifiers.
        
        Args:
            y: Effect estimates
            se: Standard errors
            X: Design matrix (bias covariates)
            Z: Effect modifier matrix
            feature_names: Names for design features
            modifier_names: Names for effect modifiers
            
        Returns:
            ModelResults
        """
        # If no effect modifiers, fall back to base model
        if Z is None:
            return super().fit(y, se, X, feature_names=feature_names, **kwargs)
        
        # Combine X and Z into full design matrix
        Z = np.asarray(Z)
        X_full = np.hstack([X, Z])
        
        # Combined feature names
        p_x = X.shape[1]
        p_z = Z.shape[1]
        
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(p_x)]
        if modifier_names is None:
            modifier_names = [f"Z{i}" for i in range(p_z)]
        
        all_names = feature_names + modifier_names
        
        # Fit combined model
        result = super().fit(y, se, X_full, feature_names=all_names, **kwargs)
        
        # Split coefficients into beta (design) and gamma (modifiers)
        if result.beta is not None and len(result.beta) >= p_z:
            gamma = result.beta[-(p_z):]
            gamma_se = result.beta_se[-(p_z):]
            gamma_ci = result.beta_ci[-(p_z):, :]
            
            # Update result with gamma
            result.gamma = gamma
            result.gamma_se = gamma_se
            result.gamma_ci = gamma_ci
            result.effect_modifier_names = modifier_names
            
            # Trim beta to exclude gamma
            result.beta = result.beta[:-(p_z)] if p_z < len(result.beta) else np.array([])
            result.beta_se = result.beta_se[:-(p_z)] if p_z < len(result.beta_se) else np.array([])
            result.beta_ci = result.beta_ci[:-(p_z), :] if p_z < len(result.beta_ci) else np.array([])
            result.feature_names = feature_names[1:] if len(feature_names) > 1 else []
        
        return result
