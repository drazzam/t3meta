"""
Base Model Classes for T3-Meta.

This module defines abstract base classes and result containers
for T3-Meta statistical models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class ModelResults:
    """
    Container for T3-Meta model results.
    
    Holds the estimated target trial effect, bias coefficients,
    heterogeneity parameters, and diagnostics.
    
    Attributes:
        theta_star: Estimated target trial effect
        theta_star_se: Standard error of theta_star
        theta_star_ci: Confidence/credible interval for theta_star
        beta: Bias coefficients for design features
        beta_se: Standard errors of bias coefficients
        beta_ci: Confidence/credible intervals for beta
        tau_squared: Residual heterogeneity variance
        tau_squared_ci: CI for tau_squared
        gamma: Effect modifier coefficients (if applicable)
        gamma_se: Standard errors of gamma
        study_effects: Study-specific effect estimates (adjusted)
        study_weights: Final study weights
        study_bias: Estimated bias for each study
        fitted_values: Model fitted values
        residuals: Model residuals
        convergence: Whether model converged
        n_iter: Number of iterations
        log_likelihood: Log-likelihood at convergence (frequentist)
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        deviance: Model deviance
        i_squared: I-squared after bias adjustment
        q_statistic: Cochran's Q after adjustment
        feature_names: Names of design features
        effect_modifier_names: Names of effect modifiers
        method: Fitting method used
        model_type: Type of model (frequentist/bayesian)
        warnings: Any warnings generated during fitting
        extra: Additional results specific to model type
    """
    
    # Primary estimates
    theta_star: float = 0.0
    theta_star_se: float = 0.0
    theta_star_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Bias coefficients
    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    beta_se: np.ndarray = field(default_factory=lambda: np.array([]))
    beta_ci: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Heterogeneity
    tau_squared: float = 0.0
    tau_squared_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Effect modifiers (transportability)
    gamma: Optional[np.ndarray] = None
    gamma_se: Optional[np.ndarray] = None
    gamma_ci: Optional[np.ndarray] = None
    
    # Study-level results
    study_effects: np.ndarray = field(default_factory=lambda: np.array([]))
    study_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    study_bias: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_values: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model diagnostics
    convergence: bool = True
    n_iter: int = 0
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    deviance: Optional[float] = None
    
    # Heterogeneity diagnostics
    i_squared: float = 0.0
    i_squared_explained: float = 0.0  # Proportion explained by design features
    q_statistic: float = 0.0
    q_df: int = 0
    q_pvalue: float = 1.0
    
    # Metadata
    feature_names: List[str] = field(default_factory=list)
    effect_modifier_names: List[str] = field(default_factory=list)
    method: str = ""
    model_type: str = ""
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tau(self) -> float:
        """Standard deviation of residual heterogeneity."""
        return np.sqrt(self.tau_squared)
    
    @property
    def n_studies(self) -> int:
        """Number of studies."""
        return len(self.study_effects)
    
    @property
    def n_features(self) -> int:
        """Number of design features."""
        return len(self.beta)
    
    def get_theta_star_exp(self) -> Tuple[float, Tuple[float, float]]:
        """Get theta_star on exponentiated scale (for ratio measures)."""
        exp_est = np.exp(self.theta_star)
        exp_ci = (np.exp(self.theta_star_ci[0]), np.exp(self.theta_star_ci[1]))
        return exp_est, exp_ci
    
    def get_beta_for_feature(self, feature_name: str) -> Tuple[float, float, Tuple[float, float]]:
        """
        Get coefficient, SE, and CI for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Tuple of (coefficient, se, (ci_lower, ci_upper))
        """
        if feature_name not in self.feature_names:
            raise KeyError(f"Feature '{feature_name}' not found")
        
        idx = self.feature_names.index(feature_name)
        return (
            self.beta[idx],
            self.beta_se[idx],
            (self.beta_ci[idx, 0], self.beta_ci[idx, 1]) if self.beta_ci.size > 0 else (np.nan, np.nan)
        )
    
    def predict_for_target(
        self,
        target_design: Optional[np.ndarray] = None,
        target_modifiers: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Predict effect for a specific target population/design.
        
        Args:
            target_design: Design feature values for target (default: ideal trial = 0)
            target_modifiers: Effect modifier values for target
            
        Returns:
            Tuple of (predicted_effect, prediction_se)
        """
        # Default: ideal trial has no design deviations
        if target_design is None:
            target_design = np.zeros(len(self.beta))
        
        # Base prediction
        pred = self.theta_star + np.dot(target_design, self.beta)
        
        # Add effect modifier contribution if applicable
        if target_modifiers is not None and self.gamma is not None:
            pred += np.dot(target_modifiers, self.gamma)
        
        # Prediction SE (approximate - ignores covariance)
        var_pred = self.theta_star_se ** 2
        var_pred += np.sum((target_design * self.beta_se) ** 2)
        if target_modifiers is not None and self.gamma_se is not None:
            var_pred += np.sum((target_modifiers * self.gamma_se) ** 2)
        
        return pred, np.sqrt(var_pred)
    
    def decompose_heterogeneity(self) -> Dict[str, float]:
        """
        Decompose total heterogeneity into explained and residual.
        
        Returns:
            Dictionary with heterogeneity decomposition
        """
        return {
            "total_tau_squared": self.tau_squared + np.var(self.study_bias),
            "explained_tau_squared": np.var(self.study_bias),
            "residual_tau_squared": self.tau_squared,
            "proportion_explained": self.i_squared_explained,
            "residual_i_squared": self.i_squared,
        }
    
    def summary_table(self) -> str:
        """Generate summary table as string."""
        lines = [
            "=" * 60,
            "T3-Meta Analysis Results",
            "=" * 60,
            "",
            f"Target Trial Effect (θ*): {self.theta_star:.4f}",
            f"  Standard Error: {self.theta_star_se:.4f}",
            f"  95% CI: [{self.theta_star_ci[0]:.4f}, {self.theta_star_ci[1]:.4f}]",
            "",
        ]
        
        # Exponentiated for ratio measures
        exp_est, exp_ci = self.get_theta_star_exp()
        lines.append(f"  Exponentiated: {exp_est:.4f} [{exp_ci[0]:.4f}, {exp_ci[1]:.4f}]")
        lines.append("")
        
        # Heterogeneity
        lines.extend([
            "Heterogeneity:",
            f"  τ² (residual): {self.tau_squared:.4f}",
            f"  τ: {self.tau:.4f}",
            f"  I² (residual): {self.i_squared * 100:.1f}%",
            f"  I² explained by design: {self.i_squared_explained * 100:.1f}%",
            f"  Q statistic: {self.q_statistic:.2f} (df={self.q_df}, p={self.q_pvalue:.4f})",
            "",
        ])
        
        # Bias coefficients
        if len(self.beta) > 0:
            lines.append("Bias Coefficients (β):")
            for i, name in enumerate(self.feature_names):
                ci_str = ""
                if self.beta_ci.size > 0:
                    ci_str = f" [{self.beta_ci[i, 0]:.4f}, {self.beta_ci[i, 1]:.4f}]"
                lines.append(
                    f"  {name}: {self.beta[i]:.4f} (SE: {self.beta_se[i]:.4f}){ci_str}"
                )
            lines.append("")
        
        # Model info
        lines.extend([
            "Model Information:",
            f"  Method: {self.method}",
            f"  Type: {self.model_type}",
            f"  N studies: {self.n_studies}",
            f"  Converged: {self.convergence}",
        ])
        
        if self.log_likelihood is not None:
            lines.append(f"  Log-likelihood: {self.log_likelihood:.2f}")
        if self.aic is not None:
            lines.append(f"  AIC: {self.aic:.2f}")
        if self.bic is not None:
            lines.append(f"  BIC: {self.bic:.2f}")
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "theta_star": float(self.theta_star),
            "theta_star_se": float(self.theta_star_se),
            "theta_star_ci": list(self.theta_star_ci),
            "beta": self.beta.tolist() if isinstance(self.beta, np.ndarray) else self.beta,
            "beta_se": self.beta_se.tolist() if isinstance(self.beta_se, np.ndarray) else self.beta_se,
            "beta_ci": self.beta_ci.tolist() if isinstance(self.beta_ci, np.ndarray) else self.beta_ci,
            "tau_squared": float(self.tau_squared),
            "tau_squared_ci": list(self.tau_squared_ci),
            "gamma": self.gamma.tolist() if self.gamma is not None else None,
            "gamma_se": self.gamma_se.tolist() if self.gamma_se is not None else None,
            "study_effects": self.study_effects.tolist(),
            "study_weights": self.study_weights.tolist(),
            "study_bias": self.study_bias.tolist(),
            "i_squared": float(self.i_squared),
            "i_squared_explained": float(self.i_squared_explained),
            "q_statistic": float(self.q_statistic),
            "q_df": int(self.q_df),
            "q_pvalue": float(self.q_pvalue),
            "feature_names": self.feature_names,
            "method": self.method,
            "model_type": self.model_type,
            "convergence": self.convergence,
            "n_iter": self.n_iter,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "warnings": self.warnings,
        }


class BaseModel(ABC):
    """
    Abstract base class for T3-Meta models.
    
    Defines the interface that all T3-Meta models must implement.
    """
    
    @abstractmethod
    def fit(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        Z: Optional[np.ndarray] = None,
        **kwargs
    ) -> ModelResults:
        """
        Fit the model.
        
        Args:
            y: Effect estimates (on analysis scale)
            se: Standard errors
            X: Design feature matrix (n_studies x n_features)
            Z: Effect modifier matrix (optional)
            **kwargs: Additional arguments
            
        Returns:
            ModelResults object
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X_new: np.ndarray,
        Z_new: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict effects for new design/modifier combinations.
        
        Args:
            X_new: Design features for prediction
            Z_new: Effect modifiers for prediction (optional)
            
        Returns:
            Tuple of (predictions, standard_errors)
        """
        pass
    
    @property
    @abstractmethod
    def results(self) -> Optional[ModelResults]:
        """Get fitted results."""
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        pass


def compute_heterogeneity_stats(
    y: np.ndarray,
    se: np.ndarray,
    fitted: np.ndarray,
    n_params: int
) -> Dict[str, float]:
    """
    Compute heterogeneity statistics.
    
    Args:
        y: Observed effects
        se: Standard errors
        fitted: Fitted values
        n_params: Number of parameters in model
        
    Returns:
        Dictionary with Q, I², df, and p-value
    """
    weights = 1 / (se ** 2)
    residuals = y - fitted
    
    # Cochran's Q
    q = np.sum(weights * residuals ** 2)
    df = len(y) - n_params
    
    # p-value
    if df > 0:
        pvalue = 1 - stats.chi2.cdf(q, df)
    else:
        pvalue = np.nan
    
    # I-squared
    if q > df:
        i_squared = (q - df) / q
    else:
        i_squared = 0.0
    
    return {
        "q_statistic": q,
        "q_df": df,
        "q_pvalue": pvalue,
        "i_squared": i_squared,
    }


def compute_tau_squared_dl(
    y: np.ndarray,
    se: np.ndarray,
    fitted: Optional[np.ndarray] = None
) -> float:
    """
    Estimate tau-squared using DerSimonian-Laird method.
    
    Args:
        y: Effect estimates
        se: Standard errors
        fitted: Fitted values (if None, uses weighted mean)
        
    Returns:
        Estimated tau-squared
    """
    weights = 1 / (se ** 2)
    
    if fitted is None:
        fitted = np.sum(weights * y) / np.sum(weights)
    
    if np.isscalar(fitted):
        residuals = y - fitted
    else:
        residuals = y - fitted
    
    q = np.sum(weights * residuals ** 2)
    k = len(y)
    
    c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    
    tau_sq = max(0, (q - (k - 1)) / c)
    
    return tau_sq


def compute_tau_squared_reml(
    y: np.ndarray,
    se: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[float, bool]:
    """
    Estimate tau-squared using REML.
    
    Args:
        y: Effect estimates
        se: Standard errors
        X: Design matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Tuple of (tau_squared, converged)
    """
    n = len(y)
    p = X.shape[1]
    variances = se ** 2
    
    # Initialize with DL estimate
    tau_sq = compute_tau_squared_dl(y, se)
    
    converged = False
    for iteration in range(max_iter):
        # Total variance
        V = variances + tau_sq
        W = np.diag(1 / V)
        
        # Weighted least squares
        XtWX = X.T @ W @ X
        XtWX_inv = np.linalg.pinv(XtWX)
        beta = XtWX_inv @ X.T @ W @ y
        
        # Residuals
        residuals = y - X @ beta
        
        # Projection matrix
        P = W - W @ X @ XtWX_inv @ X.T @ W
        
        # REML score
        score = -0.5 * np.trace(P) + 0.5 * residuals.T @ P @ P @ residuals
        
        # Fisher information
        fisher = 0.5 * np.trace(P @ P)
        
        # Update
        tau_sq_new = max(0, tau_sq + score / fisher)
        
        # Check convergence
        if abs(tau_sq_new - tau_sq) < tol:
            converged = True
            tau_sq = tau_sq_new
            break
        
        tau_sq = tau_sq_new
    
    return tau_sq, converged
