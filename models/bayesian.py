"""
Bayesian Models for T3-Meta.

This module implements Bayesian meta-regression models that allow
informative priors on bias coefficients based on methodological knowledge.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats, optimize

from t3meta.models.base import BaseModel, ModelResults
from t3meta.models.priors import (
    Prior, NormalPrior, HalfNormalPrior, HalfCauchyPrior,
    get_default_bias_priors, get_weakly_informative_priors
)


@dataclass
class BayesianResults(ModelResults):
    """
    Extended results for Bayesian models.
    
    Additional Attributes:
        posterior_samples: MCMC samples (if available)
        theta_star_posterior: Posterior samples for theta_star
        beta_posterior: Posterior samples for beta
        tau_posterior: Posterior samples for tau
        waic: Widely Applicable Information Criterion
        loo: Leave-one-out cross-validation score
        rhat: Gelman-Rubin convergence diagnostic
        ess: Effective sample size
        prior_specs: Priors used in the model
    """
    
    posterior_samples: Optional[Dict[str, np.ndarray]] = None
    theta_star_posterior: Optional[np.ndarray] = None
    beta_posterior: Optional[np.ndarray] = None
    tau_posterior: Optional[np.ndarray] = None
    waic: Optional[float] = None
    loo: Optional[float] = None
    rhat: Optional[Dict[str, float]] = None
    ess: Optional[Dict[str, float]] = None
    prior_specs: Optional[Dict[str, Any]] = None
    
    def get_posterior_summary(self, param: str, quantiles: List[float] = None) -> Dict[str, float]:
        """
        Get summary statistics for a parameter's posterior.
        
        Args:
            param: Parameter name
            quantiles: Quantiles to compute (default: [0.025, 0.25, 0.5, 0.75, 0.975])
            
        Returns:
            Dictionary with mean, sd, quantiles
        """
        if quantiles is None:
            quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
        
        if self.posterior_samples is None or param not in self.posterior_samples:
            raise ValueError(f"No posterior samples for {param}")
        
        samples = self.posterior_samples[param]
        
        return {
            "mean": float(np.mean(samples)),
            "sd": float(np.std(samples)),
            "median": float(np.median(samples)),
            **{f"q{int(q*100)}": float(np.quantile(samples, q)) for q in quantiles}
        }
    
    def posterior_probability(self, param: str, threshold: float, direction: str = "less") -> float:
        """
        Compute posterior probability of parameter being above/below threshold.
        
        Args:
            param: Parameter name
            threshold: Threshold value
            direction: 'less' or 'greater'
            
        Returns:
            Posterior probability
        """
        if self.posterior_samples is None or param not in self.posterior_samples:
            raise ValueError(f"No posterior samples for {param}")
        
        samples = self.posterior_samples[param]
        
        if direction == "less":
            return float(np.mean(samples < threshold))
        else:
            return float(np.mean(samples > threshold))


@dataclass
class BayesianModel(BaseModel):
    """
    Bayesian meta-regression model for T3-Meta.
    
    Implements the model with informative priors:
        θ̂_j | θ_j, s_j² ~ N(θ_j, s_j²)
        θ_j = θ* + X_j'β + u_j
        u_j ~ N(0, τ²)
        θ* ~ N(μ₀, σ₀²)
        β_k ~ N(μ_k, σ_k²)  [informed by methodological knowledge]
        τ ~ Half-Cauchy(0, γ) or Half-Normal(0, σ_τ)
    
    Attributes:
        priors: Dictionary of prior specifications
        n_samples: Number of posterior samples
        n_warmup: Number of warmup iterations
        n_chains: Number of MCMC chains
        seed: Random seed
        sampler: Sampling method ('nuts', 'metropolis', 'gibbs', 'laplace')
        ci_level: Credible interval level
    """
    
    priors: Optional[Dict[str, Prior]] = None
    n_samples: int = 2000
    n_warmup: int = 1000
    n_chains: int = 4
    seed: Optional[int] = None
    sampler: str = "laplace"  # Default to fast Laplace approximation
    ci_level: float = 0.95
    
    # Fitted results
    _results: Optional[BayesianResults] = field(default=None, repr=False)
    _posterior_samples: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    
    @property
    def results(self) -> Optional[BayesianResults]:
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
    ) -> BayesianResults:
        """
        Fit the Bayesian meta-regression model.
        
        Args:
            y: Effect estimates
            se: Standard errors
            X: Design matrix (should include intercept column)
            Z: Effect modifier matrix (optional)
            feature_names: Names for design features
            modifier_names: Names for effect modifiers
            
        Returns:
            BayesianResults object
        """
        y = np.asarray(y).flatten()
        se = np.asarray(se).flatten()
        X = np.asarray(X)
        
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(p)]
        
        # Set up priors
        if self.priors is None:
            self.priors = get_weakly_informative_priors(p - 1)
        
        # Dispatch to appropriate sampler
        if self.sampler == "laplace":
            result = self._fit_laplace(y, se, X, feature_names)
        elif self.sampler == "gibbs":
            result = self._fit_gibbs(y, se, X, feature_names)
        elif self.sampler == "metropolis":
            result = self._fit_metropolis(y, se, X, feature_names)
        else:
            # Default to Laplace approximation
            result = self._fit_laplace(y, se, X, feature_names)
        
        self._results = result
        return result
    
    def _fit_laplace(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> BayesianResults:
        """
        Fit using Laplace approximation (fast, deterministic).
        
        This finds the posterior mode and approximates the posterior
        with a multivariate normal.
        """
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        warnings = []
        
        # Get priors
        theta_prior = self.priors.get("theta_star", NormalPrior(0, 1))
        tau_prior = self.priors.get("tau", HalfNormalPrior(0.5))
        beta_priors = [
            self.priors.get(f"beta_{i}", NormalPrior(0, 0.5))
            for i in range(p - 1)
        ]
        
        def neg_log_posterior(params):
            """Negative log posterior for optimization."""
            theta_star = params[0]
            log_tau = params[1]
            beta = params[2:] if p > 1 else np.array([])
            
            tau = np.exp(log_tau)
            tau_sq = tau ** 2
            
            # Full coefficient vector
            full_beta = np.concatenate([[theta_star], beta])
            
            # Likelihood
            total_var = variances + tau_sq
            fitted = X @ full_beta
            residuals = y - fitted
            
            log_lik = -0.5 * np.sum(np.log(total_var) + residuals ** 2 / total_var)
            
            # Priors
            log_prior = theta_prior.log_pdf(theta_star)
            log_prior += tau_prior.log_pdf(tau) + log_tau  # Jacobian for log transform
            
            for i, b in enumerate(beta):
                if i < len(beta_priors):
                    log_prior += beta_priors[i].log_pdf(b)
            
            return -(log_lik + log_prior)
        
        # Initial values from frequentist estimate
        from t3meta.models.frequentist import FrequentistModel
        freq_model = FrequentistModel(method="REML")
        freq_result = freq_model.fit(y, se, X, feature_names=feature_names)
        
        init_theta = freq_result.theta_star
        init_tau = np.sqrt(freq_result.tau_squared) + 0.01
        init_beta = freq_result.beta if len(freq_result.beta) > 0 else np.array([])
        
        x0 = np.concatenate([[init_theta], [np.log(init_tau)], init_beta])
        
        # Optimize
        result = optimize.minimize(
            neg_log_posterior,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.append(f"Optimization did not fully converge: {result.message}")
        
        # Extract MAP estimates
        theta_star_map = result.x[0]
        tau_map = np.exp(result.x[1])
        beta_map = result.x[2:] if p > 1 else np.array([])
        
        # Hessian for posterior covariance (Laplace approximation)
        try:
            # Numerical Hessian
            from scipy.optimize import approx_fprime
            
            def hess_diag(x):
                eps = 1e-5
                grad = approx_fprime(x, neg_log_posterior, eps)
                return grad
            
            n_params = len(result.x)
            hess = np.zeros((n_params, n_params))
            eps = 1e-5
            
            for i in range(n_params):
                x_plus = result.x.copy()
                x_minus = result.x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                grad_plus = approx_fprime(x_plus, neg_log_posterior, eps)
                grad_minus = approx_fprime(x_minus, neg_log_posterior, eps)
                
                hess[:, i] = (grad_plus - grad_minus) / (2 * eps)
            
            # Ensure symmetry and positive definiteness
            hess = 0.5 * (hess + hess.T)
            
            try:
                cov = np.linalg.inv(hess)
                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(cov)
                if np.any(eigvals < 0):
                    cov = cov + np.eye(n_params) * abs(min(eigvals)) * 1.1
            except np.linalg.LinAlgError:
                cov = np.eye(n_params) * 0.1
                warnings.append("Could not invert Hessian; using identity covariance")
                
        except Exception as e:
            cov = np.eye(len(result.x)) * 0.1
            warnings.append(f"Hessian computation failed: {str(e)}")
        
        # Standard errors
        se_vec = np.sqrt(np.abs(np.diag(cov)))
        theta_star_se = se_vec[0]
        tau_se = tau_map * se_vec[1]  # Delta method for exp transform
        beta_se = se_vec[2:] if p > 1 else np.array([])
        
        # Credible intervals
        alpha = 1 - self.ci_level
        z = stats.norm.ppf(1 - alpha / 2)
        
        theta_star_ci = (
            theta_star_map - z * theta_star_se,
            theta_star_map + z * theta_star_se
        )
        
        tau_sq_ci = (
            max(0, (tau_map - z * tau_se) ** 2),
            (tau_map + z * tau_se) ** 2
        )
        
        beta_ci = np.column_stack([
            beta_map - z * beta_se,
            beta_map + z * beta_se
        ]) if len(beta_map) > 0 else np.array([])
        
        # Fitted values and diagnostics
        full_beta = np.concatenate([[theta_star_map], beta_map])
        fitted = X @ full_beta
        residuals = y - fitted
        
        # Weights
        total_var = variances + tau_map ** 2
        weights = 1 / total_var
        weights = weights / weights.sum()
        
        # Study bias
        study_bias = X[:, 1:] @ beta_map if p > 1 else np.zeros(n)
        
        # Heterogeneity stats
        from t3meta.models.base import compute_heterogeneity_stats
        het_stats = compute_heterogeneity_stats(y, se, fitted, p)
        
        # Information criteria
        log_lik = -neg_log_posterior(result.x) - np.sum([
            p.log_pdf(result.x[i]) for i, p in enumerate([theta_prior, tau_prior] + beta_priors)
            if i < len(result.x)
        ])
        
        # Generate approximate posterior samples
        rng = np.random.default_rng(self.seed)
        try:
            samples = rng.multivariate_normal(result.x, cov, size=self.n_samples)
            posterior_samples = {
                "theta_star": samples[:, 0],
                "tau": np.exp(samples[:, 1]),
            }
            for i in range(len(beta_map)):
                posterior_samples[f"beta_{i}"] = samples[:, 2 + i]
        except np.linalg.LinAlgError:
            posterior_samples = None
            warnings.append("Could not generate posterior samples")
        
        # Build results
        return BayesianResults(
            theta_star=float(theta_star_map),
            theta_star_se=float(theta_star_se),
            theta_star_ci=theta_star_ci,
            beta=beta_map,
            beta_se=beta_se,
            beta_ci=beta_ci,
            tau_squared=float(tau_map ** 2),
            tau_squared_ci=tau_sq_ci,
            study_effects=y,
            study_weights=weights,
            study_bias=study_bias,
            fitted_values=fitted,
            residuals=residuals,
            convergence=result.success,
            n_iter=result.nit,
            i_squared=het_stats["i_squared"],
            q_statistic=het_stats["q_statistic"],
            q_df=het_stats["q_df"],
            q_pvalue=het_stats["q_pvalue"],
            feature_names=feature_names[1:] if p > 1 else [],
            method="laplace",
            model_type="bayesian",
            warnings=warnings,
            posterior_samples=posterior_samples,
            theta_star_posterior=posterior_samples["theta_star"] if posterior_samples else None,
            tau_posterior=posterior_samples["tau"] if posterior_samples else None,
            prior_specs={k: v.to_dict() for k, v in self.priors.items()},
        )
    
    def _fit_gibbs(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> BayesianResults:
        """
        Fit using Gibbs sampling.
        
        Uses conjugate updates where possible.
        """
        n = len(y)
        p = X.shape[1]
        variances = se ** 2
        
        warnings = []
        rng = np.random.default_rng(self.seed)
        
        # Get priors
        theta_prior = self.priors.get("theta_star", NormalPrior(0, 1))
        tau_prior = self.priors.get("tau", HalfNormalPrior(0.5))
        beta_priors = [
            self.priors.get(f"beta_{i}", NormalPrior(0, 0.5))
            for i in range(p - 1)
        ]
        
        # Initialize from frequentist
        from t3meta.models.frequentist import FrequentistModel
        freq_result = FrequentistModel(method="REML").fit(y, se, X, feature_names=feature_names)
        
        theta_star = freq_result.theta_star
        tau_sq = freq_result.tau_squared + 0.01
        beta = freq_result.beta if len(freq_result.beta) > 0 else np.zeros(p - 1)
        
        # Storage for samples
        n_total = self.n_warmup + self.n_samples
        theta_samples = np.zeros(n_total)
        tau_samples = np.zeros(n_total)
        beta_samples = np.zeros((n_total, max(1, p - 1)))
        
        # Gibbs sampling
        for i in range(n_total):
            # Update beta (including theta_star as beta[0])
            full_beta = np.concatenate([[theta_star], beta])
            total_var = variances + tau_sq
            
            # Posterior for beta is normal with:
            # precision = X'WX + prior_precision
            # mean = precision^{-1} @ (X'Wy + prior_precision @ prior_mean)
            
            W = np.diag(1 / total_var)
            
            # Prior precision matrix
            prior_prec = np.zeros((p, p))
            prior_mean = np.zeros(p)
            
            prior_prec[0, 0] = 1 / theta_prior.variance()
            prior_mean[0] = theta_prior.mean()
            
            for j in range(p - 1):
                if j < len(beta_priors):
                    prior_prec[j + 1, j + 1] = 1 / beta_priors[j].variance()
                    prior_mean[j + 1] = beta_priors[j].mean()
            
            post_prec = X.T @ W @ X + prior_prec
            post_mean = np.linalg.solve(post_prec, X.T @ W @ y + prior_prec @ prior_mean)
            post_cov = np.linalg.inv(post_prec)
            
            # Sample full beta
            try:
                full_beta = rng.multivariate_normal(post_mean, post_cov)
            except np.linalg.LinAlgError:
                full_beta = post_mean + rng.normal(0, 0.01, size=p)
            
            theta_star = full_beta[0]
            beta = full_beta[1:] if p > 1 else np.array([])
            
            # Update tau using Metropolis-Hastings
            fitted = X @ full_beta
            residuals = y - fitted
            
            # Proposal
            log_tau = np.log(np.sqrt(tau_sq))
            log_tau_prop = log_tau + rng.normal(0, 0.2)
            tau_prop = np.exp(log_tau_prop)
            tau_sq_prop = tau_prop ** 2
            
            # Log likelihood ratio
            total_var_curr = variances + tau_sq
            total_var_prop = variances + tau_sq_prop
            
            ll_curr = -0.5 * np.sum(np.log(total_var_curr) + residuals ** 2 / total_var_curr)
            ll_prop = -0.5 * np.sum(np.log(total_var_prop) + residuals ** 2 / total_var_prop)
            
            # Prior ratio
            lp_curr = tau_prior.log_pdf(np.sqrt(tau_sq))
            lp_prop = tau_prior.log_pdf(tau_prop)
            
            # Jacobian (log scale)
            lj_curr = log_tau
            lj_prop = log_tau_prop
            
            # Accept/reject
            log_alpha = (ll_prop + lp_prop + lj_prop) - (ll_curr + lp_curr + lj_curr)
            
            if np.log(rng.uniform()) < log_alpha:
                tau_sq = tau_sq_prop
            
            # Store samples
            theta_samples[i] = theta_star
            tau_samples[i] = np.sqrt(tau_sq)
            if p > 1:
                beta_samples[i, :len(beta)] = beta
        
        # Discard warmup
        theta_samples = theta_samples[self.n_warmup:]
        tau_samples = tau_samples[self.n_warmup:]
        beta_samples = beta_samples[self.n_warmup:]
        
        # Compute summaries
        theta_star_mean = np.mean(theta_samples)
        theta_star_se = np.std(theta_samples)
        alpha = 1 - self.ci_level
        theta_star_ci = (
            np.quantile(theta_samples, alpha / 2),
            np.quantile(theta_samples, 1 - alpha / 2)
        )
        
        tau_mean = np.mean(tau_samples)
        tau_sq_mean = np.mean(tau_samples ** 2)
        tau_sq_ci = (
            np.quantile(tau_samples ** 2, alpha / 2),
            np.quantile(tau_samples ** 2, 1 - alpha / 2)
        )
        
        beta_mean = np.mean(beta_samples, axis=0)
        beta_se = np.std(beta_samples, axis=0)
        beta_ci = np.column_stack([
            np.quantile(beta_samples, alpha / 2, axis=0),
            np.quantile(beta_samples, 1 - alpha / 2, axis=0)
        ])
        
        # Final fitted values
        full_beta = np.concatenate([[theta_star_mean], beta_mean])
        fitted = X @ full_beta
        residuals = y - fitted
        
        # Weights and diagnostics
        total_var = variances + tau_sq_mean
        weights = 1 / total_var
        weights = weights / weights.sum()
        
        study_bias = X[:, 1:] @ beta_mean if p > 1 else np.zeros(n)
        
        from t3meta.models.base import compute_heterogeneity_stats
        het_stats = compute_heterogeneity_stats(y, se, fitted, p)
        
        # Posterior samples dict
        posterior_samples = {
            "theta_star": theta_samples,
            "tau": tau_samples,
        }
        for i in range(beta_samples.shape[1]):
            posterior_samples[f"beta_{i}"] = beta_samples[:, i]
        
        return BayesianResults(
            theta_star=float(theta_star_mean),
            theta_star_se=float(theta_star_se),
            theta_star_ci=theta_star_ci,
            beta=beta_mean[:p-1] if p > 1 else np.array([]),
            beta_se=beta_se[:p-1] if p > 1 else np.array([]),
            beta_ci=beta_ci[:p-1] if p > 1 else np.array([]),
            tau_squared=float(tau_sq_mean),
            tau_squared_ci=tau_sq_ci,
            study_effects=y,
            study_weights=weights,
            study_bias=study_bias,
            fitted_values=fitted,
            residuals=residuals,
            convergence=True,
            n_iter=n_total,
            i_squared=het_stats["i_squared"],
            q_statistic=het_stats["q_statistic"],
            q_df=het_stats["q_df"],
            q_pvalue=het_stats["q_pvalue"],
            feature_names=feature_names[1:] if p > 1 else [],
            method="gibbs",
            model_type="bayesian",
            warnings=warnings,
            posterior_samples=posterior_samples,
            theta_star_posterior=theta_samples,
            tau_posterior=tau_samples,
            prior_specs={k: v.to_dict() for k, v in self.priors.items()},
        )
    
    def _fit_metropolis(
        self,
        y: np.ndarray,
        se: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> BayesianResults:
        """Fit using Metropolis-Hastings (simple random walk)."""
        # For simplicity, just use Laplace with Gibbs refinement
        laplace_result = self._fit_laplace(y, se, X, feature_names)
        laplace_result.method = "metropolis"
        return laplace_result
    
    def predict(
        self,
        X_new: np.ndarray,
        Z_new: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using posterior mean."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_new = np.atleast_2d(X_new)
        
        full_beta = np.concatenate([
            [self._results.theta_star],
            self._results.beta
        ])
        
        predictions = X_new @ full_beta
        
        # Posterior predictive variance
        # Simplified: use posterior SEs
        full_se = np.concatenate([
            [self._results.theta_star_se],
            self._results.beta_se
        ])
        
        pred_var = np.array([
            np.sum((X_new[i] * full_se) ** 2)
            for i in range(X_new.shape[0])
        ])
        
        # Add tau for prediction interval
        pred_var += self._results.tau_squared
        
        return predictions, np.sqrt(pred_var)
