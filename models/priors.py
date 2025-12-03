"""
Prior Specifications for Bayesian T3-Meta Models.

This module defines prior distributions used in Bayesian meta-analysis,
with special support for informative priors on bias coefficients.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable
import numpy as np
from scipy import stats


class Prior(ABC):
    """Abstract base class for prior distributions."""
    
    @abstractmethod
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability density at x."""
        pass
    
    @abstractmethod
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples from the prior."""
        pass
    
    @abstractmethod
    def mean(self) -> float:
        """Prior mean."""
        pass
    
    @abstractmethod
    def variance(self) -> float:
        """Prior variance."""
        pass
    
    @property
    def std(self) -> float:
        """Prior standard deviation."""
        return np.sqrt(self.variance())
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        pass
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Prior:
        """Create prior from dictionary."""
        prior_type = d.get("type")
        if prior_type == "normal":
            return NormalPrior(loc=d["loc"], scale=d["scale"])
        elif prior_type == "half_normal":
            return HalfNormalPrior(scale=d["scale"])
        elif prior_type == "uniform":
            return UniformPrior(lower=d["lower"], upper=d["upper"])
        elif prior_type == "half_cauchy":
            return HalfCauchyPrior(scale=d["scale"])
        elif prior_type == "inverse_gamma":
            return InverseGammaPrior(alpha=d["alpha"], beta=d["beta"])
        elif prior_type == "exponential":
            return ExponentialPrior(rate=d["rate"])
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")


@dataclass
class NormalPrior(Prior):
    """
    Normal (Gaussian) prior distribution.
    
    Attributes:
        loc: Mean of the distribution
        scale: Standard deviation
    """
    loc: float = 0.0
    scale: float = 1.0
    
    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return stats.norm.logpdf(x, loc=self.loc, scale=self.scale)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.loc, self.scale, size=n)
    
    def mean(self) -> float:
        return self.loc
    
    def variance(self) -> float:
        return self.scale ** 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "normal", "loc": self.loc, "scale": self.scale}


@dataclass
class HalfNormalPrior(Prior):
    """
    Half-normal prior (folded normal) for positive parameters.
    
    Useful for variance components and heterogeneity parameters.
    
    Attributes:
        scale: Scale parameter (σ of the underlying normal)
    """
    scale: float = 1.0
    
    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x)
        result = np.where(
            x >= 0,
            np.log(2) + stats.norm.logpdf(x, loc=0, scale=self.scale),
            -np.inf
        )
        return result if result.ndim > 0 else float(result)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return np.abs(rng.normal(0, self.scale, size=n))
    
    def mean(self) -> float:
        return self.scale * np.sqrt(2 / np.pi)
    
    def variance(self) -> float:
        return self.scale ** 2 * (1 - 2 / np.pi)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "half_normal", "scale": self.scale}


@dataclass
class HalfCauchyPrior(Prior):
    """
    Half-Cauchy prior for positive parameters.
    
    Often recommended for variance components due to heavier tails
    than half-normal.
    
    Attributes:
        scale: Scale parameter (γ)
    """
    scale: float = 1.0
    
    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x)
        # Half-Cauchy: 2 / (π * γ * (1 + (x/γ)^2)) for x >= 0
        result = np.where(
            x >= 0,
            np.log(2) - np.log(np.pi) - np.log(self.scale) - 
            np.log(1 + (x / self.scale) ** 2),
            -np.inf
        )
        return result if result.ndim > 0 else float(result)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Half-Cauchy samples from |Cauchy|
        return np.abs(rng.standard_cauchy(size=n) * self.scale)
    
    def mean(self) -> float:
        return np.inf  # Cauchy has no finite mean
    
    def variance(self) -> float:
        return np.inf  # Cauchy has no finite variance
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "half_cauchy", "scale": self.scale}


@dataclass
class UniformPrior(Prior):
    """
    Uniform prior distribution.
    
    Attributes:
        lower: Lower bound
        upper: Upper bound
    """
    lower: float = 0.0
    upper: float = 1.0
    
    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError(f"lower must be < upper, got [{self.lower}, {self.upper}]")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return stats.uniform.logpdf(x, loc=self.lower, scale=self.upper - self.lower)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lower, self.upper, size=n)
    
    def mean(self) -> float:
        return (self.lower + self.upper) / 2
    
    def variance(self) -> float:
        return (self.upper - self.lower) ** 2 / 12
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "uniform", "lower": self.lower, "upper": self.upper}


@dataclass
class InverseGammaPrior(Prior):
    """
    Inverse-Gamma prior for variance parameters.
    
    Attributes:
        alpha: Shape parameter (α > 0)
        beta: Rate parameter (β > 0)
    """
    alpha: float = 1.0
    beta: float = 1.0
    
    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return stats.invgamma.logpdf(x, a=self.alpha, scale=self.beta)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return stats.invgamma.rvs(a=self.alpha, scale=self.beta, size=n, random_state=rng)
    
    def mean(self) -> float:
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return np.inf
    
    def variance(self) -> float:
        if self.alpha > 2:
            return self.beta ** 2 / ((self.alpha - 1) ** 2 * (self.alpha - 2))
        return np.inf
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "inverse_gamma", "alpha": self.alpha, "beta": self.beta}


@dataclass
class ExponentialPrior(Prior):
    """
    Exponential prior for positive parameters.
    
    Attributes:
        rate: Rate parameter (λ > 0)
    """
    rate: float = 1.0
    
    def __post_init__(self):
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")
    
    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return stats.expon.logpdf(x, scale=1/self.rate)
    
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.exponential(1/self.rate, size=n)
    
    def mean(self) -> float:
        return 1 / self.rate
    
    def variance(self) -> float:
        return 1 / (self.rate ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "exponential", "rate": self.rate}


@dataclass
class BiasPriorSpec:
    """
    Specification for priors on bias coefficients.
    
    In T3-Meta, bias priors encode methodological knowledge about
    how design features affect bias. For example:
    - Non-randomized studies may exaggerate benefits
    - Immortal time bias inflates effect sizes
    - Unadjusted confounding typically biases toward null or away
    
    Attributes:
        feature_name: Name of the design feature
        prior: Prior distribution for the bias coefficient
        direction: Expected direction of bias ('positive', 'negative', 'none')
        rationale: Text explaining the prior choice
        source: Source of prior information (e.g., meta-epidemiological study)
    """
    feature_name: str
    prior: Prior
    direction: str = "none"  # 'positive', 'negative', 'none'
    rationale: str = ""
    source: str = ""
    
    def __post_init__(self):
        if self.direction not in {'positive', 'negative', 'none'}:
            raise ValueError(f"direction must be 'positive', 'negative', or 'none'")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "prior": self.prior.to_dict(),
            "direction": self.direction,
            "rationale": self.rationale,
            "source": self.source,
        }


def get_default_bias_priors() -> Dict[str, BiasPriorSpec]:
    """
    Get default priors for common bias sources.
    
    These are informed by meta-epidemiological evidence about
    typical bias directions and magnitudes.
    
    Returns:
        Dictionary mapping feature names to BiasPriorSpec objects
    """
    return {
        "is_rct": BiasPriorSpec(
            feature_name="is_rct",
            prior=NormalPrior(loc=0.0, scale=0.05),
            direction="none",
            rationale="RCTs are reference; coefficient near zero expected",
        ),
        
        "has_immortal_time_bias": BiasPriorSpec(
            feature_name="has_immortal_time_bias",
            prior=NormalPrior(loc=-0.15, scale=0.10),
            direction="negative",
            rationale="Immortal time typically inflates apparent benefit (negative log-HR)",
            source="Suissa 2007, Lévesque 2010",
        ),
        
        "has_prevalent_user_bias": BiasPriorSpec(
            feature_name="has_prevalent_user_bias",
            prior=NormalPrior(loc=-0.10, scale=0.10),
            direction="negative",
            rationale="Prevalent users are survivors, biasing toward benefit",
            source="Ray 2003",
        ),
        
        "confounding_none": BiasPriorSpec(
            feature_name="confounding_none",
            prior=NormalPrior(loc=0.0, scale=0.25),
            direction="none",
            rationale="Unadjusted estimates may be biased in either direction",
        ),
        
        "confounding_iptw": BiasPriorSpec(
            feature_name="confounding_iptw",
            prior=NormalPrior(loc=0.0, scale=0.08),
            direction="none",
            rationale="IPTW provides reasonable confounding control",
        ),
        
        "confounding_doubly_robust": BiasPriorSpec(
            feature_name="confounding_doubly_robust",
            prior=NormalPrior(loc=0.0, scale=0.05),
            direction="none",
            rationale="Doubly robust methods provide strong bias reduction",
        ),
        
        "outcome_adjudicated": BiasPriorSpec(
            feature_name="outcome_adjudicated",
            prior=NormalPrior(loc=0.0, scale=0.05),
            direction="none",
            rationale="Adjudicated outcomes are reference standard",
        ),
        
        "loss_to_followup_pct": BiasPriorSpec(
            feature_name="loss_to_followup_pct",
            prior=NormalPrior(loc=0.0, scale=0.01),
            direction="none",
            rationale="Each % lost to follow-up may introduce small bias",
        ),
    }


def get_vague_priors(n_features: int, scale: float = 10.0) -> Dict[str, Prior]:
    """
    Get vague/diffuse priors for all parameters.
    
    Args:
        n_features: Number of design features (for beta coefficients)
        scale: Scale for normal priors
        
    Returns:
        Dictionary of priors
    """
    priors = {
        "theta_star": NormalPrior(loc=0.0, scale=scale),
        "tau": HalfCauchyPrior(scale=0.5),
    }
    
    for i in range(n_features):
        priors[f"beta_{i}"] = NormalPrior(loc=0.0, scale=scale)
    
    return priors


def get_weakly_informative_priors(
    n_features: int,
    effect_measure: str = "log_HR"
) -> Dict[str, Prior]:
    """
    Get weakly informative priors suitable for most meta-analyses.
    
    Based on recommendations from Röver (2020) for tau and
    general meta-analysis practice.
    
    Args:
        n_features: Number of design features
        effect_measure: Type of effect measure for scaling
        
    Returns:
        Dictionary of priors
    """
    # Prior for target effect
    if effect_measure in ["log_HR", "log_RR", "log_OR"]:
        theta_prior = NormalPrior(loc=0.0, scale=1.0)  # Centered on null
        tau_prior = HalfNormalPrior(scale=0.5)  # Reasonable heterogeneity
        beta_scale = 0.5
    elif effect_measure in ["RD"]:
        theta_prior = NormalPrior(loc=0.0, scale=0.2)
        tau_prior = HalfNormalPrior(scale=0.1)
        beta_scale = 0.1
    elif effect_measure in ["SMD", "MD"]:
        theta_prior = NormalPrior(loc=0.0, scale=1.0)
        tau_prior = HalfNormalPrior(scale=0.5)
        beta_scale = 0.5
    else:
        theta_prior = NormalPrior(loc=0.0, scale=1.0)
        tau_prior = HalfNormalPrior(scale=0.5)
        beta_scale = 0.5
    
    priors = {
        "theta_star": theta_prior,
        "tau": tau_prior,
    }
    
    for i in range(n_features):
        priors[f"beta_{i}"] = NormalPrior(loc=0.0, scale=beta_scale)
    
    return priors


def get_skeptical_priors(
    n_features: int,
    effect_measure: str = "log_HR",
    skepticism_factor: float = 0.5
) -> Dict[str, Prior]:
    """
    Get skeptical priors that shrink effects toward null.
    
    Useful for sensitivity analysis and to counteract publication bias.
    
    Args:
        n_features: Number of design features
        effect_measure: Type of effect measure
        skepticism_factor: How strongly to shrink toward null (0-1)
        
    Returns:
        Dictionary of priors
    """
    # Tighter prior on target effect, centered on null
    if effect_measure in ["log_HR", "log_RR", "log_OR"]:
        scale = 0.3 * (1 - skepticism_factor * 0.5)
    else:
        scale = 0.1 * (1 - skepticism_factor * 0.5)
    
    priors = {
        "theta_star": NormalPrior(loc=0.0, scale=scale),
        "tau": HalfNormalPrior(scale=0.3),
    }
    
    for i in range(n_features):
        priors[f"beta_{i}"] = NormalPrior(loc=0.0, scale=0.3)
    
    return priors
