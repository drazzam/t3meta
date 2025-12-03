"""
Statistical models for T3-Meta framework.

This module provides statistical models for T3-Meta analyses,
including frequentist and Bayesian implementations.
"""

from t3meta.models.base import (
    BaseModel,
    ModelResults,
    compute_heterogeneity_stats,
    compute_tau_squared_dl,
    compute_tau_squared_reml,
)
from t3meta.models.frequentist import (
    FrequentistModel,
    MixedEffectsModel,
)
from t3meta.models.bayesian import (
    BayesianModel,
    BayesianResults,
)
from t3meta.models.priors import (
    Prior,
    NormalPrior,
    HalfNormalPrior,
    HalfCauchyPrior,
    UniformPrior,
    InverseGammaPrior,
    ExponentialPrior,
    BiasPriorSpec,
    get_default_bias_priors,
    get_vague_priors,
    get_weakly_informative_priors,
    get_skeptical_priors,
)
from t3meta.models.t3_meta_analysis import T3MetaAnalysis

__all__ = [
    # Base classes
    "BaseModel",
    "ModelResults",
    "compute_heterogeneity_stats",
    "compute_tau_squared_dl",
    "compute_tau_squared_reml",
    # Frequentist models
    "FrequentistModel",
    "MixedEffectsModel",
    # Bayesian models
    "BayesianModel",
    "BayesianResults",
    # Priors
    "Prior",
    "NormalPrior",
    "HalfNormalPrior",
    "HalfCauchyPrior",
    "UniformPrior",
    "InverseGammaPrior",
    "ExponentialPrior",
    "BiasPriorSpec",
    "get_default_bias_priors",
    "get_vague_priors",
    "get_weakly_informative_priors",
    "get_skeptical_priors",
    # Main analysis class
    "T3MetaAnalysis",
]
