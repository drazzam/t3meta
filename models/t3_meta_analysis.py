"""
T3MetaAnalysis: Main Interface for T3-Meta Framework.

This module provides the primary user interface for conducting
target trial-centric meta-analyses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Literal
import numpy as np
import json
import warnings

from t3meta.core.target_trial import TargetTrial
from t3meta.core.study import Study, DesignFeatures
from t3meta.core.registry import StudyRegistry
from t3meta.core.estimand import EffectMeasure, EstimandType
from t3meta.models.base import ModelResults
from t3meta.models.frequentist import FrequentistModel, MixedEffectsModel
from t3meta.models.bayesian import BayesianModel, BayesianResults
from t3meta.models.priors import Prior, get_default_bias_priors, get_weakly_informative_priors


@dataclass
class T3MetaAnalysis:
    """
    Main class for conducting T3-Meta analyses.
    
    T3-Meta is a meta-analytic framework that treats published studies
    as imperfect emulations of a single, clearly defined target trial.
    
    Key Features:
        - Define target trial specification
        - Register studies with design features
        - Align estimands across studies
        - Fit bias-aware hierarchical models
        - Compute target trial effect estimates
        - Conduct sensitivity analyses
    
    Attributes:
        target_trial: The target trial specification
        registry: Container for registered studies
        model_type: 'frequentist' or 'bayesian'
        method: Estimation method (REML, DL, ML, FE for frequentist; 
                laplace, gibbs for bayesian)
        feature_names: Names of design features to include
        align_estimands: Whether to align effect measures
        align_time: Whether to align time horizons
        default_baseline_risk: Default baseline risk for conversions
        priors: Prior specifications for Bayesian model
        ci_level: Confidence/credible interval level
        verbose: Print progress messages
    
    Example:
        >>> from t3meta import T3MetaAnalysis, TargetTrial, Study
        >>> 
        >>> # Define target trial
        >>> target = TargetTrial(
        ...     name="Ideal GLP-1RA Trial",
        ...     population="Adults with T2DM",
        ...     intervention="GLP-1RA initiation",
        ...     comparator="No GLP-1RA",
        ...     outcome="MACE",
        ...     time_horizon_months=36,
        ...     effect_measure="HR"
        ... )
        >>> 
        >>> # Create analysis
        >>> meta = T3MetaAnalysis(target_trial=target)
        >>> 
        >>> # Add studies
        >>> meta.add_study(Study(
        ...     name="LEADER",
        ...     effect_estimate=0.87,
        ...     effect_measure="HR",
        ...     ci_lower=0.78, ci_upper=0.97,
        ...     design_features={"is_rct": True}
        ... ))
        >>> 
        >>> # Fit model
        >>> results = meta.fit()
        >>> print(results.summary_table())
    """
    
    target_trial: Optional[TargetTrial] = None
    registry: StudyRegistry = field(default_factory=StudyRegistry)
    model_type: Literal["frequentist", "bayesian"] = "frequentist"
    method: str = "REML"
    feature_names: Optional[List[str]] = None
    align_estimands: bool = True
    align_time: bool = False
    default_baseline_risk: float = 0.1
    priors: Optional[Dict[str, Prior]] = None
    ci_level: float = 0.95
    verbose: bool = False
    
    # Internal state
    _model: Optional[Union[FrequentistModel, BayesianModel]] = field(default=None, repr=False)
    _results: Optional[ModelResults] = field(default=None, repr=False)
    _aligned_estimates: Optional[np.ndarray] = field(default=None, repr=False)
    _aligned_ses: Optional[np.ndarray] = field(default=None, repr=False)
    _design_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize the analysis."""
        if self.target_trial is not None:
            self.registry.target_trial = self.target_trial
        
        if self.feature_names is None:
            self.feature_names = DesignFeatures.standard_feature_names()
    
    @property
    def n_studies(self) -> int:
        """Number of registered studies."""
        return len(self.registry)
    
    @property
    def studies(self) -> List[Study]:
        """List of registered studies."""
        return self.registry.studies
    
    @property
    def results(self) -> Optional[ModelResults]:
        """Fitted model results."""
        return self._results
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._results is not None
    
    def add_study(self, study: Study) -> T3MetaAnalysis:
        """
        Add a study to the analysis.
        
        Args:
            study: Study object to add
            
        Returns:
            self for method chaining
        """
        self.registry.add(study)
        self._results = None  # Invalidate cached results
        return self
    
    def add_studies(self, studies: List[Study]) -> T3MetaAnalysis:
        """
        Add multiple studies.
        
        Args:
            studies: List of Study objects
            
        Returns:
            self for method chaining
        """
        for study in studies:
            self.add_study(study)
        return self
    
    def remove_study(self, name: str) -> Study:
        """
        Remove a study by name.
        
        Args:
            name: Study name
            
        Returns:
            The removed study
        """
        study = self.registry.remove(name)
        self._results = None
        return study
    
    def set_target_trial(self, target: TargetTrial) -> T3MetaAnalysis:
        """
        Set or update the target trial specification.
        
        Args:
            target: TargetTrial specification
            
        Returns:
            self for method chaining
        """
        self.target_trial = target
        self.registry.target_trial = target
        self._results = None
        return self
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model fitting.
        
        Returns:
            Tuple of (estimates, standard_errors, design_matrix)
        """
        if self.n_studies == 0:
            raise ValueError("No studies registered")
        
        # Get estimates and SEs
        if self.align_estimands and self.target_trial is not None:
            estimates, ses = self._align_estimates()
        else:
            estimates = self.registry.get_estimates()
            ses = self.registry.get_standard_errors()
        
        self._aligned_estimates = estimates
        self._aligned_ses = ses
        
        # Build design matrix
        X = self.registry.get_design_matrix(
            feature_names=self.feature_names,
            include_intercept=True
        )
        self._design_matrix = X
        
        return estimates, ses, X
    
    def _align_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align study estimates to target trial estimand.
        
        Returns:
            Tuple of (aligned_estimates, aligned_ses) on analysis scale
        """
        from t3meta.alignment.effect_measures import align_to_target_measure
        
        target_measure = self.target_trial.effect_measure
        
        aligned = align_to_target_measure(
            self.registry.studies,
            target_measure,
            use_study_baseline_risk=True,
            default_baseline_risk=self.default_baseline_risk
        )
        
        estimates = np.array([a[0] for a in aligned])
        ses = np.array([a[1] for a in aligned])
        
        # Handle any NaNs
        nan_mask = np.isnan(estimates) | np.isnan(ses)
        if np.any(nan_mask):
            warnings.warn(
                f"Could not align {nan_mask.sum()} studies; using original estimates"
            )
            for i in np.where(nan_mask)[0]:
                estimates[i] = self.registry.studies[i].log_estimate
                ses[i] = self.registry.studies[i].log_se
        
        return estimates, ses
    
    def fit(
        self,
        method: Optional[str] = None,
        model_type: Optional[str] = None,
        priors: Optional[Dict[str, Prior]] = None,
        **kwargs
    ) -> ModelResults:
        """
        Fit the T3-Meta model.
        
        Args:
            method: Override estimation method
            model_type: Override model type
            priors: Override priors (for Bayesian)
            **kwargs: Additional arguments passed to model
            
        Returns:
            ModelResults object
        """
        if method is not None:
            self.method = method
        if model_type is not None:
            self.model_type = model_type
        if priors is not None:
            self.priors = priors
        
        # Prepare data
        estimates, ses, X = self._prepare_data()
        
        # Feature names (excluding intercept)
        feature_names = ["intercept"] + self.feature_names
        
        if self.verbose:
            print(f"Fitting {self.model_type} model with method={self.method}")
            print(f"  N studies: {self.n_studies}")
            print(f"  N features: {X.shape[1] - 1}")
        
        # Create and fit model
        if self.model_type == "bayesian":
            if self.priors is None:
                self.priors = get_weakly_informative_priors(
                    X.shape[1] - 1,
                    effect_measure=self.target_trial.effect_measure.value if self.target_trial else "log_HR"
                )
            
            self._model = BayesianModel(
                priors=self.priors,
                sampler=self.method if self.method in ["laplace", "gibbs", "metropolis"] else "laplace",
                ci_level=self.ci_level,
                **kwargs
            )
        else:
            self._model = FrequentistModel(
                method=self.method,
                ci_level=self.ci_level,
                **kwargs
            )
        
        self._results = self._model.fit(
            y=estimates,
            se=ses,
            X=X,
            feature_names=feature_names
        )
        
        return self._results
    
    def fit_frequentist(self, method: str = "REML", **kwargs) -> ModelResults:
        """Convenience method for frequentist fitting."""
        return self.fit(model_type="frequentist", method=method, **kwargs)
    
    def fit_bayesian(
        self, 
        method: str = "laplace",
        priors: Optional[Dict[str, Prior]] = None,
        **kwargs
    ) -> ModelResults:
        """Convenience method for Bayesian fitting."""
        return self.fit(model_type="bayesian", method=method, priors=priors, **kwargs)
    
    def predict_target_effect(
        self,
        target_design: Optional[np.ndarray] = None
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Predict effect for the ideal target trial.
        
        Args:
            target_design: Design features for target (default: all zeros = ideal)
            
        Returns:
            Tuple of (predicted_effect, (ci_lower, ci_upper))
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if target_design is None:
            # Ideal trial: intercept = 1, all design features = 0
            target_design = np.zeros(len(self.feature_names) + 1)
            target_design[0] = 1.0
        
        pred, pred_se = self._model.predict(target_design.reshape(1, -1))
        
        from scipy import stats
        z = stats.norm.ppf((1 + self.ci_level) / 2)
        ci = (float(pred[0] - z * pred_se[0]), float(pred[0] + z * pred_se[0]))
        
        return float(pred[0]), ci
    
    def get_study_bias(self) -> Dict[str, float]:
        """
        Get estimated bias for each study.
        
        Returns:
            Dictionary mapping study names to bias estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        return {
            study.name: float(bias)
            for study, bias in zip(self.registry.studies, self._results.study_bias)
        }
    
    def get_bias_adjusted_effects(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get bias-adjusted effect estimates for each study.
        
        Returns:
            Dictionary mapping study names to (adjusted_effect, original, bias)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        result = {}
        for i, study in enumerate(self.registry.studies):
            original = self._aligned_estimates[i]
            bias = self._results.study_bias[i]
            adjusted = original - bias
            result[study.name] = (float(adjusted), float(original), float(bias))
        
        return result
    
    def sensitivity_analysis(
        self,
        vary: str = "tau_prior",
        values: Optional[List[Any]] = None,
        **kwargs
    ) -> List[ModelResults]:
        """
        Conduct sensitivity analysis by varying model parameters.
        
        Args:
            vary: Parameter to vary ('tau_prior', 'method', 'feature_set')
            values: List of values to try
            **kwargs: Additional arguments
            
        Returns:
            List of ModelResults for each value
        """
        results = []
        
        if vary == "method":
            if values is None:
                values = ["REML", "DL", "ML", "FE"]
            
            for method in values:
                r = self.fit(method=method, model_type="frequentist")
                results.append(r)
        
        elif vary == "tau_prior":
            if values is None:
                from t3meta.models.priors import HalfNormalPrior, HalfCauchyPrior
                values = [
                    {"tau": HalfNormalPrior(0.25)},
                    {"tau": HalfNormalPrior(0.5)},
                    {"tau": HalfNormalPrior(1.0)},
                    {"tau": HalfCauchyPrior(0.5)},
                ]
            
            for prior_spec in values:
                priors = self.priors.copy() if self.priors else {}
                priors.update(prior_spec)
                r = self.fit(model_type="bayesian", priors=priors)
                results.append(r)
        
        elif vary == "feature_set":
            if values is None:
                values = [
                    ["is_rct"],
                    ["is_rct", "has_immortal_time_bias"],
                    ["is_rct", "has_immortal_time_bias", "confounding_iptw"],
                    self.feature_names,
                ]
            
            original_features = self.feature_names
            for features in values:
                self.feature_names = features
                r = self.fit()
                results.append(r)
            self.feature_names = original_features
        
        elif vary == "exclude_study":
            # Leave-one-out sensitivity
            for study in self.registry.studies:
                temp_registry = self.registry.filter(lambda s: s.name != study.name)
                temp_analysis = T3MetaAnalysis(
                    target_trial=self.target_trial,
                    registry=temp_registry,
                    model_type=self.model_type,
                    method=self.method,
                    feature_names=self.feature_names,
                    priors=self.priors,
                )
                r = temp_analysis.fit()
                r.extra["excluded_study"] = study.name
                results.append(r)
        
        else:
            raise ValueError(f"Unknown sensitivity parameter: {vary}")
        
        return results
    
    def leave_one_out(self) -> Dict[str, ModelResults]:
        """
        Conduct leave-one-out analysis.
        
        Returns:
            Dictionary mapping excluded study name to results
        """
        results = self.sensitivity_analysis(vary="exclude_study")
        return {
            r.extra["excluded_study"]: r for r in results
        }
    
    def cumulative_meta_analysis(
        self,
        order_by: str = "year"
    ) -> List[Tuple[str, ModelResults]]:
        """
        Conduct cumulative meta-analysis.
        
        Args:
            order_by: How to order studies ('year', 'sample_size', 'precision')
            
        Returns:
            List of (study_name, cumulative_results) tuples
        """
        # Sort studies
        if order_by == "year":
            sorted_studies = sorted(
                self.registry.studies,
                key=lambda s: s.year or 9999
            )
        elif order_by == "sample_size":
            sorted_studies = sorted(
                self.registry.studies,
                key=lambda s: s.n_total or 0
            )
        elif order_by == "precision":
            sorted_studies = sorted(
                self.registry.studies,
                key=lambda s: 1 / s.variance
            )
        else:
            sorted_studies = self.registry.studies
        
        results = []
        cumulative = StudyRegistry(target_trial=self.target_trial)
        
        for study in sorted_studies:
            cumulative.add(study)
            
            if len(cumulative) < 2:
                continue  # Need at least 2 studies
            
            temp_analysis = T3MetaAnalysis(
                target_trial=self.target_trial,
                registry=cumulative,
                model_type=self.model_type,
                method=self.method,
                feature_names=self.feature_names,
                priors=self.priors,
            )
            
            try:
                r = temp_analysis.fit()
                results.append((study.name, r))
            except Exception as e:
                warnings.warn(f"Could not fit cumulative analysis after {study.name}: {e}")
        
        return results
    
    def subgroup_analysis(
        self,
        by: str,
        categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, ModelResults]:
        """
        Conduct subgroup analysis.
        
        Args:
            by: Feature to stratify by (e.g., 'is_rct', 'geographic_region')
            categories: Optional mapping of category names to study names
            
        Returns:
            Dictionary mapping subgroup names to results
        """
        results = {}
        
        if by == "is_rct":
            for is_rct, label in [(True, "RCT"), (False, "Observational")]:
                sub_registry = self.registry.filter(
                    lambda s: s.design_map and s.design_map.features.is_rct == is_rct
                )
                
                if len(sub_registry) < 2:
                    continue
                
                temp = T3MetaAnalysis(
                    target_trial=self.target_trial,
                    registry=sub_registry,
                    model_type=self.model_type,
                    method=self.method,
                    feature_names=self.feature_names,
                    priors=self.priors,
                )
                results[label] = temp.fit()
        
        elif categories is not None:
            for cat_name, study_names in categories.items():
                sub_registry = self.registry.filter(
                    lambda s: s.name in study_names
                )
                
                if len(sub_registry) < 2:
                    continue
                
                temp = T3MetaAnalysis(
                    target_trial=self.target_trial,
                    registry=sub_registry,
                    model_type=self.model_type,
                    method=self.method,
                    feature_names=self.feature_names,
                    priors=self.priors,
                )
                results[cat_name] = temp.fit()
        
        return results
    
    def summary(self) -> str:
        """Generate summary of the analysis."""
        lines = []
        
        if self.target_trial:
            lines.append(self.target_trial.describe())
            lines.append("")
        
        lines.append(self.registry.describe())
        lines.append("")
        
        if self.is_fitted:
            lines.append(self._results.summary_table())
        else:
            lines.append("Model not yet fitted. Call .fit() to estimate effects.")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            "target_trial": self.target_trial.to_dict() if self.target_trial else None,
            "registry": self.registry.to_dict(),
            "model_type": self.model_type,
            "method": self.method,
            "feature_names": self.feature_names,
            "align_estimands": self.align_estimands,
            "align_time": self.align_time,
            "default_baseline_risk": self.default_baseline_risk,
            "ci_level": self.ci_level,
            "results": self._results.to_dict() if self._results else None,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> T3MetaAnalysis:
        """Create from dictionary."""
        target_trial = None
        if d.get("target_trial"):
            target_trial = TargetTrial.from_dict(d["target_trial"])
        
        registry = StudyRegistry.from_dict(d.get("registry", {"studies": []}))
        
        return cls(
            target_trial=target_trial,
            registry=registry,
            model_type=d.get("model_type", "frequentist"),
            method=d.get("method", "REML"),
            feature_names=d.get("feature_names"),
            align_estimands=d.get("align_estimands", True),
            align_time=d.get("align_time", False),
            default_baseline_risk=d.get("default_baseline_risk", 0.1),
            ci_level=d.get("ci_level", 0.95),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> T3MetaAnalysis:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
