"""
Study and DesignMap classes for T3-Meta.

This module defines how individual studies are represented and how their
design features are mapped relative to the target trial.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum
import numpy as np
import json

from t3meta.core.estimand import Estimand, EstimandType, EffectMeasure


class StudyType(Enum):
    """Classification of study designs."""
    RCT = "rct"
    OBSERVATIONAL_COHORT = "observational_cohort"
    CASE_CONTROL = "case_control"
    EMULATED_TARGET_TRIAL = "emulated_target_trial"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    SINGLE_ARM = "single_arm"
    CROSSOVER = "crossover"
    CLUSTER_RCT = "cluster_rct"
    PRAGMATIC_TRIAL = "pragmatic_trial"


class ConfoundingControl(Enum):
    """Methods for controlling confounding."""
    RANDOMIZATION = "randomization"
    NONE = "none"
    BASIC_REGRESSION = "basic_regression"
    MULTIVARIABLE_REGRESSION = "multivariable_regression"
    PROPENSITY_SCORE_MATCHING = "ps_matching"
    PROPENSITY_SCORE_WEIGHTING = "ps_weighting"
    PROPENSITY_SCORE_STRATIFICATION = "ps_stratification"
    INVERSE_PROBABILITY_WEIGHTING = "iptw"
    DOUBLY_ROBUST = "doubly_robust"
    INSTRUMENTAL_VARIABLE = "iv"
    REGRESSION_DISCONTINUITY = "rdd"
    DIFFERENCE_IN_DIFFERENCES = "did"
    SYNTHETIC_CONTROL = "synthetic_control"


class TimeZeroDefinition(Enum):
    """How time zero is defined in the study."""
    RANDOMIZATION = "randomization"
    FIRST_PRESCRIPTION = "first_prescription"
    ELIGIBILITY_DATE = "eligibility_date"
    DIAGNOSIS_DATE = "diagnosis_date"
    INDEX_VISIT = "index_visit"
    HOSPITAL_ADMISSION = "hospital_admission"
    UNCLEAR = "unclear"


class BlindingType(Enum):
    """Blinding status of the study."""
    DOUBLE_BLIND = "double_blind"
    SINGLE_BLIND = "single_blind"
    OPEN_LABEL = "open_label"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class DesignFeatures:
    """
    Structured representation of study design features.
    
    These features are used to construct the design covariate matrix X
    in the T3-Meta bias model.
    
    Attributes:
        is_rct: Whether study is a randomized controlled trial
        study_type: Detailed study type classification
        confounding_control: Method used to control confounding
        time_zero: How time zero is defined
        blinding: Blinding status
        has_immortal_time_bias: Risk of immortal time bias
        has_prevalent_user_bias: Risk of prevalent user bias
        outcome_adjudicated: Whether outcomes were adjudicated
        itd_source: Whether using individual-level data
        loss_to_followup_pct: Percentage lost to follow-up
        endpoint_composite: Whether endpoint is composite
        sample_size: Total sample size
        events: Number of events (if applicable)
        median_followup_months: Median follow-up duration
        max_followup_months: Maximum follow-up duration
        year_published: Publication year
        geographic_region: Geographic region(s) of study
        multicenter: Whether study is multicenter
        industry_funded: Whether industry funded
        preregistered: Whether study was preregistered
        custom: Additional custom features
    """
    
    # Core design features
    is_rct: bool = False
    study_type: Union[str, StudyType] = StudyType.OBSERVATIONAL_COHORT
    confounding_control: Union[str, ConfoundingControl] = ConfoundingControl.NONE
    time_zero: Union[str, TimeZeroDefinition] = TimeZeroDefinition.UNCLEAR
    blinding: Union[str, BlindingType] = BlindingType.NOT_APPLICABLE
    
    # Bias indicators
    has_immortal_time_bias: bool = False
    has_prevalent_user_bias: bool = False
    has_time_lag_bias: bool = False
    has_selection_bias: bool = False
    has_information_bias: bool = False
    
    # Quality features
    outcome_adjudicated: bool = False
    ipd_available: bool = False
    intention_to_treat_analysis: bool = True
    
    # Quantitative features
    loss_to_followup_pct: Optional[float] = None
    crossover_pct: Optional[float] = None
    sample_size: Optional[int] = None
    events: Optional[int] = None
    median_followup_months: Optional[float] = None
    max_followup_months: Optional[float] = None
    
    # Study metadata
    year_published: Optional[int] = None
    geographic_region: List[str] = field(default_factory=list)
    multicenter: bool = False
    industry_funded: Optional[bool] = None
    preregistered: Optional[bool] = None
    
    # Custom features for user extension
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string inputs to enums."""
        if isinstance(self.study_type, str):
            self.study_type = StudyType(self.study_type.lower())
        if isinstance(self.confounding_control, str):
            self.confounding_control = ConfoundingControl(self.confounding_control.lower())
        if isinstance(self.time_zero, str):
            try:
                self.time_zero = TimeZeroDefinition(self.time_zero.lower())
            except ValueError:
                self.time_zero = TimeZeroDefinition.UNCLEAR
        if isinstance(self.blinding, str):
            try:
                self.blinding = BlindingType(self.blinding.lower())
            except ValueError:
                self.blinding = BlindingType.NOT_APPLICABLE
    
    def to_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert to numeric feature vector for model input.
        
        Args:
            feature_names: List of feature names to include (default: all standard features)
            
        Returns:
            1D numpy array of feature values
        """
        if feature_names is None:
            feature_names = self.standard_feature_names()
        
        vector = []
        for name in feature_names:
            value = self.get_numeric_value(name)
            vector.append(value)
        
        return np.array(vector, dtype=np.float64)
    
    def get_numeric_value(self, feature_name: str) -> float:
        """Get numeric value for a feature."""
        # Binary features
        binary_features = {
            "is_rct": self.is_rct,
            "outcome_adjudicated": self.outcome_adjudicated,
            "ipd_available": self.ipd_available,
            "intention_to_treat_analysis": self.intention_to_treat_analysis,
            "has_immortal_time_bias": self.has_immortal_time_bias,
            "has_prevalent_user_bias": self.has_prevalent_user_bias,
            "has_time_lag_bias": self.has_time_lag_bias,
            "has_selection_bias": self.has_selection_bias,
            "has_information_bias": self.has_information_bias,
            "multicenter": self.multicenter,
            "industry_funded": self.industry_funded if self.industry_funded is not None else False,
            "preregistered": self.preregistered if self.preregistered is not None else False,
        }
        
        if feature_name in binary_features:
            return 1.0 if binary_features[feature_name] else 0.0
        
        # Continuous features (with defaults for missing)
        continuous_features = {
            "loss_to_followup_pct": self.loss_to_followup_pct if self.loss_to_followup_pct is not None else 0.0,
            "crossover_pct": self.crossover_pct if self.crossover_pct is not None else 0.0,
            "log_sample_size": np.log(self.sample_size) if self.sample_size and self.sample_size > 0 else 0.0,
            "log_events": np.log(self.events) if self.events and self.events > 0 else 0.0,
            "median_followup_months": self.median_followup_months if self.median_followup_months is not None else 0.0,
            "year_published": (self.year_published - 2000) / 10.0 if self.year_published else 0.0,
        }
        
        if feature_name in continuous_features:
            return continuous_features[feature_name]
        
        # Categorical features (one-hot encoded internally)
        if feature_name.startswith("study_type_"):
            target_type = feature_name.replace("study_type_", "")
            return 1.0 if self.study_type.value == target_type else 0.0
        
        if feature_name.startswith("confounding_"):
            target_method = feature_name.replace("confounding_", "")
            return 1.0 if self.confounding_control.value == target_method else 0.0
        
        if feature_name.startswith("blinding_"):
            target_blinding = feature_name.replace("blinding_", "")
            return 1.0 if self.blinding.value == target_blinding else 0.0
        
        # Custom features
        if feature_name in self.custom:
            val = self.custom[feature_name]
            if isinstance(val, bool):
                return 1.0 if val else 0.0
            if isinstance(val, (int, float)):
                return float(val)
            return 0.0
        
        return 0.0
    
    @staticmethod
    def standard_feature_names() -> List[str]:
        """Return list of standard feature names."""
        return [
            "is_rct",
            "outcome_adjudicated",
            "has_immortal_time_bias",
            "has_prevalent_user_bias",
            "loss_to_followup_pct",
            "log_sample_size",
            "median_followup_months",
            "confounding_iptw",
            "confounding_doubly_robust",
            "confounding_ps_matching",
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_rct": self.is_rct,
            "study_type": self.study_type.value if isinstance(self.study_type, StudyType) else self.study_type,
            "confounding_control": self.confounding_control.value if isinstance(self.confounding_control, ConfoundingControl) else self.confounding_control,
            "time_zero": self.time_zero.value if isinstance(self.time_zero, TimeZeroDefinition) else self.time_zero,
            "blinding": self.blinding.value if isinstance(self.blinding, BlindingType) else self.blinding,
            "has_immortal_time_bias": self.has_immortal_time_bias,
            "has_prevalent_user_bias": self.has_prevalent_user_bias,
            "has_time_lag_bias": self.has_time_lag_bias,
            "has_selection_bias": self.has_selection_bias,
            "has_information_bias": self.has_information_bias,
            "outcome_adjudicated": self.outcome_adjudicated,
            "ipd_available": self.ipd_available,
            "intention_to_treat_analysis": self.intention_to_treat_analysis,
            "loss_to_followup_pct": self.loss_to_followup_pct,
            "crossover_pct": self.crossover_pct,
            "sample_size": self.sample_size,
            "events": self.events,
            "median_followup_months": self.median_followup_months,
            "max_followup_months": self.max_followup_months,
            "year_published": self.year_published,
            "geographic_region": self.geographic_region,
            "multicenter": self.multicenter,
            "industry_funded": self.industry_funded,
            "preregistered": self.preregistered,
            "custom": self.custom,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DesignFeatures:
        """Create from dictionary."""
        return cls(**d)


@dataclass
class DesignMap:
    """
    Maps a study's design to the target trial template.
    
    For each study, we record how it approximates (or deviates from)
    the target trial across key dimensions.
    
    Attributes:
        features: Structured design features
        eligibility_match: How well eligibility criteria match target (0-1)
        intervention_match: How well intervention matches target (0-1)
        comparator_match: How well comparator matches target (0-1)
        outcome_match: How well outcome definition matches target (0-1)
        time_zero_match: How well time zero matches target (0-1)
        followup_match: How well follow-up matches target time horizon (0-1)
        estimand_match: How well estimand matches target (0-1)
        overall_similarity: Computed overall similarity to target trial
        deviations: Text description of key deviations from target
        notes: Additional notes
    """
    
    features: DesignFeatures = field(default_factory=DesignFeatures)
    
    # Match scores (0-1 scale, 1 = perfect match)
    eligibility_match: float = 0.5
    intervention_match: float = 0.5
    comparator_match: float = 0.5
    outcome_match: float = 0.5
    time_zero_match: float = 0.5
    followup_match: float = 0.5
    estimand_match: float = 0.5
    
    # Deviation descriptions
    deviations: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    
    def __post_init__(self):
        """Validate match scores."""
        for attr in ['eligibility_match', 'intervention_match', 'comparator_match',
                     'outcome_match', 'time_zero_match', 'followup_match', 'estimand_match']:
            value = getattr(self, attr)
            if value < 0 or value > 1:
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")
    
    @property
    def overall_similarity(self) -> float:
        """Compute weighted overall similarity score."""
        weights = {
            'eligibility_match': 0.15,
            'intervention_match': 0.20,
            'comparator_match': 0.15,
            'outcome_match': 0.15,
            'time_zero_match': 0.10,
            'followup_match': 0.10,
            'estimand_match': 0.15,
        }
        
        total = sum(
            weights[k] * getattr(self, k) 
            for k in weights
        )
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": self.features.to_dict(),
            "eligibility_match": self.eligibility_match,
            "intervention_match": self.intervention_match,
            "comparator_match": self.comparator_match,
            "outcome_match": self.outcome_match,
            "time_zero_match": self.time_zero_match,
            "followup_match": self.followup_match,
            "estimand_match": self.estimand_match,
            "overall_similarity": self.overall_similarity,
            "deviations": self.deviations,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DesignMap:
        """Create from dictionary."""
        features = DesignFeatures.from_dict(d.get("features", {}))
        return cls(
            features=features,
            eligibility_match=d.get("eligibility_match", 0.5),
            intervention_match=d.get("intervention_match", 0.5),
            comparator_match=d.get("comparator_match", 0.5),
            outcome_match=d.get("outcome_match", 0.5),
            time_zero_match=d.get("time_zero_match", 0.5),
            followup_match=d.get("followup_match", 0.5),
            estimand_match=d.get("estimand_match", 0.5),
            deviations=d.get("deviations", {}),
            notes=d.get("notes", ""),
        )


@dataclass
class Study:
    """
    Representation of a single study in the meta-analysis.
    
    Each study provides an estimate that is a biased, noisy measurement
    of the target effect: θ̂_j ≈ θ* + B_j + ε_j
    
    Attributes:
        name: Study identifier/name
        effect_estimate: Point estimate (on original scale)
        effect_measure: Type of effect measure (HR, RR, RD, etc.)
        se: Standard error (on analysis scale)
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        ci_level: Confidence level (default 0.95)
        design_map: DesignMap object linking to target trial
        design_features: Alternative: direct DesignFeatures (converted to DesignMap)
        effect_modifiers: Study-level effect modifier values (Z_j)
        baseline_risk: Baseline risk in control group (for conversions)
        n_treatment: Sample size in treatment arm
        n_control: Sample size in control arm
        events_treatment: Events in treatment arm
        events_control: Events in control arm
        estimand: Study's estimand specification
        authors: Author list
        year: Publication year
        journal: Publication journal
        doi: Digital object identifier
        notes: Additional notes
    """
    
    name: str
    effect_estimate: float
    effect_measure: Union[str, EffectMeasure] = EffectMeasure.HAZARD_RATIO
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    
    # Design mapping
    design_map: Optional[DesignMap] = None
    design_features: Optional[Union[Dict[str, Any], DesignFeatures]] = None
    
    # Effect modifiers for transportability
    effect_modifiers: Dict[str, float] = field(default_factory=dict)
    
    # For effect measure conversions
    baseline_risk: Optional[float] = None
    
    # Sample size and events
    n_treatment: Optional[int] = None
    n_control: Optional[int] = None
    n_total: Optional[int] = None
    events_treatment: Optional[int] = None
    events_control: Optional[int] = None
    events_total: Optional[int] = None
    
    # Study's own estimand
    estimand: Optional[Estimand] = None
    
    # Bibliographic
    authors: str = ""
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    notes: str = ""
    
    # Internal
    _log_estimate: Optional[float] = field(default=None, repr=False)
    _log_se: Optional[float] = field(default=None, repr=False)
    _weight: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and process inputs."""
        # Convert effect measure
        if isinstance(self.effect_measure, str):
            self.effect_measure = EffectMeasure.from_string(self.effect_measure)
        
        # Compute SE from CI if not provided
        if self.se is None and self.ci_lower is not None and self.ci_upper is not None:
            self._compute_se_from_ci()
        
        # Compute log-scale values for ratio measures
        self._compute_log_scale()
        
        # Convert design_features dict to DesignFeatures object
        if self.design_features is not None and not isinstance(self.design_features, DesignFeatures):
            self.design_features = DesignFeatures.from_dict(self.design_features)
        
        # Create DesignMap from design_features if not provided
        if self.design_map is None and self.design_features is not None:
            self.design_map = DesignMap(features=self.design_features)
        elif self.design_map is None:
            self.design_map = DesignMap()
        
        # Compute total sample size if not provided
        if self.n_total is None:
            if self.n_treatment is not None and self.n_control is not None:
                self.n_total = self.n_treatment + self.n_control
        
        # Compute total events if not provided
        if self.events_total is None:
            if self.events_treatment is not None and self.events_control is not None:
                self.events_total = self.events_treatment + self.events_control
    
    def _compute_se_from_ci(self):
        """Compute standard error from confidence interval."""
        from scipy import stats
        
        z = stats.norm.ppf((1 + self.ci_level) / 2)
        
        if self.effect_measure.is_ratio_measure():
            # CI is on original scale, SE is on log scale
            log_lower = np.log(self.ci_lower)
            log_upper = np.log(self.ci_upper)
            self.se = (log_upper - log_lower) / (2 * z)
        else:
            self.se = (self.ci_upper - self.ci_lower) / (2 * z)
    
    def _compute_log_scale(self):
        """Compute log-scale estimate and SE for ratio measures."""
        if self.effect_measure.is_ratio_measure():
            self._log_estimate = np.log(self.effect_estimate)
            self._log_se = self.se  # Already on log scale
        else:
            self._log_estimate = self.effect_estimate
            self._log_se = self.se
    
    @property
    def log_estimate(self) -> float:
        """Get estimate on log/analysis scale."""
        if self._log_estimate is None:
            self._compute_log_scale()
        return self._log_estimate
    
    @property
    def log_se(self) -> float:
        """Get SE on log/analysis scale."""
        if self._log_se is None:
            self._compute_log_scale()
        return self._log_se
    
    @property
    def variance(self) -> float:
        """Get sampling variance on analysis scale."""
        return self.log_se ** 2
    
    @property
    def weight(self) -> float:
        """Get inverse-variance weight."""
        if self._weight is not None:
            return self._weight
        return 1.0 / self.variance
    
    @weight.setter
    def weight(self, value: float):
        self._weight = value
    
    @property
    def sample_size(self) -> Optional[int]:
        """Get total sample size."""
        return self.n_total
    
    def get_ci(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval.
        
        Args:
            level: Confidence level
            
        Returns:
            Tuple of (lower, upper) bounds on original scale
        """
        from scipy import stats
        
        z = stats.norm.ppf((1 + level) / 2)
        
        if self.effect_measure.is_ratio_measure():
            log_lower = self.log_estimate - z * self.log_se
            log_upper = self.log_estimate + z * self.log_se
            return np.exp(log_lower), np.exp(log_upper)
        else:
            lower = self.log_estimate - z * self.log_se
            upper = self.log_estimate + z * self.log_se
            return lower, upper
    
    def get_design_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Get design feature vector for this study."""
        if self.design_map is None:
            return np.zeros(len(feature_names or DesignFeatures.standard_feature_names()))
        return self.design_map.features.to_vector(feature_names)
    
    def get_effect_modifier_vector(self, modifier_names: List[str]) -> np.ndarray:
        """Get effect modifier vector for this study."""
        return np.array([
            self.effect_modifiers.get(name, 0.0) 
            for name in modifier_names
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "effect_estimate": self.effect_estimate,
            "effect_measure": self.effect_measure.value,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "design_map": self.design_map.to_dict() if self.design_map else None,
            "effect_modifiers": self.effect_modifiers,
            "baseline_risk": self.baseline_risk,
            "n_treatment": self.n_treatment,
            "n_control": self.n_control,
            "n_total": self.n_total,
            "events_treatment": self.events_treatment,
            "events_control": self.events_control,
            "events_total": self.events_total,
            "estimand": self.estimand.to_dict() if self.estimand else None,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "notes": self.notes,
        }
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Study:
        """Create from dictionary."""
        design_map = None
        if d.get("design_map"):
            design_map = DesignMap.from_dict(d["design_map"])
        
        estimand = None
        if d.get("estimand"):
            estimand = Estimand.from_dict(d["estimand"])
        
        return cls(
            name=d["name"],
            effect_estimate=d["effect_estimate"],
            effect_measure=d.get("effect_measure", "HR"),
            se=d.get("se"),
            ci_lower=d.get("ci_lower"),
            ci_upper=d.get("ci_upper"),
            ci_level=d.get("ci_level", 0.95),
            design_map=design_map,
            effect_modifiers=d.get("effect_modifiers", {}),
            baseline_risk=d.get("baseline_risk"),
            n_treatment=d.get("n_treatment"),
            n_control=d.get("n_control"),
            n_total=d.get("n_total"),
            events_treatment=d.get("events_treatment"),
            events_control=d.get("events_control"),
            events_total=d.get("events_total"),
            estimand=estimand,
            authors=d.get("authors", ""),
            year=d.get("year"),
            journal=d.get("journal", ""),
            doi=d.get("doi", ""),
            notes=d.get("notes", ""),
        )
    
    def describe(self) -> str:
        """Generate human-readable description."""
        ci = self.get_ci()
        lines = [
            f"Study: {self.name}",
            f"Effect: {self.effect_estimate:.3f} ({self.effect_measure.value})",
            f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]",
            f"SE: {self.se:.4f}" if self.se else "",
        ]
        
        if self.n_total:
            lines.append(f"Sample size: {self.n_total}")
        if self.events_total:
            lines.append(f"Events: {self.events_total}")
        if self.design_map:
            lines.append(f"Design similarity: {self.design_map.overall_similarity:.2f}")
        
        return "\n".join(line for line in lines if line)


def register_study(
    name: str,
    effect_estimate: float,
    effect_measure: str = "HR",
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    se: Optional[float] = None,
    design_features: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Study:
    """
    Convenience function to register a study.
    
    Args:
        name: Study identifier
        effect_estimate: Point estimate
        effect_measure: Type of effect (HR, RR, RD, etc.)
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        se: Standard error
        design_features: Dict of design features
        **kwargs: Additional Study arguments
        
    Returns:
        Study instance
    """
    return Study(
        name=name,
        effect_estimate=effect_estimate,
        effect_measure=effect_measure,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        design_features=design_features,
        **kwargs
    )
