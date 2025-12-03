"""Core data structures for T3-Meta."""

from t3meta.core.target_trial import TargetTrial, TargetTrialSpec
from t3meta.core.study import Study, DesignMap, DesignFeatures
from t3meta.core.estimand import Estimand, EstimandType, EffectMeasure
from t3meta.core.registry import StudyRegistry

__all__ = [
    "TargetTrial",
    "TargetTrialSpec",
    "Study",
    "DesignMap",
    "DesignFeatures",
    "Estimand",
    "EstimandType",
    "EffectMeasure",
    "StudyRegistry",
]
