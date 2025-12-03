"""
T3-Meta: Target Trial-Centric Meta-Analysis Framework

A meta-analytic framework that treats published studies as imperfect emulations
of a single, clearly defined target trial. Instead of asking "what is the average
effect reported across studies?", T3-Meta asks: "Given a well-specified target trial,
what is our best estimate of its causal effect?"

Key Features:
    - Unify evidence (RCTs, observational, emulated trials) around a single target estimand
    - Make design & analysis differences explicit rather than unstructured heterogeneity
    - Model bias structurally via design covariates
    - Provide transportable estimates for chosen target populations

Example Usage:
    >>> from t3meta import TargetTrial, Study, T3MetaAnalysis
    >>> 
    >>> # Define target trial
    >>> target = TargetTrial(
    ...     name="Ideal GLP-1RA Trial",
    ...     population="Adults with T2DM, no prior CV events",
    ...     intervention="GLP-1RA initiation within 30 days",
    ...     comparator="No GLP-1RA",
    ...     outcome="MACE (MI, stroke, CV death)",
    ...     time_horizon_months=36,
    ...     estimand_type="ITT"
    ... )
    >>> 
    >>> # Register studies
    >>> study1 = Study(
    ...     name="LEADER Trial",
    ...     effect_estimate=0.87,
    ...     effect_measure="HR",
    ...     ci_lower=0.78, ci_upper=0.97,
    ...     design_features={"is_rct": True, "blinding": "double"}
    ... )
    >>> 
    >>> # Fit model
    >>> meta = T3MetaAnalysis(target_trial=target)
    >>> meta.add_study(study1)
    >>> results = meta.fit()

Author: Ahmed Y. Azzam
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Ahmed Y. Azzam"
__email__ = "ahmed.azzam@wvumedicine.org"

# Core classes
from t3meta.core.target_trial import TargetTrial, TargetTrialSpec
from t3meta.core.study import Study, DesignMap, DesignFeatures
from t3meta.core.estimand import Estimand, EstimandType, EffectMeasure
from t3meta.core.registry import StudyRegistry

# Alignment utilities
from t3meta.alignment.effect_measures import (
    EffectMeasureConverter,
    or_to_rr,
    rr_to_rd,
    hr_to_rd,
    log_transform,
    exp_transform
)
from t3meta.alignment.time_alignment import TimeAligner
from t3meta.alignment.standardization import BaselineRiskStandardizer

# Models
from t3meta.models.base import BaseModel, ModelResults
from t3meta.models.frequentist import FrequentistModel, MixedEffectsModel
from t3meta.models.bayesian import BayesianModel, BayesianResults
from t3meta.models.priors import Prior, NormalPrior, HalfNormalPrior, UniformPrior

# Main analysis class
from t3meta.models.t3_meta_analysis import T3MetaAnalysis

# Diagnostics
from t3meta.diagnostics.heterogeneity import (
    HeterogeneityAnalysis,
    decompose_heterogeneity,
    compute_i_squared,
    compute_tau_squared
)
from t3meta.diagnostics.influence import InfluenceAnalysis, leave_one_out
from t3meta.diagnostics.sensitivity import SensitivityAnalysis

# Visualization
from t3meta.visualization.forest import forest_plot, t3_forest_plot
from t3meta.visualization.bias import bias_contribution_plot, bias_heatmap
from t3meta.visualization.funnel import funnel_plot, contour_enhanced_funnel

# I/O
from t3meta.io.readers import (
    read_csv,
    read_json,
    read_excel,
    studies_from_dataframe
)
from t3meta.io.writers import (
    write_csv,
    write_json,
    write_excel,
    export_to_prisma
)
from t3meta.io.schema import T3MetaSchema, validate_input

# Utilities
from t3meta.utils import (
    se_from_ci,
    ci_from_se,
    weighted_mean,
    pooled_estimate_fixed,
    pooled_estimate_random,
    format_estimate
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core classes
    "TargetTrial",
    "TargetTrialSpec",
    "Study",
    "DesignMap",
    "DesignFeatures",
    "Estimand",
    "EstimandType",
    "EffectMeasure",
    "StudyRegistry",
    
    # Main analysis
    "T3MetaAnalysis",
    
    # Alignment
    "EffectMeasureConverter",
    "or_to_rr",
    "rr_to_rd",
    "hr_to_rd",
    "log_transform",
    "exp_transform",
    "TimeAligner",
    "BaselineRiskStandardizer",
    
    # Models
    "BaseModel",
    "ModelResults",
    "FrequentistModel",
    "MixedEffectsModel",
    "BayesianModel",
    "BayesianResults",
    "Prior",
    "NormalPrior",
    "HalfNormalPrior",
    "UniformPrior",
    
    # Diagnostics
    "HeterogeneityAnalysis",
    "decompose_heterogeneity",
    "compute_i_squared",
    "compute_tau_squared",
    "InfluenceAnalysis",
    "leave_one_out",
    "SensitivityAnalysis",
    
    # Visualization
    "forest_plot",
    "t3_forest_plot",
    "bias_contribution_plot",
    "bias_heatmap",
    "funnel_plot",
    "contour_enhanced_funnel",
    
    # I/O
    "read_csv",
    "read_json",
    "read_excel",
    "studies_from_dataframe",
    "write_csv",
    "write_json",
    "write_excel",
    "export_to_prisma",
    "T3MetaSchema",
    "validate_input",
    
    # Utilities
    "se_from_ci",
    "ci_from_se",
    "weighted_mean",
    "pooled_estimate_fixed",
    "pooled_estimate_random",
    "format_estimate",
]
