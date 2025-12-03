"""
Estimand definitions for T3-Meta.

This module defines the types of estimands and effect measures used in meta-analysis,
following ICH E9(R1) framework terminology and causal inference conventions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Union
import numpy as np


class EffectMeasure(Enum):
    """Types of effect measures used in meta-analysis."""
    
    # Ratio measures (typically analyzed on log scale)
    HAZARD_RATIO = "HR"
    RISK_RATIO = "RR"
    ODDS_RATIO = "OR"
    INCIDENCE_RATE_RATIO = "IRR"
    
    # Difference measures (analyzed on natural scale)
    RISK_DIFFERENCE = "RD"
    MEAN_DIFFERENCE = "MD"
    STANDARDIZED_MEAN_DIFFERENCE = "SMD"
    
    # Time-to-event measures
    RESTRICTED_MEAN_SURVIVAL_TIME = "RMST"
    MEDIAN_SURVIVAL_RATIO = "MSR"
    
    # Other
    CORRELATION = "r"
    
    @classmethod
    def from_string(cls, s: str) -> EffectMeasure:
        """Convert string to EffectMeasure enum."""
        s_upper = s.upper().strip()
        # Handle common aliases
        aliases = {
            "HAZARD RATIO": "HR",
            "RISK RATIO": "RR", 
            "RELATIVE RISK": "RR",
            "ODDS RATIO": "OR",
            "RISK DIFFERENCE": "RD",
            "MEAN DIFFERENCE": "MD",
            "STANDARDIZED MEAN DIFFERENCE": "SMD",
            "HEDGES G": "SMD",
            "HEDGES' G": "SMD",
            "COHEN D": "SMD",
            "COHEN'S D": "SMD",
            "RMST": "RMST",
            "INCIDENCE RATE RATIO": "IRR",
            "RATE RATIO": "IRR",
        }
        if s_upper in aliases:
            s_upper = aliases[s_upper]
        
        for member in cls:
            if member.value == s_upper or member.name == s_upper:
                return member
        raise ValueError(f"Unknown effect measure: {s}")
    
    def is_ratio_measure(self) -> bool:
        """Check if this is a ratio measure (analyzed on log scale)."""
        return self in {
            EffectMeasure.HAZARD_RATIO,
            EffectMeasure.RISK_RATIO,
            EffectMeasure.ODDS_RATIO,
            EffectMeasure.INCIDENCE_RATE_RATIO,
            EffectMeasure.MEDIAN_SURVIVAL_RATIO,
        }
    
    def is_difference_measure(self) -> bool:
        """Check if this is a difference measure."""
        return self in {
            EffectMeasure.RISK_DIFFERENCE,
            EffectMeasure.MEAN_DIFFERENCE,
            EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
            EffectMeasure.RESTRICTED_MEAN_SURVIVAL_TIME,
        }
    
    def null_value(self) -> float:
        """Return the null value (no effect) for this measure."""
        if self.is_ratio_measure():
            return 1.0
        return 0.0
    
    def log_null_value(self) -> float:
        """Return the null value on log scale."""
        return 0.0  # log(1) = 0 for ratios, 0 for differences


class EstimandType(Enum):
    """Types of estimands following ICH E9(R1) framework."""
    
    # Primary estimand types
    INTENTION_TO_TREAT = "ITT"
    PER_PROTOCOL = "PP"
    AS_TREATED = "AT"
    
    # Causal estimand types
    AVERAGE_TREATMENT_EFFECT = "ATE"
    AVERAGE_TREATMENT_EFFECT_TREATED = "ATT"
    AVERAGE_TREATMENT_EFFECT_CONTROLS = "ATC"
    
    # Hypothetical estimands
    HYPOTHETICAL = "HYPO"
    PRINCIPAL_STRATUM = "PS"
    
    # While-on-treatment
    WHILE_ON_TREATMENT = "WOT"
    
    # Composite strategy
    COMPOSITE = "COMP"
    
    @classmethod
    def from_string(cls, s: str) -> EstimandType:
        """Convert string to EstimandType enum."""
        s_upper = s.upper().strip().replace("-", "_").replace(" ", "_")
        aliases = {
            "INTENTION_TO_TREAT": "ITT",
            "PER_PROTOCOL": "PP",
            "AS_TREATED": "AT",
            "ON_TREATMENT": "AT",
            "ATE": "ATE",
            "ATT": "ATT",
            "ATC": "ATC",
            "HYPOTHETICAL": "HYPO",
        }
        if s_upper in aliases:
            s_upper = aliases[s_upper]
        
        for member in cls:
            if member.value == s_upper or member.name == s_upper:
                return member
        raise ValueError(f"Unknown estimand type: {s}")


class InterCurrentEventStrategy(Enum):
    """Strategies for handling intercurrent events (ICE)."""
    
    TREATMENT_POLICY = "treatment_policy"  # ITT-like: ignore ICE
    HYPOTHETICAL = "hypothetical"  # What if ICE hadn't occurred
    COMPOSITE = "composite"  # ICE is part of outcome
    WHILE_ON_TREATMENT = "while_on_treatment"  # Censor at ICE
    PRINCIPAL_STRATUM = "principal_stratum"  # Subset who wouldn't have ICE


@dataclass
class Estimand:
    """
    Full specification of a target estimand.
    
    Following ICH E9(R1), an estimand is defined by:
    - Population
    - Treatment/intervention
    - Endpoint/outcome
    - Summary measure
    - Handling of intercurrent events
    
    Attributes:
        effect_measure: Type of effect measure (HR, RR, RD, etc.)
        estimand_type: ITT, PP, AT, etc.
        population_description: Text description of target population
        treatment_description: Text description of treatment strategy
        outcome_description: Text description of outcome
        time_horizon: Follow-up time in months (or None for variable)
        ice_strategies: Dict mapping intercurrent event types to handling strategies
        summary_function: How effects are summarized (e.g., "marginal", "conditional")
    """
    
    effect_measure: EffectMeasure
    estimand_type: EstimandType
    population_description: str = ""
    treatment_description: str = ""
    outcome_description: str = ""
    time_horizon: Optional[float] = None  # months
    ice_strategies: Dict[str, InterCurrentEventStrategy] = field(default_factory=dict)
    summary_function: str = "marginal"
    
    def __post_init__(self):
        """Validate and convert string inputs."""
        if isinstance(self.effect_measure, str):
            self.effect_measure = EffectMeasure.from_string(self.effect_measure)
        if isinstance(self.estimand_type, str):
            self.estimand_type = EstimandType.from_string(self.estimand_type)
    
    def is_compatible_with(self, other: Estimand, strict: bool = False) -> bool:
        """
        Check if this estimand is compatible with another for meta-analysis.
        
        Args:
            other: Another Estimand to compare
            strict: If True, require exact match; if False, allow convertible measures
            
        Returns:
            True if estimands can be meaningfully combined
        """
        if strict:
            return (
                self.effect_measure == other.effect_measure and
                self.estimand_type == other.estimand_type
            )
        
        # Less strict: same class of effect measure
        same_measure_class = (
            (self.effect_measure.is_ratio_measure() and other.effect_measure.is_ratio_measure()) or
            (self.effect_measure.is_difference_measure() and other.effect_measure.is_difference_measure())
        )
        
        return same_measure_class
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "effect_measure": self.effect_measure.value,
            "estimand_type": self.estimand_type.value,
            "population_description": self.population_description,
            "treatment_description": self.treatment_description,
            "outcome_description": self.outcome_description,
            "time_horizon": self.time_horizon,
            "ice_strategies": {
                k: v.value for k, v in self.ice_strategies.items()
            },
            "summary_function": self.summary_function,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Estimand:
        """Create from dictionary."""
        ice_strategies = {
            k: InterCurrentEventStrategy(v) 
            for k, v in d.get("ice_strategies", {}).items()
        }
        return cls(
            effect_measure=EffectMeasure.from_string(d["effect_measure"]),
            estimand_type=EstimandType.from_string(d["estimand_type"]),
            population_description=d.get("population_description", ""),
            treatment_description=d.get("treatment_description", ""),
            outcome_description=d.get("outcome_description", ""),
            time_horizon=d.get("time_horizon"),
            ice_strategies=ice_strategies,
            summary_function=d.get("summary_function", "marginal"),
        )
    
    def describe(self) -> str:
        """Generate human-readable description of estimand."""
        parts = [
            f"Effect Measure: {self.effect_measure.value}",
            f"Estimand Type: {self.estimand_type.value}",
        ]
        
        if self.population_description:
            parts.append(f"Population: {self.population_description}")
        if self.treatment_description:
            parts.append(f"Treatment: {self.treatment_description}")
        if self.outcome_description:
            parts.append(f"Outcome: {self.outcome_description}")
        if self.time_horizon:
            parts.append(f"Time Horizon: {self.time_horizon} months")
        if self.ice_strategies:
            ice_desc = ", ".join(
                f"{k}: {v.value}" for k, v in self.ice_strategies.items()
            )
            parts.append(f"ICE Handling: {ice_desc}")
        
        return "\n".join(parts)


def convert_effect_estimate(
    estimate: float,
    se: float,
    from_measure: EffectMeasure,
    to_measure: EffectMeasure,
    baseline_risk: Optional[float] = None,
    time_horizon_from: Optional[float] = None,
    time_horizon_to: Optional[float] = None,
) -> tuple[float, float]:
    """
    Convert effect estimate between different measures.
    
    Args:
        estimate: Point estimate on original scale
        se: Standard error on original scale
        from_measure: Original effect measure
        to_measure: Target effect measure
        baseline_risk: Baseline risk in control group (for RD conversions)
        time_horizon_from: Original time horizon in months
        time_horizon_to: Target time horizon in months
        
    Returns:
        Tuple of (converted_estimate, converted_se)
    
    Note:
        This function uses approximations that may not be valid for all ranges.
        Large effects or extreme baseline risks may yield inaccurate conversions.
    """
    if from_measure == to_measure:
        return estimate, se
    
    # First convert to log-scale if ratio measure
    if from_measure.is_ratio_measure():
        log_est = np.log(estimate)
        log_se = se / estimate  # Delta method approximation
    else:
        log_est = estimate
        log_se = se
    
    # Handle specific conversions
    if from_measure == EffectMeasure.ODDS_RATIO and to_measure == EffectMeasure.RISK_RATIO:
        if baseline_risk is None:
            raise ValueError("baseline_risk required for OR to RR conversion")
        # Zhang & Yu (1998) formula: RR = OR / (1 - p0 + p0*OR)
        rr = estimate / (1 - baseline_risk + baseline_risk * estimate)
        # Approximate SE using delta method
        d_rr_d_or = (1 - baseline_risk) / ((1 - baseline_risk + baseline_risk * estimate) ** 2)
        rr_se = se * abs(d_rr_d_or)
        return rr, rr_se
    
    elif from_measure == EffectMeasure.RISK_RATIO and to_measure == EffectMeasure.RISK_DIFFERENCE:
        if baseline_risk is None:
            raise ValueError("baseline_risk required for RR to RD conversion")
        rd = baseline_risk * (estimate - 1)
        rd_se = baseline_risk * se  # Approximation
        return rd, rd_se
    
    elif from_measure == EffectMeasure.ODDS_RATIO and to_measure == EffectMeasure.RISK_DIFFERENCE:
        if baseline_risk is None:
            raise ValueError("baseline_risk required for OR to RD conversion")
        # First convert OR to RR
        rr, rr_se = convert_effect_estimate(
            estimate, se, 
            EffectMeasure.ODDS_RATIO, EffectMeasure.RISK_RATIO,
            baseline_risk=baseline_risk
        )
        # Then RR to RD
        return convert_effect_estimate(
            rr, rr_se,
            EffectMeasure.RISK_RATIO, EffectMeasure.RISK_DIFFERENCE,
            baseline_risk=baseline_risk
        )
    
    elif from_measure == EffectMeasure.HAZARD_RATIO and to_measure == EffectMeasure.RISK_RATIO:
        # HR ≈ RR for rare events or short follow-up
        # For common outcomes, this is an approximation
        return estimate, se
    
    elif from_measure == EffectMeasure.HAZARD_RATIO and to_measure == EffectMeasure.RISK_DIFFERENCE:
        if baseline_risk is None:
            raise ValueError("baseline_risk required for HR to RD conversion")
        if time_horizon_from is None or time_horizon_to is None:
            # Assume same time horizon, use cumulative incidence approximation
            # P(event by t | treatment) ≈ 1 - (1-p0)^HR
            p0 = baseline_risk
            p1 = 1 - (1 - p0) ** estimate
            rd = p1 - p0
            # Approximate SE using delta method
            d_p1_d_hr = -(1 - p0) ** estimate * np.log(1 - p0)
            rd_se = se * abs(d_p1_d_hr)
            return rd, rd_se
        else:
            # Adjust for time horizon difference
            ratio = time_horizon_to / time_horizon_from
            # Simple linear extrapolation (crude approximation)
            adjusted_estimate = estimate
            adjusted_se = se * np.sqrt(ratio)
            p0_adj = min(baseline_risk * ratio, 0.99)
            p1 = 1 - (1 - p0_adj) ** adjusted_estimate
            rd = p1 - p0_adj
            d_p1_d_hr = -(1 - p0_adj) ** adjusted_estimate * np.log(1 - p0_adj)
            rd_se = adjusted_se * abs(d_p1_d_hr)
            return rd, rd_se
    
    # If no specific conversion available, raise error
    raise ValueError(
        f"Conversion from {from_measure.value} to {to_measure.value} not implemented"
    )
