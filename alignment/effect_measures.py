"""
Effect Measure Conversion Utilities for T3-Meta.

This module provides functions for converting between different effect measures
(OR, RR, HR, RD, etc.) to align studies to a common target estimand.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import numpy as np
from scipy import stats

from t3meta.core.estimand import EffectMeasure


def log_transform(
    estimate: float, 
    se: float
) -> Tuple[float, float]:
    """
    Transform ratio measure to log scale.
    
    Args:
        estimate: Effect estimate on original scale
        se: Standard error on original scale (relative)
        
    Returns:
        Tuple of (log_estimate, log_se)
    """
    if estimate <= 0:
        raise ValueError(f"Estimate must be positive for log transform, got {estimate}")
    
    log_estimate = np.log(estimate)
    # If SE is already on log scale (as is typical for ratios), keep it
    # Otherwise, apply delta method: SE(log(X)) ≈ SE(X) / X
    log_se = se
    
    return log_estimate, log_se


def exp_transform(
    log_estimate: float, 
    log_se: float
) -> Tuple[float, float]:
    """
    Transform from log scale back to original scale.
    
    Args:
        log_estimate: Effect estimate on log scale
        log_se: Standard error on log scale
        
    Returns:
        Tuple of (estimate, se) on original scale
    """
    estimate = np.exp(log_estimate)
    # Approximate SE on original scale
    se = log_se  # Return log-scale SE as is (it's what we typically work with)
    
    return estimate, se


def or_to_rr(
    odds_ratio: float,
    se_log_or: float,
    baseline_risk: float,
    method: str = "zhang_yu"
) -> Tuple[float, float]:
    """
    Convert odds ratio to risk ratio.
    
    Args:
        odds_ratio: Odds ratio estimate
        se_log_or: Standard error of log(OR)
        baseline_risk: Risk in control/reference group (p0)
        method: Conversion method ('zhang_yu' or 'rare_disease')
        
    Returns:
        Tuple of (risk_ratio, se_log_rr)
        
    Reference:
        Zhang J, Yu KF. What's the relative risk? A method of correcting 
        the odds ratio in cohort studies of common outcomes. JAMA 1998.
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError(f"baseline_risk must be in (0, 1), got {baseline_risk}")
    
    if method == "rare_disease":
        # For rare outcomes, OR ≈ RR
        return odds_ratio, se_log_or
    
    elif method == "zhang_yu":
        # Zhang & Yu (1998) formula
        # RR = OR / (1 - p0 + p0*OR)
        denominator = 1 - baseline_risk + baseline_risk * odds_ratio
        rr = odds_ratio / denominator
        
        # Delta method for SE
        # d(RR)/d(OR) = (1 - p0) / (1 - p0 + p0*OR)^2
        d_rr_d_or = (1 - baseline_risk) / (denominator ** 2)
        
        # SE on original scale: SE(RR) = |d(RR)/d(OR)| * SE(OR)
        # SE(OR) = OR * SE(log(OR))
        se_or = odds_ratio * se_log_or
        se_rr = abs(d_rr_d_or) * se_or
        
        # Convert to log scale: SE(log(RR)) ≈ SE(RR) / RR
        se_log_rr = se_rr / rr
        
        return rr, se_log_rr
    
    else:
        raise ValueError(f"Unknown method: {method}")


def rr_to_rd(
    risk_ratio: float,
    se_log_rr: float,
    baseline_risk: float
) -> Tuple[float, float]:
    """
    Convert risk ratio to risk difference.
    
    Args:
        risk_ratio: Risk ratio estimate
        se_log_rr: Standard error of log(RR)
        baseline_risk: Risk in control/reference group (p0)
        
    Returns:
        Tuple of (risk_difference, se_rd)
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError(f"baseline_risk must be in (0, 1), got {baseline_risk}")
    
    # RD = p1 - p0 = p0 * RR - p0 = p0 * (RR - 1)
    rd = baseline_risk * (risk_ratio - 1)
    
    # Delta method for SE
    # d(RD)/d(RR) = p0
    # SE(RR) = RR * SE(log(RR))
    se_rr = risk_ratio * se_log_rr
    se_rd = baseline_risk * se_rr
    
    return rd, se_rd


def hr_to_rd(
    hazard_ratio: float,
    se_log_hr: float,
    baseline_risk: float,
    time_horizon: Optional[float] = None,
    method: str = "cumulative_incidence"
) -> Tuple[float, float]:
    """
    Convert hazard ratio to risk difference.
    
    Args:
        hazard_ratio: Hazard ratio estimate
        se_log_hr: Standard error of log(HR)
        baseline_risk: Cumulative incidence in control group at specified time
        time_horizon: Time horizon in months (optional)
        method: Conversion method
        
    Returns:
        Tuple of (risk_difference, se_rd)
        
    Note:
        This is an approximation that assumes proportional hazards and
        uses cumulative incidence transformation.
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError(f"baseline_risk must be in (0, 1), got {baseline_risk}")
    
    if method == "cumulative_incidence":
        # Under proportional hazards:
        # P(event by t | treatment) = 1 - S_0(t)^HR = 1 - (1 - p0)^HR
        p0 = baseline_risk
        p1 = 1 - (1 - p0) ** hazard_ratio
        rd = p1 - p0
        
        # Delta method for SE
        # d(p1)/d(HR) = -(1-p0)^HR * log(1-p0)
        d_p1_d_hr = -(1 - p0) ** hazard_ratio * np.log(1 - p0)
        
        # SE(HR) = HR * SE(log(HR))
        se_hr = hazard_ratio * se_log_hr
        se_rd = abs(d_p1_d_hr) * se_hr
        
        return rd, se_rd
    
    elif method == "rare_event":
        # For rare events, HR ≈ RR, and RD ≈ p0 * (HR - 1)
        rd = baseline_risk * (hazard_ratio - 1)
        se_hr = hazard_ratio * se_log_hr
        se_rd = baseline_risk * se_hr
        return rd, se_rd
    
    else:
        raise ValueError(f"Unknown method: {method}")


def rd_to_rr(
    risk_difference: float,
    se_rd: float,
    baseline_risk: float
) -> Tuple[float, float]:
    """
    Convert risk difference to risk ratio.
    
    Args:
        risk_difference: Risk difference estimate
        se_rd: Standard error of RD
        baseline_risk: Risk in control/reference group
        
    Returns:
        Tuple of (risk_ratio, se_log_rr)
    """
    if baseline_risk <= 0 or baseline_risk >= 1:
        raise ValueError(f"baseline_risk must be in (0, 1), got {baseline_risk}")
    
    # RR = (p0 + RD) / p0 = 1 + RD/p0
    rr = 1 + risk_difference / baseline_risk
    
    if rr <= 0:
        raise ValueError(f"Computed RR is non-positive: {rr}")
    
    # Delta method: d(RR)/d(RD) = 1/p0
    se_rr = se_rd / baseline_risk
    
    # Convert to log scale
    se_log_rr = se_rr / rr
    
    return rr, se_log_rr


def smd_to_or(
    smd: float,
    se_smd: float,
    method: str = "logistic"
) -> Tuple[float, float]:
    """
    Convert standardized mean difference to odds ratio.
    
    Args:
        smd: Standardized mean difference (Cohen's d or Hedges' g)
        se_smd: Standard error of SMD
        method: Conversion method ('logistic' or 'normal')
        
    Returns:
        Tuple of (odds_ratio, se_log_or)
        
    Reference:
        Chinn S. A simple method for converting an odds ratio to effect size
        for use in meta-analysis. Statistics in Medicine 2000.
    """
    if method == "logistic":
        # Chinn (2000): log(OR) = SMD * π / √3
        log_or = smd * np.pi / np.sqrt(3)
        se_log_or = se_smd * np.pi / np.sqrt(3)
        return np.exp(log_or), se_log_or
    
    elif method == "normal":
        # Cox (1970): log(OR) ≈ SMD * 1.81
        log_or = smd * 1.81
        se_log_or = se_smd * 1.81
        return np.exp(log_or), se_log_or
    
    else:
        raise ValueError(f"Unknown method: {method}")


def or_to_smd(
    odds_ratio: float,
    se_log_or: float,
    method: str = "logistic"
) -> Tuple[float, float]:
    """
    Convert odds ratio to standardized mean difference.
    
    Args:
        odds_ratio: Odds ratio estimate
        se_log_or: Standard error of log(OR)
        method: Conversion method
        
    Returns:
        Tuple of (smd, se_smd)
    """
    if method == "logistic":
        smd = np.log(odds_ratio) * np.sqrt(3) / np.pi
        se_smd = se_log_or * np.sqrt(3) / np.pi
        return smd, se_smd
    
    elif method == "normal":
        smd = np.log(odds_ratio) / 1.81
        se_smd = se_log_or / 1.81
        return smd, se_smd
    
    else:
        raise ValueError(f"Unknown method: {method}")


@dataclass
class EffectMeasureConverter:
    """
    Unified converter for effect measures.
    
    Provides a consistent interface for converting between different
    effect measures used in meta-analysis.
    
    Attributes:
        from_measure: Source effect measure
        to_measure: Target effect measure
        baseline_risk: Baseline risk for conversions (if needed)
        time_horizon: Time horizon for survival outcomes
    """
    
    from_measure: EffectMeasure
    to_measure: EffectMeasure
    baseline_risk: Optional[float] = None
    time_horizon: Optional[float] = None
    
    def __post_init__(self):
        """Validate and convert inputs."""
        if isinstance(self.from_measure, str):
            self.from_measure = EffectMeasure.from_string(self.from_measure)
        if isinstance(self.to_measure, str):
            self.to_measure = EffectMeasure.from_string(self.to_measure)
        
        # Check if baseline risk is required
        if self._requires_baseline_risk() and self.baseline_risk is None:
            raise ValueError(
                f"baseline_risk is required for conversion from "
                f"{self.from_measure.value} to {self.to_measure.value}"
            )
    
    def _requires_baseline_risk(self) -> bool:
        """Check if conversion requires baseline risk."""
        # Conversions to/from RD require baseline risk
        if self.to_measure == EffectMeasure.RISK_DIFFERENCE:
            return self.from_measure.is_ratio_measure()
        if self.from_measure == EffectMeasure.RISK_DIFFERENCE:
            return self.to_measure.is_ratio_measure()
        # OR to RR also requires baseline risk
        if (self.from_measure == EffectMeasure.ODDS_RATIO and 
            self.to_measure == EffectMeasure.RISK_RATIO):
            return True
        return False
    
    def convert(
        self, 
        estimate: float, 
        se: float
    ) -> Tuple[float, float]:
        """
        Convert effect estimate from source to target measure.
        
        Args:
            estimate: Effect estimate on original scale
            se: Standard error (on log scale for ratio measures)
            
        Returns:
            Tuple of (converted_estimate, converted_se)
        """
        if self.from_measure == self.to_measure:
            return estimate, se
        
        # Dispatch to appropriate conversion function
        from_val = self.from_measure
        to_val = self.to_measure
        
        # OR -> RR
        if from_val == EffectMeasure.ODDS_RATIO and to_val == EffectMeasure.RISK_RATIO:
            return or_to_rr(estimate, se, self.baseline_risk)
        
        # RR -> RD
        if from_val == EffectMeasure.RISK_RATIO and to_val == EffectMeasure.RISK_DIFFERENCE:
            return rr_to_rd(estimate, se, self.baseline_risk)
        
        # OR -> RD (via RR)
        if from_val == EffectMeasure.ODDS_RATIO and to_val == EffectMeasure.RISK_DIFFERENCE:
            rr, se_log_rr = or_to_rr(estimate, se, self.baseline_risk)
            return rr_to_rd(rr, se_log_rr, self.baseline_risk)
        
        # HR -> RD
        if from_val == EffectMeasure.HAZARD_RATIO and to_val == EffectMeasure.RISK_DIFFERENCE:
            return hr_to_rd(estimate, se, self.baseline_risk, self.time_horizon)
        
        # HR -> RR (approximation)
        if from_val == EffectMeasure.HAZARD_RATIO and to_val == EffectMeasure.RISK_RATIO:
            # HR ≈ RR for rare events
            return estimate, se
        
        # RD -> RR
        if from_val == EffectMeasure.RISK_DIFFERENCE and to_val == EffectMeasure.RISK_RATIO:
            return rd_to_rr(estimate, se, self.baseline_risk)
        
        # SMD -> OR
        if from_val == EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE and to_val == EffectMeasure.ODDS_RATIO:
            return smd_to_or(estimate, se)
        
        # OR -> SMD
        if from_val == EffectMeasure.ODDS_RATIO and to_val == EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE:
            return or_to_smd(estimate, se)
        
        raise NotImplementedError(
            f"Conversion from {from_val.value} to {to_val.value} not implemented"
        )
    
    def convert_many(
        self,
        estimates: np.ndarray,
        ses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert multiple estimates.
        
        Args:
            estimates: Array of effect estimates
            ses: Array of standard errors
            
        Returns:
            Tuple of (converted_estimates, converted_ses)
        """
        converted_est = np.zeros_like(estimates)
        converted_se = np.zeros_like(ses)
        
        for i in range(len(estimates)):
            converted_est[i], converted_se[i] = self.convert(estimates[i], ses[i])
        
        return converted_est, converted_se


def align_to_target_measure(
    studies: List,  # List[Study]
    target_measure: Union[str, EffectMeasure],
    use_study_baseline_risk: bool = True,
    default_baseline_risk: float = 0.1
) -> List:
    """
    Align all studies to a target effect measure.
    
    Args:
        studies: List of Study objects
        target_measure: Target effect measure
        use_study_baseline_risk: Use each study's baseline_risk if available
        default_baseline_risk: Default baseline risk if not specified
        
    Returns:
        List of aligned estimates and SEs as tuples
    """
    if isinstance(target_measure, str):
        target_measure = EffectMeasure.from_string(target_measure)
    
    results = []
    
    for study in studies:
        source_measure = study.effect_measure
        
        if source_measure == target_measure:
            results.append((study.log_estimate, study.log_se))
            continue
        
        baseline = (
            study.baseline_risk if use_study_baseline_risk and study.baseline_risk 
            else default_baseline_risk
        )
        
        converter = EffectMeasureConverter(
            from_measure=source_measure,
            to_measure=target_measure,
            baseline_risk=baseline,
        )
        
        # Get estimate on appropriate scale
        if source_measure.is_ratio_measure():
            est = study.effect_estimate
            se = study.log_se
        else:
            est = study.effect_estimate
            se = study.se or study.log_se
        
        converted_est, converted_se = converter.convert(est, se)
        
        # Return on log scale if ratio measure
        if target_measure.is_ratio_measure() and converted_est > 0:
            results.append((np.log(converted_est), converted_se))
        else:
            results.append((converted_est, converted_se))
    
    return results
