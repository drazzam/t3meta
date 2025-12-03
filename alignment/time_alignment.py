"""
Time Alignment Utilities for T3-Meta.

This module provides functions for aligning studies with different
follow-up durations to a common target time horizon.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy import stats, interpolate


@dataclass
class SurvivalData:
    """
    Container for survival data used in time alignment.
    
    Attributes:
        times: Array of time points
        survival: Survival probabilities at each time point
        at_risk: Number at risk at each time point (optional)
        events: Number of events at each time point (optional)
    """
    times: np.ndarray
    survival: np.ndarray
    at_risk: Optional[np.ndarray] = None
    events: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate inputs."""
        self.times = np.asarray(self.times)
        self.survival = np.asarray(self.survival)
        
        if len(self.times) != len(self.survival):
            raise ValueError("times and survival must have same length")
        
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("times must be non-decreasing")
        
        if not np.all((self.survival >= 0) & (self.survival <= 1)):
            raise ValueError("survival probabilities must be in [0, 1]")
    
    def get_survival_at(self, t: float) -> float:
        """
        Get survival probability at specific time point.
        
        Uses linear interpolation between observed time points.
        """
        if t <= self.times[0]:
            return 1.0
        if t >= self.times[-1]:
            return self.survival[-1]
        
        return np.interp(t, self.times, self.survival)
    
    def get_cumulative_incidence_at(self, t: float) -> float:
        """Get cumulative incidence at specific time point."""
        return 1 - self.get_survival_at(t)


@dataclass
class TimeAligner:
    """
    Align time-to-event outcomes to a common time horizon.
    
    This class provides methods for adjusting hazard ratios and other
    survival measures to account for different follow-up durations.
    
    Attributes:
        target_time: Target time horizon in months
        method: Alignment method ('exponential', 'weibull', 'linear', 'rmst')
    """
    
    target_time: float
    method: str = "exponential"
    
    def __post_init__(self):
        """Validate inputs."""
        if self.target_time <= 0:
            raise ValueError(f"target_time must be positive, got {self.target_time}")
        
        valid_methods = {"exponential", "weibull", "linear", "rmst", "proportional"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
    
    def align_hr(
        self,
        hr: float,
        se_log_hr: float,
        source_time: float,
        source_baseline_risk: Optional[float] = None,
        target_baseline_risk: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Align hazard ratio from source time horizon to target.
        
        Under proportional hazards, the HR should be constant across time.
        However, cumulative incidence and risk differences depend on time.
        This method adjusts the SE to account for different precision at
        different time horizons.
        
        Args:
            hr: Hazard ratio estimate
            se_log_hr: Standard error of log(HR)
            source_time: Source study follow-up time in months
            source_baseline_risk: Baseline risk at source time (optional)
            target_baseline_risk: Baseline risk at target time (optional)
            
        Returns:
            Tuple of (aligned_hr, aligned_se_log_hr)
        """
        if self.method == "proportional":
            # Under proportional hazards, HR is constant
            # Only adjust SE based on information content
            info_ratio = np.sqrt(source_time / self.target_time)
            aligned_se = se_log_hr * info_ratio
            return hr, aligned_se
        
        elif self.method == "exponential":
            # Assume exponential distribution for extrapolation
            # HR remains constant, but precision changes
            if source_time == self.target_time:
                return hr, se_log_hr
            
            # Rough approximation: precision ~ sqrt(events) ~ sqrt(time) for fixed rate
            time_ratio = self.target_time / source_time
            
            if time_ratio > 1:
                # Extrapolating beyond observed data - inflate SE
                se_inflation = np.sqrt(time_ratio)
            else:
                # Restricting to shorter horizon - could have more or less precision
                se_inflation = np.sqrt(1 / time_ratio) * 0.9  # Modest adjustment
            
            aligned_se = se_log_hr * se_inflation
            return hr, aligned_se
        
        elif self.method == "linear":
            # Simple linear interpolation/extrapolation of log(HR)
            # This is generally not recommended but provided for completeness
            return hr, se_log_hr * np.sqrt(self.target_time / source_time)
        
        elif self.method == "rmst":
            # RMST-based alignment (requires more data, return approximation)
            return hr, se_log_hr
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def align_risk_difference(
        self,
        rd: float,
        se_rd: float,
        source_time: float,
        baseline_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Align risk difference from source time to target time.
        
        Args:
            rd: Risk difference at source time
            se_rd: Standard error of RD
            source_time: Source time horizon in months
            baseline_rate: Baseline event rate per unit time (optional)
            
        Returns:
            Tuple of (aligned_rd, aligned_se_rd)
        """
        if source_time == self.target_time:
            return rd, se_rd
        
        time_ratio = self.target_time / source_time
        
        if self.method in ["exponential", "proportional"]:
            # Approximate: RD scales roughly linearly for small risks
            # More precisely, need to convert through cumulative incidence
            # RD(t) = p1(t) - p0(t)
            # Under proportional hazards: p(t) = 1 - S(t) = 1 - exp(-λt)
            # This is approximate
            aligned_rd = rd * time_ratio
            aligned_se = se_rd * time_ratio
            
            # Bound RD to [-1, 1]
            aligned_rd = np.clip(aligned_rd, -1, 1)
            
            return aligned_rd, aligned_se
        
        elif self.method == "linear":
            aligned_rd = rd * time_ratio
            aligned_se = se_rd * np.sqrt(time_ratio)
            return aligned_rd, aligned_se
        
        else:
            return rd, se_rd
    
    def align_cumulative_incidence(
        self,
        ci: float,
        se_ci: float,
        source_time: float,
    ) -> Tuple[float, float]:
        """
        Align cumulative incidence from source time to target time.
        
        Args:
            ci: Cumulative incidence at source time
            se_ci: Standard error of CI
            source_time: Source time horizon
            
        Returns:
            Tuple of (aligned_ci, aligned_se)
        """
        if source_time == self.target_time:
            return ci, se_ci
        
        if ci <= 0 or ci >= 1:
            return ci, se_ci
        
        # Convert to hazard rate assuming exponential
        # CI = 1 - exp(-λt) → λ = -log(1-CI)/t
        hazard_rate = -np.log(1 - ci) / source_time
        
        # Project to target time
        aligned_ci = 1 - np.exp(-hazard_rate * self.target_time)
        
        # Propagate uncertainty (delta method)
        # d(CI_target)/d(CI_source) = d(CI_target)/d(λ) * d(λ)/d(CI_source)
        # d(λ)/d(CI) = 1/((1-CI)*t)
        # d(CI_target)/d(λ) = t_target * (1 - CI_target)
        d_lambda_d_ci = 1 / ((1 - ci) * source_time)
        d_ci_target_d_lambda = self.target_time * (1 - aligned_ci)
        
        jacobian = d_ci_target_d_lambda * d_lambda_d_ci
        aligned_se = abs(jacobian) * se_ci
        
        # Bound CI to (0, 1)
        aligned_ci = np.clip(aligned_ci, 0.001, 0.999)
        
        return aligned_ci, aligned_se
    
    def estimate_rmst_difference(
        self,
        hr: float,
        se_log_hr: float,
        baseline_survival_data: Optional[SurvivalData] = None,
        baseline_median: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Estimate RMST difference from hazard ratio.
        
        RMST = integral of S(t) from 0 to tau.
        
        Args:
            hr: Hazard ratio
            se_log_hr: Standard error of log(HR)
            baseline_survival_data: Survival data for control group
            baseline_median: Median survival in control (alternative to full data)
            
        Returns:
            Tuple of (rmst_diff, se_rmst_diff)
        """
        tau = self.target_time
        
        if baseline_survival_data is not None:
            # Use actual survival data for more accurate estimation
            times = baseline_survival_data.times
            s0 = baseline_survival_data.survival
            
            # Control RMST
            rmst0 = np.trapz(s0[times <= tau], times[times <= tau])
            
            # Treatment RMST under proportional hazards
            # S1(t) = S0(t)^HR
            s1 = s0 ** hr
            rmst1 = np.trapz(s1[times <= tau], times[times <= tau])
            
            rmst_diff = rmst1 - rmst0
            
            # Approximate SE using delta method
            # This is a rough approximation
            se_rmst_diff = abs(rmst1 * np.log(s0.mean()) * se_log_hr)
            
        elif baseline_median is not None:
            # Assume exponential distribution
            # Median = log(2)/λ → λ = log(2)/median
            lambda0 = np.log(2) / baseline_median
            
            # RMST for exponential: (1 - exp(-λτ))/λ
            rmst0 = (1 - np.exp(-lambda0 * tau)) / lambda0
            
            # Treatment: λ1 = HR * λ0
            lambda1 = hr * lambda0
            rmst1 = (1 - np.exp(-lambda1 * tau)) / lambda1
            
            rmst_diff = rmst1 - rmst0
            
            # Approximate SE
            d_rmst1_d_hr = -tau * np.exp(-lambda1 * tau) / hr - rmst1 / hr
            se_rmst_diff = abs(d_rmst1_d_hr) * hr * se_log_hr
            
        else:
            # Fallback: very rough approximation
            # Assume common event rate and compute expected difference
            rmst_diff = -np.log(hr) * tau * 0.1  # Very rough
            se_rmst_diff = abs(rmst_diff) * se_log_hr / abs(np.log(hr)) if hr != 1 else 0.1
        
        return rmst_diff, se_rmst_diff


def compute_rmst(
    survival_data: SurvivalData,
    tau: float
) -> Tuple[float, float]:
    """
    Compute RMST and its standard error.
    
    Args:
        survival_data: Survival data
        tau: Restriction time
        
    Returns:
        Tuple of (rmst, se_rmst)
    """
    times = survival_data.times
    survival = survival_data.survival
    
    # Truncate at tau
    mask = times <= tau
    t = times[mask]
    s = survival[mask]
    
    # Add tau if not already included
    if t[-1] < tau:
        t = np.append(t, tau)
        s = np.append(s, survival_data.get_survival_at(tau))
    
    # RMST = integral of S(t) from 0 to tau
    rmst = np.trapz(s, t)
    
    # Greenwood-type variance estimate (simplified)
    # For full variance, would need at-risk counts and events
    if survival_data.at_risk is not None and survival_data.events is not None:
        # Proper variance estimation
        n = survival_data.at_risk[mask]
        d = survival_data.events[mask]
        
        # Greenwood's formula for S(t) variance, then integrate
        var_s = survival[mask]**2 * np.cumsum(d / (n * (n - d) + 1e-10))
        se_rmst = np.sqrt(np.trapz(var_s, t[:-1] if len(var_s) < len(t) else t))
    else:
        # Rough approximation
        se_rmst = rmst * 0.1  # 10% CV as rough guess
    
    return rmst, se_rmst


def align_studies_to_time_horizon(
    studies: List,  # List[Study]
    target_time: float,
    method: str = "exponential"
) -> List[Tuple[float, float]]:
    """
    Align multiple studies to a common time horizon.
    
    Args:
        studies: List of Study objects
        target_time: Target time horizon in months
        method: Alignment method
        
    Returns:
        List of (aligned_estimate, aligned_se) tuples
    """
    aligner = TimeAligner(target_time=target_time, method=method)
    aligned = []
    
    for study in studies:
        # Get source follow-up time
        source_time = None
        if study.design_map and study.design_map.features.median_followup_months:
            source_time = study.design_map.features.median_followup_months
        elif study.design_map and study.design_map.features.max_followup_months:
            source_time = study.design_map.features.max_followup_months
        
        if source_time is None:
            # Assume target time if not specified
            aligned.append((study.log_estimate, study.log_se))
            continue
        
        # Align based on effect measure
        from t3meta.core.estimand import EffectMeasure
        
        if study.effect_measure in [EffectMeasure.HAZARD_RATIO, 
                                     EffectMeasure.RISK_RATIO,
                                     EffectMeasure.ODDS_RATIO]:
            aligned_est, aligned_se = aligner.align_hr(
                np.exp(study.log_estimate),
                study.log_se,
                source_time,
                study.baseline_risk,
            )
            aligned.append((np.log(aligned_est), aligned_se))
        
        elif study.effect_measure == EffectMeasure.RISK_DIFFERENCE:
            aligned_est, aligned_se = aligner.align_risk_difference(
                study.effect_estimate,
                study.log_se,
                source_time,
            )
            aligned.append((aligned_est, aligned_se))
        
        else:
            # Other measures - no time alignment
            aligned.append((study.log_estimate, study.log_se))
    
    return aligned
