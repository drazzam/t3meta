"""
Baseline Risk Standardization for T3-Meta.

This module provides utilities for standardizing effect estimates
to a common baseline risk, enabling more meaningful comparisons
across studies with different populations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class BaselineRiskStandardizer:
    """
    Standardize effect estimates to a common baseline risk.
    
    When studies report relative measures (RR, OR, HR), the absolute
    benefit depends on baseline risk. This class allows standardization
    to a common target baseline risk.
    
    Attributes:
        target_baseline_risk: Target baseline risk for standardization
        method: Standardization method ('linear', 'logit', 'log')
    """
    
    target_baseline_risk: float
    method: str = "linear"
    
    def __post_init__(self):
        """Validate inputs."""
        if not 0 < self.target_baseline_risk < 1:
            raise ValueError(
                f"target_baseline_risk must be in (0, 1), got {self.target_baseline_risk}"
            )
        
        valid_methods = {"linear", "logit", "log", "probit"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
    
    def standardize_rd(
        self,
        rd: float,
        se_rd: float,
        source_baseline_risk: float,
    ) -> Tuple[float, float]:
        """
        Standardize risk difference to target baseline risk.
        
        Args:
            rd: Risk difference at source baseline
            se_rd: Standard error of RD
            source_baseline_risk: Baseline risk in source study
            
        Returns:
            Tuple of (standardized_rd, standardized_se)
        """
        if self.method == "linear":
            # Simple linear scaling
            # Assume RD scales proportionally with baseline risk
            ratio = self.target_baseline_risk / source_baseline_risk
            std_rd = rd * ratio
            std_se = se_rd * ratio
            
        elif self.method == "logit":
            # Transform through logit scale
            # Convert RD to RR, then standardize
            rr = 1 + rd / source_baseline_risk
            if rr <= 0:
                return rd, se_rd  # Cannot transform
            
            std_rd = self.target_baseline_risk * (rr - 1)
            std_se = se_rd * (self.target_baseline_risk / source_baseline_risk)
            
        else:
            # Default to no transformation
            std_rd = rd
            std_se = se_rd
        
        return std_rd, std_se
    
    def standardize_nnt(
        self,
        nnt: float,
        se_nnt: float,
        source_baseline_risk: float,
    ) -> Tuple[float, float]:
        """
        Standardize NNT to target baseline risk.
        
        Args:
            nnt: Number needed to treat at source baseline
            se_nnt: Standard error of NNT
            source_baseline_risk: Baseline risk in source study
            
        Returns:
            Tuple of (standardized_nnt, standardized_se)
        """
        # NNT = 1/ARR, so we first get ARR (absolute risk reduction = -RD)
        arr = 1 / nnt
        se_arr = se_nnt / (nnt ** 2)  # Delta method
        
        # Standardize ARR as RD
        std_arr, std_se_arr = self.standardize_rd(
            -arr, se_arr, source_baseline_risk
        )
        std_arr = -std_arr
        
        # Convert back to NNT
        if std_arr != 0:
            std_nnt = 1 / std_arr
            std_se_nnt = std_se_arr / (std_arr ** 2)
        else:
            std_nnt = np.inf
            std_se_nnt = np.inf
        
        return std_nnt, std_se_nnt
    
    def compute_target_rd_from_rr(
        self,
        rr: float,
        se_log_rr: float,
    ) -> Tuple[float, float]:
        """
        Compute risk difference at target baseline from risk ratio.
        
        Args:
            rr: Risk ratio
            se_log_rr: Standard error of log(RR)
            
        Returns:
            Tuple of (rd_at_target, se_rd)
        """
        rd = self.target_baseline_risk * (rr - 1)
        
        # Delta method: SE(RD) = |d(RD)/d(RR)| * SE(RR)
        # d(RD)/d(RR) = p0
        se_rr = rr * se_log_rr
        se_rd = self.target_baseline_risk * se_rr
        
        return rd, se_rd
    
    def compute_target_rd_from_or(
        self,
        odds_ratio: float,
        se_log_or: float,
    ) -> Tuple[float, float]:
        """
        Compute risk difference at target baseline from odds ratio.
        
        Uses Zhang-Yu conversion.
        
        Args:
            odds_ratio: Odds ratio
            se_log_or: Standard error of log(OR)
            
        Returns:
            Tuple of (rd_at_target, se_rd)
        """
        from t3meta.alignment.effect_measures import or_to_rr, rr_to_rd
        
        # Convert OR to RR at target baseline
        rr, se_log_rr = or_to_rr(odds_ratio, se_log_or, self.target_baseline_risk)
        
        # Then RR to RD
        rd, se_rd = rr_to_rd(rr, se_log_rr, self.target_baseline_risk)
        
        return rd, se_rd
    
    def compute_target_rd_from_hr(
        self,
        hr: float,
        se_log_hr: float,
        time_horizon: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute risk difference at target baseline from hazard ratio.
        
        Args:
            hr: Hazard ratio
            se_log_hr: Standard error of log(HR)
            time_horizon: Time horizon (optional)
            
        Returns:
            Tuple of (rd_at_target, se_rd)
        """
        from t3meta.alignment.effect_measures import hr_to_rd
        
        rd, se_rd = hr_to_rd(
            hr, se_log_hr, 
            self.target_baseline_risk,
            time_horizon
        )
        
        return rd, se_rd


def estimate_baseline_risk(
    events: int,
    total: int,
    method: str = "mle",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Estimate baseline risk from event counts.
    
    Args:
        events: Number of events
        total: Total sample size
        method: 'mle' for maximum likelihood, 'bayesian' for Bayesian
        prior_alpha: Beta prior alpha parameter (for Bayesian)
        prior_beta: Beta prior beta parameter (for Bayesian)
        
    Returns:
        Tuple of (estimated_risk, standard_error)
    """
    if total <= 0:
        raise ValueError("total must be positive")
    
    if method == "mle":
        risk = events / total
        # Standard error of proportion
        se = np.sqrt(risk * (1 - risk) / total)
        
    elif method == "bayesian":
        # Posterior is Beta(events + alpha, non-events + beta)
        post_alpha = events + prior_alpha
        post_beta = (total - events) + prior_beta
        
        # Posterior mean and SD
        risk = post_alpha / (post_alpha + post_beta)
        se = np.sqrt(
            post_alpha * post_beta / 
            ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
        )
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return risk, se


def pool_baseline_risks(
    risks: List[float],
    ses: List[float],
    sample_sizes: Optional[List[int]] = None,
    method: str = "inverse_variance"
) -> Tuple[float, float]:
    """
    Pool baseline risks across studies.
    
    Args:
        risks: List of baseline risk estimates
        ses: List of standard errors
        sample_sizes: List of sample sizes (for sample-size weighting)
        method: 'inverse_variance', 'sample_size', or 'equal'
        
    Returns:
        Tuple of (pooled_risk, pooled_se)
    """
    risks = np.array(risks)
    ses = np.array(ses)
    variances = ses ** 2
    
    if method == "inverse_variance":
        weights = 1 / variances
    elif method == "sample_size" and sample_sizes is not None:
        weights = np.array(sample_sizes).astype(float)
    elif method == "equal":
        weights = np.ones_like(risks)
    else:
        weights = 1 / variances
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted mean
    pooled_risk = np.sum(weights * risks)
    
    # Pooled SE (using weighted average of variances)
    pooled_var = np.sum(weights ** 2 * variances)
    pooled_se = np.sqrt(pooled_var)
    
    return pooled_risk, pooled_se


def compute_number_needed(
    rd: float,
    se_rd: float,
    measure: str = "NNT"
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute NNT or NNH from risk difference.
    
    Args:
        rd: Risk difference
        se_rd: Standard error of RD
        measure: 'NNT' (benefit) or 'NNH' (harm)
        
    Returns:
        Tuple of (nnt_or_nnh, (ci_lower, ci_upper))
    """
    if rd == 0:
        return np.inf, (np.inf, np.inf)
    
    # NNT = 1/ARR = 1/(-RD) if RD negative (treatment reduces risk)
    # NNH = 1/RD if RD positive (treatment increases risk)
    
    if measure == "NNT":
        arr = -rd  # Absolute risk reduction
        if arr <= 0:
            return np.inf, (np.inf, np.inf)
        nnt = 1 / arr
        
        # CI using Fieller's method (simplified)
        z = stats.norm.ppf(0.975)
        arr_lower = arr - z * se_rd
        arr_upper = arr + z * se_rd
        
        if arr_lower > 0:
            ci = (1 / arr_upper, 1 / arr_lower)
        else:
            ci = (1 / arr_upper, np.inf)
        
        return nnt, ci
        
    else:  # NNH
        if rd <= 0:
            return np.inf, (np.inf, np.inf)
        nnh = 1 / rd
        
        z = stats.norm.ppf(0.975)
        rd_lower = rd - z * se_rd
        rd_upper = rd + z * se_rd
        
        if rd_lower > 0:
            ci = (1 / rd_upper, 1 / rd_lower)
        else:
            ci = (1 / rd_upper, np.inf)
        
        return nnh, ci


@dataclass
class AbsoluteEffectEstimator:
    """
    Estimate absolute effects from relative effect measures.
    
    This class provides methods to convert relative measures (RR, OR, HR)
    to absolute measures (RD, NNT/NNH) at specified baseline risks.
    
    Attributes:
        baseline_risks: List of baseline risk values to compute effects at
    """
    
    baseline_risks: List[float]
    
    def __post_init__(self):
        """Validate inputs."""
        self.baseline_risks = np.array(self.baseline_risks)
        if np.any(self.baseline_risks <= 0) or np.any(self.baseline_risks >= 1):
            raise ValueError("All baseline risks must be in (0, 1)")
    
    def compute_rd_profile(
        self,
        rr: float,
        se_log_rr: float,
    ) -> Dict[str, Any]:
        """
        Compute risk difference across a range of baseline risks.
        
        Args:
            rr: Risk ratio
            se_log_rr: Standard error of log(RR)
            
        Returns:
            Dictionary with baseline_risks, rds, se_rds, nnts
        """
        rds = []
        se_rds = []
        nnts = []
        
        for p0 in self.baseline_risks:
            standardizer = BaselineRiskStandardizer(target_baseline_risk=p0)
            rd, se_rd = standardizer.compute_target_rd_from_rr(rr, se_log_rr)
            rds.append(rd)
            se_rds.append(se_rd)
            
            if rd != 0:
                nnts.append(1 / abs(rd))
            else:
                nnts.append(np.inf)
        
        return {
            "baseline_risks": self.baseline_risks.tolist(),
            "risk_differences": rds,
            "se_risk_differences": se_rds,
            "nnts": nnts,
            "relative_measure": "RR",
            "relative_estimate": rr,
        }
    
    def compute_rd_profile_from_hr(
        self,
        hr: float,
        se_log_hr: float,
        time_horizon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute risk difference profile from hazard ratio.
        
        Args:
            hr: Hazard ratio
            se_log_hr: Standard error of log(HR)
            time_horizon: Time horizon for cumulative incidence
            
        Returns:
            Dictionary with absolute effect profile
        """
        rds = []
        se_rds = []
        nnts = []
        
        for p0 in self.baseline_risks:
            standardizer = BaselineRiskStandardizer(target_baseline_risk=p0)
            rd, se_rd = standardizer.compute_target_rd_from_hr(
                hr, se_log_hr, time_horizon
            )
            rds.append(rd)
            se_rds.append(se_rd)
            
            if rd != 0:
                nnts.append(1 / abs(rd))
            else:
                nnts.append(np.inf)
        
        return {
            "baseline_risks": self.baseline_risks.tolist(),
            "risk_differences": rds,
            "se_risk_differences": se_rds,
            "nnts": nnts,
            "relative_measure": "HR",
            "relative_estimate": hr,
            "time_horizon": time_horizon,
        }
