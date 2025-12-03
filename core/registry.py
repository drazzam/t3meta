"""
Study Registry for T3-Meta.

This module provides a container for managing collections of studies
in a meta-analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Iterator, Union, Callable
import numpy as np
import json

from t3meta.core.study import Study, DesignFeatures
from t3meta.core.target_trial import TargetTrial
from t3meta.core.estimand import EffectMeasure


@dataclass
class StudyRegistry:
    """
    Container for managing studies in a T3-Meta analysis.
    
    Provides methods for adding, removing, filtering, and accessing
    studies, as well as computing aggregate statistics.
    
    Attributes:
        target_trial: The target trial specification
        studies: List of registered studies
        name: Optional name for the registry
        notes: Additional notes
    """
    
    target_trial: Optional[TargetTrial] = None
    studies: List[Study] = field(default_factory=list)
    name: str = ""
    notes: str = ""
    
    # Index for quick lookup
    _name_index: Dict[str, int] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Build name index."""
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the name-to-index mapping."""
        self._name_index = {
            study.name: i for i, study in enumerate(self.studies)
        }
    
    def add(self, study: Study) -> StudyRegistry:
        """
        Add a study to the registry.
        
        Args:
            study: Study to add
            
        Returns:
            self for chaining
        """
        if study.name in self._name_index:
            raise ValueError(f"Study '{study.name}' already exists in registry")
        
        self._name_index[study.name] = len(self.studies)
        self.studies.append(study)
        return self
    
    def add_many(self, studies: List[Study]) -> StudyRegistry:
        """
        Add multiple studies to the registry.
        
        Args:
            studies: List of studies to add
            
        Returns:
            self for chaining
        """
        for study in studies:
            self.add(study)
        return self
    
    def remove(self, name: str) -> Study:
        """
        Remove a study by name.
        
        Args:
            name: Name of study to remove
            
        Returns:
            The removed study
        """
        if name not in self._name_index:
            raise KeyError(f"Study '{name}' not found in registry")
        
        idx = self._name_index[name]
        study = self.studies.pop(idx)
        self._rebuild_index()
        return study
    
    def get(self, name: str) -> Study:
        """
        Get a study by name.
        
        Args:
            name: Study name
            
        Returns:
            The study
        """
        if name not in self._name_index:
            raise KeyError(f"Study '{name}' not found in registry")
        return self.studies[self._name_index[name]]
    
    def __getitem__(self, key: Union[str, int]) -> Study:
        """Get study by name or index."""
        if isinstance(key, str):
            return self.get(key)
        return self.studies[key]
    
    def __len__(self) -> int:
        """Number of studies in registry."""
        return len(self.studies)
    
    def __iter__(self) -> Iterator[Study]:
        """Iterate over studies."""
        return iter(self.studies)
    
    def __contains__(self, name: str) -> bool:
        """Check if study name exists."""
        return name in self._name_index
    
    @property
    def names(self) -> List[str]:
        """Get list of study names."""
        return [s.name for s in self.studies]
    
    @property
    def n_studies(self) -> int:
        """Number of studies."""
        return len(self.studies)
    
    @property
    def n_rct(self) -> int:
        """Number of RCTs."""
        return sum(
            1 for s in self.studies 
            if s.design_map and s.design_map.features.is_rct
        )
    
    @property
    def n_observational(self) -> int:
        """Number of observational studies."""
        return self.n_studies - self.n_rct
    
    @property
    def total_sample_size(self) -> int:
        """Total sample size across all studies."""
        return sum(
            s.n_total or 0 for s in self.studies
        )
    
    @property
    def total_events(self) -> int:
        """Total events across all studies."""
        return sum(
            s.events_total or 0 for s in self.studies
        )
    
    def filter(self, predicate: Callable[[Study], bool]) -> StudyRegistry:
        """
        Filter studies by a predicate function.
        
        Args:
            predicate: Function that returns True for studies to keep
            
        Returns:
            New StudyRegistry with filtered studies
        """
        filtered = [s for s in self.studies if predicate(s)]
        return StudyRegistry(
            target_trial=self.target_trial,
            studies=filtered,
            name=f"{self.name} (filtered)",
            notes=self.notes,
        )
    
    def filter_rct(self) -> StudyRegistry:
        """Get registry with only RCTs."""
        return self.filter(
            lambda s: s.design_map and s.design_map.features.is_rct
        )
    
    def filter_observational(self) -> StudyRegistry:
        """Get registry with only observational studies."""
        return self.filter(
            lambda s: s.design_map and not s.design_map.features.is_rct
        )
    
    def filter_by_year(
        self, 
        min_year: Optional[int] = None, 
        max_year: Optional[int] = None
    ) -> StudyRegistry:
        """Filter studies by publication year."""
        def predicate(s: Study) -> bool:
            if s.year is None:
                return False
            if min_year and s.year < min_year:
                return False
            if max_year and s.year > max_year:
                return False
            return True
        return self.filter(predicate)
    
    def filter_by_sample_size(self, min_n: int) -> StudyRegistry:
        """Filter studies by minimum sample size."""
        return self.filter(
            lambda s: s.n_total is not None and s.n_total >= min_n
        )
    
    def get_estimates(self) -> np.ndarray:
        """Get array of effect estimates (on analysis scale)."""
        return np.array([s.log_estimate for s in self.studies])
    
    def get_variances(self) -> np.ndarray:
        """Get array of sampling variances."""
        return np.array([s.variance for s in self.studies])
    
    def get_standard_errors(self) -> np.ndarray:
        """Get array of standard errors."""
        return np.array([s.log_se for s in self.studies])
    
    def get_weights(self, method: str = "inverse_variance") -> np.ndarray:
        """
        Get array of study weights.
        
        Args:
            method: Weighting method ('inverse_variance', 'sample_size', 'equal')
            
        Returns:
            Array of weights (sum to 1)
        """
        if method == "inverse_variance":
            w = 1.0 / self.get_variances()
        elif method == "sample_size":
            sizes = np.array([s.n_total or 1 for s in self.studies])
            w = sizes.astype(float)
        elif method == "equal":
            w = np.ones(len(self.studies))
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        return w / w.sum()
    
    def get_design_matrix(
        self, 
        feature_names: Optional[List[str]] = None,
        include_intercept: bool = True
    ) -> np.ndarray:
        """
        Get design covariate matrix X for all studies.
        
        Args:
            feature_names: List of feature names to include
            include_intercept: Whether to add intercept column
            
        Returns:
            2D array of shape (n_studies, n_features)
        """
        if feature_names is None:
            feature_names = DesignFeatures.standard_feature_names()
        
        X = np.vstack([
            s.get_design_vector(feature_names) for s in self.studies
        ])
        
        if include_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack([intercept, X])
        
        return X
    
    def get_effect_modifier_matrix(
        self, 
        modifier_names: List[str]
    ) -> np.ndarray:
        """
        Get effect modifier matrix Z for all studies.
        
        Args:
            modifier_names: List of modifier names
            
        Returns:
            2D array of shape (n_studies, n_modifiers)
        """
        return np.vstack([
            s.get_effect_modifier_vector(modifier_names) 
            for s in self.studies
        ])
    
    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the registry.
        
        Returns:
            Dictionary of summary statistics
        """
        estimates = self.get_estimates()
        variances = self.get_variances()
        weights = 1.0 / variances
        
        # Fixed-effect estimate
        fe_estimate = np.sum(weights * estimates) / np.sum(weights)
        fe_variance = 1.0 / np.sum(weights)
        
        # Heterogeneity
        q = np.sum(weights * (estimates - fe_estimate) ** 2)
        df = len(self.studies) - 1
        
        # I-squared
        i_squared = max(0, (q - df) / q) if q > 0 else 0
        
        return {
            "n_studies": self.n_studies,
            "n_rct": self.n_rct,
            "n_observational": self.n_observational,
            "total_sample_size": self.total_sample_size,
            "total_events": self.total_events,
            "fixed_effect_estimate": fe_estimate,
            "fixed_effect_se": np.sqrt(fe_variance),
            "cochran_q": q,
            "q_df": df,
            "i_squared": i_squared,
            "min_estimate": estimates.min(),
            "max_estimate": estimates.max(),
            "estimate_range": estimates.max() - estimates.min(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target_trial": self.target_trial.to_dict() if self.target_trial else None,
            "studies": [s.to_dict() for s in self.studies],
            "name": self.name,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StudyRegistry:
        """Create from dictionary."""
        target_trial = None
        if d.get("target_trial"):
            target_trial = TargetTrial.from_dict(d["target_trial"])
        
        studies = [Study.from_dict(s) for s in d.get("studies", [])]
        
        return cls(
            target_trial=target_trial,
            studies=studies,
            name=d.get("name", ""),
            notes=d.get("notes", ""),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> StudyRegistry:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def to_dataframe(self):
        """
        Convert to pandas DataFrame.
        
        Returns:
            DataFrame with study information
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        
        records = []
        for s in self.studies:
            ci_lower, ci_upper = s.get_ci()
            record = {
                "name": s.name,
                "estimate": s.effect_estimate,
                "log_estimate": s.log_estimate,
                "se": s.log_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_total": s.n_total,
                "events": s.events_total,
                "is_rct": s.design_map.features.is_rct if s.design_map else None,
                "year": s.year,
            }
            
            # Add design features
            if s.design_map:
                for name in DesignFeatures.standard_feature_names():
                    record[f"design_{name}"] = s.design_map.features.get_numeric_value(name)
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def describe(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Study Registry: {self.name}" if self.name else "Study Registry",
            "=" * 50,
            f"Number of studies: {self.n_studies}",
            f"  - RCTs: {self.n_rct}",
            f"  - Observational: {self.n_observational}",
            f"Total sample size: {self.total_sample_size:,}",
            f"Total events: {self.total_events:,}",
            "",
            "Studies:",
        ]
        
        for s in self.studies:
            ci = s.get_ci()
            design = "RCT" if (s.design_map and s.design_map.features.is_rct) else "OBS"
            lines.append(
                f"  [{design}] {s.name}: {s.effect_estimate:.3f} "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] (n={s.n_total or '?'})"
            )
        
        if self.target_trial:
            lines.append("")
            lines.append(f"Target Trial: {self.target_trial.name}")
        
        return "\n".join(lines)
