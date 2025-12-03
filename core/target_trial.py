"""
Target Trial Specification for T3-Meta.

This module defines the target trial template that serves as the gold standard
against which all included studies are compared. The target trial represents
the ideal causal study we wish we had conducted.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import json

from t3meta.core.estimand import Estimand, EstimandType, EffectMeasure, InterCurrentEventStrategy


@dataclass
class TargetTrialSpec:
    """
    Detailed specification of a target trial component.
    
    Used for structured specification of population, intervention, etc.
    """
    description: str
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TargetTrialSpec:
        return cls(
            description=d.get("description", ""),
            inclusion_criteria=d.get("inclusion_criteria", []),
            exclusion_criteria=d.get("exclusion_criteria", []),
            notes=d.get("notes", ""),
        )


@dataclass
class TargetTrial:
    """
    Target Trial Specification for T3-Meta.
    
    Before touching the literature, T3-Meta requires defining the ideal trial
    that we wish existed. Every real study is then viewed as an approximation
    of this target trial with some similarity and some deviations.
    
    Attributes:
        name: Descriptive name for the target trial
        population: Target population specification
        intervention: Treatment/intervention strategy
        comparator: Control/comparator strategy
        outcome: Primary outcome definition
        time_horizon_months: Follow-up duration in months
        estimand_type: ITT, PP, AT, etc.
        effect_measure: HR, RR, RD, etc.
        time_zero: Definition of when follow-up starts
        intercurrent_events: How ICEs are handled
        target_estimand: Full estimand specification (optional, computed from above)
        
    Example:
        >>> target = TargetTrial(
        ...     name="Ideal SGLT2i Cardiovascular Trial",
        ...     population="Adults with T2DM and established CVD",
        ...     intervention="SGLT2i initiation within 30 days of eligibility",
        ...     comparator="No SGLT2i use during follow-up",
        ...     outcome="Composite MACE (CV death, MI, stroke)",
        ...     time_horizon_months=36,
        ...     estimand_type="ITT",
        ...     effect_measure="HR",
        ...     time_zero="Date of eligibility (first T2DM diagnosis after CVD)"
        ... )
    """
    
    name: str
    population: Union[str, TargetTrialSpec] = ""
    intervention: Union[str, TargetTrialSpec] = ""
    comparator: Union[str, TargetTrialSpec] = ""
    outcome: Union[str, TargetTrialSpec] = ""
    time_horizon_months: Optional[float] = None
    estimand_type: Union[str, EstimandType] = EstimandType.INTENTION_TO_TREAT
    effect_measure: Union[str, EffectMeasure] = EffectMeasure.HAZARD_RATIO
    time_zero: str = "randomization"
    intercurrent_events: Dict[str, str] = field(default_factory=dict)
    secondary_outcomes: List[Union[str, TargetTrialSpec]] = field(default_factory=list)
    target_population_characteristics: Dict[str, Any] = field(default_factory=dict)
    analysis_plan: str = ""
    version: str = "1.0"
    created_at: Optional[str] = None
    notes: str = ""
    
    # Computed
    _estimand: Optional[Estimand] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and process inputs."""
        # Convert string types to enums
        if isinstance(self.estimand_type, str):
            self.estimand_type = EstimandType.from_string(self.estimand_type)
        if isinstance(self.effect_measure, str):
            self.effect_measure = EffectMeasure.from_string(self.effect_measure)
        
        # Set creation timestamp if not provided
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        # Build the full estimand
        self._build_estimand()
    
    def _build_estimand(self):
        """Construct the full Estimand object from components."""
        pop_desc = (
            self.population.description 
            if isinstance(self.population, TargetTrialSpec) 
            else self.population
        )
        int_desc = (
            self.intervention.description 
            if isinstance(self.intervention, TargetTrialSpec) 
            else self.intervention
        )
        out_desc = (
            self.outcome.description 
            if isinstance(self.outcome, TargetTrialSpec) 
            else self.outcome
        )
        
        # Convert ICE strategies
        ice_strategies = {}
        for event, strategy in self.intercurrent_events.items():
            if isinstance(strategy, str):
                try:
                    ice_strategies[event] = InterCurrentEventStrategy(strategy)
                except ValueError:
                    ice_strategies[event] = InterCurrentEventStrategy.TREATMENT_POLICY
            else:
                ice_strategies[event] = strategy
        
        self._estimand = Estimand(
            effect_measure=self.effect_measure,
            estimand_type=self.estimand_type,
            population_description=pop_desc,
            treatment_description=int_desc,
            outcome_description=out_desc,
            time_horizon=self.time_horizon_months,
            ice_strategies=ice_strategies,
        )
    
    @property
    def estimand(self) -> Estimand:
        """Get the full estimand specification."""
        if self._estimand is None:
            self._build_estimand()
        return self._estimand
    
    def population_text(self) -> str:
        """Get population description as text."""
        if isinstance(self.population, TargetTrialSpec):
            return self.population.description
        return self.population
    
    def intervention_text(self) -> str:
        """Get intervention description as text."""
        if isinstance(self.intervention, TargetTrialSpec):
            return self.intervention.description
        return self.intervention
    
    def comparator_text(self) -> str:
        """Get comparator description as text."""
        if isinstance(self.comparator, TargetTrialSpec):
            return self.comparator.description
        return self.comparator
    
    def outcome_text(self) -> str:
        """Get outcome description as text."""
        if isinstance(self.outcome, TargetTrialSpec):
            return self.outcome.description
        return self.outcome
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def spec_to_dict(x):
            if isinstance(x, TargetTrialSpec):
                return x.to_dict()
            return x
        
        return {
            "name": self.name,
            "population": spec_to_dict(self.population),
            "intervention": spec_to_dict(self.intervention),
            "comparator": spec_to_dict(self.comparator),
            "outcome": spec_to_dict(self.outcome),
            "time_horizon_months": self.time_horizon_months,
            "estimand_type": self.estimand_type.value,
            "effect_measure": self.effect_measure.value,
            "time_zero": self.time_zero,
            "intercurrent_events": self.intercurrent_events,
            "secondary_outcomes": [spec_to_dict(x) for x in self.secondary_outcomes],
            "target_population_characteristics": self.target_population_characteristics,
            "analysis_plan": self.analysis_plan,
            "version": self.version,
            "created_at": self.created_at,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TargetTrial:
        """Create from dictionary."""
        def parse_spec(x):
            if isinstance(x, dict) and "description" in x:
                return TargetTrialSpec.from_dict(x)
            return x
        
        secondary = d.get("secondary_outcomes", [])
        if secondary:
            secondary = [parse_spec(x) for x in secondary]
        
        return cls(
            name=d["name"],
            population=parse_spec(d.get("population", "")),
            intervention=parse_spec(d.get("intervention", "")),
            comparator=parse_spec(d.get("comparator", "")),
            outcome=parse_spec(d.get("outcome", "")),
            time_horizon_months=d.get("time_horizon_months"),
            estimand_type=d.get("estimand_type", "ITT"),
            effect_measure=d.get("effect_measure", "HR"),
            time_zero=d.get("time_zero", "randomization"),
            intercurrent_events=d.get("intercurrent_events", {}),
            secondary_outcomes=secondary,
            target_population_characteristics=d.get("target_population_characteristics", {}),
            analysis_plan=d.get("analysis_plan", ""),
            version=d.get("version", "1.0"),
            created_at=d.get("created_at"),
            notes=d.get("notes", ""),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> TargetTrial:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def describe(self) -> str:
        """Generate human-readable description."""
        lines = [
            f"Target Trial: {self.name}",
            "=" * (len(self.name) + 15),
            "",
            f"Population: {self.population_text()}",
            f"Intervention: {self.intervention_text()}",
            f"Comparator: {self.comparator_text()}",
            f"Outcome: {self.outcome_text()}",
            f"Time Horizon: {self.time_horizon_months} months" if self.time_horizon_months else "Time Horizon: Not specified",
            f"Time Zero: {self.time_zero}",
            f"Estimand Type: {self.estimand_type.value}",
            f"Effect Measure: {self.effect_measure.value}",
        ]
        
        if self.intercurrent_events:
            lines.append("")
            lines.append("Intercurrent Event Handling:")
            for event, strategy in self.intercurrent_events.items():
                lines.append(f"  - {event}: {strategy}")
        
        if self.target_population_characteristics:
            lines.append("")
            lines.append("Target Population Characteristics:")
            for key, value in self.target_population_characteristics.items():
                lines.append(f"  - {key}: {value}")
        
        if self.notes:
            lines.append("")
            lines.append(f"Notes: {self.notes}")
        
        return "\n".join(lines)
    
    def validate(self) -> List[str]:
        """
        Validate target trial specification.
        
        Returns:
            List of warning/error messages. Empty list means valid.
        """
        issues = []
        
        if not self.name:
            issues.append("Target trial name is required")
        
        if not self.population:
            issues.append("Population specification is required")
        
        if not self.intervention:
            issues.append("Intervention specification is required")
        
        if not self.comparator:
            issues.append("Comparator specification is required")
        
        if not self.outcome:
            issues.append("Outcome specification is required")
        
        if self.time_horizon_months is None:
            issues.append("Warning: Time horizon not specified; may limit estimand alignment")
        
        if not self.time_zero:
            issues.append("Warning: Time zero not explicitly defined")
        
        return issues
    
    def copy(self, **overrides) -> TargetTrial:
        """Create a copy with optional overrides."""
        d = self.to_dict()
        d.update(overrides)
        return TargetTrial.from_dict(d)


def define_target_trial(
    name: str,
    population: str,
    intervention: str,
    comparator: str,
    outcome: str,
    time_horizon_months: Optional[float] = None,
    estimand_type: str = "ITT",
    effect_measure: str = "HR",
    time_zero: str = "randomization",
    **kwargs
) -> TargetTrial:
    """
    Convenience function to define a target trial.
    
    This is the primary entry point for creating a target trial specification.
    
    Args:
        name: Descriptive name
        population: Target population description
        intervention: Treatment strategy description
        comparator: Control strategy description
        outcome: Primary outcome description
        time_horizon_months: Follow-up duration in months
        estimand_type: ITT, PP, AT, etc.
        effect_measure: HR, RR, RD, MD, SMD, etc.
        time_zero: Definition of when follow-up starts
        **kwargs: Additional arguments passed to TargetTrial
        
    Returns:
        TargetTrial instance
    """
    return TargetTrial(
        name=name,
        population=population,
        intervention=intervention,
        comparator=comparator,
        outcome=outcome,
        time_horizon_months=time_horizon_months,
        estimand_type=estimand_type,
        effect_measure=effect_measure,
        time_zero=time_zero,
        **kwargs
    )
