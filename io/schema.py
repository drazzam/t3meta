"""
Schema validation for T3-Meta.

This module provides schema definitions and validation
for T3-Meta data inputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import json


@dataclass
class FieldSpec:
    """Specification for a data field."""
    
    name: str
    dtype: str  # 'float', 'int', 'str', 'bool', 'list', 'dict'
    required: bool = True
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        Validate a value against this field spec.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if required
        if value is None:
            if self.required:
                return False, f"{self.name} is required"
            return True, ""
        
        # Check dtype
        dtype_map = {
            'float': (int, float),
            'int': int,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
        }
        
        expected_types = dtype_map.get(self.dtype)
        if expected_types and not isinstance(value, expected_types):
            return False, f"{self.name} must be {self.dtype}, got {type(value).__name__}"
        
        # Check range
        if self.min_value is not None and value < self.min_value:
            return False, f"{self.name} must be >= {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"{self.name} must be <= {self.max_value}"
        
        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"{self.name} must be one of {self.allowed_values}"
        
        return True, ""


@dataclass
class T3MetaSchema:
    """
    Schema definition for T3-Meta data structures.
    
    Provides validation for study data, target trial specifications,
    and design features.
    """
    
    # Study schema
    STUDY_FIELDS: List[FieldSpec] = field(default_factory=lambda: [
        FieldSpec("name", "str", required=True, description="Study identifier"),
        FieldSpec("effect_estimate", "float", required=True, description="Effect estimate"),
        FieldSpec("effect_measure", "str", required=True,
                  allowed_values=["HR", "RR", "OR", "IRR", "RD", "MD", "SMD", "RMST"],
                  description="Effect measure type"),
        FieldSpec("se", "float", required=False, min_value=0,
                  description="Standard error"),
        FieldSpec("ci_lower", "float", required=False,
                  description="Lower confidence interval"),
        FieldSpec("ci_upper", "float", required=False,
                  description="Upper confidence interval"),
        FieldSpec("ci_level", "float", required=False, default=0.95,
                  min_value=0.5, max_value=0.999,
                  description="Confidence level"),
        FieldSpec("n_total", "int", required=False, min_value=1,
                  description="Total sample size"),
        FieldSpec("n_treatment", "int", required=False, min_value=0,
                  description="Treatment group sample size"),
        FieldSpec("n_control", "int", required=False, min_value=0,
                  description="Control group sample size"),
        FieldSpec("year", "int", required=False, min_value=1900, max_value=2100,
                  description="Publication year"),
        FieldSpec("baseline_risk", "float", required=False, min_value=0, max_value=1,
                  description="Baseline risk in control group"),
    ])
    
    # Target trial schema
    TARGET_TRIAL_FIELDS: List[FieldSpec] = field(default_factory=lambda: [
        FieldSpec("name", "str", required=True, description="Target trial name"),
        FieldSpec("population", "str", required=True, description="Target population"),
        FieldSpec("intervention", "str", required=True, description="Intervention"),
        FieldSpec("comparator", "str", required=True, description="Comparator"),
        FieldSpec("outcome", "str", required=True, description="Outcome"),
        FieldSpec("time_horizon_months", "int", required=True, min_value=1,
                  description="Follow-up duration in months"),
        FieldSpec("effect_measure", "str", required=True,
                  allowed_values=["HR", "RR", "OR", "IRR", "RD", "MD", "SMD", "RMST"],
                  description="Effect measure type"),
    ])
    
    # Design features schema
    DESIGN_FEATURE_FIELDS: List[FieldSpec] = field(default_factory=lambda: [
        FieldSpec("is_rct", "bool", required=False, description="Is randomized trial"),
        FieldSpec("outcome_adjudicated", "bool", required=False,
                  description="Was outcome adjudicated"),
        FieldSpec("has_immortal_time_bias", "bool", required=False,
                  description="Has immortal time bias"),
        FieldSpec("has_prevalent_user_bias", "bool", required=False,
                  description="Has prevalent user bias"),
        FieldSpec("loss_to_followup_pct", "float", required=False,
                  min_value=0, max_value=100,
                  description="Loss to follow-up percentage"),
        FieldSpec("sample_size", "int", required=False, min_value=1,
                  description="Sample size"),
        FieldSpec("median_followup_months", "float", required=False, min_value=0,
                  description="Median follow-up in months"),
    ])
    
    def validate_study(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate study data against schema.
        
        Args:
            data: Dictionary with study data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field_spec in self.STUDY_FIELDS:
            value = data.get(field_spec.name)
            is_valid, error = field_spec.validate(value)
            if not is_valid:
                errors.append(error)
        
        # Custom validation: either SE or CI must be provided
        if data.get("se") is None and (data.get("ci_lower") is None or data.get("ci_upper") is None):
            errors.append("Either 'se' or both 'ci_lower' and 'ci_upper' must be provided")
        
        # Check CI ordering
        if data.get("ci_lower") is not None and data.get("ci_upper") is not None:
            if data["ci_lower"] > data["ci_upper"]:
                errors.append("ci_lower must be <= ci_upper")
        
        return len(errors) == 0, errors
    
    def validate_target_trial(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate target trial specification against schema.
        
        Args:
            data: Dictionary with target trial data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field_spec in self.TARGET_TRIAL_FIELDS:
            value = data.get(field_spec.name)
            is_valid, error = field_spec.validate(value)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def validate_design_features(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate design features against schema.
        
        Args:
            data: Dictionary with design features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field_spec in self.DESIGN_FEATURE_FIELDS:
            if field_spec.name in data:
                value = data[field_spec.name]
                is_valid, error = field_spec.validate(value)
                if not is_valid:
                    errors.append(error)
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "study_fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "required": f.required,
                    "description": f.description,
                }
                for f in self.STUDY_FIELDS
            ],
            "target_trial_fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "required": f.required,
                    "description": f.description,
                }
                for f in self.TARGET_TRIAL_FIELDS
            ],
            "design_feature_fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "required": f.required,
                    "description": f.description,
                }
                for f in self.DESIGN_FEATURE_FIELDS
            ],
        }


def validate_input(
    data: Union[Dict, List[Dict]],
    schema_type: str = "study"
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate input data.
    
    Args:
        data: Data to validate (single dict or list of dicts)
        schema_type: Type of schema ('study', 'target_trial', 'design')
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    schema = T3MetaSchema()
    
    if isinstance(data, list):
        all_errors = []
        for i, item in enumerate(data):
            if schema_type == "study":
                is_valid, errors = schema.validate_study(item)
            elif schema_type == "target_trial":
                is_valid, errors = schema.validate_target_trial(item)
            elif schema_type == "design":
                is_valid, errors = schema.validate_design_features(item)
            else:
                return False, [f"Unknown schema type: {schema_type}"]
            
            if errors:
                all_errors.extend([f"Item {i}: {e}" for e in errors])
        
        return len(all_errors) == 0, all_errors
    
    else:
        if schema_type == "study":
            return schema.validate_study(data)
        elif schema_type == "target_trial":
            return schema.validate_target_trial(data)
        elif schema_type == "design":
            return schema.validate_design_features(data)
        else:
            return False, [f"Unknown schema type: {schema_type}"]


def generate_template(schema_type: str = "study", format: str = "dict") -> Any:
    """
    Generate a template for data entry.
    
    Args:
        schema_type: Type of schema ('study', 'target_trial', 'design')
        format: Output format ('dict', 'json', 'csv_header')
        
    Returns:
        Template in requested format
    """
    schema = T3MetaSchema()
    
    if schema_type == "study":
        fields = schema.STUDY_FIELDS
    elif schema_type == "target_trial":
        fields = schema.TARGET_TRIAL_FIELDS
    elif schema_type == "design":
        fields = schema.DESIGN_FEATURE_FIELDS
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")
    
    if format == "dict":
        return {
            f.name: f.default if f.default is not None else f"<{f.dtype}>"
            for f in fields
        }
    
    elif format == "json":
        template = {
            f.name: f.default if f.default is not None else None
            for f in fields
        }
        return json.dumps(template, indent=2)
    
    elif format == "csv_header":
        return ",".join(f.name for f in fields)
    
    else:
        raise ValueError(f"Unknown format: {format}")
