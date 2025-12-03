"""
Data readers for T3-Meta.

This module provides functions for reading study data from
various file formats and converting them to T3-Meta objects.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import csv
from pathlib import Path
import numpy as np

from t3meta.core.study import Study, DesignFeatures, DesignMap
from t3meta.core.estimand import EffectMeasure


def read_csv(
    filepath: Union[str, Path],
    effect_col: str = "effect",
    se_col: str = "se",
    ci_lower_col: Optional[str] = "ci_lower",
    ci_upper_col: Optional[str] = "ci_upper",
    name_col: str = "study",
    effect_measure_col: Optional[str] = None,
    default_effect_measure: str = "HR",
    design_cols: Optional[List[str]] = None,
    delimiter: str = ",",
    **kwargs
) -> List[Study]:
    """
    Read study data from CSV file.
    
    Args:
        filepath: Path to CSV file
        effect_col: Column name for effect estimates
        se_col: Column name for standard errors
        ci_lower_col: Column name for lower CI bounds
        ci_upper_col: Column name for upper CI bounds
        name_col: Column name for study names
        effect_measure_col: Column name for effect measure
        default_effect_measure: Default effect measure if not in file
        design_cols: Column names to use as design features
        delimiter: CSV delimiter
        **kwargs: Additional arguments passed to Study
        
    Returns:
        List of Study objects
    """
    filepath = Path(filepath)
    
    studies = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            # Extract core data
            name = row.get(name_col, f"Study_{len(studies)+1}")
            effect = float(row[effect_col])
            
            # SE can be computed from CI if not provided
            se = None
            if se_col in row and row[se_col]:
                se = float(row[se_col])
            
            ci_lower = None
            ci_upper = None
            if ci_lower_col and ci_lower_col in row and row[ci_lower_col]:
                ci_lower = float(row[ci_lower_col])
            if ci_upper_col and ci_upper_col in row and row[ci_upper_col]:
                ci_upper = float(row[ci_upper_col])
            
            # Effect measure
            if effect_measure_col and effect_measure_col in row:
                effect_measure = row[effect_measure_col]
            else:
                effect_measure = default_effect_measure
            
            # Design features
            design_features = None
            if design_cols:
                features_dict = {}
                for col in design_cols:
                    if col in row:
                        val = row[col]
                        # Try to convert to appropriate type
                        try:
                            val = float(val)
                            if val == int(val):
                                val = int(val)
                        except ValueError:
                            if val.lower() in ('true', 'yes', '1'):
                                val = True
                            elif val.lower() in ('false', 'no', '0'):
                                val = False
                        features_dict[col] = val
                
                if features_dict:
                    design_features = DesignFeatures(**features_dict)
            
            # Create study
            study = Study(
                name=name,
                effect_estimate=effect,
                effect_measure=effect_measure,
                se=se,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                design_features=design_features,
                **kwargs
            )
            studies.append(study)
    
    return studies


def read_json(
    filepath: Union[str, Path],
    **kwargs
) -> List[Study]:
    """
    Read study data from JSON file.
    
    Expected format:
    {
        "studies": [
            {
                "name": "Study 1",
                "effect_estimate": 0.85,
                "effect_measure": "HR",
                "se": 0.1,
                ...
            },
            ...
        ]
    }
    
    Args:
        filepath: Path to JSON file
        **kwargs: Additional arguments
        
    Returns:
        List of Study objects
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    studies = []
    
    study_list = data.get("studies", data) if isinstance(data, dict) else data
    
    for study_data in study_list:
        study = Study.from_dict(study_data)
        studies.append(study)
    
    return studies


def read_excel(
    filepath: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    effect_col: str = "effect",
    se_col: str = "se",
    name_col: str = "study",
    **kwargs
) -> List[Study]:
    """
    Read study data from Excel file.
    
    Requires openpyxl or xlrd to be installed.
    
    Args:
        filepath: Path to Excel file
        sheet_name: Sheet name or index
        effect_col: Column name for effect estimates
        se_col: Column name for standard errors
        name_col: Column name for study names
        **kwargs: Additional arguments passed to read_csv logic
        
    Returns:
        List of Study objects
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Excel file reading")
    
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    return studies_from_dataframe(
        df,
        effect_col=effect_col,
        se_col=se_col,
        name_col=name_col,
        **kwargs
    )


def read_revman(
    filepath: Union[str, Path],
    outcome_id: Optional[str] = None
) -> List[Study]:
    """
    Read study data from RevMan/Cochrane XML file.
    
    Args:
        filepath: Path to RevMan XML file
        outcome_id: Specific outcome ID to extract (optional)
        
    Returns:
        List of Study objects
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree is required for RevMan file reading")
    
    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    studies = []
    
    # Find all study data elements
    # RevMan structure varies, so we try multiple paths
    for study_elem in root.findall('.//STUDY'):
        name = study_elem.get('NAME', study_elem.get('ID', f"Study_{len(studies)+1}"))
        
        # Look for effect data
        for outcome in study_elem.findall('.//OUTCOME'):
            if outcome_id and outcome.get('ID') != outcome_id:
                continue
            
            effect = outcome.get('EFFECT')
            se = outcome.get('SE')
            ci_lower = outcome.get('CI_START')
            ci_upper = outcome.get('CI_END')
            
            if effect is not None:
                study = Study(
                    name=name,
                    effect_estimate=float(effect),
                    effect_measure="OR",  # RevMan default
                    se=float(se) if se else None,
                    ci_lower=float(ci_lower) if ci_lower else None,
                    ci_upper=float(ci_upper) if ci_upper else None,
                )
                studies.append(study)
    
    # Alternative structure for newer RevMan formats
    if not studies:
        for study_elem in root.findall('.//*[@STUDY_ID]'):
            name = study_elem.get('STUDY_ID')
            effect = study_elem.get('EFFECT_SIZE') or study_elem.get('EFFECT')
            se = study_elem.get('SE')
            
            if effect is not None:
                study = Study(
                    name=name,
                    effect_estimate=float(effect),
                    effect_measure="OR",
                    se=float(se) if se else None,
                )
                studies.append(study)
    
    return studies


def studies_from_dataframe(
    df,
    effect_col: str = "effect",
    se_col: str = "se",
    ci_lower_col: Optional[str] = "ci_lower",
    ci_upper_col: Optional[str] = "ci_upper",
    name_col: str = "study",
    effect_measure_col: Optional[str] = None,
    default_effect_measure: str = "HR",
    design_cols: Optional[List[str]] = None,
    year_col: Optional[str] = None,
    n_total_col: Optional[str] = None,
    **kwargs
) -> List[Study]:
    """
    Convert pandas DataFrame to list of Study objects.
    
    Args:
        df: pandas DataFrame with study data
        effect_col: Column name for effect estimates
        se_col: Column name for standard errors
        ci_lower_col: Column name for lower CI bounds
        ci_upper_col: Column name for upper CI bounds
        name_col: Column name for study names
        effect_measure_col: Column name for effect measure
        default_effect_measure: Default effect measure if not in DataFrame
        design_cols: Column names to use as design features
        year_col: Column name for publication year
        n_total_col: Column name for total sample size
        **kwargs: Additional arguments passed to Study
        
    Returns:
        List of Study objects
    """
    studies = []
    
    for idx, row in df.iterrows():
        # Extract core data
        name = row.get(name_col, f"Study_{idx+1}")
        effect = float(row[effect_col])
        
        se = None
        if se_col in row and pd.notna(row[se_col]):
            se = float(row[se_col])
        
        ci_lower = None
        ci_upper = None
        if ci_lower_col and ci_lower_col in row and pd.notna(row[ci_lower_col]):
            ci_lower = float(row[ci_lower_col])
        if ci_upper_col and ci_upper_col in row and pd.notna(row[ci_upper_col]):
            ci_upper = float(row[ci_upper_col])
        
        # Effect measure
        if effect_measure_col and effect_measure_col in row:
            effect_measure = row[effect_measure_col]
        else:
            effect_measure = default_effect_measure
        
        # Design features
        design_features = None
        if design_cols:
            features_dict = {}
            for col in design_cols:
                if col in row and pd.notna(row[col]):
                    features_dict[col] = row[col]
            if features_dict:
                design_features = DesignFeatures(**features_dict)
        
        # Optional fields
        year = int(row[year_col]) if year_col and year_col in row and pd.notna(row[year_col]) else None
        n_total = int(row[n_total_col]) if n_total_col and n_total_col in row and pd.notna(row[n_total_col]) else None
        
        study = Study(
            name=name,
            effect_estimate=effect,
            effect_measure=effect_measure,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            design_features=design_features,
            year=year,
            n_total=n_total,
            **kwargs
        )
        studies.append(study)
    
    return studies


# Check for pandas availability
try:
    import pandas as pd
except ImportError:
    pd = None
