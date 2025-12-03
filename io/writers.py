"""
Data writers for T3-Meta.

This module provides functions for exporting T3-Meta analyses
to various file formats.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING
import json
import csv
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from t3meta.core.study import Study
    from t3meta.core.registry import StudyRegistry
    from t3meta.models.base import ModelResults


def write_csv(
    studies: Union[List['Study'], 'StudyRegistry'],
    filepath: Union[str, Path],
    include_design: bool = True,
    include_results: bool = False,
    results: Optional['ModelResults'] = None,
    delimiter: str = ",",
    **kwargs
) -> None:
    """
    Write study data to CSV file.
    
    Args:
        studies: List of Study objects or StudyRegistry
        filepath: Output file path
        include_design: Include design feature columns
        include_results: Include fitted results
        results: ModelResults object (if include_results)
        delimiter: CSV delimiter
        **kwargs: Additional arguments
    """
    filepath = Path(filepath)
    
    # Handle StudyRegistry
    if hasattr(studies, 'studies'):
        studies = studies.studies
    
    if not studies:
        raise ValueError("No studies to write")
    
    # Build column headers
    columns = [
        "study", "effect_estimate", "effect_measure", "se",
        "ci_lower", "ci_upper", "year", "n_total"
    ]
    
    if include_design and studies[0].design_features is not None:
        design_cols = list(studies[0].design_features.to_dict().keys())
        columns.extend([f"design_{c}" for c in design_cols])
    else:
        design_cols = []
    
    if include_results and results is not None:
        columns.extend(["weight", "bias", "fitted", "residual"])
    
    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(columns)
        
        for i, study in enumerate(studies):
            row = [
                study.name,
                study.effect_estimate,
                study.effect_measure.value if hasattr(study.effect_measure, 'value') else study.effect_measure,
                study.se,
                study.ci_lower,
                study.ci_upper,
                study.year,
                study.n_total,
            ]
            
            if design_cols and study.design_features:
                design_dict = study.design_features.to_dict()
                for col in design_cols:
                    row.append(design_dict.get(col, ""))
            
            if include_results and results is not None:
                row.extend([
                    results.study_weights[i] if i < len(results.study_weights) else "",
                    results.study_bias[i] if i < len(results.study_bias) else "",
                    results.fitted_values[i] if i < len(results.fitted_values) else "",
                    results.residuals[i] if i < len(results.residuals) else "",
                ])
            
            writer.writerow(row)


def write_json(
    data: Union[Dict, List, 'StudyRegistry', 'ModelResults'],
    filepath: Union[str, Path],
    indent: int = 2,
    **kwargs
) -> None:
    """
    Write data to JSON file.
    
    Args:
        data: Data to write (dict, list, or T3-Meta object)
        filepath: Output file path
        indent: JSON indentation
        **kwargs: Additional arguments
    """
    filepath = Path(filepath)
    
    # Convert T3-Meta objects to dict
    if hasattr(data, 'to_dict'):
        data = data.to_dict()
    
    # Handle numpy arrays
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        return obj
    
    data = convert_for_json(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def write_excel(
    studies: Union[List['Study'], 'StudyRegistry'],
    filepath: Union[str, Path],
    sheet_name: str = "Studies",
    include_design: bool = True,
    include_results: bool = False,
    results: Optional['ModelResults'] = None,
    **kwargs
) -> None:
    """
    Write study data to Excel file.
    
    Requires openpyxl to be installed.
    
    Args:
        studies: List of Study objects or StudyRegistry
        filepath: Output file path
        sheet_name: Excel sheet name
        include_design: Include design feature columns
        include_results: Include fitted results
        results: ModelResults object (if include_results)
        **kwargs: Additional arguments
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Excel file writing")
    
    filepath = Path(filepath)
    
    # Handle StudyRegistry
    if hasattr(studies, 'studies'):
        studies = studies.studies
    
    # Build DataFrame
    data = []
    for i, study in enumerate(studies):
        row = {
            "study": study.name,
            "effect_estimate": study.effect_estimate,
            "effect_measure": study.effect_measure.value if hasattr(study.effect_measure, 'value') else study.effect_measure,
            "se": study.se,
            "ci_lower": study.ci_lower,
            "ci_upper": study.ci_upper,
            "year": study.year,
            "n_total": study.n_total,
        }
        
        if include_design and study.design_features:
            design_dict = study.design_features.to_dict()
            for k, v in design_dict.items():
                row[f"design_{k}"] = v
        
        if include_results and results is not None:
            row["weight"] = results.study_weights[i] if i < len(results.study_weights) else None
            row["bias"] = results.study_bias[i] if i < len(results.study_bias) else None
            row["fitted"] = results.fitted_values[i] if i < len(results.fitted_values) else None
            row["residual"] = results.residuals[i] if i < len(results.residuals) else None
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_excel(filepath, sheet_name=sheet_name, index=False)


def export_to_revman(
    studies: Union[List['Study'], 'StudyRegistry'],
    filepath: Union[str, Path],
    review_title: str = "T3-Meta Analysis",
    outcome_name: str = "Primary Outcome",
    **kwargs
) -> None:
    """
    Export study data to RevMan-compatible XML format.
    
    Args:
        studies: List of Study objects or StudyRegistry
        filepath: Output file path
        review_title: Title of the review
        outcome_name: Name of the outcome
        **kwargs: Additional arguments
    """
    filepath = Path(filepath)
    
    # Handle StudyRegistry
    if hasattr(studies, 'studies'):
        studies = studies.studies
    
    # Build XML structure
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<COCHRANE_REVIEW>',
        f'  <COVER_SHEET>',
        f'    <TITLE>{review_title}</TITLE>',
        f'  </COVER_SHEET>',
        f'  <ANALYSES_AND_DATA>',
        f'    <COMPARISON ID="CMP-001" NO="1">',
        f'      <NAME>{outcome_name}</NAME>',
        f'      <OUTCOME ID="OUT-001" NO="1">',
        f'        <NAME>{outcome_name}</NAME>',
    ]
    
    for i, study in enumerate(studies):
        study_id = f"STD-{i+1:03d}"
        effect = study.effect_estimate
        se = study.se or 0
        
        lines.append(f'        <STUDY ID="{study_id}">')
        lines.append(f'          <NAME>{study.name}</NAME>')
        lines.append(f'          <EFFECT>{effect}</EFFECT>')
        lines.append(f'          <SE>{se}</SE>')
        if study.ci_lower is not None:
            lines.append(f'          <CI_START>{study.ci_lower}</CI_START>')
        if study.ci_upper is not None:
            lines.append(f'          <CI_END>{study.ci_upper}</CI_END>')
        lines.append(f'        </STUDY>')
    
    lines.extend([
        f'      </OUTCOME>',
        f'    </COMPARISON>',
        f'  </ANALYSES_AND_DATA>',
        '</COCHRANE_REVIEW>',
    ])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_to_prisma(
    registry: 'StudyRegistry',
    results: Optional['ModelResults'],
    filepath: Union[str, Path],
    format: str = "markdown",
    **kwargs
) -> None:
    """
    Export PRISMA-style summary report.
    
    Args:
        registry: StudyRegistry object
        results: ModelResults object (optional)
        filepath: Output file path
        format: Output format ('markdown', 'html', 'txt')
        **kwargs: Additional arguments
    """
    filepath = Path(filepath)
    
    lines = []
    
    if format == "markdown":
        lines.append("# PRISMA-Style Summary Report")
        lines.append("")
        lines.append("## Study Characteristics")
        lines.append("")
        lines.append("| Study | Effect | 95% CI | Design | Year |")
        lines.append("|-------|--------|--------|--------|------|")
        
        for study in registry.studies:
            effect_str = f"{study.effect_estimate:.3f}"
            ci_str = f"[{study.ci_lower:.3f}, {study.ci_upper:.3f}]" if study.ci_lower else "N/A"
            design = "RCT" if study.design_features and study.design_features.is_rct else "Obs"
            year = study.year or "N/A"
            lines.append(f"| {study.name} | {effect_str} | {ci_str} | {design} | {year} |")
        
        lines.append("")
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Total studies**: {len(registry)}")
        lines.append(f"- **Total participants**: {registry.total_sample_size or 'N/A'}")
        
        if results is not None:
            exp_est, exp_ci = results.get_theta_star_exp()
            lines.append("")
            lines.append("## Pooled Results")
            lines.append("")
            lines.append(f"- **Target Trial Effect (θ*)**: {results.theta_star:.4f}")
            lines.append(f"- **95% CI**: [{results.theta_star_ci[0]:.4f}, {results.theta_star_ci[1]:.4f}]")
            lines.append(f"- **Exponentiated**: {exp_est:.4f} [{exp_ci[0]:.4f}, {exp_ci[1]:.4f}]")
            lines.append(f"- **Heterogeneity (I²)**: {results.i_squared * 100:.1f}%")
            lines.append(f"- **τ²**: {results.tau_squared:.4f}")
    
    elif format == "html":
        lines.append("<!DOCTYPE html>")
        lines.append("<html><head><title>PRISMA Summary</title></head><body>")
        lines.append("<h1>PRISMA-Style Summary Report</h1>")
        lines.append("<h2>Study Characteristics</h2>")
        lines.append("<table border='1'>")
        lines.append("<tr><th>Study</th><th>Effect</th><th>95% CI</th><th>Design</th><th>Year</th></tr>")
        
        for study in registry.studies:
            effect_str = f"{study.effect_estimate:.3f}"
            ci_str = f"[{study.ci_lower:.3f}, {study.ci_upper:.3f}]" if study.ci_lower else "N/A"
            design = "RCT" if study.design_features and study.design_features.is_rct else "Obs"
            year = study.year or "N/A"
            lines.append(f"<tr><td>{study.name}</td><td>{effect_str}</td><td>{ci_str}</td><td>{design}</td><td>{year}</td></tr>")
        
        lines.append("</table>")
        lines.append("</body></html>")
    
    else:  # txt
        lines.append("PRISMA-Style Summary Report")
        lines.append("=" * 50)
        lines.append("")
        lines.append("Study Characteristics:")
        for study in registry.studies:
            lines.append(f"  - {study.name}: {study.effect_estimate:.3f}")
        
        if results is not None:
            lines.append("")
            lines.append("Pooled Results:")
            lines.append(f"  Target Trial Effect: {results.theta_star:.4f}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_results_table(
    results: 'ModelResults',
    filepath: Union[str, Path],
    format: str = "csv",
    **kwargs
) -> None:
    """
    Export model results to tabular format.
    
    Args:
        results: ModelResults object
        filepath: Output file path
        format: Output format ('csv', 'latex', 'markdown')
        **kwargs: Additional arguments
    """
    filepath = Path(filepath)
    
    # Build data rows
    rows = [
        ("Target Trial Effect (θ*)", f"{results.theta_star:.4f}",
         f"[{results.theta_star_ci[0]:.4f}, {results.theta_star_ci[1]:.4f}]"),
        ("Standard Error", f"{results.theta_star_se:.4f}", ""),
        ("τ²", f"{results.tau_squared:.4f}",
         f"[{results.tau_squared_ci[0]:.4f}, {results.tau_squared_ci[1]:.4f}]"),
        ("I²", f"{results.i_squared * 100:.1f}%", ""),
        ("Q statistic", f"{results.q_statistic:.2f}", f"p = {results.q_pvalue:.4f}"),
    ]
    
    # Add bias coefficients
    for i, name in enumerate(results.feature_names):
        coef = results.beta[i]
        se = results.beta_se[i]
        ci = results.beta_ci[i] if len(results.beta_ci) > i else (np.nan, np.nan)
        rows.append((f"β ({name})", f"{coef:.4f} (SE: {se:.4f})",
                    f"[{ci[0]:.4f}, {ci[1]:.4f}]"))
    
    if format == "csv":
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Estimate", "95% CI"])
            writer.writerows(rows)
    
    elif format == "latex":
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\begin{tabular}{lcc}",
            "\\hline",
            "Parameter & Estimate & 95\\% CI \\\\",
            "\\hline",
        ]
        for row in rows:
            lines.append(f"{row[0]} & {row[1]} & {row[2]} \\\\")
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{T3-Meta Analysis Results}",
            "\\end{table}",
        ])
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    elif format == "markdown":
        lines = [
            "| Parameter | Estimate | 95% CI |",
            "|-----------|----------|--------|",
        ]
        for row in rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
