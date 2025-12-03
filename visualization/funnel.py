"""
Funnel Plot Visualization for T3-Meta.

This module provides functions for creating funnel plots
for assessing publication bias in meta-analyses.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from scipy import stats


def funnel_plot(
    estimates: np.ndarray,
    se: np.ndarray,
    pooled_estimate: Optional[float] = None,
    study_names: Optional[List[str]] = None,
    color_by: Optional[np.ndarray] = None,
    color_label: str = "",
    title: str = "Funnel Plot",
    xlabel: str = "Effect Estimate",
    ylabel: str = "Standard Error",
    show_ci: bool = True,
    ci_levels: List[float] = [0.90, 0.95, 0.99],
    invert_y: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    **kwargs
) -> Any:
    """
    Create a standard funnel plot.
    
    Args:
        estimates: Effect estimates
        se: Standard errors
        pooled_estimate: Pooled effect (vertical line)
        study_names: Names for hover/annotation
        color_by: Values for coloring points
        color_label: Label for color scale
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_ci: Show confidence interval contours
        ci_levels: Confidence levels for contours
        invert_y: Invert Y-axis (SE increases downward)
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure or data dictionary
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        return _funnel_plot_data(
            estimates, se, pooled_estimate, study_names,
            title, xlabel, ylabel
        )
    
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    n = len(estimates)
    
    # Compute pooled estimate if not provided
    if pooled_estimate is None:
        weights = 1 / (se ** 2)
        pooled_estimate = np.sum(weights * estimates) / np.sum(weights)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Confidence interval funnel regions
    if show_ci:
        se_range = np.linspace(0, max(se) * 1.2, 100)
        
        for i, level in enumerate(sorted(ci_levels)):
            z = stats.norm.ppf((1 + level) / 2)
            
            lower = pooled_estimate - z * se_range
            upper = pooled_estimate + z * se_range
            
            alpha = 0.15 + 0.1 * i
            color = plt.cm.Blues(0.3 + 0.2 * i)
            
            ax.fill_betweenx(
                se_range, lower, upper,
                alpha=alpha, color=color,
                label=f'{int(level*100)}% CI'
            )
    
    # Plot points
    if color_by is not None:
        scatter = ax.scatter(
            estimates, se, c=color_by, s=50,
            cmap='RdYlBu_r', edgecolors='black', linewidth=0.5, zorder=5
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            estimates, se, c='steelblue', s=50,
            edgecolors='black', linewidth=0.5, zorder=5
        )
    
    # Pooled estimate line
    ax.axvline(x=pooled_estimate, color='darkred', linestyle='-', linewidth=1.5,
               label='Pooled estimate')
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Invert Y-axis (larger SE at bottom)
    if invert_y:
        ax.invert_yaxis()
    
    # Set Y-axis to start at 0
    if invert_y:
        ax.set_ylim(max(se) * 1.2, 0)
    else:
        ax.set_ylim(0, max(se) * 1.2)
    
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    return fig


def contour_enhanced_funnel(
    estimates: np.ndarray,
    se: np.ndarray,
    pooled_estimate: Optional[float] = None,
    tau_squared: float = 0.0,
    p_value_contours: List[float] = [0.01, 0.05, 0.1],
    title: str = "Contour-Enhanced Funnel Plot",
    xlabel: str = "Effect Estimate",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> Any:
    """
    Create contour-enhanced funnel plot showing significance regions.
    
    Args:
        estimates: Effect estimates
        se: Standard errors
        pooled_estimate: Pooled effect estimate
        tau_squared: Between-study variance
        p_value_contours: P-value levels for contours
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _funnel_plot_data(
            estimates, se, pooled_estimate, None,
            title, xlabel, "Standard Error"
        )
    
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    
    if pooled_estimate is None:
        weights = 1 / (se ** 2 + tau_squared)
        pooled_estimate = np.sum(weights * estimates) / np.sum(weights)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Significance contours (testing against null = 0)
    se_range = np.linspace(0.001, max(se) * 1.3, 200)
    
    colors = plt.cm.RdYlGn_r([0.1, 0.3, 0.5])
    
    for i, p in enumerate(sorted(p_value_contours)):
        z = stats.norm.ppf(1 - p / 2)
        
        # Contours for positive significant effects
        upper_pos = z * se_range
        # Contours for negative significant effects
        upper_neg = -z * se_range
        
        ax.plot(upper_pos, se_range, color=colors[i], linestyle='--',
                linewidth=1.5, label=f'p = {p}')
        ax.plot(upper_neg, se_range, color=colors[i], linestyle='--',
                linewidth=1.5)
    
    # Fill significance regions
    ax.fill_betweenx(
        se_range,
        -stats.norm.ppf(0.975) * se_range,
        stats.norm.ppf(0.975) * se_range,
        alpha=0.1, color='gray', label='Non-significant (p > 0.05)'
    )
    
    # Plot points
    # Color by significance
    z_scores = estimates / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    colors_points = np.where(p_values < 0.05, 'darkred', 'steelblue')
    
    for i in range(len(estimates)):
        ax.scatter(
            estimates[i], se[i], c=colors_points[i], s=60,
            edgecolors='black', linewidth=0.5, zorder=5
        )
    
    # Pooled estimate line
    ax.axvline(x=pooled_estimate, color='black', linestyle='-', linewidth=1.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Standard Error")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_ylim(max(se) * 1.3, 0)
    
    ax.legend(loc='lower left', fontsize=8)
    
    plt.tight_layout()
    
    return fig


def trim_fill_funnel(
    estimates: np.ndarray,
    se: np.ndarray,
    imputed_estimates: np.ndarray,
    imputed_se: np.ndarray,
    original_pooled: float,
    adjusted_pooled: float,
    title: str = "Trim-and-Fill Funnel Plot",
    xlabel: str = "Effect Estimate",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> Any:
    """
    Create funnel plot showing trim-and-fill results.
    
    Args:
        estimates: Original effect estimates
        se: Original standard errors
        imputed_estimates: Imputed effect estimates
        imputed_se: Imputed standard errors
        original_pooled: Original pooled estimate
        adjusted_pooled: Adjusted pooled estimate
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "type": "trim_fill_funnel",
            "original_estimates": estimates.tolist(),
            "original_se": se.tolist(),
            "imputed_estimates": imputed_estimates.tolist(),
            "imputed_se": imputed_se.tolist(),
            "original_pooled": original_pooled,
            "adjusted_pooled": adjusted_pooled
        }
    
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    imputed_estimates = np.asarray(imputed_estimates).flatten()
    imputed_se = np.asarray(imputed_se).flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # CI funnel for adjusted estimate
    all_se = np.concatenate([se, imputed_se])
    se_range = np.linspace(0, max(all_se) * 1.2, 100)
    
    for level in [0.95]:
        z = stats.norm.ppf((1 + level) / 2)
        lower = adjusted_pooled - z * se_range
        upper = adjusted_pooled + z * se_range
        ax.fill_betweenx(se_range, lower, upper, alpha=0.15, color='blue')
    
    # Original studies
    ax.scatter(
        estimates, se, c='steelblue', s=60,
        edgecolors='black', linewidth=0.5, zorder=5,
        label=f'Original (n={len(estimates)})'
    )
    
    # Imputed studies
    if len(imputed_estimates) > 0:
        ax.scatter(
            imputed_estimates, imputed_se, c='white', s=60,
            edgecolors='steelblue', linewidth=1.5, zorder=5,
            marker='o', label=f'Imputed (n={len(imputed_estimates)})'
        )
    
    # Pooled estimates
    ax.axvline(x=original_pooled, color='darkred', linestyle='-',
               linewidth=1.5, label=f'Original: {original_pooled:.3f}')
    ax.axvline(x=adjusted_pooled, color='darkblue', linestyle='--',
               linewidth=1.5, label=f'Adjusted: {adjusted_pooled:.3f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Standard Error")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_ylim(max(all_se) * 1.2, 0)
    
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def precision_funnel(
    estimates: np.ndarray,
    se: np.ndarray,
    pooled_estimate: Optional[float] = None,
    title: str = "Precision Funnel Plot",
    xlabel: str = "Effect Estimate",
    ylabel: str = "Precision (1/SE)",
    figsize: Tuple[float, float] = (8, 8),
    **kwargs
) -> Any:
    """
    Create funnel plot with precision (1/SE) on Y-axis.
    
    Args:
        estimates: Effect estimates
        se: Standard errors
        pooled_estimate: Pooled effect estimate
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "type": "precision_funnel",
            "estimates": estimates.tolist(),
            "precision": (1 / se).tolist()
        }
    
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    precision = 1 / se
    
    if pooled_estimate is None:
        weights = precision ** 2
        pooled_estimate = np.sum(weights * estimates) / np.sum(weights)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # CI contours
    prec_range = np.linspace(0.1, max(precision) * 1.1, 100)
    
    for level in [0.95]:
        z = stats.norm.ppf((1 + level) / 2)
        se_equiv = 1 / prec_range
        lower = pooled_estimate - z * se_equiv
        upper = pooled_estimate + z * se_equiv
        ax.fill_betweenx(prec_range, lower, upper, alpha=0.15, color='blue')
    
    # Plot points
    ax.scatter(
        estimates, precision, c='steelblue', s=60,
        edgecolors='black', linewidth=0.5, zorder=5
    )
    
    ax.axvline(x=pooled_estimate, color='darkred', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(precision) * 1.1)
    
    plt.tight_layout()
    
    return fig


def regression_funnel(
    estimates: np.ndarray,
    se: np.ndarray,
    egger_intercept: float,
    egger_slope: float,
    title: str = "Egger's Regression Funnel Plot",
    xlabel: str = "Precision (1/SE)",
    ylabel: str = "Standardized Effect (Effect/SE)",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> Any:
    """
    Create Egger's regression funnel plot.
    
    Args:
        estimates: Effect estimates
        se: Standard errors
        egger_intercept: Egger's test intercept
        egger_slope: Egger's test slope
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "type": "regression_funnel",
            "estimates": estimates.tolist(),
            "se": se.tolist(),
            "egger_intercept": egger_intercept,
            "egger_slope": egger_slope
        }
    
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    
    precision = 1 / se
    standardized = estimates / se
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(
        precision, standardized, c='steelblue', s=60,
        edgecolors='black', linewidth=0.5, zorder=5
    )
    
    # Egger's regression line
    x_line = np.linspace(0, max(precision) * 1.1, 100)
    y_line = egger_intercept + egger_slope * x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'Egger: intercept = {egger_intercept:.3f}')
    
    # Reference line (no bias: intercept = 0)
    y_ref = egger_slope * x_line
    ax.plot(x_line, y_ref, 'k--', linewidth=1, alpha=0.5,
            label='No asymmetry (intercept = 0)')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, max(precision) * 1.1)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig


def _funnel_plot_data(
    estimates, se, pooled_estimate, study_names,
    title, xlabel, ylabel
) -> Dict[str, Any]:
    """Return funnel plot data for non-matplotlib rendering."""
    estimates = np.asarray(estimates).flatten()
    se = np.asarray(se).flatten()
    
    if pooled_estimate is None:
        weights = 1 / (se ** 2)
        pooled_estimate = float(np.sum(weights * estimates) / np.sum(weights))
    
    return {
        "type": "funnel_plot",
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "pooled_estimate": pooled_estimate,
        "studies": [
            {
                "name": study_names[i] if study_names else f"Study_{i+1}",
                "estimate": float(estimates[i]),
                "se": float(se[i])
            }
            for i in range(len(estimates))
        ]
    }
