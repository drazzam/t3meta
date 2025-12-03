"""
Bias Visualization for T3-Meta.

This module provides functions for visualizing bias contributions
and design feature effects in T3-Meta analyses.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np


def bias_contribution_plot(
    feature_names: List[str],
    coefficients: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    prior_means: Optional[np.ndarray] = None,
    title: str = "Bias Coefficients",
    xlabel: str = "Coefficient (β)",
    figsize: Tuple[float, float] = (8, 6),
    sort_by_magnitude: bool = True,
    show_priors: bool = True,
    **kwargs
) -> Any:
    """
    Create coefficient plot showing bias contributions.
    
    Args:
        feature_names: Names of design features
        coefficients: Estimated coefficients
        ci_lower: Lower CI bounds
        ci_upper: Upper CI bounds
        prior_means: Prior means for comparison
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        sort_by_magnitude: Sort by absolute coefficient value
        show_priors: Show prior means as reference points
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object or data dict
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _bias_plot_data(
            feature_names, coefficients, ci_lower, ci_upper,
            prior_means, title, xlabel
        )
    
    coefficients = np.asarray(coefficients).flatten()
    ci_lower = np.asarray(ci_lower).flatten()
    ci_upper = np.asarray(ci_upper).flatten()
    n = len(coefficients)
    
    # Sort by magnitude if requested
    if sort_by_magnitude:
        order = np.argsort(np.abs(coefficients))[::-1]
    else:
        order = np.arange(n)
    
    coefficients = coefficients[order]
    ci_lower = ci_lower[order]
    ci_upper = ci_upper[order]
    feature_names = [feature_names[i] for i in order]
    if prior_means is not None:
        prior_means = np.asarray(prior_means)[order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(n)
    
    # Error bars
    xerr = np.array([coefficients - ci_lower, ci_upper - coefficients])
    
    # Determine colors based on sign and significance
    colors = []
    for i in range(n):
        if ci_lower[i] > 0:
            colors.append('darkred')  # Significant positive bias
        elif ci_upper[i] < 0:
            colors.append('darkblue')  # Significant negative bias
        else:
            colors.append('gray')  # Not significant
    
    # Plot coefficients with error bars
    ax.errorbar(
        coefficients, y_pos, xerr=xerr,
        fmt='o', capsize=3, capthick=1,
        color='black', ecolor='gray', markersize=8
    )
    
    # Color the markers
    for i, (coef, color) in enumerate(zip(coefficients, colors)):
        ax.scatter([coef], [y_pos[i]], c=color, s=80, zorder=5)
    
    # Show prior means
    if show_priors and prior_means is not None:
        ax.scatter(
            prior_means, y_pos, marker='|', s=200,
            c='green', alpha=0.7, label='Prior mean', zorder=4
        )
    
    # Reference line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
               markersize=10, label='Positive bias (sig.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
               markersize=10, label='Negative bias (sig.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Not significant'),
    ]
    if show_priors and prior_means is not None:
        legend_elements.append(
            Line2D([0], [0], marker='|', color='w', markerfacecolor='green',
                   markersize=15, label='Prior mean')
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig


def bias_heatmap(
    study_names: List[str],
    feature_names: List[str],
    design_matrix: np.ndarray,
    coefficients: np.ndarray,
    show_contribution: bool = True,
    title: str = "Design Feature Matrix",
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = "RdBu_r",
    **kwargs
) -> Any:
    """
    Create heatmap showing design features and bias contributions.
    
    Args:
        study_names: Names of studies
        feature_names: Names of design features
        design_matrix: Design matrix (studies x features)
        coefficients: Bias coefficients
        show_contribution: Show feature*coefficient product
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object or data dict
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "type": "bias_heatmap",
            "study_names": study_names,
            "feature_names": feature_names,
            "design_matrix": design_matrix.tolist(),
            "coefficients": coefficients.tolist(),
        }
    
    design_matrix = np.asarray(design_matrix)
    coefficients = np.asarray(coefficients).flatten()
    n_studies, n_features = design_matrix.shape
    
    # Compute contributions if requested
    if show_contribution:
        contributions = design_matrix * coefficients
        display_data = contributions
        cbar_label = "Bias Contribution (X·β)"
    else:
        display_data = design_matrix
        cbar_label = "Feature Value"
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [4, 1]}
    )
    
    # Main heatmap
    vmax = np.max(np.abs(display_data))
    im = ax1.imshow(
        display_data, aspect='auto', cmap=cmap,
        vmin=-vmax, vmax=vmax
    )
    
    # Labels
    ax1.set_xticks(np.arange(n_features))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_yticks(np.arange(n_studies))
    ax1.set_yticklabels(study_names)
    ax1.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label(cbar_label)
    
    # Row sums (total bias per study)
    if show_contribution:
        total_bias = np.sum(contributions, axis=1)
        
        ax2.barh(np.arange(n_studies), total_bias, color='steelblue')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_yticks([])
        ax2.set_xlabel("Total Bias")
        ax2.set_title("Study Bias")
        
        # Color bars by sign
        for i, bias in enumerate(total_bias):
            color = 'darkred' if bias > 0 else 'darkblue'
            ax2.barh([i], [bias], color=color, alpha=0.7)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    
    return fig


def bias_decomposition_plot(
    study_names: List[str],
    original_effects: np.ndarray,
    bias_components: Dict[str, np.ndarray],
    adjusted_effects: np.ndarray,
    effect_measure: str = "Effect",
    title: str = "Bias Decomposition",
    figsize: Tuple[float, float] = (12, 8),
    **kwargs
) -> Any:
    """
    Create stacked bar chart showing bias decomposition.
    
    Args:
        study_names: Study names
        original_effects: Original effect estimates
        bias_components: Dict mapping feature names to bias arrays
        adjusted_effects: Bias-adjusted effects
        effect_measure: Label for effect measure
        title: Plot title
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return {
            "type": "bias_decomposition",
            "study_names": study_names,
            "original_effects": original_effects.tolist(),
            "bias_components": {k: v.tolist() for k, v in bias_components.items()},
            "adjusted_effects": adjusted_effects.tolist(),
        }
    
    n_studies = len(study_names)
    n_components = len(bias_components)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(n_studies)
    
    # Plot adjusted effects as baseline
    ax.barh(y_pos, adjusted_effects, height=0.6, color='steelblue',
            label='Adjusted effect (θ*)', alpha=0.8)
    
    # Stack bias components
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_components))
    cumulative = adjusted_effects.copy()
    
    for i, (name, component) in enumerate(bias_components.items()):
        component = np.asarray(component)
        ax.barh(y_pos, component, height=0.6, left=cumulative,
                color=colors[i], label=name, alpha=0.7)
        cumulative = cumulative + component
    
    # Mark original effects
    ax.scatter(original_effects, y_pos, marker='|', s=200, c='black',
               label='Original estimate', zorder=5)
    
    # Reference line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_names)
    ax.set_xlabel(effect_measure)
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig


def coefficient_comparison_plot(
    feature_names: List[str],
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Coefficient Comparison Across Models",
    xlabel: str = "Coefficient (β)",
    figsize: Tuple[float, float] = (10, 6),
    **kwargs
) -> Any:
    """
    Compare coefficients across different models.
    
    Args:
        feature_names: Names of features
        models: Dict mapping model name to (coefficients, SEs)
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "type": "coefficient_comparison",
            "feature_names": feature_names,
            "models": {k: {"coef": v[0].tolist(), "se": v[1].tolist()}
                      for k, v in models.items()},
        }
    
    n_features = len(feature_names)
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(n_features)
    height = 0.8 / n_models
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model_name, (coefs, ses)) in enumerate(models.items()):
        coefs = np.asarray(coefs)
        ses = np.asarray(ses)
        
        offset = (i - n_models / 2 + 0.5) * height
        
        ax.errorbar(
            coefs, y_pos + offset,
            xerr=1.96 * ses,
            fmt='o', capsize=2, capthick=1,
            label=model_name, color=colors[i], markersize=6
        )
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig


def _bias_plot_data(
    feature_names, coefficients, ci_lower, ci_upper,
    prior_means, title, xlabel
) -> Dict[str, Any]:
    """Return bias plot data for non-matplotlib rendering."""
    return {
        "type": "bias_contribution",
        "title": title,
        "xlabel": xlabel,
        "features": [
            {
                "name": name,
                "coefficient": float(coef),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "prior_mean": float(prior) if prior_means is not None else None
            }
            for name, coef, lower, upper, prior in zip(
                feature_names, coefficients, ci_lower, ci_upper,
                prior_means if prior_means is not None else [None] * len(coefficients)
            )
        ]
    }
