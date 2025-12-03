"""
Forest Plot Visualization for T3-Meta.

This module provides functions for creating publication-quality
forest plots, with extensions for T3-Meta bias visualization.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np


def forest_plot(
    estimates: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    study_names: List[str],
    weights: Optional[np.ndarray] = None,
    pooled_estimate: Optional[float] = None,
    pooled_ci: Optional[Tuple[float, float]] = None,
    effect_measure: str = "HR",
    exponentiate: bool = True,
    title: str = "Forest Plot",
    xlabel: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    show_weights: bool = True,
    sort_by: Optional[str] = None,
    color_by: Optional[np.ndarray] = None,
    color_label: str = "",
    null_value: float = 1.0,
    **kwargs
) -> Any:
    """
    Create a standard forest plot.
    
    Args:
        estimates: Effect estimates (on display scale)
        ci_lower: Lower CI bounds
        ci_upper: Upper CI bounds
        study_names: Names for each study
        weights: Study weights (for marker sizing)
        pooled_estimate: Pooled effect estimate
        pooled_ci: Pooled CI (lower, upper)
        effect_measure: Label for effect measure (HR, RR, OR, etc.)
        exponentiate: Whether estimates are on log scale (exp for display)
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size (width, height)
        show_weights: Show weight percentages
        sort_by: Sort studies by 'effect', 'weight', 'name', or None
        color_by: Array of values for coloring points
        color_label: Label for color scale
        null_value: Null effect value (1.0 for ratios, 0.0 for differences)
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object (or dict for non-matplotlib backends)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        # Return data structure for alternative rendering
        return _forest_plot_data(
            estimates, ci_lower, ci_upper, study_names, weights,
            pooled_estimate, pooled_ci, effect_measure, exponentiate,
            title, xlabel, null_value
        )
    
    estimates = np.asarray(estimates).flatten()
    ci_lower = np.asarray(ci_lower).flatten()
    ci_upper = np.asarray(ci_upper).flatten()
    n = len(estimates)
    
    # Transform if needed
    if exponentiate:
        display_est = np.exp(estimates)
        display_lower = np.exp(ci_lower)
        display_upper = np.exp(ci_upper)
        if pooled_estimate is not None:
            display_pooled = np.exp(pooled_estimate)
            display_pooled_ci = (np.exp(pooled_ci[0]), np.exp(pooled_ci[1])) if pooled_ci else None
    else:
        display_est = estimates
        display_lower = ci_lower
        display_upper = ci_upper
        display_pooled = pooled_estimate
        display_pooled_ci = pooled_ci
    
    # Sort if requested
    if sort_by == "effect":
        order = np.argsort(display_est)
    elif sort_by == "weight" and weights is not None:
        order = np.argsort(weights)
    elif sort_by == "name":
        order = np.argsort(study_names)
    else:
        order = np.arange(n)
    
    # Apply sorting
    display_est = display_est[order]
    display_lower = display_lower[order]
    display_upper = display_upper[order]
    study_names = [study_names[i] for i in order]
    if weights is not None:
        weights = weights[order]
    if color_by is not None:
        color_by = np.asarray(color_by)[order]
    
    # Normalize weights for marker sizing
    if weights is not None:
        marker_sizes = 50 + 200 * (weights / weights.max())
    else:
        marker_sizes = np.full(n, 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Y positions
    y_pos = np.arange(n)
    
    # Plot error bars
    for i in range(n):
        ax.plot(
            [display_lower[i], display_upper[i]], [y_pos[i], y_pos[i]],
            color='black', linewidth=1
        )
        ax.plot(
            [display_lower[i], display_lower[i]], [y_pos[i] - 0.1, y_pos[i] + 0.1],
            color='black', linewidth=1
        )
        ax.plot(
            [display_upper[i], display_upper[i]], [y_pos[i] - 0.1, y_pos[i] + 0.1],
            color='black', linewidth=1
        )
    
    # Plot points
    if color_by is not None:
        scatter = ax.scatter(
            display_est, y_pos, s=marker_sizes, c=color_by,
            cmap='RdYlBu_r', edgecolors='black', linewidth=0.5, zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label(color_label)
    else:
        ax.scatter(
            display_est, y_pos, s=marker_sizes, c='steelblue',
            edgecolors='black', linewidth=0.5, zorder=3
        )
    
    # Add pooled estimate
    if display_pooled is not None:
        y_pooled = -1.5
        
        # Diamond for pooled
        diamond_half_width = (display_pooled_ci[1] - display_pooled_ci[0]) / 2 if display_pooled_ci else 0.1
        diamond = mpatches.FancyBboxPatch(
            (display_pooled - diamond_half_width, y_pooled - 0.3),
            2 * diamond_half_width, 0.6,
            boxstyle="square,pad=0",
            facecolor='darkred', edgecolor='black', linewidth=1
        )
        ax.add_patch(diamond)
        
        # Pooled label
        ax.text(
            ax.get_xlim()[0] - 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            y_pooled, "Pooled",
            va='center', ha='right', fontweight='bold'
        )
    
    # Reference line at null value
    ax.axvline(x=null_value, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_names)
    ax.set_xlabel(xlabel or effect_measure)
    ax.set_title(title)
    
    # Add weight annotations
    if show_weights and weights is not None:
        weight_pct = 100 * weights / weights.sum()
        for i, (est, w) in enumerate(zip(display_est, weight_pct)):
            ax.text(
                ax.get_xlim()[1] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                y_pos[i],
                f'{w:.1f}%',
                va='center', ha='left', fontsize=8
            )
    
    # Extend y-axis for pooled
    if display_pooled is not None:
        ax.set_ylim(-2.5, n - 0.5)
    else:
        ax.set_ylim(-0.5, n - 0.5)
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig


def t3_forest_plot(
    estimates: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    study_names: List[str],
    study_bias: np.ndarray,
    weights: Optional[np.ndarray] = None,
    pooled_estimate: Optional[float] = None,
    pooled_ci: Optional[Tuple[float, float]] = None,
    is_rct: Optional[np.ndarray] = None,
    effect_measure: str = "HR",
    exponentiate: bool = True,
    title: str = "T3-Meta Forest Plot",
    show_bias_adjusted: bool = True,
    figsize: Tuple[float, float] = (12, 10),
    **kwargs
) -> Any:
    """
    Create T3-Meta enhanced forest plot with bias visualization.
    
    Shows original estimates, bias contributions, and adjusted estimates.
    
    Args:
        estimates: Effect estimates (on log scale for ratios)
        ci_lower: Lower CI bounds
        ci_upper: Upper CI bounds
        study_names: Study names
        study_bias: Estimated bias for each study
        weights: Study weights
        pooled_estimate: Pooled target trial effect
        pooled_ci: Pooled CI
        is_rct: Boolean array indicating RCT vs observational
        effect_measure: Effect measure label
        exponentiate: Whether to exponentiate for display
        title: Plot title
        show_bias_adjusted: Show bias-adjusted estimates
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        return _forest_plot_data(
            estimates, ci_lower, ci_upper, study_names, weights,
            pooled_estimate, pooled_ci, effect_measure, exponentiate,
            title, None, 1.0 if exponentiate else 0.0
        )
    
    estimates = np.asarray(estimates).flatten()
    ci_lower = np.asarray(ci_lower).flatten()
    ci_upper = np.asarray(ci_upper).flatten()
    study_bias = np.asarray(study_bias).flatten()
    n = len(estimates)
    
    # Compute bias-adjusted estimates
    adjusted = estimates - study_bias
    
    # Transform for display
    if exponentiate:
        display_est = np.exp(estimates)
        display_lower = np.exp(ci_lower)
        display_upper = np.exp(ci_upper)
        display_adj = np.exp(adjusted)
        null_val = 1.0
        if pooled_estimate is not None:
            display_pooled = np.exp(pooled_estimate)
            display_pooled_ci = (np.exp(pooled_ci[0]), np.exp(pooled_ci[1])) if pooled_ci else None
    else:
        display_est = estimates
        display_lower = ci_lower
        display_upper = ci_upper
        display_adj = adjusted
        null_val = 0.0
        display_pooled = pooled_estimate
        display_pooled_ci = pooled_ci
    
    # Marker sizes from weights
    if weights is not None:
        marker_sizes = 30 + 150 * (weights / weights.max())
    else:
        marker_sizes = np.full(n, 80)
    
    # Colors for RCT vs observational
    if is_rct is not None:
        colors = np.where(is_rct, 'darkblue', 'darkorange')
    else:
        colors = np.full(n, 'steelblue')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(n)
    
    # Plot CIs
    for i in range(n):
        ax.plot(
            [display_lower[i], display_upper[i]], [y_pos[i], y_pos[i]],
            color='gray', linewidth=1, alpha=0.5
        )
    
    # Plot original estimates
    ax.scatter(
        display_est, y_pos, s=marker_sizes, c=colors,
        edgecolors='black', linewidth=0.5, marker='s',
        label='Original', alpha=0.7, zorder=3
    )
    
    # Plot bias-adjusted estimates
    if show_bias_adjusted:
        ax.scatter(
            display_adj, y_pos, s=marker_sizes * 0.6, c='red',
            edgecolors='black', linewidth=0.5, marker='o',
            label='Bias-adjusted', zorder=4
        )
        
        # Draw arrows showing bias correction
        for i in range(n):
            if abs(display_est[i] - display_adj[i]) > 0.01:
                ax.annotate(
                    '', xy=(display_adj[i], y_pos[i]),
                    xytext=(display_est[i], y_pos[i]),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1)
                )
    
    # Pooled estimate
    if display_pooled is not None:
        y_pooled = -1.5
        
        if display_pooled_ci:
            diamond_width = display_pooled_ci[1] - display_pooled_ci[0]
        else:
            diamond_width = 0.2
        
        diamond_x = [
            display_pooled,
            display_pooled + diamond_width / 2,
            display_pooled,
            display_pooled - diamond_width / 2
        ]
        diamond_y = [y_pooled - 0.3, y_pooled, y_pooled + 0.3, y_pooled]
        ax.fill(diamond_x, diamond_y, color='darkred', edgecolor='black')
        
        ax.text(
            ax.get_xlim()[0] - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            y_pooled, "Target Trial\nEffect (θ*)",
            va='center', ha='right', fontweight='bold', fontsize=9
        )
    
    # Reference line
    ax.axvline(x=null_val, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_names)
    ax.set_xlabel(effect_measure)
    ax.set_title(title)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
               markersize=10, label='Original estimate'),
    ]
    if show_bias_adjusted:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Bias-adjusted')
        )
    if is_rct is not None:
        legend_elements.extend([
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue',
                   markersize=10, label='RCT'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange',
                   markersize=10, label='Observational'),
        ])
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Y-axis limits
    if display_pooled is not None:
        ax.set_ylim(-2.5, n - 0.5)
    else:
        ax.set_ylim(-0.5, n - 0.5)
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    return fig


def _forest_plot_data(
    estimates, ci_lower, ci_upper, study_names, weights,
    pooled_estimate, pooled_ci, effect_measure, exponentiate,
    title, xlabel, null_value
) -> Dict[str, Any]:
    """Return forest plot data for non-matplotlib rendering."""
    estimates = np.asarray(estimates).flatten()
    ci_lower = np.asarray(ci_lower).flatten()
    ci_upper = np.asarray(ci_upper).flatten()
    
    if exponentiate:
        display_est = np.exp(estimates)
        display_lower = np.exp(ci_lower)
        display_upper = np.exp(ci_upper)
        display_pooled = np.exp(pooled_estimate) if pooled_estimate else None
        display_pooled_ci = tuple(np.exp(x) for x in pooled_ci) if pooled_ci else None
    else:
        display_est = estimates
        display_lower = ci_lower
        display_upper = ci_upper
        display_pooled = pooled_estimate
        display_pooled_ci = pooled_ci
    
    return {
        "type": "forest_plot",
        "title": title,
        "xlabel": xlabel or effect_measure,
        "null_value": null_value,
        "studies": [
            {
                "name": name,
                "estimate": float(est),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "weight": float(w) if weights is not None else None
            }
            for name, est, lower, upper, w in zip(
                study_names, display_est, display_lower, display_upper,
                weights if weights is not None else [None] * len(estimates)
            )
        ],
        "pooled": {
            "estimate": float(display_pooled) if display_pooled else None,
            "ci_lower": float(display_pooled_ci[0]) if display_pooled_ci else None,
            "ci_upper": float(display_pooled_ci[1]) if display_pooled_ci else None,
        } if display_pooled else None
    }
