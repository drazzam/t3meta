# T3-Meta: Target Trial-Centric Meta-Analysis Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/yourusername/t3meta)

A Python framework for conducting meta-analyses that treats published studies as imperfect emulations of a single, clearly defined **target trial**. Instead of asking *"what is the average effect reported across studies?"*, T3-Meta asks: *"Given a well-specified target trial, what is our best estimate of its causal effect?"*

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [Mathematical Framework](#mathematical-framework)
- [Package Structure](#package-structure)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

---

## Overview

Traditional meta-analysis pools effect estimates across studies, treating heterogeneity as unexplained noise. **T3-Meta** takes a fundamentally different approach by:

1. **Defining a Target Trial**: Explicitly specifying the ideal randomized trial you wish you could conduct
2. **Mapping Study Designs**: Quantifying how each study's design deviates from the target
3. **Modeling Bias Structurally**: Using meta-regression to adjust for design-induced biases
4. **Estimating the Target Estimand**: Recovering what the target trial would have shown

This approach unifies evidence from RCTs, observational studies, and target trial emulations within a coherent causal inference framework.

### When to Use T3-Meta

- Combining RCTs with observational evidence
- Synthesizing studies with heterogeneous designs
- Adjusting for known sources of bias (immortal time, prevalent user, etc.)
- Transporting effects to a specific target population
- Resolving clinical equipoise with bias-adjusted estimates

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Target Trial Specification** | Define PICO elements, estimand type (ITT/PP/AT), and time horizon |
| **Design Mapping** | Automatically encode study design features as covariates |
| **Effect Measure Alignment** | Convert between HR, RR, OR, RD, SMD with proper variance propagation |
| **Time Alignment** | Harmonize studies with different follow-up durations |
| **Bias Priors** | Incorporate meta-epidemiological evidence on bias magnitudes |
| **Frequentist Models** | REML, DL, ML, Paule-Mandel, Knapp-Hartung adjustment |
| **Bayesian Models** | Laplace approximation, Gibbs sampling with informative priors |
| **Diagnostics** | I² decomposition, influence analysis, sensitivity analysis |
| **Visualization** | Forest plots, bias heatmaps, funnel plots |
| **I/O Support** | CSV, JSON, Excel, RevMan XML import/export |

---

## Installation

### From Source

```bash
git clone https://github.com/yourusername/t3meta.git
cd t3meta
pip install -e .
```

### Dependencies

```bash
pip install numpy scipy
```

Optional dependencies for full functionality:

```bash
pip install pandas matplotlib openpyxl
```

---

## Quick Start

```python
from t3meta import TargetTrial, Study, DesignFeatures, T3MetaAnalysis

# 1. Define your target trial
target = TargetTrial(
    name="Ideal GLP-1RA Cardiovascular Outcomes Trial",
    population="Adults with T2DM, no prior CV events",
    intervention="GLP-1RA initiation within 30 days of eligibility",
    comparator="No GLP-1RA (standard care)",
    outcome="MACE (MI, stroke, CV death)",
    time_horizon_months=36,
    effect_measure="HR",
    estimand_type="ITT"
)

# 2. Register studies with design features
studies = [
    Study(
        name="LEADER",
        effect_estimate=0.87,
        effect_measure="HR",
        ci_lower=0.78, ci_upper=0.97,
        design_features=DesignFeatures(is_rct=True, sample_size=9340)
    ),
    Study(
        name="Real-World Cohort",
        effect_estimate=0.82,
        effect_measure="HR",
        ci_lower=0.75, ci_upper=0.90,
        design_features=DesignFeatures(
            is_rct=False, 
            sample_size=45000,
            has_immortal_time_bias=True
        )
    ),
]

# 3. Fit the model
meta = T3MetaAnalysis(
    target_trial=target,
    feature_names=["is_rct", "has_immortal_time_bias"]
)
for s in studies:
    meta.add_study(s)

results = meta.fit()

# 4. Get results
print(f"Target Trial Effect: HR = {results.get_theta_star_exp()[0]:.3f}")
print(f"95% CI: {results.get_theta_star_exp()[1]}")
print(f"Residual I²: {results.i_squared * 100:.1f}%")
```

---

## Core Concepts

### Target Trial

The **target trial** is the hypothetical ideal RCT you wish to conduct. It defines:

- **Population**: Who would be enrolled
- **Intervention/Comparator**: Treatment strategies being compared
- **Outcome**: Primary endpoint and its measurement
- **Time Horizon**: Duration of follow-up
- **Estimand**: ITT, per-protocol, or as-treated effect

```python
target = TargetTrial(
    name="Ideal Trial",
    population="Adults aged 40-75 with condition X",
    intervention="Drug A 100mg daily",
    comparator="Placebo",
    outcome="Time to first event",
    time_horizon_months=24,
    effect_measure="HR",
    estimand_type="ITT"
)
```

### Design Features

Each study is characterized by **design features** that may introduce bias relative to the target trial:

```python
features = DesignFeatures(
    is_rct=False,                    # Observational study
    outcome_adjudicated=True,        # Blinded endpoint adjudication
    has_immortal_time_bias=True,     # Immortal time in exposure definition
    has_prevalent_user_bias=False,   # New users only
    loss_to_followup_pct=12.5,       # 12.5% LTFU
    sample_size=15000,
    median_followup_months=28
)
```

### The T3-Meta Model

The framework models each study's estimate as:

```
θ̂_j = θ* + X_j'β + u_j + ε_j
```

Where:
- `θ*` = Target trial effect (the estimand of interest)
- `X_j` = Design feature vector for study j
- `β` = Bias coefficients (estimated from data + priors)
- `u_j ~ N(0, τ²)` = Residual heterogeneity
- `ε_j ~ N(0, s_j²)` = Sampling error

---

## Usage Examples

### Frequentist Analysis with REML

```python
from t3meta import T3MetaAnalysis

meta = T3MetaAnalysis(
    target_trial=target,
    feature_names=["is_rct", "has_immortal_time_bias", "loss_to_followup_pct"]
)
meta.add_studies(studies)

# Fit with REML and Knapp-Hartung adjustment
results = meta.fit_frequentist(
    method="REML",
    knapp_hartung=True,
    ci_level=0.95
)

print(results.summary_table())
```

### Bayesian Analysis with Informative Priors

```python
from t3meta import T3MetaAnalysis, get_default_bias_priors

# Get meta-epidemiological priors
priors = get_default_bias_priors()

meta = T3MetaAnalysis(
    target_trial=target,
    feature_names=["is_rct", "has_immortal_time_bias"],
    priors=priors
)
meta.add_studies(studies)

# Fit with Gibbs sampling
results = meta.fit_bayesian(
    method="gibbs",
    n_samples=5000,
    n_warmup=1000
)

# Posterior probability of benefit
p_benefit = results.posterior_probability("theta_star", threshold=0)
print(f"P(HR < 1) = {p_benefit:.3f}")
```

### Leave-One-Out Sensitivity Analysis

```python
from t3meta.diagnostics import InfluenceAnalysis

influence = InfluenceAnalysis(
    estimates=estimates,
    se=standard_errors,
    study_names=names
)

# Identify influential studies
loo_results = meta.leave_one_out()
for study, result in loo_results.items():
    change = result.theta_star - results.theta_star
    print(f"Excluding {study}: Δθ* = {change:+.4f}")
```

### Heterogeneity Decomposition

```python
from t3meta.diagnostics import HeterogeneityAnalysis, decompose_heterogeneity

# Decompose I² into explained and residual
decomp = decompose_heterogeneity(
    estimates=estimates,
    se=standard_errors,
    design_matrix=X,
    coefficients=results.beta
)

print(f"Total I²: {decomp['i_squared_total']*100:.1f}%")
print(f"Explained I²: {decomp['i_squared_explained']*100:.1f}%")
print(f"Residual I²: {decomp['i_squared_residual']*100:.1f}%")
print(f"R² (variance explained): {decomp['r_squared']*100:.1f}%")
```

### Publication Bias Assessment

```python
from t3meta.diagnostics import SensitivityAnalysis

sens = SensitivityAnalysis(estimates, se)

# Egger's test
egger = sens.egger_test()
print(f"Egger's intercept: {egger['intercept']:.3f} (p = {egger['p_value']:.3f})")

# Trim-and-fill
tf = sens.trim_and_fill()
print(f"Imputed studies: {tf['n_imputed']}")
print(f"Adjusted estimate: {tf['adjusted_estimate']:.4f}")
```

### Visualization

```python
from t3meta.visualization import t3_forest_plot, funnel_plot

# T3-Meta forest plot with bias adjustment
fig = t3_forest_plot(
    estimates=results.study_effects,
    ci_lower=ci_lower,
    ci_upper=ci_upper,
    study_names=names,
    study_bias=results.study_bias,
    is_rct=is_rct_array,
    pooled_estimate=results.theta_star,
    pooled_ci=results.theta_star_ci,
    title="T3-Meta Forest Plot: GLP-1RA and MACE"
)
fig.savefig("forest_plot.png", dpi=300)

# Contour-enhanced funnel plot
fig = funnel_plot(
    estimates=estimates,
    se=se,
    pooled_estimate=results.theta_star,
    title="Funnel Plot"
)
```

---

## Mathematical Framework

### The T3-Meta Model

**Observation Model:**
```
θ̂_j | θ_j, s_j² ~ N(θ_j, s_j²)
```

**Structural Model:**
```
θ_j = θ* + X_j'β + u_j,  u_j ~ N(0, τ²)
```

**Target Trial Effect:**
```
θ* = E[θ_j | X_j = 0]
```

### Estimation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **REML** | Restricted maximum likelihood | Default; unbiased τ² estimation |
| **DL** | DerSimonian-Laird | Quick moment estimator |
| **ML** | Maximum likelihood | Large sample settings |
| **PM** | Paule-Mandel | Iterative, robust to outliers |
| **Laplace** | Posterior mode + Hessian | Fast Bayesian approximation |
| **Gibbs** | MCMC with conjugate updates | Full posterior inference |

### Heterogeneity Statistics

- **Cochran's Q**: `Q = Σ w_j(θ̂_j - θ̄)²`
- **I²**: `I² = max(0, (Q - (k-1))/Q)`
- **τ² (DL)**: `τ² = max(0, (Q - (k-1))/c)` where `c = Σw - Σw²/Σw`
- **Prediction Interval**: `θ* ± t_{k-2} × √(SE² + τ²)`

### Effect Measure Conversions

- **OR → RR** (Zhang-Yu): `RR = OR / (1 - p₀ + p₀ × OR)`
- **SMD → OR** (Chinn): `OR = exp(π × d / √3)`
- **HR → RD**: Via cumulative incidence assuming exponential survival

---

## Package Structure

```
t3meta/
├── __init__.py              # Main exports
├── core/                    # Core data structures
│   ├── target_trial.py      # TargetTrial, TargetTrialSpec
│   ├── study.py             # Study, DesignFeatures, DesignMap
│   ├── estimand.py          # Estimand, EffectMeasure, EstimandType
│   └── registry.py          # StudyRegistry
├── alignment/               # Effect alignment utilities
│   ├── effect_measures.py   # EffectMeasureConverter
│   ├── time_alignment.py    # TimeAligner
│   └── standardization.py   # BaselineRiskStandardizer
├── models/                  # Statistical models
│   ├── base.py              # BaseModel, ModelResults
│   ├── frequentist.py       # FrequentistModel, MixedEffectsModel
│   ├── bayesian.py          # BayesianModel, BayesianResults
│   ├── priors.py            # Prior specifications
│   └── t3_meta_analysis.py  # T3MetaAnalysis (main interface)
├── diagnostics/             # Model diagnostics
│   ├── heterogeneity.py     # HeterogeneityAnalysis
│   ├── influence.py         # InfluenceAnalysis
│   └── sensitivity.py       # SensitivityAnalysis
├── visualization/           # Plotting functions
│   ├── forest.py            # forest_plot, t3_forest_plot
│   ├── bias.py              # bias_contribution_plot, bias_heatmap
│   └── funnel.py            # funnel_plot, contour_enhanced_funnel
├── io/                      # Input/output utilities
│   ├── readers.py           # read_csv, read_json, read_excel
│   ├── writers.py           # write_csv, write_json, export_to_prisma
│   └── schema.py            # T3MetaSchema, validate_input
└── utils/                   # Statistical utilities
    └── __init__.py          # Helper functions
```

---

## API Reference

### Main Classes

| Class | Description |
|-------|-------------|
| `TargetTrial` | Specification of the ideal target trial |
| `Study` | Individual study with effect estimate and design features |
| `DesignFeatures` | Design characteristics that may introduce bias |
| `T3MetaAnalysis` | Main analysis class orchestrating the workflow |
| `ModelResults` | Container for fitted model results |

### Key Methods

```python
# T3MetaAnalysis
meta.add_study(study)              # Register a study
meta.add_studies(study_list)       # Register multiple studies
meta.fit()                         # Fit model (auto-selects frequentist)
meta.fit_frequentist(method="REML") # Frequentist fitting
meta.fit_bayesian(method="gibbs")  # Bayesian fitting
meta.leave_one_out()               # LOO sensitivity analysis
meta.get_study_bias()              # Extract study-level bias estimates
meta.predict_target_effect()       # Predict for ideal design

# ModelResults
results.theta_star                 # Target trial effect (log scale)
results.get_theta_star_exp()       # Exponentiated effect with CI
results.beta                       # Bias coefficients
results.tau_squared                # Residual heterogeneity
results.i_squared                  # I² statistic
results.summary_table()            # Formatted results table
```

---

## Dependencies

### Required
- `numpy >= 1.20`
- `scipy >= 1.7`

### Optional
- `pandas >= 1.3` (Excel I/O, DataFrame support)
- `matplotlib >= 3.4` (Visualization)
- `openpyxl >= 3.0` (Excel file reading)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/t3meta.git
cd t3meta
pip install -e ".[dev]"
pytest tests/
```

---

## Citation

If you use T3-Meta in your research, please cite:

```bibtex
@software{t3meta2025,
  author = {Azzam, Ahmed Y.},
  title = {T3-Meta: Target Trial-Centric Meta-Analysis Framework},
  version = {1.0.0},
  year = {2025},
  url = {https://github.com/yourusername/t3meta}
}
```

### Related Publications

The methodological foundations of T3-Meta draw from:

- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. *Am J Epidemiol*. 2016.
- Turner RM, et al. Predicting the extent of heterogeneity in meta-analysis. *Stat Med*. 2012.
- Rhodes KM, et al. Predictive distributions were developed for the extent of heterogeneity in meta-analyses of continuous outcome data. *J Clin Epidemiol*. 2015.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Ahmed Y. Azzam, MD, MEng, DSc(h.c.), FRCP**

Research Fellow, Department of Neuroradiology  
WVU Medicine

- Email: ahmed.azzam@wvumedicine.org
- GitHub: [@ahmedyazzam](https://github.com/ahmedyazzam)

---

## Acknowledgments

- The target trial emulation framework was inspired by the work of Hernán and Robins
- Meta-epidemiological prior specifications adapted from Turner et al. and Rhodes et al.
- Statistical methods follow Cochrane Handbook recommendations where applicable

---

<p align="center">
  <i>Unifying evidence through the lens of a target trial.</i>
</p>
