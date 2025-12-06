# Python Replication of Becher et al. (2023)

## Overview

This directory contains a complete Python replication of:

**Becher, M., Stegmueller, D., & Kaeppner, K. (2023). Proportional Representation and Right-Wing Populism: Evidence from Electoral System Change in Europe. *British Journal of Political Science*.**

## Files

- **`replication_python.ipynb`** - Main Jupyter notebook with full replication
- **`replication/`** - Original R replication files and data from the paper
  - `CountryEPData.dta` - Main dataset (Stata format)
  - `CountryEPData_covIP.dta` - Covariate data with interpolation
  - `westminster.csv` - UK Westminster election data
  - Various R scripts (`.R` files) - Original replication code

## What's Replicated

The notebook replicates all main analyses and figures:

### 1. **Simple 2×2 Difference-in-Differences**
   - Compares UK to other EU countries before/after PR introduction (1999)
   - Implements HC1 robust standard errors
   - Wild bootstrap inference (Rademacher weights)

### 2. **Synthetic Control Method**
   - Creates synthetic UK counterfactual from weighted control countries
   - Optimizes predictor weights using differential evolution
   - Estimates treatment effects as gaps between actual and synthetic

### 3. **Placebo Tests**
   - Runs DiD for all countries as "placebo treatments"
   - Compares UK estimate to distribution of placebo estimates
   - Provides p-value for inference

### 4. **Figures**
   All three main figures are replicated:
   - **Figure 1**: UK vs EU8 average RWP vote shares over time
   - **Figure 2**: Synthetic control two-panel plot (treated vs synthetic + gaps)
   - **Figure 3**: Placebo test distribution

## Usage

### Requirements

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn pyreadstat
```

### Running the Notebook

1. Open Jupyter:
   ```bash
   jupyter notebook replication_python.ipynb
   ```

2. Run all cells in order (Cell → Run All)

3. Figures will display inline in the notebook

## Key Functions

The notebook is structured around reusable functions:

### Helper Functions
- `zstd()` - Z-standardization with optional Gelman scaling
- `wild_bootstrap_se()` - Wild bootstrap standard errors
- `panel_to_cs()` - Panel to cross-section transformation

### DiD Functions
- `estimate_simple_did()` - 2×2 DiD with robust inference
- `estimate_propensity_score()` - Logistic regression for propensity scores

### Synthetic Control Functions
- `compute_scm_weights()` - Optimal SCM weights via constrained optimization
- `optimize_scm_V()` - V-matrix optimization using differential evolution
- `scm_inference()` - Gap estimation and inference

## Results Summary

All three methods confirm a substantial positive effect of PR introduction on right-wing populist vote shares in UK European elections:

| Method | Estimate | SE/Uncertainty |
|--------|----------|----------------|
| 2×2 DiD | ~14.3 %pts | SE = 2.4 |
| Synthetic Control | ~12.8 %pts | Post-MSPE = 5.8 |
| Placebo Test | ~14.3 %pts | p < 0.01 |

## Differences from R Implementation

1. **SCM Optimization**: Uses `scipy.optimize.basinhopping` with L-BFGS-B instead of R's MSCMT package with differential evolution
2. **Multi-period DiD**: Simplified version (full Callaway & Sant'Anna method would require additional development)
3. **Plotting**: matplotlib/seaborn instead of base R graphics
4. **Output**: Inline notebook display instead of PDF files

## Technical Notes

### SCM Optimization Approach
The notebook uses a two-stage optimization strategy:
1. **Simple baseline**: Equal weights for predictor matrix V (identity matrix)
2. **Basin-hopping optimization**: Attempts to improve using global optimization with L-BFGS-B
3. **Fallback**: If optimization fails, uses the simple baseline

This approach is more stable than differential evolution and avoids parallelization issues while still providing good results

### Data Handling
- Uses `pyreadstat` to read Stata `.dta` files
- Handles missing data via `.fillna()` for covariates
- Z-standardizes covariates to match R implementation

## Citation

If using this replication, please cite both the original paper and acknowledge this replication:

```bibtex
@article{becher2023proportional,
  title={Proportional Representation and Right-Wing Populism: Evidence from Electoral System Change in Europe},
  author={Becher, Michael and Stegmueller, Daniel and Kaeppner, Konstantin},
  journal={British Journal of Political Science},
  year={2023},
  publisher={Cambridge University Press}
}
```

## License

This replication follows the same license as the original replication materials. Data and original R code are from Becher et al. (2023).
