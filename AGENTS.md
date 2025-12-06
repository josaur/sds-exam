# AGENTS.md - Pepinsky et al. (2024) Replication Study

## Purpose
This file provides structured guidance for AI agents to efficiently generate Python code for data analysis tasks related to the Pepinsky, Goodman & Ziller (2024) replication study: "Modeling Spatial Heterogeneity and Historical Persistence: Nazi Concentration Camps and Contemporary Intolerance" (APSR).

**Repository:** `/home/ich/VSC/sds-exam/`
**Main notebook:** `exam/Exam_Notebook.ipynb`
**Data directory:** `exam/data/replication_archive/`

### Repository Structure
```
sds-exam/
├── README.md
├── AGENTS.md (this file)
├── exam/
│   ├── Exam_Notebook.ipynb (main analysis notebook)
│   ├── data/
│   │   └── replication_archive/
│   │       ├── tables/
│   │       │   ├── EVS_main.csv
│   │       │   ├── evs_weimar.csv
│   │       │   └── elections_2017.csv
│   │       ├── figure3/
│   │       │   └── camp_coords.csv
│   │       └── dags/
│   ├── study/
│   │   ├── Pepinsky et al. - 2024 - ...pdf (main paper)
│   │   └── Appendix.pdf
│   └── [generated figures: dag_*.pdf, *.pdf]
└── .venv/ (Python virtual environment)
```

### Exam_Notebook.ipynb Structure
The main notebook contains the following sections:
1. **Load and Prepare Data** - Import datasets and create derived variables
2. **Descriptive Statistics** - Summary statistics and distributions
3. **DAG Visualizations** (Figure 1) - Causal diagrams for post-treatment bias
4. **Geographic Visualization** (Figure 3) - Map of concentration camps
5. **Table 1, Panel A** - Main regression results (Intolerance outcome)
6. **Table 1, Panel B** - Immigrant Resentment outcome
7. **Table 1, Panel C** - Far-right party support outcome
8. **Table A2** - Hausman tests (FE vs RE)
9. **Table 2** - 2017 Election results (AfD vote share)
10. **Visualization of Results** - Coefficient plots and comparisons
11. **Summary and Interpretation** - Key findings
12. **Conclusion** - Overall assessment

---

## 1. Study Overview

**Full Citation:**
Pepinsky, T. B., Goodman, S. W., & Ziller, C. (2024). Modeling Spatial Heterogeneity and Historical Persistence: Nazi Concentration Camps and Contemporary Intolerance. *American Political Science Review*, 118(1), 519–528. https://doi.org/10.1017/S0003055423000072

**Data Repository:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0PY9EA

### 1.1 Research Question
**Original claim (HPT 2020):** Germans living closer to former Nazi concentration camps display higher intolerance toward outgroups today.

**Pepinsky et al. critique:** This effect disappears when controlling for German state (Länder) fixed effects, suggesting the original finding was driven by unobserved spatial heterogeneity.

### 1.2 Key Theoretical Framework
```
Temporal vs. Causal Ordering:
t₁: U (unobserved factors)
t₂: T (Distance to camps) ← Treatment
t₃: F (Länder boundaries) ← Fixed Effects  
t₄: Y (Intolerance) ← Outcome

Core Debate: Are Länder-FE "post-treatment" variables?
- Pepinsky: NO → F is NOT caused by T → Control for F is valid
- Alternative: YES → F is caused by T → Control creates bias
```

### 1.3 Main Finding
With state fixed effects, the Distance coefficient:
- Becomes statistically insignificant (EVS data)
- Reverses sign to positive (Election data)

---

## 2. Data Structure

### 2.1 Datasets

| Dataset | File | Path | N | Unit | Source |
|---------|------|------|---|------|--------|
| EVS Survey | `EVS_main.csv` | `data/replication_archive/tables/EVS_main.csv` | ~2,075 | Individual | European Values Study |
| EVS Weimar | `evs_weimar.csv` | `data/replication_archive/tables/evs_weimar.csv` | ~2,052 | Individual | EVS + Weimar boundaries |
| Elections 2017 | `elections_2017.csv` | `data/replication_archive/tables/elections_2017.csv` | ~10,870 | Electoral district | Bundestagswahl |
| Camp Coordinates | `camp_coords.csv` | `data/replication_archive/figure3/camp_coords.csv` | ~70 | Concentration camp | Historical records |

**Note:** All paths are relative to the `exam/` directory.

### 2.2 Variable Dictionary

#### Outcome Variables (Y)
| Variable | Description | Scale | Dataset |
|----------|-------------|-------|---------|
| `intolerance` | Outgroup intolerance index | Continuous | EVS |
| `resentment` | Immigrant resentment | Continuous | EVS |
| `far_right` | Far-right party support | Binary/Continuous | EVS |
| `AfDshare` | AfD vote share 2017 | Percentage | Elections |
| `AfDNPDshare` | AfD + NPD vote share | Percentage | Elections |

#### Treatment Variable (T)
| Variable | Description | Dataset | Notes |
|----------|-------------|---------|-------|
| `Distance` | Distance to nearest concentration camp (km) | EVS | Primary treatment variable |
| `distance2` | Distance to nearest camp (km) | Elections | Same construct, different naming |
| `distance` | Raw distance in evs_weimar | EVS Weimar | Gets renamed to `Distance` |

#### Pretreatment Controls (X) - measured before Nazi era
| Variable | Description | Year | Dataset |
|----------|-------------|------|---------|
| `prop_jewish25` / `prop_juden` | % Jewish population | 1925 | Both |
| `unemployment33` / `unemp33` | Unemployment rate | 1933 | Both |
| `population25` / `pop25` | Population size | 1925 | Both |
| `nazishare33` / `vshare33` | NSDAP vote share | 1933 | Both |

#### Posttreatment Controls - measured after 1945
| Variable | Description | Dataset |
|----------|-------------|---------|
| `west` | West Germany dummy | EVS |
| `urban_scale` | Urbanization level | EVS |
| `immigrants07` | Immigrant share 2007 | EVS |
| `unemployment07` | Unemployment 2007 | EVS |
| `educ` | Education level | EVS |
| `age` | Age | EVS |
| `female` | Female dummy | EVS |
| `lr` | Left-right self-placement | EVS |
| `unemp` | Currently unemployed | EVS |

#### Fixed Effects (F)
| Variable | Description | Dataset |
|----------|-------------|---------|
| `state` / `f.state` | 16 German Länder | EVS |
| `NAME_1` | State name | Elections |
| `weimarprov` | Weimar-era boundaries | EVS Weimar |

### 2.3 State Codes (EVS)
```python
STATE_MAPPING = {
    'DE1': 'Baden-Württemberg',  # West
    'DE2': 'Bavaria',            # West
    'DE3': 'Berlin',             # East
    'DE4': 'Brandenburg',        # East
    'DE5': 'Bremen',             # West
    'DE6': 'Hamburg',            # West
    'DE7': 'Hessen',             # West
    'DE8': 'Mecklenburg-Vorpommern',  # East
    'DE9': 'Lower Saxony',       # West
    'DEA': 'North Rhine-Westphalia',  # West
    'DEB': 'Rhineland-Palatinate',    # West
    'DEC': 'Saarland',           # West
    'DED': 'Saxony',             # East
    'DEE': 'Saxony-Anhalt',      # East
    'DEF': 'Schleswig-Holstein', # West
    'DEG': 'Thuringia'           # East
}
```

---

## 3. Python Code Templates

### 3.1 Data Loading and Preparation

**Note:** The actual `Exam_Notebook.ipynb` uses a `ModelSuite` class to organize and store all fitted regression models. This provides cleaner code organization and easier model comparison. See the notebook for the full implementation.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility (matches the notebook)
np.random.seed(12435)

# Load data (paths relative to exam/ directory)
evs = pd.read_csv('data/replication_archive/tables/EVS_main.csv')
evs_weimar = pd.read_csv('data/replication_archive/tables/evs_weimar.csv')
elections = pd.read_csv('data/replication_archive/tables/elections_2017.csv')

# Create state factor
evs['f_state'] = evs['state'].astype('category')

# Prepare Weimar-era data
evs_weimar['Distance'] = evs_weimar['distance']
evs_weimar.loc[evs_weimar['oldland_pruprov'] == 2000, 'oldland_pruprov'] = 2001
evs_weimar['weimarprov'] = evs_weimar['oldland']
mask = evs_weimar['weimarprov'] == 1000
evs_weimar.loc[mask, 'weimarprov'] = evs_weimar.loc[mask, 'oldland_pruprov']

# For reweighting analysis: exclude Berlin and Hamburg (no internal variation)
elections_bfe = elections[~elections['NAME_1'].isin(['Berlin', 'Hamburg'])].copy()
```

### 3.2 OLS Models with Fixed Effects

```python
def run_ols_models(data, outcome_var, treatment_var='Distance', 
                   pretreatment_controls=None, fe_var=None):
    """
    Run OLS models with and without fixed effects.
    
    Parameters:
    -----------
    data : DataFrame
    outcome_var : str - dependent variable name
    treatment_var : str - treatment variable name
    pretreatment_controls : list - pretreatment control variables
    fe_var : str - fixed effects variable name
    
    Returns:
    --------
    dict with model results
    """
    results = {}
    
    if pretreatment_controls is None:
        pretreatment_controls = ['prop_jewish25', 'unemployment33', 
                                  'population25', 'nazishare33']
    
    # Bivariate model
    formula_biv = f'{outcome_var} ~ {treatment_var}'
    results['bivariate'] = smf.ols(formula_biv, data=data).fit()
    
    # Bivariate + FE
    if fe_var:
        formula_biv_fe = f'{outcome_var} ~ {treatment_var} + C({fe_var})'
        results['bivariate_fe'] = smf.ols(formula_biv_fe, data=data).fit()
    
    # With pretreatment controls
    controls_str = ' + '.join(pretreatment_controls)
    formula_pre = f'{outcome_var} ~ {treatment_var} + {controls_str}'
    results['pretreatment'] = smf.ols(formula_pre, data=data).fit()
    
    # With pretreatment controls + FE
    if fe_var:
        formula_pre_fe = f'{outcome_var} ~ {treatment_var} + {controls_str} + C({fe_var})'
        results['pretreatment_fe'] = smf.ols(formula_pre_fe, data=data).fit()
    
    return results

# Example usage
results = run_ols_models(evs, 'intolerance', fe_var='state')
```

### 3.3 G-Estimation (Sequential g-estimator)

```python
def g_estimation(data, outcome_var, treatment_var, 
                 pretreatment_controls, posttreatment_controls,
                 n_bootstrap=1000, seed=543):
    """
    Implement G-estimation following Acharya, Blackwell & Sen (2016).
    
    Steps:
    1. Estimate full model with all controls
    2. Compute adjusted outcome: Y_tilde = Y - gamma_post * X_post
    3. Regress Y_tilde on treatment and pretreatment controls
    4. Bootstrap for standard errors
    """
    np.random.seed(seed)
    
    # Stage 1: Full model with post-treatment controls
    all_controls = pretreatment_controls + posttreatment_controls
    formula_full = f'{outcome_var} ~ {treatment_var} + ' + ' + '.join(all_controls)
    full_model = smf.ols(formula_full, data=data).fit()
    
    # Stage 2: Adjust outcome
    # Remove effect of post-treatment variables
    adjustment = np.zeros(len(data))
    for var in posttreatment_controls:
        if var in full_model.params.index:
            adjustment += full_model.params[var] * data[var].values
    
    data = data.copy()
    data['y_adjusted'] = data[outcome_var] - adjustment
    
    # Stage 3: Regress adjusted outcome on treatment + pretreatment
    formula_adjusted = f'y_adjusted ~ {treatment_var} + ' + ' + '.join(pretreatment_controls)
    g_est_model = smf.ols(formula_adjusted, data=data).fit()
    
    # Bootstrap for standard errors
    n = len(data)
    boot_coefs = np.zeros((n_bootstrap, len(g_est_model.params)))
    
    for b in range(n_bootstrap):
        # Resample
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data.iloc[idx].copy()
        
        # Stage 1
        boot_full = smf.ols(formula_full, data=boot_data).fit()
        
        # Stage 2
        boot_adjustment = np.zeros(len(boot_data))
        for var in posttreatment_controls:
            if var in boot_full.params.index:
                boot_adjustment += boot_full.params[var] * boot_data[var].values
        boot_data['y_adjusted'] = boot_data[outcome_var] - boot_adjustment
        
        # Stage 3
        boot_g_est = smf.ols(formula_adjusted, data=boot_data).fit()
        boot_coefs[b, :] = boot_g_est.params.values
    
    # Bootstrap standard errors
    boot_se = boot_coefs.std(axis=0)
    
    return {
        'model': g_est_model,
        'coefficients': g_est_model.params,
        'boot_se': pd.Series(boot_se, index=g_est_model.params.index),
        'boot_coefs': boot_coefs
    }

# Example usage
pretreatment = ['prop_jewish25', 'unemployment33', 'population25', 'nazishare33']
posttreatment = ['lr', 'immigrants07', 'unemployment07', 'unemp', 'educ', 'urban_scale']

g_results = g_estimation(evs, 'intolerance', 'Distance', 
                         pretreatment, posttreatment, n_bootstrap=1000)
```

### 3.4 Hausman Test (FE vs. Pooled/RE)

```python
from linearmodels.panel import compare

def hausman_test(data, formula, entity_var, time_var=None):
    """
    Perform Hausman test comparing Fixed Effects vs Random Effects/Pooled.
    
    For cross-sectional data with groups, use entity_var as group identifier.
    """
    # Set panel index
    if time_var:
        data = data.set_index([entity_var, time_var])
    else:
        # Create dummy time variable for cross-sectional data
        data = data.copy()
        data['_time'] = 1
        data = data.set_index([entity_var, '_time'])
    
    # Fit models
    fe_model = PanelOLS.from_formula(formula + ' + EntityEffects', data=data).fit()
    re_model = RandomEffects.from_formula(formula, data=data).fit()
    pooled_model = PooledOLS.from_formula(formula, data=data).fit()
    
    # Hausman test: compare FE and RE
    # H0: RE is consistent (both FE and RE are consistent)
    # H1: RE is inconsistent (only FE is consistent)
    
    b_fe = fe_model.params
    b_re = re_model.params
    
    # Common parameters only
    common_params = b_fe.index.intersection(b_re.index)
    common_params = [p for p in common_params if p != 'Intercept']
    
    diff = b_fe[common_params] - b_re[common_params]
    
    # Variance of difference
    var_fe = fe_model.cov[common_params].loc[common_params]
    var_re = re_model.cov[common_params].loc[common_params]
    var_diff = var_fe - var_re
    
    # Hausman statistic
    try:
        H = float(diff.T @ np.linalg.inv(var_diff) @ diff)
        df = len(common_params)
        p_value = 1 - stats.chi2.cdf(H, df)
    except:
        H, p_value = np.nan, np.nan
    
    return {
        'fe_model': fe_model,
        're_model': re_model,
        'pooled_model': pooled_model,
        'hausman_stat': H,
        'p_value': p_value,
        'df': len(common_params)
    }
```

### 3.5 Machine Learning Methods

#### 3.5.1 Random Forest for Feature Importance

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
import shap

def ml_feature_importance(data, outcome_var, feature_vars, 
                          method='random_forest', standardize=True):
    """
    Analyze feature importance using ML methods.
    
    Parameters:
    -----------
    method : str - 'random_forest', 'gradient_boosting', 'lasso', 'ridge'
    """
    # Prepare data
    df = data[feature_vars + [outcome_var]].dropna()
    X = df[feature_vars]
    y = df[outcome_var]
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_vars)
    else:
        X_scaled = X
    
    if method == 'random_forest':
        model = RandomForestRegressor(n_estimators=500, random_state=42, 
                                       min_samples_leaf=5, n_jobs=-1)
        model.fit(X_scaled, y)
        
        importance = pd.DataFrame({
            'feature': feature_vars,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    elif method == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=500, random_state=42,
                                           max_depth=3, learning_rate=0.1)
        model.fit(X_scaled, y)
        
        importance = pd.DataFrame({
            'feature': feature_vars,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    elif method in ['lasso', 'ridge']:
        from sklearn.linear_model import LassoCV, RidgeCV
        
        if method == 'lasso':
            model = LassoCV(cv=5, random_state=42)
        else:
            model = RidgeCV(cv=5)
        
        model.fit(X_scaled, y)
        
        importance = pd.DataFrame({
            'feature': feature_vars,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    return {
        'model': model,
        'importance': importance,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }

# Example usage
features = ['Distance', 'prop_jewish25', 'unemployment33', 
            'population25', 'nazishare33', 'west', 'urban_scale']
ml_results = ml_feature_importance(evs, 'intolerance', features, method='random_forest')
```

#### 3.5.2 Causal Forest

```python
# Note: Requires econml package
# pip install econml

from econml.dml import CausalForestDML
from econml.grf import CausalForest

def causal_forest_analysis(data, outcome_var, treatment_var, control_vars,
                           n_estimators=2000):
    """
    Estimate heterogeneous treatment effects using Causal Forest.
    
    Returns CATE (Conditional Average Treatment Effect) estimates.
    """
    df = data[[outcome_var, treatment_var] + control_vars].dropna()
    
    Y = df[outcome_var].values
    T = df[treatment_var].values
    X = df[control_vars].values
    
    # Binarize treatment for causal forest (above/below median)
    T_binary = (T < np.median(T)).astype(int)  # 1 = close to camp
    
    # Fit Causal Forest
    cf = CausalForestDML(
        model_y='auto',
        model_t='auto',
        n_estimators=n_estimators,
        random_state=42,
        cv=5
    )
    cf.fit(Y, T_binary, X=X)
    
    # Get treatment effects
    tau_hat = cf.effect(X)
    
    # Confidence intervals
    tau_lower, tau_upper = cf.effect_interval(X, alpha=0.05)
    
    # Average treatment effect
    ate = cf.ate(X)
    ate_se = cf.ate_stderr(X)
    
    return {
        'model': cf,
        'tau_hat': tau_hat,
        'tau_lower': tau_lower,
        'tau_upper': tau_upper,
        'ate': ate,
        'ate_se': ate_se,
        'X': X,
        'feature_names': control_vars
    }
```

#### 3.5.3 Double Machine Learning

```python
from econml.dml import LinearDML, SparseLinearDML
from sklearn.ensemble import RandomForestRegressor

def double_ml_estimation(data, outcome_var, treatment_var, control_vars,
                         ml_method='random_forest', n_folds=5):
    """
    Implement Double/Debiased Machine Learning.
    
    Model:
    Y = θ·T + f(X) + ε
    T = g(X) + u
    
    Uses cross-fitting and orthogonalization for valid inference on θ.
    """
    df = data[[outcome_var, treatment_var] + control_vars].dropna()
    
    Y = df[outcome_var].values
    T = df[treatment_var].values.reshape(-1, 1)
    X = df[control_vars].values
    
    # Define ML models for nuisance functions
    if ml_method == 'random_forest':
        model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, 
                                         random_state=42, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                         random_state=42, n_jobs=-1)
    
    # Fit DML
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        random_state=42
    )
    dml.fit(Y, T, X=X)
    
    # Get coefficient and inference
    theta = dml.coef_[0]
    theta_se = dml.coef__stderr()[0]
    theta_ci = dml.coef__interval(alpha=0.05)
    
    return {
        'model': dml,
        'theta': theta,
        'se': theta_se,
        'ci_lower': theta_ci[0][0],
        'ci_upper': theta_ci[1][0],
        't_stat': theta / theta_se,
        'p_value': 2 * (1 - stats.norm.cdf(abs(theta / theta_se)))
    }

# Example usage
controls = ['prop_jewish25', 'unemployment33', 'population25', 
            'nazishare33', 'west', 'urban_scale', 'age', 'educ', 'female']
dml_results = double_ml_estimation(evs, 'intolerance', 'Distance', controls)
```

### 3.6 Bootstrapping

```python
def bootstrap_coefficient(data, formula, var_of_interest, 
                          n_bootstrap=1000, seed=42, 
                          cluster_var=None):
    """
    Bootstrap standard errors for regression coefficients.
    
    Strategies:
    1. Simple bootstrap: resample observations
    2. Cluster bootstrap: resample clusters
    3. Block bootstrap: for time series
    4. Wild bootstrap: for heteroskedasticity
    """
    np.random.seed(seed)
    n = len(data)
    boot_coefs = []
    
    for b in range(n_bootstrap):
        if cluster_var is None:
            # Simple bootstrap
            idx = np.random.choice(n, size=n, replace=True)
            boot_data = data.iloc[idx]
        else:
            # Cluster bootstrap
            clusters = data[cluster_var].unique()
            boot_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
            boot_data = pd.concat([data[data[cluster_var] == c] for c in boot_clusters])
        
        try:
            model = smf.ols(formula, data=boot_data).fit()
            boot_coefs.append(model.params[var_of_interest])
        except:
            continue
    
    boot_coefs = np.array(boot_coefs)
    
    return {
        'mean': boot_coefs.mean(),
        'se': boot_coefs.std(),
        'ci_lower': np.percentile(boot_coefs, 2.5),
        'ci_upper': np.percentile(boot_coefs, 97.5),
        'boot_coefs': boot_coefs
    }
```

### 3.7 DAG Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_dag(edges, node_positions=None, title=""):
    """
    Create and visualize a Directed Acyclic Graph.
    
    Parameters:
    -----------
    edges : list of tuples - (from_node, to_node)
    node_positions : dict - {node: (x, y)}
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    if node_positions is None:
        node_positions = nx.spring_layout(G)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=node_positions, with_labels=True, 
            node_color='lightblue', node_size=2000, 
            font_size=12, font_weight='bold',
            arrows=True, arrowsize=20,
            edge_color='gray', width=2)
    plt.title(title)
    plt.tight_layout()
    return G

# Example: DAG 1(a) from the paper
edges_1a = [
    ('U', 'F'), ('U', 'T'),
    ('X', 'T'), ('X', 'Y'),
    ('T', 'Y'), ('F', 'Y')
]
positions_1a = {
    'U': (-1, 1), 'T': (0, 1), 'F': (1, 2),
    'Y': (2, 1), 'X': (0, 0)
}
dag_1a = create_dag(edges_1a, positions_1a, "DAG 1(a): F is not post-treatment")
```

### 3.8 Publication-Quality Tables

```python
from stargazer.stargazer import Stargazer

def create_regression_table(models, model_names, 
                            covariate_labels=None, 
                            custom_notes=None):
    """
    Create publication-quality regression tables.
    """
    table = Stargazer(models)
    table.custom_columns(model_names)
    
    if covariate_labels:
        table.rename_covariates(covariate_labels)
    
    if custom_notes:
        table.add_custom_notes(custom_notes)
    
    return table.render_html()

# Alternative: Manual table creation
def format_coef(coef, se, stars=True):
    """Format coefficient with standard error and significance stars."""
    t_stat = abs(coef / se)
    if stars:
        if t_stat > 2.576:
            star = '***'
        elif t_stat > 1.96:
            star = '**'
        elif t_stat > 1.645:
            star = '*'
        else:
            star = ''
    else:
        star = ''
    
    return f"{coef:.3f}{star}\n({se:.3f})"
```

---

## 4. Typical Exam Tasks and Solutions

### 4.1 Task Type: Creating Informative Figures

**Common requests:**
- Visualize results from a regression table
- Show coefficient comparisons across specifications
- Display regional variation

```python
def plot_coefficient_comparison(models_dict, var_name, 
                                 model_labels=None, title=""):
    """
    Compare coefficients across model specifications.
    """
    coefs = []
    ci_lower = []
    ci_upper = []
    labels = []
    
    for name, model in models_dict.items():
        coefs.append(model.params[var_name])
        ci = model.conf_int().loc[var_name]
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
        labels.append(name if model_labels is None else model_labels.get(name, name))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(coefs))
    
    ax.errorbar(coefs, y_pos, 
                xerr=[np.array(coefs) - np.array(ci_lower), 
                      np.array(ci_upper) - np.array(coefs)],
                fmt='o', capsize=5, capthick=2, markersize=8)
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Coefficient Estimate')
    ax.set_title(title)
    plt.tight_layout()
    return fig
```

### 4.2 Task Type: Good vs. Bad Controls

**Framework for evaluation:**

```python
def evaluate_control_variable(var_name, dag_structure):
    """
    Evaluate whether a variable is a good or bad control.
    
    Bad controls:
    1. Post-treatment variables (descendants of treatment)
    2. Colliders (conditioning opens spurious paths)
    3. Mediators (blocks causal pathway if interested in total effect)
    
    Good controls:
    1. Confounders (common causes of T and Y)
    2. Variables on backdoor paths
    3. Precision variables (predict Y, uncorrelated with T)
    """
    evaluation = {
        'variable': var_name,
        'is_pretreatment': None,
        'is_confounder': None,
        'is_mediator': None,
        'is_collider': None,
        'recommendation': None,
        'reasoning': ""
    }
    
    # Check DAG structure
    # ... (implementation depends on DAG specification)
    
    return evaluation

# Pretreatment controls in Pepinsky study (GOOD)
pretreatment_controls = {
    'prop_jewish25': 'Confounder - affects camp location and possibly current attitudes',
    'unemployment33': 'Confounder - economic conditions affect Nazi support and persistence',
    'population25': 'Confounder - urbanization affects both',
    'nazishare33': 'Confounder - Nazi support predicts camp proximity and attitude persistence'
}

# Posttreatment controls (POTENTIALLY PROBLEMATIC)
posttreatment_controls = {
    'west': 'Potentially problematic if Distance → West → Y (mediator)',
    'immigrants07': 'Potentially problematic - may be on causal path',
    'urban_scale': 'Ambiguous - time-constant or changed?',
    'education': 'Could be mediator if regional factors affect education'
}
```

### 4.3 Task Type: Machine Learning Analysis

**Template for ML exploration:**

```python
def ml_exploration_template(data, outcome_var, features, 
                            categorical_vars=None, year_filter=None):
    """
    Standard ML exploration pipeline.
    
    Steps:
    1. Data preparation (filter, standardize)
    2. Run ML model
    3. Extract feature importance
    4. Cross-validation
    5. Interpret results
    """
    # Filter by year if needed
    if year_filter:
        data = data[data['year'] == year_filter].copy()
    
    # Prepare features
    df = data[features + [outcome_var]].dropna()
    
    # Standardize continuous variables
    continuous_vars = [v for v in features if v not in (categorical_vars or [])]
    
    scaler = StandardScaler()
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    
    # Handle categorical variables
    if categorical_vars:
        df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    
    # Split features and outcome
    feature_cols = [c for c in df.columns if c != outcome_var]
    X = df[feature_cols]
    y = df[outcome_var]
    
    # Run Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    
    return {
        'model': rf,
        'importance': importance,
        'cv_r2': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
        'n_obs': len(df)
    }
```

### 4.4 Task Type: Bootstrapping

```python
def bootstrap_strategies_comparison(data, formula, var_of_interest,
                                    cluster_var=None, n_bootstrap=1000):
    """
    Compare different bootstrapping strategies.
    
    Strategies:
    1. Nonparametric (pairs) bootstrap
    2. Residual bootstrap
    3. Cluster bootstrap
    4. Wild bootstrap
    """
    results = {}
    
    # 1. Nonparametric bootstrap
    results['nonparametric'] = bootstrap_coefficient(
        data, formula, var_of_interest, n_bootstrap, cluster_var=None
    )
    
    # 2. Cluster bootstrap (if cluster_var provided)
    if cluster_var:
        results['cluster'] = bootstrap_coefficient(
            data, formula, var_of_interest, n_bootstrap, cluster_var=cluster_var
        )
    
    # 3. Residual bootstrap
    base_model = smf.ols(formula, data=data).fit()
    residuals = base_model.resid
    fitted = base_model.fittedvalues
    
    boot_coefs_resid = []
    for _ in range(n_bootstrap):
        boot_resid = np.random.choice(residuals, size=len(residuals), replace=True)
        boot_data = data.copy()
        boot_data[formula.split('~')[0].strip()] = fitted + boot_resid
        boot_model = smf.ols(formula, data=boot_data).fit()
        boot_coefs_resid.append(boot_model.params[var_of_interest])
    
    results['residual'] = {
        'se': np.std(boot_coefs_resid),
        'ci_lower': np.percentile(boot_coefs_resid, 2.5),
        'ci_upper': np.percentile(boot_coefs_resid, 97.5)
    }
    
    # 4. Wild bootstrap (Rademacher weights)
    boot_coefs_wild = []
    for _ in range(n_bootstrap):
        weights = np.random.choice([-1, 1], size=len(residuals))
        boot_resid = residuals * weights
        boot_data = data.copy()
        boot_data[formula.split('~')[0].strip()] = fitted + boot_resid
        boot_model = smf.ols(formula, data=boot_data).fit()
        boot_coefs_wild.append(boot_model.params[var_of_interest])
    
    results['wild'] = {
        'se': np.std(boot_coefs_wild),
        'ci_lower': np.percentile(boot_coefs_wild, 2.5),
        'ci_upper': np.percentile(boot_coefs_wild, 97.5)
    }
    
    return results
```

---

## 5. Key Formulas (with Color Coding for Clarity)

### 5.1 OLS with Fixed Effects

$$Y_{is} = \color{blue}{\beta} \cdot Distance_{is} + \color{green}{\gamma} X_{is} + \color{red}{\phi_s} + \varepsilon_{is}$$

- $\color{blue}{\beta}$: Treatment effect (parameter of interest)
- $\color{green}{\gamma}$: Pretreatment control coefficients
- $\color{red}{\phi_s}$: State fixed effects

### 5.2 G-Estimation

**Stage 1:** Full model
$$Y_i = \alpha + \beta T_i + \color{green}{\gamma_{pre}} X_i^{pre} + \color{orange}{\gamma_{post}} X_i^{post} + u_i$$

**Stage 2:** Adjusted outcome
$$\tilde{Y}_i = Y_i - \color{orange}{\hat{\gamma}_{post}} X_i^{post}$$

**Stage 3:** Final estimation
$$\tilde{Y}_i = \alpha + \color{blue}{\beta} T_i + \color{green}{\gamma_{pre}} X_i^{pre} + v_i$$

### 5.3 Hausman Test

$$H = (\color{blue}{\hat{\beta}_{FE}} - \color{red}{\hat{\beta}_{RE}})' [\color{blue}{Var(\hat{\beta}_{FE})} - \color{red}{Var(\hat{\beta}_{RE})}]^{-1} (\color{blue}{\hat{\beta}_{FE}} - \color{red}{\hat{\beta}_{RE}}) \sim \chi^2_k$$

### 5.4 Double Machine Learning

$$\hat{\theta}_{DML} = \frac{\sum_i (\color{blue}{Y_i - \hat{f}(X_i)}) \cdot (\color{green}{T_i - \hat{g}(X_i)})}{\sum_i T_i \cdot (\color{green}{T_i - \hat{g}(X_i)})}$$

- $\color{blue}{\hat{f}(X_i)}$: ML prediction of Y from X
- $\color{green}{\hat{g}(X_i)}$: ML prediction of T from X

---

## 6. Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Missing values in covariates | Use `dropna()` or imputation |
| Categorical variables in ML | One-hot encode with `pd.get_dummies()` |
| Multicollinearity with FE | Check VIF, consider clustering SE |
| Small within-variation | Report warning, consider alternative FE |
| Bootstrap with clusters | Use cluster bootstrap |
| Standardization with categoricals | Only standardize continuous variables |
| Memory issues with large bootstrap | Use chunking or reduce `n_bootstrap` |

---

## 7. Required Python Packages

```python
# Core
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

# Causal Inference (optional)
# from econml.dml import LinearDML, CausalForestDML

# Utilities
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```

---

## 8. Quick Reference: Model Specifications from Paper

### Table 1: EVS Survey Data (Intolerance)
```python
# Bivariate
"intolerance ~ Distance"

# Bivariate + FE
"intolerance ~ Distance + C(state)"

# Pretreatment controls
"intolerance ~ Distance + prop_jewish25 + unemployment33 + population25 + nazishare33"

# Pretreatment + FE
"intolerance ~ Distance + prop_jewish25 + unemployment33 + population25 + nazishare33 + C(state)"

# G-estimation (full controls then adjust)
"intolerance ~ Distance + prop_jewish25 + unemployment33 + population25 + nazishare33 + lr + immigrants07 + unemployment07 + unemp + educ + female + age + urban_scale + west"
```

### Table 2: Election Data (AfD Share)
```python
# Pooled OLS
"AfDshare ~ distance2 + prop_juden + unemp33 + pop25 + vshare33"

# Fixed Effects (by state)
# Use PanelOLS with EntityEffects or dummy variables for NAME_1
```

---

## 9. Working with This Repository

### 9.1 Running the Analysis
```bash
# From repository root
cd exam
jupyter notebook Exam_Notebook.ipynb

# Or using Python directly (if converted to .py)
python Exam_Notebook.py
```

### 9.2 Key Files to Reference
- **Main paper:** `exam/study/Pepinsky et al. - 2024 - ...pdf`
- **Appendix:** `exam/study/Appendix.pdf`
- **Main replication notebook:** `exam/Exam_Notebook.ipynb`
- **This guide:** `AGENTS.md`

### 9.3 Important Notes
- All code assumes working directory is `exam/`
- Paths to data files are relative: `data/replication_archive/tables/`
- The notebook uses a `ModelSuite` class for model organization
- Random seed is set to 12435 for reproducibility
- State fixed effects use the `state` variable (DE1, DE2, etc.)
- Weimar province fixed effects require data preparation (see section 3.1)

---

*Last updated: 2025-12-04*
*Repository: https://github.com/[user]/sds-exam*
*For: SDS Exam - Replication Study*
