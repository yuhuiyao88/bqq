# bqq: Bayesian Quintuple Quantile Chart

**bqq** implements a Bayesian quintuple quantile (BQQ) charting approach for Phase I statistical process monitoring. It fits Bayesian multi-quantile regression models with interquantile shrinkage, Bayesian LASSO-type priors, and non-crossing penalties via [Stan](https://mc-stan.org/), and provides control charts and change-point detection tools based on predictive quantile vectors.

## Installation

Install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("yuhuiyao88/bqq")
```

### Requirements

- R >= 3.5.0
- [rstan](https://mc-stan.org/rstan/) >= 2.21.0 (and a working C++ toolchain for Stan)

## Overview

The BQQ methodology monitors a process by fitting a multi-quantile regression model jointly across quantile levels, then testing whether the fitted quantile vectors deviate from in-control behavior. The package provides:

- **Model fitting** via MAP estimation, MCMC, or MAP-initialized MCMC with a built-in Stan program
- **Interquantile shrinkage** that borrows strength across quantiles to stabilize outer-quantile estimates
- **Bayesian LASSO-type priors** on both user covariates and shift coefficients, with the intercept retained under a separate weakly informative prior
- **Non-crossing penalties** to maintain quantile monotonicity
- **Change-point detection** by posterior whitening of the shift coefficients followed by union-intersection (UI) and Hotelling $T^2$ block tests, with family-wise error (FAP) control across blocks — run on both the raw quantile basis and the QSS shape basis
- **Quantile Shape Statistics (QSS)**: location, scale, skewness, and kurtosis derived from the fitted quantile function, for both distributional profiling and shape-shift inference
- **Cross-validation** for hyperparameter tuning (non-crossing penalty, LASSO rate, IQ shrinkage rate)
- **Visualization** functions for quantile charts, control charts, QSS time series, and detection barplots

## Quick Start

```r
library(bqq)
set.seed(123)

# 1. Simulate data with a sustained mean shift
n <- 360
y <- rnorm(n)
shift_start <- 252
y[shift_start:n] <- y[shift_start:n] + 1

# 2. Set up design matrix and quantile levels
taus <- c(0.025, 0.25, 0.5, 0.75, 0.975)
l <- 30   # block length
w <- 30   # warm-up period
H <- getSustainedShift(n, l = l, w = w)

# 3. Fit the model (MAP for speed)
fit <- getModel(y, taus, H = H, w = w,
                fit_method = "map",
                map_hessian = FALSE, map_iter = 2000,
                lambda_nc = 100,
                adaptive_gamma = TRUE,
                lambda_lasso2_b = 0.5,
                adaptive_iq = TRUE,
                lambda_iq2_b = 0.1)

# 4. Extract predictive quantiles
eta <- getEta(fit, H = H)

# 5. Change-point detection: posterior whitening + union-intersection (UI) and
#    Hotelling T^2 block tests, on the quantile and QSS (shape) bases
detection <- detectChangepoints_gamma(fit, taus = taus, l = l, w = w,
                                      basis = c("quantile", "qss"),
                                      statistic = c("ui", "hotelling_t2"))
plotGammaHeatmap(fit, detection)              # block-shift heatmap with significance borders

# 6. Distributional profiling via Quantile Shape Statistics (QSS) over time
plotQSSProcess(fit, H = H, detection = detection)   # location/scale/skewness/kurtosis bands
plotQuantileProcess(fit)                            # fitted quantile process
```

## Core Functions

### Design Matrices

| Function | Description |
|---|---|
| `getSustainedShift(n, l, w)` | Lower-triangular block design matrix for persistent shifts |
| `getIsolatedShift(n, l, w)` | Block-diagonal design matrix for transient/windowed shifts |

### Model Fitting

| Function | Description |
|---|---|
| `getModel()` | Fit the multi-quantile regression model via MAP, MCMC, or MAP+MCMC |
| `getLaplaceSamples()` | Generate approximate posterior samples from a MAP fit |
| `getEta()` | Extract the 3D predictive quantile array `[iterations x quantiles x time]` |

### Inference

| Function | Description |
|---|---|
| `getQSS()` | Compute Quantile Shape Statistics (location, scale, skewness, kurtosis) from predictive quantiles |
| `detectChangepoints_gamma()` | Change-point detection: posterior whitening + UI / Hotelling $T^2$ block tests on the quantile and QSS bases, with FAP control and multiplicity adjustment |

### Cross-Validation

| Function | Description |
|---|---|
| `cv_copss_map()` | COPSS-style 2-fold CV for `lambda_nc` (MAP fits) |
| `cv_copss_grid()` | Grid search CV over multiple hyperparameters (MAP fits) |
| `cv_copss_mcmc()` | Grid search CV using MCMC fits |

### Visualization

| Function | Description |
|---|---|
| `plotQuantileProcess()` | Fitted quantile process (the five quantile curves over time) |
| `plotGammaHeatmap()` | Block-shift heatmap (quantile and/or QSS panels) with significance borders from the UI / $T^2$ detection |
| `plotQSSProcess()` | QSS distributional profiling over time (location, scale, skewness, kurtosis) with posterior credible bands and change onsets |

## Model Details

The conditional quantile function at level $\tau_q$ is modeled as:

$$\eta_{q,i} = \mu_{q,i} + x_i^\top \beta_q + h_i^\top \gamma_q + \text{offset}_i$$

where:

- $\mu_{q,\cdot}$ is a quantile-specific random walk capturing smooth temporal trends
- $\beta_{0,q}$ are intercept coefficients with weakly informative normal priors
- $\beta_{X,q}$ are optional covariate coefficients with selectable priors (normal, lasso, adaptive lasso, spike-and-slab, group lasso, or heterogeneous group lasso)
- $\gamma_q$ are shift coefficients penalized by a Bayesian LASSO-type prior
- The loss function uses the smoothed check (pinball) loss

### Penalties

- **Interquantile shrinkage**: Penalizes $|\gamma_q - \gamma_{q-1}|$ with weights that increase for outer quantiles, borrowing strength from the center of the quantile distribution (Jiang, Wang, & Bondell, 2013)
- **Bayesian LASSO-type priors**: Supports grouped, adaptive, and heterogeneous-group shrinkage on the shift coefficients
- **Non-crossing penalty**: L1 hinge penalty on finite differences in $\tau$ to enforce quantile ordering

### Estimation Methods

- `fit_method = "map"`: Fast MAP optimization with Laplace approximation for posterior samples
- `fit_method = "mcmc"`: Full MCMC via Stan; posterior median as point estimate
- `fit_method = "map_mcmc"`: MAP point estimates with MCMC posterior draws (MAP used as initialization)

## References

- Jiang, L., Wang, H. J., & Bondell, H. D. (2013). Interquantile Shrinkage in Regression Models. *Journal of Computational and Graphical Statistics*, 22(4), 970-986.

## License

GPL-3
