# bqq: Bayesian Quintuple Quantile Chart

**bqq** implements a Bayesian quintuple quantile (BQQ) charting approach for Phase I statistical process monitoring. It fits Bayesian multi-quantile regression models with interquantile shrinkage, grouped Bayesian LASSO, and non-crossing penalties via [Stan](https://mc-stan.org/), and provides control charts and change-point detection tools based on predictive quantile vectors.

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
- **Grouped Bayesian LASSO** on shift coefficients with data-adaptive penalty learning
- **Non-crossing penalties** to maintain quantile monotonicity
- **Gamma-coefficient decorrelation-based** change-point detection
- **Quantile Shape Statistics (QSS)**: location, scale, skewness, and kurtosis derived from the fitted quantile function
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

# 5. Gamma-based change-point detection (decorrelation approach)
gamma_result <- detectChangepoints_gamma(fit, taus = taus, l = l, w = w)
plot_gamma_detection(gamma_result, true_shift = shift_start)

# 6. Quantile Shape Statistics
qss <- getQSS(eta, taus = taus)
plot_qss_series(qss, w = w, true_shift = shift_start)
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
| `getQSS()` | Compute Quantile Shape Statistics from predictive quantiles |
| `detectChangepoints_gamma()` | Decorrelation-based change-point detection using gamma coefficients |

### Cross-Validation

| Function | Description |
|---|---|
| `cv_copss_map()` | COPSS-style 2-fold CV for `lambda_nc` (MAP fits) |
| `cv_copss_grid()` | Grid search CV over multiple hyperparameters (MAP fits) |
| `cv_copss_mcmc()` | Grid search CV using MCMC fits |

### Visualization

| Function | Description |
|---|---|
| `plot_quantile_chart()` | Fitted quantile curves overlaid on data |
| `plot_gamma_detection()` | Block-level significance barplot from gamma detection |
| `plot_qss_series()` | QSS time series (location, scale, skewness, kurtosis) with posterior bands |

## Model Details

The conditional quantile function at level $\tau_q$ is modeled as:

$$\eta_{q,i} = \mu_{q,i} + x_i^\top \beta_q + h_i^\top \gamma_q + \text{offset}_i$$

where:

- $\mu_{q,\cdot}$ is a quantile-specific random walk capturing smooth temporal trends
- $\beta_q$ are fixed-effect coefficients (optional covariates)
- $\gamma_q$ are shift coefficients penalized by a grouped Bayesian LASSO
- The loss function uses the smoothed check (pinball) loss

### Penalties

- **Interquantile shrinkage**: Penalizes $|\gamma_q - \gamma_{q-1}|$ with weights that increase for outer quantiles, borrowing strength from the center of the quantile distribution (Jiang, Wang, & Bondell, 2013)
- **Grouped Bayesian LASSO**: Shrinks each column of gamma jointly across quantile levels toward zero
- **Non-crossing penalty**: L1 hinge penalty on finite differences in $\tau$ to enforce quantile ordering

### Estimation Methods

- `fit_method = "map"`: Fast MAP optimization with Laplace approximation for posterior samples
- `fit_method = "mcmc"`: Full MCMC via Stan; posterior median as point estimate
- `fit_method = "map_mcmc"`: MAP point estimates with MCMC posterior draws (MAP used as initialization)

## References

- Jiang, L., Wang, H. J., & Bondell, H. D. (2013). Interquantile Shrinkage in Regression Models. *Journal of Computational and Graphical Statistics*, 22(4), 970-986.

## License

GPL-3
