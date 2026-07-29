# bqq: Bayesian Quintuple Quantile Chart

**bqq** implements a Bayesian quintuple quantile (BQQ) charting approach for Phase I
statistical process monitoring. It fits a multi-quantile regression model jointly
across quantile levels — a smoothed score likelihood with non-crossing and
interquantile-shrinkage penalties and sparsity-inducing priors on blockwise shift
coefficients, computed via [Stan](https://mc-stan.org/) — and detects distributional
change-points through calibrated block tests on the shift coefficients.

## Installation

Install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("yuhuiyao88/bqq")
```

### Requirements

- R >= 3.5.0
- [rstan](https://mc-stan.org/rstan/) >= 2.21.0 (and a working C++ toolchain for Stan)
- Recommended: [quantreg](https://cran.r-project.org/package=quantreg) (pilot LASSO
  quantile-regression initialization and adaptive IQ weights; the package falls back
  gracefully without it), ggplot2 and patchwork (plots)

## Overview

The BQQ methodology fits the five conditional quantiles (0.025, 0.25, 0.5, 0.75,
0.975) jointly over time, anchors their in-control levels with a prior elicited from
a warm-up period, and represents distributional changes through blockwise shift
coefficients, so change-point detection becomes structured variable selection.
The package provides:

- **Model fitting** via MAP estimation (with Laplace posterior draws), MCMC, or
  MAP-initialized MCMC, from a built-in Stan program
- **An informative warm-up prior for the intercepts** (the default): each
  per-quantile intercept is centered at the warm-up-window empirical quantile with
  unit-information scale — equivalently, a power prior on the warm-up period with
  discount `a0 = 1/w`
- **Sparsity-inducing priors** on the shift coefficients (spike-and-slab primary;
  LASSO, adaptive LASSO, group LASSO, heterogeneous group LASSO, and
  spike-and-slab LASSO alternatives)
- **Interquantile (IQ) shrinkage** that fuses adjacent-quantile coefficients with
  data-adaptive weights, and a **non-crossing penalty** preserving quantile ordering
- **Change-point detection** by posterior whitening of the shift coefficients
  followed by union–intersection (UI) and Hotelling T² block tests, run on the raw
  quantile basis and/or the QSS shape basis, with the full across-block adjustment
  family (raw, Holm, Bonferroni, BH, and calibrated single-step charting constants
  controlling the family-wise false-alarm probability)
- **Quantile Shape Statistics (QSS)**: location, scale, skewness, and kurtosis
  profiles derived from the fitted quantile process
- **Cross-validation** for hyperparameter tuning and **visualization** for quantile
  processes, QSS profiles, and shift heatmaps

## Quick Start

```r
library(bqq)
set.seed(123)

# 1. Simulate data with a sustained mean shift
n <- 360
y <- rnorm(n)
y[252:n] <- y[252:n] + 1

# 2. Quantile levels, block design, warm-up period
taus <- c(0.025, 0.25, 0.5, 0.75, 0.975)
l <- 30   # block length
w <- 30   # warm-up period (in-control reference window)
H <- getSustainedShift(n, l = l, w = w)

# 3. Fit. The defaults implement the full method: unit-information warm-up prior
#    on the intercepts, marginal LASSO quantile-regression initialization,
#    tight-tolerance L-BFGS (history 25), spike-and-slab prior on the shifts.
fit <- getModel(y, taus, H = H, w = w,
                fit_method = "map",          # MAP + Laplace draws
                prior_gamma = "spike_slab",
                lambda_iq = 0.2, seed = 1)

fit$map$termination   # human-readable optimizer exit status
fit$map$coverage      # per-quantile empirical coverage of the fitted curves
                      # (a bad fit warns automatically)

# 4. Predictive quantile draws [iterations x quantiles x time]
eta <- getEta(fit, H = H)

# 5. Change-point detection: both bases, both statistics. `adjust` is the
#    across-block decision rule of record ("calib" default; also "raw",
#    "holm", "bonf", "bh") -- every plot renders exactly this rule.
det <- detectChangepoints_gamma(fit, taus = taus, l = l, w = w,
                                basis = c("quantile", "qss"),
                                statistic = c("ui", "hotelling_t2"),
                                adjust = "calib",
                                y = y, eta = eta)

# 6. Plots: pure renderers of the fit and the recorded detection decisions.
plotQuantileProcess(fit, detection = det)        # bands + localized change-points
plotQSSProcess(fit, eta = eta, detection = det)  # QSS profiles + detected blocks
plotGammaHeatmap(fit, det)                       # blocks (grey) + cells (black)
```

## Core Functions

### Design Matrices

| Function | Description |
|---|---|
| `getSustainedShift(n, l, w)` | Cumulative step design: each column is 1 from its block start to the end (coefficients are shift *increments*) |
| `getIsolatedShift(n, l, w)` | Block-diagonal design for transient/windowed shifts |

### Model Fitting

| Function | Description |
|---|---|
| `getModel()` | Fit the joint multi-quantile model via MAP, MCMC, or MAP+MCMC; returns the fit, Laplace samples, `stan_data` (the exact prior/bandwidth used), and MAP diagnostics (`termination`, `coverage`) |
| `getLaplaceSamples()` | Approximate posterior samples from a MAP fit |
| `getEta()` | Predictive quantile array `[iterations x quantiles x time]` |

### Inference

| Function | Description |
|---|---|
| `getQSS()` | Quantile Shape Statistics (location, scale, skewness, kurtosis) from predictive quantiles |
| `detectChangepoints_gamma()` | Posterior whitening + UI / Hotelling T² block tests on the quantile and/or QSS bases; computes the full adjustment family (raw / Holm / Bonferroni / BH / calibrated) with the matching cell-level constants and flags, and records the decision rule (`adjust`) that all plots render |

### Cross-Validation

| Function | Description |
|---|---|
| `cv_copss_map()` | Order-preserved 2-fold CV (MAP fits) |
| `cv_copss_grid()` | Grid-search CV over hyperparameters (MAP fits); extra `getModel` arguments pass through `base_args` |
| `cv_copss_mcmc()` | Grid-search CV using MCMC fits |

### Visualization

| Function | Description |
|---|---|
| `plotQuantileProcess()` | The five fitted quantile curves over time |
| `plotGammaHeatmap()` | Shift heatmap: whitened-z fill, grey borders on OOC blocks, black borders on localized cells — all decisions (basis, statistic, `adjust`, constants) taken from the `detection` object; only display options (colors, labels, `mark_cells`) are settable |
| `plotQSSProcess()` | QSS profiles over time with credible bands and detected blocks |

## Model Details

The conditional quantile at level τ_q and time i is modeled as

$$\eta_{q,i} = \beta_{0,q} + x_i^\top \beta_{X,q} + h_i^\top \gamma_q + \mathrm{offset}_i ,$$

estimated through a score-based likelihood with logistic smoothing of the check-loss
indicator (bandwidth by the Fernandes–Guerre–Horta rule of thumb) and a quantile
kernel `min(τ,τ′) − ττ′` coupling the levels.

### Priors (defaults)

- **Intercepts** `β0[q] ~ Normal(beta0_loc[q], beta0_scale[q])` with, by default,
  `beta0_loc` = the empirical τ-quantiles of the warm-up period and `beta0_scale` =
  the **unit-information** scale `sqrt(τ(1−τ)) / f̂` (Kass & Wasserman, 1995), where
  `f̂` is a kernel density estimate of the warm-up period. Together this equals the
  power prior on the warm-up period with discount `a0 = 1/w` (Ibrahim & Chen, 2000;
  Bourazas, Kiagias & Tsiamyrtzis, 2022), and anchors each intercept in the spirit
  of the empirical-quantile anchoring of Yang & He (2012). Both are overridable
  (`beta0_loc`, `beta0_scale`); with `log_flag = 1` everything is computed on the
  log (modeling) scale.
- **Shift coefficients** γ: spike-and-slab by default; five LASSO-type alternatives.
- **IQ shrinkage** fuses `|γ_q − γ_{q−1}|` with adaptive weights from pilot quantile
  regressions (Jiang, Wang & Bondell, 2013); the intercept is never IQ-penalized.
- **Non-crossing penalty**: L1 hinge on finite differences in τ.

### Computation

- `fit_method = "map"` (recommended): L-BFGS with **tight convergence tolerances**
  (`tol_rel_obj = tol_rel_grad = 1e2`, `iter = 10000`, `history_size = 25`). The
  smoothed-score objective has near-flat plateaus; loose tolerances can stop there
  prematurely while reporting convergence.
- **Initialization** (`map_init`): `"pilot"` (default) starts at marginal LASSO
  quantile-regression estimates per level (`quantreg::rq.fit.lasso` on `[1|X|H]`,
  intercept unpenalized, penalty scaling `sqrt(τ(1−τ) n log d)` following Belloni &
  Chernozhukov, 2011) — the LASSO-initialization strategy with oracle support in
  Fan, Xue & Zou (2014). `"prior_center"` starts at the prior mode (in-control
  state); `"random"` restores rstan's default.
- **Diagnostics**: every MAP fit reports `fit$map$termination` (translated
  optimizer exit status; exit code 70 = line search exhausted, the expected ending
  at a converged optimum) and `fit$map$coverage` (empirical coverage of the fitted
  curves), warning automatically when a fit looks wrong.

### Detection

Shift coefficients are posterior-whitened (`z̃ = Σ^{−1/2} γ̄`), then combined per
block by the UI statistic (max |z̃|) and/or Hotelling T² (sum z̃²). Each test returns
the full across-block family — raw, Holm, Bonferroni, BH, and the **calibrated**
single-step rule using analytic charting constants (Šidák-type) that control the
probability of any false alarm across all blocks and cells jointly.

### Deprecations (0.4.6)

- 0.4.8: plots are pure renderers. `plotGammaHeatmap()` lost `adjust`, `basis`,
  `sig_block`, and `alpha`; `plotQuantileProcess()`/`plotQSSProcess()` lost
  `taus`, `alpha`, `adjust`, `basis` (and the redundant `y` override) and gained
  the display toggles `show_onset`/`show_located`. The decision rule moved into
  `detectChangepoints_gamma(adjust = ...)`, which now also returns cell-level
  constants and flags (`$cell_c`, `$cells`) implementing the manuscript's
  Eqs. (21)-(25). The long-deprecated `calibrated`/`block_test`/`qss` arguments
  were removed. `getModel()` now records `taus` in its return value.
- 0.4.6: `detectChangepoints_gamma()` `n_calib` removed; `plotGammaHeatmap()`
  `taus`, `scale`, `whiten` removed (levels and fill follow `detection`).

## References

- Belloni, A., & Chernozhukov, V. (2011). ℓ1-Penalized Quantile Regression in
  High-Dimensional Sparse Models. *Annals of Statistics*, 39(1), 82–130.
- Bourazas, K., Kiagias, D., & Tsiamyrtzis, P. (2022). Predictive Control Charts
  (PCC): A Bayesian Approach in Online Monitoring of Short Runs. *Journal of
  Quality Technology*, 54(4), 367–391.
- Fan, J., Xue, L., & Zou, H. (2014). Strong Oracle Optimality of Folded Concave
  Penalized Estimation. *Annals of Statistics*, 42(3), 819–849.
- Fernandes, M., Guerre, E., & Horta, E. (2021). Smoothing Quantile Regressions.
  *Journal of Business & Economic Statistics*, 39(1), 338–357.
- Ibrahim, J. G., & Chen, M.-H. (2000). Power Prior Distributions for Regression
  Models. *Statistical Science*, 15(1), 46–60.
- Jiang, L., Wang, H. J., & Bondell, H. D. (2013). Interquantile Shrinkage in
  Regression Models. *Journal of Computational and Graphical Statistics*, 22(4),
  970–986.
- Kass, R. E., & Wasserman, L. (1995). A Reference Bayesian Test for Nested
  Hypotheses and Its Relationship to the Schwarz Criterion. *JASA*, 90(431),
  928–934.
- Yang, Y., & He, X. (2012). Bayesian Empirical Likelihood for Quantile Regression.
  *Annals of Statistics*, 40(2), 1102–1131.

## License

GPL-3
