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
#    tight-tolerance L-BFGS (history 25), spike-and-slab prior on the shifts,
#    and an EM-learned interquantile fusion weight (adaptive_iq = TRUE).
fit <- getModel(y, taus, H = H, w = w,
                fit_method = "map",          # MAP + Laplace draws
                prior_gamma = "spike_slab",
                seed = 1)

fit$map$termination   # human-readable optimizer exit status
fit$map$coverage      # per-quantile empirical coverage of the fitted curves
                      # (a bad fit warns automatically)
fit$iq_em$lambda_iq2  # the learned squared IQ weight
fit$iq_em$trace       # one row per EM iteration

# To pin the IQ weight instead of learning it, give the SQUARED value.
# `lambda_iq2` replaces the old `lambda_iq`: the effective fusion rate is
# sqrt(lambda_iq2), so the former `lambda_iq = 0.2` is now `lambda_iq2 = 0.04`.
fit_fixed <- getModel(y, taus, H = H, w = w,
                      fit_method = "map", prior_gamma = "spike_slab",
                      adaptive_iq = FALSE, lambda_iq2 = 0.04, seed = 1)

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
| `cv_copss_grid()` | Grid-search CV over hyperparameters (MAP fits); extra `getModel` arguments pass through `base_args` |

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

### 0.6.8

- **LASSO-type priors are fitted with their scale latents integrated out.** The joint
  MAP of the normal scale-mixture hierarchies (`lasso`, `group_lasso`, `het_group_lasso`,
  `adaptive_lasso`, and the same priors on `betaX`) does not exist: the joint density is
  unbounded as a scale latent tends to 0 together with its coefficients, and the optimizer
  occasionally reached that spike (a het_group_lasso fit on a 90-point series failed with
  "Initialization failed" on every retry). The Stan program now uses the marginal priors:
  `lasso` is Laplace with rate `sqrt(lambda_lasso2)` (Park and Casella, 2008); `group_lasso`
  is `lambda^m exp(-lambda ||gamma_j||_2)` with its normalizing constant (Kyung et al., 2010,
  hierarchy (6)); `adaptive_lasso` is Laplace with local rate `sqrt(lambda2_qj)` and keeps
  `lambda2_qj ~ Gamma(a, b)`; `het_group_lasso` is Laplace with block rate `sqrt(omega_j)` and
  keeps `omega_j ~ InvGamma(1/2, c/2)`. `.bqq_lp17()` matches term by term. The group norm is
  smoothed in the optimizer with the IQ constant (`sqrt(||g||^2 + iq_smooth^2)`). Results
  change for these four priors relative to 0.6.7; spike-and-slab priors are unaffected.

### 0.6.7

- **Cross-validation has one function.** `cv_copss_map()` and `cv_copss_mcmc()` were
  removed. `cv_copss_grid(y, taus, H, X, w, grid, base_args, loss, seed, verbose)` calls
  `getModel()` directly with each grid row merged into `base_args`, so a CV fit is the
  same chain (inner optimization to convergence, one EM update, repeat, stop on the
  relative gain of the complete-data log posterior) as the final fit. Only the columns
  of `grid` are tuned; the manuscript tunes `spike_sd` and `lambda_lasso2_b`.
- **EM stopping rule.** The chain stops when the relative gain of the complete-data log
  posterior (Eq. 17 of the manuscript plus `r (m - 1) log lambda_iq`) falls below
  `iq_em_lp_tol` (default 1e-2). `iq_em_tol`, `iq_em_mc_tol`, `iq_em_switch_tol`,
  `iq_em_warm` and the `"hybrid"` value of `iq_em_step` were removed;
  `iq_em_step = c("fixedpoint", "recursion")` selects Appendix C (C.9) or (C.8).
  `iq_em_estep = "closed"` (default) uses the folded-normal mean of |d| under the
  Laplace approximation; `"draws"` uses posterior draws.
- **Detection defaults.** `detectChangepoints_gamma(adjust = "raw")` is the default
  decision rule; `basis` is `c("quantile", "qss", "lmom")` and all three are always
  reported. The `"maxent"` basis was retired and `plotGammaHeatmap()` no longer draws
  its panel.

### Wider spike defaults (0.5.2)

- `getModel()`'s `spike_sd` and `beta_spike_sd` both default to **0.1**, raised from
  0.05. A narrower spike makes the null mixture component close to a point mass, so
  any noise-driven coefficient is pushed into the slab and flagged.
- The evidence is from the `lmom_3cfg` simulation, null arm, pooled over three
  settings and both spike priors: `spike_sd <= 0.05` flagged **71 of 155**
  replications (0.458) against **1 of 85** (0.012) for `spike_sd >= 0.10` — Fisher
  exact p = 6.5e-16, odds ratio 70 (95% CI 12–2833). It also accounts for
  `spike_slab_lasso`'s 0.526 false-alarm rate, which is not a distinct failure: that
  prior simply lands on a tight spike far more often (553/598 fits vs 190/598).
- **No multiplicity correction repairs this.** Size is identical at `raw`, `bonf`,
  `holm`, `bh` and `calib` — those detections survive every threshold adjustment.
- **Detections change**, so `spike_slab` / `spike_slab_lasso` results from 0.5.1 and
  earlier are not comparable unless `spike_sd` is passed explicitly. Pass
  `spike_sd = 0.05` to reproduce an older fit.
- The direct evidence concerns `spike_sd`; `beta_spike_sd` was raised with it for
  consistency of spike width across the two coefficient blocks, without an
  equivalent study of the covariate side.

### Monitoring bases (0.5.1)

- `detectChangepoints_gamma(basis = ...)` accepts **`"lmom"`** in addition to
  `"quantile"` and `"qss"` (the `"maxent"` basis added here was retired in 0.6.7).
  Results appear at `det$tests$lmom`, with the flat alias `z_white_lmom`, and
  `plotGammaHeatmap()` renders one panel per basis. All bases are 4-cell linear
  contrasts on the block gammas and are scored by identical code, so any difference
  between them comes from the weights alone.
- **0.5.1 changes what `"lmom"` means.** It now integrates
  `lambda_{r+1} = int_0^1 Q(u) P*_r(u) du` over the **whole** unit interval, as the
  definition requires, using a surrogate quantile function
  (piecewise-uniform interior, continuity-matched exponential tails). The 0.5.0
  version integrated only `[tau_1, tau_m]` and then projected each shape row off the
  location row to repair the resulting location leak. **The two give different
  weights and different detections** -- e.g. the first L-skewness weight moves from
  0.095133 to 0.079479 -- so results from 0.5.0 and 0.5.1 are not comparable. The
  projection is gone; location invariance now holds by construction.
- The `"lmom"` weights depend on `taus` alone (no baseline).
  Derivations: `simulation_study/lmom_3cfg/MONITORING_BASES_math.md`.

### Breaking changes (0.5.0)

- `getModel()`'s `lambda_iq` is renamed **`lambda_iq2`** and is now the
  **squared** interquantile fusion weight: the rate applied to
  `|gamma[q] - gamma[q-1]|` is `sqrt(lambda_iq2)`. This matches the existing
  `lambda_lasso2` / `lambda_beta2` convention. **To reproduce a previous fit,
  square the old value** -- `lambda_iq = 0.5` becomes `lambda_iq2 = 0.25`
  (verified bit-identical on the ARCOS fit).
- `getModel(adaptive_iq = TRUE)` is the **new default**: `lambda_iq2` is learned
  by an EM recursion run between refits, controlled by `iq_em_max_iter` (and, since
  0.6.7, `iq_em_lp_tol`). Diagnostics are returned in `fit$iq_em` (including a per-iteration
  `trace`). Because every EM iteration is a *full refit*, this makes a default
  `getModel()` call several times more expensive; pass `adaptive_iq = FALSE` for
  the old single-fit behavior.
  The E-step uses Laplace-approximation draws rather than the exact conditional
  posterior, so this is an *approximate* empirical-Bayes EM with no
  monotone-ascent guarantee.
- **`iq_em_update` was removed in 0.5.2.** It chose between the creeping EM
  recursion `lambda2_{s+1} = 2N*lambda2_s/(lambda_s*Sbar + N)` and a direct jump
  to its fixed point `(N/Sbar)^2`. Solving the first for its fixed point *gives*
  the second, so both converge to the same value from any start (verified from
  1e-4, 1 and 1e6) — but the creeping form needs ~44 refits where the jump needs
  1, and `"em"` was the **default**. Its only claim was monotone ascent under an
  exact E-step, which a Monte-Carlo E-step does not provide. The M-step now
  always jumps. The outer loop still iterates, because `Sbar` is recomputed from
  a refit at the updated `lambda`.
- The CV helper (`cv_copss_grid`) defaults
  `adaptive_iq = FALSE` so a tuning sweep is not silently multiplied by the EM,
  and they now **error** on an unrecognized tuning name instead of silently
  dropping it -- a grid still carrying `lambda_iq` would otherwise have tuned
  nothing while appearing to run.

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
