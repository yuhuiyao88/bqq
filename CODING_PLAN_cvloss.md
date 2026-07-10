# Coding Plan — CV Validation-Loss Monitoring for the `bqq` Package

Goal: add the validation-loss model-comparison monitor (full `X+H` vs reduced `X`-only,
scored by held-out predictive loss over posterior draws) **as an additional path**, without
changing the existing multiplicity-adjustment monitor. Version 0.3.7.

---

## 0. What already exists (verified in the repo)

- `getModel()` (getModel.R:166) — the fit. Args used here: `H`, `X`, `w`, `taus`, `lambda_nc`,
  `eps_rel`, `prior_beta`, `prior_gamma`, `fit_method ∈ {mcmc, map_mcmc, map}`, `map_hessian`.
  Returns `$map$par`, `$map`, `$hessian`, `$laplace_samples` (if map_hessian), `$fit` (stanfit),
  `$H`, `$X`, `$fit_method`.
- `getLaplaceSamples()` (getInference.R:49) — MAP+Hessian → draws `list(beta[S,m,p], gamma[S,m,r])`.
  Tier-1 uses the Hessian (needs `map_hessian=TRUE`); Tier-2 is a heuristic fallback.
- `getEta()` (getInference.R:235) — draws → fitted-quantile array `[iterations, quantiles, time]`;
  accepts arbitrary `H`, `X` (so it can predict on a held-out fold). **This is the reuse point.**
- `detectChangepoints_gamma()` (getInference.R:496) — existing monitor: whitens gamma draws,
  per-block tests, multiplicity adjustment, selection. **Unchanged.**
- `cv_copss_map/grid/mcmc()` (cv_copss.R) — odd/even CV, scores only the point estimate
  (`map_hessian=FALSE`), returns a scalar `val_loss` per grid row. **Unchanged.**
- `pinball_loss()` (cv_copss.R:18) — RAW check loss (no smoothing).
- Smoothing (getModel.R:434,550-552): `smooth_T = base_scale*eps_rel`; `Ilt = inv_logit(-r/smooth_T)`;
  score `psi = tau - Ilt`. The likelihood is a quadratic form of the aggregated scores (score-based),
  **not** a summed pinball loss.
- `plotBQQ.R` — `plotQuantileProcess`, `plotQSSProcess`, `plotGammaHeatmap` (+ internal
  `.bqq_sig_blocks`, `.bqq_fitted_quantiles`). Existing visualization to reuse.
- No `tests/` directory yet.

---

## 1. Component (1) — CV that outputs posterior draws + per-time-point held-out loss

Do **not** touch grid selection (keep the cheap point-estimate `cv_copss_*`). Add, in `cv_copss.R`:

`cv_holdout_pointwise(y, taus, H, X, w, lambda_nc, eps_rel, prior_*, fit_method, n_draws, seed, ...)`
- Runs the **same** odd/even split. For each fold, fit at the **selected** hyperparameters with draws:
  - MAP: `getModel(..., fit_method="map", map_hessian=TRUE)` → draws via `$laplace_samples` /
    `getLaplaceSamples`.
  - MCMC: `getModel(..., fit_method="mcmc")` → draws directly.
- Get per-draw fitted quantiles on the **held-out** fold with `getEta(fit, H=H_val, X=X_val)`
  → `[S, m, n_val]`.
- Compute per-observation, per-draw **raw** check loss (`pinball_loss_pointwise`, Component 4) →
  `v[S, n_val]`; stitch the two folds back into **time order** → `v_full[S, n]`.
- Return `v_full` (draws × time), plus `taus`.

## 2. Component (2) — reduced-model held-out loss (on demand only)  [decisions 1, 5]

`cv_holdout_pointwise_reduced(...)` — same as Component 1 but **drop `H`** (`gamma = 0`):
- **Always a 1-column `getModel` fit** (`getModel(H=NULL, X=X, ...)`), including when `X` is NULL
  (then the design is the intercept-only `X0` column) — one consistent Laplace draw path, no
  closed-form branch. **[decision 1]**
- The reduced fit is a **separate** fit, and its `X`-regularization uses the **same beta
  hyperparameters as the full fit** — `prior_beta`, `adaptive_beta`, `lambda_beta2_*` — carried over
  verbatim. Gamma hyperparameters are irrelevant (no `H`). **[decision 5]**
- **Not automatic:** only invoked by the new monitor when the validation-loss chart is requested;
  never in the standard `getModel`→`detectChangepoints_gamma` pipeline.

## 3. Component (3) — new monitor `detectChangepoints_cvloss()` (additional, in getInference.R)

New exported function parallel to `detectChangepoints_gamma`; existing monitor untouched.
- Inputs: `v_full[S,n]`, `v_reduced[S,n]`, `l`, `w`, `alpha=0.05`, `n_pairs=1000`, `seed`.
- Observed: draw SRS pairs `(s, s*)`; `d_t = |v_reduced[s,t] - v_full[s*,t]|`;
  block max `D_j = max_{t∈block j} d_t`; overall `D = max_j D_j`.
- Null: `d*_t = |v_reduced[s,t] - v_reduced[s*,t]|`; `D* = max_j max_{t} d*_t`.
- Cutoff `c_alpha = (1-alpha)`-quantile of `D*` over SRS pairs; **flag if `D > c_alpha`**; report
  `p = mean(D* >= D)`, `changepoint = argmax block/point`, per-block `D_j`. Same block indexing
  (`l`, `w`) as `detectChangepoints_gamma`.

## 4. Component (4) — RAW check loss for scoring/charting (align with literature)  [decision 2, revised]

The loss used to **score and chart** is the **raw check (pinball) loss**, not a smoothed surrogate —
matching the change-point CV literature (Pein and Shah 2025 recommend absolute/raw check loss). The
model's internal logistic smoothing (`smooth_T = base_scale*eps_rel`) stays where it belongs: inside
the *fitting* likelihood only. Concretely:
- Add `pinball_loss_pointwise(y_val, qhat, taus)` in cv_copss.R = the existing raw check loss but
  returned **per observation** (mean over `taus` only, length-`n_val` vector) for the per-time-point
  chart. `loss_i = mean_q (u_iq * (tau_q - I(u_iq < 0)))`, `u = y - qhat`. No smoothing.
- **No `pinball_loss_smooth`.** The existing `pinball_loss` (scalar) and the existing
  `cv_copss_map/grid/mcmc` scoring stay **unchanged** (raw) — this reverses the earlier smoothing
  change, so the existing CV output is untouched.

## 5. Component (5) — smoke test (`tests/smoke_cvloss.R`, run with Rscript)

- Small synthetic series (`n≈120`, one sustained shift), `taus=c(.1,.5,.9)`, `H=getSustainedShift`.
- **New path runs clean:** grid CV → pick `lambda_nc` → `cv_holdout_pointwise` (full) +
  `_reduced` → `detectChangepoints_cvloss`; assert output structure + dims, `D`, `c_alpha`, flag.
- **Old path not broken (regression):** on a fixed seed, `getModel` → `detectChangepoints_gamma`
  → `getQSS` → `plotBQQ.*` produce the **same** key outputs as before the change (compare a saved
  reference). New functions must not alter any existing signature or output.

## 6. Component (6) — mimic the simulation and run TWO series (after smoke test)

Reproduce the `simulation_study` data-generating process and run the **new** pipeline end-to-end on
**two** series:
- **(a) no-shift (null):** confirm the monitor does **not** flag (and that the realized false-alarm
  behavior is sensible — the one-time size check).
- **(b) with a shift:** confirm the monitor **flags** and **localizes** the change near the true point.

Report `D`, `c_alpha`, the flag, and `hat{changepoint}` for both.

## 7. Component (7) — visualize with the existing methods  [decision: (A) only]

Reuse the existing `plotBQQ.R` (no new heatmap). Alignment (A): the $\gamma$ **fill is unchanged**
(model-based); the CV-loss result supplies the **block-level significance** for the borders.
- `detectChangepoints_cvloss()` returns, in addition to `D`, `c_alpha`, `flag`, `p`, `changepoint`:
  a per-block vector `D_block` (`= D_j`), a length-`r` logical `sig_block = (D_block > c_alpha)`, and a
  `detected_blocks`-compatible frame (`obs_start`, `signal_obs`, `significant`) for the flagged blocks,
  with the argmax block `hat_j` as the primary localized change.
- `plotGammaHeatmap()` gains an optional `sig_block` argument: when supplied, border the **whole
  column** of each block with `sig_block == TRUE` (instead of per-cell borders). Fill stays `gamma`.
  Border rule: **all** blocks with `D_j > c_alpha` (family-wise valid, mirrors the current per-cell
  behavior); `hat_j` is the primary localized change.
- `plotQuantileProcess()` / `plotQSSProcess()` overlay onset lines + the localized point via the same
  `.bqq_sig_blocks()` path — so the CV-loss result's `detected_blocks` frame must expose
  `obs_start` / `signal_obs`, letting those two plots work unchanged.
- **No companion predictive-gain heatmap (B dropped).**

---

## Component (4b) — real Laplace only, no heuristic draws  [decision 4]

Package-wide, draws must always be the **real Hessian-based Laplace** (Tier 1). Remove the Tier-2
heuristic-perturbation fallback in `getLaplaceSamples()`: if the Hessian is `NULL` or its inversion
fails, **error** with a clear message instead of silently returning perturbation draws. Because Tier 1
already uses the Moore-Penrose pseudo-inverse (`MASS::ginv`, getInference.R:109), which is robust to the
flat zero-curvature directions of unused-prior latents, genuine inversion failures are rare and subtle
— so this error path is a **near-dead edge case**, and removing the heuristic is low-impact. All
draw-using paths fit with `map_hessian = TRUE` (already the `getModel` default) and `n_draws = 1000`
**[decision 3]**. *Behavior change (low-impact):* a would-be heuristic fallback now errors instead of
degrading.

## Files touched / added
- `R/cv_copss.R`: `+ pinball_loss_pointwise` (raw, per-observation); `+ cv_holdout_pointwise`;
  `+ cv_holdout_pointwise_reduced`. Existing `pinball_loss` and `cv_copss_map/grid/mcmc` **unchanged**.
- `R/getInference.R`: `+ detectChangepoints_cvloss` (returns `D`, `c_alpha`, `flag`, `p`,
  `changepoint`, `D_block`, `sig_block`, `detected_blocks`); **remove the Tier-2 heuristic** from
  `getLaplaceSamples` (error instead) [decision 4].
- `R/plotBQQ.R`: `plotGammaHeatmap()` gains an **optional** `sig_block` arg (whole-column borders when
  supplied); backward-compatible (NULL → current per-cell behavior). `plotQuantileProcess` /
  `plotQSSProcess` unchanged — they consume the CV-loss `detected_blocks` via `.bqq_sig_blocks`.
- `NAMESPACE` + Roxygen: export the new user-facing functions (`devtools::document()`).
- `tests/smoke_cvloss.R` (new).
- **Unchanged behavior:** `getModel.R`, `getDesignMatrix.R`, all `cv_copss_*`, and
  `detectChangepoints_gamma`. The **only** intended existing-behavior change is the real-Laplace-only
  edit to `getLaplaceSamples` (low-impact; see 4b), which the smoke-test regression baseline must reflect.

## Resolved decisions
1. **Reduced draws:** always a 1-column `getModel(H=NULL)` fit (no closed-form branch even when `X` NULL).
2. **Loss:** **raw** check (pinball) loss for scoring and charting, everywhere — aligning with the
   change-point CV literature (Pein and Shah 2025). No smoothed surrogate; the model's `smooth_T` stays
   inside the fitting likelihood only. Existing `cv_copss_*` untouched.
3. **Defaults:** `n_draws = 1000`, `n_pairs = 1000`.
4. **Laplace:** real Hessian-based only (`MASS::ginv`), no heuristic fallback; error if the Hessian
   truly fails (rare, given the pseudo-inverse).
5. **Reduced fit:** separate fit; its regularization uses the **same beta hyperparameters** as the full
   fit (`prior_beta`, `adaptive_beta`, `lambda_beta2_*`).

## Order of work
1. `pinball_loss_smooth` (verify against the Stan `psi`/`smooth_T`).
2. `cv_holdout_pointwise` (reuse `getEta`).
3. `cv_holdout_pointwise_reduced`.
4. `detectChangepoints_cvloss`.
5. Roxygen + NAMESPACE.
6. Smoke test (new clean + old regression).
7. Mini-sim run + existing-viz.
