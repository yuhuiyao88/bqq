# --- Session-level cache for compiled Stan model ---
# The Stan code is a fixed string; only stan_data changes between calls.
# Compiling once per R session avoids redundant 30-60s compilations.
.bqq_stan_cache <- new.env(parent = emptyenv())


# ====================================================================
# EM for the squared interquantile penalty weight, lambda_iq2
# ====================================================================
# Derivation: bqq/simulation_study/lmom_3cfg/EM_FOR_LAMBDA_IQ_math.md
#
# The Stan model subtracts  lambda * P  from the target, where
# lambda = sqrt(lambda_iq2) and P = pen_iq_total is the NORMALIZED weighted
# penalty actually accumulated in the model block:
#
#   P = (1/n_comp) * [ (1/(r(m-1))) sum_{j,q} w^g_{q-1,j} |gamma[q,j]-gamma[q-1,j]|
#                    + (1/(p_s(m-1))) sum_{j,q} w^b_{q-1,j} |beta[q,j+1]-beta[q-1,j+1]| ]
#
# Writing the contribution as -lambda * sum_k c_k |d_k| over the N differences,
# the Laplace rate of difference k is a_k = lambda * c_k. Reading the fused factor
# as a Laplace prior on the differences and applying the Gaussian scale-mixture
# augmentation d_k | t_k ~ N(0, t_k), t_k | lambda ~ Exp(a_k^2 / 2) gives the M-step
#
#   lambda^2 = 2N / sum_k c_k^2 E[t_k],     E[t_k] = E|d_k| / a_k + 1 / a_k^2
#
# and, since sum_k c_k^2 E[t_k] = Sbar / lambda^(s) + N / (lambda^(s))^2 with
# Sbar := sum_k c_k E|d_k| = E[P], the closed-form recursion on the SQUARED scale:
#
#   lambda_iq2^(s+1) = 2 N lambda_iq2^(s) / ( sqrt(lambda_iq2^(s)) * Sbar + N )
#
# whose fixed point is lambda* = N / Sbar, i.e. lambda_iq2* = (N / Sbar)^2.
#
# Sbar is therefore just the posterior mean of the very quantity the Stan model
# computes, which is what .bqq_iq_Sbar() evaluates from the draws.
#
# IMPORTANT (see EM_FOR_LAMBDA_IQ_math.md sections 4 and 7): the scale-mixture
# augmentation is exact, but this E-step uses a Laplace approximation to
# p(gamma | y, lambda), not the exact conditional. The procedure is therefore an
# APPROXIMATE empirical-Bayes EM and carries no monotone-ascent guarantee.

# Number of penalized adjacent-quantile differences (the N above).
.bqq_iq_n_diff <- function(r, p_slope, m) {
  if (m < 2L) return(0L)
  as.integer((if (r > 0) r * (m - 1L) else 0L) +
             (if (p_slope > 0) p_slope * (m - 1L) else 0L))
}

# Posterior mean of pen_iq_total, i.e. Sbar = sum_k c_k E|d_k|.
# draws$gamma is [S, m, r]; draws$beta is [S, m, p_total] with column 1 the
# intercept, which is NOT penalized (mirrors the Stan model block exactly).
.bqq_iq_Sbar <- function(draws, w_iq_gamma, w_iq_beta, r, p_slope, m) {
  if (m < 2L) return(NA_real_)
  qs <- 2:m
  qm <- 1:(m - 1L)
  total <- 0
  n_comp <- 0L

  if (r > 0 && !is.null(draws$gamma)) {
    g <- draws$gamma
    dg <- abs(g[, qs, , drop = FALSE] - g[, qm, , drop = FALSE])   # [S, m-1, r]
    Ed <- apply(dg, c(2, 3), mean)                                  # [m-1, r]
    total <- total + sum(w_iq_gamma * Ed) / (r * (m - 1L))
    n_comp <- n_comp + 1L
  }

  if (p_slope > 0 && !is.null(draws$beta)) {
    b <- draws$beta[, , 2:(p_slope + 1L), drop = FALSE]             # drop intercept
    db <- abs(b[, qs, , drop = FALSE] - b[, qm, , drop = FALSE])
    Ed <- apply(db, c(2, 3), mean)
    total <- total + sum(w_iq_beta * Ed) / (p_slope * (m - 1L))
    n_comp <- n_comp + 1L
  }

  if (n_comp == 0L) return(NA_real_)
  total / n_comp
}

# One update of lambda_iq2.
#   "em"         : the EM recursion above (monotone in theory for an exact E-step).
#   "fixedpoint" : jump straight to the fixed point (N/Sbar)^2. Same limit, far
#                  fewer refits, but it is NOT the EM iteration.
# M-step for lambda_iq2.  The EM recursion is
#     lambda2_{s+1} = 2 N lambda2_s / (lambda_s Sbar + N),
# whose fixed point solves lambda Sbar = N, i.e. lambda2 = (N / Sbar)^2.  We jump
# straight there.  Both routes converge to the SAME value from any start (checked
# from 1e-4, 1 and 1e6), but the creeping form needs ~44 refits where this needs 1,
# and each refit is a full model fit.  The creeping form's only advantage is
# monotone ascent under an EXACT E-step, and this E-step is Monte-Carlo from
# Laplace draws, so that guarantee does not hold here either.  Removed in 0.5.2.
#
# Note the OUTER loop still iterates: Sbar is recomputed from a refit at the new
# lambda, so this is a fixed-point iteration on Sbar(lambda), not a one-step solve.
.bqq_iq_em_step <- function(lambda_iq2, Sbar, N) {
  if (!is.finite(Sbar) || !is.finite(lambda_iq2) || N <= 0) return(NA_real_)
  if (Sbar <= 0) return(NA_real_)          # no fusion signal; caller keeps current value
  (N / Sbar)^2
}

# Posterior draws of beta/gamma from whichever machinery the fit_method used.
# Laplace draws (fit_method = "map") and MCMC draws share the [S, m, .] layout.
.bqq_iq_draws <- function(res) {
  ls <- res$laplace_samples
  if (!is.null(ls) && (!is.null(ls$gamma) || !is.null(ls$beta))) {
    return(list(beta = ls$beta, gamma = ls$gamma))
  }
  if (!is.null(res$fit)) {
    d <- tryCatch(rstan::extract(res$fit, pars = c("beta", "gamma")),
                  error = function(e) NULL)
    if (!is.null(d)) return(list(beta = d$beta, gamma = d$gamma))
  }
  NULL
}

#' Smoothed Quantile Regression with Interquantile Shrinkage (Stan)
#'
#' Fits a multi-quantile (\eqn{m}) regression model where the conditional
#' quantile function is modeled with a per-quantile intercept (in time or index)
#' with optional fixed effects \eqn{X} and structured effects \eqn{H}.
#' The \eqn{H}-coefficients are shrunk via **Bayesian LASSO-type priors**
#' (group, elementwise, heterogeneous-group, or adaptive), and adjacent quantiles
#' are softly penalized to discourage crossings. **Interquantile shrinkage**
#' stabilizes outer quantiles by penalizing differences between adjacent quantile
#' coefficients.
#'
#' @section Model (high level):
#' \describe{
#'   \item{Data & design}{
#'     \itemize{
#'       \item \eqn{y_i} is optionally jittered (\code{u ~ Beta(1,1)}) and/or log-transformed.
#'       \item Combined linear predictor
#'       \eqn{\eta_{qi} = \beta_{0,q} + x_i^\top \beta_{X,q} + h_i^\top \gamma_q + \mathrm{offset}_i}.
#'       \item \eqn{\beta_{0,q}} is a per-quantile intercept whose default prior is
#'       centered at the warm-up-window empirical quantiles with unit-information
#'       scale (Kass & Wasserman 1995) — equivalently, a power prior on the warm-up
#'       window with discount \eqn{a_0 = 1/w} (Ibrahim & Chen 2000).
#'     }
#'   }
#'   \item{Interquantile shrinkage}{
#'     Penalizes differences between adjacent quantile coefficients for gamma and beta slopes
#'     using data-driven adaptive weights from pilot quantile regression estimates:
#'     \eqn{\text{pen}_{\text{IQ}} = \sum_{q=2}^m \sum_j w_{q,j} |\theta_{q,j} - \theta_{q-1,j}|}
#'     where \eqn{w_{q,j} = (|\tilde{\theta}_{q,j} - \tilde{\theta}_{q-1,j}| + \epsilon_w)^{-1}}
#'     and \eqn{\tilde{\theta}} are pilot estimates from separate quantile regressions
#'     (Jiang, Wang, & Bondell 2013).
#'     Weights are applied separately to gamma and beta slopes.
#'     Falls back to uniform weights (all 1) when \pkg{quantreg} is not available.
#'     Note: Intercept (beta0) is NOT penalized (per Jiang, Wang, & Bondell 2013).
#'   }
#'   \item{Non-crossing penalty}{
#'     Adds an L1 hinge on the finite-difference derivative in \eqn{\tau},
#'     scaled by \code{lambda_nc}.
#'   }
#' }
#'
#' @param y Numeric vector of responses of length \eqn{n}.
#' @param taus Numeric vector of target quantile levels in \eqn{(0,1)}, length \eqn{m}.
#' @param H Numeric matrix \eqn{n \times r} of structured predictors for group-lasso
#'   coefficients \eqn{\gamma}. If \eqn{r = 0}, pass a zero-column matrix.
#' @param w Integer \eqn{\ge 1}. Used for initial quantile estimation from first w observations.
#' @param X Optional numeric matrix \eqn{n \times p_x} of user-supplied covariates.
#'   An intercept column is added internally and assigned its own weakly informative prior.
#' @param offset Optional numeric vector of length \eqn{n} added to the linear predictor.
#' @param alpha Deprecated scalar retained for backward compatibility. It is no
#'   longer used in the \code{adaptive_lasso} or \code{het_group_lasso} prior
#'   construction.
#' @param eps_w Positive scalar FLOOR on the pilot adjacent-difference used in the
#'   adaptive IQ weight. Appendix C / manuscript Eq. (15) define
#'   w_{q,j} = |gamma^pilot_{q,j} - gamma^pilot_{q-1,j}|^{-1}; this floor reproduces
#'   that exactly whenever the pilot difference is at least eps_w, and only caps the
#'   weight at 1/eps_w for degenerate (near-zero) differences. Before 0.6.1 the code
#'   used (diff + eps_w)^{-1}, which perturbed EVERY weight rather than only the
#'   degenerate ones
#'   in the IQ shrinkage weights (default 1e-6).
#' @param c_sigma Positive scalar scaling factor for the base scale (default 1.0).
#' @param base_scale Optional positive scalar smoothing bandwidth. Defaults to the
#'   Fernandes, Guerre & Horta (2021) rule-of-thumb
#'   \code{1.06 * s * length(y)^(-1/5)} with \code{s = min(sd, IQR/1.38898)} of the
#'   standard median (tau = 0.5) regression residuals, fitted on the ordinary
#'   predictors [intercept | X] only (the change-point design H is excluded). Used
#'   as the smoothing temperature \code{smooth_T = base_scale}; it does NOT set
#'   \code{beta0_scale}.
#' @param beta0_loc Prior location for the per-quantile intercept \code{beta0};
#'   \code{NULL} (default) or a length-\code{m} numeric vector on the modeling
#'   scale of \code{y}. When \code{NULL}, set to the empirical \code{taus}-quantiles
#'   of the warm-up period \code{y_model[1:w]} — the empirical-quantile anchoring
#'   of the tau-specific intercept of Yang and He (2012, Annals of Statistics).
#' @param beta0_scale Prior std dev for the per-quantile intercept \code{beta0};
#'   \code{NULL} (default), a positive scalar (recycled), or a length-\code{m}
#'   vector, on the modeling scale of \code{y} (log scale when \code{log_flag = 1}).
#'   When \code{NULL}, set to the unit-information prior of Kass and Wasserman
#'   (1995): \code{sqrt(taus * (1 - taus)) / f_hat}, with \code{f_hat} a kernel
#'   density estimate of the warm-up period evaluated at \code{beta0_loc}, so the
#'   prior carries the information of a single warm-up observation about each
#'   quantile. Together with the default \code{beta0_loc} this equals the power
#'   prior on the warm-up period with discount \code{a0 = 1/w} (Ibrahim and Chen
#'   2000; Bourazas, Kiagias and Tsiamyrtzis 2022). The prior is
#'   \code{beta0[q] ~ Normal(beta0_loc[q], beta0_scale[q])}.
#' @param beta_sd Positive scalar prior std dev for \code{betaX} coefficients under
#'   \code{prior_beta = "normal"} (default 1.0).
#' @param lambda_nc Positive scalar weight for the non-crossing penalty (larger is stricter).
#' @param lambda_iq2 Non-negative scalar, the \strong{squared} interquantile (IQ)
#'   shrinkage weight \eqn{\lambda_{iq}^2}. The penalty applied to the target is
#'   \eqn{\sqrt{\lambda_{iq}^2}\sum_q |\gamma_q - \gamma_{q-1}|}, i.e. the effective
#'   L1 fusion rate is the \emph{square root} of this argument. This matches the
#'   \code{lambda_lasso2} / \code{lambda_beta2} convention, where the stored quantity
#'   is \eqn{\lambda^2} and the Laplace rate is \eqn{\lambda}. Larger = stronger
#'   fusion; 0 = none. When \code{adaptive_iq = FALSE} this is the fixed value used;
#'   when \code{adaptive_iq = TRUE} it is the \emph{starting value} of the EM
#'   recursion and must be strictly positive (default 1).
#' @param adaptive_iq Logical; if TRUE (default), \eqn{\lambda_{iq}^2} is learned from
#'   the data by an EM recursion run \emph{between} refits, in the same spirit as
#'   \code{adaptive_gamma} / \code{adaptive_beta} learn \eqn{\lambda^2} for the
#'   coefficients. If FALSE, the supplied \code{lambda_iq2} is used as a fixed value
#'   and the model is fitted once.
#'
#'   Note that \eqn{\lambda_{iq}^2} cannot be learned \emph{inside} the fit the way
#'   the coefficient-side rates are: the fused factor enters the target as a bare
#'   penalty without its \eqn{N\log\lambda} normalizer, so its joint-MAP mode is
#'   degenerate and collapses to 0 under \code{fit_method = "map"}. The EM route
#'   restores the normalizer by working with the marginal likelihood, so it does not
#'   suffer that collapse. Each EM iteration is a \strong{full refit}, so
#'   \code{adaptive_iq = TRUE} costs up to \code{iq_em_max_iter} times a single fit.
#' @param iq_em_max_iter Maximum number of EM iterations (refits) when
#'   \code{adaptive_iq = TRUE} (default 60). Since 0.5.2 the M-step jumps straight
#'   to its fixed point, so a converging run typically needs only a handful of
#'   refits; the outer loop still iterates because \eqn{\bar S} is recomputed from
#'   a refit at the updated \eqn{\lambda}.
#' @param iq_em_tol Relative-change tolerance on \eqn{\lambda_{iq}^2} for declaring EM
#'   convergence (default 1e-3).
#' @param iq_em_mc_tol Monte-Carlo noise-floor tolerance (default 0.02). The E-step
#'   estimates \eqn{\bar S} from a finite set of Laplace draws, so
#'   \eqn{\lambda_{iq}^2} cannot settle more tightly than the sampling noise in
#'   \eqn{\bar S}: past that point successive updates simply bounce up and down about
#'   the fixed point instead of shrinking. When the update has changed direction on
#'   two consecutive iterations and every relative move involved is below
#'   \code{iq_em_mc_tol}, the recursion is declared converged \emph{at the Monte-Carlo
#'   floor} and stops, rather than burning the remaining refits chasing a tolerance
#'   the E-step cannot deliver. This is recorded in \code{iq_em$note} and
#'   \code{iq_em$stop_reason}. Tighten the floor by raising
#'   \code{laplace_n_samples}, not by lowering \code{iq_em_tol}. Set to 0 to disable.

#' @param adaptive_beta Logical; if TRUE (default), the beta-side shrinkage level
#'   \eqn{\lambda_\beta^2} is learned from data for LASSO-type priors. If FALSE,
#'   \code{lambda_beta2_fixed} is used.
#' @param lambda_beta2_a,lambda_beta2_b Positive shape/rate hyperparameters for the
#'   beta-side LASSO-type shrinkage hierarchy.
#' @param lambda_beta2_fixed Positive scalar; fixed value for the beta-side shrinkage
#'   level \eqn{\lambda_\beta^2} when \code{adaptive_beta = FALSE} (default 1).
#' @param lambda_lasso2_a,lambda_lasso2_b Positive shape/rate hyperparameters for the
#'   LASSO-type shrinkage hierarchy. Their exact role depends on \code{prior_gamma}:
#'   they govern the global \eqn{\lambda^2} prior for \code{"lasso"}, \code{"group_lasso"},
#'   and \code{"het_group_lasso"}, and the global hyperprior driving the local
#'   coefficient-specific shrinkage in \code{"adaptive_lasso"}.
#' @param adaptive_gamma Logical; if TRUE (default), the global shrinkage level
#'   \eqn{\lambda^2} is learned from data via a Gamma prior. If FALSE, the fixed
#'   value \code{lambda_lasso2_fixed} is used.
#' @param lambda_lasso2_fixed Positive scalar; fixed value for the global shrinkage
#'   level \eqn{\lambda^2} when \code{adaptive_gamma = FALSE} (default 1).
#' @param log_flag Integer \code{0/1}. If 1, fit on \code{log(y)} (or
#'   \code{log(y + u)} with jittering). All data-driven inputs -- the \code{beta0}
#'   prior location, the \code{base_scale} bandwidth, and the pilot fits behind the
#'   adaptive IQ weights -- are then computed on the log scale as well (jitter
#'   approximated by its mean 0.5), so user-supplied \code{beta0_scale} /
#'   \code{base_scale} must also be given on the log scale.
#' @param jittering Integer \code{0/1}. If 1, add \eqn{u \sim \mathrm{Beta}(1,1)} to \eqn{y}.
#' @param chains Number of MCMC chains.
#' @param iter Total iterations per chain.
#' @param warmup Warmup iterations per chain.
#' @param control Optional list passed to \code{rstan::sampling()}.
#' @param seed RNG seed.
#' @param verbose show the log.
#' @param map_hessian Logical; if \code{TRUE} (default), compute the Hessian in the
#'   MAP step. The Hessian enables the standard Laplace approximation for posterior
#'   sampling (see \code{\link{getLaplaceSamples}}). When \code{FALSE}, the Hessian
#'   is not computed and posterior samples are generated using a heuristic perturbation
#'   fallback instead.
#' @param map_tol_obj,map_tol_grad,map_tol_rel_grad,map_tol_rel_obj,map_tol_param
#'   MAP optimizer (L-BFGS) tolerances. Defaults are deliberately tight
#'   (\code{map_tol_rel_grad = 1e2}, \code{map_tol_rel_obj = 1e2}, with
#'   \code{map_iter = 10000}): the smoothed-score objective has near-flat plateaus
#'   (see \code{map_init}), and looser relative tolerances (e.g., rstan's default
#'   \code{tol_rel_obj = 1e4}) can terminate there prematurely while reporting
#'   convergence, leaving outer-quantile coefficients stranded far from the optimum.
#' @param map_history_size L-BFGS history size (default 25; rstan's default is 5).
#'   A larger history improves the curvature approximation on this ridge-shaped
#'   objective, typically reducing the iteration count without loosening any
#'   stopping tolerance.
#' @param map_iter Maximum iterations for MAP optimization.
#' @param map_init Initialization for MAP optimization. \code{"pilot"} (default)
#'   fits a marginal LASSO quantile regression at each quantile level on
#'   \code{[1 | X | H]} (\code{quantreg::rq.fit.lasso}; intercept unpenalized,
#'   slope penalty at the \code{sqrt(tau(1-tau) n log d)} rate) and uses those
#'   estimates as the starting coefficients (clamped to a generous data range).
#'   The sparse, data-proximal start begins where the smoothed score has live
#'   gradients, matches the spike-slab prior's sparsity, and typically reduces
#'   the iteration count. It falls back to \code{"prior_center"} (with a
#'   warning) when the pilot fits are unavailable (no \code{X}/\code{H} columns,
#'   or \pkg{quantreg} missing).
#'   \code{"prior_center"} starts at the prior mode -- \code{beta0 = beta0_loc},
#'   \code{betaX = 0}, \code{gamma = 0}, hierarchy scales at 1 -- i.e., the
#'   in-control state. Mode-selection nuance: \code{"pilot"} approaches the
#'   optimum from the data-fitting side while \code{"prior_center"} grows the
#'   solution from the null; for multimodal spike-slab posteriors the two can
#'   select different modes, so compare the final log-posterior
#'   (\code{$map$value}) when in doubt. \code{"random"} restores rstan's default
#'   random initialization. Background: the logistic score saturates (clamped at
#'   |z| = 20) once a curve is ~20 bandwidths away from the data, leaving zero
#'   gradient; random inits can start (or wander) into such flat regions and
#'   produce runaway MAP modes.
#' @param fit_method One of "mcmc", "map_mcmc", or "map":
#'   \itemize{
#'     \item "mcmc": Estimators are posterior median from MCMC; posterior draws from MCMC.
#'     \item "map_mcmc": Estimators are MAP; posterior draws from MCMC (MAP used as init).
#'     \item "map": Estimators are MAP; posterior draws from Laplacian approximation.
#'   }
#' @param laplace_n_samples Number of samples for Laplacian approximation (when fit_method = "map").
#' @param laplace_noise_scale Scale factor for parameter perturbation in Laplacian approximation.
#' @param prior_beta Prior type for \code{betaX}: \code{"normal"}, \code{"lasso"},
#'   \code{"spike_slab"}, \code{"group_lasso"}, \code{"het_group_lasso"}, or
#'   \code{"adaptive_lasso"}. The intercept \code{beta0} always retains its own
#'   normal prior \code{beta0[q] ~ Normal(quantile(y[1:w], tau_q), beta0_scale[q])}.
#' @param prior_gamma Prior type for gamma: \code{"spike_slab"} (default),
#'   \code{"group_lasso"}, \code{"lasso"}, \code{"het_group_lasso"}, \code{"adaptive_lasso"}, or
#'   \code{"spike_slab_lasso"} (a continuous Laplace spike + Laplace slab mixture,
#'   Rockova & George 2018; reuses \code{spike_sd}/\code{slab_sd} as Laplace scales).
#'   The \code{"adaptive_lasso"} option follows a Leng et al. (2014)-style hierarchy
#'   with coefficient-specific local shrinkage parameters. The
#'   \code{"het_group_lasso"} option combines group-level shrinkage with
#'   coefficient-level local scales, without pilot weights.
#' @param beta_spike_sd,beta_slab_sd,beta_slab_pi_a,beta_slab_pi_b Spike-and-slab
#'   hyperparameters for \code{betaX}.
#' @param spike_sd,slab_sd,slab_pi_a,slab_pi_b Spike-and-slab hyperparameters.
#'   NOTE: \code{spike_sd}/\code{slab_sd} are STANDARD DEVIATIONS; the
#'   manuscript's spike-and-slab specification is written in variances
#'   (variance = sd^2).
#'
#'   \code{spike_sd} defaults to \strong{0.1} (raised from 0.05 in 0.5.2). A
#'   narrower spike makes the null mixture component close to a point mass, so
#'   any noise-driven \code{gamma} is pushed into the slab and flagged. In the
#'   lmom_3cfg simulation this dominated the false-alarm rate: on the null arm,
#'   \code{spike_sd <= 0.05} flagged 71 of 155 replications (0.458) against 1 of
#'   85 (0.012) for \code{spike_sd >= 0.10} -- Fisher exact p = 6.5e-16, odds
#'   ratio 70. Values below 0.1 are still accepted but will inflate the
#'   false-alarm rate of \code{prior_gamma = "spike_slab"} and
#'   \code{"spike_slab_lasso"}; the effect is not removable by any multiplicity
#'   correction, since those detections survive Bonferroni, Holm, BH and the
#'   calibrated constant identically.
#'
#'   \code{beta_spike_sd} was raised to 0.1 alongside it, for consistency of the
#'   spike width across the two coefficient blocks. Note the direct evidence
#'   above concerns \code{spike_sd} only: it governs the block coefficients that
#'   drive detection, whereas \code{beta_spike_sd} governs \code{betaX}, and no
#'   equivalent false-alarm study was run for the covariate side.
#'
#' @return A list with components:
#'   \itemize{
#'     \item fit: stanfit object (NULL if fit_method = "map")
#'     \item map: MAP estimates (contains $par with parameter values)
#'     \item y, H, X: Input data and design matrices
#'     \item hessian: Hessian at MAP (if computed)
#'     \item fit_method: The estimation method used
#'     \item laplace_samples: Pre-generated Laplacian samples (if fit_method = "map")
#'     \item stan_data: The data list passed to Stan. Its \code{lambda_iq2} entry is
#'       the value the returned fit was actually computed at.
#'     \item iq_em: Diagnostics for the \eqn{\lambda_{iq}^2} EM: \code{adaptive},
#'       \code{update}, \code{n_diff} (the number of penalized differences \eqn{N}),
#'       \code{lambda_iq2} (final, matching the returned fit), \code{lambda_iq2_next}
#'       (the update the last iteration proposed), \code{Sbar}, \code{converged},
#'       \code{n_iter}, \code{note}, and \code{trace} -- a data frame with one row per
#'       EM iteration recording \code{lambda_iq2}, \code{lambda_iq}, \code{Sbar},
#'       \code{lambda_iq2_next} and \code{rel_change}.
#'   }
#'
#' @references
#' Jiang, L., Wang, H. J., & Bondell, H. D. (2013). Interquantile Shrinkage in Regression Models.
#' Journal of Computational and Graphical Statistics, 22(4), 970-986.
#'
#' Leng, C., Tran, M. N., & Nott, D. (2014). Bayesian adaptive lasso.
#' Annals of the Institute of Statistical Mathematics, 66(2), 221-244.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' }
#'
#' @importFrom rstan stan_model sampling stan
#' @importFrom stats lm resid sd rnorm quantile mad median coef cov pchisq qchisq p.adjust
#' @importFrom graphics plot lines points abline polygon legend par barplot
#' @importFrom grDevices rgb rainbow
#' @importFrom MASS ginv
#' @export

getModel <- function(y, taus, H = NULL, X = NULL, offset = NULL, w = 0,
                        alpha = 0.75, eps_w = 1e-6, c_sigma = 1.0,
                        base_scale = NULL, beta0_loc = NULL, beta0_scale = NULL,
                        beta_sd = 1.0,
                        lambda_nc = 2,
                        lambda_iq2 = 1,
                        adaptive_iq = TRUE,
                        iq_em_max_iter = 60,
                        iq_em_tol = 1e-3,
                        iq_em_mc_tol = 0.02,
                        adaptive_beta = TRUE,
                        lambda_beta2_a = 1, lambda_beta2_b = 0.05,
                        lambda_beta2_fixed = 1,
                        adaptive_gamma = TRUE,
                        lambda_lasso2_a = 1, lambda_lasso2_b = 0.05,
                        lambda_lasso2_fixed = 1,
                        log_flag = 0, jittering = 0,
                        chains = 1, iter = 1500, warmup = 500,
                        control = list(adapt_delta = 0.99),
                        seed = 123, verbose = FALSE,
                        map_hessian = TRUE,
                        map_init = c("pilot", "prior_center", "random"),
                        map_tol_obj = 1e-12, map_tol_grad = 1e-8,
                        map_tol_rel_grad = 1e2, map_tol_param = 1e-8,
                        map_tol_rel_obj = 1e2,
                        map_iter = 10000,
                        map_history_size = 25,
                        fit_method = c("mcmc", "map_mcmc", "map"),
                        laplace_n_samples = 1000,
                        laplace_noise_scale = 0.1,
                        prior_beta = c("normal", "lasso", "spike_slab",
                                       "group_lasso", "het_group_lasso", "adaptive_lasso"),
                        prior_gamma = c("spike_slab", "group_lasso", "lasso",
                                        "het_group_lasso", "adaptive_lasso",
                                        "spike_slab_lasso"),
                        beta_spike_sd = 0.1, beta_slab_sd = 2.0,
                        beta_slab_pi_a = 1, beta_slab_pi_b = 1,
                        spike_sd = 0.1, slab_sd = 2.0,
                        slab_pi_a = 1, slab_pi_b = 1) {

  prior_beta   <- match.arg(prior_beta)
  prior_gamma  <- match.arg(prior_gamma)
  fit_method   <- match.arg(fit_method)
  map_init     <- match.arg(map_init)

  if (!is.numeric(lambda_iq2) || length(lambda_iq2) != 1L ||
      is.na(lambda_iq2) || lambda_iq2 < 0) {
    stop("lambda_iq2 must be a single non-negative scalar (it is lambda^2, not lambda).")
  }
  if (isTRUE(adaptive_iq) && lambda_iq2 <= 0) {
    stop("lambda_iq2 must be > 0 when adaptive_iq = TRUE: it is the EM starting value, ",
         "and the recursion has an absorbing fixed point at 0.")
  }
  prior_beta_code <- switch(
    prior_beta,
    normal            = 1L,
    lasso             = 2L,
    spike_slab        = 3L,
    group_lasso       = 4L,
    het_group_lasso   = 5L,
    adaptive_lasso    = 6L
  )
  prior_code <- switch(
    prior_gamma,
    group_lasso        = 1L,
    lasso              = 2L,
    spike_slab         = 3L,
    het_group_lasso    = 4L,
    adaptive_lasso     = 5L,
    spike_slab_lasso   = 6L
  )

  safe_pilot_coefs <- function(y, Z, tau, eps_w = 1e-3, lambda_lasso = NULL) {
    d <- ncol(Z)
    if (d == 0) return(numeric(0))

    fit_q <- try(
      suppressWarnings(quantreg::rq(y ~ Z - 1, tau = tau, method = "fn")),
      silent = TRUE
    )

    if (inherits(fit_q, "try-error")) {
      n <- length(y)
      if (is.null(lambda_lasso)) {
        lambda_lasso <- sqrt(log(d + 1L) / n)
      }
      fit_q <- try(
        suppressWarnings(
          quantreg::rq(y ~ Z - 1, tau = tau,
                       method = "lasso", lambda = lambda_lasso)
        ),
        silent = TRUE
      )
      if (inherits(fit_q, "try-error")) {
        warning("Both rq(method = 'fn') and rq(method = 'lasso') failed at tau=",
                tau, "; using zero pilot coefficients.")
        return(rep(0, d))
      }
    }

    coef_hat <- as.numeric(stats::coef(fit_q))
    if (length(coef_hat) != d) {
      warning("Pilot fit returned length mismatch at tau=", tau,
              "; using zero pilot coefficients.")
      return(rep(0, d))
    }
    coef_hat
  }

  n <- length(y)

  stan_code <- "
  data {
      int<lower=1> n;                  // observations
      int<lower=0> px;                 // user predictors in eta (X), excluding intercept
      int<lower=2> m;                  // quantiles
      int<lower=0> r;                  // predictors in eta (H)

      matrix[n, px] X;                 // n x px
      matrix[n, r] H;                  // n x r
      vector[n] y;
      vector[n] offset;
      vector[m] tau_q;
      vector[m] beta0_loc;
      vector<lower=0>[m] beta0_scale;

      real<lower=1e-12> base_scale;
      real<lower=0>      c_sigma;
      real<lower=0>      beta_sd;

      real<lower=0> lambda_nc;         // non-crossing penalty weight

      // Interquantile (fused lasso) shrinkage: SQUARED penalty weight.
      // Convention matches lambda_lasso2 / lambda_beta2: the stored quantity is
      // lambda^2 and the Laplace rate applied to |gamma[q]-gamma[q-1]| is its
      // square root. Held fixed within a fit; when adaptive_iq = 1 the value is
      // updated between fits by the EM recursion in getModel() (see .bqq_iq_em_step).
      real<lower=0> lambda_iq2;        // SQUARED L1 fusion weight on adjacent-quantile differences

      real<lower=0> lambda_lasso2_a;
      real<lower=0> lambda_lasso2_b;
      int<lower=0, upper=1> adaptive_gamma;      // 1 = data-adaptive, 0 = fixed
      real<lower=0> lambda_lasso2_fixed;         // fixed value when adaptive_gamma = 0

      real<lower=0> lambda_beta2_a;
      real<lower=0> lambda_beta2_b;
      int<lower=0, upper=1> adaptive_beta;       // 1 = data-adaptive, 0 = fixed
      real<lower=0> lambda_beta2_fixed;          // fixed value when adaptive_beta = 0

      real<lower=0, upper = 1> jittering;
      real<lower=0, upper = 1> log_flag;

      // prior selectors
      int<lower=1, upper=6> prior_beta_code;
      int<lower=1, upper=6> prior_code;

      // beta spike-and-slab hyperparameters
      real<lower=0> beta_spike_sd;
      real<lower=0> beta_slab_sd;
      real<lower=0> beta_slab_pi_a;
      real<lower=0> beta_slab_pi_b;

      // spike-and-slab hyperparameters
      real<lower=0> spike_sd;
      real<lower=0> slab_sd;
      real<lower=0> slab_pi_a;
      real<lower=0> slab_pi_b;

      // Data-driven IQ shrinkage weights (from pilot quantile regressions)
      int<lower=0> p_slope;
      matrix[m-1, r] w_iq_gamma;
      matrix[m-1, p_slope] w_iq_beta;
  }

  transformed data {
      // Quantile kernel Q[a,b] = min(tau_a, tau_b) - tau_a * tau_b
      matrix[m, m] Q;
      for (a in 1:m)
        for (b in 1:m)
          Q[a, b] = fmin(tau_q[a], tau_q[b]) - tau_q[a] * tau_q[b];

      // Combined X design with explicit intercept
      int p = px + 1;
      matrix[n, p] X_design;
      for (i in 1:n) {
        X_design[i, 1] = 1;
        if (px > 0) {
          for (j in 1:px)
            X_design[i, j + 1] = X[i, j];
        }
      }

      // Combined design Z = [X | H] (n x pr)
      int pr = p + r;
      matrix[n, pr] Z;
      {
        for (j in 1:p)
          for (i in 1:n)
            Z[i, j] = X_design[i, j];

        if (r > 0) {
          for (j in 1:r)
            for (i in 1:n)
              Z[i, p + j] = H[i, j];
        }
      }

      // Gram for score: Gs = Z'Z / n and its Cholesky
      matrix[pr, pr] Gs;
      matrix[pr, pr] L_Gs;
      if (pr > 0) {
        matrix[pr, n] Zt = Z';
        Gs = (Zt * Z) / n;
        for (k in 1:pr) Gs[k, k] = Gs[k, k] + 1e-8;  // tiny ridge
        L_Gs = cholesky_decompose(Gs);
      } else {
        Gs   = rep_matrix(0, 0, 0);
        L_Gs = rep_matrix(0, 0, 0);
      }

      // IQ shrinkage weights are now data-driven (passed as w_iq_gamma, w_iq_beta)
      // Computed from pilot quantile regression estimates in R
  }

  parameters {
      // Per-quantile intercept and user X-coefficients
      vector[m] beta0;
      matrix[m, px] betaX;

      // H-coefficients
      matrix[m, r] gamma;

      // Group-level scale for beta group lasso (one per X column)
      vector<lower=0>[px] sigma2_beta_group;

      // Element-wise local scales for beta lasso/adaptive lasso
      matrix<lower=0>[m, px] sigma2_beta;

      // Local adaptive shrinkage rates for beta adaptive lasso
      matrix<lower=0>[m, px] lambda2_beta_local;

      // Group-level scale for group lasso (one per H column)
      vector<lower=0>[r] sigma2_gamma_group;

      // Element-wise local scales for lasso/adaptive lasso
      matrix<lower=0>[m, r] sigma2_gamma;

      // Local adaptive shrinkage rates for adaptive lasso
      matrix<lower=0>[m, r] lambda2_gamma_local;

      // Global beta shrinkage rate (learned when adaptive_beta = 1)
      real<lower=0> lambda_beta2;

      // Global LASSO rate (learned when adaptive_gamma = 1)
      real<lower=0> lambda_lasso2;

      // Spike-and-slab mixing weights
      real<lower=0, upper=1> pi_slab_beta;
      real<lower=0, upper=1> pi_slab;

      // Group-level mixer for beta hetero group lasso
      vector<lower=0>[px] omega_beta_group;

      // Group-level mixer for hetero group lasso (Levy)
      // One per time block (consistent with group lasso grouping)
      vector<lower=0>[r] omega_group;

      // jitter variable
      vector<lower=1e-12, upper = 1>[n] u;
  }

  transformed parameters {
      matrix[m, px + 1] beta;
      for (q in 1:m) {
        beta[q, 1] = beta0[q];
        if (px > 0) {
          for (j in 1:px)
            beta[q, j + 1] = betaX[q, j];
        }
      }

      // Smoothing bandwidth (FGH rule-of-thumb; computed on the R side as base_scale)
      real<lower=1e-12> smooth_T = base_scale;

      vector[n] y_eff;
      y_eff = y;

      if (jittering == 1) {
        y_eff = y_eff + u;
      }
      if (log_flag == 1) {
        y_eff = log(y_eff);
      }

      vector[m-1] dtau;
      for (q in 1:(m-1)) dtau[q] = tau_q[q+1] - tau_q[q];
  }

  model {

      // jitter prior
      u ~ beta(1, 1);

      // beta0 prior (per-quantile intercept; informative warm-up-window prior)
      beta0 ~ normal(beta0_loc, beta0_scale);

      // betaX prior (user covariates only)
      if (px > 0) {
        if (prior_beta_code == 1) {
          to_vector(betaX) ~ normal(0, beta_sd);
        } else if (prior_beta_code == 2) {
          real lambda_beta2_eff;
          if (adaptive_beta == 1) {
            lambda_beta2 ~ gamma(lambda_beta2_a, lambda_beta2_b);
            lambda_beta2_eff = lambda_beta2;
          } else {
            lambda_beta2_eff = lambda_beta2_fixed;
          }
          for (j in 1:m) {
            for (i in 1:px) {
              sigma2_beta[j, i] ~ exponential(0.5 * lambda_beta2_eff);
              betaX[j, i] ~ normal(0, sqrt(sigma2_beta[j, i]));
            }
          }
        } else if (prior_beta_code == 6) {
          real lambda_beta2_eff;
          if (adaptive_beta == 1) {
            lambda_beta2 ~ gamma(lambda_beta2_a, lambda_beta2_b);
            lambda_beta2_eff = lambda_beta2;
          } else {
            lambda_beta2_eff = lambda_beta2_fixed;
          }
          for (j in 1:m) {
            for (i in 1:px) {
              lambda2_beta_local[j, i] ~ gamma(1, lambda_beta2_eff);
              sigma2_beta[j, i] ~ exponential(0.5 * lambda2_beta_local[j, i]);
              betaX[j, i] ~ normal(0, sqrt(sigma2_beta[j, i]));
            }
          }
        } else if (prior_beta_code == 3) {
          pi_slab_beta ~ beta(beta_slab_pi_a, beta_slab_pi_b);
          for (j in 1:m) {
            for (i in 1:px) {
              target += log_mix(
                pi_slab_beta,
                normal_lpdf(betaX[j, i] | 0, beta_slab_sd),
                normal_lpdf(betaX[j, i] | 0, beta_spike_sd)
              );
            }
          }
        } else if (prior_beta_code == 4) {
          real lambda_beta2_eff;
          if (adaptive_beta == 1) {
            lambda_beta2 ~ gamma(lambda_beta2_a, lambda_beta2_b);
            lambda_beta2_eff = lambda_beta2;
          } else {
            lambda_beta2_eff = lambda_beta2_fixed;
          }
          for (i in 1:px) {
            sigma2_beta_group[i] ~ gamma((m + 1) / 2, 0.5 * lambda_beta2_eff);
            for (j in 1:m) {
              betaX[j, i] ~ normal(0, sqrt(sigma2_beta_group[i]));
            }
          }
        } else if (prior_beta_code == 5) {
          real c_levy_beta;
          real lambda_beta2_eff;
          if (adaptive_beta == 1) {
            lambda_beta2 ~ gamma(lambda_beta2_a, lambda_beta2_b);
            lambda_beta2_eff = lambda_beta2;
          } else {
            lambda_beta2_eff = lambda_beta2_fixed;
          }
          c_levy_beta = lambda_beta2_eff;
          for (i in 1:px) {
            omega_beta_group[i] ~ inv_gamma(0.5, 0.5 * c_levy_beta);
            for (j in 1:m) {
              sigma2_beta[j, i] ~ exponential(0.5 * omega_beta_group[i]);
              betaX[j, i] ~ normal(0, sqrt(sigma2_beta[j, i]));
            }
          }
        }
      }

      // ----- Score-based likelihood using Z = [X | H] with logit smoothing -----
      {
        if ((p + r) > 0) {
          matrix[pr, m] S;

          for (q in 1:m) {
            vector[pr] s_q = rep_vector(0.0, pr);
            for (i in 1:n) {
              real xb = dot_product(to_vector(row(X_design, i)), to_vector(beta[q]));
              real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;

              real eta = xb + hb + offset[i];
              real r_i = y_eff[i] - eta;

              real z  = fmin(20, fmax(-20, -r_i / smooth_T));
              real Ilt = inv_logit(z);
              real psi = tau_q[q] - Ilt;

              s_q[1:p]      += to_vector(row(X_design, i)) * psi;
              if (r > 0) s_q[(p+1):pr] += to_vector(row(H, i)) * psi;
            }
            S[, q] = s_q;
          }

          matrix[m, m] L_Q = cholesky_decompose(Q);
          matrix[pr, m] A = mdivide_left_tri_low(L_Gs, S);
          matrix[m, pr] B = mdivide_left_tri_low(L_Q, A');

          target += -0.5 * dot_self(to_vector(B)) / n;
        }
      }

      // ----- Priors on gamma (H-coefficients) -----
      if (r > 0) {

        // Determine effective lambda_lasso2 value for lasso-type priors.
        real lambda_lasso2_eff;
        if (prior_code != 3 && prior_code != 6) {
          if (adaptive_gamma == 1) {
            lambda_lasso2 ~ gamma(lambda_lasso2_a, lambda_lasso2_b);
            lambda_lasso2_eff = lambda_lasso2;
          } else {
            lambda_lasso2_eff = lambda_lasso2_fixed;
          }
        } else {
          lambda_lasso2_eff = lambda_lasso2_fixed;
        }

        // 1 = group lasso
        if (prior_code == 1) {
          for (i in 1:r) {
            sigma2_gamma_group[i] ~ gamma( (m + 1) / 2, 0.5 * lambda_lasso2_eff );
            for (j in 1:m) {
              gamma[j, i] ~ normal(0, sqrt(sigma2_gamma_group[i]));
            }
          }

        // 2 = lasso
        } else if (prior_code == 2) {
          for (j in 1:m) {
            for (i in 1:r) {
              sigma2_gamma[j, i] ~ exponential(0.5 * lambda_lasso2_eff);
              gamma[j, i] ~ normal(0, sqrt(sigma2_gamma[j, i]));
            }
          }

        // 5 = adaptive lasso with coefficient-specific local shrinkage
        } else if (prior_code == 5) {
          for (j in 1:m) {
            for (i in 1:r) {
              lambda2_gamma_local[j, i] ~ gamma(1, lambda_lasso2_eff);
              sigma2_gamma[j, i] ~ exponential(0.5 * lambda2_gamma_local[j, i]);
              gamma[j, i] ~ normal(0, sqrt(sigma2_gamma[j, i]));
            }
          }

        // 3 = spike-and-slab
        } else if (prior_code == 3) {
          pi_slab ~ beta(slab_pi_a, slab_pi_b);
          for (j in 1:m) {
            for (i in 1:r) {
              target += log_mix(
                pi_slab,
                normal_lpdf(gamma[j, i] | 0, slab_sd),
                normal_lpdf(gamma[j, i] | 0, spike_sd)
              );
            }
          }

        // 4 = heterogeneous group lasso with Levy mixing
        // Groups by time block (consistent with group lasso prior_code=1)
        // omega_group[i]: block-level Levy scale (one per time block)
        // sigma2_gamma[j, i]: element-specific scale (one per quantile x block)
        } else if (prior_code == 4) {
          real c_levy = lambda_lasso2_eff;
          for (i in 1:r) {
            omega_group[i] ~ inv_gamma(0.5, 0.5 * c_levy);
            for (j in 1:m) {
              sigma2_gamma[j, i] ~ exponential(0.5 * omega_group[i]);
              gamma[j, i] ~ normal(0, sqrt(sigma2_gamma[j, i]));
            }
          }

        // 6 = spike-and-slab LASSO (Laplace spike + Laplace slab; Rockova & George 2018).
        // The Laplace spike contributes zero curvature away from 0, so the Laplace-
        // approximation posterior variance is driven by the likelihood (honest) rather
        // than a tiny prior variance -- avoids the near-Dirac Gaussian-spike degeneracy.
        } else if (prior_code == 6) {
          pi_slab ~ beta(slab_pi_a, slab_pi_b);
          for (j in 1:m) {
            for (i in 1:r) {
              target += log_mix(
                pi_slab,
                double_exponential_lpdf(gamma[j, i] | 0, slab_sd),
                double_exponential_lpdf(gamma[j, i] | 0, spike_sd)
              );
            }
          }
        }
      }

      // ---- Non-crossing penalty ----
      {
        real pen = 0;

        for (i in 1:n) {
          vector[m] eta_row;
          for (q in 1:m) {
            real xb = dot_product(to_vector(row(X_design, i)), to_vector(beta[q]));
            real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;
            eta_row[q] = xb + hb + offset[i];
          }

          // Non-crossing penalty: penalize negative derivatives
          for (q in 1:(m-1)) {
            real dfdtau = (eta_row[q + 1] - eta_row[q]) / dtau[q];
            pen += fmax(0, -dfdtau);
          }
        }

        pen /= (n * (m - 1));
        target += - lambda_nc * pen;
      }

      // ---- Interquantile (fused lasso) shrinkage penalty ----
      // Penalizes |coef[q] - coef[q-1]| with IQ weights (more shrinkage at outer quantiles)
      // Applied to gamma and betaX slopes (NOT the intercept beta0)
      {
        // Effective IQ penalty weight is the SQUARE ROOT of lambda_iq2, mirroring
        // the lambda_lasso2 convention where the stored quantity is lambda^2 and
        // the Laplace rate is lambda. Fixed within a fit; updated across fits by EM.
        real lambda_iq_eff = sqrt(lambda_iq2);

        if (lambda_iq_eff > 0) {
          real pen_iq_gamma = 0;
          real pen_iq_beta = 0;
          int n_components = 0;

          // Penalty on gamma (H-coefficients / change-point effects)
          if (r > 0) {
            for (j in 1:r) {
              for (q in 2:m) {
                pen_iq_gamma += w_iq_gamma[q-1, j] * fabs(gamma[q, j] - gamma[q-1, j]);
              }
            }
            pen_iq_gamma /= (r * (m - 1));
            n_components += 1;
          }

          // Penalty on beta (X-coefficients EXCLUDING intercept)
          // Note: Column 1 of X is the intercept - do NOT penalize it
          // (Jiang, Wang, & Bondell 2013 only penalize slope coefficients)
          if (p_slope > 0) {
            for (j in 1:p_slope) {
              for (q in 2:m) {
                pen_iq_beta += w_iq_beta[q-1, j] * fabs(beta[q, j+1] - beta[q-1, j+1]);
              }
            }
            pen_iq_beta /= (p_slope * (m - 1));
            n_components += 1;
          }

          // Average across components
          if (n_components > 0) {
            real pen_iq_total = (pen_iq_gamma + pen_iq_beta) / n_components;
            target += - lambda_iq_eff * pen_iq_total;
          }
        }
      }

      // Jacobian adjustment for log transform
      if (log_flag == 1) {
        if (jittering == 1) {
          target += -sum(log(y + u));
        } else {
          target += -sum(log(y));
        }
      }
  }
  "

  # ---------------- R-side pre-processing ----------------

  n <- length(y)
  m <- length(taus)

  if (is.null(offset)) {
    offset <- rep(0, n)
  }

  if (is.null(X)) {
    px <- 0L
    X <- matrix(0, nrow = n, ncol = 0)
  } else {
    X <- as.matrix(X)
    px <- ncol(X)
  }

  if (is.null(H)) {
    r <- 0
    H <- matrix(0, n, 0)
  } else {
    r <- ncol(H)
    if (r == 0) {
      H <- matrix(0, n, 0)
    }
  }

  # Modeling-scale response: the Stan likelihood operates on y_eff = log(y (+ u))
  # when log_flag = 1, so every data-driven quantity below -- the beta0 prior
  # location, the smoothing bandwidth, and the pilot fits for the adaptive IQ
  # weights -- must be computed on that same scale. The jitter u ~ Beta(1,1) is
  # approximated by its prior mean 0.5.
  if (log_flag == 1) {
    y_shift <- if (jittering == 1) 0.5 else 0
    if (any(y + y_shift <= 0)) {
      stop("log_flag = 1 requires y (plus the 0.5 jitter midpoint when ",
           "jittering = 1) to be strictly positive.")
    }
    y_model <- log(y + y_shift)
  } else {
    y_model <- y
  }

  # Smoothing bandwidth (Fernandes, Guerre & Horta 2021 rule-of-thumb; Silverman 1986):
  #   base_scale = 1.06 * s * n^(-1/5),  s = min(sd, IQR / 1.38898) of the standard
  # median (tau = 0.5) regression residuals. The pilot median fit uses only the
  # ordinary predictors [intercept | X]; the change-point design H is excluded so the
  # shift structure being detected cannot absorb the residual scale. Overridable via
  # the base_scale argument.
  if (is.null(base_scale)) {
    if (!is.null(X) && ncol(as.matrix(X)) > 0 &&
        requireNamespace("quantreg", quietly = TRUE)) {
      Xd_med <- cbind(1, as.matrix(X))
      fit_med <- try(suppressWarnings(
        quantreg::rq(y_model ~ Xd_med - 1, tau = 0.5, method = "fn")), silent = TRUE)
      med_resid <- if (inherits(fit_med, "try-error")) {
        y_model - as.numeric(stats::quantile(y_model, 0.5))
      } else {
        y_model - as.numeric(Xd_med %*% as.numeric(stats::coef(fit_med)))
      }
    } else {
      med_resid <- y_model - as.numeric(stats::quantile(y_model, 0.5))
    }
    s_disp <- min(stats::sd(med_resid), stats::IQR(med_resid) / 1.38898)
    base_scale <- max(1e-8, 1.06 * s_disp * length(y)^(-1 / 5))
  }

  # beta0 prior location: defaults to the per-quantile empirical level of the
  # warm-up period, on the modeling scale (log scale when log_flag = 1) --
  # empirical-quantile anchoring of the tau-specific intercept in the sense of
  # Yang & He (2012, Ann. Statist., p. 1107).
  if (is.null(beta0_loc)) {
    if (w > 0) {
      beta0_loc <- stats::quantile(y_model[1:w], probs = taus)
    } else {
      beta0_loc <- stats::quantile(y_model, probs = taus)
    }
  }
  beta0_loc <- as.vector(beta0_loc)
  if (length(beta0_loc) != m || anyNA(beta0_loc)) {
    stop("beta0_loc must be NULL or a length-m numeric vector without NAs.")
  }

  # beta0 prior scale: defaults to the unit-information prior (Kass & Wasserman
  # 1995) -- the prior carries the information of a single warm-up observation
  # about each quantile: sd = sqrt(tau (1 - tau)) / f_hat(beta0_loc), with f_hat
  # a kernel density estimate of the warm-up period. Equivalently, the power
  # prior on the warm-up period with discount a0 = 1/w (Ibrahim & Chen 2000;
  # Bourazas, Kiagias & Tsiamyrtzis 2022). Override with a scalar or length-m
  # vector on the modeling scale of y.
  if (is.null(beta0_scale)) {
    yw <- if (w > 0) y_model[1:w] else y_model
    if (length(unique(yw)) < 2L) {
      stop("The warm-up period is (nearly) constant; supply beta0_scale explicitly.")
    }
    dw <- stats::density(yw)
    f_hat <- stats::approx(dw$x, dw$y, xout = beta0_loc, rule = 2)$y
    f_hat <- pmax(f_hat, 0.05 / diff(range(yw)))   # numerical floor for sparse tails
    beta0_scale <- sqrt(taus * (1 - taus)) / f_hat
  } else if (length(beta0_scale) == 1L) {
    beta0_scale <- rep(beta0_scale, m)
  }
  if (length(beta0_scale) != m || any(beta0_scale <= 0)) {
    stop("beta0_scale must be a positive scalar or a length-m positive vector.")
  }

  # ---- IQ weight matrices construction (data-driven adaptive weights) ----
  p_total <- px + 1L
  p_slope <- px
  w_iq_gamma <- matrix(1, nrow = m - 1L, ncol = max(r, 0L))
  w_iq_beta  <- matrix(1, nrow = m - 1L, ncol = max(p_slope, 0L))

  if ((r > 0 || p_slope > 0) && requireNamespace("quantreg", quietly = TRUE)) {
    X_design <- cbind(Intercept = 1, X)
    Z_pilot <- if (r > 0) cbind(X_design, H) else X_design
    d_pilot <- ncol(Z_pilot)

    pilot_coefs <- matrix(NA, nrow = m, ncol = d_pilot)
    for (q in seq_len(m)) {
      pilot_coefs[q, ] <- safe_pilot_coefs(
        y = y_model, Z = Z_pilot, tau = taus[q], eps_w = eps_w
      )
    }

    if (r > 0) {
      gamma_pilot <- pilot_coefs[, (p_total + 1):(p_total + r), drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(r)) {
          diff_val <- abs(gamma_pilot[q, j] - gamma_pilot[q - 1, j])
          w_iq_gamma[q - 1, j] <- 1 / max(diff_val, eps_w)
        }
      }
    }

    if (p_slope > 0) {
      beta_pilot <- pilot_coefs[, 2:p_total, drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(p_slope)) {
          diff_val <- abs(beta_pilot[q, j] - beta_pilot[q - 1, j])
          w_iq_beta[q - 1, j] <- 1 / max(diff_val, eps_w)
        }
      }
    }
  } else if ((r > 0 || p_slope > 0) && !requireNamespace("quantreg", quietly = TRUE)) {
    warning("Package 'quantreg' not available; using uniform IQ shrinkage weights.")
  }


  stan_data <- list(
    n = n, px = px, m = m, r = r,
    X = X, H = H,
    y = y, offset = offset, tau_q = taus,
    beta0_loc = beta0_loc, beta0_scale = beta0_scale,
    base_scale = base_scale, c_sigma = c_sigma, beta_sd = beta_sd,
    lambda_nc = lambda_nc,
    lambda_iq2 = lambda_iq2,                      # SQUARED IQ fusion weight (rate = sqrt)
    lambda_beta2_a = lambda_beta2_a, lambda_beta2_b = lambda_beta2_b,
    adaptive_beta = as.integer(adaptive_beta),
    lambda_beta2_fixed = lambda_beta2_fixed,
    lambda_lasso2_a = lambda_lasso2_a, lambda_lasso2_b = lambda_lasso2_b,
    adaptive_gamma = as.integer(adaptive_gamma),
    lambda_lasso2_fixed = lambda_lasso2_fixed,
    log_flag = log_flag, jittering = jittering,
    prior_beta_code = prior_beta_code,
    prior_code = prior_code,
    beta_spike_sd = beta_spike_sd,
    beta_slab_sd = beta_slab_sd,
    beta_slab_pi_a = beta_slab_pi_a,
    beta_slab_pi_b = beta_slab_pi_b,
    spike_sd = spike_sd,
    slab_sd  = slab_sd,
    slab_pi_a = slab_pi_a,
    slab_pi_b = slab_pi_b,
    p_slope = as.integer(p_slope),
    w_iq_gamma = if (r > 0) w_iq_gamma else matrix(0, m - 1L, 0L),
    w_iq_beta  = if (p_slope > 0) w_iq_beta else matrix(0, m - 1L, 0L)
  )

  # Prior-center initialization for MAP optimization: start beta0 at its prior
  # location, all shift/slope coefficients at 0, and the shrinkage hierarchy at
  # neutral values. The smoothed score saturates (|z| clamped at 20) once a curve
  # sits ~20 bandwidths from the data, leaving zero gradient there; random inits
  # can start in (or reach) these flat regions and yield runaway MAP modes.
  init_prior_center <- list(
    beta0 = as.array(beta0_loc),
    betaX = matrix(0, m, px),
    gamma = matrix(0, m, r),
    sigma2_beta_group = as.array(rep(1, px)),
    sigma2_beta = matrix(1, m, px),
    lambda2_beta_local = matrix(1, m, px),
    sigma2_gamma_group = as.array(rep(1, r)),
    sigma2_gamma = matrix(1, m, r),
    lambda2_gamma_local = matrix(1, m, r),
    lambda_beta2 = 1, lambda_lasso2 = 1,
    pi_slab_beta = 0.5, pi_slab = 0.5,
    omega_beta_group = as.array(rep(1, px)),
    omega_group = as.array(rep(1, r)),
    u = as.array(rep(0.5, n))
  )

  # Pilot-based initialization (map_init = "pilot"): marginal LASSO quantile
  # regression per quantile level on [1 | X | H] via quantreg::rq.fit.lasso,
  # with the intercept unpenalized and the slope penalty at the
  # sqrt(tau (1 - tau) n log d) rate (cf. Belloni & Chernozhukov 2011). The
  # sparse, data-proximal start begins inside the region where the smoothed
  # score has live gradients and matches the spike-slab prior's sparsity.
  # (The adaptive IQ weights keep their own pilot fits; this block is
  # initialization only.) Mode-selection nuance: "pilot" approaches the optimum
  # from the data-fitting side, whereas "prior_center" grows the solution from
  # the in-control state; for multimodal spike-slab posteriors compare final lp.
  init_pilot <- init_prior_center
  if (map_init == "pilot") {
    pilot_init <- NULL
    if ((px + r) > 0 && requireNamespace("quantreg", quietly = TRUE)) {
      Z_init <- cbind(1, X, H)
      d_init <- ncol(Z_init)
      pilot_init <- matrix(NA_real_, m, d_init)
      for (q in seq_len(m)) {
        lam_q <- sqrt(taus[q] * (1 - taus[q]) * n * log(d_init))
        lam_vec <- c(0, rep(lam_q, d_init - 1L))  # do not penalize the intercept
        fit_l <- try(suppressWarnings(
          quantreg::rq.fit.lasso(Z_init, y_model, tau = taus[q], lambda = lam_vec)
        ), silent = TRUE)
        cf <- if (inherits(fit_l, "try-error")) NULL else as.numeric(fit_l$coefficients)
        if (!is.null(cf) && length(cf) == d_init && all(is.finite(cf))) {
          pilot_init[q, ] <- cf
        }
      }
      if (anyNA(pilot_init)) pilot_init <- NULL
    }
    if (!is.null(pilot_init)) {
      cap <- 5 * diff(range(y_model))
      clamp <- function(x, ref = 0) ref + pmin(pmax(x - ref, -cap), cap)
      init_pilot$beta0 <- as.array(clamp(pilot_init[, 1], beta0_loc))
      if (px > 0) init_pilot$betaX <-
        matrix(clamp(pilot_init[, 2:(px + 1), drop = FALSE]), m, px)
      if (r > 0) init_pilot$gamma <-
        matrix(clamp(pilot_init[, (px + 2):(px + 1 + r), drop = FALSE]), m, r)
    } else {
      warning("map_init = 'pilot' requested but rq.fit.lasso pilot fits are ",
              "unavailable; falling back to 'prior_center'.")
      map_init <- "prior_center"
    }
  }

  # Run rstan::optimizing with its generic "non-zero return code" warning
  # replaced by an informative termination record. Under the tight default
  # tolerances, L-BFGS commonly exits with code 70 (line search cannot further
  # improve the objective) -- the expected exit at a converged optimum for this
  # objective. Exit code 0 only means a stopping tolerance fired first; it is
  # not by itself evidence of a better fit (see map_init/tolerance docs).
  run_optimizing <- function(opt_args) {
    res <- withCallingHandlers(
      do.call(rstan::optimizing, opt_args),
      warning = function(w) {
        if (grepl("non-zero return code", conditionMessage(w), fixed = TRUE)) {
          invokeRestart("muffleWarning")
        }
      }
    )
    rc <- res$return_code
    res$termination <- if (identical(rc, 0L)) {
      "converged: a stopping tolerance was met"
    } else if (identical(rc, 70L)) {
      paste0("line search exhausted (exit code 70): no direction improves the ",
             "objective further; expected at a converged optimum")
    } else {
      sprintf("optimizer exit code %s", rc)
    }
    if (!identical(rc, 0L) && !identical(rc, 70L)) {
      warning("MAP optimization ended with unusual exit code ", rc,
              "; check fit quality via fit$map$coverage and fit$map$value.")
    }
    res
  }

  # Fit-quality diagnostic: empirical coverage of the fitted quantile curves,
  # colMeans(y < eta_tau), which should sit near tau. Guards against silently
  # bad MAP modes (curves stranded away from the data) -- loud where the
  # optimizer's exit code is uninformative.
  map_coverage_check <- function(par_vec) {
    tryCatch({
      b0 <- par_vec[sprintf("beta0[%d]", seq_len(m))]
      eta <- matrix(rep(b0, each = n), n, m)
      if (px > 0) {
        bX <- matrix(par_vec[grep("^betaX\\[", names(par_vec))], nrow = m)
        eta <- eta + X %*% t(bX)
      }
      if (r > 0) {
        gm <- matrix(par_vec[grep("^gamma\\[", names(par_vec))], nrow = m)
        eta <- eta + H %*% t(gm)
      }
      eta <- eta + offset
      cov_hat <- colMeans(y_model < eta)
      names(cov_hat) <- paste0("tau=", taus)
      tol_cov <- pmax(0.10, 4 * sqrt(taus * (1 - taus) / n))
      if (any(abs(cov_hat - taus) > tol_cov)) {
        warning("MAP fit-quality check: empirical coverage of the fitted ",
                "quantile curves deviates from the target levels (",
                paste(sprintf("%.3f vs %.3f", cov_hat, taus), collapse = "; "),
                "). The optimizer may have stopped in a poor mode; consider ",
                "map_init = 'prior_center', a larger map_iter, or tighter ",
                "tolerances.", call. = FALSE)
      }
      cov_hat
    }, error = function(e) NULL)
  }

  # Compile Stan model once per session (cached)
  if (is.null(.bqq_stan_cache$sm)) {
    if (verbose) message("Compiling BQQ Stan model (one-time per session)...")
    .bqq_stan_cache$sm <- rstan::stan_model(model_code = stan_code)
  }
  sm <- .bqq_stan_cache$sm

  # Initialize outputs
  fit <- NULL
  map_fit <- NULL
  hessian <- NULL
  laplace_samples <- NULL

  # Helper: parse 2D parameter indices from names like "gamma[1,2]"
  parse_2d_idx <- function(par_names, idx, prefix) {
    dims_str <- gsub(paste0(prefix, "\\[|\\]"), "", par_names[idx])
    dims_split <- strsplit(dims_str, ",")
    list(
      row = as.integer(sapply(dims_split, `[`, 1)),
      col = as.integer(sapply(dims_split, `[`, 2))
    )
  }

  # Helper: scatter flat sample vector into 3D array
  scatter_to_array <- function(samples_mat, row_idx, col_idx, n_row, n_col, n_samples) {
    arr <- array(NA, dim = c(n_samples, n_row, n_col))
    for (i in seq_along(row_idx)) {
      arr[, row_idx[i], col_idx[i]] <- samples_mat[, i]
    }
    arr
  }

  # Helper: generate Laplace-approximation samples from the MAP fit. The Hessian
  # from rstan::optimizing() is on the UNCONSTRAINED scale (k x k, raw parameters
  # only). The eta-relevant parameters are the intercept beta0, user slopes betaX,
  # and the change-point effects gamma; the baseline lives entirely in beta0.
  generate_laplace_samples <- function(par_map, hessian, n_samples, noise_scale, seed_val) {
    if (!is.null(seed_val)) set.seed(seed_val)

    par_names <- names(par_map)
    n_par <- length(par_map)

    # --- Identify raw parameter indices (first k entries of par_map) ---
    # The Hessian has k rows/cols corresponding to the raw (unconstrained) parameters.
    # par_map[1:k] = raw parameters on CONSTRAINED scale
    # par_map[(k+1):n_par] = transformed parameters
    k <- if (!is.null(hessian)) nrow(hessian) else 0
    raw_par_names <- if (k > 0) names(par_map)[1:k] else character(0)

    # Indices within raw parameters for the eta-relevant components
    beta0_idx  <- grep("^beta0\\[",  raw_par_names)
    betaX_idx  <- grep("^betaX\\[",  raw_par_names)
    gamma_idx  <- grep("^gamma\\[",  raw_par_names)
    eta_param_idx <- c(beta0_idx, betaX_idx, gamma_idx)

    # Parse dimensions from parameter names
    beta0_parsed  <- if (length(beta0_idx) > 0) {
      list(idx = as.integer(gsub("beta0\\[|\\]", "", raw_par_names[beta0_idx])))
    } else NULL
    betaX_parsed  <- if (length(betaX_idx) > 0) parse_2d_idx(raw_par_names, betaX_idx, "betaX") else NULL
    gamma_parsed  <- if (length(gamma_idx) > 0) parse_2d_idx(raw_par_names, gamma_idx, "gamma") else NULL

    # --- Try proper Hessian-based Laplace approximation ---
    laplace_ok <- FALSE
    beta_array <- NULL
    gamma_array <- NULL

    if (!is.null(hessian) && k > 0 && length(eta_param_idx) > 0) {
      laplace_result <- tryCatch({

        # Build unconstrained mean vector. beta0/betaX/gamma are unbounded;
        # lower-bounded scale params are log-transformed below for correct inversion.
        theta_unc_full <- as.numeric(par_map[1:k])

        # sigma2_beta_group (<lower=0>): log
        tbg_idx <- grep("^sigma2_beta_group", raw_par_names)
        if (length(tbg_idx) > 0) theta_unc_full[tbg_idx] <- log(pmax(par_map[tbg_idx], 1e-10))
        # sigma2_gamma_group (<lower=0>): log
        tgg_idx <- grep("^sigma2_gamma_group", raw_par_names)
        if (length(tgg_idx) > 0) theta_unc_full[tgg_idx] <- log(pmax(par_map[tgg_idx], 1e-10))
        # sigma2_beta (<lower=0>): log
        tb_idx <- grep("^sigma2_beta\\[", raw_par_names)
        if (length(tb_idx) > 0) theta_unc_full[tb_idx] <- log(pmax(par_map[tb_idx], 1e-10))
        # sigma2_gamma (<lower=0>): log
        tg_idx <- grep("^sigma2_gamma\\[", raw_par_names)
        if (length(tg_idx) > 0) theta_unc_full[tg_idx] <- log(pmax(par_map[tg_idx], 1e-10))
        # lambda_lasso2, lambda_beta2, omega_* (<lower=0>): log
        for (pat in c("^lambda_lasso2$", "^lambda_beta2$", "^lambda2_beta_local\\[", "^lambda2_gamma_local\\[", "^omega_group", "^omega_beta_group")) {
          idx_tmp <- grep(pat, raw_par_names)
          if (length(idx_tmp) > 0) theta_unc_full[idx_tmp] <- log(pmax(par_map[idx_tmp], 1e-10))
        }
        # pi_slab_beta (<lower=0, upper=1>): logit
        pi_beta_idx <- grep("^pi_slab_beta$", raw_par_names)
        if (length(pi_beta_idx) > 0) {
          pv <- pmin(pmax(par_map[pi_beta_idx], 1e-10), 1 - 1e-10)
          theta_unc_full[pi_beta_idx] <- log(pv / (1 - pv))
        }
        # pi_slab (<lower=0, upper=1>): logit
        pi_idx <- grep("^pi_slab$", raw_par_names)
        if (length(pi_idx) > 0) {
          pv <- pmin(pmax(par_map[pi_idx], 1e-10), 1 - 1e-10)
          theta_unc_full[pi_idx] <- log(pv / (1 - pv))
        }
        # u (<lower=1e-12, upper=1>): logit (approx)
        u_idx <- grep("^u\\[", raw_par_names)
        if (length(u_idx) > 0) {
          uv <- pmin(pmax(par_map[u_idx], 1e-10), 1 - 1e-10)
          theta_unc_full[u_idx] <- log(uv / (1 - uv))
        }

        # Invert full Hessian to get posterior covariance on unconstrained scale.
        # Moore-Penrose pseudo-inverse is robust to the zero-curvature directions
        # of unused-prior scale latents (their flat Hessian rows otherwise make
        # solve() report a computationally singular system). eta is decoupled from
        # those latents, so its submatrix below is the exact marginal covariance.
        H_neg <- -(hessian + t(hessian)) / 2  # ensure symmetry
        H_neg_reg <- H_neg + diag(1e-6, k)
        Sigma_full <- MASS::ginv(H_neg_reg)

        # Extract marginal covariance for the eta-related parameters
        Sigma_sub <- Sigma_full[eta_param_idx, eta_param_idx]

        # Ensure positive definiteness
        eig <- eigen(Sigma_sub, symmetric = TRUE)
        eig$values <- pmax(eig$values, 1e-8)
        n_eta <- length(eig$values)
        L <- t(eig$vectors %*% diag(sqrt(eig$values), nrow = n_eta, ncol = n_eta))

        # Sample from MVN on unconstrained scale
        theta_unc_sub <- theta_unc_full[eta_param_idx]
        z_mat <- matrix(rnorm(n_samples * length(eta_param_idx)), n_samples, length(eta_param_idx))
        samples_unc <- sweep(z_mat %*% L, 2, theta_unc_sub, "+")

        # --- Map columns back to parameter blocks (order of eta_param_idx) ---
        beta0_cols  <- seq_along(beta0_idx)
        betaX_cols  <- length(beta0_idx) + seq_along(betaX_idx)
        gamma_cols  <- length(beta0_idx) + length(betaX_idx) + seq_along(gamma_idx)

        # --- Assemble beta and gamma arrays ---
        beta_arr <- if (length(beta0_idx) > 0 || length(betaX_idx) > 0) {
          m_beta <- if (!is.null(beta0_parsed)) {
            max(beta0_parsed$idx)
          } else {
            max(betaX_parsed$row)
          }
          p_beta <- 1 + if (!is.null(betaX_parsed)) max(betaX_parsed$col) else 0
          arr <- array(0, dim = c(n_samples, m_beta, p_beta))
          if (!is.null(beta0_parsed)) {
            for (i in seq_along(beta0_parsed$idx)) {
              arr[, beta0_parsed$idx[i], 1] <- samples_unc[, beta0_cols[i]]
            }
          }
          if (!is.null(betaX_parsed)) {
            for (i in seq_along(betaX_parsed$row)) {
              arr[, betaX_parsed$row[i], betaX_parsed$col[i] + 1] <- samples_unc[, betaX_cols[i]]
            }
          }
          arr
        } else NULL

        gamma_arr <- if (length(gamma_idx) > 0) {
          scatter_to_array(samples_unc[, gamma_cols, drop = FALSE],
                           gamma_parsed$row, gamma_parsed$col,
                           max(gamma_parsed$row), max(gamma_parsed$col), n_samples)
        } else NULL

        list(beta = beta_arr, gamma = gamma_arr)

      }, error = function(e) {
        warning("Hessian-based Laplace failed: ", conditionMessage(e),
                ". Falling back to heuristic noise.")
        NULL
      })

      if (!is.null(laplace_result)) {
        beta_array <- laplace_result$beta
        gamma_array <- laplace_result$gamma
        laplace_ok <- TRUE
      }
    }

    # --- Fallback: heuristic noise (when Hessian is unavailable or inversion fails) ---
    if (!laplace_ok) {

      beta_full_idx <- grep("^beta\\[", par_names)
      beta_array <- if (length(beta_full_idx) > 0) {
        beta_full_parsed <- parse_2d_idx(par_names, beta_full_idx, "beta")
        m_beta <- max(beta_full_parsed$row); p_beta <- max(beta_full_parsed$col)
        beta_map <- matrix(NA, m_beta, p_beta)
        for (i in seq_along(beta_full_idx)) beta_map[beta_full_parsed$row[i], beta_full_parsed$col[i]] <- par_map[beta_full_idx[i]]
        beta_sd <- pmax(abs(beta_map) * noise_scale, 0.05)
        arr <- array(NA, dim = c(n_samples, m_beta, p_beta))
        for (s in 1:n_samples) arr[s, , ] <- beta_map + matrix(rnorm(m_beta * p_beta, 0, beta_sd), m_beta, p_beta)
        arr
      }

      gamma_full_idx <- grep("^gamma\\[", par_names)
      gamma_array <- if (length(gamma_full_idx) > 0) {
        gamma_full_parsed <- parse_2d_idx(par_names, gamma_full_idx, "gamma")
        m_gamma <- max(gamma_full_parsed$row); r_gamma <- max(gamma_full_parsed$col)
        gamma_map <- matrix(NA, m_gamma, r_gamma)
        for (i in seq_along(gamma_full_idx)) gamma_map[gamma_full_parsed$row[i], gamma_full_parsed$col[i]] <- par_map[gamma_full_idx[i]]
        gamma_sd <- pmax(abs(gamma_map) * noise_scale, 0.02)
        arr <- array(NA, dim = c(n_samples, m_gamma, r_gamma))
        for (s in 1:n_samples) arr[s, , ] <- gamma_map + matrix(rnorm(m_gamma * r_gamma, 0, gamma_sd), m_gamma, r_gamma)
        arr
      }
    }

    list(mu = NULL, beta = beta_array, gamma = gamma_array)
  }

  # ------------------------------------------------------------------
  # One complete fit at a given stan_data -- in particular, at a given
  # lambda_iq2. Factored out so the EM recursion below can call it
  # repeatedly. EM sits OUTSIDE the fit: it updates lambda_iq2 between
  # refits and touches no Stan code.
  # ------------------------------------------------------------------
  run_one_fit <- function(stan_data) {
    fit <- NULL
    map_fit <- NULL
    hessian <- NULL
    laplace_samples <- NULL

  # ------------------------------------------------------------------
  # fit_method = "mcmc": MCMC only, estimators are posterior median
  # ------------------------------------------------------------------
  if (fit_method == "mcmc") {
    fit <- rstan::sampling(
      sm, data = stan_data,
      chains = chains, iter = iter, warmup = warmup,
      control = control, seed = seed, verbose = verbose
    )

    # Extract posterior median as point estimates
    draws <- rstan::extract(fit, pars = c("beta", "gamma"))
    map_fit <- list(par = list())
    if (!is.null(draws$beta)) map_fit$par$beta <- apply(draws$beta, c(2, 3), median)
    if (!is.null(draws$gamma)) map_fit$par$gamma <- apply(draws$gamma, c(2, 3), median)
    map_fit$estimator <- "posterior_median"

  # ------------------------------------------------------------------
  # fit_method = "map_mcmc": MAP estimators, MCMC posterior draws
  # ------------------------------------------------------------------
  } else if (fit_method == "map_mcmc") {
    # First get MAP estimates
    opt_args <- list(
      object = sm,
      data = stan_data,
      hessian = map_hessian,
      as_vector = FALSE,
      seed = seed,
      verbose = verbose
    )
    if (map_init == "prior_center") opt_args$init <- init_prior_center
    else if (map_init == "pilot")   opt_args$init <- init_pilot
    if (!is.null(map_tol_obj))       opt_args$tol_obj       <- map_tol_obj
    if (!is.null(map_tol_grad))      opt_args$tol_grad      <- map_tol_grad
    if (!is.null(map_tol_rel_grad))  opt_args$tol_rel_grad  <- map_tol_rel_grad
    if (!is.null(map_tol_param))     opt_args$tol_param     <- map_tol_param
    if (!is.null(map_tol_rel_obj))   opt_args$tol_rel_obj   <- map_tol_rel_obj
    if (!is.null(map_history_size)) opt_args$history_size  <- map_history_size
    if (!is.null(map_iter))          opt_args$iter          <- map_iter
    map_fit <- run_optimizing(opt_args)
    map_fit$estimator <- "map"

    hessian <- if (map_hessian && !is.null(map_fit$hessian)) map_fit$hessian else NULL

    # Then run MCMC with MAP as initialization
    init_theta <- map_fit$par
    fit <- rstan::sampling(
      sm, data = stan_data,
      chains = chains, iter = iter, warmup = warmup,
      init = function() init_theta, init_r = 0.01,
      control = control, seed = seed, verbose = verbose
    )

  # ------------------------------------------------------------------
  # fit_method = "map": MAP estimators, Laplacian approximation draws
  # ------------------------------------------------------------------
  } else if (fit_method == "map") {
    opt_args <- list(
      object = sm,
      data = stan_data,
      hessian = map_hessian,
      as_vector = TRUE,
      seed = seed,
      verbose = verbose
    )
    if (map_init == "prior_center") opt_args$init <- init_prior_center
    else if (map_init == "pilot")   opt_args$init <- init_pilot
    if (!is.null(map_tol_obj))       opt_args$tol_obj       <- map_tol_obj
    if (!is.null(map_tol_grad))      opt_args$tol_grad      <- map_tol_grad
    if (!is.null(map_tol_rel_grad))  opt_args$tol_rel_grad  <- map_tol_rel_grad
    if (!is.null(map_tol_param))     opt_args$tol_param     <- map_tol_param
    if (!is.null(map_tol_rel_obj))   opt_args$tol_rel_obj   <- map_tol_rel_obj
    if (!is.null(map_history_size)) opt_args$history_size  <- map_history_size
    if (!is.null(map_iter))          opt_args$iter          <- map_iter
    map_fit <- run_optimizing(opt_args)
    map_fit$estimator <- "map"
    map_fit$coverage <- map_coverage_check(map_fit$par)

    hessian <- if (map_hessian && !is.null(map_fit$hessian)) map_fit$hessian else NULL

    # Generate Laplace approximation samples (uses Hessian if available)
    laplace_samples <- generate_laplace_samples(
      par_map = map_fit$par,
      hessian = hessian,
      n_samples = laplace_n_samples,
      noise_scale = laplace_noise_scale,
      seed_val = seed
    )
  }

    list(fit = fit, map = map_fit, hessian = hessian,
         laplace_samples = laplace_samples)
  }

  # ------------------------------------------------------------------
  # lambda_iq2: EM across refits, or a single fit at the supplied value
  # ------------------------------------------------------------------
  n_iq_diff <- .bqq_iq_n_diff(r, p_slope, m)
  iq_em <- list(
    adaptive        = isTRUE(adaptive_iq),
    update          = "fixedpoint",
    n_diff          = n_iq_diff,
    lambda_iq2      = lambda_iq2,
    lambda_iq2_next = NA_real_,
    Sbar            = NA_real_,
    converged       = NA,
    stop_reason     = NA_character_,
    n_iter          = 0L,
    trace           = NULL,
    note            = NULL
  )

  if (isTRUE(adaptive_iq) && n_iq_diff > 0L) {
    cur <- lambda_iq2
    lam_used <- cur
    nxt <- NA_real_
    Sbar <- NA_real_
    converged <- FALSE
    tr <- NULL
    res <- NULL
    # Monte-Carlo floor detection: once the E-step noise dominates, the updates
    # alternate direction about the fixed point instead of shrinking.
    prev_sign <- NA_integer_
    flips <- 0L
    recent_rel <- numeric(0)
    stop_reason <- "max_iter"

    for (s in seq_len(iq_em_max_iter)) {
      stan_data$lambda_iq2 <- cur
      res <- run_one_fit(stan_data)
      lam_used <- cur

      draws <- .bqq_iq_draws(res)
      Sbar <- if (is.null(draws)) NA_real_ else
        .bqq_iq_Sbar(draws, w_iq_gamma, w_iq_beta, r, p_slope, m)
      nxt <- .bqq_iq_em_step(cur, Sbar, n_iq_diff)
      rel <- if (is.finite(nxt)) abs(nxt - cur) / max(cur, .Machine$double.eps) else NA_real_

      tr <- rbind(tr, data.frame(
        iter = s, lambda_iq2 = lam_used, lambda_iq = sqrt(lam_used),
        Sbar = Sbar, lambda_iq2_next = nxt, rel_change = rel
      ))
      if (verbose) {
        message(sprintf("[iq-EM %02d] lambda_iq2 = %.6g  (lambda = %.6g)  Sbar = %.6g  -> %.6g",
                        s, lam_used, sqrt(lam_used), Sbar, nxt))
      }

      if (!is.finite(nxt)) {
        stop_reason <- "no_usable_update"
        iq_em$note <- sprintf(
          "EM halted at iteration %d: Sbar = %s produced no usable update; keeping lambda_iq2 = %.6g.",
          s, format(Sbar), lam_used)
        break
      }
      if (is.finite(rel) && rel < iq_em_tol) {
        converged <- TRUE
        stop_reason <- "tol"
        break
      }

      # Monte-Carlo floor: direction reversals with only small moves mean the
      # E-step noise, not the recursion, is now driving lambda_iq2.
      sgn <- sign(nxt - cur)
      if (!is.na(prev_sign) && sgn != 0L && sgn == -prev_sign) flips <- flips + 1L
      else if (sgn != 0L) flips <- 0L
      if (sgn != 0L) prev_sign <- sgn
      recent_rel <- c(recent_rel, rel)
      if (length(recent_rel) > 3L) recent_rel <- recent_rel[-1L]
      if (iq_em_mc_tol > 0 && flips >= 2L && length(recent_rel) >= 3L &&
          all(is.finite(recent_rel)) && max(recent_rel) < iq_em_mc_tol) {
        converged <- TRUE
        stop_reason <- "mc_floor"
        iq_em$note <- sprintf(
          paste0("Stopped at the Monte-Carlo floor after %d iterations: lambda_iq2 is oscillating ",
                 "about its fixed point by at most %.2g (relative), which is E-step sampling noise ",
                 "from laplace_n_samples = %d, not a failure to converge. iq_em_tol = %g is below ",
                 "that floor and cannot be met. Raise laplace_n_samples to tighten it."),
          s, max(recent_rel), laplace_n_samples, iq_em_tol)
        break
      }
      cur <- nxt
    }

    # The returned fit is the one actually computed at lam_used, so the model
    # and the reported lambda_iq2 are self-consistent; no extra refit is done.
    stan_data$lambda_iq2 <- lam_used
    fit             <- res$fit
    map_fit         <- res$map
    hessian         <- res$hessian
    laplace_samples <- res$laplace_samples

    iq_em$lambda_iq2      <- lam_used
    iq_em$lambda_iq2_next <- nxt
    iq_em$Sbar            <- Sbar
    iq_em$converged       <- converged
    iq_em$stop_reason     <- stop_reason
    iq_em$n_iter          <- if (is.null(tr)) 0L else nrow(tr)
    iq_em$trace           <- tr

    if (!converged && is.null(iq_em$note)) {
      iq_em$note <- sprintf(
        paste0("EM did not meet iq_em_tol = %g within iq_em_max_iter = %d (last relative change ",
               "%.3g). Reported lambda_iq2 is the last fitted value. If the trace is oscillating ",
               "rather than drifting, this is E-step Monte-Carlo noise: raise laplace_n_samples ",
               "(currently %d) or iq_em_mc_tol."),
        iq_em_tol, iq_em_max_iter, tr$rel_change[nrow(tr)], laplace_n_samples)
      warning(iq_em$note, call. = FALSE)
    }

  } else {
    if (isTRUE(adaptive_iq) && n_iq_diff == 0L) {
      iq_em$adaptive <- FALSE
      iq_em$note <- "adaptive_iq = TRUE ignored: there are no penalized adjacent-quantile differences (need m >= 2 and r > 0 or p_slope > 0)."
    }
    res <- run_one_fit(stan_data)
    fit             <- res$fit
    map_fit         <- res$map
    hessian         <- res$hessian
    laplace_samples <- res$laplace_samples
  }

  list(
    fit = fit,
    map = map_fit,
    y = y, H = H, X = X, taus = taus,
    hessian = hessian,
    fit_method = fit_method,
    laplace_samples = laplace_samples,
    stan_data = stan_data,
    iq_em = iq_em
  )
}
