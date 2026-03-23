#' Smoothed Quantile Regression with Interquantile Shrinkage (Stan)
#'
#' Fits a multi-quantile (\eqn{m}) regression model where the conditional
#' quantile function is modeled as a latent random walk in time (or index)
#' with optional fixed effects \eqn{X} and structured effects \eqn{H}.
#' The \eqn{H}-coefficients are shrunk via a **grouped Bayesian LASSO**
#' (column-wise sharing across quantiles), and adjacent quantiles are softly
#' penalized to discourage crossings. **Interquantile shrinkage** stabilizes
#' outer quantiles by penalizing differences between adjacent quantile coefficients.
#'
#' @section Model (high level):
#' \describe{
#'   \item{Data & design}{
#'     \itemize{
#'       \item \eqn{y_i} is optionally jittered (\code{u ~ Beta(1,1)}) and/or log-transformed.
#'       \item Combined linear predictor \eqn{\eta_{qi} = \mu_{q,i} + x_i^\top \beta_q + h_i^\top \gamma_q + \mathrm{offset}_i}.
#'       \item \eqn{\mu_{q,\cdot}} is a random walk per quantile: \eqn{\mu_{q,t} = \mu_{q,t-1} + \tau^{(rw)}_q z_{q,t-1}}.
#'     }
#'   }
#'   \item{Interquantile shrinkage}{
#'     Penalizes differences between adjacent quantile coefficients for gamma and beta slopes
#'     using data-driven adaptive weights from pilot quantile regression estimates:
#'     \eqn{\text{pen}_{\text{IQ}} = \sum_{q=2}^m \sum_j w_{q,j} |\theta_{q,j} - \theta_{q-1,j}|}
#'     where \eqn{w_{q,j} = (|\tilde{\theta}_{q,j} - \tilde{\theta}_{q-1,j}| + \epsilon_w)^{-1}}
#'     and \eqn{\tilde{\theta}} are pilot estimates from separate quantile regressions
#'     (Jiang, Wang, & Bondell 2013).
#'     Weights are median-normalized for scale invariance and applied separately to gamma and beta slopes.
#'     Falls back to uniform weights (all 1) when quantreg is not available.
#'     Note: Intercept is NOT penalized (per Jiang, Wang, & Bondell 2013).
#'     Note: mu (random walk) is NOT penalized to allow quantile-specific temporal evolution.
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
#' @param X Optional numeric matrix \eqn{n \times p_x} of additional predictors.
#' @param offset Optional numeric vector of length \eqn{n} added to the linear predictor.
#' @param alpha Positive scalar exponent for adaptive LASSO weights (default 0.75).
#' @param eps_w Positive scalar added to pilot estimates for numerical stability
#'   in both adaptive LASSO weights and IQ shrinkage weights (default 1e-6).
#' @param c_sigma Positive scalar scaling factor for the base scale (default 1.0).
#' @param beta_sd Positive scalar prior std dev for \code{beta} coefficients (default 1.0).
#' @param lambda_nc Positive scalar weight for the non-crossing penalty (larger is stricter).
#' @param adaptive_iq Logical; if TRUE (default), the IQ shrinkage rate lambda_iq2
#'   is learned from data via a Gamma prior. If FALSE, lambda_iq2_fixed is used.
#' @param lambda_iq2_a,lambda_iq2_b Positive shape/rate hyperparameters for the
#'   IQ shrinkage rate \eqn{\lambda_{iq}^2} (used when adaptive_iq = TRUE).
#'   Prior: \eqn{\lambda_{iq}^2 \sim \mathrm{Gamma}(a, b)}, mean = a/b.
#'   Effective penalty weight is \eqn{\sqrt{\lambda_{iq}^2}}.
#' @param lambda_iq2_fixed Positive scalar; fixed value for \eqn{\lambda_{iq}^2} when
#'   adaptive_iq = FALSE (default 1). Effective penalty = \eqn{\sqrt{\lambda_{iq2\_fixed}}}.
#' @param eps_rel Positive scalar "smoothing temperature" (dimensionless).
#' @param lambda_lasso2_a,lambda_lasso2_b Positive shape/rate hyperparameters for the
#'   global LASSO rate \eqn{\lambda} (used when adaptive_gamma = TRUE).
#' @param adaptive_gamma Logical; if TRUE (default), the global LASSO rate lambda_lasso2
#'   is learned from data via a Gamma prior. If FALSE, lambda_lasso2_fixed is used as a fixed value.
#' @param lambda_lasso2_fixed Positive scalar; fixed value for global LASSO rate when
#'   adaptive_gamma = FALSE (default 1).
#' @param log_flag Integer \code{0/1}. If 1, fit on \code{log(y)}.
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
#' @param map_tol_obj,map_tol_grad,map_tol_rel_grad,map_tol_param MAP optimizer tolerances.
#' @param map_iter Maximum iterations for MAP optimization.
#' @param fit_method One of "mcmc", "map_mcmc", or "map":
#'   \itemize{
#'     \item "mcmc": Estimators are posterior median from MCMC; posterior draws from MCMC.
#'     \item "map_mcmc": Estimators are MAP; posterior draws from MCMC (MAP used as init).
#'     \item "map": Estimators are MAP; posterior draws from Laplacian approximation.
#'   }
#' @param laplace_n_samples Number of samples for Laplacian approximation (when fit_method = "map").
#' @param laplace_noise_scale Scale factor for parameter perturbation in Laplacian approximation.
#' @param prior_gamma Prior type for gamma: "group_lasso", "lasso", "spike_slab", "het_group_lasso", "adaptive_lasso".
#' @param spike_sd,slab_sd,slab_pi_a,slab_pi_b Spike-and-slab hyperparameters.
#'
#' @return A list with components:
#'   \itemize{
#'     \item fit: stanfit object (NULL if fit_method = "map")
#'     \item map: MAP estimates (contains $par with parameter values)
#'     \item y, H, X: Input data and design matrices
#'     \item hessian: Hessian at MAP (if computed)
#'     \item fit_method: The estimation method used
#'     \item laplace_samples: Pre-generated Laplacian samples (if fit_method = "map")
#'   }
#'
#' @references
#' Jiang, L., Wang, H. J., & Bondell, H. D. (2013). Interquantile Shrinkage in Regression Models.
#' Journal of Computational and Graphical Statistics, 22(4), 970-986.
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
#
# --- Session-level cache for compiled Stan model ---
# The Stan code is a fixed string; only stan_data changes between calls.
# Compiling once per R session avoids redundant 30-60s compilations.
.bqq_stan_cache <- new.env(parent = emptyenv())

getModel <- function(y, taus, H = NULL, X = NULL, offset = NULL, w = 0,
                        alpha = 0.75, eps_w = 1e-6, c_sigma = 1.0,
                        beta_sd = 1.0,
                        lambda_nc = 2, eps_rel = 0.1,
                        adaptive_iq = TRUE,
                        lambda_iq2_a = 1, lambda_iq2_b = 0.1,
                        lambda_iq2_fixed = 1,
                        adaptive_gamma = TRUE,
                        lambda_lasso2_a = 1, lambda_lasso2_b = 0.05,
                        lambda_lasso2_fixed = 1,
                        log_flag = 0, jittering = 0,
                        chains = 1, iter = 1500, warmup = 500,
                        control = list(adapt_delta = 0.99),
                        seed = 123, verbose = FALSE,
                        map_hessian = TRUE,
                        map_tol_obj = 1e-12, map_tol_grad = 1e-8,
                        map_tol_rel_grad = 1e4, map_tol_param = 1e-8,
                        map_iter = 2000,
                        fit_method = c("mcmc", "map_mcmc", "map"),
                        laplace_n_samples = 1000,
                        laplace_noise_scale = 0.1,
                        prior_gamma = c("group_lasso", "lasso", "spike_slab",
                                        "het_group_lasso", "adaptive_lasso"),
                        spike_sd = 0.05, slab_sd = 2.0,
                        slab_pi_a = 1, slab_pi_b = 1) {

  prior_gamma <- match.arg(prior_gamma)
  fit_method  <- match.arg(fit_method)
  prior_code <- switch(
    prior_gamma,
    group_lasso        = 1L,
    lasso              = 2L,
    spike_slab         = 3L,
    het_group_lasso    = 4L,
    adaptive_lasso     = 5L
  )

  safe_gamma_weights <- function(y, H, tau, alpha, eps_w, lambda_lasso = NULL) {
    r <- ncol(H)
    if (r == 0) return(numeric(0))

    fit_q <- try(
      suppressWarnings(quantreg::rq(y ~ H - 1, tau = tau, method = "fn")),
      silent = TRUE
    )

    if (inherits(fit_q, "try-error")) {
      n <- length(y)
      if (is.null(lambda_lasso)) {
        lambda_lasso <- sqrt(log(r + 1L) / n)
      }
      fit_q <- try(
        suppressWarnings(
          quantreg::rq(y ~ H - 1, tau = tau,
                       method = "lasso", lambda = lambda_lasso)
        ),
        silent = TRUE
      )
      if (inherits(fit_q, "try-error")) {
        warning("Both rq(method = 'fn') and rq(method = 'lasso') failed; using w = 1 for this tau.")
        return(rep(1, r))
      }
    }

    gamma_hat <- as.numeric(stats::coef(fit_q))
    if (length(gamma_hat) != r) {
      warning("Pilot quantile fit returned a length mismatch; using w = 1 for this tau.")
      return(rep(1, r))
    }
    (abs(gamma_hat) + eps_w)^(-alpha)
  }

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
      int<lower=0> p;                  // predictors in eta (X)
      int<lower=2> m;                  // quantiles
      int<lower=0> r;                  // predictors in eta (H)

      matrix[n, p] X;                  // n x p
      matrix[n, r] H;                  // n x r
      vector[n] y;
      vector[n] offset;
      vector[m] tau_q;
      vector[m] mu0_init;

      real<lower=1e-12> base_scale;
      real<lower=0>      c_sigma;
      real<lower=0>      beta_sd;

      real<lower=0> lambda_nc;         // non-crossing penalty weight

      // Interquantile (fused lasso) shrinkage
      real<lower=0> lambda_iq2_a;      // Gamma prior shape for lambda_iq^2 (when adaptive_iq = 1)
      real<lower=0> lambda_iq2_b;      // Gamma prior rate  for lambda_iq^2 (when adaptive_iq = 1)
      int<lower=0, upper=1> adaptive_iq;  // 1 = data-adaptive, 0 = fixed
      real<lower=0> lambda_iq2_fixed;  // fixed value for lambda_iq^2 when adaptive_iq = 0

      real eps_rel;                      // smoothing temperature (dimensionless)

      real<lower=0> lambda_lasso2_a;
      real<lower=0> lambda_lasso2_b;
      int<lower=0, upper=1> adaptive_gamma;      // 1 = data-adaptive, 0 = fixed
      real<lower=0> lambda_lasso2_fixed;         // fixed value when adaptive_gamma = 0

      real<lower=0, upper = 1> jittering;
      real<lower=0, upper = 1> log_flag;

      // prior selector
      int<lower=1, upper=5> prior_code;

      // weights for lasso / adaptive lasso / hetero group lasso
      matrix[m, r] w_gamma;

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

      // Combined design Z = [X | H] (n x pr)
      int pr = p + r;
      matrix[n, pr] Z;
      {
        for (j in 1:p)
          for (i in 1:n)
            Z[i, j] = X[i, j];

        for (j in 1:r)
          for (i in 1:n)
            Z[i, p + j] = H[i, j];
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
      // Random-walk increments (non-centered)
      // z_incr and tau_rw removed: mu[q,t] = mu[q,t-1] = mu0[q]
      vector[m]       mu0;

      // X-coefficients
      matrix[m, p] beta;

      // H-coefficients
      matrix[m, r] gamma;

      // Group-level scale for group lasso (one per H column)
      vector<lower=0>[r] sigma2_gamma_group;

      // Element-wise local scales for lasso/adaptive lasso
      matrix<lower=0>[m, r] sigma2_gamma;

      // Global LASSO rate (learned when adaptive_gamma = 1)
      real<lower=0> lambda_lasso2;

      // IQ shrinkage rate squared (learned when adaptive_iq = 1)
      real<lower=0> lambda_iq2;

      // Spike-and-slab mixing weight
      real<lower=0, upper=1> pi_slab;

      // Group-level mixer for hetero group lasso (Levy)
      // One per time block (consistent with group lasso grouping)
      vector<lower=0>[r] omega_group;

      // jitter variable
      vector<lower=1e-12, upper = 1>[n] u;
  }

  transformed parameters {
      // RW paths
      matrix[m, n] mu;
      for (q in 1:m) {
        mu[q,1] = mu0[q];
        for (t in 2:n)
          mu[q,t] = mu[q,t-1];
      }

      // Smoothing temperature on data scale
      real<lower=1e-12> smooth_T = base_scale * eps_rel;

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

      // mu0 prior (no random walk innovation)
      mu0 ~ normal(mu0_init, 2 * base_scale);

      // beta prior
      if (p > 0) to_vector(beta) ~ normal(0, beta_sd);

      // ----- Score-based likelihood using Z = [X | H] with logit smoothing -----
      {
        if ((p + r) > 0) {
          matrix[pr, m] S;

          for (q in 1:m) {
            vector[pr] s_q = rep_vector(0.0, pr);
            for (i in 1:n) {
              real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
              real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;

              real eta = mu[q, i] + xb + hb + offset[i];
              real r_i = y_eff[i] - eta;

              real z  = fmin(20, fmax(-20, -r_i / smooth_T));
              real Ilt = inv_logit(z);
              real psi = tau_q[q] - Ilt;

              if (p > 0) s_q[1:p]      += to_vector(row(X, i)) * psi;
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

        // Determine effective lambda_lasso2 value
        real lambda_lasso2_eff;
        if (adaptive_gamma == 1) {
          // Data-adaptive: learn lambda_lasso2 from data via Gamma prior
          if (prior_code != 3) {
            lambda_lasso2 ~ gamma(lambda_lasso2_a, lambda_lasso2_b);
          }
          lambda_lasso2_eff = lambda_lasso2;
        } else {
          // Fixed: use the user-specified value
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

        // 2 = lasso or 5 = adaptive lasso
        } else if (prior_code == 2 || prior_code == 5) {
          for (j in 1:m) {
            for (i in 1:r) {
              sigma2_gamma[j, i] ~ exponential(0.5 * lambda_lasso2_eff * square(w_gamma[j, i]));
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
          real c_levy = lambda_lasso2_eff / 2;
          for (i in 1:r) {
            omega_group[i] ~ inv_gamma(0.5, 0.5 * c_levy);
            for (j in 1:m) {
              sigma2_gamma[j, i] ~ exponential(0.5 * square(omega_group[i] * w_gamma[j, i]));
              gamma[j, i] ~ normal(0, sqrt(sigma2_gamma[j, i]));
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
            real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
            real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;
            eta_row[q] = mu[q, i] + xb + hb + offset[i];
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
      // Applied to gamma and beta (NOT mu, as random walk should track quantile-specific evolution)
      {
        // Determine effective lambda_iq^2 value, then take sqrt for the penalty
        real lambda_iq2_eff;
        if (adaptive_iq == 1) {
          // Data-adaptive: learn lambda_iq^2 from data via Gamma prior
          lambda_iq2 ~ gamma(lambda_iq2_a, lambda_iq2_b);
          lambda_iq2_eff = lambda_iq2;
        } else {
          // Fixed: use the user-specified value
          lambda_iq2_eff = lambda_iq2_fixed;
        }
        real lambda_iq_eff = sqrt(lambda_iq2_eff);

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

          // Note: mu (random walk) is NOT penalized to allow quantile-specific evolution

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

  X0 <- matrix(1, nrow = n, ncol = 1)

  if (is.null(X)) {
    X <- X0
  } else {
    X <- cbind(X0, X)
  }
  p <- ncol(X)

  if (is.null(H)) {
    r <- 0
    H <- matrix(0, n, 0)
  } else {
    r <- ncol(H)
    if (r == 0) {
      H <- matrix(0, n, 0)
    }
  }

  # Base scale for smoothing (robust)
  base_scale <- max(1e-8, 1.4826 * stats::mad(y))

  # Initial mu0 by quantiles
  if (w > 0) {
    mu0_init <- stats::quantile(y[1:w], probs = taus)
  } else {
    mu0_init <- stats::quantile(y, probs = taus)
  }

  # ---- w_gamma construction ----
  w_gamma <- matrix(1, nrow = m, ncol = r)

  if (r > 0 && prior_gamma %in% c("het_group_lasso", "adaptive_lasso")) {
    if (!requireNamespace("quantreg", quietly = TRUE)) {
      stop("Package 'quantreg' is required for het_group_lasso / adaptive_lasso weights.")
    }

    for (q in seq_len(m)) {
      w_gamma[q, ] <- safe_gamma_weights(
        y   = y,
        H   = H,
        tau = taus[q],
        alpha  = alpha,
        eps_w  = eps_w
      )
    }

    # No median normalization: use raw adaptive weights per Zou (2006).
    # lambda_lasso2 (learned from data via Gamma prior) handles overall penalty scale.
  }

  # ---- IQ weight matrices construction (data-driven adaptive weights) ----
  p_slope <- max(p - 1L, 0L)
  w_iq_gamma <- matrix(1, nrow = m - 1L, ncol = max(r, 0L))
  w_iq_beta  <- matrix(1, nrow = m - 1L, ncol = max(p_slope, 0L))

  if ((r > 0 || p_slope > 0) && requireNamespace("quantreg", quietly = TRUE)) {
    Z_pilot <- if (r > 0) cbind(X, H) else X
    d_pilot <- ncol(Z_pilot)

    pilot_coefs <- matrix(NA, nrow = m, ncol = d_pilot)
    for (q in seq_len(m)) {
      pilot_coefs[q, ] <- safe_pilot_coefs(
        y = y, Z = Z_pilot, tau = taus[q], eps_w = eps_w
      )
    }

    if (r > 0) {
      gamma_pilot <- pilot_coefs[, (p + 1):(p + r), drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(r)) {
          diff_val <- abs(gamma_pilot[q, j] - gamma_pilot[q - 1, j])
          w_iq_gamma[q - 1, j] <- (diff_val + eps_w)^(-1)
        }
      }
      med_w_iq_gamma <- stats::median(w_iq_gamma)
      if (is.finite(med_w_iq_gamma) && med_w_iq_gamma > 0) {
        w_iq_gamma <- w_iq_gamma / med_w_iq_gamma
      }
    }

    if (p_slope > 0) {
      beta_pilot <- pilot_coefs[, 2:p, drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(p_slope)) {
          diff_val <- abs(beta_pilot[q, j] - beta_pilot[q - 1, j])
          w_iq_beta[q - 1, j] <- (diff_val + eps_w)^(-1)
        }
      }
      med_w_iq_beta <- stats::median(w_iq_beta)
      if (is.finite(med_w_iq_beta) && med_w_iq_beta > 0) {
        w_iq_beta <- w_iq_beta / med_w_iq_beta
      }
    }
  } else if ((r > 0 || p_slope > 0) && !requireNamespace("quantreg", quietly = TRUE)) {
    warning("Package 'quantreg' not available; using uniform IQ shrinkage weights.")
  }


  stan_data <- list(
    n = n, p = p, m = m, r = r,
    X = X, H = H,
    y = y, offset = offset, tau_q = taus,
    mu0_init = as.vector(mu0_init),
    base_scale = base_scale, c_sigma = c_sigma, beta_sd = beta_sd,
    lambda_nc = lambda_nc, eps_rel = eps_rel,
    lambda_iq2_a = lambda_iq2_a, lambda_iq2_b = lambda_iq2_b,
    adaptive_iq = as.integer(adaptive_iq),
    lambda_iq2_fixed = lambda_iq2_fixed,
    lambda_lasso2_a = lambda_lasso2_a, lambda_lasso2_b = lambda_lasso2_b,
    adaptive_gamma = as.integer(adaptive_gamma),
    lambda_lasso2_fixed = lambda_lasso2_fixed,
    log_flag = log_flag, jittering = jittering,
    prior_code = prior_code,
    w_gamma = if (r > 0) w_gamma else matrix(0, m, 0),
    spike_sd = spike_sd,
    slab_sd  = slab_sd,
    slab_pi_a = slab_pi_a,
    slab_pi_b = slab_pi_b,
    p_slope = as.integer(p_slope),
    w_iq_gamma = if (r > 0) w_iq_gamma else matrix(0, m - 1L, 0L),
    w_iq_beta  = if (p_slope > 0) w_iq_beta else matrix(0, m - 1L, 0L)
  )

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

  # Helper function to generate Laplace approximation samples from MAP
  #
  # The Hessian from rstan::optimizing() is on the UNCONSTRAINED parameter scale
  # and has dimensions k x k where k = number of raw parameters (NOT including

  # transformed parameters). par_map includes both raw + transformed parameters,
  # so length(par_map) > k.
  #
  # mu is a TRANSFORMED parameter (mu[q,t] = mu[q,t-1], mu[q,1] = mu0[q]).
  # Since mu[q,t] = mu0[q] for all t, we sample mu0 from the Hessian
  # and replicate across time points.
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

    # Indices within raw parameters for the components we need for eta
    z_incr_idx <- grep("^z_incr\\[", raw_par_names)
    tau_rw_idx <- grep("^tau_rw",    raw_par_names)
    mu0_idx    <- grep("^mu0\\[",    raw_par_names)
    beta_idx   <- grep("^beta\\[",   raw_par_names)
    gamma_idx  <- grep("^gamma\\[",  raw_par_names)
    eta_param_idx <- c(z_incr_idx, tau_rw_idx, mu0_idx, beta_idx, gamma_idx)

    # Also find mu (transformed parameter) indices for the heuristic fallback
    mu_tp_idx <- grep("^mu\\[", par_names)

    # Parse dimensions from parameter names
    z_incr_parsed <- if (length(z_incr_idx) > 0) parse_2d_idx(raw_par_names, z_incr_idx, "z_incr") else NULL
    mu0_parsed    <- if (length(mu0_idx) > 0) {
      # mu0 is a vector: mu0[1], mu0[2], ... - parse as 1D
      dims_str <- gsub("mu0\\[|\\]", "", raw_par_names[mu0_idx])
      list(idx = as.integer(dims_str))
    } else NULL
    tau_rw_parsed <- if (length(tau_rw_idx) > 0) {
      dims_str <- gsub("tau_rw\\[|\\]", "", raw_par_names[tau_rw_idx])
      list(idx = as.integer(dims_str))
    } else NULL
    beta_parsed   <- if (length(beta_idx) > 0) parse_2d_idx(raw_par_names, beta_idx, "beta") else NULL
    gamma_parsed  <- if (length(gamma_idx) > 0) parse_2d_idx(raw_par_names, gamma_idx, "gamma") else NULL
    mu_tp_parsed  <- if (length(mu_tp_idx) > 0) parse_2d_idx(par_names, mu_tp_idx, "mu") else NULL

    # --- Try proper Hessian-based Laplace approximation ---
    laplace_ok <- FALSE
    mu_array <- NULL
    beta_array <- NULL
    gamma_array <- NULL

    if (!is.null(hessian) && k > 0 && length(eta_param_idx) > 0) {
      laplace_result <- tryCatch({

        # Build unconstrained mean vector for the full raw parameter space
        # par_map[1:k] is on constrained scale; Hessian is on unconstrained scale
        # For tau_rw (<lower=0>): unconstrained = log(constrained)
        # For z_incr, mu0, beta, gamma (no bounds): unconstrained = constrained
        theta_unc_full <- as.numeric(par_map[1:k])
        theta_unc_full[tau_rw_idx] <- log(pmax(par_map[tau_rw_idx], 1e-10))

        # Also transform other constrained params for correct Hessian inversion:
        # sigma2_gamma_group (<lower=0>): log
        tgg_idx <- grep("^sigma2_gamma_group", raw_par_names)
        if (length(tgg_idx) > 0) theta_unc_full[tgg_idx] <- log(pmax(par_map[tgg_idx], 1e-10))
        # sigma2_gamma (<lower=0>): log
        tg_idx <- grep("^sigma2_gamma\\[", raw_par_names)
        if (length(tg_idx) > 0) theta_unc_full[tg_idx] <- log(pmax(par_map[tg_idx], 1e-10))
        # lambda_lasso2, lambda_iq2, omega_group (<lower=0>): log
        for (pat in c("^lambda_lasso2$", "^lambda_iq2$", "^omega_group")) {
          idx_tmp <- grep(pat, raw_par_names)
          if (length(idx_tmp) > 0) theta_unc_full[idx_tmp] <- log(pmax(par_map[idx_tmp], 1e-10))
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

        # Invert full Hessian to get posterior covariance on unconstrained scale
        H_neg <- -(hessian + t(hessian)) / 2  # ensure symmetry
        H_neg_reg <- H_neg + diag(1e-6, k)
        Sigma_full <- solve(H_neg_reg)

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

        # --- Map columns back to parameter blocks ---
        # Column offsets within samples_unc (follows the order of eta_param_idx)
        z_incr_cols <- seq_along(z_incr_idx)
        tau_rw_cols <- length(z_incr_idx) + seq_along(tau_rw_idx)
        mu0_cols    <- length(z_incr_idx) + length(tau_rw_idx) + seq_along(mu0_idx)
        beta_cols   <- length(z_incr_idx) + length(tau_rw_idx) + length(mu0_idx) + seq_along(beta_idx)
        gamma_cols  <- length(z_incr_idx) + length(tau_rw_idx) + length(mu0_idx) + length(beta_idx) + seq_along(gamma_idx)

        # --- Reconstruct mu from mu0 (mu[q,t] = mu0[q] for all t) ---
        m_q <- max(mu0_parsed$idx)
        # Get n_time from transformed mu parameter in par_map
        n_time <- if (!is.null(mu_tp_parsed)) max(mu_tp_parsed$col) else 1

        mu_arr <- array(NA, dim = c(n_samples, m_q, n_time))

        for (s in 1:n_samples) {
          # Extract mu0
          mu0_s <- numeric(m_q)
          for (i in seq_along(mu0_parsed$idx)) {
            mu0_s[mu0_parsed$idx[i]] <- samples_unc[s, mu0_cols[i]]
          }

          # mu[q,t] = mu0[q] for all t (no random walk)
          for (q in 1:m_q) {
            mu_arr[s, q, ] <- mu0_s[q]
          }
        }

        # --- Assemble beta and gamma arrays ---
        beta_arr <- if (length(beta_idx) > 0) {
          scatter_to_array(samples_unc[, beta_cols, drop = FALSE],
                           beta_parsed$row, beta_parsed$col,
                           max(beta_parsed$row), max(beta_parsed$col), n_samples)
        } else NULL

        gamma_arr <- if (length(gamma_idx) > 0) {
          scatter_to_array(samples_unc[, gamma_cols, drop = FALSE],
                           gamma_parsed$row, gamma_parsed$col,
                           max(gamma_parsed$row), max(gamma_parsed$col), n_samples)
        } else NULL

        list(mu = mu_arr, beta = beta_arr, gamma = gamma_arr)

      }, error = function(e) {
        warning("Hessian-based Laplace failed: ", conditionMessage(e),
                ". Falling back to heuristic noise.")
        NULL
      })

      if (!is.null(laplace_result)) {
        mu_array <- laplace_result$mu
        beta_array <- laplace_result$beta
        gamma_array <- laplace_result$gamma
        laplace_ok <- TRUE
      }
    }

    # --- Fallback: heuristic noise (when Hessian is unavailable or inversion fails) ---
    if (!laplace_ok) {

      mu_array <- if (length(mu_tp_idx) > 0) {
        m_mu <- max(mu_tp_parsed$row); n_mu <- max(mu_tp_parsed$col)
        mu_map <- matrix(NA, m_mu, n_mu)
        for (i in seq_along(mu_tp_idx)) mu_map[mu_tp_parsed$row[i], mu_tp_parsed$col[i]] <- par_map[mu_tp_idx[i]]
        mu_sd <- matrix(NA, m_mu, n_mu)
        for (q in 1:m_mu) {
          d_sd <- sd(diff(mu_map[q, ]), na.rm = TRUE)
          if (is.na(d_sd) || d_sd < 1e-6) d_sd <- 0.1
          mu_sd[q, ] <- d_sd * noise_scale
        }
        arr <- array(NA, dim = c(n_samples, m_mu, n_mu))
        for (s in 1:n_samples) arr[s, , ] <- mu_map + matrix(rnorm(m_mu * n_mu, 0, mu_sd), m_mu, n_mu)
        arr
      }

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

    list(mu = mu_array, beta = beta_array, gamma = gamma_array)
  }

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
    draws <- rstan::extract(fit, pars = c("mu", "beta", "gamma"))
    map_fit <- list(par = list())
    map_fit$par$mu <- apply(draws$mu, c(2, 3), median)
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
    if (!is.null(map_tol_obj))       opt_args$tol_obj       <- map_tol_obj
    if (!is.null(map_tol_grad))      opt_args$tol_grad      <- map_tol_grad
    if (!is.null(map_tol_rel_grad))  opt_args$tol_rel_grad  <- map_tol_rel_grad
    if (!is.null(map_tol_param))     opt_args$tol_param     <- map_tol_param
    if (!is.null(map_iter))          opt_args$iter          <- map_iter
    map_fit <- do.call(rstan::optimizing, opt_args)
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
    if (!is.null(map_tol_obj))       opt_args$tol_obj       <- map_tol_obj
    if (!is.null(map_tol_grad))      opt_args$tol_grad      <- map_tol_grad
    if (!is.null(map_tol_rel_grad))  opt_args$tol_rel_grad  <- map_tol_rel_grad
    if (!is.null(map_tol_param))     opt_args$tol_param     <- map_tol_param
    if (!is.null(map_iter))          opt_args$iter          <- map_iter
    map_fit <- do.call(rstan::optimizing, opt_args)
    map_fit$estimator <- "map"

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

  list(
    fit = fit,
    map = map_fit,
    y = y, H = H, X = X,
    hessian = hessian,
    fit_method = fit_method,
    laplace_samples = laplace_samples
  )
}
