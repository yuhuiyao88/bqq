# --- Session-level cache for compiled Stan model ---
# The Stan code is a fixed string; only stan_data changes between calls.
# Compiling once per R session avoids redundant 30-60s compilations.
.bqq_stan_cache <- new.env(parent = emptyenv())

#' Smoothed Quantile Regression with Interquantile Shrinkage (Stan)
#'
#' Fits a multi-quantile (\eqn{m}) regression model where the conditional
#' quantile function is modeled as a latent random walk in time (or index)
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
#'       \eqn{\eta_{qi} = \mu_{q,i} + \beta_{0,q} + x_i^\top \beta_{X,q} + h_i^\top \gamma_q + \mathrm{offset}_i}.
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
#'     Weights are applied separately to gamma and beta slopes.
#'     Falls back to uniform weights (all 1) when \pkg{quantreg} is not available.
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
#' @param X Optional numeric matrix \eqn{n \times p_x} of user-supplied covariates.
#'   An intercept column is added internally and assigned its own weakly informative prior.
#' @param offset Optional numeric vector of length \eqn{n} added to the linear predictor.
#' @param alpha Deprecated scalar retained for backward compatibility. It is no
#'   longer used in the \code{adaptive_lasso} or \code{het_group_lasso} prior
#'   construction.
#' @param eps_w Positive scalar added to pilot estimates for numerical stability
#'   in the IQ shrinkage weights (default 1e-6).
#' @param c_sigma Positive scalar scaling factor for the base scale (default 1.0).
#' @param beta0_sd Positive scalar prior std dev for the intercept coefficients
#'   \code{beta0} (default 1.0).
#' @param beta_sd Positive scalar prior std dev for \code{betaX} coefficients under
#'   \code{prior_beta = "normal"} (default 1.0).
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
#' @param prior_beta Prior type for \code{betaX}: \code{"normal"}, \code{"lasso"},
#'   \code{"spike_slab"}, \code{"group_lasso"}, \code{"het_group_lasso"}, or
#'   \code{"adaptive_lasso"}. The intercept \code{beta0} always retains the weakly
#'   informative normal prior specified by \code{beta0_sd}.
#' @param prior_gamma Prior type for gamma: \code{"group_lasso"}, \code{"lasso"},
#'   \code{"spike_slab"}, \code{"het_group_lasso"}, or \code{"adaptive_lasso"}.
#'   The \code{"adaptive_lasso"} option follows a Leng et al. (2014)-style hierarchy
#'   with coefficient-specific local shrinkage parameters. The
#'   \code{"het_group_lasso"} option combines group-level shrinkage with
#'   coefficient-level local scales, without pilot weights.
#' @param beta_spike_sd,beta_slab_sd,beta_slab_pi_a,beta_slab_pi_b Spike-and-slab
#'   hyperparameters for \code{betaX}.
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
                        beta0_sd = 1.0, beta_sd = 1.0,
                        lambda_nc = 2, eps_rel = 0.1,
                        adaptive_iq = TRUE,
                        lambda_iq2_a = 1, lambda_iq2_b = 0.1,
                        lambda_iq2_fixed = 1,
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
                        map_tol_obj = 1e-12, map_tol_grad = 1e-8,
                        map_tol_rel_grad = 1e4, map_tol_param = 1e-8,
                        map_iter = 2000,
                        fit_method = c("mcmc", "map_mcmc", "map"),
                        laplace_n_samples = 1000,
                        laplace_noise_scale = 0.1,
                        prior_beta = c("normal", "lasso", "spike_slab",
                                       "group_lasso", "het_group_lasso", "adaptive_lasso"),
                        prior_gamma = c("group_lasso", "lasso", "spike_slab",
                                        "het_group_lasso", "adaptive_lasso"),
                        beta_spike_sd = 0.05, beta_slab_sd = 2.0,
                        beta_slab_pi_a = 1, beta_slab_pi_b = 1,
                        spike_sd = 0.05, slab_sd = 2.0,
                        slab_pi_a = 1, slab_pi_b = 1) {

  prior_beta  <- match.arg(prior_beta)
  prior_gamma <- match.arg(prior_gamma)
  fit_method  <- match.arg(fit_method)
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
    adaptive_lasso     = 5L
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
      vector[m] mu0_init;

      real<lower=1e-12> base_scale;
      real<lower=0>      c_sigma;
      real<lower=0>      beta0_sd;
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

      real<lower=0> lambda_beta2_a;
      real<lower=0> lambda_beta2_b;
      int<lower=0, upper=1> adaptive_beta;       // 1 = data-adaptive, 0 = fixed
      real<lower=0> lambda_beta2_fixed;          // fixed value when adaptive_beta = 0

      real<lower=0, upper = 1> jittering;
      real<lower=0, upper = 1> log_flag;

      // prior selectors
      int<lower=1, upper=6> prior_beta_code;
      int<lower=1, upper=5> prior_code;

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
      // Random-walk increments (non-centered)
      // z_incr and tau_rw removed: mu[q,t] = mu[q,t-1] = mu0[q]
      vector[m]       mu0;

      // Intercept and user X-coefficients
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

      // IQ shrinkage rate squared (learned when adaptive_iq = 1)
      real<lower=0> lambda_iq2;

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
      // RW paths
      matrix[m, n] mu;
      matrix[m, px + 1] beta;
      for (q in 1:m) {
        mu[q,1] = mu0[q];
        for (t in 2:n)
          mu[q,t] = mu[q,t-1];
        beta[q, 1] = beta0[q];
        if (px > 0) {
          for (j in 1:px)
            beta[q, j + 1] = betaX[q, j];
        }
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

      // beta0 prior (intercept only)
      beta0 ~ normal(0, beta0_sd);

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

              real eta = mu[q, i] + xb + hb + offset[i];
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
        if (prior_code != 3) {
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

  # Base scale for smoothing (robust)
  base_scale <- max(1e-8, 1.4826 * stats::mad(y))

  # Initial mu0 by quantiles
  if (w > 0) {
    mu0_init <- stats::quantile(y[1:w], probs = taus)
  } else {
    mu0_init <- stats::quantile(y, probs = taus)
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
        y = y, Z = Z_pilot, tau = taus[q], eps_w = eps_w
      )
    }

    if (r > 0) {
      gamma_pilot <- pilot_coefs[, (p_total + 1):(p_total + r), drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(r)) {
          diff_val <- abs(gamma_pilot[q, j] - gamma_pilot[q - 1, j])
          w_iq_gamma[q - 1, j] <- (diff_val + eps_w)^(-1)
        }
      }
    }

    if (p_slope > 0) {
      beta_pilot <- pilot_coefs[, 2:p_total, drop = FALSE]
      for (q in 2:m) {
        for (j in seq_len(p_slope)) {
          diff_val <- abs(beta_pilot[q, j] - beta_pilot[q - 1, j])
          w_iq_beta[q - 1, j] <- (diff_val + eps_w)^(-1)
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
    mu0_init = as.vector(mu0_init),
    base_scale = base_scale, c_sigma = c_sigma, beta0_sd = beta0_sd, beta_sd = beta_sd,
    lambda_nc = lambda_nc, eps_rel = eps_rel,
    lambda_iq2_a = lambda_iq2_a, lambda_iq2_b = lambda_iq2_b,
    adaptive_iq = as.integer(adaptive_iq),
    lambda_iq2_fixed = lambda_iq2_fixed,
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
    beta0_idx  <- grep("^beta0\\[",  raw_par_names)
    betaX_idx  <- grep("^betaX\\[",  raw_par_names)
    gamma_idx  <- grep("^gamma\\[",  raw_par_names)
    eta_param_idx <- c(z_incr_idx, tau_rw_idx, mu0_idx, beta0_idx, betaX_idx, gamma_idx)

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
    beta0_parsed  <- if (length(beta0_idx) > 0) {
      dims_str <- gsub("beta0\\[|\\]", "", raw_par_names[beta0_idx])
      list(idx = as.integer(dims_str))
    } else NULL
    betaX_parsed  <- if (length(betaX_idx) > 0) parse_2d_idx(raw_par_names, betaX_idx, "betaX") else NULL
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
        # lambda_lasso2, lambda_beta2, lambda_iq2, omega_* (<lower=0>): log
        for (pat in c("^lambda_lasso2$", "^lambda_beta2$", "^lambda2_beta_local\\[", "^lambda2_gamma_local\\[", "^lambda_iq2$", "^omega_group", "^omega_beta_group")) {
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
        beta0_cols  <- length(z_incr_idx) + length(tau_rw_idx) + length(mu0_idx) + seq_along(beta0_idx)
        betaX_cols  <- length(z_incr_idx) + length(tau_rw_idx) + length(mu0_idx) + length(beta0_idx) + seq_along(betaX_idx)
        gamma_cols  <- length(z_incr_idx) + length(tau_rw_idx) + length(mu0_idx) + length(beta0_idx) + length(betaX_idx) + seq_along(gamma_idx)

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
