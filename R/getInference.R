# Statistical Inference Functions for Interquantile Shrinkage Model
#
# Implements the Bayesian Predictive Quantile-Based Charting approach
# as described in the BQQ methodology.
#
# Two types of inference:
# 1. Predictive quantile-based inference (chi-squared tests on quantile vectors)
# 2. Predictive distributional statistics-based inference (location, scale, skew, kurtosis)

library(MASS)  # For mvrnorm

# =============================================================================
# Laplacian Approximation for MAP Inference
# =============================================================================

#' Generate posterior samples using Laplacian approximation from MAP fit
#'
#' Generates approximate posterior samples from a MAP fit using a two-tier strategy:
#'
#' **Tier 1 (Hessian-based Laplace approximation):** When a Hessian matrix is provided
#' and its inversion succeeds, the function constructs a standard Laplace approximation:
#' a multivariate normal centered at the MAP estimate with covariance equal to the
#' inverse of the negative Hessian. Sampling is performed on the unconstrained parameter
#' scale (log for lower-bounded parameters, logit for unit-bounded parameters) and
#' transformed back. A small ridge regularization (1e-6) is added for numerical
#' stability, and eigenvalues are clamped at 1e-8 to ensure positive definiteness.
#'
#' **Tier 2 (Heuristic fallback):** When the Hessian is unavailable (\code{NULL}) or
#' its inversion fails, the function falls back to a heuristic perturbation scheme:
#' each parameter is perturbed independently with standard deviation proportional to
#' \code{noise_scale} times the absolute MAP estimate (with small floors to prevent
#' degenerate samples). This is a numerical safeguard, not a formal Laplace approximation.
#'
#' Note: When using \code{getModel()} with \code{fit_method = "map"} and
#' \code{map_hessian = TRUE} (the default), the Hessian is computed automatically and
#' Tier 1 is used. Setting \code{map_hessian = FALSE} bypasses Hessian computation
#' and triggers the Tier 2 fallback.
#'
#' @param map_fit MAP optimization result from getModel (the $map element)
#' @param hessian Hessian matrix at MAP estimate. When provided and invertible,
#'   enables the standard Hessian-based Laplace approximation (Tier 1).
#'   When \code{NULL} or inversion fails, the heuristic fallback (Tier 2) is used.
#' @param n_samples Number of posterior samples to generate (default: 1000)
#' @param noise_scale Scale factor for parameter perturbation in the heuristic
#'   fallback (Tier 2 only). Default: 0.1. Ignored when Hessian-based Laplace succeeds.
#' @param seed Random seed for reproducibility
#' @return List with parameter samples in the same structure as rstan::extract()
#' @export
getLaplaceSamples <- function(map_fit, hessian = NULL, n_samples = 1000,
                              noise_scale = 0.1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  par_map <- map_fit$par
  par_names <- names(par_map)
  k <- if (!is.null(hessian)) nrow(hessian) else 0
  raw_par_names <- if (k > 0) par_names[1:k] else character(0)

  # Real Hessian-based Laplace only; the heuristic-perturbation fallback has been
  # removed package-wide (decision: no heuristic draws). Fit with map_hessian = TRUE.
  if (is.null(hessian) || k == 0)
    stop("getLaplaceSamples: a Hessian is required (fit with map_hessian = TRUE); ",
         "the heuristic fallback has been removed.", call. = FALSE)

  parse_2d <- function(names_vec, idx, prefix) {
    dims_str <- gsub(paste0(prefix, "\\[|\\]"), "", names_vec[idx])
    dims_split <- strsplit(dims_str, ",")
    list(
      row = as.integer(sapply(dims_split, `[`, 1)),
      col = as.integer(sapply(dims_split, `[`, 2))
    )
  }

  scatter <- function(samp_mat, parsed, n_samples, n_row = NULL, n_col = NULL) {
    nr <- if (is.null(n_row)) max(parsed$row) else n_row
    nc <- if (is.null(n_col)) max(parsed$col) else n_col
    arr <- array(NA, dim = c(n_samples, nr, nc))
    for (i in seq_along(parsed$row)) {
      arr[, parsed$row[i], parsed$col[i]] <- samp_mat[, i]
    }
    arr
  }

  parse_1d <- function(names_vec, idx, prefix) {
    dims_str <- gsub(paste0(prefix, "\\[|\\]"), "", names_vec[idx])
    as.integer(dims_str)
  }

  beta_idx_tp  <- grep("^beta\\[", par_names)
  gamma_idx_tp <- grep("^gamma\\[", par_names)

  beta0_idx  <- grep("^beta0\\[", raw_par_names)
  betaX_idx  <- grep("^betaX\\[", raw_par_names)
  gamma_idx  <- grep("^gamma\\[", raw_par_names)
  eta_param_idx <- c(beta0_idx, betaX_idx, gamma_idx)

  beta_tp_parsed  <- if (length(beta_idx_tp) > 0) parse_2d(par_names, beta_idx_tp, "beta") else NULL
  gamma_tp_parsed <- if (length(gamma_idx_tp) > 0) parse_2d(par_names, gamma_idx_tp, "gamma") else NULL

  beta0_parsed <- if (length(beta0_idx) > 0) parse_1d(raw_par_names, beta0_idx, "beta0") else NULL
  betaX_parsed <- if (length(betaX_idx) > 0) parse_2d(raw_par_names, betaX_idx, "betaX") else NULL
  gamma_parsed <- if (length(gamma_idx) > 0) parse_2d(raw_par_names, gamma_idx, "gamma") else NULL

  samples_unc <- NULL
  beta_array <- NULL
  gamma_array <- NULL
  if (!is.null(hessian) && k > 0 && length(eta_param_idx) > 0) {
    samples_unc <- tryCatch({
      theta_unc_full <- as.numeric(par_map[1:k])

      H_neg <- -(hessian + t(hessian)) / 2
      H_neg_reg <- H_neg + diag(1e-6, k)
      # Moore-Penrose pseudo-inverse: robust to the flat (zero-curvature)
      # directions of unused-prior scale latents, which make solve() report a
      # singular system. eta is decoupled from them, so its submatrix is exact.
      Sigma_full <- MASS::ginv(H_neg_reg)
      Sigma_unc <- Sigma_full[eta_param_idx, eta_param_idx, drop = FALSE]
      eig <- eigen(Sigma_unc, symmetric = TRUE)
      eig$values <- pmax(eig$values, 1e-8)
      L <- t(eig$vectors %*% diag(sqrt(eig$values)))
      theta_map_unc <- theta_unc_full[eta_param_idx]
      z_mat <- matrix(rnorm(n_samples * length(eta_param_idx)), n_samples, length(eta_param_idx))
      sweep(z_mat %*% L, 2, theta_map_unc, "+")
    }, error = function(e) {
      stop("getLaplaceSamples: Hessian-based Laplace failed: ",
           conditionMessage(e), call. = FALSE)
    })
  }
  if (is.null(samples_unc))
    stop("getLaplaceSamples: Laplace sampling produced no draws (degenerate Hessian ",
         "or no eta parameters).", call. = FALSE)

  beta0_cols  <- seq_along(beta0_idx)
  betaX_cols  <- length(beta0_idx) + seq_along(betaX_idx)
  gamma_cols  <- length(beta0_idx) + length(betaX_idx) + seq_along(gamma_idx)

  if (!is.null(samples_unc)) {
    beta_array <- if (length(beta0_idx) > 0 || length(betaX_idx) > 0) {
      m_beta <- max(c(beta0_parsed, if (!is.null(betaX_parsed)) betaX_parsed$row else integer(0)))
      p_beta <- 1L + if (!is.null(betaX_parsed)) max(betaX_parsed$col) else 0L
      arr <- array(0, dim = c(n_samples, m_beta, p_beta))
      if (length(beta0_idx) > 0) {
        for (i in seq_along(beta0_parsed)) {
          arr[, beta0_parsed[i], 1] <- samples_unc[, beta0_cols[i]]
        }
      }
      if (length(betaX_idx) > 0) {
        for (i in seq_along(betaX_parsed$row)) {
          arr[, betaX_parsed$row[i], betaX_parsed$col[i] + 1L] <- samples_unc[, betaX_cols[i]]
        }
      }
      arr
    } else NULL

    gamma_array <- if (length(gamma_idx) > 0) {
      scatter(samples_unc[, gamma_cols, drop = FALSE], gamma_parsed, n_samples)
    } else NULL
  } else {
    beta_array <- if (length(beta_idx_tp) > 0) {
      mb <- max(beta_tp_parsed$row)
      pb <- max(beta_tp_parsed$col)
      beta_map <- matrix(NA, mb, pb)
      for (i in seq_along(beta_idx_tp)) {
        beta_map[beta_tp_parsed$row[i], beta_tp_parsed$col[i]] <- par_map[beta_idx_tp[i]]
      }
      beta_sd <- pmax(abs(beta_map) * noise_scale, 0.05)
      arr <- array(NA, dim = c(n_samples, mb, pb))
      for (s in seq_len(n_samples)) {
        arr[s, , ] <- beta_map + matrix(rnorm(mb * pb, 0, beta_sd), mb, pb)
      }
      arr
    } else if (length(beta0_idx) > 0 || length(betaX_idx) > 0) {
      m_beta <- max(c(beta0_parsed, if (!is.null(betaX_parsed)) betaX_parsed$row else integer(0)))
      p_beta <- 1L + if (!is.null(betaX_parsed)) max(betaX_parsed$col) else 0L
      beta_map <- matrix(0, m_beta, p_beta)
      if (length(beta0_idx) > 0) beta_map[beta0_parsed, 1] <- par_map[beta0_idx]
      if (length(betaX_idx) > 0) {
        for (i in seq_along(betaX_parsed$row)) {
          beta_map[betaX_parsed$row[i], betaX_parsed$col[i] + 1L] <- par_map[betaX_idx[i]]
        }
      }
      beta_sd <- pmax(abs(beta_map) * noise_scale, 0.05)
      arr <- array(NA, dim = c(n_samples, m_beta, p_beta))
      for (s in seq_len(n_samples)) {
        arr[s, , ] <- beta_map + matrix(rnorm(m_beta * p_beta, 0, beta_sd), m_beta, p_beta)
      }
      arr
    } else NULL

    gamma_array <- if (length(gamma_idx_tp) > 0) {
      mg <- max(gamma_tp_parsed$row)
      rg <- max(gamma_tp_parsed$col)
      gamma_map <- matrix(NA, mg, rg)
      for (i in seq_along(gamma_idx_tp)) {
        gamma_map[gamma_tp_parsed$row[i], gamma_tp_parsed$col[i]] <- par_map[gamma_idx_tp[i]]
      }
      gamma_sd <- pmax(abs(gamma_map) * noise_scale, 0.02)
      arr <- array(NA, dim = c(n_samples, mg, rg))
      for (s in seq_len(n_samples)) {
        arr[s, , ] <- gamma_map + matrix(rnorm(mg * rg, 0, gamma_sd), mg, rg)
      }
      arr
    } else if (length(gamma_idx) > 0) {
      mg <- max(gamma_parsed$row)
      rg <- max(gamma_parsed$col)
      gamma_map <- matrix(NA, mg, rg)
      for (i in seq_along(gamma_idx)) {
        gamma_map[gamma_parsed$row[i], gamma_parsed$col[i]] <- par_map[gamma_idx[i]]
      }
      gamma_sd <- pmax(abs(gamma_map) * noise_scale, 0.02)
      arr <- array(NA, dim = c(n_samples, mg, rg))
      for (s in seq_len(n_samples)) {
        arr[s, , ] <- gamma_map + matrix(rnorm(mg * rg, 0, gamma_sd), mg, rg)
      }
      arr
    } else NULL

  }

  list(mu = NULL, beta = beta_array, gamma = gamma_array)
}


# =============================================================================
# Extract Predictive Quantiles
# =============================================================================

#' Extract eta (fitted quantiles) from posterior samples or Laplace approximation
#'
#' Automatically detects the fit_method used and extracts posterior samples accordingly:
#' - "mcmc" or "map_mcmc": Uses MCMC posterior draws from stanfit object
#' - "map": Uses pre-generated Laplacian approximation samples
#'
#' @param fit_result Full result from getModel()
#' @param H Design matrix for gamma coefficients. If \code{NULL}, uses \code{fit_result$H}.
#' @param X Design matrix for user beta covariates (excluding intercept). If \code{NULL},
#'   uses \code{fit_result$X}. For backward compatibility, a matrix that already includes
#'   the intercept is also accepted.
#' @param n_samples Number of samples for Laplace approximation (only used if fit_result
#'   doesn't have laplace_samples and needs to generate them)
#' @param seed Random seed (only used if generating new Laplace samples)
#' @return 3D array [iterations, quantiles, time]
#' @export
getEta <- function(fit_result, H = NULL, X = NULL, offset = NULL, n_samples = 1000, seed = NULL) {

  if (is.null(H)) {
    H <- fit_result$H
  }
  if (is.null(X)) {
    X <- fit_result$X
  }

  n <- if (!is.null(H)) nrow(H) else if (!is.null(X)) nrow(X) else length(fit_result$y)
  r <- if (!is.null(H)) ncol(H) else 0

  if (is.null(X)) {
    X <- matrix(0, n, 0)
  } else {
    X <- as.matrix(X)
  }

  # Offset defaults to zero
  if (is.null(offset)) {
    offset <- rep(0, n)
  }

  # Determine fit method (backward compatible)
  fit_method <- fit_result$fit_method
  if (is.null(fit_method)) {
    # Infer from available data for backward compatibility
    if (!is.null(fit_result$fit) && inherits(fit_result$fit, "stanfit")) {
      fit_method <- "mcmc"
    } else if (!is.null(fit_result$laplace_samples)) {
      fit_method <- "map"
    } else if (!is.null(fit_result$map)) {
      fit_method <- "map"
    } else {
      stop("Cannot determine fit_method from fit_result")
    }
  }

  # Extract posterior draws based on fit_method
  if (fit_method %in% c("mcmc", "map_mcmc")) {
    # Use MCMC samples
    if (is.null(fit_result$fit) || !inherits(fit_result$fit, "stanfit")) {
      stop("fit_method is '", fit_method, "' but no stanfit object found")
    }

    draws <- rstan::extract(fit_result$fit)
    n_iter <- dim(draws$beta)[1]
    m <- dim(draws$beta)[2]

    # Handle dimension collapse for beta
    beta_draws <- draws$beta
    if (length(dim(beta_draws)) == 2) {
      beta_draws <- array(beta_draws, dim = c(n_iter, m, 1))
    }

    # Handle dimension collapse for gamma
    gamma_draws <- NULL
    if (r > 0) {
      gamma_draws <- draws$gamma
      if (length(dim(gamma_draws)) == 2) {
        gamma_draws <- array(gamma_draws, dim = c(n_iter, m, 1))
      }
    }

  } else if (fit_method == "map") {
    # Use Laplacian approximation samples
    if (!is.null(fit_result$laplace_samples)) {
      # Use pre-generated samples from getModel
      laplace_samples <- fit_result$laplace_samples
    } else if (!is.null(fit_result$map)) {
      # Generate samples on the fly (backward compatibility)
      laplace_samples <- getLaplaceSamples(fit_result$map, fit_result$hessian,
                                           n_samples = n_samples, seed = seed)
    } else {
      stop("fit_method is 'map' but no MAP estimates or laplace_samples found")
    }

    n_iter <- dim(laplace_samples$beta)[1]
    m <- dim(laplace_samples$beta)[2]

    beta_draws <- laplace_samples$beta
    gamma_draws <- laplace_samples$gamma

  } else {
    stop("Unknown fit_method: ", fit_method)
  }

  p <- dim(beta_draws)[3]
  if (ncol(X) == p) {
    X_design <- X
  } else if (ncol(X) + 1 == p) {
    X_design <- cbind(Intercept = 1, X)
  } else if (p == 1 && ncol(X) == 0) {
    X_design <- matrix(1, n, 1)
  } else {
    stop("X has incompatible number of columns for the extracted beta coefficients.")
  }

  # Initialize eta array
  eta <- array(NA, dim = c(n_iter, m, n))

  for (s in 1:n_iter) {
    # Handle beta
    if (length(dim(beta_draws)) == 3) {
      beta_s <- beta_draws[s, , , drop = FALSE]
      beta_s <- matrix(beta_s, nrow = m, ncol = p)
    } else {
      beta_s <- matrix(beta_draws[s, ], nrow = m, ncol = p)
    }

    for (q in 1:m) {
      xb <- as.numeric(X_design %*% beta_s[q, ])
      hg <- 0
      if (r > 0 && !is.null(gamma_draws)) {
        if (length(dim(gamma_draws)) == 3) {
          gamma_s <- gamma_draws[s, , , drop = FALSE]
          gamma_s <- matrix(gamma_s, nrow = m, ncol = r)
        } else {
          gamma_s <- matrix(gamma_draws[s, ], nrow = m, ncol = r)
        }
        hg <- as.numeric(H %*% gamma_s[q, ])
      }
      # baseline (intercept) is column 1 of beta (beta0), already included in xb
      eta[s, q, ] <- xb + hg + offset
    }
  }

  eta
}


# =============================================================================
# Internal: coherent point estimates (MAP mode / posterior median)
# =============================================================================

# Coefficient point estimates beta (m x p) and gamma (m x r) from a getModel()
# fit, read from fit_result$map$par. getModel() already stores the coherent point
# estimate there for each fit method, so every consumer that routes through this
# helper uses one estimator:
#   - fit_method = "mcmc":              posterior MEDIAN of the coefficients
#                                       (map$estimator == "posterior_median")
#   - fit_method = "map" / "map_mcmc":  the MAP (posterior mode)
# This is the single source of truth: detection/localization (.bqq_point_eta),
# cross-validation (cv_copss), and the plots (plotBQQ) all read from it rather
# than averaging Laplace/MCMC draws (a posterior mean).
#
# map$par has two storage formats: a list with $beta/$gamma (fit_method %in%
# c("mcmc", "map_mcmc")), or a named vector "beta[i,j]"/"gamma[i,j]"
# (fit_method "map", optimizing with as_vector = TRUE).
.bqq_coefs <- function(fit_result, m, r = NULL) {
  par <- fit_result$map$par
  if (is.null(par)) stop("fit_result$map$par is missing")
  if (is.null(r)) r <- if (!is.null(fit_result$H)) ncol(fit_result$H) else 0L

  as_mat <- function(vec) {
    ij <- regmatches(names(vec), regexec("\\[([0-9]+),([0-9]+)\\]", names(vec)))
    rows <- as.integer(vapply(ij, function(z) z[2], character(1)))
    cols <- as.integer(vapply(ij, function(z) z[3], character(1)))
    M <- matrix(0, max(rows), max(cols))
    M[cbind(rows, cols)] <- as.numeric(vec)
    M
  }

  if (is.list(par)) {
    beta <- as.matrix(par$beta)
    gamma <- if (!is.null(par$gamma)) as.matrix(par$gamma) else matrix(0, m, 0)
  } else {
    bn <- grep("^beta\\[", names(par))
    gn <- grep("^gamma\\[", names(par))
    beta <- if (length(bn)) as_mat(par[bn]) else matrix(0, m, 1)
    gamma <- if (length(gn)) as_mat(par[gn]) else matrix(0, m, 0)
  }
  list(beta = beta, gamma = gamma)
}

# Build the m x n matrix of point-estimate fitted quantiles eta_hat[q, t] from a
# getModel() fit: eta_hat = X %*% beta + H %*% gamma, using the .bqq_coefs point
# estimate (MAP mode under MAP, posterior median under MCMC). Used for within-block
# change-time localization. offset defaults to 0, matching getEta().
.bqq_point_eta <- function(fit_result, taus) {
  H <- fit_result$H
  X <- fit_result$X
  m <- length(taus)

  n <- if (!is.null(H)) nrow(H) else if (!is.null(X)) nrow(as.matrix(X)) else length(fit_result$y)
  r <- if (!is.null(H)) ncol(H) else 0
  if (is.null(X)) X <- matrix(0, n, 0) else X <- as.matrix(X)
  offset <- rep(0, n)

  co <- .bqq_coefs(fit_result, m, r)
  beta <- co$beta
  gamma <- co$gamma

  p <- ncol(beta)
  if (ncol(X) == p) {
    X_design <- X
  } else if (ncol(X) + 1 == p) {
    X_design <- cbind(1, X)
  } else if (p == 1 && ncol(X) == 0) {
    X_design <- matrix(1, n, 1)
  } else {
    stop("X has incompatible number of columns for the point-estimate beta.")
  }

  eta_hat <- matrix(0, m, n)
  for (q in seq_len(m)) {
    xb <- as.numeric(X_design %*% beta[q, ])
    hg <- if (r > 0 && ncol(gamma) == r) as.numeric(H %*% gamma[q, ]) else 0
    eta_hat[q, ] <- xb + hg + offset
  }
  eta_hat
}


# =============================================================================
# Gamma-Based Change-Point Detection
# =============================================================================

# Shared engine for ONE basis (quantile or qss). Aligns the UI and Hotelling-T^2
# tests AT THE CELL LEVEL. `cv` is n_iter x (ncell*r), cell (c,j) at column
# (j-1)*ncell + c. cbar = posterior mean of each cell over the gamma draws.
# POSTERIOR WHITENING (Section 3.1): z_tilde = Sigma^{-1/2} cbar, the eigen-based (ZCA)
# inverse square root of the posterior covariance Sigma, so the cells are ~ N(0, I)
# (independent) under H0. The cell-specific statistic is z_tilde^2 (chi-square_1):
#   block   T^2_j = sum_{cells in block j} z_tilde^2   ~ chi-square_ncell
#   overall T^2   = max_j T^2_j  (Section 3.2), p^{T2} = 1 - (P(chi2_ncell <= T^2))^r
# The UI test uses the SAME whitened cells but aggregates by MAX instead of SUM:
#   block   UI_j  = max_{cells in block j} |z_tilde|,  overall UI = max over all cells.
# After whitening the cells are independent, so the per-block p-values are exact
# (chi-square for T^2; max-of-iid-normals for UI) and the block-level adjustment
# family {raw, Bonferroni, Holm, BH, calibrated} is analytic (no Monte Carlo).
# `z` (studentized cbar/sd) is retained for interpretable display only.
.bqq_block_tests <- function(cv, ncell, r, alpha, want_ui, want_t2, rowlab = NULL) {
  cbar <- colMeans(cv)
  S    <- stats::cov(cv)
  sdv  <- sqrt(pmax(diag(S), 1e-12))
  R    <- S / tcrossprod(sdv)               # correlation, kept only for the returned $R
  z_vec <- cbar / sdv                        # studentized cells (interpretable display only)
  # Posterior whitening (Section 3.1): z_tilde = Sigma^{-1/2} cbar, the eigen-based (ZCA)
  # inverse square root of the posterior covariance Sigma, so cells are ~ N(0, I)
  # (independent) under H0 and T^2 = sum z_tilde^2 = cbar' Sigma^-1 cbar is the EXACT
  # Hotelling statistic with exact chi-square / max-of-iid p-values. Moore-Penrose form
  # (ginv-style): drop the numerically-null eigen-directions instead of flooring them, so
  # a rank-deficient / collinear Sigma (e.g. the highly correlated quantile gammas) does
  # not blow up the whitening.
  eig <- eigen((S + t(S)) / 2, symmetric = TRUE)
  lam <- eig$values
  tol <- max(lam) * sqrt(.Machine$double.eps)
  inv_sqrt <- ifelse(lam > tol, 1 / sqrt(lam), 0)
  Sinvsqrt <- eig$vectors %*% (inv_sqrt * t(eig$vectors))   # V diag(inv_sqrt) V'
  zt  <- as.numeric(Sinvsqrt %*% cbar)
  z_mat    <- matrix(z_vec, ncell, r)
  zt_mat   <- matrix(zt,    ncell, r)
  cellstat <- zt_mat^2                      # cell-specific test: whitened squared z
  if (!is.null(rowlab))
    rownames(z_mat) <- rownames(zt_mat) <- rownames(cellstat) <- rowlab
  blk <- lapply(seq_len(r), function(j) ((j - 1L) * ncell + 1L):(j * ncell))

  out <- list(z = z_mat, z_white = zt_mat, cellstat = cellstat, R = R,
              overall_t2 = NA_real_, overall_t2_p = NA_real_,
              overall_ui = NA_real_, overall_ui_p = NA_real_,
              ui = NULL, hotelling_t2 = NULL)

  if (want_t2) {
    W  <- vapply(blk, function(ix) sum(zt[ix]^2), 0)               # block T^2 = sum whitened z^2
    p  <- stats::pchisq(W, df = ncell, lower.tail = FALSE)         # block posterior prob p_j^{T2} (Eq. 23)
    c_block <- stats::qchisq(1 - alpha, df = ncell)                # block charting constant c_j^{T2} (Eq. 23)
    c_ss <- stats::qchisq((1 - alpha)^(1 / r), df = ncell)         # full (process) charting constant c^{T2}:
                                                                   #   Sidak (1967) single-step over r blocks (Eq. 24)
    out$hotelling_t2 <- .bqq_adj_family(W, p, alpha, c_ss, c_block)
    out$overall_t2   <- max(W)                                     # overall T^2 = max_j T^2_j (Sec 3.2)
    out$overall_t2_p <- 1 - stats::pchisq(out$overall_t2, df = ncell)^r  # p^{T2}=1-(P(chi2_m<=T^2))^r (Eq. 24)
  }
  if (want_ui) {
    M  <- vapply(blk, function(ix) max(abs(zt[ix])), 0)            # block UI = max |whitened z|
    p  <- 1 - (2 * stats::pnorm(M) - 1)^ncell                      # block posterior prob p_j^{UI} (Eq. 21)
    c_block <- stats::qnorm((1 + (1 - alpha)^(1 / ncell)) / 2)     # block charting constant c_j^{UI} (Eq. 21)
    c_ss <- stats::qnorm((1 + (1 - alpha)^(1 / (ncell * r))) / 2)  # full (process) charting constant c^{UI}:
                                                                   #   order-statistic single-step over all cells (Eq. 22)
    out$ui <- .bqq_adj_family(M, p, alpha, c_ss, c_block)
    out$overall_ui   <- max(abs(zt))
    out$overall_ui_p <- 1 - (2 * stats::pnorm(out$overall_ui) - 1)^(ncell * r)  # p^{UI} (Eq. 22)
  }
  out
}

# Given a per-block statistic + per-block p-value, build the across-block adjustment
# family over the r blocks. `raw` uses the block posterior prob p_j vs the block charting
# constant c_block (Eqs. 21/23), with NO across-block adjustment; `Bonferroni`/`Holm`/`BH`
# adjust the block p-values; `calibrated` uses the full (process) single-step charting
# constant c_ss (Eqs. 22/24; the T^2 form is the Sidak (1967) correction) -- a block is
# flagged if its statistic exceeds c_ss.
.bqq_adj_family <- function(stat, p, alpha, c_ss, c_block = NA_real_) {
  ah <- stats::p.adjust(p, "holm")
  ab <- stats::p.adjust(p, "bonferroni")
  az <- stats::p.adjust(p, "BH")
  list(stat = stat, pvalue = p, adjp_holm = ah, adjp_bonf = ab, adjp_bh = az,
       c_block = c_block, c_calib = c_ss,
       raw   = which(p  < alpha),
       holm  = which(ah < alpha),
       bonf  = which(ab < alpha),
       bh    = which(az < alpha),
       calib = which(stat > c_ss))
}

#' Detect change points using gamma coefficients
#'
#' Uses the H-matrix gamma coefficients directly for change-point detection.
#' This approach is more aligned with the model design where gamma explicitly
#' represents shift effects at each block.
#'
#' Automatically detects the fit_method used and extracts gamma samples accordingly:
#' - "mcmc" or "map_mcmc": Uses MCMC posterior draws
#' - "map": Uses pre-generated Laplacian approximation samples
#'
#' @param fit_result Full result from getModel()
#' @param taus Quantile levels
#' @param l Block length (for converting H-column to observation)
#' @param w Warm-up period
#' @param signal_position Method to determine signal position within a significant block:
#'   - "first": First observation in the block (default)
#'   - "last": Last observation in the block
#'   - "middle": Middle observation in the block
#'   - "max_deviation": Observation with maximum deviation from the predictive median (fitted eta at tau = 0.5)
#'   - "pinball": Observation that splits the block to minimize the equally weighted pinball (check) loss between the pre-block and block fitted quantile vectors (the same loss used for cross-validation)
#' @param y Original data (required for signal_position = "max_deviation" or "pinball")
#' @param eta Predictive quantiles array (required for signal_position = "max_deviation" or "pinball")
#' @param laplace_n_samples Number of Laplace draws generated on the fly, used only
#'   for backward compatibility when \code{fit_result$laplace_samples} is absent
#'   (a normal MAP fit already carries the draws, so this is otherwise ignored).
#' @param alpha Significance level for all cell and block decisions (two-sided).
#' @param basis Character vector selecting which test family/families to compute:
#'   \code{"quantile"} (the raw quantile block-gammas) and/or \code{"qss"} (the four
#'   shape contrasts derived from those quantiles). Default is both; \code{"both"} is
#'   also accepted. The QSS family is simply the quantile family rotated into an
#'   interpretable, moment-aligned basis (L, S, Sk, K).
#' @param statistic Character vector selecting how a block's cells are combined into a
#'   block-level test. Both are built from the same posterior-whitened cell z-scores
#'   \eqn{\tilde z = \Sigma_\gamma^{-1/2}\bar\gamma} (Section 3.1, always applied): \code{"ui"}
#'   (default) is the union-intersection / max test \eqn{\max_k |\tilde z_k|}, and
#'   \code{"hotelling_t2"} is the sum \eqn{\sum_k \tilde z_k^2}. The cell statistic
#'   \eqn{\tilde z_k^2} sums to the block \eqn{T^2_j}; the overall \eqn{T^2 = \max_j T^2_j}
#'   with \eqn{p^{T^2} = 1 - (P(\chi^2_{ncell} \le T^2))^r} (Section 3.2; the across-block
#'   single-step is the Sidak (1967) correction). \code{"cell_max"}
#'   is a deprecated alias for \code{"ui"}. The legacy \code{calibrated}/\code{block_test}/
#'   \code{qss} flags are derived from \code{basis} and \code{statistic} unless passed
#'   explicitly.
#' @param seed Random seed (only used when generating new Laplace samples).
#' @param calibrated,block_test,qss DEPRECATED — use \code{basis} and \code{statistic},
#'   which express every test combination (\code{calibrated} = UI on the quantile
#'   basis, \code{block_test} = Hotelling \eqn{T^2} on the quantile basis,
#'   \code{qss} = the QSS shape-contrast family). When left \code{NULL} (default)
#'   they are derived from \code{basis} x \code{statistic}; passing them explicitly
#'   still overrides for now, with a deprecation warning, and will stop being
#'   accepted in a future version.
#' @return A list. \code{tests[[basis]]} (\code{"quantile"} / \code{"qss"}) carries
#'   \code{$z} (studentized cells), \code{$z_white} (whitened), \code{$cellstat}
#'   (whitened \eqn{\tilde z^2}), and \code{$ui} / \code{$hotelling_t2} — each with
#'   \code{stat}, \code{pvalue}, \code{adjp_holm/bonf/bh}, \code{c_block}, \code{c_calib}, and
#'   \code{raw/holm/bonf/bh/calib} over the r blocks — plus \code{$overall_ui(_p)}
#'   and \code{$overall_t2(_p)}. \code{detected_blocks} carries the significance flags
#'   for all four charts (\code{significant_*}, \code{significant_wald_*},
#'   \code{significant_qss_*}, \code{significant_qss_t2_*}). Flat aliases and
#'   \code{basis}/\code{statistic} are also returned.
#' @export
detectChangepoints_gamma <- function(fit_result, taus, l, w,
                                     signal_position = c("first", "last", "middle", "max_deviation", "pinball"),
                                     y = NULL, eta = NULL,
                                     laplace_n_samples = 1000, alpha = 0.05,
                                     basis = c("quantile", "qss"),
                                     statistic = c("ui", "hotelling_t2"),
                                     calibrated = NULL, block_test = NULL, qss = NULL,
                                     seed = NULL) {

  # Validate signal_position argument
  signal_position <- match.arg(signal_position)

  # Legacy test-selection flags: accepted with a warning for now (they override
  # the basis/statistic interface); scheduled for removal.
  if (!is.null(calibrated) || !is.null(block_test) || !is.null(qss)) {
    warning("'calibrated', 'block_test', and 'qss' are deprecated in ",
            "detectChangepoints_gamma(); use 'basis' and 'statistic' instead. ",
            "Explicit values still override for now but will be removed.",
            call. = FALSE)
  }

  # ---- Two-family / two-statistic API ---------------------------------------
  # `basis`     : which test family to run — "quantile" (the raw quantile gammas)
  #               and/or "qss" (the four shape contrasts). Default = both.
  # `statistic` : how cells are combined into a block-level test — "cell_max"
  #               (default; the max-type adjustment family, higher power on sparse
  #               single-moment shifts) and/or "hotelling_t2" (the omnibus quadratic
  #               alternative). Default = cell_max only.
  # The legacy flags calibrated/block_test/qss are DERIVED from these unless the
  # caller passes them explicitly (backward compatibility).
  if (length(basis) == 1L && identical(basis, "both")) basis <- c("quantile", "qss")
  basis     <- match.arg(basis, c("quantile", "qss"), several.ok = TRUE)
  statistic <- if (missing(statistic)) "ui" else statistic
  statistic[statistic == "cell_max"] <- "ui"          # backward-compat alias
  statistic <- match.arg(statistic, c("ui", "hotelling_t2"), several.ok = TRUE)
  want_ui <- "ui"           %in% statistic
  want_t2 <- "hotelling_t2" %in% statistic
  # Derive the legacy flags from basis x statistic unless the caller sets them.
  if (is.null(calibrated)) calibrated <- ("quantile" %in% basis) && want_ui
  if (is.null(block_test)) block_test <- ("quantile" %in% basis) && want_t2
  if (is.null(qss))        qss        <- ("qss" %in% basis)
  # Per-family / per-statistic gates (respect explicit legacy overrides).
  do_q_ui     <- isTRUE(calibrated)
  do_q_t2     <- isTRUE(block_test)
  do_quantile <- do_q_ui || do_q_t2
  do_qss_fam  <- isTRUE(qss)
  do_qss_ui   <- do_qss_fam && want_ui
  do_qss_t2   <- do_qss_fam && want_t2

  m <- length(taus)

  # Determine fit method (backward compatible)
  fit_method <- fit_result$fit_method
  if (is.null(fit_method)) {
    # Infer from available data
    if (!is.null(fit_result$fit) && inherits(fit_result$fit, "stanfit")) {
      fit_method <- "mcmc"
    } else if (!is.null(fit_result$laplace_samples)) {
      fit_method <- "map"
    } else if (!is.null(fit_result$map)) {
      fit_method <- "map"
    } else {
      stop("Cannot determine fit_method from fit_result")
    }
  }

  # Extract gamma samples based on fit_method
  if (fit_method %in% c("mcmc", "map_mcmc")) {
    if (is.null(fit_result$fit) || !inherits(fit_result$fit, "stanfit")) {
      stop("fit_method is '", fit_method, "' but no stanfit object found")
    }
    draws <- rstan::extract(fit_result$fit)
    gamma_samples <- draws$gamma
    if (length(dim(gamma_samples)) == 2) {
      gamma_samples <- array(gamma_samples, dim = c(dim(gamma_samples)[1], m, 1))
    }
  } else if (fit_method == "map") {
    if (!is.null(fit_result$laplace_samples)) {
      gamma_samples <- fit_result$laplace_samples$gamma
    } else if (!is.null(fit_result$map)) {
      # Backward compatibility: generate samples on the fly
      laplace <- getLaplaceSamples(fit_result$map, fit_result$hessian,
                                   n_samples = laplace_n_samples, seed = seed)
      gamma_samples <- laplace$gamma
    } else {
      stop("fit_method is 'map' but no laplace_samples or MAP estimates found")
    }
    if (is.null(gamma_samples)) {
      stop("No gamma coefficients found in laplace_samples")
    }
  } else {
    stop("Unknown fit_method: ", fit_method)
  }

  # Coherent point estimate of the fitted quantiles (m x n) for within-block
  # localization: MAP mode under fit_method = "map"/"map_mcmc", posterior median
  # under "mcmc" (see .bqq_point_eta). This replaces averaging the eta draws
  # (a posterior mean) for signal_position = "pinball"/"max_deviation".
  eta_point <- tryCatch(
    .bqq_point_eta(fit_result, taus),
    error = function(e) {
      warning("Could not build point-estimate fitted quantiles (",
              conditionMessage(e), "); localization falls back to the sample ",
              "mean of `eta` when provided.")
      NULL
    }
  )

  n_iter <- dim(gamma_samples)[1]
  r <- dim(gamma_samples)[3]  # number of H columns

  # ============================================================
  # Two families (quantile / qss) x two tests (ui / hotelling_t2), ALIGNED AT THE
  # CELL LEVEL. Each family's cells are studentized then globally whitened; the
  # cell-specific statistic is the whitened squared z (chi-square_1). Hotelling T^2
  # SUMS those squares (block and overall); the UI takes their MAX. Both feed the
  # same block-level adjustment family {raw, Bonferroni, Holm, BH, calibrated}.
  # See .bqq_block_tests().
  # ============================================================
  q_res <- NULL; qss_res <- NULL

  if (do_quantile) {
    cvq <- matrix(NA_real_, n_iter, m * r)          # cell (q,j) -> (j-1)*m + q
    for (j in seq_len(r)) cvq[, ((j - 1L) * m + 1L):(j * m)] <- gamma_samples[, , j]
    q_res <- .bqq_block_tests(cvq, m, r, alpha, do_q_ui, do_q_t2, format(taus))
  }

  if (do_qss_fam) {
    if (m < 5) {
      warning("qss basis requires the five-quantile grid ",
              "(0.025, 0.25, 0.5, 0.75, 0.975); skipping the QSS family.")
    } else {
      qi <- vapply(c(0.025, 0.25, 0.5, 0.75, 0.975),
                   function(t) which.min(abs(taus - t)), integer(1))
      # QSS shift statistics = linear contrasts of the block gammas (Appendix C):
      #   LS  = gamma_3                                  (location, median)
      #   ScS = gamma_4 - gamma_2                         (scale, IQR)
      #   SkS = gamma_2 - 2 gamma_3 + gamma_4             (skewness, Bowley numerator)
      #   KS  = gamma_5 - gamma_4 + gamma_2 - gamma_1     (kurtosis, tail excess beyond the quartiles)
      cvc <- matrix(NA_real_, n_iter, 4L * r)         # cell (c,j) -> (j-1)*4 + c
      for (j in seq_len(r)) {
        g <- gamma_samples[, , j]
        cvc[, ((j - 1L) * 4L + 1L):(j * 4L)] <- cbind(
          g[, qi[3]],                                                  # L : median
          g[, qi[4]] - g[, qi[2]],                                     # S : IQR
          g[, qi[4]] + g[, qi[2]] - 2 * g[, qi[3]],                    # Sk: Bowley numerator
          (g[, qi[5]] - g[, qi[4]]) + (g[, qi[2]] - g[, qi[1]]))       # K : tail excess (gamma_5-gamma_4+gamma_2-gamma_1)
      }
      qss_res <- .bqq_block_tests(cvc, 4L, r, alpha, do_qss_ui, do_qss_t2,
                                  c("L", "S", "Sk", "K"))
    }
  }

  # nested, fully-symmetric results: tests[[basis]] has $z, $z_white, $cellstat,
  # $ui, $hotelling_t2 (each with stat/pvalue/adjp_*/sig_*), and $overall_{ui,t2}(_p).
  keep <- c("z", "z_white", "cellstat", "ui", "hotelling_t2",
            "overall_ui", "overall_ui_p", "overall_t2", "overall_t2_p")
  tests <- list(quantile = if (!is.null(q_res)) q_res[keep] else NULL,
                qss      = if (!is.null(qss_res)) qss_res[keep] else NULL)

  # ---- flat aliases (NA / empty when a family or statistic is off) ----
  gu <- function(res, st, f, d) if (!is.null(res) && !is.null(res[[st]]) && !is.null(res[[st]][[f]])) res[[st]][[f]] else d
  gz <- function(res, f, d) if (!is.null(res) && !is.null(res[[f]])) res[[f]] else d
  ei <- integer(0); NAr <- rep(NA_real_, r)
  # quantile — cells
  z_raw    <- gz(q_res, "z", NULL); z_white <- gz(q_res, "z_white", NULL); cellstat <- gz(q_res, "cellstat", NULL)
  # quantile — UI
  ui_block          <- gu(q_res, "ui", "stat", NAr)
  pvalue_ui         <- gu(q_res, "ui", "pvalue", NAr)
  adjp_holm         <- gu(q_res, "ui", "adjp_holm", NULL)
  adjp_bonf         <- gu(q_res, "ui", "adjp_bonf", NULL)
  adjp_bh           <- gu(q_res, "ui", "adjp_bh", NULL)
  c_calib           <- gu(q_res, "ui", "c_calib", NA_real_)
  sig_blocks_raw    <- gu(q_res, "ui", "raw", ei)
  significant_holm  <- gu(q_res, "ui", "holm",ei)
  significant_bonf  <- gu(q_res, "ui", "bonf",ei)
  significant_bh    <- gu(q_res, "ui", "bh",ei)
  significant_calib <- gu(q_res, "ui", "calib",ei)
  # quantile — Hotelling T^2
  W_block                <- gu(q_res, "hotelling_t2", "stat", NAr)
  pvalue_wald            <- gu(q_res, "hotelling_t2", "pvalue", NAr)
  adjp_wald_holm         <- gu(q_res, "hotelling_t2", "adjp_holm", NULL)
  c_wald_calib           <- gu(q_res, "hotelling_t2", "c_calib", NA_real_)
  significant_wald_raw   <- gu(q_res, "hotelling_t2", "raw", ei)
  significant_wald_holm  <- gu(q_res, "hotelling_t2", "holm",ei)
  significant_wald_bonf  <- gu(q_res, "hotelling_t2", "bonf",ei)
  significant_wald_bh    <- gu(q_res, "hotelling_t2", "bh",ei)
  significant_wald_calib <- gu(q_res, "hotelling_t2", "calib",ei)
  overall_t2   <- gz(q_res, "overall_t2", NA_real_);   overall_t2_p <- gz(q_res, "overall_t2_p", NA_real_)
  overall_ui   <- gz(q_res, "overall_ui", NA_real_);   overall_ui_p <- gz(q_res, "overall_ui_p", NA_real_)
  # qss — cells
  z_qss    <- gz(qss_res, "z", NULL); z_white_qss <- gz(qss_res, "z_white", NULL); cellstat_qss <- gz(qss_res, "cellstat", NULL)
  qss_stat <- if (!is.null(z_qss)) max(abs(z_qss)) else NA_real_   # raw studentized global max
  # qss — UI
  ui_block_qss          <- gu(qss_res, "ui", "stat", NAr)
  pvalue_qss            <- gu(qss_res, "ui", "pvalue", NAr)
  adjp_qss_holm         <- gu(qss_res, "ui", "adjp_holm", NULL)
  adjp_qss_bonf         <- gu(qss_res, "ui", "adjp_bonf", NULL)
  adjp_qss_bh           <- gu(qss_res, "ui", "adjp_bh", NULL)
  c_qss                 <- gu(qss_res, "ui", "c_calib", NA_real_)
  significant_qss_raw   <- gu(qss_res, "ui", "raw", ei)
  significant_qss_holm  <- gu(qss_res, "ui", "holm",ei)
  significant_qss_bonf  <- gu(qss_res, "ui", "bonf",ei)
  significant_qss_bh    <- gu(qss_res, "ui", "bh",ei)
  significant_qss_calib <- gu(qss_res, "ui", "calib",ei)
  # qss — Hotelling T^2
  W_qss                    <- gu(qss_res, "hotelling_t2", "stat", NAr)
  pvalue_qss_t2            <- gu(qss_res, "hotelling_t2", "pvalue", NAr)
  adjp_qss_t2_holm         <- gu(qss_res, "hotelling_t2", "adjp_holm", NULL)
  c_qss_t2                 <- gu(qss_res, "hotelling_t2", "c_calib", NA_real_)
  significant_qss_t2_raw   <- gu(qss_res, "hotelling_t2", "raw", ei)
  significant_qss_t2_holm  <- gu(qss_res, "hotelling_t2", "holm",ei)
  significant_qss_t2_bonf  <- gu(qss_res, "hotelling_t2", "bonf",ei)
  significant_qss_t2_bh    <- gu(qss_res, "hotelling_t2", "bh",ei)
  significant_qss_t2_calib <- gu(qss_res, "hotelling_t2", "calib",ei)
  significant_qss_t2       <- significant_qss_t2_calib   # back-compat alias (calibrated member)
  overall_t2_qss   <- gz(qss_res, "overall_t2", NA_real_); overall_t2_qss_p <- gz(qss_res, "overall_t2_p", NA_real_)
  overall_ui_qss   <- gz(qss_res, "overall_ui", NA_real_); overall_ui_qss_p <- gz(qss_res, "overall_ui_p", NA_real_)

  # Convert H column to observation number.
  # For combined designs (e.g., sustained + isolated + drift), H has multiple
  # column groups sharing the same time blocks. Map h_col back to the actual
  # time block using modular arithmetic so observation indices stay within [1, n].
  n_data <- if (!is.null(y)) length(y) else NULL
  # blocks per design type, matching the design-matrix column count
  # (ceil((n-w)/l), as in getSustainedShift/Isolated/GradualDrift). Using floor
  # here previously wrapped the final block (h_col = r) back onto block 1 when
  # (n-w) was not a multiple of l, mislocating that block's signal observation.
  r_per_type <- if (!is.null(n_data)) as.integer((n_data - w) / l) + as.numeric((((n_data - w) %% l) > 0)) else r

  h_to_obs <- function(h_col) {
    # Map H column to actual time block (handles combined design)
    actual_block <- ((h_col - 1) %% r_per_type) + 1
    obs_start <- w + (actual_block - 1) * l + 1
    obs_end_raw <- w + actual_block * l
    obs_end <- if (!is.null(n_data)) min(obs_end_raw, n_data) else obs_end_raw
    c(obs_start, obs_end)
  }

  # Helper function to determine signal observation within a block
  get_signal_obs <- function(h_col, position_method, y_data = NULL, eta_data = NULL, tau_levels = NULL, eta_hat = NULL) {
    obs_range <- h_to_obs(h_col)
    obs_start <- obs_range[1]
    obs_end <- obs_range[2]

    if (position_method == "first") {
      return(obs_start)

    } else if (position_method == "last") {
      return(obs_end)

    } else if (position_method == "middle") {
      return(floor((obs_start + obs_end) / 2))

    } else if (position_method == "max_deviation") {
      # Find observation with maximum deviation from the predictive median
      # (point-estimate fitted eta at tau = 0.5). The point estimate is the MAP
      # mode (MAP) or posterior median (MCMC); fall back to the eta-draw mean
      # only if no point estimate is available.
      qmat <- if (!is.null(eta_hat)) eta_hat
              else if (!is.null(eta_data)) apply(eta_data, c(2, 3), mean)
              else NULL
      if (is.null(y_data) || is.null(qmat)) {
        warning("y and a fitted quantile estimate required for max_deviation; using 'first' instead")
        return(obs_start)
      }

      # Find the index of the median quantile (tau = 0.5)
      if (!is.null(tau_levels)) {
        median_idx <- which.min(abs(tau_levels - 0.5))
      } else {
        # Fallback to middle index if taus not provided
        median_idx <- ceiling(nrow(qmat) / 2)
      }

      eta_median <- qmat[median_idx, ]

      # Calculate deviations for observations in this block
      block_obs <- obs_start:obs_end
      deviations <- abs(y_data[block_obs] - eta_median[block_obs])

      # Return observation with maximum deviation
      max_dev_idx <- which.max(deviations)
      return(block_obs[max_dev_idx])

    } else if (position_method == "pinball") {
      # Refine the change time within the block by minimizing the equally
      # weighted pinball (check) loss: assign the pre-block fitted quantile
      # vector to observations before the split and the block's fitted quantile
      # vector at/after it, then pick the split with the lowest loss (the same
      # check loss used by cv_copss). The fitted quantiles use the coherent
      # point estimate (MAP mode under MAP, posterior median under MCMC); the
      # eta-draw mean is used only as a fallback when no point estimate exists.
      qhat <- if (!is.null(eta_hat)) eta_hat
              else if (!is.null(eta_data)) apply(eta_data, c(2, 3), mean)
              else NULL
      if (is.null(y_data) || is.null(qhat) || is.null(tau_levels)) {
        warning("y, a fitted quantile estimate, and taus required for pinball; using 'first' instead")
        return(obs_start)
      }
      if (obs_start < 2) return(obs_start)
      pre <- qhat[, obs_start - 1]; post <- qhat[, obs_start]
      block_obs <- obs_start:obs_end
      best_c <- obs_start; best_L <- Inf
      for (cc in block_obs) {
        L <- 0
        for (q in seq_along(tau_levels)) {
          qh <- ifelse(block_obs < cc, pre[q], post[q])
          u <- y_data[block_obs] - qh
          L <- L + sum(u * (tau_levels[q] - (u < 0)))
        }
        if (L < best_L) { best_L <- L; best_c <- cc }
      }
      return(best_c)
    }
  }

  # Compile results
  detected_blocks <- data.frame(
    h_col = 1:r,
    # Quantile basis - UI test (max of whitened |z|), full adjustment family
    significant_raw   = 1:r %in% sig_blocks_raw,
    significant_holm  = 1:r %in% significant_holm,
    significant_bonf  = 1:r %in% significant_bonf,
    significant_bh    = 1:r %in% significant_bh,
    significant_calib = 1:r %in% significant_calib,
    # Quantile basis - Hotelling T^2 (sum of whitened z^2), full adjustment family
    significant_wald_raw   = 1:r %in% significant_wald_raw,
    significant_wald_holm  = 1:r %in% significant_wald_holm,
    significant_wald_bonf  = 1:r %in% significant_wald_bonf,
    significant_wald_bh    = 1:r %in% significant_wald_bh,
    significant_wald_calib = 1:r %in% significant_wald_calib,
    # QSS basis - UI test
    significant_qss_raw   = 1:r %in% significant_qss_raw,
    significant_qss_holm  = 1:r %in% significant_qss_holm,
    significant_qss_bonf  = 1:r %in% significant_qss_bonf,
    significant_qss_bh    = 1:r %in% significant_qss_bh,
    significant_qss_calib = 1:r %in% significant_qss_calib,
    # QSS basis - Hotelling T^2
    significant_qss_t2_raw   = 1:r %in% significant_qss_t2_raw,
    significant_qss_t2_holm  = 1:r %in% significant_qss_t2_holm,
    significant_qss_t2_bonf  = 1:r %in% significant_qss_t2_bonf,
    significant_qss_t2_bh    = 1:r %in% significant_qss_t2_bh,
    significant_qss_t2_calib = 1:r %in% significant_qss_t2_calib
  )

  # Add observation ranges
  detected_blocks$obs_start <- sapply(detected_blocks$h_col, function(j) h_to_obs(j)[1])
  detected_blocks$obs_end <- sapply(detected_blocks$h_col, function(j) h_to_obs(j)[2])

  # Add signal observation based on signal_position method
  detected_blocks$signal_obs <- sapply(detected_blocks$h_col, function(j) {
    get_signal_obs(j, signal_position, y, eta, taus, eta_point)
  })

  # First detection (using signal_obs) for the quantile Holm and calibrated members.
  first_sig <- function(idx) if (length(idx) > 0) detected_blocks$signal_obs[min(idx)] else NA
  first_signal_holm  <- first_sig(significant_holm)
  first_signal_calib <- first_sig(significant_calib)

  list(
    detected_blocks = detected_blocks,
    # Nested, fully-symmetric results: tests[[basis]] carries $z (studentized),
    # $z_white (whitened), $cellstat (whitened z^2 = cell-level Hotelling), and
    # $ui / $hotelling_t2 (each: stat, pvalue, adjp_holm/bonf/bh, c_calib, sig_*),
    # plus $overall_ui(_p) and $overall_t2(_p).
    tests = tests,
    # ---- quantile basis (flat aliases) ----
    z_raw = z_raw, z_white = z_white, cellstat = cellstat,
    ui_block = ui_block, pvalue_ui = pvalue_ui,
    adjp_holm = adjp_holm, adjp_bonf = adjp_bonf, adjp_bh = adjp_bh, c_calib = c_calib,
    significant_raw = sig_blocks_raw, significant_holm = significant_holm,
    significant_bonf = significant_bonf, significant_bh = significant_bh,
    significant_calib = significant_calib,
    n_significant_holm = length(significant_holm), n_significant_calib = length(significant_calib),
    first_signal_holm = first_signal_holm, first_signal_calib = first_signal_calib,
    W_block = W_block, pvalue_wald = pvalue_wald, adjp_wald_holm = adjp_wald_holm,
    c_wald_calib = c_wald_calib,
    significant_wald_raw = significant_wald_raw, significant_wald_holm = significant_wald_holm,
    significant_wald_bonf = significant_wald_bonf, significant_wald_bh = significant_wald_bh,
    significant_wald_calib = significant_wald_calib,
    overall_ui = overall_ui, overall_ui_p = overall_ui_p,
    overall_t2 = overall_t2, overall_t2_p = overall_t2_p,
    # ---- qss basis (flat aliases) ----
    z_qss = z_qss, z_white_qss = z_white_qss, cellstat_qss = cellstat_qss, qss_stat = qss_stat,
    ui_block_qss = ui_block_qss, pvalue_qss = pvalue_qss,
    adjp_qss_holm = adjp_qss_holm, adjp_qss_bonf = adjp_qss_bonf, adjp_qss_bh = adjp_qss_bh, c_qss = c_qss,
    significant_qss_raw = significant_qss_raw, significant_qss_holm = significant_qss_holm,
    significant_qss_bonf = significant_qss_bonf, significant_qss_bh = significant_qss_bh,
    significant_qss_calib = significant_qss_calib,
    n_significant_qss_holm = length(significant_qss_holm), n_significant_qss_calib = length(significant_qss_calib),
    W_qss = W_qss, pvalue_qss_t2 = pvalue_qss_t2, adjp_qss_t2_holm = adjp_qss_t2_holm, c_qss_t2 = c_qss_t2,
    significant_qss_t2_raw = significant_qss_t2_raw, significant_qss_t2_holm = significant_qss_t2_holm,
    significant_qss_t2_bonf = significant_qss_t2_bonf, significant_qss_t2_bh = significant_qss_t2_bh,
    significant_qss_t2_calib = significant_qss_t2_calib,
    significant_qss_t2 = significant_qss_t2, n_significant_qss_t2 = length(significant_qss_t2),
    overall_ui_qss = overall_ui_qss, overall_ui_qss_p = overall_ui_qss_p,
    overall_t2_qss = overall_t2_qss, overall_t2_qss_p = overall_t2_qss_p,
    # ---- configuration ----
    signal_position = signal_position, alpha = alpha,
    basis = basis, statistic = statistic, taus = taus,
    calibrated = calibrated, block_test = block_test, qss = qss
  )
}


# =============================================================================
# QSS: Quantile Shape Statistics (Location, Scale, Skewness, Kurtosis)
# =============================================================================

#' Compute Quantile Shape Statistics (QSS) from predictive quantiles
#'
#' Using quintuple quantiles (tau = 0.05, 0.25, 0.5, 0.75, 0.95), computes:
#' - Location: L_t = Q_t(0.5) (median)
#' - Scale: S_t = Q_t(0.75) - Q_t(0.25) (IQR)
#' - Skewness: Sk_t = [(Q_t(0.75) - Q_t(0.5)) - (Q_t(0.5) - Q_t(0.25))] / IQR (Bowley)
#' - Kurtosis: K_t = [Q_t(0.975) - Q_t(0.025)] / IQR
#'
#' @param eta 3D array [iterations, quantiles, time] from getEta()
#' @param taus Vector of quantile levels (must include 0.025, 0.25, 0.5, 0.75, 0.975 or similar)
#' @return 3D array [iterations, 4, time] containing QSS statistics
#' @export
getQSS <- function(eta, taus = c(0.025, 0.25, 0.5, 0.75, 0.975)) {

  n_iter <- dim(eta)[1]
  m <- dim(eta)[2]
  n <- dim(eta)[3]

  if (m != 5) {
    warning("QSS expects 5 quantile levels. Attempting to use available quantiles.")
  }

  # Find indices for key quantiles
  # Assuming order: tau1 < tau2 < tau3 < tau4 < tau5
  idx_lo <- 1      # lowest (e.g., 0.025)
  idx_q1 <- 2      # first quartile (e.g., 0.25)
  idx_med <- 3     # median (e.g., 0.5)
  idx_q3 <- 4      # third quartile (e.g., 0.75)
  idx_hi <- 5      # highest (e.g., 0.975)

  # Initialize QSS array: [iterations, 4 stats, time]
  qss <- array(NA, dim = c(n_iter, 4, n))
  dimnames(qss) <- list(NULL, c("Location", "Scale", "Skewness", "Kurtosis"), NULL)

  for (s in 1:n_iter) {
    for (t in 1:n) {
      q_lo <- eta[s, idx_lo, t]
      q1 <- eta[s, idx_q1, t]
      med <- eta[s, idx_med, t]
      q3 <- eta[s, idx_q3, t]
      q_hi <- eta[s, idx_hi, t]

      iqr <- q3 - q1

      # Location (median)
      qss[s, 1, t] <- med

      # Scale (IQR)
      qss[s, 2, t] <- iqr

      # Skewness (Bowley coefficient)
      if (abs(iqr) > 1e-10) {
        qss[s, 3, t] <- ((q3 - med) - (med - q1)) / iqr
      } else {
        qss[s, 3, t] <- 0
      }

      # Kurtosis (tail weight ratio)
      if (abs(iqr) > 1e-10) {
        qss[s, 4, t] <- (q_hi - q_lo) / iqr
      } else {
        qss[s, 4, t] <- NA
      }
    }
  }

  qss
}


