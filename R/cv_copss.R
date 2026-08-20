#' Cross-Validation for Interquantile Shrinkage Model (getModel)
#'
#' COPSS-style order-preserved 2-fold CV for tuning hyperparameters
#' in the interquantile shrinkage quantile regression model. Two validation
#' losses are available: the exact unconstrained score-likelihood criterion
#' (default) and the average pinball (check) loss.
#'
#' @name cv_copss
NULL

#' Pinball (check) loss for quantile regression
#'
#' Computes the average pinball loss across observations and quantile levels.
#'
#' @param y_val Numeric vector of validation responses.
#' @param qhat Numeric matrix of predicted quantiles (n x m).
#' @param taus Numeric vector of quantile levels.
#' @return Scalar average pinball loss.
#' @keywords internal
pinball_loss <- function(y_val, qhat, taus) {
  n <- length(y_val)
  m <- length(taus)
  losses <- matrix(0, n, m)
  for (j in seq_len(m)) {
    u <- y_val - qhat[, j]
    losses[, j] <- u * (taus[j] - as.numeric(u < 0))
  }
  mean(losses)
}


#' Exact unconstrained score-likelihood loss for quantile regression
#'
#' Held-out version of the unrestricted score likelihood (manuscript Eq. (6)):
#' the quadratic form vec(S)' (Q kron G)^{-1} vec(S) / (2 n), where S stacks the
#' EXACT (unsmoothed) quantile scores psi_qt = tau_q - 1\{y_t < qhat_qt\} along
#' the design directions, Q[a,b] = min(tau_a, tau_b) - tau_a tau_b is the
#' quantile kernel (the covariance of the exact score, which is why no smoothing
#' is applied here), and G = Z'Z/n is the Gram of the evaluation fold's design.
#' Equals the negative held-out score log-likelihood up to the model's own 1/(2n)
#' scaling, so lower is better. The non-crossing indicator and all penalties are
#' deliberately excluded: validation evaluates the data-fit term only.
#'
#' Mirrors the Stan likelihood block of \code{getModel} (same Cholesky solves,
#' same 1e-8 Gram ridge), with the smoothed sigmoid replaced by the exact
#' indicator because no gradients are needed at validation.
#'
#' @param y_val Numeric vector of validation responses.
#' @param qhat Numeric matrix of predicted quantiles (n x m).
#' @param taus Numeric vector of quantile levels.
#' @param Z_val Design matrix of the evaluation fold (n x (p + r)), the same
#'   \code{[1 | X | H]} layout used in the Stan score.
#' @return Scalar loss (negative held-out score log-likelihood), or NA if the
#'   whitening factorization fails.
#' @keywords internal
score_loss <- function(y_val, qhat, taus, Z_val) {
  n <- length(y_val)
  m <- length(taus)
  psi <- matrix(0, n, m)
  for (j in seq_len(m)) {
    psi[, j] <- taus[j] - as.numeric(y_val < qhat[, j])
  }
  tryCatch({
    S <- crossprod(Z_val, psi)                       # (p+r) x m score matrix
    G <- crossprod(Z_val) / n
    diag(G) <- diag(G) + 1e-8                        # same tiny ridge as Stan
    Qk <- outer(taus, taus, pmin) - tcrossprod(taus)
    A <- forwardsolve(t(chol(G)), S)                 # L_G^{-1} S
    B <- forwardsolve(t(chol(Qk)), t(A))             # L_Q^{-1} (L_G^{-1} S)'
    0.5 * sum(B * B) / n                             # tr(S' G^-1 S Q^-1) / (2n)
  }, error = function(e) {
    warning("score_loss failed: ", e$message)
    NA_real_
  })
}


# Both losses are computed for every fit (the fits dominate the cost); `loss`
# only decides which one fills train_loss/val_loss and ranks the grid.
.bqq_cv_losses <- function(y_tr, eta_tr, y_val, eta_val, taus, Z_tr, Z_val) {
  list(
    train_pinball = pinball_loss(y_tr, eta_tr, taus),
    val_pinball   = pinball_loss(y_val, eta_val, taus),
    train_score   = score_loss(y_tr, eta_tr, taus, Z_tr),
    val_score     = score_loss(y_val, eta_val, taus, Z_val)
  )
}

.bqq_cv_losses_na <- function() {
  list(train_pinball = NA_real_, val_pinball = NA_real_,
       train_score = NA_real_, val_score = NA_real_)
}


# Tuning grids and base_args are matched to getModel() formals BY NAME, and any
# name that is not a formal is silently dropped when the argument list is built.
# That silence is dangerous after a rename: a grid still carrying the old
# `lambda_iq` column would tune nothing while appearing to run normally. Fail
# loudly instead, and point at the rename when that is what happened.
.bqq_check_tuning_names <- function(grid_names, base_names, formal_names) {
  unknown <- setdiff(c(grid_names, base_names), formal_names)
  unknown <- unknown[nzchar(unknown)]
  if (!length(unknown)) return(invisible(TRUE))
  hint <- ""
  if ("lambda_iq" %in% unknown) {
    hint <- paste0(
      "\n  `lambda_iq` was renamed to `lambda_iq2` and is now the SQUARED weight: ",
      "the effective fusion rate is sqrt(lambda_iq2).\n  Replace lambda_iq = v with ",
      "lambda_iq2 = v^2 to reproduce previous results.")
  }
  stop("Unknown tuning argument(s) for getModel(): ",
       paste(sQuote(unknown), collapse = ", "), hint, call. = FALSE)
}



#' Order-preserved 2-fold CV for getModel (MAP-only)
#'
#' Implements the COPSS-style split (odds vs evens) and evaluates a grid of
#' \code{lambda_nc} using MAP fits from \code{getModel}.
#' Scoring uses the validation loss selected by \code{loss}: the exact
#' unconstrained score-likelihood criterion (default; see \code{score_loss}) or
#' the average pinball (check) loss across all taus on the held-out fold.
#' IQ shrinkage uses its own penalty lambda_iq2 (adaptive or fixed).
#'
#' @param y Numeric vector of responses.
#' @param taus Numeric vector of quantile levels.
#' @param H,X Design matrices (already aligned with y).
#' @param w Integer; passed to \code{getModel}.
#' @param grid_lambda_nc Numeric vector of candidate \code{lambda_nc} (non-crossing penalty).
#' @param lambda_iq2 Squared IQ fusion weight passed to \code{getModel} (the effective
#'   rate is its square root). Used as the fixed value when \code{adaptive_iq = FALSE},
#'   or as the EM starting value when TRUE.
#' @param adaptive_iq Logical; whether each CV fit learns \eqn{\lambda_{iq}^2} by EM.
#'   Defaults to FALSE here, unlike \code{getModel}, because every EM iteration is a
#'   full refit: leaving it on multiplies the cost of the CV sweep by up to
#'   \code{iq_em_max_iter}. Set TRUE only if the IQ level must be retuned per fold.
#' @param iq_em_max_iter,iq_em_tol,iq_em_mc_tol EM controls forwarded to
#'   \code{getModel}; see there.
#' @param prior_beta Prior type for betaX (default "normal").
#' @param prior_gamma Prior type for gamma (default "spike_slab").
#' @param map_iter Maximum iterations for MAP optimization.
#' @param loss Validation loss used to rank the grid: \code{"score"} (default)
#'   is the exact unconstrained score-likelihood criterion of \code{score_loss}
#'   (exact indicator score, no smoothing, no penalties); \code{"pinball"}
#'   restores the previous average pinball (check) loss. Both losses are always
#'   computed and returned; \code{loss} decides which one fills
#'   \code{train_loss}/\code{val_loss} and sorts the result.
#' @param seed Random seed.
#' @param verbose Print progress messages.
#'
#' @return A data.frame of grid values and CV losses (lower is better), sorted
#'   by val_loss. \code{train_loss}/\code{val_loss} carry the selected loss;
#'   \code{train_pinball}/\code{val_pinball} and \code{train_score}/
#'   \code{val_score} report both criteria, and \code{cv_loss} records which
#'   one did the ranking.
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' y <- rnorm(n)
#' y[150:200] <- y[150:200] + 2
#' H <- getSustainedShift(n, l = 20, w = 30)
#' taus <- c(0.1, 0.5, 0.9)
#'
#' cv_result <- cv_copss_map(
#'   y = y, taus = taus, H = H, X = NULL, w = 30,
#'   grid_lambda_nc = c(5, 10, 20)
#' )
#' print(cv_result)
#' }
#'
#' @export
cv_copss_map <- function(y, taus, H, X = NULL, w,
                            grid_lambda_nc,
                            prior_beta = "normal",
                            prior_gamma = "spike_slab",
                            map_iter = 2000,
                            lambda_iq2 = 1,
                            adaptive_iq = FALSE,
                            iq_em_max_iter = 30,
                            iq_em_tol = 1e-3,
                            iq_em_mc_tol = 0.02,
                            loss = c("score", "pinball"),
                            seed = 123,
                            verbose = TRUE) {

  loss <- match.arg(loss)

  fit_and_score <- function(idx_train, idx_val, lnc) {
    y_tr <- y[idx_train]
    H_tr <- H[idx_train, , drop = FALSE]
    H_val <- H[idx_val, , drop = FALSE]
    X_tr <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])

    # Fit model using MAP
    fit <- tryCatch({
      getModel(
        y = y_tr, taus = taus, H = H_tr, X = X_tr, w = w,
        lambda_nc = lnc,
        lambda_iq2 = lambda_iq2,
        adaptive_iq = adaptive_iq,
        iq_em_max_iter = iq_em_max_iter,
        iq_em_tol = iq_em_tol,
        iq_em_mc_tol = iq_em_mc_tol,
        prior_beta = prior_beta,
        prior_gamma = prior_gamma,
        fit_method = "map",
        map_hessian = FALSE,
        map_iter = map_iter,
        seed = seed,
        verbose = FALSE
      )
    }, error = function(e) {
      # A wall-clock / CPU time-limit hit (from setTimeLimit in a caller) must
      # abort the whole CV rather than be silently downgraded to a skipped grid
      # point: the caller treats a timeout as an invalid run and redoes the
      # simulation from fresh data. Genuine numerical fit failures are still
      # tolerated (skip this grid point and continue).
      if (grepl("reached elapsed time limit|reached CPU time limit",
                conditionMessage(e))) stop(e)
      warning("Model fitting failed: ", e$message)
      return(NULL)
    })

    if (is.null(fit) || is.null(fit$map)) {
      return(.bqq_cv_losses_na())
    }

    par <- fit$map$par

    # Design used in Stan: X0 column of 1s plus user X (if any)
    X_tr_design <- {
      X0 <- matrix(1, nrow = length(idx_train), ncol = 1)
      out <- if (is.null(X_tr) || ncol(X_tr) == 0) X0 else cbind(X0, X_tr)
      storage.mode(out) <- "double"
      out
    }

    m <- length(taus)
    p <- ncol(X_tr_design)
    r <- ncol(H_tr)

    # Extract parameters
    beta_vec <- par[grep("^beta\\[", names(par))]
    gamma_vec <- par[grep("^gamma\\[", names(par))]

    # Handle empty parameters
    if (length(beta_vec) == 0) beta_vec <- rep(0, m * p)
    if (length(gamma_vec) == 0) gamma_vec <- rep(0, m * r)

    beta <- matrix(beta_vec, m, p, byrow = FALSE)
    gamma <- if (r > 0) matrix(gamma_vec, m, r, byrow = FALSE) else matrix(0, m, 0)

    # Training predictions: X*beta[q,] (incl. intercept) + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- as.numeric(X_tr_design %*% beta[j, ]) +
        if (r > 0) as.numeric(H_tr %*% gamma[j, ]) else 0
    }

    # Validation predictions: X*beta[q,] (incl. intercept) + H*gamma[q,]
    X_val_raw <- if (is.null(X)) NULL else as.matrix(X[idx_val, , drop = FALSE])
    X_val <- {
      X0 <- matrix(1, nrow = length(idx_val), ncol = 1)
      out <- if (is.null(X_val_raw) || ncol(X_val_raw) == 0) X0 else cbind(X0, X_val_raw)
      storage.mode(out) <- "double"
      out
    }

    n_val <- length(idx_val)

    eta_val <- matrix(0, n_val, m)
    for (j in seq_len(m)) {
      eta_val[, j] <- as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    .bqq_cv_losses(y[idx_train], eta_tr, y[idx_val], eta_val, taus,
                   Z_tr = cbind(X_tr_design, H_tr),
                   Z_val = cbind(X_val, H_val))
  }

  # COPSS split: odds vs evens
  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  # Create grid (1D: lambda_nc only; IQ shrinkage via adaptive/fixed lambda_iq2)
  grid <- data.frame(lambda_nc = grid_lambda_nc)
  grid$train_loss <- NA_real_
  grid$val_loss <- NA_real_
  grid$train_pinball <- NA_real_
  grid$val_pinball <- NA_real_
  grid$train_score <- NA_real_
  grid$val_score <- NA_real_

  tr_sel <- paste0("train_", loss)
  val_sel <- paste0("val_", loss)

  for (k in seq_len(nrow(grid))) {
    lnc <- grid$lambda_nc[k]

    # Fold 1: train on odd, validate on even
    a <- fit_and_score(idx_odd, idx_even, lnc)

    # Fold 2: train on even, validate on odd
    b <- fit_and_score(idx_even, idx_odd, lnc)

    grid$train_pinball[k] <- (a$train_pinball + b$train_pinball) / 2
    grid$val_pinball[k]   <- (a$val_pinball + b$val_pinball) / 2
    grid$train_score[k]   <- (a$train_score + b$train_score) / 2
    grid$val_score[k]     <- (a$val_score + b$val_score) / 2
    grid$train_loss[k] <- grid[[tr_sel]][k]
    grid$val_loss[k]   <- grid[[val_sel]][k]

    if (verbose) {
      msg <- sprintf(
        "[cv_copss] iter %d/%d | lambda_nc=%.2f | %s: train=(%.4f, %.4f) avg=%.4f | val=(%.4f, %.4f) avg=%.4f",
        k, nrow(grid), lnc, loss, a[[tr_sel]], b[[tr_sel]], grid$train_loss[k],
        a[[val_sel]], b[[val_sel]], grid$val_loss[k]
      )
      message(msg)
    }
  }

  # Sort by the selected validation loss
  grid <- grid[order(grid$val_loss), ]
  rownames(grid) <- NULL

  # Add best indicator
  grid$is_best <- FALSE
  grid$is_best[1] <- TRUE

  grid$cv_loss <- loss

  grid
}


#' General grid search CV for getModel
#'
#' More flexible version that accepts a data.frame grid of hyperparameters.
#'
#' @param y Numeric vector of responses.
#' @param taus Numeric vector of quantile levels.
#' @param H,X Design matrices.
#' @param w Integer warm-up period.
#' @param grid data.frame with columns for hyperparameters (lambda_nc, etc.)
#' @param base_args Named list of additional arguments passed to getModel.
#' @param loss Validation loss used to rank the grid: \code{"score"} (default)
#'   is the exact unconstrained score-likelihood criterion of \code{score_loss};
#'   \code{"pinball"} restores the previous average pinball (check) loss. Both
#'   are always computed and returned; \code{loss} decides which one fills
#'   \code{train_loss}/\code{val_loss} and sorts the result.
#' @param seed Random seed.
#' @param verbose Print progress.
#'
#' @return data.frame with grid and CV losses (see \code{cv_copss_map} for the
#'   loss columns).
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getSustainedShift(n, l = 20, w = 30)
#' grid <- data.frame(lambda_nc = c(2, 5, 10))
#' cv_result <- cv_copss_grid(y = y, taus = taus, H = H, w = 30, grid = grid)
#' }
#'
#' @export
cv_copss_grid <- function(y, taus, H, X = NULL, w, grid,
                              base_args = list(),
                              loss = c("score", "pinball"),
                              seed = 123,
                              verbose = TRUE) {

  loss <- match.arg(loss)

  # Get getModel formals for default filling
  gm_formals <- as.list(formals(getModel))
  .bqq_check_tuning_names(names(grid), names(base_args), names(gm_formals))

  fit_and_score <- function(idx_train, idx_val, row_args) {
    # Convert factors to character
    row_args <- lapply(row_args, function(x) if (is.factor(x)) as.character(x) else x)
    base_args_l <- lapply(base_args, function(x) if (is.factor(x)) as.character(x) else x)

    H_tr <- H[idx_train, , drop = FALSE]
    H_val <- H[idx_val, , drop = FALSE]

    # Build full argument list
    full_args <- gm_formals
    full_args["y"] <- list(y[idx_train])
    full_args["taus"] <- list(taus)
    full_args["H"] <- list(H_tr)
    X_tr <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])
    full_args["X"] <- list(X_tr)
    full_args["w"] <- list(w)
    full_args["fit_method"] <- list("map")
    full_args["map_hessian"] <- list(FALSE)
    full_args["seed"] <- list(seed)
    full_args["verbose"] <- list(FALSE)
    # getModel() defaults to adaptive_iq = TRUE, but each EM iteration is a FULL
    # refit; inheriting that here would multiply the cost of the whole CV sweep
    # by up to iq_em_max_iter. Default it off for tuning. Callers who really want
    # the IQ level relearned per fold can override via base_args or a grid column.
    full_args["adaptive_iq"] <- list(FALSE)

    # Override with base_args then row_args
    for (nm in names(base_args_l)) full_args[nm] <- list(base_args_l[[nm]])
    for (nm in names(row_args)) full_args[nm] <- list(row_args[[nm]])

    # Keep only valid arguments
    full_args <- full_args[intersect(names(full_args), names(gm_formals))]

    fit <- tryCatch({
      do.call(getModel, full_args)
    }, error = function(e) {
      # A wall-clock / CPU time-limit hit (from setTimeLimit in a caller) must
      # abort the whole CV rather than be silently downgraded to a skipped grid
      # point: the caller treats a timeout as an invalid run and redoes the
      # simulation from fresh data. Genuine numerical fit failures are still
      # tolerated (skip this grid point and continue).
      if (grepl("reached elapsed time limit|reached CPU time limit",
                conditionMessage(e))) stop(e)
      warning("Model fitting failed: ", e$message)
      return(NULL)
    })

    if (is.null(fit) || is.null(fit$map)) {
      return(.bqq_cv_losses_na())
    }

    par <- fit$map$par

    # Reconstruct design
    X_tr_design <- {
      X0 <- matrix(1, nrow = length(idx_train), ncol = 1)
      X_raw <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])
      out <- if (is.null(X_raw) || ncol(X_raw) == 0) X0 else cbind(X0, X_raw)
      storage.mode(out) <- "double"
      out
    }

    m <- length(taus)
    p <- ncol(X_tr_design)
    r <- ncol(H_tr)

    beta_vec <- par[grep("^beta\\[", names(par))]
    gamma_vec <- par[grep("^gamma\\[", names(par))]

    if (length(beta_vec) == 0) beta_vec <- rep(0, m * p)
    if (length(gamma_vec) == 0) gamma_vec <- rep(0, m * r)

    beta <- matrix(beta_vec, m, p, byrow = FALSE)
    gamma <- if (r > 0) matrix(gamma_vec, m, r, byrow = FALSE) else matrix(0, m, 0)

    # Training predictions: X*beta[q,] (incl. intercept) + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- as.numeric(X_tr_design %*% beta[j, ]) +
        if (r > 0) as.numeric(H_tr %*% gamma[j, ]) else 0
    }

    # Validation predictions: X*beta[q,] (incl. intercept) + H*gamma[q,]
    X_val_raw <- if (is.null(X)) NULL else as.matrix(X[idx_val, , drop = FALSE])
    X_val <- {
      X0 <- matrix(1, nrow = length(idx_val), ncol = 1)
      out <- if (is.null(X_val_raw) || ncol(X_val_raw) == 0) X0 else cbind(X0, X_val_raw)
      storage.mode(out) <- "double"
      out
    }

    n_val <- length(idx_val)

    eta_val <- matrix(0, n_val, m)
    for (j in seq_len(m)) {
      eta_val[, j] <- as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    .bqq_cv_losses(y[idx_train], eta_tr, y[idx_val], eta_val, taus,
                   Z_tr = cbind(X_tr_design, H_tr),
                   Z_val = cbind(X_val, H_val))
  }

  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  tr_sel <- paste0("train_", loss)
  val_sel <- paste0("val_", loss)

  train_pinballs <- val_pinballs <- numeric(nrow(grid))
  train_scores <- val_scores <- numeric(nrow(grid))

  for (k in seq_len(nrow(grid))) {
    row_args <- as.list(grid[k, , drop = FALSE])

    a <- fit_and_score(idx_odd, idx_even, row_args)
    b <- fit_and_score(idx_even, idx_odd, row_args)

    train_pinballs[k] <- (a$train_pinball + b$train_pinball) / 2
    val_pinballs[k]   <- (a$val_pinball + b$val_pinball) / 2
    train_scores[k]   <- (a$train_score + b$train_score) / 2
    val_scores[k]     <- (a$val_score + b$val_score) / 2

    if (verbose) {
      hp_str <- paste(
        sprintf("%s=%s", names(row_args), vapply(row_args, function(x) format(x, digits = 3), "")),
        collapse = ", "
      )
      msg <- sprintf(
        "[cv_copss_grid] iter %d/%d | %s | %s: train=%.4f | val=%.4f",
        k, nrow(grid), hp_str, loss,
        (a[[tr_sel]] + b[[tr_sel]]) / 2, (a[[val_sel]] + b[[val_sel]]) / 2
      )
      message(msg)
    }
  }

  out <- cbind(grid,
               train_loss = if (loss == "score") train_scores else train_pinballs,
               val_loss   = if (loss == "score") val_scores else val_pinballs,
               train_pinball = train_pinballs, val_pinball = val_pinballs,
               train_score = train_scores, val_score = val_scores,
               cv_loss = loss)
  out <- out[order(out$val_loss), ]
  rownames(out) <- NULL

  out
}


#' MCMC-based grid search CV for getModel
#'
#' Uses MCMC with shorter draws instead of MAP for hyperparameter selection.
#' This can improve results when MAP estimation is unreliable.
#'
#' @param y Numeric vector of responses.
#' @param taus Numeric vector of quantile levels.
#' @param H,X Design matrices.
#' @param w Integer warm-up period.
#' @param grid data.frame with columns for hyperparameters (lambda_nc, etc.)
#' @param base_args Named list of additional arguments passed to getModel.
#' @param mcmc_warmup Number of MCMC warmup iterations (default 200).
#' @param mcmc_draws Number of MCMC sampling iterations (default 300).
#' @param loss Validation loss used to rank the grid: \code{"score"} (default)
#'   is the exact unconstrained score-likelihood criterion of \code{score_loss};
#'   \code{"pinball"} restores the previous average pinball (check) loss. Both
#'   are always computed and returned; \code{loss} decides which one fills
#'   \code{train_loss}/\code{val_loss} and sorts the result.
#' @param seed Random seed.
#' @param verbose Print progress.
#'
#' @return data.frame with grid and CV losses (see \code{cv_copss_map} for the
#'   loss columns).
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 200
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getSustainedShift(n, l = 20, w = 30)
#' grid <- data.frame(lambda_nc = c(2, 5))
#' cv_result <- cv_copss_mcmc(y = y, taus = taus, H = H, w = 30, grid = grid,
#'                             mcmc_warmup = 100, mcmc_draws = 100)
#' }
#'
#' @export
cv_copss_mcmc <- function(y, taus, H, X = NULL, w, grid,
                              base_args = list(),
                              mcmc_warmup = 200,
                              mcmc_draws = 300,
                              loss = c("score", "pinball"),
                              seed = 123,
                              verbose = TRUE) {

  loss <- match.arg(loss)

  # Get getModel formals for default filling
  gm_formals <- as.list(formals(getModel))
  .bqq_check_tuning_names(names(grid), names(base_args), names(gm_formals))

  fit_and_score_mcmc <- function(idx_train, idx_val, row_args) {
    # Convert factors to character
    row_args <- lapply(row_args, function(x) if (is.factor(x)) as.character(x) else x)
    base_args_l <- lapply(base_args, function(x) if (is.factor(x)) as.character(x) else x)

    H_tr <- H[idx_train, , drop = FALSE]
    H_val <- H[idx_val, , drop = FALSE]

    # Build full argument list
    full_args <- gm_formals
    full_args["y"] <- list(y[idx_train])
    full_args["taus"] <- list(taus)
    full_args["H"] <- list(H_tr)
    X_tr <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])
    full_args["X"] <- list(X_tr)
    full_args["w"] <- list(w)

    # Use MCMC instead of MAP
    full_args["fit_method"] <- list("mcmc")
    full_args["mcmc_warmup"] <- list(mcmc_warmup)
    full_args["mcmc_draws"] <- list(mcmc_draws)
    full_args["mcmc_chains"] <- list(1)  # Single chain for speed
    full_args["mcmc_parallel_chains"] <- list(1)

    full_args["seed"] <- list(seed)
    full_args["verbose"] <- list(FALSE)
    # getModel() defaults to adaptive_iq = TRUE, but each EM iteration is a FULL
    # refit; inheriting that here would multiply the cost of the whole CV sweep
    # by up to iq_em_max_iter. Default it off for tuning. Callers who really want
    # the IQ level relearned per fold can override via base_args or a grid column.
    full_args["adaptive_iq"] <- list(FALSE)

    # Override with base_args then row_args
    for (nm in names(base_args_l)) full_args[nm] <- list(base_args_l[[nm]])
    for (nm in names(row_args)) full_args[nm] <- list(row_args[[nm]])

    # Keep only valid arguments
    full_args <- full_args[intersect(names(full_args), names(gm_formals))]

    fit <- tryCatch({
      do.call(getModel, full_args)
    }, error = function(e) {
      warning("MCMC fitting failed: ", e$message)
      return(NULL)
    })

    if (is.null(fit) || is.null(fit$map) || is.null(fit$map$par)) {
      return(.bqq_cv_losses_na())
    }

    # Point estimate: getModel() stores the MCMC estimator as the posterior
    # MEDIAN of the coefficients in fit$map$par (estimator == "posterior_median").
    # Use it directly so CV scoring uses the same estimator as detection and
    # change-time localization, rather than a posterior mean of the draws.
    par <- fit$map$par

    # Reconstruct design
    X_tr_design <- {
      X0 <- matrix(1, nrow = length(idx_train), ncol = 1)
      X_raw <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])
      out <- if (is.null(X_raw) || ncol(X_raw) == 0) X0 else cbind(X0, X_raw)
      storage.mode(out) <- "double"
      out
    }

    m <- length(taus)
    p <- ncol(X_tr_design)
    r <- ncol(H_tr)

    beta <- if (!is.null(par$beta)) matrix(as.numeric(par$beta), m, p) else matrix(0, m, p)
    gamma <- if (r > 0 && !is.null(par$gamma)) matrix(as.numeric(par$gamma), m, r) else matrix(0, m, 0)

    # Training predictions: X*beta[q,] (incl. intercept) + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- as.numeric(X_tr_design %*% beta[j, ]) +
        if (r > 0) as.numeric(H_tr %*% gamma[j, ]) else 0
    }

    # Validation predictions
    X_val_raw <- if (is.null(X)) NULL else as.matrix(X[idx_val, , drop = FALSE])
    X_val <- {
      X0 <- matrix(1, nrow = length(idx_val), ncol = 1)
      out <- if (is.null(X_val_raw) || ncol(X_val_raw) == 0) X0 else cbind(X0, X_val_raw)
      storage.mode(out) <- "double"
      out
    }

    n_val <- length(idx_val)

    eta_val <- matrix(0, n_val, m)
    for (j in seq_len(m)) {
      eta_val[, j] <- as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    .bqq_cv_losses(y[idx_train], eta_tr, y[idx_val], eta_val, taus,
                   Z_tr = cbind(X_tr_design, H_tr),
                   Z_val = cbind(X_val, H_val))
  }

  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  tr_sel <- paste0("train_", loss)
  val_sel <- paste0("val_", loss)

  train_pinballs <- val_pinballs <- numeric(nrow(grid))
  train_scores <- val_scores <- numeric(nrow(grid))

  for (k in seq_len(nrow(grid))) {
    row_args <- as.list(grid[k, , drop = FALSE])

    a <- fit_and_score_mcmc(idx_odd, idx_even, row_args)
    b <- fit_and_score_mcmc(idx_even, idx_odd, row_args)

    train_pinballs[k] <- (a$train_pinball + b$train_pinball) / 2
    val_pinballs[k]   <- (a$val_pinball + b$val_pinball) / 2
    train_scores[k]   <- (a$train_score + b$train_score) / 2
    val_scores[k]     <- (a$val_score + b$val_score) / 2

    if (verbose) {
      hp_str <- paste(
        sprintf("%s=%s", names(row_args), vapply(row_args, function(x) format(x, digits = 3), "")),
        collapse = ", "
      )
      msg <- sprintf(
        "[cv_copss_mcmc] iter %d/%d | %s | %s: train=%.4f | val=%.4f",
        k, nrow(grid), hp_str, loss,
        (a[[tr_sel]] + b[[tr_sel]]) / 2, (a[[val_sel]] + b[[val_sel]]) / 2
      )
      message(msg)
    }
  }

  out <- cbind(grid,
               train_loss = if (loss == "score") train_scores else train_pinballs,
               val_loss   = if (loss == "score") val_scores else val_pinballs,
               train_pinball = train_pinballs, val_pinball = val_pinballs,
               train_score = train_scores, val_score = val_scores,
               cv_loss = loss)
  out <- out[order(out$val_loss), ]
  rownames(out) <- NULL

  out
}


