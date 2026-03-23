#' Cross-Validation for Interquantile Shrinkage Model (getModel)
#'
#' COPSS-style order-preserved 2-fold CV for tuning lambda_nc
#' in the interquantile shrinkage quantile regression model.
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


#' Order-preserved 2-fold CV for getModel (MAP-only)
#'
#' Implements the COPSS-style split (odds vs evens) and evaluates a grid of
#' \code{lambda_nc} using MAP fits from \code{getModel}.
#' Scoring uses the average pinball (check) loss across all taus on the held-out fold.
#' IQ shrinkage uses its own penalty lambda_iq2 (adaptive or fixed).
#'
#' @param y Numeric vector of responses.
#' @param taus Numeric vector of quantile levels.
#' @param H,X Design matrices (already aligned with y).
#' @param w Integer; passed to \code{getModel}.
#' @param grid_lambda_nc Numeric vector of candidate \code{lambda_nc} (non-crossing penalty).
#' @param eps_rel Smoothing temperature (default 0.1).
#' @param prior_gamma Prior type for gamma (default "group_lasso").
#' @param map_iter Maximum iterations for MAP optimization.
#' @param seed Random seed.
#' @param verbose Print progress messages.
#'
#' @return A data.frame of grid values and CV losses (lower is better), sorted by val_loss.
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
                            eps_rel = 0.1,
                            prior_gamma = "group_lasso",
                            map_iter = 2000,
                            seed = 123,
                            verbose = TRUE) {

  fit_and_score <- function(idx_train, idx_val, lnc) {
    y_tr <- y[idx_train]
    H_tr <- H[idx_train, , drop = FALSE]
    H_val <- H[idx_val, , drop = FALSE]
    X_tr <- if (is.null(X)) NULL else as.matrix(X[idx_train, , drop = FALSE])

    # Fit model using MAP
    fit <- tryCatch({
      getModel(
        y = y_tr, taus = taus, H = H_tr, X = X_tr, w = w,
        lambda_nc = lnc, eps_rel = eps_rel,
        prior_gamma = prior_gamma,
        fit_method = "map",
        map_hessian = FALSE,
        map_iter = map_iter,
        seed = seed,
        verbose = FALSE
      )
    }, error = function(e) {
      warning("Model fitting failed: ", e$message)
      return(NULL)
    })

    if (is.null(fit) || is.null(fit$map)) {
      return(list(train = NA, val = NA))
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
    mu0_vec <- par[grep("^mu0\\[", names(par))]

    # Handle empty parameters
    if (length(beta_vec) == 0) beta_vec <- rep(0, m * p)
    if (length(gamma_vec) == 0) gamma_vec <- rep(0, m * r)

    beta <- matrix(beta_vec, m, p, byrow = FALSE)
    gamma <- if (r > 0) matrix(gamma_vec, m, r, byrow = FALSE) else matrix(0, m, 0)
    mu0 <- as.numeric(mu0_vec)  # length m

    # Training predictions: mu0[q] + X*beta[q,] + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- mu0[j] +
        as.numeric(X_tr_design %*% beta[j, ]) +
        if (r > 0) as.numeric(H_tr %*% gamma[j, ]) else 0
    }

    # Validation predictions: mu0[q] + X*beta[q,] + H*gamma[q,]
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
      eta_val[, j] <- mu0[j] +
        as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    list(
      train = pinball_loss(y[idx_train], eta_tr, taus),
      val   = pinball_loss(y[idx_val], eta_val, taus)
    )
  }

  # COPSS split: odds vs evens
  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  # Create grid (1D: lambda_nc only; IQ shrinkage via adaptive/fixed lambda_iq2)
  grid <- data.frame(lambda_nc = grid_lambda_nc)
  grid$train_loss <- NA_real_
  grid$val_loss <- NA_real_

  for (k in seq_len(nrow(grid))) {
    lnc <- grid$lambda_nc[k]

    # Fold 1: train on odd, validate on even
    a <- fit_and_score(idx_odd, idx_even, lnc)

    # Fold 2: train on even, validate on odd
    b <- fit_and_score(idx_even, idx_odd, lnc)

    grid$train_loss[k] <- (a$train + b$train) / 2
    grid$val_loss[k] <- (a$val + b$val) / 2

    if (verbose) {
      msg <- sprintf(
        "[cv_copss] iter %d/%d | lambda_nc=%.2f | train=(%.4f, %.4f) avg=%.4f | val=(%.4f, %.4f) avg=%.4f",
        k, nrow(grid), lnc, a$train, b$train, grid$train_loss[k],
        a$val, b$val, grid$val_loss[k]
      )
      message(msg)
    }
  }

  # Sort by validation loss
  grid <- grid[order(grid$val_loss), ]
  rownames(grid) <- NULL

  # Add best indicator
  grid$is_best <- FALSE
  grid$is_best[1] <- TRUE

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
#' @param grid data.frame with columns for hyperparameters (lambda_nc, eps_rel, etc.)
#' @param base_args Named list of additional arguments passed to getModel.
#' @param seed Random seed.
#' @param verbose Print progress.
#'
#' @return data.frame with grid and CV losses.
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
                              seed = 123,
                              verbose = TRUE) {

  # Get getModel formals for default filling
  gm_formals <- as.list(formals(getModel))

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

    # Override with base_args then row_args
    for (nm in names(base_args_l)) full_args[nm] <- list(base_args_l[[nm]])
    for (nm in names(row_args)) full_args[nm] <- list(row_args[[nm]])

    # Keep only valid arguments
    full_args <- full_args[intersect(names(full_args), names(gm_formals))]

    fit <- tryCatch({
      do.call(getModel, full_args)
    }, error = function(e) {
      warning("Model fitting failed: ", e$message)
      return(NULL)
    })

    if (is.null(fit) || is.null(fit$map)) {
      return(list(train = NA, val = NA))
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
    mu0_vec <- par[grep("^mu0\\[", names(par))]

    if (length(beta_vec) == 0) beta_vec <- rep(0, m * p)
    if (length(gamma_vec) == 0) gamma_vec <- rep(0, m * r)

    beta <- matrix(beta_vec, m, p, byrow = FALSE)
    gamma <- if (r > 0) matrix(gamma_vec, m, r, byrow = FALSE) else matrix(0, m, 0)
    mu0 <- as.numeric(mu0_vec)  # length m

    # Training predictions: mu0[q] + X*beta[q,] + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- mu0[j] +
        as.numeric(X_tr_design %*% beta[j, ]) +
        if (r > 0) as.numeric(H_tr %*% gamma[j, ]) else 0
    }

    # Validation predictions: mu0[q] + X*beta[q,] + H*gamma[q,]
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
      eta_val[, j] <- mu0[j] +
        as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    list(
      train = pinball_loss(y[idx_train], eta_tr, taus),
      val   = pinball_loss(y[idx_val], eta_val, taus)
    )
  }

  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  train_losses <- numeric(nrow(grid))
  val_losses <- numeric(nrow(grid))

  for (k in seq_len(nrow(grid))) {
    row_args <- as.list(grid[k, , drop = FALSE])

    a <- fit_and_score(idx_odd, idx_even, row_args)
    b <- fit_and_score(idx_even, idx_odd, row_args)

    train_losses[k] <- (a$train + b$train) / 2
    val_losses[k] <- (a$val + b$val) / 2

    if (verbose) {
      hp_str <- paste(
        sprintf("%s=%s", names(row_args), vapply(row_args, function(x) format(x, digits = 3), "")),
        collapse = ", "
      )
      msg <- sprintf(
        "[cv_copss_grid] iter %d/%d | %s | train=%.4f | val=%.4f",
        k, nrow(grid), hp_str, train_losses[k], val_losses[k]
      )
      message(msg)
    }
  }

  out <- cbind(grid, train_loss = train_losses, val_loss = val_losses)
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
#' @param grid data.frame with columns for hyperparameters (lambda_nc, eps_rel, etc.)
#' @param base_args Named list of additional arguments passed to getModel.
#' @param mcmc_warmup Number of MCMC warmup iterations (default 200).
#' @param mcmc_draws Number of MCMC sampling iterations (default 300).
#' @param seed Random seed.
#' @param verbose Print progress.
#'
#' @return data.frame with grid and CV losses.
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
                              seed = 123,
                              verbose = TRUE) {

  # Get getModel formals for default filling
  gm_formals <- as.list(formals(getModel))

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

    if (is.null(fit) || is.null(fit$draws)) {
      return(list(train = NA, val = NA))
    }

    # Extract posterior means from MCMC draws
    draws <- fit$draws

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

    # Get posterior means from draws
    beta_vars <- grep("^beta\\[", colnames(draws), value = TRUE)
    gamma_vars <- grep("^gamma\\[", colnames(draws), value = TRUE)
    mu0_vars <- grep("^mu0\\[", colnames(draws), value = TRUE)

    beta_vec <- if (length(beta_vars) > 0) colMeans(draws[, beta_vars, drop = FALSE]) else rep(0, m * p)
    gamma_vec <- if (length(gamma_vars) > 0) colMeans(draws[, gamma_vars, drop = FALSE]) else rep(0, m * r)
    mu0_vec <- if (length(mu0_vars) > 0) colMeans(draws[, mu0_vars, drop = FALSE]) else rep(0, m)

    beta <- matrix(beta_vec, m, p, byrow = FALSE)
    gamma <- if (r > 0) matrix(gamma_vec, m, r, byrow = FALSE) else matrix(0, m, 0)
    mu0 <- as.numeric(mu0_vec)  # length m

    # Training predictions: mu0[q] + X*beta[q,] + H*gamma[q,]
    eta_tr <- matrix(0, nrow = length(idx_train), ncol = m)
    for (j in seq_len(m)) {
      eta_tr[, j] <- mu0[j] +
        as.numeric(X_tr_design %*% beta[j, ]) +
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
      eta_val[, j] <- mu0[j] +
        as.numeric(X_val %*% beta[j, ]) +
        if (r > 0) as.numeric(H_val %*% gamma[j, ]) else 0
    }

    list(
      train = pinball_loss(y[idx_train], eta_tr, taus),
      val   = pinball_loss(y[idx_val], eta_val, taus)
    )
  }

  idx_odd  <- seq(1, length(y), by = 2)
  idx_even <- seq(2, length(y), by = 2)

  train_losses <- numeric(nrow(grid))
  val_losses <- numeric(nrow(grid))

  for (k in seq_len(nrow(grid))) {
    row_args <- as.list(grid[k, , drop = FALSE])

    a <- fit_and_score_mcmc(idx_odd, idx_even, row_args)
    b <- fit_and_score_mcmc(idx_even, idx_odd, row_args)

    train_losses[k] <- (a$train + b$train) / 2
    val_losses[k] <- (a$val + b$val) / 2

    if (verbose) {
      hp_str <- paste(
        sprintf("%s=%s", names(row_args), vapply(row_args, function(x) format(x, digits = 3), "")),
        collapse = ", "
      )
      msg <- sprintf(
        "[cv_copss_mcmc] iter %d/%d | %s | train=%.4f | val=%.4f",
        k, nrow(grid), hp_str, train_losses[k], val_losses[k]
      )
      message(msg)
    }
  }

  out <- cbind(grid, train_loss = train_losses, val_loss = val_losses)
  out <- out[order(out$val_loss), ]
  rownames(out) <- NULL

  out
}
