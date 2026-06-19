# =============================================================================
# BQQ visualization (ggplot2)
# =============================================================================
# Three generalized graph types, mirroring the package's worked demos:
#   (1) plotQuantileProcess() - data with fitted quantile bands over time
#   (2) plotQSSProcess()      - quantile shape statistics over time (ribbons)
#   (3) plotGammaHeatmap()    - block-shift coefficient diagnosis (heatmap)
# ggplot2 is an optional (Suggests) dependency; each function checks for it at
# call time so the package still loads and fits without ggplot2 installed.

# UA / crimson palette shared across the three graphs
.bqq_pal <- list(ink = "#2A2123", steel = "#5B666D", crimson = "#9E1B32",
                 brick = "#76232F", rose = "#C46A78", gray = "#9AA5B1")

.bqq_need_ggplot2 <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for BQQ plots. Install it with install.packages(\"ggplot2\").",
         call. = FALSE)
  }
}

.bqq_theme <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      strip.placement = "outside",
      strip.text.y.left = ggplot2::element_text(angle = 0)
    )
}

# Quantile levels: prefer an explicit argument, else recover from the fit.
.bqq_taus <- function(fit, taus) {
  if (!is.null(taus)) return(taus)
  for (nm in c("taus", "tau_q", "tau")) if (!is.null(fit[[nm]])) return(fit[[nm]])
  stop("Quantile levels not found in fit; please pass `taus`.", call. = FALSE)
}

# MAP fitted quantiles, n x m, reconstructed as X %*% beta + H %*% gamma.
.bqq_fitted_quantiles <- function(fit, taus) {
  par <- fit$map$par
  if (is.null(par)) stop("BQQ plots require a MAP fit (fit$map$par).", call. = FALSE)
  m <- length(taus)
  X <- if (is.null(fit$X)) NULL else as.matrix(fit$X)
  px <- if (is.null(X)) 0L else ncol(X)
  beta <- matrix(par[grep("^beta\\[", names(par))], m, px + 1L)
  r <- if (!is.null(fit$H)) ncol(fit$H) else 0L
  gamma <- if (r > 0) matrix(par[grep("^gamma\\[", names(par))], m, r) else matrix(0, m, 0)
  n <- length(fit$y)
  Xd <- cbind(1, if (is.null(X)) matrix(0, n, 0) else X)
  sapply(seq_len(m), function(j)
    as.numeric(Xd %*% beta[j, ]) + (if (r > 0) as.numeric(fit$H %*% gamma[j, ]) else 0))
}

# Significant blocks + their onset/localized observations from a
# detectChangepoints_gamma() result (block significant if any quantile cell is
# Holm-significant, matching the worked demo).
.bqq_sig_blocks <- function(detection, alpha) {
  db <- detection$detected_blocks
  sb <- if (!is.null(detection$adjp_holm)) which(apply(detection$adjp_holm < alpha, 2, any)) else which(db$significant_holm)
  list(blocks = sb, onset = db$obs_start[sb], located = db$signal_obs[sb])
}


#' Plot the data process with fitted quantile bands over time
#'
#' Graph type 1: the observations with the five fitted quantile bands, and
#' (optionally) crimson onset lines and within-block localized change points
#' from a \code{detectChangepoints_gamma()} result.
#'
#' @param fit A MAP fit from \code{getModel()}.
#' @param time Optional x-axis vector (default \code{seq_len(n)}).
#' @param taus Quantile levels (default recovered from \code{fit}).
#' @param center,scale Map the fit-scale quantiles/data back to the display scale
#'   as \code{value * scale + center} (e.g. the standardization used before fitting).
#' @param detection Optional \code{detectChangepoints_gamma()} result; adds onset
#'   lines and localized-change-point markers.
#' @param alpha Significance level for the block decision (default 0.05).
#' @param y Optional observed series override (default \code{fit$y}).
#' @param title Optional plot title.
#' @return A ggplot object.
#' @export
plotQuantileProcess <- function(fit, time = NULL, taus = NULL, center = 0, scale = 1,
                                detection = NULL, alpha = 0.05, y = NULL, title = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- .bqq_taus(fit, taus); m <- length(taus)
  q <- .bqq_fitted_quantiles(fit, taus) * scale + center
  n <- nrow(q)
  yv <- (if (is.null(y)) fit$y else y) * scale + center
  if (is.null(time)) time <- seq_len(n)
  df <- data.frame(time = time, y = yv,
                   lo = q[, 1], q1 = q[, 2], med = q[, ceiling(m / 2)], q3 = q[, m - 1], hi = q[, m])
  p <- ggplot2::ggplot(df, ggplot2::aes(x = time)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi), fill = pal$gray, alpha = 0.10) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = q1, ymax = q3), fill = pal$crimson, alpha = 0.17) +
    ggplot2::geom_point(ggplot2::aes(y = y), color = pal$gray, alpha = 0.5, size = 1) +
    ggplot2::geom_line(ggplot2::aes(y = lo), color = pal$brick, linetype = "dashed", linewidth = 0.4) +
    ggplot2::geom_line(ggplot2::aes(y = hi), color = pal$brick, linetype = "dashed", linewidth = 0.4) +
    ggplot2::geom_line(ggplot2::aes(y = q1), color = pal$steel, linewidth = 0.5) +
    ggplot2::geom_line(ggplot2::aes(y = q3), color = pal$steel, linewidth = 0.5) +
    ggplot2::geom_line(ggplot2::aes(y = med), color = pal$ink, linewidth = 0.9)
  if (!is.null(detection)) {
    loc <- .bqq_sig_blocks(detection, alpha)
    if (length(loc$onset) > 0) {
      p <- p + ggplot2::geom_vline(xintercept = time[loc$onset], color = pal$crimson,
                                   linewidth = 0.6, alpha = 0.85)
      lp <- loc$located[loc$located >= 1 & loc$located <= n]
      if (length(lp) > 0) {
        p <- p + ggplot2::geom_point(
          data = data.frame(x = time[lp], y = yv[lp]),
          ggplot2::aes(x = x, y = y), shape = 21, fill = pal$crimson, color = "black",
          size = 2.6, stroke = 0.8)
      }
    }
  }
  p + ggplot2::labs(x = "time", y = "value", title = title) + .bqq_theme()
}


#' Plot the quantile-shape-statistic (QSS) process over time
#'
#' Graph type 2: posterior Location, Scale, Skewness and Kurtosis over time, each
#' as a median line with a credible-band ribbon. Posterior draws are sorted within
#' each draw (non-crossing) before the shape statistics are formed, so the ratios
#' stay well defined.
#'
#' @param fit A MAP fit from \code{getModel()}.
#' @param eta Optional posterior predictive-quantile array from \code{getEta()};
#'   computed internally if not supplied.
#' @param H,X Optional design matrices passed to \code{getEta()} (default from fit).
#' @param time Optional x-axis vector (default \code{seq_len(n)}).
#' @param taus Quantile levels (default recovered from \code{fit}).
#' @param center,scale Map the fit-scale quantiles to the display scale.
#' @param level Credible-band level (default 0.95).
#' @param detection Optional \code{detectChangepoints_gamma()} result; adds onset lines.
#' @param alpha Significance level for the block decision (default 0.05).
#' @param seed Optional seed for \code{getEta()}.
#' @param title Optional plot title.
#' @return A ggplot object (four stacked, free-y facets).
#' @export
plotQSSProcess <- function(fit, eta = NULL, H = NULL, X = NULL, time = NULL, taus = NULL,
                           center = 0, scale = 1, level = 0.95, detection = NULL,
                           alpha = 0.05, seed = NULL, title = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- .bqq_taus(fit, taus)
  if (is.null(eta)) eta <- getEta(fit, H = H, X = X, seed = seed)
  eta <- eta * scale + center
  eta <- aperm(apply(eta, c(1, 3), sort), c(2, 1, 3))  # enforce non-crossing per draw
  qss <- getQSS(eta, taus = taus)                      # [iters, 4, n]
  a <- (1 - level) / 2
  qmid <- apply(qss, c(2, 3), stats::median, na.rm = TRUE)
  qlo  <- apply(qss, c(2, 3), stats::quantile, probs = a, na.rm = TRUE)
  qhi  <- apply(qss, c(2, 3), stats::quantile, probs = 1 - a, na.rm = TRUE)
  n <- dim(qss)[3]
  if (is.null(time)) time <- seq_len(n)
  labs4 <- c("Location", "Scale", "Skewness", "Kurtosis")
  df <- do.call(rbind, lapply(seq_len(4), function(k) data.frame(
    time = time, stat = factor(labs4[k], levels = labs4),
    mid = qmid[k, ], lo = qlo[k, ], hi = qhi[k, ])))
  cols <- c(Location = pal$ink, Scale = pal$crimson, Skewness = pal$rose, Kurtosis = pal$brick)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = time))
  if (!is.null(detection)) {
    loc <- .bqq_sig_blocks(detection, alpha)
    if (length(loc$onset) > 0)
      p <- p + ggplot2::geom_vline(xintercept = time[loc$onset], color = pal$crimson,
                                   linetype = "dashed", linewidth = 0.35, alpha = 0.55)
  }
  p +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi, fill = stat), alpha = 0.22) +
    ggplot2::geom_line(ggplot2::aes(y = mid, color = stat), linewidth = 0.8) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_color_manual(values = cols, guide = "none") +
    ggplot2::facet_wrap(~stat, ncol = 1, scales = "free_y", strip.position = "left") +
    ggplot2::labs(x = "time", y = NULL, title = title) + .bqq_theme() +
    ggplot2::theme(
      panel.border = ggplot2::element_rect(color = "grey55", fill = NA, linewidth = 0.5),
      panel.spacing.y = ggplot2::unit(0.6, "lines"))
}


#' Plot block-shift coefficient diagnosis (heatmap)
#'
#' Graph type 3: the estimated block-shift coefficients gamma as a
#' quantile-by-block heatmap (diverging fill), with a black border on the cells
#' that are significant under the BQQ test.
#'
#' @param fit A MAP fit from \code{getModel()}.
#' @param detection Optional \code{detectChangepoints_gamma()} result; cells with
#'   Holm-adjusted p < \code{alpha} get a black border.
#' @param taus Quantile levels (default recovered from \code{fit}).
#' @param scale Multiply coefficients for display (e.g. the fitting SD).
#' @param alpha Significance level for the cell borders (default 0.05).
#' @param block_labels Optional labels for the block (x) axis (default block index).
#' @param title Optional plot title.
#' @return A ggplot object.
#' @export
plotGammaHeatmap <- function(fit, detection = NULL, taus = NULL, scale = 1,
                             alpha = 0.05, block_labels = NULL, title = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- .bqq_taus(fit, taus); m <- length(taus)
  par <- fit$map$par
  r <- if (!is.null(fit$H)) ncol(fit$H) else 0L
  if (r == 0) stop("No block-shift design (fit$H has no columns).", call. = FALSE)
  gamma <- matrix(par[grep("^gamma\\[", names(par))], m, r) * scale
  sig <- matrix(FALSE, m, r)
  if (!is.null(detection) && !is.null(detection$adjp_holm)) sig <- detection$adjp_holm < alpha
  blk <- if (!is.null(block_labels)) block_labels else seq_len(r)
  df <- expand.grid(qi = seq_len(m), bj = seq_len(r))
  df$gamma <- gamma[cbind(df$qi, df$bj)]
  df$sig <- sig[cbind(df$qi, df$bj)]
  df$tau <- factor(format(taus[df$qi]), levels = format(taus))
  df$block <- factor(blk[df$bj], levels = blk)
  lim <- max(abs(gamma), na.rm = TRUE); if (!is.finite(lim) || lim == 0) lim <- 1
  ggplot2::ggplot(df, ggplot2::aes(x = block, y = tau)) +
    ggplot2::geom_tile(ggplot2::aes(fill = gamma)) +
    ggplot2::geom_tile(data = df[df$sig, , drop = FALSE], fill = NA, color = "black", linewidth = 0.6) +
    ggplot2::scale_fill_gradient2(low = pal$steel, mid = "white", high = pal$crimson,
                                  midpoint = 0, limits = c(-lim, lim)) +
    ggplot2::labs(x = "block", y = expression(tau), fill = expression(gamma), title = title) +
    .bqq_theme()
}
