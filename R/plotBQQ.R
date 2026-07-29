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

# Point-estimate fitted quantiles, n x m, reconstructed as X %*% beta + H %*% gamma.
# Uses the coherent point estimate (MAP mode under MAP, posterior median under
# MCMC) via .bqq_point_eta / .bqq_coefs, so the plotted central line matches the
# estimator used for detection and localization.
.bqq_fitted_quantiles <- function(fit, taus) {
  if (is.null(fit$map$par))
    stop("BQQ plots require a fit with fit$map$par (MAP mode or MCMC posterior median).",
         call. = FALSE)
  t(.bqq_point_eta(fit, taus))   # m x n -> n x m
}

# Significant blocks + their onset/localized observations from a
# detectChangepoints_gamma() result (block significant if any quantile cell is
# Holm-significant, matching the worked demo).
.bqq_sig_blocks <- function(detection, alpha) {
  db <- detection$detected_blocks
  sb <- if (length(detection$significant_holm)) detection$significant_holm else which(db$significant_holm)
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
#' Graph type 3: block-shift heatmap(s) with black borders on the significant
#' cells. The function auto-detects, from the \code{detection} object, which test
#' family/families were run and shows the matching panel(s): a \strong{quantile}
#' panel (the block-shift gammas as a quantile-by-block map) and/or a \strong{QSS}
#' panel (the four studentized shape contrasts L, S, Sk, K by block). If only one
#' family was computed, only that panel is drawn; if both, both are stacked (via
#' \pkg{patchwork} when available). Borders follow the statistic recorded in
#' \code{detection}: per-cell where \eqn{|z|} exceeds the calibrated cell-max
#' threshold, or whole significant columns under the Hotelling \eqn{T^2}.
#'
#' @param fit A MAP fit from \code{getModel()}.
#' @param detection Optional \code{detectChangepoints_gamma()} result. Its
#'   \code{basis}/\code{statistic} fields drive which panels appear and how cells
#'   are bordered. Older results without those fields fall back to a quantile panel
#'   (Holm-bordered), plus a QSS panel if \code{z_qss} is present.
#' @param block_labels Optional labels for the block (x) axis (default block index).
#'   Must be unique (they become factor levels).
#' @param adjust Which member of the across-block adjustment family determines the
#'   outlined (significant) blocks: \code{"calib"} (default; the calibrated
#'   single-step charting-constant rule), \code{"raw"}, \code{"holm"},
#'   \code{"bonf"}, or \code{"bh"}, as computed by
#'   \code{detectChangepoints_gamma()}. The quantile levels, significance level,
#'   and significance level are inherited from \code{detection}. The recorded
#'   statistic selects the BLOCK test only: the display is identical for
#'   \code{"ui"} and \code{"hotelling_t2"}. The fill is always the whitened
#'   \eqn{\tilde z} on a diverging scale centred at zero, and the cell-level
#'   statistic is always the whitened \eqn{\tilde z^2}; only which blocks are
#'   bordered depends on the statistic. Detection objects predating the
#'   whitened-cell fields fall back to the studentized z, which the panel
#'   subtitle records.
#' @param title Optional plot title.
#' @param sig_block Optional length-\code{r} logical vector of significant blocks.
#'   When supplied it takes precedence for the quantile panel: every cell in a
#'   significant block gets the grey border, so flagged blocks read as whole
#'   bordered columns. Default \code{NULL} keeps the detection-driven behavior.
#' @param basis Optional character vector (\code{"quantile"}, \code{"qss"}) to force
#'   which panel(s) to draw, overriding the auto-detection from \code{detection}.
#' @param mark_cells Logical; when \code{TRUE} (default) the cells responsible for a
#'   flagged block are given a second, darker border. A block that the block-level
#'   rule flags is bordered in \code{block_color}; inside it, each cell whose
#'   cell-level posterior probability
#'   \eqn{p_{q,j} = 1 - P(\chi^2_1 \le \tilde z^2_{q,j})} falls below \code{alpha}
#'   is bordered in \code{cell_color}. This is the within-block localization step of
#'   Section 3.2: it runs only inside blocks that have already signaled, so it
#'   localizes a detected shift rather than adding a new family of tests, and the
#'   false alarm probability remains that of the block-level rule.
#' @param alpha Significance level for the cell-level localization. Default
#'   \code{NULL} inherits \code{detection$alpha}, falling back to 0.05.
#' @param block_color Border colour for cells of a flagged block (default grey).
#' @param cell_color Border colour for the localized cells within a flagged block
#'   (default black).
#' @return A ggplot object when one family is shown; a \pkg{patchwork} of two panels
#'   when both are shown (or a named list of ggplots if \pkg{patchwork} is absent).
#' @export
plotGammaHeatmap <- function(fit, detection = NULL, block_labels = NULL,
                             title = NULL, sig_block = NULL, basis = NULL,
                             adjust = c("calib", "raw", "holm", "bonf", "bh"),
                             mark_cells = TRUE, alpha = NULL,
                             block_color = NULL, cell_color = "black") {
  adjust <- match.arg(adjust)
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- if (!is.null(detection) && !is.null(detection$taus)) detection$taus
          else .bqq_taus(fit, NULL)
  m <- length(taus)
  r <- if (!is.null(fit$H)) ncol(fit$H) else 0L
  if (r == 0) stop("No block-shift design (fit$H has no columns).", call. = FALSE)
  blk <- if (!is.null(block_labels)) block_labels else seq_len(r)

  ## ---- which family/families to show: honor `basis`, else auto-detect from `detection` ----
  fams <- basis
  if (is.null(fams)) {
    if (is.null(detection)) fams <- "quantile"
    else if (!is.null(detection$basis)) fams <- detection$basis
    else { fams <- "quantile"; if (!is.null(detection$z_qss)) fams <- c(fams, "qss") }
  }
  fams <- intersect(c("quantile", "qss"), fams)
  if ("qss" %in% fams && (is.null(detection) || is.null(detection$z_qss))) fams <- setdiff(fams, "qss")
  if (length(fams) == 0) fams <- "quantile"
  stat <- if (!is.null(detection) && !is.null(detection$statistic)) detection$statistic else "ui"
  use_t2 <- ("hotelling_t2" %in% stat) && !("ui" %in% stat)   # UI wins if both requested
  stat_name <- if (use_t2) "hotelling_t2" else "ui"
  # The statistic selects the BLOCK test only (UI or T^2). The display is the same
  # either way: the fill is the whitened z, and the cell-level statistic is the
  # whitened z^2 (chi-square_1). Only the block borders depend on `stat_name`.

  ## ---- significant blocks for a family: sig_block override wins; otherwise the
  ## requested member of the across-block adjustment family {raw, holm, bonf, bh,
  ## calib} from detection$tests; older detection objects without $tests fall
  ## back to the flat calibrated aliases (calib only). ----
  get_sig <- function(fam) {
    if (!is.null(sig_block)) return(which(as.logical(sig_block)))
    if (is.null(detection)) return(integer(0))
    fam_res <- detection$tests[[fam]]
    if (!is.null(fam_res) && !is.null(fam_res[[stat_name]]) &&
        !is.null(fam_res[[stat_name]][[adjust]])) {
      return(fam_res[[stat_name]][[adjust]])
    }
    if (adjust != "calib") {
      warning("adjust = '", adjust, "' requested but this detection object does ",
              "not carry the adjustment family; using the calibrated flags.",
              call. = FALSE)
    }
    if (fam == "qss") {
      if (use_t2) detection$significant_qss_t2_calib else detection$significant_qss_calib
    } else {
      if (use_t2) detection$significant_wald_calib else detection$significant_calib
    }
  }
  outline_lab <- paste0("  (block: ", if (use_t2) "T2" else "UI", "/", adjust,
                        if (isTRUE(mark_cells)) "; cell: chi-square_1" else "", ")")

  ## ---- within-block localization (Section 3.2, Eq. 25) ----
  ## A block flagged by the block-level rule is bordered in block_color. Inside it,
  ## a cell is bordered in cell_color when its cell-level posterior probability
  ##   p_{q,j} = 1 - P(chi-square_1 <= z~^2_{q,j})
  ## falls below alpha. Restricting the search to already-flagged blocks is what
  ## keeps this a localization step rather than a new family of tests, so the
  ## false alarm probability stays that of the block-level rule.
  blk_col <- if (!is.null(block_color)) block_color else pal$gray
  a_cell <- if (!is.null(alpha)) alpha
            else if (!is.null(detection) && !is.null(detection$alpha)) detection$alpha
            else 0.05
  get_sig_cells <- function(fam, sig_cols, nr) {
    out <- matrix(FALSE, nr, r)
    if (!isTRUE(mark_cells) || is.null(detection) || length(sig_cols) == 0) return(out)
    cs <- if (fam == "qss") detection$cellstat_qss else detection$cellstat
    if (is.null(cs) || !is.matrix(cs) || nrow(cs) != nr || ncol(cs) != r) {
      warning("cell-level statistics are unavailable for the '", fam,
              "' family; blocks are bordered but cells are not localized.",
              call. = FALSE)
      return(out)
    }
    p <- stats::pchisq(cs, df = 1, lower.tail = FALSE)
    out[, sig_cols] <- p[, sig_cols, drop = FALSE] < a_cell
    out
  }

  ## ---- single-panel builder; sig_cols = significant blocks (whole-column border),
  ## sig_cells = logical ncell x r matrix of localized cells within those blocks ----
  heat <- function(vals, rowlab, sig_cols, sig_cells, fill_lab, subtitle, diverging) {
    d <- expand.grid(ri = seq_len(nrow(vals)), bj = seq_len(r))
    d$val   <- vals[cbind(d$ri, d$bj)]
    d$sig   <- d$bj %in% sig_cols
    d$cell  <- sig_cells[cbind(d$ri, d$bj)]
    d$row   <- factor(rowlab[d$ri], levels = rowlab)
    d$block <- factor(blk[d$bj], levels = blk)
    g <- ggplot2::ggplot(d, ggplot2::aes(x = block, y = row)) +
      ggplot2::geom_tile(ggplot2::aes(fill = val)) +
      ggplot2::geom_tile(data = d[d$sig, , drop = FALSE], fill = NA,
                         color = blk_col, linewidth = 0.6) +
      ggplot2::geom_tile(data = d[d$cell, , drop = FALSE], fill = NA,
                         color = cell_color, linewidth = 1.0)
    if (diverging) {
      lim <- max(abs(vals), na.rm = TRUE); if (!is.finite(lim) || lim == 0) lim <- 1
      g <- g + ggplot2::scale_fill_gradient2(low = pal$steel, mid = "white", high = pal$crimson,
                                             midpoint = 0, limits = c(-lim, lim))
    } else {
      g <- g + ggplot2::scale_fill_gradient(low = "white", high = pal$crimson)
    }
    g + ggplot2::labs(x = "block", y = NULL, fill = fill_lab, subtitle = subtitle) + .bqq_theme()
  }

  panels <- list()

  ## ---- quantile panel ----
  if ("quantile" %in% fams) {
    if (!is.null(detection) && !is.null(detection$z_raw)) {
      if (!is.null(detection$z_white)) {
        vals <- detection$z_white
        flab <- expression(tilde(z)); sub <- "quantile: whitened z"
      } else {
        vals <- detection$z_raw
        flab <- "z"; sub <- "quantile: studentized z (whitened cells unavailable)"
      }
      rl <- rownames(vals); if (is.null(rl)) rl <- format(taus)
    } else {
      vals <- .bqq_coefs(fit, m, r)$gamma; rl <- format(taus)
      flab <- expression(gamma); sub <- "quantile: block-shift gamma"
    }
    if (!is.null(detection)) sub <- paste0(sub, outline_lab)
    sc <- get_sig("quantile")
    panels$quantile <- heat(vals, rl, sc, get_sig_cells("quantile", sc, nrow(vals)),
                            flab, sub, diverging = TRUE)
  }

  ## ---- QSS panel ----
  if ("qss" %in% fams) {
    if (!is.null(detection$z_white_qss)) {
      vals <- detection$z_white_qss
      flab <- expression(tilde(z)); sub <- "QSS: whitened shape contrasts"
    } else {
      vals <- detection$z_qss
      flab <- "z"; sub <- "QSS: studentized shape contrasts (whitened cells unavailable)"
    }
    rl <- rownames(vals); if (is.null(rl)) rl <- c("L", "S", "Sk", "K")
    sub <- paste0(sub, outline_lab)
    sc <- get_sig("qss")
    panels$qss <- heat(vals, rl, sc, get_sig_cells("qss", sc, nrow(vals)),
                       flab, sub, diverging = TRUE)
  }

  ## ---- return one panel, or stack both ----
  if (length(panels) == 1L) {
    p <- panels[[1]]
    if (!is.null(title)) p <- p + ggplot2::labs(title = title)
    return(p)
  }
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combo <- patchwork::wrap_plots(panels$quantile, panels$qss, ncol = 1L)
    if (!is.null(title)) combo <- combo + patchwork::plot_annotation(title = title)
    return(combo)
  }
  message("Both quantile and QSS results are present; install 'patchwork' to stack ",
          "them into one figure. Returning a named list of ggplot objects instead.")
  panels
}
