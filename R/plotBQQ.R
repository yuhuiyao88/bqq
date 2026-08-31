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
                 brick = "#76232F", rose = "#C46A78", gray = "#9AA5B1",
                 # Muted, low-chroma Morandi-style diverging pair for the heatmap:
                 # positive shifts read red, negative shifts read blue. There is no
                 # canonical hex for "Morandi red/blue"; these are representative
                 # tones, matched in lightness, and overridable per call.
                 morandi_red = "#AD6A6C", morandi_blue = "#6E8CA0")

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
# Significant blocks + their onsets and localized change-points, resolved from the
# family/statistic/adjustment actually recorded in `detection`. Reading a hardcoded
# member (as this once did) silently returns nothing whenever the caller ran a basis
# other than "quantile", which is why a QSS-only detection drew no lines at all.
.bqq_sig_blocks <- function(detection, basis = NULL) {
  empty <- list(blocks = integer(0), onset = numeric(0), located = numeric(0))
  db <- detection$detected_blocks
  if (is.null(db)) return(empty)

  fam <- if (!is.null(detection$basis)) detection$basis else "quantile"
  # `basis` lets a caller pin the family so a plot marks the blocks flagged by
  # the SAME test its panel displays. Without it the old precedence applies and
  # quantile wins whenever it was run, which silently made an L-moment or QSS
  # panel carry quantile-basis change points.
  if (!is.null(basis) && length(basis) == 1L &&
      !is.null(detection$tests) && !is.null(detection$tests[[basis]])) {
    fam <- basis
  } else {
    fam <- if ("quantile" %in% fam) "quantile" else "qss"   # quantile wins if both were run
  }
  stat <- if (!is.null(detection$statistic)) detection$statistic else "ui"
  stat_name <- if (("hotelling_t2" %in% stat) && !("ui" %in% stat)) "hotelling_t2" else "ui"
  adjust <- if (!is.null(detection$adjust)) detection$adjust else "calib"

  sb <- detection$tests[[fam]][[stat_name]][[adjust]]
  if (is.null(sb)) {                                      # older detection objects
    nm <- if (fam == "qss") {
      if (stat_name == "hotelling_t2") paste0("significant_qss_t2_", adjust)
      else                             paste0("significant_qss_", adjust)
    } else {
      if (stat_name == "hotelling_t2") paste0("significant_wald_", adjust)
      else                             paste0("significant_", adjust)
    }
    sb <- detection[[nm]]
    if (is.null(sb) && !is.null(db[[nm]])) sb <- which(db[[nm]])
  }
  if (is.null(sb) || length(sb) == 0) return(empty)
  list(blocks = sb, onset = db$obs_start[sb], located = db$signal_obs[sb])
}


#' Plot the data process with fitted quantile bands over time
#'
#' Graph type 1: the observations with the five fitted quantile bands, and
#' (optionally) crimson onset lines and within-block localized change-points
#' from a \code{detectChangepoints_gamma()} result.
#'
#' @param fit A MAP fit from \code{getModel()}.
#' @param time Optional x-axis vector (default \code{seq_len(n)}).
#' @param center,scale Map the fit-scale quantiles/data back to the display scale
#'   as \code{value * scale + center} (e.g. the standardization used before fitting).
#' @param detection Optional \code{detectChangepoints_gamma()} result. Its recorded
#'   \code{basis} and \code{statistic} select which block test flags the blocks, and
#'   its \code{detected_blocks} supplies the block onset (vertical line) and the
#'   localized change-point \code{signal_obs} (circled point on the series), the
#'   latter obtained under whichever \code{signal_position} was passed to
#'   \code{detectChangepoints_gamma()}.
#' @param show_onset,show_located Logical display toggles (default TRUE): draw
#'   the block-onset marks and the localized change-points recorded in
#'   \code{detection}. They only hide layers; the decisions themselves are
#'   made (and recorded) by \code{detectChangepoints_gamma()}.
#' @param title Optional plot title.
#' @param xlab Label for the x axis (default \code{"time"}).
#' @param ylab Label for the y axis (default \code{"value"}). Set it to name the
#'   series actually plotted, e.g. \code{"residual"} when the model was fitted to
#'   regression residuals, or \code{"difference"} when it was fitted to their
#'   first-order differences. Use \code{NULL} to drop the label.
#' @return A ggplot object.
#' @export
plotQuantileProcess <- function(fit, time = NULL, center = 0, scale = 1,
                                detection = NULL, title = NULL,
                                xlab = "time", ylab = "value",
                                show_onset = TRUE, show_located = TRUE, basis = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- if (!is.null(detection) && !is.null(detection$taus)) detection$taus
          else .bqq_taus(fit, NULL)
  m <- length(taus)
  q <- .bqq_fitted_quantiles(fit, taus) * scale + center
  n <- nrow(q)
  yv <- fit$y * scale + center
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
    loc <- .bqq_sig_blocks(detection, basis = basis)
    if (isTRUE(show_onset) && length(loc$onset) > 0) {
      p <- p + ggplot2::geom_vline(xintercept = time[loc$onset], color = pal$crimson,
                                   linewidth = 0.6, alpha = 0.85)
    }
    lp <- loc$located[!is.na(loc$located) & loc$located >= 1 & loc$located <= n]
    if (isTRUE(show_located) && length(lp) > 0) {
      p <- p + ggplot2::geom_point(
        data = data.frame(x = time[lp], y = yv[lp]),
        ggplot2::aes(x = x, y = y), shape = 21, fill = pal$crimson, color = "black",
        size = 2.6, stroke = 0.8)
    }
  }
  p + ggplot2::labs(x = xlab, y = ylab, title = title) + .bqq_theme()
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
#' @param center,scale Map the fit-scale quantiles to the display scale.
#' @param level Credible-band level (default 0.95).
#' @param detection Optional \code{detectChangepoints_gamma()} result. Its recorded
#'   \code{basis} and \code{statistic} select which block test flags the blocks, and
#'   its \code{detected_blocks} supplies both the block onset (dashed line) and the
#'   localized change-point \code{signal_obs} (solid line), the latter obtained under
#'   whichever \code{signal_position} was passed to \code{detectChangepoints_gamma()}.
#' @param show_onset,show_located Logical display toggles (default TRUE): draw
#'   the block-onset marks and the localized change-points recorded in
#'   \code{detection}. They only hide layers; the decisions themselves are
#'   made (and recorded) by \code{detectChangepoints_gamma()}.
#' @param seed Optional seed for \code{getEta()}.
#' @param title Optional plot title.
#' @param xlab Label for the x axis (default \code{"time"}).
#' @param ylab Label for the shared y axis. Default \code{NULL} (no label), since
#'   the four panels are already named by their facet strips and each has its own
#'   free scale.
#' @return A ggplot object (four stacked, free-y facets).
#' @export
plotQSSProcess <- function(fit, eta = NULL, H = NULL, X = NULL, time = NULL,
                           center = 0, scale = 1, level = 0.95, detection = NULL,
                           seed = NULL, title = NULL,
                           xlab = "time", ylab = NULL,
                           show_onset = TRUE, show_located = TRUE, basis = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- if (!is.null(detection) && !is.null(detection$taus)) detection$taus
          else .bqq_taus(fit, NULL)
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
    loc <- .bqq_sig_blocks(detection, basis = basis)
    ob <- loc$onset[!is.na(loc$onset) & loc$onset >= 1 & loc$onset <= n]
    if (isTRUE(show_onset) && length(ob) > 0)
      p <- p + ggplot2::geom_vline(xintercept = time[ob], color = pal$crimson,
                                   linetype = "dashed", linewidth = 0.35, alpha = 0.55)
    # localized change-point within each OOC block (Eq. 27): solid, so it is
    # distinguishable from the dashed block onset.
    lp <- loc$located[!is.na(loc$located) & loc$located >= 1 & loc$located <= n]
    if (isTRUE(show_located) && length(lp) > 0)
      p <- p + ggplot2::geom_vline(xintercept = time[lp], color = pal$crimson,
                                   linewidth = 0.6, alpha = 0.9)
  }
  p +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi, fill = stat), alpha = 0.22) +
    ggplot2::geom_line(ggplot2::aes(y = mid, color = stat), linewidth = 0.8) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_color_manual(values = cols, guide = "none") +
    ggplot2::facet_wrap(~stat, ncol = 1, scales = "free_y", strip.position = "left") +
    ggplot2::labs(x = xlab, y = ylab, title = title) + .bqq_theme() +
    ggplot2::theme(
      panel.border = ggplot2::element_rect(color = "grey55", fill = NA, linewidth = 0.5),
      panel.spacing.y = ggplot2::unit(0.6, "lines"))
}

#' L-moment shape profile over time
#'
#' The L-moment counterpart of \code{\link{plotQSSProcess}}: posterior median and
#' credible band for each of the four approximate L-moments over time, computed by
#' \code{\link{getLmom}} from the fitted quantiles.
#'
#' Change-point marks default to \code{basis = "lmom"}, so the vertical rules come
#' from the SAME L-moment UI test that flags the L-moment heatmap panel. That is the
#' whole point of this function: an L-moment profile carrying quantile-basis change
#' points would be internally inconsistent.
#'
#' @inheritParams plotQSSProcess
#' @param basis Which detection family supplies the change-point marks. Defaults to
#'   \code{"lmom"}; pass another family name to override, or \code{NULL} for the
#'   package's legacy precedence (quantile wins when it was run).
#'
#' @note The panels are the L-moments \eqn{\lambda_r}, not the scale-free ratios
#'   \eqn{\tau_r}; see \code{\link{getLmom}}.
#'
#' @export
plotLmomProcess <- function(fit, eta = NULL, H = NULL, X = NULL, time = NULL,
                            center = 0, scale = 1, level = 0.95, detection = NULL,
                            seed = NULL, title = NULL,
                            xlab = "time", ylab = NULL,
                            show_onset = TRUE, show_located = TRUE,
                            basis = "lmom") {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- if (!is.null(detection) && !is.null(detection$taus)) detection$taus
          else .bqq_taus(fit, NULL)
  if (is.null(eta)) eta <- getEta(fit, H = H, X = X, seed = seed)
  eta <- eta * scale + center
  eta <- aperm(apply(eta, c(1, 3), sort), c(2, 1, 3))   # non-crossing per draw
  lm4 <- getLmom(eta, taus = taus)                      # [iters, 4, n]
  a <- (1 - level) / 2
  mid <- apply(lm4, c(2, 3), stats::median, na.rm = TRUE)
  lo  <- apply(lm4, c(2, 3), stats::quantile, probs = a, na.rm = TRUE)
  hi  <- apply(lm4, c(2, 3), stats::quantile, probs = 1 - a, na.rm = TRUE)
  n <- dim(lm4)[3]
  if (is.null(time)) time <- seq_len(n)
  labs4 <- dimnames(lm4)[[2]]
  df <- do.call(rbind, lapply(seq_len(4), function(k) data.frame(
    time = time, stat = factor(labs4[k], levels = labs4),
    mid = mid[k, ], lo = lo[k, ], hi = hi[k, ])))
  cols <- stats::setNames(c(pal$ink, pal$crimson, pal$rose, pal$brick), labs4)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = time))
  if (!is.null(detection)) {
    loc <- .bqq_sig_blocks(detection, basis = basis)
    ob <- loc$onset[!is.na(loc$onset) & loc$onset >= 1 & loc$onset <= n]
    if (isTRUE(show_onset) && length(ob) > 0)
      p <- p + ggplot2::geom_vline(xintercept = time[ob], color = pal$crimson,
                                   linetype = "dashed", linewidth = 0.35, alpha = 0.55)
    lp <- loc$located[!is.na(loc$located) & loc$located >= 1 & loc$located <= n]
    if (isTRUE(show_located) && length(lp) > 0)
      p <- p + ggplot2::geom_vline(xintercept = time[lp], color = pal$crimson,
                                   linewidth = 0.6, alpha = 0.9)
  }
  p +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi, fill = stat), alpha = 0.22) +
    ggplot2::geom_line(ggplot2::aes(y = mid, color = stat), linewidth = 0.8) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_color_manual(values = cols, guide = "none") +
    ggplot2::facet_wrap(~stat, ncol = 1, scales = "free_y", strip.position = "left") +
    ggplot2::labs(x = xlab, y = ylab, title = title) + .bqq_theme() +
    ggplot2::theme(
      panel.border = ggplot2::element_rect(color = "grey55", fill = NA, linewidth = 0.5),
      panel.spacing.y = ggplot2::unit(0.6, "lines"))
}


#' Plot block-shift coefficient diagnosis (heatmap)
#'
#' Graph type 3: block-shift heatmap(s) with black borders on the significant
#' cells. The function auto-detects, from the \code{detection} object, which test
#' family/families were run and shows the matching panel(s): a \strong{quantile}
#' panel (the quantile shift coefficients as a quantile-by-block map) and/or a \strong{QSS}
#' panel (the four QSS shift coefficients LS, ScS, SkS, KS by block). If only one
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
#' @param show_onset,show_located Logical display toggles (default TRUE): draw
#'   the block-onset marks and the localized change-points recorded in
#'   \code{detection}. They only hide layers; the decisions themselves are
#'   made (and recorded) by \code{detectChangepoints_gamma()}.
#' @param block_labels Optional labels for the block (x) axis (default block index).
#'   Must be unique (they become factor levels).
#' @param title Optional plot title.
#' @param mark_cells Logical; when \code{TRUE} (default) the cells responsible for a
#'   OOC block are given a second, darker border. A block that the block-level
#'   rule flags is bordered in \code{block_color}; inside it, each cell whose
#'   cell-level posterior probability
#'   \eqn{p_{q,j} = 1 - P(\chi^2_1 \le \tilde z^2_{q,j})} falls below \code{alpha}
#'   is bordered in \code{cell_color}. This is the within-block localization step of
#'   Section 3.2: it runs only inside blocks that have already signaled, so it
#'   localizes a detected shift rather than adding a new family of tests, and the
#'   false alarm probability remains that of the block-level rule.
#' @param block_color Border color for cells of a OOC block (default grey).
#' @param cell_color Border color for the localized cells within a OOC block
#'   (default black).
#' @param xlab Label for the x axis (default \code{"block"}).
#' @param ylab Label for the y axis. Default \code{NULL} (no label), since the rows
#'   are already named by the quantile levels or the QSS contrasts. When both panels
#'   are stacked, \code{xlab}/\code{ylab} apply to each panel; \code{title} is
#'   applied once to the combined figure.
#' @param z_limit Positive scalar fixing the diverging fill scale to
#'   \code{c(-z_limit, z_limit)} whenever the fill is on a z scale -- the whitened
#'   \eqn{\tilde z} cells, or the studentized \code{z} fallback (default 3). A FIXED
#'   scale is the point: with a data-driven limit the same color means a different
#'   number in every figure, so two fits cannot be compared by eye. Cells beyond the
#'   limit are clipped to the end color (not dropped), and the subtitle records that
#'   clipping occurred and how far out the extreme cell was. Set \code{NULL} to
#'   restore the old data-driven symmetric limit. Ignored when the fill is the raw
#'   posterior-mean \eqn{\gamma}, where a fixed \eqn{\pm 3} would be meaningless --
#'   those panels always use a data-driven limit.
#' @param pos_color,neg_color End colors of the diverging fill: positive
#'   \eqn{\tilde z} and negative \eqn{\tilde z} respectively, white at zero.
#'   Defaults are a muted, lightness-matched Morandi-style red and blue. There is
#'   no canonical hex for those, so pass your own values to match a house palette.
#' @return A ggplot object when one family is shown; a \pkg{patchwork} of two panels
#'   when both are shown (or a named list of ggplots if \pkg{patchwork} is absent).
#' @export
plotGammaHeatmap <- function(fit, detection = NULL, block_labels = NULL,
                             title = NULL, mark_cells = TRUE,
                             block_color = NULL, cell_color = "black",
                             pos_color = NULL, neg_color = NULL,
                             z_limit = 3,
                             xlab = "block", ylab = NULL) {
  .bqq_need_ggplot2()
  pal <- .bqq_pal
  taus <- if (!is.null(detection) && !is.null(detection$taus)) detection$taus
          else .bqq_taus(fit, NULL)
  m <- length(taus)
  r <- if (!is.null(fit$H)) ncol(fit$H) else 0L
  if (r == 0) stop("No block-shift design (fit$H has no columns).", call. = FALSE)
  blk <- if (!is.null(block_labels)) block_labels else seq_len(r)

  ## ---- everything below renders the decisions RECORDED in `detection`: which
  ## families (basis), which block statistic, and which across-block rule
  ## (adjust) were chosen when detectChangepoints_gamma() was run. The plot has
  ## no decision arguments of its own. ----
  ALL_FAMS <- c("quantile", "qss", "lmom", "maxent")
  fams <- if (is.null(detection)) "quantile"
          else if (!is.null(detection$basis)) detection$basis
          else c("quantile", if (!is.null(detection$z_qss)) "qss")
  fams <- intersect(ALL_FAMS, fams)
  # keep only families the detection object actually carries cells for
  have <- function(f) switch(f,
    quantile = TRUE,
    qss      = !is.null(detection$z_qss),
    lmom     = !is.null(detection$z_lmom),
    maxent   = !is.null(detection$z_maxent),
    FALSE)
  if (!is.null(detection)) fams <- fams[vapply(fams, have, logical(1))]
  if (length(fams) == 0) fams <- "quantile"
  stat <- if (!is.null(detection) && !is.null(detection$statistic)) detection$statistic else "ui"
  use_t2 <- ("hotelling_t2" %in% stat) && !("ui" %in% stat)   # UI wins if both were run
  stat_name <- if (use_t2) "hotelling_t2" else "ui"
  adjust <- if (!is.null(detection) && !is.null(detection$adjust)) detection$adjust else "calib"

  ## ---- OOC blocks under the recorded rule; older detection objects fall
  ## back to the flat calibrated aliases (calib only). ----
  get_sig <- function(fam) {
    if (is.null(detection)) return(integer(0))
    fam_res <- detection$tests[[fam]]
    if (!is.null(fam_res) && !is.null(fam_res[[stat_name]]) &&
        !is.null(fam_res[[stat_name]][[adjust]])) {
      return(fam_res[[stat_name]][[adjust]])
    }
    if (adjust != "calib") {
      warning("this detection object does not carry the '", adjust,
              "' adjustment member; using the calibrated flags.", call. = FALSE)
    }
    if (fam == "qss") {
      if (use_t2) detection$significant_qss_t2_calib else detection$significant_qss_calib
    } else {
      if (use_t2) detection$significant_wald_calib else detection$significant_calib
    }
  }
  outline_lab <- paste0(": ", if (use_t2) "T2" else "UI", "/", adjust)

  ## ---- within-block localization (manuscript Sec 3.2): a OOC block is
  ## bordered in block_color; inside it, a cell is bordered in cell_color when
  ## its whitened statistic exceeds the SAME charting constant that flagged the
  ## block -- Eqs. (21)/(23) for the raw rule, (22)/(24) for the calibrated
  ## rule, and Eq. (25)'s adjusted constants under Bonferroni/Holm/BH. The flags
  ## are computed by detectChangepoints_gamma() ($cells); a cell exceedance
  ## implies the block exceedance, so cells localize the block decisions without
  ## adding a hypothesis family. ----
  blk_col   <- if (!is.null(block_color)) block_color else pal$gray
  pos_color <- if (!is.null(pos_color))   pos_color   else pal$morandi_red
  neg_color <- if (!is.null(neg_color))   neg_color   else pal$morandi_blue
  get_sig_cells <- function(fam, sig_cols, nr) {
    out <- matrix(FALSE, nr, r)
    if (!isTRUE(mark_cells) || is.null(detection) || length(sig_cols) == 0) return(out)
    cm <- detection$tests[[fam]][[stat_name]]$cells[[adjust]]
    if (is.null(cm) || !is.matrix(cm) || nrow(cm) != nr || ncol(cm) != r) {
      warning("cell-level flags are unavailable for the '", fam,
              "' family (older detection object?); blocks are bordered but ",
              "cells are not localized.", call. = FALSE)
      return(out)
    }
    cm
  }

  ## ---- single-panel builder; sig_cols = significant blocks (whole-column border),
  ## sig_cells = logical ncell x r matrix of localized cells within those blocks ----
  heat <- function(vals, rowlab, sig_cols, sig_cells, fill_lab, subtitle, diverging,
                   fixed_lim = NULL) {
    d <- expand.grid(ri = seq_len(nrow(vals)), bj = seq_len(r))
    d$val   <- vals[cbind(d$ri, d$bj)]
    d$sig   <- d$bj %in% sig_cols
    d$cell  <- sig_cells[cbind(d$ri, d$bj)]
    d$row   <- factor(rowlab[d$ri], levels = rowlab)
    d$block <- factor(blk[d$bj], levels = blk)

    # A FIXED fill scale is what makes two figures comparable; a data-driven one
    # silently redefines what a color means. Values beyond the fixed limit are
    # clipped to the end color rather than dropped (ggplot renders out-of-limits
    # as grey NA, which would read as "missing" instead of "extreme"), and the
    # clipping is disclosed in the subtitle.
    obs_max <- suppressWarnings(max(abs(vals), na.rm = TRUE))
    clipped <- 0L
    if (diverging && !is.null(fixed_lim) && is.finite(fixed_lim) && fixed_lim > 0) {
      lim <- fixed_lim
      clipped <- sum(abs(d$val) > lim, na.rm = TRUE)
      d$fill_val <- pmin(pmax(d$val, -lim), lim)
    } else {
      lim <- obs_max; if (!is.finite(lim) || lim == 0) lim <- 1
      d$fill_val <- d$val
    }
    if (clipped > 0L) {
      subtitle <- paste0(subtitle, sprintf("  (fill fixed at +/-%g; %d cell%s clipped, max |z| = %.2f)",
                                           lim, clipped, if (clipped == 1L) "" else "s", obs_max))
    }

    g <- ggplot2::ggplot(d, ggplot2::aes(x = block, y = row)) +
      ggplot2::geom_tile(ggplot2::aes(fill = fill_val)) +
      ggplot2::geom_tile(data = d[d$sig, , drop = FALSE], fill = NA,
                         color = blk_col, linewidth = 0.6) +
      ggplot2::geom_tile(data = d[d$cell, , drop = FALSE], fill = NA,
                         color = cell_color, linewidth = 1.0)
    if (diverging) {
      g <- g + ggplot2::scale_fill_gradient2(low = neg_color, mid = "white", high = pos_color,
                                             midpoint = 0, limits = c(-lim, lim))
    } else {
      g <- g + ggplot2::scale_fill_gradient(low = "white", high = pal$crimson)
    }
    g + ggplot2::labs(x = xlab, y = ylab, fill = fill_lab, subtitle = subtitle) + .bqq_theme()
  }

  panels <- list()

  ## ---- quantile panel ----
  if ("quantile" %in% fams) {
    if (!is.null(detection) && !is.null(detection$z_raw)) {
      if (!is.null(detection$z_white)) {
        vals <- detection$z_white
        flab <- expression(tilde(z)); sub <- "Quantile Shift Coefficient"; note <- ""
      } else {
        vals <- detection$z_raw
        flab <- "z"; sub <- "Quantile Shift Coefficient"
        note <- "  (studentized; whitened cells unavailable)"
      }
      rl <- rownames(vals); if (is.null(rl)) rl <- format(taus)
      zscale <- TRUE                  # z-tilde or studentized z: fix the fill scale
    } else {
      vals <- .bqq_coefs(fit, m, r)$gamma; rl <- format(taus)
      flab <- expression(gamma); sub <- "Quantile Shift Coefficient"
      note <- "  (posterior mean)"
      zscale <- FALSE                 # raw coefficients: +/-3 would be meaningless
    }
    sub <- paste0(sub, if (!is.null(detection)) outline_lab else "", note)
    sc <- get_sig("quantile")
    panels$quantile <- heat(vals, rl, sc, get_sig_cells("quantile", sc, nrow(vals)),
                            flab, sub, diverging = TRUE,
                            fixed_lim = if (zscale) z_limit else NULL)
  }

  ## ---- QSS panel ----
  if ("qss" %in% fams) {
    if (!is.null(detection$z_white_qss)) {
      vals <- detection$z_white_qss
      flab <- expression(tilde(z)); sub <- "QSS Shift Coefficient"; note <- ""
    } else {
      vals <- detection$z_qss
      flab <- "z"; sub <- "QSS Shift Coefficient"
      note <- "  (studentized; whitened cells unavailable)"
    }
    rl <- rownames(vals); if (is.null(rl)) rl <- c("LS", "ScS", "SkS", "KS")
    sub <- paste0(sub, outline_lab, note)
    sc <- get_sig("qss")
    panels$qss <- heat(vals, rl, sc, get_sig_cells("qss", sc, nrow(vals)),
                       flab, sub, diverging = TRUE, fixed_lim = z_limit)
  }

  ## ---- alternative shape bases: identical rendering, different rotation ----
  shape_panel <- function(fam, zt, zs, default_rows, label) {
    if (!(fam %in% fams)) return(invisible(NULL))
    if (!is.null(zt)) {
      vals <- zt; flab <- expression(tilde(z)); note <- ""
    } else {
      vals <- zs; flab <- "z"; note <- "  (studentized; whitened cells unavailable)"
    }
    if (is.null(vals)) return(invisible(NULL))
    rl <- rownames(vals); if (is.null(rl)) rl <- default_rows
    sc <- get_sig(fam)
    panels[[fam]] <<- heat(vals, rl, sc, get_sig_cells(fam, sc, nrow(vals)),
                           flab, paste0(label, outline_lab, note),
                           diverging = TRUE, fixed_lim = z_limit)
  }
  shape_panel("lmom", detection$z_white_lmom, detection$z_lmom,
              c("L-location", "L-scale", "L-skewness", "L-kurtosis"),
              "L-moment Shift Coefficient")
  shape_panel("maxent", detection$z_white_maxent, detection$z_maxent,
              c("ME-location", "ME-scale", "ME-skewness", "ME-kurtosis"),
              "Maximum-entropy Shift Coefficient")

  ## ---- return one panel, or stack them in the order requested ----
  panels <- panels[intersect(fams, names(panels))]
  if (length(panels) == 1L) {
    p <- panels[[1]]
    if (!is.null(title)) p <- p + ggplot2::labs(title = title)
    return(p)
  }
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combo <- patchwork::wrap_plots(panels, ncol = 1L)
    if (!is.null(title)) combo <- combo + patchwork::plot_annotation(title = title)
    return(combo)
  }
  message(length(panels), " basis panels are present; install 'patchwork' to stack ",
          "them into one figure. Returning a named list of ggplot objects instead.")
  panels
}
