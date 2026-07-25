# Smoke test for the manuscript-alignment fixes (Section 3 / Appendix C).
# Run: Rscript tests/smoke_alignment.R    (from the package root)
# Verifies: (1) covariance whitening z~ = Sigma^{-1/2} gammabar; (2) new kurtosis
# contrast KS = g5 - g4 + g2 - g1; (3) manuscript charting constants; (4) new
# overall T^2 = max_j T^2_j with p = 1 - (P(chi2 <= T^2))^r; (5) k0_hat / whiten removed.

suppressWarnings(suppressMessages(source("R/getInference.R")))
ok <- function(msg, cond) {
  cat(if (isTRUE(cond)) "PASS" else "FAIL", "-", msg, "\n")
  if (!isTRUE(cond)) stop("assertion failed: ", msg)
}
approx <- function(a, b, tol = 1e-8) all(abs(a - b) <= tol * pmax(1, abs(b)))

set.seed(1)
alpha <- 0.05
r <- 6L                 # blocks
n_iter <- 1500L
sdc <- 0.12

## ---------- Part A: parse/source clean ----------
ok("package inference sources without error", exists("detectChangepoints_gamma") &&
     exists(".bqq_block_tests") && exists("getQSS"))
ok("whiten arg removed from detectChangepoints_gamma",
   !("whiten" %in% names(formals(detectChangepoints_gamma))))
ok("whiten arg removed from .bqq_block_tests",
   !("whiten" %in% names(formals(.bqq_block_tests))))

## ---------- Part B: .bqq_block_tests on a QSS contrast matrix ----------
# 4 shape cells x r blocks. Kurtosis-only shift in block kb: K mean high, L/S/Sk ~ 0.
kb <- 3L
mu <- matrix(0, 4L, r); mu[4L, kb] <- 2.0        # cell 4 = K
cvc <- matrix(rnorm(n_iter * 4L * r, 0, sdc), n_iter, 4L * r)
for (j in seq_len(r)) for (c in 1:4)
  cvc[, (j - 1L) * 4L + c] <- cvc[, (j - 1L) * 4L + c] + mu[c, j]
res <- .bqq_block_tests(cvc, 4L, r, alpha, want_ui = TRUE, want_t2 = TRUE,
                        rowlab = c("L", "S", "Sk", "K"))

# manuscript charting constants
ok("c_T2 == qchisq((1-a)^{1/r}, df=4)",
   approx(res$hotelling_t2$c_calib, qchisq((1 - alpha)^(1 / r), df = 4L)))
ok("c_UI == qnorm((1+(1-a)^{1/(4r)})/2)",
   approx(res$ui$c_calib, qnorm((1 + (1 - alpha)^(1 / (4L * r))) / 2)))
# new overall T^2 = max block, p = 1 - (P(chi2_4 <= T2))^r
Wblk <- res$hotelling_t2$stat
ok("overall_t2 == max_j T^2_j", approx(res$overall_t2, max(Wblk)))
ok("overall_t2_p == 1 - pchisq(max, df=4)^r",
   approx(res$overall_t2_p, 1 - pchisq(res$overall_t2, df = 4L)^r))
# covariance whitening: sum of whitened z^2 == Mahalanobis gammabar' Sigma^-1 gammabar
cbar <- colMeans(cvc); Sig <- cov(cvc)
maha <- as.numeric(t(cbar) %*% solve(Sig) %*% cbar)
ok("sum(z~^2) == cbar' Sigma^-1 cbar (covariance whitening)",
   approx(sum(res$cellstat), maha, tol = 1e-6))
# the kurtosis block is flagged and driven by the K cell
ok("kurtosis block flagged by T^2 (calibrated)", kb %in% res$hotelling_t2$sig_calib)
ok("kurtosis block flagged by UI (calibrated)",  kb %in% res$ui$sig_calib)
ok("K cell dominates |z~| in the shifted block",
   which.max(abs(res$z_white[, kb])) == 4L)
ok("no shift block is NOT flagged (block 1)",
   !(1L %in% res$hotelling_t2$sig_calib) && !(1L %in% res$ui$sig_calib))

## ---------- Part C: end-to-end detectChangepoints_gamma (new K contrast) ----------
m <- 5L; taus <- c(0.025, 0.25, 0.5, 0.75, 0.975)
# gamma draws [n_iter, 5, r]; block kb: pure tail widening g1=-1, g5=+1 (KS ~ 2), others 0
gam <- array(rnorm(n_iter * m * r, 0, sdc), dim = c(n_iter, m, r))
gam[, 1, kb] <- gam[, 1, kb] - 1.0      # g1
gam[, 5, kb] <- gam[, 5, kb] + 1.0      # g5
fit <- list(fit_method = "map", laplace_samples = list(gamma = gam))
det <- detectChangepoints_gamma(fit, taus = taus, l = 30L, w = 30L,
                                signal_position = "first",
                                basis = c("quantile", "qss"),
                                statistic = c("ui", "hotelling_t2"))
ok("end-to-end runs; qss tests present", !is.null(det$tests$qss))
ok("k0_hat removed from output", is.null(det$k0_hat))
ok("whiten removed from output config", is.null(det$whiten))
# The K contrast should be the dominant QSS cell in block kb (rows L,S,Sk,K)
zq <- det$tests$qss$z          # studentized shape contrasts, 4 x r
ok("KS = g5-g4+g2-g1 drives block kb (K row largest |z|)",
   which.max(abs(zq[, kb])) == 4L)
ok("qss T^2 flags the kurtosis block", kb %in% det$tests$qss$hotelling_t2$sig_calib)
ok("quantile basis still computed", !is.null(det$tests$quantile$ui))

## ---------- Part C2: each shape contrast isolates its own shift ----------
# Pure shifts, one per block: L (g3), S (g4-g2), Sk (g2=g4 up, median fixed), K (g1/g5 tails).
gam2 <- array(rnorm(n_iter * m * r, 0, sdc), dim = c(n_iter, m, r))
gam2[, 3, 1] <- gam2[, 3, 1] + 1                                  # block 1: pure location (LS)
gam2[, 4, 2] <- gam2[, 4, 2] + 1; gam2[, 2, 2] <- gam2[, 2, 2] - 1 # block 2: pure scale  (ScS)
gam2[, 2, 3] <- gam2[, 2, 3] + 1; gam2[, 4, 3] <- gam2[, 4, 3] + 1 # block 3: pure skewness (SkS)
gam2[, 1, 4] <- gam2[, 1, 4] - 1; gam2[, 5, 4] <- gam2[, 5, 4] + 1 # block 4: pure kurtosis (KS)
det2 <- suppressWarnings(detectChangepoints_gamma(
  list(fit_method = "map", laplace_samples = list(gamma = gam2)),
  taus = taus, l = 30L, w = 30L, signal_position = "first",
  basis = "qss", statistic = c("ui", "hotelling_t2")))
z2 <- det2$tests$qss$z    # rows L, S, Sk, K
ok("LS  (gamma_3) isolates the Location cell in block 1",   which.max(abs(z2[, 1])) == 1L)
ok("ScS (gamma_4-gamma_2) isolates the Scale cell in block 2", which.max(abs(z2[, 2])) == 2L)
ok("SkS (gamma_2-2gamma_3+gamma_4) isolates the Skewness cell in block 3", which.max(abs(z2[, 3])) == 3L)
ok("KS  (gamma_5-gamma_4+gamma_2-gamma_1) isolates the Kurtosis cell in block 4", which.max(abs(z2[, 4])) == 4L)
ok("all four shape blocks flagged (T^2)",
   all(1:4 %in% det2$tests$qss$hotelling_t2$sig_calib))

## ---------- Part D: getQSS visualization kurtosis = (h5-h1)/IQR ----------
eta <- array(NA_real_, dim = c(3L, 5L, 2L))
eta[, , 1] <- matrix(c(-2, -0.6, 0, 0.6, 2), 3, 5, byrow = TRUE)
eta[, , 2] <- matrix(c(-3, -0.6, 0, 0.6, 3), 3, 5, byrow = TRUE)  # wider tails
q <- getQSS(eta, taus = taus)
ok("getQSS kurtosis == (h5-h1)/IQR (Table 2 standardized form)",
   approx(q[1, 4, 1], (2 - (-2)) / (0.6 - (-0.6))) &&
     approx(q[1, 4, 2], (3 - (-3)) / (0.6 - (-0.6))))

cat("\nALL SMOKE-TEST ASSERTIONS PASSED\n")
