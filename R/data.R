#' Daily oxycodone shipments to Alabama pharmacies, 2015-2019 (ARCOS)
#'
#' The series of the ARCOS illustration in the BQQ manuscript and the JSM 2026 talk:
#' morphine milligram equivalents (MME) of oxycodone shipped by distributors to
#' Alabama pharmacies per day, per state resident, built from the DEA Automation of
#' Reports and Consolidated Orders System (ARCOS) transaction records released by
#' The Washington Post, aggregated by transaction date and divided by the state
#' population of the year. The analyses fit the residuals of a linear regression of
#' \code{mme_per_capita} on the holiday indicator and the day of the week.
#'
#' @format A data frame with 1826 rows (2015-01-01 to 2019-12-31) and 4 columns:
#' \describe{
#'   \item{date}{Date.}
#'   \item{mme_per_capita}{Oxycodone MME shipped that day per state resident.}
#'   \item{holiday}{1 on an Alabama public holiday, 0 otherwise.}
#'   \item{weekday}{Day of the week, 1 = Sunday to 7 = Saturday.}
#' }
#' @source DEA ARCOS transaction data, 2006-2019 release by The Washington Post
#'   (\url{https://www.washingtonpost.com/national/2019/07/18/how-download-use-dea-pain-pills-database/}),
#'   oxycodone transactions with Alabama buyers; annual state population from the
#'   U.S. Census Bureau.
#' @examples
#' data(arcos_al)
#' y <- residuals(lm(mme_per_capita ~ holiday + factor(weekday), data = arcos_al))
"arcos_al"
