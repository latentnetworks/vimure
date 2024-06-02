#' Generating Observed Networks - X
#'
#' Given a network `Y`, we can generate N observed adjacency matrices as would
#' have been reported by reporting nodes $m$ ($m in N$).
#'
#' @param synthetic A synthetic model
#' @param mutuality The mutuality parameter (from 0 to 1)
#' @param sh_theta Shape of gamma distribution from which to draw theta. The 'reliability' of nodes is represented by the parameter $theta_{lm}$ and by default are modelled as a gamma function with shape `sh_theta` and scale `sc_theta`.
#' @param sc_theta Scale of gamma distribution from which to draw theta.
#' @param theta Custom values of theta, if provided sh_theta and sc_theta will be ignored.
#' @param reporting_scenario The reporting scenario to be used. Default is 2.
#' - Scenario 1: Reporters report only on ties they are involved in
#' as ego (excluding self-loops).
#' - Scenario 2: Reporters report on all ties they are involved in,
#' as either ego or alter (double sampling, excluding self-loops).
#' - Scenario 3: Ego reports on any ties,
#' even those they are not involved in (excludes self-loops).
#' @param Q Maximum value of X entries. If None, it will use the network's K parameter
#' @param cutoff_X Whether to set X as a binary
#' @param lambda_diff The difference between each subsequent K
#' @param verbose Provides additional details
#' @param ...  Additional args of synthetic$build_X() method.
#'
#' @return A sptensor
#' @export
#' @examples
#'
#' random_net <- gm_crep(N=100, M=100, L=1, eta=0.99)
#' X <- build_X(random_net, reporting_scenario=2, cutoff_X=FALSE, seed=10L)
#' dim(X)
build_X <- function(
  synthetic,
  mutuality = 0.5,
  sh_theta = 2,
  sc_theta = 0.5,
  theta = NULL,
  reporting_scenario = 2,
  Q = NULL,
  cutoff_X = FALSE,
  lambda_diff = NULL,
  verbose = TRUE,
  ...) {
  synthetic$`_build_X`(
    mutuality = mutuality,
    sh_theta = sh_theta,
    sc_theta = sc_theta,
    theta = theta,
    reporting_scenario = reporting_scenario,
    Q = Q,
    cutoff_X = cutoff_X,
    lambda_diff = lambda_diff,
    verbose = verbose,
    ...
  )

  return(py_to_r.sktensor.sptensor.sptensor(synthetic$X))
}


#' Extract the Xavg matrix
#'
#' Average value of observed network X over reporter mask.
#'
#' @param synthetic A synthetic model
#'
#' @return A matrix
#' @export
#'
#' @examples
#' random_net <- gm_crep(N=100, M=100, L=1, eta=0.99)
#' X <- build_X(random_net)
#' Xavg <- extract_Xavg(random_net)
#' dim(Xavg)
extract_Xavg <- function(synthetic){
  Xavg <- vimureP$utils$calculate_average_over_reporter_mask(
    synthetic$X, synthetic$R
  )

  return(Xavg)
}
