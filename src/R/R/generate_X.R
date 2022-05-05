
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
#' @param flag_self_reporter Indicates whether a node can only report about their own ties (default is true).
#' @param Q Maximum value of X entries. If None, it will use the network's K parameter
#' @param cutoff_X Whether to set X as a binary
#' @param lambda_diff The difference between each subsequent K
#' @param seed  Pseudo random generator seed to use
#' @param verbose Provides additional details
#'
#' @return A character vector.
#' @export
#' @examples
#'
#' random_net <- GMReciprocity(N=100, M=100, L=1, eta=0.99)
#' X <- build_X(random_net, flag_self_reporter=TRUE, cutoff_X=FALSE, seed=10L)
#' dim(X)
build_X <- function(
  synthetic, mutuality=0.5, sh_theta=2, sc_theta=0.5, theta=reticulate::py_none(), flag_self_reporter=T,
  Q=reticulate::py_none(), cutoff_X=F, lambda_diff=reticulate::py_none(), seed=reticulate::py_none(), verbose=T
){
  synthetic$build_X(
    mutuality=mutuality,
    sh_theta=sh_theta,
    sc_theta=sc_theta,
    theta=theta,
    flag_self_reporter=flag_self_reporter,
    Q=Q,
    cutoff_X=cutoff_X,
    lambda_diff=lambda_diff,
    seed=seed,
    verbose=verbose
  )
}


#' Extract the Xavg matrix (considering the reporter's mask)
#'
#' @param synthetic A synthetic model
#'
#' @return A matrix
#' @export
#'
#' @examples
#' random_net <- GMReciprocity(N=100, M=100, L=1, eta=0.99)
#' X <- build_X(random_net)
#' Xavg <- extract_Xavg(synthetic)
#' dim(Xavg)
extract_Xavg <- function(synthetic){
  Xavg <- vimureP$utils$calculate_average_over_reporter_mask(
    synthetic$X, synthetic$R
  )

  Xavg <- extract_data(Xavg)
  return(Xavg)
}
