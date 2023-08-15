#' Standard stochastic blockmodel
#'
#' A generative graph model which assumes the probability of connecting two
#' nodes in a graph is determined entirely by their block assignments.
#' For more information about this model, see
#' Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983).
#' _Stochastic blockmodels: First steps. Social networks_, 5(2), 109-137.
#' DOI:10.1016/0378-8733(83)90021-7
#'
#' @family generative models
#'
#' @param N Number of nodes
#' @param M Number of reporters
#' @param L Number of layers (it has only been tested for L=1)
#' @param C Number of communities
#' @param K Maximum edge weight in the adjacency matrix.
#'    When K=2 (default), the adjacency matrix will contain some
#'    Y(ij)=0 and Y(ij)=1.
#' @param avg_degree Desired average degree for the network.
#'    It is not guaranteed that the ultimate network will have that exact
#'    average degree value. Try tweaking this parameter if you want to increase
#'    or decrease the density of the network.
#' @param sparsify If True (default), enforce sparsity.
#' @param overlapping Fraction of nodes with mixed membership.
#'    It has to be in [0, 1).
#' @param seed Pseudo random generator seed to use
#'
#' @return gm_StandardSBM model
#' @export
#'
#' @examples
#' random_net <- gm_StandardSBM(N = 100, M = 100, L = 1)
#' Y <- random_net$Y$toarray()
#' dim(Y)
gm_standard_sbm <- function(N = 100,
                            M = 100,
                            K = 2,
                            L = 1,
                            C = 2,
                            avg_degree = 10,
                            sparsify = TRUE,
                            overlapping = 0,
                            seed = 10) {

  synthetic <- vimureP$synthetic$StandardSBM(
    N = as.integer(N),
    M = as.integer(M),
    K = as.integer(K),
    L = as.integer(L),
    C = as.integer(C),
    avg_degree = as.numeric(avg_degree),
    sparsify = as.logical(sparsify),
    overlapping = as.numeric(overlapping),
    seed = as.integer(seed)
  )

  return(synthetic)
}

#' Degree-corrected stochastic blockmodel
#'
#' A generative model that incorporates heterogeneous vertex degrees into
#' stochastic blockmodels, improving the performance of the models for
#' statistical inference of group structure.
#' For more information about this model, see
#' Karrer, B., & Newman, M. E. (2011).
#' _Stochastic blockmodels and community structure in networks_.
#' Physical review E, 83(1), 016107. DOI:10.1103/PhysRevE.83.016107
#'
#' @family generative models
#'
#' @param exp_in Exponent power law of in-degree distribution
#' @param exp_out Exponent power law of out-degree distribution
#' @inheritParams gm_StandardSBM
#'
#' @return gm_DegreeCorrectedSBM model
#' @export
#'
#' @examples
#' random_net <- gm_DegreeCorrectedSBM(exp_in = 2, exp_out = 2.5)
#' Y <- random_net$Y$toarray()
#' dim(Y)
gm_degree_corrected_sbm <- function(N = 100,
                                    M = 100,
                                    K = 2,
                                    L = 1,
                                    C = 2,
                                    avg_degree = 10,
                                    exp_in = 2,
                                    exp_out = 2.5,
                                    sparsify = TRUE,
                                    seed = 10) {


  synthetic <- vimureP$synthetic$DegreeCorrectedSBM(
    N = as.integer(N),
    M = as.integer(M),
    K = as.integer(K),
    L = as.integer(L),
    C = as.integer(C),
    avg_degree = as.numeric(avg_degree),
    sparsify = as.logical(sparsify),
    exp_in = as.numeric(exp_in),
    exp_out = as.numeric(exp_out),
    seed = as.integer(seed)
  )

  return(synthetic)
}

#' Generative Model with Reciprocity (CRep)
#'
#' A mathematically principled generative model for capturing both community
#' and reciprocity patterns in directed networks.
#' For more information about this model, see
#' Safdari, H., Contisciani, M., & De Bacco, C. (2021).
#' _Generative model for reciprocity and community detection in networks_.
#' Physical Review Research, 3(2), 023209.
#' DOI:10.1103/PhysRevResearch.3.023209
#'
#' @family generative models
#'
#' @param eta Initial value for the reciprocity coefficient.
#'  Eta has to be in [0, 1).
#' @inheritParams gm_StandardSBM
#'
#' @return gm_Multitensor model
#' @export
#'
#' @examples
#' random_net <- gm_Multitensor(N = 100, M = 100, L = 1, eta = 0.5)
#' Y <- random_net$Y$toarray()
#' dim(Y)
gm_crep <- function(N = 100,
                    M = 100,
                    K = 2,
                    L = 1,
                    C = 2,
                    avg_degree = 10,
                    sparsify = TRUE,
                    eta = 0.99,
                    seed = 10) {

  synthetic <- vimureP$synthetic$Multitensor(
    N = as.integer(N),
    M = as.integer(M),
    K = as.integer(K),
    L = as.integer(L),
    C = as.integer(C),
    avg_degree = as.numeric(avg_degree),
    sparsify = as.logical(sparsify),
    eta = as.numeric(eta),
    seed = as.integer(seed)
  )

  return(synthetic)
}

#' Customize 'Reliability' parameter theta
#'
#' Instead of the regular generative model for theta (theta ~ Gamma(sh, sc)),
#' create a more extreme scenario where some percentage of reporters exaggerates
#'
#' @param synthetic A synthetic model
#' @param theta_ratio Percentage of reporters who exaggerate \[0,1]
#' @param exaggeration_type "over" or "under"
#' @param seed Pseudo random generator seed to use
#'
#' @return A L x M matrix for theta
#' @export
build_custom_theta <- function(synthetic, theta_ratio = 0.5,
                               exaggeration_type = c("over", "under"),
                               seed = NULL) {
  exaggeration_type <- match.arg(exaggeration_type)
  custom_theta <- vimureP$synthetic$build_custom_theta(
    gt_network = synthetic,
    theta_ratio = theta_ratio,
    exaggeration_type = exaggeration_type,
    seed = as.integer(seed)
  )
  return(custom_theta)
}
