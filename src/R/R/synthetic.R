#' Standard stochastic blockmodel
#'
#' A generative graph model which assumes the probability of connecting two nodes in a graph is determined entirely by their block assignments.
#' For more information about this model, see Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). _Stochastic blockmodels: First steps. Social networks_, 5(2), 109-137.
#' [DOI:10.1016/0378-8733(83)90021-7](https://www.sciencedirect.com/science/article/abs/pii/0378873383900217)
#'
#' @family generative models
#'
#' @param N Number of nodes
#' @param M Number of reporters
#' @param L Number of layers (it has only been tested for L=1)
#' @param C Number of communities
#' @param K Maximum edge weight in the adjacency matrix. When K=2 (default), the adjacency matrix will contain some Y(ij)=0 and Y(ij)=1.
#' @param avg_degree Desired average degree for the network. It is not guaranteed that the ultimate network will have that exact average degree value. Try tweaking this parameter if you want to increase or decrease the density of the network.
#' @param sparsify If True (default), enforce sparsity.
#' @param overlapping Fraction of nodes with mixed membership. It has to be in [0, 1).
#' @param seed Pseudo random generator seed to use
#'
#' @return gm_StandardSBM model
#' @export
#'
#' @examples
#' random_net <- gm_StandardSBM(N=100, M=100, L=1)
#' Y <- extract_Y(random_net)
#' dim(Y)
gm_StandardSBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, overlapping=0, seed=10){
  INT_ARGS <- c(N=N, M=M, K=K, L=L, C=C, avg_degree=avg_degree, seed=seed)
  mode(INT_ARGS) <- "integer"

  synthetic <- vimureP$synthetic$StandardSBM(
    N=INT_ARGS["N"],
    M=INT_ARGS["M"],
    K=INT_ARGS["K"],
    L=INT_ARGS["L"],
    C=INT_ARGS["C"],
    avg_degree=INT_ARGS["avg_degree"],
    sparsify=sparsify,
    overlapping=overlapping,
    seed=INT_ARGS["seed"])

  return(synthetic)
}

#' Degree-corrected stochastic blockmodel
#'
#' A generative model that incorporates heterogeneous vertex degrees into stochastic blockmodels, improving the performance of the models for statistical inference of group structure.
#' For more information about this model, see Karrer, B., & Newman, M. E. (2011). _Stochastic blockmodels and community structure in networks_. Physical review E, 83(1), 016107.
#' [DOI:10.1103/PhysRevE.83.016107](https://arxiv.org/pdf/1008.3926.pdf)
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
#' Y <- extract_Y(random_net)
#' dim(Y)
gm_DegreeCorrectedSBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, exp_in=2, exp_out=2.5, sparsify=TRUE, seed=10){
  INT_ARGS <- c(N=N, M=M, K=K, C=C, L=L, seed=seed)
  mode(INT_ARGS) <- "integer"

  synthetic <- vimureP$synthetic$DegreeCorrectedSBM(
    N=INT_ARGS["N"],
    M=INT_ARGS["M"],
    K=INT_ARGS["K"],
    L=INT_ARGS["L"],
    C=INT_ARGS["C"],
    avg_degree=avg_degree,
    sparsify=sparsify,
    exp_in=exp_in,
    exp_out=exp_out,
    seed=INT_ARGS["seed"])

  return(synthetic)
}

#' Generative Model with Reciprocity (CRep)
#'
#' A mathematically principled generative model for capturing both community and reciprocity patterns in directed networks.
#' For more information about this model, see Safdari, H., Contisciani, M., & De Bacco, C. (2021). _Generative model for reciprocity and community detection in networks_. Physical Review Research, 3(2), 023209.
#' [DOI:10.1103/PhysRevResearch.3.023209](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023209).
#'
#' @family generative models
#'
#' @param eta Initial value for the reciprocity coefficient. Eta has to be in [0, 1).
#' @inheritParams gm_StandardSBM
#'
#' @return gm_CReciprocity model
#' @export
#'
#' @examples
#' random_net <- gm_CReciprocity(N=100, M=100, L=1, eta=0.5)
#' Y <- extract_Y(random_net)
#' dim(Y)
gm_CReciprocity <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, eta=0.99, seed=10){
  INT_ARGS <- c(N=N, M=M, K=K, L=L, C=C, avg_degree=avg_degree, seed=seed)
  mode(INT_ARGS) <- "integer"

  synthetic <- vimureP$synthetic$GMReciprocity(
    N=INT_ARGS["N"],
    M=INT_ARGS["M"],
    K=INT_ARGS["K"],
    L=INT_ARGS["L"],
    C=INT_ARGS["C"],
    avg_degree=INT_ARGS["avg_degree"],
    sparsify=sparsify,
    eta=eta,
    seed=INT_ARGS["seed"])

  return(synthetic)
}

#' Extract the Y matrix
#'
#' @param synthetic A synthetic model
#'
#' @return A sptensor
#' @export
#'
#' @examples
#' random_net <- gm_CReciprocity(N=100, M=100, L=1, eta=0.99)
#' Y <- extract_Y(random_net)
#' dim(Y)
extract_Y <- function(synthetic){
  py_to_r.sktensor.sptensor.sptensor(synthetic$Y)
}
