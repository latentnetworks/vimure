#' Standard stochastic block-model synthetic network
#'
#' More info [here](https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0170-z.pdf#page=22&zoom=100,148,205)
#'
#' @param N Number of nodes
#' @param M Number of reporters
#' @param L Number of layers (it has only been tested for L=1)
#' @param C Number of communities
#' @param K Maximum edge weight in the adjacency matrix. When K=2 (default), the adjacency matrix will contain some Y(ij)=0 and Y(ij)=1.
#' @param avg_degree Desired average degree for the network. It is not guaranteed that the ultimate network will have that exact average degree value. Try tweaking this parameter if you want to increase or decrease the density of the network.
#' @param sparsify If True (default), enforce sparsity.
#' @param overlapping n/a
#' @param seed Pseudo random generator seed to use
#'
#' @return synthetic_SBM model
#' @export
#'
#' @examples
#' random_net <- synthetic_SBM(N=100, M=100, L=1)
#' Y <- extract_Y(random_net)
#' dim(Y)
synthetic_SBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, overlapping=0, seed=10){
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

#' DegreeCorrectedSBM
#'
#' More info [here](https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0170-z.pdf#page=22&zoom=100,148,205)
#'
#' @param exp_in n/a
#' @param exp_out n/a
#' @inheritParams synthetic_SBM
#'
#' @return synthetic_DegreeSBM model
#' @export
#'
#' @examples
#' random_net <- synthetic_DegreeSBM(exp_in = 2, exp_out = 2.5)
#' Y <- extract_Y(random_net)
#' dim(Y)
synthetic_DegreeSBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, exp_in=2, exp_out=2.5, sparsify=TRUE, seed=10){
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
#' For more information about this model, see Safdari, H., Contisciani, M., & De Bacco, C. (2021). _Generative model for reciprocity and community detection in networks_. Physical Review Research, 3(2), 023209.
#' [DOI:10.1103/PhysRevResearch.3.023209](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023209).
#'
#' @param eta Initial value for the reciprocity coefficient. Eta has to be in [0, 1).
#' @inheritParams synthetic_SBM
#'
#' @return synthetic_CRep model
#' @export
#'
#' @examples
#' random_net <- synthetic_CRep(N=100, M=100, L=1, eta=0.5)
#' Y <- extract_Y(random_net)
#' dim(Y)
synthetic_CRep <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, eta=0.99, seed=10){
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
#' @return A matrix
#' @export
#'
#' @examples
#' random_net <- synthetic_CRep(N=100, M=100, L=1, eta=0.99)
#' Y <- extract_Y(random_net)
#' dim(Y)
extract_Y <- function(synthetic){
  Y <- synthetic$Y$toarray()
  Y
}
