#' Generative Model with Reciprocity (CRep)
#'
#' For more information about this model, see Safdari, H., Contisciani, M., & De Bacco, C. (2021). _Generative model for reciprocity and community detection in networks_. Physical Review Research, 3(2), 023209.
#' [DOI:10.1103/PhysRevResearch.3.023209](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023209).
#'
#' @param N Number of nodes
#' @param M Number of reporters
#' @param L Number of layers (it has only been tested for L=1)
#' @param K Maximum edge weight in the adjacency matrix. When K=2 (default), the adjacency matrix will contain some Y(ij)=0 and Y(ij)=1.
#' @param C a
#' @param avg_degree Desired average degree for the network. It is not guaranteed that the ultimate network will have that exact average degree value. Try tweaking this parameter if you want to increase or decrease the density of the network.
#' @param sparsify a
#' @param eta a
#' @param seed Pseudo random generator seed to use
#'
#' @return GMReciprocity model
#' @export
#'
#' @examples
#' random_net <- GMReciprocity(N=100, M=100, L=1, eta=0.99)
#' Y <- extract_Y(random_net)
#' dim(Y)
GMReciprocity <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, eta=0.99, seed=10){
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
#' random_net <- GMReciprocity(N=100, M=100, L=1, eta=0.99)
#' Y <- extract_Y(random_net)
#' dim(Y)
extract_Y <- function(synthetic){
  Y <- synthetic$Y$toarray()
  extract_data(Y)
}


#' Standard stochastic block-model synthetic network
#' @return DegreeCorrectedSBM model
#' @export
StandardSBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, sparsify=TRUE, overlapping=0, seed=10){
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
#' @return DegreeCorrectedSBM model
#' @export
DegreeCorrectedSBM <- function(N=100, M=100, K=2, L=1, C=2, avg_degree=10, exp_in=2, exp_out=2.5, sparsify=TRUE, seed=10){
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

extract_data <- function(data){
  dim <- dim(data)
  if( (dim[1] == 1)){
    dim(data) <- utils::tail(dim, length(dim)-1)
  }

  return(data)
}

py_to_r.sktensor.sptensor.sptensor <- function(x){
  vals <- as.vector(x$vals)
  dims <- unlist(x$shape)

  subs <- list()
  for(i in 1:length(vals)){
    sub <- list(c(x$subs[[1]][i]+1, x$subs[[2]][i]+1, x$subs[[3]][i]+1))
    subs <- append(sub, subs)
  }

  tensorr::sptensor(subs, vals, dims)
}

