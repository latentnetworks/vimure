#' Check reporters mask
#'
#' Check if reporters mask is valid. That is, if it is an array with four
#' dimensions, if the second, third, and fourth dimensions have the same
#' length, and if it is either entirely TRUE/FALSE or entirely 0s/1s.
#'
#' @family vimure
#' @param R Reporters mask (TRUE/FALSE) indicating whether a node
#' _CAN_ report on a particular tie, with dimensions L x N x N x N.
#' @return TRUE if reporters mask is valid, FALSE otherwise.
#'
#' @export
#'
check_reporters_mask <- function(R){
  # CHECK 01: if R is an array
  is_sptensor <- inherits(R, "sktensor.sptensor.sptensor")
  is_R_array  <- is.array(R)

  if (!(is_sptensor || is_R_array)) {
      stop("Invalid type='", class(R), "' for parameter R. R must be an array!")
  }

  # CHECK 02: if R has four dimensions
  valid_mask <- TRUE
  if (is_sptensor) {
    if (R$ndim != 4) {
      valid_mask <- FALSE
    }
  } else if (length(dim(R)) != 4) {
    valid_mask <- FALSE
  }

  if (!valid_mask) {
    stop("R must have four dimensions.")
  }

  # CHECK 03: if second, third, and fourth dimensions have the same length
  dims <- if (is_sptensor) {
    unlist(R$shape)
  } else {
    dim(R)
  }
  N <- dims[2]
  if (any(dims[2:4] != N)) {
    stop("The 2nd, 3rd, and 4th dimensions of R must have the same length.")
  }

  # CHECK 04: if R is either entirely TRUE/FALSE or entirely 0s/1s
  vals <- if (is_sptensor) {
    R$vals
  } else {
    R
  }

  if (!all(vals == TRUE | vals == FALSE) && !all(vals == 0 | vals == 1)) {
    stop("R must be either entirely made of TRUE/FALSE or of 0s/1s.")
  }

  # if all conditions are met, return TRUE
  return(TRUE)
}


#' ViMuRe
#'
#' Fit a probabilistic generative model to double sampled networks.
#' It returns reliability parameters for the reporters (theta),
#' average interactions for the links (lambda) and the estimate of
#' the true and unknown network (rho). 
#' The inference is performed with a Variational Inference approach.
#'
#' @family vimure
#'
#' @param x An adjancency matrix with dimensions L x N x N x N
#' or a igraph object.
#' @param R Reporters mask (TRUE/FALSE) indicating whether a node
#' _CAN_ report on a particular tie, with dimensions L x N x N x N.
#' If reporters mask was not informed, the model will assume that
#' every reporter can report on any tie.
#' @param K Value of the maximum entry of the network - i.
#' @param undirected Whether the network is undirected.
#' @param mutuality Whether to use the mutuality parameter.
#' @param theta_prior Shape and scale hyperparameters for variable theta.
#' @param lambda_prior Shape and scale hyperparameters for variable lambda.
#' @param eta_prior Shape and scale hyperparameters for variable eta.
#' @param rho_prior Array with prior values of the rho parameter - if ndarray.
#' @param etol controls when to stop the optimisation algorithm (CAVI).
#' @param seed Pseudo random generator seed to use.
#' @param verbose Provides additional details.
#' @param ... Additional args of model$fit() method.
#'
#' @return vimure model
#' @export
vimure <-
  function(x,
           R = NULL,
           mutuality = TRUE,
           undirected = FALSE,
           theta_prior = c(0.1, 0.1),
           K = NULL,
           lambda_prior = c(10.0, 10.0),
           eta_prior = c(0.5, 1.0),
           rho_prior = NULL,
           seed = NULL,
           etol = 0.1,
           verbose = FALSE,
           ...) {

  valid_classes <- c("sktensor.sptensor.sptensor", "array",
                     "igraph", "data.frame", "tbl_df")
  if (!any(class(x) %in% valid_classes)) {
    stop("invalid 'type'", class(x), "of argument")
  }

  model <- vimureP$model$VimureModel(mutuality = mutuality,
                                     undirected = undirected,
                                     convergence_tol = etol,
                                     verbose = verbose)

  if (length(seed) > 0) {
    seed <- as.integer(seed)
  }

  if (length(K) > 0) {
    K <- as.integer(K)
  }

  if (length(theta_prior) > 0) {
    theta_prior <- reticulate::tuple(as.list(theta_prior))
  }

  if (length(lambda_prior) > 0) {
    lambda_prior <- reticulate::tuple(as.list(lambda_prior))
  }

  if (length(eta_prior) > 0) {
    eta_prior <- reticulate::tuple(as.list(eta_prior))
  }

  if (length(R) > 0) {
    check_reporters_mask(R)

    model$fit(
      X = x, R = R, K = K, theta_prior = theta_prior,
      lambda_prior = lambda_prior, eta_prior = eta_prior,
      rho_prior = rho_prior, seed = seed, ...
    )
  }else {
    model$fit(
      X = x, K = K, theta_prior = theta_prior,
      lambda_prior = lambda_prior, eta_prior = eta_prior,
      rho_prior = rho_prior, seed = seed, ...
    )
  }

  return(model)
}

#' Estimate Y
#'
#' @family vimure
#'
#' @details Use this function to reconstruct the Y matrix
#' with a fitted vimure model.
#' It will use `model$rho_f` values to extract an estimated Y matrix.
#' \itemize{
#'  \item{*rho_max*: }{Assign the value of the highest probability}
#'  \item{*rho_mean*: }{Expected value of the discrete distribution}
#'  \item{*fixed_threshold*: }{Check if the probability is higher than
#' a threshold (Only for 2 categories)}
#'  \item{*heuristic_threshold*: }{Calculate and use the best threshold
#' (Only for 2 categories)}
#' }
#'
#'
#' @param object A "vimure" object.
#' @param method A character string indicating which method is to be computed.
#' One of "rho_max" (default), "rho_mean", "fixed_threshold"
#' or "heuristic_threshold".
#' @param threshold A threshold to be used when method = "fixed_threshold".
#'
#' @export
get_inferred_model <- function(object, method = "rho_max", threshold = NULL) {
  if ("rho_f" %in% names(object)) {
    return(object$get_inferred_model(method, threshold))
  } else {
    stop('"object" is not a fitted vimure model')
  }
}

#' Sample Y trials from rho distribution

#' Use this function to sample Y trials with a fitted vimure model.
#' It will use `model.rho_f` as the probabilities of a discrete distribution.
#'
#' @param object A "vimure" object.
#' @param N Number of trials.
#' @param seed A pseudo generator seed.
#'
#' @export
sample_inferred_model <- function(object, N = 1, seed = NULL){
  N <- as.integer(N)
  if( "rho_f" %in% names(object)) {
    return(object$sample_inferred_model(N = N, seed = seed))
  } else {
    stop('"object" is not a fitted vimure model')
  }
}

#' Diagnostics metrics
#' @family vimure
#'
#' @param object A "vimure" object.
#' @param net A vm.io.baseNetwork object.
#' @param ... Additional arguments affecting the summary produced.
#'
#' @rdname summary.vimure.model.VimureModel
#' @export
summary.vimure.model.VimureModel <- function(object, net = NULL, ...) {
  if (length(net) == 0) {
    net <- reticulate::py_none()
  }

  diag <- vimureP$diagnostics$Diagnostics(object, net)
  print(diag)
  return(vimure_training_history(diag))
}

vimure_training_history <- function(diag) {
  params <- strsplit(diag$model_str, "\n")[[1]]
  priors <- list()
  for (param in params){
    if (grepl("Posterior", param)) {
      break
    }
    else{ # nolint: brace_linter.
      param <- gsub("-|^\\s*", "", param)
      param <- gsub("\\s+", " ", param)
      prior <- strsplit(param, ": ")[[1]]
      if (length(prior) == 2 && grepl("=", prior[2])){
        result <- eval(parse(text = paste0(
          "list(", prior[1], "=c(", gsub(" ", ",", prior[2]), "))"))
          )
        priors <- append(priors, result)
      }
    }
  }
  priors$rho <- diag$model$pr_rho

  posteriors <- list(
    G_exp_lambda_f = diag$model$G_exp_lambda_f,
    G_exp_nu_f = diag$model$G_exp_nu_f,
    G_exp_theta_f = diag$model$G_exp_theta_f
  )

  if(reticulate::py_has_attr(diag$model, "rho_f")){
    posteriors$rho_f <- diag$model$rho_f
  }

  # Reliability dataframe
  lambda <- posteriors$G_exp_lambda_f[, 2]
  theta_matrix <- t(posteriors$G_exp_theta_f)
  reliability <- theta_matrix * lambda
  reliability_df <- data.frame(layer = reliability)
  reliability_df$node <- row.names(reliability_df)

  structure(class = "vimure_training_history", list(
    priors = priors,
    posteriors = posteriors,
    trace = diag$model$trace,
    reliability = reliability_df,
    model_str = diag$model_str
  ))
}

#' @export
print.vimure_training_history <- function(x, ...) {
  cat(x$model_str, "\n")
}
