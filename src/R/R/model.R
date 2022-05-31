#' ViMuRe
#'
#' Fit a probabilistic generative model to double sampled networks. It returns reliability parameters for the
#' reporters (theta), average interactions for the links (lambda) and the estimate of the true and unknown
#' network (rho). The inference is performed with a Variational Inference approach.
#'
#' @param x An adjancency matrix with dimensions L x N x N x N or a igraph object.
#' @param R Reporters mask (TRUE/FALSE) indicating whether a node _CAN_ report on a particular tie, with dimensions L x N x N x N.
#' If reporters mask was not informed, the model will assume that every reporter can report on any tie.
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
vimure <- function(x, R=NULL, mutuality=T, undirected=F, theta_prior=c(0.1, 0.1), K=NULL,
                   lambda_prior=c(10.0, 10.0), eta_prior=c(0.5, 1.0), rho_prior=NULL,
                   seed=NULL, etol=0.1, verbose=T, ...){
  if(!any(class(x) %in% c("sktensor.sptensor.sptensor", "array", "igraph"))){
    stop("invalid 'type'", class(x), "of argument")
  }

  if("igraph" %in% class(x)){
    net <- parse_graph_from_igraph(x)
    x <- net$X
    R <- net$R
  }

  model <- vimureP$model$VimureModel(mutuality=mutuality, undirected=undirected, convergence_tol=etol, verbose=verbose)

  if(length(seed) > 0){seed <- as.integer(seed)}
  if(length(K) > 0){K <- as.integer(K)}
  if(length(theta_prior) > 0){theta_prior <- reticulate::tuple(as.list(theta_prior))}
  if(length(lambda_prior) > 0){lambda_prior <- reticulate::tuple(as.list(lambda_prior))}
  if(length(eta_prior) > 0){eta_prior <- reticulate::tuple(as.list(eta_prior))}

  if(length(R) > 0){
    if(!any(class(R) %in% c("sktensor.sptensor.sptensor", "array"))){
      stop("invalid 'type'", class(R), "of argument")
    }

    model$fit(
      X=x, R=R, K=K, theta_prior=theta_prior,
      lambda_prior=lambda_prior, eta_prior=eta_prior,
      rho_prior=rho_prior, seed=seed, ...
    )
  }else{
    model$fit(
      X=x, K=K, theta_prior=theta_prior,
      lambda_prior=lambda_prior, eta_prior=eta_prior,
      rho_prior=rho_prior, seed=seed, ...
    )
  }

  return(model)
}

#' Diagnostics metrics
#'
#'
#' @param object A "vimure" object.
#' @param net A vm.io.baseNetwork object.
#' @param ... Additional arguments affecting the summary produced.
#'
#' @rdname summary.vimure.model.VimureModel
#' @export
summary.vimure.model.VimureModel <- function(object, net=NULL, ...){
  if(length(net) == 0){
    net <- reticulate::py_none()
  }

  diag <- vimureP$diagnostics$Diagnostics(object, net)
  print(diag)
  return(vimure_training_history(diag))
}

vimure_training_history <- function(diag){
  params <- strsplit(diag$model_str, "\n")[[1]]
  priors <- list()
  for(param in params){
    if(grepl("Posterior", param)){
      break
    }
    else{
      param <- gsub("-|^\\s*", "", param)
      param <- gsub("\\s+", " ", param)
      prior <- strsplit(param, ": ")[[1]]
      if(length(prior) == 2 & grepl("=", prior[2])){
        result <- eval(parse(text=paste0(
          "list(", prior[1],"=c(", gsub(" ", ",", prior[2]), "))"))
          )
        priors <- append(priors, result)
      }
    }
  }
  priors$rho <- diag$model$pr_rho

  posteriors <- list(
    G_exp_lambda_f=diag$model$G_exp_lambda_f,
    G_exp_nu_f=diag$model$G_exp_nu_f,
    G_exp_theta_f=diag$model$G_exp_theta_f
  )

  if(reticulate::py_has_attr(diag$model, "rho_f")){
    posteriors$rho_f <- diag$model$rho_f
  }

  # Realibility dataframe
  lambda <- posteriors$G_exp_lambda_f[,2]
  theta_matrix <- t(posteriors$G_exp_theta_f)
  reliability <- theta_matrix*lambda
  reliability_df <- data.frame(layer=reliability)
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
print.vimure_training_history <- function(x, ...){
  cat(x$model_str, "\n")
}

