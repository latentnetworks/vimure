#' ViMuRe
#'
#' Fit a probabilistic generative model to double sampled networks. It returns reliability parameters for the
#' reporters (theta), average interactions for the links (lambda) and the estimate of the true and unknown
#' network (rho). The inference is performed with a Variational Inference approach.
#'
#' @param X The observed data, with dimensions L x N x N x N
#' @param R Reporters mask (TRUE/FALSE) indicating whether a node _CAN_ report on a particular tie, with dimensions L x N x N x N.
#' If reporters mask was not informed, the model will assume that every reporter can report on any tie.
#' @param K Value of the maximum entry of the network - i
#' @param undirected Whether the network is undirected
#' @param mutuality Whether to use the mutuality parameter
#' @param theta_prior Shape and scale hyperparameters for variable theta
#' @param lambda_prior Shape and scale hyperparameters for variable lambda
#' @param eta_prior Shape and scale hyperparameters for variable eta
#' @param rho_prior Array with prior values of the rho parameter - if ndarray.
#' @param etol controls when to stop the optimisation algorithm (CAVI)
#' @param seed Pseudo random generator seed to use
#' @param verbose Provides additional details
#'
#' @return vimure model
#' @export
vimure <- function(X, R=NULL, mutuality=T, undirected=F, theta_prior=c(0.1, 0.1), K=reticulate::py_none(),
                   lambda_prior=c(10.0, 10.0), eta_prior=c(0.5, 1.0), rho_prior=reticulate::py_none(),
                   seed=reticulate::py_none(),etol=0.1, verbose=T){
  model <- vimureP$model$VimureModel(mutuality=mutuality, undirected=undirected, convergence_tol=etol, verbose=verbose)

  if(length(R)){
    model$fit(
      X=X, R=R, K=K, theta_prior=theta_prior,
      lambda_prior=lambda_prior, eta_prior=eta_prior,
      rho_prior=rho_prior, seed=seed
    )
  }else{
    model$fit(
      X=X, K=K, theta_prior=theta_prior,
      lambda_prior=lambda_prior, eta_prior=eta_prior,
      rho_prior=rho_prior, seed=seed
    )
  }

  return(model)
}

#' Diagnostics metrics
#'
#'
#' @param object a "vimure" object
#' @param net a "baseNetwork" object
#' @param ... additional arguments affecting the summary produced.
#'
#' @rdname summary.vimure.model.VimureModel
#' @export
summary.vimure.model.VimureModel <- function(object, net, ...){
  diag <- vimureP$diagnostics$Diagnostics(object, net)
  print(diag)
  print(diag$plot_elbo_values())
  if(reticulate::py_has_attr(net, "theta")){
    print(diag$plot_theta(theta_GT = net$theta))
  } else {
    print(diag$plot_theta())
  }
  vimure_training_history(diag)
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

  structure(class = "vimure_training_history", list(
    priors = priors,
    posteriors = posteriors,
    trace = diag$model$trace
  ), comment= diag$reliability_interactions)
}



