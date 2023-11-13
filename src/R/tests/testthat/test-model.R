require(testthat)
require(reticulate)

skip_if_no_vimure <- function() {
  have_vimure <- reticulate::py_module_available("vimure")
  if (!have_vimure)
    skip("Vimure not available for testing")
}

check_parameters <- function(model, synth_net) {
  expect_false(is.null(model$rho))
  expect_equal(dim(model$rho),
               c(synth_net$L, synth_net$N, synth_net$N, synth_net$K))
  expect_gt(sum(model$rho), 0)

  expect_false(is.null(model$gamma_shp))
  expect_equal(dim(model$gamma_shp), c(synth_net$L, synth_net$M))
  expect_gt(sum(model$gamma_shp), 0)

  expect_false(is.null(model$gamma_rte))
  expect_equal(dim(model$gamma_rte), c(synth_net$L, synth_net$M))
  expect_gt(sum(model$gamma_rte), 0)

  expect_false(is.null(model$phi_shp))
  expect_equal(dim(model$phi_shp), c(synth_net$L, synth_net$K))
  expect_gt(sum(model$phi_shp), 0)

  expect_false(is.null(model$phi_rte))
  expect_equal(dim(model$phi_rte), c(synth_net$L, synth_net$K))
  expect_gt(sum(model$phi_rte), 0)

  expect_false(is.null(model$nu_shp))
  expect_gt(sum(model$nu_shp), 0)

  expect_false(is.null(model$nu_rte))
  expect_gt(sum(model$nu_rte), 0)
}

check_extreme_scenarios <- function(exaggeration_type, f1_expected){
  eta <- 0.2
  theta_ratio <- 0.1
  seed <- 25
  K <- 2

  synth_net <- gm_crep(N=100, M=100, L=1, C=2,
                               K=K, avg_degree=5, sparsify=T, seed=seed, eta=eta
  )

  custom_theta <- build_custom_theta(
    synth_net,
    theta_ratio=theta_ratio,
    exaggeration_type=exaggeration_type,
    seed = seed
  )

  LAMBDA_0 <- 0.01
  LAMBDA_DIFF <- 0.99
  X <- build_X(synth_net, mutuality=eta, theta=custom_theta, cutoff_X=F, lambda_diff=LAMBDA_DIFF,
               flag_self_reporter=T, verbose=T
  )

  lambda_k_GT <- matrix(c(LAMBDA_0, LAMBDA_0 + LAMBDA_DIFF), ncol=K)
  beta_lambda <- matrix(10000 * rep(1, length(lambda_k_GT)), ncol=K)
  alpha_lambda <- lambda_k_GT * beta_lambda

  model <- vimure(synth_net$X, R=synth_net$R, K=K, seed=seed, mutuality=T,
                  alpha_lambda=alpha_lambda, beta_lambda=beta_lambda,
                  num_realisations=2, max_iter=21, verbose=F
  )

  Y_true <- as.vector(synth_net$Y$toarray())
  Y_rec <- as.integer(model$rho_f[,,,2] >= 0.5)
  tab <- table(Y_true, Y_rec)
  f1_score <- (tab[2,2])/(tab[2,2] + 0.5*(tab[1,2] + tab[2,1]))
  expect_lt(abs(f1_expected-f1_score), 1e-2)
}

check_output <- function(Y, model){
  expect_gt(sum(Y), 0)
  expect_equal(dim(Y), c(model$L, model$N, model$N))
}

test_that("Check reporters' mask", {
  skip_if_no_vimure()

  N <- 20
  nodes <- 1:N
  synth_net <- gm_standard_sbm(N = N, M = 20, L = 1, K = 3, C = 2,
                              avg_degree = 2, sparsify = FALSE)

  # specify some sort of mask
  # for example, only egos of synth_net are reporters
  adj_matrix <- synth_net$Y$toarray()[1, , ]
  reporters <- which(rowSums(adj_matrix) > 0)

  reporter_mask <- array(0,
                        dim = c(1, N, N, N),
                        dimnames = list("layer", nodes, nodes, nodes))

  reporter_mask[, , , reporters] <- "clearly_wrong"

  expect_error(vimure(adj_matrix, R = reporter_mask), 
               "R must be either entirely made of TRUE/FALSE or of 0s/1s.")

})

test_that("Check vimure model with standard_sbm", {
  skip_if_no_vimure()

  synth_net <- gm_standard_sbm(N=20, M=20, L=1, K=3, C=2, avg_degree=2, sparsify=F)
  X <- build_X(synth_net, flag_self_reporter=T)
  model <- vimure(synth_net$X, R=synth_net$R, K=synth_net$K, verbose=F)
  check_parameters(model, synth_net)
})

# Motivated by https://github.com/latentnetworks/vimure/issues/92
test_that("Check custom reporters' mask are enforced", {
  skip_if_no_vimure()

  L <- 1
  N <- 20
  nodes <- 1:N
  synth_net <- gm_standard_sbm(N = N, M = 20, L = L, K = 3, C = 2,
                               avg_degree = 2, sparsify = FALSE)

  X <- build_X(synth_net, flag_self_reporter = TRUE)

  # Essentially all 1:M are reporters
  all_reporters <- unique(synth_net$R$subs[[4]])

  reporter_mask <- array(0, dim = c(1, N, N, N),
                         dimnames = list("layer", 1:N, 1:N, 1:N))

  # I made it so that only egos are reporters
  for (l in 1:L) {
    for (i in 1:N) {
      for (j in 1:N) {
        for (m in seq_along(all_reporters)) {
          if (i == all_reporters[m]) {
            reporter_mask[l, i, j, m] <- 1
          }
        }
      }
    }
  }

  model <- vimure(synth_net$X,
                  R = reporter_mask,
                  K = synth_net$K,
                  verbose = TRUE)
  check_parameters(model, synth_net)

  # The object reporter_mask is already a native R matrix
  # Now I need to convert model$R from python to R

  # I will need numpy
  np <- import("numpy", convert = FALSE)

  # Initialise an empty 4D numpy array
  model_R <- np$zeros(shape = as.integer(c(L, N, N, N)))

  # Get the keys (a 4D list) of the python object model$R
  model_R_keys <- model$R$subs

  #For each key, assign the corresponding value to the numpy array
  for (i in seq_along(model_R_keys[[1]])) {
    model_R[model_R_keys[[1]][i], model_R_keys[[2]][i],
             model_R_keys[[3]][i], model_R_keys[[4]][i]] <- 1
  }

  model_R <- as.array(model_R)
  dimnames(model_R) <- list("layer", 1:N, 1:N, 1:N)

  expect_equal(model_R, reporter_mask)

})

test_that("Check vimure model in extreme scenarios of under reporting", {
  check_extreme_scenarios("under", 0.97)
})

test_that("Check vimure model in extreme scenarios of over reporting", {
  check_extreme_scenarios("over", 0.92)
})

test_that("Check vimure model for invalid inputs", {
  expect_error(vimure("a"), "invalid 'type'")
  expect_error(vimure(list(a = "a")), "invalid 'type'")
  expect_error(vimure(mtcars),
               "ValueError: Required columns not found in data frame")
  expect_error(vimure(1:10), "invalid 'type'")
})


test_that("Check inferred model", {
  synth_net <- gm_standard_sbm(N=20, M=20, sparsify = F, K=2)
  X <- build_X(synth_net, flag_self_reporter = T, cutoff_X = T)

  model <- vimure(synth_net$X, synth_net$R, num_realisations=1, max_iter=500, mutuality = T, verbose=F)
  expect_error(get_inferred_model(model, method = "NonImplemented"), "'method' should be one of")
  expect_error(get_inferred_model(model, method = "fixed_threshold"), 'you must set the threshold to a value')
  expect_error(get_inferred_model(model, method = "fixed_threshold", threshold=2), 'you must set the threshold to a value')

  OPTIONS <- c("rho_max", "rho_mean", "fixed_threshold", "heuristic_threshold")
  for(method in OPTIONS){
    Y_hat <- get_inferred_model(model, method = method, threshold = 0.5)
    check_output(Y_hat, model)
  }

  N <- 10L
  Y_hat <- sample_inferred_model(model, N=N)
  expect_equal(length(Y_hat), N)

  for(y in Y_hat){
    check_output(y, model)
  }
})
