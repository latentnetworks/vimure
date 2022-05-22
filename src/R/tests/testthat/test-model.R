skip_if_no_vimure <- function() {
  have_vimure <- reticulate::py_module_available("vimure")
  if (!have_vimure)
    skip("Vimure not available for testing")
}

check_parameters <- function(model, synth_net){
  expect_false(is.null(model$rho))
  expect_equal(dim(model$rho), c(synth_net$L, synth_net$N, synth_net$N, synth_net$K))
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

  synth_net <- gm_CReciprocity(N=100, M=100, L=1, C=2,
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

  Y_true <- as.vector(extract_Y(synth_net))
  Y_rec <- as.integer(model$rho_f[,,,2] >= 0.5)
  tab <- table(Y_true, Y_rec)
  f1_score <- (tab[2,2])/(tab[2,2] + 0.5*(tab[1,2] + tab[2,1]))
  expect_lt(abs(f1_expected-f1_score), 1e-2)
}

test_that("Check vimure model with standard_sbm", {
  skip_if_no_vimure()

  synth_net <- gm_StandardSBM(N=20, M=20, L=1, K=3, C=2, avg_degree=2, sparsify=F)
  X <- build_X(synth_net, flag_self_reporter=T)
  model <- vimure(synth_net$X, R=synth_net$R, K=synth_net$K, verbose=F)
  check_parameters(model, synth_net)
})

test_that("Check vimure model in extreme scenarios of under reporting", {
  check_extreme_scenarios("under", 0.97)
})

test_that("Check vimure model in extreme scenarios of over reporting", {
  check_extreme_scenarios("over", 0.92)
})

test_that("Check vimure model for invalid inputs", {
  INPUTS <- list("a", list(a="a"), mtcars, 1:10)
  for(x in INPUTS)
  expect_error(vimure(x), "invalid 'type'")
})
