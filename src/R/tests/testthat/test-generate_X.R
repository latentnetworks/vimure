skip_if_no_vimure <- function() {
  have_vimure <- reticulate::py_module_available("vimure")
  if (!have_vimure)
    skip("Vimure not available for testing")
}

check_X_matrix <- function(synth_net){
  expect_false(is.null(synth_net$X))
  expect_equal(unlist(synth_net$X$shape), c(synth_net$L, synth_net$N, synth_net$N, synth_net$M))
  expect_gt(sum(synth_net$X$vals), 0)
}

check_theta <- function(synth_net){
  expect_false(is.null(synth_net$theta))
  expect_equal(dim(synth_net$theta), c(synth_net$L, synth_net$M))
  expect_true(all(synth_net$theta > 0))
}

check_lambda_k <- function(synth_net){
  expect_false(is.null(synth_net$lambda_k))
  expect_equal(dim(synth_net$lambda_k), unlist(synth_net$Y$shape))
  expect_true(all(synth_net$lambda_k > 0))
  expect_lte(max(synth_net$lambda_k), synth_net$K)
}

check_R <- function(synth_net){
  expect_false(is.null(synth_net$R))
  expect_equal(dim(synth_net$R), c(synth_net$L, synth_net$N, synth_net$N, synth_net$M))
  expect_gt(sum(synth_net$R), 0)
}

test_that("multiplication works", {
  skip_if_no_vimure()

  synth_net <- gm_StandardSBM(
    N=20,
    M=20,
    L=1,
    K=4,
    C=2,
    avg_degree=4,
    sparsify=F,
    seed=10,
  )

  X <- build_X(synth_net, flag_self_reporter=F, cutoff_X=F, mutuality=0.5, seed=20L)

  check_X_matrix(synth_net)
  check_theta(synth_net)
  check_lambda_k(synth_net)
  check_R(synth_net)
})
