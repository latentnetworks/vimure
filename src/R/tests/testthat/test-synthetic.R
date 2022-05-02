DEFAULT_C <- vimureP$synthetic$DEFAULT_C
DEFAULT_N <- vimureP$synthetic$DEFAULT_N
DEFAULT_M <- vimureP$synthetic$DEFAULT_M
DEFAULT_K <- vimureP$synthetic$DEFAULT_K
DEFAULT_L <- vimureP$synthetic$DEFAULT_L
DEFAULT_AVG_DEGREE <- vimureP$synthetic$DEFAULT_AVG_DEGREE
DEFAULT_ETA <- vimureP$synthetic$DEFAULT_ETA
DEFAULT_C <- vimureP$synthetic$DEFAULT_C
DEFAULT_EXP_IN <- vimureP$synthetic$DEFAULT_EXP_IN
DEFAULT_EXP_OUT <- vimureP$synthetic$DEFAULT_EXP_OUT


check_default_values <- function(synth_net){
  expect_equal(synth_net$C, DEFAULT_C)
  expect_equal(synth_net$N, DEFAULT_N)
  expect_equal(synth_net$M, DEFAULT_M)
  expect_equal(synth_net$K, DEFAULT_K)
  expect_equal(synth_net$L, DEFAULT_L)
  expect_equal(synth_net$avg_degree, DEFAULT_AVG_DEGREE)
  expect_equal(unlist(synth_net$Y$shape), c(DEFAULT_L, DEFAULT_N, DEFAULT_N))

  Y <- extract_Y(synth_net)
  expect_gt(sum(Y), 0)
  expect_lt(max(Y), synth_net$K)
}

check_affinity_matrix <- function(synth_net){
  L <- synth_net$L

  # Check affinity matrix dimensions
  for(i in 1:L){
    current_w <- synth_net$w[i, ,]
    expect_equal(dim(current_w), c(DEFAULT_C, DEFAULT_C))

    diag_w <- diag(current_w)
    non_diag_w <- current_w[as.logical(lower.tri(current_w) + upper.tri(current_w))]
    if(synth_net$structure[i] == "assortative"){
      expect_true(all(sapply(diag_w, function(x) all(x > non_diag_w))))
    } else{
      expect_true(all(sapply(diag_w, function(x) all(x < non_diag_w))))
    }

  }
}

check_Y_change_for_exp_in_ou_change <- function(synth_net, sparse_synth_net){
  another_synth_net <- DegreeCorrectedSBM(
    N=DEFAULT_N,
    M=DEFAULT_M,
    L=DEFAULT_L,
    K=DEFAULT_K,
    C=DEFAULT_C,
    avg_degree=DEFAULT_AVG_DEGREE,
    exp_in=4,
    exp_out=6,
    sparsify=FALSE
  )

  another_sparse_synth_net <- DegreeCorrectedSBM(
    N=DEFAULT_N,
    M=DEFAULT_M,
    L=DEFAULT_L,
    K=DEFAULT_K,
    C=DEFAULT_C,
    avg_degree=DEFAULT_AVG_DEGREE,
    exp_in=4,
    exp_out=6,
    sparsify=TRUE
  )

  Y <- extract_Y(synth_net)
  another_Y <- extract_Y(another_synth_net)
  sparse_Y <- extract_Y(sparse_synth_net)
  another_sparse_Y <- extract_Y(another_sparse_synth_net)

  expect_false(all(Y == another_Y))
  expect_false(all(sparse_Y == another_sparse_Y))
  expect_gt(sum(Y), sum(sparse_Y))
  expect_gt(sum(Y), sum(another_Y))
  expect_gt(sum(another_Y), sum(another_sparse_Y))
}

test_that("GMReciprocity - Check default values and adjacency matrix structure", {
  synth_net <- GMReciprocity(
    N = DEFAULT_N,
    M = DEFAULT_M,
    K = DEFAULT_K,
    L = DEFAULT_L,
    C = DEFAULT_C,
    eta = DEFAULT_ETA,
    avg_degree = DEFAULT_AVG_DEGREE
  )

  check_default_values(synth_net)
  expect_equal(synth_net$eta, DEFAULT_ETA)
  expect_false(is.null(synth_net$ExpM))
})

test_that("StandardSBM - Check default values, adjacency matrix and affinity matrix structure", {
  synth_net <- StandardSBM(
    N=DEFAULT_N,
    M=DEFAULT_M,
    L=DEFAULT_L,
    K=DEFAULT_K,
    C=DEFAULT_C,
    avg_degree=DEFAULT_AVG_DEGREE,
    sparsify=FALSE,
  )

  check_default_values(synth_net)
  check_affinity_matrix(synth_net)
})

test_that("DegreeCorrectedSBM -", {
  synth_net <- DegreeCorrectedSBM(
    N=DEFAULT_N,
    M=DEFAULT_M,
    L=DEFAULT_L,
    K=DEFAULT_K,
    avg_degree=DEFAULT_AVG_DEGREE,
    C=DEFAULT_C,
    sparsify=FALSE
  )

  sparse_synth_net <- DegreeCorrectedSBM(
    N=DEFAULT_N,
    M=DEFAULT_M,
    L=DEFAULT_L,
    K=DEFAULT_K,
    C=DEFAULT_C,
    avg_degree=DEFAULT_AVG_DEGREE,
    sparsify=TRUE
  )

  expect_equal(synth_net$exp_in, DEFAULT_EXP_IN)
  expect_equal(synth_net$exp_out, DEFAULT_EXP_OUT)

  expect_equal(length(synth_net$d_in), synth_net$N)
  expect_equal(length(synth_net$d_out), synth_net$N)

  check_Y_change_for_exp_in_ou_change(synth_net, sparse_synth_net)
})