test_that("Parse graph from a non-dataframe", {
  INPUTS <- list("a", list(a="a"), matrix(c(1,2,3,4)))
  for(edges in INPUTS){
    msg <- paste0("invalid 'type'")
    expect_error(parse_graph_from_edgelist(edges), msg)
  }
})

test_that("Parse graph with invalid column names", {
  data <- mtcars
  expect_error(parse_graph_from_edgelist(data), "invalid columns")
})

test_that("Parse graph without Ego and Alter columns", {
  data <- mtcars
  expect_error(
    parse_graph_from_edgelist(data, ego="Ego", alter="Alter", reporter="mpg", layer="cyl"),
    "`edges` do not have columns")
})
