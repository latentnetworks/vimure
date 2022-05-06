#' @export
py_to_r.sktensor.sptensor.sptensor <- function(x){
  vals <- as.vector(x$vals)
  dims <- unlist(x$shape)
  subsa <- do.call(rbind, x$subs)
  subs <- lapply(seq(dim(subsa)[2]), function(i) subsa[, i]+1)
  tensorr::sptensor(subs, vals, dims)
}

