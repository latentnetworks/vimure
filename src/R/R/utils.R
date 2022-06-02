#' Parses an edgelist dataframe to vm.io.RealNetwork class
#'
#' parse_graph_from_edgelist creates a vimure.io.RealNetwork from an edgelist dataframe.
#' Its argument is a dataframe with at least Ego and Alter columns, each row defines one edge.
#' The dataframe can contains three optional columns:
#' 'reporter' for identify who reports each edge, 'layer' for multidimensions networks and 'weight' for
#' a weighted adjancency matrix
#' Use the params `ego`, `alter`, `reporter`, `weight` and `layer` to map the dataframe columns.
#'
#' @family parse data input
#'
#' @param edges data.frame representing the edgelist.
#' @param nodes List of all nodes.
#' @param reporters List of the nodes who took the survey.
#' @param weighted 	Whether to add weights to the adjancency matrix.
#' @param directed Whether to create a directed graph.
#' @param ego Column name for mapping Ego.
#' @param alter Column name for mapping Alter.
#' @param reporter Column name for mapping reporter.
#' @param layer Column name for mapping layer.
#' @param weight Column name for mapping weight.
#' @param ... Additional args of vm.io.RealNetwork class
#'
#' @return vimureP$io$RealNetwork
#' @export
parse_graph_from_edgelist <- function(
  edges, nodes=NULL, reporters=NULL, weighted=F, directed=T, ego="Ego", alter="Alter",
  reporter="reporter", layer="layer", weight="weight",
  ...
){
  if(!("data.frame" %in% class(edges))){
    stop("invalid 'type' (", class(edges)[1], ") of argument")
  }

  col_expected <- c(ego, alter, reporter, layer, weight)
  col_real <- colnames(edges)
  col_diff <- col_expected[!col_expected %in% col_real]

  # Convert factors to characters
  i <- sapply(edges, is.factor)
  edges[i] <- lapply(edges[i], as.character)

  if(length(col_diff)==length(col_expected)){
    stop("invalid columns in `edges`.
         Hint: Use params ego,alter,... for mapping column names.")
  }

  if(sum(c(ego, alter) %in% col_diff) > 0){
    stop("`edges` do not have columns '", ego, "' and '", alter, "'.
         Hint: Use params ego,alter,... for mapping column names.")
  }

  if(reporter %in% col_diff){
    warning("'", reporter, "' column not found in `edges`. Using '", ego, "' column as reporter.")
    edges[reporter] <- edges[ego]
  }

  if(layer %in% col_diff){
    edges[layer] <- "1"
  }

  if(weight %in% col_diff){
    edges[weight] <- 1
  }

  if(length(nodes) == 0){
    # Infer nodes from ego and alter columns
    nodes <- unique(unlist(c(edges[ego],edges[alter],edges[reporter])))
  }

  if(length(reporters) == 0){
    reporters <- unique(unlist(edges[ego]))
  }

  vimureP$io$parse_graph_from_edgelist(
    edges, nodes, reporters, is_undirected=!directed, is_weighted=weighted,
    ego=ego, alter=alter, weight=weight, reporter=reporter, layer=layer, ...
  )
}

#' Parses a csv file to vm.io.RealNetwork class
#'
#' parse_graph_from_csv creates a vimure.io.RealNetwork from a csv file
#' Its argument is a csv file with at least Ego and Alter columns, each row defines one edge.
#' The dataframe can contains three optional columns:
#' 'reporter' for identify who reports each edge, 'layer' for multidimensions networks and 'weight' for
#' a weighted adjancency matrix
#' Use the params `ego`, `alter`, `reporter`, `weight` and `layer` to map the dataframe columns.
#'
#' @family parse data input
#'
#' @param file data.frame representing the edgelist.
#' @param weighted 	Whether to add weights to the adjancency matrix.
#' @param directed Whether to create a directed graph.
#' @param ego Column name for mapping Ego.
#' @param alter Column name for mapping Alter.
#' @param reporter Column name for mapping reporter.
#' @param layer Column name for mapping layer.
#' @param weight Column name for mapping weight.
#' @param ... Additional args of read.csv function
#'
#' @return vimureP$io$RealNetwork
#' @export
parse_graph_from_csv <- function(
  file, weighted=F, directed=T, ego="Ego", alter="Alter", reporter="reporter", layer="layer", weight="weight", ...){
  edges <- utils::read.csv(file, ...)
  parse_graph_from_edgelist(
    edges, weighted=weighted, directed=directed, ego=ego,
    alter=alter, reporter=reporter, layer=layer, weight=weight
  )
}

#' Parses an igraph object to vm.io.RealNetwork class
#'
#' parse_graph_from_igraph creates a vimure.io.RealNetwork from an igraph object.
#' The igraph can contains three optional edge attributes:
#' 'reporter' for identify who reports each edge, 'layer' for multidimensions networks and 'weight' for
#' a weighted adjancency matrix
#' Use the params `reporter`, `weight` and `layer` to map the edge attribute names.
#'
#' @family parse data input
#'
#' @param graph An igraph object.
#' @param directed Whether to create a directed graph.
#' @param weighted 	Whether to add weights to the adjancency matrix.
#' @param reporter Edge attribute name for mapping reporter.
#' @param weight Edge attribute name for mapping weight.
#' @param layer Edge attribute name for mapping layer.
#'
#' @return vimureP$io$RealNetwork
#' @export
parse_graph_from_igraph <- function(
  graph, directed = T, weighted=F, weight="weight", reporter="reporter", layer="layer"){
  if(class(graph) != "igraph"){
    stop("invalid 'type' (", class(graph), ") of argument")
  }

  df_edges <- data.frame(igraph::as_edgelist(graph))
  colnames(df_edges) <- c("Ego", "Alter")

  edges <- igraph::E(graph)
  df_edges['weight'] <- eval(parse(text=paste0("edges$",weight)))
  df_edges['reporter'] <- eval(parse(text=paste0("edges$",reporter)))
  df_edges['layer'] <- eval(parse(text=paste0("edges$",layer)))

  parse_graph_from_edgelist(df_edges, directed=directed, weighted=weighted)
}

#' Reciprocity of networks
#'
#' Calculates the reciprocity of adjancency matrix
#'
#' @param X A matrix
#'
#' @export
overall_reciprocity <- function(X){
  vimureP$utils$calculate_overall_reciprocity(X)
}

py_to_r.sktensor.sptensor.sptensor <- function(x){
  x$toarray()
}

r_to_py.dgCMatrix <- function(x){
  vimureP$utils$sk
}
