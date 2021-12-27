import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx

import scipy.cluster

from plotnine import *


def plot_adjacency_matrix(
    A,
    node_order=None,
    return_node_order=False,
    gradient_low="#fcfcfc",
    gradient_mid="#86cefa",
    gradient_high="#003396",
    hide_legend=False,
):
    """
    Plot adjacency matrix of layer l
    """

    """
    A heuristic to order nodes in the plot
    """
    if node_order is None:
        h_clust_mat = sp.cluster.hierarchy.complete(1 - A)
        h_clust_order = sp.cluster.hierarchy.fcluster(Z=h_clust_mat, t=0.1, criterion="distance").argsort()
        node_order = list(map(str, np.arange(0, A.shape[0])[h_clust_order]))

    g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

    plot_df = nx.edges(g).data("weight")
    plot_df = pd.DataFrame(
        plot_df,
        columns=["node_from", "node_to", "weight"],
    )
    plot_df["node_from"] = plot_df["node_from"].astype(str)
    plot_df["node_to"] = plot_df["node_to"].astype(str)

    # Ref:https://matthewlincoln.net/2014/12/20/adjacency-matrix-plots-with-r-and-ggplot2.html
    g = (
        ggplot(plot_df, aes(x="node_from", y="node_to", fill="weight"))
        + geom_raster()
        + theme_bw()
        + scale_fill_gradient2(low=gradient_low, mid=gradient_mid, high=gradient_high)
        +
        # Because we need the x and y axis to display every node,
        # not just the nodes that have connections to each other,
        # make sure that ggplot does not drop unused factor levels
        scale_x_discrete(drop=False, breaks=node_order)
        + scale_y_discrete(drop=False, breaks=node_order)
        + theme(
            # Rotate the x-axis lables so they are legible
            axis_text_x=element_text(angle=270, hjust=0, size=5),
            axis_text_y=element_text(size=5),
            # Force the plot into a square aspect ratio
            aspect_ratio=1,
            # Figure Size
            figure_size=(7, 7),
        )
    )

    if hide_legend:
        # for other options: https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.themeable.legend_position.html
        g = g + theme(legend_position="none")

    if return_node_order:
        return g, node_order
    else:
        return g
