import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx

from .log import setup_logging
from .model import VimureModel
from .synthetic import BaseSyntheticNetwork
from .utils import apply_rho_threshold

from plotnine import *


class Diagnostics:
    def __init__(
        self, model: VimureModel = None, net: BaseSyntheticNetwork = None, verbose: bool = None,
    ):
        self.model = model

        if hasattr(net, "Y"):
            # If it is a ground truth network, keep relevant info
            self.Y = net.Y
            self.eta = net.eta

        self.L = net.L
        self.M = net.M
        self.N = net.N

        if self.model.mutuality:
            self.model_str = "ViMuRe(T)"
        else:
            self.model_str = "ViMuRe(F)"

        self.model_str += "\n\n  Priors:\n"

        self.model_str += "   - eta:    shp=%.2f rte=%.2f\n" % (
            self.model.alpha_mutuality,
            self.model.beta_mutuality,
        )
        try:
            self.model_str += "   - theta:  shp=%.2f rte=%.2f\n" % (
                self.model.alpha_theta,
                self.model.beta_theta,
            )
        except Exception:
            self.model_str += "   - theta:  shp=%.2f rte=%.2f\n" % (
                self.model.alpha_theta.mean(),
                self.model.beta_theta.mean(),
            )
        self.model_str += "   - lambda: shp=%s rte=%s\n" % (
            str(self.model.alpha_lambda),
            str(self.model.beta_lambda),
        )

        self.model_str += "   - rho:    a %s tensor " % str(self.model.pr_rho.shape)
        self.model_str += "(to inspect it, run <diag_obj>.model.pr_rho)\n"

        self.model_str += "\n  Posteriors:\n"
        self.model_str += "   - G_exp_lambda_f: %s\n" % str(self.model.G_exp_lambda_f)
        self.model_str += "   - G_exp_nu_f: %.2f\n" % self.model.G_exp_nu_f
        self.model_str += "   - G_exp_theta_f: a %s" % str(self.model.G_exp_theta_f.shape)
        self.model_str += " tensor (to inspect it, run <diag_obj>.model.G_exp_theta_f)\n"

        self.model_str += "   - rho_f: a %s" % str(self.model.rho_f.shape)
        self.model_str += " tensor (to inspect it, run <diag_obj>.model.rho_f)\n"

        # TODO: Add reached_convergence variable to model_str in Diagnostics

        if verbose is None:
            self.verbose = self.model.verbose

        self.logger = setup_logging("vm.diagnostics.Diagnostics", verbose)

    def __str__(self):
        msg = "---------------\n- DIAGNOSTICS -\n---------------\n\n"
        msg += f"Model: {self.model_str}\n"
        msg += "Optimisation:\n\n   Elbo: %.12f\n" % self.model.maxL
        return msg

    def __repr__(self):
        msg = "---------------\n- DIAGNOSTICS -\n---------------\n\n"
        msg += f"Model: {self.model_str}\n"
        msg += "Optimisation:\n\n   Elbo: %.12f\n" % self.model.maxL
        return msg

    def plot_elbo_values(self):
        """
        Plot the multiple realisations of ELBO values.
        """

        plot_df = self.model.trace
        plot_df["realisation"] = plot_df["realisation"].astype(str)

        g = (
            ggplot(plot_df, aes(x="iter", y="elbo", group_by="seed", color="realisation"))
            + geom_point()
            + geom_line()
            + scale_x_continuous(name="Iter")
            + scale_y_continuous(name="ELBO")
            + scale_color_discrete(name="Realisation")
            + theme_bw()
            + theme(figure_size=(10, 4))
        )

        return g

    def plot_theta(
        self,
        theta_GT: np.ndarray = None,
        node_order: list = None,
        return_node_order: bool = False,
        return_plot_df: bool = False,
        remove_nodes: list = None,
    ):
        """
        Scatterplot showing theta distribution (reporters' reciprocity),
            compared to ground truth theta (theta_GT) when that is available
        """

        if node_order is None:
            if theta_GT is None:
                node_order = list(map(str, reversed(np.argsort(self.model.G_exp_theta_f[0]))))
            else:
                node_order = list(map(str, reversed(np.argsort(theta_GT[0]))))

        plot_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "layer": layer,
                        "variable": r"$\theta_{est}$",
                        "node": range(self.N),
                        "value": self.model.G_exp_theta_f[layer],
                    }
                )
                for layer in range(self.L)
            ]
        )

        # Add lambda * theta
        aux_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "layer": layer,
                        "variable": r"$\theta_{est} \times \lambda_{k=1}$",
                        "node": range(self.N),
                        "value": self.model.G_exp_theta_f[layer] * self.model.G_exp_lambda_f[layer][1],
                    }
                )
                for layer in range(self.L)
            ]
        )
        plot_df = pd.concat([plot_df, aux_df])

        if theta_GT is not None:
            aux_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "layer": layer,
                            "variable": r"$\theta_{GT}$",
                            "node": range(self.N),
                            "value": theta_GT[layer],
                        }
                    )
                    for layer in range(self.L)
                ]
            )
            plot_df = pd.concat([plot_df, aux_df])
            title = r"Estimated vs ground-truth reliability | Experiment:" + r"$\eta_{est} = $ %.2f"
            title = title % self.model.G_exp_nu_f
        else:
            title = r"Reporters' estimated reliability | $\eta_{est} = $ %.2f"
            title = title % (self.model.G_exp_nu_f)

        plot_df["layer"] = plot_df["layer"].astype(str)
        plot_df["node"] = plot_df["node"].astype(str).astype("category").cat.reorder_categories(node_order)

        if remove_nodes is not None:
            plot_df = plot_df[~plot_df["node"].isin(remove_nodes)].copy()

        g = (
            ggplot(plot_df, aes(x="node", y="value", fill="variable"))
            + geom_point(alpha=0.6, size=2)
            + scale_x_discrete(name="Nodes sorted from over- to under- reporting")
            + ylab(r"Reliability ($\theta$) values")
            + scale_fill_discrete(name=None)
            + theme_bw()
            + theme(
                figure_size=(11, 5),
                axis_text_x=element_blank(),
                axis_title_y=element_text(margin={"r": 15}),
                axis_title_x=element_text(margin={"t": 10}),
            )
            + facet_wrap("~ variable", labeller="label_both", scales="free_y", ncol=1)
            + ggtitle(title)
        )

        if return_node_order:
            if return_plot_df:
                return g, node_order, plot_df
            else:
                return g, node_order
        else:
            if return_plot_df:
                return g, plot_df
            else:
                return g

    def reliability_interactions(self, rep_id):
        raise NotImplementedError

    def plot_baseline(self):
        # TODO: Check if net is a synthetic network, raise error otherwise
        # TODO: Calculate union
        # TODO: Calculate average
        raise NotImplementedError

    def plot_adjacency_matrix(
        self,
        layer,
        node_order=None,
        return_node_order=False,
        gradient_low="#fcfcfc",
        gradient_mid="#86cefa",
        gradient_high="#003396",
        hide_legend=False,
        include_ground_truth=True,
        additional_Ys={},
    ):
        """
        Plot adjacency matrix of layer l, compared to ground truth
        """

        plotnames = ["Y_rec"]

        def calculate_node_order(A):
            h_clust_mat = sp.cluster.hierarchy.complete(1 - A)
            h_clust_order = sp.cluster.hierarchy.fcluster(Z=h_clust_mat, t=0.1, criterion="distance")
            h_clust_order = h_clust_order.argsort()
            node_order = list(map(str, np.arange(0, A.shape[0])[h_clust_order]))
            return node_order

        def get_network_df(networkx_obj, name):
            cols = ["node_from", "node_to", "weight"]

            # Convert networkx object to a pandas DataFrame
            plot_df = nx.edges(networkx_obj).data("weight")
            plot_df = pd.DataFrame(plot_df, columns=cols)
            plot_df["matrix_source"] = name

            return plot_df

        # TODO: In the future, instead of self.model.rho_sub, read from a function (ex: self.model.get_Yrec())
        A = apply_rho_threshold(self.model)
        g = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        plot_df = get_network_df(g, name="Y_rec")

        # Double check if self.net was a synthetic network and actually contains a ground truth Y
        if include_ground_truth:
            if hasattr(self, "Y"):

                g = self.layer_graphs[layer]
                A = self.Y[layer]  # Overwrite A so node_order makes more sense
                aux_df = get_network_df(g, name="Y_true")
                plotnames = ["Y_true", "Y_rec"]

                # Convert networkx object to a pandas DataFrame and concat
                plot_df = pd.concat([plot_df, aux_df])
            else:
                msg = "include_ground_truth=True but network does not contain Y"
                self.logger.warn(msg)

        if len(additional_Ys) > 0:
            for net_name, add_Y in additional_Ys.items():
                g = nx.from_numpy_matrix(add_Y, create_using=nx.DiGraph)
                aux_df = get_network_df(g, name=net_name)

                plotnames.extend([net_name])
                # Convert networkx object to a pandas DataFrame and concat
                plot_df = pd.concat([plot_df, aux_df])

        """
        A heuristic to order nodes in the plot
        """
        if node_order is None:
            node_order = calculate_node_order(A)

        plot_df["matrix_source"] = (
            plot_df["matrix_source"].astype("category").cat.reorder_categories(plotnames)
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
                figure_size=(12, 6),
            )
            + facet_wrap("~ matrix_source", labeller="label_value", ncol=2)
        )

        if hide_legend:
            # for other options: https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.themeable.legend_position.html
            g = g + theme(legend_position="none")

        if return_node_order:
            return g, node_order
        else:
            return g

    def plot_posteriors(self, params: dict = {}):
        # TODO: Borrow interface from posterior package in R
        raise NotImplementedError

    def plot_pairs(self, params: dict = {}):
        # TODO: Borrow interface from posterior package in R
        raise NotImplementedError
