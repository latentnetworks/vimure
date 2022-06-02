"""Read and parse data"""
from asyncio.log import logger

import numpy as np
import pandas as pd
import sktensor as skt
import networkx as nx
import warnings

from abc import ABCMeta
from scipy import sparse

from .log import setup_logging
from .utils import sparse_max, sptensor_from_list

"""
CONSTANTS
"""

DEFAULT_SEED = 10

"""
SETUP
"""
module_logger = setup_logging("vm.io")


class BaseNetwork(metaclass=ABCMeta):
    """
    A base abstract class for generation and management of networks, in an adequate format for this project.
    Suitable for representing any type of network, synthetic or real.
    """

    def __init__(
        self, N: int, M: int, L: int, K: int, seed: int = DEFAULT_SEED, **kwargs,
    ):
        """
        Parameters
        ------------
        N: int
            Number of nodes.
        M: int
            Number of reporters.
        L: int
            Number of layers.
        K: int
            Maximum edge weight in the adjacency matrix. 
            When `K=2`, the adjacency matrix will contain some `Y_{ij}=0` and `Y_{ij}=1`.
        seed: int
            Pseudo random generator seed to use.

        """

        self.N = N
        self.M = M
        self.L = L
        self.K = K

        self.seed = seed
        self.prng = np.random.RandomState(self.seed)

    def get_layer(self, layer: int, return_matrix: bool = True):
        """
        Return adjacency matrix or nx.Digraph object relative to the requested layer of the observed graph (X)
        """
        if return_matrix:
            ret_obj = self.getX()
        else:
            ret_obj = self.getGraphs()

        return ret_obj[layer]

    def setX(self, X):
        dimX = (self.L, self.N, self.N, self.N)
        error_msg = "X has to be a tensor of %s dimensions!" % str(dimX)
        if not (isinstance(X, np.ndarray) or isinstance(X, skt.dtensor) or isinstance(X, skt.sptensor)):
            raise ValueError(error_msg)
        elif X.shape != dimX:
            raise ValueError(error_msg)

        self.X = X

    def getX(self):
        return getattr(self, "X")
    
    def __repr__(self):
        return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, seed={self.seed})"
    
    def __str__(self):
        return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, seed={self.seed})"

    
def parse_graph_from_networkx(G):
    df = nx.to_pandas_edgelist(G)

    raise NotImplementedError


def parse_graph_from_csv(
    filename: str,
    is_weighted: bool = False,
    is_undirected: bool = True,
    reporter="reporter",
    layer="layer",
    ego="Ego",
    alter="Alter",
    weight="weight",
    K=None,
    **kwargs,
):
    """
    Parameters
    -----------

    filename: str
        path to CSV file
    """

    df = pd.read_csv(filename)
    return parse_graph_from_edgelist(
        df,
        is_weighted=is_weighted,
        is_undirected=is_undirected,
        reporter=reporter,
        layer=layer,
        ego=ego,
        alter=alter,
        weight=weight,
        **kwargs,
    )


def parse_graph_from_edgelist(
    df: pd.DataFrame,
    nodes: list = None,
    reporters: list = None,
    is_weighted: bool = False,
    is_undirected: bool = False,
    reporter="reporter",
    layer="layer",
    ego="Ego",
    alter="Alter",
    weight="weight",
    K=None,
    R=None,
    **kwargs,
):
    """
    Parameters
    -----------

    df: pd.DataFrame
        DataFrame representing the edgelist
    nodes: list
        list of all nodes
    reporters: list
        list of the nodes who took the survey
    is_weighted: bool
        True if we should add weights to adjacency matrices
    reporter: str
        reporter column
    layer: str
        layer column
    ego: str
        ego column
    alter: str
        alter column
    weight: str
        weight column
    K: int
        maximum value on the adjacency matrix
    R: list of list of sparse COO array NxN, tot dimension is MxLxNxN (same dimension of the data)
        If this is None, we assume reporters only reports their own ties (of any type)
    **kwargs:
        any other parameters to be sent to RealNetwork.__init__
    """

    if not isinstance(df, pd.DataFrame):
        msg = "Invalid 'type' ({}) of argument 'df'".format(type(df))
        raise ValueError(msg)

    expected_columns = [ego, alter, reporter, layer, weight]
    real_columns = df.columns.values
    diff_columns = list(set(expected_columns) - set(real_columns))

    # Dataframe has no expected column
    if len(diff_columns) == len(expected_columns):
        msg = "Invalid columns in 'df'. Hint: Use params"
        msg += " ego,alter,... for mapping column names."
        raise ValueError(msg)
        
    if ego in diff_columns or alter in diff_columns:
        msg = "Columns '{}' or '{}' were not found in 'df'. Hint: Use params".format(ego, alter)
        msg += " ego,alter,... for mapping column names."
        raise ValueError(msg)
        
    if reporter in diff_columns:
        msg = "'{}' column not found in 'df'. Using '{}' columns as reporter.".format(reporter, ego)
        warnings.warn(msg, UserWarning)
        df.loc[:, reporter] = df[ego]
        
    if layer in diff_columns:
        df.loc[:, layer] = "1"
        
    if weight in diff_columns:
        df.loc[:, weight] = 1

    # Put nodes and reporters in alphabetical order
    layers = sorted(df[layer].unique())
    
    if nodes is None:
        msg = "The set of nodes was not informed, "
        msg += "using {} and {} columns to infer nodes.".format(ego, alter)
        warnings.warn(msg, UserWarning)
        nodes = set(sum(df[[ego, alter]].values.tolist(), []))

    if np.logical_or(~df[ego].isin(nodes), ~df[alter].isin(nodes)).any():
        msg = "Some nodes in the edgelist are not listed in the `nodes` variable."
        raise ValueError(msg)

    L = len(layers)
    N = len(nodes)
    if reporters is None:
        msg = "The set of reporters was not informed, "
        msg += "assuming set(reporters) = set(nodes) and N = M."
        warnings.warn(msg, UserWarning)

        reporters = nodes

    if not set(reporters).issubset(nodes):
        raise ValueError("Set of reporters is not a subset of nodes!")
    M = len(reporters)

    # Remove duplicates
    df = df[expected_columns].drop_duplicates()

    """
    Configure mappers
    """
    # map str to id
    nodeName2Id = {}
    nodeId2Name = {}
    for i, l in enumerate(nodes):
        nodeName2Id[l] = i
        nodeId2Name[i] = l

    layerName2Id = {}
    layerId2Name = {}
    for i, l in enumerate(layers):
        layerName2Id[l] = i
        layerId2Name[i] = l

    if R is None:
        msg = "Reporters Mask was not informed (parameter R). "
        msg += "Parser will build it from reporter column, "
        msg += "assuming a reporter can only report their own ties."

        warnings.warn(msg, UserWarning)

        """
        Infer R
        """

        # While in theory R would be of dimension (M x L), we chose it to be (N x L)
        #  since R and X are later converted to sparse tensors
        #  and then we don't need to create a reporterName2Id mapping.
        #  We can use the same mapping of nodeName2Id
        R = [[[] for _ in range(L)] for _ in range(N)]

        for layerName, layerIdx in layerName2Id.items():
            for repName in reporters:
                rep = nodeName2Id[repName]
                # This reporter can report on any ties involving themselves
                row = rep * np.ones(N - 1)
                col = np.array(list(set(np.arange(N)) - set([rep])))
                data = np.ones(N - 1)

                R[rep][layerIdx] = sparse.coo_matrix((data, (row, col)), shape=(N, N))
                R[rep][layerIdx] = sparse_max(R[rep][layerIdx], R[rep][layerIdx].T,)
                R[rep][layerIdx] = R[rep][layerIdx].tocoo()

        # TODO: If some reporters do not appear in the edgelit, we should also warn the user
    elif R.shape != (L, N, N, M):
        msg = "Dimensions of reporter mask (R) do not match L x N x N x M"
        module_logger.error(msg)
        raise ValueError(msg)

    """
    Set X
    """
    g = df.groupby(by=[reporter, layer])
    X = [[[] for _ in range(L)] for _ in range(N)]  # M x L list
    for idx, n in g:
        row = n[ego].map(nodeName2Id).values
        col = n[alter].map(nodeName2Id).values

        if is_weighted:
            data = n[weight].values
        else:
            data = (n[weight].values > 0).astype("int")

        data_nnz = data > 0  # Keep track of nonzero entries
        rel_data = data[data_nnz]  # Relevant data
        rep = nodeName2Id[idx[0]]  # Current reporter
        layer = layerName2Id[idx[1]]  # Current layer

        X[rep][layer] = sparse.coo_matrix((rel_data, (row[data_nnz], col[data_nnz])), (N, N))
        if is_undirected:
            X[rep][layer] = sparse_max(X[rep][layer], X[rep][layer].T,)
            X[rep][layer] = X[rep][layer].tocoo()

    # Convert to sptensor object
    X = sptensor_from_list(X)

    if isinstance(R, list):
        R = sptensor_from_list(R)

    """
    Set K
    """
    if K is None:
        K = np.max(X.vals) + 1
        msg = f"Parameter K was None. Defaulting to: {K}"
        warnings.warn(msg, UserWarning)
    else:
        K = np.max(X) + 1

    # TODO: For future users, we might want to keep track of nodeName2Id too (nodeId2Name)
    network = RealNetwork(X=X, R=R, L=L, N=N, M=M, K=K, nodeNames=nodeId2Name, **kwargs)
    return network


class RealNetwork(BaseNetwork):
    def __init__(self, X, R=None, **kwargs):
        """
        Parameters
        ------------

        X: tensor of dimensions L x N x N x M
            Represents the observed network as reported by each reporter M
        R: list of list of sparse COO array NxN, tot dimension is L x N x N x M (same dimension of the data)
            If this is None, we assume reporters only reports their own ties (of any type)
        """

        super().__init__(**kwargs)

        self.setX(X)
        self.R = R

        if "nodeNames" in kwargs:
            self.nodeNames = pd.DataFrame(kwargs["nodeNames"].items(), columns = ["id", "name"])

        def __repr__(self):
            return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, number_ties={self.X.vals.sum()})"

        def __str__(self):
            return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, number_ties={self.X.vals.sum()})"
