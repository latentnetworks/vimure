"""Read and parse data"""
from asyncio.log import logger
from typing import Optional

import torch

import numpy as np
import pandas as pd
import sktensor as skt
import networkx as nx
import warnings

from abc import ABCMeta
from scipy import sparse

from ._log import setup_logging
from .utils import sparse_max, sptensor_from_list

"""
CONSTANTS
"""


"""
SETUP
"""
module_logger = setup_logging("vm.io", False)

### Classes ###

class BaseNetwork(metaclass=ABCMeta):
    """
    A base abstract class for generation and management of networks, in an adequate format for this project.
    Suitable for representing any type of network, synthetic or real.
    """

    def __init__(
        self, N: int, M: int, L: int, K: int, seed: int = 10, **kwargs,
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
            Maximum edge weight in the adjacency matrix. When `K=2`, the adjacency matrix will contain some `Y_{ij}=0` and `Y_{ij}=1`.
        seed: int
            Pseudo random generator seed to use.

        """

        self.N = N
        self.M = M
        self.L = L
        self.K = K

        self.seed = seed
        self.prng = torch.Generator()
        self.prng.manual_seed(self.seed)

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


class RealNetwork(BaseNetwork):
    """
    Class to represent all elements of a real network, and distinguish 
    it from synthetic networks (vimure.synthetic module).

    Note: this class is not meant to be used directly by users.
    """

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

        if "layerNames" in kwargs:
            self.layerNames = kwargs["layerNames"]

        def __repr__(self):
            return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, number_ties={self.X.vals.sum()})"

        def __str__(self):
            return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, number_ties={self.X.vals.sum()})"


### I/O Functions ###

def read_from_edgelist(
    df: pd.DataFrame,
    nodes: list = [],
    reporters: list = [],
    is_weighted: bool = False,
    is_undirected: bool = False,
    reporter: str ="reporter",
    layer: str = "layer",
    ego: str = "ego",
    alter: str = "alter",
    weight: str = "weight",
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

    Returns
    -----------

        A RealNetwork object that is used to represent the observed network

    """

    all_params = {key: value for key, value in locals().items() if key != 'self'}
    df, nodes, reporters = _check_params_consistency(**all_params) # type: ignore

    # Put nodes and reporters in alphabetical order
    layers = sorted(df[layer].unique())

    L = len(layers)
    N = len(nodes)
    M = len(reporters)

    # Remove duplicates
    all_columns = [ego, alter, reporter, layer, weight]
    df = df[all_columns].drop_duplicates()

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
        msg = (
            "Reporters Mask was not informed (parameter R). "
            "Parser will build it from reporter column, "
            "assuming a reporter can only report their own ties."
        )

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
                R[rep][layerIdx] = sparse_max(R[rep][layerIdx], R[rep][layerIdx].T,) # type: ignore
                R[rep][layerIdx] = R[rep][layerIdx].tocoo() # type: ignore

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
            data = (n[weight].values > 0).astype("int") # type: ignore

        # Keep track of nonzero entries 
        data_nnz = data > 0 # type: ignore
        rel_data = data[data_nnz]  # Relevant data
        
        # Current reporter
        # TODO: #71 In the future, we might want to allow reporters to be externals (not in the network)
        rep = nodeName2Id[idx[0]]  
        layer = layerName2Id[idx[1]]  # Current layer

        X[rep][layer] = sparse.coo_matrix((rel_data, # type: ignore
                                           (row[data_nnz], col[data_nnz])), 
                                           (N, N)) 
        if is_undirected:
            X[rep][layer] = sparse_max(X[rep][layer], X[rep][layer].T,) # type: ignore
            X[rep][layer] = X[rep][layer].tocoo() # type: ignore

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

    # TODO: For future users, we might want to keep track of nodeName2Id too (nodeId2Name)
    network = RealNetwork(X=X, R=R, L=L, N=N, M=M, K=K, nodeNames=nodeId2Name, layerNames=layers, **kwargs)
    return network

def read_from_csv(
    filename: str,
    **kwargs,
) -> RealNetwork: # type: ignore
    """
    Reads a csv file and returns a RealNetwork object

    Parameters
    ----------
    filename : str
        Path to the csv file
    **kwargs:
        Any other parameters to be sent to read_from_edgelist

    Returns
    -------

    RealNetwork
        A RealNetwork object
    """  

    df = pd.read_csv(filename)

    net_obj = read_from_edgelist(df, **kwargs)
    return net_obj

def read_from_igraph(G, **kwargs):
    """
    Reads an igraph object and returns a RealNetwork object

    Parameters
    ----------
    G : igraph.Graph
        An igraph object
    **kwargs:
        Any other parameters to be sent to read_from_edgelist
    
    Returns
    -------
    RealNetwork
        A RealNetwork object
    """

    # get the edgelist as a list of tuples
    edgelist = G.get_edgelist()

    # extract the edge atgtributes as a list of dictionaries
    edge_attrs = [{attr: G.es[edge_idx][attr] for attr in G.es.attributes()} 
                  for edge_idx in range(G.ecount())]

    # create a pandas dataframe from the edgelist and edge_attrs
    df = pd.DataFrame(edgelist, columns=['source', 'target'])
    for attr in edge_attrs[0]:
        df[attr] = [edge[attr] for edge in edge_attrs]

    df = df.rename(columns={"source":"ego", "target":"alter"})\
        .assign(ego=lambda x: G.vs[x.ego]["name"], 
                alter=lambda x: G.vs[x.alter]["name"])
    
    return read_from_edgelist(df, **kwargs)

def parse_graph_from_networkx(G, **kwargs):
    df = nx.to_pandas_edgelist(G)
    return read_from_edgelist(df, **kwargs)

### Utility functions ###

def _check_params_consistency(**kwargs):
    """
    Check if the parameters are consistent with each other.
    """

    df = kwargs.get("df")

    # Required columns
    ego = kwargs.get("ego")
    alter = kwargs.get("alter")
    reporter = kwargs.get("reporter")

    # Optional columns
    layer = kwargs.get("layer")
    weight = kwargs.get("weight")

    # Optional parameters
    nodes = kwargs.get("nodes")
    reporters = kwargs.get("reporters")

    if not isinstance(df, pd.DataFrame):
        msg = f"'df' should be a DataFrame, instead it is of type: {type(df)}."
        raise ValueError(msg)
    
    # Throws a ValueError if the required columns are not present in df
    dict_required_columns = {"ego": ego, "alter": alter, "reporter": reporter}
    _check_required_columns(df, dict_required_columns)

    if nodes is not None and not isinstance(nodes, list):
        msg = f"'nodes' should be a list, instead it is of type: {type(nodes)}."
        raise ValueError(msg)
    
    if reporters is not None and not isinstance(reporters, list):
        msg = f"'reporters' should be a list, instead it is of type: {type(reporters)}."
        raise ValueError(msg)

    if nodes == []:
        msg = (
            "The set of nodes was not informed, "
            f"using {ego} and {alter} columns to infer nodes."
        )
        warnings.warn(msg, UserWarning)
        nodes = pd.concat([df[ego], df[alter]]).unique().tolist() # type: ignore

    if np.logical_or(~df[ego].isin(nodes), ~df[alter].isin(nodes)).any(): # type: ignore
        msg = (
            "A list of nodes was informed, "
            "but it does not contain all nodes in the data frame."
        )
        raise ValueError(msg)
    
    # Check optional columns. Does not throw an error if they are not present.
    dict_optional_columns = {"layer": layer, "weight": weight}
    missing_optional_columns = _check_optional_columns(df, dict_optional_columns)

    if len(missing_optional_columns) > 0:
        if layer in missing_optional_columns:
            df.loc[:, layer] = "1" # type: ignore
        
        if weight in missing_optional_columns:
            df.loc[:, weight] = 1 # type: ignore

    if reporters is None or reporters == []:
        msg = (
            "The set of reporters was not informed, "
            "assuming set(reporters) = set(nodes) and N = M."
        )
        warnings.warn(msg, UserWarning)
        
        # https://stackoverflow.com/a/40382592/843365
        reporters = nodes[:] # type: ignore

        # Reporters were inferred from nodes,
        #  so we need to check if they are all reporters
        reporters_in_df = df[reporter].unique().tolist() # type: ignore
        if not set(reporters_in_df).issubset(reporters):
            error_msg = (
                "This survey setup is not currently supported by the package: "
                " some reporters are not nodes in the network. "
                "Hint: If this is unexpected behaviour, "
                f"compare the unique values of the `{str(reporter)}` column "
                f"with those of the `{str(ego)}` and `{str(alter)}` columns."
            )
            raise ValueError(error_msg)
    else:
        reporters_in_df = df[reporter].unique().tolist() # type: ignore
        if not set(reporters_in_df).issubset(reporters):
            error_msg = (
                "Some reporters in the data frame do not appear "
                "in the list of reporters provided. "
                f"Hint: Compare the unique values of the `{str(reporter)}` column "
                "with the list of reporters passed as parameter."
            )
            raise ValueError(error_msg)

    if not set(reporters).issubset(nodes): # type: ignore
        error_msg = (
            "This survey setup is not currently supported by the package: "
            " some reporters are not nodes in the network. "
            "Hint: If this is unexpected behaviour, "
            f"compare the unique values of the `{str(reporter)}` column "
            f"with those of the `{str(ego)}` and `{str(alter)}` columns."
        )
        raise ValueError(error_msg)

    if not set(nodes).issubset(reporters): # type: ignore
        warnings.warn(
            "Not necessarily a problem, but"
            " some of the nodes are not reporters.",
            UserWarning
        )

    return df, nodes, reporters

def _check_required_columns(df, dict_required_columns):
    """
    Check if the required columns are present in the dataframe.
    """

    required_columns = list(dict_required_columns.values())

    # Collect which required columns are not present in df
    missing_columns = [col for col in required_columns if col not in df.columns]

    if len(missing_columns) > 0:
        error_msg = (
            f"Required columns not found in data frame: {', '.join(missing_columns)}. "
            "Mapping used: "
            f"ego='{dict_required_columns['ego']}', "
            f"alter='{dict_required_columns['alter']}', "
            f"reporter='{dict_required_columns['reporter']}'. "
            "Hint: Use params ego,alter,... for mapping column names."
        )
        raise ValueError(error_msg)
    
    return True

def _check_optional_columns(df, dict_optional_columns):
    """
    Check if the optional columns are present in the dataframe.
    """

    optional_columns = list(dict_optional_columns.values())

    # Collect which optional columns are not present in df
    missing_columns = [col for col in optional_columns if col not in df.columns]

    return missing_columns


