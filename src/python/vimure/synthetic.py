"""Code to generate synthetic networks that emulates directed double-sample questions networks

You can read more about our synthetic network generation in our paper: [@de_bacco_latent_2023]

"""

# Paper link: https://doi.org/10.1093/jrsssa/qnac004

import math
import torch

import numpy as np
import pandas as pd
import networkx as nx
import sktensor as skt

from abc import ABCMeta, abstractmethod
from ._io import BaseNetwork
from ._log import setup_logging
from .utils import preprocess, sptensor_from_dense_array, find_indices_of_value

DEFAULT_N = 100
DEFAULT_M = 100
DEFAULT_L = 1
DEFAULT_K = 2

DEFAULT_C = 2
DEFAULT_STRUCTURE = None
DEFAULT_SPARSIFY = True
DEFAULT_OVERLAPPING = 0.0

DEFAULT_EXP_IN = 2
DEFAULT_EXP_OUT = 2.5

DEFAULT_ETA = 0.5
DEFAULT_AVG_DEGREE = 2

module_logger = setup_logging("vm.synthetic", False)

class BaseSyntheticNetwork(BaseNetwork, metaclass=ABCMeta):
    """
    A base abstract class for generation and management of synthetic networks. 
    Suitable for representing any type of synthetic network (whether SBM or not).
    """
    def __init__(
        self,
        N: int = DEFAULT_N,
        M: int = DEFAULT_M,
        L: int = DEFAULT_L,
        K: int = DEFAULT_K,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(N=N, M=M, L=L, K=K, seed=seed, **kwargs)

    @abstractmethod
    def _generate_lv(self):
        pass

    @abstractmethod
    def _build_Y(self):
        pass

    def _build_X(
        self,
        mutuality: float = 0.5,
        sh_theta: float = 2.0,
        sc_theta: float = 0.5,
        flag_self_reporter: bool = True,
        cutoff_X: bool = False,
        lambda_diff: float = None,
        Q: int = None,
        seed: int = None,
        theta: np.ndarray = None,
        verbose: bool = False,
    ):
        """
        Any object inhereted from BaseSyntheticNetwork will have a ground truth network Y.
        Given that Y, generate the observed network X.
        
        Parameters
        ----------

        mutuality : float
            The mutuality parameter from 0 to 1.
        sh_theta : float
            Shape of gamma distribution from which to draw theta. 
            The 'reliability' of nodes is represented by the parameter `theta_{lm}`
            and by default are modelled as a gamma function with shape `sh_theta` and scale `sc_theta`.
        sc_theta : float
            Scale of gamma distribution from which to draw theta.
        flag_self_reporter : bool
            Indicates whether a node can only report about their own ties.
        Q : int
            Maximum value of X entries. If None, it will use the network's K parameter.
        cutoff_X : bool
            Whether to set X as a binary.
        lambda_diff : float
            The difference between each subsequent K.
        seed : int
            Pseudo random generator seed to use.
        verbose : bool
            Provides additional details.

        Returns
        -------
        X : sktensor
            Observed network.
        """

        logger = setup_logging("vm.synthetic.generate_X", verbose)

        if mutuality < 0 or mutuality >= 1:
            msg = "The mutuality parameter has to be in [0, 1)!"
            raise ValueError(msg)

        # generate_x uses its own pseudo-random seed generator, so we can better control the generation of theta.
        # We can generate the same theta, while varying mutuality (to check how X change);
        if seed is None:
            seed = self.seed
        
        prng = torch.Generator()
        prng.manual_seed(seed)

        Y = self.Y
        N = self.N
        M = self.M
        L = self.L
        K = self.K

        if theta is not None:
            warn_msg = "Ignoring sh_theta and sc_theta since a full theta matrix was informed"
            logger.debug(warn_msg)

            if type(theta) != np.ndarray or theta.shape != (L, M):
                msg = "theta matrix is not valid. When using this parameter, make sure to inform a %d x %d matrix."
                raise ValueError(msg % (L, M))
        else:
            # Generate theta (reliability)
            gamma = torch.distributions.gamma.Gamma(concentration=torch.tensor([sh_theta]), 
                                                    rate=torch.tensor([1/sc_theta]))
            theta = gamma.rsample((L, M))[:,:,0]

        LAMBDA_0 = 0.01
        # Generate theta (ties average interactions)
        lambda_k = torch.ones(Y.shape).float() * LAMBDA_0

        if lambda_diff is not None:
            if lambda_diff <= 0:
                msg = "lambda_diff is optional but when set should be higher than 0!"
                raise ValueError(msg)

        for k in range(1, K):
            indices_val_equals_k = find_indices_of_value(Y, k)
            lij = (indices_val_equals_k[0], 
                    indices_val_equals_k[1], 
                    indices_val_equals_k[2])
            if lambda_diff is not None:
                lambda_k[lij] = LAMBDA_0 + lambda_diff
            else:
                lambda_k[lij] = K

        M_X = torch.einsum("lm,lij->lijm", theta, lambda_k)
        MM = (M_X + mutuality * M_X.transpose(1, 2)) / (1.0 - mutuality * mutuality)

        X = torch.zeros(MM.shape).to(torch.int16)

        if cutoff_X and Q is None:
            Q = self.K

        if flag_self_reporter:
            #TODO: convert to torch
            R = build_self_reporter_mask(self)

            for l in range(L):
                for m in range(M):
                    subs_nz = np.where(R[l, :, :, m] > 0)
                    len_subs_nz = subs_nz[0].shape[0]
                    for n in range(len_subs_nz):
                        i, j = subs_nz[0][n], subs_nz[1][n]
                        r = prng.rand(1)[0]

                        # for those reporters that report perfectly, i.e. theta=1, do not extract from a poisson.
                        # Rather, assign the X deterministically using the mean of the poisson
                        if np.allclose(theta[l, m], 1.0) == False:
                            if r < 0.5:
                                X[l, i, j, m] = prng.poisson(MM[l, i, j, m] * R[l, i, j, m])
                                if cutoff_X:
                                    if X[l, i, j, m] > Q - 1:
                                        X[l, i, j, m] = Q - 1
                                cond_exp = M_X[l, j, i, m] * R[l, i, j, m] + mutuality * X[l, i, j, m]
                                X[l, j, i, m] = prng.poisson(cond_exp)
                            else:
                                X[l, j, i, m] = prng.poisson(MM[l, j, i, m] * R[l, j, i, m])
                                if cutoff_X:
                                    if X[l, j, i, m] > Q - 1:
                                        X[l, j, i, m] = Q - 1
                                cond_exp = M_X[l, i, j, m] * R[l, j, i, m] + mutuality * X[l, j, i, m]
                                X[l, i, j, m] = prng.poisson(cond_exp)
                        else:
                            if r < 0.5:
                                X[l, i, j, m] = MM[l, i, j, m] * R[l, i, j, m]
                                if cutoff_X == True:
                                    if X[l, i, j, m] > Q - 1:
                                        X[l, i, j, m] = Q - 1
                                cond_exp = M_X[l, j, i, m] * R[l, i, j, m] + mutuality * X[l, i, j, m]
                                X[l, j, i, m] = cond_exp
                            else:
                                X[l, j, i, m] = MM[l, j, i, m] * R[l, j, i, m]
                                if cutoff_X == True:
                                    if X[l, j, i, m] > Q - 1:
                                        X[l, j, i, m] = Q - 1
                                cond_exp = M_X[l, i, j, m] * R[l, j, i, m] + mutuality * X[l, j, i, m]
                                X[l, i, j, m] = cond_exp

        else:
            R = np.ones((L, N, N, M))
            for l in range(L):
                for m in range(M):
                    for i in range(N):
                        for j in range(i + 1, N):
                            r = prng.rand(1)[0]
                            if r < 0.5:
                                X[l, i, j, m] = prng.poisson(MM[l, i, j, m])
                                if cutoff_X:
                                    if X[l, i, j, m] > Q - 1:
                                        X[l, i, j, m] = Q - 1
                                cond_exp = M_X[l, j, i, m] + mutuality * X[l, i, j, m]
                                X[l, j, i, m] = prng.poisson(cond_exp)
                            else:
                                X[l, j, i, m] = prng.poisson(MM[l, j, i, m])
                                if cutoff_X:
                                    if X[l, j, i, m] > Q - 1:
                                        X[l, j, i, m] = Q - 1
                                cond_exp = M_X[l, i, j, m] + mutuality * X[l, j, i, m]
                                X[l, i, j, m] = prng.poisson(cond_exp)

        if cutoff_X:
            X[X > Q - 1] = Q - 1  # cut-off, max entry has to be equal to K - 1

        self.X = preprocess(X)
        self.R = preprocess(R)
        self.theta = theta
        self.lambda_k = lambda_k
        self.mutuality = mutuality

        subs_lijm = self.X.subs

        """
        BASELINE: UNION

        This is one of the simplest possible ways of collating all networks reported by independent reporters (X) 
        into a single adjacency matrix (Y).

        The baseline union we use here takes the union of all ties (l, i, j) that were reported at least once by someone
        regardless of who reported it or the strength given to each tie.

        Note: the output adjacency matrix is binary

        # TODO: Maybe the calculation of baseline adjacency matrices shouldn't be inside the _build_X() function, 
        #       since it isn't a feature of the package itself?
        """
        lij = (subs_lijm[0], subs_lijm[1], subs_lijm[2])
        all_ties = np.stack(lij).T  # dim ((lij)_subs, 3)
        union_ties, count_ties = np.unique(all_ties, axis=0, return_counts=True)

        # Convert (l,i,j) to format understood by sktensor
        union_subs = tuple(np.moveaxis(union_ties, 1, 0))
        X_union = skt.sptensor(
            subs=union_subs,
            vals=np.ones(union_ties.shape[0]),
            shape=(self.L, self.N, self.N),
            dtype=np.int8,
        )
        self.X_union = X_union

        """
        BASELINE: INTERSECTION

        The intersection is another simple way to combine all networks reported by independent reporters (X) 
        into a single adjacency matrix (Y).

        Here, a tie (l, i, j) is only considered if all reporters **who were allowed to report on that tie, 
        as indicated by the reporter's mask R,** reported this tie.

        Similar to the union baseline, here we disregard the strength of ties. 

        # TODO: Maybe the calculation of baseline adjacency matrices shouldn't be inside the _build_X() function, 
        #       since it isn't a feature of the package itself?
        """

        """
        Intersection Baseline | Step 1
        
        First create a skt.sptensor called max_reports_lij to hold the maximum number of reports a tie (l, i, j) could receive
        """
        if isinstance(R, skt.dtensor) or isinstance(R, np.ndarray):
            # Since R is a dense tensor of dimensions (l, i, j, m), we just need to sum over the m dimension
            max_reports_lij = sptensor_from_dense_array(R.sum(axis=3))

            df_max_reports_lij = pd.DataFrame(max_reports_lij.subs).T
            df_max_reports_lij.columns = ["l", "i", "j"]
            df_max_reports_lij["max"] = max_reports_lij.vals
        elif isinstance(R, skt.sptensor):
            # Because R is a sparse tensor, we need to work out the sum another way.
            
            # When performing operations with skt.sptensor, I (@jonjoncardoso) find it more intuitive -- and faster -- 
            # to work with pandas groupby -> apply() than manipulating their subs+vals with numpy directly.
            R_vals_df = pd.DataFrame(np.stack(self.R.subs).T, columns=["l", "i", "j", "m"])

            df_max_reports_lij = R_vals_df.groupby(["l", "i", "j"])\
                .apply(lambda x: pd.Series({"max": sum([R[l, i, j, m]
                                                   if type(R[l, i, j, m]) == int else R[l, i, j, m][0] 
                                                   for m in range(self.M)])}))


        """
        Intersection Baseline | Step 2

        Convert `union_ties` to pd.DataFrame
        """
        df_union_ties = pd.DataFrame(union_ties, columns=["l", "i", "j"])
        df_union_ties["count"] = count_ties

        """
        Intersection Baseline | Step 3

        Find out how many, of the union ties, were unanimous. 
        That is: all reporters **that could report** on the tie (l, i, j) did agree that this tie existed and have reported it.

        NOTE: When performing operations with skt.sptensor, I (@jonjoncardoso) find it more intuitive -- and faster -- 
              to work with pandas groupby -> apply() or pandas merge than manipulating their subs+vals with numpy directly.
        """
        aux_df = pd.merge(df_union_ties, df_max_reports_lij, how="left")
        # Results DataFrame has columns: [l, i, j, count, max]
        df_intersection_ties = aux_df[aux_df["count"] == aux_df["max"]]
        
        """
        Intersection Baseline | Step 4

        Create a skt.sptensor to represent the intersection of X ties. 
        """

        if df_intersection_ties.empty:
            # If NO REPORTER has agreed on any ties, than intersection is empty
            self.X_intersection = None
        else:
            self.X_intersection = skt.sptensor(
                subs=tuple(df_intersection_ties[["l", "i", "j"]].T.values.tolist()),
                vals=np.ones(df_intersection_ties.shape[0]).tolist(),
                shape=(self.L, self.N, self.N),
                dtype=np.int8,
            )          

        # TODO: Rethink the baseline union & intersections for the case that X is not a binary matrix (or even if Y is binary)

        return self

    def __repr__(self):
        return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, seed={self.seed})"

    def __str__(self):
        return f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, K={self.K}, seed={self.seed})"


class StandardSBM(BaseSyntheticNetwork):
    """
    **Creates a standard stochastic block-model synthetic network.**

    A generative graph model which assumes the probability of connecting two nodes in a graph is determined entirely by their block assignments.
    For more information about this model, see Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). _Stochastic blockmodels: First steps. Social networks_, 5(2), 109-137.
    [DOI:10.1016/0378-8733(83)90021-7](https://www.sciencedirect.com/science/article/abs/pii/0378873383900217)
    """

    # TODO: Document overlapping communities separately as it involves setting several other parameters

    def __init__(
        self,
        C: int = DEFAULT_C,
        structure: str = DEFAULT_STRUCTURE,
        avg_degree: float = DEFAULT_AVG_DEGREE,
        sparsify: bool = DEFAULT_SPARSIFY,
        overlapping: float = DEFAULT_OVERLAPPING,
        **kwargs,
    ):
        """
        Parameters
        ----------
        C : int
            Number of communities
        structure: str
            Structures for the affinity tensor `w`. It can be 'assortative' or 'disassortative'. 
            It can be a list to map structure for each layer in a multilayer graph.
        avg_degree : float
            Desired average degree for the network. It is not guaranteed that the 
            ultimate network will have that exact average degree value. 
            Try tweaking this parameter if you want to increase or decrease the 
            density of the network.
        sparsify : bool
            If True (default), enforce sparsity.
        overlapping : float
            Fraction of nodes with mixed membership. It has to be in `[0, 1)`.
        kwargs : 
            Additional arguments of `vimure.synthetic.StandardSBM` 
        """
        self._init_sbm_params(
            C=C,
            structure=structure,
            avg_degree=avg_degree,
            sparsify=sparsify,
            overlapping=overlapping,
            **kwargs,
        )
        self._build_Y()

    def _init_sbm_params(self, **kwargs):
        """
        Check SBM-specific parameters
        """

        super().__init__(**kwargs)

        self.u = np.zeros((self.N, self.K), dtype=float)  # out-going membership
        self.v = np.zeros((self.N, self.K), dtype=float)  # in-going membership

        if "C" in kwargs:
            self.C = kwargs["C"]  # number of communities
        else:
            msg = "C parameter was not set. Defaulting to C=%d" % DEFAULT_C
            module_logger.warning(msg)
            self.C = DEFAULT_C

        if "avg_degree" in kwargs:
            avg_degree = kwargs["avg_degree"]
        else:
            msg = "avg_degree parameter was not set. Defaulting to avg_degree=%d" % DEFAULT_AVG_DEGREE
            module_logger.warning(msg)
            avg_degree = DEFAULT_AVG_DEGREE
        self.avg_degree = avg_degree

        if "sparsify" in kwargs:
            sparsify = kwargs["sparsify"]
        else:
            msg = "sparsify parameter was not set. Defaulting to sparsify=False"
            module_logger.warning(msg)
            sparsify = False
        self.sparsify = sparsify

        """
        SETUP overlapping communities
        """
        if "overlapping" in kwargs:
            overlapping = kwargs["overlapping"]
            # fraction of nodes with mixed membership
            if (overlapping < 0) or (overlapping > 1):
                err_msg = "The overlapping parameter has to be in [0, 1]!"
                raise ValueError(err_msg)
        else:
            overlapping = False
        self.overlapping = overlapping
        if self.overlapping:

            if "corr" in kwargs:
                # correlation between u and v synthetically generated
                if (kwargs["corr"] < 0) or (kwargs["corr"] > 1):
                    msg = "The correlation parameter corr has to be in [0, 1]!"
                    raise ValueError(msg)

                corr = float(kwargs["corr"])
            else:
                msg = "corr parameter for overlapping communities was not set. Defaulting to corr=0."
                module_logger.warning(msg)
                corr = 0
            self.corr = corr

        if self.overlapping > 0:
            if "normalization" in kwargs:
                self.normalization = bool(kwargs["normalization"])
            else:
                msg = "Normalization parameter was not set. Defaulting to normalization=False (Dirichlet overlapping communities)"
                module_logger.warning(msg)
                self.normalization = False

            if self.normalization:
                if "ag" in kwargs:
                    self.ag = float(kwargs["ag"])
                else:
                    msg = "Parameter alpha for the Gamma distribution was not set. Defaulting to alpha=0.1"
                    module_logger.warning(msg)
                    self.ag = 0.1

                if "beta" in kwargs:
                    self.beta = float(kwargs["beta"])
                else:
                    msg = "Parameter beta for the Gamma distribution was not set. Defaulting to beta=0.1"

                    module_logger.warning(msg)
                    self.beta = 0.1
            else:
                if "alpha" in kwargs:
                    self.alpha = float(kwargs["alpha"])
                else:
                    msg = (
                        "Parameter alpha for the Dirichlet distribution was not set. Defaulting to alpha=0.1"
                    )
                    module_logger.warning(msg)
                    self.alpha = 0.1

        """
        SETUP informed structure
        """
        if "structure" in kwargs:
            structure = kwargs["structure"]
        else:
            structure = None

        if structure is None:
            structure = ["assortative"] * self.L
        elif type(structure) == str:
            if structure not in ["assortative", "disassortative"]:
                msg = "The available structures for the affinity tensor w are: assortative, disassortative!"
                raise ValueError(msg)
            else:
                structure = [structure] * self.L
        elif len(structure) != self.L:  # list of structures of the affinity tensor w
            msg = (
                "The parameter structure should be a list of length L. "
                "Each entry defines the structure of the corresponding layer!"
            )
            raise ValueError(msg)
        for e in structure:
            if e not in ["assortative", "disassortative"]:
                msg = (
                    "The available structures for the affinity tensor w are: " "assortative, disassortative.!"
                )
                raise ValueError(msg)
        self.structure = structure

    def __repr__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping})"
        return return_str

    def __str__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping})"
        return return_str

    def _build_Y(self):
        """
        Latent variables
        """
        self.u, self.v, self.w = self._generate_lv()
        """
        Generate Y
        """
        M_Y = torch.einsum("ik,jq->ijkq", self.u, self.v)
        M_Y = torch.einsum("ijkq,akq->aij", M_Y, self.w)

        # sparsity parameter for Y
        if self.sparsify:
            # TODO: Explain rationale behind this particular formula
            c = (float(self.N) * self.avg_degree) / M_Y.sum()
            M_Y *= c
            self.w *= c

        Y = torch.poisson(M_Y, self.prng)

        # Instead of np.fill_diagonal(Y, 0), 
        # we use the following https://stackoverflow.com/a/49512781/843365
        for l in range(self.L):
            Y[l][torch.eye(self.N).bool()] = 0

        # cut-off, max entry has to be equal to K - 1
        Y[Y > self.K - 1] = self.K - 1  

        self.Y = Y.to_sparse_coo()

    def __sample_membership_vectors(self):
        """
        Compute the NxK membership vectors u, v using a Dirichlet distribution.

        INPUT
        ----------
        prng: Numpy Random object
              Random number generator container.
        alpha : float
                Parameter for Dirichlet.
        N : int
            Number of nodes.
        C : int
            Number of communities.
        corr : float
               Correlation between u and v synthetically generated.
        over : float
               Fraction of nodes with mixed membership.

        OUTPUT
        -------
        u : Numpy array
            Matrix NxC of out-going membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.

        v : Numpy array
            Matrix NxC of in-coming membership vectors, positive element-wise.
            With unitary L1 norm computed row-wise.
        """

        # Generate equal-size unmixed group membership
        size = int(self.N / self.C)
        u = torch.zeros((self.N, self.C), dtype=torch.float32)
        v = torch.zeros((self.N, self.C), dtype=torch.float32)

        for i in range(self.N):
            q = int(math.floor(float(i) / float(size)))
            if q == self.C:
                u[i:, self.C - 1] = 1
                v[i:, self.C - 1] = 1
            else:
                for j in range(q * size, q * size + size):
                    u[j, q] = 1
                    v[j, q] = 1

        return u, v

    def __normalize_nonzero_membership(self, u):
        """
        Given a matrix, it returns the same matrix normalized by row.

        INPUT
        ----------
        u: Numpy array
           Numpy Matrix.

        OUTPUT
        -------
        The matrix normalized by row.
        """

        den1 = u.sum(axis=1, keepdims=True)
        nzz = den1 == 0.0
        den1[nzz] = 1.0

        return u / den1

    def __compute_affinity_matrix(self, structure, a=0.1):
        """
        Compute the CxC affinity matrix w with probabilities between and within groups.

        Parameters
        ----------
        structure : string
            Structure of the network.
        a : float
            Parameter for secondary probabilities.

        Returns
        -------
        p : Numpy array
            Array with probabilities between and within groups. Element (k,h)
            gives the density of edges going from the nodes of group k to nodes of group h.
        """

        p1 = self.avg_degree * self.C / self.N

        if structure == "assortative":
            p = p1 * a * np.ones((self.C, self.C))  # secondary-probabilities
            np.fill_diagonal(p, p1 * np.ones(self.C))  # primary-probabilities

        elif structure == "disassortative":
            p = p1 * np.ones((self.C, self.C))  # primary-probabilities
            np.fill_diagonal(p, a * p1 * np.ones(self.C))  # secondary-probabilities

        return p

    def __apply_overlapping(self, u, v):
        overlapping = int(self.N * self.overlapping)  # number of nodes belonging to more communities
        ind_over = torch.randint(len(u), size=(overlapping,), dtype=torch.long)

        if not self.normalization:
            # u and v from a Dirichlet distribution
            dirichlet_dist = torch.distributions.dirichlet.Dirichlet(self.alpha * torch.ones(self.C))
            u[ind_over] = dirichlet_dist.sample((overlapping,))
            v[ind_over] = self.corr * u[ind_over] + (1.0 - self.corr) * dirichlet_dist.sample((overlapping,))
            if self.corr == 1.0:
                assert np.allclose(u, v)
            if self.corr > 0:
                v = self.__normalize_nonzero_membership(v)
        else:
            # u and v from a Gamma distribution
            u[ind_over] = self.prng.gamma(self.ag, 1.0 / self.beta, size=(overlapping, self.C))
            v[ind_over] = self.corr * self.u[ind_over] + (1.0 - self.corr) * self.prng.gamma(
                self.ag, 1.0 / self.beta, size=(overlapping, self.C)
            )
            u = self.__normalize_nonzero_membership(u)
            v = self.__normalize_nonzero_membership(v)

        return u, v

    def _generate_lv(self):
        """
        Generate latent variables for a Stochastic BlockModel, assuming network layers are independent.
        """

        # Generate u, v for overlapping communities
        u, v = self.__sample_membership_vectors()

        if self.overlapping > 0:
            u, v = self.__apply_overlapping(u, v)

        # Generate w
        w = np.zeros((self.L, self.C, self.C))
        for l in range(self.L):
            w[l, :, :] = self.__compute_affinity_matrix(self.structure[l])

        w = torch.from_numpy(w).to(torch.float32)

        return u, v, w


class DegreeCorrectedSBM(StandardSBM):
    """
    **Degree-corrected stochastic blockmodel.**

    A generative model that incorporates heterogeneous vertex degrees into stochastic blockmodels, improving the performance of the models for statistical inference of group structure.
    For more information about this model, see [@karrer_stochastic_2011].
    """
    def __init__(
            self, exp_in: float = DEFAULT_EXP_IN,
            exp_out: float = DEFAULT_EXP_OUT,
            **kwargs):
        """
        Parameters
        ----------
        exp_in : float
            Exponent power law of in-degree distribution.
        exp_out : float
            Exponent power law of out-degree distribution.
        kwargs : dict
            Additional arguments of `vimure.synthetic.StandardSBM`.
        """

        self.exp_in = exp_in  # exponent power law distribution in degree
        self.exp_out = exp_out  # exponent power law distribution out degree

        # Initialize all other variables
        super().__init__(**kwargs)

    def _generate_lv(self):
        """
        Overwrite standard SBM model to add degree distribution
        """

        u, v, w = super()._generate_lv()

        # We add +1 to the degree distributions to avoid creating disconnected nodes
        self.d_in = np.array(
            [int(x) + 2 for x in nx.utils.powerlaw_sequence(self.N, exponent=self.exp_in, seed=self.seed)]
        )

        self.d_out = np.array(
            [int(x) + 1 for x in nx.utils.powerlaw_sequence(self.N, exponent=self.exp_out, seed=self.seed)]
        )

        u_hat = u * torch.from_numpy(self.d_out[:, np.newaxis]).to(torch.int8)
        v_hat = v * torch.from_numpy(self.d_in[:, np.newaxis]).to(torch.int8)

        return u_hat, v_hat, w

    def __repr__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping}, "
        return_str += f"exp_in={self.exp_in}, exp_out={self.exp_out})"
        return return_str

    def __str__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping}, "
        return_str += f"exp_in={self.exp_in}, exp_out={self.exp_out})"
        return return_str


class Multitensor(StandardSBM):
    """
    **A generative model with reciprocity**

    A mathematically principled generative model for capturing both community and reciprocity patterns in directed networks.
    Adapted from [@safdari_generative_2021].
    
    Generate a directed, possibly weighted network by using the reciprocity generative model.
    Can be used to generate benchmarks for networks with reciprocity.

    **Steps:**
    
    1. Generate the latent variables.
    2. Extract `A_{ij}` entries (network edges) from a Poisson distribution; its mean depends on the latent variables.
        
    .. note:: Open Source code available at https://github.com/mcontisc/CRep and modified in accordance with its [license](https://github.com/mcontisc/CRep/blob/master/LICENSE).

    -------------------------------------------------------------------------------
    Copyright (c) 2020 Hadiseh Safdari, Martina Contisciani and Caterina De Bacco.
    """

    def __init__(self, eta=DEFAULT_ETA, ExpM=None, **kwargs):
        """       
        Parameters
        ----------
        eta : float
            Initial value for the reciprocity coefficient. Eta has to be in [0, 1).
        ExpM : int
            Expected number of edges
        kwargs : 
            Additional arguments of `vimure.synthetic.StandardSBM` 
        """
        super()._init_sbm_params(**kwargs)

        if eta < 0 or eta >= 1:
            msg = "The reciprocity parameter eta has to be in [0, 1)!"
            raise ValueError(msg)
        self.eta = eta
        if ExpM is None:  # expected number of edges
            self.ExpM = int(self.N * self.avg_degree / 2.0)
        else:
            self.ExpM = int(ExpM)
            self.avg_degree = 2 * self.ExpM / float(self.N)

        self._build_Y()

    def _Exp_ija_matrix(self, u, v, w):
        """
        Compute the mean lambda0_ij for all entries.

        Parameters
        ----------
        u : torch.Tensor
            Out-going membership matrix.
        v : torch.Tensor
            In-coming membership matrix.
        w : torch.Tensor
            Affinity matrix.
            
        Returns
        -------
        M : torch.Tensor
            Mean `\lambda^{0}_{ij}` for all entries.
        """

        M = torch.einsum("ik,jq->ijkq", u, v)
        M = torch.einsum("ijkq,kq->ij", M, w)

        return M

    def _build_Y(self):
        """
        Generate network layers G and adjacency matrix A using the latent variables,
        with the generative model `(A_{ij},A_{ji}) ~ P(A_{ij}|u,v,w,eta) P(A_{ji}|A_{ij},u,v,w,eta)`
        """

        Y = torch.zeros(size=(self.L, self.N, self.N), dtype=torch.float32)

        self.u, self.v, self.w = self._generate_lv()

        for l in range(self.L):

            # whose elements are lambda0_{ij}
            M0 = self._Exp_ija_matrix(self.u, self.v, self.w[l])  

            # Instead of np.fill_diagonal(Y, 0),
            # we use the following https://stackoverflow.com/a/49512781/843365
            M0[torch.eye(self.N).bool()] = 0

            if self.sparsify:
                # constant to enforce sparsity
                c = (self.ExpM * (1.0 - self.eta)) / M0.sum()

            # whose elements are m_{ij}
            MM = (M0 + self.eta * M0.transpose(0,1)) / (1.0 - self.eta * self.eta)
            Mt = MM.transpose(0,1)

            # equivalent of M0.copy()
            # as per https://stackoverflow.com/a/62496418/843365
            MM0 = M0.detach().clone()  # to be not influenced by c_lambda

            if self.sparsify:
                M0 *= c
                # To allow reconstruction of the network from u, v, w
                self.w *= c  

            # whose elements are lambda0_{ji}
            M0t = M0.transpose(0,1)

            # whose elements are m_{ij}
            M = (M0 + self.eta * M0t) / (1.0 - self.eta * self.eta)
            # Instead of np.fill_diagonal(Y, 0),
            # we use the following https://stackoverflow.com/a/49512781/843365
            M[torch.eye(self.N).bool()] = 0

            # expected reciprocity
            rw = self.eta + ((MM0 * Mt + self.eta * Mt ** 2).sum() / MM.sum())

            """
            # TODO:Document this section
            """

            G = nx.DiGraph()
            for i in range(self.N):
                G.add_node(i)

            counter = 0
            totM = 0

            for i in range(self.N):
                for j in range(i + 1, self.N):
                    r = torch.rand(1, generator=self.prng).item()
                    if r < 0.5:
                        # draw A_ij from P(A_ij) = Poisson(m_ij)
                        A_ij = torch.poisson(M[i, j], generator=self.prng).item()
                        if A_ij > 0:
                            G.add_edge(i, j, weight=A_ij)
                        lambda_ji = M0[j, i] + self.eta * A_ij

                        # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                        A_ji = torch.poisson(lambda_ji, generator=self.prng).item()
                        if A_ji > 0:
                            G.add_edge(j, i, weight=A_ji)
                    else:
                        # draw A_ij from P(A_ij) = Poisson(m_ij)
                        A_ji = torch.poisson(M[j, i], generator=self.prng).item()
                        if A_ji > 0:
                            G.add_edge(j, i, weight=A_ji)
                        lambda_ij = M0[i, j] + self.eta * A_ji

                        # draw A_ji from P(A_ji|A_ij) = Poisson(lambda0_ji + eta*A_ij)
                        A_ij = torch.poisson(lambda_ij, generator=self.prng).item()
                        if A_ij > 0:
                            G.add_edge(i, j, weight=A_ij)
                    counter += 1
                    totM += A_ij + A_ji

            # number of connected components
            n_connected_comp = len(list(nx.weakly_connected_components(G)))
            if n_connected_comp > 1:
                msg = f"Multitensor has produced a network with {n_connected_comp} connected components. "
                msg += "You can try increasing avg_degree and/or running with different seeds "
                msg += "until you get a network with just a single giant component."
                module_logger.warning(msg)

            # Adjacency matrix
            Y_l = torch.from_numpy(nx.to_numpy_array(G)).int()

            # cut-off, max entry has to be equal to K - 1
            Y_l[Y_l > self.K - 1] = self.K - 1 
            Y[l] = Y_l

        self.Y = Y.to_sparse_coo()

    def __repr__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping}, "
        return_str += f", eta={self.eta}, ExpM={self.ExpM})"
        return return_str

    def __str__(self):
        return_str = f"{self.__class__.__name__} (N={self.N}, M={self.M}, L={self.L}, "
        return_str + f"K={self.K}, seed={self.seed}, "
        return_str += f"C={self.C}, structure={self.structure}, avg_degree={self.avg_degree}, "
        return_str += f"sparsify={self.sparsify}, overlapping={self.overlapping}, "
        return_str += f", eta={self.eta}, ExpM={self.ExpM})"
        return return_str


class HollandLaskeyLeinhardtModel(BaseSyntheticNetwork):
    def __init__(self, **kwargs):
        raise NotImplementedError


"""
FUNCTIONS TO GENERATE X (OBSERVED NETWORK) FROM A GENERATED GROUND TRUTH NETWORK, Y
"""


def build_self_reporter_mask(gt_network):
    """
    Build the reporters' mask in a way such that:

    - A reporter `m` can report ties in which she is ego:   `m --> i`
    - A reporter `m` can report ties in which she is alter: `i --> m`
    - A reporter `m` **cannot** report ties she is not involved, that is `i --> j` where `i != m` and `j != m`
    
    Parameters
    ----------
    gt_network : vimure.synthetic.BaseSyntheticNetwork
        Generative ground truth model.
    """

    # TODO: Use sparse matrices instead

    R = np.zeros((gt_network.L, gt_network.N, gt_network.N, gt_network.M))
    R[:, np.arange(gt_network.M), :, np.arange(gt_network.M)] = 1
    R[:, :, np.arange(gt_network.M), np.arange(gt_network.M)] = 1

    return R


def build_custom_theta(
    gt_network: BaseSyntheticNetwork,
    theta_ratio: float = 0.5,
    exaggeration_type: str = "over",
    seed: int = None,
):
    """
    Instead of the regular generative model for `theta ~ Gamma(sh,sc)`,
    create a more extreme scenario where some percentage of reporters are exaggerating.

    Parameters
    ----------
    gt_network : vimure.synthetic.BaseSyntheticNetwork
        A network generated from a generative model.
    theta_ratio : float
        Percentage of reporters who exaggerate.
    exaggeration_type : str
        ("over", "under")
    seed : int
        If not set, use gt_network.prng instead.

    Returns
    ----------
    theta : numpy.array
        A L x M matrix for theta.

    """

    if theta_ratio < 0 or theta_ratio > 1:
        raise ValueError("theta_ratio should be in the interval [0, 1]")

    if exaggeration_type not in ["over", "under"]:
        raise ValueError("Unrecognised exaggeration_type: %s" % exaggeration_type)

    prng = np.random.RandomState(seed)

    nodes = np.arange(gt_network.M)
    theta = np.ones((gt_network.L, gt_network.M))

    # Number of "unreliable" reporters
    N_exa = int(gt_network.M * theta_ratio)
    selected_reporters = prng.choice(nodes, size=N_exa, replace=False)
    if N_exa > 0.0:
        if exaggeration_type == "under":
            theta[:, selected_reporters] = 0.50 * np.ones((gt_network.L, N_exa))
        else:
            theta[:, selected_reporters] = 50.0 * np.ones((gt_network.L, N_exa))
    return theta
